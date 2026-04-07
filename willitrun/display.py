"""Rich-based terminal output formatting."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .estimator import Estimate, get_best_tflops
from .recommender import Recommendation

console = Console()

_VERDICT_STYLES = {
    "RUNS_GREAT": ("bold green", "[green]\u2705 Runs great[/green]"),
    "RUNS": ("bold yellow", "[yellow]\u2714 Runs[/yellow]"),
    "NEEDS_QUANT": ("bold yellow", "[yellow]\u26a0\ufe0f  Needs quantization[/yellow]"),
    "TIGHT": ("bold yellow", "[yellow]\u26a0\ufe0f  Tight fit[/yellow]"),
    "WONT_FIT": ("bold red", "[red]\u274c Won't fit[/red]"),
    "?": ("bold white", "[white]? Unknown[/white]"),
}


def display_result(est: Estimate, rec: Recommendation) -> None:
    """Display the estimation result as a beautiful Rich panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan", width=18)
    table.add_column("Value")

    # Parameters
    params_str = est.parameters_human
    if est.is_moe and est.active_params is not None:
        active_b = est.active_params / 1_000_000_000
        params_str += f" total  [dim]({active_b:.1f}B active per token)[/dim]"
    if est.model_source == "huggingface":
        params_str += "  [dim](estimated from HuggingFace config)[/dim]"
    table.add_row("Parameters:", params_str)

    # FLOPs (vision models)
    if est.flops:
        table.add_row("FLOPs:", est.flops_human)

    # Memory at each precision
    if est.memory_by_precision:
        first = True
        for prec, mem_str in est.memory_by_precision.items():
            fit = getattr(est, f"fits_{prec}", None)
            if fit is True:
                indicator = " [green]\u2714[/green]"
            elif fit is False:
                indicator = " [red]\u274c[/red]"
            else:
                indicator = ""

            label = "Model Memory:" if first else ""
            table.add_row(label, f"~{mem_str} ({prec.upper()}){indicator}")
            first = False

    # Performance
    if est.tier == 1 and est.benchmark:
        b = est.benchmark
        is_llm = est.model_type == "llm"

        if is_llm:
            # Text generation (primary — what users experience during chat)
            tg_perf = f"{b.value:.0f} {b.metric}"
            if b.framework:
                tg_perf += f"  [dim]({b.framework} {b.precision.upper()})[/dim]"
            table.add_row("Text gen:", tg_perf)

            # Prompt processing (secondary — context ingestion speed)
            if est.secondary_benchmark:
                pp = est.secondary_benchmark
                table.add_row(
                    "Prompt proc:",
                    f"[dim]{pp.value:.0f} {pp.metric}  (context loading)[/dim]",
                )
        else:
            perf = f"{b.value:.0f} {b.metric}"
            if b.framework:
                perf += f"  [dim]({b.framework} {b.precision.upper()})[/dim]"
            table.add_row(f"Speed:", perf)

        table.add_row("Confidence:", "[bold green]● Measured[/bold green]")
        if b.source:
            table.add_row("Source:", f"[dim]{b.source}[/dim]")

    elif est.tier == 2 and est.estimated_fps is not None:
        metric_label = "Text gen:" if est.model_type == "llm" else "Speed:"
        perf = f"~{est.estimated_fps:.0f} {est.metric}"
        table.add_row(metric_label, perf)

        if est.estimated_fps_range:
            low, high = est.estimated_fps_range
            pct = int(round((high - est.estimated_fps) / est.estimated_fps * 100))
            table.add_row("Range:", f"[dim]{low:.0f}–{high:.0f} {est.metric}  (±{pct}%)[/dim]")

        _strategy_labels = {
            1: "Parameter-ratio scaling from same-device benchmark",
            2: "TFLOPS scaling from same-model on another device",
            3: "Rough FLOPs scaling — different architecture",
        }
        method_str = _strategy_labels.get(est.tier2_strategy, "Scaling estimate")
        table.add_row("Confidence:", "[bold yellow]◎ Scaled estimate[/bold yellow]")
        table.add_row("Method:", f"[dim]{method_str}[/dim]")
        if est.scaling_notes:
            table.add_row("", f"[dim]{est.scaling_notes}[/dim]")

    else:
        # No benchmark data at all
        table.add_row("Speed:", "[dim]—[/dim]")
        table.add_row("Confidence:", "[dim]○ No benchmark data[/dim]")
        if est.memory_by_precision:
            table.add_row("", "[dim]Memory fit above is reliable. Speed requires benchmark data.[/dim]")

    # Device memory
    if est.device_memory_gb:
        table.add_row("Device RAM:", f"{est.device_memory_gb} GB")

    # Verdict
    _, verdict_rich = _VERDICT_STYLES.get(
        rec.verdict_emoji, ("bold white", f"[white]{rec.verdict_text}[/white]")
    )
    table.add_row("Verdict:", verdict_rich)

    # Title
    title = f"[bold]{est.model_name}[/bold] on [bold]{est.device_name}[/bold]"

    panel = Panel(
        table,
        title=title,
        border_style="blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)

    # Suggestions
    if rec.suggestions:
        console.print()
        for s in rec.suggestions:
            console.print(f"  [dim]\u2192 {s}[/dim]")
        console.print()


def display_devices(devices: dict) -> None:
    """Display available devices in a table."""
    table = Table(title="Available Devices", border_style="blue")
    table.add_column("ID", style="bold cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Memory")
    table.add_column("Best TFLOPS")

    for device_id, spec in sorted(devices.items()):
        mem = spec.get("memory", {}).get("total_gb")
        mem_str = f"{mem} GB" if mem is not None else "?"

        # Use get_best_tflops so Pascal GPUs (fp32 only) and
        # devices with only fp16 both show a meaningful number.
        tflops = get_best_tflops(spec)
        tflops_str = f"{tflops:.1f}" if tflops else "?"

        table.add_row(
            device_id,
            spec.get("name", ""),
            spec.get("type", ""),
            mem_str,
            tflops_str,
        )

    console.print()
    console.print(table)
    console.print()


def display_models(models: dict) -> None:
    """Display available models in a table."""
    table = Table(title="Available Models", border_style="blue")
    table.add_column("ID", style="bold cyan")
    table.add_column("Type")
    table.add_column("Architecture")
    table.add_column("Parameters")
    table.add_column("Aliases")

    for model_id, spec in sorted(models.items()):
        if not isinstance(spec, dict):
            continue
        params = spec.get("parameters")
        if params and params >= 1_000_000_000:
            params_str = f"{params / 1_000_000_000:.1f}B"
        elif params and params >= 1_000_000:
            params_str = f"{params / 1_000_000:.1f}M"
        elif params:
            params_str = f"{params / 1_000:.0f}K"
        else:
            params_str = "?"

        aliases = spec.get("aliases", [])
        aliases_str = ", ".join(aliases[:2])
        if len(aliases) > 2:
            aliases_str += f" (+{len(aliases) - 2})"

        table.add_row(
            model_id,
            spec.get("model_type", ""),
            spec.get("architecture", ""),
            params_str,
            aliases_str,
        )

    console.print()
    console.print(table)
    console.print()
