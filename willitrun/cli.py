"""Click-based CLI entry point for willitrun."""

from __future__ import annotations

from pathlib import Path

import click

from . import __version__
from . import data_access
from .display import console, display_devices, display_models, display_result
from .estimator import estimate
from .interactive import run_interactive
from .loader import resolve_model
from .profiler import profile_model
from .recommender import recommend


def _status_msg_for(model: str) -> str:
    """Pick a spinner label based on what kind of input the model string looks like."""
    from pathlib import Path as _Path
    if _Path(model).exists():
        return "[bold blue]Reading local file...[/bold blue]"
    if "/" in model:
        return "[bold blue]Fetching from HuggingFace...[/bold blue]"
    return "[bold blue]Looking up model...[/bold blue]"


@click.command()
@click.argument("model", required=False)
@click.option(
    "--device", "-d",
    required=False,
    help="Target device ID (e.g. jetson-orin-nano-8gb, rtx-3060-12gb, apple-m2-8gb).",
)
@click.option(
    "--list-devices", is_flag=True,
    help="List all available devices.",
)
@click.option(
    "--list-models", is_flag=True,
    help="List all models in the benchmark database.",
)
@click.option(
    "--precision", "-p",
    type=click.Choice(["fp32", "fp16", "int8", "4bit"]),
    default=None,
    help="Override precision for estimation.",
)
@click.option(
    "--json", "output_json", is_flag=True,
    help="Output result as JSON.",
)
@click.version_option(version=__version__, prog_name="willitrun")
def main(
    model: str | None,
    device: str | None,
    list_devices: bool,
    list_models: bool,
    precision: str | None,
    output_json: bool,
) -> None:
    """Will your ML model run on your device? Find out in one command.

    MODEL can be a model name (yolov8n), a local file (model.pt, model.onnx),
    or a HuggingFace model ID (meta-llama/Llama-3-8B).
    """
    if list_devices:
        display_devices({d.device_id: d.model_dump() for d in data_access.list_devices()})
        return

    if list_models:
        display_models({m.model_id: m.model_dump() for m in data_access.list_models()})
        return

    # ── Interactive mode ────────────────────────────────────────────────────
    model_info = None
    if model is None and device is None:
        model_info, device = run_interactive(
            {m.model_id: m.model_dump() for m in data_access.list_models()},
            {d.device_id: d.model_dump() for d in data_access.list_devices()},
        )
        # model_info is already fully resolved; skip the resolve block below

    # ── Direct mode: require both arguments ─────────────────────────────────
    if model is None and model_info is None:
        console.print("[bold red]Error:[/bold red] MODEL argument is required.")
        console.print()
        console.print("Usage: [bold]willitrun MODEL --device DEVICE[/bold]")
        console.print()
        console.print("Discover what's available:")
        console.print("  willitrun --list-models    # models in the database")
        console.print("  willitrun --list-devices   # supported hardware")
        raise SystemExit(1)

    if device is None:
        console.print("[bold red]Error:[/bold red] --device is required.")
        console.print()
        console.print("Use [bold]--list-devices[/bold] to see available devices, e.g.:")
        console.print("  willitrun --list-devices")
        console.print()
        console.print("Example:")
        console.print("  willitrun yolov8n --device jetson-orin-nano-8gb")
        raise SystemExit(1)

    # ── Validate device (both paths) ────────────────────────────────────────
    devices = {d.device_id: d.model_dump() for d in data_access.list_devices()}
    if device not in devices:
        console.print(f"[bold red]Error:[/bold red] Unknown device [bold]'{device}'[/bold].")
        console.print()
        suggestions = [d for d in devices if device.lower() in d.lower()]
        if suggestions:
            console.print("Did you mean?")
            for s in suggestions[:3]:
                console.print(f"  [cyan]{s}[/cyan]  {devices[s].get('name', '')}")
        else:
            console.print("Run [bold]willitrun --list-devices[/bold] to see all supported hardware.")
        raise SystemExit(1)

    # ── Resolve model (direct mode only) ────────────────────────────────────
    if model_info is None:
        with console.status(_status_msg_for(model)):
            model_info = resolve_model(model)

        if model_info.source == "unknown" and model_info.parameters is None:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Could not resolve [bold]'{model}'[/bold]."
            )
            console.print(
                "  Not found in the local database, as a file, or on HuggingFace."
            )
            console.print(
                "  Use [bold]--list-models[/bold] to see available models."
            )
            console.print()

    # ── Estimation pipeline ─────────────────────────────────────────────────
    profile = profile_model(model_info)
    est     = estimate(profile, device)
    rec     = recommend(est)

    if output_json:
        _print_json(est, rec)
    else:
        display_result(est, rec)


def _print_json(est, rec):
    """Output result as JSON."""
    import json

    secondary = None
    if est.secondary_benchmark:
        secondary = {
            "value": est.secondary_benchmark.value,
            "metric": est.secondary_benchmark.metric,
            "label": "prompt_processing",
        }

    data = {
        "model": est.model_name,
        "device": est.device_name,
        "device_id": est.device_id,
        "parameters": est.parameters,
        "parameters_human": est.parameters_human,
        "flops": est.flops,
        "model_type": est.model_type,
        "tier": est.tier,
        "metric": est.metric,
        "estimated_value": est.estimated_fps,
        "estimated_range": list(est.estimated_fps_range) if est.estimated_fps_range else None,
        "secondary_metric": secondary,
        "memory": est.memory_by_precision,
        "device_memory_gb": est.device_memory_gb,
        "fits": {
            "fp32": est.fits_fp32,
            "fp16": est.fits_fp16,
            "int8": est.fits_int8,
            "4bit": est.fits_4bit,
        },
        "verdict": rec.verdict,
        "verdict_text": rec.verdict_text,
        "best_precision": rec.best_precision,
        "suggestions": rec.suggestions,
    }
    click.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
