"""Interactive mode for willitrun — triggered when no arguments are given.

Uses questionary.text() + a lazy completer for live-filtered fzf-style pickers
when questionary is installed; falls back to a Rich-rendered numbered list otherwise.

Lazy completer behaviour:
  - Empty input + typing  → no dropdown (clean prompt)
  - Empty input + Tab     → full list appears (escape hatch)
  - Any text typed        → filtered matches appear

Returns (ModelInfo, device_id) so the caller already has a resolved model —
no second network trip in cli.py.
"""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel

from .display import console, display_ranked_models
from .loader import (
    ModelInfo,
    resolve_from_database,
    resolve_from_file,
    resolve_from_huggingface,
)


# ---------------------------------------------------------------------------
# Lazy completer — core of the fzf-style UX
# ---------------------------------------------------------------------------

def _make_lazy_completer(choices: list[str]):
    """Return a prompt_toolkit Completer that only fires after the user types.

    Pressing Tab on an empty prompt still expands the full list so users who
    want to browse have an escape hatch.

    Uses a hand-rolled substring match so that typing '409' matches
    'rtx-4090-24gb', '7b' matches every 7B LLM, etc.  WordCompleter with
    sentence=True has position-arithmetic issues for mid-string hits; we
    avoid that by always replacing the full current input with the chosen
    display string (then _slug_from_choice strips the padding on confirm).
    """
    from prompt_toolkit.completion import Completer, Completion  # noqa: PLC0415

    class _LazyCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text
            show_all = complete_event.completion_requested
            if not show_all and not text.strip():
                return                          # clean prompt, nothing typed yet

            needle = text.lower()
            for choice in choices:
                if show_all or needle in choice.lower():
                    # Replace the entire current input with the full display string
                    yield Completion(
                        choice,
                        start_position=-len(text),
                        display=choice,
                    )

    return _LazyCompleter()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_params(params: int | None) -> str:
    if params is None:
        return "?"
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    return f"{params / 1_000:.0f}K"


def _model_choices(models: dict) -> list[str]:
    """Padded display strings: slug   params   type"""
    rows = [(slug, _fmt_params(s.get("parameters")), s.get("model_type", ""))
            for slug, s in sorted(models.items())]
    slug_w = max(len(r[0]) for r in rows) + 2
    param_w = max(len(r[1]) for r in rows) + 2
    return [f"{slug:<{slug_w}}{params:<{param_w}}{mtype}" for slug, params, mtype in rows]


def _device_choices(devices: dict) -> list[str]:
    """Padded display strings: device-id   XGB   type"""
    rows = []
    for did, spec in sorted(devices.items()):
        mem = spec.get("memory", {}).get("total_gb")
        rows.append((did, f"{mem}GB" if mem is not None else "?", spec.get("type", "")))
    did_w  = max(len(r[0]) for r in rows) + 2
    mem_w  = max(len(r[1]) for r in rows) + 2
    return [f"{did:<{did_w}}{mem:<{mem_w}}{dtype}" for did, mem, dtype in rows]


def _slug_from_choice(choice: str) -> str:
    """Extract the leading slug/id from a padded display string."""
    return choice.split()[0]


def _looks_like_hf_id(text: str) -> bool:
    return "/" in text and not Path(text).exists()


def _looks_like_path(text: str) -> bool:
    return text.endswith((".pt", ".onnx", ".bin")) or Path(text).exists()


# ---------------------------------------------------------------------------
# Intro panel
# ---------------------------------------------------------------------------

def _print_intro() -> None:
    console.print()
    console.print(Panel(
        "[dim]Check if an ML model fits and runs on your hardware.[/dim]\n"
        "[dim]Type a slug, a HuggingFace ID [bold](user/model)[/bold],"
        " or a local file path.[/dim]",
        title="[bold]willitrun[/bold]",
        border_style="blue",
        padding=(0, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Model resolution with inline feedback
# ---------------------------------------------------------------------------

def _resolve_with_feedback(model_str: str) -> ModelInfo:
    """Resolve model string → ModelInfo, printing status as we go."""
    # 1. Local database (instant — no spinner needed)
    info = resolve_from_database(model_str)
    if info:
        return info

    # 2. Local file
    if Path(model_str).exists():
        info = resolve_from_file(model_str)
        if info:
            return info

    # 3. HuggingFace — announce it so the user knows a network call is happening
    if "/" in model_str:
        console.print(
            "  [dim]Not in local database"
            " — fetching from HuggingFace...[/dim]"
        )
        info = resolve_from_huggingface(model_str)
        if info:
            return info
        console.print(
            "  [yellow]⚠[/yellow]  [dim]Could not fetch config from HuggingFace."
            " Estimation will be limited.[/dim]"
        )

    return ModelInfo(name=model_str, source="unknown")


def _print_model_confirmation(info: ModelInfo) -> None:
    """One-line summary printed after model is resolved."""
    params = _fmt_params(info.parameters)
    mtype  = info.model_type or ""

    parts = [f"[bold]{info.name}[/bold]"]
    if info.parameters:
        parts.append(f"{params} params")
    if mtype:
        parts.append(mtype)

    detail = "  ·  ".join(parts)

    if info.source == "database":
        badge = "[dim]local database[/dim]"
        icon  = "[green]✓[/green]"
    elif info.source == "huggingface":
        badge = "[dim]HuggingFace[/dim]"
        icon  = "[green]✓[/green]"
    elif info.source == "file":
        badge = "[dim]local file[/dim]"
        icon  = "[green]✓[/green]"
    else:
        badge = "[dim]not found — estimation may be limited[/dim]"
        icon  = "[yellow]⚠[/yellow]"

    console.print(f"  {icon}  {detail}  [dim]({badge})[/dim]" if info.source != "unknown"
                  else f"  {icon}  [bold]{info.name}[/bold]  {badge}")
    console.print()


# ---------------------------------------------------------------------------
# questionary path
# ---------------------------------------------------------------------------

_STYLE_DEF = [
    ("qmark",       "fg:#5f8700 bold"),
    ("question",    "bold"),
    ("answer",      "fg:#875fd7 bold"),
    ("pointer",     "fg:#875fd7 bold"),
    ("highlighted", "fg:#875fd7 bold"),
    ("selected",    "fg:#875fd7"),
    ("separator",   "fg:#444444"),
    ("instruction", "fg:#444444"),
    ("text",        ""),
    ("disabled",    "fg:#858585 italic"),
]


def _run_inverse_questionary(devices: dict, style) -> None:
    """Interactive flow for 'Find best models for my device'."""
    import questionary  # noqa: PLC0415
    from .ranker import list_categories, get_best_models_for_device  # noqa: PLC0415

    device_display = _device_choices(devices)

    # ── Step 1: device ─────────────────────────────────────────────
    raw_device = questionary.text(
        "Device",
        instruction="(type to filter · Tab for full list)",
        completer=_make_lazy_completer(device_display),
        complete_while_typing=True,
        style=style,
        validate=lambda _: True,
    ).ask()

    if raw_device is None:
        raise SystemExit(0)

    raw_device = raw_device.strip()
    device_id  = _slug_from_choice(raw_device) if " " in raw_device else raw_device
    device_name = devices.get(device_id, {}).get("name", device_id)

    console.print()

    # ── Step 2: category ───────────────────────────────────────────
    categories = list_categories()
    if not categories:
        console.print("[yellow]No model categories found in the database.[/yellow]")
        return

    category = questionary.select(
        "Category",
        choices=categories,
        style=style,
    ).ask()

    if category is None:
        raise SystemExit(0)

    console.print()

    # ── Step 3: compute and display ────────────────────────────────
    with console.status(f"[bold blue]Ranking {category} models for {device_name}...[/bold blue]"):
        results = get_best_models_for_device(device_id, category)

    if not results:
        console.print(
            f"[yellow]No models or benchmarks found for category '{category}'.[/yellow]"
        )
        return

    display_ranked_models(results, device_name, category)


def _run_questionary(models: dict, devices: dict) -> tuple[ModelInfo | None, str | None]:
    import questionary  # noqa: PLC0415

    style          = questionary.Style(_STYLE_DEF)
    model_display  = _model_choices(models)
    device_display = _device_choices(devices)

    _print_intro()

    # ── Mode selection ──────────────────────────────────────────────
    mode = questionary.select(
        "What do you want to do?",
        choices=[
            "Check a specific model on a device",
            "Find best models for my device",
        ],
        style=style,
    ).ask()

    if mode is None:
        raise SystemExit(0)

    console.print()

    if mode == "Find best models for my device":
        _run_inverse_questionary(devices, style)
        return None, None

    # ── Step 1: model ──────────────────────────────────────────────
    raw_model = questionary.text(
        "Model",
        instruction="(type to filter · Tab for full list · HuggingFace ID or file path accepted)",
        completer=_make_lazy_completer(model_display),
        complete_while_typing=True,
        style=style,
        validate=lambda _: True,
    ).ask()

    if raw_model is None:           # Ctrl-C
        raise SystemExit(0)

    raw_model = raw_model.strip()
    # File paths may contain spaces — preserve them as-is.
    # Everything else (dropdown selections with padding, free-typed slugs,
    # HF IDs) should have the leading token extracted so the padded columns
    # don't leak into the resolver.
    if _looks_like_path(raw_model):
        model_str = raw_model
    elif " " in raw_model:
        model_str = _slug_from_choice(raw_model)   # strip padding from dropdown
    else:
        model_str = raw_model                       # clean slug or HF ID typed directly

    console.print()
    model_info = _resolve_with_feedback(model_str)
    _print_model_confirmation(model_info)

    # ── Step 2: device ─────────────────────────────────────────────
    raw_device = questionary.text(
        "Device",
        instruction="(type to filter · Tab for full list)",
        completer=_make_lazy_completer(device_display),
        complete_while_typing=True,
        style=style,
        validate=lambda _: True,
    ).ask()

    if raw_device is None:
        raise SystemExit(0)

    raw_device = raw_device.strip()
    device_id  = _slug_from_choice(raw_device) if " " in raw_device else raw_device
    # (devices are never file paths, so no path exception needed)

    console.print()
    return model_info, device_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_interactive(
    models: dict, devices: dict
) -> tuple[ModelInfo | None, str | None]:
    """Run interactive mode and return (model_info, device_id).

    Returns (None, None) when the inverse flow ('Find best models') was
    selected — results are displayed inside this function and the caller
    should exit cleanly without further processing.

    The model is fully resolved here (database / file / HuggingFace) so the
    caller does not need a second resolution pass.
    """
    return _run_questionary(models, devices)
