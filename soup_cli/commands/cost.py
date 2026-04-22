"""soup cost — estimate training cost in USD."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.utils.gpu import model_size_from_name
from soup_cli.utils.profiler import (
    estimate_speed,
    estimate_training_time,
)

console = Console()

# GPU pricing and speed multipliers (relative to A100)
GPU_PRICING = [
    {"provider": "RunPod", "gpu": "A100 80G", "cost_per_hr": 1.89, "speed_mult": 1.0},
    {"provider": "Lambda", "gpu": "A100 40G", "cost_per_hr": 1.10, "speed_mult": 1.0},
    {"provider": "Modal", "gpu": "H100", "cost_per_hr": 5.92, "speed_mult": 2.5},
    {"provider": "Vast.ai", "gpu": "RTX 4090", "cost_per_hr": 0.35, "speed_mult": 0.7},
    {"provider": "CoreWeave", "gpu": "A100 80G", "cost_per_hr": 2.21, "speed_mult": 1.0},
]


def _get_dataset_size(cfg) -> int:
    """Estimate dataset size."""
    train_path = cfg.data.train
    path = Path(train_path)

    # Local file
    if path.exists():
        from soup_cli.data.loader import load_raw_data
        try:
            data = load_raw_data(path)
            # If val_split is used, training set is smaller
            split = 1.0 - cfg.data.val_split
            return int(len(data) * split)
        except Exception:
            pass

    # Try HF dataset (only if not a local file path with extension)
    if not path.suffix:
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(train_path)
            size = builder.info.splits["train"].num_examples
            split = 1.0 - cfg.data.val_split
            return int(size * split)
        except Exception:
            pass

    # Fallback default
    return 10000


def cost(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c", help="Path to soup.yaml config file"
    ),
    gpu: str = typer.Option(
        None, "--gpu", "-g",
        help="Filter by specific GPU (e.g., A100, H100, RTX 4090)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON for scripting"
    ),
):
    """Estimate training cost in USD."""
    from soup_cli.config.loader import load_config

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/] {config}")
        raise typer.Exit(1)

    cfg = load_config(config_path)
    model_params_b = model_size_from_name(cfg.base)

    batch_size = cfg.training.batch_size
    if batch_size == "auto":
        batch_size = 4
    else:
        batch_size = int(batch_size)

    dataset_size = _get_dataset_size(cfg)
    epochs = cfg.training.epochs

    # Base speed (A100)
    base_tokens_per_sec = estimate_speed(
        model_params_b, cfg.training.quantization, batch_size
    )
    base_samples_per_sec = base_tokens_per_sec / max(cfg.data.max_length, 1)

    results = []

    for provider_info in GPU_PRICING:
        if gpu:
            gpu_norm = gpu.lower().replace(" ", "")
            p_gpu_norm = provider_info["gpu"].lower().replace(" ", "")
            if gpu_norm not in p_gpu_norm:
                continue

        speed_mult = provider_info["speed_mult"]
        samples_per_sec = base_samples_per_sec * speed_mult

        # Duration in minutes
        duration_mins = estimate_training_time(dataset_size, epochs, samples_per_sec)
        duration_hrs = duration_mins / 60.0

        total_cost = duration_hrs * provider_info["cost_per_hr"]

        results.append({
            "provider": provider_info["provider"],
            "gpu": provider_info["gpu"],
            "cost_per_hr": provider_info["cost_per_hr"],
            "duration_hrs": duration_hrs,
            "total_cost": total_cost,
        })

    if not results:
        console.print(f"[red]No matching GPUs found for:[/] {gpu}")
        raise typer.Exit(1)

    if json_output:
        console.print(json.dumps(results, indent=2))
        return

    # Render table
    table = Table(title="Training Cost Estimate", title_justify="left", box=None, padding=(0, 2))
    table.add_column("Provider", style="cyan")
    table.add_column("GPU", style="green")
    table.add_column("$/hr", justify="right")
    table.add_column("Total Cost", justify="right")

    for r in results:
        cost_str = f"~${r['total_cost']:.2f}"

        # Format duration to match example, e.g. (4h) or (1.5h)
        # If it's very small, maybe (<1h)
        if r['duration_hrs'] < 1.0:
            dur_str = "(<1h)"
        else:
            dur_str = f"({r['duration_hrs']:.0f}h)"

        table.add_row(
            r["provider"],
            r["gpu"],
            f"${r['cost_per_hr']:.2f}",
            f"{cost_str} [dim]{dur_str}[/dim]"
        )

    console.print(table)
    console.print()
