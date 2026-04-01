"""soup deploy — deploy models to inference runtimes (Ollama)."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)


@app.command()
def ollama(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to GGUF model file",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Ollama model name (e.g. soup-my-model)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the model",
    ),
    template: str = typer.Option(
        "auto",
        "--template",
        "-t",
        help="Chat template: auto, chatml, llama, mistral, vicuna, zephyr",
    ),
    parameter: Optional[List[str]] = typer.Option(
        None,
        "--parameter",
        "-p",
        help="Ollama parameter (repeatable): temperature=0.7, top_p=0.9, etc.",
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List Soup-deployed models in Ollama",
    ),
    remove: Optional[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Remove a model from Ollama by name",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Deploy a GGUF model to local Ollama instance."""
    from soup_cli.utils.ollama import (
        OLLAMA_TEMPLATES,
        create_modelfile,
        deploy_to_ollama,
        detect_ollama,
        list_soup_models,
        remove_model,
        validate_gguf_path,
        validate_model_name,
    )

    # --- List mode ---
    if list_models:
        version = detect_ollama()
        if not version:
            console.print("[red]Ollama not found.[/] Install from https://ollama.com")
            raise typer.Exit(1)

        models = list_soup_models()
        if not models:
            console.print("[yellow]No Soup-deployed models found in Ollama.[/]")
            console.print("[dim]Deploy a model with: soup deploy ollama --model <gguf>[/]")
            raise typer.Exit(0)

        table = Table(title="Soup Models in Ollama")
        table.add_column("Name", style="bold cyan")
        table.add_column("Size", style="green")
        for entry in models:
            table.add_row(entry["name"], entry["size"])
        console.print(table)
        raise typer.Exit(0)

    # --- Remove mode ---
    if remove:
        version = detect_ollama()
        if not version:
            console.print("[red]Ollama not found.[/] Install from https://ollama.com")
            raise typer.Exit(1)

        if not yes:
            confirm = typer.confirm(f"Remove model '{remove}' from Ollama?")
            if not confirm:
                raise typer.Exit(0)

        success, message = remove_model(remove)
        if success:
            console.print(f"[green]{message}[/]")
        else:
            console.print(f"[red]{message}[/]")
            raise typer.Exit(1)
        raise typer.Exit(0)

    # --- Deploy mode: require --model and --name ---
    if not model:
        console.print("[red]--model is required for deploy.[/]")
        console.print("[dim]Usage: soup deploy ollama --model <gguf> --name <name>[/]")
        raise typer.Exit(1)

    if not name:
        console.print("[red]--name is required for deploy.[/]")
        console.print("[dim]Usage: soup deploy ollama --model <gguf> --name <name>[/]")
        raise typer.Exit(1)

    # Validate model name
    valid_name, name_err = validate_model_name(name)
    if not valid_name:
        console.print(f"[red]Invalid model name:[/] {name_err}")
        raise typer.Exit(1)

    # Validate GGUF path
    gguf_path = Path(model)
    valid_path, path_err = validate_gguf_path(gguf_path)
    if not valid_path:
        console.print(f"[red]{path_err}[/]")
        raise typer.Exit(1)

    # Check Ollama is installed
    version = detect_ollama()
    if not version:
        console.print(
            "[red]Ollama not found.[/]\n"
            "Install from: [bold]https://ollama.com[/]"
        )
        raise typer.Exit(1)

    # Resolve template
    resolved_template = None
    if template == "auto":
        # Try to infer from soup.yaml in cwd
        resolved_template = _auto_detect_template()
        if not resolved_template:
            resolved_template = "chatml"  # Default fallback
    elif template in OLLAMA_TEMPLATES:
        resolved_template = template
    else:
        console.print(
            f"[red]Unknown template: {template}[/]\n"
            f"Available: auto, {', '.join(OLLAMA_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    # Parse parameters
    params = {}
    if parameter:
        for param_str in parameter:
            if "=" not in param_str:
                console.print(f"[red]Invalid parameter format: {param_str}[/]")
                console.print("[dim]Expected format: key=value (e.g. temperature=0.7)[/]")
                raise typer.Exit(1)
            key, value = param_str.split("=", 1)
            params[key.strip()] = value.strip()

    # Show deploy plan
    console.print(
        Panel(
            f"Model:    [bold]{name}[/]\n"
            f"GGUF:     [bold]{gguf_path}[/]\n"
            f"Template: [bold]{resolved_template}[/]"
            + (f"\nSystem:   [bold]{system}[/]" if system else "")
            + (f"\nParams:   [bold]{params}[/]" if params else ""),
            title="Deploy to Ollama",
        )
    )

    # Confirmation — warn that this overwrites an existing model
    if not yes:
        console.print(
            "[yellow]Warning:[/] This will overwrite any existing Ollama model "
            f"named '{name}'."
        )
        confirm = typer.confirm("Proceed?")
        if not confirm:
            raise typer.Exit(0)

    # Generate Modelfile
    console.print(f"[green]\u2713[/] Ollama v{version} detected")
    try:
        modelfile = create_modelfile(
            gguf_path=gguf_path,
            template=resolved_template,
            system_prompt=system,
            parameters=params,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid parameter:[/] {exc}")
        raise typer.Exit(1)
    console.print("[green]\u2713[/] Modelfile generated")

    # Deploy
    console.print("[dim]Creating model in Ollama...[/]")
    success, message = deploy_to_ollama(name, modelfile)
    if not success:
        console.print(f"[red]Deploy failed:[/] {message}")
        raise typer.Exit(1)

    console.print(f"[green]\u2713[/] Model created: [bold]{name}[/]")
    console.print(
        Panel(
            f"Run: [bold]ollama run {name}[/]",
            title="[bold green]Deploy Complete![/]",
        )
    )


def _auto_detect_template() -> Optional[str]:
    """Try to infer chat template from soup.yaml in cwd."""
    from soup_cli.utils.ollama import infer_chat_template

    config_path = Path("soup.yaml")
    if not config_path.exists():
        return None

    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            return None
        data_section = config.get("data", {})
        if isinstance(data_section, dict):
            fmt = data_section.get("format")
            return infer_chat_template(fmt)
    except (yaml.YAMLError, OSError, KeyError, ImportError):
        return None
    return None
