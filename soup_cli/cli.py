"""Main CLI entry point — all commands registered here."""

import sys

import typer
from rich.console import Console

from soup_cli import __version__
from soup_cli.commands import (
    chat,
    data,
    diff,
    eval,
    export,
    generate,
    init,
    merge,
    push,
    runs,
    serve,
    sweep,
    train,
)
from soup_cli.commands import doctor as doctor_cmd
from soup_cli.commands import quickstart as quickstart_cmd

console = Console()

# Global verbose flag — set via callback, read by error handler
_verbose = False

app = typer.Typer(
    name="soup",
    help="Fine-tune LLMs in one command. No SSH, no config hell.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command()(init.init)
app.command()(train.train)
app.command()(chat.chat)
app.command()(push.push)
app.command(name="export")(export.export)
app.command()(merge.merge)
app.add_typer(
    data.app, name="data",
    help="Dataset tools: inspect, convert, merge, dedup, validate, stats.",
)
app.add_typer(runs.app, name="runs", help="Experiment tracking: list, show, compare runs.")
app.command(name="eval")(eval.eval_model)
app.command()(serve.serve)
app.command()(sweep.sweep)
app.command(name="diff")(diff.diff)
app.command()(doctor_cmd.doctor)
app.command()(quickstart_cmd.quickstart)

# Register data generate as a subcommand of data
data.app.command(name="generate")(generate.generate)


@app.command()
def version():
    """Show Soup CLI version."""
    console.print(f"[bold green]soup[/] v{__version__}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show full traceback on errors",
    ),
):
    """Soup — fine-tune LLMs in one command."""
    global _verbose
    _verbose = verbose


def run():
    """Entry point with friendly error handling."""
    try:
        app()
    except SystemExit:
        raise
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)
    except Exception as exc:
        from soup_cli.utils.errors import format_friendly_error

        format_friendly_error(exc, verbose=_verbose)
        sys.exit(1)


# When invoked via `soup` entry point, use run() for error handling.
# When invoked via `python -m soup_cli`, __main__.py calls run() directly.
if __name__ == "__main__":
    run()
