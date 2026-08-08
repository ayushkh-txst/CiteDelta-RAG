"""Command-line entry point. Subcommands get added as blocks land."""

from __future__ import annotations

import typer

from citedelta import __version__

app = typer.Typer(
    name="citedelta",
    help="Bitemporal hybrid search over versioned regulations.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the version and exit."""
    typer.echo(f"citedelta {__version__}")


if __name__ == "__main__":
    app()
