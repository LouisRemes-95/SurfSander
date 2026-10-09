import sys
import argparse

from rich.traceback import install
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

install(show_locals=True)

from surfsander.common import console

class RichArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        console.print(f"[bold red]✖ Argument error:[/bold red] {message}\n")
        self.print_help()
        raise SystemExit(2)


def build_parser() -> RichArgumentParser:
    parser = RichArgumentParser(
        prog="SurfSander",
        description="Local curvature based surface triangulation smoothing module",
        formatter_class=RichHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to the input triangulation or voxelisation",
    )

    return parser


def main() -> int:
    parser = build_parser()

    console.print(
        Panel(
            "[bold cyan]SurfSander[/bold cyan]\n"
            "[dim]Curvature-driven surface smoothing[/dim]",
            border_style="cyan",
            expand=False,
        )
    )

    args = parser.parse_args()

    return 0


if __name__ == "__main__":
    sys.exit(main())