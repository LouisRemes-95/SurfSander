import sys
import argparse
from pathlib import Path

from rich.traceback import install
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

install(show_locals=True)

from surfsander.common import UserError, console

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
        type=Path,
        default=None,
        help="Path to the input triangulation or voxelisation",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Path to the directory where outputs should be saved",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the output file without suffix",
    )

    return parser


def resolve_input(path: Path) -> Path:
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise UserError("input_dir must be an existing directory")
    return path

def resolve_output(input_path: Path, output_dir: Path | None, name: str | None) -> Path:
    output_dir = (output_dir or input_path.parent).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = input_path.name.removesuffix("".join(input_path.suffixes))

    return output_dir / name


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

    try:
        input_path = resolve_input(args.input)
        output_path = resolve_output(input_path, args.output_dir, args.name)

    except UserError as e:
        console.print(f"[bold red]✖ Error:[/bold red] {e}")
        raise SystemExit(1)

    print(input_path)
    print(output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())