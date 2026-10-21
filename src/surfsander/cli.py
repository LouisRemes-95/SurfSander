import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.traceback import install
from rich_argparse import RichHelpFormatter

from surfsander.common import UserError, console
from surfsander.pipeline import AlgConfig, run

install(show_locals=True)


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
        allow_abbrev=False,
    )

    parser.add_argument(
        "input_path",
        type=Path,
        default=None,
        help="Path to the input voxelisation in .npz file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Path to the directory where outputs should be saved",
    )
    parser.add_argument(
        "-on",
        "--output-name",
        type=str,
        default=None,
        help="Name of the output file without suffix",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Physical size of one voxel dimension in mm (default: %(default)s)",
    )
    parser.add_argument(
        "--k1-max",
        type=float,
        default=None,
        help="Maximum allowed K1 curvature in mm⁻¹ (default: None)",
    )
    parser.add_argument(
        "--k2-max",
        type=float,
        default=None,
        help="Maximum allowed K2 curvature in mm⁻¹ (default: None)",
    )
    parser.add_argument(
        "--adaptive-zoom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adaptive zooming on the voxelisation (default: True)",
    )

    return parser


def resolve_input(path: Path) -> Path:
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise UserError("input_path must be an existing file")
    return path


def resolve_output(input_path: Path, dir: Path | None, name: str | None) -> Path:
    dir = (dir or input_path.parent).resolve()

    dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = input_path.name.removesuffix("".join(input_path.suffixes))

    return (dir / name).with_suffix(".ply")


def main() -> int:
    parser = build_parser()

    console.print(
        Panel(
            "[bold cyan]SurfSander[/bold cyan]\nCurvature-driven surface smoothing",
            border_style="cyan",
            expand=False,
        )
    )

    args = parser.parse_args()

    try:
        input_path = resolve_input(args.input_path)
        output_path = resolve_output(input_path, args.output_dir, args.output_name)

        alg_config = AlgConfig(
            input_path,
            output_path,
            args.voxel_size,
            args.k1_max,
            args.k2_max,
            args.adaptive_zoom,
        )

        return run(alg_config)

    except UserError as e:
        console.print(f"[bold red]✖ Error:[/bold red] {e}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    sys.exit(main())
