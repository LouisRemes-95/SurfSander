import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import trimesh
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

from surfsander.common import UserError, console
from surfsander.surface_state import SurfaceState


@dataclass(frozen=True)
class AlgConfig:
    input_dir: Path
    output_dir: Path
    unit_size: float


def build_iso_surface(input_path: Path, unit_size: float) -> trimesh:
    match input_path.suffix:
        case ".npz":
            with np.load(input_path) as data:
                if len(data.files) != 1:
                    raise UserError(f"Expected a single array in '{input_path}', found {len(data.files)}: {data.files}")
                
                voxelisation = data[data.files[0]]

                if voxelisation.ndim != 3:
                    raise ValueError(f"Expected a 3D volume, got shape {voxelisation.shape}")

                threshold = threshold_otsu(voxelisation)

                vertices, faces, normals, values = marching_cubes(
                    voxelisation,
                    level=threshold,
                    spacing=(unit_size,) * 3,
                )

                # Create and export triangle mesh
                return trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=True,
                )

        case ".ply":
            return trimesh.load_mesh(input_path)

        case _:
            raise UserError(
                f"Unsupported input format '{input_path.suffix}'. "
            )


def local_smoothing(alg_config: AlgConfig) -> int:
    surface_state = SurfaceState(build_iso_surface(alg_config.input_dir, alg_config.unit_size))

    surface_state.mesh.export(alg_config.output_dir.with_suffix(".ply"))
