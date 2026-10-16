import sys
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import trimesh
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from scipy.ndimage import zoom

from surfsander.common import UserError, console
from surfsander.surface import Surface
from surfsander.voxelisation_correction import make_well_composed


@dataclass(frozen=True)
class AlgConfig:
    input_path: Path
    output_dir: Path
    voxel_size: float
    k1_max: float | None
    k2_max: float | None
    adaptive_zoom: bool

    def __post_init__(self) -> None:
        if self.k1_max is not None and self.k2_max is not None and self.k2_max > self.k1_max:
            raise UserError(f"k2_max ({self.k2_max} [mm⁻¹]) cannot be greater than k1_max ({self.k1_max} [mm⁻¹])")


def build_iso_surface(alg_config: AlgConfig) -> trimesh:
    voxelisation = _load_voxelisation(alg_config.input_path)

    zoom_factor = 1

    if alg_config.adaptive_zoom:
        k_max = alg_config.k1_max if alg_config.k1_max is not None else alg_config.k2_max

        if k_max is not None:
            zoom_factor = k_max * alg_config.voxel_size / 2

    voxelisation = zoom(voxelisation, zoom_factor, order=1)

    threshold = threshold_otsu(voxelisation)
    binary = voxelisation > threshold
    well_composed_binary, _ = make_well_composed(binary)

    voxelisation[well_composed_binary & ~ binary] = threshold * 1.01
    voxelisation[~ well_composed_binary & binary] = threshold * .99

    vertices, faces, _, _ = marching_cubes(
        voxelisation,
        level=threshold,
        spacing=(alg_config.voxel_size / zoom_factor,) * 3,
        allow_degenerate=False,
    )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


def _load_voxelisation(input_path: Path) -> np.ndarray:
    if input_path.suffix != ".npz":
        raise UserError("Only supports .npz files")
    
    with np.load(input_path) as data:
        if len(data.files) != 1:
            raise UserError(f"Expected a single array in '{input_path}', found {len(data.files)}: {data.files}")
        
        voxelisation = data[data.files[0]]

    if voxelisation.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {voxelisation.shape}")

    return voxelisation


def run(alg_config: AlgConfig) -> int:
    surface_state = Surface(build_iso_surface(alg_config))

    surface_state.locally_smooth(
        k1_max=alg_config.k1_max or float("inf"),
        k2_max=alg_config.k2_max or float("inf"),
    )

    surface_state.mesh.export(alg_config.output_dir.with_suffix(".ply"))

    return 0