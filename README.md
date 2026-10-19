# SurfSander

SurfSander converts a voxel volume into a smooth triangular surface mesh. It
reads a three-dimensional voxelisation, builds an isosurface, repairs voxel
configurations that would make the surface topologically ambiguous, and
smooths regions whose local curvature exceeds the requested limits.

The command writes a `.ply` triangle mesh suitable for inspection or a
downstream meshing workflow.

## Installation

SurfSander uses [uv](https://docs.astral.sh/uv/) for dependency and environment
management, and requires Python 3.13 or newer.

```bash
git clone https://github.com/LouisRemes-95/SurfSander.git
cd SurfSander
uv sync
```

## Quick start

Run it on any compatible voxelisation:

```bash
uv run surfsander path/to/voxelisation.npz \
  --voxel-size 0.025 \
  --k1-max 1.0 \
  --k2-max 0.5
```

This creates `voxelisation.ply` beside the input file. To choose a
destination and basename:

```bash
uv run surfsander path/to/voxelisation.npz \
  --output-dir ./surface_output \
  --name smoothed_interface \
  --voxel-size 0.025 \
  --k1-max 1.0 \
  --k2-max 0.5
```

The output in this example is `surface_output/smoothed_interface.ply`.

## Command-line reference

```text
surfsander INPUT [options]
```

| Option | Description |
| --- | --- |
| `INPUT` | Required `.npz` voxelisation file. It must contain exactly one three-dimensional NumPy array. |
| `-o`, `--output-dir PATH` | Directory for the `.ply` file. Defaults to the input file's directory. |
| `-n`, `--name NAME` | Output basename, without an extension. Defaults to the input basename. |
| `--voxel-size VALUE` | Physical cubic voxel size in millimetres. Default: `1.0`. |
| `--k1-max VALUE` | Maximum permitted first principal curvature in mm⁻¹. Omit for no limit. |
| `--k2-max VALUE` | Maximum permitted second principal curvature in mm⁻¹. Omit for no limit. |
| `--adaptive-zoom`, `--no-adaptive-zoom` | Enable or disable curvature-dependent voxel resampling before surface extraction. Enabled by default. |

When both curvature limits are supplied, `k2-max` must not exceed `k1-max`.
Curvature limits, when supplied, must be positive for smoothing to run.

## Input and output

### Voxelisation input

The input format is a `.npz` archive with exactly one 3-D array. Array names
are not significant. SurfSander derives an Otsu threshold from the volume; it
therefore accepts either a binary volume or scalar voxel data with separable
foreground and background values.

The voxel spacing is not stored in the archive. Always pass the source scan's
physical voxel size with `--voxel-size` when millimetre-scale geometry or
curvature limits matter.

### PLY output

SurfSander writes one triangular `.ply` mesh. Coordinates use the same voxel
axis order as the input volume and are scaled using `--voxel-size`. Boundary
vertices are retained on the extracted volume bounds during smoothing.

## Processing pipeline

1. Load the voxelisation and choose an Otsu isovalue.
2. Optionally resample the volume using the requested curvature scale.
3. Convert the thresholded volume to a well-composed binary volume, repairing
   critical local voxel configurations before surface extraction.
4. Extract an isosurface with marching cubes.
5. Estimate principal curvatures and iteratively smooth high-curvature peaks,
   saddle regions, and their surroundings. The mesh topology may be adjusted
   by splitting or collapsing edges as part of this process.
6. Export the resulting mesh as PLY.

With neither curvature limit set, SurfSander still performs voxel-topology
repair and surface extraction, but does not need to change the surface for a
curvature constraint.

## Project layout

- `src/surfsander/cli.py` — command-line interface and path handling.
- `src/surfsander/pipeline.py` — volume loading, isosurface extraction, and output.
- `src/surfsander/voxelisation_correction.py` — well-composedness repair for binary volumes.
- `src/surfsander/surface.py` — curvature estimation, smoothing, and mesh-topology operations.
