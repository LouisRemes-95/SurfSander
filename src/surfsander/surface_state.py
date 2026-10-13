from dataclasses import dataclass, field

import numpy as np
import trimesh

@dataclass
class SurfaceState:
    mesh: trimesh.Trimesh

    _edge_tagging: np.ndarray = field(init=False)
    _k1: np.ndarray = field(init=False)
    _k2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_vertices = len(self.mesh.vertices)
        n_edges = len(self.mesh.edges_unique)

        self._edge_tagging = np.zeros(n_edges, dtype=bool)
        self._k1 = np.zeros(n_vertices, dtype=np.float64)
        self._k2 = np.zeros(n_vertices, dtype=np.float64)