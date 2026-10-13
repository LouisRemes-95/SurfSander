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
        self.remove_duplicate_faces()

        n_vertices = len(self.mesh.vertices)
        n_edges = len(self.mesh.edges_unique)

        self._edge_tagging = np.zeros(n_edges, dtype=bool)
        self._k1 = np.zeros(n_vertices, dtype=np.float64)
        self._k2 = np.zeros(n_vertices, dtype=np.float64)

    
    def remove_duplicate_faces(self) -> None:
        sorted_faces = np.sort(self.mesh.faces, axis=1)

        _, inverse, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)

        self.mesh.update_faces(counts[inverse] == 1)

        self.mesh.remove_unreferenced_vertices()