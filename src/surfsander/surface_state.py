from dataclasses import dataclass, field

import numpy as np
import trimesh

from rich.progress import track
from surfsander.common import console

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

    def _remove_multifolds(self) -> None:
        vertices = self.mesh.vertices.copy()
        faces = self.mesh.faces.copy()

        n_vertices = len(vertices)

        for vertex in track(range(n_vertices), description="[cyan]Removing multifolds...", console=console, transient=True):
            vertex_queue = [vertex]
            vertices_to_dissociate = []

            while vertex_queue:
                current_vertex = vertex_queue.pop(0)

                # Add the current vertex.
                vertices_to_dissociate.append(current_vertex)

                # Find surrounding faces.
                surrounding_faces = faces[np.any(faces == current_vertex, axis=1)]

                # Find surrounding vertices.
                surrounding_vertices = surrounding_faces.reshape(-1)
                surrounding_vertices = surrounding_vertices[
                    ~np.isin(surrounding_vertices, vertices_to_dissociate)
                ]

                # Identify vertices appearing exactly four times.
                unique_vertices, counts = np.unique(
                    surrounding_vertices,
                    return_counts=True,
                )

                vertex_queue.extend(unique_vertices[counts == 4].tolist())

            vertices, faces = self._dissociate_vertices(
                vertices,
                faces,
                vertices_to_dissociate,
            )

        self.mesh.vertices = vertices
        self.mesh.faces = faces

    @staticmethod
    def _dissociate_vertices(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertices_to_dissociate: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:

        vertices_still_to_dissociate = vertices_to_dissociate.copy()

        for vertex in vertices_to_dissociate:
            # Find surrounding faces.
            surrounding_face_indices = np.flatnonzero(
                np.any(faces == vertex, axis=1)
            )
            surrounding_faces = faces[surrounding_face_indices]

            used_faces = np.zeros(len(surrounding_faces), dtype=bool)

            while not np.all(used_faces):
                start = np.flatnonzero(~used_faces)[0]
                used_faces[start] = True

                cycle = np.zeros(len(used_faces), dtype=bool)
                cycle[start] = True

                analysed_faces = [start]

                while analysed_faces:
                    current = analysed_faces.pop(0)

                    similar_vertices = np.isin(
                        surrounding_faces,
                        surrounding_faces[current],
                    )

                    adjacent_faces = np.flatnonzero(
                        (np.sum(similar_vertices, axis=1) == 2)
                        & (
                            np.sum(
                                similar_vertices
                                & np.isin(
                                    surrounding_faces,
                                    vertices_still_to_dissociate,
                                ),
                                axis=1,
                            ) != 2
                        )
                    )

                    adjacent_faces = adjacent_faces[
                        ~used_faces[adjacent_faces]
                    ]

                    analysed_faces.extend(adjacent_faces.tolist())
                    used_faces[adjacent_faces] = True
                    cycle[adjacent_faces] = True

                # If multiple face cycles exist, separate them.
                if not np.all(used_faces):
                    new_vertex = len(vertices)

                    vertices = np.vstack([
                        vertices,
                        vertices[vertex],
                    ])

                    selected_faces = surrounding_face_indices[cycle]

                    # Rereference the current vertex in selected faces.
                    selected = faces[selected_faces].copy()
                    selected[selected == vertex] = new_vertex
                    faces[selected_faces] = selected

            vertices_still_to_dissociate.pop(0)

        return vertices, faces



