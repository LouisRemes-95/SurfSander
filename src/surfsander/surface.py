import heapq
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import trimesh
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from surfsander.common import console


class CustomBarColumn(BarColumn):
    def render(self, task):
        if task.total is None:
            self.complete_style = "white"
            self.pulse_style = "white"
        else:
            self.complete_style = "blue"
            self.pulse_style = "blue"
        return super().render(task)
    

class _DynamicTopology:
    """Mutable triangle connectivity for local split/collapse operations.

    Vertex and face IDs remain stable until ``to_arrays`` compacts the mesh.
    Edge priorities are updated only in modified neighborhoods.
    """

    def __init__(self, surface: "Surface") -> None:
        self.vertices = [point.copy() for point in np.asarray(surface.mesh.vertices)]
        self.faces = [tuple(map(int, face)) for face in surface.mesh.faces]
        self.tags = list(map(int, surface.mesh.vertex_attributes["boundary_tag"]))
        self.k1 = list(map(float, surface._k1))
        self.k2 = list(map(float, surface._k2))

        self.vertex_faces = [set() for _ in self.vertices]
        self.edge_faces = defaultdict(set)
        self.tagged_edges = {tuple(map(int, edge)) for edge, tagged in zip(surface.mesh.edges_unique, surface._edge_tagging) if tagged}
        self.parked_edges = set()
        self.lengths = {}
        self.versions = defaultdict(int)
        self.min_heap = []
        self.max_heap = []
        self.length_sum = 0.0

        for index, face in enumerate(self.faces):
            self._attach(index, face)
        self._refresh(set(self.edge_faces))

    @staticmethod
    def _edges(face):
        a, b, c = face
        return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a))))

    def _attach(self, index, face):
        for vertex in face:
            self.vertex_faces[vertex].add(index)
        for edge in self._edges(face):
            self.edge_faces[edge].add(index)

    def _detach(self, index, face):
        for vertex in face:
            self.vertex_faces[vertex].discard(index)
        for edge in self._edges(face):
            self.edge_faces[edge].discard(index)
            if not self.edge_faces[edge]:
                del self.edge_faces[edge]

    def _refresh(self, edges, moved_vertices=()):
        for edge in edges:
            exists = edge in self.edge_faces
            previous = self.lengths.get(edge)

            # Only new, removed or geometrically changed edges need refreshing.
            if exists and previous is not None and edge[0] not in moved_vertices and edge[1] not in moved_vertices:
                continue

            if previous is not None:
                self.length_sum -= self.lengths.pop(edge)

            self.versions[edge] += 1

            if not exists:
                self.tagged_edges.discard(edge)
                self.parked_edges.discard(edge)
                continue

            a, b = edge
            length = float(np.sqrt(np.dot((self.vertices[a] - self.vertices[b]) ** 2, [1, 1, 1])))
            self.lengths[edge] = length
            self.length_sum += length

            version = self.versions[edge]
            if edge not in self.parked_edges:
                heapq.heappush(self.min_heap, (length, edge, version))
            heapq.heappush(self.max_heap, (-length, edge, version))

    def _replace_faces(self, replacements, additions=(), moved_vertices=()):
        affected = set()
        for index, face in replacements.items():
            previous = self.faces[index]
            if previous is not None:
                affected.update(self._edges(previous))
                self._detach(index, previous)
            self.faces[index] = tuple(map(int, face)) if face is not None else None
            if face is not None:
                self._attach(index, self.faces[index])
                affected.update(self._edges(self.faces[index]))

        for face in additions:
            face = tuple(map(int, face))
            index = len(self.faces)
            self.faces.append(face)
            self._attach(index, face)
            affected.update(self._edges(face))

        self._refresh(affected, moved_vertices=moved_vertices)

    def _reset_edge_tags(self):
        # Reconsider parked edges only when MATLAB resets edge tagging.
        self.tagged_edges.clear()

        for edge in self.parked_edges:
            if edge in self.lengths:
                heapq.heappush(self.min_heap, (self.lengths[edge], edge, self.versions[edge]))

        self.parked_edges.clear()

    def _append_vertex(self, vertex, tag, k1=0.0, k2=0.0):
        index = len(self.vertices)
        self.vertices.append(np.asarray(vertex, dtype=np.float64).copy())
        self.tags.append(int(tag))
        self.k1.append(float(k1))
        self.k2.append(float(k2))
        self.vertex_faces.append(set())
        return index

    def mean_length(self):
        return self.length_sum / len(self.lengths) if self.lengths else 0.0

    def next_edge(self, shortest, threshold):
        heap = self.min_heap if shortest else self.max_heap

        while heap:
            priority, edge, version = heap[0]
            if self.versions[edge] != version or edge not in self.lengths:
                heapq.heappop(heap)
                continue

            length = self.lengths[edge]
            if (length >= threshold) if shortest else (length <= threshold):
                return None

            if shortest and edge in self.tagged_edges:
                # Keep rejected edges out of the active heap until tags reset.
                self.parked_edges.add(edge)
                heapq.heappop(heap)
                continue

            return edge

        return None

    def remaining(self, low, high):
        return sum(length < low or length > high for length in self.lengths.values())

    def split(self, edge):
        a, b = edge
        midpoint = (self.vertices[a] + self.vertices[b]) / 2
        new_vertex = self._append_vertex(midpoint, self.tags[a] & self.tags[b])
        replacements = {}
        additions = []

        for index in sorted(self.edge_faces[edge]):
            original = self.faces[index]
            replacements[index] = tuple(new_vertex if vertex == b else vertex for vertex in original)
            additions.append(tuple(new_vertex if vertex == a else vertex for vertex in original))

        self._replace_faces(replacements, additions)
        # MATLAB's long-edge handling resets edge tags.
        self._reset_edge_tags()

    def _pipe_candidates(self, edge):
        a, b = edge
        if len(self.edge_faces[edge]) != 2:
            return []

        shared = set().union(*(self.faces[i] for i in self.edge_faces[edge]))
        neighbors_a = set().union(*(self.faces[i] for i in self.vertex_faces[a])) - {a, b}
        neighbors_b = set().union(*(self.faces[i] for i in self.vertex_faces[b])) - {a, b}
        return sorted((neighbors_a & neighbors_b) - shared)

    def _dissociate_vertex(self, vertex):
        # Split edge-disconnected face fans, following DissociateMesh(fill=False).
        indices = sorted(self.vertex_faces[vertex])
        if not indices:
            return
        used = set()
        replacements = {}

        while len(used) < len(indices):
            start = next(index for index in indices if index not in used)
            visited = {start}
            used.add(start)
            queue = deque([start])

            while queue:
                current = queue.popleft()
                face = self.faces[current]
                for neighbor in indices:
                    if neighbor not in used and len(set(face) & set(self.faces[neighbor])) == 2:
                        used.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(used) < len(indices):
                new_vertex = self._append_vertex(self.vertices[vertex], self.tags[vertex])
                for index in visited:
                    replacements[index] = tuple(new_vertex if v == vertex else v for v in self.faces[index])

        if replacements:
            self._replace_faces(replacements)

    def collapse(self, edge, ignore_pipe=False):
        a, b = edge
        touching = self.vertex_faces[a] | self.vertex_faces[b]
        shared = set(self.edge_faces[edge])
        con1 = [self.faces[i] for i in touching if i not in shared]

        # Same small tetrahedral-cluster guard as HandleShortEdge.m.
        if len(con1) == 2 and len(set().union(*con1)) == 4 and sum(a in face for face in con1) == 1 and sum(b in face for face in con1) == 1:
            self.tagged_edges.add(edge)
            return "rejected"

        if not ignore_pipe and self._pipe_candidates(edge):
            return "pipe"

        midpoint = (self.vertices[a] + self.vertices[b]) / 2
        replacements = {index: None for index in shared}
        for index in touching - shared:
            replacements[index] = tuple(a if vertex == b else vertex for vertex in self.faces[index])

        self.vertices[a] = midpoint
        self.tags[a] |= self.tags[b]
        # Update the modified neighborhood in one pass, including moved edges.
        self._replace_faces(replacements, moved_vertices={a})
        self._dissociate_vertex(a)
        return "collapsed"

    def to_arrays(self):
        active_faces = np.asarray([face for face in self.faces if face is not None], dtype=np.int64).reshape(-1, 3)
        used = np.zeros(len(self.vertices), dtype=bool)
        if len(active_faces):
            used[active_faces.ravel()] = True

        mapping = np.full(len(used), -1, dtype=np.int64)
        mapping[used] = np.arange(np.count_nonzero(used))
        vertices = np.asarray([self.vertices[i] for i in np.flatnonzero(used)], dtype=np.float64).reshape(-1, 3)
        tags = np.asarray(self.tags, dtype=np.uint8)[used]
        k1 = np.asarray(self.k1)[used]
        k2 = np.asarray(self.k2)[used]
        faces = mapping[active_faces]
        edge_tags = {tuple(sorted(map(int, mapping[list(edge)]))) for edge in self.tagged_edges if np.all(mapping[list(edge)] >= 0)}
        return vertices, faces, tags, k1, k2, edge_tags, mapping

@dataclass
class Surface:
    mesh: trimesh.Trimesh

    _edge_tagging: np.ndarray = field(init=False)
    _k1: np.ndarray = field(init=False)
    _k2: np.ndarray = field(init=False)
    _boundaries: np.ndarray = field(init=False)
    _progress: Progress | None = field(init=False, default=None, repr=False)
    _stage_task: int | None = field(init=False, default=None, repr=False)
    _stage_initial: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._remove_small_clusters()
        domain_boundaries = self.mesh.metadata.get("domain boundaries")
        self._boundaries = np.asarray(domain_boundaries if domain_boundaries is not None else self.mesh.bounds, dtype=float).copy()
        self._set_boundaries()

        n_vertices = len(self.mesh.vertices)
        n_edges = len(self.mesh.edges_unique)

        self._edge_tagging = np.zeros(n_edges, dtype=bool)
        self._k1 = np.zeros(n_vertices, dtype=float)
        self._k2 = np.zeros(n_vertices, dtype=float)

    def _set_boundaries(self) -> None:
        vertices = np.asarray(self.mesh.vertices)
        bounds = self._boundaries
        tolerance = 0.1 * float(self.mesh.metadata.get("voxel_size", 0.0))

        tags = np.zeros(len(vertices), dtype=np.uint8)
        for axis in range(3):
            tags |= (np.isclose(vertices[:, axis], bounds[0, axis], rtol=0, atol=tolerance).astype(np.uint8) << (2 * axis))
            tags |= (np.isclose(vertices[:, axis], bounds[1, axis], rtol=0, atol=tolerance).astype(np.uint8) << (2 * axis + 1))

        self.mesh.vertex_attributes["boundary_tag"] = tags

    def _remove_small_clusters(self) -> None:
        n_faces = len(self.mesh.faces)
        if not n_faces:
            return

        threshold = max(n_faces * 0.001, 4)

        components = trimesh.graph.connected_components(self.mesh.face_adjacency, nodes=np.arange(n_faces))

        keep = np.ones(n_faces, dtype=bool)

        for component in components:
            if len(component) <= threshold:
                keep[component] = False

        if not np.all(keep):
            # Keep curvature arrays in sync when called after initialization.
            old_faces = np.asarray(self.mesh.faces).copy()
            referenced = np.zeros(len(self.mesh.vertices), dtype=bool)
            referenced[old_faces[keep].ravel()] = True

            self.mesh.update_faces(keep)
            self.mesh.remove_unreferenced_vertices()

            if hasattr(self, "_k1"):
                self._k1 = self._k1[referenced]
                self._k2 = self._k2[referenced]

            if hasattr(self, "_edge_tagging"):
                self._edge_tagging = np.zeros(len(self.mesh.edges_unique), dtype=bool)

    def _commit(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tags: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray,
        *,
        old_edges: np.ndarray | None = None,
        old_edge_tags: np.ndarray | None = None,
    ) -> None:
        self.mesh.vertices = vertices
        self.mesh.faces = faces
        self.mesh.vertex_attributes["boundary_tag"] = tags
        self._k1 = k1
        self._k2 = k2

        if old_edges is None:
            self._edge_tagging = np.zeros(len(self.mesh.edges_unique), dtype=bool)
        else:
            # Preserve tags on surviving *unchanged* edges; new edges untagged.
            tagged = {tuple(edge) for edge, flag in zip(old_edges, old_edge_tags) if flag}
            self._edge_tagging = np.array([tuple(edge) in tagged for edge in self.mesh.edges_unique], dtype=bool)

    def _find_pipe_cycle(self, selection: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        # MATLAB DissociateMesh.m (Fill=true): search a complete face cycle.
        faces = np.asarray(self.mesh.faces)
        indices = np.flatnonzero(np.any(np.isin(faces, selection), axis=1))
        surrounding_faces = faces[indices]
        used = np.zeros(len(indices), dtype=bool)
        selected_vertices = np.asarray(selection, dtype=np.intp)

        while np.any(~used):
            start = int(np.flatnonzero(~used)[0])
            used[start] = True
            cycle = np.zeros(len(indices), dtype=bool)
            cycle[start] = True
            analysed_faces = [start]
            loop_found = True

            while analysed_faces:
                current = analysed_faces.pop(0)

                # Find edge-adjacent faces without crossing the empty triangle.
                similar = np.isin(surrounding_faces, surrounding_faces[current])
                adjacent = np.flatnonzero((similar.sum(axis=1) == 2) & ((similar & np.isin(surrounding_faces, selected_vertices)).sum(axis=1) != 2))

                # MATLAB tests adjacency before discarding already-used faces.
                if len(adjacent) < 2:
                    loop_found = False

                adjacent = adjacent[~used[adjacent]]
                analysed_faces.extend(adjacent.tolist())
                used[adjacent] = True
                cycle[adjacent] = True

            if loop_found and np.any((np.isin(surrounding_faces, selected_vertices).sum(axis=1) == 2) & cycle):
                return indices, surrounding_faces, cycle

        return None

    def _dissociate_vertices(self, vertex_indices, fill: bool = False) -> bool:
        vertices = np.asarray(self.mesh.vertices).copy()
        faces = np.asarray(self.mesh.faces).copy()
        tags = np.asarray(self.mesh.vertex_attributes["boundary_tag"]).copy()
        k1 = self._k1.copy()
        k2 = self._k2.copy()
        old_edges = self.mesh.edges_unique.copy()
        old_tags = self._edge_tagging.copy()
        selection = [int(vertex) for vertex in np.atleast_1d(vertex_indices)]

        if fill:
            if len(selection) != 3:
                raise ValueError("Filling a pipe requires exactly three vertices.")

            result = self._find_pipe_cycle(selection)
            if result is None:
                return False

            indices, surrounding_faces, cycle = result

            # Find vertex ordering for consistent face normals (MATLAB).
            connected = surrounding_faces[(np.isin(surrounding_faces, selection).sum(axis=1) == 2) & cycle][0]
            ordered = connected.copy()
            ordered[~np.isin(ordered, selection)] = [vertex for vertex in selection if vertex not in ordered]
            ordered = ordered[[1, 0, 2]]

            new_vertex = len(vertices)
            vertices = np.vstack((vertices, vertices[ordered]))
            tags = np.append(tags, tags[ordered])
            k1 = np.append(k1, np.zeros(3))
            k2 = np.append(k2, np.zeros(3))

            # Cap the original opening and the detached side.
            faces = np.vstack((faces, ordered[[0, 2, 1]], np.arange(new_vertex, new_vertex + 3)))
            selected_faces = indices[cycle]

            for offset, vertex in enumerate(ordered):
                selected = faces[selected_faces].copy()
                selected[selected == vertex] = new_vertex + offset
                faces[selected_faces] = selected

        else:
            remaining = selection.copy()

            for vertex in selection:
                # Find surrounding faces.
                indices = np.flatnonzero(np.any(faces == vertex, axis=1))
                surrounding_faces = faces[indices].copy()
                used = np.zeros(len(indices), dtype=bool)

                while not np.all(used):
                    start = int(np.flatnonzero(~used)[0])
                    used[start] = True
                    cycle = np.zeros(len(indices), dtype=bool)
                    cycle[start] = True
                    analysed_faces = [start]

                    while analysed_faces:
                        current = analysed_faces.pop(0)
                        similar = np.isin(surrounding_faces, surrounding_faces[current])
                        adjacent = np.flatnonzero((similar.sum(axis=1) == 2) & ((similar & np.isin(surrounding_faces, remaining)).sum(axis=1) != 2))
                        adjacent = adjacent[~used[adjacent]]
                        analysed_faces.extend(adjacent.tolist())
                        used[adjacent] = True
                        cycle[adjacent] = True

                    # If multiple face cycles exist, separate them.
                    if not np.all(used):
                        new_vertex = len(vertices)
                        vertices = np.vstack((vertices, vertices[vertex]))
                        tags = np.append(tags, tags[vertex])
                        k1 = np.append(k1, 0.0)
                        k2 = np.append(k2, 0.0)

                        # Rereference the current vertex in selected faces.
                        selected_faces = indices[cycle]
                        selected = faces[selected_faces].copy()
                        selected[selected == vertex] = new_vertex
                        faces[selected_faces] = selected

                remaining.pop(0)

        self._commit(vertices, faces, tags, k1, k2, old_edges=old_edges, old_edge_tags=old_tags)
        return True

    def _dissociate_if_pipe(self, edge_index: int) -> bool:
        edges = self.mesh.edges_unique
        v1, v2 = map(int, edges[edge_index])
        faces = np.asarray(self.mesh.faces)

        # Find faces containing both endpoints.
        shared = np.flatnonzero(np.isin(faces, [v1, v2]).sum(axis=1) == 2)
        if len(shared) != 2:
            return False

        # Find third vertices completing empty triangles (DissociateIfPipe.m).
        connected_edges = edges[np.any(np.isin(edges, [v1, v2]), axis=1)]
        connected_vertices = connected_edges.ravel()
        connected_vertices = connected_vertices[~np.isin(connected_vertices, faces[shared])]

        # FindSecondOccurenceIndices: duplicates excluding first occurrence.
        seen = set()
        candidates = []
        for vertex in connected_vertices:
            vertex = int(vertex)
            if vertex in seen and vertex not in candidates:
                candidates.append(vertex)
            seen.add(vertex)

        if not candidates:
            return False

        # The source uses the first candidate. Validate before editing: some
        # non-manifold neighborhoods do not contain a closed detachable cycle.
        return self._dissociate_vertices([v1, v2, candidates[0]], fill=True)

    def _handle_short_edge(self, edge_index: int) -> bool:
        v1, v2 = map(int, self.mesh.edges_unique[edge_index])
        faces = np.asarray(self.mesh.faces)

        counts = np.isin(faces, [v1, v2]).sum(axis=1)
        con1 = faces[counts == 1]

        # MATLAB's tetrahedral small-cluster guard.
        if (
            len(con1) == 2
            and len(np.unique(con1)) == 4
            and np.count_nonzero(con1 == v1) == 1
            and np.count_nonzero(con1 == v2) == 1
        ):
            self._edge_tagging[edge_index] = True
            return False

        # Dissociate a pipe-like configuration before collapsing.
        if self._dissociate_if_pipe(edge_index):
            return True  # Changed topology; no edge was collapsed.

        old_edges = self.mesh.edges_unique.copy()
        old_tag = self._edge_tagging.copy()

        # Collapse the edge to its midpoint.
        vertices = np.asarray(self.mesh.vertices).copy()
        vertices[v1] = (vertices[v1] + vertices[v2]) / 2
        new_faces = faces[counts != 2].copy()
        new_faces[new_faces == v2] = v1

        # Combine boundary tags.
        tags = np.asarray(self.mesh.vertex_attributes["boundary_tag"]).copy()
        tags[v1] |= tags[v2]

        # Identify and remove unreferenced vertices.
        referenced = np.zeros(len(vertices), dtype=bool)
        if len(new_faces):
            referenced[new_faces.ravel()] = True

        inverse = np.full(len(vertices), -1, dtype=np.intp)
        inverse[referenced] = np.arange(np.count_nonzero(referenced))
        surviving_index = int(inverse[v1]) if referenced[v1] else -1

        # Map surviving previously tagged edges by old vertex indices.
        mapped_edges = inverse[old_edges.copy()]
        remapped_flags = old_tag & np.all(mapped_edges >= 0, axis=1)
        mapped_edges = np.sort(mapped_edges[remapped_flags], axis=1)

        self._commit(
            vertices[referenced],
            inverse[new_faces],
            tags[referenced],
            self._k1[referenced],
            self._k2[referenced],
            old_edges=mapped_edges,
            old_edge_tags=np.ones(len(mapped_edges), dtype=bool),
        )

        if surviving_index >= 0:
            self._dissociate_vertices([surviving_index], fill=False)

        return True

    def _handle_long_edge(self, edge_index: int) -> None:
        v1, v2 = map(int, self.mesh.edges_unique[edge_index])
        faces = np.asarray(self.mesh.faces)

        # Find faces containing both endpoints.
        shared = np.isin(faces, [v1, v2]).sum(axis=1) == 2
        if not np.any(shared):
            raise ValueError("Selected edge has no incident faces.")

        # Create the midpoint vertex and split surrounding faces.
        new_vertex = len(self.mesh.vertices)
        new_faces = faces[shared].copy()
        new_faces[new_faces == v1] = new_vertex

        updated = faces.copy()
        replacement = updated[shared].copy()
        replacement[replacement == v2] = new_vertex
        updated[shared] = replacement

        # Update geometry and connectivity.
        vertices = np.vstack(
            (self.mesh.vertices, (self.mesh.vertices[v1] + self.mesh.vertices[v2]) / 2)
        )
        # Propagate boundary tagging and initialize new curvature values.
        tags = np.asarray(self.mesh.vertex_attributes["boundary_tag"])
        self._commit(
            vertices,
            np.vstack((updated, new_faces)),
            np.r_[tags, np.uint8(tags[v1] & tags[v2])],
            np.r_[self._k1, 0.0],
            np.r_[self._k2, 0.0],
        )

    def _back_on_boundary(self) -> None:
        tags = np.asarray(self.mesh.vertex_attributes["boundary_tag"], dtype=np.uint8)
        vertices = np.asarray(self.mesh.vertices).copy()

        for axis in range(3):
            low = np.uint8(1 << (2 * axis))
            high = np.uint8(1 << (2 * axis + 1))

            if np.any(((tags & low) != 0) & ((tags & high) != 0)):
                raise ValueError(f"Impossible boundary tagging on axis {axis}.")

            vertices[(tags & low) != 0, axis] = self._boundaries[0, axis]
            vertices[(tags & high) != 0, axis] = self._boundaries[1, axis]

        self.mesh.vertices = vertices

    def _merge_split_by_length(self, factor: float = 2.0) -> None:
        if factor <= 1:
            raise ValueError("Factor must be greater than 1.")

        if not len(self.mesh.edges_unique):
            return

        topology = _DynamicTopology(self)
        mean = topology.mean_length()
        initial = topology.remaining(mean / factor, mean * factor)
        operation_count = 0

        # Reuse the smoothing display when available.
        standalone = self._progress is None
        context = Progress(
            TextColumn("{task.description}"),
            BarColumn(
                complete_style="blue",
                finished_style="blue",
                pulse_style="blue",
            ),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) if standalone else nullcontext(self._progress)

        with context as progress:
            task = progress.add_task("Merging / splitting edges", total=max(initial, 1), unit="edges")

            while topology.lengths:
                mean = topology.mean_length()
                low, high = mean / factor, mean * factor

                # Collapse the shortest eligible edges.
                while (edge := topology.next_edge(True, low)) is not None:
                    result = topology.collapse(edge)

                    if result == "pipe":
                        # Compact vertex IDs before the MATLAB-style pipe procedure.
                        mapping = self._load_topology(topology)
                        compacted_edge = mapping[list(edge)]
                        index = np.flatnonzero(np.all(self.mesh.edges_unique == compacted_edge, axis=1))

                        if len(index) and self._dissociate_if_pipe(int(index[0])):
                            topology = _DynamicTopology(self)
                        else:
                            # The existing topology still uses the original vertex IDs.
                            topology.collapse(edge, ignore_pipe=True)

                    operation_count += 1
                    if operation_count % 100 == 0:
                        remaining = topology.remaining(low, high)
                        progress.update(task, completed=int(np.clip(initial - remaining, 0, max(initial, 1))))

                # Split the longest edges.
                pass_splits = 0

                while (edge := topology.next_edge(False, high)) is not None:
                    topology.split(edge)
                    pass_splits += 1
                    operation_count += 1

                    if operation_count % 100 == 0:
                        remaining = topology.remaining(low, high)
                        progress.update(task, completed=int(np.clip(initial - remaining, 0, max(initial, 1))))

                if not pass_splits:
                    break

            self._load_topology(topology)
            self._remove_small_clusters()
            self._back_on_boundary()
            progress.update(task, completed=max(initial, 1))
            progress.remove_task(task)

    def _load_topology(self, topology: _DynamicTopology) -> np.ndarray:
        vertices, faces, tags, k1, k2, tagged_edges, mapping = topology.to_arrays()

        self.mesh.vertices = vertices
        self.mesh.faces = faces
        self.mesh.vertex_attributes["boundary_tag"] = tags
        self._k1 = k1
        self._k2 = k2
        self._edge_tagging = np.array(
            [tuple(map(int, edge)) in tagged_edges for edge in self.mesh.edges_unique],
            dtype=bool,
        )

        return mapping

    def _stage_progress(self, remaining: int) -> None:
        if self._progress is None or self._stage_task is None:
            return

        total = max(self._stage_initial, 1)
        completed = int(np.clip(self._stage_initial - remaining, 0, total))
        self._progress.update(self._stage_task, completed=completed)

    def _run_stage(self, name: str, method, *args) -> None:
        self._stage_initial = 0
        self._stage_task = self._progress.add_task(name, total=1, unit="resolved")
        try:
            method(*args)
            self._progress.update(self._stage_task, completed=max(self._stage_initial, 1))
        finally:
            self._progress.remove_task(self._stage_task)
            self._stage_task = None

    def _compute_curvatures(self) -> None:
        # Compute mixed Voronoi areas, Gaussian curvature and mean curvature.
        faces = np.asarray(self.mesh.faces)
        vertices = np.asarray(self.mesh.vertices)
        n_vertices = len(vertices)

        if not len(faces):
            self._k1 = np.zeros(n_vertices)
            self._k2 = np.zeros(n_vertices)
            return

        triangles = vertices[faces]
        edge_next = np.roll(triangles, -1, axis=1) - triangles
        edge_prev = np.roll(triangles, 1, axis=1) - triangles

        cross = np.cross(edge_next, edge_prev)
        double_areas = np.linalg.norm(cross, axis=2)
        dot = np.sum(edge_next * edge_prev, axis=2)
        angles = np.arctan2(double_areas, dot)
        cotangents = np.divide(dot, double_areas, out=np.zeros_like(dot), where=double_areas > 0)

        areas = double_areas[:, 0] / 2
        obtuse_angles = angles > np.pi / 2
        obtuse_faces = np.any(obtuse_angles, axis=1)

        length_next = np.sum(edge_next ** 2, axis=2)
        length_prev = np.sum(edge_prev ** 2, axis=2)

        # Mixed Voronoi cell area: acute cotangent formula, obtuse area splitting.
        mixed = (np.roll(cotangents, -1, axis=1) * length_prev + np.roll(cotangents, 1, axis=1) * length_next) / 8
        mixed = np.where(obtuse_faces[:, None], np.where(obtuse_angles, areas[:, None] / 2, areas[:, None] / 4), mixed)

        vertex_area = np.zeros(n_vertices, dtype=np.float64)
        angle_sum = np.zeros(n_vertices, dtype=np.float64)
        np.add.at(vertex_area, faces.ravel(), mixed.ravel())
        np.add.at(angle_sum, faces.ravel(), angles.ravel())

        # Cotangent Laplacian, accumulated by triangle corner.
        laplace_corner = (np.roll(cotangents, -1, axis=1)[:, :, None] * edge_prev + np.roll(cotangents, 1, axis=1)[:, :, None] * edge_next)
        laplace = np.zeros((n_vertices, 3), dtype=np.float64)
        np.add.at(laplace, faces.ravel(), laplace_corner.reshape(-1, 3))
        laplace = np.divide(laplace, 2 * vertex_area[:, None], out=np.zeros_like(laplace), where=vertex_area[:, None] > 0)

        gaussian = np.divide(2 * np.pi - angle_sum, vertex_area, out=np.zeros(n_vertices), where=vertex_area > 0)
        mean_absolute = np.linalg.norm(laplace, axis=1) / 2

        # As in MATLAB, boundary vertices have zero curvature.
        boundary = self.mesh.vertex_attributes["boundary_tag"] != 0
        gaussian[boundary] = 0
        mean_absolute[boundary] = 0

        normals = self.mesh.vertex_normals
        mean = np.where(np.sum(normals * -laplace, axis=1) < 0, -mean_absolute, mean_absolute)
        roots = np.sqrt(np.maximum(mean ** 2 - gaussian, 0))

        k1 = mean + roots
        k2 = mean - roots
        exchange = np.abs(k1) < np.abs(k2)
        self._k1 = np.where(exchange, k2, k1)
        self._k2 = np.where(exchange, k1, k2)

    def _extend_mask_to_boundaries(self, mask: np.ndarray) -> np.ndarray:
        # MATLAB copies the mask of the nearest nonboundary vertex to boundary vertices.
        mask = np.asarray(mask).copy()
        tags = self.mesh.vertex_attributes["boundary_tag"]
        boundary = np.flatnonzero(tags != 0)
        interior = np.flatnonzero(tags == 0)

        if not len(boundary) or not len(interior):
            return mask

        faces = np.asarray(self.mesh.faces)
        near_boundary = np.any(np.isin(faces, boundary), axis=1)
        candidates = np.unique(faces[near_boundary])
        candidates = candidates[tags[candidates] == 0]

        if not len(candidates):
            return mask

        from scipy.spatial import cKDTree

        nearest = cKDTree(np.asarray(self.mesh.vertices)[candidates]).query(np.asarray(self.mesh.vertices)[boundary])[1]
        mask[boundary] = mask[candidates[nearest]]
        return mask

    def _smooth_mesh(self, mask: np.ndarray) -> None:
        # MATLAB SmoothMesh with method='Average' and one iteration.
        vertices = np.asarray(self.mesh.vertices)
        edges = self.mesh.edges_unique
        totals = vertices.copy()
        counts = np.ones(len(vertices), dtype=np.int64)

        np.add.at(totals, edges[:, 0], vertices[edges[:, 1]])
        np.add.at(totals, edges[:, 1], vertices[edges[:, 0]])
        np.add.at(counts, edges[:, 0], 1)
        np.add.at(counts, edges[:, 1], 1)

        updated = vertices.copy()
        updated[mask] = (totals / counts[:, None])[mask]
        self.mesh.vertices = updated

    def _swell_mesh(self, mask: np.ndarray, factor: float) -> None:
        # Move selected vertices along their area-weighted normals.
        lengths = self.mesh.edges_unique_length
        if not len(lengths):
            return

        normals = self.mesh.vertex_normals
        self.mesh.vertices = np.asarray(self.mesh.vertices) + factor * lengths.mean() * normals * np.asarray(mask)[:, None]

    def _correct_boundary_tagging(self) -> None:
        # Drop boundary bits not shared by at least one adjacent vertex.
        tags = np.asarray(self.mesh.vertex_attributes["boundary_tag"]).copy()
        edges = self.mesh.edges_unique
        neighbors = np.zeros(len(tags), dtype=np.uint8)
        np.bitwise_or.at(neighbors, edges[:, 0], tags[edges[:, 1]])
        np.bitwise_or.at(neighbors, edges[:, 1], tags[edges[:, 0]])
        self.mesh.vertex_attributes["boundary_tag"] = tags & neighbors

    def _smooth_peaks(self, k2_max: float) -> None:
        # MATLAB SmartSmoothPeeks.m.
        while True:
            self._compute_curvatures()
            mask = (np.abs(self._k2) > k2_max) & ((self._k1 > 0) == (self._k2 > 0))
            mask = self._extend_mask_to_boundaries(mask)

            count = int(np.count_nonzero(mask))
            if self._stage_initial == 0 and count:
                self._stage_initial = count
                if self._stage_task is not None:
                    self._progress.update(self._stage_task, total=count)
            self._stage_progress(count)

            if not count:
                return

            self._smooth_mesh(mask)
            self._back_on_boundary()
            self._merge_split_by_length()
            self._compute_curvatures()
            self._correct_boundary_tagging()

    def _swell_saddles(self, k2_max: float) -> None:
        # MATLAB SmartSwellSadel.m.
        while True:
            self._compute_curvatures()
            saddle = (self._k1 > 0) != (self._k2 > 0)
            mask1 = self._extend_mask_to_boundaries((self._k2 > k2_max) & saddle)
            mask2 = self._extend_mask_to_boundaries((self._k2 < -k2_max) & saddle)
            overlapping = mask1 & mask2
            mask1 &= ~overlapping
            mask2 &= ~overlapping

            count = int(np.count_nonzero(mask1 | mask2))
            if self._stage_initial == 0 and count:
                self._stage_initial = count
                if self._stage_task is not None:
                    self._progress.update(self._stage_task, total=count)
            self._stage_progress(count)

            if not count:
                return

            self._swell_mesh(mask1, -0.1)
            self._swell_mesh(mask2, 0.1)
            self._back_on_boundary()
            self._merge_split_by_length()

    def _surround_swelling(self, k1_max: float, k2_max: float) -> None:
        # MATLAB SurroundSwelling.m: swell neighbors and smooth problem vertices.
        while True:
            self._compute_curvatures()
            problem = (np.abs(self._k1) > k1_max) & (np.abs(self._k2) < k2_max)
            problem = self._extend_mask_to_boundaries(problem)

            count = int(np.count_nonzero(problem))
            if self._stage_initial == 0 and count:
                self._stage_initial = count
                if self._stage_task is not None:
                    self._progress.update(self._stage_task, total=count)
            self._stage_progress(count)

            if not count:
                return

            faces = np.asarray(self.mesh.faces)
            curved = np.flatnonzero(problem)
            mask = np.zeros(len(self.mesh.vertices), dtype=np.float64)

            for vertex in curved:
                surrounding = np.unique(faces[np.any(faces == vertex, axis=1)])
                surrounding = surrounding[surrounding != vertex]
                mask[surrounding] += self._k1[vertex]

            mask = self._extend_mask_to_boundaries(np.sign(mask))
            self._swell_mesh(mask, 0.1)
            self._smooth_mesh(problem)
            self._back_on_boundary()
            self._merge_split_by_length()

    def locally_smooth(self, k1_max: float, k2_max: float) -> bool:
        if k1_max <= 0 or k2_max <= 0:
            raise ValueError("Curvature limits must be positive.")

        self._compute_curvatures()

        with Progress(
            TextColumn("{task.description}"),
            CustomBarColumn(
                complete_style="blue",
                finished_style="blue",
                pulse_style="blue",
            ),
            TextColumn("{task.completed:,.0f} {task.fields[unit]}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            self._progress = progress
            iteration_task = progress.add_task("Full smoothing iterations", total=None, unit="iterations")

            try:
                while np.any((np.abs(self._k1) > k1_max) | (np.abs(self._k2) > k2_max)):
                    previous_vertices = np.asarray(self.mesh.vertices).copy()
                    previous_faces = np.asarray(self.mesh.faces).copy()

                    # MATLAB smoothing order: peaks, saddles, surrounding swelling.
                    self._run_stage("Smoothing curvature peaks", self._smooth_peaks, k2_max)
                    self._run_stage("Swelling saddle regions", self._swell_saddles, k2_max)
                    self._run_stage("Swelling surrounding regions", self._surround_swelling, k1_max, k2_max)

                    self._compute_curvatures()
                    progress.advance(iteration_task)

                    lengths = self.mesh.edges_unique_length
                    unchanged = np.array_equal(previous_faces, self.mesh.faces) and previous_vertices.shape == self.mesh.vertices.shape
                    if unchanged and len(lengths) and np.all(np.abs(previous_vertices - self.mesh.vertices) < 0.001 * lengths.min()):
                        console.print("[yellow]Smoothing stagnated; MATLAB's corrective pipe search is not yet translated.[/yellow]")
                        return False
            finally:
                self._progress = None
                self._stage_task = None

        iterations = int(progress.tasks[iteration_task].completed)
        console.print(
            f"[green]✔ Local smoothing converged after {iterations:,} full smoothing iterations.[/green]"
        )
        return True