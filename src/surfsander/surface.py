from dataclasses import dataclass, field

import numpy as np
import trimesh
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from surfsander.common import console

@dataclass
class Surface:
    mesh: trimesh.Trimesh

    _edge_tagging: np.ndarray = field(init=False)
    _k1: np.ndarray = field(init=False)
    _k2: np.ndarray = field(init=False)
    _boundaries: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._remove_small_clusters()
        self._boundaries = self.mesh.bounds.copy()
        self._set_boundaries()

        n_vertices = len(self.mesh.vertices)
        n_edges = len(self.mesh.edges_unique)

        self._edge_tagging = np.zeros(n_edges, dtype=bool)
        self._k1 = np.zeros(n_vertices, dtype=float)
        self._k2 = np.zeros(n_vertices, dtype=float)

    def _set_boundaries(self) -> None:
        vertices = np.asarray(self.mesh.vertices)
        bounds = self._boundaries

        tags = np.zeros(len(vertices), dtype=np.uint8)
        for axis in range(3):
            tags |= ((vertices[:, axis] == bounds[0, axis]).astype(np.uint8) << (2 * axis))
            tags |= ((vertices[:, axis] == bounds[1, axis]).astype(np.uint8) << (2 * axis + 1))

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

        lengths = self.mesh.edges_unique_length
        mean = lengths.mean()

        initial = int(np.count_nonzero((lengths < mean / factor) | (lengths > mean * factor)))

        collapsed = 0
        split = 0

        with Progress(
            TextColumn("[cyan]Merging/splitting edges:"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Adjusting mesh", total=max(initial, 1))

            while True:
                lengths = self.mesh.edges_unique_length
                mean = lengths.mean()
                low, high = mean / factor, mean * factor

                # Collapse short edges.
                while True:
                    lengths = self.mesh.edges_unique_length
                    options = np.flatnonzero((lengths < low) & ~self._edge_tagging)

                    if not len(options):
                        break

                    candidate = int(options[np.argmin(lengths[options])])
                    before = len(self.mesh.vertices)
                    self._handle_short_edge(candidate)

                    if len(self.mesh.vertices) < before:
                        collapsed += 1

                    now = self.mesh.edges_unique_length
                    remaining = np.count_nonzero((now < low) | (now > high))
                    progress.update(task, completed=np.clip(initial - remaining, 0, max(initial, 1)))

                # Split long edges.
                pass_splits = 0

                while True:
                    lengths = self.mesh.edges_unique_length
                    candidate = int(np.argmax(lengths))

                    if lengths[candidate] <= high:
                        break

                    self._handle_long_edge(candidate)
                    split += 1
                    pass_splits += 1

                    now = self.mesh.edges_unique_length
                    remaining = np.count_nonzero((now < low) | (now > high))
                    progress.update(task, completed=np.clip(initial - remaining, 0, max(initial, 1)))

                # Stop when no more edges were split.
                if not pass_splits:
                    break

            self._remove_small_clusters()
            self._back_on_boundary()
            progress.update(task, completed=max(initial, 1))

        console.print(
            f"[green]✔[/green] Mesh adjusted: "
            f"{collapsed:,} collapsed, {split:,} split."
        )

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

    def _smooth_peaks(self, k2_max: float, max_iterations: int) -> bool:
        # MATLAB SmartSmoothPeeks.m.
        for _ in range(max_iterations):
            self._compute_curvatures()
            mask = (np.abs(self._k2) > k2_max) & ((self._k1 > 0) == (self._k2 > 0))
            mask = self._extend_mask_to_boundaries(mask)

            if not np.any(mask):
                return True

            self._smooth_mesh(mask)
            self._back_on_boundary()
            self._merge_split_by_length()
            self._compute_curvatures()
            self._correct_boundary_tagging()

        return False

    def _swell_saddles(self, k2_max: float, max_iterations: int) -> bool:
        # MATLAB SmartSwellSadel.m.
        for _ in range(max_iterations):
            self._compute_curvatures()
            saddle = (self._k1 > 0) != (self._k2 > 0)
            mask1 = self._extend_mask_to_boundaries((self._k2 > k2_max) & saddle)
            mask2 = self._extend_mask_to_boundaries((self._k2 < -k2_max) & saddle)
            overlapping = mask1 & mask2
            mask1 &= ~overlapping
            mask2 &= ~overlapping

            if not np.any(mask1 | mask2):
                return True

            self._swell_mesh(mask1, -0.1)
            self._swell_mesh(mask2, 0.1)
            self._back_on_boundary()
            self._merge_split_by_length()

        return False

    def _surround_swelling(self, k1_max: float, k2_max: float, max_iterations: int) -> bool:
        # MATLAB SurroundSwelling.m: swell neighbors and smooth problem vertices.
        for _ in range(max_iterations):
            self._compute_curvatures()
            problem = (np.abs(self._k1) > k1_max) & (np.abs(self._k2) < k2_max)
            problem = self._extend_mask_to_boundaries(problem)

            if not np.any(problem):
                return True

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

        return False

    def locally_smooth(self, k1_max: float, k2_max: float, max_iterations: int = 100, max_stage_iterations: int = 100) -> bool:
        # Main smoothing loop corresponding to LocalSmoothing.m.
        if k1_max <= 0 or k2_max <= 0:
            raise ValueError("Curvature limits must be positive.")

        self._compute_curvatures()

        with Progress(
            TextColumn("[cyan]Curvature-driven smoothing:"),
            BarColumn(),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f} iterations"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Smoothing", total=max_iterations)

            for iteration in range(max_iterations):
                problem = (np.abs(self._k1) > k1_max) | (np.abs(self._k2) > k2_max)
                if not np.any(problem):
                    return True

                previous_vertices = np.asarray(self.mesh.vertices).copy()
                previous_faces = np.asarray(self.mesh.faces).copy()

                # MATLAB smoothing order: peaks, saddle swelling, surrounding swelling.
                if not self._smooth_peaks(k2_max, max_stage_iterations):
                    console.print("[yellow]Peak smoothing reached the iteration limit.[/yellow]")
                    return False

                if not self._swell_saddles(k2_max, max_stage_iterations):
                    console.print("[yellow]Saddle swelling reached the iteration limit.[/yellow]")
                    return False

                if not self._surround_swelling(k1_max, k2_max, max_stage_iterations):
                    console.print("[yellow]Surround swelling reached the iteration limit.[/yellow]")
                    return False

                self._compute_curvatures()
                progress.advance(task)

                # MATLAB invokes CorrectiveMeasure if the mesh ceases to change.
                lengths = self.mesh.edges_unique_length
                unchanged = (np.array_equal(previous_faces, self.mesh.faces) and previous_vertices.shape == self.mesh.vertices.shape)
                if unchanged and len(lengths) and np.all(np.abs(previous_vertices - self.mesh.vertices) < 0.001 * lengths.min()):
                    console.print("[yellow]Smoothing stagnated; MATLAB's corrective pipe search is not yet translated.[/yellow]")
                    return False

        self._compute_curvatures()
        return not np.any((np.abs(self._k1) > k1_max) | (np.abs(self._k2) > k2_max))
