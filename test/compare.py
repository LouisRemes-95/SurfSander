import numpy as np
import trimesh

a = trimesh.load("test/output/doi_voxelisation.ply", process=False)
b = trimesh.load("test/output/doi_voxelisation_working_version.ply", process=False)

print("Vertices:", len(a.vertices), len(b.vertices))
print("Faces:   ", len(a.faces), len(b.faces))

if a.vertices.shape == b.vertices.shape:
    print("Vertices identical:", np.array_equal(a.vertices, b.vertices))
    print("Max vertex difference:", np.max(np.abs(a.vertices - b.vertices)) if a.vertices.size else 0)

if a.faces.shape == b.faces.shape:
    print("Faces identical:", np.array_equal(a.faces, b.faces))