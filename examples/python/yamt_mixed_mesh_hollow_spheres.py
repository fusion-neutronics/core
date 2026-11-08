"""
Mixed mesh: tet-meshed + surface-only nested hollow spheres.

Pipeline:  CadQuery (geometry) -> cad_to_yamc (surface mesh + selective tet meshing)
           -> Arrow IPC -> YAMC (transport) -> VTK HDF5 (ParaView)

Demonstrates YAMC's ability to mix tet-meshed and surface-only volumes in a
single simulation.  Transport always uses surface BVHs for ray-tracing;
tet walking is only engaged for tally scoring.

Geometry -- three concentric spheres at r = 5, 10, 15 cm:
  * Inner shell (radius=5 -> radius=10): full tet mesh  -> per-tet flux tally
  * Outer shell (radius=10 -> radius=15): surface-only  -> material-level flux tally
  * Vacuum boundary on the outer sphere (radius=15)
  * Point source at radius=6 (inside the inner shell)

Requirements:
    pip install yamc[cad] wildmeshing h5py numpy
"""

import os
import tempfile

import cadquery as cq
import h5py
import numpy as np
import yamc

from yamc.cad import CadToYamc

# ---------------------------------------------------------------------------
# 1. CadQuery -- build two concentric hollow shells
# ---------------------------------------------------------------------------

r_inner, r_mid, r_outer = 5.0, 10.0, 15.0  # cm

sphere_inner = cq.Workplane("XY").sphere(r_inner)
sphere_mid = cq.Workplane("XY").sphere(r_mid)
sphere_outer = cq.Workplane("XY").sphere(r_outer)

# Hollow shells via boolean cuts
inner_shell = sphere_mid.cut(sphere_inner)   # radius=5 to radius=10
outer_shell = sphere_outer.cut(sphere_mid)   # radius=10 to radius=15

# Build assembly with named parts
assy = cq.Assembly()
assy.add(inner_shell, name="inner")
assy.add(outer_shell, name="outer")

# ---------------------------------------------------------------------------
# 2. cad_to_yamc -- surface mesh + selective tet mesh + Arrow export
# ---------------------------------------------------------------------------

c2y = CadToYamc()
c2y.add_cadquery_object(assy, material_tags="assembly_names")

# Coarse surfaces for outer shell, fine surfaces + tets for inner shell
mesh = c2y.mesh(tet_volumes=["inner"], target_edge_length=2.0)
print(f"Surface mesh: {len(mesh.vertices)} vertices, "
      f"{len(mesh.triangles)} triangles")
for solid_id, (tv, tt) in c2y._tet_data.items():
    print(f"  Volume {solid_id}: {len(tv)} tet vertices, {len(tt)} tetrahedra")

# Export to Arrow IPC
output_dir = os.path.join(tempfile.gettempdir(), "yamc_mixed_mesh")
arrow_path = os.path.join(output_dir, "model.arrow")
c2y.to_arrow(arrow_path)  # creates output_dir if needed
print(f"Arrow file written to {arrow_path}")

# ---------------------------------------------------------------------------
# 3. YAMC -- define materials, build model, run transport
# ---------------------------------------------------------------------------

# Inner shell: lithium-6
inner_mat = yamc.Material(
    composition={"Li6": 1.0},
    density=0.534,
    name="inner",
)
inner_mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})

# Outer shell: beryllium
outer_mat = yamc.Material(
    composition={"Be9": 1.0},
    density=1.85,
    name="outer",
)
outer_mat.read_nuclear_data({"Be9": "tests/Be9.arrow"})

# Build mesh geometry from Arrow file
materials = {"inner": inner_mat, "outer": outer_mat}
mesh_geom = yamc.MeshGeometry(arrow_path, materials)
print(mesh_geom)

# Source: 14.1 MeV point source at radius=6 (inside the inner shell)
source = yamc.NeutronSource(
    energy=14.06e6,
    position=(6, 0, 0),
)

# --- Per-tet flux tally on the inner shell (tet-meshed) ---
tet_tally = yamc.Tally(scores=["flux"], unstructured_mesh=(mesh_geom, "inner"), name="inner_tet_flux")

# --- Material-level flux tally on the outer shell (surface-only) ---
outer_tally = yamc.Tally(scores=["flux"], materials=outer_mat, name="outer_flux")

model = yamc.Model(
    geometry=mesh_geom,
    tallies=[tet_tally, outer_tally],
    source=source,
)
results = model.simulate_transport(total_particles=25000, seed=42)

# ---------------------------------------------------------------------------
# 4. Post-process: print summary
# ---------------------------------------------------------------------------

tet_result = results[tet_tally]
tet_mean = np.array(tet_result.mean)
tet_std = np.array(tet_result.standard_deviation)
n_tets = len(tet_mean)
nonzero_count = np.count_nonzero(tet_mean)

print(f"\n--- Inner shell (tet-meshed, radius={r_inner}->{r_mid} cm) ---")
print(f"  Tetrahedra:          {n_tets}")
print(f"  Non-zero flux tets:  {nonzero_count}/{n_tets}")
print(f"  Total flux (sum):    {tet_mean.sum():.6e}")

outer_result = results[outer_tally]
print(f"\n--- Outer shell (surface-only, radius={r_mid}->{r_outer} cm) ---")
print(f"  Flux mean:           {outer_result.mean[0]:.6e}")
print(f"  Flux std dev:        {outer_result.standard_deviation[0]:.6e}")

# ---------------------------------------------------------------------------
# 5. Write inner-shell tet tally to VTK HDF5
# ---------------------------------------------------------------------------


def tet_to_vtkhdf(filename, tet_vertices, tet_connectivity, datasets, volume_normalization=True):
    """Write tet-mesh tally data to a VTK HDF5 file.

    Args:
        filename: Output path (should end in .vtkhdf)
        tet_vertices: array of vertex coordinates
        tet_connectivity: array of tet vertex indices (n_tets x 4)
        datasets: dict mapping dataset names to numpy arrays (one value per tet)
        volume_normalization: If True, divide values by tet volume
    """
    VTK_TETRA = 10
    POINTS_PER_TET = 4

    vertices = np.array(tet_vertices, dtype="float64")
    connectivity = np.array(tet_connectivity, dtype="int64").flatten()
    n_elements = len(tet_connectivity)
    n_points = len(vertices)

    offsets = np.arange(
        0, n_elements * POINTS_PER_TET + 1, POINTS_PER_TET, dtype="int64"
    )
    types = np.full(n_elements, VTK_TETRA, dtype="uint8")

    if volume_normalization:
        # Compute tet volumes from vertices and connectivity
        def tet_volume(v0, v1, v2, v3):
            return abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
        conn = np.array(tet_connectivity, dtype="int64")
        volumes = np.array([
            tet_volume(vertices[c[0]], vertices[c[1]], vertices[c[2]], vertices[c[3]])
            for c in conn
        ])

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 1], dtype="int64")
        ascii_type = "UnstructuredGrid".encode("ascii")
        root.attrs.create(
            "Type",
            ascii_type,
            dtype=h5py.string_dtype("ascii", len(ascii_type)),
        )

        root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype="int64"))
        root.create_dataset(
            "NumberOfCells", data=np.array([n_elements], dtype="int64")
        )
        root.create_dataset(
            "NumberOfConnectivityIds",
            data=np.array([len(connectivity)], dtype="int64"),
        )
        root.create_dataset("Points", data=vertices)
        root.create_dataset("Connectivity", data=connectivity)
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Types", data=types)

        cell_data = root.create_group("CellData")
        for name, data in datasets.items():
            values = np.array(data, dtype="float64")
            if volume_normalization:
                values = values / volumes
            cell_data.create_dataset(name, data=values)

    print(f"Wrote {filename}  ({n_elements} tets, {n_points} vertices)")


# ---------------------------------------------------------------------------
# 6. Write surface mesh geometry to VTK HDF5
# ---------------------------------------------------------------------------


def surface_to_vtkhdf(filename, mesh_geom, cell_data_dict=None):
    """Write the surface triangle mesh to a VTK HDF5 file.

    Produces a triangle-only unstructured grid that can be opened in
    ParaView to inspect the geometry.  An optional *cell_data_dict* maps
    dataset names to per-triangle arrays.

    A ``volume_id`` field is always written so that each triangle can be
    coloured by the volume it belongs to.

    Args:
        filename: Output path (should end in .vtkhdf)
        mesh_geom: yamc.MeshGeometry with surface mesh data
        cell_data_dict: Optional dict of dataset name -> per-triangle array
    """
    VTK_TRIANGLE = 5
    POINTS_PER_TRI = 3

    all_vertices = np.array(mesh_geom.vertices, dtype="float64")
    all_triangles = np.array(mesh_geom.triangles, dtype="int64")
    n_triangles = len(all_triangles)
    n_points = len(all_vertices)

    connectivity = all_triangles.flatten()
    offsets = np.arange(
        0, n_triangles * POINTS_PER_TRI + 1, POINTS_PER_TRI, dtype="int64"
    )
    types = np.full(n_triangles, VTK_TRIANGLE, dtype="uint8")

    # Build a volume_id label per triangle
    volume_id_arr = np.full(n_triangles, -1, dtype="int32")
    for vol_id in range(mesh_geom.num_volumes):
        tri_indices = mesh_geom.volume_triangle_indices(vol_id)
        for ti in tri_indices:
            volume_id_arr[ti] = vol_id

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 1], dtype="int64")
        ascii_type = "UnstructuredGrid".encode("ascii")
        root.attrs.create(
            "Type",
            ascii_type,
            dtype=h5py.string_dtype("ascii", len(ascii_type)),
        )

        root.create_dataset(
            "NumberOfPoints", data=np.array([n_points], dtype="int64")
        )
        root.create_dataset(
            "NumberOfCells", data=np.array([n_triangles], dtype="int64")
        )
        root.create_dataset(
            "NumberOfConnectivityIds",
            data=np.array([len(connectivity)], dtype="int64"),
        )
        root.create_dataset("Points", data=all_vertices)
        root.create_dataset("Connectivity", data=connectivity)
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Types", data=types)

        cell_data = root.create_group("CellData")
        cell_data.create_dataset("volume_id", data=volume_id_arr)

        if cell_data_dict:
            for name, data in cell_data_dict.items():
                cell_data.create_dataset(
                    name, data=np.array(data, dtype="float64")
                )

    print(f"Wrote {filename}  ({n_triangles} triangles, {n_points} vertices)")


# Get tet mesh data for the inner shell from cad_to_yamc
inner_solid_id = [sid for sid in c2y._tet_data][0]  # first tet-meshed volume
inner_tet_verts, inner_tet_conn = c2y._tet_data[inner_solid_id]

tet_vtkhdf_path = os.path.join(output_dir, "inner_shell_flux.vtkhdf")
tet_to_vtkhdf(
    tet_vtkhdf_path,
    inner_tet_verts,
    inner_tet_conn,
    datasets={
        "flux": tet_mean,
        "standard_deviation": tet_std,
    },
    volume_normalization=True,
)

surface_vtkhdf_path = os.path.join(output_dir, "surface_mesh.vtkhdf")
surface_to_vtkhdf(surface_vtkhdf_path, mesh_geom)

print("\nOpen in ParaView:")
print(f"  Tet tally:     paraview {tet_vtkhdf_path}")
print(f"  Surface mesh:  paraview {surface_vtkhdf_path}")
