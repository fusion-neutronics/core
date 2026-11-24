# Coupling to a heat solve

This example couples yamc's nuclear heating to a steady heat conduction solve in
[FEniCSx](https://fenicsproject.org/), and feeds the temperature back into
yamc as the density of a liquid breeder. A sphere of PbLi sits in a steel box
with a 14.06 MeV point source at its centre:

1. yamc tallies the heating in every tet of the mesh.
2. FEniCSx solves for the temperature on the same tets, with the box's outer
   faces held at 600 K.
3. The PbLi's density at each tet's temperature goes back into yamc with
   ``Model.set_tet_density_factors``, and the loop repeats until the
   temperature stops changing.

Both codes run on one tet mesh, made from the CAD by
``MeshGeometry.from_cadquery``, so nothing is interpolated between them: each
tet carries one heating value, one temperature and one density factor. Lengths
are in cm throughout, including in the heat solve.

## Installing

yamc with its CAD extra, and FEniCSx with PETSc. FEniCSx is not on PyPI yet;
[these wheels](https://github.com/shimwell/wheels) install it with pip on
Linux, for Python 3.12 or newer and glibc 2.34 or newer (Ubuntu 22.04 on):

```bash
pip install "yamc[cad]"
pip install --extra-index-url https://shimwell.github.io/wheels mpich "fenics-dolfinx[petsc4py]"
```

## Materials

The PbLi density follows a linear fit in temperature. The material is built at
the wall temperature, and the density factors are relative to that.

```python
import time

import basix.ufl
import cadquery as cq
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

import yamc

yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"

T_WALL = 600.0  # K


def pbli_density(T):
    """PbLi density in g/cm3 at T in K (Mas de les Valls et al. 2008)."""
    return 10.52035 - 1.19051e-3 * T


breeder = yamc.Material(
    composition={"Pb": 0.83, "Li": 0.17}, density=pbli_density(T_WALL), temperature=T_WALL
)
steel = yamc.materials.pnnl.material("Steel, Stainless 316", temperature=T_WALL)
```

## Geometry and mesh

``from_cadquery`` imprints the assembly, meshes its surfaces and fills the
listed volumes with tets of about ``target_edge_length``, all in one call. The
materials are keyed by the assembly's part names. Both volumes are tet-meshed,
the breeder for its density feedback and the steel because the heat solve needs
every region the heat flows through.

```python
sphere = cq.Workplane("XY").sphere(10.0)
box = cq.Workplane("XY").box(30, 30, 30).cut(sphere)
assembly = cq.Assembly()
assembly.add(sphere, name="breeder")
assembly.add(box, name="structure")

geometry = yamc.MeshGeometry.from_cadquery(
    assembly,
    materials={"breeder": breeder, "structure": steel},
    tet_volumes=["breeder", "structure"],
    target_edge_length=2.5,  # cm
)
tets = np.array(geometry.tetrahedra)
tet_names = np.array(geometry.tet_volume_names)
in_breeder = tet_names == "breeder"
tet_volume = np.array(geometry.tet_volumes)  # cm3
```

The tets of the two volumes meet face to face on the sphere, and their outer
faces are exactly the surface triangles transport tracks against.
``geometry.to_arrow("blanket.arrow")`` would save the mesh, for
``MeshGeometry.from_arrow`` to load later, but nothing here needs the file.

## Neutronics model

One heating tally covers every tet: given the geometry, a tet-mesh tally has a
bin for each of its tets, in the order of ``geometry.tetrahedra``.

```python
heating_tally = yamc.Tally(scores=["heating"], mesh=geometry)
model = yamc.Model(
    geometry=geometry,
    source=yamc.NeutronSource(),
    tallies=[heating_tally],
    transport_secondary_photons=True,
    tracking_mode="hybrid",  # per-tet densities need delta tracking
)


def heating_density(seed):
    """Nuclear heating in W/cm3 in every tet."""
    results = model.simulate_transport(total_particles=100_000, seed=seed)
    ev_per_source = np.array(results[heating_tally].mean)
    source_rate = 5e15  # neutrons per second
    return ev_per_source * source_rate * 1.602176634e-19 / tet_volume  # eV to J
```

A tet-mesh tally scores eV per source neutron in each tet, integrated over the
tet. Multiplying by the source rate and dividing by the tet volume gives W/cm3.

## Thermal model on the same tets

``dolfinx.mesh.create_mesh`` builds the FEniCSx mesh from yamc's vertices and
tets. FEniCSx reorders the cells, and ``original_cell_index`` maps each of its
cells back to a yamc tet. Heating and conductivity are constant per tet (DG0),
the temperature is continuous (CG1), and each tet's temperature for the
feedback is the CG1 field at its centre.

```python
CONDUCTIVITY = {"breeder": 0.15, "structure": 0.25}  # W/(cm K)

used, cells = np.unique(tets, return_inverse=True)
coordinates = np.array(geometry.vertices)[used]
element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
domain = mesh.create_mesh(MPI.COMM_WORLD, cells.reshape(-1, 4), ufl.Mesh(element), coordinates)
yamc_tet = domain.topology.original_cell_index

DG0 = fem.functionspace(domain, ("DG", 0))
V = fem.functionspace(domain, ("Lagrange", 1))
kappa = fem.Function(DG0)
kappa.x.array[:] = [CONDUCTIVITY[name] for name in tet_names[yamc_tet]]
q = fem.Function(DG0)

# the mesh is conformal, so its boundary is just the box's outer faces
domain.topology.create_connectivity(2, 3)
outer = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(T_WALL, fem.locate_dofs_topological(V, 2, outer), V)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
problem = LinearProblem(
    kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx,
    q * v * ufl.dx,
    bcs=[bc],
    petsc_options_prefix="heat_",
    petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-10},
)
temperature_per_tet = fem.Function(DG0)


def tet_temperatures(heating):
    """Steady temperature in K at the centre of every tet."""
    q.x.array[:] = heating[yamc_tet]
    temperature_per_tet.interpolate(problem.solve())
    T = np.empty(len(tets))
    T[yamc_tet] = temperature_per_tet.x.array
    return T
```

Every facet of the breeder/steel interface is shared by a breeder tet and a
steel tet, so FEniCSx's exterior facets are exactly the box's outer faces and
the boundary condition goes on all of them, with no geometric test. Heat
crosses the whole interface.

## Iterating

Each transport run is noisy, so the heating is a running mean over the
iterations rather than the latest run alone. That averages the noise down as
the temperature converges. Only the breeder's tets get density factors; the
steel's stay at 1.

```python
heating = None
T_old = np.full(len(tets), T_WALL)
for iteration in range(1, 7):
    start = time.perf_counter()
    factors = np.where(in_breeder, pbli_density(T_old) / pbli_density(T_WALL), 1.0)
    model.set_tet_density_factors(factors)
    new = heating_density(seed=iteration)
    heating = new if heating is None else heating + (new - heating) / iteration
    T = tet_temperatures(heating)
    print(
        f"iteration {iteration}: power {np.sum(heating * tet_volume) / 1e3:.2f} kW, "
        f"peak T {T.max():.1f} K, breeder density {factors[in_breeder].min():.4f}"
        f"-{factors[in_breeder].max():.4f} of 600 K, max |dT| {np.abs(T - T_old).max():.2f} K, "
        f"{time.perf_counter() - start:.0f} s"
    )
    T_old = T
```

```text
iteration 1: power 4.39 kW, peak T 784.8 K, breeder density 1.0000-1.0000 of 600 K, max |dT| 184.82 K, 44 s
iteration 2: power 4.37 kW, peak T 782.8 K, breeder density 0.9776-0.9954 of 600 K, max |dT| 2.24 K, 43 s
iteration 3: power 4.38 kW, peak T 782.4 K, breeder density 0.9778-0.9954 of 600 K, max |dT| 0.54 K, 42 s
iteration 4: power 4.38 kW, peak T 781.8 K, breeder density 0.9779-0.9954 of 600 K, max |dT| 0.62 K, 44 s
iteration 5: power 4.38 kW, peak T 781.7 K, breeder density 0.9779-0.9954 of 600 K, max |dT| 0.23 K, 40 s
iteration 6: power 4.38 kW, peak T 781.5 K, breeder density 0.9779-0.9954 of 600 K, max |dT| 0.17 K, 41 s
```

The first iteration has no feedback, as every factor is 1. From the second on
the hottest PbLi is 2.2% less dense than at the wall, and the temperature has
settled to within the transport noise by the third. The feedback lowers the
peak temperature by about 3 K here: PbLi's density changes little over a few
hundred kelvin, so density feedback is a small correction for this breeder.
