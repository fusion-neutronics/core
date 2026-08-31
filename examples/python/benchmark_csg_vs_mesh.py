"""
Benchmark: CSG vs Mesh geometry for neutron transport in YAMC.

Takes geometry definitions from model classes and runs each model
in two configurations:

  1. YAMC CSG:  analytic surfaces via yamc.Sphere / yamc.ZCylinder / yamc.Plane
  2. YAMC Mesh: CadQuery CAD -> CadToYamc (yamm surface mesh) -> Arrow IPC -> yamc.MeshGeometry (yamt)

All use the same material (Be9), source (14.1 MeV DT at origin), and
energy-binned flux tally (VITAMIN-J-175).  Results are compared with
reduced chi-squared and plotted.

Requirements:
    pip install yamc[cad] matplotlib numpy
"""

import tempfile
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import cadquery as cq
import yamc
from yamc.cad import CadToYamc

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
PARTICLES = 50_000
BATCHES = 10
SEED = 42
ENERGY_GROUP = "VITAMIN-J-175"
SOURCE_ENERGY = 14.06e6  # eV (DT fusion neutron)
NUCLIDE = "Be9"
NUCLIDE_H5 = "tests/Be9.arrow"
DENSITY = 1.85  # g/cm3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_material_id_counter = 0

def make_material(name="beryllium"):
    global _material_id_counter
    _material_id_counter += 1
    mat = yamc.Material(
        composition={NUCLIDE: 1.0},
        density=DENSITY,
        name=name,
        temperature=294)
    mat.read_nuclear_data({NUCLIDE: NUCLIDE_H5})
    return mat


def make_source(point=None):
    if point is None:
        point = [0, 0, 0]
    return yamc.NeutronSource(
        energy=SOURCE_ENERGY,
        position=point)


def plane_from_points(p1, p2, p3, boundary=None):
    """Create a Plane from three points."""
    v1 = [p2[i] - p1[i] for i in range(3)]
    v2 = [p3[i] - p1[i] for i in range(3)]
    a = v1[1] * v2[2] - v1[2] * v2[1]
    b = v1[2] * v2[0] - v1[0] * v2[2]
    c = v1[0] * v2[1] - v1[1] * v2[0]
    d = a * p1[0] + b * p1[1] + c * p1[2]
    norm = (a * a + b * b + c * c) ** 0.5
    return yamc.Plane(axis=(a, b, c), offset=d / norm, boundary=boundary)


def cadquery_to_mesh_geometry(assembly, material_names, mesh_size=None):
    """Convert a CadQuery Assembly to a yamc.MeshGeometry."""
    tmpdir = tempfile.mkdtemp(prefix="yamc_bench_")
    arrow_path = os.path.join(tmpdir, "model.arrow")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=material_names)

    tolerance = mesh_size * 0.05 if mesh_size else 0.1
    c2y.mesh(tolerance=tolerance, angular_tolerance=0.1)
    c2y.to_arrow(arrow_path)

    # Build yamc materials map
    materials_map = {}
    for name in material_names:
        if name not in materials_map:
            materials_map[name] = make_material(name)

    mesh_geom = yamc.MeshGeometry(arrow_path, materials_map)
    return mesh_geom, arrow_path


def run_simulation(geometry, label, source_point=None):
    """Run a YAMC transport simulation and return energy-binned flux results."""
    source = make_source(source_point)

    tally = yamc.Tally(
        scores=["flux"],
        name=f"flux_{label}",
        energy_group_structure=ENERGY_GROUP)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)

    t0 = time.time()
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
    dt = time.time() - t0

    tally_result = results[tally]
    mean = np.array(tally_result.mean)
    std = np.array(tally_result.standard_deviation)

    rate = PARTICLES * BATCHES / dt
    print(f"  {label}: {dt:.2f}s  ({rate:,.0f} p/s)  total_flux={np.sum(mean):.4e}")
    return mean, std, dt


# ---------------------------------------------------------------------------
# Benchmark model definitions (yamc CSG + CadQuery for mesh)
# ---------------------------------------------------------------------------

class BenchmarkModel:
    """Base class for benchmark models."""
    name: str

    def csg_geometry(self):
        raise NotImplementedError

    def cadquery_assembly(self):
        raise NotImplementedError

    def mesh_size(self):
        return None

    def source_point(self):
        return [0, 0, 0]


class SphereModel(BenchmarkModel):
    name = "Sphere"

    def __init__(self, radius=10):
        self.radius = radius

    def csg_geometry(self):
        s = yamc.Sphere(radius=self.radius, boundary="vacuum")
        mat = make_material()
        cell = yamc.Cell(name="sphere", region=s.below, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="sphere")
        assembly.add(cq.Workplane().sphere(self.radius))
        return assembly, ["beryllium"]

    def mesh_size(self):
        return self.radius / 5


class CuboidModel(BenchmarkModel):
    name = "Cuboid"

    def __init__(self, width=10):
        self.width = width

    def csg_geometry(self):
        hw = self.width / 2
        xp = yamc.Plane(axis="x", offset=hw, boundary="vacuum")
        xn = yamc.Plane(axis="x", offset=-hw, boundary="vacuum")
        yp = yamc.Plane(axis="y", offset=hw, boundary="vacuum")
        yn = yamc.Plane(axis="y", offset=-hw, boundary="vacuum")
        zp = yamc.Plane(axis="z", offset=hw, boundary="vacuum")
        zn = yamc.Plane(axis="z", offset=-hw, boundary="vacuum")
        region = xp.below & xn.above & yp.below & yn.above & zp.below & zn.above
        mat = make_material()
        cell = yamc.Cell(name="cuboid", region=region, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="cuboid")
        assembly.add(cq.Workplane().box(self.width, self.width, self.width))
        return assembly, ["beryllium"]

    def mesh_size(self):
        return self.width / 5


class CylinderModel(BenchmarkModel):
    name = "Cylinder"

    def __init__(self, radius=5, height=20):
        self.radius = radius
        self.height = height

    def csg_geometry(self):
        cyl = yamc.Cylinder(axis="z", radius=self.radius, boundary="vacuum")
        top = yamc.Plane(axis="z", offset=self.height / 2, boundary="vacuum")
        bot = yamc.Plane(axis="z", offset=-self.height / 2, boundary="vacuum")
        region = cyl.below & top.below & bot.above
        mat = make_material()
        cell = yamc.Cell(name="cylinder", region=region, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="cylinder")
        assembly.add(
            cq.Workplane("XY").circle(self.radius).extrude(self.height / 2, both=True)
        )
        return assembly, ["beryllium"]

    def mesh_size(self):
        return self.radius / 3


class NestedSphereModel(BenchmarkModel):
    name = "Nested Sphere"

    def __init__(self, radius_inner=5, radius_outer=10):
        self.r1 = radius_inner
        self.r2 = radius_outer

    def csg_geometry(self):
        s1 = yamc.Sphere(radius=self.r1)
        s2 = yamc.Sphere(radius=self.r2, boundary="vacuum")
        mat = make_material("inner")
        mat2 = make_material("outer")
        cell1 = yamc.Cell(name="inner", region=s1.below, material=mat)
        cell2 = yamc.Cell(name="outer", region=s1.above & s2.below, material=mat2)
        return yamc.Geometry([cell1, cell2])

    def cadquery_assembly(self):
        inner = cq.Workplane().sphere(self.r1)
        outer_shell = cq.Workplane().sphere(self.r2).cut(cq.Workplane().sphere(self.r1))
        assembly = cq.Assembly(name="nested_sphere")
        assembly.add(inner)
        assembly.add(outer_shell)
        return assembly, ["inner", "outer"]

    def mesh_size(self):
        return self.r1 / 3


class TwoTouchingCuboidsModel(BenchmarkModel):
    name = "Two Touching Cuboids"

    def __init__(self, width1=10, width2=4):
        self.w1 = width1
        self.w2 = width2

    def csg_geometry(self):
        hw1 = self.w1 / 2
        s1 = yamc.Plane(axis="z", offset=hw1, boundary="vacuum")
        s2 = yamc.Plane(axis="z", offset=-hw1, boundary="vacuum")
        s3 = yamc.Plane(axis="x", offset=hw1, boundary="vacuum")
        s4 = yamc.Plane(axis="x", offset=-hw1, boundary="vacuum")
        s5 = yamc.Plane(axis="y", offset=hw1)  # shared interface
        s6 = yamc.Plane(axis="y", offset=-hw1, boundary="vacuum")
        s7 = yamc.Plane(axis="y", offset=hw1 + self.w2, boundary="vacuum")

        region1 = s1.below & s2.above & s3.below & s4.above & s5.below & s6.above
        region2 = s1.below & s2.above & s3.below & s4.above & s7.below & s5.above

        mat1 = make_material("mat1")
        mat2 = make_material("mat2")
        cell1 = yamc.Cell(name="cuboid1", region=region1, material=mat1)
        cell2 = yamc.Cell(name="cuboid2", region=region2, material=mat2)
        return yamc.Geometry([cell1, cell2])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="two_touching_cuboids")
        c1 = cq.Workplane().box(self.w1, self.w1, self.w1)
        c2 = (
            cq.Workplane()
            .transformed(offset=cq.Vector(0, self.w1 / 2 + self.w2 / 2, 0))
            .box(self.w1, self.w2, self.w1)
        )
        assembly.add(c1)
        assembly.add(c2)
        return assembly, ["mat1", "mat2"]

    def mesh_size(self):
        return self.w2 / 2


class NestedCylinderModel(BenchmarkModel):
    """Two coaxial cylinders."""
    name = "Nested Cylinder"

    def __init__(self, radius1=15, radius2=8, height1=15, height2=8):
        self.r1 = radius1
        self.r2 = radius2
        self.h1 = height1
        self.h2 = height2

    def csg_geometry(self):
        cyl_outer = yamc.Cylinder(axis="z", radius=self.r1, boundary="vacuum")
        top_outer = yamc.Plane(axis="z", offset=self.h1 / 2, boundary="vacuum")
        bot_outer = yamc.Plane(axis="z", offset=-self.h1 / 2, boundary="vacuum")
        cyl_inner = yamc.Cylinder(axis="z", radius=self.r2)
        top_inner = yamc.Plane(axis="z", offset=self.h2 / 2)
        bot_inner = yamc.Plane(axis="z", offset=-self.h2 / 2)

        inner_region = cyl_inner.below & top_inner.below & bot_inner.above
        outer_region = (cyl_outer.below & top_outer.below & bot_outer.above) & ~inner_region

        mat_inner = make_material("cyl_inner")
        mat_outer = make_material("cyl_outer")
        cell1 = yamc.Cell(name="inner_cyl", region=inner_region, material=mat_inner)
        cell2 = yamc.Cell(name="outer_cyl", region=outer_region, material=mat_outer)
        return yamc.Geometry([cell1, cell2])

    def cadquery_assembly(self):
        inner = cq.Workplane("XY").circle(self.r2).extrude(self.h2 / 2, both=True)
        outer_shell = (
            cq.Workplane("XY").circle(self.r1).extrude(self.h1 / 2, both=True)
            .cut(cq.Workplane("XY").circle(self.r2).extrude(self.h2 / 2, both=True))
        )
        assembly = cq.Assembly(name="nested_cylinder")
        assembly.add(inner)
        assembly.add(outer_shell)
        return assembly, ["cyl_inner", "cyl_outer"]

    def mesh_size(self):
        return self.r2 / 3


class SimpleTokamakModel(BenchmarkModel):
    """Simplified tokamak.

    4 regions: plasma (inner sphere), blanket (sphere shell),
    center column (cylinder), outer vessel (remaining).
    """
    name = "Simple Tokamak"

    def __init__(self, radius=20, blanket=5, center_column=6):
        self.radius = radius
        self.blanket = blanket
        self.cc = center_column

    def csg_geometry(self):
        r = self.radius
        b = self.blanket
        cc_height = (r + b + 1) * 2
        outer_r = r + b + 2

        s_inner = yamc.Sphere(radius=r)
        s_outer = yamc.Sphere(radius=r + b)
        cyl_cc = yamc.Cylinder(axis="z", radius=self.cc)
        top = yamc.Plane(axis="z", offset=cc_height / 2, boundary="vacuum")
        bot = yamc.Plane(axis="z", offset=-cc_height / 2, boundary="vacuum")
        cyl_outer = yamc.Cylinder(axis="z", radius=outer_r, boundary="vacuum")

        region_plasma = s_inner.below & cyl_cc.above
        region_blanket = s_inner.above & s_outer.below & cyl_cc.above
        region_cc = top.below & bot.above & cyl_cc.below
        region_vessel = s_outer.above & top.below & bot.above & cyl_cc.above & cyl_outer.below

        mat1 = make_material("plasma")
        mat2 = make_material("blanket")
        mat3 = make_material("column")
        mat4 = make_material("vessel")

        cell1 = yamc.Cell(name="plasma", region=region_plasma, material=mat1)
        cell2 = yamc.Cell(name="blanket", region=region_blanket, material=mat2)
        cell3 = yamc.Cell(name="center_col", region=region_cc, material=mat3)
        cell4 = yamc.Cell(name="vessel", region=region_vessel, material=mat4)
        return yamc.Geometry([cell1, cell2, cell3, cell4])

    def cadquery_assembly(self):
        r = self.radius
        b = self.blanket
        cc_height = (r + b + 1) * 2
        outer_r = r + b + 2

        inner_sphere = cq.Workplane().sphere(r)
        outer_sphere = cq.Workplane().sphere(r + b)
        cc_cyl = cq.Workplane("XY").circle(self.cc).extrude(cc_height / 2, both=True)
        outer_cyl = cq.Workplane("XY").circle(outer_r).extrude(cc_height / 2, both=True)

        plasma = inner_sphere.cut(cc_cyl)
        blanket = outer_sphere.cut(inner_sphere).cut(cc_cyl)
        column = cc_cyl
        vessel = outer_cyl.cut(outer_sphere).cut(cc_cyl)

        assembly = cq.Assembly(name="simple_tokamak")
        assembly.add(plasma)
        assembly.add(blanket)
        assembly.add(column)
        assembly.add(vessel)
        return assembly, ["plasma", "blanket", "column", "vessel"]

    def mesh_size(self):
        return self.radius / 4


class CircularTorusModel(BenchmarkModel):
    """Circular torus. Single-material torus with equal minor radii (b == c)."""
    name = "Circular Torus"

    def __init__(self, major_radius=10, minor_radius=3):
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def source_point(self):
        return [self.major_radius, 0, 0]

    def csg_geometry(self):
        torus = yamc.Torus(
            axis="z",
            r_major=self.major_radius,
            r_minor=self.minor_radius,
            boundary="vacuum")
        mat = make_material()
        cell = yamc.Cell(name="torus", region=torus.below, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="circular_torus")
        torus = cq.Solid.makeTorus(self.major_radius, self.minor_radius)
        assembly.add(torus)
        return assembly, ["beryllium"]

    def mesh_size(self):
        return self.minor_radius / 2


class EllipticalTorusModel(BenchmarkModel):
    """Elliptical torus. Different minor radii in z and xy directions."""
    name = "Elliptical Torus"

    def __init__(self, major_radius=10, minor_radius_z=4, minor_radius_xy=2):
        self.major_radius = major_radius
        self.minor_z = minor_radius_z
        self.minor_xy = minor_radius_xy

    def source_point(self):
        return [self.major_radius, 0, 0]

    def csg_geometry(self):
        torus = yamc.Torus(
            axis="z",
            r_major=self.major_radius,
            r_minor=self.minor_z,
            r_minor_2=self.minor_xy,
            boundary="vacuum")
        mat = make_material()
        cell = yamc.Cell(name="elliptical_torus", region=torus.below, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="elliptical_torus")
        half1 = (
            cq.Workplane("XZ", origin=(self.major_radius, 0, 0))
            .ellipse(self.minor_xy, self.minor_z)
            .revolve(180, (-self.major_radius, 0, 0), (-self.major_radius, 1, 0))
        )
        half2 = (
            cq.Workplane("XZ", origin=(-self.major_radius, 0, 0))
            .ellipse(self.minor_xy, self.minor_z)
            .revolve(180, (self.major_radius, 0, 0), (self.major_radius, 1, 0))
        )
        torus = half1.union(half2)
        assembly.add(torus)
        return assembly, ["beryllium"]

    def mesh_size(self):
        return min(self.minor_z, self.minor_xy) / 2


class NestedTorusModel(BenchmarkModel):
    """Nested concentric tori. Multiple shells sharing the same major radius."""
    name = "Nested Torus"

    def __init__(self, major_radius=10, minor_radii=None):
        self.major_radius = major_radius
        self.minor_radii = minor_radii or [4, 3, 2, 1]

    def source_point(self):
        return [self.major_radius, 0, 0]

    def csg_geometry(self):
        surfaces = []
        for i, r in enumerate(self.minor_radii):
            bt = "vacuum" if i == 0 else "transmission"
            surfaces.append(yamc.Torus(axis="z", r_major=self.major_radius, r_minor=r, boundary=bt))

        cells = []
        for i in range(len(self.minor_radii) - 1):
            name = f"torus_{i}"
            mat = make_material(name)
            region = surfaces[i + 1].above & surfaces[i].below
            cells.append(yamc.Cell(id=i + 1, name=f"shell_{i}", region=region, material=mat))
        core_name = f"torus_{len(self.minor_radii) - 1}"
        mat = make_material(core_name)
        cells.append(yamc.Cell(
            id=len(self.minor_radii), name="core",
            region=surfaces[-1].below, material=mat))
        return yamc.Geometry(cells)

    def cadquery_assembly(self):
        assembly = cq.Assembly(name="nested_torus")
        mat_names = []
        radii_asc = list(reversed(self.minor_radii))
        for i, r in enumerate(radii_asc):
            idx = len(self.minor_radii) - 1 - i
            torus = cq.Solid.makeTorus(self.major_radius, r)
            if i > 0:
                inner_torus = cq.Solid.makeTorus(self.major_radius, radii_asc[i - 1])
                torus = torus.cut(inner_torus)
            assembly.add(torus)
            mat_names.append(f"torus_{idx}")
        return assembly, mat_names

    def mesh_size(self):
        return min(self.minor_radii) / 2


class TetrahedralModel(BenchmarkModel):
    """Single tetrahedron defined by 4 planes."""
    name = "Tetrahedron"

    def __init__(self, length=10):
        self.length = length

    def source_point(self):
        L = self.length
        return [L / 4, L / 4, L / 4]

    def csg_geometry(self):
        L = self.length
        A, B, C, D = (0, 0, 0), (L, 0, 0), (0, L, 0), (0, 0, L)

        plane1 = plane_from_points(B, C, D, boundary="vacuum")
        plane2 = plane_from_points(A, C, D, boundary="vacuum")
        plane3 = plane_from_points(A, B, D, boundary="vacuum")
        plane4 = plane_from_points(A, B, C, boundary="vacuum")

        region = plane1.below & plane2.above & plane3.below & plane4.above
        mat = make_material()
        cell = yamc.Cell(name="tetrahedron", region=region, material=mat)
        return yamc.Geometry([cell])

    def cadquery_assembly(self):
        L = self.length
        A = cq.Vector(0, 0, 0)
        B = cq.Vector(L, 0, 0)
        C = cq.Vector(0, L, 0)
        D = cq.Vector(0, 0, L)

        f1 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B, C, A]))
        f2 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B, D, A]))
        f3 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, C, D, A]))
        f4 = cq.Face.makeFromWires(cq.Wire.makePolygon([B, C, D, B]))
        shell = cq.Shell.makeShell([f1, f2, f3, f4])
        solid = cq.Solid.makeSolid(shell)

        assembly = cq.Assembly(name="tetrahedron")
        assembly.add(solid)
        return assembly, ["beryllium"]

    def mesh_size(self):
        return self.length / 5


class TwoTetrahedronsModel(BenchmarkModel):
    """Two tetrahedrons sharing a face. Mirror-image at x=0 plane."""
    name = "Two Tetrahedrons"

    def __init__(self, length=10):
        self.length = length

    def source_point(self):
        L = self.length
        return [L * 0.1, L / 4, L / 4]

    def csg_geometry(self):
        L = self.length
        plane_x0 = yamc.Plane(axis="x", offset=0)
        plane_y0 = yamc.Plane(axis="y", offset=0, boundary="vacuum")
        plane_z0 = yamc.Plane(axis="z", offset=0, boundary="vacuum")
        plane_1 = yamc.Plane(axis=(1, 1, 1), offset=L / ((1**2 + 1**2 + 1**2) ** 0.5), boundary="vacuum")
        plane_2 = yamc.Plane(axis=(-1, 1, 1), offset=L / (((-1)**2 + 1**2 + 1**2) ** 0.5), boundary="vacuum")

        region1 = plane_x0.above & plane_y0.above & plane_z0.above & plane_1.below
        region2 = plane_x0.below & plane_y0.above & plane_z0.above & plane_2.below

        mat1 = make_material("tet1")
        mat2 = make_material("tet2")
        cell1 = yamc.Cell(name="tet1", region=region1, material=mat1)
        cell2 = yamc.Cell(name="tet2", region=region2, material=mat2)
        return yamc.Geometry([cell1, cell2])

    def cadquery_assembly(self):
        L = self.length
        A = cq.Vector(0, 0, 0)
        B = cq.Vector(L, 0, 0)
        C = cq.Vector(0, L, 0)
        D = cq.Vector(0, 0, L)
        B_prime = cq.Vector(-L, 0, 0)

        f1_1 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B, C, A]))
        f1_2 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B, D, A]))
        f1_3 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, C, D, A]))
        f1_4 = cq.Face.makeFromWires(cq.Wire.makePolygon([B, C, D, B]))
        shell1 = cq.Shell.makeShell([f1_1, f1_2, f1_3, f1_4])
        solid1 = cq.Solid.makeSolid(shell1)

        f2_1 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B_prime, C, A]))
        f2_2 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, B_prime, D, A]))
        f2_3 = cq.Face.makeFromWires(cq.Wire.makePolygon([A, C, D, A]))
        f2_4 = cq.Face.makeFromWires(cq.Wire.makePolygon([B_prime, C, D, B_prime]))
        shell2 = cq.Shell.makeShell([f2_1, f2_2, f2_3, f2_4])
        solid2 = cq.Solid.makeSolid(shell2)

        assembly = cq.Assembly(name="two_tetrahedrons")
        assembly.add(solid1)
        assembly.add(solid2)
        return assembly, ["tet1", "tet2"]

    def mesh_size(self):
        return self.length / 5


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

MODELS = [
    SphereModel(radius=10),
    CuboidModel(width=10),
    CylinderModel(radius=5, height=20),
    NestedSphereModel(radius_inner=5, radius_outer=10),
    TwoTouchingCuboidsModel(width1=10, width2=4),
    NestedCylinderModel(radius1=15, radius2=8, height1=15, height2=8),
    SimpleTokamakModel(radius=20, blanket=5, center_column=6),
    CircularTorusModel(major_radius=10, minor_radius=3),
    EllipticalTorusModel(major_radius=10, minor_radius_z=4, minor_radius_xy=2),
    NestedTorusModel(major_radius=10, minor_radii=[4, 3, 2, 1]),
    TetrahedralModel(length=10),
    TwoTetrahedronsModel(length=10),
]

_ref_tally = yamc.Tally(scores=["flux"], energy_group_structure=ENERGY_GROUP)
energy_bins = np.array(_ref_tally.energy_bins)

all_results = []

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"  {model.name}")
    print(f"{'='*60}")

    sp = model.source_point()

    # --- YAMC CSG ---
    csg_geom = model.csg_geometry()
    csg_mean, csg_std, csg_time = run_simulation(csg_geom, "CSG", source_point=sp)

    # --- YAMC Mesh ---
    assembly, mat_names = model.cadquery_assembly()
    mesh_geom, _ = cadquery_to_mesh_geometry(
        assembly, mat_names, mesh_size=model.mesh_size()
    )
    print(f"  {mesh_geom}")
    mesh_mean, mesh_std, mesh_time = run_simulation(mesh_geom, "Mesh", source_point=sp)

    # --- Chi-squared (YAMC CSG vs YAMC Mesh) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        combined_var = csg_std**2 + mesh_std**2
        chi2_terms = np.where(
            combined_var > 0, (mesh_mean - csg_mean) ** 2 / combined_var, 0
        )
    mask = (csg_mean > 0) | (mesh_mean > 0)
    n_bins = int(np.sum(mask))
    chi2 = float(np.sum(chi2_terms[mask]))
    red_chi2 = chi2 / n_bins if n_bins > 0 else 0.0

    total_csg = np.sum(csg_mean)
    total_mesh = np.sum(mesh_mean)
    rel_diff = (total_mesh - total_csg) / total_csg * 100 if total_csg else 0.0

    print(f"  chi2/dof={red_chi2:.2f}  flux_diff={rel_diff:+.2f}%  "
          f"speedup={csg_time/mesh_time:.2f}x (CSG/Mesh)")

    all_results.append(dict(
        name=model.name,
        csg_mean=csg_mean, csg_std=csg_std, csg_time=csg_time,
        mesh_mean=mesh_mean, mesh_std=mesh_std, mesh_time=mesh_time,
        chi2=red_chi2, rel_diff=rel_diff))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
n = PARTICLES * BATCHES

hdr = (f"{'Model':<25} {'chi2/dof':>10} {'Flux diff':>10} "
       f"{'CSG (p/s)':>12} {'Mesh (p/s)':>12} {'CSG/Mesh':>10}")
sep_len = 92

print(f"\n{'='*sep_len}")
print(hdr)
print(f"{'-'*sep_len}")
for r in all_results:
    csg_rate = n / r["csg_time"]
    mesh_rate = n / r["mesh_time"]
    csg_mesh = csg_rate / mesh_rate if mesh_rate > 0 else float("inf")
    print(
        f"{r['name']:<25} {r['chi2']:>10.2f} {r['rel_diff']:>+9.2f}% "
        f"{csg_rate:>12,.0f} {mesh_rate:>12,.0f} {csg_mesh:>9.1f}x"
    )
print(f"{'='*sep_len}")

# ---------------------------------------------------------------------------
# Plot all models
# ---------------------------------------------------------------------------
n_models = len(all_results)
fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models),
                         gridspec_kw={"width_ratios": [3, 1]})
if n_models == 1:
    axes = axes[np.newaxis, :]

for i, r in enumerate(all_results):
    ax_flux = axes[i, 0]
    ax_ratio = axes[i, 1]

    # Flux spectrum
    ax_flux.step(energy_bins[:-1], r["csg_mean"], where="post",
                 label="CSG", linewidth=1.5, color="C0")
    ax_flux.step(energy_bins[:-1], r["mesh_mean"], where="post",
                 label="Mesh", linewidth=1.5, linestyle="--", color="C1")
    ax_flux.set_xscale("log")
    ax_flux.set_yscale("log")
    nonzero = (r["csg_mean"] > 0) | (r["mesh_mean"] > 0)
    if np.any(nonzero):
        ymin = min(r["csg_mean"][nonzero].min(), r["mesh_mean"][nonzero].min()) * 0.3
        ax_flux.set_ylim(bottom=ymin)
    ax_flux.set_ylabel("Flux")

    # Title with speed info
    csg_rate = n / r["csg_time"]
    mesh_rate = n / r["mesh_time"]
    title = f"{r['name']}  |  chi2/dof={r['chi2']:.2f}  diff={r['rel_diff']:+.1f}%"
    ax_flux.set_title(title, fontsize=9)
    ax_flux.legend(fontsize=8)
    ax_flux.grid(True, alpha=0.3)
    if i == n_models - 1:
        ax_flux.set_xlabel("Energy (eV)")

    # Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r["csg_mean"] > 0, r["mesh_mean"] / r["csg_mean"], np.nan)
    ax_ratio.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax_ratio.step(energy_bins[:-1], ratio, where="post", color="C2", linewidth=1)
    ax_ratio.set_xscale("log")
    ax_ratio.set_ylabel("Mesh/CSG")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, alpha=0.3)
    if i == n_models - 1:
        ax_ratio.set_xlabel("Energy (eV)")

plt.tight_layout()
out_path = "benchmark_csg_vs_mesh.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.close()
