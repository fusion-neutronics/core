//! A solid cut into parts by internal faces: surfaces with the solid on both
//! sides (`[v, v]` in `surface_volumes`), as a file from another writer can
//! have (a MOAB surface with sense both ways, or a gmsh model converted with
//! one volume per material group). An internal face bounds nothing, so it
//! must change neither which volume a point is in nor where a ray leaves it.
//!
//! The meshes are those of issues #517 and #533: a box with one internal
//! face, its octants, a 27-cell cube (also turned off the axes), slabs in
//! four layers, a torus shell in four layers, cylinder shells in radial and
//! axial segments, and a closed internal surface. Point location passed over
//! every ray whose nearest crossing was an internal face, so a region enclosed
//! by them, and much of each middle layer of a slab, could not be placed.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, Beyond, MeshGeometry, Sense};

type P = [f64; 3];

/// One volume as a triangle soup, with the `surface_volumes` entry of each
/// of its surfaces.
#[derive(Default)]
struct Soup {
    vertices: Vec<P>,
    triangles: Vec<[u32; 3]>,
    surface_ids: Vec<u32>,
    surface_volumes: Vec<&'static str>,
}

impl Soup {
    /// A new surface with this `surface_volumes` entry, by its 1-based id.
    fn surface(&mut self, entry: &'static str) -> u32 {
        self.surface_volumes.push(entry);
        self.surface_volumes.len() as u32
    }

    fn quad(&mut self, q: [u32; 4], surface: u32) {
        self.triangles.push([q[0], q[1], q[2]]);
        self.triangles.push([q[0], q[2], q[3]]);
        self.surface_ids.extend([surface, surface]);
    }

    /// Points on the internal faces: the corners of their triangles, the
    /// midpoints of their edges, and points inside them, on up to 100 of the
    /// triangles spread over the faces.
    fn internal_face_points(&self) -> Vec<P> {
        let internal: Vec<[u32; 3]> = self
            .triangles
            .iter()
            .zip(&self.surface_ids)
            .filter(|&(_, &s)| self.surface_volumes[s as usize - 1] == "[0, 0]")
            .map(|(&t, _)| t)
            .collect();
        let mut points = Vec::new();
        for t in internal.iter().step_by(internal.len().div_ceil(100).max(1)) {
            let [a, b, c] = t.map(|i| self.vertices[i as usize]);
            for w in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.2, 0.3, 0.5],
                [0.7, 0.2, 0.1],
            ] {
                points.push([0, 1, 2].map(|k| w[0] * a[k] + w[1] * b[k] + w[2] * c[k]));
            }
        }
        points
    }

    fn geometry(&self) -> MeshGeometry {
        self.geometry_with(r#"{"1": {"name": "mat:a", "dim": 3}}"#)
    }

    fn geometry_with(&self, physical_groups_json: &str) -> MeshGeometry {
        let data = ArrowMeshData {
            triangle_aabbs: build_triangle_aabbs(&self.triangles, &self.vertices),
            tet_aabbs: build_tet_aabbs(&[], &self.vertices),
            triangle_physical_groups: vec![1; self.triangles.len()],
            triangle_surface_ids: self.surface_ids.clone(),
            triangles: self.triangles.clone(),
            vertices: self.vertices.clone(),
            physical_groups_json: physical_groups_json.to_string(),
            surface_volumes_json: format!("[{}]", self.surface_volumes.join(", ")),
            imprinted: Some(true),
            ..Default::default()
        };
        MeshGeometry::from_topology(build_topology(data).unwrap())
    }
}

/// The box [0, n0 h0] x [0, n1 h1] x [0, n2 h2] on a lattice of cells of size
/// `h`. Its outside is surface 1, wound outward; with `cut[a]` each lattice
/// plane inside it across axis `a` is an internal face of its own.
fn lattice_box(n: [usize; 3], h: P, cut: [bool; 3]) -> Soup {
    let id = |i: [usize; 3]| ((i[2] * (n[1] + 1) + i[1]) * (n[0] + 1) + i[0]) as u32;
    let mut s = Soup::default();
    for k in 0..=n[2] {
        for j in 0..=n[1] {
            for i in 0..=n[0] {
                s.vertices
                    .push([i as f64 * h[0], j as f64 * h[1], k as f64 * h[2]]);
            }
        }
    }
    let outside = s.surface("[0, null]");
    for a in 0..3 {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        for plane in 0..=n[a] {
            let surface = if plane == 0 || plane == n[a] {
                outside
            } else if cut[a] {
                s.surface("[0, 0]")
            } else {
                continue;
            };
            for jb in 0..n[b] {
                for jc in 0..n[c] {
                    let corner = |db: usize, dc: usize| {
                        let mut i = [0; 3];
                        (i[a], i[b], i[c]) = (plane, jb + db, jc + dc);
                        id(i)
                    };
                    // Faces +a as listed, so the plane at 0 is turned round.
                    let q = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
                    s.quad(
                        if plane == 0 {
                            [q[0], q[3], q[2], q[1]]
                        } else {
                            q
                        },
                        surface,
                    );
                }
            }
        }
    }
    s
}

/// A shell about the z axis between radii `r`, from the first level of `z`
/// to the last, with `n` facets round. Its outside is surface 1, wound
/// outward; with `cut_r` each cylinder between the first and last radius is
/// an internal face, and with `cut_z` each annulus between the end levels.
fn cylinder_shell(r: &[f64], z: &[f64], n: u32, cut_r: bool, cut_z: bool) -> Soup {
    let (k, m) = (r.len() - 1, z.len() - 1);
    let mut s = Soup::default();
    for &ri in r {
        for &zl in z {
            for j in 0..n {
                let th = std::f64::consts::TAU * f64::from(j) / f64::from(n);
                s.vertices.push([ri * th.cos(), ri * th.sin(), zl]);
            }
        }
    }
    let id = |i: usize, l: usize, j: u32| (i * (m + 1) + l) as u32 * n + j % n;
    let outside = s.surface("[0, null]");
    for i in 0..=k {
        let surface = if i == 0 || i == k {
            outside
        } else if cut_r {
            s.surface("[0, 0]")
        } else {
            continue;
        };
        for l in 0..m {
            for j in 0..n {
                // Faces away from the axis as listed.
                let q = [
                    id(i, l, j),
                    id(i, l, j + 1),
                    id(i, l + 1, j + 1),
                    id(i, l + 1, j),
                ];
                s.quad(if i == 0 { [q[0], q[3], q[2], q[1]] } else { q }, surface);
            }
        }
    }
    for l in 0..=m {
        let surface = if l == 0 || l == m {
            outside
        } else if cut_z {
            s.surface("[0, 0]")
        } else {
            continue;
        };
        for i in 0..k {
            for j in 0..n {
                // Faces -z as listed.
                let q = [
                    id(i, l, j),
                    id(i, l, j + 1),
                    id(i + 1, l, j + 1),
                    id(i + 1, l, j),
                ];
                s.quad(if l == m { [q[0], q[3], q[2], q[1]] } else { q }, surface);
            }
        }
    }
    s
}

/// A torus shell of major radius `rr` between minor radii `a`, with `nt`
/// facets the short way round and `np` the long way. Its outside is surface
/// 1, wound outward; with `cut` each torus between the first and last minor
/// radius is an internal face.
fn torus_shell(rr: f64, a: &[f64], nt: u32, np: u32, cut: bool) -> Soup {
    let k = a.len() - 1;
    let mut s = Soup::default();
    for &ai in a {
        for p in 0..nt {
            let th = std::f64::consts::TAU * f64::from(p) / f64::from(nt);
            for q in 0..np {
                let ph = std::f64::consts::TAU * f64::from(q) / f64::from(np);
                let rho = rr + ai * th.cos();
                s.vertices
                    .push([rho * ph.cos(), rho * ph.sin(), ai * th.sin()]);
            }
        }
    }
    let id = |i: usize, p: u32, q: u32| (i as u32 * nt + p % nt) * np + q % np;
    let outside = s.surface("[0, null]");
    for i in 0..=k {
        let surface = if i == 0 || i == k {
            outside
        } else if cut {
            s.surface("[0, 0]")
        } else {
            continue;
        };
        for p in 0..nt {
            for q in 0..np {
                // Faces away from the core circle as listed.
                let f = [
                    id(i, p, q),
                    id(i, p, q + 1),
                    id(i, p + 1, q + 1),
                    id(i, p + 1, q),
                ];
                s.quad(if i == 0 { [f[0], f[3], f[2], f[1]] } else { f }, surface);
            }
        }
    }
    s
}

/// The cube [0, 30]^3, with the closed surface of [10, 20]^3 inside it as one
/// internal face when `internal` is set.
fn nested_cube(internal: bool) -> Soup {
    fn cube(s: &mut Soup, lo: f64, hi: f64, surface: u32) {
        let o = s.vertices.len() as u32;
        for i in 0..8 {
            let c = |bit: u32| if i & bit == 0 { lo } else { hi };
            s.vertices.push([c(1), c(2), c(4)]);
        }
        for q in [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ] {
            s.quad(q.map(|v| o + v), surface);
        }
    }
    let mut s = Soup::default();
    let outside = s.surface("[0, null]");
    cube(&mut s, 0.0, 30.0, outside);
    if internal {
        let face = s.surface("[0, 0]");
        cube(&mut s, 10.0, 20.0, face);
    }
    s
}

/// Rows of a rotation by a generic angle about a generic axis.
fn rotation() -> [P; 3] {
    let n = (0.31f64 * 0.31 + 0.52 * 0.52 + 0.79 * 0.79).sqrt();
    let [x, y, z] = [0.31 / n, -0.52 / n, 0.79 / n];
    let (s, c) = 0.7371f64.sin_cos();
    [
        [
            c + x * x * (1. - c),
            x * y * (1. - c) - z * s,
            x * z * (1. - c) + y * s,
        ],
        [
            y * x * (1. - c) + z * s,
            c + y * y * (1. - c),
            y * z * (1. - c) - x * s,
        ],
        [
            z * x * (1. - c) - y * s,
            z * y * (1. - c) + x * s,
            c + z * z * (1. - c),
        ],
    ]
}

/// Whether a value lies in `(lo, hi)`; `None` within `margin` of either end.
fn in_range(x: f64, lo: f64, hi: f64, margin: f64) -> Option<bool> {
    if x > lo + margin && x < hi - margin {
        Some(true)
    } else if x < lo - margin || x > hi + margin {
        Some(false)
    } else {
        None
    }
}

/// Whether `p` is in the box `[0, hi]`; `None` within `margin` of its faces.
fn in_box(p: P, hi: P, margin: f64) -> Option<bool> {
    let each = [0, 1, 2].map(|k| in_range(p[k], 0.0, hi[k], margin));
    if each.contains(&Some(false)) {
        Some(false)
    } else if each.contains(&None) {
        None
    } else {
        Some(true)
    }
}

/// A solid cut by internal faces, the same solid without them, the box
/// points are drawn from, and whether a point is in the solid: `None` within
/// round-off of its faceted outside, where either answer will do.
struct Case {
    name: &'static str,
    cut: Soup,
    control: Soup,
    lo: P,
    hi: P,
    inside: Box<dyn Fn(P) -> Option<bool>>,
}

fn lattice_case(name: &'static str, n: [usize; 3], h: P, cut: [bool; 3]) -> Case {
    let hi = [0, 1, 2].map(|k| n[k] as f64 * h[k]);
    Case {
        name,
        cut: lattice_box(n, h, cut),
        control: lattice_box(n, h, [false; 3]),
        lo: hi.map(|x| -0.05 * x),
        hi: hi.map(|x| 1.05 * x),
        inside: Box::new(move |p| in_box(p, hi, 1e-6 * hi[0].max(hi[1]).max(hi[2]))),
    }
}

/// A shell between radii `r` and levels `z`, as [`cylinder_shell`] makes.
fn cylinder_case(name: &'static str, r: &[f64], z: &[f64], cut_r: bool, cut_z: bool) -> Case {
    let (r0, r1, z0, z1) = (r[0], r[r.len() - 1], z[0], z[z.len() - 1]);
    Case {
        name,
        cut: cylinder_shell(r, z, 96, cut_r, cut_z),
        control: cylinder_shell(r, z, 96, false, false),
        lo: [-1.1 * r1, -1.1 * r1, z0 - 10.0],
        hi: [1.1 * r1, 1.1 * r1, z1 + 10.0],
        inside: Box::new(move |p| {
            match (
                in_range(p[0].hypot(p[1]), r0, r1, 0.5),
                in_range(p[2], z0, z1, 1e-4),
            ) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        }),
    }
}

fn cases() -> Vec<Case> {
    let r = rotation();
    let mut rotated = lattice_case("27-cell cube, turned", [3, 3, 3], [10.0; 3], [true; 3]);
    for s in [&mut rotated.cut, &mut rotated.control] {
        for v in &mut s.vertices {
            *v = r.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
        }
    }
    let corners = &rotated.control.vertices;
    rotated.lo =
        [0, 1, 2].map(|k| corners.iter().map(|v| v[k]).fold(f64::INFINITY, f64::min) - 3.0);
    rotated.hi = [0, 1, 2].map(|k| {
        corners
            .iter()
            .map(|v| v[k])
            .fold(f64::NEG_INFINITY, f64::max)
            + 3.0
    });
    rotated.inside = Box::new(move |p| {
        let back = [0, 1, 2].map(|k| r[0][k] * p[0] + r[1][k] * p[1] + r[2][k] * p[2]);
        in_box(back, [30.0; 3], 1e-5)
    });
    let torus = [200.0, 225.0, 250.0, 275.0, 300.0];
    vec![
        lattice_case(
            "box with one internal face",
            [1, 1, 2],
            [2.0, 2.0, 1.0],
            [false, false, true],
        ),
        lattice_case("octants of a box", [2, 2, 2], [1.0; 3], [true; 3]),
        lattice_case("27-cell cube", [3, 3, 3], [10.0; 3], [true; 3]),
        rotated,
        lattice_case(
            "100x100x40 slab in four layers",
            [1, 1, 4],
            [100.0, 100.0, 10.0],
            [false, false, true],
        ),
        lattice_case(
            "10x10x4 slab in four layers",
            [1, 1, 4],
            [10.0, 10.0, 1.0],
            [false, false, true],
        ),
        Case {
            name: "torus shell in four layers",
            cut: torus_shell(600.0, &torus, 32, 96, true),
            control: torus_shell(600.0, &torus, 32, 96, false),
            lo: [-920.0, -920.0, -320.0],
            hi: [920.0, 920.0, 320.0],
            // The facets lie within 2 cm of the tori they are cut from.
            inside: Box::new(move |p| {
                let minor = (p[0].hypot(p[1]) - 600.0).hypot(p[2]);
                in_range(minor, torus[0], torus[4], 3.0)
            }),
        },
        cylinder_case(
            "cylinder shell in three radial layers",
            &[100.0, 120.0, 140.0, 160.0],
            &[-100.0, 100.0],
            true,
            false,
        ),
        cylinder_case(
            "cylinder shell in four axial segments",
            &[100.0, 160.0],
            &[-100.0, -50.0, 0.0, 50.0, 100.0],
            false,
            true,
        ),
        Case {
            name: "cube with a closed internal surface",
            cut: nested_cube(true),
            control: nested_cube(false),
            lo: [-3.0; 3],
            hi: [33.0; 3],
            inside: Box::new(|p| in_box(p, [30.0; 3], 1e-5)),
        },
    ]
}

/// Deterministic uniform numbers in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn point(&mut self, lo: P, hi: P) -> P {
        [0, 1, 2].map(|k| lo[k] + (hi[k] - lo[k]) * self.next())
    }

    fn direction(&mut self) -> P {
        let cos = 2.0 * self.next() - 1.0;
        let (s, c) = (std::f64::consts::TAU * self.next()).sin_cos();
        let sin = (1.0 - cos * cos).sqrt();
        [sin * c, sin * s, cos]
    }
}

/// The points that some query puts in the wrong volume or cannot place,
/// given whether each is in volume 0.
fn misplaced(geom: &MeshGeometry, points: &[(P, bool)]) -> Vec<P> {
    let ic = geom.implicit_complement();
    let placed = |p: P, inside: bool| {
        let want = if inside { 0 } else { ic };
        geom.point_in_volume(0, p) == inside
            && geom.point_in_volume_checked(0, p) == Some(inside)
            && geom.find_volume(p) == want
            && [0, ic, u32::MAX].into_iter().all(|hint| {
                geom.try_find_volume_from(hint, p) == Some(want)
                    && geom.find_volume_from(hint, p) == want
            })
    };
    points
        .iter()
        .filter(|&&(p, inside)| !placed(p, inside))
        .map(|&(p, _)| p)
        .collect()
}

fn report(case: &Case, wrong: &[P], of: usize, failures: &mut Vec<String>) {
    if let Some(p) = wrong.first() {
        failures.push(format!(
            "{}: {} of {of} points misplaced, e.g. {p:?}",
            case.name,
            wrong.len()
        ));
    }
}

#[test]
fn every_point_is_placed_in_its_volume_whatever_internal_faces_cut_it() {
    let mut rng = Lcg(0x5eed);
    let mut failures = Vec::new();
    for case in cases() {
        let points: Vec<(P, bool)> = std::iter::repeat_with(|| rng.point(case.lo, case.hi))
            .filter_map(|p| (case.inside)(p).map(|inside| (p, inside)))
            .take(1500)
            .collect();
        let wrong = misplaced(&case.cut.geometry(), &points);
        report(&case, &wrong, points.len(), &mut failures);
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn a_point_on_an_internal_face_its_edges_or_corners_is_in_the_solid() {
    let mut failures = Vec::new();
    for case in cases() {
        let points: Vec<(P, bool)> = case
            .cut
            .internal_face_points()
            .into_iter()
            .filter(|&p| (case.inside)(p) == Some(true))
            .map(|p| (p, true))
            .collect();
        assert!(!points.is_empty(), "{}", case.name);
        let wrong = misplaced(&case.cut.geometry(), &points);
        report(&case, &wrong, points.len(), &mut failures);
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn internal_faces_change_no_answer_about_the_solid() {
    // The same solid without its internal faces answers the same: its
    // measure, and where a ray from a point in it or outside it next meets
    // its surface, by every query transport makes.
    let mut rng = Lcg(0xface);
    let mut failures = Vec::new();
    for case in cases() {
        let (geom, control) = (case.cut.geometry(), case.control.geometry());
        let topo = &geom.topology;
        // The topology keeps the internal faces, once each way.
        assert!(topo.volume_surfaces[0].contains(&(1, Sense::Forward)));
        assert!(topo.volume_surfaces[0].contains(&(1, Sense::Reverse)));
        if topo.volume_measures != control.topology.volume_measures
            || geom.measure_volume(0) != control.measure_volume(0)
        {
            failures.push(format!(
                "{}: measures {:?}, without the internal faces {:?}",
                case.name, topo.volume_measures, control.topology.volume_measures
            ));
        }
        let ic = geom.implicit_complement();
        let (mut asked, mut differ) = (0, Vec::new());
        while asked < 500 {
            let p = rng.point(case.lo, case.hi);
            let Some(inside) = (case.inside)(p) else {
                continue;
            };
            asked += 1;
            let (v, d) = (if inside { 0 } else { ic }, rng.direction());
            // Debug output prints each float exactly, so equal text is an
            // equal answer. A triangle is told by its corners: the internal
            // faces' triangles come between the outside's in the file.
            let answers = |g: &MeshGeometry| {
                let crossing = g.next_crossing(v, p, d).map(|(t, s, beyond)| match beyond {
                    Beyond::Relocate(tri) => {
                        let corners = g.topology.triangles[tri as usize];
                        (
                            t,
                            s,
                            format!("{:?}", corners.map(|i| g.topology.vertices[i as usize])),
                        )
                    }
                    Beyond::Volume(next) => (t, s, format!("{next:?}")),
                });
                [
                    format!("ray_fire {:?}", g.ray_fire(v, p, d, None)),
                    format!("ray_fire_leaving {:?}", g.ray_fire_leaving(v, p, d)),
                    format!("next_crossing {crossing:?}"),
                    format!("closest_to_surface {:?}", g.closest_to_surface(v, p)),
                ]
            };
            let (got, want) = (answers(&geom), answers(&control));
            if let Some(k) = (0..got.len()).find(|&k| got[k] != want[k]) {
                differ.push((p, d, got[k].clone(), want[k].clone()));
            }
        }
        if let Some((p, d, got, want)) = differ.first() {
            failures.push(format!(
                "{}: {} of {asked} queries answered otherwise, e.g. from {p:?} along {d:?}: \
                 {got}, without the internal faces {want}",
                case.name,
                differ.len()
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn an_internal_face_tagged_as_a_vacuum_boundary_is_no_way_out() {
    // Delta tracking ends a flight at the first vacuum-boundary facet ahead,
    // where surface tracking would have killed the particle. An internal face
    // bounds nothing, so whatever its tag it is not that facet: surface
    // tracking passes it by.
    let groups = r#"{"1": {"name": "mat:a", "dim": 3},
        "2": {"name": "boundary:vacuum", "dim": 2, "surface_ids": [1, 2]}}"#;
    let geom = lattice_box([1, 1, 2], [2.0, 2.0, 1.0], [false, false, true]).geometry_with(groups);
    let (p, up) = ([0.7, 0.9, 0.5], [0.0, 0.0, 1.0]);
    let (t, surface) = geom.ray_fire(0, p, up, None).unwrap();
    assert!((t - 1.5).abs() < 1e-12 && surface == 0, "{t} {surface}");
    let d = geom.distance_to_vacuum(p, up, 10.0).unwrap();
    assert!((d - 1.5).abs() < 1e-12, "{d}");
}
