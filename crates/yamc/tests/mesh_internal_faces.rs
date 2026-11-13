//! Transport through a solid cut into parts by internal faces: surfaces with
//! the solid on both sides (`[v, v]` in `surface_volumes`), as a file from
//! another writer can have (issues #517 and #533).
//!
//! Point location could not place a point where its every test ray met an
//! internal face first, so a particle born there, or landing there after a
//! delta-tracking flight, was lost: every one born in a cell enclosed by
//! internal faces, and up to nine in ten in a torus shell in four layers. Now
//! no particle is lost, and the flux is that of the same solid without the
//! internal faces, in surface, hybrid and Woodcock tracking and in a mesh
//! filling a CSG cell, which used to refuse such a mesh as not watertight. A
//! plot takes the solid as the same solid without them too. The transport
//! tests self-skip if the endf-b8.1 Fe56 fixture is absent.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::conversion::mesh_geometry_to_geo_mesh;
use yamc::geometry::fill::CellFillSpec;
use yamc::geometry::mesh::MeshGeometry;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::MaterialFilter;
use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData};

type P = [f64; 3];
const NUCLIDE: &str = "Fe56";
const HISTORIES: usize = 2000;

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

fn iron() -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let nm = HashMap::from([(NUCLIDE.to_string(), yamc_test_cache::nuclide_path(NUCLIDE))]);
    m.read_nuclear_data(&nm, None).unwrap();
    Arc::new(m)
}

/// One iron volume as a triangle soup, with the `surface_volumes` entry of
/// each of its surfaces and the surfaces tagged as vacuum boundaries.
#[derive(Default)]
struct Soup {
    vertices: Vec<P>,
    triangles: Vec<[u32; 3]>,
    surface_ids: Vec<u32>,
    surface_volumes: Vec<&'static str>,
    vacuum: Vec<u32>,
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

    /// The same soup with its internal faces tagged as vacuum boundaries.
    fn internal_faces_vacuum(mut self) -> Self {
        self.vacuum = (1..=self.surface_volumes.len() as u32)
            .filter(|&s| self.surface_volumes[s as usize - 1] == "[0, 0]")
            .collect();
        self
    }

    /// The solid, iron.
    fn mesh(&self, iron: &Arc<Material>) -> MeshGeometry {
        let materials = HashMap::from([("iron".to_string(), iron.clone())]);
        MeshGeometry::new(self.yamt(), &materials, None).unwrap()
    }

    fn yamt(&self) -> yamt::MeshGeometry {
        let iron = r#""1": {"name": "mat:iron", "dim": 3}"#;
        let vacuum = format!(
            r#""2": {{"name": "boundary:vacuum", "dim": 2, "surface_ids": {:?}}}"#,
            self.vacuum
        );
        let data = ArrowMeshData {
            triangle_aabbs: build_triangle_aabbs(&self.triangles, &self.vertices),
            tet_aabbs: build_tet_aabbs(&[], &self.vertices),
            triangle_physical_groups: vec![1; self.triangles.len()],
            triangle_surface_ids: self.surface_ids.clone(),
            triangles: self.triangles.clone(),
            vertices: self.vertices.clone(),
            physical_groups_json: if self.vacuum.is_empty() {
                format!("{{{iron}}}")
            } else {
                format!("{{{iron}, {vacuum}}}")
            },
            surface_volumes_json: format!("[{}]", self.surface_volumes.join(", ")),
            imprinted: Some(true),
            ..Default::default()
        };
        yamt::MeshGeometry::from_topology(build_topology(data).unwrap())
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

/// A torus shell of major radius 600 cm between minor radii 200 and 300 cm,
/// with 32 facets the short way round and 96 the long way. Its outside is
/// surface 1, wound outward; with `cut` it is in four layers, the tori
/// between them internal faces.
fn torus_shell(cut: bool) -> Soup {
    let (nt, np) = (32u32, 96u32);
    let a = [200.0, 225.0, 250.0, 275.0, 300.0];
    let mut s = Soup::default();
    for ai in a {
        for p in 0..nt {
            let th = std::f64::consts::TAU * f64::from(p) / f64::from(nt);
            for q in 0..np {
                let ph = std::f64::consts::TAU * f64::from(q) / f64::from(np);
                let rho = 600.0 + ai * th.cos();
                s.vertices
                    .push([rho * ph.cos(), rho * ph.sin(), ai * th.sin()]);
            }
        }
    }
    let id = |i: u32, p: u32, q: u32| (i * nt + p % nt) * np + q % np;
    let outside = s.surface("[0, null]");
    for i in 0..5 {
        let surface = if i == 0 || i == 4 {
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
}

fn point_source(p: P) -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new(p)),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn flux_tally(iron: &Arc<Material>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters = vec![Filter::Material(MaterialFilter::new(iron).unwrap())];
    t.scores = vec![Score::Flux(FluxScore)];
    t.reset_accumulation();
    Arc::new(t)
}

/// Lost histories and the flux in the iron. Every loss is counted rather
/// than ending the run, and one thread adds the scores in a fixed order, so
/// the same histories give the same flux to the last bit.
fn run(mut model: Model, tally: &Tally, mode: TrackingMode) -> (usize, f64) {
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(HISTORIES),
            seed: 1,
            threads: Some(1),
            ..Default::default()
        })
        .unwrap();
    (model.lost_particles.len(), tally.get_mean()[0])
}

#[test]
fn transport_through_internal_faces_loses_nothing_and_scores_as_without_them() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let mut rng = Lcg(7);
    let mut uniform = |sample: &mut dyn FnMut(&mut Lcg) -> P| -> Vec<ParticleSource> {
        (0..400).map(|_| point_source(sample(&mut rng))).collect()
    };
    let slab = uniform(&mut |r| [100.0 * r.next(), 100.0 * r.next(), 40.0 * r.next()]);
    let blanket = uniform(&mut |r| {
        let minor = 201.0 + 98.0 * r.next();
        let (th, ph) = (
            std::f64::consts::TAU * r.next(),
            std::f64::consts::TAU * r.next(),
        );
        let rho = 600.0 + minor * th.cos();
        [rho * ph.cos(), rho * ph.sin(), minor * th.sin()]
    });
    type Mesh = fn(bool) -> Soup;
    let cases: [(&str, Mesh, Vec<ParticleSource>); 5] = [
        (
            "27-cell cube, source in the centre cell",
            |cut| lattice_box([3, 3, 3], [10.0; 3], [cut; 3]),
            vec![point_source([15.0; 3])],
        ),
        (
            "100x100x40 slab in four layers, uniform source",
            |cut| lattice_box([1, 1, 4], [100.0, 100.0, 10.0], [false, false, cut]),
            slab.clone(),
        ),
        // A tag on a face that bounds nothing is no way out: the slab is
        // closed by the model's vacuum box, as without the faces.
        (
            "the same slab, its internal faces tagged as vacuum boundaries",
            |cut| {
                lattice_box([1, 1, 4], [100.0, 100.0, 10.0], [false, false, cut])
                    .internal_faces_vacuum()
            },
            slab,
        ),
        (
            "torus shell in four layers, uniform source",
            torus_shell,
            blanket,
        ),
        (
            "cube with a closed internal surface, source inside it",
            nested_cube,
            vec![point_source([13.7, 15.2, 16.1])],
        ),
    ];
    let fe = iron();
    let mut failures = Vec::new();
    for (name, mesh, sources) in &cases {
        for mode in [
            TrackingMode::Surface,
            TrackingMode::Hybrid,
            TrackingMode::Woodcock,
        ] {
            let [(lost, flux), (control_lost, control_flux)] = [true, false].map(|cut| {
                let tally = flux_tally(&fe);
                let geometry = mesh(cut).mesh(&fe);
                let model = Model::new_with_mesh(geometry, sources.clone(), vec![tally.clone()]);
                run(model, &tally, mode)
            });
            assert_eq!(
                control_lost, 0,
                "{name}, {mode:?}: the control loses histories"
            );
            if lost != 0 || flux != control_flux {
                failures.push(format!(
                    "{name}, {mode:?}: {lost} of {HISTORIES} histories lost, flux {flux:.6e} \
                     against {control_flux:.6e} without the internal faces"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn a_mesh_with_internal_faces_fills_a_csg_cell_as_one_without_them() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    // The 27-cell cube in a void sphere with a vacuum boundary, the source in
    // its centre cell. The fill used to be refused: each edge of an internal
    // face is met four times, so the solid did not read as watertight.
    let fe = iron();
    let fluxes = [true, false].map(|cut| {
        let sphere = Arc::new(Surface {
            surface_id: None,
            kind: SurfaceKind::Sphere {
                x0: 15.0,
                y0: 15.0,
                z0: 15.0,
                radius: 40.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        });
        let host = Cell::new(
            Some(1),
            Region::new_from_halfspace(HalfspaceType::Below(sphere)),
            Some("chamber".to_string()),
            None,
        );
        let geometry = Geometry::new_with_fills(
            vec![host],
            vec![],
            vec![CellFillSpec {
                host_cell_index: 0,
                mesh_geometry: lattice_box([3, 3, 3], [10.0; 3], [cut; 3]).mesh(&fe),
                translation: [0.0; 3],
                rotation_degrees: [0.0; 3],
                allow_clipping: false,
            }],
        )
        .unwrap_or_else(|e| panic!("internal faces {cut}: {e}"));
        let tally = flux_tally(&fe);
        let model = Model::new(geometry, vec![point_source([15.0; 3])], vec![tally.clone()]);
        run(model, &tally, TrackingMode::Surface)
    });
    assert_eq!(fluxes[1].0, 0);
    assert_eq!(
        fluxes[0], fluxes[1],
        "(lost, flux) with internal faces and without: {fluxes:?}"
    );
}

#[test]
fn a_plot_takes_the_solid_as_bounded_by_its_outside_alone() {
    // The browser viewer places points by parity over each volume's
    // triangles, which listed an internal face twice, once each way.
    let [cut, control] = [true, false].map(|cut| {
        let solid = lattice_box([3, 3, 3], [10.0; 3], [cut; 3]).yamt();
        mesh_geometry_to_geo_mesh(&MeshGeometry::new(solid, &HashMap::new(), None).unwrap())
    });
    assert_eq!(cut.volumes[0].triangles, control.volumes[0].triangles);
}
