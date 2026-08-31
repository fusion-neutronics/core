//! Scale-up regression for the concentric-shell lost-particle bug
//! (issue #223, fixed by #226).
//!
//! With the source inside a scattering MATERIAL cell bounded by
//! transmission surfaces to further material shells, near-tangent
//! boundary crossings used to mishandle the cell handoff and lose
//! ~0.1% of particles -- enough to abort at the default
//! max_lost_particles once particle counts exceeded ~1000. The
//! existing 2-shell tests ran only 100 particles and never tripped it.
//! This runs the original reproducer topology at 100k particles in all
//! three tracking modes; completion (no lost-particle abort) is the
//! assertion. Weight-window isosurfaces (WW PR4) depend on this
//! staying fixed.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};

fn li6_material(id: u32) -> Arc<Material> {
    let mut mat = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    mat.set_material_id(id);
    mat.set_temperature("294");
    let map = HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]);
    mat.read_nuclear_data(&map, None).unwrap();
    Arc::new(mat)
}

fn sphere(id: usize, radius: f64, vacuum: bool) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: if vacuum {
            BoundaryType::Vacuum
        } else {
            BoundaryType::Transmission
        },
        name: None,
    })
}

/// N concentric Li6 shells out to r=50, all material (the #223 trigger
/// needs the source inside a scattering material cell).
fn nested_shells_geometry(n_shells: usize) -> Geometry {
    let spheres: Vec<Arc<Surface>> = (1..=n_shells)
        .map(|i| {
            let r = 50.0 * i as f64 / n_shells as f64;
            sphere(i, r, i == n_shells)
        })
        .collect();

    let mut cells = vec![Cell::new(
        Some(1),
        Region {
            expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
                spheres[0].clone(),
            )))),
        },
        Some("shell_0".to_string()),
        Some(0),
    )];
    for i in 1..n_shells {
        cells.push(Cell::new(
            Some((i + 1) as u32),
            Region {
                expr: RegionExpr::Intersection(
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
                        spheres[i - 1].clone(),
                    ))),
                    Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                        HalfspaceType::Above(spheres[i].clone()),
                    )))),
                ),
            },
            Some(format!("shell_{i}")),
            Some(i as u32),
        ));
    }
    let materials = (1..=n_shells).map(|i| li6_material(i as u32)).collect();
    Geometry::new(cells, materials).unwrap()
}

fn make_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

#[test]
fn nested_material_shells_lose_no_particles_at_scale() {
    // 100k particles trips the historical ~0.1% loss rate hundreds of
    // times over the default max_lost_particles=10; simulate_transport
    // returning Ok IS the assertion.
    let settings = TransportSettings {
        total_particles: Some(100_000),
        seed: 42,
        ..Default::default()
    };
    for n_shells in [3, 5] {
        for mode in [
            TrackingMode::Surface,
            TrackingMode::Woodcock,
            TrackingMode::Hybrid,
        ] {
            let mut model = Model::new(
                nested_shells_geometry(n_shells),
                vec![make_source()],
                vec![],
            );
            model.verbose = yamc::model::Verbose::silent();
            model.tracking_mode = mode;
            model.simulate_transport(&settings).unwrap_or_else(|e| {
                panic!("{n_shells} shells / {mode:?}: lost-particle regression (#223): {e}")
            });
        }
    }
}
