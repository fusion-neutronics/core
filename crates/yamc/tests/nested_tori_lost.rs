//! Nested tori lose no particles at their surfaces.
//!
//! Four nested z-tori of major radius 907.2 cm and minor radii 317.7 to
//! 467.7 cm, iron between them and a vacuum boundary on the outermost,
//! with 14.1 MeV neutrons born on the ring of the major radius. The ray-
//! torus distance used to lose about 2% of these histories ("Surface
//! crossed: unknown"), in every tracking mode: its quartic was solved
//! in absolute units, where the coefficients span twelve decades, by a
//! closed-form solver with absolute thresholds, and a particle that had
//! just crossed a torus could read that crossing as a tiny positive
//! root. 4000 histories expect about 80 losses at that rate, so a
//! single loss now fails the test.
//!
//! The elliptical variant (axial minor radii 600 to 750 cm) lost a
//! little more. The GPU kernel carries its own port of the torus
//! distance, checked here too when built with `--features gpu` and an
//! f64 GPU is present. Self-skips if the endf-b8.1 Fe56 fixture is
//! absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const NUCLIDE: &str = "Fe56";
const MAJOR_RADIUS: f64 = 907.2;
const PARTICLES: usize = 4000;

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

fn iron() -> Material {
    let mut material = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.8),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(NUCLIDE.to_string(), yamc_test_cache::nuclide_path(NUCLIDE))]);
    material.read_nuclear_data(&nm, None).unwrap();
    material
}

/// Four nested z-tori; `axial` is the axial minor radius of the
/// innermost (equal to its radial minor radius for circular tori),
/// growing by 50 cm per torus like the radial one.
fn geometry(axial: f64) -> Geometry {
    let surfaces: Vec<Arc<Surface>> = (0..4)
        .map(|i| {
            let grow = 50.0 * i as f64;
            Arc::new(Surface {
                surface_id: Some(i + 1),
                kind: SurfaceKind::ZTorus {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    a: MAJOR_RADIUS,
                    b: axial + grow,
                    c: 317.7 + grow,
                },
                boundary: if i == 3 {
                    BoundaryType::Vacuum
                } else {
                    BoundaryType::Transmission
                },
                name: None,
            })
        })
        .collect();
    let mut cells = vec![Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(surfaces[0].clone())),
        Some("inner".into()),
        None,
    )];
    for i in 0..3 {
        let region = Region::new_from_halfspace(HalfspaceType::Above(surfaces[i].clone()))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                surfaces[i + 1].clone(),
            )));
        cells.push(Cell::new(
            Some(i as u32 + 2),
            region,
            Some(format!("shell{i}")),
            Some(0),
        ));
    }
    Geometry::new(cells, vec![Arc::new(iron())]).unwrap()
}

fn model(axial: f64, mode: TrackingMode) -> Model {
    // A point on the source ring: by the tori's symmetry its histories
    // are those of the whole ring.
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([MAJOR_RADIUS, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry(axial), vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    model
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(PARTICLES),
        seed: 1,
        threads: Some(8),
        ..Default::default()
    }
}

fn lost(axial: f64, mode: TrackingMode) -> usize {
    let mut model = model(axial, mode);
    model.simulate_transport(&settings()).unwrap();
    model.lost_particles.len()
}

#[test]
fn nested_tori_lose_no_particles() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for (shape, axial) in [("circular", 317.7), ("elliptical", 600.0)] {
        for mode in [
            TrackingMode::Surface,
            TrackingMode::Hybrid,
            TrackingMode::Woodcock,
        ] {
            let n = lost(axial, mode);
            assert_eq!(
                n, 0,
                "{shape} tori, {mode:?}: {n} of {PARTICLES} histories lost"
            );
        }
    }
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
#[test]
fn nested_tori_lose_no_particles_on_gpu() {
    if !data_present() || yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: {NUCLIDE} fixture or f64 GPU absent");
        return;
    }
    for (shape, axial) in [("circular", 317.7), ("elliptical", 600.0)] {
        let mut model = model(axial, TrackingMode::Surface);
        let run = yamc::gpu::run_on_gpu(&mut model, &settings()).expect("GPU run");
        assert_eq!(
            run.lost_count, 0,
            "{shape} tori on the GPU: {} of {PARTICLES} histories lost",
            run.lost_count
        );
    }
}
