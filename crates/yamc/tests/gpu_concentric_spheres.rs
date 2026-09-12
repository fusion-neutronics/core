//! GPU point-in-region cell finding for NESTED (concentric-shell) geometry.
//!
//! Three concentric spheres make four cells whose AABBs are fully nested:
//! the core's AABB sits inside shell 1's, shell 1's inside shell 2's, and so
//! on. A point in the core is inside every shell's AABB too, so AABB-only
//! cell finding (the GPU's original scheme) cannot tell the cells apart and
//! mis-assigns material + tally. The region path evaluates each cell's actual
//! CSG predicate (`-s_inner & +s_outer`), so a point resolves to the one cell
//! that genuinely contains it -- matching the CPU.
//!
//! Each shell is a DIFFERENT material (same nuclide, different densities, so
//! all share one energy grid -- the GPU material translation collapses
//! materials onto the first one's grid) so a mis-assignment shifts both the
//! per-cell flux and the macroscopic reaction rate, making a wrong cell
//! impossible to mistake for noise. The test asserts per-cell track-length
//! flux agrees between CPU and GPU within a breadth band.
//!
//! Needs a real f64 GPU adapter; self-skips otherwise (run single-threaded
//! via `cargo test-gpu`).

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
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
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const N_PER_BATCH: usize = 4_000;
const N_BATCHES: usize = 8;

/// A sphere of radius `r` at the origin. Outermost (id 4) is the vacuum
/// boundary; the inner shells are transmission surfaces.
fn sphere(id: usize, r: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        r,
        Some(id),
        Some(boundary),
    ))
}

/// Fe56 at a given density. Using one nuclide across all shells keeps a
/// single shared energy grid (the GPU translate collapses materials onto the
/// first's grid), while the differing density gives each shell a distinct
/// macroscopic cross-section -- so a mis-located cell shows up as a wrong
/// per-cell flux, not noise.
fn material(id: u32, density_g_cm3: f64) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density_g_cm3),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

/// Four nested cells: core (r < 1), shell 1 (1 < r < 2), shell 2 (2 < r < 3),
/// shell 3 (3 < r < 4, vacuum boundary). Materials differ per shell. The
/// cells are stored OUTERMOST-FIRST so that even with the AABB-only tie-break
/// the (broken) baseline would resolve to the outer cell -- making the
/// region-fix effect unambiguous: only the region test puts a core point in
/// the core cell.
fn build_geometry() -> (Geometry, [u32; 4]) {
    let s1 = sphere(1, 1.0, BoundaryType::Transmission);
    let s2 = sphere(2, 2.0, BoundaryType::Transmission);
    let s3 = sphere(3, 3.0, BoundaryType::Transmission);
    let s4 = sphere(4, 4.0, BoundaryType::Vacuum);

    let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s1)));
    let shell1 = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s1))).intersection(
        &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s2))),
    );
    let shell2 = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s2))).intersection(
        &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s3))),
    );
    let shell3 = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s3))).intersection(
        &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s4))),
    );

    // material_idx into the materials vec below: distinct Fe56 densities.
    let core_cell = Cell::new(Some(1), core, Some("core".into()), Some(0));
    let shell1_cell = Cell::new(Some(2), shell1, Some("shell1".into()), Some(1));
    let shell2_cell = Cell::new(Some(3), shell2, Some("shell2".into()), Some(2));
    let shell3_cell = Cell::new(Some(4), shell3, Some("shell3".into()), Some(3));

    let materials = vec![
        material(1, 1.0),
        material(2, 3.0),
        material(3, 5.0),
        material(4, 2.0),
    ];

    // Outermost-first ordering (see doc comment).
    let geometry = Geometry::new(
        vec![shell3_cell, shell2_cell, shell1_cell, core_cell],
        materials,
    )
    .unwrap();
    (geometry, [1u32, 2u32, 3u32, 4u32])
}

/// Isotropic 14 MeV point source at the centre so histories cross every
/// shell, exercising the per-step cell finding in all four cells.
fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn make_flux_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec!["flux".parse::<Score>().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.name = Some(format!("flux_cell_{cell_id}"));
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn model(geometry: Geometry, tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut m = Model::new(geometry, vec![neutron_source()], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: 42,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

fn run_cpu(geometry: Geometry, cell_ids: &[u32]) -> Vec<f64> {
    let tallies: Vec<Arc<Tally>> = cell_ids.iter().map(|&id| make_flux_tally(id)).collect();
    let (mut m, settings) = model(geometry, tallies.clone());
    m.simulate_transport(&settings).expect("CPU run");
    tallies.iter().map(|t| t.get_mean().iter().sum()).collect()
}

fn run_gpu(geometry: Geometry, cell_ids: &[u32]) -> Vec<f64> {
    let tallies: Vec<Arc<Tally>> = cell_ids.iter().map(|&id| make_flux_tally(id)).collect();
    let (mut m, settings) = model(geometry, tallies.clone());
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");
    tallies.iter().map(|t| t.get_mean().iter().sum()).collect()
}

fn assert_match(names: &[&str], cpu: &[f64], gpu: &[f64]) {
    let mut bad = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let (c, g) = (cpu[i], gpu[i]);
        assert!(
            g.is_finite() && g >= 0.0,
            "{name}: GPU value not finite/non-negative ({g})"
        );
        let r = g / c;
        if !(0.85..=1.18).contains(&r) {
            bad.push(format!(
                "{name}: GPU/CPU ratio {r:.3} out of [0.85, 1.18] (CPU {c:.5e}, GPU {g:.5e})"
            ));
        }
        eprintln!("  {name}: CPU {c:.5e}  GPU {g:.5e}  ratio {r:.3}");
    }
    assert!(
        bad.is_empty(),
        "GPU disagrees with CPU:\n  {}",
        bad.join("\n  ")
    );
}

/// Nested concentric shells: the GPU must reproduce the CPU per-cell flux.
/// The AABB-only cell finding fails here (every shell's AABB contains the
/// core point); the region path passes.
#[test]
fn gpu_concentric_spheres_flux_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let cell_ids = [1u32, 2u32, 3u32, 4u32];
    let (geom_cpu, _) = build_geometry();
    let (geom_gpu, _) = build_geometry();
    let cpu = run_cpu(geom_cpu, &cell_ids);
    let gpu = run_gpu(geom_gpu, &cell_ids);

    // Every shell must see real traversal flux on both backends -- the
    // isotropic source streams outward through all four cells.
    for (i, v) in cpu.iter().enumerate() {
        assert!(*v > 0.0, "CPU cell {} flux must be > 0, got {v}", i + 1);
    }
    for (i, v) in gpu.iter().enumerate() {
        assert!(*v > 0.0, "GPU cell {} flux must be > 0, got {v}", i + 1);
    }

    assert_match(
        &["core_flux", "shell1_flux", "shell2_flux", "shell3_flux"],
        &cpu,
        &gpu,
    );
}
