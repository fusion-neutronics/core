//! Regression: GPU photon transport must resolve nested / overlapping cell
//! AABBs with the exact CSG region test, not the bounding box alone.
//!
//! In a geometry of concentric shells, every shell's cell AABB is the cube of
//! its OUTER sphere, so the boxes are deeply nested and many contain the same
//! point. The photon cell-find previously used an AABB-only membership test
//! (the `region_contains` conjunct the neutron kernel applies had been dropped),
//! so a photon far from the source was mis-assigned to an inner cell whose
//! bounding cube still contained the point. That inner cell's nonzero sigma_t
//! triggered fake collisions in what should be void, and the photon flux in
//! distant cells collapsed with radius (~0.4x at 25 cm down to ~0.02x at 100 cm).
//!
//! This runs a direct 2 MeV photon point source through a steel core, void
//! gaps, and a distant Be detector shell, and asserts the GPU photon flux there
//! matches the CPU. Without the fix the ratio is ~0.06; with it, ~1.0.
//!
//! Self-skips without the bundled Fe/Be data or an f64 GPU. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_nested_cells -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;

fn data_present() -> bool {
    ["Fe56", "Fe", "Be9", "Be"]
        .iter()
        .all(|n| std::path::Path::new(&format!("tests/{n}.arrow")).is_dir())
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn material(nuclide: &str, elem: &str, density: f64, id: u32) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    let nm = HashMap::from([(nuclide.to_string(), format!("tests/{nuclide}.arrow"))]);
    let pp = HashMap::from([(elem.to_string(), format!("tests/{elem}.arrow"))]);
    m.read_nuclear_data(&nm, Some(&pp)).unwrap();
    m.init_photon_data(&pp).unwrap();
    Arc::new(m)
}

fn sphere(id: usize, r: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r,
        },
        boundary,
        name: None,
    })
}

/// Concentric geometry: steel core (r<15), void gap, Be detector shell
/// (r=95..98) far from the source, vacuum boundary at r=105. The nested cell
/// AABBs are what previously broke the AABB-only photon cell-find.
fn nested_geometry() -> (Geometry, u32) {
    let steel = material("Fe56", "Fe", 7.874, 1);
    let be = material("Be9", "Be", 1.85, 2);

    let s_core = sphere(1, 15.0, BoundaryType::Transmission);
    let s_gap = sphere(2, 95.0, BoundaryType::Transmission);
    let s_det = sphere(3, 98.0, BoundaryType::Transmission);
    let s_bnd = sphere(4, 105.0, BoundaryType::Vacuum);

    // material_idx indexes the materials vec: 0 = steel, 1 = Be; None = void.
    let core = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s_core))),
        Some("core".into()),
        Some(0),
    );
    let gap = Cell::new(
        Some(2),
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s_core))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s_gap))),
        ),
        Some("gap".into()),
        None,
    );
    let det = Cell::new(
        Some(3),
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s_gap))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s_det))),
        ),
        Some("det".into()),
        Some(1),
    );
    let outer = Cell::new(
        Some(4),
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s_det))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s_bnd))),
        ),
        Some("outer".into()),
        None,
    );
    let det_id = det.cell_id.unwrap();
    let geo = Geometry::new(vec![core, gap, det, outer], vec![steel, be]).unwrap();
    (geo, det_id)
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![2.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

fn build(det_id: u32) -> (Model, Arc<Tally>, TransportSettings) {
    let (geo, _) = nested_geometry();
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(det_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Photon,
    )));
    t.scores = vec!["flux".parse().unwrap()];
    t.initialize_batches(10);
    let tally = Arc::new(t);
    let mut model = Model::new(geo, vec![photon_source()], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 5_000;
    model.tracking_mode = TrackingMode::Surface;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(200_000),
        seed: 24680,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

#[test]
fn gpu_cpu_distant_photon_flux_through_nested_cells() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping gpu_cpu_distant_photon_flux_through_nested_cells: data / GPU absent");
        return;
    }
    let (_, det_id) = nested_geometry();

    let (mut cpu_m, cpu_t, cs) = build(det_id);
    cpu_m.simulate_transport(&cs).expect("CPU run failed");
    let cpu: f64 = cpu_t.get_mean().iter().sum();

    let (mut gpu_m, gpu_t, gs) = build(det_id);
    yamc::gpu::run_on_gpu(&mut gpu_m, &gs).expect("GPU run failed");
    let gpu: f64 = gpu_t.get_mean().iter().sum();

    assert!(cpu > 0.0, "CPU distant photon flux is zero -- setup wrong");
    let ratio = gpu / cpu;
    eprintln!("distant (r=95-98) photon flux through nested cells: CPU={cpu:.5e} GPU={gpu:.5e} ratio={ratio:.4}");
    // Without the region_contains fix the GPU collapses to ~0.06x; the exact
    // membership test restores parity. Band covers the approximate GPU photon
    // kernel systematic.
    assert!(
        (0.85..=1.15).contains(&ratio),
        "GPU/CPU distant photon flux ratio {ratio:.4} outside [0.85, 1.15] -- \
         nested-cell photon cell-find regressed (AABB-only membership?)"
    );
}
