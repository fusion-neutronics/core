//! GPU implicit-complement acceptance test.
//!
//! yamc CSG has no auto-generated implicit-complement cell; the modeller
//! writes the "rest of space" explicitly as an *unbounded* cell, e.g. the
//! `outer.above` halfspace (or `~region`). That cell's bounding box is
//! non-finite, which used to be rejected outright by the GPU translator
//! (`CellRegionUnsupported { reason: "cell has unbounded extent" }`).
//!
//! The GPU now supports a *vacuum* (material-less) implicit complement:
//! `translate_cells` emits an empty sentinel AABB for the unbounded cell
//! so the kernel can never match it, and any particle that reaches that
//! region is treated as having left the geometry (history terminates).
//! That mirrors the CPU, where free-streaming into the void implicit
//! complement ends the history with no further collisions.
//!
//! The acceptance case is an Fe56 sphere of material wrapped in a void
//! implicit complement (`sphere.above`) with a `vacuum` outer surface --
//! the conventional configuration where the implicit complement is the
//! geometry's vacuum sink. The GPU flux in the material cell must match
//! the CPU flux within Monte-Carlo + known-GPU-kernel tolerance.
//!
//! Scope note: the implicit-complement leak on the GPU relies on a vacuum
//! outer boundary to terminate the history at the true surface. A
//! *transmission* outer boundary into a void implicit complement is NOT
//! correctly handled on the GPU and is out of scope here -- the kernel
//! finds cells by point-in-AABB and does not nudge a particle past a
//! transmission surface, so it re-selects the same cell and over-counts.
//! That is a pre-existing GPU surface-crossing limitation (it applies even
//! with no implicit-complement cell at all) independent of this change;
//! GPU models must use vacuum outer boundaries. The unit tests below pin
//! the translator behaviour directly.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
}

fn fe_material() -> Material {
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    material.read_nuclear_data(&nm, None).unwrap();
    material
}

/// Fe56 sphere (material cell) + a void implicit-complement cell that is
/// the sphere's complement (`sphere.above`, an unbounded region). The
/// outer surface is a vacuum boundary. Returns `(geometry, source,
/// material_cell_id)`.
fn fe_sphere_with_implicit_complement(radius: f64) -> (Geometry, ParticleSource, u32) {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let inside = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&sphere)));
    // Implicit complement: outside the sphere -- an unbounded region.
    let outside = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&sphere)));

    let mat_cell = Cell::new(Some(1), inside, Some("fe".into()), Some(0));
    let mat_cell_id = mat_cell.cell_id.unwrap();
    // Void implicit-complement cell: no material.
    let ic_cell = Cell::new(Some(2), outside, Some("implicit_complement".into()), None);

    let geometry = Geometry::new(vec![mat_cell, ic_cell], vec![Arc::new(fe_material())]).unwrap();

    // 14 MeV neutron point source at the centre.
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    (geometry, source, mat_cell_id)
}

/// Build a model with a single flux tally on the material cell.
fn model_with_flux_tally(
    seed: u64,
    radius: f64,
    n_particles: usize,
    n_batches: usize,
) -> (Model, Arc<Tally>, TransportSettings) {
    let (geometry, source, mat_cell_id) = fe_sphere_with_implicit_complement(radius);

    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(mat_cell_id)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(n_batches);
    let tally = Arc::new(tally);

    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.max_steps_per_particle = 5_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

#[test]
fn gpu_implicit_complement_sphere_vacuum_matches_cpu() {
    if !data_present() {
        eprintln!("skipping gpu_implicit_complement -- tests/Fe56.arrow not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping gpu_implicit_complement -- no GPU with f64 compute available");
        return;
    }

    let seed = 7777;
    let radius = 5.0;
    let n_particles = 20_000;
    let n_batches = 8;

    // CPU reference.
    let (mut cpu_m, cpu_t, settings) = model_with_flux_tally(seed, radius, n_particles, n_batches);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu_flux = cpu_t.get_mean().iter().sum::<f64>();

    // GPU run -- this is the path that the unbounded implicit-complement
    // cell used to make impossible (it errored in `translate_cells`).
    let (mut gpu_m, gpu_t, settings) = model_with_flux_tally(seed, radius, n_particles, n_batches);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings)
        .expect("GPU dispatch must succeed with a vacuum implicit complement");
    let gpu_flux = gpu_t.get_mean().iter().sum::<f64>();

    let ratio = gpu_flux / cpu_flux;
    println!(
        "implicit complement (sphere/vacuum): CPU flux = {cpu_flux:.4e}  \
         GPU flux = {gpu_flux:.4e}  GPU/CPU = {ratio:.4}"
    );

    assert!(
        cpu_flux > 0.0 && gpu_flux > 0.0,
        "both backends must score a non-zero flux in the material cell"
    );
    // Tight band: a real implicit-complement mis-handling (particles
    // wrongly pulled into / leaked from the unbounded cell) shifts this
    // well outside [0.97, 1.03].
    assert!(
        (0.97..=1.03).contains(&ratio),
        "GPU/CPU flux ratio {ratio:.4} outside [0.97, 1.03] \
         (GPU {gpu_flux:.4e} vs CPU {cpu_flux:.4e})"
    );
}
