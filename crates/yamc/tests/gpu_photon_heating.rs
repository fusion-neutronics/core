//! GPU photon heating / KERMA scoring parity test.
//!
//! Companion to `gpu_photon_co60.rs` (flux) and
//! `gpu_photon_components.rs` (component XS). This exercises the
//! `Score::Heating` photon tally, which routes through `SCORE_PER_MT`
//! on the photon kernel with `Mt::HEATING` (301). The kernel reads the
//! per-material macroscopic heating / KERMA array
//! (`heating_xs_per_material`) built by
//! `extract_photon_material_xs` -- a density-weighted sum of each
//! element's `ElementMicroXS.heating`, the SAME aggregation the CPU
//! `Material::calculate_photon_xs().heating` uses (issue #356, the CPU
//! track-length photon-heating estimate). Because GPU and CPU draw
//! from the identical heating table and interpolate it the same way,
//! the heating tally should agree to within MC noise + the kernel's
//! transport approximations (Compton sampler, secondary cascade).
//!
//! A coexisting flux tally checks that adding the heating score leaves
//! the flux path unaffected.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
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
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, HeatingScore, Score};
use yamc_tallies::tally::Tally;

/// Build a Fe sphere with a centred monoenergetic photon point source
/// and two cell-only photon tallies: flux and heating. Mirrors
/// `gpu_photon_co60.rs` / `gpu_photon_components.rs` setup exactly.
fn fe_sphere_with_heating(
    seed: u64,
    radius: f64,
    source_energy: f64,
) -> (Model, Arc<Tally>, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_energy], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let n_particles = 20_000;
    let n_batches = 4;

    let mk = |score: Score| {
        let mut t = Tally::new();
        t.filters
            .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
        t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
            yamc_particle::particle::ParticleType::Photon,
        )));
        t.scores = vec![score];
        t.initialize_batches(n_batches);
        Arc::new(t)
    };
    let flux = mk(Score::Flux(FluxScore));
    let heating = mk(Score::Heating(HeatingScore));

    let mut model = Model::new(
        geometry,
        vec![source],
        vec![Arc::clone(&flux), Arc::clone(&heating)],
    );
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, flux, heating, settings)
}

#[test]
fn gpu_photon_heating_fe_sphere_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // 1.25 MeV -- Co60-mean, same as the parent tests. Compton
    // dominates; the recoil-electron energy deposited locally drives
    // the heating tally. Both CPU and GPU read the identical
    // density-weighted heating / KERMA table for Fe.
    let source_e = 1.25e6_f64;

    let (mut cpu_m, cpu_flux, cpu_heat, settings) = fe_sphere_with_heating(42, 1.0, source_e);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu_flux_v = cpu_flux.get_mean().iter().sum::<f64>();
    let cpu_heat_v = cpu_heat.get_mean().iter().sum::<f64>();

    let (mut gpu_m, gpu_flux, gpu_heat, settings) = fe_sphere_with_heating(42, 1.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu_flux_v = gpu_flux.get_mean().iter().sum::<f64>();
    let gpu_heat_v = gpu_heat.get_mean().iter().sum::<f64>();

    let heat_ratio = gpu_heat_v / cpu_heat_v;
    let flux_ratio = gpu_flux_v / cpu_flux_v;
    eprintln!(
        "Fe r=1 ({source_e:.2e} eV) heating: CPU = {cpu_heat_v:.4e}  GPU = {gpu_heat_v:.4e}  ratio = {heat_ratio:.3}"
    );
    eprintln!(
        "Fe r=1 ({source_e:.2e} eV) flux:    CPU = {cpu_flux_v:.4e}  GPU = {gpu_flux_v:.4e}  ratio = {flux_ratio:.3}"
    );

    // Both sides score from the same heating / KERMA table, so the
    // ratio is bounded by MC noise plus the transport approximations
    // (Kahn vs incoherent-form-factor Compton, secondary cascade
    // ordering). The heating tally must be clearly non-zero -- the
    // pre-fix kernel returned a flat 0 for the HEATING MT.
    assert!(
        cpu_heat_v > 0.0 && gpu_heat_v > 0.0,
        "photon heating must be non-zero on both paths (CPU = {cpu_heat_v:.4e}, GPU = {gpu_heat_v:.4e})"
    );
    assert!(
        (0.85..1.20).contains(&heat_ratio),
        "Fe r=1 GPU/CPU photon heating ratio {heat_ratio:.3} outside [0.85, 1.20] -- \
         photon heating scoring may have regressed"
    );

    // The flux tally must be unaffected by the coexisting heating
    // score (heating is additive and MT-gated). Same band as
    // `gpu_photon_co60`.
    assert!(
        (0.75..1.25).contains(&flux_ratio),
        "Fe r=1 GPU/CPU photon flux ratio {flux_ratio:.3} outside [0.75, 1.25] -- \
         coexisting heating tally should not perturb flux"
    );
}
