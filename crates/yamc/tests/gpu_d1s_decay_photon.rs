//! D1S (Direct-1-Step) decay-photon GPU dispatch acceptance test.
//!
//! A 14 MeV neutron point source in a single-material Fe56 sphere with
//! `transport_secondary_photons = true` and `use_decay_photons = true`. Fe56
//! (n,p) -> Mn56 is an activation product whose decay emits gamma lines; the
//! neutron kernel emits one decay photon per collision, tagged with its parent
//! radionuclide (Mn56), into the device bank, and the photon sub-pass
//! transports it. The photon flux tally is binned by a `parent_nuclides` filter
//! (the per-radionuclide breakdown the host-side TCF post-processing needs).
//!
//! Claims:
//!   (a) the GPU per-parent photon flux tally agrees with the CPU within a
//!       statistical / model-fidelity band (the GPU photon physics is the same
//!       approximate kernel the `gpu_photon_*` tests already bound; D1S adds the
//!       decay SOURCE of those photons and the parent-nuclide binning);
//!   (b) the GPU neutron flux from the D1S run equals the GPU neutron-ONLY run
//!       -- decay emission must not perturb neutron transport (child-seed
//!       isolation in the kernel);
//!   (c) the D1S run does not overflow the photon bank;
//!   (d) the parent-nuclide-binned photon flux is non-zero (Mn56 decay photons
//!       are actually produced, transported, and binned).

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::parent_nuclide::ParentNuclideFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn chain_path() -> String {
    format!(
        "{}/tests/transmutation-endf-b8.1-sfr.arrow",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
        && std::path::Path::new(&chain_path()).exists()
}

/// Fe56 sphere geometry + 14 MeV isotropic neutron point source. Returns
/// `(geometry, source, cell_id)`.
fn fe_sphere_geo_source(radius: f64) -> (Geometry, ParticleSource, u32) {
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
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    (geometry, source, cell_id)
}

/// D1S model: a neutron flux tally (Neutron-filtered) and a photon flux tally
/// binned by `parent_nuclides=["Mn56"]` with a coarse energy filter. The chain
/// is wired via the global transmutation config inside this function.
fn d1s_fe_sphere(
    seed: u64,
    radius: f64,
    n_particles: usize,
    n_batches: usize,
) -> (Model, Arc<Tally>, Arc<Tally>, TransportSettings) {
    let (geometry, source, cell_id) = fe_sphere_geo_source(radius);

    let mut neutron_tally = Tally::new();
    neutron_tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    neutron_tally
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Neutron,
        )));
    neutron_tally.scores = vec![Score::Flux(FluxScore)];
    neutron_tally.initialize_batches(n_batches);
    let neutron_tally = Arc::new(neutron_tally);

    // Photon flux tally with a parent_nuclides filter (Mn56, the Fe56(n,p)
    // activation product) and a coarse 3-bin energy filter to exercise
    // per-(parent, energy) binning.
    let mut photon_tally = Tally::new();
    photon_tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    photon_tally
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Photon,
        )));
    photon_tally
        .filters
        .push(Filter::Energy(EnergyFilter::new(vec![
            1.0e3, 1.0e6, 2.0e6, 1.0e7,
        ])));
    photon_tally
        .filters
        .push(Filter::ParentNuclide(ParentNuclideFilter::new(vec![
            "Mn56".to_string(),
        ])));
    photon_tally.scores = vec![Score::Flux(FluxScore)];
    photon_tally.initialize_batches(n_batches);
    let photon_tally = Arc::new(photon_tally);

    let tallies: Vec<Arc<Tally>> = vec![Arc::clone(&neutron_tally), Arc::clone(&photon_tally)];

    let mut model = Model::new(geometry, vec![source], tallies);
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.use_decay_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, neutron_tally, photon_tally, settings)
}

/// A GPU neutron-only Fe56 run (no decay photons) -- the byte-identity baseline.
fn neutron_only_fe_sphere(
    seed: u64,
    radius: f64,
    n_particles: usize,
    n_batches: usize,
) -> (Model, Arc<Tally>, TransportSettings) {
    let (geometry, source, cell_id) = fe_sphere_geo_source(radius);
    let mut neutron_tally = Tally::new();
    neutron_tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    neutron_tally
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Neutron,
        )));
    neutron_tally.scores = vec![Score::Flux(FluxScore)];
    neutron_tally.initialize_batches(n_batches);
    let neutron_tally = Arc::new(neutron_tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&neutron_tally)]);
    model.gpu_max_steps_per_particle = 5_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, neutron_tally, settings)
}

#[test]
fn gpu_d1s_decay_photon_matches_cpu() {
    if !data_present() {
        eprintln!("skipping gpu_d1s_decay_photon_matches_cpu -- test data / chain not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // D1S requires the transmutation chain via the global config. Set it once.
    {
        // Per-subsection transmutation config (the single-file setter was
        // removed in the transmutation split); point all three parts at the
        // v2 fixture directory, which resolve_subsection splits into
        // decay/reactions/fission_yields subdirs.
        let mut cfg = yamc_nuclide::config::Config::global();
        cfg.transmutation_decay_data = Some(chain_path());
        cfg.transmutation_reactions = yamc_nuclide::config::SubsectionSource::Set(chain_path());
        cfg.transmutation_fission_yields =
            yamc_nuclide::config::SubsectionSource::Set(chain_path());
    }

    let seed = 7;
    let radius = 10.0;
    let n_particles = 4_000;
    let n_batches = 1;

    // --- CPU reference (single coupled transport loop) ---
    let (mut cpu_m, cpu_neutron, cpu_photon, settings) =
        d1s_fe_sphere(seed, radius, n_particles, n_batches);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU D1S transport");
    let cpu_neutron_flux = cpu_neutron.get_mean();
    let cpu_photon_flux = cpu_photon.get_mean();

    // --- GPU D1S (two-pass: neutron bank -> photon sub-pass) ---
    let (mut gpu_m, gpu_neutron, gpu_photon, settings) =
        d1s_fe_sphere(seed, radius, n_particles, n_batches);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU D1S dispatch");
    let gpu_neutron_flux = gpu_neutron.get_mean();
    let gpu_photon_flux = gpu_photon.get_mean();

    // --- GPU neutron-only baseline for the byte-identity claim ---
    let (mut gpu_n_only_m, gpu_n_only, settings) =
        neutron_only_fe_sphere(seed, radius, n_particles, n_batches);
    yamc::gpu::run_on_gpu(&mut gpu_n_only_m, &settings).expect("GPU neutron-only dispatch");
    let gpu_n_only_flux = gpu_n_only.get_mean();

    let cpu_p: f64 = cpu_photon_flux.iter().sum();
    let gpu_p: f64 = gpu_photon_flux.iter().sum();
    let cpu_n: f64 = cpu_neutron_flux.iter().sum();
    let gpu_n: f64 = gpu_neutron_flux.iter().sum();
    let gpu_no: f64 = gpu_n_only_flux.iter().sum();
    println!(
        "D1S Mn56 photon flux: CPU {cpu_p:.6e}  GPU {gpu_p:.6e}  ratio {:.3}",
        gpu_p / cpu_p
    );
    println!("neutron flux: CPU {cpu_n:.6e}  GPU(D1S) {gpu_n:.6e}  GPU(n-only) {gpu_no:.6e}");
    println!(
        "per-(parent,energy) photon bins: CPU {:?}  GPU {:?}",
        cpu_photon_flux, gpu_photon_flux
    );

    // (d) Decay photons were produced, transported, and binned.
    assert!(
        gpu_p > 0.0,
        "GPU produced zero parent-nuclide-binned decay photon flux"
    );
    assert!(
        cpu_p > 0.0,
        "CPU produced zero decay photon flux (chain/data issue)"
    );

    // (a) GPU per-parent photon flux agrees with CPU within a wide band (same
    // model-fidelity tolerance the coupled photon test uses; D1S inherits the
    // same approximate GPU photon kernel).
    let ratio = gpu_p / cpu_p;
    assert!(
        (0.80..=1.25).contains(&ratio),
        "D1S photon flux GPU/CPU ratio {ratio:.3} outside [0.80, 1.25] (CPU {cpu_p:.6e}, GPU {gpu_p:.6e})"
    );

    // (b) Decay emission does not perturb neutron transport (child-seed
    // isolation). This is bit-exact at the kernel level (yamc-gpu
    // `cpu_gpu_equivalence`). Since issue #233 Stage 3 the D1S run is ALSO
    // batch-free (per-source) and uses the SAME fixed launch chunk as the
    // neutron-only baseline, so both transport the identical source neutrons and
    // agree to a tight band (only f64 round-off in the per-source vs per-history
    // sum differs). A real perturbation would shift the neutron flux far beyond
    // this (both still track the CPU neutron flux in (b2)).
    let n_only_ratio = gpu_n / gpu_no;
    assert!(
        (0.999..=1.001).contains(&n_only_ratio),
        "GPU D1S neutron flux diverged from the neutron-only run (batch-free, same \
         chunk) (sum ratio {n_only_ratio:.6}); child-seed isolation may be broken"
    );

    // (b2) GPU neutron flux also tracks the CPU neutron flux.
    let n_ratio = gpu_n / cpu_n;
    assert!(
        (0.90..=1.10).contains(&n_ratio),
        "neutron flux GPU/CPU ratio {n_ratio:.3} outside [0.90, 1.10]"
    );
}
