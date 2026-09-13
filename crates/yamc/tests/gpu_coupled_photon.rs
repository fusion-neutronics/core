//! Coupled neutron->photon GPU dispatch acceptance test (slices S6 + S7).
//!
//! A 14 MeV neutron point source in a single-material Fe56 sphere with
//! `transport_secondary_photons = true`. The neutron kernel emits secondary
//! (n,gamma) / (n,xn) photons into a device bank; the coupled dispatch drains
//! that bank into the photon kernel so the produced photons are transported.
//!
//! Three claims:
//!   (a) the GPU photon flux tally agrees with the CPU photon flux tally
//!       within a statistical / model-fidelity band (the GPU photon physics is
//!       the same approximate kernel the `gpu_photon_*` tests already bound to
//!       roughly +/-25%; the coupled SOURCE of those photons is what S6 adds);
//!   (b) the GPU neutron flux from the coupled run equals the GPU neutron-ONLY
//!       run -- coupling must not perturb neutron transport (child-seed
//!       isolation in the kernel);
//!   (c) the coupled run does not overflow the photon bank (a clean dispatch,
//!       not a `PhotonBankOverflow` error).

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
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
}

/// A neutron flux tally on `cell_id` (no particle filter -- analog neutron
/// scoring) and, if `with_photon`, a photon flux tally (ParticleType(Photon)).
/// Returns the model plus the (neutron, photon) tally handles for readback.
/// The shared Fe56 sphere geometry + 14 MeV isotropic neutron point source.
/// Returns `(geometry, source, cell_id)`.
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

    // 14 MeV neutron point source at the centre.
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

fn coupled_fe_sphere(
    seed: u64,
    radius: f64,
    n_particles: usize,
    n_batches: usize,
    with_photon: bool,
) -> (Model, Arc<Tally>, Option<Arc<Tally>>, TransportSettings) {
    let (geometry, source, cell_id) = fe_sphere_geo_source(radius);

    // Neutron flux tally with an explicit Neutron particle filter so it counts
    // ONLY neutrons on both backends. Without the filter the CPU (single
    // coupled transport loop) also scores the drained secondary photons' track
    // length onto this tally, while the GPU (two-pass) scores neutrons only --
    // an apples-to-oranges comparison. The filter makes both neutron-only.
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

    let mut tallies: Vec<Arc<Tally>> = vec![Arc::clone(&neutron_tally)];

    let photon_tally = if with_photon {
        let mut t = Tally::new();
        t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
        t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Photon,
        )));
        t.scores = vec![Score::Flux(FluxScore)];
        t.initialize_batches(n_batches);
        let t = Arc::new(t);
        tallies.push(Arc::clone(&t));
        Some(t)
    } else {
        None
    };

    let mut model = Model::new(geometry, vec![source], tallies);
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, neutron_tally, photon_tally, settings)
}

#[test]
fn gpu_coupled_neutron_photon_matches_cpu() {
    if !data_present() {
        eprintln!("skipping gpu_coupled_neutron_photon_matches_cpu -- test data not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let seed = 4242;
    let radius = 5.0;
    let n_particles = 20_000;
    let n_batches = 8;

    // --- CPU coupled reference ---
    let (mut cpu_m, cpu_n_t, cpu_p_t, settings) =
        coupled_fe_sphere(seed, radius, n_particles, n_batches, true);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu_neutron = cpu_n_t.get_mean().iter().sum::<f64>();
    let cpu_p_t = cpu_p_t.unwrap();
    let cpu_photon = cpu_p_t.get_mean().iter().sum::<f64>();

    // --- GPU coupled run (neutron + photon tally) ---
    let (mut gpu_m, gpu_n_t, gpu_p_t, settings) =
        coupled_fe_sphere(seed, radius, n_particles, n_batches, true);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings)
        .expect("coupled GPU dispatch must succeed (no overflow)");
    let gpu_neutron_coupled_mean = gpu_n_t.get_mean();
    let gpu_neutron_coupled = gpu_neutron_coupled_mean.iter().sum::<f64>();
    let gpu_p_t = gpu_p_t.unwrap();
    let gpu_photon = gpu_p_t.get_mean().iter().sum::<f64>();

    // --- GPU neutron-ONLY run (transport_secondary_photons still on, but no
    //     photon tally so only the neutron kernel scores) for the neutron
    //     identity check. The neutron tally must be unaffected by coupling, so
    //     it must also equal a coupled run's neutron tally. We compare the
    //     coupled neutron tally to a run with photon production OFF: that is
    //     the strongest "coupling did not perturb neutron transport" check. ---
    let (mut gpu_neutron_only_m, gpu_no_t, _none, settings) =
        coupled_fe_sphere(seed, radius, n_particles, n_batches, false);
    gpu_neutron_only_m.transport_secondary_photons = false;
    yamc::gpu::run_on_gpu(&mut gpu_neutron_only_m, &settings).expect("neutron-only GPU dispatch");
    let gpu_neutron_only_mean = gpu_no_t.get_mean();
    let gpu_neutron_only = gpu_neutron_only_mean.iter().sum::<f64>();

    let photon_ratio = gpu_photon / cpu_photon;
    let neutron_ratio = gpu_neutron_coupled / gpu_neutron_only;
    let neutron_cpu_gpu_ratio = gpu_neutron_coupled / cpu_neutron;
    println!(
        "coupled Fe r={radius}: CPU photon = {cpu_photon:.4e}  GPU photon = {gpu_photon:.4e}  \
         photon ratio = {photon_ratio:.3}"
    );
    println!(
        "coupled Fe r={radius}: CPU neutron = {cpu_neutron:.4e}  \
         GPU neutron (coupled) = {gpu_neutron_coupled:.4e}  \
         GPU neutron (analog) = {gpu_neutron_only:.4e}  \
         GPU/CPU neutron = {neutron_cpu_gpu_ratio:.3}  (coupled==analog ratio {neutron_ratio:.4})"
    );

    // (c) Non-zero photon tally of the right order of magnitude: the coupled
    //     source actually produced + transported photons.
    assert!(
        gpu_photon > 0.0,
        "coupled GPU run produced a zero photon flux tally -- the bank drain / \
         photon sub-pass did not transport any secondary photons"
    );
    assert!(
        cpu_photon > 0.0,
        "CPU coupled run produced a zero photon flux tally -- test setup issue"
    );

    // (b) Neutron-identity: coupling must NOT perturb neutron transport. The
    //     neutron kernel forks a child PCG state for photon sampling
    //     (`pcg_next(state ^ ...)`) and never advances the neutron `state`, so
    //     coupling-on vs coupling-off is bit-identical FOR THE SAME SEEDS. Since
    //     issue #233 Stage 3 the coupled path is ALSO batch-free (per-source),
    //     using the SAME fixed launch chunk as the neutron-only baseline, so the
    //     two runs transport the identical source neutrons and agree to a very
    //     tight band (only f64 round-off in the per-source vs per-history sum
    //     differs). A real coupling perturbation would shift the flux far beyond
    //     this. (The kernel-level bit-exactness is the yamc-gpu `cpu_gpu_equivalence`
    //     suite's strong guarantee.)
    assert!(
        (0.999..=1.001).contains(&neutron_ratio),
        "coupled GPU neutron flux must match the neutron-only GPU flux (batch-free, \
         same chunk) (sum ratio {neutron_ratio:.6}) -- coupling perturbed neutron transport"
    );

    // (b2) GPU neutron flux must also match the CPU neutron flux. Both tallies
    //     are Neutron-filtered, so this is a clean neutron-only comparison (an
    //     UNFILTERED tally would make the CPU's single coupled loop also score
    //     the drained photons' track length, inflating it ~2x -- the GPU
    //     two-pass scores neutrons only). Filtered, GPU/CPU agrees to well
    //     within Monte-Carlo noise; the band is loose enough for the known GPU
    //     neutron-kernel approximations yet tight enough to catch a real gap.
    assert!(
        (0.90..=1.10).contains(&neutron_cpu_gpu_ratio),
        "coupled GPU/CPU neutron flux ratio {neutron_cpu_gpu_ratio:.3} outside [0.90, 1.10] \
         (GPU {gpu_neutron_coupled:.4e} vs CPU {cpu_neutron:.4e})"
    );

    // (a) Photon flux GPU-vs-CPU band. Both sides are deterministic for a fixed
    //     seed (CPU MC; GPU fixed-point order-independent tallies + per-photon
    //     seeds stored in the bank), so the only spread is the systematic
    //     GPU-photon-kernel approximation (Kahn free-electron Compton, no
    //     atomic-relaxation cascade, no Compton-electron secondaries) -- the
    //     same kernel the gpu_photon_* tests bound to ~+/-25%. This coupled
    //     Fe56 case sits at ~1.03, so a [0.90, 1.15] band is a tight regression
    //     guard on the coupled SOURCE (wrong spectrum / weight / drained count
    //     shifts it well outside) while retaining margin for that documented
    //     kernel systematic.
    assert!(
        (0.90..=1.15).contains(&photon_ratio),
        "coupled GPU/CPU photon flux ratio {photon_ratio:.3} outside [0.90, 1.15] -- \
         coupled photon source or sub-pass may have regressed"
    );
}

/// Build the Fe56 sphere coupled model with three flux tallies on the same
/// cell: unfiltered (ALL particles), Neutron-filtered, and Photon-filtered.
fn three_tally_model(
    seed: u64,
    radius: f64,
    n_particles: usize,
    n_batches: usize,
) -> (Model, Arc<Tally>, Arc<Tally>, Arc<Tally>, TransportSettings) {
    let (geometry, source, cell_id) = fe_sphere_geo_source(radius);
    let flux_tally = |pt: Option<ParticleType>| {
        let mut t = Tally::new();
        t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
        if let Some(p) = pt {
            t.filters
                .push(Filter::ParticleType(ParticleTypeFilter::new(p)));
        }
        t.scores = vec![Score::Flux(FluxScore)];
        t.initialize_batches(n_batches);
        Arc::new(t)
    };
    let unfiltered = flux_tally(None);
    let neutron = flux_tally(Some(ParticleType::Neutron));
    let photon = flux_tally(Some(ParticleType::Photon));
    let tallies = vec![
        Arc::clone(&unfiltered),
        Arc::clone(&neutron),
        Arc::clone(&photon),
    ];
    let mut model = Model::new(geometry, vec![source], tallies);
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, unfiltered, neutron, photon, settings)
}

/// In coupled mode an UNFILTERED flux tally means ALL particles, so it must
/// score the neutron + photon SUM (matching CPU/OpenMC). The GPU two-pass
/// dispatch scores such a tally in BOTH passes and sums them per batch. This
/// pins (1) the dual-pass sum identity -- the unfiltered tally equals the
/// neutron-filtered + photon-filtered tallies from the SAME run -- and (2) that
/// it agrees with the CPU's unfiltered all-particle flux.
#[test]
fn gpu_coupled_unfiltered_flux_sums_both_particles() {
    if !data_present() {
        eprintln!("skipping gpu_coupled_unfiltered_flux_sums_both_particles -- data not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let seed = 4242;
    let radius = 5.0;
    let n_particles = 20_000;
    let n_batches = 8;

    // CPU reference: an unfiltered tally scores all particles (n + gamma).
    let (mut cpu_m, cpu_unf, _n, _p, settings) =
        three_tally_model(seed, radius, n_particles, n_batches);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu_unfiltered = cpu_unf.get_mean().iter().sum::<f64>();

    // GPU coupled run with all three tallies.
    let (mut gpu_m, gpu_unf, gpu_n, gpu_p, settings) =
        three_tally_model(seed, radius, n_particles, n_batches);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("coupled GPU dispatch");
    let gpu_unfiltered = gpu_unf.get_mean().iter().sum::<f64>();
    let gpu_neutron = gpu_n.get_mean().iter().sum::<f64>();
    let gpu_photon = gpu_p.get_mean().iter().sum::<f64>();
    let sum = gpu_neutron + gpu_photon;

    println!(
        "coupled unfiltered: GPU unfiltered = {gpu_unfiltered:.4e}  \
         (n {gpu_neutron:.4e} + g {gpu_photon:.4e} = {sum:.4e})  \
         CPU unfiltered = {cpu_unfiltered:.4e}"
    );

    assert!(
        gpu_photon > 0.0,
        "no photons scored -- coupled source produced none"
    );

    // (1) Dual-pass sum identity: the unfiltered (all-particle) tally equals the
    //     neutron-filtered + photon-filtered tallies from the SAME run. The
    //     batch mean is linear, so this holds to summation round-off.
    assert!(
        (gpu_unfiltered - sum).abs() <= 1e-9 * sum.abs().max(1.0),
        "GPU unfiltered flux {gpu_unfiltered:.6e} != neutron+photon {sum:.6e} -- \
         dual-pass sum is broken (unfiltered tally not scored in both passes)"
    );

    // (2) Matches the CPU all-particle flux (same n+gamma quantity).
    let ratio = gpu_unfiltered / cpu_unfiltered;
    assert!(
        (0.90..=1.15).contains(&ratio),
        "coupled GPU/CPU unfiltered (all-particle) flux ratio {ratio:.3} outside [0.90, 1.15] \
         (GPU {gpu_unfiltered:.4e} vs CPU {cpu_unfiltered:.4e})"
    );
}
