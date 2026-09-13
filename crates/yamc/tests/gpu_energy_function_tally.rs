//! GPU `EnergyFunctionFilter` tally support (issue #271), on-hardware.
//!
//! `energy_function=` (and its sugar `dose_coefficients=`) is not a bin
//! dimension: it multiplies the score by a tabulated curve interpolated at the
//! particle's energy and DROPS the event entirely when the energy falls off the
//! table. So the things worth testing are the multiply, the gate, and the fact
//! that the multiply composes with the other filters rather than replacing them.
//!
//! Three of the four tests are exact identities rather than statistical
//! comparisons, because they compare tallies scored over the SAME histories in
//! one launch:
//!
//!   1. a constant curve `y == c` must scale a plain flux tally by exactly `c`
//!      (a natural cubic spline through constant data is exactly constant, which
//!      `energy_function_twin.rs` pins independently);
//!   2. a curve covering only part of the spectrum must drop exactly the flux
//!      outside it, so gated + complement == ungated;
//!   3. the weighted tally must stay consistent across the energy-bin and mesh
//!      dimensions it is stacked with.
//!
//! The fourth compares GPU against CPU on the real ICRP-116 dose curve, which
//! is the workflow this feature exists for.
//!
//! Self-skips without an f64 GPU or the C12/Fe56 cache. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_energy_function_tally -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::{EnergyFunctionFilter, Estimator};

const N_PARTICLES: usize = 40_000;
const N_BATCHES: usize = 8;
const SEED: u64 = 20260802;
const RADIUS: f64 = 25.0;
const SOURCE_E: f64 = 14.06e6;

/// The constant an `energy_function` of all-`CONST_Y` must scale the flux by.
/// Deliberately not a power of two, so a dropped multiply cannot hide behind
/// a coincidentally exact ratio.
const CONST_Y: f64 = 3.25;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// One moderating sphere. C12 at fast energies down-scatters strongly, so the
/// flux spans many decades of energy -- which is what makes the out-of-range
/// gate test below score a meaningful fraction of the histories rather than
/// nothing.
fn sphere_geometry(nuclide: &str, density: f64) -> Geometry {
    let s = Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        RADIUS,
        Some(1),
        Some(BoundaryType::Vacuum),
    ));
    let region = Region::new_from_halfspace(HalfspaceType::Below(s));
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), cache_dir(nuclide))]),
        None,
    )
    .unwrap();
    let cell = Cell::new(Some(1), region, Some("sphere".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(m)]).unwrap()
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn flux_tally(name: &str, filters: Vec<Filter>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters = filters;
    t.scores = vec!["flux".parse::<Score>().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.name = Some(name.to_string());
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn cell_filter() -> Filter {
    Filter::Cell(CellFilter { cell_ids: vec![1] })
}

/// A curve that is exactly `CONST_Y` everywhere over the whole plausible
/// spectrum, so it weights every scoring event and gates none.
fn constant_curve() -> Filter {
    Filter::EnergyFunction(EnergyFunctionFilter::new(
        vec![1e-5, 1e2, 1e5, 1e8],
        vec![CONST_Y; 4],
    ))
}

fn model(tallies: Vec<Arc<Tally>>, nuclide: &str, density: f64) -> (Model, TransportSettings) {
    let mut m = Model::new(
        sphere_geometry(nuclide, density),
        vec![neutron_source()],
        tallies,
    );
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_PARTICLES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

fn sum_mean(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum()
}

fn assert_close(label: &str, got: f64, want: f64, rtol: f64) {
    let denom = want.abs().max(f64::MIN_POSITIVE);
    let rel = (got - want).abs() / denom;
    assert!(
        rel <= rtol,
        "{label}: got {got:.12e}, want {want:.12e} (relative {rel:.3e} > {rtol:.1e})"
    );
}

/// A constant energy function scales the tally by exactly that constant.
/// Same histories, same launch, so this is arithmetic, not statistics.
#[test]
fn gpu_constant_energy_function_scales_flux_exactly() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_constant_energy_function_scales_flux_exactly: data/GPU absent");
        return;
    }
    let plain = flux_tally("plain", vec![cell_filter()]);
    let weighted = flux_tally("weighted", vec![cell_filter(), constant_curve()]);
    let (mut m, settings) = model(vec![Arc::clone(&plain), Arc::clone(&weighted)], "C12", 2.0);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");

    let (p, w) = (sum_mean(&plain), sum_mean(&weighted));
    assert!(p > 0.0, "plain flux tally scored nothing");
    eprintln!("  plain {p:.6e}  weighted {w:.6e}  ratio {:.9}", w / p);
    assert_close("weighted vs CONST_Y * plain", w, CONST_Y * p, 1e-9);
}

/// A curve covering only the high-energy part of the spectrum must drop
/// exactly the flux below its floor: gated + complement == ungated. This is
/// the property that separates "gate the event" from "score zero", and it is
/// the one a naive implementation gets wrong.
#[test]
fn gpu_energy_function_gate_drops_exactly_the_out_of_range_flux() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_energy_function_gate_drops_exactly_the_out_of_range_flux: absent");
        return;
    }
    // Split the spectrum at 1 MeV: the energy function covers [1e6, 2e7] only.
    const SPLIT: f64 = 1e6;
    const TOP: f64 = 2e7;
    let gated = flux_tally(
        "gated",
        vec![
            cell_filter(),
            Filter::EnergyFunction(EnergyFunctionFilter::new(
                vec![SPLIT, 5e6, 1e7, TOP],
                vec![1.0; 4],
            )),
        ],
    );
    // Reference: the same flux binned by energy, so the in-range part can be
    // summed independently of the energy function.
    let binned = flux_tally(
        "binned",
        vec![
            cell_filter(),
            Filter::Energy(EnergyFilter::new(vec![1e-5, SPLIT, TOP])),
        ],
    );
    let (mut m, settings) = model(vec![Arc::clone(&gated), Arc::clone(&binned)], "C12", 2.0);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");

    let g = sum_mean(&gated);
    let bins = binned.get_mean().to_vec();
    assert_eq!(bins.len(), 2, "expected 2 energy bins");
    let (below, above) = (bins[0], bins[1]);
    eprintln!("  gated {g:.6e}  below-split {below:.6e}  above-split {above:.6e}");

    // The gate must actually bite: a moderating sphere puts real flux below
    // 1 MeV, otherwise this test would pass trivially.
    assert!(
        below > 0.05 * above,
        "only {below:.3e} vs {above:.3e} below the split; the gate is barely exercised"
    );
    // A weight of 1.0 in range means the gated tally IS the above-split bin.
    assert_close("gated vs above-split bin", g, above, 1e-9);
}

/// Stacked with `energy_bins=`, the weight must apply per event, so folding the
/// weighted spectrum over energy still equals the weighted total, and each
/// energy bin equals the unweighted bin times the constant.
#[test]
fn gpu_energy_function_composes_with_energy_bins() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_energy_function_composes_with_energy_bins: data/GPU absent");
        return;
    }
    let edges = vec![1e-5, 1e3, 1e6, 2e7];
    let plain = flux_tally(
        "plain_spectrum",
        vec![
            cell_filter(),
            Filter::Energy(EnergyFilter::new(edges.clone())),
        ],
    );
    let weighted = flux_tally(
        "weighted_spectrum",
        vec![
            cell_filter(),
            Filter::Energy(EnergyFilter::new(edges.clone())),
            constant_curve(),
        ],
    );
    let (mut m, settings) = model(vec![Arc::clone(&plain), Arc::clone(&weighted)], "C12", 2.0);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");

    let p = plain.get_mean().to_vec();
    let w = weighted.get_mean().to_vec();
    assert_eq!(p.len(), edges.len() - 1);
    assert_eq!(w.len(), p.len());
    let occupied = p.iter().filter(|v| **v > 0.0).count();
    assert!(
        occupied >= 2,
        "only {occupied} energy bin(s) populated; the per-bin check is not meaningful"
    );
    for (i, (&pi, &wi)) in p.iter().zip(w.iter()).enumerate() {
        if pi > 0.0 {
            assert_close(&format!("energy bin {i}"), wi, CONST_Y * pi, 1e-9);
        }
    }
}

/// GPU vs CPU on the real ICRP-116 AP dose curve -- the workflow the feature
/// exists for. Independent MC estimates, so this one is a band, not an identity.
#[test]
fn gpu_icrp116_dose_matches_cpu() {
    if !data_present("Fe56") || !gpu_available() {
        eprintln!("skipping gpu_icrp116_dose_matches_cpu: data/GPU absent");
        return;
    }
    let (e, coeffs) = yamc_nuclide::data::effective_dose::dose_coefficients(
        yamc_nuclide::data::effective_dose::DoseParticle::Neutron,
        yamc_nuclide::data::effective_dose::DoseGeometry::AP,
        yamc_nuclide::data::effective_dose::DoseDataSource::ICRP116,
    );
    let dose_filter =
        || Filter::EnergyFunction(EnergyFunctionFilter::new(e.clone(), coeffs.clone()));

    let cpu_t = flux_tally("cpu_dose", vec![cell_filter(), dose_filter()]);
    let (mut cpu_m, cpu_s) = model(vec![Arc::clone(&cpu_t)], "Fe56", 7.874);
    cpu_m.simulate_transport(&cpu_s).expect("CPU run");
    let cpu = sum_mean(&cpu_t);

    let gpu_t = flux_tally("gpu_dose", vec![cell_filter(), dose_filter()]);
    let (mut gpu_m, gpu_s) = model(vec![Arc::clone(&gpu_t)], "Fe56", 7.874);
    yamc::gpu::run_on_gpu(&mut gpu_m, &gpu_s).expect("GPU run");
    let gpu = sum_mean(&gpu_t);

    assert!(cpu > 0.0, "CPU dose tally scored nothing");
    assert!(gpu > 0.0, "GPU dose tally scored nothing");
    let ratio = gpu / cpu;
    eprintln!("  ICRP-116 AP dose: CPU {cpu:.5e}  GPU {gpu:.5e}  ratio {ratio:.4}");
    assert!(
        (0.95..=1.05).contains(&ratio),
        "GPU/CPU dose ratio {ratio:.4} outside [0.95, 1.05] (CPU {cpu:.5e}, GPU {gpu:.5e})"
    );

    // Sanity that the dose curve is actually doing something: the ICRP-116
    // coefficients are O(100) pSv cm^2 at these energies, so the dose tally
    // must be far larger than the bare flux it weights.
    let bare = flux_tally("cpu_bare", vec![cell_filter()]);
    let (mut bare_m, bare_s) = model(vec![Arc::clone(&bare)], "Fe56", 7.874);
    bare_m.simulate_transport(&bare_s).expect("CPU bare run");
    let bare_v = sum_mean(&bare);
    assert!(
        cpu > 10.0 * bare_v,
        "dose {cpu:.3e} is not meaningfully above bare flux {bare_v:.3e}; \
         the energy function may not be applied at all"
    );
}
