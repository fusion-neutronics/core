//! Regression guard for the #111 elastic-unification (CPU elastic routed
//! through the shared `yamc_physics::gpu::flat::{elastic_mu_cm,
//! free_gas_elastic}` samplers + a 32-bit PCG stream).
//!
//! Runs the production CPU neutron transport on single-isotope Cr52/Mn55/Mo98
//! spheres + an SS316 sphere (14 MeV point source, 20 cm sphere) and asserts
//! the integrated flux stays within 5% of the pre-refactor, OpenMC-validated
//! CPU baseline. The pre-refactor CPU matched OpenMC; this confirms routing
//! elastic through the shared samplers preserves that agreement (the
//! OpenMC-regression localizer for #88 -- a break here would mean the elastic
//! orchestration is the #88 bug). Prints the full spectrum with `--nocapture`.
//! Self-skips when the cached isotope data is absent (e.g. CI).

// Per-bin spectrum loops index `flux` and `edges` in lockstep; a range loop
// reads more clearly than zipped iterators here.
#![allow(clippy::needless_range_loop)]

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
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn build(comp: &[(&str, f64)], density: f64) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if comp
        .iter()
        .any(|(n, _)| !std::path::Path::new(&cache(n)).exists())
    {
        return None;
    }
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 20.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let composition: HashMap<String, f64> = comp.iter().map(|(n, f)| (n.to_string(), *f)).collect();
    let data: HashMap<String, String> = comp
        .iter()
        .map(|(n, _)| (n.to_string(), cache(n)))
        .collect();
    let mut material = Material::new(composition, "atom", "g/cm3", Some(density)).ok()?;
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(&data, None).ok()?;
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut edges = vec![1.0e-5];
    let mut e = 1.0e-1;
    while e < 2.0e7 {
        edges.push(e);
        e *= 10f64.powf(0.5); // ~half-decade bins
    }
    edges.push(2.0e7);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.filters.push(Filter::Energy(EnergyFilter::new(edges)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(400_000),
        seed: 77_88,
        threads: Some(1),
        ..Default::default()
    };
    Some((model, tally, settings))
}

fn report(name: &str, comp: &[(&str, f64)], density: f64, baseline_sum: f64) {
    let Some((mut cpu, ct, settings)) = build(comp, density) else {
        eprintln!("{name}: skip (data missing)");
        return;
    };
    cpu.simulate_transport(&settings).unwrap();
    let c = ct.get_mean();
    let edges = if let Filter::Energy(ef) = ct
        .filters
        .iter()
        .find(|f| matches!(f, Filter::Energy(_)))
        .unwrap()
    {
        ef.bins.clone()
    } else {
        vec![]
    };
    let total_c: f64 = c.iter().sum();
    // Epithermal fraction (1 - 300 eV), the #88 diagnostic band.
    let mut epi = 0.0;
    for i in 0..c.len() {
        let e_lo = edges.get(i).copied().unwrap_or(0.0);
        if (1.0..=300.0).contains(&e_lo) {
            epi += c[i];
        }
    }
    eprintln!(
        "=== {name} (sum CPU={total_c:.6e}, epithermal[1-300eV] frac={:.6})",
        epi / total_c
    );
    eprintln!("  E_lo(eV)        CPU_flux       %oftot");
    for i in 0..c.len() {
        if c[i] <= 0.0 {
            continue;
        }
        let frac = c[i] / total_c;
        eprintln!(
            "  {:>12.4e}  {:>14.6e}   {:>7.4}%",
            edges.get(i).copied().unwrap_or(0.0),
            c[i],
            frac * 100.0
        );
    }
    // Pre-refactor (OpenMC-validated) CPU integrated flux. A 5% band is far
    // above the ~0.15% Monte-Carlo noise at 400k histories but well below a
    // real spectrum shift, so it catches a broken elastic orchestration.
    let ratio = total_c / baseline_sum;
    assert!(
        (0.95..=1.05).contains(&ratio),
        "{name}: integrated CPU flux {total_c:.4e} vs baseline {baseline_sum:.4e} (ratio {ratio:.4}) outside [0.95, 1.05]"
    );
}

// Heavy CPU transport (4 spheres x 400k histories): run explicitly with
// `cargo test --release -p yamc --test elastic_unification_baseline -- --ignored`.
// Self-skips without the cached isotope data, so it is inert in CI regardless.
#[test]
#[ignore = "heavy CPU transport regression; run with --release --ignored"]
fn cpu_elastic_spectrum_matches_baseline() {
    report("Cr52", &[("Cr52", 1.0)], 7.19, 4.379e1);
    report("Mn55", &[("Mn55", 1.0)], 7.21, 6.688e1);
    report("Mo98", &[("Mo98", 1.0)], 10.28, 1.135e2);
    report(
        "SS316",
        &[
            ("Fe54", 0.03812),
            ("Fe56", 0.59850),
            ("Fe57", 0.01382),
            ("Fe58", 0.00184),
            ("Cr50", 0.00712),
            ("Cr52", 0.13724),
            ("Cr53", 0.01556),
            ("Cr54", 0.00387),
            ("Ni58", 0.07721),
            ("Ni60", 0.02974),
            ("Ni61", 0.00129),
            ("Ni62", 0.00412),
            ("Ni64", 0.00105),
            ("Mo92", 0.00373),
            ("Mo94", 0.00233),
            ("Mo95", 0.00401),
            ("Mo96", 0.00420),
            ("Mo97", 0.00241),
            ("Mo98", 0.00609),
            ("Mo100", 0.00243),
            ("Mn55", 0.02000),
        ],
        7.99,
        5.769e1,
    );
}
