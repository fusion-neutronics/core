//! Where `Material::transmute` spends its time, phase by phase.
//!
//! Run as: `cargo run --release -p yani-transmute --example transmute_perf`,
//! and with `RAYON_NUM_THREADS=1` beside it once any of issue #576's parallel
//! work has landed, so a speedup can be told apart from a core count.
//!
//! Three shapes, because they load the phases differently:
//!
//! * **Fe56** -- one nuclide, one long grid. The collapse per nuclide.
//! * **steel** -- six elements, so the reachable chain closure and the Arrow
//!   decode that comes with it. The shape `tools/bench_transmute.py` times.
//! * **14 MeV** -- the same steel under a monoenergetic source in CCFE-709,
//!   where all but a couple of groups carry no flux at all.
//!
//! The split is by difference rather than by instrumentation. `preload` and
//! `collapse` are measured directly, by doing the same work the driver does;
//! `steps` is a plain `transmute_material` on the already-loaded material minus
//! that collapse, and `replicas` is what an uncertainty request adds to the
//! same call. Both differences are between two timings taken in the same cache
//! state, which is what "the cold total minus the phases measured after it" is
//! not: measuring a phase warms the cache the total already paid for. The cold
//! number is `tools/bench_transmute.py`'s `once`, which is the caller's own.
//!
//! Skips when the nuclear-data fixtures are not cached. `python
//! scripts/fetch_test_fixtures.py` fetches them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use yamc_materials::Material;
use yani_transmute::uncertainty::DataUncertainty;
use yani_transmute::{
    compute_multigroup_reaction_rates, transmute_material, MultigroupSpectrum, TransmuteStep,
};

const GROUPS: &str = "CCFE-709";

/// Fixed rather than adaptive: an adaptive count makes the replica phase's cost
/// depend on how fast the sigmas settle, which is not what is being timed.
const SAMPLES: usize = 32;

const STEEL: [(&str, f64); 6] = [
    ("Fe", 0.65),
    ("Cr", 0.17),
    ("Ni", 0.12),
    ("Mo", 0.025),
    ("Mn", 0.02),
    ("Si", 0.01),
];

fn main() {
    let boundaries = match yamc_nuclide::group_structures::get_group_structure(GROUPS) {
        Ok(b) => b.to_vec(),
        Err(e) => {
            eprintln!("no {GROUPS} group structure: {e}");
            return;
        }
    };

    let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = Arc::new(yani::parse_chain_arrow(&chain_path).expect("parse chain"));

    // Point the loader at the fixture cache, entry by entry, rather than at a
    // library keyword: the example then reads only what this machine already
    // has and never reaches the network. How much of the library is cached is
    // therefore part of the measurement, which is why the nuclide count is a
    // column.
    {
        let mut cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for name in chain.keys() {
            if let Some(path) = yamc_test_cache::nuclide(name) {
                cfg.set_cross_section(name, Some(&path));
            }
        }
    }

    let fusion = fusion_spectrum(&boundaries);
    let monoenergetic = fourteen_mev(&boundaries);

    println!(
        "{:<10}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "shape", "preload", "collapse", "steps", "replicas", "nuclides"
    );
    for (label, composition, flux) in [
        ("Fe56", vec![("Fe56", 1.0)], &fusion),
        ("steel", STEEL.to_vec(), &fusion),
        ("14 MeV", STEEL.to_vec(), &monoenergetic),
    ] {
        match run(label, &composition, flux, &boundaries, &chain) {
            Some(row) => println!("{row}"),
            None => println!("{label:<10}  skipped (nuclear-data fixtures missing)"),
        }
    }
}

fn run(
    label: &str,
    composition: &[(&str, f64)],
    flux: &[f64],
    boundaries: &[f64],
    chain: &Arc<HashMap<String, yani::ChainNuclide>>,
) -> Option<String> {
    // Elements are expanded here rather than left as `"Fe"`, because
    // `Material::new` takes nuclides: the expansion is the Python binding's job
    // on the usual path, and without it the chain closure would be empty and
    // every shape would report itself skipped.
    let mut nuclides: HashMap<String, f64> = HashMap::new();
    for (name, fraction) in composition {
        if name.chars().any(|c| c.is_ascii_digit()) {
            nuclides.insert(name.to_string(), *fraction);
        } else {
            let expanded =
                yamc_nuclide::composition::expand_element(name, *fraction, "mass").ok()?;
            yamc_nuclide::composition::merge_nuclides(&mut nuclides, &expanded);
        }
    }
    let mut material = Material::new(nuclides, "mass", "g/cm3", Some(7.9)).ok()?;
    material.volume = Some(1.0);
    material.set_temperature("294");

    let total: f64 = flux.iter().sum();
    let spectra = vec![MultigroupSpectrum {
        boundaries: boundaries.to_vec(),
        masses: flux.iter().map(|f| f / total).collect(),
        relative_std_dev: Some(vec![0.05; flux.len()]),
    }];
    let steps = vec![
        TransmuteStep {
            dt: 3600.0,
            irradiation: Some((0, 1.0e14)),
        },
        TransmuteStep {
            dt: 3600.0,
            irradiation: None,
        },
    ];

    // Phase 1, by calling the driver's own preload rather than a copy of it, so
    // this column tracks whatever that does. The paths come from CONFIG, which
    // `main` has pointed at the fixture cache entry by entry, so how much of the
    // library is cached is part of the measurement -- which is why the nuclide
    // count is a column.
    //
    // Preloaded WITH the uncertainty request, so it reads the covariance too.
    // Without that, the uncertainty call below re-reads every nuclide to pick
    // it up and the `replicas` column -- a difference between the two calls --
    // silently carries a second full load.
    let request = DataUncertainty {
        seed: 1,
        samples: Some(SAMPLES),
        ..Default::default()
    };
    let start = Instant::now();
    let mut loaded = material.clone().to_sum_mode().ok()?;
    yani_transmute::preload_activation_data(
        &mut loaded,
        chain,
        &Default::default(),
        Some(&request),
        None,
    )
    .ok()?;
    let preload = start.elapsed();
    let nuclides = loaded.nuclide_data.len();
    if nuclides == 0 {
        return None;
    }

    // Phase 2, on that same loaded material.
    let start = Instant::now();
    let _ = compute_multigroup_reaction_rates(
        &loaded,
        chain,
        &spectra[0].masses,
        &spectra[0].boundaries,
        1.0,
    );
    let collapse = start.elapsed();

    // Phases 3 and 4, by difference. Both calls are handed the material that
    // already carries the data, so neither pays the load again and the two
    // differences are each between two timings taken in the same state --
    // which is what the "cold total minus the phases measured after it" split
    // is not, because measuring the phases warms the cache the total paid for.
    let start = Instant::now();
    transmute_material(
        &mut loaded.clone(),
        &spectra,
        &steps,
        Arc::clone(chain),
        &Default::default(),
        Default::default(),
        None,
    )
    .ok()?;
    let plain = start.elapsed();

    let start = Instant::now();
    transmute_material(
        &mut loaded.clone(),
        &spectra,
        &steps,
        Arc::clone(chain),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .ok()?;
    let with_uncertainty = start.elapsed();

    let steps_phase = plain.saturating_sub(collapse);
    let replicas = with_uncertainty.saturating_sub(plain);
    Some(format!(
        "{label:<10}{:>10}{:>10}{:>10}{:>10}{nuclides:>10}",
        secs(preload),
        secs(collapse),
        secs(steps_phase),
        secs(replicas),
    ))
}

fn secs(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64())
}

/// A 1/E slowing-down tail with a 14 MeV peak: every group carries flux, so
/// nothing in the collapse can be skipped as a zero.
fn fusion_spectrum(boundaries: &[f64]) -> Vec<f64> {
    boundaries
        .windows(2)
        .map(|w| {
            let mid = (w[0].max(1.0e-5) * w[1]).sqrt();
            let value = (w[1] - w[0]) / mid;
            if (1.3e7..1.5e7).contains(&mid) {
                value + 50.0
            } else {
                value
            }
        })
        .collect()
}

/// All of the flux in the one group holding 14.06 MeV, which is the case the
/// zero-flux skip of finding 2 exists for.
fn fourteen_mev(boundaries: &[f64]) -> Vec<f64> {
    boundaries
        .windows(2)
        .map(|w| {
            if w[0] <= 1.406e7 && 1.406e7 < w[1] {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}
