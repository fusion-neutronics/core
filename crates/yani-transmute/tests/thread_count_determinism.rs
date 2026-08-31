//! The answer must not depend on how many cores ran it.
//!
//! Issue #576 parallelises the Arrow decode, the per-nuclide collapse and the
//! uncertainty replicas, under a hard "no accuracy loss" constraint. The whole
//! argument for each of them is that the work is independent and merged in a
//! fixed order, so nothing about the answer can depend on the thread count.
//!
//! This is the guard on that. The same case runs in an explicit 1-thread pool
//! and an explicit 7-thread pool -- `install` scopes the nested `par_iter`s the
//! driver runs -- and everything the caller can read must agree **to the last
//! bit**: the inventories, the per-replica ensemble, the sigmas, the sample
//! count and the truncation counters.
//!
//! Seven rather than a power of two on purpose: a bug that only shows when the
//! work does not divide evenly is exactly the kind this is looking for.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::DataUncertainty;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];
const DAY: f64 = 86400.0;

/// Fixed rather than adaptive, so the two runs are compared over the same
/// number of replicas by construction rather than by the assertion below.
const SAMPLES: usize = 24;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Iron, with CONFIG pointed at the fixture cache so the driver's own preload
/// -- the parallel one -- is what resolves and reads the data.
fn iron() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    {
        let mut cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        cfg.set_cross_section("Fe56", Some(&path));
    }
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.volume = Some(1.0);
    m.set_temperature("294");
    Some(m)
}

fn spectra(with_sigma: bool) -> Vec<MultigroupSpectrum> {
    let total: f64 = FLUX.iter().sum();
    vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: with_sigma.then(|| vec![0.05; FLUX.len()]),
    }]
}

fn steps() -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: DAY,
            irradiation: Some((0, FLUX.iter().sum())),
        },
        TransmuteStep {
            dt: DAY,
            irradiation: None,
        },
        TransmuteStep {
            dt: 7.0 * DAY,
            irradiation: Some((0, FLUX.iter().sum())),
        },
    ]
}

/// Everything a caller can read, as bits.
#[derive(Debug, PartialEq)]
struct Answer {
    densities: Vec<(usize, String, u64)>,
    replicas: Vec<(usize, String, u64)>,
    sigmas: Vec<(usize, String, u64)>,
    samples: usize,
    converged: bool,
    truncations: Option<String>,
}

fn run(threads: usize, uncertainty: bool) -> Answer {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");

    pool.install(|| {
        let mut material = iron().expect("checked by the caller");
        let request = uncertainty.then(|| DataUncertainty {
            seed: 7,
            samples: Some(SAMPLES),
            ..Default::default()
        });
        let results = transmute_material(
            &mut material,
            &spectra(uncertainty),
            &steps(),
            chain(),
            &Default::default(),
            Default::default(),
            request.as_ref(),
        )
        .expect("transmute");

        let id = material.material_id.unwrap_or(0);
        let n_steps = results.num_steps();

        let mut densities = Vec::new();
        let mut sigmas = Vec::new();
        let mut replicas = Vec::new();
        for step in 0..=n_steps {
            let inventory = match results.get_material(id, step) {
                Some(m) => m.nuclides.clone(),
                None => continue,
            };
            let mut names: Vec<&String> = inventory.keys().collect();
            names.sort();
            for name in names {
                densities.push((step, name.clone(), inventory[name].to_bits()));
                if let Some(sigma) = results.get_nuclide_uncertainty(id, name, step) {
                    sigmas.push((step, name.clone(), sigma.to_bits()));
                }
            }
            // The per-replica inventories, not just their moments: the ensemble
            // index IS the replica index, so a merge in the wrong order shows
            // here even when the mean does not.
            if let Some(ensemble) = results.uncertainty_inventories(id, step) {
                for (replica, inventory) in ensemble.iter().enumerate() {
                    let mut names: Vec<&String> = inventory.keys().collect();
                    names.sort();
                    for name in names {
                        replicas.push((replica, name.clone(), inventory[name].to_bits()));
                    }
                }
            }
        }

        let info = results.uncertainty_info.as_ref();
        Answer {
            densities,
            replicas,
            sigmas,
            samples: info.map_or(0, |i| i.samples),
            converged: info.is_some_and(|i| i.converged),
            // The counters a replica merges back, which is where a parallel
            // merge that folded a local's zeros over the driver's totals would
            // show.
            truncations: info.map(|i| {
                format!(
                    "{} {} {} {} {}",
                    i.rates_floored,
                    i.rates_sampled,
                    i.flux_bins_floored,
                    i.flux_bins_sampled,
                    i.spectra_with_flux_sigma,
                )
            }),
        }
    })
}

#[test]
fn the_plain_path_does_not_depend_on_the_thread_count() {
    if iron().is_none() {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    }
    assert_eq!(
        run(1, false),
        run(7, false),
        "the inventories moved between a 1-thread and a 7-thread run"
    );
}

#[test]
fn the_uncertainty_path_does_not_depend_on_the_thread_count() {
    if iron().is_none() {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    }
    let one = run(1, true);
    let seven = run(7, true);
    assert_eq!(
        one.samples, SAMPLES,
        "the fixed sample count must be honoured"
    );
    assert!(
        !one.replicas.is_empty(),
        "the ensemble must be populated for this test to mean anything"
    );
    assert_eq!(
        one, seven,
        "the ensemble moved between a 1-thread and a 7-thread run"
    );
}
