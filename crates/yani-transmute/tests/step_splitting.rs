//! Splitting a step changes nothing, because the burnup matrix does not depend
//! on the composition (issue #563).
//!
//! `accumulate_matrix` (`crates/yani/src/matrix.rs`) takes the chain, the
//! reaction rates and the fission-yield weights. It does NOT take the
//! composition: every entry is a decay constant or a `sigma*phi` rate, and none
//! of them depend on `N`.
//!
//! So for a supplied, fixed flux the matrix `A` is constant over a step and
//! `exp(A*dt)*N(t)` is the exact solution of the Bateman system at any `dt`.
//! There is no step-length error to shorten your way out of, and the docs used
//! to advise shortening steps as though there were.
//!
//! What this does NOT say: that a real irradiation is insensitive to step
//! length. It is not, because the flux itself changes as the composition does.
//! But that is the caller's input to `Material.transmute`, not something the
//! stepper approximates, and the `method="coupled"` path is where the
//! distinction actually bites.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];
const YEAR: f64 = 365.0 * 24.0 * 3600.0;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn iron(data: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Fe56");
    m
}

fn spectra() -> Vec<MultigroupSpectrum> {
    let total: f64 = FLUX.iter().sum();
    vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        // This spectrum is a fixture with no stated error, which is the common
        // case and is not the same as one measured to be exact.
        relative_std_dev: None,
    }]
}

/// `n` equal irradiation steps covering `YEAR` in total.
fn steps(n: usize) -> Vec<TransmuteStep> {
    let rate: f64 = FLUX.iter().sum();
    (0..n)
        .map(|_| TransmuteStep {
            dt: YEAR / n as f64,
            irradiation: Some((0, rate)),
        })
        .collect()
}

fn final_inventory(material: &mut Material, n: usize) -> HashMap<String, f64> {
    let results = transmute_material(
        material,
        &spectra(),
        &steps(n),
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute");
    results
        .get_final_material(material.material_id.unwrap_or(0))
        .expect("a final material")
        .nuclides
        .clone()
}

/// One year in a single step equals one year in many, to round-off.
///
/// The property the corrected documentation claims. A user shortening steps on
/// a fixed spectrum is buying nothing, and this is what says so.
#[test]
fn splitting_a_step_does_not_change_the_answer() {
    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping splitting_a_step_does_not_change_the_answer (fixtures missing)");
        return;
    };
    let mut material = iron(Path::new(&dir));

    let one = final_inventory(&mut material, 1);
    let twelve = final_inventory(&mut material, 12);
    let many = final_inventory(&mut material, 365);

    assert!(!one.is_empty(), "the run produced nothing to compare");

    // Compared against the largest density rather than each nuclide's own, so
    // a trace many decades down cannot fail this on its own noise: at 1e-30 of
    // the total it carries no significant figures to disagree about.
    let scale = one.values().cloned().fold(0.0_f64, f64::max);
    let mut worst = 0.0_f64;
    let mut worst_name = String::new();

    for (name, &a) in &one {
        for other in [&twelve, &many] {
            let b = other.get(name).copied().unwrap_or(0.0);
            let d = (a - b).abs() / scale;
            if d > worst {
                worst = d;
                worst_name = name.clone();
            }
        }
    }

    assert!(
        worst < 1.0e-12,
        "splitting the step moved the inventory by {worst:.3e} of the largest \
         density (worst nuclide {worst_name}); the burnup matrix does not \
         depend on composition, so a fixed-flux step is exact at any length \
         and this should be round-off"
    );
}

/// The same, with a decay-only tail, so the split falls inside a cooling step
/// as well as inside an irradiation.
///
/// Decay-only steps take a different path through the stepper (an empty rate
/// map and the base chain rather than the folded one), so they are worth
/// splitting separately.
#[test]
fn splitting_a_cooling_step_does_not_change_the_answer() {
    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!(
            "skipping splitting_a_cooling_step_does_not_change_the_answer (fixtures missing)"
        );
        return;
    };
    let mut material = iron(Path::new(&dir));
    let rate: f64 = FLUX.iter().sum();

    let mut run = |cooling_steps: usize| {
        let mut s = vec![TransmuteStep {
            dt: YEAR,
            irradiation: Some((0, rate)),
        }];
        s.extend((0..cooling_steps).map(|_| TransmuteStep {
            dt: YEAR / cooling_steps as f64,
            irradiation: None,
        }));
        let results = transmute_material(
            &mut material,
            &spectra(),
            &s,
            chain(),
            &Default::default(),
            Default::default(),
            None,
        )
        .expect("transmute");
        results
            .get_final_material(material.material_id.unwrap_or(0))
            .expect("a final material")
            .nuclides
            .clone()
    };

    let one = run(1);
    let many = run(52);
    let scale = one.values().cloned().fold(0.0_f64, f64::max);

    let worst = one
        .iter()
        .map(|(name, &a)| (a - many.get(name).copied().unwrap_or(0.0)).abs() / scale)
        .fold(0.0_f64, f64::max);

    assert!(
        worst < 1.0e-12,
        "splitting the cooling step moved the inventory by {worst:.3e} of the \
         largest density; decay is a constant matrix too"
    );
}
