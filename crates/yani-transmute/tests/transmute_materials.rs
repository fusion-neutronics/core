//! Several materials transmuted in one call (issue #146).
//!
//! The plural entry point must be a faster way to get the answers the
//! single-material one gives, never different answers. So the main check is
//! that every material in a plural solve comes out bit-identical to solving it
//! alone, with its own spectrum and flux magnitude, and that the collapse
//! sharing it does is reported and never shares across materials that differ.
//!
//! Self-skips when the Fe56 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{
    transmute_material, transmute_materials, CollapseReuse, MultigroupSpectrum,
    TransmutationResults, TransmuteCase, TransmuteStep,
};

const DAY: f64 = 86400.0;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Fe56 at `density` atoms/b-cm, with id `id`.
fn iron(id: u32, density: f64) -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.nuclides.insert("Fe56".to_string(), density);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.set_material_id(id);
    m.read_nuclear_data(&HashMap::from([("Fe56".to_string(), path)]), None)
        .expect("read Fe56");
    Some(m)
}

fn spectrum(masses: &[f64]) -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![1.0e-5, 1.0e2, 1.0e5, 1.0e6, 2.0e7],
        masses: masses.to_vec(),
        relative_std_dev: None,
    }
}

/// Irradiate for two days at `rate`, then cool for one.
fn steps(rate: f64) -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: 2.0 * DAY,
            irradiation: Some((0, rate)),
        },
        TransmuteStep {
            dt: DAY,
            irradiation: None,
        },
    ]
}

fn alone(mut material: Material, s: &MultigroupSpectrum, rate: f64) -> TransmutationResults {
    transmute_material(
        &mut material,
        std::slice::from_ref(s),
        &steps(rate),
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute alone")
}

/// Every nuclide density at every step, bit for bit.
fn assert_same_material(plural: &TransmutationResults, single: &TransmutationResults, id: u32) {
    for step in 0..=2 {
        let a = &plural.get_material(id, step).expect("plural step").nuclides;
        let b = &single.get_material(id, step).expect("single step").nuclides;
        assert_eq!(
            a.len(),
            b.len(),
            "material {id} step {step}: nuclide sets differ"
        );
        for (name, density) in b {
            assert_eq!(
                a[name].to_bits(),
                density.to_bits(),
                "material {id} step {step} {name}: {} vs {density}",
                a[name]
            );
        }
    }
    assert_eq!(plural.get_source_rates(id), single.get_source_rates(id));
    // HashMaps compare by content, and the f64 rates bit for bit through
    // `==` since none of them is NaN.
    assert_eq!(
        plural.get_reaction_rates(id, 0).expect("plural rates"),
        single.get_reaction_rates(id, 0).expect("single rates"),
        "material {id}: per-edge rates differ"
    );
}

/// Each material in a plural solve equals its own solve, and the two that
/// share a spectrum and a composition share one collapse.
#[test]
fn each_material_matches_its_own_solve() {
    // One load, cloned, so the three hold the same `Arc` per nuclide. The
    // collapse key compares loaded data by pointer, and the process-wide
    // nuclide cache hands out a *new* `Arc` when a later request widens the
    // load scope (a sibling test asking for covariance is enough). Loading
    // three times would leave whether they match up to which tests ran in
    // between, and the reuse asserted below is what this test is about.
    let Some(base) = iron(1, 8.5e-2) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let same_steel = |id: u32| {
        let mut m = base.clone();
        m.set_material_id(id);
        m
    };
    let (a, b, c) = (same_steel(1), same_steel(2), same_steel(3));
    // The precondition for the reuse below, so a future split back into three
    // loads fails here with a reason rather than intermittently on the count.
    assert!(
        Arc::ptr_eq(&a.nuclide_data["Fe56"], &b.nuclide_data["Fe56"])
            && Arc::ptr_eq(&a.nuclide_data["Fe56"], &c.nuclide_data["Fe56"]),
        "clones must share one decoded Fe56"
    );
    let fast = spectrum(&[0.0, 0.1, 0.3, 0.6]);
    let soft = spectrum(&[0.5, 0.3, 0.2, 0.0]);

    let expected = [
        alone(a.clone(), &fast, 1.0e14),
        alone(b.clone(), &fast, 3.0e13),
        alone(c.clone(), &soft, 2.0e14),
    ];

    let (mut a, mut b, mut c) = (a, b, c);
    let cases = vec![
        TransmuteCase {
            material: &mut a,
            spectra: vec![fast.clone()],
            steps: steps(1.0e14),
            shielding: None,
        },
        TransmuteCase {
            material: &mut b,
            spectra: vec![fast.clone()],
            steps: steps(3.0e13),
            shielding: None,
        },
        TransmuteCase {
            material: &mut c,
            spectra: vec![soft.clone()],
            steps: steps(2.0e14),
            shielding: None,
        },
    ];
    let plural = transmute_materials(
        cases,
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute together");

    let mut ids: Vec<u32> = plural.materials.keys().copied().collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
    for (id, single) in [1, 2, 3].into_iter().zip(&expected) {
        assert_same_material(&plural, single, id);
    }
    assert_eq!(plural.get_source_rates(2), Some(&[3.0e13, 0.0][..]));
    // Materials 1 and 2 are the same steel under the same spectrum at
    // different flux magnitudes, which is one collapse between them.
    assert_eq!(
        plural.collapse_reuse,
        Some(CollapseReuse {
            performed: 2,
            requested: 3
        })
    );
    // Each material keeps its own spectrum for re-deriving rate spectra.
    assert!(plural
        .get_reaction_rate_spectrum(3, "Fe56", "(n,gamma)", 0)
        .is_some());
}

/// A collapse is shared only between identical inputs: the same spectrum on a
/// different composition is collapsed again, because the report it carries
/// depends on the densities.
#[test]
fn a_different_composition_is_collapsed_again() {
    let (Some(mut a), Some(mut b)) = (iron(1, 8.5e-2), iron(2, 1.0e-3)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let fast = spectrum(&[0.0, 0.1, 0.3, 0.6]);
    let plural = transmute_materials(
        vec![
            TransmuteCase {
                material: &mut a,
                spectra: vec![fast.clone()],
                steps: steps(1.0e14),
                shielding: None,
            },
            TransmuteCase {
                material: &mut b,
                spectra: vec![fast],
                steps: steps(1.0e14),
                shielding: None,
            },
        ],
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute together");
    assert_eq!(
        plural.collapse_reuse,
        Some(CollapseReuse {
            performed: 2,
            requested: 2
        })
    );
    assert_eq!(plural.shielding_info.len(), 2, "one report per material");
}

/// The uncertainty replicas of a material in a plural solve are those of its
/// own solve: each material's replicas see only its own spectra.
#[test]
fn uncertainty_matches_each_materials_own_solve() {
    let (Some(a), Some(b)) = (iron(1, 8.5e-2), iron(2, 8.5e-2)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let with_sigma = |masses: &[f64], sigma: f64| MultigroupSpectrum {
        relative_std_dev: Some(vec![sigma; masses.len()]),
        ..spectrum(masses)
    };
    let fast = with_sigma(&[0.0, 0.1, 0.3, 0.6], 0.05);
    let soft = with_sigma(&[0.5, 0.3, 0.2, 0.0], 0.2);
    let request = DataUncertainty {
        seed: 7,
        samples: Some(64),
        sources: vec![Source::FluxSpectrum],
    };
    let solo = |mut m: Material, s: &MultigroupSpectrum| {
        transmute_material(
            &mut m,
            std::slice::from_ref(s),
            &steps(1.0e14),
            chain(),
            &Default::default(),
            Default::default(),
            Some(&request),
        )
        .expect("alone")
    };
    let expected = [solo(a.clone(), &fast), solo(b.clone(), &soft)];

    let (mut a, mut b) = (a, b);
    let plural = transmute_materials(
        vec![
            TransmuteCase {
                material: &mut a,
                spectra: vec![fast],
                steps: steps(1.0e14),
                shielding: None,
            },
            TransmuteCase {
                material: &mut b,
                spectra: vec![soft],
                steps: steps(1.0e14),
                shielding: None,
            },
        ],
        chain(),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute together");

    for (id, single) in [1u32, 2].into_iter().zip(&expected) {
        let got = plural.uncertainty[&id].std_dev_at(0);
        let want = single.uncertainty[&id].std_dev_at(0);
        assert!(!want.is_empty());
        for (name, sigma) in &want {
            assert_eq!(got[name].to_bits(), sigma.to_bits(), "material {id} {name}");
        }
        assert_eq!(
            plural.uncertainty_info[&id].samples,
            single.uncertainty_info[&id].samples
        );
    }
}

/// Two materials with one id cannot both be in a result keyed by id.
#[test]
fn a_repeated_id_is_refused() {
    let (Some(mut a), Some(mut b)) = (iron(4, 8.5e-2), iron(4, 8.5e-2)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let fast = spectrum(&[0.0, 0.1, 0.3, 0.6]);
    let err = transmute_materials(
        vec![
            TransmuteCase {
                material: &mut a,
                spectra: vec![fast.clone()],
                steps: steps(1.0e14),
                shielding: None,
            },
            TransmuteCase {
                material: &mut b,
                spectra: vec![fast],
                steps: steps(1.0e14),
                shielding: None,
            },
        ],
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect_err("duplicate ids");
    assert!(err.to_string().contains("both have id 4"), "{err}");
}

/// The result has one series of times, so the timelines must agree on the
/// durations and on which steps irradiate.
#[test]
fn a_different_timeline_is_refused() {
    let (Some(mut a), Some(mut b)) = (iron(1, 8.5e-2), iron(2, 8.5e-2)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let fast = spectrum(&[0.0, 0.1, 0.3, 0.6]);
    let mut longer = steps(1.0e14);
    longer[1].dt = 2.0 * DAY;
    let mut run = |other: Vec<TransmuteStep>| {
        transmute_materials(
            vec![
                TransmuteCase {
                    material: &mut a,
                    spectra: vec![fast.clone()],
                    steps: steps(1.0e14),
                    shielding: None,
                },
                TransmuteCase {
                    material: &mut b,
                    spectra: vec![fast.clone()],
                    steps: other,
                    shielding: None,
                },
            ],
            chain(),
            &Default::default(),
            Default::default(),
            None,
        )
        .expect_err("timelines differ")
        .to_string()
    };
    let err = run(longer);
    assert!(err.contains("step 1 lasts"), "{err}");
    let mut cooled = steps(1.0e14);
    cooled[0].irradiation = None;
    let err = run(cooled);
    assert!(err.contains("step 0 is a cooldown"), "{err}");
}
