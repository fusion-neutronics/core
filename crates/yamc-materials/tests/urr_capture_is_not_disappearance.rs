//! An `(n,gamma)` tally must score MT 102 even where probability tables are in
//! range (fusion-neutronics/core#106).
//!
//! `compute_urr_macro_xs` used one accumulator for both the capture score and
//! the absorption score, filled from MT 101 (disappearance) for every nuclide
//! whose tables are not in range, because MT 27 is built from it as
//! `disappearance + fission`. `macro_xs_by_mt` then answered MT 102 from it as
//! well, so inside the band an `(n,gamma)` tally collected every
//! charged-particle channel of every other nuclide in the material. On the
//! Fe58 / Be9 mixture below that reads 3.4x high, Be9's `(n,alpha)` being the
//! whole of the excess; OpenMC substitutes disappearance for capture only on
//! the nuclide whose own `use_ptable` is set.
//!
//! The material is Fe58 with Be9. Fe58 has tables over [350 keV, 3 MeV] and no
//! charged-particle channel inside them (MT 103 to MT 107 are exactly zero
//! there against an MT 102 of a few millibarns), so its disappearance IS its
//! capture and the perturbed value is the same either way. Be9 has no tables
//! and, at the 2 MeV table energy this reads, an `(n,alpha)` of 45 mb against
//! an Fe58 `(n,gamma)` of 2.1 mb, so the material's MT 101 is 22x its MT 102
//! and whichever accumulator the capture score reads is unmistakable.
//!
//! The check is a band average rather than a single draw. Each table's
//! probability-weighted factor is exactly 1.0 at every table energy, so
//! averaging the perturbed capture over the band distribution has to come back
//! to the smooth value, and the two scores then have to come back to two
//! DIFFERENT smooth values: MT 102 and MT 101. A single draw could not tell a
//! wrong accumulator from an ordinary fluctuation.

use std::collections::HashMap;

use yamc_materials::Material;

/// A Fe58 table energy, so the probability-weighted factor is exactly 1.0
/// rather than interpolated between two energies at which it is. 2 MeV rather
/// than one of the lower ones because Be9's `(n,alpha)` rises steeply through
/// the band: at 350 keV it is identically zero and the two scores cannot be
/// told apart at all, at 1 MeV the material's MT 101 is only 1.26x its MT 102,
/// and here it is 22x.
const ENERGY: f64 = 2.0e6;

/// Base randoms swept over (0, 1). `compute_urr_macro_xs` hashes this per
/// nuclide (issue #204), so a uniform sweep gives each nuclide a uniform draw
/// over its own bands. Deterministic, so the test cannot flake.
const SAMPLES: usize = 20_000;

fn composition() -> HashMap<String, f64> {
    HashMap::from([("Fe58".to_string(), 0.5), ("Be9".to_string(), 0.5)])
}

fn build() -> Option<Material> {
    let paths: HashMap<String, String> = composition()
        .keys()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.clone(), p)))
        .collect::<Option<_>>()?;
    let mut m = Material::new(composition(), "atom", "g/cm3", Some(2.0)).expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(&paths, None).expect("nuclear data");
    // The smooth macroscopic tables the comparison reads are built on demand,
    // not by the load, and `compute_urr_macro_xs` needs the per-temperature
    // fast grid this also builds.
    m.calculate_macroscopic_xs(&vec![1, 2, 18, 101, 102], true);
    Some(m)
}

#[test]
fn a_capture_score_inside_the_band_is_mt_102_and_not_disappearance() {
    let Some(material) = build() else {
        eprintln!("SKIP: Fe58 or Be9 is not in the test cache; this test checked nothing");
        return;
    };

    let smooth_capture = material.lookup_xs_by_mt(102, ENERGY);
    let smooth_disappearance = material.lookup_xs_by_mt(101, ENERGY);
    assert!(
        smooth_capture > 0.0 && smooth_disappearance > 5.0 * smooth_capture,
        "this material cannot tell the two scores apart: MT 102 {smooth_capture:.4e}, \
         MT 101 {smooth_disappearance:.4e}. Be9's (n,alpha) is what separates them, so a \
         fixture change that dropped it would make this test vacuous."
    );

    let mut capture = 0.0;
    let mut disappearance = 0.0;
    let mut sampled = 0usize;
    for i in 0..SAMPLES {
        let base = (i as f64 + 0.5) / SAMPLES as f64;
        let urr = material
            .compute_urr_macro_xs(ENERGY, base)
            .expect("Fe58's tables cover 1 MeV, so the material has an URR sample here");
        capture += urr.capture;
        disappearance += urr.absorption - urr.fission;
        sampled += 1;
    }
    let capture = capture / sampled as f64;
    let disappearance = disappearance / sampled as f64;

    // The band average of the capture score is the smooth MT 102, because the
    // table's own probability-weighted factor is 1.0 here. Before the fix this
    // came back as the smooth MT 101 instead, 22x larger on this material.
    assert!(
        (capture / smooth_capture - 1.0).abs() < 0.01,
        "band-averaged capture {capture:.6e} against the smooth MT 102 {smooth_capture:.6e} \
         (ratio {:.4}); the smooth MT 101 is {smooth_disappearance:.6e}, so a capture score \
         reading disappearance lands at a ratio of {:.2}",
        capture / smooth_capture,
        smooth_disappearance / smooth_capture,
    );

    // And the absorption score still means MT 27: disappearance plus fission.
    assert!(
        (disappearance / smooth_disappearance - 1.0).abs() < 0.01,
        "band-averaged disappearance {disappearance:.6e} against the smooth MT 101 \
         {smooth_disappearance:.6e} (ratio {:.4})",
        disappearance / smooth_disappearance,
    );
}

/// The single-nuclide case the V&V sweep runs, which is why this went
/// unnoticed: with only Fe58 in the material there is no other nuclide to
/// donate its charged-particle channels, so both scores agree to the last bit
/// and the defect is invisible.
#[test]
fn a_single_urr_nuclide_cannot_show_the_difference() {
    let Some(path) = yamc_test_cache::nuclide("Fe58") else {
        eprintln!("SKIP: Fe58 is not in the test cache; this test checked nothing");
        return;
    };
    let composition = HashMap::from([("Fe58".to_string(), 1.0)]);
    let mut material = Material::new(composition, "atom", "g/cm3", Some(1.0)).expect("material");
    material.set_temperature("294");
    material
        .read_nuclear_data(&HashMap::from([("Fe58".to_string(), path)]), None)
        .expect("nuclear data");
    material.calculate_macroscopic_xs(&vec![1, 2, 18, 101, 102], true);

    let urr = material
        .compute_urr_macro_xs(ENERGY, 0.5)
        .expect("Fe58's tables cover 1 MeV");
    assert_eq!(
        urr.capture,
        urr.absorption - urr.fission,
        "Fe58 has no charged-particle channel inside its band, so its capture and its \
         disappearance are the same number"
    );
}
