//! Issue #369: read the `fission_photon.arrow` section the converter writes and
//! reproduce OpenMC's delayed-photon scaling exactly.
//!
//! This is a CROSS-REPO integration test. The fixtures are real files produced by
//! `nuclear_data_to_yamc_format`'s `_fission_photon_rows` + `FISSION_PHOTON_SCHEMA`
//! from the ENDF/B-VIII.1 evaluations, so it pins the thing unit tests on either
//! side cannot: that the column names, types and row layout the converter WRITES
//! are the ones yamc READS. A rename on one side breaks this test rather than
//! silently disabling the scaling again.
//!
//! The two fixtures cover both representations, which is the whole reason the
//! section stores functions rather than coefficients:
//!   * U235  -- prompt is Tabulated1D (18 points, lin-lin), delayed is Polynomial
//!   * Ac226 -- both Polynomial
//!
//! Expected values come from OpenMC itself: `fe.prompt_photons(E)` and
//! `fe.delayed_photons(E)` on `openmc.data.IncidentNeutron`, so this is a
//! cross-code check and not a self-consistency one.

use std::path::Path;

use yamc_nuclide::arrow_helpers::{get_f64_list, get_i32_list, get_str, read_arrow_file};
use yamc_nuclide::fission_photon::{FissionPhotonRelease, ReleaseFunction};

/// Rebuild the release from a section file. Mirrors the private reader in
/// `nuclide_arrow.rs`; kept in step by asserting the same column names.
fn load(fixture: &str) -> FissionPhotonRelease {
    let path = format!("tests/fission_photon_{fixture}.arrow");
    let batch = read_arrow_file(Path::new(&path)).expect("fixture readable");
    assert_eq!(batch.num_rows(), 2, "one row per term");

    let mut prompt = None;
    let mut delayed = None;
    for row in 0..batch.num_rows() {
        let role = get_str(&batch, "role", row).expect("role column");
        let kind = get_str(&batch, "kind", row).expect("kind column");
        let func = match kind.as_str() {
            "polynomial" => ReleaseFunction::Polynomial(
                get_f64_list(&batch, "coefficients", row).expect("coefficients"),
            ),
            "tabulated" => ReleaseFunction::from_tabulated(
                get_f64_list(&batch, "x", row).expect("x"),
                get_f64_list(&batch, "y", row).expect("y"),
                &get_i32_list(&batch, "interpolation", row).expect("interpolation"),
                &get_i32_list(&batch, "breakpoints", row).expect("breakpoints"),
                &format!("{fixture} {role}"),
            )
            .expect("a published table must be a single lin-lin region"),
            other => panic!("unexpected kind {other:?}"),
        };
        match role.as_str() {
            "prompt_photons" => prompt = Some(func),
            "delayed_photons" => delayed = Some(func),
            other => panic!("unexpected role {other:?}"),
        }
    }
    FissionPhotonRelease {
        prompt: prompt.expect("prompt_photons row"),
        delayed: delayed.expect("delayed_photons row"),
    }
}

/// U235's prompt term is tabulated. This is the case a polynomial-only reader
/// gets catastrophically wrong: Horner over the flattened 36 table values at
/// 2 MeV gives 4.3e227 instead of 7.99e6.
#[test]
fn u235_tabulated_prompt_matches_openmc() {
    let release = load("U235");
    assert!(
        matches!(release.prompt, ReleaseFunction::LinLinTable { .. }),
        "U235's prompt photon release is tabulated in ENDF/B-VIII.1; reading it as a \
         polynomial is the #369 hazard"
    );
    assert!(
        matches!(release.delayed, ReleaseFunction::Polynomial(_)),
        "U235's delayed photon release is a polynomial, so the two terms must be \
         allowed to differ in representation"
    );
    for (e, expected) in [(2.0e6_f64, 1.7734235644_f64), (1.4e7, 1.5130148366)] {
        let f = release.scaling(e);
        assert!(
            (f - expected).abs() < 1.0e-9,
            "U235 scaling at {e:.3e} eV: expected OpenMC's {expected:.10}, got {f:.10}"
        );
    }
}

/// Ac226 has both terms as polynomials, the majority case (76 of 79).
#[test]
fn ac226_polynomial_terms_match_openmc() {
    let release = load("Ac226");
    assert!(matches!(release.prompt, ReleaseFunction::Polynomial(_)));
    assert!(matches!(release.delayed, ReleaseFunction::Polynomial(_)));
    for (e, expected) in [(2.0e6_f64, 1.5255580929_f64), (1.4e7, 1.3829800673)] {
        let f = release.scaling(e);
        assert!(
            (f - expected).abs() < 1.0e-9,
            "Ac226 scaling at {e:.3e} eV: expected OpenMC's {expected:.10}, got {f:.10}"
        );
    }
}

/// The scaling can only ever increase fission photon production, and it is a
/// large effect -- which is why leaving it at 1.0 showed up as a 1.4 to 2.6%
/// secondary-photon flux deficit in the V&V.
#[test]
fn the_scaling_is_always_above_one_and_substantial() {
    for fixture in ["U235", "Ac226"] {
        let release = load(fixture);
        for e in [0.0253_f64, 1.0e3, 1.0e6, 1.4e7] {
            let f = release.scaling(e);
            assert!(
                f > 1.3 && f < 2.0,
                "{fixture} at {e:.3e} eV gave f = {f}, outside the range these \
                 evaluations can produce; the factor is real and large, not a nudge"
            );
        }
    }
}
