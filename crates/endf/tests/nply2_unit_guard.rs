//! The NPLY=2 unit correction fires exactly where its comment says.
//!
//! ENDF/B-VII.1 left some second-order MT=458 coefficients in MeV, so
//! `FissionEnergyRelease::from_material` divides a second-order coefficient by
//! 1e6 when it is too large to be physics. "Too large" is spelled out as
//! `|c2| * (5 MeV)^2 > 100 MeV` in eV, which puts the boundary at `4e-6`.
//!
//! A report once had the guard firing an order of magnitude below that, which
//! would have meant a legitimately small coefficient in a modern evaluation was
//! being silently divided. It does not. The measurement behind that report injected
//! into the fixed-format ENDF text, where MT=458's LIST is laid out order-major
//! (a value and an uncertainty for each of the nine components, 18 numbers per
//! order), so component `i` at order `k` is raw index `2i + 18k` rather than
//! anywhere near the component's other orders. An edit made at the intuitive
//! position lands in a different component.
//!
//! These inject at the parsed level for that reason: it is the same guard, with
//! none of the column arithmetic that made the original measurement wrong.

use endf::Material;
use std::path::Path;

fn read_text(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut out = Vec::new();
    std::io::Read::read_to_end(
        &mut lzma_rust2::XzReader::new(raw.as_slice(), true),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("decompressing {}: {e}", path.display()));
    String::from_utf8(out).expect("a fixture is not UTF-8")
}

/// Am244 is NPLY=2 with a zero second-order prompt-photon coefficient, so it
/// reaches the guard and carries nothing there to begin with.
const PROMPT_PHOTONS: usize = 3;

/// Read back the prompt-photon coefficients after setting the second-order one.
fn coeffs_after_injecting(c2: f64) -> Vec<f64> {
    let mut material = Material::from_str(&read_text("n-095_Am_244.endf.xz")).unwrap();
    match material.section_data.get_mut(&(1, 458)) {
        Some(endf::Section::Mf1Mt458(section)) => {
            assert_eq!(section.nply, 2, "the fixture must reach the NPLY=2 guard");
            match &mut section.components[PROMPT_PHOTONS] {
                endf::mf::mf1::FissionEnergyRelease::Polynomial(pairs) => pairs[2].0 = c2,
                other => panic!("prompt photons is not a polynomial: {other:?}"),
            }
        }
        other => panic!("no MF=1 MT=458 section: {other:?}"),
    }
    let released = endf::FissionEnergyRelease::from_material(&material, None).unwrap();
    match &released.prompt_photons {
        endf::fission_energy::Component::Polynomial(p) => p.coefficients.clone(),
        other => panic!("prompt photons came back tabulated: {other:?}"),
    }
}

/// Below the boundary the coefficient is physics and must survive untouched.
///
/// `3.9e-6` is the largest of these that was reported as divided, so it is the
/// case that would fail if the guard really did fire early.
#[test]
fn a_small_second_order_coefficient_is_left_alone() {
    for c2 in [0.0, 1e-9, 1e-6, 3.9e-6] {
        let got = coeffs_after_injecting(c2);
        assert_eq!(
            got[2],
            c2,
            "|c2| * (5e6)^2 = {:e} is under the 1e8 threshold, so {c2:e} is physics \
             and must not be divided",
            c2.abs() * (5.0e6f64).powi(2)
        );
    }
}

/// Above it, the coefficient is the ENDF/B-VII.1 units error and is converted.
#[test]
fn a_second_order_coefficient_too_large_for_physics_is_converted_from_mev() {
    for c2 in [4.1e-6, 1.7e-5, 1.0e-3] {
        let got = coeffs_after_injecting(c2);
        assert!(
            (got[2] - c2 / 1.0e6).abs() <= (c2 / 1.0e6).abs() * 1e-12,
            "|c2| * (5e6)^2 = {:e} exceeds the 1e8 threshold, so {c2:e} is the MeV \
             units error and should have become {:e}, got {:e}",
            c2.abs() * (5.0e6f64).powi(2),
            c2 / 1.0e6,
            got[2]
        );
    }
}

/// The boundary is where the comment's arithmetic puts it, not an order out.
///
/// This is the assertion the report actually turned on: it claimed the crossing was
/// "somewhere below 1e-6" rather than at 4e-6.
#[test]
fn the_boundary_sits_at_four_micro_ev_per_ev_squared() {
    let below = coeffs_after_injecting(3.999e-6);
    assert_eq!(below[2], 3.999e-6, "just below the threshold, untouched");

    let above = coeffs_after_injecting(4.001e-6);
    assert!(
        (above[2] - 4.001e-12).abs() <= 4.001e-24,
        "just above the threshold, divided; got {:e}",
        above[2]
    );

    // The negative side uses the same magnitude test, so it crosses together.
    let negative = coeffs_after_injecting(-4.001e-6);
    assert!(
        (negative[2] + 4.001e-12).abs() <= 4.001e-24,
        "the guard tests |c2|, so the negative side crosses at the same place; got {:e}",
        negative[2]
    );
}
