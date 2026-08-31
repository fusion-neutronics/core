//! The macroscopic cross section must be bit-reproducible (issue #598).
//!
//! Same defect as #502 (`matrix.rs`), #576 (`composition.rs`) and the four
//! sites fixed in #597, one layer further in. `Material::nuclides` is a
//! `HashMap`, Rust seeds each instance separately, and the accumulation loop
//! in `macro_xs.rs` walked it directly:
//!
//! ```text
//! for nuclide in self.nuclides.keys() {   // HashMap iteration order
//!     ...
//!     macro_values[i] += atoms_per_bcm * xs;   // every nuclide sums here
//! }
//! ```
//!
//! Every nuclide accumulates into the same slot, so the summation order, and
//! with it the last bit of the material's macroscopic cross section at every
//! grid point, differed between runs.
//!
//! This one feeds transport, which is why it was held back from #597: the
//! cross sections these tests pin are what a collision samples against.
//!
//! Rebuilt rather than compared with itself: the map is constructed per
//! material, so a fresh one is the only way to get a fresh iteration order.

use std::collections::HashMap;

use yamc_materials::Material;

/// A simplified stainless steel: the eight nuclides of it that the fixture
/// set (`scripts/fetch_test_fixtures.py`) carries with full transport
/// sections, at their natural-abundance fractions.
///
/// Restricted to those on purpose, so this runs in CI rather than self-
/// skipping. Eight is enough for the iteration order to vary between two
/// `HashMap` instances, and the fractions span four orders of magnitude, so
/// adding the small terms before or after the large ones rounds differently.
/// The full 24-nuclide version is in `composition_reproducibility.rs`, which
/// needs no nuclear data at all.
fn steel() -> HashMap<String, f64> {
    HashMap::from(
        [
            ("Fe54", 0.0380),
            ("Fe56", 0.5966),
            ("Fe57", 0.0138),
            ("Fe58", 0.0018),
            ("Cr52", 0.1423),
            ("Si28", 0.0092),
            ("Si29", 0.0005),
            ("Si30", 0.0003),
        ]
        .map(|(n, f)| (n.to_string(), f)),
    )
}

/// Cache paths for every nuclide of the composition, or `None` if any is absent.
fn data_paths(composition: &HashMap<String, f64>) -> Option<HashMap<String, String>> {
    composition
        .keys()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.clone(), p)))
        .collect()
}

/// Built from a FRESH `steel()` every time, never from a clone.
///
/// `HashMap::clone` copies the hasher along with the contents, so a cloned
/// composition iterates in exactly the order its source did. Cloning one map
/// per build would hold the iteration order fixed and the test would pass
/// against the unfixed code, which is what it did when first written.
fn build(paths: &HashMap<String, String>) -> Material {
    let mut m = Material::new(steel(), "atom", "g/cm3", Some(7.9)).expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(paths, None).expect("nuclear data");
    m
}

/// Every value of every MT, as raw bits, in a deterministic order.
fn bits(macro_xs: &HashMap<i32, Vec<f64>>) -> Vec<(i32, Vec<u64>)> {
    let mut out: Vec<(i32, Vec<u64>)> = macro_xs
        .iter()
        .map(|(&mt, xs)| (mt, xs.iter().map(|v| v.to_bits()).collect()))
        .collect();
    out.sort_by_key(|(mt, _)| *mt);
    out
}

/// The whole point: the same material built twice gives the same numbers.
#[test]
fn the_macroscopic_cross_section_is_bit_identical_across_builds() {
    let composition = steel();
    let Some(paths) = data_paths(&composition) else {
        eprintln!("SKIP: the steel nuclides are not in the test cache; this test checked nothing");
        return;
    };

    let mts = vec![1, 2, 102];
    let (grid, first) = build(&paths).calculate_macroscopic_xs(&mts, false);
    let expected = bits(&first);
    assert!(
        expected.iter().any(|(_, xs)| xs.len() > 1000),
        "the union grid is too small for this to mean anything: {:?}",
        expected
            .iter()
            .map(|(mt, xs)| (mt, xs.len()))
            .collect::<Vec<_>>()
    );

    for i in 1..8 {
        let (grid_again, again) = build(&paths).calculate_macroscopic_xs(&mts, false);
        assert_eq!(
            grid_again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            grid.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the unified energy grid on build {i} differs from the first"
        );

        let again = bits(&again);
        for ((mt, want), (_, got)) in expected.iter().zip(again.iter()) {
            let differing = want.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "MT {mt}: {differing} of {} grid points differ on build {i}",
                want.len()
            );
        }
    }
}
