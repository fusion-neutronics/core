//! The atom densities a material starts from must be bit-reproducible.
//!
//! Same defect as issue #502, one layer up. `Material::nuclides` is a
//! `HashMap`, Rust seeds each instance separately, and
//! `get_atoms_per_barn_cm` summed the fractions straight out of it. Two
//! materials built from the same composition therefore normalized by totals
//! that differed in the last bit, and every atom density -- and so every
//! inventory the transmutation solve produced from it -- differed with them.
//!
//! Measured on the steel `tools/bench_transmute.py` benchmarks: 103 of 200
//! final densities disagreed at ~1e-15 between one process and the next, with
//! no Monte Carlo anywhere in the path. That is small, but it is also exactly
//! the size of the difference issue #576's speed work has to prove it does NOT
//! make, so it had to stop moving before any of that could be checked.
//!
//! Rebuilt rather than compared with itself: the map is constructed per
//! material, so a fresh one is the only way to get a fresh iteration order.

use std::collections::HashMap;

use yamc_materials::material::Material;

/// A composition wide enough for the iteration order to actually vary: 24
/// nuclides is the natural-abundance expansion of a stainless steel, and the
/// magnitudes are spread over four orders of magnitude, so adding the small
/// terms before or after the large ones rounds differently.
fn steel() -> HashMap<String, f64> {
    HashMap::from(
        [
            ("Fe54", 0.0380),
            ("Fe56", 0.5966),
            ("Fe57", 0.0138),
            ("Fe58", 0.0018),
            ("Cr50", 0.0071),
            ("Cr52", 0.1423),
            ("Cr53", 0.0164),
            ("Cr54", 0.0042),
            ("Ni58", 0.0806),
            ("Ni60", 0.0318),
            ("Ni61", 0.0014),
            ("Ni62", 0.0046),
            ("Ni64", 0.0012),
            ("Mo92", 0.0035),
            ("Mo94", 0.0022),
            ("Mo95", 0.0039),
            ("Mo96", 0.0042),
            ("Mo97", 0.0024),
            ("Mo98", 0.0062),
            ("Mo100", 0.0025),
            ("Mn55", 0.0200),
            ("Si28", 0.0092),
            ("Si29", 0.0005),
            ("Si30", 0.0003),
        ]
        .map(|(n, f)| (n.to_string(), f)),
    )
}

fn bits(densities: &HashMap<String, f64>) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = densities
        .iter()
        .map(|(n, v)| (n.clone(), v.to_bits()))
        .collect();
    out.sort();
    out
}

#[test]
fn atom_densities_are_bit_identical_across_builds() {
    for fraction_type in ["mass", "atom"] {
        let first = Material::new(steel(), fraction_type, "g/cm3", Some(7.9))
            .expect("steel")
            .get_atoms_per_barn_cm()
            .expect("atom densities");
        let expected = bits(&first);

        for i in 1..25 {
            let again = Material::new(steel(), fraction_type, "g/cm3", Some(7.9))
                .expect("steel")
                .get_atoms_per_barn_cm()
                .expect("atom densities");
            assert_eq!(
                bits(&again),
                expected,
                "{fraction_type} fractions: build {i} differs from the first"
            );
        }
    }
}

#[test]
fn the_derived_scalars_are_bit_identical_across_builds() {
    for fraction_type in ["mass", "atom"] {
        let build = || Material::new(steel(), fraction_type, "g/cm3", Some(7.9)).expect("steel");
        let molar = build().average_molar_mass().expect("molar mass").to_bits();
        let total = build().total_atom_density().expect("total").to_bits();

        for i in 1..25 {
            assert_eq!(
                build().average_molar_mass().expect("molar mass").to_bits(),
                molar,
                "{fraction_type} fractions: molar mass on build {i} differs"
            );
            assert_eq!(
                build().total_atom_density().expect("total").to_bits(),
                total,
                "{fraction_type} fractions: total atom density on build {i} differs"
            );
        }
    }
}

/// A "sum"-mode material's mass density is the other sum over the same map.
#[test]
fn the_sum_mode_mass_density_is_bit_identical_across_builds() {
    let build = || {
        let mut m = Material::new(steel(), "atom", "g/cm3", Some(7.9)).expect("steel");
        m = m.to_sum_mode().expect("sum mode");
        m
    };
    let expected = build().get_mass_density().expect("mass density").to_bits();
    for i in 1..25 {
        assert_eq!(
            build().get_mass_density().expect("mass density").to_bits(),
            expected,
            "build {i} differs from the first"
        );
    }
}

/// `mix_materials` is the one function in `composition.rs` that the name-order
/// fix did not reach: it summed its two per-cc maps straight out of `HashMap`
/// iteration order, and those totals normalize every nuclide fraction and
/// become the mixed material's density. Measured before the fix, mixing the
/// same two materials gave two different densities across eight runs.
#[test]
fn a_mixed_material_is_bit_identical_across_builds() {
    // Two multi-isotope elements, so both maps carry several terms to add.
    let iron = || Material::new(steel(), "mass", "g/cm3", Some(7.9)).expect("steel");
    let water = || {
        Material::new(
            HashMap::from([("H1", 0.1119), ("O16", 0.8881)].map(|(n, f)| (n.to_string(), f))),
            "mass",
            "g/cm3",
            Some(1.0),
        )
        .expect("water")
    };

    for fraction_type in ["volume", "mass", "atom"] {
        let build = || {
            let a = iron();
            let b = water();
            Material::mix_materials(&[&a, &b], &[0.6, 0.4], fraction_type, None, None).expect("mix")
        };

        let first = build();
        let density = first.get_mass_density().expect("density").to_bits();
        let fractions = bits(&first.nuclides);
        assert!(
            first.nuclides.len() > 2,
            "the mix must carry several nuclides for this to mean anything"
        );

        for i in 1..25 {
            let again = build();
            assert_eq!(
                again.get_mass_density().expect("density").to_bits(),
                density,
                "{fraction_type}: the mixed density on build {i} differs"
            );
            assert_eq!(
                bits(&again.nuclides),
                fractions,
                "{fraction_type}: the mixed fractions on build {i} differ"
            );
        }
    }
}
