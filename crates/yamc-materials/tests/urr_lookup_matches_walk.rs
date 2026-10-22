//! The per-step URR queries read a [`yamc_materials::material::UrrLookup`]
//! resolved once per cross-section build, instead of walking `nuclide_data`
//! and comparing temperature labels on every step. That walk is still what a
//! material without the lookup runs, so clearing the lookup on a clone gives
//! the reference: every query must answer exactly what it does, to the bit,
//! and draw exactly the same random numbers.
//!
//! The material mixes three nuclides with probability tables, whose ranges
//! overlap and leave gaps (Fe58 from 350 keV, W184 and Mo92 in the keV range),
//! with Be9, which has none. The energies probed are every range's two end
//! points, which a table does NOT cover, points just inside and outside them,
//! their midpoints, and energies below and above every range.

use std::collections::HashMap;

use yamc_materials::Material;

const NUCLIDES: [&str; 4] = ["Fe58", "W184", "Mo92", "Be9"];

fn build() -> Option<Material> {
    let paths: HashMap<String, String> = NUCLIDES
        .iter()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.to_string(), p)))
        .collect::<Option<_>>()?;
    let composition: HashMap<String, f64> = NUCLIDES
        .iter()
        .zip([0.4, 0.2, 0.1, 0.3])
        .map(|(n, f)| (n.to_string(), f))
        .collect();
    let mut m = Material::new(composition, "atom", "g/cm3", Some(7.0)).expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(&paths, None).expect("nuclear data");
    m.calculate_macroscopic_xs(&vec![1, 2, 18, 101, 102], true);
    assert!(
        m.urr_lookup.is_some(),
        "calculate_macroscopic_xs builds the lookup"
    );
    Some(m)
}

/// The same material answering every query by walking its nuclides.
fn walking(material: &Material) -> Material {
    let mut walk = material.clone();
    walk.urr_lookup = None;
    walk
}

/// Energies around every probability-table range in the material.
fn probes(material: &Material) -> Vec<f64> {
    let mut energies = vec![1e-5, 1.0, 100.0, 5e6, 2e7];
    for nuclide in material.nuclide_data.values() {
        let temp_idx = nuclide.get_temp_idx("294").expect("loaded at 294 K");
        let Some(Some(urr)) = nuclide.urr_data.get(temp_idx) else {
            continue;
        };
        let (lo, hi) = (urr.energy[0], *urr.energy.last().unwrap());
        for e in [lo, hi] {
            energies.extend([e, e * (1.0 - 1e-12), e * (1.0 + 1e-12), e * 0.9, e * 1.1]);
        }
        energies.push(0.5 * (lo + hi));
        energies.extend(urr.energy.iter().copied());
    }
    energies
}

fn skip() {
    eprintln!("SKIP: Fe58, W184, Mo92 or Be9 is not in the test cache; this test checked nothing");
}

#[test]
fn the_material_has_tables_and_gaps_between_them() {
    let Some(material) = build() else {
        return skip();
    };
    let energies = probes(&material);
    let inside = energies
        .iter()
        .filter(|&&e| material.has_urr_in_range(e))
        .count();
    assert!(
        inside > 10 && inside < energies.len() - 10,
        "{inside} of {} probes are inside a table: the fixture no longer exercises both sides",
        energies.len()
    );
}

#[test]
fn has_urr_in_range_matches_the_walk() {
    let Some(material) = build() else {
        return skip();
    };
    let walk = walking(&material);
    for e in probes(&material) {
        assert_eq!(
            material.has_urr_in_range(e),
            walk.has_urr_in_range(e),
            "has_urr_in_range({e:e})"
        );
    }
}

#[test]
fn compute_urr_macro_xs_matches_the_walk_bit_for_bit() {
    let Some(material) = build() else {
        return skip();
    };
    let walk = walking(&material);
    for e in probes(&material) {
        for i in 0..25 {
            let base = (i as f64 + 0.5) / 25.0;
            let got = material.compute_urr_macro_xs(e, base);
            let want = walk.compute_urr_macro_xs(e, base);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    for (name, a, b) in [
                        ("total", g.total, w.total),
                        ("elastic", g.elastic, w.elastic),
                        ("fission", g.fission, w.fission),
                        ("capture", g.capture, w.capture),
                        ("absorption", g.absorption, w.absorption),
                    ] {
                        assert_eq!(a.to_bits(), b.to_bits(), "{name} at {e:e}, base {base}");
                    }
                }
                (g, w) => panic!(
                    "at {e:e}, base {base}: lookup gave {:?}, walk gave {:?}",
                    g.is_some(),
                    w.is_some()
                ),
            }
        }
    }
}

#[test]
fn flight_and_nuclide_selection_match_the_walk_bit_for_bit() {
    let Some(material) = build() else {
        return skip();
    };
    let walk = walking(&material);
    for e in probes(&material) {
        for i in 0..25u64 {
            // A held band on even draws, a fresh one drawn by the flight on odd.
            let held = (i % 2 == 0).then(|| (i as f64 + 0.5) / 25.0);
            let seed = 0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(i + 1);
            let (mut s_got, mut s_want) = (seed, seed);
            let got = material.smooth_flight(e, held, &mut s_got);
            let want = walk.smooth_flight(e, held, &mut s_want);
            assert_eq!(s_got, s_want, "flight draws at {e:e}");
            let (Some(got), Some(want)) = (got, want) else {
                assert_eq!(got.is_some(), want.is_some(), "flight at {e:e}");
                continue;
            };
            assert_eq!(
                got.distance.to_bits(),
                want.distance.to_bits(),
                "distance at {e:e}"
            );
            assert_eq!(
                got.sigma_t.to_bits(),
                want.sigma_t.to_bits(),
                "sigma_t at {e:e}"
            );
            assert_eq!(
                (got.i_grid, got.f.to_bits()),
                (want.i_grid, want.f.to_bits())
            );
            assert_eq!(
                got.urr_random.map(f64::to_bits),
                want.urr_random.map(f64::to_bits),
                "band at {e:e}"
            );

            let picked =
                material.select_nuclide_smooth(e, got.i_grid, got.f, got.urr_random, &mut s_got);
            let expected =
                walk.select_nuclide_smooth(e, want.i_grid, want.f, want.urr_random, &mut s_want);
            assert_eq!(s_got, s_want, "selection draws at {e:e}");
            assert_eq!(
                picked.map(|(name, _, id)| (name.to_string(), id)),
                expected.map(|(name, _, id)| (name.to_string(), id)),
                "struck nuclide at {e:e}"
            );
        }
    }
}
