//! Inside a nuclide's unresolved range a `nuclides=` bin scores that
//! nuclide's cross section at the band the flight sampled, so the nuclide bins
//! add up to the material bin for every MT the bands move, as they do outside
//! the tables. Fe58 and W184 have tables, Be9 has none.

use std::collections::HashMap;

use yamc_materials::Material;

const NUCLIDES: [&str; 3] = ["Fe58", "W184", "Be9"];
const MTS: [i32; 5] = [1, 2, 18, 27, 102];

fn build() -> Option<Material> {
    let paths: HashMap<String, String> = NUCLIDES
        .iter()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.to_string(), p)))
        .collect::<Option<_>>()?;
    let composition: HashMap<String, f64> = NUCLIDES
        .iter()
        .zip([0.5, 0.3, 0.2])
        .map(|(n, f)| (n.to_string(), f))
        .collect();
    let mut m = Material::new(composition, "atom", "g/cm3", Some(7.0)).expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(&paths, None).expect("nuclear data");
    m.calculate_macroscopic_xs(&MTS.to_vec(), true);
    m.populate_per_nuclide_xs();
    Some(m)
}

#[test]
fn nuclide_bins_add_up_to_the_material_inside_the_tables() {
    let Some(material) = build() else {
        eprintln!("SKIP: Fe58, W184 or Be9 is not in the test cache; this test checked nothing");
        return;
    };
    let mut inside = 0;
    for nuclide in ["Fe58", "W184"] {
        let data = &material.nuclide_data[nuclide];
        let urr = data.urr_data[data.get_temp_idx("294").unwrap()]
            .as_ref()
            .expect("tables at 294 K");
        for w in urr.energy.windows(2) {
            let e = 0.5 * (w[0] + w[1]);
            for i in 0..8 {
                let Some(u) = material.compute_urr_macro_xs(e, (i as f64 + 0.5) / 8.0) else {
                    continue;
                };
                inside += 1;
                for mt in MTS {
                    let total = material.macro_xs_by_mt(mt, e, Some(&u));
                    let sum: f64 = NUCLIDES
                        .iter()
                        .map(|n| material.macro_xs_by_mt_for_nuclide(n, mt, e, Some(&u)))
                        .sum();
                    assert!(
                        (sum - total).abs() <= 1e-6 * total.abs().max(1e-12),
                        "MT {mt} at {e:e} eV: nuclide bins {sum} vs material {total}"
                    );
                }
            }
        }
    }
    assert!(inside > 0, "no energy probed was inside a table");
    // A nuclide with no table, or an MT the bands do not move, is the smooth
    // per-nuclide value whatever the band.
    let e = 1.0e5;
    let u = material.compute_urr_macro_xs(e, 0.5);
    for mt in [102, 301] {
        assert_eq!(
            material.macro_xs_by_mt_for_nuclide("Be9", mt, e, u.as_ref()),
            material.lookup_xs_by_mt_for_nuclide("Be9", mt, e)
        );
    }
}
