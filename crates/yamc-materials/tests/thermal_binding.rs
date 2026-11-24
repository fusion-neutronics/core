//! Which nuclides a material's thermal scattering tables apply to, and what a
//! material accepts as `s_alpha_beta`.

use std::collections::HashMap;
use yamc_materials::material::{bind_thermal, Material, MaterialSerde};

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_table_binds_the_nuclides_it_lists_that_the_material_has() {
    let water = names(&["H1", "H2", "O16", "O17"]);
    let h = names(&["H1"]);
    let bound = bind_thermal(&[("c_H_in_H2O", &h)], &water).unwrap();
    assert_eq!(bound, vec![names(&["H1"])]);
    let graphite = names(&["C12", "C13"]);
    let bound = bind_thermal(&[("c_Graphite", &graphite)], &names(&["C12", "C13"])).unwrap();
    assert_eq!(bound, vec![names(&["C12", "C13"])]);
}

#[test]
fn a_table_binding_nothing_and_a_nuclide_claimed_twice_are_refused() {
    let water = names(&["H1", "O16"]);
    let carbon = names(&["C12", "C13"]);
    let e = bind_thermal(&[("c_Graphite", &carbon)], &water).unwrap_err();
    assert!(
        e.contains("c_Graphite") && e.contains("none of them"),
        "{e}"
    );
    let h = names(&["H1"]);
    let e = bind_thermal(&[("c_H_in_H2O", &h), ("c_H_in_CH2", &h)], &water).unwrap_err();
    assert!(e.contains("H1") && e.contains("both"), "{e}");
}

#[test]
fn s_alpha_beta_takes_a_name_once_with_a_fraction_in_zero_to_one() {
    let mut m = Material::new(
        HashMap::from([("H1".to_string(), 2.0), ("O16".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    m.set_s_alpha_beta(vec![("c_H_in_H2O".into(), 1.0)])
        .unwrap();
    assert_eq!(m.s_alpha_beta(), [("c_H_in_H2O".to_string(), 1.0)]);
    assert!(m
        .set_s_alpha_beta(vec![("c_H_in_ZrH".into(), 0.0)])
        .is_err());
    assert!(m
        .set_s_alpha_beta(vec![("c_H_in_ZrH".into(), 1.5)])
        .is_err());
    assert!(m
        .set_s_alpha_beta(vec![("a".into(), 1.0), ("a".into(), 0.5)])
        .is_err());
    assert!(m.set_s_alpha_beta(vec![(" ".into(), 1.0)]).is_err());
    // It survives the serialized form.
    m.set_s_alpha_beta(vec![("c_H_in_ZrH".into(), 0.6)])
        .unwrap();
    let back = Material::try_from(MaterialSerde::from(m)).unwrap();
    assert_eq!(back.s_alpha_beta(), [("c_H_in_ZrH".to_string(), 0.6)]);
}
