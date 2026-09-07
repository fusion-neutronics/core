//! The branching converter says how each excited production level was matched
//! to an isomer, so a rebuild can be checked for levels that changed route.

use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

/// In115's evaluation gives isomer production for (n,n'), (n,2n) and (n,gamma).
/// With decay data for In116's two isomers only, the first two products have no
/// isomer table and the capture level is matched by its level index, since
/// In116_m1 decays by beta- alone and so has no transition energy to match.
#[test]
fn routes_are_counted_and_nothing_here_is_flagged() {
    let neutron = vec![material(fixture!("n-049_In-115_trimmed.endf.xz"))];
    let decay = vec![
        material(fixture!("dec-049_In_116m1.endf.xz")),
        material(fixture!("dec-049_In_116m2.endf.xz")),
    ];
    let (rows, stats) = yani_convert::branching::extract_branching(
        &neutron,
        &decay,
        endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
        yani_convert::branching::DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts");
    assert!(!rows.is_empty());
    assert_eq!(stats.metastable_targets, vec!["In116_m1".to_string()]);
    assert_eq!(stats.level_routes.get("level_index"), Some(&1));
    assert_eq!(stats.level_routes.get("no_isomers"), Some(&2));
    assert_eq!(stats.level_routes.values().sum::<usize>(), 3);
    assert!(
        stats.flagged_levels.is_empty(),
        "{:?}",
        stats.flagged_levels
    );
}
