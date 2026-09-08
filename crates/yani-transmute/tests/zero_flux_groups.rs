//! A group with no flux in it contributes nothing, so skipping it changes
//! nothing (issue #576, finding 2).
//!
//! The collapse is `sigma_eff = sum_g sigma_g phi_g / sum_g phi_g` and the rate
//! it produces is `sigma_eff * 1e-24 * sum_g phi_g`, so the total flux cancels
//! and the rate is `1e-24 * sum_g sigma_g phi_g`. A group with `phi_g = 0`
//! therefore adds exactly `+0.0`, and the whole structure outside the groups
//! that carry flux is dead weight -- which on a monoenergetic 14 MeV source in
//! CCFE-709 is 708 groups of 709.
//!
//! Rather than assert that reasoning, this runs the same physical spectrum two
//! ways: padded out with empty groups, and as the narrow structure of the groups
//! that actually carry flux. The rates must agree to the bit.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;

use yamc_materials::Material;
use yani_transmute::compute_multigroup_reaction_rates;

fn chain() -> HashMap<String, yani::ChainNuclide> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    yani::parse_chain_arrow(&path).expect("parse chain")
}

fn iron(data: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), data.to_string())]),
        None,
    )
    .expect("read Fe56");
    m
}

fn bits(rates: &yani::ReactionRates) -> Vec<(String, String, u64)> {
    let mut out: Vec<(String, String, u64)> = rates
        .iter()
        .flat_map(|(n, per_kind)| {
            per_kind
                .iter()
                .map(move |(k, v)| (n.clone(), k.clone(), v.to_bits()))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn padding_a_spectrum_with_empty_groups_changes_no_bits() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let material = iron(&data);
    let chain = chain();

    // CCFE-709, with flux in three groups spread across the structure and
    // nothing anywhere else -- the shape a monoenergetic or few-line source has.
    // The highest is the last group wholly under 20 MeV, where the ENDF/B-VIII.1
    // evaluation ends: a group above it carries no cross section and so no rate,
    // which would leave the threshold channels with nothing to compare.
    let boundaries = yamc_nuclide::group_structures::get_group_structure("CCFE-709")
        .expect("CCFE-709")
        .to_vec();
    let n = boundaries.len() - 1;
    let top = boundaries.iter().rposition(|&e| e <= 2.0e7).unwrap() - 1;
    let carrying = [12usize, 400, top];
    let mut padded = vec![0.0; n];
    for (i, &g) in carrying.iter().enumerate() {
        padded[g] = (i + 1) as f64;
    }

    let (wide, wide_fy) =
        compute_multigroup_reaction_rates(&material, &chain, &padded, &boundaries, 1.0);

    // The same physical spectrum, as the structure of the groups that carry it.
    // Each group is integrated on its own, so the surviving groups' averages are
    // untouched by the removal of their empty neighbours.
    let mut narrow_boundaries = Vec::new();
    let mut narrow_flux = Vec::new();
    for (i, &g) in carrying.iter().enumerate() {
        narrow_boundaries.push(boundaries[g]);
        narrow_boundaries.push(boundaries[g + 1]);
        narrow_flux.push((i + 1) as f64);
    }
    // Three disjoint groups need four boundaries plus the gaps between them; the
    // gaps carry no flux, which is exactly the case under test.
    let mut merged_boundaries = vec![narrow_boundaries[0]];
    let mut merged_flux = Vec::new();
    for (i, pair) in narrow_boundaries.chunks(2).enumerate() {
        if *merged_boundaries.last().expect("non-empty") < pair[0] {
            merged_boundaries.push(pair[0]);
            merged_flux.push(0.0);
        }
        merged_boundaries.push(pair[1]);
        merged_flux.push((i + 1) as f64);
    }

    let (narrow, narrow_fy) =
        compute_multigroup_reaction_rates(&material, &chain, &merged_flux, &merged_boundaries, 1.0);

    assert!(!wide.is_empty(), "the padded spectrum must produce rates");
    assert_eq!(
        bits(&wide),
        bits(&narrow),
        "the 709-group spectrum and its 5-group equivalent disagree"
    );
    assert_eq!(wide_fy.len(), narrow_fy.len());

    // Against an independent reference, because the comparison above is between
    // two runs of the same skip and would agree even if the skip dropped a
    // group that carries flux. `per_group_reaction_rates` walks every group
    // unconditionally and its terms are `sigma_g * phi_g * 1e-24`, which is the
    // rate the collapse produces once the total flux cancels out of it.
    let per_group = yani_transmute::multigroup::per_group_reaction_rates(
        &material,
        &chain,
        &padded,
        &boundaries,
        None,
    );
    let mut checked = 0;
    for (nuclide, per_kind) in &wide {
        for (kind, &rate) in per_kind {
            let terms = &per_group[nuclide][kind];
            assert_eq!(terms.len(), n, "one term per group");
            let reference: f64 = terms.iter().sum();
            assert!(
                (rate - reference).abs() <= 1.0e-12 * reference.abs(),
                "{nuclide} {kind}: collapse gives {rate}, the per-group sum gives {reference}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 3,
        "expected several reactions to check, got {checked}"
    );
}
