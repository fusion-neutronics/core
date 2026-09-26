//! Isomeric branching uncertainty from MF=40, end to end.
//!
//! Two cases pin what the source has to keep. Pb208 (n,2n) splits into Pb207
//! and Pb207_m1 by two flat partials, 1.5 b and 0.5 b, so the isomer takes
//! f = 0.25 of it; a 10% MF=40 block on the isomer's partial alone moves the
//! isomer's share by (1 - f) x 10% = 7.5% and leaves the reaction's total
//! alone, which is the transport total's. Co58 (n,n') has no transport total:
//! its rate is the Co58_m1 partial itself, so a 10% block on it moves the
//! isomer's production by the full 10%.
//!
//! The branching curves and the blocks are made up; the cross sections, the
//! chain and the converter that writes `branching_covariance.arrow` are the
//! real ones. Self-skips when the Pb208 or Co58 fixture is missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use endf::mf::covariance::{Mf33Subsection, NiSubsection};
use yamc_materials::Material;
use yani::{BranchCurve, BranchQuantity, BranchTable, ChainNuclide, ChainReaction};
use yani_convert::branching::{write_branching_covariance, BranchingCovarianceRow};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{
    transmute_material, transport_replicas, MultigroupSpectrum, TransmuteStep, TransportTallied,
};

const HOUR: f64 = 3600.0;

/// The committed chain, with the two isomeric edges the overlay would graft.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let edge = |kind: &str, target: &str, branching: f64| ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching,
        q_value: None,
    };
    let pb208 = chain.get_mut("Pb208").expect("Pb208 in the chain");
    pb208.reactions.push(edge("(n,2n)", "Pb207_m1", 0.0));
    pb208.reactions.push(edge("(n,np)", "Tl207_m1", 0.0));
    let co58 = chain.get_mut("Co58").expect("Co58 in the chain");
    co58.reactions.push(edge("(n,n')", "Co58_m1", 1.0));
    Arc::new(chain)
}

fn material(name: &str, density: f64) -> Option<Material> {
    let path = yamc_test_cache::nuclide(name)?;
    let mut m = Material::new(
        HashMap::from([(name.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.nuclides.insert(name.to_string(), density);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([(name.to_string(), path)]), None)
        .expect("read nuclear data");
    Some(m)
}

fn curve(target: &str, quantity: BranchQuantity, from: f64, value: f64) -> BranchCurve {
    BranchCurve {
        target: target.to_string(),
        quantity,
        energy: vec![from, 2.0e7],
        values: vec![value, value],
    }
}

/// One MF=40 self block with a flat relative `variance` over `[from, 20 MeV]`.
fn block(
    parent: &str,
    kind: &str,
    target: &str,
    mt: i32,
    from: f64,
    variance: f64,
) -> BranchingCovarianceRow {
    BranchingCovarianceRow {
        nuclide: parent.to_string(),
        reaction: kind.to_string(),
        target: target.to_string(),
        lfs: 1,
        target1: Some(target.to_string()),
        energy: None,
        values: None,
        mt,
        subsection_idx: 0,
        subsection: Mf33Subsection {
            xmf1: 10.0,
            xlfs1: 1.0,
            mat1: 0,
            mt1: mt as i64,
            nc: 0,
            ni: 1,
            nc_subsections: vec![],
            ni_subsections: vec![NiSubsection {
                lb: 5,
                ls: 1,
                nt: 3,
                ne: 2,
                ek: vec![from, 2.0e7],
                fkk: vec![variance],
                ..Default::default()
            }],
        },
    }
}

/// A branching directory holding just the covariance, as the converter writes it.
fn covariance(rows: &[BranchingCovarianceRow]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("scratch dir");
    assert!(write_branching_covariance(rows, dir.path()).expect("writes"));
    dir
}

/// Pb208 (n,2n) at f = 0.25, optionally with two lists MF=9 gives: (n,np)
/// split between Tl207 and Tl207_m1, and (n,gamma) to Pb209 alone, which is
/// no split at all.
fn lead_overlay(covariance: Option<&Path>, with_yields: bool) -> BranchTable {
    let mut branch = BranchTable::new();
    let kinds = branch.entry("Pb208".to_string()).or_default();
    kinds.insert(
        "(n,2n)".to_string(),
        vec![
            curve("Pb207", BranchQuantity::CrossSection, 7.4e6, 1.5),
            curve("Pb207_m1", BranchQuantity::CrossSection, 7.4e6, 0.5),
        ],
    );
    if with_yields {
        kinds.insert(
            "(n,np)".to_string(),
            vec![
                curve("Tl207", BranchQuantity::Yield, 1.0e-5, 0.9),
                curve("Tl207_m1", BranchQuantity::Yield, 1.0e-5, 0.1),
            ],
        );
        kinds.insert(
            "(n,gamma)".to_string(),
            vec![curve("Pb209", BranchQuantity::Yield, 1.0e-5, 1.0)],
        );
    }
    if let Some(path) = covariance {
        branch.set_covariance_path(path.join("branching_covariance.arrow"));
    }
    branch
}

fn lead_covariance() -> tempfile::TempDir {
    covariance(&[block("Pb208", "(n,2n)", "Pb207_m1", 16, 7.4e6, 0.01)])
}

/// Co58 (n,n') Co58_m1 by one 0.3 b partial, with a 10% block on it.
fn cobalt_overlay(covariance: &Path) -> BranchTable {
    let mut branch = BranchTable::new();
    branch.entry("Co58".to_string()).or_default().insert(
        "(n,n')".to_string(),
        vec![curve("Co58_m1", BranchQuantity::CrossSection, 1.0e5, 0.3)],
    );
    branch.set_covariance_path(covariance.join("branching_covariance.arrow"));
    branch
}

fn cobalt_covariance() -> tempfile::TempDir {
    covariance(&[block("Co58", "(n,n')", "Co58_m1", 4, 1.0e5, 0.01)])
}

fn fourteen_mev() -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![1.3e7, 1.5e7],
        masses: vec![1.0],
        flux_error: None,
    }
}

fn one_hour() -> Vec<TransmuteStep> {
    vec![TransmuteStep {
        dt: HOUR,
        irradiation: Some((0, 1.0e14)),
    }]
}

fn request(sources: Vec<Source>, attribution: bool) -> DataUncertainty {
    DataUncertainty {
        seed: 5,
        samples: Some(1024),
        sources,
        attribution,
    }
}

fn run(
    mut m: Material,
    branch: &BranchTable,
    uncertainty: Option<&DataUncertainty>,
) -> Result<yani_transmute::TransmutationResults, Box<dyn std::error::Error>> {
    transmute_material(
        &mut m,
        &[fourteen_mev()],
        &one_hour(),
        chain(),
        branch,
        Default::default(),
        uncertainty,
    )
}

fn relative_sigma(results: &yani_transmute::TransmutationResults, nuclide: &str) -> f64 {
    let nominal = results.get_material(0, 1).unwrap().nuclides[nuclide];
    results.uncertainty[&0].std_dev_at(0)[nuclide] / nominal
}

#[test]
fn a_perturbed_partial_moves_the_split_by_one_minus_f() {
    let Some(lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let dir = lead_covariance();
    let branch = lead_overlay(Some(dir.path()), false);
    let results = run(
        lead,
        &branch,
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("transmute");
    let info = &results.uncertainty_info[&0];
    assert!(
        info.isomeric_channels_perturbed.contains("Pb208 (n,2n)"),
        "{info:?}"
    );
    assert_eq!(info.isomeric_partials_sampled, 1024);
    assert!(
        (info.isomeric_rate_fraction_covered
            [&("Pb208".to_string(), "(n,2n) Pb207_m1".to_string())]
            - 1.0)
            .abs()
            < 1e-12
    );

    // Pb207_m1 lives 0.8 s, so it is saturated and its density is its share.
    let rel = relative_sigma(&results, "Pb207_m1");
    assert!(
        (rel / 0.075 - 1.0).abs() < 0.1,
        "relative sigma {rel:.4} against (1 - f) x 10% = 0.075"
    );

    // The (n,2n) total is untouched in every replica: Pb208 burns the same,
    // and ground plus isomer are made the same, whatever the split.
    let ensemble = &results.uncertainty[&0];
    let nominal = &results.get_material(0, 1).unwrap().nuclides;
    let made = |n: &HashMap<String, f64>| n["Pb207"] + n["Pb207_m1"];
    for replica in ensemble.inventories_at(0) {
        assert!((replica["Pb208"] / nominal["Pb208"] - 1.0).abs() < 1e-12);
        assert!(
            (made(replica) / made(nominal) - 1.0).abs() < 1e-10,
            "the total moved: {} against {}",
            made(replica),
            made(nominal)
        );
    }
}

#[test]
fn the_nnprime_rate_moves_with_its_partial() {
    let Some(cobalt) = material("Co58", 1.0e-3) else {
        eprintln!("skipping -- Co58 fixture absent");
        return;
    };
    let dir = cobalt_covariance();
    let results = run(
        cobalt,
        &cobalt_overlay(dir.path()),
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("transmute");
    assert!(results.uncertainty_info[&0]
        .isomeric_channels_perturbed
        .contains("Co58 (n,n')"));
    // One hour of a 9.1 h isomer: its density is linear in the rate, and the
    // rate is the partial.
    let rel = relative_sigma(&results, "Co58_m1");
    assert!(
        (rel / 0.10 - 1.0).abs() < 0.1,
        "relative sigma {rel:.4} against the partial's 10%"
    );
}

#[test]
fn the_covariance_file_is_never_opened_unless_asked() {
    let Some(lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let dir = tempfile::tempdir().expect("scratch dir");
    let garbage = dir.path().join("branching_covariance.arrow");
    std::fs::write(&garbage, b"this is not an Arrow file").unwrap();
    let branch = lead_overlay(Some(dir.path()), false);
    assert_eq!(branch.covariance_path(), Some(garbage.as_path()));

    run(lead.clone(), &branch, None).expect("no uncertainty, nothing opened");
    run(
        lead.clone(),
        &branch,
        Some(&request(vec![Source::CrossSections], false)),
    )
    .expect("cross sections alone never open it");
    let err = run(
        lead,
        &branch,
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect_err("asked for, the garbage is an error")
    .to_string();
    assert!(
        err.contains(&garbage.display().to_string()),
        "the error names the file: {err}"
    );
    assert!(err.contains("isomeric_branching"), "{err}");
}

#[test]
fn the_means_are_bit_identical_with_and_without_the_source() {
    let Some(lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let dir = lead_covariance();
    let branch = lead_overlay(Some(dir.path()), true);
    let without = run(lead.clone(), &branch, None).expect("nominal");
    let with = run(
        lead,
        &branch,
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("with the source");
    assert!(!with.uncertainty[&0].std_dev_at(0).is_empty());
    for step in 0..=1 {
        let (a, b) = (
            &without.get_material(0, step).unwrap().nuclides,
            &with.get_material(0, step).unwrap().nuclides,
        );
        assert_eq!(a.len(), b.len());
        for (name, n) in a {
            assert_eq!(n.to_bits(), b[name].to_bits(), "{name} at step {step}");
        }
    }
}

#[test]
fn the_report_names_what_had_no_covariance() {
    let Some(lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let dir = lead_covariance();
    let results = run(
        lead.clone(),
        &lead_overlay(Some(dir.path()), true),
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("transmute");
    let info = &results.uncertainty_info[&0];
    // MF=9 yields have no covariance format. A list naming one state has no
    // split to be without one.
    assert!(
        info.no_isomeric_branching_uncertainty
            .contains("Pb208 (n,np)"),
        "{:?}",
        info.no_isomeric_branching_uncertainty
    );
    assert!(!info
        .no_isomeric_branching_uncertainty
        .contains("Pb208 (n,gamma)"));
    // The ground's partial has no block, and moves only as the isomer's does.
    assert!(info
        .isomeric_partials_without_covariance
        .contains("Pb208 (n,2n) Pb207"));
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "isomeric-branching x cross-section correlation (none published)"));
    assert!(!info.not_perturbed.iter().any(|s| s.contains("MF=9/MF=10")));
    assert!(info.has_gaps());

    // A branching library without the file perturbs nothing and says so for
    // every channel, rather than reading as an exact split.
    let bare = run(
        lead.clone(),
        &lead_overlay(None, true),
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("transmute");
    let info = &bare.uncertainty_info[&0];
    assert!(info.isomeric_channels_perturbed.is_empty());
    for channel in ["Pb208 (n,2n)", "Pb208 (n,np)"] {
        assert!(
            info.no_isomeric_branching_uncertainty.contains(channel),
            "{channel}: {:?}",
            info.no_isomeric_branching_uncertainty
        );
    }
    // Nothing was sampled, so there is no sampled split whose correlation
    // with MF=33 could go unstated.
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "isomeric branching (MF=9/MF=10)"));
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.contains("none published")));

    // And with the source off, the split is listed as held at nominal.
    let off = run(
        lead,
        &lead_overlay(Some(dir.path()), true),
        Some(&request(vec![Source::CrossSections], false)),
    )
    .expect("transmute");
    assert!(off.uncertainty_info[&0]
        .not_perturbed
        .iter()
        .any(|s| s == "isomeric branching (MF=9/MF=10)"));
}

/// With no overlay the chain's own splits are all there is, and nothing
/// samples them: each one the material drives is named as held at nominal,
/// and the source reads as having perturbed nothing, not as a sampled split.
#[test]
fn without_an_overlay_the_chains_own_splits_are_named_as_held() {
    let Some(mut lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let mut chain = (*chain()).clone();
    for r in &mut chain.get_mut("Pb208").unwrap().reactions {
        if r.kind == "(n,2n)" {
            r.branching = if r.target.as_deref() == Some("Pb207_m1") {
                0.25
            } else {
                0.75
            };
        }
    }
    let results = transmute_material(
        &mut lead,
        &[fourteen_mev()],
        &one_hour(),
        Arc::new(chain),
        &BranchTable::new(),
        Default::default(),
        Some(&request(vec![Source::IsomericBranching], false)),
    )
    .expect("transmute");
    let info = &results.uncertainty_info[&0];
    assert!(info.isomeric_channels_perturbed.is_empty());
    assert!(
        info.no_isomeric_branching_uncertainty
            .contains("Pb208 (n,2n)"),
        "{:?}",
        info.no_isomeric_branching_uncertainty
    );
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "isomeric branching (MF=9/MF=10)"));
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.contains("none published")));
    assert!(info.has_gaps());
}

#[test]
fn the_first_order_contributor_matches_the_source_alone() {
    let Some(lead) = material("Pb208", 3.3e-2) else {
        eprintln!("skipping -- Pb208 fixture absent");
        return;
    };
    let dir = lead_covariance();
    let results = run(
        lead,
        &lead_overlay(Some(dir.path()), false),
        Some(&request(vec![Source::IsomericBranching], true)),
    )
    .expect("transmute");
    let b = results
        .uncertainty_breakdown(0, "Pb207_m1", 1)
        .expect("asked for");
    let exact = b.by_source["isomeric_branching"];
    let first_order = |reaction: Option<&str>| {
        b.contributors
            .iter()
            .find(|(s, n, r, _)| {
                s == "isomeric_branching" && n == "Pb208" && r.as_deref() == reaction
            })
            .unwrap_or_else(|| panic!("no contributor for {reaction:?}: {:?}", b.contributors))
            .3
    };
    let whole = first_order(None);
    assert!(
        (whole / exact - 1.0).abs() < 0.12,
        "first order {whole:.3e} against exact {exact:.3e}"
    );
    // One partial carries it all.
    assert!((first_order(Some("(n,2n) Pb207_m1")) / whole - 1.0).abs() < 1e-9);
}

/// A transport run's tallied Co58 (n,n') partial, per source particle, with
/// the tally's flux shape for the MF=40 fold.
fn tallied(partial: f64) -> TransportTallied {
    TransportTallied {
        rates: HashMap::from([(
            "Co58".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 1.0e-27)]),
        )]),
        partials: HashMap::from([(
            "Co58".to_string(),
            HashMap::from([("(n,n')".to_string(), vec![("Co58_m1".to_string(), partial)])]),
        )]),
        fy_weights: HashMap::new(),
        spectrum: fourteen_mev(),
        statistics: None,
    }
}

#[test]
fn the_transport_path_scales_the_tallied_partials() {
    let Some(cobalt) = material("Co58", 1.0e-3) else {
        eprintln!("skipping -- Co58 fixture absent");
        return;
    };
    let dir = cobalt_covariance();
    let (ensemble, info) = transport_replicas(
        &cobalt,
        &tallied(3.0e-26),
        &[HOUR],
        &[1.0e14],
        &chain(),
        &cobalt_overlay(dir.path()),
        Default::default(),
        &request(vec![Source::IsomericBranching], false),
    )
    .expect("transport replicas");
    assert_eq!(info.sources, vec!["isomeric_branching".to_string()]);
    assert!(info.isomeric_channels_perturbed.contains("Co58 (n,n')"));
    assert_eq!(info.isomeric_partials_sampled, 1024);
    let samples = ensemble.samples_at(0, "Co58_m1");
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let rel = ensemble.std_dev_at(0)["Co58_m1"] / mean;
    assert!(
        (rel / 0.10 - 1.0).abs() < 0.1,
        "relative sigma {rel:.4} against the partial's 10%"
    );
}

#[test]
fn the_transport_attribution_matches_the_source_alone() {
    let Some(cobalt) = material("Co58", 1.0e-3) else {
        eprintln!("skipping -- Co58 fixture absent");
        return;
    };
    let dir = cobalt_covariance();
    let (ensemble, _) = transport_replicas(
        &cobalt,
        &tallied(3.0e-26),
        &[HOUR],
        &[1.0e14],
        &chain(),
        &cobalt_overlay(dir.path()),
        Default::default(),
        &request(vec![Source::IsomericBranching], true),
    )
    .expect("transport replicas");
    let attribution = ensemble.attribution.as_ref().expect("asked for");
    let exact = attribution.by_source["isomeric_branching"][0]["Co58_m1"];
    let whole = attribution
        .contributors
        .iter()
        .find(|c| c.source == "isomeric_branching" && c.nuclide == "Co58" && c.reaction.is_none())
        .expect("Co58's MF=40 contributes")
        .variance[0]["Co58_m1"];
    assert!(
        (whole / exact - 1.0).abs() < 0.12,
        "first order {whole:.3e} against exact {exact:.3e}"
    );
}
