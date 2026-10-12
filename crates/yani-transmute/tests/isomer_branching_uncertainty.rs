//! The `isomer_branching` source: the MF=40 covariance of each final state's
//! production cross section, drawn and re-folded into the isomeric split, end
//! to end through `transmute_material`.
//!
//! The synthetic cases always run. Their `branching_covariance.arrow` is
//! written by the converter's own writer and read back as the chain loader
//! holds it, so the blocks drawn are the file's. In115 has a flat 1 b `(n,2n)`
//! over one group from 1 to 20 MeV, split 0.6 b to In114 and 0.4 b to
//! In114_m1, and an `(n,n')` partial to In115_m1; every product is stable, so
//! over a short step each density is its production rate times the step.
//!
//! The ENDF/B-VIII.1 case reads the In115 fixture and the chain fixture, which
//! `scripts/fetch_test_fixtures.py` fetches, and self-skips on a machine that
//! has not run it. Its MF=40 rows are the In115 and Mo92 rows of
//! ENDF/B-VIII.1's `branching_covariance.arrow`, committed beside the
//! branching rows in `tests/fixtures/endf-b8.1-branching-In115-Mo92`.

use std::collections::HashMap;
use std::sync::Arc;

use endf::mf::covariance::{Mf33Subsection, NiSubsection};
use yamc_materials::Material;
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yani::{BranchCurve, BranchQuantity, BranchState, BranchTable, ChainNuclide, ChainReaction};
use yani_convert::branching::{write_branching_covariance, BranchingCovarianceRow};
use yani_transmute::uncertainty::{DataUncertainty, Info, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const LOW: f64 = 1.0e6;
const HIGH: f64 = 2.0e7;
/// The two MF=40 intervals of the synthetic blocks, cut inside the group.
const CUT: f64 = 1.0e7;

fn reaction(mt: i32, energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
    Reaction {
        cross_section: cross_section.into(),
        threshold_idx: 0,
        energy: energy.into(),
        mt_number: mt,
        q_value: 0.0,
        products: vec![].into(),
        scatter_in_cm: false,
        redundant: false,
    }
}

/// An LB=5 relative covariance on `grid`, upper triangle row by row.
fn lb5(grid: Vec<f64>, upper: Vec<f64>) -> NiSubsection {
    NiSubsection {
        lb: 5,
        ls: 1,
        ne: grid.len() as i64,
        ek: grid,
        fkk: upper,
        ..Default::default()
    }
}

/// In115 with a flat 1 b `(n,2n)` and a flat 2 b inelastic over the group,
/// and, when `mf33` is given, that relative sigma on its `(n,2n)` as one
/// MF=33 block over the whole range.
fn indium(mf33: Option<f64>) -> Material {
    let temperature = "294".to_string();
    let reactions: HashMap<i32, Arc<Reaction>> = HashMap::from([
        (4, Arc::new(reaction(4, vec![LOW, HIGH], vec![2.0, 2.0]))),
        (16, Arc::new(reaction(16, vec![LOW, HIGH], vec![1.0, 1.0]))),
    ]);
    let covariance = mf33.map(|s| {
        Arc::new(vec![CovarianceBlock {
            mt: 16,
            subsection_idx: 0,
            block_idx: 0,
            mat1: 0,
            mt1: 16,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat: 0,
            data: CovarianceData::Ni(lb5(vec![1.0e-5, HIGH], vec![s * s])),
        }])
    });
    let nuclide = Nuclide {
        name: Some("In115".to_string()),
        element: None,
        atomic_symbol: Some("In".to_string()),
        atomic_number: Some(49),
        neutron_number: Some(66),
        mass_number: Some(115),
        atomic_weight_ratio: Some(113.9),
        library: None,
        energy: None,
        reactions: vec![reactions],
        fissionable: false,
        available_temperatures: vec![temperature.clone()],
        loaded_temperatures: vec![temperature.clone()],
        data_path: None,
        data_source: None,
        fission_nu: None,
        fast_xs: vec![],
        urr_data: vec![],
        urr_present: false,
        fission_photon_release: None,
        covariance,
        angular_covariance: None,
        nubar_covariance: None,
        spectrum_covariance: None,
        resonance_parameters: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: Default::default(),
    };
    let mut m = Material::new(
        HashMap::from([("In115".to_string(), 3.8e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("indium");
    m.set_temperature(&temperature);
    m.nuclide_data
        .insert("In115".to_string(), Arc::new(nuclide));
    m.volume = Some(1.0);
    m
}

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: vec![],
        fission_yields: None,
        sources: Vec::new(),
    }
}

/// In115's `(n,2n)` to the ground at 1.0 with the isomer grafted at 0.0, and
/// its `(n,n')` grafted to its isomer.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let edge = |kind: &str, target: &str, branching: f64| ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching,
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
        q_value: Some(0.0),
    };
    let mut map = HashMap::new();
    map.insert(
        "In115".to_string(),
        nuclide(
            "In115",
            vec![
                edge("(n,n')", "In115_m1", 0.0),
                edge("(n,2n)", "In114", 1.0),
                edge("(n,2n)", "In114_m1", 0.0),
            ],
        ),
    );
    for name in ["In115_m1", "In114", "In114_m1"] {
        map.insert(name.to_string(), nuclide(name, vec![]));
    }
    Arc::new(map)
}

fn facts(mt: i32, lfs: i32, complete: bool) -> Arc<[BranchState]> {
    Arc::from(vec![BranchState {
        mt,
        lfs,
        lmf: Some(10),
        list_complete: complete,
        level_route: if lfs == 0 { "ground" } else { "energy" }.to_string(),
        level_energy: 0.0,
        level_energy_difference: Some(0.0),
        mf3_cross_section: None,
    }])
}

fn flat(target: &str, mt: i32, lfs: i32, value: f64, complete: bool) -> BranchCurve {
    BranchCurve {
        target: target.to_string(),
        quantity: BranchQuantity::CrossSection,
        energy: vec![LOW, HIGH],
        values: vec![value, value],
        states: facts(mt, lfs, complete),
        normalisation: None,
    }
}

/// One MF=40 sub-subsection of In115: state `(mt, lfs)` of `target` with
/// partner `(mt1, xlfs1)`, an LB=5 block over the two intervals either side
/// of [`CUT`].
fn mf40(
    target: &str,
    reaction: &str,
    (mt, lfs): (i32, i64),
    (mt1, xlfs1): (i64, f64),
    upper: Vec<f64>,
) -> BranchingCovarianceRow {
    let grid = if upper.len() == 1 {
        vec![1.0e-5, HIGH]
    } else {
        vec![1.0e-5, CUT, HIGH]
    };
    BranchingCovarianceRow {
        nuclide: "In115".to_string(),
        reaction: Some(reaction.to_string()),
        target: Some(target.to_string()),
        target1: Some(target.to_string()),
        energy: None,
        values: None,
        quantity: Some("cross_section".to_string()),
        mat: 4931,
        mt,
        za: 49115,
        awr: 113.9,
        lis: 0,
        state_idx: lfs as usize,
        qm: 0.0,
        qi: 0.0,
        izap: 49114,
        lfs,
        subsection_idx: 0,
        subsection: Mf33Subsection {
            xmf1: 10.0,
            xlfs1,
            mat1: 0,
            mt1,
            nc: 0,
            ni: 1,
            nc_subsections: vec![],
            ni_subsections: vec![lb5(grid, upper)],
        },
    }
}

/// The branching table: the `(n,2n)` list complete (ground and isomer) or of
/// the isomer alone, the `(n,n')` isomer partial, and `rows` as its
/// `branching_covariance.arrow`, written and read back.
fn branch(complete: bool, rows: &[BranchingCovarianceRow]) -> BranchTable {
    let mut branch = BranchTable::new();
    let kinds = branch.curves_mut().entry("In115".to_string()).or_default();
    let mut n2n = vec![flat("In114_m1", 16, 1, 0.4, complete)];
    if complete {
        n2n.insert(0, flat("In114", 16, 0, 0.6, true));
    }
    kinds.insert("(n,2n)".to_string(), n2n);
    kinds.insert(
        "(n,n')".to_string(),
        vec![flat("In115_m1", 4, 1, 0.5, false)],
    );
    if !rows.is_empty() {
        let dir = tempfile::tempdir().expect("tmp");
        assert!(write_branching_covariance(rows, dir.path()).expect("write"));
        let file =
            std::fs::File::open(dir.path().join("branching_covariance.arrow")).expect("written");
        let reader = arrow_ipc::reader::FileReader::try_new(file, None).expect("arrow");
        let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
        branch.set_covariance(batches);
    }
    branch
}

fn spectrum() -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![LOW, HIGH],
        masses: vec![1.0],
        flux_error: None,
    }
}

fn steps() -> [TransmuteStep; 1] {
    [TransmuteStep {
        dt: 60.0,
        irradiation: Some((0, 1.0e14)),
    }]
}

fn request(sources: Vec<Source>, samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: 11,
        samples: Some(samples),
        sources,
        attribution: false,
        ..Default::default()
    }
}

const PRODUCTS: [&str; 3] = ["In114", "In114_m1", "In115_m1"];

/// Each product's relative sigma, its replicas, and the run's report.
struct Run {
    relative: HashMap<String, f64>,
    replicas: HashMap<String, Vec<f64>>,
    nominal: HashMap<String, f64>,
    info: Info,
}

fn run(material: &mut Material, branch: &BranchTable, request: Option<&DataUncertainty>) -> Run {
    let results = transmute_material(
        material,
        &[spectrum()],
        &steps(),
        chain(),
        branch,
        Default::default(),
        request,
    )
    .expect("transmute");
    let nominal: HashMap<String, f64> =
        results.get_material(0, 1).expect("step 1").nuclides.clone();
    let Some(ensemble) = results.uncertainty.get(&0) else {
        return Run {
            relative: HashMap::new(),
            replicas: HashMap::new(),
            nominal,
            info: Info::default(),
        };
    };
    let sd = ensemble.std_dev_at(0);
    let relative = PRODUCTS
        .iter()
        .map(|n| {
            (
                n.to_string(),
                sd.get(*n).copied().unwrap_or(0.0) / nominal[*n],
            )
        })
        .collect();
    let replicas = PRODUCTS
        .iter()
        .map(|n| (n.to_string(), ensemble.samples_at(0, n)))
        .collect();
    Run {
        relative,
        replicas,
        nominal,
        info: results.uncertainty_info[&0].clone(),
    }
}

/// `(b - a) / (HIGH - LOW)` of the group's flat production lies in `[a, b]`.
fn width(a: f64, b: f64) -> f64 {
    (b - a) / (HIGH - LOW)
}

/// An isomer listed alone moves by its own MF=40, folded over the two
/// intervals the group straddles, and the ground takes the rest: its
/// absolute change is the isomer's, with the sign turned.
#[test]
fn an_isomer_listed_alone_moves_by_its_folded_sigma() {
    let (s1, s2) = (0.05, 0.10);
    let rows = [mf40(
        "In114_m1",
        "(n,2n)",
        (16, 1),
        (16, 1.0),
        vec![s1 * s1, 0.0, s2 * s2],
    )];
    let branch = branch(false, &rows);
    let r = run(
        &mut indium(None),
        &branch,
        Some(&request(vec![Source::IsomerBranching], 1000)),
    );
    let (w1, w2) = (width(LOW, CUT), width(CUT, HIGH));
    let folded = ((w1 * s1).powi(2) + (w2 * s2).powi(2)).sqrt();
    let key = (
        "In115".to_string(),
        "(n,2n)".to_string(),
        "In114_m1".to_string(),
    );
    let stated = r.info.isomer_production_sigma[&key];
    assert!(
        (stated - folded).abs() < 1e-12,
        "the report's {stated} against the fold's {folded}"
    );
    assert!(
        (r.relative["In114_m1"] / folded - 1.0).abs() < 0.08,
        "In114_m1 relative sigma {} against {folded}",
        r.relative["In114_m1"]
    );
    let ground = 0.4 / 0.6 * folded;
    assert!(
        (r.relative["In114"] / ground - 1.0).abs() < 0.08,
        "In114 relative sigma {} against {ground}",
        r.relative["In114"]
    );
    // The reaction's total is MF=33's alone, so the two products sum to the
    // nominal in every replica.
    let total = r.nominal["In114"] + r.nominal["In114_m1"];
    for (g, m) in r.replicas["In114"].iter().zip(&r.replicas["In114_m1"]) {
        assert!(((g + m) / total - 1.0).abs() < 1e-9, "{g} + {m} != {total}");
    }
    // No MF=40 on the (n,n') isomer: it moves only through In115's depletion.
    assert!(
        r.relative["In115_m1"] < 1.0e-6,
        "{}",
        r.relative["In115_m1"]
    );
}

/// A list that names its ground: only the split moves. With the two states
/// independent at 5% each, the isomer's share `f` moves by about
/// `(1 - f) sqrt(s_g^2 + s_m^2)`, and ground and isomer always sum to the
/// reaction.
#[test]
fn a_complete_list_moves_only_its_split() {
    let s = 0.05;
    let rows = [
        mf40("In114", "(n,2n)", (16, 0), (16, 0.0), vec![s * s]),
        mf40("In114_m1", "(n,2n)", (16, 1), (16, 1.0), vec![s * s]),
    ];
    let branch = branch(true, &rows);
    let r = run(
        &mut indium(None),
        &branch,
        Some(&request(vec![Source::IsomerBranching], 1000)),
    );
    let expected = 0.6 * (2.0 * s * s).sqrt();
    assert!(
        (r.relative["In114_m1"] / expected - 1.0).abs() < 0.08,
        "In114_m1 relative sigma {} against {expected}",
        r.relative["In114_m1"]
    );
    let total = r.nominal["In114"] + r.nominal["In114_m1"];
    for (g, m) in r.replicas["In114"].iter().zip(&r.replicas["In114_m1"]) {
        assert!(((g + m) / total - 1.0).abs() < 1e-9, "{g} + {m} != {total}");
        assert!(*g >= 0.0 && *m >= 0.0 && *m <= total * (1.0 + 1e-12));
    }
}

/// With the cross sections alone the isomer listed alone moves with its
/// reaction's MF=33 draw, lent the total's 10%. With MF=40 as well it is made
/// at its own stated production, 3%, and the total still moves by MF=33, so
/// the ground carries the rest of the total's spread.
#[test]
fn mf40_replaces_the_lent_mf33_sigma_of_an_isomer_listed_alone() {
    let (s33, s40) = (0.10, 0.03);
    let rows = [mf40(
        "In114_m1",
        "(n,2n)",
        (16, 1),
        (16, 1.0),
        vec![s40 * s40],
    )];
    let branch = branch(false, &rows);
    let lent = run(
        &mut indium(Some(s33)),
        &branch,
        Some(&request(vec![Source::CrossSections], 1000)),
    );
    assert!(
        (lent.relative["In114_m1"] / s33 - 1.0).abs() < 0.08,
        "lent: {}",
        lent.relative["In114_m1"]
    );
    let both = run(
        &mut indium(Some(s33)),
        &branch,
        Some(&request(
            vec![Source::CrossSections, Source::IsomerBranching],
            1000,
        )),
    );
    assert!(
        (both.relative["In114_m1"] / s40 - 1.0).abs() < 0.08,
        "stated: {}",
        both.relative["In114_m1"]
    );
    // The ground is the total, moved by MF=33, less the isomer, moved by
    // MF=40, the two independent.
    let ground = ((1.0 * s33).powi(2) + (0.4 * s40).powi(2)).sqrt() / 0.6;
    assert!(
        (both.relative["In114"] / ground - 1.0).abs() < 0.08,
        "In114 relative sigma {} against {ground}",
        both.relative["In114"]
    );
}

/// An `(n,n')` isomer's rate is its MF=10 partial, so it is made at the drawn
/// partial whatever MT=4's MF=33 says.
#[test]
fn an_inelastic_isomer_is_made_at_its_drawn_partial() {
    let s = 0.07;
    let rows = [mf40("In115_m1", "(n,n')", (4, 1), (4, 1.0), vec![s * s])];
    let branch = branch(false, &rows);
    let r = run(
        &mut indium(None),
        &branch,
        Some(&request(vec![Source::IsomerBranching], 1000)),
    );
    assert!(
        (r.relative["In115_m1"] / s - 1.0).abs() < 0.08,
        "In115_m1 relative sigma {}",
        r.relative["In115_m1"]
    );
    // Only through In115's own depletion, which its drawn (n,n') rate moves.
    assert!(
        r.relative["In114_m1"] < 1.0e-6,
        "{}",
        r.relative["In114_m1"]
    );
}

/// The nominal does not move with the source on, and a zero covariance
/// re-folds every replica to exactly the nominal.
#[test]
fn a_zero_covariance_reproduces_the_nominal() {
    let rows = [
        mf40("In114", "(n,2n)", (16, 0), (16, 0.0), vec![0.0]),
        mf40("In114_m1", "(n,2n)", (16, 1), (16, 1.0), vec![0.0]),
    ];
    let branch = branch(true, &rows);
    let plain = run(&mut indium(None), &branch, None);
    let r = run(
        &mut indium(None),
        &branch,
        Some(&request(vec![Source::IsomerBranching], 8)),
    );
    assert_eq!(r.nominal, plain.nominal);
    for n in PRODUCTS {
        for replica in &r.replicas[n] {
            assert!(
                (replica - plain.nominal[n]).abs() <= 1e-12 * plain.nominal[n],
                "{n}: replica {replica} against {}",
                plain.nominal[n]
            );
        }
    }
}

/// The report names what was drawn, what was held for want of MF=40, and the
/// blocks it could not use.
#[test]
fn the_report_says_what_was_drawn_and_what_was_held() {
    let s = 0.05;
    let mut other_file = mf40("In114_m1", "(n,2n)", (16, 1), (16, 1.0), vec![s * s]);
    other_file.subsection.xmf1 = 3.0;
    let rows = [
        mf40("In114_m1", "(n,2n)", (16, 1), (16, 1.0), vec![s * s]),
        other_file,
    ];
    let branch = branch(false, &rows);
    let samples = 8;
    let r = run(
        &mut indium(None),
        &branch,
        Some(&request(Source::IMPLEMENTED.to_vec(), samples)),
    );
    let info = &r.info;
    assert!(info.sources.contains(&"isomer_branching".to_string()));
    let n2n = ("In115".to_string(), "(n,2n)".to_string());
    let inelastic = ("In115".to_string(), "(n,n')".to_string());
    assert!(info.isomer_branchings_perturbed.contains(&n2n));
    assert!(info.no_isomer_branching_uncertainty.contains(&inelastic));
    assert_eq!(
        info.isomer_branching_blocks_not_used["In115"]
            ["the partner is not an MF=10 production (XMF1 not 10)"],
        1
    );
    assert_eq!(info.isomer_branchings_sampled, samples);
    assert!(info.has_gaps());
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("isomeric branching (MF=9/MF=10) of a reaction with no")));

    // Switched off, every split is held and said to be.
    let off = run(
        &mut indium(None),
        &branch,
        Some(&request(vec![Source::HalfLife], samples)),
    );
    assert!(off
        .info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("isomeric branching covariance (MF=40)")));
    assert!(off.info.isomer_branchings_perturbed.is_empty());
}

/// ENDF/B-VIII.1's chain and In115 with its own MF=40: under a 14 MeV group
/// the In114_m1 and In115_m1 inventories move by the folded MF=40 sigma the
/// report states, a few percent, where they read zero without the source.
#[test]
fn endf_b8_1_in115_isomers_move_by_their_mf40() {
    let Some(path) = yamc_test_cache::nuclide("In115") else {
        eprintln!("skipping -- In115 fixture absent");
        return;
    };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let chain_fixture = manifest.join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    let dirs = [
        chain_fixture.join("decay"),
        chain_fixture.join("reactions"),
        chain_fixture.join("fission_yields"),
        manifest.join("tests/fixtures/endf-b8.1-branching-In115-Mo92"),
    ];
    if let Some(missing) = dirs.iter().find(|d| !d.is_dir()) {
        eprintln!("skipping -- {} absent", missing.display());
        return;
    }
    let dir = |i: usize| dirs[i].to_string_lossy().into_owned();
    let loaded = yani::load_chain_parts(&dir(0), Some(&dir(1)), Some(&dir(2)), Some(&dir(3)))
        .expect("chain");
    assert!(loaded.branch.covariance().is_some(), "the fixture's MF=40");
    let mut material = Material::new(
        HashMap::from([("In115".to_string(), 3.8e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    material.set_temperature("294");
    material.volume = Some(1.0);
    let scope = yamc_nuclide::load_scope::LoadScope::activation(yani_transmute::activation_mts(
        &loaded.chain,
        &loaded.branch,
    ));
    let data = yamc_nuclide::nuclide::get_or_load_nuclide(
        "In115",
        &HashMap::from([("In115".to_string(), path)]),
        &scope,
    )
    .expect("load");
    material.nuclide_data.insert("In115".to_string(), data);

    let dt = MultigroupSpectrum {
        boundaries: vec![1.35e7, 1.45e7],
        masses: vec![1.0],
        flux_error: None,
    };
    let results = transmute_material(
        &mut material,
        &[dt],
        &steps(),
        Arc::clone(&loaded.chain),
        &loaded.branch,
        loaded.parts,
        Some(&request(vec![Source::IsomerBranching], 600)),
    )
    .expect("transmute");
    let nominal = &results.get_material(0, 1).expect("step 1").nuclides;
    let sd = results.uncertainty[&0].std_dev_at(0);
    let info = &results.uncertainty_info[&0];
    for (reaction, product) in [("(n,2n)", "In114_m1"), ("(n,n')", "In115_m1")] {
        let key = (
            "In115".to_string(),
            reaction.to_string(),
            product.to_string(),
        );
        let stated = info.isomer_production_sigma[&key];
        let relative = sd[product] / nominal[product];
        eprintln!(
            "ENDF/B-VIII.1 In115 {reaction} {product} at 13.5-14.5 MeV: MF=40 folded \
             {stated:.4}, inventory {relative:.4}"
        );
        assert!(
            (0.005..0.2).contains(&stated),
            "{product}: a stated sigma of a few percent, got {stated}"
        );
        assert!(
            (relative / stated - 1.0).abs() < 0.1,
            "{product}: inventory sigma {relative} against the stated {stated}"
        );
    }
    assert!(info
        .isomer_branchings_perturbed
        .contains(&("In115".to_string(), "(n,2n)".to_string())));
}
