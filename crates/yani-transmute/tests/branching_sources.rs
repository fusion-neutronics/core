//! Several isomeric-branching sources, in order, end to end.
//!
//! `transmutation_branch_ratios` can name more than one branching subsection.
//! Each `(parent, reaction)` channel then takes its split from the first source
//! that evaluates it, all of it, and the MF=40 covariance of that split from
//! the same source and no other. These tests hold the loader and the solve to
//! that, and check the numbers against the evaluations' own.
//!
//! The synthetic case always runs. The others read
//! `tests/fixtures/irdff-ii-branching-In115-Nb93` and
//! `tests/fixtures/endf-b8.1-branching-In115-Nb93`: the In115 and Nb93 rows of
//! the IRDFF-II and ENDF/B-VIII.1 `branching.arrow` and
//! `branching_covariance.arrow` of release 2026-10-12.1, cut out unmodified,
//! with each release's `provenance.json`. IRDFF-II's are its `(n,n')` and
//! `(n,2n)` channels, so its In115 `(n,gamma)` (two 33,000-point curves) is
//! left out and ENDF/B-VIII.1's In115 `(n,gamma)` is the only one here.
//! ENDF/B-VIII.1 evaluates no Nb93 `(n,2n)` isomer. They also read the chain
//! fixture and, for the solve, the In115 cross sections, which
//! `scripts/fetch_test_fixtures.py` fetches, and self-skip without them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use endf::radionuclide_production::LevelRoute;
use yamc_materials::Material;
use yani::{ChainNuclide, ChainReaction};
use yani_convert::branching::{write_branching, BranchingRow, StateFacts};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, BranchingReport, MultigroupSpectrum, TransmuteStep};

const IRDFF: &str = "irdff-ii";
const ENDF: &str = "endf-b8.1";
/// The flux magnitude of the one irradiation step, n/cm2/s.
const FLUX: f64 = 1.0e14;

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(library: &str) -> PathBuf {
    manifest().join(format!("tests/fixtures/{library}-branching-In115-Nb93"))
}

/// The ENDF/B-VIII.1 chain fixture's decay, reactions and fission-yield
/// subsections, or `None` on a machine that has not fetched it.
fn chain_dirs() -> Option<[String; 3]> {
    let root = manifest().join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    let dirs = ["decay", "reactions", "fission_yields"].map(|s| root.join(s));
    if let Some(missing) = dirs.iter().find(|d| !d.is_dir()) {
        eprintln!("skipping -- {} absent", missing.display());
        return None;
    }
    Some(dirs.map(|d| d.to_string_lossy().into_owned()))
}

/// The chain with the fixture branching of `sources` laid over it in order.
fn load(chain: &[String; 3], sources: &[&str]) -> yani::LoadedChain {
    let dirs: Vec<(String, String)> = sources
        .iter()
        .map(|s| (s.to_string(), fixture(s).to_string_lossy().into_owned()))
        .collect();
    let branching: Vec<(&str, &str)> = dirs.iter().map(|(s, d)| (s.as_str(), d.as_str())).collect();
    yani::load_chain_parts(&chain[0], Some(&chain[1]), Some(&chain[2]), &branching).expect("chain")
}

/// The `(energy, values)` of each curve of a channel, in order.
fn curves(loaded: &yani::LoadedChain, parent: &str, kind: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    loaded.branch.curves()[parent][kind]
        .iter()
        .map(|c| (c.energy.clone(), c.values.clone()))
        .collect()
}

fn source<'a>(loaded: &'a yani::LoadedChain, parent: &str, kind: &str) -> Option<&'a str> {
    loaded
        .branch
        .source_of(parent, kind)
        .map(|s| s.name.as_str())
}

#[test]
fn each_channel_takes_all_its_curves_from_the_first_source_that_evaluates_it() {
    let Some(chain) = chain_dirs() else {
        return;
    };
    let irdff = load(&chain, &[IRDFF]);
    let endf = load(&chain, &[ENDF]);

    let first = load(&chain, &[IRDFF, ENDF]);
    let names: Vec<&str> = first
        .branch
        .sources()
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(names, [IRDFF, ENDF]);
    // Named by their provenance.json as well as by the run.
    let libraries: Vec<Option<&str>> = first
        .branch
        .sources()
        .iter()
        .map(|s| s.library.as_deref())
        .collect();
    assert_eq!(libraries, [Some(IRDFF), Some(ENDF)]);
    for kind in ["(n,n')", "(n,2n)"] {
        assert_eq!(source(&first, "In115", kind), Some(IRDFF), "In115 {kind}");
        assert_eq!(
            curves(&first, "In115", kind),
            curves(&irdff, "In115", kind),
            "In115 {kind} is IRDFF-II's, curve for curve"
        );
        assert_ne!(curves(&first, "In115", kind), curves(&endf, "In115", kind));
    }
    // The one In115 channel the IRDFF-II fixture does not carry.
    assert_eq!(source(&first, "In115", "(n,gamma)"), Some(ENDF));
    assert_eq!(
        curves(&first, "In115", "(n,gamma)"),
        curves(&endf, "In115", "(n,gamma)")
    );
    assert_eq!(source(&first, "Nb93", "(n,2n)"), Some(IRDFF));

    let second = load(&chain, &[ENDF, IRDFF]);
    for kind in ["(n,n')", "(n,2n)"] {
        assert_eq!(source(&second, "In115", kind), Some(ENDF), "In115 {kind}");
        assert_eq!(curves(&second, "In115", kind), curves(&endf, "In115", kind));
    }
    assert_eq!(source(&second, "Nb93", "(n,n')"), Some(ENDF));
    // ENDF/B-VIII.1 evaluates no Nb93 (n,2n) isomer, so IRDFF-II's is the
    // first, and its metastable channel is grafted from it.
    assert_eq!(source(&second, "Nb93", "(n,2n)"), Some(IRDFF));
    assert!(second.chain["Nb93"]
        .reactions
        .iter()
        .any(|r| r.kind == "(n,2n)" && r.target.as_deref() == Some("Nb92_m1")));

    // With no source there is no overlay: no branching curves but MT=5's, and
    // no source to name.
    let none = load(&chain, &[]);
    assert!(none.branch.sources().is_empty());
    assert!(none
        .branch
        .curves()
        .values()
        .all(|kinds| kinds.keys().all(|k| k == yani::reactions::ANYTHING)));
}

fn chain_nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_evaluated: true,
        excitation_energy: None,
        level_index: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: vec![],
        fission_yields: None,
        sources: Default::default(),
    }
}

/// An isomer-only MF=10 row, one state.
fn isomer_row(parent: &str, reaction: &str, mt: i32, target: &str, lfs: i64) -> BranchingRow {
    BranchingRow {
        nuclide: parent.to_string(),
        reaction: reaction.to_string(),
        target: target.to_string(),
        quantity: "cross_section".to_string(),
        energy: vec![1.0e6, 2.0e7],
        values: vec![0.1, 0.1],
        states: vec![StateFacts {
            mt,
            lfs,
            lmf: Some(10),
            list_complete: false,
            level_route: LevelRoute::Energy,
            level_energy: 1.0e5,
            level_energy_difference: Some(0.0),
            match_energy: None,
            mf3: None,
        }],
        normalisation: None,
    }
}

/// A source's rows are passed over whole for a channel an earlier source
/// evaluates, grafting nothing, and every channel names its source.
#[test]
fn a_later_source_reaches_only_the_channels_no_earlier_one_evaluates() {
    let tmp = tempfile::tempdir().expect("tmp");
    let n2n_ground = ChainReaction {
        kind: "(n,2n)".to_string(),
        target: Some("Ag106".to_string()),
        branching: 1.0,
        q_value: Some(-9.5e6),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    let chain: HashMap<String, ChainNuclide> = [
        chain_nuclide("Ag107", vec![n2n_ground]),
        chain_nuclide("Ag107_m1", vec![]),
        chain_nuclide("Ag106", vec![]),
        chain_nuclide("Ag106_m1", vec![]),
        chain_nuclide("Ag106_m2", vec![]),
    ]
    .into_iter()
    .map(|n| (n.name.clone(), n))
    .collect();
    yani::export_chain_parts(&chain, tmp.path(), Some("test"), "test").expect("chain exports");
    let (first, second, empty) = (
        tmp.path().join("first"),
        tmp.path().join("second"),
        tmp.path().join("empty"),
    );
    write_branching(&[isomer_row("Ag107", "(n,2n)", 16, "Ag106_m1", 1)], &first).expect("first");
    write_branching(
        &[
            isomer_row("Ag107", "(n,2n)", 16, "Ag106_m2", 2),
            isomer_row("Ag107", "(n,n')", 4, "Ag107_m1", 1),
        ],
        &second,
    )
    .expect("second");
    std::fs::create_dir_all(&empty).expect("empty");
    let decay = tmp.path().join("decay");
    let reactions = tmp.path().join("reactions");
    let parse = |sources: &[(&str, &Path)]| {
        yani::parse_chain_parts(&decay, Some(&reactions), None, sources)
    };
    let targets = |chain: &HashMap<String, ChainNuclide>, kind: &str| -> Vec<String> {
        let mut t: Vec<String> = chain["Ag107"]
            .reactions
            .iter()
            .filter(|r| r.kind == kind)
            .filter_map(|r| r.target.clone())
            .collect();
        t.sort();
        t
    };

    let (chain, branch) =
        parse(&[("first", first.as_path()), ("second", second.as_path())]).expect("load");
    assert_eq!(
        branch.source_of("Ag107", "(n,2n)").map(|s| s.name.as_str()),
        Some("first")
    );
    let n2n = &branch.curves()["Ag107"]["(n,2n)"];
    assert_eq!(n2n.len(), 1);
    assert_eq!(n2n[0].target, "Ag106_m1");
    // The second source's (n,2n) row was passed over, so its state was not
    // grafted either.
    assert_eq!(targets(&chain, "(n,2n)"), ["Ag106", "Ag106_m1"]);
    // A channel only the second source evaluates is its.
    assert_eq!(
        branch.source_of("Ag107", "(n,n')").map(|s| s.name.as_str()),
        Some("second")
    );
    assert_eq!(targets(&chain, "(n,n')"), ["Ag107_m1"]);
    // Neither carries a provenance.json, so neither names a library.
    assert!(branch.sources().iter().all(|s| s.library.is_none()));

    let (chain, branch) =
        parse(&[("second", second.as_path()), ("first", first.as_path())]).expect("load");
    assert_eq!(
        branch.source_of("Ag107", "(n,2n)").map(|s| s.name.as_str()),
        Some("second")
    );
    assert_eq!(targets(&chain, "(n,2n)"), ["Ag106", "Ag106_m2"]);

    let twice = parse(&[("first", first.as_path()), ("first", second.as_path())])
        .expect_err("a source named twice");
    assert!(twice.to_string().contains("given twice"), "{twice}");
    let nothing = parse(&[("first", first.as_path()), ("typo", empty.as_path())])
        .expect_err("a source with no branching.arrow");
    assert!(
        nothing
            .to_string()
            .contains("'typo' has no branching.arrow"),
        "{nothing}"
    );
}

/// `(energy, values)` of `parent`'s `reaction` row to `target` in `library`'s
/// fixture `branching.arrow`, read straight from the file.
fn fixture_curve(
    library: &str,
    parent: &str,
    reaction: &str,
    target: &str,
) -> (Vec<f64>, Vec<f64>) {
    use arrow_array::{Array, Float64Array, ListArray, StringArray};
    let file = std::fs::File::open(fixture(library).join("branching.arrow")).expect("fixture");
    let reader = arrow_ipc::reader::FileReader::try_new(file, None).expect("arrow");
    for batch in reader {
        let batch = batch.expect("batch");
        let text = |name: &str| {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .expect("a string column")
                .clone()
        };
        let list = |name: &str, i: usize| -> Vec<f64> {
            let l = batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<ListArray>())
                .expect("a list column");
            let v = l.value(i);
            let v = v.as_any().downcast_ref::<Float64Array>().expect("f64");
            (0..v.len()).map(|k| v.value(k)).collect()
        };
        let (n, r, t) = (text("nuclide"), text("reaction"), text("target"));
        for i in 0..batch.num_rows() {
            if n.value(i) == parent && r.value(i) == reaction && t.value(i) == target {
                return (list("energy", i), list("values", i));
            }
        }
    }
    panic!("{library} has no {parent} {reaction} {target} row");
}

/// The average of a lin-lin curve over `[lo, hi]`, flat in energy, as the
/// solve's default within-group weight takes it.
fn group_average((energy, values): &(Vec<f64>, Vec<f64>), lo: f64, hi: f64) -> f64 {
    let at = |e: f64| -> f64 {
        if e < energy[0] {
            return 0.0;
        }
        let i = energy.partition_point(|&x| x <= e);
        if i >= energy.len() {
            return values[energy.len() - 1];
        }
        let (e0, e1, v0, v1) = (energy[i - 1], energy[i], values[i - 1], values[i]);
        v0 + (v1 - v0) * (e - e0) / (e1 - e0)
    };
    let mut nodes: Vec<f64> = vec![lo, hi];
    nodes.extend(energy.iter().copied().filter(|&e| e > lo && e < hi));
    nodes.sort_by(f64::total_cmp);
    nodes
        .windows(2)
        .map(|w| 0.5 * (at(w[0]) + at(w[1])) * (w[1] - w[0]))
        .sum::<f64>()
        / (hi - lo)
}

/// The relative sigma of `parent`'s state `(mt, lfs)` production that
/// `library`'s own MF=40 block states, over `groups` of `(lo, hi, production)`:
/// the block's relative covariance on its grid, each group's production split
/// across the block's intervals by energy width, as the solve splits a group a
/// covariance edge cuts.
fn stated_sigma(library: &str, parent: &str, mt: i32, lfs: i32, groups: &[(f64, f64, f64)]) -> f64 {
    use yamc_nuclide::covariance::expand::expand_ni;
    use yamc_nuclide::covariance::CovarianceData;
    let file = std::fs::File::open(fixture(library).join("branching_covariance.arrow"))
        .expect("fixture covariance");
    let batches: Vec<_> = arrow_ipc::reader::FileReader::try_new(file, None)
        .expect("arrow")
        .map(|b| b.expect("batch"))
        .collect();
    let blocks = yamc_nuclide::arrow::covariance_arrow::branching_covariance_blocks(&batches)
        .expect("blocks");
    let block = blocks
        .iter()
        .find(|b| b.nuclide == parent && b.block.mt == mt && b.lfs == lfs)
        .unwrap_or_else(|| panic!("{library} states no MF=40 for {parent} MT{mt} LFS{lfs}"));
    let CovarianceData::Ni(ni) = &block.block.data else {
        panic!("an NI block");
    };
    let e = expand_ni(ni).expect("expands");
    let grid = &e.row_energies;
    let w: Vec<f64> = grid
        .windows(2)
        .map(|cell| {
            groups
                .iter()
                .map(|&(lo, hi, p)| p * (hi.min(cell[1]) - lo.max(cell[0])).max(0.0) / (hi - lo))
                .sum()
        })
        .collect();
    let total: f64 = groups.iter().map(|g| g.2).sum();
    let mut var = 0.0;
    for i in 0..w.len() {
        for j in 0..w.len() {
            var += w[i] * w[j] * e.get(i, j);
        }
    }
    var.sqrt() / total
}

/// What one solve says about a parent's isomers.
struct Outcome {
    /// `(reaction, product) -> per-atom production rate`, 1/s, where the
    /// solve made the product by that reaction at all.
    rates: HashMap<(String, String), f64>,
    /// `(reaction, product) -> the stated relative sigma of its production`.
    sigma: HashMap<(String, String), f64>,
    report: BranchingReport,
}

impl Outcome {
    fn rate(&self, reaction: &str, product: &str) -> Option<f64> {
        self.rates
            .get(&(reaction.to_string(), product.to_string()))
            .copied()
    }

    fn sigma(&self, reaction: &str, product: &str) -> Option<f64> {
        self.sigma
            .get(&(reaction.to_string(), product.to_string()))
            .copied()
    }

    fn source(&self, parent: &str, reaction: &str) -> Option<&str> {
        self.report
            .channels
            .iter()
            .find(|c| c.parent == parent && c.reaction == reaction)
            .unwrap_or_else(|| panic!("{parent} {reaction} is not reported"))
            .source
            .as_deref()
    }
}

/// Transmute a one-nuclide `material` over one step of `spectrum` with the
/// MF=40 source drawn, and read what it says about `isomers`.
fn solve(
    loaded: &yani::LoadedChain,
    mut material: Material,
    spectrum: MultigroupSpectrum,
    isomers: &[(&str, &str)],
) -> Outcome {
    let parent = material.nuclide_data.keys().next().expect("one").clone();
    let request = DataUncertainty {
        seed: 11,
        samples: Some(16),
        sources: vec![Source::IsomerBranching],
        attribution: false,
        ..Default::default()
    };
    let results = transmute_material(
        &mut material,
        &[spectrum],
        &[TransmuteStep {
            dt: 60.0,
            irradiation: Some((0, FLUX)),
        }],
        Arc::clone(&loaded.chain),
        &loaded.branch,
        loaded.parts,
        Some(&request),
    )
    .expect("transmute");
    let edges = results.get_reaction_rates(0, 0).expect("step 0");
    let info = &results.uncertainty_info[&0];
    let mut rates = HashMap::new();
    let mut sigma = HashMap::new();
    for &(reaction, product) in isomers {
        let key = (reaction.to_string(), product.to_string());
        let rate = edges
            .get(&parent)
            .and_then(|kinds| kinds.get(reaction))
            .and_then(|targets| targets.iter().find(|(t, _)| t.as_deref() == Some(product)))
            .map(|(_, r)| *r);
        if let Some(rate) = rate.filter(|r| *r > 0.0) {
            rates.insert(key.clone(), rate);
        }
        let stated = (parent.clone(), reaction.to_string(), product.to_string());
        if let Some(s) = info.isomer_production_sigma.get(&stated) {
            sigma.insert(key, *s);
        }
    }
    Outcome {
        rates,
        sigma,
        report: results.get_branching_report(0, 0).expect("report").clone(),
    }
}

fn reaction(
    mt: i32,
    energy: Vec<f64>,
    cross_section: Vec<f64>,
) -> yamc_nuclide::reaction::Reaction {
    yamc_nuclide::reaction::Reaction {
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

/// Nb93 with a flat 2 b inelastic from 30 keV and a flat 1 b `(n,2n)` from
/// 9 MeV, both to 20 MeV, as loaded from an ENDF/B-VIII.1 folder: totals for
/// the isomer partials to sit on, well above them.
fn niobium() -> Material {
    let temperature = "294".to_string();
    let reactions = HashMap::from([
        (4, Arc::new(reaction(4, vec![3.0e4, 2.0e7], vec![2.0, 2.0]))),
        (
            16,
            Arc::new(reaction(16, vec![9.0e6, 2.0e7], vec![1.0, 1.0])),
        ),
    ]);
    let nuclide = yamc_nuclide::nuclide::Nuclide {
        name: Some("Nb93".to_string()),
        element: None,
        atomic_symbol: Some("Nb".to_string()),
        atomic_number: Some(41),
        neutron_number: Some(52),
        mass_number: Some(93),
        atomic_weight_ratio: Some(92.1),
        library: Some("endfb-8.1".to_string()),
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
        covariance: None,
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
        HashMap::from([("Nb93".to_string(), 5.5e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("niobium");
    m.set_temperature(&temperature);
    m.volume = Some(1.0);
    m.nuclide_data.insert("Nb93".to_string(), Arc::new(nuclide));
    m
}

/// Two groups, 1-2 MeV and 13.5-14.5 MeV, each half the flux.
fn two_groups() -> (MultigroupSpectrum, [(f64, f64, f64); 2]) {
    let spectrum = MultigroupSpectrum {
        boundaries: vec![1.0e6, 2.0e6, 1.35e7, 1.45e7],
        masses: vec![0.5, 0.0, 0.5],
        flux_error: None,
    };
    (spectrum, [(1.0e6, 2.0e6, 0.5), (1.35e7, 1.45e7, 0.5)])
}

const NB93_ISOMERS: [(&str, &str); 2] = [("(n,n')", "Nb93_m1"), ("(n,2n)", "Nb92_m1")];

/// Nb93 `(n,n')` Nb93_m1, which IRDFF-II and ENDF/B-VIII.1 both evaluate with
/// an MF=40 covariance, and differently (IRDFF-II's partial is 0.855 of
/// ENDF/B-VIII.1's over 1-2 MeV), and Nb93 `(n,2n)` Nb92_m1, which only
/// IRDFF-II evaluates. With IRDFF-II first, both splits and both covariances
/// are IRDFF-II's own numbers, bit for bit what IRDFF-II alone gives; with
/// ENDF/B-VIII.1 first, `(n,n')` is ENDF/B-VIII.1's, bit for bit, and
/// `(n,2n)` still IRDFF-II's, the one source that evaluates it.
///
/// The branching rows and their MF=40 blocks are the evaluations' own; the
/// chain and the transport totals are made up, so this runs on any machine.
#[test]
fn nb93_takes_each_split_and_its_covariance_from_the_first_source_that_evaluates_it() {
    let tmp = tempfile::tempdir().expect("tmp");
    let n2n_ground = ChainReaction {
        kind: "(n,2n)".to_string(),
        target: Some("Nb92".to_string()),
        branching: 1.0,
        q_value: Some(-8.8e6),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    let chain: HashMap<String, ChainNuclide> = [
        chain_nuclide("Nb93", vec![n2n_ground]),
        chain_nuclide("Nb93_m1", vec![]),
        chain_nuclide("Nb92", vec![]),
        chain_nuclide("Nb92_m1", vec![]),
    ]
    .into_iter()
    .map(|n| (n.name.clone(), n))
    .collect();
    yani::export_chain_parts(&chain, tmp.path(), Some("test"), "test").expect("chain exports");
    let dir = |sub: &str| tmp.path().join(sub).to_string_lossy().into_owned();
    let (irdff_dir, endf_dir) = (
        fixture(IRDFF).to_string_lossy().into_owned(),
        fixture(ENDF).to_string_lossy().into_owned(),
    );
    let run = |sources: &[(&str, &str)]| {
        let loaded = yani::load_chain_parts(&dir("decay"), Some(&dir("reactions")), None, sources)
            .expect("chain");
        solve(&loaded, niobium(), two_groups().0, &NB93_ISOMERS)
    };
    let irdff_alone = run(&[(IRDFF, &irdff_dir)]);
    let irdff_first = run(&[(IRDFF, &irdff_dir), (ENDF, &endf_dir)]);
    let endf_alone = run(&[(ENDF, &endf_dir)]);
    let endf_first = run(&[(ENDF, &endf_dir), (IRDFF, &irdff_dir)]);

    // Each library's own numbers: an isomer's production is its absolute
    // MF=10 partial averaged over each group, and its sigma what its MF=40
    // block states over the groups.
    let groups = two_groups().1;
    let own = |library: &str, reaction: &str, product: &str, mt: i32| -> (f64, f64) {
        let curve = fixture_curve(library, "Nb93", reaction, product);
        let per_group: Vec<(f64, f64, f64)> = groups
            .iter()
            .map(|&(lo, hi, flux)| (lo, hi, flux * group_average(&curve, lo, hi)))
            .collect();
        let rate = per_group.iter().map(|g| g.2).sum::<f64>() * 1.0e-24 * FLUX;
        (rate, stated_sigma(library, "Nb93", mt, 1, &per_group))
    };
    let close = |got: Option<f64>, want: f64, tolerance: f64, what: &str| {
        let got = got.unwrap_or_else(|| panic!("{what}: none"));
        assert!(
            (got / want - 1.0).abs() < tolerance,
            "{what}: {got}, the evaluation gives {want}"
        );
    };

    let (irdff_rate, irdff_sigma) = own(IRDFF, "(n,n')", "Nb93_m1", 4);
    let (endf_rate, endf_sigma) = own(ENDF, "(n,n')", "Nb93_m1", 4);
    let (n2n_rate, n2n_sigma) = own(IRDFF, "(n,2n)", "Nb92_m1", 16);
    for (name, outcome) in [
        ("IRDFF-II alone", &irdff_alone),
        ("IRDFF-II first", &irdff_first),
    ] {
        close(outcome.rate("(n,n')", "Nb93_m1"), irdff_rate, 1e-12, name);
        close(outcome.sigma("(n,n')", "Nb93_m1"), irdff_sigma, 1e-9, name);
        assert_eq!(outcome.source("Nb93", "(n,n')"), Some(IRDFF), "{name}");
    }
    for (name, outcome) in [
        ("ENDF/B-VIII.1 alone", &endf_alone),
        ("ENDF/B-VIII.1 first", &endf_first),
    ] {
        close(outcome.rate("(n,n')", "Nb93_m1"), endf_rate, 1e-12, name);
        close(outcome.sigma("(n,n')", "Nb93_m1"), endf_sigma, 1e-9, name);
        assert_eq!(outcome.source("Nb93", "(n,n')"), Some(ENDF), "{name}");
    }
    for (name, outcome) in [
        ("IRDFF-II alone", &irdff_alone),
        ("IRDFF-II first", &irdff_first),
        ("ENDF/B-VIII.1 first", &endf_first),
    ] {
        close(outcome.rate("(n,2n)", "Nb92_m1"), n2n_rate, 1e-12, name);
        close(outcome.sigma("(n,2n)", "Nb92_m1"), n2n_sigma, 1e-9, name);
        assert_eq!(outcome.source("Nb93", "(n,2n)"), Some(IRDFF), "{name}");
    }
    // ENDF/B-VIII.1 alone evaluates no Nb93 (n,2n) isomer, so it makes none.
    assert_eq!(endf_alone.rate("(n,2n)", "Nb92_m1"), None);
    assert_eq!(endf_alone.sigma("(n,2n)", "Nb92_m1"), None);

    // Listed first or alone, a source gives the same numbers to the bit.
    for (reaction, product) in NB93_ISOMERS {
        assert_eq!(
            irdff_first.rate(reaction, product),
            irdff_alone.rate(reaction, product)
        );
        assert_eq!(
            irdff_first.sigma(reaction, product),
            irdff_alone.sigma(reaction, product)
        );
    }
    assert_eq!(
        endf_first.rate("(n,n')", "Nb93_m1"),
        endf_alone.rate("(n,n')", "Nb93_m1")
    );
    assert_eq!(
        endf_first.sigma("(n,n')", "Nb93_m1"),
        endf_alone.sigma("(n,n')", "Nb93_m1")
    );
    assert!(irdff_rate != endf_rate && irdff_sigma != endf_sigma);
    eprintln!(
        "Nb93 (n,n') Nb93_m1 over 1-2 and 13.5-14.5 MeV: IRDFF-II {:.6} b, sigma {:.4}; \
         ENDF/B-VIII.1 {:.6} b, sigma {:.4}. Nb93 (n,2n) Nb92_m1, IRDFF-II only: {:.6} b, \
         sigma {:.4}",
        irdff_rate / (1.0e-24 * FLUX),
        irdff_sigma,
        endf_rate / (1.0e-24 * FLUX),
        endf_sigma,
        n2n_rate / (1.0e-24 * FLUX),
        n2n_sigma,
    );

    // The report names the sources, and a run whose splits are not all from
    // its cross-section library says it is mixed.
    assert_eq!(irdff_first.report.sources, [IRDFF]);
    assert_eq!(endf_first.report.sources, [ENDF, IRDFF]);
    assert_eq!(endf_alone.report.sources, [ENDF]);
    assert!(!endf_alone.report.mixed, "{:?}", endf_alone.report.channels);
    assert!(endf_first.report.mixed);
    assert!(
        irdff_alone.report.mixed,
        "IRDFF-II splits on ENDF/B-VIII.1 totals"
    );
    for channel in &endf_alone.report.channels {
        assert_eq!(channel.cross_section_library.as_deref(), Some(ENDF));
    }
}

/// In115 with its real ENDF/B-VIII.1 cross sections under a 13.5-14.5 MeV
/// group. IRDFF-II listed first supplies its `(n,n')` and `(n,2n)` splits,
/// and the report says so, but the numbers are ENDF/B-VIII.1's to the bit:
/// below 20 MeV ENDF/B-VIII.1's In115 `(n,n')` In115_m1 and `(n,2n)` In114_m1
/// partials and their MF=40 covariances are IRDFF-II's (IRDFF-II carries both
/// on to 60 MeV), so where the two are the same data the choice changes
/// nothing but the record of it.
#[test]
fn in115_names_irdff_ii_where_it_supplies_the_split_whatever_the_numbers() {
    let Some(chain) = chain_dirs() else {
        return;
    };
    let Some(in115) = yamc_test_cache::nuclide("In115") else {
        eprintln!("skipping -- In115 fixture absent");
        return;
    };
    let isomers = [("(n,n')", "In115_m1"), ("(n,2n)", "In114_m1")];
    let run = |sources: &[&str]| {
        let loaded = load(&chain, sources);
        let mut material = Material::new(
            HashMap::from([("In115".to_string(), 3.8e-2)]),
            "atom",
            "sum",
            None,
        )
        .expect("material");
        material.set_temperature("294");
        material.volume = Some(1.0);
        let scope = yamc_nuclide::load_scope::LoadScope::activation(
            yani_transmute::activation_mts(&loaded.chain, &loaded.branch),
        );
        let data = yamc_nuclide::nuclide::get_or_load_nuclide(
            "In115",
            &HashMap::from([("In115".to_string(), in115.clone())]),
            &scope,
        )
        .expect("load");
        material.nuclide_data.insert("In115".to_string(), data);
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.35e7, 1.45e7],
            masses: vec![1.0],
            flux_error: None,
        };
        solve(&loaded, material, spectrum, &isomers)
    };
    let irdff_first = run(&[IRDFF, ENDF]);
    let endf_alone = run(&[ENDF]);
    for (reaction, product) in isomers {
        assert_eq!(irdff_first.source("In115", reaction), Some(IRDFF));
        assert_eq!(endf_alone.source("In115", reaction), Some(ENDF));
        assert!(irdff_first.rate(reaction, product).is_some_and(|r| r > 0.0));
        assert_eq!(
            irdff_first.rate(reaction, product),
            endf_alone.rate(reaction, product)
        );
        assert_eq!(
            irdff_first.sigma(reaction, product),
            endf_alone.sigma(reaction, product)
        );
    }
    // The IRDFF-II fixture carries no In115 (n,gamma), so ENDF/B-VIII.1's is
    // the one used, and the run is mixed.
    assert_eq!(irdff_first.source("In115", "(n,gamma)"), Some(ENDF));
    assert_eq!(irdff_first.report.sources, [IRDFF, ENDF]);
    assert!(irdff_first.report.mixed);
    assert!(!endf_alone.report.mixed, "{:?}", endf_alone.report.channels);
}

/// With no branching source, and with one, a run gives what it gave before
/// sources could be listed, to the bit: the hex values below are what
/// origin/main at 9edb57a0 gives for the same Nb93 case, two one-hour steps
/// under the two-group spectrum and then a 13.5-14.5 MeV group, with the
/// MF=40 source drawn. (The same comparison on ENDF/B-VIII.1 In115, with no
/// source, ENDF/B-VIII.1 alone and IRDFF-II alone, nominal and drawn, every
/// density, edge rate, sigma and standard deviation: 481 numbers, all
/// identical.)
#[test]
fn no_source_and_one_source_give_what_they_gave_before_to_the_bit() {
    let tmp = tempfile::tempdir().expect("tmp");
    let n2n_ground = ChainReaction {
        kind: "(n,2n)".to_string(),
        target: Some("Nb92".to_string()),
        branching: 1.0,
        q_value: Some(-8.8e6),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    let chain: HashMap<String, ChainNuclide> = [
        chain_nuclide("Nb93", vec![n2n_ground]),
        chain_nuclide("Nb93_m1", vec![]),
        chain_nuclide("Nb92", vec![]),
        chain_nuclide("Nb92_m1", vec![]),
    ]
    .into_iter()
    .map(|n| (n.name.clone(), n))
    .collect();
    yani::export_chain_parts(&chain, tmp.path(), Some("test"), "test").expect("chain exports");
    let dir = |sub: &str| tmp.path().join(sub).to_string_lossy().into_owned();
    let irdff = fixture(IRDFF).to_string_lossy().into_owned();
    let spectra = [
        two_groups().0,
        MultigroupSpectrum {
            boundaries: vec![1.35e7, 1.45e7],
            masses: vec![1.0],
            flux_error: None,
        },
    ];
    let steps: Vec<TransmuteStep> = (0..2)
        .map(|i| TransmuteStep {
            dt: 3600.0,
            irradiation: Some((i, FLUX)),
        })
        .collect();
    let request = DataUncertainty {
        seed: 11,
        samples: Some(16),
        sources: vec![Source::IsomerBranching],
        attribution: false,
        ..Default::default()
    };
    let run = |sources: &[(&str, &str)]| {
        let loaded = yani::load_chain_parts(&dir("decay"), Some(&dir("reactions")), None, sources)
            .expect("chain");
        transmute_material(
            &mut niobium(),
            &spectra,
            &steps,
            Arc::clone(&loaded.chain),
            &loaded.branch,
            loaded.parts,
            Some(&request),
        )
        .expect("transmute")
    };
    let bits = |results: &yani_transmute::TransmutationResults| -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = results
            .get_material(0, 2)
            .expect("step 2")
            .nuclides
            .iter()
            .map(|(n, v)| (n.clone(), v.to_bits()))
            .collect();
        for ((parent, kind, product), sigma) in
            &results.uncertainty_info[&0].isomer_production_sigma
        {
            out.push((format!("{parent} {kind} {product}"), sigma.to_bits()));
        }
        out.sort();
        out
    };

    let none = bits(&run(&[]));
    let want_none: Vec<(String, u64)> = vec![
        ("Nb92".to_string(), 0x3e654294a214c5b6),
        ("Nb93".to_string(), 0x3fac28f46e661208),
    ];
    assert_eq!(none, want_none);

    let one = bits(&run(&[(IRDFF, &irdff)]));
    let want_one: Vec<(String, u64)> = vec![
        ("Nb92".to_string(), 0x3e5be842291d9d43),
        ("Nb92_m1".to_string(), 0x3e4d39ce0ff7c28b),
        ("Nb93".to_string(), 0x3fac28f4590bec5c),
        ("Nb93 (n,2n) Nb92_m1".to_string(), 0x3f7634714a31bdfb),
        ("Nb93 (n,n') Nb93_m1".to_string(), 0x3fb28a5ae3e5ca75),
        ("Nb93_m1".to_string(), 0x3e255a264537f102),
    ];
    assert_eq!(one, want_one);
}
