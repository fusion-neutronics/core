//! Convert real evaluations, then read the result back with yani's own reader.
//!
//! The point is to cross the language-free equivalent of the boundary that
//! issue #379 went through: a writer and a reader that share assumptions can be
//! wrong together and stay green. Here `yani_convert` writes the files and
//! `yani::parse_chain_parts` takes them back, which is the reader a real
//! transmutation run uses.
//!
//! Fixtures are pulled in with `include_bytes!`, so a fixture that goes missing
//! is a compile error rather than a test that quietly checks nothing.

use std::collections::BTreeMap;

use endf::chain::Chain;
use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

/// The Cs137 to Ba137m chain plus the In115 and Xe136 activation paths: small,
/// committed, and between them they exercise a decay mode with a target, a
/// metastable state, discrete photon sources and neutron reactions with Q.
const DECAY: &[&[u8]] = &[
    fixture!("dec-055_Cs_137.endf.xz"),
    fixture!("dec-054_Xe_136.endf.xz"),
    fixture!("dec-054_Xe_137.endf.xz"),
    fixture!("dec-049_In_115.endf.xz"),
    fixture!("dec-049_In_116.endf.xz"),
    fixture!("dec-049_In_116m1.endf.xz"),
    fixture!("dec-049_In_116m2.endf.xz"),
    fixture!("dec-050_Sn_115.endf.xz"),
    fixture!("dec-050_Sn_116.endf.xz"),
    fixture!("dec-048_Cd_116.endf.xz"),
];
const NEUTRON: &[&[u8]] = &[
    fixture!("n-049_In-115_trimmed.endf.xz"),
    fixture!("n-054_Xe_136_trimmed.endf.xz"),
];
const FPY: &[&[u8]] = &[fixture!("synthetic-nfy.endf.xz")];

fn provenance() -> yani_convert::Provenance {
    yani_convert::Provenance {
        library: "endf-b8.1".to_string(),
        // The case this field exists for is a chain whose decay half came from
        // elsewhere; here it did not, so it stays empty and the written
        // provenance records that nobody said otherwise.
        decay_library: String::new(),
        data_version: "2026-08-09.1".to_string(),
        created_utc: "2026-08-09T00:00:00+00:00".to_string(),
    }
}

fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

fn materials(blobs: &[&[u8]]) -> Vec<Material> {
    blobs
        .iter()
        .map(|b| Material::from_str(&text(b)).expect("fixture parses"))
        .collect()
}

struct Converted {
    dir: std::path::PathBuf,
    chain: Chain,
    sources: BTreeMap<String, Vec<yani_convert::SourceRow>>,
}

fn convert(name: &str) -> Converted {
    let decay = materials(DECAY);
    let fpy = materials(FPY);
    let neutron = materials(NEUTRON);

    let q_values = endf::chain::q_values(&neutron);
    let chain = Chain::from_endf(&decay, &fpy, &q_values, &endf::chain::DEFAULT_REACTIONS)
        .expect("chain builds from the fixtures");
    let sources = yani_convert::decay_sources(&decay).expect("decay sources read");

    let dir = std::env::temp_dir().join(format!("yani-convert-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::write_decay(&chain, &sources, &dir.join("decay")).expect("decay written");
    yani_convert::write_reactions(&chain, &dir.join("reactions")).expect("reactions written");
    yani_convert::write_fission_yields(&chain, &dir.join("fission_yields"))
        .expect("fission yields written");

    Converted {
        dir,
        chain,
        sources,
    }
}

/// The floor that stops everything below passing vacuously.
///
/// A converter that produced an empty chain would satisfy every "did it round
/// trip" assertion perfectly.
#[test]
fn the_fixtures_produce_a_chain_worth_testing() {
    let c = convert("floor");
    assert!(
        c.chain.nuclides.len() >= 10,
        "only {} nuclides; the fixture set has shrunk and the round-trip tests \
         below would pass on almost nothing",
        c.chain.nuclides.len()
    );
    let decays: usize = c.chain.nuclides.iter().map(|n| n.decay_modes.len()).sum();
    let reactions: usize = c.chain.nuclides.iter().map(|n| n.reactions.len()).sum();
    assert!(decays > 0, "no decay modes in the chain");
    assert!(reactions > 0, "no neutron reactions in the chain");
    assert!(!c.sources.is_empty(), "no decay spectra were read");
    let _ = std::fs::remove_dir_all(&c.dir);
}

/// Everything written must come back through the reader a real run uses.
#[test]
fn yani_reads_what_the_converter_writes() {
    let c = convert("roundtrip");
    let (back, _branch) = yani::parse_chain_parts(
        &c.dir.join("decay"),
        Some(&c.dir.join("reactions")),
        Some(&c.dir.join("fission_yields")),
        None,
    )
    .expect("yani reads the converted chain");

    assert_eq!(
        back.len(),
        c.chain.nuclides.len(),
        "the reader lost or invented nuclides"
    );
    for written in &c.chain.nuclides {
        let read = back
            .get(&written.name)
            .unwrap_or_else(|| panic!("{} did not survive the round trip", written.name));
        assert_eq!(read.half_life, written.half_life, "{}", written.name);
        assert_eq!(
            read.decays.len(),
            written.decay_modes.len(),
            "{} decay modes",
            written.name
        );
        assert_eq!(
            read.reactions.len(),
            written.reactions.len(),
            "{} reactions",
            written.name
        );
    }
    let _ = std::fs::remove_dir_all(&c.dir);
}

/// Q survives, which the previous writer could not manage.
///
/// `reactions/reactions.arrow` declares Q non-nullable, and the value only
/// exists in the evaluation: nothing downstream can reconstruct it, so losing
/// it here is silent and permanent.
#[test]
fn reaction_q_values_survive() {
    let c = convert("q");
    let (back, _) = yani::parse_chain_parts(
        &c.dir.join("decay"),
        Some(&c.dir.join("reactions")),
        Some(&c.dir.join("fission_yields")),
        None,
    )
    .expect("reads back");

    let mut compared = 0;
    for written in &c.chain.nuclides {
        let read = back.get(&written.name).expect("present");
        for (w, r) in written.reactions.iter().zip(read.reactions.iter()) {
            assert_eq!(
                r.q_value,
                Some(w.q_value),
                "{} {} lost its Q",
                written.name,
                w.kind
            );
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no reactions were compared, so this test proved nothing"
    );
    let _ = std::fs::remove_dir_all(&c.dir);
}

/// A borrowed-yield nuclide is recorded as an alias, not as a copy.
#[test]
fn borrowed_yields_are_written_as_aliases() {
    let c = convert("aliases");
    let borrowed: Vec<&str> = c
        .chain
        .nuclides
        .iter()
        .filter(|n| n.borrowed_yields_from.is_some())
        .map(|n| n.name.as_str())
        .collect();

    let aliases = c.dir.join("fission_yields/aliases.arrow");
    if borrowed.is_empty() {
        assert!(
            !aliases.exists(),
            "an aliases file was written for a chain with no borrowed yields"
        );
    } else {
        assert!(
            aliases.exists(),
            "{} nuclides borrow yields and no aliases file was written",
            borrowed.len()
        );
    }
    let _ = std::fs::remove_dir_all(&c.dir);
}

/// The one-call entry point produces a directory yani reads, with the
/// provenance a published library carries.
#[test]
fn convert_transmutation_writes_a_complete_directory() {
    let decay = materials(DECAY);
    let fpy = materials(FPY);
    let neutron = materials(NEUTRON);
    let dir = std::env::temp_dir().join(format!("yani-convert-full-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let q_values = endf::chain::q_values(&neutron);
    let chain = yani_convert::convert_transmutation(
        &yani_convert::Inputs {
            decay: &decay,
            fpy: &fpy,
            q_values: &q_values,
            decay_fill: &[],
            decay_fill_library: "",
        },
        &endf::chain::DEFAULT_REACTIONS,
        None,
        &["decay", "reactions", "fission_yields"],
        &dir,
        &provenance(),
    )
    .expect("conversion succeeds");
    assert!(
        chain.nuclides.len() >= 10,
        "chain is too small to prove anything"
    );

    let (back, _) = yani::parse_chain_parts(
        &dir.join("decay"),
        Some(&dir.join("reactions")),
        Some(&dir.join("fission_yields")),
        None,
    )
    .expect("yani reads it");
    assert_eq!(back.len(), chain.nuclides.len());

    // data_version is what yamc compares a cached copy against (#366), so an
    // unstamped directory is a cache that can never be invalidated.
    for subsection in ["decay", "reactions", "fission_yields"] {
        let text = std::fs::read_to_string(dir.join(subsection).join("provenance.json"))
            .unwrap_or_else(|_| panic!("{subsection} has no provenance.json"));
        let v: serde_json::Value = serde_json::from_str(&text).expect("provenance is JSON");
        assert_eq!(v["subsection"], subsection);
        assert_eq!(v["library"], "endf-b8.1");
        assert_eq!(v["data_version"], "2026-08-09.1");
    }

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest"),
    )
    .expect("manifest is JSON");
    assert_eq!(manifest["format_version"], 2);
    assert_eq!(manifest["data_version"], "2026-08-09.1");
    for subsection in ["decay", "reactions", "fission_yields"] {
        assert_eq!(
            manifest["subsections"][subsection]["path"], subsection,
            "manifest does not list {subsection}"
        );
    }

    // Reproducible when the timestamp is supplied, judged on decoded content
    // rather than bytes.
    //
    // The bytes are NOT stable across processes, and that is not this
    // converter's doing: nuclear_data_schema::meta builds the section metadata
    // as a std HashMap, arrow-ipc serialises it in iteration order, and
    // RandomState reseeds every process. So two identical runs emit the same
    // values with `filetype` and `version` in a different order in the schema.
    // yani::export_chain_parts has the same property.
    //
    // Worth knowing beyond this test: it rules out content hashing as a way to
    // detect that published data changed, which was one of the options weighed
    // for issue #366. A hash over these files churns on every rebuild for no
    // reason. The stamped data_version that was chosen instead does not.
    let again = std::env::temp_dir().join(format!("yani-convert-again-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&again);
    yani_convert::convert_transmutation(
        &yani_convert::Inputs {
            decay: &decay,
            fpy: &fpy,
            q_values: &q_values,
            decay_fill: &[],
            decay_fill_library: "",
        },
        &endf::chain::DEFAULT_REACTIONS,
        None,
        &["decay", "reactions", "fission_yields"],
        &again,
        &provenance(),
    )
    .expect("second conversion succeeds");

    assert_eq!(
        std::fs::read(dir.join("manifest.json")).expect("manifest"),
        std::fs::read(again.join("manifest.json")).expect("manifest"),
        "manifest.json is plain JSON and must be byte-stable"
    );

    let (a, _) = yani::parse_chain_parts(
        &dir.join("decay"),
        Some(&dir.join("reactions")),
        Some(&dir.join("fission_yields")),
        None,
    )
    .expect("first reads");
    let (b, _) = yani::parse_chain_parts(
        &again.join("decay"),
        Some(&again.join("reactions")),
        Some(&again.join("fission_yields")),
        None,
    )
    .expect("second reads");
    assert_eq!(
        a.len(),
        b.len(),
        "two identical runs disagree on nuclide count"
    );
    for (name, first) in &a {
        let second = b.get(name).expect("same nuclides");
        assert_eq!(first.half_life, second.half_life, "{name} half life");
        assert_eq!(
            first.decay_energy, second.decay_energy,
            "{name} decay energy"
        );
        assert_eq!(
            first.reactions.len(),
            second.reactions.len(),
            "{name} reactions"
        );
        for (x, y) in first.reactions.iter().zip(second.reactions.iter()) {
            assert_eq!(x.q_value, y.q_value, "{name} {} Q", x.kind);
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&again);
}

/// Branching is written as its own subsection and merged into the manifest.
///
/// The fixtures are the In115 and Xe136 neutron evaluations, which is what
/// makes this worth running: In115 (n,gamma) populates In116 and In116_m1, so
/// the level-to-isomer resolution has something real to resolve.
#[test]
fn branching_is_written_and_joins_the_manifest() {
    let dir = std::env::temp_dir().join(format!("yani-convert-branch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");

    let plain = |names: &[&[u8]], stem: &str| -> Vec<String> {
        names
            .iter()
            .enumerate()
            .map(|(i, blob)| {
                let p = dir.join(format!("{stem}{i}.endf"));
                std::fs::write(&p, text(blob)).expect("write fixture");
                p.to_string_lossy().into_owned()
            })
            .collect()
    };
    let neutron_files = plain(NEUTRON, "n");
    let decay_files = plain(DECAY, "d");

    let out = dir.join("out");

    // The three chain subsections first, so the branching call has a manifest
    // to merge into rather than one to create. That is the real sequence: a
    // library is assembled by several calls, and each must leave the others'
    // entries alone.
    let fpy_files = plain(FPY, "f");
    yani_convert::convert_transmutation_files(
        &decay_files,
        &fpy_files,
        &neutron_files,
        &[],
        "",
        None,
        None,
        None,
        &out,
        &provenance(),
    )
    .expect("chain conversion succeeds");

    let stats = yani_convert::convert_branching_files(
        &neutron_files,
        &decay_files,
        &out,
        &provenance(),
        3000.0,
        yani_convert::branching::DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching conversion succeeds");

    assert!(
        stats.parents >= 2,
        "only {} parents were read; the fixtures are not being found",
        stats.parents
    );
    assert!(
        out.join("branching/branching.arrow").is_file(),
        "no branching.arrow was written"
    );
    assert!(
        out.join("branching/provenance.json").is_file(),
        "branching has no provenance"
    );

    // All four subsections in one manifest: the branching call must not have
    // overwritten what the chain call recorded.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    for subsection in ["decay", "reactions", "fission_yields", "branching"] {
        assert_eq!(
            manifest["subsections"][subsection]["path"], subsection,
            "the manifest lost {subsection}"
        );
    }

    // yani reads branching through a separate path from the three chain
    // subsections, so exercise it rather than assuming a file that exists loads.
    let (chain, branch) = yani::parse_chain_parts(
        &out.join("decay"),
        Some(&out.join("reactions")),
        Some(&out.join("fission_yields")),
        Some(&out.join("branching")),
    )
    .expect("yani reads the chain and its branching subsection");
    assert!(!chain.is_empty(), "no chain came back");
    assert!(
        !branch.is_empty(),
        "yani read the directory but found no branching curves"
    );
    let curves: usize = branch.values().map(|by_reaction| by_reaction.len()).sum();
    assert!(curves > 0, "the branching table has no curves in it");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Passing no reaction list must follow every reaction, not the six-name set.
///
/// `endf::chain::DEFAULT_REACTIONS` is six names. Defaulting to it drops 2554
/// of the 5175 reaction rows ENDF/B-VIII.1 produces: every (n,na), (n,np),
/// (n,2na) and the rest are simply absent from the network, and nothing says
/// so. Measured against the published data before this was fixed.
#[test]
fn the_default_reaction_set_is_not_the_short_one() {
    let decay = materials(DECAY);
    let fpy = materials(FPY);
    let neutron = materials(NEUTRON);

    let count = |reactions: Option<&[String]>, name: &str| -> usize {
        let dir = std::env::temp_dir().join(format!("yani-rx-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        yani_convert::convert_transmutation_files(
            &DECAY
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let p = dir.join(format!("d{i}.endf"));
                    std::fs::create_dir_all(&dir).ok();
                    std::fs::write(&p, text(b)).expect("write");
                    p.to_string_lossy().into_owned()
                })
                .collect::<Vec<_>>(),
            &FPY.iter()
                .enumerate()
                .map(|(i, b)| {
                    let p = dir.join(format!("f{i}.endf"));
                    std::fs::write(&p, text(b)).expect("write");
                    p.to_string_lossy().into_owned()
                })
                .collect::<Vec<_>>(),
            &NEUTRON
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let p = dir.join(format!("n{i}.endf"));
                    std::fs::write(&p, text(b)).expect("write");
                    p.to_string_lossy().into_owned()
                })
                .collect::<Vec<_>>(),
            &[],
            "",
            reactions,
            None,
            None,
            &dir.join("out"),
            &provenance(),
        )
        .expect("converts");
        let (chain, _) = yani::parse_chain_parts(
            &dir.join("out/decay"),
            Some(&dir.join("out/reactions")),
            Some(&dir.join("out/fission_yields")),
            None,
        )
        .expect("reads");
        let n = chain.values().map(|c| c.reactions.len()).sum();
        let _ = std::fs::remove_dir_all(&dir);
        n
    };

    let short: Vec<String> = endf::chain::DEFAULT_REACTIONS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let with_short = count(Some(&short), "short");
    let with_default = count(None, "default");
    assert!(
        with_default > with_short,
        "the default followed {with_default} reactions and the six-name set \
         followed {with_short}; the default is the short set again"
    );
    let _ = (decay, fpy, neutron);
}

/// Only the reactions subsection, with no fission yields to give.
///
/// The case a real workflow needs and the API refused until now: a chain
/// assembled from more than one library takes its reaction topology from the
/// neutron library and its decay data from elsewhere, so the caller has no
/// fission yields at all. Requiring them turned "I want one subsection" into an
/// error, and the FNS example works around it by passing an empty list.
#[test]
fn a_single_subsection_can_be_written_without_the_other_inputs() {
    let dir = std::env::temp_dir().join(format!("yani-subsec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let write = |names: &[&[u8]], stem: &str| -> Vec<String> {
        names
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let p = dir.join(format!("{stem}{i}.endf"));
                std::fs::write(&p, text(b)).expect("write");
                p.to_string_lossy().into_owned()
            })
            .collect()
    };
    let decay_files = write(DECAY, "d");
    let neutron_files = write(NEUTRON, "n");
    let out = dir.join("out");

    let n = yani_convert::convert_transmutation_files(
        &decay_files,
        &[],
        &neutron_files,
        &[],
        "",
        None,
        None,
        Some(&["reactions".to_string()]),
        &out,
        &provenance(),
    )
    .expect("a reactions-only conversion with no fission yields");
    assert!(n >= 10, "only {n} nuclides");

    assert!(out.join("reactions/reactions.arrow").is_file());
    assert!(
        !out.join("decay").exists(),
        "decay was written when only reactions was asked for"
    );
    assert!(
        !out.join("fission_yields").exists(),
        "fission_yields was written with no fission yield inputs"
    );

    // The manifest must advertise only what is there. A manifest naming a
    // subsection whose directory is absent is how a consumer ends up reading
    // nothing and being told everything is fine.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    let listed: Vec<&str> = manifest["subsections"]
        .as_object()
        .expect("subsections")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(listed, vec!["reactions"], "manifest lists {listed:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Streaming the neutron files writes exactly what holding them all wrote.
///
/// [`yani_convert::convert_transmutation_files`] reads each neutron evaluation,
/// takes its channels' Q values and drops it, so that a sublibrary far larger
/// than memory can be converted: TENDL's 2848 files parse to about 39 GB held
/// all at once, which was killed three times on a 45 GB machine (issue #53).
/// That is only a safe trade if the result is unchanged, so this drives the
/// same fixtures down both routes and compares the trees byte for byte.
#[test]
fn streaming_the_neutron_files_writes_the_same_tree_as_holding_them() {
    let dir = std::env::temp_dir().join(format!("yani-convert-stream-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");

    let write = |blobs: &[&[u8]], stem: &str| -> Vec<String> {
        blobs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let p = dir.join(format!("{stem}{i}.endf"));
                std::fs::write(&p, text(b)).expect("write fixture");
                p.to_string_lossy().into_owned()
            })
            .collect()
    };
    let decay_files = write(DECAY, "d");
    let fpy_files = write(FPY, "f");
    let neutron_files = write(NEUTRON, "n");

    // Named explicitly rather than left to default, because the two entry
    // points default differently: the files route takes every reaction the
    // chain builder knows, and the in-memory route takes what it is given.
    let reactions: Vec<String> = endf::chain::DEFAULT_REACTIONS
        .iter()
        .map(|r| r.to_string())
        .collect();
    let subsections: Vec<String> = ["decay", "reactions", "fission_yields"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let streamed = dir.join("streamed");
    yani_convert::convert_transmutation_files(
        &decay_files,
        &fpy_files,
        &neutron_files,
        &[],
        "",
        Some(&reactions),
        None,
        Some(&subsections),
        &streamed,
        &provenance(),
    )
    .expect("streamed conversion succeeds");

    let held = dir.join("held");
    let decay = materials(DECAY);
    let fpy = materials(FPY);
    let neutron = materials(NEUTRON);
    let q_values = endf::chain::q_values(&neutron);
    yani_convert::convert_transmutation(
        &yani_convert::Inputs {
            decay: &decay,
            fpy: &fpy,
            q_values: &q_values,
            decay_fill: &[],
            decay_fill_library: "",
        },
        &endf::chain::DEFAULT_REACTIONS,
        None,
        &["decay", "reactions", "fission_yields"],
        &held,
        &provenance(),
    )
    .expect("in-memory conversion succeeds");

    let files = |root: &std::path::Path| -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).expect("read output directory") {
                let path = entry.expect("directory entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let rel = path.strip_prefix(root).expect("under root");
                    out.push(rel.to_string_lossy().into_owned());
                }
            }
        }
        out.sort();
        out
    };

    let written = files(&streamed);
    assert_eq!(
        written,
        files(&held),
        "the two routes wrote different sets of files"
    );
    assert!(
        written.len() >= 6,
        "only {} files written, too few to prove anything: {written:?}",
        written.len()
    );
    for rel in &written {
        let a = std::fs::read(streamed.join(rel)).expect("streamed file");
        let b = std::fs::read(held.join(rel)).expect("held file");
        assert_eq!(a, b, "{rel} differs between the streamed and held routes");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Extracting evaluations one at a time and absorbing them in file order gives
/// exactly what adding them one at a time gives.
///
/// That equivalence is what lets `convert_branching_files` parse in parallel.
/// The rows, the flagged levels and the partial-sum lines are all ordered, and
/// the counters are sums, so a merge that lost the order or forgot a statistic
/// would change the written subsection without changing anything else.
#[test]
fn absorbing_partials_in_file_order_matches_adding_one_at_a_time() {
    use yani_convert::branching::{BranchingExtractor, DEFAULT_LINEARIZE_TOL};

    let decay = materials(DECAY);
    let neutron = materials(NEUTRON);
    let build = || BranchingExtractor::new(&decay, 3000.0, DEFAULT_LINEARIZE_TOL);

    let mut sequential = build();
    for material in &neutron {
        sequential.add(material);
    }

    // What the parallel driver does: every evaluation worked out against the
    // same isomer table, then merged in the order the files were read.
    let base = build();
    let partials: Vec<_> = neutron.iter().map(|m| base.extract_one(m)).collect();
    let mut merged = base;
    for partial in partials {
        merged.absorb(partial);
    }

    let (rows_one_at_a_time, stats_one_at_a_time) = sequential.finish();
    let (rows_merged, stats_merged) = merged.finish();
    assert!(
        !rows_one_at_a_time.is_empty(),
        "the fixtures produced no branching rows, so this proves nothing"
    );
    assert_eq!(rows_merged, rows_one_at_a_time);
    assert_eq!(stats_merged, stats_one_at_a_time);
}
