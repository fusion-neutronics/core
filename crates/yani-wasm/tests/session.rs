//! Drive the wasm API exactly as JS would, natively.
//!
//! `wasm-bindgen` methods are ordinary Rust methods when the crate is built for
//! the host, so the assertions can live here rather than behind a headless
//! browser. This is the pattern `crates/yamc/tests/wasm_simulation.rs` uses,
//! and it is what catches a boundary that compiles but does nothing.
//!
//! These tests share process-global state -- `set_storage` and the nuclear-data
//! config are both global -- so they must not run concurrently. Building a
//! session per test is not enough: last-writer-wins applies across the whole
//! process, so a test that loads its data and then transmutes can have the
//! storage replaced in between and read zero. They are serialised on
//! `exclusive()` below rather than relying on `--test-threads=1`, which no
//! caller is obliged to pass -- neither `cargo test --workspace` nor the
//! coverage run does.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use yani_wasm::YaniSession;

/// Serialises every test in this binary.
///
/// `YaniSession::new` installs a *process-global* storage backend (see the
/// crate docs: "a second `YaniSession` takes the first one's data with it").
/// Cargo runs a test binary's tests as threads in one process, so two tests
/// that each build a session and then transmute will clobber each other's
/// storage: the loser reads an empty backend, finds no cross sections, and
/// reports zero activity rather than failing. With two such tests in this file
/// that is not a rare race -- it was every run, on every platform that got
/// there first.
///
/// Every test takes it, including the ones that touch no session, so the rule
/// is "every test in this file starts with this line" rather than a judgement
/// about which globals a test reaches.
fn exclusive() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    // A failing test poisons the lock. The next test installs its own storage
    // regardless, so recovering keeps one failure from being reported as
    // several.
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn chain_fixture() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    dir.join("decay/nuclides.arrow").exists().then_some(dir)
}

fn nuclide_fixture(nuclide: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests")
        .join(format!("{nuclide}.arrow"));
    dir.join("nuclide.arrow").exists().then_some(dir)
}

/// A session with the fixture chain loaded, as a host would after fetching.
fn session_with_chain() -> Option<(YaniSession, usize)> {
    let root = chain_fixture()?;
    let mut session = YaniSession::new();
    for (subsection, file) in [
        ("decay", "nuclides.arrow"),
        ("decay", "decay_modes.arrow"),
        ("decay", "sources.arrow"),
        ("reactions", "reactions.arrow"),
        ("fission_yields", "fission_yields.arrow"),
        ("fission_yields", "aliases.arrow"),
    ] {
        let path = root.join(subsection).join(file);
        if path.exists() {
            session
                .add_chain_section(subsection, file, std::fs::read(&path).unwrap())
                .unwrap();
        }
    }
    let n = session.load_chain().expect("chain loads");
    Some((session, n))
}

#[test]
fn a_chain_loads_from_bytes_the_host_supplied() {
    let _exclusive = exclusive();
    let Some((_session, n)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    assert!(n > 100, "fixture chain should be substantial, got {n}");
}

#[test]
fn loading_a_chain_before_its_sections_is_an_error() {
    let _exclusive = exclusive();
    let mut session = YaniSession::new();
    let err = session
        .load_chain()
        .expect_err("an empty chain must not load");
    assert!(
        err.contains("decay/nuclides.arrow"),
        "error should name the missing section, got: {err}",
    );
}

#[test]
fn an_unknown_subsection_is_rejected() {
    let _exclusive = exclusive();
    let mut session = YaniSession::new();
    let err = session
        .add_chain_section("decays", "nuclides.arrow", Vec::new())
        .expect_err("'decays' is not a subsection");
    assert!(err.contains("decays"), "got: {err}");
}

#[test]
fn the_pnnl_compendium_needs_no_data() {
    let _exclusive = exclusive();
    // Compiled in, so it works before a single byte is fetched -- which is what
    // lets the UI offer a material picker on first paint.
    let names: Vec<String> = serde_json::from_str(&YaniSession::pnnl_names()).unwrap();
    assert!(
        names.iter().any(|n| n == "Steel, Stainless 316"),
        "expected SS316",
    );
    assert!(
        names.len() > 300,
        "expected the full compendium, got {}",
        names.len()
    );

    // The report's own names carry commas, which is why these come back as
    // JSON: joining them on one would split "Steel, Stainless 316" in two.
    assert!(
        names.iter().any(|n| n.contains(',')),
        "a joined-string API would be lossy here",
    );

    let hits: Vec<String> = serde_json::from_str(&YaniSession::pnnl_search("stainless")).unwrap();
    assert!(!hits.is_empty(), "search should find stainless steels");
    assert!(hits.iter().all(|n| n.to_lowercase().contains("stainless")));
}

#[test]
fn a_material_can_be_built_from_a_composition() {
    let _exclusive = exclusive();
    let mut session = YaniSession::new();
    session
        .build_material(r#"{"Fe": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .expect("iron builds");
    let json: serde_json::Value = serde_json::from_str(&session.material_json().unwrap()).unwrap();
    let densities = json["atoms_per_barn_cm"].as_object().unwrap();
    // Natural iron expands over its four stable isotopes.
    for isotope in ["Fe54", "Fe56", "Fe57", "Fe58"] {
        assert!(densities.contains_key(isotope), "missing {isotope}");
    }
    assert_eq!(json["volume"].as_f64(), Some(1000.0));
}

/// A nuclide name means that nuclide, not a formula that happens to parse.
///
/// The formula parser accepts "Fe56" and reads it as Fe with a subscript of
/// 56, which expands over natural abundance: the material comes back as iron,
/// 8.8% short on Fe56 and carrying Fe54, Fe57 and Fe58 that were never asked
/// for. It is silent, and the browser's front end had it for as long as the
/// binding did -- caught by running the same setup through this API and the
/// Python one and finding a constant 8.8% between them.
#[test]
fn a_nuclide_name_is_not_read_as_a_formula() {
    let _exclusive = exclusive();
    let mut session = YaniSession::new();
    session
        .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .expect("Fe56 builds");
    let json: serde_json::Value = serde_json::from_str(&session.material_json().unwrap()).unwrap();
    let densities = json["atoms_per_barn_cm"].as_object().unwrap();

    assert_eq!(
        densities.keys().collect::<Vec<_>>(),
        vec!["Fe56"],
        "a single nuclide must stay a single nuclide",
    );
    // rho * N_A / M for Fe56 (55.934936 g/mol), which natural iron misses by
    // the 8.4% of its atoms that are not Fe56.
    let fe56 = densities["Fe56"].as_f64().unwrap();
    assert!(
        (fe56 - 0.0847741).abs() < 1e-6,
        "expected the Fe56 atom density, got {fe56}",
    );

    // Formulas and elements still reach the parsers that should have them.
    session
        .build_material(r#"{"H2O": 1.0}"#, 1.0, "g/cm3", "atom", 1.0)
        .expect("H2O builds");
    let water: serde_json::Value = serde_json::from_str(&session.material_json().unwrap()).unwrap();
    let water = water["atoms_per_barn_cm"].as_object().unwrap();
    assert!(
        water.contains_key("O16"),
        "H2O should expand, got {water:?}"
    );
    assert!(
        water.keys().any(|k| k.starts_with("H")),
        "H2O should expand to hydrogen too, got {water:?}",
    );

    session
        .build_material(r#"{"Fe": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .expect("Fe builds");
    let iron: serde_json::Value = serde_json::from_str(&session.material_json().unwrap()).unwrap();
    let iron = iron["atoms_per_barn_cm"].as_object().unwrap();
    assert!(
        iron.contains_key("Fe54") && iron.contains_key("Fe56"),
        "an element still expands over natural abundance, got {iron:?}",
    );
}

#[test]
fn the_download_budget_is_knowable_before_fetching() {
    let _exclusive = exclusive();
    let Some((mut session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    session
        .build_material(r#"{"Fe": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .unwrap();

    let names: Vec<String> =
        serde_json::from_str(&session.required_nuclides().expect("budget")).unwrap();
    assert!(
        names.len() > 50,
        "an iron chain reaches far more than its own isotopes, got {}",
        names.len()
    );
    assert!(
        names.iter().any(|n| n == "Fe56"),
        "the seed itself must be in the set",
    );
    // No cross sections have been supplied yet, which is the point: the host
    // learns what to fetch before fetching any of it.
    assert_eq!(session.file_count(), 0);
}

#[test]
fn the_mts_a_run_reads_are_knowable_before_fetching() {
    let _exclusive = exclusive();
    let Some((session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };

    let mts: Vec<i32> = serde_json::from_str(&session.required_mts().expect("mts")).unwrap();

    // The host turns this into byte ranges of each nuclide's reactions.arrow,
    // so a missing MT is not a 404, it is a reaction rate silently reading
    // zero. Capture and (n,2n) are in every chain worth the name.
    assert!(mts.contains(&102), "(n,gamma) must be named, got {mts:?}");
    assert!(mts.contains(&16), "(n,2n) must be named, got {mts:?}");

    // Sorted and unique: the host may cache or diff it, and it comes out of a
    // HashSet upstream.
    let mut sorted = mts.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(mts, sorted, "must be sorted with no repeats");

    // Only the activation channels. The full-grid transport MTs are the bulk of
    // every reactions.arrow and naming one here would fetch it for every
    // nuclide, which is the entire cost this set exists to avoid.
    for transport_only in [1, 2, 3, 4, 27, 101, 301, 444] {
        assert!(
            !mts.contains(&transport_only),
            "MT {transport_only} is a transport lookup, not a chain reaction",
        );
    }

    // Knowable with no cross sections in hand, same as the nuclide budget.
    assert_eq!(session.file_count(), 0);
}

/// The chain has to be in before the MTs can be read off it, and saying so is
/// better than answering with an empty set that fetches nothing.
#[test]
fn the_mt_set_needs_a_chain_first() {
    let _exclusive = exclusive();
    let err = YaniSession::new().required_mts().unwrap_err();
    assert!(err.contains("loadChain"), "unhelpful error: {err}");
}

/// The browser's byte-range path, end to end and natively.
///
/// The page fetches only the MTs `requiredMts` names out of each
/// `reactions.arrow`, using the byte ranges `version.json` publishes, and
/// splices them into an Arrow IPC *stream* before handing them over. That is a
/// different framing from the published file, so this asserts the thing that
/// actually matters to a user: the spliced stream transmutes to the same
/// numbers the whole file does.
///
/// Fed through `add_nuclide_data`, which is the same entry point the page uses,
/// so nothing here is a shortcut around the boundary being tested.
///
/// Compared to a relative tolerance rather than bit-for-bit. The two loads
/// build reaction maps of different sizes -- the whole file carries every MT
/// and the splice carries the handful the chain names -- so the collapse sums
/// them in a different order and the totals land up to an ulp apart. That is
/// summation order, not different cross sections; the batches themselves are
/// asserted byte-identical in yamc-convert's `reaction_ranges` tests.
#[test]
fn a_spliced_stream_transmutes_to_the_same_answer_as_the_whole_file() {
    let _exclusive = exclusive();
    let Some(fe56) = nuclide_fixture("Fe56") else {
        eprintln!("skipping: no Fe56 cross-section fixture");
        return;
    };
    let Some((session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };

    let reactions = fe56.join("reactions.arrow");
    let bytes = std::fs::read(&reactions).unwrap();
    let index = yamc_convert::reaction_ranges::index_reactions(&reactions).unwrap();
    let cut = |(off, len): (u64, u64)| bytes[off as usize..(off + len) as usize].to_vec();

    // The set the page asks the solver for, intersected with what this nuclide
    // publishes: an MT the chain names but Fe56 has no channel for simply has
    // no batch to fetch.
    let wanted: Vec<i32> = serde_json::from_str::<Vec<i32>>(&session.required_mts().unwrap())
        .unwrap()
        .into_iter()
        .filter(|mt| index.mts.contains_key(mt))
        .collect();
    assert!(!wanted.is_empty(), "chain should name MTs Fe56 carries");

    // File order, not MT order: the loader fuses the batches it reads, so the
    // order they arrive in is the row order of the fused batch.
    let mut ranges: Vec<(u64, u64)> = wanted.iter().map(|mt| index.mts[mt]).collect();
    ranges.sort_unstable();
    let spliced = yamc_convert::reaction_ranges::splice_stream(
        &cut(index.schema),
        &ranges.into_iter().map(cut).collect::<Vec<_>>(),
    );
    assert!(
        spliced.len() * 2 < bytes.len(),
        "spliced {} of {} bytes; the subset should be far smaller",
        spliced.len(),
        bytes.len(),
    );

    // One year at 1e14 on a three-group spectrum, then a day of cooling. Same
    // as the whole-file transmutation test above, so the two are comparable.
    let spectra = r#"[{"boundaries": [1e-5, 1e5, 1e6, 1.5e7], "values": [1e12, 1e13, 1e14]}]"#;
    let year = 365.25 * 86400.0;
    let schedule =
        format!(r#"[{{"dt": {year}, "rate": 1.11e14, "spectrum": 0}}, {{"dt": 86400.0}}]"#);

    let run = |reactions_bytes: Vec<u8>| -> serde_json::Value {
        let mut session = session_with_chain().expect("chain").0;
        session
            .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
            .unwrap();
        let nuclide = std::fs::read(fe56.join("nuclide.arrow")).unwrap();
        session.add_nuclide_data("Fe56", "nuclide.arrow", nuclide);
        session.add_nuclide_data("Fe56", "reactions.arrow", reactions_bytes);
        serde_json::from_str(&session.run(spectra, &schedule).expect("transmutation runs")).unwrap()
    };
    let activity = |v: &serde_json::Value| v[0]["activity"].as_f64().unwrap();

    let from_whole = run(bytes.clone());
    let from_spliced = run(spliced);

    assert!(
        activity(&from_whole) > 0.0,
        "the whole-file run must activate, or this compares two zeroes",
    );
    let (a, b) = (activity(&from_spliced), activity(&from_whole));
    assert!(
        (a - b).abs() <= 1e-12 * b.abs(),
        "spliced activity {a} against whole-file {b}",
    );
    assert_eq!(
        from_spliced[0]["activity_by_nuclide"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        from_whole[0]["activity_by_nuclide"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        "the two runs should produce the same inventory",
    );

    // Negative control, and the reason the assertions above can be trusted.
    // `get_or_load_nuclide` caches parsed nuclides process-globally by path, and
    // both runs above hand their bytes over at the same virtual path, so an
    // agreement could in principle be the second run being served the first
    // one's parse rather than reading the splice at all. Starving the splice of
    // capture must move the answer; if it does not, nothing here is being read.
    let starved =
        yamc_convert::reaction_ranges::splice_stream(&cut(index.schema), &[cut(index.mts[&16])]);
    let c = activity(&run(starved));
    assert!(
        (c - b).abs() > 1e-6 * b.abs(),
        "dropping (n,gamma) left the activity at {c} against {b}, so the spliced \
         bytes are not reaching the solver",
    );
}

#[test]
fn a_transmutation_runs_and_reports_every_series_the_plots_need() {
    let _exclusive = exclusive();
    let Some((mut session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let Some(fe56) = nuclide_fixture("Fe56") else {
        eprintln!("skipping: no Fe56 cross-section fixture");
        return;
    };

    session
        .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .unwrap();

    for file in ["version.json", "nuclide.arrow", "reactions.arrow"] {
        let path = fe56.join(file);
        if path.exists() {
            session.add_nuclide_data("Fe56", file, std::fs::read(&path).unwrap());
        }
    }
    assert!(
        session.file_count() >= 2,
        "cross-section files should be held"
    );

    // One year at 1e14 on a three-group spectrum, then a day of cooling.
    let spectra = r#"[{"boundaries": [1e-5, 1e5, 1e6, 1.5e7], "values": [1e12, 1e13, 1e14]}]"#;
    let year = 365.25 * 86400.0;
    let schedule =
        format!(r#"[{{"dt": {year}, "rate": 1.11e14, "spectrum": 0}}, {{"dt": 86400.0}}]"#);

    let out = session.run(spectra, &schedule).expect("transmutation runs");
    let steps: Vec<serde_json::Value> = serde_json::from_str(&out).unwrap();
    assert_eq!(steps.len(), 2, "one result per schedule step");

    assert_eq!(steps[0]["irradiated"].as_bool(), Some(true));
    assert_eq!(steps[1]["irradiated"].as_bool(), Some(false));
    assert!(
        (steps[1]["time"].as_f64().unwrap() - (year + 86400.0)).abs() < 1.0,
        "times should be cumulative",
    );

    // Irradiating Fe56 must actually activate it. A zero here is the failure
    // mode the docs warn about: unconfigured data reads as zero rather than
    // erroring, so an empty result looks like a successful run.
    let activity = steps[0]["activity"].as_f64().unwrap();
    let heat = steps[0]["decay_heat"].as_f64().unwrap();
    assert!(activity > 0.0, "expected activity, got {activity}");
    assert!(heat > 0.0, "expected decay heat, got {heat}");

    // Mn56 is the (n,p) product and the dominant short-lived activity.
    let by_nuclide = steps[0]["activity_by_nuclide"].as_object().unwrap();
    assert!(
        by_nuclide.contains_key("Mn56") || by_nuclide.contains_key("Fe55"),
        "expected an activation product, got {:?}",
        by_nuclide.keys().collect::<Vec<_>>(),
    );

    // The totals must be the sum of their own breakdowns, or the two plots the
    // UI draws from them disagree.
    let summed: f64 = by_nuclide.values().map(|v| v.as_f64().unwrap()).sum();
    assert!(
        (summed - activity).abs() <= activity * 1e-9,
        "total {activity} != sum of by-nuclide {summed}",
    );

    // Decay photon lines come back as paired arrays for the stem plot.
    let energy = steps[1]["photon_energy"].as_array().unwrap();
    let intensity = steps[1]["photon_intensity"].as_array().unwrap();
    assert_eq!(energy.len(), intensity.len(), "lines must be paired");

    // Contact dose, in Gy/h, from the same inventory. It needs no volume, so
    // the one thing that could silently zero it is the material carrying no
    // photon emitter -- which irradiated iron certainly does.
    let dose = steps[0]["contact_dose"].as_f64().unwrap();
    assert!(dose > 0.0, "expected a contact dose, got {dose}");
    let dose_by_nuclide = steps[0]["contact_dose_by_nuclide"].as_object().unwrap();
    let summed: f64 = dose_by_nuclide.values().map(|v| v.as_f64().unwrap()).sum();
    assert!(
        (summed - dose).abs() <= dose * 1e-9,
        "total {dose} != sum of by-nuclide {summed}",
    );

    // Only the photon emitters contribute, so the dose breakdown is a subset of
    // the activity breakdown rather than a second list of everything.
    for nuclide in dose_by_nuclide.keys() {
        assert!(
            by_nuclide.contains_key(nuclide),
            "{nuclide} carries dose but no activity",
        );
    }
}

/// The same run, repeated on one session, must give the same answer.
///
/// It did not. The temperature a nuclide was read at came from
/// `energy.keys().next()`, an arbitrary key that differs per parse because
/// every `HashMap` is seeded separately. ENDF/B-8.1 Fe56 carries a `0` grid
/// that no reaction set backs, so drawing it left the nuclide with no reaction
/// rates: the irradiation activated nothing and reported zero, for unchanged
/// input, about one run in seven. A single run passes seven times in eight,
/// which is why this is a loop.
#[test]
fn repeated_runs_of_one_session_agree() {
    let _exclusive = exclusive();
    let Some((mut session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let Some(fe56) = nuclide_fixture("Fe56") else {
        eprintln!("skipping: no Fe56 cross-section fixture");
        return;
    };
    session
        .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .unwrap();
    for file in ["version.json", "nuclide.arrow", "reactions.arrow"] {
        let path = fe56.join(file);
        if path.exists() {
            session.add_nuclide_data("Fe56", file, std::fs::read(&path).unwrap());
        }
    }

    let spectra = r#"[{"boundaries": [1e-5, 1e5, 1e6, 1.5e7], "values": [1e12, 1e13, 1e14]}]"#;
    let schedule = r#"[{"dt": 31557600.0, "rate": 1.11e14, "spectrum": 0}]"#;

    let mut activities = Vec::new();
    for _ in 0..10 {
        let out = session.run(spectra, schedule).expect("transmutation runs");
        let steps: Vec<serde_json::Value> = serde_json::from_str(&out).unwrap();
        activities.push(steps[0]["activity"].as_f64().unwrap());
    }

    let first = activities[0];
    assert!(first > 0.0, "expected activity, got {activities:?}");
    for (i, a) in activities.iter().enumerate() {
        // Not bit-equality: the rates are summed in parallel, so the last digit
        // or two moves between runs. Anything larger is a different answer.
        assert!(
            (a - first).abs() <= first * 1e-6,
            "run {i} gave {a}, run 0 gave {first}",
        );
    }
}

#[test]
fn a_schedule_naming_a_missing_spectrum_is_rejected() {
    let _exclusive = exclusive();
    let Some((mut session, _)) = session_with_chain() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    session
        .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .unwrap();
    let err = session
        .run("[]", r#"[{"dt": 3600.0, "rate": 1e14, "spectrum": 0}]"#)
        .expect_err("spectrum 0 does not exist");
    assert!(err.contains("out of range"), "got: {err}");
}

#[test]
fn durations_use_the_same_unit_spellings_as_the_wheel() {
    let _exclusive = exclusive();
    assert_eq!(yani_wasm::duration_to_seconds(1.0, "h"), Ok(3600.0));
    assert_eq!(yani_wasm::duration_to_seconds(1.0, "d"), Ok(86400.0));
    assert_eq!(
        yani_wasm::duration_to_seconds(1.0, "a"),
        Ok(365.25 * 86400.0)
    );
    assert!(yani_wasm::duration_to_seconds(1.0, "fortnight").is_err());
}

#[test]
fn named_group_structures_come_from_the_wheels_own_tables() {
    let _exclusive = exclusive();
    let session = YaniSession::new();
    // n+1 boundaries for n groups, which is what the names promise and what a
    // pasted flux is checked against.
    // Every name the registry lists, so a structure added there cannot reach the
    // browser without this passing.
    let registry = yamc_nuclide::group_structures::available_group_structures();
    assert_eq!(registry.len(), 12);
    for name in registry {
        let groups: usize = name
            .rsplit('-')
            .find_map(|part| part.parse().ok())
            .unwrap_or_else(|| panic!("{name} carries no group count"));
        let edges: Vec<f64> =
            serde_json::from_str(&session.group_structure(name).unwrap()).unwrap();
        assert_eq!(edges.len(), groups + 1, "{name} boundary count");
        assert!(
            edges.windows(2).all(|w| w[1] > w[0]),
            "{name} must be ascending",
        );
    }
    // A typo must name the alternatives rather than silently becoming one.
    assert!(session.group_structure("CCFE-708").is_err());
}
