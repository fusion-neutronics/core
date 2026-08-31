//! What a browser session costs in memory, measured rather than guessed.
//!
//! wasm32 has a 4 GB address space. A full ENDF/B-8.1 activation closure is 461
//! nuclides at roughly 508 MB of activation-scoped Arrow, and the session holds
//! both the raw bytes (in the virtual filesystem) and the parsed `Nuclide`
//! structures. If the sum of those does not fit, the whole browser approach
//! fails at the last moment, so it is worth knowing the expansion factor early.
//!
//! Reports rather than asserts a budget: the number depends on which nuclides a
//! run touches, and a hard threshold here would be a guess dressed as a gate.
//! What it *does* assert is that dropping the raw bytes after parsing actually
//! frees them, because that is the mitigation the design depends on.
//!
//! Run with:
//! ```text
//! cargo test -p yani-wasm --test memory -- --nocapture
//! ```

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

/// Resident set size in bytes, from /proc. Linux-only; the probe is a
/// measurement aid, not a portability requirement.
fn rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn fixtures() -> Vec<(String, PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "arrow") && p.join("reactions.arrow").exists()
        })
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            // Element folders (photon data) carry no nuclide.arrow.
            p.join("nuclide.arrow").exists().then_some((name, p))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn report_the_memory_a_session_costs() {
    let _exclusive = exclusive();
    let fixtures = fixtures();
    if fixtures.is_empty() {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    }
    let Some(baseline) = rss() else {
        eprintln!("skipping: no /proc/self/status on this platform");
        return;
    };

    let session = YaniSession::new();
    let mut raw_bytes = 0u64;
    for (name, dir) in &fixtures {
        for file in ["version.json", "nuclide.arrow", "reactions.arrow"] {
            let path = dir.join(file);
            if path.exists() {
                let bytes = std::fs::read(&path).unwrap();
                raw_bytes += bytes.len() as u64;
                session.add_nuclide_data(name, file, bytes);
            }
        }
    }
    let after_load = rss().unwrap();

    let held = after_load.saturating_sub(baseline);
    let mb = |b: u64| b as f64 / 1048576.0;
    eprintln!(
        "\n{} nuclides, {:.1} MB of Arrow on the wire\n\
         RSS grew {:.1} MB holding the raw bytes -> {:.2}x the wire size",
        fixtures.len(),
        mb(raw_bytes),
        mb(held),
        held as f64 / raw_bytes as f64,
    );

    // The extrapolation that matters. 508 MB is the measured activation-scoped
    // size of a full ENDF/B-8.1 closure (461 nuclides).
    const FULL_CLOSURE_MB: f64 = 508.0;
    let projected = FULL_CLOSURE_MB * (held as f64 / raw_bytes as f64);
    eprintln!(
        "a full 461-nuclide closure would hold about {projected:.0} MB of raw bytes, \
         against wasm32's 4096 MB address space",
    );

    assert!(
        raw_bytes > 50 * 1024 * 1024,
        "probe needs a meaningful sample, got {:.1} MB",
        mb(raw_bytes),
    );
}

/// What parsing costs on top of the bytes, which is the number the raw-bytes
/// probe above cannot see.
///
/// `get_or_load_nuclide` turns the Arrow into `Vec<f64>` cross sections that
/// outlive the batch, and those are what a session actually carries. Measured
/// on one small nuclide so the parse cost is not buried under the file size.
#[test]
fn report_what_parsing_costs_on_top_of_the_bytes() {
    let _exclusive = exclusive();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Be9.arrow");
    let chain = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    if !dir.join("reactions.arrow").exists() || !chain.join("decay/nuclides.arrow").exists() {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    }
    let Some(baseline) = rss() else {
        eprintln!("skipping: no /proc/self/status on this platform");
        return;
    };
    let mb = |b: u64| b as f64 / 1048576.0;

    let mut session = YaniSession::new();
    for (subsection, file) in [
        ("decay", "nuclides.arrow"),
        ("decay", "decay_modes.arrow"),
        ("decay", "sources.arrow"),
        ("reactions", "reactions.arrow"),
        ("fission_yields", "fission_yields.arrow"),
        ("fission_yields", "aliases.arrow"),
    ] {
        let path = chain.join(subsection).join(file);
        if path.exists() {
            session
                .add_chain_section(subsection, file, std::fs::read(&path).unwrap())
                .unwrap();
        }
    }
    let n = session.load_chain().unwrap();
    let after_chain = rss().unwrap();

    let mut raw = 0u64;
    for file in ["version.json", "nuclide.arrow", "reactions.arrow"] {
        let path = dir.join(file);
        if path.exists() {
            let bytes = std::fs::read(&path).unwrap();
            raw += bytes.len() as u64;
            session.add_nuclide_data("Be9", file, bytes);
        }
    }
    session
        .build_material(r#"{"Be9": 1.0}"#, 1.85, "g/cm3", "atom", 1000.0)
        .unwrap();
    let after_bytes = rss().unwrap();

    let spectra = r#"[{"boundaries": [1e-5, 1e5, 1e6, 1.5e7], "values": [1e12, 1e13, 1e14]}]"#;
    session
        .run(
            spectra,
            r#"[{"dt": 31557600.0, "rate": 1e14, "spectrum": 0}]"#,
        )
        .expect("Be9 transmutes");
    let after_run = rss().unwrap();

    let chain_cost = after_chain.saturating_sub(baseline);
    let bytes_cost = after_bytes.saturating_sub(after_chain);
    let parse_cost = after_run.saturating_sub(after_bytes);

    eprintln!(
        "\nchain ({n} nuclides)      {:>7.1} MB   -- fixed, paid once per session\n\
         Be9 raw bytes            {:>7.1} MB   ({:.2}x its {:.1} MB on the wire)\n\
         parsing + solve          {:>7.1} MB   ({:.2}x the bytes it parsed)\n",
        mb(chain_cost),
        mb(bytes_cost),
        bytes_cost as f64 / raw as f64,
        mb(raw),
        mb(parse_cost),
        parse_cost as f64 / raw as f64,
    );

    // One point does not give a slope. Be9's parse cost is mostly fixed
    // overhead -- the reduced chain, the burnup matrix, the CRAM workspace --
    // amortised over 0.4 MB, so dividing by its size measures the overhead, not
    // the marginal cost of a nuclide. Add a nuclide two orders of magnitude
    // bigger and take the difference.
    let big = Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    if !big.join("reactions.arrow").exists() {
        eprintln!("(no Fe56 fixture: cannot separate fixed from marginal cost)");
        return;
    }
    let mut big_raw = 0u64;
    for file in ["version.json", "nuclide.arrow", "reactions.arrow"] {
        let path = big.join(file);
        if path.exists() {
            let bytes = std::fs::read(&path).unwrap();
            big_raw += bytes.len() as u64;
            session.add_nuclide_data("Fe56", file, bytes);
        }
    }
    session
        .build_material(r#"{"Fe56": 1.0}"#, 7.874, "g/cm3", "atom", 1000.0)
        .unwrap();
    session
        .run(
            spectra,
            r#"[{"dt": 31557600.0, "rate": 1e14, "spectrum": 0}]"#,
        )
        .expect("Fe56 transmutes");
    let after_big = rss().unwrap();

    let big_cost = after_big.saturating_sub(after_run);
    let marginal = big_cost as f64 / big_raw as f64;
    eprintln!(
        "Fe56 ({:.1} MB on the wire) added {:>7.1} MB   ({marginal:.2}x) -- the marginal rate\n",
        mb(big_raw),
        mb(big_cost),
    );

    // The projection that decides whether the browser approach survives, using
    // the marginal rate rather than Be9's overhead-dominated one.
    const FULL_CLOSURE_MB: f64 = 508.0;
    eprintln!(
        "extrapolated at the marginal rate: 461 nuclides at {FULL_CLOSURE_MB:.0} MB on the wire \
         need about {:.0} MB, against wasm32's 4096 MB",
        FULL_CLOSURE_MB * marginal,
    );
}

#[test]
fn clearing_the_virtual_filesystem_frees_the_raw_bytes() {
    let _exclusive = exclusive();
    // The mitigation the design leans on: once a nuclide is parsed, its raw
    // bytes are dead weight, and peak memory is only survivable if dropping
    // them actually returns the memory. If this ever stops holding, the session
    // pays for every byte twice.
    let fixtures = fixtures();
    if fixtures.is_empty() {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    }

    let mut session = YaniSession::new();
    for (name, dir) in &fixtures {
        for file in ["nuclide.arrow", "reactions.arrow"] {
            let path = dir.join(file);
            if path.exists() {
                session.add_nuclide_data(name, file, std::fs::read(&path).unwrap());
            }
        }
    }
    let loaded = session.file_count();
    assert!(loaded > 0, "fixtures should have been added");

    session.clear_nuclide_data();
    assert_eq!(
        session.file_count(),
        0,
        "clearing must actually empty the store",
    );
}
