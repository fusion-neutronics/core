//! What makes a converted `.arrow` directory complete, and how to say so.
//!
//! `version.json` is the completion marker: the converters put it down as the
//! very last thing they do, so its presence means the conversion ran all the
//! way through, and a resume reads it back to decide what to skip.
//!
//! The marker alone would be enough for anything written since it started
//! being written last. The table check below is what catches the directories
//! an older ordering already left on disk: the writers used to put
//! `version.json` down FIRST, so every directory an interrupted run left
//! behind looked finished to the skip check. A rebuild killed partway through
//! U238 left two of its eight tables there, and a plain re-run skipped it and
//! called the library complete.
//!
//! The rule came from the retired Python converter's `completion.py`, which
//! was where both halves lived so they could not disagree. It is a test rather
//! than a function because the resume loop that consumes it lives in
//! `nuclear_data_generation_scripts`, not here; what has to stay true on this
//! side is that a conversion which finished satisfies the rule.
//!
//! One correction the port had to make: the Python rule had a single neutron
//! set, `(nuclide, fast_xs, reactions)`, from the days when a neutron
//! conversion meant one thing. The Rust converter has two neutron scopes and
//! `fast_xs.arrow` belongs only to the wider one, so conflating them would
//! call every cross-section-only directory incomplete.

use std::path::{Path, PathBuf};

/// The completion marker, written last.
const MARKER: &str = "version.json";

/// Tables every complete cross-section-only neutron directory has.
///
/// `urr.arrow` is not here: it is carried when the evaluation has resonance
/// data and absent otherwise, so its absence proves nothing. Neither is
/// `covariance.arrow`, which is written only when it was asked for.
const NEUTRON_XS_REQUIRED_TABLES: &[&str] = &["nuclide.arrow", "reactions.arrow"];

/// Tables every complete transport neutron directory has.
///
/// The cross-section set plus what a transport run cannot start without. The
/// fissile-only sections (`total_nu`, `fission_photon`) are excluded for the
/// same reason `urr` is.
const NEUTRON_TRANSPORT_REQUIRED_TABLES: &[&str] = &[
    "nuclide.arrow",
    "reactions.arrow",
    "products.arrow",
    "distributions.arrow",
    "fast_xs.arrow",
];

/// Tables every complete photon directory has.
///
/// `subshells`, `compton` and `bremsstrahlung` are written only when the
/// evaluation carries them, so they are not proof of anything either way.
const PHOTON_REQUIRED_TABLES: &[&str] = &["element.arrow"];

/// Is *dir* a finished conversion, safe for a resume to skip?
fn is_complete(dir: &Path, required_tables: &[&str]) -> bool {
    dir.join(MARKER).is_file()
        && required_tables
            .iter()
            .all(|table| dir.join(table).is_file())
}

/// Which of *required_tables* are missing, for a failure message worth reading.
fn missing(dir: &Path, required_tables: &[&str]) -> Vec<String> {
    std::iter::once(MARKER)
        .chain(required_tables.iter().copied())
        .filter(|f| !dir.join(f).is_file())
        .map(str::to_string)
        .collect()
}

fn fixture(name: &str, into: &Path) -> PathBuf {
    let compressed: &[u8] = match name {
        "Li6.ace" => include_bytes!("../../endf/fixtures/Li6.ace.xz"),
        "photoat-001_H_000.endf" => include_bytes!("../../endf/fixtures/photoat-001_H_000.endf.xz"),
        "atom-001_H_000.endf" => include_bytes!("../../endf/fixtures/atom-001_H_000.endf.xz"),
        other => panic!("no such fixture: {other}"),
    };
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    let path = into.join(name);
    std::fs::write(&path, out).expect("fixture is written");
    path
}

/// A cross-section conversion that ran to the end satisfies the rule.
#[test]
fn a_finished_neutron_xs_conversion_is_complete() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ace = fixture("Li6.ace", tmp.path());

    let dir = yamc_convert::entry::convert_neutron_xs(
        &yamc_convert::entry::Source::Ace { path: &ace },
        tmp.path(),
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the ACE route converts");

    assert!(
        is_complete(&dir, NEUTRON_XS_REQUIRED_TABLES),
        "a conversion that returned Ok is not complete; missing {:?}",
        missing(&dir, NEUTRON_XS_REQUIRED_TABLES)
    );

    // The scopes are not interchangeable, which is the correction the port had
    // to make to the Python rule. Holding a cross-section directory to the
    // transport set would reject every one of them.
    assert!(
        !dir.join("fast_xs.arrow").is_file(),
        "convert_neutron_xs wrote fast_xs.arrow, so the two neutron scopes now \
         have the same required set and this test's split is stale"
    );
}

/// A photon conversion that ran to the end satisfies the rule.
#[test]
fn a_finished_photon_conversion_is_complete() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let photoatomic = fixture("photoat-001_H_000.endf", tmp.path());
    let relaxation = fixture("atom-001_H_000.endf", tmp.path());

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).expect("mkdir");
    let written = yamc_convert::entry::convert_photon(
        &photoatomic,
        Some(&relaxation),
        None,
        &out,
        &yamc_convert::entry::Provenance::default(),
    )
    .expect("the photon route converts");

    assert!(!written.is_empty(), "the photon route wrote no directory");
    for dir in &written {
        assert!(
            is_complete(dir, PHOTON_REQUIRED_TABLES),
            "{} returned Ok but is not complete; missing {:?}",
            dir.display(),
            missing(dir, PHOTON_REQUIRED_TABLES)
        );
    }
}

/// The half the marker cannot do on its own.
///
/// This is the case the rule exists for: the marker is present and the tables
/// are not. A skip check that trusts the marker alone reads such a directory
/// as finished, which is how a partial rebuild got published.
#[test]
fn a_marked_directory_missing_a_required_table_is_not_complete() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ace = fixture("Li6.ace", tmp.path());

    let dir = yamc_convert::entry::convert_neutron_xs(
        &yamc_convert::entry::Source::Ace { path: &ace },
        tmp.path(),
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the ACE route converts");

    std::fs::remove_file(dir.join("reactions.arrow")).expect("the table is removed");

    assert!(
        dir.join(MARKER).is_file(),
        "the marker has to survive for this to be the case under test"
    );
    assert!(
        !is_complete(&dir, NEUTRON_XS_REQUIRED_TABLES),
        "a directory with the marker but no reactions.arrow was called complete"
    );
}

/// The transport scope's wider set, when this machine can produce one.
///
/// Needs NJOY and a real evaluation: the transport route refuses ACE, since an
/// ACE table carries no MT 901 heating and no MF=1/MT=458. Announces a loud
/// skip rather than passing quietly, as the other NJOY-gated tests here do.
#[test]
fn a_finished_neutron_transport_conversion_is_complete() {
    let evaluation =
        yamc_test_cache::endf_evaluations().join("neutrons-version.VIII.1/n-003_Li_006.endf");
    if !evaluation.is_file() {
        eprintln!("SKIP: no ENDF/B-VIII.1 Li6 evaluation; this test checked nothing");
        return;
    }
    if std::process::Command::new("njoy")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP: no njoy on PATH; this test checked nothing");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = yamc_convert::entry::convert_neutron_transport(
        &yamc_convert::entry::Source::Endf {
            path: &evaluation,
            njoy_exec: "njoy",
            temperatures: vec![294.0],
        },
        tmp.path(),
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the ENDF route converts");

    assert!(
        is_complete(&dir, NEUTRON_TRANSPORT_REQUIRED_TABLES),
        "a transport conversion that returned Ok is not complete; missing {:?}",
        missing(&dir, NEUTRON_TRANSPORT_REQUIRED_TABLES)
    );
}
