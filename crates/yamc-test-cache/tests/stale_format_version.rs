//! A cache directory stamped with an unreadable `format_version` must not look
//! present.
//!
//! `nuclide` answers "can this machine run that case", and a directory the
//! loader refuses cannot. The refusal is real: the 2026-09-18 republish moved
//! the libraries to format 2, `read_nuclide_from_arrow` rejects a format-1
//! directory outright ("Data published before the energy grids moved to
//! energy.arrow"), and any entry the fetch script does not refresh stays at
//! format 1. A test written to skip on an absent nuclide then panics instead.
//! `gpu_mixed_fissile_yield` and `gpu_survival_fissile_scheme` both did on a
//! developer box carrying a pre-republish U235.
//!
//! Same shape as the activation-scope case in `transport_scope.rs`: the
//! directory is there, so a bare `is_dir` says yes, and the loader is the one
//! that finds out.

use std::path::Path;

fn write_marker(dir: &Path, format_version: &str) {
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(
        dir.join("version.json"),
        format!(
            "{{\n  \"data_version\": \"2026-09-18\",\n  \"format_version\": {format_version},\n  \
             \"library\": \"endfb-8.1\"\n}}\n"
        ),
    )
    .expect("write version.json");
}

#[test]
fn the_current_format_version_reads() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("current");
    write_marker(
        &dir,
        &yamc_nuclide::arrow::nuclide_arrow::FORMAT_VERSION.to_string(),
    );
    assert!(
        yamc_test_cache::format_version_is_readable(&dir),
        "a directory stamped with the build's own format_version must be readable"
    );
}

#[test]
fn a_superseded_format_version_does_not() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("v1");
    write_marker(&dir, "1");
    assert!(
        !yamc_test_cache::format_version_is_readable(&dir),
        "format 1 is the pre-republish layout the loader refuses, so it must not \
         read as present"
    );
}

/// No marker, an unparseable one, and one without the field all read as
/// current. The loader only applies its check when `version.json` is there and
/// says something, so inventing a stricter rule here would skip directories it
/// would have loaded.
#[test]
fn an_absent_or_unreadable_marker_reads_as_current() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let bare = tmp.path().join("bare");
    std::fs::create_dir_all(&bare).expect("mkdir");
    assert!(yamc_test_cache::format_version_is_readable(&bare));

    let broken = tmp.path().join("broken");
    std::fs::create_dir_all(&broken).expect("mkdir");
    std::fs::write(broken.join("version.json"), "{not json").expect("write");
    assert!(yamc_test_cache::format_version_is_readable(&broken));

    let fieldless = tmp.path().join("fieldless");
    std::fs::create_dir_all(&fieldless).expect("mkdir");
    std::fs::write(
        fieldless.join("version.json"),
        "{\"library\": \"endfb-8.1\"}",
    )
    .expect("write");
    assert!(yamc_test_cache::format_version_is_readable(&fieldless));
}
