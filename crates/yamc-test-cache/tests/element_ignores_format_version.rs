//! A photon element stamped with a superseded `format_version` is still a
//! fixture; a nuclide stamped the same way is not.
//!
//! `nuclide` refuses a directory whose `format_version` the nuclide loader
//! would refuse, so a stale entry skips instead of panicking
//! (`stale_format_version.rs`). `element` must NOT inherit that: the element
//! loader applies no such gate, because the format-2 change (energy grids into
//! `energy.arrow`, `reactions.arrow` split per (MT, temperature)) touched only
//! the neutron layout. The converter stamps every directory it writes, so a
//! pre-republish element is stamped 1 and reads exactly as one stamped 2 does.
//! Routing elements through the nuclide check reported the W the photon Z
//! sweep had just run on as absent.
//!
//! One test in its own binary because it points the cache root at a temporary
//! directory through `YAMC_CACHE_DIR`, which `cache_root` re-reads on every
//! call, and a process-wide variable is not something to share with parallel
//! tests.

use std::path::Path;

fn stamp(dir: &Path, format_version: i64) {
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(
        dir.join("version.json"),
        format!(
            "{{\"data_version\": \"2026-09-08\", \"format_version\": {format_version}, \
             \"library\": \"endfb-8.1\"}}"
        ),
    )
    .expect("write version.json");
}

#[test]
fn a_superseded_element_stamp_is_still_present_and_a_nuclide_one_is_not() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::env::set_var("YAMC_CACHE_DIR", tmp.path());

    let lib = yamc_test_cache::LIBRARY;
    stamp(&tmp.path().join(format!("{lib}-W.arrow")), 1);
    stamp(&tmp.path().join(format!("{lib}-W184.arrow")), 1);

    assert!(
        yamc_test_cache::element("W").is_some(),
        "a format-1 element directory reads today, so it must still be reported"
    );
    assert!(
        yamc_test_cache::nuclide("W184").is_none(),
        "a format-1 nuclide directory is refused by the loader, so it must not be"
    );
    // And the two agree on the current stamp, so this is about the gate and not
    // about the element path resolving somewhere else.
    stamp(
        &tmp.path().join(format!("{lib}-Fe.arrow")),
        yamc_nuclide::arrow::nuclide_arrow::FORMAT_VERSION,
    );
    stamp(
        &tmp.path().join(format!("{lib}-Fe56.arrow")),
        yamc_nuclide::arrow::nuclide_arrow::FORMAT_VERSION,
    );
    assert!(yamc_test_cache::element("Fe").is_some());
    assert!(yamc_test_cache::nuclide("Fe56").is_some());
}
