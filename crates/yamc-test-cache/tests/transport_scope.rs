//! A cache directory left at activation scope must not look transport-ready.
//!
//! `nuclide` answers "is the fixture here", which a directory holding
//! `reactions.arrow` and none of the products satisfies. The loader does not
//! object to it either: `narrow_to_present_sections` narrows a `Full` request
//! to the sections on disk rather than refusing it, so reading the directory
//! succeeds and returns a nuclide with no secondary distributions. A test that
//! then samples an `(n,2n)` panics with "Missing product distributions"
//! instead of skipping, which is how a windows CI runner whose restored
//! fixture cache carried an activation-scope Ar38 failed
//! `parity_ar38_collision0_spectrum` while every other runner skipped it.
//!
//! So `transport_ready` reads the directory and looks at the scope that came
//! back, and this builds both shapes out of one fixture to pin the difference.

use std::path::Path;

/// The sections a conversion writes that an activation load does not need.
const TRANSPORT_ONLY: [&str; 3] = ["products.arrow", "distributions.arrow", "fast_xs.arrow"];

/// Copy `src` into a new directory, dropping the names in `skip`.
fn copy_without(src: &Path, dst: &Path, skip: &[&str]) {
    std::fs::create_dir_all(dst).expect("mkdir");
    for entry in std::fs::read_dir(src).expect("read fixture") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if skip.contains(&name.as_str()) {
            continue;
        }
        let from = entry.path();
        if from.is_dir() {
            continue;
        }
        std::fs::copy(&from, dst.join(&name)).expect("copy section");
    }
}

#[test]
fn an_activation_scope_directory_is_not_transport_ready() {
    // Any fixture with the full transport set. Fe56 is in the CI fixture list.
    let Some(full) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("SKIP: Fe56 is not in the test cache; this test checked nothing");
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");

    let whole = tmp.path().join("whole");
    copy_without(Path::new(&full), &whole, &[]);
    assert!(
        yamc_test_cache::transport_ready(&whole),
        "a complete copy of the fixture must read as transport-ready, or this test is \
         measuring the copy rather than the scope"
    );

    let narrowed = tmp.path().join("activation");
    copy_without(Path::new(&full), &narrowed, &TRANSPORT_ONLY);
    // The loader still reads it, which is the whole problem: absence of the
    // transport sections narrows the request instead of failing it.
    assert!(
        yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow(
            &narrowed,
            &yamc_nuclide::LoadScope::full(),
        )
        .is_ok(),
        "the loader is expected to narrow a Full request to what is on disk; if it now \
         refuses instead, `transport_ready` can go back to asking whether the read succeeded"
    );
    assert!(
        !yamc_test_cache::transport_ready(&narrowed),
        "a directory with no products or distributions must not read as transport-ready"
    );
}
