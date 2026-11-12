//! A bare name that is both a library keyword and a directory is refused.
//!
//! With a folder called `endf-b8.1` in the working directory,
//! `neutron_cross_section_data = "endf-b8.1"` could mean either, and reading one of
//! them would be a guess. The test changes the working directory, which is
//! process-global, so it is the only test in this binary.

use yamc_nuclide::url_cache::{
    resolve_data_path, resolve_path_or_url, resolve_subsection, DataKind,
};
use yamc_nuclide::LoadScope;

#[test]
fn a_keyword_that_names_a_directory_too_is_refused() {
    let dir = std::env::temp_dir().join(format!("yamc-shadowed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("endf-b8.1/decay")).unwrap();
    let before = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();

    let scope = LoadScope::full();
    let refusals = [
        resolve_subsection("endf-b8.1", "decay").map(|_| ()),
        resolve_path_or_url("endf-b8.1", "Fe56", DataKind::Neutron, &scope).map(|_| ()),
        resolve_data_path("endf-b8.1", Some("Fe"), DataKind::Photon, &scope).map(|_| ()),
    ];
    // Written as a path, the directory is read.
    let as_path = resolve_subsection("./endf-b8.1", "decay");

    std::env::set_current_dir(&before).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    for refusal in refusals {
        let err = refusal.expect_err("an ambiguous source").to_string();
        assert!(
            err.contains("'endf-b8.1' is both a library keyword and the directory"),
            "{err}"
        );
        assert!(err.contains("\"./endf-b8.1\""), "{err}");
    }
    assert_eq!(
        as_path.unwrap(),
        std::path::Path::new("./endf-b8.1").join("decay")
    );
}
