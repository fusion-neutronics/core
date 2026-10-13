//! Positive-orientation invariant on the exact conforming carve, the third path
//! `mesh_volume` can emit from (see `tet_orientation.rs` for why the invariant
//! matters).
//!
//! The carve is selected by `YAMM_CONFORMING`, which `mesh_volume` reads from
//! the PROCESS environment. Setting it here would steer every other test in the
//! binary, so the test re-runs itself as a child of the test binary with the
//! variable set on the child's environment only.

mod common;

use std::process::Command;

use common::{assert_positively_oriented, cube, cylinder, icosphere};

#[test]
fn conforming_carve_tets_are_positively_oriented() {
    if std::env::var("YAMM_CONFORMING").is_err() {
        let output = Command::new(std::env::current_exe().expect("path of the test binary"))
            .args([
                "conforming_carve_tets_are_positively_oriented",
                "--exact",
                "--nocapture",
            ])
            .env("YAMM_CONFORMING", "1")
            .output()
            .expect("re-run the test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        eprint!("{stdout}");
        assert!(
            output.status.success(),
            "the YAMM_CONFORMING=1 child failed"
        );
        assert!(
            stdout.contains("test result: ok. 1 passed"),
            "the YAMM_CONFORMING=1 child did not run the test"
        );
        return;
    }

    let (v, t) = cube(5.0);
    assert_positively_oriented("conforming cube s=5 tel=1.5", v, t, 1.5);

    let (v, t) = icosphere(5.0, 2);
    assert_positively_oriented("conforming sphere r=5 tel=1.5", v, t, 1.5);

    let (v, t) = cylinder(1.0, 4.0, 24);
    assert_positively_oriented("conforming cylinder r=1 h=4 tel=0.5", v, t, 0.5);
}
