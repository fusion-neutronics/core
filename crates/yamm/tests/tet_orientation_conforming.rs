//! Positive-orientation invariant on the exact conforming carve, the third path
//! `mesh_volume` can emit from (see `tet_orientation.rs` for why the invariant
//! matters).
//!
//! The carve is selected by `YAMM_CONFORMING`, which `mesh_volume` reads from
//! the PROCESS environment. A test that sets it would therefore steer every
//! other test running concurrently in the same binary, so this file holds one
//! test and nothing else: cargo gives each integration-test file its own
//! process, which is the isolation the variable needs.

mod common;

use common::{assert_positively_oriented, cube, cylinder, icosphere};

#[test]
fn conforming_carve_tets_are_positively_oriented() {
    std::env::set_var("YAMM_CONFORMING", "1");

    let (v, t) = cube(5.0);
    assert_positively_oriented("conforming cube s=5 tel=1.5", v, t, 1.5);

    let (v, t) = icosphere(5.0, 2);
    assert_positively_oriented("conforming sphere r=5 tel=1.5", v, t, 1.5);

    let (v, t) = cylinder(1.0, 4.0, 24);
    assert_positively_oriented("conforming cylinder r=1 h=4 tel=0.5", v, t, 0.5);
}
