//! Inspecting a photon directory must not install its grids process-wide.
//!
//! `read_photon_interaction_from_arrow` publishes three process-wide,
//! first-write-wins grids: the Compton profile momentum grid and the two TTB
//! grids. That is right for a simulation, where one library is loaded and every
//! element shares them, and wrong for a caller holding a candidate conversion up
//! for inspection: the first read in the process decides them, so validating a
//! directory would hand its grids to every later element load and every photon
//! collision. A different-length grid then panics out of the Doppler sampler; a
//! same-length one with different values changes every sample and says nothing.
//!
//! `yamc.read_element_from_arrow` (issue #443) is exactly that caller, so it
//! goes through `inspect_photon_interaction_from_arrow` instead.
//!
//! One test, not two: the grids are process-global and first-write-wins, so the
//! order of two tests in one binary would decide the result of both.

use yamc_element::photon::compton_profile_pz;
use yamc_element::photon_arrow::{
    inspect_photon_interaction_from_arrow, read_photon_interaction_from_arrow,
};

#[test]
fn inspecting_leaves_the_shared_grids_to_the_real_load() {
    let Some(dir) = yamc_test_cache::element("Fe") else {
        eprintln!("skip -- no endf-b8.1 Fe fixture; run scripts/fetch_test_fixtures.py");
        return;
    };
    let dir = std::path::Path::new(&dir);

    // Nothing has loaded an element yet in this binary, so the grid is unset.
    assert!(
        compton_profile_pz().is_empty(),
        "a shared grid was already published; this test must run first in its binary"
    );

    // An inspection reads everything and publishes nothing.
    let inspected =
        inspect_photon_interaction_from_arrow(dir).expect("the Fe fixture reads for inspection");
    assert!(
        !inspected.profile_pdf.is_empty(),
        "the inspection read no Compton profiles, so it would not have had a grid to publish \
         and this test proves nothing"
    );
    assert!(
        compton_profile_pz().is_empty(),
        "inspecting a directory published the shared Compton momentum grid"
    );

    // An ordinary load still does publish it, which is what transport needs.
    let loaded = read_photon_interaction_from_arrow(dir).expect("the Fe fixture reads normally");
    assert!(
        !compton_profile_pz().is_empty(),
        "an ordinary load stopped publishing the shared Compton momentum grid"
    );

    // And the two reads produce the same data: suppressing publication changes
    // what the process remembers, not what the caller is handed.
    assert_eq!(inspected.name, loaded.name);
    assert_eq!(inspected.atomic_number, loaded.atomic_number);
    assert_eq!(inspected.energy.len(), loaded.energy.len());
    assert_eq!(inspected.shells.len(), loaded.shells.len());
    assert_eq!(inspected.profile_pdf, loaded.profile_pdf);
    assert_eq!(inspected.profile_cdf, loaded.profile_cdf);
    assert_eq!(inspected.dcs, loaded.dcs);
}
