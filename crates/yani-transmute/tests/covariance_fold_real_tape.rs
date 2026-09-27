//! The MF=33 fold on a real evaluation, at an energy where every block is flat.
//!
//! ENDF/B-VIII.1 Cr52 states its (n,p) covariance in LB=0, LB=1 and LB=8
//! blocks that run to 20 MeV. A narrow group at 14.1 MeV lies inside one
//! interval of every one of them, so the folded relative variance of the rate
//! is the tape's own diagonal there, whatever the cross section does inside the
//! group. That makes this a check on the whole path from the tape to the fold
//! rather than on the fold's arithmetic alone: the old LB=0 to 4 split cut the
//! two LB=1 tables off at 3.3 and 8 MeV, and the same fold gave 0.4%, the LB=8
//! block on its own.
//!
//! The fixture is built the way `data_uncertainty.rs` builds Fe56: the cached
//! Cr52 directory is copied and the real converter writes `covariance.arrow`
//! into it from the committed trimmed evaluation.
//!
//! Self-skips when the nuclear-data fixtures are missing, the way every other
//! test that needs them does.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use yamc_materials::Material;
use yani_transmute::compute_multigroup_reaction_rates;
use yani_transmute::covariance_fold::fold_rate_covariance;

const CR52_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-024_Cr_052_trimmed.endfb81.endf.xz");

/// One group around 14.1 MeV, inside [14, 16] MeV, which is an interval of
/// every Cr52 (n,p) block.
const GROUPS: [f64; 2] = [14.0e6, 14.2e6];
const FLUX: [f64; 1] = [1.0];

fn chain() -> HashMap<String, yani::ChainNuclide> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    yani::parse_chain_arrow(&path).expect("parse chain")
}

/// A copy of the cached Cr52 directory with `covariance.arrow` written into it.
fn cr52_with_covariance(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("Cr52")?);
    let dir = tmp.join("Cr52.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached Cr52") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() && entry.file_name() != "covariance.arrow" {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }

    let evaluation = tmp.join("cr52.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &CR52_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let material = endf::Material::from_file(&evaluation).expect("Cr52 parses");
    assert!(
        yamc_convert::covariance::write_covariance(&material, &dir).expect("covariance writes"),
        "the fixture must carry MF=33 for this test to mean anything"
    );
    Some(dir)
}

fn chromium(data: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Cr52".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Cr52 material");
    m.density = Some(7.19);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Cr52".to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Cr52");
    // A plain read leaves MF=33 out; `transmute` widens it the same way when
    // an uncertainty is asked for.
    m.ensure_covariance_loaded().expect("read Cr52 covariance");
    m
}

#[test]
fn cr52_np_folds_to_the_tapes_sigma_at_14_mev() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = cr52_with_covariance(tmp.path()) else {
        eprintln!(
            "skipping: cr52_np_folds_to_the_tapes_sigma_at_14_mev (nuclear-data fixtures missing)"
        );
        return;
    };
    let material = chromium(&dir);
    let chain = chain();

    let (rates, _) = compute_multigroup_reaction_rates(&material, &chain, &FLUX, &GROUPS, 1.0);
    let (folded, coverage) = fold_rate_covariance(&material, &chain, &rates, &FLUX, &GROUPS);
    assert_eq!(coverage.malformed, 0, "every block expands");
    assert!(coverage.unsupported_layouts.is_empty());

    let cov = folded.get("Cr52").expect("Cr52 carries MF=33 (n,p)");
    let i = cov
        .kinds
        .iter()
        .position(|k| k == "(n,p)")
        .expect("(n,p) is in the chain and folded");
    assert_eq!(
        coverage
            .rate_fraction_covered
            .get(&("Cr52".to_string(), "(n,p)".to_string())),
        Some(&1.0),
        "the group is inside every block's grid"
    );

    // The tape's relative components on the interval holding 14.1 MeV: LB=1
    // on [4, 20] MeV, LB=1 on [14, 16] MeV and LB=8 on [14, 16] MeV. The
    // LB=0 block is absolute, 1.1e-14 b^2 on [4, 20] MeV, and relativizes by
    // the group's own cross section; it is eleven orders of magnitude smaller.
    let sigma_eff = rates["Cr52"]["(n,p)"] / 1.0e-24;
    let want = 1.125e-2 + 1.8e-2 + 1.5842e-5 + 1.1e-14 / (sigma_eff * sigma_eff);
    let got = cov.get(i, i);
    assert!(
        (got - want).abs() <= 1e-9 * want,
        "Cr52 (n,p) at 14.1 MeV folds to {got} (sigma {:.2}%); the tape states {want} (sigma {:.2}%)",
        100.0 * got.max(0.0).sqrt(),
        100.0 * want.sqrt(),
    );
}
