//! The covariance fold under the `1/E` within-group weight.
//!
//! Its own test binary because the weight is process-wide: set in a binary
//! whose other tests collapse rates in parallel, it would change their
//! answers under them.
//!
//! The fixture is the one `data_uncertainty.rs` builds: cached Fe56 with the
//! committed evaluation's MF=33 written into it. Its `(n,p)` covariance has an
//! edge at 4.3 MeV, inside the fast group, which is the case the weight
//! affects. Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::multigroup::{set_within_group_weight, Weighting};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const FE56_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");

const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];

fn fe56_with_covariance(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("Fe56")?);
    let dir = tmp.join("Fe56.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached Fe56") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }
    let evaluation = tmp.join("fe56.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &FE56_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let material = endf::Material::from_file(&evaluation).expect("Fe56 parses");
    assert!(
        yamc_convert::covariance::write_covariance(&material, &dir).expect("covariance writes"),
        "the fixture must carry MF=33 for this test to mean anything"
    );
    Some(dir)
}

fn fold_info(data: &Path) -> yani_transmute::uncertainty::Info {
    let mut material = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    material.density = Some(7.87);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("Fe56".to_string(), data.to_string_lossy().into_owned())]),
            None,
        )
        .expect("read Fe56");
    let chain = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let total: f64 = FLUX.iter().sum();
    let results = transmute_material(
        &mut material,
        &[MultigroupSpectrum {
            boundaries: GROUPS.to_vec(),
            masses: FLUX.iter().map(|f| f / total).collect(),
            flux_error: None,
        }],
        &[TransmuteStep {
            dt: 3600.0,
            irradiation: Some((0, total)),
        }],
        Arc::new(yani::parse_chain_arrow(&chain).expect("parse chain")),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(8),
            sources: vec![Source::CrossSections],
            ..Default::default()
        }),
    )
    .expect("transmute");
    results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported")
}

/// Under the flat-within-group weight the fold's partials and the collapsed
/// rate are one integral, so nothing is listed. Under the `1/E` weight they
/// are not: a partial over the part of the fast group above the 4.3 MeV edge
/// is that part's lethargy average times its share of the group's energy
/// width, while the collapse lethargy-averages the whole group, which a
/// threshold reaction fills only at its top. The partials then sum to about
/// ten times the rate, so the relative sigma is overstated by that much, and
/// the check must say so rather than let a dilute run read as consistent.
/// When the fold weights its partials the way the collapse does, this is the
/// assertion to turn around.
#[test]
fn a_one_over_e_fold_with_an_edge_inside_a_group_is_reported() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let key = ("Fe56".to_string(), "(n,p)".to_string());

    let flat = fold_info(&dir);
    assert!(
        flat.partials_above_rate.is_empty(),
        "{:?}",
        flat.partials_above_rate
    );

    set_within_group_weight(Weighting::OneOverE);
    let one_over_e = fold_info(&dir);
    set_within_group_weight(Weighting::FlatInEnergy);

    let ratio = one_over_e.partials_above_rate[&key];
    assert!(ratio > 5.0, "{ratio}");
    assert!(one_over_e.has_gaps());
    let share = one_over_e.rate_fraction_covered[&key];
    assert!((0.0..=1.0).contains(&share), "{share}");
}
