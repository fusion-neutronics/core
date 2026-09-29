//! Fixtures shared by more than one test binary.

use std::path::{Path, PathBuf};

const FE56_ENDF: &[u8] = include_bytes!("../../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");

/// A copy of the cached Fe56 directory with `covariance.arrow` written into it
/// from the committed ENDF evaluation, or `None` when the nuclear-data
/// fixtures are missing.
pub fn fe56_with_covariance(tmp: &Path) -> Option<PathBuf> {
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
