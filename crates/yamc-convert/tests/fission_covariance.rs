//! `nubar_covariance.arrow` and `spectrum_covariance.arrow`: MF=31 and MF=35
//! written by the converter and read back by the loader, block for block.

use std::path::Path;

use endf::Material;
use yamc_nuclide::arrow::covariance_arrow::{read_nubar_covariance, read_spectrum_covariance};
use yamc_nuclide::covariance::CovarianceData;

/// ENDF/B-VIII.1 Ac225's MF=1, MF=31 and MF=35 only. MF=31 MT=452 is an NC
/// block (the sum of 455 and 456), MT=455 an LB=1 block and MT=456 an LB=5
/// block; MF=35 MT=18 has four LB=7 blocks, one per incident energy range.
const AC225_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-089_Ac_225_mf31_mf35.endf.xz");
/// In115 has neither, which is what "absent" is tested against.
const IN115_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-049_In-115_trimmed.endf.xz");

fn material(compressed: &[u8], dir: &Path, name: &str) -> Material {
    let path = dir.join(format!("{name}.endf"));
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&path, raw).expect("fixture writes");
    Material::from_file(&path).expect("evaluation parses")
}

#[test]
fn ac225_nubar_covariance_round_trips_through_the_section() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(AC225_ENDF, tmp.path(), "Ac225");

    let written = yamc_convert::nubar_covariance::write_nubar_covariance(&material, tmp.path())
        .expect("nubar covariance writes");
    assert!(written, "Ac225 carries MF=31, so a file must be written");
    let blocks = read_nubar_covariance(tmp.path(), "Ac225")
        .expect("reads")
        .expect("the file is there");

    // Every block the parser holds, in tape order (NC before NI within a
    // subsection), and nothing else.
    let mut expected = Vec::new();
    for mt in [452, 455, 456] {
        let mf31 = material.mf31(mt).expect("the fixture carries this MT");
        for (s, sub) in mf31.subsections.iter().enumerate() {
            let nc = sub.nc_subsections.iter().cloned().map(CovarianceData::Nc);
            let ni = sub.ni_subsections.iter().cloned().map(CovarianceData::Ni);
            for (b, data) in nc.chain(ni).enumerate() {
                expected.push((mt, s, b, sub, data));
            }
        }
    }
    assert_eq!(expected.len(), 3);
    assert_eq!(blocks.len(), expected.len());

    for (got, (mt, s, b, sub, data)) in blocks.iter().zip(expected) {
        assert_eq!(
            (got.mt, got.subsection_idx, got.block_idx),
            (mt, s as i32, b as i32)
        );
        assert_eq!((got.mat1, got.mt1), (sub.mat1 as i32, sub.mt1 as i32));
        assert_eq!((got.xmf1, got.xlfs1), (sub.xmf1, sub.xlfs1));
        assert_eq!((got.mtl, got.mat), (0, material.mat));
        assert_eq!(got.data, data, "block ({mt}, {s}, {b})");
    }
}

#[test]
fn ac225_spectrum_covariance_round_trips_through_the_section() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(AC225_ENDF, tmp.path(), "Ac225");
    let mf35 = material.mf35(18).expect("the fixture carries MF=35 MT=18");

    let written =
        yamc_convert::spectrum_covariance::write_spectrum_covariance(&material, tmp.path())
            .expect("spectrum covariance writes");
    assert!(written, "Ac225 carries MF=35, so a file must be written");
    let blocks = read_spectrum_covariance(tmp.path())
        .expect("reads")
        .expect("the file is there");

    assert_eq!(blocks.len(), 4);
    assert_eq!(blocks.len(), mf35.blocks.len());
    for (k, (got, want)) in blocks.iter().zip(&mf35.blocks).enumerate() {
        assert_eq!((got.mt, got.block_idx), (18, k as i32));
        assert_eq!((got.e1, got.e2), (want.e1, want.e2));
        assert_eq!(
            (got.ls, got.lb, got.ne),
            (want.ls as i32, want.lb as i32, want.ne as i32)
        );
        assert_eq!(got.ek, want.ek);
        assert_eq!(got.fkk, want.fkk);
        // LB=7: NE boundaries, and the upper triangle of the (NE - 1) square
        // matrix of the bins between them.
        let bins = got.ek.len() - 1;
        assert_eq!(got.ek.len(), got.ne as usize);
        assert_eq!(got.fkk.len(), bins * (bins + 1) / 2);
    }
}

#[test]
fn an_evaluation_without_mf31_or_mf35_writes_no_file_and_reads_as_none() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(IN115_ENDF, tmp.path(), "In115");
    assert!(
        !yamc_convert::nubar_covariance::write_nubar_covariance(&material, tmp.path())
            .expect("writes nothing without error")
    );
    assert!(
        !yamc_convert::spectrum_covariance::write_spectrum_covariance(&material, tmp.path())
            .expect("writes nothing without error")
    );
    assert!(!tmp.path().join("nubar_covariance.arrow").exists());
    assert!(!tmp.path().join("spectrum_covariance.arrow").exists());
    assert_eq!(
        read_nubar_covariance(tmp.path(), "In115").expect("reads"),
        None
    );
    assert_eq!(read_spectrum_covariance(tmp.path()).expect("reads"), None);
}
