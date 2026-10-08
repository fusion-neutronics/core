//! `angular_covariance.arrow`: MF=34 written by the converter and read back
//! by the loader, block for block.

use std::path::Path;

use endf::mf::covariance::split_mf34_block;
use endf::Material;
use yamc_nuclide::arrow::covariance_arrow::read_angular_covariance;

/// ENDF/B-VIII.1 U235, trimmed. It carries MF=34 for MT=51: NL=2, so three
/// (L, L1) pairs of its covariance with itself.
const U235_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-092_U_235_trimmed.endf.xz");
/// In115 has no MF=34, which is what "absent" is tested against.
const IN115_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-049_In-115_trimmed.endf.xz");

fn material(compressed: &[u8], dir: &Path, name: &str) -> Material {
    let path = dir.join(format!("{name}.endf"));
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&path, raw).expect("fixture writes");
    Material::from_file(&path).expect("evaluation parses")
}

#[test]
fn u235_angular_covariance_round_trips_through_the_section() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(U235_ENDF, tmp.path(), "U235");
    let mf34 = material.mf34(51).expect("the fixture carries MF=34 MT=51");

    let written = yamc_convert::angular_covariance::write_angular_covariance(&material, tmp.path())
        .expect("angular covariance writes");
    assert!(written, "U235 carries MF=34, so a file must be written");
    let blocks = read_angular_covariance(tmp.path())
        .expect("reads")
        .expect("the file is there");

    // Every block the parser holds, in tape order, and nothing else.
    let mut expected = Vec::new();
    for (s, sub) in mf34.subsections.iter().enumerate() {
        for (p, pair) in sub.subsubsections.iter().enumerate() {
            for (b, values) in pair.data.iter().enumerate() {
                expected.push((s, p, b, sub, pair, values));
            }
        }
    }
    assert!(!expected.is_empty());
    assert_eq!(blocks.len(), expected.len());

    for (got, (s, p, b, sub, pair, values)) in blocks.iter().zip(expected) {
        assert_eq!(
            (got.mt, got.subsection_idx, got.pair_idx, got.block_idx),
            (51, s as i32, p as i32, b as i32)
        );
        assert_eq!((got.mat1, got.mt1), (sub.mat1 as i32, sub.mt1 as i32));
        assert_eq!((got.nl, got.nl1), (sub.nl as i32, sub.nl1 as i32));
        assert_eq!((got.l, got.l1), (sub.l[p] as i32, sub.l1[p] as i32));
        assert_eq!(got.lct, pair.lct as i32);
        assert_eq!(got.ltt, mf34.ltt as i32);
        assert_eq!(got.mat, material.mat);
        let want = split_mf34_block(
            pair.ls[b] as i64,
            pair.lb[b] as i64,
            pair.nt[b] as i64,
            pair.ne[b] as i64,
            values,
        )
        .expect("the block splits");
        assert_eq!(got.block, want, "block ({s}, {p}, {b})");
    }
}

#[test]
fn an_evaluation_without_mf34_writes_no_file_and_reads_as_none() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(IN115_ENDF, tmp.path(), "In115");
    let written = yamc_convert::angular_covariance::write_angular_covariance(&material, tmp.path())
        .expect("writes nothing without error");
    assert!(!written);
    assert!(!tmp.path().join("angular_covariance.arrow").exists());
    assert_eq!(read_angular_covariance(tmp.path()).expect("reads"), None);
}
