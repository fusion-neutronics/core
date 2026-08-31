//! Reading the compressed fixtures, for the unit tests.
//!
//! The fixtures in `tests/` are stored xz-compressed: an evaluation is highly
//! repetitive and compresses about six to one, and the ACE table alone is
//! 1.8 MB uncompressed. They are embedded with `include_bytes!` as before and
//! decompressed on first use, so a test still needs no working directory.
//!
//! Compiled only for tests, so the decompressor is a dev-dependency and no
//! consumer of this crate pays for it.

/// Decompress an embedded fixture to the text the readers take.
pub fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut { compressed }, &mut out).expect("fixture is valid xz");
    String::from_utf8(out).expect("fixture is valid UTF-8")
}

/// The ACE tables of an embedded fixture.
pub fn ace_tables(compressed: &[u8]) -> Vec<crate::ace::Table> {
    crate::ace::tables_from_str(&text(compressed), None).expect("fixture parses")
}

/// The Li6 ACE table, which several modules test against.
pub const LI6_ACE: &[u8] = include_bytes!("../fixtures/Li6.ace.xz");
