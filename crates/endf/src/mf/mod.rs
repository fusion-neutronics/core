//! One module per ENDF file (MF), each turning a section's records into a
//! typed struct.
//!
//! # Porting the rest
//!
//! The Python package covers MF 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15,
//! 23, 26, 27, 28, 33, 34 and 40. [`mf3`] is the worked example: a module
//! exposes a struct and a `parse_mfN(&mut Reader) -> Result<MfN>` function,
//! gains a variant in [`crate::material::Section`], and an arm in that module's
//! dispatch. Sections without a Rust parser stay
//! [`crate::material::Section::Unparsed`] with their text intact, so the two
//! readers can run side by side while the port is checked file by file.

pub mod atomic;
pub mod covariance;
pub mod mf1;
pub mod mf2;
pub mod mf3;
pub mod mf4;
pub mod mf5;
pub mod mf6;
pub mod mf7;
pub mod mf8;
pub mod photon;
