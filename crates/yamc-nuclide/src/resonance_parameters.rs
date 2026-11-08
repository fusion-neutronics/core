//! MF=2 and MF=32 resonance data, as it comes off `resonance_parameters.arrow`.
//!
//! The file keeps each section's ENDF-6 text verbatim, and this keeps it the
//! same way: parsing is left to `endf`'s own MF=2 and MF=32 readers, through
//! [`ResonanceParameters::mf2`] and [`ResonanceParameters::mf32`], so what a
//! consumer samples from is exactly what the converter read off the tape.
//! Nothing here interprets a formalism, and nothing on the load path parses
//! the text: a nuclide carries it until something asks.

use endf::mf::mf2::Mf2;
use endf::mf::mf32::Mf32;
use endf::resonance_covariance::RangeCovariance;
use endf::Reader;

/// One evaluation's MF=2 MT=151 and MF=32 MT=151 sections, as ENDF-6 text.
///
/// Each is the section's records, one newline-terminated line apiece with its
/// control columns, up to but not including the SEND record: what
/// [`endf::Material::section_text`] holds, and what the parsers read.
#[derive(Debug, Clone, PartialEq)]
pub struct ResonanceParameters {
    /// MF=2 MT=151, the resonance parameters.
    pub mf2_text: String,
    /// MF=32 MT=151, their covariance.
    pub mf32_text: String,
}

impl ResonanceParameters {
    /// Parse the MF=2 text into `endf`'s resonance parameters.
    pub fn mf2(&self) -> endf::Result<Mf2> {
        endf::mf::mf2::parse_mf2(&mut Reader::new(&self.mf2_text))
    }

    /// Parse the MF=32 text into `endf`'s resonance parameter covariance.
    pub fn mf32(&self) -> endf::Result<Mf32> {
        endf::mf::mf32::parse_mf32(&mut Reader::new(&self.mf32_text))
    }

    /// Both sections parsed, MF=2 then MF=32.
    pub fn parse(&self) -> endf::Result<(Mf2, Mf32)> {
        Ok((self.mf2()?, self.mf32()?))
    }

    /// Each MF=32 range's parameter covariance, matched to its MF=2 range:
    /// [`endf::resonance_covariance::range_covariances`] on the parsed
    /// sections.
    pub fn range_covariances(&self) -> endf::Result<Vec<RangeCovariance>> {
        let (mf2, mf32) = self.parse()?;
        endf::resonance_covariance::range_covariances(&mf2, &mf32)
    }
}
