//! `resonance_parameters.arrow`: MF=2 MT=151 and MF=32 MT=151, the resonance
//! parameters and their covariance, as the evaluation's own ENDF-6 text.
//!
//! One row per section, holding every record of it as it is on the tape. The
//! reader hands that text straight back to `endf`'s MF=2 and MF=32 parsers,
//! so the stored form is lossless by construction: whatever formalism or
//! covariance form the evaluator used, the consumer sees the same parameters
//! the converter did, with nothing transcribed in between.
//!
//! Only an evaluation with MF=32 MT=151 gets the file. MF=2 alone is already
//! reconstructed into the pointwise cross sections, and the section exists so
//! the parameters can be sampled from their covariance, which needs both.

use std::error::Error;
use std::path::Path;

use endf::Material;

use crate::sections::{ints, strings, write_section};

/// The sections the file holds, in the order they are written.
const SECTIONS: [(i32, i32); 2] = [(2, 151), (32, 151)];

/// Write `resonance_parameters.arrow`, one row per section: MF=2 MT=151 and
/// MF=32 MT=151.
///
/// Returns whether a file was written: an evaluation with no MF=32 MT=151
/// writes nothing, and a reader takes the missing file as "no resonance
/// parameter covariance". An evaluation with MF=32 and no MF=2 is refused,
/// since MF=32 is a covariance of MF=2's parameters and means nothing alone.
///
/// No formalism is filtered out here: the parameters are stored whatever
/// they are, and what can be sampled is the reader's decision.
pub fn write_resonance_parameters(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    if !material.section_text.contains_key(&(32, 151)) {
        return Ok(false);
    }
    let mut mf = Vec::new();
    let mut mt = Vec::new();
    let mut text = Vec::new();
    for (section_mf, section_mt) in SECTIONS {
        let Some(body) = material.section_text.get(&(section_mf, section_mt)) else {
            return Err(format!(
                "MAT {} has MF=32 MT=151 but no MF={section_mf} MT={section_mt}; \
                 a resonance parameter covariance needs the parameters it is of",
                material.mat
            )
            .into());
        };
        mf.push(section_mf);
        mt.push(section_mt);
        text.push(body.clone());
    }
    write_section(
        &dir.join("resonance_parameters.arrow"),
        "resonance_parameters.arrow",
        vec![ints(&mf), ints(&mt), strings(&text)],
    )?;
    Ok(true)
}
