//! `nubar_covariance.arrow`: MF=31, the covariance of the fission neutron
//! multiplicities, straight off the tape.
//!
//! MF=31 is MF=33's format (ENDF-102 chapter 31), so its blocks are written
//! through the same [`CovarianceRows`] as `covariance.arrow`, one row per NC or
//! NI sub-subsection of one subsection, with the evaluation's own MAT last.
//! The MTs are multiplicities: 452 total, 455 delayed and 456 prompt.
//!
//! A section of its own rather than rows of `covariance.arrow`: every row there
//! is read as a cross-section covariance, and an MF=31 MT=452 block would be
//! folded as one.

use std::error::Error;
use std::path::Path;

use endf::Material;

use crate::covariance::CovarianceRows;
use crate::sections::{opt_ints, write_section};

/// Write `nubar_covariance.arrow`, one row per MF=31 covariance block.
///
/// Returns whether a file was written: an evaluation with no MF=31 block
/// writes nothing, and a reader takes the missing file as "no multiplicity
/// covariance", as it does `covariance.arrow`.
///
/// A section with a nonzero MTL is refused. ENDF-102 gives MF=31 the field
/// but no tape uses it, and a lumped component has no subsections, so writing
/// its blocks would silently drop the one statement it makes.
pub fn write_nubar_covariance(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let mut rows = CovarianceRows::default();
    let mts: Vec<i32> = material
        .sections()
        .into_iter()
        .filter(|(mf, _)| *mf == 31)
        .map(|(_, mt)| mt)
        .collect();
    for mt in mts {
        let Some(mf31) = material.mf31(mt) else {
            continue;
        };
        if mf31.mtl != 0 {
            return Err(format!(
                "MAT {} MF=31 MT={mt} names lumped reaction MT={}; lumped MF=31 \
                 sections are not supported",
                material.mat, mf31.mtl
            )
            .into());
        }
        for (subsection_idx, sub) in mf31.subsections.iter().enumerate() {
            rows.push_subsection(mt, subsection_idx, Some(mf31.mtl), sub);
        }
    }
    if rows.is_empty() {
        return Ok(false);
    }
    let mut columns = rows.columns();
    columns.push(opt_ints(&vec![Some(material.mat); rows.len()]));
    write_section(
        &dir.join("nubar_covariance.arrow"),
        "nubar_covariance.arrow",
        columns,
    )?;
    Ok(true)
}
