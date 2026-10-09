//! `spectrum_covariance.arrow`: MF=35, the covariance of secondary energy
//! distributions, straight off the tape.
//!
//! One row per covariance block, which is one LIST record of one MT's section:
//! the LB=7 matrix of one incident energy range, with its outgoing energy bin
//! boundaries and the packed upper triangle kept as the parser holds them.
//! On the tapes the only MT is 18, the prompt fission neutron spectrum.

use std::error::Error;
use std::path::Path;

use endf::Material;

use crate::covariance::narrow;
use crate::sections::{float_lists_or_null, ints, opt_floats, opt_ints, write_section};

/// Write `spectrum_covariance.arrow`, one row per MF=35 covariance block.
///
/// Returns whether a file was written: an evaluation with no MF=35 block
/// writes nothing, and a reader takes the missing file as "no spectrum
/// covariance".
pub fn write_spectrum_covariance(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let mut mt = Vec::new();
    let mut block_idx = Vec::new();
    let mut e1 = Vec::new();
    let mut e2 = Vec::new();
    let mut ls = Vec::new();
    let mut lb = Vec::new();
    let mut ne = Vec::new();
    let mut ek = Vec::new();
    let mut fkk = Vec::new();
    let mts: Vec<i32> = material
        .sections()
        .into_iter()
        .filter(|(mf, _)| *mf == 35)
        .map(|(_, mt)| mt)
        .collect();
    for section_mt in mts {
        let Some(mf35) = material.mf35(section_mt) else {
            continue;
        };
        for (idx, block) in mf35.blocks.iter().enumerate() {
            mt.push(section_mt);
            block_idx.push(idx as i32);
            e1.push(Some(block.e1));
            e2.push(Some(block.e2));
            ls.push(Some(narrow(block.ls)));
            lb.push(Some(narrow(block.lb)));
            ne.push(Some(narrow(block.ne)));
            ek.push(block.ek.clone());
            fkk.push(block.fkk.clone());
        }
    }
    if mt.is_empty() {
        return Ok(false);
    }
    write_section(
        &dir.join("spectrum_covariance.arrow"),
        "spectrum_covariance.arrow",
        vec![
            ints(&mt),
            ints(&block_idx),
            opt_floats(&e1),
            opt_floats(&e2),
            opt_ints(&ls),
            opt_ints(&lb),
            opt_ints(&ne),
            float_lists_or_null(&ek),
            float_lists_or_null(&fkk),
        ],
    )?;
    Ok(true)
}
