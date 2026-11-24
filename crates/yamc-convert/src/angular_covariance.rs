//! `angular_covariance.arrow`: MF=34, the covariance of the Legendre
//! coefficients of angular distributions, straight off the tape.
//!
//! One row per covariance block, which is one LIST record of one (L, L1) pair
//! of one subsection of one MT's section. The block is split by its `lb` into
//! the shape `covariance.arrow`'s NI blocks take
//! ([`endf::mf::covariance::split_mf34_block`]), and the subsection's partner,
//! Legendre order counts and frame ride on every row, so a reader needs nothing
//! but the row to place it.
//!
//! A section of its own rather than rows of `covariance.arrow`: every row there
//! is read as a cross-section covariance, and an MF=34 MT=2 block would be
//! folded as the elastic cross section's.

use std::error::Error;
use std::path::Path;

use endf::mf::covariance::{split_mf34_block, Mf34};
use endf::Material;

use crate::covariance::narrow;
use crate::sections::{float_lists_or_null, ints, opt_ints, write_section};

/// The rows of `angular_covariance.arrow`, as parallel columns.
#[derive(Default)]
struct AngularRows {
    mt: Vec<i32>,
    subsection_idx: Vec<i32>,
    pair_idx: Vec<i32>,
    block_idx: Vec<i32>,
    mat1: Vec<Option<i32>>,
    mt1: Vec<Option<i32>>,
    ltt: Vec<Option<i32>>,
    nl: Vec<Option<i32>>,
    nl1: Vec<Option<i32>>,
    l: Vec<Option<i32>>,
    l1: Vec<Option<i32>>,
    lct: Vec<Option<i32>>,
    lb: Vec<Option<i32>>,
    ls: Vec<Option<i32>>,
    nt: Vec<Option<i32>>,
    ne: Vec<Option<i32>>,
    ner: Vec<Option<i32>>,
    nec: Vec<Option<i32>>,
    ek: Vec<Vec<f64>>,
    fk: Vec<Vec<f64>>,
    fkk: Vec<Vec<f64>>,
    er: Vec<Vec<f64>>,
    ec: Vec<Vec<f64>>,
    fkl: Vec<Vec<f64>>,
}

impl AngularRows {
    /// Every block of one MF=34 section, in tape order.
    fn push_section(&mut self, mt: i32, mf34: &Mf34) -> Result<(), Box<dyn Error>> {
        for (subsection_idx, sub) in mf34.subsections.iter().enumerate() {
            for (pair_idx, pair) in sub.subsubsections.iter().enumerate() {
                let l = sub.l.get(pair_idx).map(|v| *v as i64).unwrap_or(0);
                let l1 = sub.l1.get(pair_idx).map(|v| *v as i64).unwrap_or(0);
                for (block_idx, values) in pair.data.iter().enumerate() {
                    let field = |v: &[f64]| v.get(block_idx).map(|x| *x as i64).unwrap_or(0);
                    let block = split_mf34_block(
                        field(&pair.ls),
                        field(&pair.lb),
                        field(&pair.nt),
                        field(&pair.ne),
                        values,
                    )?;
                    self.mt.push(mt);
                    self.subsection_idx.push(subsection_idx as i32);
                    self.pair_idx.push(pair_idx as i32);
                    self.block_idx.push(block_idx as i32);
                    self.mat1.push(Some(narrow(sub.mat1)));
                    self.mt1.push(Some(narrow(sub.mt1)));
                    self.ltt.push(Some(narrow(mf34.ltt)));
                    self.nl.push(Some(narrow(sub.nl)));
                    self.nl1.push(Some(narrow(sub.nl1)));
                    self.l.push(Some(narrow(l)));
                    self.l1.push(Some(narrow(l1)));
                    self.lct.push(Some(narrow(pair.lct)));
                    self.lb.push(Some(narrow(block.lb)));
                    self.ls.push(Some(narrow(block.ls)));
                    self.nt.push(Some(narrow(block.nt)));
                    self.ne.push(Some(narrow(block.ne)));
                    self.ner.push(Some(narrow(block.ner)));
                    self.nec.push(Some(narrow(block.nec)));
                    self.ek.push(block.ek);
                    self.fk.push(block.fk);
                    self.fkk.push(block.fkk);
                    self.er.push(block.er);
                    self.ec.push(block.ec);
                    self.fkl.push(block.fkl);
                }
            }
        }
        Ok(())
    }

    /// The columns in the schema's order, with the evaluation's own MAT last.
    fn columns(&self, mat: i32) -> Vec<arrow_array::ArrayRef> {
        vec![
            ints(&self.mt),
            ints(&self.subsection_idx),
            ints(&self.pair_idx),
            ints(&self.block_idx),
            opt_ints(&self.mat1),
            opt_ints(&self.mt1),
            opt_ints(&self.ltt),
            opt_ints(&self.nl),
            opt_ints(&self.nl1),
            opt_ints(&self.l),
            opt_ints(&self.l1),
            opt_ints(&self.lct),
            opt_ints(&self.lb),
            opt_ints(&self.ls),
            opt_ints(&self.nt),
            opt_ints(&self.ne),
            opt_ints(&self.ner),
            opt_ints(&self.nec),
            float_lists_or_null(&self.ek),
            float_lists_or_null(&self.fk),
            float_lists_or_null(&self.fkk),
            float_lists_or_null(&self.er),
            float_lists_or_null(&self.ec),
            float_lists_or_null(&self.fkl),
            opt_ints(&vec![Some(mat); self.mt.len()]),
        ]
    }
}

/// Write `angular_covariance.arrow`, one row per MF=34 covariance block.
///
/// Returns whether a file was written: an evaluation with no MF=34 block
/// writes nothing, and a reader takes the missing file as "no angular
/// covariance", as it does `covariance.arrow`. Independent of MF=33, which an
/// evaluation can carry without MF=34 and the other way round.
pub fn write_angular_covariance(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let mut rows = AngularRows::default();
    let mts: Vec<i32> = material
        .sections()
        .into_iter()
        .filter(|(mf, _)| *mf == 34)
        .map(|(_, mt)| mt)
        .collect();
    for mt in mts {
        if let Some(mf34) = material.mf34(mt) {
            rows.push_section(mt, mf34)?;
        }
    }
    if rows.mt.is_empty() {
        return Ok(false);
    }
    write_section(
        &dir.join("angular_covariance.arrow"),
        "angular_covariance.arrow",
        rows.columns(material.mat),
    )?;
    Ok(true)
}
