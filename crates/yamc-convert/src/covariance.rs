//! `covariance.arrow`: MF=33 cross-section covariance, straight off the tape.
//!
//! A faithful dump, not a transformation. Every column is a field of
//! [`endf::mf::covariance::Mf33Subsection`], [`NiSubsection`] or
//! [`NcSubsection`], written in the parser's own order and units, so reading
//! the file back reconstructs exactly what the parser produced.
//!
//! Nothing is reshaped on the way out. `fkk` keeps the format's packed order
//! with its own `ls` beside it, because `ls=1` is a triangle whose transpose is
//! implied and `ls=0` is a genuinely asymmetric block: folding one into the
//! other would lose numbers. Likewise a subsection with `mt1 != mt` is a
//! cross-reaction block whose transpose is the (`mt1`, `mt`) block rather than
//! itself, so no symmetry is assumed across rows either. Interpreting any of
//! that is the reader's job (issue #514).
//!
//! One row per covariance block, which is one NC or NI sub-subsection of one
//! subsection. That granularity IS the sparse form: most (MT, MT1) pairs have
//! no cross terms and simply have no row, with nothing thresholded and no small
//! value dropped.

use std::error::Error;
use std::path::Path;

use endf::mf::covariance::{Mf33, NcSubsection, NiSubsection};
use endf::Material;

use crate::sections::{float_lists_or_null, ints, opt_floats, opt_ints, strings, write_section};

/// The columns of `covariance.arrow`, accumulated one row at a time.
///
/// A struct of parallel vectors rather than a vector of structs, because that
/// is the shape [`write_section`] wants and building it directly avoids a
/// transpose whose only job would be to reintroduce the column order the schema
/// already fixes.
#[derive(Default)]
struct Rows {
    mt: Vec<i32>,
    subsection_idx: Vec<i32>,
    block_idx: Vec<i32>,
    kind: Vec<String>,
    mat1: Vec<Option<i32>>,
    mt1: Vec<Option<i32>>,
    xmf1: Vec<Option<f64>>,
    xlfs1: Vec<Option<f64>>,
    mtl: Vec<Option<i32>>,

    // kind = "ni"
    lb: Vec<Option<i32>>,
    ls: Vec<Option<i32>>,
    lt: Vec<Option<i32>>,
    nt: Vec<Option<i32>>,
    np: Vec<Option<i32>>,
    ne: Vec<Option<i32>>,
    ner: Vec<Option<i32>>,
    nec: Vec<Option<i32>>,
    ek: Vec<Vec<f64>>,
    fk: Vec<Vec<f64>>,
    el: Vec<Vec<f64>>,
    fl: Vec<Vec<f64>>,
    fkk: Vec<Vec<f64>>,
    er: Vec<Vec<f64>>,
    ec: Vec<Vec<f64>>,
    fkl: Vec<Vec<f64>>,

    // kind = "nc"
    lty: Vec<Option<i32>>,
    e1: Vec<Option<f64>>,
    e2: Vec<Option<f64>>,
    nci: Vec<Option<i32>>,
    ci: Vec<Vec<f64>>,
    xmti: Vec<Vec<f64>>,
    mats: Vec<Option<i32>>,
    mts: Vec<Option<i32>>,
    nei: Vec<Option<i32>>,
    xmfs: Vec<Option<f64>>,
    xlfss: Vec<Option<f64>>,
    ei: Vec<Vec<f64>>,
    wei: Vec<Vec<f64>>,
}

/// A count that came off the tape as `i64`, narrowed for the column.
///
/// Saturating rather than wrapping: these are ENDF counts and flags, none of
/// which can legitimately exceed `i32`, and a corrupt tape should not silently
/// become a plausible small number.
fn narrow(v: i64) -> i32 {
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

impl Rows {
    /// The columns every row carries, whatever its kind.
    #[allow(clippy::too_many_arguments)]
    fn push_common(
        &mut self,
        mt: i32,
        subsection_idx: usize,
        block_idx: usize,
        kind: &str,
        mat1: i64,
        mt1: i64,
        xmf1: f64,
        xlfs1: f64,
        mtl: i64,
    ) {
        self.mt.push(mt);
        self.subsection_idx.push(subsection_idx as i32);
        self.block_idx.push(block_idx as i32);
        self.kind.push(kind.to_string());
        self.mat1.push(Some(narrow(mat1)));
        self.mt1.push(Some(narrow(mt1)));
        self.xmf1.push(Some(xmf1));
        self.xlfs1.push(Some(xlfs1));
        self.mtl.push(Some(narrow(mtl)));
    }

    /// Null every `kind = "ni"` column, for a row that is an NC block.
    fn push_ni_null(&mut self) {
        self.lb.push(None);
        self.ls.push(None);
        self.lt.push(None);
        self.nt.push(None);
        self.np.push(None);
        self.ne.push(None);
        self.ner.push(None);
        self.nec.push(None);
        self.ek.push(Vec::new());
        self.fk.push(Vec::new());
        self.el.push(Vec::new());
        self.fl.push(Vec::new());
        self.fkk.push(Vec::new());
        self.er.push(Vec::new());
        self.ec.push(Vec::new());
        self.fkl.push(Vec::new());
    }

    /// Null every `kind = "nc"` column, for a row that is an NI block.
    fn push_nc_null(&mut self) {
        self.lty.push(None);
        self.e1.push(None);
        self.e2.push(None);
        self.nci.push(None);
        self.ci.push(Vec::new());
        self.xmti.push(Vec::new());
        self.mats.push(None);
        self.mts.push(None);
        self.nei.push(None);
        self.xmfs.push(None);
        self.xlfss.push(None);
        self.ei.push(Vec::new());
        self.wei.push(Vec::new());
    }

    /// One NI block: a covariance given explicitly.
    ///
    /// Every NI scalar is written as the parser left it, including the ones
    /// this block's `lb` does not use (which the parser leaves at zero). That
    /// keeps the round trip exact without the reader having to know which
    /// fields each `lb` populates in order to reconstruct a default.
    fn push_ni(&mut self, s: &NiSubsection) {
        self.lb.push(Some(narrow(s.lb)));
        self.ls.push(Some(narrow(s.ls)));
        self.lt.push(Some(narrow(s.lt)));
        self.nt.push(Some(narrow(s.nt)));
        self.np.push(Some(narrow(s.np)));
        self.ne.push(Some(narrow(s.ne)));
        self.ner.push(Some(narrow(s.ner)));
        self.nec.push(Some(narrow(s.nec)));
        self.ek.push(s.ek.clone());
        self.fk.push(s.fk.clone());
        self.el.push(s.el.clone());
        self.fl.push(s.fl.clone());
        self.fkk.push(s.fkk.clone());
        self.er.push(s.er.clone());
        self.ec.push(s.ec.clone());
        self.fkl.push(s.fkl.clone());
        self.push_nc_null();
    }

    /// One NC block: a covariance derived from other reactions.
    fn push_nc(&mut self, s: &NcSubsection) {
        self.push_ni_null();
        self.lty.push(Some(narrow(s.lty)));
        self.e1.push(Some(s.e1));
        self.e2.push(Some(s.e2));
        self.nci.push(Some(narrow(s.nci)));
        self.ci.push(s.ci.clone());
        self.xmti.push(s.xmti.clone());
        self.mats.push(Some(narrow(s.mats)));
        self.mts.push(Some(narrow(s.mts)));
        self.nei.push(Some(narrow(s.nei)));
        self.xmfs.push(Some(s.xmfs));
        self.xlfss.push(Some(s.xlfss));
        self.ei.push(s.ei.clone());
        self.wei.push(s.wei.clone());
    }

    fn is_empty(&self) -> bool {
        self.mt.is_empty()
    }

    /// The columns in the schema's own order.
    fn columns(&self) -> Vec<arrow_array::ArrayRef> {
        vec![
            ints(&self.mt),
            ints(&self.subsection_idx),
            ints(&self.block_idx),
            strings(&self.kind),
            opt_ints(&self.mat1),
            opt_ints(&self.mt1),
            opt_floats(&self.xmf1),
            opt_floats(&self.xlfs1),
            opt_ints(&self.mtl),
            opt_ints(&self.lb),
            opt_ints(&self.ls),
            opt_ints(&self.lt),
            opt_ints(&self.nt),
            opt_ints(&self.np),
            opt_ints(&self.ne),
            opt_ints(&self.ner),
            opt_ints(&self.nec),
            float_lists_or_null(&self.ek),
            float_lists_or_null(&self.fk),
            float_lists_or_null(&self.el),
            float_lists_or_null(&self.fl),
            float_lists_or_null(&self.fkk),
            float_lists_or_null(&self.er),
            float_lists_or_null(&self.ec),
            float_lists_or_null(&self.fkl),
            opt_ints(&self.lty),
            opt_floats(&self.e1),
            opt_floats(&self.e2),
            opt_ints(&self.nci),
            float_lists_or_null(&self.ci),
            float_lists_or_null(&self.xmti),
            opt_ints(&self.mats),
            opt_ints(&self.mts),
            opt_ints(&self.nei),
            opt_floats(&self.xmfs),
            opt_floats(&self.xlfss),
            float_lists_or_null(&self.ei),
            float_lists_or_null(&self.wei),
        ]
    }
}

/// Every MT this evaluation carries an MF=33 section for, ascending.
///
/// From `section_data` rather than a fixed MT list: which reactions an
/// evaluation gives covariance for is the evaluator's choice, and a hard-coded
/// set would silently drop whatever was not anticipated.
fn covariance_mts(material: &Material) -> Vec<i32> {
    material
        .sections()
        .into_iter()
        .filter(|(mf, _)| *mf == 33)
        .map(|(_, mt)| mt)
        .collect()
}

/// Accumulate one MF=33 section's blocks, in tape order.
///
/// NC blocks precede NI blocks within a subsection because that is the order
/// they appear on the tape and the order the parser reads them, so `block_idx`
/// is a running index over both rather than one per kind. Two subsections of
/// one section may name the same (MAT1, MT1), which is why the position is
/// carried explicitly instead of being recovered from the keys.
fn push_section(rows: &mut Rows, mt: i32, mf33: &Mf33) {
    for (subsection_idx, sub) in mf33.subsections.iter().enumerate() {
        let mut block_idx = 0;
        for block in &sub.nc_subsections {
            rows.push_common(
                mt,
                subsection_idx,
                block_idx,
                "nc",
                sub.mat1,
                sub.mt1,
                sub.xmf1,
                sub.xlfs1,
                mf33.mtl,
            );
            rows.push_nc(block);
            block_idx += 1;
        }
        for block in &sub.ni_subsections {
            rows.push_common(
                mt,
                subsection_idx,
                block_idx,
                "ni",
                sub.mat1,
                sub.mt1,
                sub.xmf1,
                sub.xlfs1,
                mf33.mtl,
            );
            rows.push_ni(block);
            block_idx += 1;
        }
    }
}

/// Write `covariance.arrow`, one row per covariance block.
///
/// Returns whether a file was written. An evaluation with no MF=33 at all, or
/// one whose MF=33 sections hold no blocks, writes nothing: absence is how this
/// section says "no covariance", and the reader treats a missing file that way
/// rather than as an error. No `.absent` marker is written here, since that is
/// a download-cache record of a settled 404 rather than anything a conversion
/// produces.
pub fn write_covariance(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let mut rows = Rows::default();
    for mt in covariance_mts(material) {
        if let Some(mf33) = material.mf33(mt) {
            push_section(&mut rows, mt, mf33);
        }
    }

    if rows.is_empty() {
        return Ok(false);
    }

    write_section(
        &dir.join("covariance.arrow"),
        "covariance.arrow",
        rows.columns(),
    )?;
    Ok(true)
}
