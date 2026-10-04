//! `covariance.arrow`: MF=33 cross-section covariance, straight off the tape.
//!
//! A faithful dump, not a transformation. Every column is a field of
//! [`endf::mf::covariance::Mf33Subsection`], [`NiSubsection`] or
//! [`NcSubsection`], written in the parser's own order and units, so reading
//! the file back reconstructs exactly what the parser produced. The one
//! addition is `mat`, the evaluation's own MAT from the tape's control
//! columns: `mat1` is kept as written, and ENDF-102 33.3.1 lets it name this
//! material by its MAT rather than by 0, which a reader can only recognise
//! with the MAT beside it.
//!
//! Nothing is reshaped on the way out. `fkk` keeps the format's packed order
//! with its own `ls` beside it, because `ls=1` is a triangle whose transpose is
//! implied and `ls=0` is a genuinely asymmetric block: folding one into the
//! other would lose numbers. Likewise a subsection with `mt1 != mt` is a
//! cross-reaction block whose transpose is the (`mt1`, `mt`) block rather than
//! itself, so no symmetry is assumed across rows either. Interpreting any of
//! that is the reader's job.
//!
//! One row per covariance block, which is one NC or NI sub-subsection of one
//! subsection.
//!
//! # The resonance-parameter contribution
//!
//! The one set of rows not on the tape. ENDF-102 section 32 makes a resolved
//! range's cross-section covariance the MF=32 resonance-parameter part plus
//! MF=33, and many evaluations put the whole resolved-range uncertainty in
//! MF=32 (ENDF/B-VIII.1 W, Cu, Cr, Ni, Pb and Ti among them). So for each
//! resolved MF=32 range whose formalism [`endf::resonance`] reconstructs
//! (Reich-Moore, and R-matrix limited without charged-particle channels), the
//! covariance of its elastic, capture and fission group cross sections
//! ([`endf::resonance_covariance::group_covariance`], one group per resonance,
//! 1/E weight, infinite dilution, 0 K) is written as NI blocks: LB=5 for a
//! reaction with itself, LB=6 for elastic with capture and the like. They are
//! relative to the whole cross section, resonance part plus MF=3 background,
//! since that is what a perturbation multiplies, as NJOY's ERRORR takes it.
//! These rows have `subsection_idx = -1`, which no tape subsection has, so
//! they read as the ordinary NI blocks they are (a reader adds them to
//! MF=33's, as the format says) and are still told apart from the tape's. That granularity IS the sparse form: most (MT, MT1) pairs have
//! no cross terms and simply have no row, with nothing thresholded and no small
//! value dropped.
//!
//! The one row that is not a block is a component of a lumped reaction
//! (ENDF-102 33.2.3): a section with a nonzero MTL and no subsections, whose
//! HEAD record is the only statement of which reactions the lumped MT 851-870
//! sums. Dropping it would lose that index, so it is written as a
//! `kind = "lumped"` row.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use endf::mf::covariance::{Mf33, Mf33Subsection, NcSubsection, NiSubsection};
use endf::Material;

use crate::sections::{float_lists_or_null, ints, opt_floats, opt_ints, strings, write_section};

/// The columns of `covariance.arrow`, accumulated one row at a time.
///
/// A struct of parallel vectors rather than a vector of structs, because that
/// is the shape [`write_section`] wants and building it directly avoids a
/// transpose whose only job would be to reintroduce the column order the schema
/// already fixes.
///
/// Public because `branching/branching_covariance.arrow` is the same blocks
/// with a key in front: MF=40 writes its sub-subsections in MF=33's format, and
/// the transmutation converter builds those rows here rather than through a
/// second copy of this layout.
#[derive(Default)]
pub struct CovarianceRows {
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
pub fn narrow(v: i64) -> i32 {
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

impl CovarianceRows {
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
        mtl: Option<i64>,
    ) {
        self.mt.push(mt);
        self.subsection_idx.push(subsection_idx as i32);
        self.block_idx.push(block_idx as i32);
        self.kind.push(kind.to_string());
        self.mat1.push(Some(narrow(mat1)));
        self.mt1.push(Some(narrow(mt1)));
        self.xmf1.push(Some(xmf1));
        self.xlfs1.push(Some(xlfs1));
        self.mtl.push(mtl.map(narrow));
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

    /// A lumped reaction's component: its section's HEAD record and nothing
    /// else, since the format gives a component no subsections.
    ///
    /// Only `mt` and `mtl` carry anything here, and `mat` beside them in
    /// `covariance.arrow`. The HEAD record has no MAT1, MT1, XMF1 or XLFS1,
    /// so those are null rather than a zero the tape never wrote, and the two
    /// indices are 0 because the section has no subsection or block for them
    /// to count.
    fn push_lumped(&mut self, mt: i32, mtl: i64) {
        self.mt.push(mt);
        self.subsection_idx.push(0);
        self.block_idx.push(0);
        self.kind.push("lumped".to_string());
        self.mat1.push(None);
        self.mt1.push(None);
        self.xmf1.push(None);
        self.xlfs1.push(None);
        self.mtl.push(Some(narrow(mtl)));
        self.push_ni_null();
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

    /// One MF=33-format subsection's blocks, in tape order, returning how
    /// many rows that was.
    ///
    /// NC blocks precede NI blocks within a subsection because that is the
    /// order they appear on the tape and the order the parser reads them, so
    /// `block_idx` is a running index over both rather than one per kind.
    /// `mtl` is the section HEAD's lumped-reaction MT, `None` for a file that
    /// has no such field (MF=40), which is written as null rather than as a
    /// zero the tape never stated.
    pub fn push_subsection(
        &mut self,
        mt: i32,
        subsection_idx: usize,
        mtl: Option<i64>,
        sub: &Mf33Subsection,
    ) -> usize {
        let mut block_idx = 0;
        for block in &sub.nc_subsections {
            self.push_common(
                mt,
                subsection_idx,
                block_idx,
                "nc",
                sub.mat1,
                sub.mt1,
                sub.xmf1,
                sub.xlfs1,
                mtl,
            );
            self.push_nc(block);
            block_idx += 1;
        }
        for block in &sub.ni_subsections {
            self.push_common(
                mt,
                subsection_idx,
                block_idx,
                "ni",
                sub.mat1,
                sub.mt1,
                sub.xmf1,
                sub.xlfs1,
                mtl,
            );
            self.push_ni(block);
            block_idx += 1;
        }
        block_idx
    }

    /// One NI block derived from MF=32 rather than read off the tape: the
    /// covariance of `mt` with `mt1` in this evaluation, written with
    /// `subsection_idx = -1` (see the module documentation).
    fn push_derived(&mut self, mt: i32, block_idx: usize, mt1: i32, block: &NiSubsection) {
        self.push_common(mt, 0, block_idx, "ni", 0, mt1 as i64, 0.0, 0.0, Some(0));
        *self.subsection_idx.last_mut().expect("just pushed") = -1;
        self.push_ni(block);
    }

    pub fn is_empty(&self) -> bool {
        self.mt.is_empty()
    }

    /// The number of rows, one per block.
    pub fn len(&self) -> usize {
        self.mt.len()
    }

    /// The columns in the schema's own order, all but `covariance.arrow`'s
    /// trailing `mat`: a file of one evaluation's blocks writes it once per row
    /// beside these, and `branching_covariance.arrow` has its own in its key.
    pub fn columns(&self) -> Vec<arrow_array::ArrayRef> {
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
/// Two subsections of one section may name the same (MAT1, MT1), which is why
/// the position is carried explicitly instead of being recovered from the keys.
///
/// A section with a nonzero MTL is a lumped reaction's component, and its one
/// row is its HEAD. ENDF-102 33.2.3 gives such a section no subsections
/// (NL=0), and one that has them is refused: the lump's component list is read
/// off these HEAD rows, and a component that also stated a covariance of its
/// own would leave the fold no exact reading of either. `mat` names the
/// evaluation in that error.
fn push_section(
    rows: &mut CovarianceRows,
    mat: i32,
    mt: i32,
    mf33: &Mf33,
) -> Result<(), Box<dyn Error>> {
    if mf33.mtl != 0 {
        if !mf33.subsections.is_empty() {
            return Err(format!(
                "MAT {mat} MF=33 MT={mt} is a component of lumped reaction MT={} but has {} \
                 subsections; ENDF-102 33.2.3 requires NL=0 for a lumped reaction's component",
                mf33.mtl,
                mf33.subsections.len()
            )
            .into());
        }
        rows.push_lumped(mt, mf33.mtl);
        return Ok(());
    }
    for (subsection_idx, sub) in mf33.subsections.iter().enumerate() {
        rows.push_subsection(mt, subsection_idx, Some(mf33.mtl), sub);
    }
    Ok(())
}

/// Write `covariance.arrow`, one row per covariance block, plus one per
/// lumped reaction's component HEAD.
///
/// Returns whether a file was written. An evaluation with no MF=33 at all, or
/// one whose MF=33 sections hold no blocks and name no lumped reaction, writes
/// nothing: absence is how this section says "no covariance", and the reader
/// treats a missing file that way rather than as an error. No `.absent` marker
/// is written here, since that is a download-cache record of a settled 404
/// rather than anything a conversion produces.
pub fn write_covariance(material: &Material, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let mut rows = CovarianceRows::default();
    for mt in covariance_mts(material) {
        if let Some(mf33) = material.mf33(mt) {
            push_section(&mut rows, material.mat, mt, mf33)?;
        }
    }
    push_resonance_blocks(&mut rows, material)?;

    if rows.is_empty() {
        return Ok(false);
    }

    // The evaluation's own MAT, on every row, so a reader can tell a `mat1`
    // naming this material from one naming another.
    let mut columns = rows.columns();
    columns.push(opt_ints(&vec![Some(material.mat); rows.len()]));
    write_section(&dir.join("covariance.arrow"), "covariance.arrow", columns)?;
    Ok(true)
}

/// The 1/E average of `sigma` over each group of `edges`: trapezoids in
/// `ln E` on the function's own points and the edges.
fn group_average(sigma: &endf::function::Tabulated1D, edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|w| {
            let (lo, hi) = (w[0], w[1]);
            let mut points: Vec<f64> = sigma
                .x
                .iter()
                .copied()
                .filter(|&e| e > lo && e < hi)
                .collect();
            points.insert(0, lo);
            points.push(hi);
            let mut total = 0.0;
            for p in points.windows(2) {
                total += 0.5 * (p[1] / p[0]).ln() * (sigma.eval(p[0]) + sigma.eval(p[1]));
            }
            total / (hi / lo).ln()
        })
        .collect()
}

/// The resonance-parameter contribution, as NI blocks (see the module
/// documentation). A range whose formalism is not reconstructed yet is
/// skipped, as is an evaluation with no MF=32.
fn push_resonance_blocks(
    rows: &mut CovarianceRows,
    material: &Material,
) -> Result<(), Box<dyn Error>> {
    use endf::resonance::{RMatrixRange, ReichMooreRange, ResolvedRange};
    use endf::resonance_covariance::{group_covariance, resolved_covariances, resonance_edges};
    let (Some(mf2), Some(mf32)) = (material.mf2(), material.mf32()) else {
        return Ok(());
    };
    let mut next_block: BTreeMap<i32, usize> = BTreeMap::new();
    for cov in resolved_covariances(mf2, mf32)? {
        if cov.is_empty() {
            continue;
        }
        let range = &mf2.isotopes[cov.isotope].ranges[cov.mf2_range];
        // A formalism or feature endf does not reconstruct yet (charged
        // particle channels, for one) leaves the range to MF=33 alone; any
        // other failure is an error in the evaluation or the reader, and is
        // not quietly dropped.
        let reconstruction: Box<dyn ResolvedRange> = match range.lrf {
            3 => match ReichMooreRange::new(range) {
                Ok(r) => Box::new(r),
                Err(endf::Error::Unsupported { .. }) => continue,
                Err(e) => return Err(e.into()),
            },
            7 => match RMatrixRange::new(range) {
                Ok(r) => Box::new(r),
                Err(endf::Error::Unsupported { .. }) => continue,
                Err(e) => return Err(e.into()),
            },
            _ => continue,
        };
        // One group per resonance, within the covariance's own range (which
        // can stop short of MF=2's).
        let mut edges: Vec<f64> = resonance_edges(reconstruction.as_ref())
            .into_iter()
            .filter(|&e| e > cov.el && e < cov.eh)
            .collect();
        edges.insert(0, cov.el.max(range.el));
        edges.push(cov.eh.min(range.eh));
        let resonance = group_covariance(&cov, reconstruction.as_ref(), &edges)?;
        let totals: Vec<Vec<f64>> = resonance
            .reactions
            .iter()
            .zip(&resonance.cross_sections)
            .map(|(mt, xs)| {
                let background = material
                    .mf3(*mt)
                    .map_or_else(|| vec![0.0; xs.len()], |b| group_average(&b.sigma, &edges));
                xs.iter().zip(background).map(|(r, b)| r + b).collect()
            })
            .collect();
        let g = resonance.relative_to(&totals);
        let n = g.groups();
        for (a, &mt) in g.reactions.iter().enumerate() {
            for (b, &mt1) in g.reactions.iter().enumerate().skip(a) {
                let values: Vec<f64> = if a == b {
                    (0..n)
                        .flat_map(|h| (h..n).map(move |k| (h, k)))
                        .map(|(h, k)| g.get(a, h, a, k))
                        .collect()
                } else {
                    (0..n)
                        .flat_map(|h| (0..n).map(move |k| (h, k)))
                        .map(|(h, k)| g.get(a, h, b, k))
                        .collect()
                };
                if values.iter().all(|v| *v == 0.0) {
                    continue;
                }
                let block = if a == b {
                    NiSubsection {
                        lb: 5,
                        ls: 1,
                        ne: (n + 1) as i64,
                        nt: values.len() as i64 + (n + 1) as i64,
                        ek: g.edges.clone(),
                        fkk: values,
                        ..Default::default()
                    }
                } else {
                    NiSubsection {
                        lb: 6,
                        ner: (n + 1) as i64,
                        nec: (n + 1) as i64,
                        nt: 1 + 2 * (n + 1) as i64 + values.len() as i64,
                        er: g.edges.clone(),
                        ec: g.edges.clone(),
                        fkl: values,
                        ..Default::default()
                    }
                };
                let idx = next_block.entry(mt).or_insert(0);
                rows.push_derived(mt, *idx, mt1, &block);
                *idx += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use endf::mf::covariance::Mf33Subsection;

    /// A lumped reaction's component is its HEAD alone. One that also carries
    /// subsections is out of format and refused, not written as blocks whose
    /// lump the fold would never see.
    #[test]
    fn a_component_with_subsections_is_refused() {
        let mut rows = CovarianceRows::default();
        let head = Mf33 {
            mtl: 852,
            ..Mf33::default()
        };
        push_section(&mut rows, 7443, 16, &head).expect("a HEAD alone is a component");
        assert_eq!(rows.mtl, [Some(852)]);
        let with_blocks = Mf33 {
            subsections: vec![Mf33Subsection::default()],
            ..head
        };
        let err = push_section(&mut rows, 7443, 16, &with_blocks).unwrap_err();
        assert!(err.to_string().contains("NL=0"), "{err}");
    }
}
