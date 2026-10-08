//! MF=33 cross-section covariance, as it comes off `covariance.arrow`.
//!
//! The blocks here are the evaluation's own, unreshaped: one per NC or NI
//! sub-subsection of one MF=33 subsection, each keeping the energy grid the
//! evaluator wrote it on. Turning them into a matrix is [`expand`]'s job, and
//! it is deliberately separate, because how a block becomes a matrix depends on
//! its `lb` and getting that wrong should not be hidden inside a reader.
//!
//! The block payloads are [`endf`]'s own `NiSubsection` and `NcSubsection`
//! rather than a second pair of structs with the same fields. That is what
//! makes "the file round-trips the parser" a statement about types rather than
//! about a transcription someone has to keep in step.

pub mod expand;

use endf::mf::covariance::{NcSubsection, NiSubsection};

/// Which kind of block this is, and its payload.
///
/// The discriminant is the `kind` column, and it selects which columns of the
/// row were populated. NC is a covariance derived from other reactions; NI is
/// one given explicitly. `Lumped` is not a covariance at all but the section
/// of a lumped reaction's component, see [`CovarianceBlock::lumped_into`].
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceData {
    /// Given explicitly. `lb` selects the layout within it.
    Ni(NiSubsection),
    /// Derived from other reactions. `lty` selects the derivation.
    Nc(NcSubsection),
    /// Reaction `mt` is a component of the lumped reaction `mtl` and has no
    /// covariance of its own (ENDF-102 33.2.3). The section's HEAD record is
    /// all there is, so the block's other fields are 0. Those zeros are the
    /// spellings of "this evaluation's cross section, the same reaction", so
    /// [`CovarianceBlock::is_same_evaluation`] and
    /// [`CovarianceBlock::names_cross_section`] hold for it. A consumer that
    /// wants covariance blocks matches on the data or excludes
    /// [`CovarianceBlock::lumped_into`]; only
    /// [`CovarianceBlock::is_diagonal`] turns it away itself.
    Lumped,
}

/// One MF=34 covariance block: the covariance of Legendre coefficient `l` of
/// reaction `mt`'s angular distribution with coefficient `l1` of `mt1`'s.
///
/// Read from `angular_covariance.arrow`, which keeps these apart from
/// `covariance.arrow`'s cross-section blocks. Nothing samples them yet; they
/// are carried so the data is read faithfully and reported.
#[derive(Debug, Clone, PartialEq)]
pub struct AngularCovarianceBlock {
    /// The reaction this block's section belongs to.
    pub mt: i32,
    /// Which subsection of that section, in tape order (one per MAT1, MT1).
    pub subsection_idx: i32,
    /// Which (L, L1) pair within that subsection, in tape order.
    pub pair_idx: i32,
    /// Which block within that pair, in tape order.
    pub block_idx: i32,
    /// The material `mt1` belongs to. `0`, or `mat`, means this evaluation.
    pub mat1: i32,
    /// The reaction this one is correlated with.
    pub mt1: i32,
    /// LTT of the section: the representation MF=4 uses.
    pub ltt: i32,
    /// Legendre orders of `mt` and `mt1` the subsection covers.
    pub nl: i32,
    pub nl1: i32,
    /// The Legendre orders this block correlates.
    pub l: i32,
    pub l1: i32,
    /// The frame of the coefficients: 1 laboratory, 2 centre of mass, 0 the
    /// same as MF=4's.
    pub lct: i32,
    /// The evaluation's own MAT, `0` when the file does not state it.
    pub mat: i32,
    /// The block itself, in MF=33's NI shape (`lb` 0-2, 5 or 6).
    pub block: NiSubsection,
}

/// One covariance block: the covariance of `mt` with `mt1`, on one grid.
///
/// # Which blocks are this evaluation's own
///
/// `mat1` and `xmf1` are kept as the tape wrote them, and ENDF-102 33.3.1
/// gives each two spellings for "this evaluation's cross section": `mat1` 0 or
/// the evaluation's own MAT, and `xmf1` 0 or 3. FENDL-3.2d writes its Pt, Re,
/// Lu, S and O17 cross-reaction blocks with `mat1` equal to their own MAT.
/// `xmf1 = 3` appears on JEFF-4.0 H1 and Li6 (one block each) and on TENDL-2017
/// H1, Li6 and B10 (five blocks, with `mat1` their own MAT as well).
/// [`CovarianceBlock::is_same_evaluation`] reads both spellings, so no caller
/// compares `mat1` with 0.
///
/// # Which blocks are symmetric
///
/// Only the diagonal ones, a reaction with itself within this evaluation
/// ([`CovarianceBlock::is_diagonal`]). An off-diagonal block's transpose is
/// the (`mt1`, `mt`) block, not itself, so nothing here may be mirrored on the
/// assumption that a covariance matrix is symmetric. The same caution applies
/// one level down, inside an `Ni` block: `ls == 1` is an upper triangle whose
/// transpose is implied, `ls == 0` is a full asymmetric matrix as written, and
/// folding one into the other loses numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceBlock {
    /// The reaction this block's section belongs to.
    pub mt: i32,
    /// Which subsection of that section, in tape order.
    ///
    /// Carried explicitly because two subsections of one section may name the
    /// same (`mat1`, `mt1`), so the pair is not an identity.
    pub subsection_idx: i32,
    /// Which block within that subsection, in tape order: NC blocks first,
    /// then NI, as one running index.
    pub block_idx: i32,
    /// The material `mt1` belongs to. `0`, or [`CovarianceBlock::mat`], means
    /// this evaluation.
    pub mat1: i32,
    /// The reaction this one is correlated with. `0` means `mt` itself.
    pub mt1: i32,
    /// MF of the second quantity. `0` and `3` both mean a cross section.
    pub xmf1: f64,
    /// Final excited state of the second quantity.
    pub xlfs1: f64,
    /// MT this reaction is lumped into, from the section HEAD. `0` when it is
    /// not lumped.
    pub mtl: i32,
    /// The evaluation's own MAT. `0` when the file does not state it, as one
    /// written before the column existed does not, and then only `mat1 == 0`
    /// can be told to be this evaluation.
    pub mat: i32,
    /// The block itself.
    pub data: CovarianceData,
}

impl CovarianceBlock {
    /// The reaction this block correlates `mt` with.
    ///
    /// `mt1 == 0` on the tape means "the same reaction", so this resolves it
    /// rather than leaving every caller to remember the convention. MF=33's
    /// convention only: an MF=40 block goes through
    /// [`BranchingCovarianceBlock`] instead.
    pub fn partner_mt(&self) -> i32 {
        if self.mt1 == 0 {
            self.mt
        } else {
            self.mt1
        }
    }

    /// Whether this block is a reaction's covariance with itself.
    ///
    /// The only blocks for which the matrix is symmetric in itself, and the
    /// only ones a variance can be read off directly. A lumped reaction's
    /// component states no covariance, so it is not one, although its HEAD
    /// reads as the same reaction. MF=33's convention: an MF=40 partner (XMF1
    /// 10) is never a cross section, so this is false for every MF=40 block;
    /// use [`BranchingCovarianceBlock::is_self_block`].
    pub fn is_diagonal(&self) -> bool {
        !matches!(self.data, CovarianceData::Lumped)
            && self.is_same_evaluation()
            && self.partner_mt() == self.mt
    }

    /// The lumped reaction (MT 851-870) this block's `mt` is a component of,
    /// when the block is that component's section.
    ///
    /// ENDF-102 33.2.3 lists a lumped reaction's components nowhere but on
    /// their own HEAD records, so the lumped reactions of an evaluation are
    /// these blocks grouped by what this returns. The lumped reaction's own
    /// section gives the covariance of the SUM of its components; none of
    /// them has one of its own.
    pub fn lumped_into(&self) -> Option<i32> {
        matches!(self.data, CovarianceData::Lumped).then_some(self.mtl)
    }

    /// Whether `mt1` is a cross section of this same evaluation.
    ///
    /// `mat1` names this material when it is 0 or this evaluation's own MAT,
    /// and `xmf1` names a cross section when it is 0 or 3 (ENDF-102 33.3.1).
    /// `xlfs1` must be 0, since a final excited state belongs to an MF=10
    /// partner and not to a cross section.
    pub fn is_same_evaluation(&self) -> bool {
        let this_material = self.mat1 == 0 || (self.mat != 0 && self.mat1 == self.mat);
        this_material && self.names_cross_section()
    }

    /// Whether this block correlates with a reaction in ANOTHER evaluation.
    ///
    /// Not consumed today: using it would mean sampling two nuclides' cross
    /// sections from one joint distribution, and the fold is per nuclide.
    /// Counted and reported rather than silently dropped. In a file without
    /// the own MAT (`mat == 0`), every block with `mat1 != 0` counts as
    /// another evaluation's, since nothing says that `mat1` names this
    /// material. For MF=40 use [`BranchingCovarianceBlock::is_cross_material`].
    pub fn is_cross_material(&self) -> bool {
        self.mat1 != 0 && self.mat1 != self.mat
    }

    /// Whether the second quantity is a cross section (`xmf1` 0 or 3, with no
    /// final state), rather than, say, an MF=10 isomer production.
    pub fn names_cross_section(&self) -> bool {
        (self.xmf1 == 0.0 || self.xmf1 == 3.0) && self.xlfs1 == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cross-reaction block of MAT 7837 (FENDL-3.2d Pt194) on `mt`, naming
    /// `mat1` and `xmf1` as a tape might.
    fn block(mat: i32, mat1: i32, xmf1: f64) -> CovarianceBlock {
        CovarianceBlock {
            mt: 16,
            subsection_idx: 0,
            block_idx: 0,
            mat1,
            mt1: 102,
            xmf1,
            xlfs1: 0.0,
            mtl: 0,
            mat,
            data: CovarianceData::Ni(NiSubsection::default()),
        }
    }

    /// ENDF-102 33.3.1 gives two spellings each for "this material" and "a
    /// cross section". FENDL-3.2d and JEFF-4.0 write `mat1` as their own MAT,
    /// and JEFF-4.0 H1 and Li6 and TENDL-2017 H1, Li6 and B10 write `xmf1 = 3`.
    #[test]
    fn own_mat_and_xmf1_three_are_this_evaluation() {
        for (mat1, xmf1) in [(0, 0.0), (7837, 0.0), (0, 3.0), (7837, 3.0)] {
            let b = block(7837, mat1, xmf1);
            assert!(b.is_same_evaluation(), "mat1 {mat1}, xmf1 {xmf1}");
            assert!(!b.is_cross_material(), "mat1 {mat1}, xmf1 {xmf1}");
        }
    }

    /// A component's HEAD has every other field 0, which would read as the
    /// reaction with itself, but it states no covariance.
    #[test]
    fn a_lumped_component_is_not_a_diagonal_block() {
        let b = CovarianceBlock {
            mt1: 0,
            mtl: 851,
            data: CovarianceData::Lumped,
            ..block(7837, 0, 0.0)
        };
        assert!(!b.is_diagonal());
        assert_eq!(b.lumped_into(), Some(851));
    }

    #[test]
    fn another_mat_is_another_evaluation() {
        let b = block(7837, 9228, 0.0);
        assert!(b.is_cross_material());
        assert!(!b.is_same_evaluation());
    }

    /// A file written before the `mat` column has no own MAT, so a nonzero
    /// `mat1` cannot be shown to be this material and stays another's.
    #[test]
    fn without_the_own_mat_only_zero_is_this_material() {
        assert!(block(0, 0, 0.0).is_same_evaluation());
        let b = block(0, 7837, 0.0);
        assert!(b.is_cross_material());
        assert!(!b.is_same_evaluation());
    }

    /// An MF=10 partner is this material but not a cross section, so the
    /// block is neither folded as one nor counted as another evaluation's.
    #[test]
    fn another_file_is_neither() {
        let b = block(7837, 7837, 10.0);
        assert!(!b.is_same_evaluation());
        assert!(!b.is_cross_material());
        assert!(!b.names_cross_section());
    }

    /// Only a component's own section names the lump it belongs to: a block
    /// carrying a nonzero `mtl` is still a block. Conversion refuses such a
    /// section (ENDF-102 33.2.3 gives a component NL=0), so none is written.
    #[test]
    fn only_a_component_section_is_lumped() {
        let mut b = block(7443, 0, 0.0);
        b.mtl = 852;
        assert_eq!(b.lumped_into(), None);
        b.data = CovarianceData::Lumped;
        assert_eq!(b.lumped_into(), Some(852));
    }

    /// Diagonal needs the same evaluation as well as the same reaction.
    #[test]
    fn a_diagonal_block_may_name_its_own_mat() {
        let mut b = block(7837, 7837, 0.0);
        b.mt1 = 16;
        assert!(b.is_diagonal());
        b.mat = 0;
        assert!(!b.is_diagonal());
    }
}

/// One block of `branching/branching_covariance.arrow`: an MF=40 covariance of
/// one product state's MF=10 partial with another's, with the key the
/// converter put in front of it.
///
/// Every tape value is as the tape gives it, so `mat1` may name the evaluation
/// itself: compare it with `mat`, not with zero, before taking a block for a
/// cross-material one (JEFF-4.0 U235 MT 4). `izap` may be 0 for the target
/// itself on MT 4, and `lfs` need not be MF=10's number for the same level;
/// `target` is where the converter matched it.
///
/// The block is a [`CovarianceBlock`] exactly as `covariance.arrow` reads,
/// because MF=40 writes its sub-subsections in MF=33's format. Its `xmf1`,
/// `xlfs1`, `mat1` and `mt1` name the partner state, and its `subsection_idx`
/// is the sub-subsection's position within the product state. Its `mtl` is
/// always 0 and means nothing here: MF=40 has no lumped-reaction flag, and the
/// file stores the column as null.
///
/// So [`CovarianceBlock`]'s own helpers, which read MF=33's conventions, are
/// wrong on it and are not to be called through `block`: `partner_mt` reads
/// an MT1 of 0 as this MT, which the manual gives no meaning in MF=40, and
/// `is_diagonal` and `is_same_evaluation` take only a cross section (XMF1 0
/// or 3, XLFS1 0) for a partner, so they are false on every MF=40 block. Use
/// [`BranchingCovarianceBlock::is_self_block`] and
/// [`BranchingCovarianceBlock::is_cross_material`] instead. The block's `mat`
/// is the key's `mat`, read from the same column.
///
/// Several levels can resolve to one chain nuclide, so (`target`, `target1`)
/// does not identify the pair of states a block correlates: key on (`mt`,
/// `lfs`, `mt1`, `xlfs1`). JEFF-4.0 U235 MT 4 has a block between its ground
/// (LFS 0) and its 77 eV isomer (XLFS1 1), and both resolve to U235.
///
/// A product state with no sub-subsection, or a sub-subsection with no block,
/// has no block here: it holds no covariance, and the converter lists it under
/// `mf40_without_blocks` in `branching/provenance.json` rather than in the file.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchingCovarianceBlock {
    /// The parent, as in `branching.arrow`.
    pub nuclide: String,
    /// The chain reaction kind of the section's MT, `None` for an MT with none.
    pub reaction: Option<String>,
    /// The chain nuclide this block's product state is the partial of, `None`
    /// when the converter matched it to no MF=9 or MF=10 state.
    pub target: Option<String>,
    /// The chain nuclide the partner state resolved to, `None` when it is in
    /// another material, `xmf1` is not 10, `mt1` is 0, or the tape does not
    /// pin level `xlfs1` of `mt1` to one state.
    pub target1: Option<String>,
    /// This state's own curve, linearized as `branching.arrow` has it, when
    /// several states share `target` in the same `quantity` and the
    /// `branching.arrow` curve is their sum: its MF=10 partial, or its MF=9
    /// yield. `None` means the `branching.arrow` curve is this state's own.
    pub energy: Option<Vec<f64>>,
    pub values: Option<Vec<f64>>,
    /// `"cross_section"` or `"yield"`, as in `branching.arrow`: which of
    /// `target`'s curves is this state's, set for every placed state. `None`
    /// only when `target` is `None`. With `energy` and `values` null it names
    /// the `branching.arrow` row that is the state's own curve.
    pub quantity: Option<String>,
    /// The evaluation's own MAT.
    pub mat: i32,
    /// The section HEAD.
    pub za: i32,
    pub awr: f64,
    pub lis: i32,
    /// The product state's position in the section, and its CONT.
    pub state_idx: i32,
    pub qm: f64,
    pub qi: f64,
    pub izap: i32,
    pub lfs: i32,
    /// The block itself.
    pub block: CovarianceBlock,
}

impl BranchingCovarianceBlock {
    /// Whether this block is its product state's covariance with itself: an
    /// MF=10 partial of this evaluation (MAT1 0 or `mat`, XMF1 10) at this
    /// state's own MT and level.
    ///
    /// MT1 is compared as written, so an MT1 of 0 is not read as this MT, and
    /// the level is XLFS1 against `lfs`, both as the tape numbers them. The
    /// converter's `mf40_cross_state_blocks` counts every block this is false
    /// for.
    pub fn is_self_block(&self) -> bool {
        !self.is_cross_material()
            && self.block.xmf1 == 10.0
            && self.block.mt1 == self.block.mt
            && self.block.xlfs1 == self.lfs as f64
    }

    /// Whether the partner state is in another evaluation: MAT1 is neither 0
    /// nor this evaluation's own MAT, which JEFF-4.0 U235 MT 4 writes there.
    pub fn is_cross_material(&self) -> bool {
        self.block.mat1 != 0 && self.block.mat1 != self.mat
    }
}
