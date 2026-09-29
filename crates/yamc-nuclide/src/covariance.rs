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
/// one given explicitly.
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceData {
    /// Given explicitly. `lb` selects the layout within it.
    Ni(NiSubsection),
    /// Derived from other reactions. `lty` selects the derivation.
    Nc(NcSubsection),
}

/// One covariance block: the covariance of `mt` with `mt1`, on one grid.
///
/// # Which blocks are symmetric
///
/// Only the ones with `mat1 == 0 && (mt1 == 0 || mt1 == mt)`. An off-diagonal
/// block's transpose is the (`mt1`, `mt`) block, not itself, so nothing here
/// may be mirrored on the assumption that a covariance matrix is symmetric. The
/// same caution applies one level down, inside an `Ni` block: `ls == 1` is an
/// upper triangle whose transpose is implied, `ls == 0` is a full asymmetric
/// matrix as written, and folding one into the other loses numbers.
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
    /// The material `mt1` belongs to. `0` means this evaluation.
    pub mat1: i32,
    /// The reaction this one is correlated with. `0` means `mt` itself.
    pub mt1: i32,
    /// MF of the second quantity, when it is not a cross section.
    pub xmf1: f64,
    /// Final excited state of the second quantity.
    pub xlfs1: f64,
    /// MT this reaction is lumped into, from the section HEAD. `0` when it is
    /// not lumped.
    pub mtl: i32,
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
    /// only ones a variance can be read off directly. Blind to XLFS1, so not
    /// for MF=40: use [`BranchingCovarianceBlock::is_self_block`].
    pub fn is_diagonal(&self) -> bool {
        self.mat1 == 0 && self.partner_mt() == self.mt
    }

    /// Whether this block correlates with a reaction in ANOTHER evaluation.
    ///
    /// Not consumed today: using it would mean sampling two nuclides' cross
    /// sections from one joint distribution, and the fold is per nuclide.
    /// Counted and reported rather than silently dropped. MF=40 can write the
    /// evaluation's own MAT here, so use
    /// [`BranchingCovarianceBlock::is_cross_material`] for it.
    pub fn is_cross_material(&self) -> bool {
        self.mat1 != 0
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
/// an MT1 of 0 as this MT, which the manual gives no meaning in MF=40;
/// `is_diagonal` never compares XLFS1 with the state's LFS, so it takes the
/// rectangular ground-to-isomer block of JEFF-4.0 U235 MT 4 for a self block;
/// and `is_cross_material` takes JEFF-4.0 U235's MAT1 of its own 9228 for
/// another material. Use [`BranchingCovarianceBlock::is_self_block`] and
/// [`BranchingCovarianceBlock::is_cross_material`] instead.
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
    /// `"cross_section"` or `"yield"`, as in `branching.arrow`, whenever
    /// `energy` and `values` are set.
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
