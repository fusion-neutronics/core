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
    /// rather than leaving every caller to remember the convention.
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
    /// only ones a variance can be read off directly.
    pub fn is_diagonal(&self) -> bool {
        self.mat1 == 0 && self.partner_mt() == self.mt
    }

    /// Whether this block correlates with a reaction in ANOTHER evaluation.
    ///
    /// Not consumed today: using it would mean sampling two nuclides' cross
    /// sections from one joint distribution, and the fold is per nuclide.
    /// Counted and reported rather than silently dropped.
    pub fn is_cross_material(&self) -> bool {
        self.mat1 != 0
    }
}
