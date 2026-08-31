//! Turn an MF=33 block into a matrix on the grid the evaluator wrote it on.
//!
//! `lb` selects the layout, and the layouts are not variations on a theme: one
//! is a diagonal, one is an outer product, one is a full matrix, one is
//! rectangular, and two of them are not symmetric. This module is separate from
//! the reader for that reason, so the interpretation is somewhere it can be
//! tested against hand-computed matrices rather than buried in a column loop.
//!
//! # No regridding happens here
//!
//! Each block keeps its own energy grid. Nothing is projected onto a common
//! structure and blocks are never summed together, because they do not need to
//! be: a reaction rate is linear in the cross section, so the covariance of two
//! collapsed rates is
//!
//! ```text
//! Cov(R_i, R_j) = Σ_blocks  r_i(b)ᵀ · C(b) · r_j(b)
//! ```
//!
//! with `r_i(b)[k]` the partial rate of reaction `i` over block `b`'s own
//! interval `k`. Every block contributes on its own grid and the contributions
//! add. A union grid would be arithmetic with no purpose.
//!
//! # Relative versus absolute
//!
//! `lb = 0` is an absolute covariance in barns squared; every other layout is
//! relative. The distinction is carried out on [`Scale`] rather than resolved
//! here, because relativizing needs the cross section and this module
//! deliberately does not have it.

use endf::mf::covariance::NiSubsection;

/// What the values in an [`ExpandedBlock`] are covariances OF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Relative covariance: divide out the cross sections already.
    /// `Cov(σ_i, σ_j) = value · σ_i · σ_j`.
    Relative,
    /// Absolute covariance, in barns squared. `lb = 0` only.
    Absolute,
}

/// A block's covariance as a matrix, on its own row and column grids.
///
/// The grids are BOUNDARIES: `row_energies` has `n_rows + 1` entries and
/// `values[i * n_cols + j]` is the covariance between row interval
/// `[row_energies[i], row_energies[i+1])` and column interval
/// `[col_energies[j], col_energies[j+1])`.
///
/// Rows and columns are separate because two layouts need them to be: `lb = 3`
/// pairs the first (E, F) table against the second, and `lb = 6` is explicitly
/// rectangular. For the rest the two grids are the same vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedBlock {
    pub row_energies: Vec<f64>,
    pub col_energies: Vec<f64>,
    /// Row-major, `n_rows * n_cols`.
    pub values: Vec<f64>,
    pub scale: Scale,
}

impl ExpandedBlock {
    pub fn n_rows(&self) -> usize {
        self.row_energies.len().saturating_sub(1)
    }

    pub fn n_cols(&self) -> usize {
        self.col_energies.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The covariance between row interval `i` and column interval `j`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.n_cols() + j]
    }
}

/// Why a block could not be turned into a matrix.
///
/// Returned rather than silently producing zeros. A caller that cannot expand a
/// block has less uncertainty than the evaluation states, and that has to be
/// reportable: a missing contribution and a genuinely zero one must never look
/// the same downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// An `lb` layout this does not implement.
    Layout(i64),
    /// A block whose arrays do not match the sizes its own header declares.
    Malformed,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::Layout(lb) => write!(f, "MF=33 LB={lb} is not implemented"),
            Unsupported::Malformed => {
                write!(f, "the block's arrays do not match its declared sizes")
            }
        }
    }
}

/// The number of intervals a boundary vector defines.
fn intervals(grid: &[f64]) -> usize {
    grid.len().saturating_sub(1)
}

/// Expand one NI block.
///
/// # Layouts
///
/// | `lb` | matrix | scale |
/// | --- | --- | --- |
/// | 0 | `C[k,k] = fk[k]`, diagonal | absolute |
/// | 1 | `C[k,k] = fk[k]`, diagonal | relative |
/// | 2 | `C[k,l] = fk[k]·fk[l]`, fully correlated | relative |
/// | 3 | `C[k,l] = fk[k]·fl[l]`, first table against second | relative |
/// | 4 | `C[k,l] = fk[k]·fk[l]·Σ fl[m]` over second-table intervals holding both | relative |
/// | 5 | `fkk` as written; `ls=1` an upper triangle, `ls=0` a full matrix | relative |
/// | 6 | `fkl` on `er` × `ec`, rectangular | relative |
/// | 8 | `C[k,k] = fk[k]`, diagonal, short-range | relative |
///
/// `lb = 9` is not implemented: the parser reads it with `lb = 8`'s layout, but
/// its meaning is not the same and guessing would put numbers in a covariance
/// matrix on the strength of a shared record shape.
pub fn expand_ni(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    match s.lb {
        0 => diagonal(s, Scale::Absolute),
        1 => diagonal(s, Scale::Relative),
        2 => outer_product(s),
        3 => two_table_product(s),
        4 => interval_weighted(s),
        5 => explicit(s),
        6 => rectangular(s),
        // Short-range self-scaling. Its magnitude is defined relative to the
        // width of the interval the cross section is averaged over, and the
        // fold integrates over the evaluation's own intervals, so the ratio is
        // one and the contribution is `fk` on the diagonal. Averaging over
        // anything wider would give less; this is the fine-grid value.
        8 => diagonal(s, Scale::Relative),
        other => Err(Unsupported::Layout(other)),
    }
}

/// `lb` 0, 1 and 8: one value per interval, uncorrelated between them.
fn diagonal(s: &NiSubsection, scale: Scale) -> Result<ExpandedBlock, Unsupported> {
    let n = intervals(&s.ek);
    if s.fk.len() < n {
        return Err(Unsupported::Malformed);
    }
    let mut values = vec![0.0; n * n];
    for k in 0..n {
        values[k * n + k] = s.fk[k];
    }
    Ok(ExpandedBlock {
        row_energies: s.ek.clone(),
        col_energies: s.ek.clone(),
        values,
        scale,
    })
}

/// `lb = 2`: one table, fully correlated across the whole range.
fn outer_product(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n = intervals(&s.ek);
    if s.fk.len() < n {
        return Err(Unsupported::Malformed);
    }
    let mut values = vec![0.0; n * n];
    for k in 0..n {
        for l in 0..n {
            values[k * n + l] = s.fk[k] * s.fk[l];
        }
    }
    Ok(ExpandedBlock {
        row_energies: s.ek.clone(),
        col_energies: s.ek.clone(),
        values,
        scale: Scale::Relative,
    })
}

/// `lb = 3`: rows from the first (E, F) table, columns from the second.
///
/// Asymmetric and generally not square. This is a cross-reaction layout, and
/// its transpose is the partner subsection's own block rather than anything
/// recoverable from here.
fn two_table_product(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n_rows = intervals(&s.ek);
    let n_cols = intervals(&s.el);
    if s.fk.len() < n_rows || s.fl.len() < n_cols {
        return Err(Unsupported::Malformed);
    }
    let mut values = vec![0.0; n_rows * n_cols];
    for k in 0..n_rows {
        for l in 0..n_cols {
            values[k * n_cols + l] = s.fk[k] * s.fl[l];
        }
    }
    Ok(ExpandedBlock {
        row_energies: s.ek.clone(),
        col_energies: s.el.clone(),
        values,
        scale: Scale::Relative,
    })
}

/// `lb = 4`: the first table's fractions, correlated only within each of the
/// second table's intervals.
///
/// Two first-table intervals are correlated exactly when some second-table
/// interval contains both, and the strength is that interval's own `fl`. Where
/// several do, they add, which is why this accumulates rather than assigns.
fn interval_weighted(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n = intervals(&s.ek);
    let n_weight = intervals(&s.el);
    if s.fk.len() < n || s.fl.len() < n_weight {
        return Err(Unsupported::Malformed);
    }

    // Which weighting interval each first-table interval falls in. Membership
    // is by the interval's own span, so an interval straddling a weighting
    // boundary belongs to every weighting interval it overlaps.
    let mut values = vec![0.0; n * n];
    for m in 0..n_weight {
        let (lo, hi) = (s.el[m], s.el[m + 1]);
        let inside: Vec<usize> = (0..n)
            .filter(|&k| s.ek[k] < hi && s.ek[k + 1] > lo)
            .collect();
        for &k in &inside {
            for &l in &inside {
                values[k * n + l] += s.fl[m] * s.fk[k] * s.fk[l];
            }
        }
    }

    Ok(ExpandedBlock {
        row_energies: s.ek.clone(),
        col_energies: s.ek.clone(),
        values,
        scale: Scale::Relative,
    })
}

/// `lb = 5`: the matrix as written, in the format's packed order.
///
/// `ls = 1` is the upper triangle including the diagonal, row by row, and its
/// transpose is implied. `ls = 0` is the full matrix, row-major, and is NOT
/// symmetric: mirroring it would overwrite half its numbers with the other
/// half's.
fn explicit(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n = intervals(&s.ek);
    let mut values = vec![0.0; n * n];

    if s.ls == 0 {
        if s.fkk.len() < n * n {
            return Err(Unsupported::Malformed);
        }
        values.copy_from_slice(&s.fkk[..n * n]);
    } else {
        if s.fkk.len() < n * (n + 1) / 2 {
            return Err(Unsupported::Malformed);
        }
        let mut at = 0;
        for k in 0..n {
            for l in k..n {
                let v = s.fkk[at];
                values[k * n + l] = v;
                values[l * n + k] = v;
                at += 1;
            }
        }
    }

    Ok(ExpandedBlock {
        row_energies: s.ek.clone(),
        col_energies: s.ek.clone(),
        values,
        scale: Scale::Relative,
    })
}

/// `lb = 6`: a rectangular matrix on two independent grids.
///
/// Never square and never symmetric, by construction: `er` carries the rows and
/// `ec` the columns, and the two describe different reactions.
fn rectangular(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n_rows = intervals(&s.er);
    let n_cols = intervals(&s.ec);
    if s.fkl.len() < n_rows * n_cols {
        return Err(Unsupported::Malformed);
    }
    Ok(ExpandedBlock {
        row_energies: s.er.clone(),
        col_energies: s.ec.clone(),
        values: s.fkl[..n_rows * n_cols].to_vec(),
        scale: Scale::Relative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layout is checked against a matrix small enough to write out by
    /// hand, because the fixtures available here use only `lb = 5`: without
    /// these the other layouts would be code nothing has ever run.
    fn ni(lb: i64) -> NiSubsection {
        NiSubsection {
            lb,
            ..Default::default()
        }
    }

    #[test]
    fn lb0_is_a_diagonal_on_the_absolute_scale() {
        let mut s = ni(0);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![0.04, 0.09];
        let e = expand_ni(&s).expect("lb=0 expands");
        assert_eq!(e.scale, Scale::Absolute);
        assert_eq!(e.values, vec![0.04, 0.0, 0.0, 0.09]);
    }

    #[test]
    fn lb1_is_the_same_matrix_on_the_relative_scale() {
        let mut s = ni(1);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![0.04, 0.09];
        let e = expand_ni(&s).expect("lb=1 expands");
        assert_eq!(e.scale, Scale::Relative);
        assert_eq!(e.values, vec![0.04, 0.0, 0.0, 0.09]);
    }

    #[test]
    fn lb2_is_the_outer_product_of_one_table() {
        let mut s = ni(2);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![2.0, 3.0];
        let e = expand_ni(&s).expect("lb=2 expands");
        assert_eq!(e.values, vec![4.0, 6.0, 6.0, 9.0]);
    }

    #[test]
    fn lb3_pairs_the_first_table_against_the_second_and_is_not_square() {
        let mut s = ni(3);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![2.0, 3.0];
        s.el = vec![1.0, 50.0, 500.0, 5000.0];
        s.fl = vec![5.0, 7.0, 11.0];
        let e = expand_ni(&s).expect("lb=3 expands");
        assert_eq!((e.n_rows(), e.n_cols()), (2, 3));
        assert_eq!(e.values, vec![10.0, 14.0, 22.0, 15.0, 21.0, 33.0]);
    }

    #[test]
    fn lb4_correlates_only_within_a_second_table_interval() {
        let mut s = ni(4);
        // Three first-table intervals: [1,10), [10,100), [100,1000).
        s.ek = vec![1.0, 10.0, 100.0, 1000.0];
        s.fk = vec![1.0, 2.0, 3.0];
        // One weighting interval covering only the first two.
        s.el = vec![1.0, 100.0];
        s.fl = vec![0.5];
        let e = expand_ni(&s).expect("lb=4 expands");
        assert_eq!(e.n_rows(), 3);
        // 0.5 * fk[i] * fk[j] for i, j in {0, 1}; the third interval is outside
        // the weighting interval and so correlates with nothing, itself
        // included.
        assert_eq!(e.values, vec![0.5, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn lb5_ls1_unpacks_an_upper_triangle_and_mirrors_it() {
        let mut s = ni(5);
        s.ls = 1;
        s.ek = vec![1.0, 10.0, 100.0, 1000.0];
        // Upper triangle of a 3x3, row by row: (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)
        s.fkk = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e = expand_ni(&s).expect("lb=5 ls=1 expands");
        assert_eq!(e.values, vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]);
    }

    /// The case a global "dense upper triangle" layout would have destroyed.
    #[test]
    fn lb5_ls0_keeps_an_asymmetric_matrix_asymmetric() {
        let mut s = ni(5);
        s.ls = 0;
        s.ek = vec![1.0, 10.0, 100.0];
        s.fkk = vec![1.0, 2.0, 3.0, 4.0];
        let e = expand_ni(&s).expect("lb=5 ls=0 expands");
        assert_eq!(e.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_ne!(e.get(0, 1), e.get(1, 0), "ls=0 must not be symmetrized");
    }

    #[test]
    fn lb6_is_rectangular_on_two_grids() {
        let mut s = ni(6);
        s.er = vec![1.0, 10.0, 100.0];
        s.ec = vec![1.0, 5.0, 25.0, 125.0];
        s.fkl = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e = expand_ni(&s).expect("lb=6 expands");
        assert_eq!((e.n_rows(), e.n_cols()), (2, 3));
        assert_eq!(e.get(1, 2), 6.0);
    }

    #[test]
    fn lb8_is_a_relative_diagonal() {
        let mut s = ni(8);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![0.01, 0.02];
        let e = expand_ni(&s).expect("lb=8 expands");
        assert_eq!(e.scale, Scale::Relative);
        assert_eq!(e.values, vec![0.01, 0.0, 0.0, 0.02]);
    }

    #[test]
    fn lb9_is_reported_rather_than_guessed() {
        assert_eq!(expand_ni(&ni(9)), Err(Unsupported::Layout(9)));
    }

    #[test]
    fn a_block_shorter_than_its_header_claims_is_malformed_not_padded() {
        let mut s = ni(5);
        s.ls = 1;
        s.ek = vec![1.0, 10.0, 100.0, 1000.0];
        s.fkk = vec![1.0, 2.0];
        assert_eq!(expand_ni(&s), Err(Unsupported::Malformed));
    }
}
