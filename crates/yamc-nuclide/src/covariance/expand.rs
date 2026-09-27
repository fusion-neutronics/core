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
    /// A block whose arrays do not match the sizes its own header declares,
    /// or the layout its `lb` names.
    Malformed,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::Layout(lb) => write!(f, "MF=33 LB={lb} is not implemented"),
            Unsupported::Malformed => {
                write!(
                    f,
                    "the block's arrays do not match its declared sizes or its LB"
                )
            }
        }
    }
}

/// The number of intervals a boundary vector defines.
fn intervals(grid: &[f64]) -> usize {
    grid.len().saturating_sub(1)
}

/// Whether an `lb` 0 to 4 block's two tables hold exactly the pairs its
/// `np` and `lt` declare.
fn tables_match_header(s: &NiSubsection) -> bool {
    let (Ok(first), Ok(second)) = (usize::try_from(s.np - s.lt), usize::try_from(s.lt)) else {
        return false;
    };
    s.ek.len() == first && s.fk.len() == first && s.el.len() == second && s.fl.len() == second
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
/// | 4 | `C[u,v] = fk[k]·fl[l(u)]·fl[l(v)]` when one first-table interval `k` holds both, on the union of the two grids | relative |
/// | 5 | `fkk` as written; `ls=1` an upper triangle, `ls=0` a full matrix | relative |
/// | 6 | `fkl` on `er` × `ec`, rectangular | relative |
/// | 8 | `C[k,k] = fk[k]`, diagonal, short-range | relative |
///
/// `lb = 9` is not implemented: the parser reads it with `lb = 8`'s layout, but
/// its meaning is not the same and guessing would put numbers in a covariance
/// matrix on the strength of a shared record shape.
///
/// `lb` 0 to 4 store `np - lt` (E, F) pairs in the first table and `lt` in
/// the second (ENDF-102 section 33.2.2.2), and a block whose tables are not
/// exactly those lengths is malformed rather than expanded from whatever it
/// holds. `lb` 0 to 2 have one table, so `lt` must also be 0. A block split
/// at the wrong place arrives exactly this way: every `lb` 0 to 2 block in a
/// `covariance.arrow` converted before fusion-neutronics/core#166 was fixed
/// has the upper part of its only table in `el`/`fl`, and folding `ek` alone
/// would drop it without a word.
pub fn expand_ni(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    match s.lb {
        0..=4 if !tables_match_header(s) => Err(Unsupported::Malformed),
        0..=2 if s.lt != 0 => Err(Unsupported::Malformed),
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

/// `lb = 4`: the second table's fractions, fully correlated within each of the
/// first table's intervals and weighted by that interval's `fk`.
///
/// ENDF-102 section 33.2.2.2 defines the contribution as
/// `Σ_{k,l,l'} S_ik S_il S_jk S_jl' · Fk · Fl · Fl'`: two energies are
/// correlated exactly when one first-table interval holds both, and the
/// strength is that interval's `Fk` times the second-table fraction each one
/// falls in. `Fk` is not squared, and it can be negative, which is how an
/// evaluator takes out part of what another block states; its sign is kept.
///
/// The value is constant wherever neither table has a boundary, so the matrix
/// is written on the union of the two grids, over the range both tables span.
/// That is still the block's own grid rather than a regridding: every union
/// interval lies inside one interval of each table, so every entry is exact,
/// including where a second-table interval straddles a first-table boundary.
/// Outside the common range one of the `S` factors is zero, and so is the
/// contribution.
fn interval_weighted(s: &NiSubsection) -> Result<ExpandedBlock, Unsupported> {
    let n_outer = intervals(&s.ek);
    let n_inner = intervals(&s.el);
    if s.fk.len() < n_outer || s.fl.len() < n_inner {
        return Err(Unsupported::Malformed);
    }
    // Both tables are boundaries in ascending order on any tape. Checked
    // rather than assumed, because the interval lookup below bisects them.
    let ascending = |t: &[f64]| t.windows(2).all(|w| w[0] <= w[1]);
    if !ascending(&s.ek) || !ascending(&s.el) {
        return Err(Unsupported::Malformed);
    }
    if n_outer == 0 || n_inner == 0 {
        return Ok(ExpandedBlock {
            row_energies: Vec::new(),
            col_energies: Vec::new(),
            values: Vec::new(),
            scale: Scale::Relative,
        });
    }

    let lo = s.ek[0].max(s.el[0]);
    let hi = s.ek[n_outer].min(s.el[n_inner]);
    let mut grid: Vec<f64> =
        s.ek.iter()
            .chain(&s.el)
            .copied()
            .filter(|&e| e >= lo && e <= hi)
            .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let n = intervals(&grid);

    // The interval of `table` that union interval `u` lies in. Its left edge
    // is inside the common range, so the index is always a real interval; a
    // repeated energy in `table` is a zero-width interval, and taking the last
    // boundary at or below the edge steps over it.
    let within = |table: &[f64], u: usize| table.partition_point(|&e| e <= grid[u]) - 1;
    let outer: Vec<usize> = (0..n).map(|u| within(&s.ek, u)).collect();
    let inner: Vec<f64> = (0..n).map(|u| s.fl[within(&s.el, u)]).collect();

    let mut values = vec![0.0; n * n];
    for u in 0..n {
        for v in 0..n {
            if outer[u] == outer[v] {
                values[u * n + v] = s.fk[outer[u]] * inner[u] * inner[v];
            }
        }
    }

    Ok(ExpandedBlock {
        row_energies: grid.clone(),
        col_energies: grid,
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
    /// hand. The real-tape fixtures reach only `lb` 0, 1, 4, 5 and 8 (see
    /// `yamc-convert/tests/covariance.rs`), so without these the other layouts
    /// would be code nothing has ever run.
    fn ni(lb: i64) -> NiSubsection {
        NiSubsection {
            lb,
            ..Default::default()
        }
    }

    /// An `lb` 0 to 4 block as a tape carries it: the tests write one `F` per
    /// interval, and a tape pairs every boundary with an `F`, the last one
    /// zero, and states both tables' lengths in `np` and `lt`.
    fn on_tape(mut s: NiSubsection) -> NiSubsection {
        if !s.ek.is_empty() {
            s.fk.push(0.0);
        }
        if !s.el.is_empty() {
            s.fl.push(0.0);
        }
        s.lt = s.el.len() as i64;
        s.np = (s.ek.len() + s.el.len()) as i64;
        s
    }

    #[test]
    fn lb0_is_a_diagonal_on_the_absolute_scale() {
        let mut s = ni(0);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![0.04, 0.09];
        let e = expand_ni(&on_tape(s)).expect("lb=0 expands");
        assert_eq!(e.scale, Scale::Absolute);
        assert_eq!(e.values, vec![0.04, 0.0, 0.0, 0.09]);
    }

    #[test]
    fn lb1_is_the_same_matrix_on_the_relative_scale() {
        let mut s = ni(1);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![0.04, 0.09];
        let e = expand_ni(&on_tape(s)).expect("lb=1 expands");
        assert_eq!(e.scale, Scale::Relative);
        assert_eq!(e.values, vec![0.04, 0.0, 0.0, 0.09]);
    }

    #[test]
    fn lb2_is_the_outer_product_of_one_table() {
        let mut s = ni(2);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![2.0, 3.0];
        let e = expand_ni(&on_tape(s)).expect("lb=2 expands");
        assert_eq!(e.values, vec![4.0, 6.0, 6.0, 9.0]);
    }

    #[test]
    fn lb3_pairs_the_first_table_against_the_second_and_is_not_square() {
        let mut s = ni(3);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![2.0, 3.0];
        s.el = vec![1.0, 50.0, 500.0, 5000.0];
        s.fl = vec![5.0, 7.0, 11.0];
        let e = expand_ni(&on_tape(s)).expect("lb=3 expands");
        assert_eq!((e.n_rows(), e.n_cols()), (2, 3));
        assert_eq!(e.values, vec![10.0, 14.0, 22.0, 15.0, 21.0, 33.0]);
    }

    #[test]
    fn lb4_weights_the_second_table_by_the_first() {
        let mut s = ni(4);
        // Two first-table intervals: [1,100) and [100,1000).
        s.ek = vec![1.0, 100.0, 1000.0];
        s.fk = vec![0.5, 2.0];
        // Three second-table intervals, two inside the first of those.
        s.el = vec![1.0, 10.0, 100.0, 1000.0];
        s.fl = vec![1.0, 2.0, 3.0];
        let e = expand_ni(&on_tape(s)).expect("lb=4 expands");
        assert_eq!(e.row_energies, vec![1.0, 10.0, 100.0, 1000.0]);
        assert_eq!(e.col_energies, e.row_energies);
        // fk[k] * fl[l] * fl[l'] where one first-table interval holds both:
        // the first two intervals share k = 0, and the third is alone in
        // k = 1, so it correlates only with itself.
        assert_eq!(e.values, vec![0.5, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 18.0]);
    }

    /// The only LB=4 block on any tape (FENDL-3.2d Ni58 MT=103) has the same
    /// grid in both tables and a negative `fk` in every interval (mostly -1,
    /// down to -0.21 near 14 MeV): it subtracts `|fk|·fl²` on the diagonal.
    /// Squaring `fk`, or taking `fl` as the weight, turns that into an addition.
    #[test]
    fn lb4_keeps_the_sign_of_a_negative_weight() {
        let mut s = ni(4);
        s.ek = vec![1.0, 10.0, 100.0];
        s.fk = vec![-1.0, -0.21];
        s.el = s.ek.clone();
        s.fl = vec![0.1, 0.2];
        let e = expand_ni(&on_tape(s)).expect("lb=4 expands");
        assert_eq!(e.n_rows(), 2);
        assert_eq!(e.get(0, 0), -(0.1 * 0.1));
        assert_eq!(e.get(1, 1), -0.21 * 0.2 * 0.2);
        assert_eq!((e.get(0, 1), e.get(1, 0)), (0.0, 0.0));
    }

    /// A second-table interval that straddles a first-table boundary is split
    /// there: each half is correlated only with what shares its own `k`.
    #[test]
    fn lb4_splits_an_interval_that_straddles_a_first_table_boundary() {
        let mut s = ni(4);
        s.ek = vec![1.0, 50.0, 100.0];
        s.fk = vec![1.0, 2.0];
        s.el = vec![1.0, 100.0];
        s.fl = vec![3.0];
        let e = expand_ni(&on_tape(s)).expect("lb=4 expands");
        assert_eq!(e.row_energies, vec![1.0, 50.0, 100.0]);
        assert_eq!(e.values, vec![9.0, 0.0, 0.0, 18.0]);
    }

    /// Outside the range both tables span, one of the indicator factors is
    /// zero, so the matrix covers only the common range.
    #[test]
    fn lb4_is_written_only_where_both_tables_are_defined() {
        let mut s = ni(4);
        s.ek = vec![1.0, 10.0];
        s.fk = vec![2.0];
        s.el = vec![5.0, 20.0];
        s.fl = vec![0.5];
        let e = expand_ni(&on_tape(s)).expect("lb=4 expands");
        assert_eq!(e.row_energies, vec![5.0, 10.0]);
        assert_eq!(e.values, vec![2.0 * 0.5 * 0.5]);
    }

    /// What the old parser split made of ENDF/B-VIII.1 Ni58 MT=107's LB=1
    /// block with NP=3: half a table in each array. Folding `ek` alone would
    /// have dropped the 10.6% component on [0.8, 20] MeV without a word.
    #[test]
    fn lb0_to_2_with_a_second_table_is_malformed_not_truncated() {
        for lb in 0..=2 {
            let mut s = ni(lb);
            s.np = 3;
            s.ek = vec![1.0e-5, 8.0e5];
            s.fk = vec![0.0];
            s.el = vec![1.125e-2, 0.0];
            s.fl = vec![2.0e7];
            assert_eq!(expand_ni(&s), Err(Unsupported::Malformed), "lb={lb}");

            // Tables that agree with the header, but a second table the
            // layout does not have.
            let mut s = ni(lb);
            s.ek = vec![1.0, 10.0];
            s.fk = vec![0.5, 0.0];
            s.el = vec![1.0];
            s.fl = vec![0.0];
            s.lt = 1;
            s.np = 3;
            assert_eq!(expand_ni(&s), Err(Unsupported::Malformed), "lb={lb}, lt=1");
        }
    }

    /// A split that disagrees with `np` and `lt` is caught for the two-table
    /// layouts too, not only where it leaves a stray second table.
    #[test]
    fn lb3_and_lb4_tables_that_disagree_with_the_header_are_malformed() {
        for lb in 3..=4 {
            // np = 5, lt = 1 puts four pairs in the first table; the old
            // NT - NP split put 2.5 pairs there.
            let mut s = ni(lb);
            s.np = 5;
            s.lt = 1;
            s.ek = vec![1.0, 10.0];
            s.fk = vec![0.5, 0.2];
            s.el = vec![100.0, 1000.0, 10.0];
            s.fl = vec![0.0, 0.1, 0.0];
            assert_eq!(expand_ni(&s), Err(Unsupported::Malformed), "lb={lb}");
        }
    }

    #[test]
    fn lb4_tables_out_of_order_are_malformed() {
        let mut s = ni(4);
        s.ek = vec![10.0, 1.0];
        s.fk = vec![1.0];
        s.el = vec![1.0, 10.0];
        s.fl = vec![1.0];
        assert_eq!(expand_ni(&on_tape(s)), Err(Unsupported::Malformed));
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
