//! The isomeric split's uncertainty, from MF=40: each final state's production
//! cross section drawn from its stated covariance, and the split re-folded from
//! the draw by the nominal branching rule.
//!
//! # What MF=40 states
//!
//! ENDF-6 File 40 (ENDF-102 chapter 40) is the covariance of the radionuclide
//! production cross sections File 10 tabulates. A section is one reaction MT;
//! within it each final state of the product (IZAP, LFS) has sub-subsections in
//! File 33's format (NI blocks with the LB layouts, NC blocks), each correlating
//! that state's production with a partner's: another final state, named by
//! XMF1 = 10, MT1 and XLFS1, of the same or another reaction, or with a cross
//! section of another evaluation (MAT1). Relative layouts (LB 1 to 6) are
//! relative to the partials; LB=0 is absolute and LB=8 an absolute short-range
//! variance, as in MF=33.
//!
//! The converter writes every block as `branching/branching_covariance.arrow`
//! (see [`yamc_nuclide::covariance::BranchingCovarianceBlock`]), with the chain
//! nuclide each state was placed on and, where several states make one chain
//! nuclide and `branching.arrow` carries their sum, the state's own curve. In
//! the published libraries nearly every block is an LB=5 block of one state
//! with itself: ENDF/B-VIII.1 states 29 parents (Nb93 and In115 `(n,n')`, In115
//! `(n,2n)` to In114_m1), JEFF-4.0 440 parents and 6512 blocks (a ground and
//! an isomer for most reactions of most evaluations, one cross-state block,
//! U235 MT=4), and IRDFF-II 25 blocks, of which the MT=5 ones place on no
//! state.
//!
//! # How a draw moves the split
//!
//! The nominal split is [`crate::branching_rule`]'s: the branching evaluation
//! gives each final state's production curve, the transport library gives the
//! reaction's total, and what the curve means depends on the list (see the
//! rule's module documentation). A replica draws each stated state's
//! production and applies that same rule to the draw, so the split stays a
//! split however wide the draw:
//!
//! * A list that names its ground state (a share list): each state's share is
//!   its drawn production over the drawn sum of the listed states, and the
//!   reaction's rate stays the transport total, with its MF=33 draw. Shares
//!   stay in [0, 1] and sum to one. A factor common to every listed state
//!   cancels, so the part of MF=40 that is the uncertainty of the reaction's
//!   total is not counted a second time on top of MF=33's: only the split
//!   moves.
//! * A list of isomers only, and every `(n,n')` list: a stated isomer's
//!   production is the evaluation's absolute partial, so the isomer is made at
//!   its drawn production whatever the replica's MF=33 draw did to the
//!   reaction total, and the ground state takes the rest. Above the drawn
//!   total the stated productions are scaled back to it, and each is
//!   counted by name (`Info::isomer_branchings_clipped`). A state with no MF=40 keeps
//!   its nominal share of the total, as without this source, and so moves
//!   with the total's MF=33 draw.
//!
//! The second is the one place the two sources meet. Without MF=40, holding
//! the share of an isomer-only list lends the isomer its reaction's MF=33
//! uncertainty, a stand-in for the uncertainty its evaluation states nowhere;
//! that is what the cross-section source alone still does. Where MF=40 states
//! the isomer's own, it replaces the stand-in rather than adding to it, since
//! the two are not independent and their correlation is stated nowhere either
//! (no published block correlates an MF=40 state with an MF=33 total). So for
//! such an isomer the sigma with both sources on is MF=40's, it can be smaller
//! than with the cross sections alone, and `Attribution::by_source` counts
//! it under both. A share list never meets MF=33 that way: its ground and
//! isomers share one total, which only MF=33 moves.
//!
//! # Folding against the spectrum
//!
//! MF=40 is constant on each interval of its own grid, as MF=33 is, so a
//! state's drawn production is its nominal one plus `sum_j w_j (m_j - 1)`,
//! `m_j` the multiplier on interval `j` and `w_j` the state's nominal
//! production there: the production the nominal fold took, group by group, in
//! the same walk and under the same weight and shielded flux shape as the
//! transport total (see [`crate::multigroup::per_group_list_rates`]). A group
//! a covariance edge cuts is split between the two intervals by energy width,
//! the flux being flat inside a group. Where `branching.arrow` carries several
//! states summed into one curve, a state's part of the curve's production in a
//! piece of a group is the ratio of the two curves' integrals over that piece.
//! Nothing is linearized: a production is linear in its cross section, and
//! each multiplier is constant on its interval.
//!
//! # Sampling
//!
//! A parent's states and their intervals are one field, and a replica draws
//! it once, shared by every spectrum, step and material of the run, exactly
//! as a nuclide's MF=33 cells are drawn: a multivariate lognormal with mean
//! one and the evaluation's covariance, after the same nearest-correlation
//! repair where the evaluated matrix is not positive semi-definite, every
//! interval keeping its evaluated sigma (see [`crate::covariance_sample`]).
//! A multiplier is positive, so no production is ever negative. The stream is
//! the parent's own and apart from its MF=33 stream, so a draw is a pure
//! function of `(seed, replica, parent)`.
//!
//! # What is not used
//!
//! A block correlating a state with another evaluation (MAT1), with a quantity
//! that is not an MF=10 production (XMF1 not 10), an NC block, a layout that is
//! not relative (LB 0 and 8) or not implemented, and a state the branching rule
//! folds no curve for, are counted per parent and reason in
//! `Info::isomer_branching_blocks_not_used`. The `(n,n')` ground state, the
//! parent itself, makes no product, so its blocks have nothing to move and
//! are passed over without a count. A pair of states stated in both
//! orientations is used once, the lower `(MT, LFS)` as the row.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use yamc_nuclide::covariance::expand::{expand_ni, ExpandedBlock, Scale, Unsupported};
use yamc_nuclide::covariance::{BranchingCovarianceBlock, CovarianceData};
use yani::BranchQuantity;

use crate::branching_rule::{curve_interp, ListRates, ListRule, Lists, ANYTHING};
use crate::covariance_sample::{FieldRepair, LognormalLimit, RelativeLognormal};

/// Keeps a parent's MF=40 stream clear of its MF=33 one, which draws on the
/// parent's own ordinal.
const ISOMER_STREAM: u32 = 0x1503_40B1;

/// A final state of one parent's reaction: the key MF=40 gives it.
type StateKey = (i32, i32);

/// Where one MF=40 state sits in the branching lists, and the cells it owns.
#[derive(Debug, Clone, PartialEq)]
struct State {
    /// `(MT, LFS)` as the MF=40 section numbers them.
    key: StateKey,
    /// The list in `lists[parent]` and the curve in that list's `curves`.
    list: usize,
    curve: usize,
    /// The state's own curve, when the list's curve is several states summed.
    own: Option<(Vec<f64>, Vec<f64>)>,
    /// The boundaries of its intervals, ascending: the union of every used
    /// block's grid on its side.
    edges: Vec<f64>,
    /// Its first interval's index in the parent's field.
    first: usize,
}

impl State {
    fn cells(&self) -> usize {
        self.edges.len().saturating_sub(1)
    }
}

/// One parent's stated MF=40 covariance, on its states' intervals.
#[derive(Debug, Clone, PartialEq)]
struct ParentField {
    states: Vec<State>,
    /// Row-major relative covariance over every state's intervals in turn.
    covariance: Vec<f64>,
    n: usize,
}

/// Every chain parent's MF=40 covariance, placed on the branching lists the
/// rule folds, and what of it could not be used.
///
/// Built once per run from the branching table, independent of any spectrum
/// or material. [`IsomerRun`] reads it against a material's spectra.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IsomerCovariance {
    fields: BTreeMap<String, ParentField>,
    /// Per parent, the blocks not used by reason (see the module
    /// documentation).
    not_used: BTreeMap<String, BTreeMap<String, usize>>,
}

/// How an MF=40 state maps onto the branching lists.
#[derive(Clone, Copy)]
enum Placement {
    Curve {
        list: usize,
        curve: usize,
    },
    /// The `(n,n')` ground state, the parent itself: no product to move.
    NoProduct,
    Unplaced(&'static str),
}

const UNPLACED_TARGET: &str =
    "the state was placed on no chain nuclide (target null in branching_covariance.arrow)";
const UNPLACED_LIST: &str = "the chain folds no list for the state's reaction";
const UNPLACED_PASSED_OVER: &str =
    "the state is an MF=9 yield the rule passes over for the reaction's MF=10 partials";
const UNPLACED_CURVE: &str = "the reaction's list has no curve for the state";
const UNPLACED_LEVEL: &str =
    "the state is one of several summed into a curve, and its level is not among them";
const PARTNER_UNPLACED: &str = "the partner state is not one this parent's lists fold";
const CROSS_MATERIAL: &str = "the partner is in another evaluation (MAT1)";
const OTHER_FILE: &str = "the partner is not an MF=10 production (XMF1 not 10)";
const NC_BLOCK: &str = "an NC block, derived from other reactions";
const NOT_RELATIVE: &str = "an absolute or short-range layout (LB=0 or 8)";
const MALFORMED: &str = "the block's arrays do not match its declared sizes or its LB";
const LAYOUT: &str = "an LB layout that is not implemented";
const TRANSPOSE: &str = "the transpose of a cross-state block already used";

/// Where `block`'s own state sits in `rules`.
fn place(block: &BranchingCovarianceBlock, rules: &[ListRule<'_>]) -> Placement {
    let (Some(target), Some(reaction), Some(quantity)) = (
        block.target.as_deref(),
        block.reaction.as_deref(),
        block.quantity.as_deref(),
    ) else {
        return Placement::Unplaced(UNPLACED_TARGET);
    };
    let quantity = match quantity {
        "cross_section" => BranchQuantity::CrossSection,
        "yield" => BranchQuantity::Yield,
        _ => return Placement::Unplaced(UNPLACED_TARGET),
    };
    let Some(list) = rules.iter().position(|r| r.kind == reaction && !r.emits) else {
        return Placement::Unplaced(UNPLACED_LIST);
    };
    let rule = &rules[list];
    let same = |c: &yani::BranchCurve| c.target == target && c.quantity == quantity;
    let Some(curve) = rule.curves.iter().position(|c| same(c)) else {
        return Placement::Unplaced(if rule.passed_over.iter().any(|c| same(c)) {
            UNPLACED_PASSED_OVER
        } else {
            UNPLACED_CURVE
        });
    };
    if !rule.produces[curve] {
        return Placement::NoProduct;
    }
    // The curve is this state's own when the converter wrote no curve of its
    // own beside the block, and then the level is taken as the curve's whatever
    // MF=40 numbers it: ENDF/B-VIII.1 Pb204 MT=4 numbers its isomer LFS 1 in
    // MF=40 and 21 in MF=10.
    let c = &rule.curves[curve];
    let listed = c
        .states
        .iter()
        .any(|s| s.mt == block.block.mt && s.lfs == block.lfs);
    if listed || block.energy.is_none() {
        Placement::Curve { list, curve }
    } else {
        Placement::Unplaced(UNPLACED_LEVEL)
    }
}

/// The interval of ascending `edges` holding `[a, b]`, which lies inside one.
fn interval_of(edges: &[f64], a: f64, b: f64) -> Option<usize> {
    let mid = 0.5 * (a + b);
    let i = edges.partition_point(|&e| e <= mid);
    (i >= 1 && i < edges.len() && edges[i - 1] <= a && b <= edges[i]).then(|| i - 1)
}

impl IsomerCovariance {
    /// Place `blocks` on the lists of `lists` and build each parent's field.
    ///
    /// Parents with no list in `lists` (not in the chain) are left out. A
    /// block that cannot be used is counted against its parent rather than
    /// dropped without a word.
    pub(crate) fn new(blocks: &[BranchingCovarianceBlock], lists: &Lists<'_>) -> Self {
        let mut by_parent: BTreeMap<&str, Vec<&BranchingCovarianceBlock>> = BTreeMap::new();
        for b in blocks {
            if lists.contains_key(&b.nuclide) {
                by_parent.entry(b.nuclide.as_str()).or_default().push(b);
            }
        }
        let mut out = IsomerCovariance::default();
        for (parent, blocks) in by_parent {
            let (field, not_used) = parent_field(&blocks, &lists[parent]);
            if !not_used.is_empty() {
                out.not_used.insert(parent.to_string(), not_used);
            }
            if let Some(field) = field {
                out.fields.insert(parent.to_string(), field);
            }
        }
        out
    }

    /// Read the blocks out of `branch`'s `branching_covariance.arrow` and place
    /// them, or an empty covariance where the subsection carries none.
    pub(crate) fn from_table(
        branch: &yani::BranchTable,
        lists: &Lists<'_>,
    ) -> Result<Self, String> {
        let Some(batches) = branch.covariance() else {
            return Ok(Self::default());
        };
        let blocks = yamc_nuclide::arrow::covariance_arrow::branching_covariance_blocks(batches)
            .map_err(|e| format!("reading the MF=40 isomer production covariance: {e}"))?;
        Ok(Self::new(&blocks, lists))
    }

    /// Whether `parent` has a field to draw.
    pub(crate) fn has(&self, parent: &str) -> bool {
        self.fields.contains_key(parent)
    }
}

/// One parent's field from its blocks, and the blocks not used, by reason.
fn parent_field(
    blocks: &[&BranchingCovarianceBlock],
    rules: &[ListRule<'_>],
) -> (Option<ParentField>, BTreeMap<String, usize>) {
    let mut not_used: BTreeMap<String, usize> = BTreeMap::new();
    let mut skip = |reason: &str| *not_used.entry(reason.to_string()).or_insert(0) += 1;

    // Where each state sits, from its first block: every block of one state
    // places it the same way, since the placement reads only the state.
    let mut states: BTreeMap<StateKey, Placement> = BTreeMap::new();
    for b in blocks {
        states
            .entry((b.block.mt, b.lfs))
            .or_insert_with(|| place(b, rules));
    }

    // Each block, expanded, or the reason it is not used.
    let mut used: Vec<(StateKey, StateKey, ExpandedBlock)> = Vec::new();
    let mut pairs: BTreeSet<(StateKey, StateKey)> = BTreeSet::new();
    for b in blocks {
        let row = (b.block.mt, b.lfs);
        match &states[&row] {
            Placement::Curve { .. } => {}
            // A block of the `(n,n')` ground state has nothing to move.
            Placement::NoProduct => continue,
            Placement::Unplaced(reason) => {
                skip(reason);
                continue;
            }
        }
        if b.is_cross_material() {
            skip(CROSS_MATERIAL);
            continue;
        }
        if b.block.xmf1 != 10.0 {
            skip(OTHER_FILE);
            continue;
        }
        let col = (b.block.mt1, b.block.xlfs1 as i32);
        match states.get(&col) {
            Some(Placement::Curve { .. }) => {}
            // The partner is the `(n,n')` ground: nothing to correlate with.
            Some(Placement::NoProduct) => continue,
            _ => {
                skip(PARTNER_UNPLACED);
                continue;
            }
        }
        let expanded = match &b.block.data {
            CovarianceData::Ni(ni) => match expand_ni(ni) {
                Ok(e) => e,
                Err(Unsupported::Layout(_)) => {
                    skip(LAYOUT);
                    continue;
                }
                Err(Unsupported::Malformed) => {
                    skip(MALFORMED);
                    continue;
                }
            },
            _ => {
                skip(NC_BLOCK);
                continue;
            }
        };
        if expanded.scale != Scale::Relative {
            skip(NOT_RELATIVE);
            continue;
        }
        // One orientation of a pair of states: the first met is used, and
        // the blocks arrive in the file's order, the lower MT first.
        if row != col {
            let pair = (row.min(col), row.max(col));
            if pairs.contains(&pair) && !used.iter().any(|(r, c, _)| (*r, *c) == (row, col)) {
                skip(TRANSPOSE);
                continue;
            }
            pairs.insert(pair);
        }
        used.push((row, col, expanded));
    }

    // Each state's intervals: the union of the grids of the blocks it is on.
    let mut edges: BTreeMap<StateKey, Vec<f64>> = BTreeMap::new();
    for (row, col, e) in &used {
        edges.entry(*row).or_default().extend(&e.row_energies);
        edges.entry(*col).or_default().extend(&e.col_energies);
    }
    let mut placed: Vec<State> = Vec::new();
    let mut n = 0;
    for (key, mut grid) in edges {
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        if grid.len() < 2 {
            continue;
        }
        let Some(&Placement::Curve { list, curve }) = states.get(&key) else {
            continue;
        };
        let own = blocks
            .iter()
            .find(|b| (b.block.mt, b.lfs) == key)
            .and_then(|b| b.energy.clone().zip(b.values.clone()));
        let state = State {
            key,
            list,
            curve,
            own,
            first: n,
            edges: grid,
        };
        n += state.cells();
        placed.push(state);
    }
    if n == 0 {
        return (None, not_used);
    }
    let index: HashMap<StateKey, usize> =
        placed.iter().enumerate().map(|(i, s)| (s.key, i)).collect();
    let mut covariance = vec![0.0; n * n];
    for (row, col, e) in &used {
        let (r, c) = (&placed[index[row]], &placed[index[col]]);
        for i in 0..r.cells() {
            let (a, b) = (r.edges[i], r.edges[i + 1]);
            let Some(bi) = interval_of(&e.row_energies, a, b) else {
                continue;
            };
            for j in 0..c.cells() {
                let (p, q) = (c.edges[j], c.edges[j + 1]);
                let Some(bj) = interval_of(&e.col_energies, p, q) else {
                    continue;
                };
                let v = e.get(bi, bj);
                let (x, y) = (r.first + i, c.first + j);
                covariance[x * n + y] += v;
                // A cross-state block states its transpose too.
                if row != col {
                    covariance[y * n + x] += v;
                }
            }
        }
    }
    // A covariance is symmetric; an LS=0 self block written as a full matrix
    // need not be to the last bit.
    for x in 0..n {
        for y in (x + 1)..n {
            let v = 0.5 * (covariance[x * n + y] + covariance[y * n + x]);
            covariance[x * n + y] = v;
            covariance[y * n + x] = v;
        }
    }
    (
        Some(ParentField {
            states: placed,
            covariance,
            n,
        }),
        not_used,
    )
}

/// The integral of a branching curve over `[a, b]`, read as the rule reads it
/// ([`curve_interp`]: zero below the first point, flat above the last), exact
/// for its lin-lin segments.
fn curve_integral(energy: &[f64], values: &[f64], a: f64, b: f64) -> f64 {
    if energy.is_empty() || b <= a {
        return 0.0;
    }
    let a = a.max(energy[0]);
    if b <= a {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut prev = (a, curve_interp(energy, values, a));
    let inside = energy.iter().copied().filter(|&e| e > a && e < b);
    for e in inside.chain(std::iter::once(b)) {
        let v = curve_interp(energy, values, e);
        sum += 0.5 * (prev.1 + v) * (e - prev.0);
        prev = (e, v);
    }
    sum
}

/// One curve's stated part under one spectrum.
#[derive(Debug, Clone)]
struct CurveTerms {
    list: usize,
    curve: usize,
    /// The share of the curve's nominal production its stated states carry.
    stated: f64,
    /// Per interval of the parent's field, its nominal production over the
    /// curve's: `(cell, w_j / P)`.
    cells: Vec<(usize, f64)>,
}

/// One spectrum's view of the run's fields.
struct SpectrumTerms {
    /// Every list's nominal fold, which a replica perturbs where no flux draw
    /// already has.
    nominal: HashMap<String, Vec<Option<ListRates>>>,
    /// Per drawn parent, its stated curves.
    curves: BTreeMap<String, Vec<CurveTerms>>,
}

/// What [`IsomerRun::new`] found to report, for `Info`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct IsomerReport {
    pub(crate) perturbed: BTreeSet<(String, String)>,
    pub(crate) without: BTreeSet<(String, String)>,
    pub(crate) not_used: BTreeMap<String, BTreeMap<String, usize>>,
    pub(crate) repaired: BTreeMap<String, FieldRepair>,
    pub(crate) lognormal_not_carried: BTreeMap<String, LognormalLimit>,
    pub(crate) production_sigma: BTreeMap<(String, String, String), f64>,
}

/// One run's MF=40 draw: the fields of the parents the material reaches, read
/// against each of its spectra.
pub(crate) struct IsomerRun {
    parents: Vec<(String, u32, RelativeLognormal)>,
    spectra: Vec<SpectrumTerms>,
}

/// One spectrum's inputs to [`IsomerRun::new`]: the per-group terms of every
/// list (those of the drawn parents kept) and its group boundaries.
pub(crate) struct IsomerSpectrum<'a> {
    pub(crate) lists: &'a crate::flux_uncertainty::PerGroupLists,
    pub(crate) boundaries: &'a [f64],
}

/// One replica's draw: per drawn parent, its multipliers minus one.
pub(crate) type IsomerDraw = Vec<crate::covariance_sample::GaussianDraw>;

/// Per `(parent, reaction, product)`, the list draws whose stated production
/// of the product was scaled back to the drawn reaction total.
pub(crate) type Clipped = BTreeMap<(String, String, String), usize>;

impl IsomerRun {
    /// Read `covariance` against `spectra` for the parents in `reachable`.
    ///
    /// `None` when no reachable parent has a stated state its lists produce
    /// under any spectrum, so a run with nothing to draw takes the path it
    /// took before this source existed. The report is returned either way.
    pub(crate) fn new(
        covariance: &IsomerCovariance,
        lists: &Lists<'_>,
        spectra: &[IsomerSpectrum<'_>],
        reachable: &BTreeSet<String>,
    ) -> (Option<Self>, IsomerReport) {
        let mut report = IsomerReport {
            not_used: covariance
                .not_used
                .iter()
                .filter(|(p, _)| reachable.contains(*p))
                .map(|(p, r)| (p.clone(), r.clone()))
                .collect(),
            ..Default::default()
        };
        let mut terms: Vec<SpectrumTerms> = Vec::with_capacity(spectra.len());
        for s in spectra {
            let nominal: HashMap<String, Vec<Option<ListRates>>> = s
                .lists
                .iter()
                .map(|(p, per)| {
                    let per = per.iter().map(|l| l.as_ref().map(|l| l.folded.clone()));
                    (p.clone(), per.collect())
                })
                .collect();
            let mut curves: BTreeMap<String, Vec<CurveTerms>> = BTreeMap::new();
            for (parent, field) in &covariance.fields {
                if !reachable.contains(parent) {
                    continue;
                }
                let Some(per) = s.lists.get(parent) else {
                    continue;
                };
                let mut out: Vec<CurveTerms> = Vec::new();
                for state in &field.states {
                    let Some(Some(list)) = per.get(state.list) else {
                        continue;
                    };
                    let Some(groups) = list.production.get(state.curve) else {
                        continue;
                    };
                    let total = list.folded.production[state.curve];
                    if total <= 0.0 {
                        continue;
                    }
                    let rule = &lists[parent][state.list];
                    let curve = &rule.curves[state.curve];
                    let weights = cell_weights(state, groups, s.boundaries, curve);
                    let entry = match out
                        .iter_mut()
                        .position(|c| c.list == state.list && c.curve == state.curve)
                    {
                        Some(i) => &mut out[i],
                        None => {
                            out.push(CurveTerms {
                                list: state.list,
                                curve: state.curve,
                                stated: 0.0,
                                cells: Vec::new(),
                            });
                            out.last_mut().expect("just pushed")
                        }
                    };
                    for (j, w) in weights.into_iter().enumerate() {
                        if w > 0.0 {
                            entry.stated += w / total;
                            entry.cells.push((state.first + j, w / total));
                        }
                    }
                }
                out.retain(|c| !c.cells.is_empty());
                for c in &out {
                    let rule = &lists[parent][c.list];
                    report.perturbed.insert((parent.clone(), rule.kind.clone()));
                    let var: f64 = c
                        .cells
                        .iter()
                        .flat_map(|&(i, a)| {
                            c.cells
                                .iter()
                                .map(move |&(j, b)| a * b * field.covariance[i * field.n + j])
                        })
                        .sum();
                    let key = (
                        parent.clone(),
                        rule.kind.clone(),
                        rule.curves[c.curve].target.clone(),
                    );
                    let sigma = var.max(0.0).sqrt();
                    let slot = report.production_sigma.entry(key).or_insert(0.0);
                    *slot = slot.max(sigma);
                }
                if !out.is_empty() {
                    curves.insert(parent.clone(), out);
                }
            }
            terms.push(SpectrumTerms { nominal, curves });
        }

        // A reachable list whose split can move, that made something, with
        // nothing stated on it, is held: a gap. A list of one state that is
        // the whole reaction cannot move, and MT=5's residuals and particles
        // are held under their own entry.
        for t in &terms {
            for (parent, per) in &t.nominal {
                if !reachable.contains(parent) {
                    continue;
                }
                let Some(rules) = lists.get(parent) else {
                    continue;
                };
                for (rule, folded) in rules.iter().zip(per) {
                    let Some(folded) = folded else { continue };
                    if rule.emits || rule.kind == ANYTHING {
                        continue;
                    }
                    let made: f64 = folded
                        .production
                        .iter()
                        .zip(&rule.produces)
                        .filter(|(_, p)| **p)
                        .map(|(v, _)| v)
                        .sum();
                    let targets: BTreeSet<&str> = rule
                        .curves
                        .iter()
                        .zip(&rule.produces)
                        .filter(|(_, p)| **p)
                        .map(|(c, _)| c.target.as_str())
                        .collect();
                    let movable = rule.has_remainder() || targets.len() > 1;
                    let key = (parent.clone(), rule.kind.clone());
                    if made > 0.0 && movable && !report.perturbed.contains(&key) {
                        report.without.insert(key);
                    }
                }
            }
        }

        let drawn: BTreeSet<&String> = terms.iter().flat_map(|t| t.curves.keys()).collect();
        let parents: Vec<(String, u32, RelativeLognormal)> = drawn
            .into_iter()
            .map(|p| {
                let field = &covariance.fields[p];
                let sampler = RelativeLognormal::new(&field.covariance);
                if let Some(r) = sampler.repair() {
                    report.repaired.insert(p.clone(), r);
                }
                if let Some(l) = sampler.lognormal_limit() {
                    report.lognormal_not_carried.insert(p.clone(), l);
                }
                let ordinal = crate::covariance_sample::name_ordinal(p) ^ ISOMER_STREAM;
                (p.clone(), ordinal, sampler)
            })
            .collect();
        if parents.is_empty() {
            return (None, report);
        }
        (
            Some(IsomerRun {
                parents,
                spectra: terms,
            }),
            report,
        )
    }

    /// How many parents a replica draws.
    pub(crate) fn parents(&self) -> usize {
        self.parents.len()
    }

    /// One replica's draw of every drawn parent, shared by every spectrum.
    pub(crate) fn draw(&self, seed: u64, replica: u64) -> IsomerDraw {
        self.parents
            .iter()
            .map(|(_, ordinal, sampler)| sampler.draw(*ordinal, seed, replica))
            .collect()
    }

    /// Spectrum `idx`'s nominal list folds, for a replica no flux draw moved.
    pub(crate) fn nominal(&self, idx: usize) -> &HashMap<String, Vec<Option<ListRates>>> {
        &self.spectra[idx].nominal
    }

    /// `base`, spectrum `idx`'s list folds as this replica's other draws left
    /// them, with `draw` applied by the branching rule (see the module
    /// documentation), ready to re-fold into the chain.
    ///
    /// `total_ratio(parent, kind)` is the replica's draw of the reaction rate
    /// the list is a split of, over the rate before that draw: the MF=33 and
    /// resonance draws, one where neither moved it. A stated production of a
    /// list whose ground takes the rest does not follow it; everything else
    /// on such a list does, so the shares nothing states are kept. Returns the
    /// folds and, keyed `(parent, reaction, product)`, the stated productions
    /// that had to be scaled back to the drawn total (one each).
    pub(crate) fn perturb(
        &self,
        idx: usize,
        draw: &IsomerDraw,
        lists: &Lists<'_>,
        base: &HashMap<String, Vec<Option<ListRates>>>,
        total_ratio: &dyn Fn(&str, &str) -> f64,
    ) -> (HashMap<String, Vec<Option<ListRates>>>, Clipped) {
        let spectrum = &self.spectra[idx];
        let mut clipped = Clipped::new();
        let mut out = base.clone();
        for (parent, per) in out.iter_mut() {
            let Some(rules) = lists.get(parent) else {
                continue;
            };
            let stated = spectrum.curves.get(parent);
            let deviates = stated.and_then(|_| {
                let i = self.parents.iter().position(|(p, _, _)| p == parent)?;
                Some(&draw[i].0)
            });
            for (li, (rule, folded)) in rules.iter().zip(per.iter_mut()).enumerate() {
                let Some(folded) = folded else { continue };
                // Per curve: the stated share `s` of its nominal production
                // and the draw's relative change `d` to it.
                let mut s = vec![0.0; folded.production.len()];
                let mut d = vec![0.0; folded.production.len()];
                if let (Some(curves), Some(dev)) = (stated, deviates) {
                    for c in curves.iter().filter(|c| c.list == li) {
                        s[c.curve] = c.stated;
                        d[c.curve] = c.cells.iter().map(|&(j, w)| w * dev[j]).sum();
                    }
                }
                let remainder = rule.has_remainder() && !rule.emits;
                if !remainder {
                    for (k, p) in folded.production.iter_mut().enumerate() {
                        *p *= 1.0 + d[k];
                    }
                    folded.clipped = 0.0;
                    folded.extrapolated = 0.0;
                    continue;
                }
                let f = total_ratio(parent, &rule.kind);
                let mut held = 0.0;
                let mut drawn = 0.0;
                let mut parts = Vec::with_capacity(folded.production.len());
                for (k, &p) in folded.production.iter().enumerate() {
                    let rest = p * (1.0 - s[k]) * f;
                    let own = p * (s[k] + d[k]);
                    if rule.produces[k] {
                        held += rest;
                        drawn += own;
                    }
                    parts.push((rest, own));
                }
                folded.total *= f;
                // More than the reaction makes cannot be made: the stated
                // productions give way to the total, as the nominal rule
                // clips an absolute partial above it.
                let room = folded.total - held;
                let scale = if folded.has_total && drawn > 0.0 && held + drawn > folded.total {
                    for (k, curve) in rule.curves.iter().enumerate() {
                        if rule.produces[k] && s[k] > 0.0 {
                            let key = (parent.clone(), rule.kind.clone(), curve.target.clone());
                            *clipped.entry(key).or_insert(0) += 1;
                        }
                    }
                    room.max(0.0) / drawn
                } else {
                    1.0
                };
                for (p, (rest, own)) in folded.production.iter_mut().zip(parts) {
                    *p = rest + own * scale;
                }
                folded.clipped = 0.0;
                folded.extrapolated = 0.0;
            }
        }
        (out, clipped)
    }
}

/// `state`'s nominal production on each of its intervals under one spectrum:
/// the curve's per-group production `groups` (in the fold's units), each
/// group split across an interval edge by energy width, and for a state
/// summed with others into `curve`, its own part of each piece.
fn cell_weights(
    state: &State,
    groups: &[f64],
    boundaries: &[f64],
    curve: &yani::BranchCurve,
) -> Vec<f64> {
    let edges = &state.edges;
    let mut w = vec![0.0; state.cells()];
    for (g, &p) in groups.iter().enumerate() {
        if p == 0.0 || g + 1 >= boundaries.len() {
            continue;
        }
        let (lo, hi) = (boundaries[g], boundaries[g + 1]);
        if hi <= lo {
            continue;
        }
        let first = edges.partition_point(|&e| e <= lo).saturating_sub(1);
        for j in first..w.len() {
            let (a, b) = (lo.max(edges[j]), hi.min(edges[j + 1]));
            if edges[j] >= hi {
                break;
            }
            if b <= a {
                continue;
            }
            let share = match &state.own {
                None => 1.0,
                Some((energy, values)) => {
                    let sum = curve_integral(&curve.energy, &curve.values, a, b);
                    if sum > 0.0 {
                        (curve_integral(energy, values, a, b) / sum).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                }
            };
            w[j] += p * (b - a) / (hi - lo) * share;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use endf::mf::covariance::NiSubsection;
    use yamc_nuclide::covariance::CovarianceBlock;
    use yani::{BranchCurve, BranchState};

    use crate::branching_rule::ListRule;

    fn state(mt: i32, lfs: i32) -> BranchState {
        BranchState {
            mt,
            lfs,
            lmf: None,
            list_complete: true,
            level_route: "ground".to_string(),
            level_energy: 0.0,
            level_energy_difference: None,
            mf3_cross_section: None,
        }
    }

    fn curve(target: &str, mt: i32, lfs: i32, complete: bool) -> BranchCurve {
        BranchCurve {
            target: target.to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e6, 2.0e7],
            values: vec![1.0, 1.0],
            states: Arc::from(vec![BranchState {
                list_complete: complete,
                ..state(mt, lfs)
            }]),
            normalisation: None,
        }
    }

    /// An LB=5 symmetric block (upper triangle, row by row) on `ek`.
    fn lb5(ek: Vec<f64>, upper: Vec<f64>) -> NiSubsection {
        NiSubsection {
            lb: 5,
            ls: 1,
            ne: ek.len() as i64,
            ek,
            fkk: upper,
            ..Default::default()
        }
    }

    fn block(
        parent: &str,
        reaction: &str,
        target: &str,
        (mt, lfs): StateKey,
        (mt1, xlfs1): StateKey,
        ni: NiSubsection,
    ) -> BranchingCovarianceBlock {
        BranchingCovarianceBlock {
            nuclide: parent.to_string(),
            reaction: Some(reaction.to_string()),
            target: Some(target.to_string()),
            target1: Some(target.to_string()),
            energy: None,
            values: None,
            quantity: Some("cross_section".to_string()),
            mat: 4125,
            za: 41093,
            awr: 92.1,
            lis: 0,
            state_idx: 0,
            qm: 0.0,
            qi: 0.0,
            izap: 41092,
            lfs,
            block: CovarianceBlock {
                mt,
                subsection_idx: 0,
                block_idx: 0,
                mat1: 0,
                mt1,
                xmf1: 10.0,
                xlfs1: xlfs1 as f64,
                mtl: 0,
                mat: 4125,
                data: CovarianceData::Ni(ni),
            },
        }
    }

    /// Nb93 (n,2n) to Nb92 and Nb92_m1, a complete MF=10 list.
    fn n2n_curves() -> Vec<BranchCurve> {
        vec![curve("Nb92", 16, 0, true), curve("Nb92_m1", 16, 1, true)]
    }

    fn lists_for<'a>(parent: &str, kind: &str, curves: &'a [BranchCurve]) -> Lists<'a> {
        let rule = ListRule::new(parent, kind, curves)
            .expect("rule")
            .expect("a list");
        HashMap::from([(parent.to_string(), vec![rule])])
    }

    #[test]
    fn a_self_block_becomes_a_field_on_its_own_grid() {
        let curves = n2n_curves();
        let lists = lists_for("Nb93", "(n,2n)", &curves);
        let ground = block(
            "Nb93",
            "(n,2n)",
            "Nb92",
            (16, 0),
            (16, 0),
            lb5(vec![1.0e-5, 1.0e7, 2.0e7], vec![0.0, 0.0, 0.04]),
        );
        let isomer = block(
            "Nb93",
            "(n,2n)",
            "Nb92_m1",
            (16, 1),
            (16, 1),
            lb5(vec![1.0e-5, 1.5e7, 2.0e7], vec![0.0, 0.0, 0.09]),
        );
        let cov = IsomerCovariance::new(&[ground, isomer], &lists);
        let field = &cov.fields["Nb93"];
        assert_eq!(field.n, 4);
        assert_eq!(field.states[0].edges, vec![1.0e-5, 1.0e7, 2.0e7]);
        assert_eq!(field.states[1].edges, vec![1.0e-5, 1.5e7, 2.0e7]);
        assert_eq!(field.covariance[3], 0.0);
        assert_eq!(field.covariance[4 + 1], 0.04);
        assert_eq!(field.covariance[3 * 4 + 3], 0.09);
        assert!(cov.not_used.is_empty());
    }

    /// A cross-state block is used once, in both triangles, and its transpose
    /// stated again is counted rather than added twice.
    #[test]
    fn a_cross_state_block_fills_both_triangles_once() {
        let curves = n2n_curves();
        let lists = lists_for("Nb93", "(n,2n)", &curves);
        let grid = vec![1.0e-5, 2.0e7];
        let blocks = [
            block(
                "Nb93",
                "(n,2n)",
                "Nb92",
                (16, 0),
                (16, 0),
                lb5(grid.clone(), vec![0.04]),
            ),
            block(
                "Nb93",
                "(n,2n)",
                "Nb92_m1",
                (16, 1),
                (16, 1),
                lb5(grid.clone(), vec![0.09]),
            ),
            block(
                "Nb93",
                "(n,2n)",
                "Nb92",
                (16, 0),
                (16, 1),
                lb5(grid.clone(), vec![-0.03]),
            ),
            block(
                "Nb93",
                "(n,2n)",
                "Nb92_m1",
                (16, 1),
                (16, 0),
                lb5(grid.clone(), vec![-0.03]),
            ),
        ];
        let cov = IsomerCovariance::new(&blocks, &lists);
        let field = &cov.fields["Nb93"];
        assert_eq!(field.covariance, vec![0.04, -0.03, -0.03, 0.09]);
        assert_eq!(cov.not_used["Nb93"][TRANSPOSE], 1);
    }

    /// Blocks naming another evaluation, another file, an absolute layout or
    /// a state with no curve are counted by reason, never folded.
    #[test]
    fn unusable_blocks_are_counted_by_reason() {
        let curves = n2n_curves();
        let lists = lists_for("Nb93", "(n,2n)", &curves);
        let grid = vec![1.0e-5, 2.0e7];
        let mut other_mat = block(
            "Nb93",
            "(n,2n)",
            "Nb92",
            (16, 0),
            (16, 0),
            lb5(grid.clone(), vec![0.04]),
        );
        other_mat.block.mat1 = 9228;
        let mut other_file = block(
            "Nb93",
            "(n,2n)",
            "Nb92",
            (16, 0),
            (16, 0),
            lb5(grid.clone(), vec![0.04]),
        );
        other_file.block.xmf1 = 3.0;
        let absolute = block(
            "Nb93",
            "(n,2n)",
            "Nb92",
            (16, 0),
            (16, 0),
            NiSubsection {
                lb: 0,
                np: 2,
                ek: grid.clone(),
                fk: vec![0.01, 0.0],
                ..Default::default()
            },
        );
        let mut nowhere = block(
            "Nb93",
            "(n,2n)",
            "Nb92_m2",
            (16, 2),
            (16, 2),
            lb5(grid.clone(), vec![0.04]),
        );
        nowhere.target = None;
        let cov = IsomerCovariance::new(&[other_mat, other_file, absolute, nowhere], &lists);
        assert!(cov.fields.is_empty());
        let reasons = &cov.not_used["Nb93"];
        assert_eq!(reasons[CROSS_MATERIAL], 1);
        assert_eq!(reasons[OTHER_FILE], 1);
        assert_eq!(reasons[NOT_RELATIVE], 1);
        assert_eq!(reasons[UNPLACED_TARGET], 1);
    }

    /// The `(n,n')` ground state makes no product, so its block is passed
    /// over without a count.
    #[test]
    fn the_inelastic_ground_is_passed_over() {
        let curves = vec![curve("Nb93", 4, 0, true), curve("Nb93_m1", 4, 1, true)];
        let lists = lists_for("Nb93", "(n,n')", &curves);
        let grid = vec![1.0e-5, 2.0e7];
        let blocks = [
            block(
                "Nb93",
                "(n,n')",
                "Nb93",
                (4, 0),
                (4, 0),
                lb5(grid.clone(), vec![0.04]),
            ),
            block(
                "Nb93",
                "(n,n')",
                "Nb93_m1",
                (4, 1),
                (4, 1),
                lb5(grid.clone(), vec![0.09]),
            ),
        ];
        let cov = IsomerCovariance::new(&blocks, &lists);
        assert!(cov.not_used.is_empty());
        let field = &cov.fields["Nb93"];
        assert_eq!(field.n, 1);
        assert_eq!(field.states[0].key, (4, 1));
    }

    #[test]
    fn a_curve_integral_is_exact_on_a_linear_curve() {
        let (e, v) = (vec![1.0, 3.0], vec![1.0, 3.0]);
        assert!((curve_integral(&e, &v, 1.0, 3.0) - 4.0).abs() < 1e-12);
        // Zero below the first point, flat above the last.
        assert!((curve_integral(&e, &v, 0.0, 1.0)).abs() < 1e-12);
        assert!((curve_integral(&e, &v, 3.0, 5.0) - 6.0).abs() < 1e-12);
    }

    /// A group a covariance edge cuts is split by energy width, and a state
    /// summed with another takes its own curve's part of each piece.
    #[test]
    fn cell_weights_split_groups_and_merged_curves() {
        let mut s = State {
            key: (16, 1),
            list: 0,
            curve: 0,
            own: None,
            edges: vec![0.0, 1.5, 4.0],
            first: 0,
        };
        let c = BranchCurve {
            target: "X".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![0.0, 4.0],
            values: vec![2.0, 2.0],
            states: Arc::from(Vec::new()),
            normalisation: None,
        };
        let w = cell_weights(&s, &[10.0, 20.0], &[0.0, 2.0, 4.0], &c);
        assert!((w[0] - 7.5).abs() < 1e-12, "{w:?}");
        assert!((w[1] - 22.5).abs() < 1e-12, "{w:?}");
        s.own = Some((vec![0.0, 4.0], vec![0.5, 0.5]));
        let w = cell_weights(&s, &[10.0, 20.0], &[0.0, 2.0, 4.0], &c);
        assert!((w[0] - 7.5 / 4.0).abs() < 1e-12, "{w:?}");
        assert!((w[1] - 22.5 / 4.0).abs() < 1e-12, "{w:?}");
    }

    /// One flat group, a share list of two states, each with one interval.
    /// Returns the run and its list.
    fn run_on<'a>(
        curves: &'a [BranchCurve],
        kind: &str,
        sigmas: (f64, f64),
        correlation: f64,
        terms: &'a mut crate::flux_uncertainty::PerGroupLists,
    ) -> (IsomerRun, Lists<'a>) {
        let lists = lists_for("Nb93", kind, curves);
        let grid = vec![1.0e-5, 2.0e7];
        let (a, b) = sigmas;
        let mut blocks = vec![
            block(
                "Nb93",
                kind,
                &curves[0].target,
                (curves[0].states[0].mt, curves[0].states[0].lfs),
                (curves[0].states[0].mt, curves[0].states[0].lfs),
                lb5(grid.clone(), vec![a * a]),
            ),
            block(
                "Nb93",
                kind,
                &curves[1].target,
                (curves[1].states[0].mt, curves[1].states[0].lfs),
                (curves[1].states[0].mt, curves[1].states[0].lfs),
                lb5(grid.clone(), vec![b * b]),
            ),
        ];
        if correlation != 0.0 {
            blocks.push(block(
                "Nb93",
                kind,
                &curves[0].target,
                (curves[0].states[0].mt, curves[0].states[0].lfs),
                (curves[1].states[0].mt, curves[1].states[0].lfs),
                lb5(grid.clone(), vec![correlation * a * b]),
            ));
        }
        let cov = IsomerCovariance::new(&blocks, &lists);
        let folded = ListRates {
            production: vec![3.0, 1.0],
            to_rate: 1.0e-24,
            total: 4.0,
            has_total: true,
            ..Default::default()
        };
        terms.insert(
            "Nb93".to_string(),
            vec![Some(crate::flux_uncertainty::ListTerms {
                folded,
                production: vec![vec![3.0], vec![1.0]],
                total: vec![4.0],
            })],
        );
        let reachable = BTreeSet::from(["Nb93".to_string()]);
        let (run, report) = IsomerRun::new(
            &cov,
            &lists,
            &[IsomerSpectrum {
                lists: terms,
                boundaries: &[1.0e6, 2.0e7],
            }],
            &reachable,
        );
        assert!(report
            .perturbed
            .contains(&("Nb93".to_string(), kind.to_string())));
        (run.expect("something to draw"), lists)
    }

    fn moments(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    /// The drawn production reproduces the stated relative sigma, with mean
    /// one, and on a share list the shares stay in [0, 1] and sum to one.
    #[test]
    fn draws_reproduce_the_stated_sigma_and_keep_a_split() {
        let curves = n2n_curves();
        let mut terms = HashMap::new();
        let (run, lists) = run_on(&curves, "(n,2n)", (0.2, 0.3), 0.0, &mut terms);
        let base = run.nominal(0).clone();
        let (mut ground, mut isomer) = (Vec::new(), Vec::new());
        for replica in 0..4000 {
            let draw = run.draw(7, replica);
            let (out, clipped) = run.perturb(0, &draw, &lists, &base, &|_, _| 1.0);
            assert!(clipped.is_empty());
            let p = &out["Nb93"][0].as_ref().unwrap().production;
            ground.push(p[0] / 3.0);
            isomer.push(p[1]);
            let share = p[1] / (p[0] + p[1]);
            assert!((0.0..=1.0).contains(&share));
        }
        let (m0, s0) = moments(&ground);
        let (m1, s1) = moments(&isomer);
        assert!((m0 - 1.0).abs() < 0.02, "{m0}");
        assert!((m1 - 1.0).abs() < 0.03, "{m1}");
        assert!((s0 / 0.2 - 1.0).abs() < 0.06, "{s0}");
        assert!((s1 / 0.3 - 1.0).abs() < 0.06, "{s1}");
    }

    /// A stated correlation between two states is carried into the draws.
    #[test]
    fn a_stated_correlation_is_respected() {
        let curves = n2n_curves();
        let mut terms = HashMap::new();
        let (run, lists) = run_on(&curves, "(n,2n)", (0.1, 0.1), 0.8, &mut terms);
        let base = run.nominal(0).clone();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for replica in 0..4000 {
            let draw = run.draw(3, replica);
            let (out, _) = run.perturb(0, &draw, &lists, &base, &|_, _| 1.0);
            let p = &out["Nb93"][0].as_ref().unwrap().production;
            a.push(p[0] / 3.0);
            b.push(p[1]);
        }
        let (ma, sa) = moments(&a);
        let (mb, sb) = moments(&b);
        let cov: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / (a.len() as f64 - 1.0);
        let rho = cov / (sa * sb);
        assert!((rho - 0.8).abs() < 0.03, "{rho}");
    }

    /// With a zero covariance every draw is the nominal fold, bit for bit.
    #[test]
    fn zero_covariance_reproduces_the_nominal() {
        let curves = n2n_curves();
        let mut terms = HashMap::new();
        let lists = lists_for("Nb93", "(n,2n)", &curves);
        let grid = vec![1.0e-5, 2.0e7];
        let blocks = [block(
            "Nb93",
            "(n,2n)",
            "Nb92_m1",
            (16, 1),
            (16, 1),
            lb5(grid, vec![0.0]),
        )];
        let cov = IsomerCovariance::new(&blocks, &lists);
        terms.insert(
            "Nb93".to_string(),
            vec![Some(crate::flux_uncertainty::ListTerms {
                folded: ListRates {
                    production: vec![3.0, 1.0],
                    to_rate: 1.0e-24,
                    total: 4.0,
                    has_total: true,
                    ..Default::default()
                },
                production: vec![vec![3.0], vec![1.0]],
                total: vec![4.0],
            })],
        );
        let reachable = BTreeSet::from(["Nb93".to_string()]);
        let (run, _) = IsomerRun::new(
            &cov,
            &lists,
            &[IsomerSpectrum {
                lists: &terms,
                boundaries: &[1.0e6, 2.0e7],
            }],
            &reachable,
        );
        let run = run.expect("a stated state");
        let base = run.nominal(0).clone();
        for replica in 0..16 {
            let draw = run.draw(1, replica);
            let (out, _) = run.perturb(0, &draw, &lists, &base, &|_, _| 1.0);
            let p = &out["Nb93"][0].as_ref().unwrap().production;
            assert_eq!(p, &vec![3.0, 1.0]);
        }
    }

    /// An isomer-only list: the stated isomer is made at its drawn production
    /// whatever the total's draw, the rest of the total follows that draw,
    /// and a draw above the total is scaled back to it.
    #[test]
    fn an_isomer_only_list_takes_the_drawn_production_not_the_total() {
        let curves = vec![curve("Nb92", 16, 0, false), curve("Nb92_m1", 16, 1, false)];
        let mut terms = HashMap::new();
        let (run, lists) = run_on(&curves, "(n,2n)", (0.0001, 0.5), 0.0, &mut terms);
        assert!(lists["Nb93"][0].has_remainder());
        let base = run.nominal(0).clone();
        let mut clipped_any = 0;
        for replica in 0..400 {
            let draw = run.draw(5, replica);
            let (out, clipped) = run.perturb(0, &draw, &lists, &base, &|_, _| 1.25);
            if !clipped.is_empty() {
                // Each stated product of the list is named, once per
                // clipped draw: the ground states its (negligible) MF=40 too.
                let key = |p: &str| ("Nb93".to_string(), "(n,2n)".to_string(), p.to_string());
                assert_eq!(
                    clipped,
                    Clipped::from([(key("Nb92"), 1), (key("Nb92_m1"), 1)])
                );
                clipped_any += 1;
            }
            let folded = out["Nb93"][0].as_ref().unwrap();
            assert!((folded.total - 5.0).abs() < 1e-12);
            let sum: f64 = folded.production.iter().sum();
            assert!(
                sum <= folded.total * (1.0 + 1e-12),
                "{sum} > {}",
                folded.total
            );
            if clipped.is_empty() {
                // The ground's stated draw is negligible, so it is about
                // three times its 1.0 multiplier: unmoved by the total's draw.
                assert!((folded.production[0] / 3.0 - 1.0).abs() < 0.01);
            }
        }
        // A 50% lognormal on a quarter of the total exceeds the 25% headroom
        // only rarely: once the isomer draw is above 2.25.
        assert!(clipped_any < 40, "{clipped_any}");
    }

    /// A list without any stated state follows the total's draw untouched,
    /// so the cross-section source alone behaves as it did.
    #[test]
    fn an_unstated_list_keeps_its_shares() {
        let curves = vec![curve("Nb92", 16, 0, false), curve("Nb92_m1", 16, 1, false)];
        let rules = vec![ListRule::new("Nb93", "(n,2n)", &curves).unwrap().unwrap()];
        let lists: Lists<'_> = HashMap::from([("Nb93".to_string(), rules)]);
        let run = IsomerRun {
            parents: Vec::new(),
            spectra: vec![SpectrumTerms {
                nominal: HashMap::new(),
                curves: BTreeMap::new(),
            }],
        };
        let base = HashMap::from([(
            "Nb93".to_string(),
            vec![Some(ListRates {
                production: vec![3.0, 1.0],
                to_rate: 1.0e-24,
                total: 4.0,
                has_total: true,
                ..Default::default()
            })],
        )]);
        let (out, _) = run.perturb(0, &Vec::new(), &lists, &base, &|_, _| 2.0);
        let folded = out["Nb93"][0].as_ref().unwrap();
        assert_eq!(folded.production, vec![6.0, 2.0]);
        assert_eq!(folded.total, 8.0);
    }
}
