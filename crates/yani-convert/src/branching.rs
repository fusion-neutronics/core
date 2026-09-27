//! The isomeric branching subsection.
//!
//! Which fraction of a reaction leaves the product in a metastable state, as a
//! function of incident energy. It is the one transmutation subsection that is
//! not a projection of the chain: it comes from MF=8/9/10 radionuclide
//! production in each parent's own neutron evaluation, joined to an isomer
//! table built from the decay files.
//!
//! Two pieces of care, both load-bearing:
//!
//! * **Linearization.** The format stores lin-lin pairs, and a consumer reading
//!   them as piecewise linear silently reshapes any region the evaluation
//!   declared under another law. Regions are resampled here until linear
//!   interpolation reproduces the declared law. Kept in this crate rather than
//!   added to `endf`, so that crate stays close to the upstream it is synced
//!   from.
//! * **Duplicate merging.** Two nuclear levels can resolve to the same isomeric
//!   state, and the physical production is their sum. Merging on the union grid
//!   under the consumer's own conventions is what makes folding the merged
//!   curve equal folding the originals.
//!
//! MF=40, the covariance of those MF=10 partials, is written beside it as
//! `branching_covariance.arrow` when the evaluation carries any: every block
//! of every section as the tape gives it, keyed to the `branching.arrow` row
//! it is the covariance of where there is one.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use endf::function::Tabulated1D;
use endf::mf::covariance::{Mf33Subsection, Mf40, Mf40Subsection};
use endf::radionuclide_production::{LevelRoute, RadionuclideProduction};
use endf::Material;
use yamc_convert::covariance::narrow;
use yamc_convert::sections::ints;

use crate::{floats, list_of, opt_list_of, opt_strings, strings, write_section};

/// Relative tolerance for resampling a non-lin-lin region.
pub const DEFAULT_LINEARIZE_TOL: f64 = 1e-3;

/// Depth cap on the adaptive bisection, so a pathological interval terminates.
const MAX_DEPTH: u32 = 24;

/// How far a reaction's summed partials may sit from the total they should
/// reconstruct (MF=10 cross sections from MF=3, MF=9 yields from one) before
/// the reaction is listed, as a fraction of the nonelastic cross section at
/// the same energy. Measured against the nonelastic rather than against the
/// reaction itself so that a channel which barely happens cannot fill the
/// list: TENDL-2017's (n,2alpha) yields on exotic nuclides sum to anything
/// from 0.04 to 2.9, on cross sections of microbarns.
pub const PARTIAL_SUM_TOLERANCE: f64 = 0.02;

/// Incident energies above this are not held to [`PARTIAL_SUM_TOLERANCE`].
/// TENDL evaluates to 200 MeV and its partials up there routinely stop short
/// of the total, and nothing a fission or fusion device makes gets there.
const PARTIAL_SUM_TOP_EV: f64 = 2.0e7;

/// One row of `branching/branching.arrow`.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchingRow {
    pub nuclide: String,
    pub reaction: String,
    pub target: String,
    /// `"yield"` (MF=9) or `"cross_section"` (MF=10).
    pub quantity: String,
    pub energy: Vec<f64>,
    pub values: Vec<f64>,
}

/// One MF=40 sub-subsection, verbatim, with the key the writer puts in front
/// of each of its blocks. See `branching_branching_covariance` in
/// `nuclear-data-schema` for what each key means.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchingCovarianceRow {
    /// The parent, as in `branching.arrow`.
    pub nuclide: String,
    /// The chain kind the MT maps to, `None` for an MT with none.
    pub reaction: Option<String>,
    /// The chain nuclide this row's product state is the partial of, `None`
    /// when no MF=9 or MF=10 state of the MT matched it.
    pub target: Option<String>,
    /// The chain nuclide the partner state resolved to, `None` when it is in
    /// another material, XMF1 is not 10, MT1 is 0, or no single state of MT1 at level
    /// XLFS1 is found (see `partner_target`). Several levels can resolve to
    /// one chain nuclide, so (`target`, `target1`) does not identify the pair
    /// of states: (`mt`, `lfs`, `mt1`, `xlfs1`) does.
    pub target1: Option<String>,
    /// This state's own MF=10 partial, linearized exactly as `branching.arrow`
    /// has it, only when several MF=10 states resolved to `target` and
    /// `branching.arrow` carries their sum. Always `None` for a state MF=9
    /// gives as a yield, which has no MF=10 partial.
    pub energy: Option<Vec<f64>>,
    pub values: Option<Vec<f64>>,
    /// The evaluation's own MAT.
    pub mat: i32,
    /// The section's MT and HEAD.
    pub mt: i32,
    pub za: i64,
    pub awr: f64,
    pub lis: i64,
    /// Which product state of the section, in tape order, and its CONT.
    pub state_idx: usize,
    pub qm: f64,
    pub qi: f64,
    pub izap: i64,
    pub lfs: i64,
    /// Which sub-subsection of the state, in tape order.
    pub subsection_idx: usize,
    /// The sub-subsection itself, exactly as parsed.
    pub subsection: Mf33Subsection,
}

/// Whether every interpolation region is lin-lin (ENDF law 2).
fn is_linear(t: &Tabulated1D) -> bool {
    t.interpolation.iter().all(|&law| law == 2)
}

/// Evaluate an ENDF interpolation law on one interval.
fn law_value(law: i32, x0: f64, y0: f64, x1: f64, y1: f64, xm: f64) -> f64 {
    match law {
        1 => y0,
        // y linear in ln(x)
        3 => y0 + (xm / x0).ln() / (x1 / x0).ln() * (y1 - y0),
        // ln(y) linear in x
        4 => y0 * ((xm - x0) / (x1 - x0) * (y1 / y0).ln()).exp(),
        // ln(y) linear in ln(x)
        5 => y0 * ((xm / x0).ln() / (x1 / x0).ln() * (y1 / y0).ln()).exp(),
        _ => y0 + (xm - x0) / (x1 - x0) * (y1 - y0),
    }
}

/// Whether a law is defined on an interval. A log of a non-positive number is
/// not an error in the evaluation, it just means this pair cannot be resampled
/// under that law, so the stored pair stands.
fn law_defined(law: i32, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    if matches!(law, 3 | 5) && (x0 <= 0.0 || x1 <= 0.0) {
        return false;
    }
    if matches!(law, 4 | 5) && (y0 <= 0.0 || y1 <= 0.0) {
        return false;
    }
    true
}

/// Resample onto a single lin-lin region.
///
/// Law 2 passes through untouched, so an already-linear function keeps its
/// exact points. Law 1 is exact rather than approximated: each step becomes a
/// duplicated breakpoint carrying the jump, which is the same convention the
/// merge below relies on. The smooth laws are adaptively bisected.
pub fn linearize(t: &Tabulated1D, rel_tol: f64) -> (Vec<f64>, Vec<f64>) {
    if is_linear(t) {
        return (t.x.clone(), t.y.clone());
    }

    let mut out_x: Vec<f64> = Vec::new();
    let mut out_y: Vec<f64> = Vec::new();
    let emit = |px: f64, py: f64, out_x: &mut Vec<f64>, out_y: &mut Vec<f64>| {
        // Skip an exact repeat of the previous pair, which is what a shared
        // region boundary produces, while keeping a deliberate duplicate-x
        // jump pair.
        if out_x.last() == Some(&px) && out_y.last() == Some(&py) {
            return;
        }
        out_x.push(px);
        out_y.push(py);
    };

    for (k, &law) in t.interpolation.iter().enumerate() {
        let i_begin = if k > 0 {
            t.breakpoints[k - 1] as usize - 1
        } else {
            0
        };
        let i_end = t.breakpoints[k] as usize - 1;
        for i in i_begin..i_end {
            let (x0, y0, x1, y1) = (t.x[i], t.y[i], t.x[i + 1], t.y[i + 1]);
            emit(x0, y0, &mut out_x, &mut out_y);
            if x1 <= x0 || law == 2 {
                continue;
            }
            if law == 1 {
                // Hold y0 up to x1 and jump there.
                emit(x1, y0, &mut out_x, &mut out_y);
                continue;
            }
            if !law_defined(law, x0, y0, x1, y1) {
                continue;
            }
            // Depth first, left interval first, so the emitted points come out
            // in increasing x.
            let mut stack = vec![(x0, y0, x1, y1, 0u32)];
            while let Some((a, fa, b, fb, depth)) = stack.pop() {
                let m = 0.5 * (a + b);
                if m <= a || m >= b || depth >= MAX_DEPTH {
                    emit(b, fb, &mut out_x, &mut out_y);
                    continue;
                }
                let fm = law_value(law, x0, y0, x1, y1, m);
                let f_lin = 0.5 * (fa + fb);
                if (fm - f_lin).abs() <= rel_tol * fm.abs().max(f_lin.abs()) {
                    emit(b, fb, &mut out_x, &mut out_y);
                    continue;
                }
                stack.push((m, fm, b, fb, depth + 1));
                stack.push((a, fa, m, fm, depth + 1));
            }
        }
        emit(t.x[i_end], t.y[i_end], &mut out_x, &mut out_y);
    }
    (out_x, out_y)
}

/// Left and right limits of a stored curve at `u`, under the conventions a
/// consumer applies: zero below the first point, lin-lin between points, flat
/// above the last, and a duplicated breakpoint carrying a step jump.
fn eval_left_right(energy: &[f64], values: &[f64], u: f64) -> (f64, f64) {
    let lo = energy.partition_point(|&e| e < u);
    let hi = energy.partition_point(|&e| e <= u);
    if hi == 0 {
        return (0.0, 0.0);
    }
    if lo >= energy.len() {
        let last = *values.last().expect("non-empty");
        return (last, last);
    }
    if lo == hi {
        // Strictly inside a segment.
        let (x0, x1) = (energy[lo - 1], energy[lo]);
        let (y0, y1) = (values[lo - 1], values[lo]);
        let v = if x1 == x0 {
            y0
        } else {
            y0 + (u - x0) / (x1 - x0) * (y1 - y0)
        };
        return (v, v);
    }
    // `u` coincides with breakpoints lo..hi-1.
    let left = if lo > 0 { values[lo] } else { 0.0 };
    (left, values[hi - 1])
}

/// Sum rows sharing `(nuclide, reaction, target, quantity)`.
///
/// Duplicates are real rather than a bug to deduplicate away: two levels can
/// map to one isomeric state, and the production is their sum. Returns the
/// merged rows and how many groups were merged.
pub fn merge_duplicates(rows: Vec<BranchingRow>) -> (Vec<BranchingRow>, usize) {
    let mut order: Vec<(String, String, String, String)> = Vec::new();
    let mut groups: BTreeMap<(String, String, String, String), Vec<BranchingRow>> = BTreeMap::new();
    for row in rows {
        let key = (
            row.nuclide.clone(),
            row.reaction.clone(),
            row.target.clone(),
            row.quantity.clone(),
        );
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(row);
    }

    let mut merged_rows = Vec::new();
    let mut merged = 0;
    for key in order {
        let group = groups.remove(&key).expect("present");
        if group.len() == 1 {
            merged_rows.push(group.into_iter().next().expect("one"));
            continue;
        }
        merged += 1;

        let mut grid: Vec<f64> = group
            .iter()
            .flat_map(|r| r.energy.iter().copied())
            .collect();
        grid.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in an energy grid"));
        grid.dedup();

        let mut out_x = Vec::new();
        let mut out_y = Vec::new();
        for (i, &u) in grid.iter().enumerate() {
            let (mut left, mut right) = (0.0, 0.0);
            for r in &group {
                let (l, s) = eval_left_right(&r.energy, &r.values, u);
                left += l;
                right += s;
            }
            // A jump in the sum needs the duplicated-breakpoint form, except at
            // the first grid point where the below-threshold zero is implicit.
            if i > 0 && left != right {
                out_x.push(u);
                out_y.push(left);
            }
            out_x.push(u);
            out_y.push(right);
        }
        let mut row = group.into_iter().next().expect("one");
        row.energy = out_x;
        row.values = out_y;
        merged_rows.push(row);
    }
    (merged_rows, merged)
}

/// MT number to transmutation reaction name.
///
/// Built from the chain's own reaction set so the names match the reactions
/// subsection exactly, plus MT 4 for inelastic isomeric transitions, which the
/// chain's set omits because it produces no new nuclide.
pub(crate) fn mt_to_type() -> BTreeMap<i64, String> {
    let mut map: BTreeMap<i64, String> = BTreeMap::new();
    for info in endf::chain::REACTIONS.iter() {
        for &mt in info.mts.iter() {
            map.entry(mt as i64)
                .or_insert_with(|| info.name.to_string());
        }
    }
    map.entry(4).or_insert_with(|| "(n,n')".to_string());
    map
}

/// What a branching extraction found, so a caller can say so rather than
/// guessing from the row count.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BranchingStats {
    pub parents: usize,
    pub parents_with_data: usize,
    pub linearized_curves: usize,
    pub merged_duplicate_groups: usize,
    pub metastable_targets: Vec<String>,
    /// How many excited production levels each route of
    /// `endf::radionuclide_production::resolve_level` accounted for, by the
    /// route's label. A rebuild against another decay library, or a newer
    /// TENDL, shows up here as levels moving from `energy` to `level_index`
    /// or `unresolved`, which is the regression the plain counts above hide.
    pub level_routes: BTreeMap<String, usize>,
    /// The levels worth a look, one line each: unresolved and so taken as
    /// ground, matched only by the looser energy pass, or matched by energy
    /// while the level index pointed at another isomer.
    pub flagged_levels: Vec<String>,
    /// The reactions whose partials do not reconstruct their total within
    /// [`PARTIAL_SUM_TOLERANCE`], one line each with the worst point. yani
    /// shares the MF=3 total out in the proportions of the partials, so a
    /// defect here does not move yani's numbers; it does move a code that
    /// folds the partials as they stand, which is one way two codes on the
    /// same library come to disagree. TENDL-2017's Ir191 (n,2n), whose
    /// partials sum to 95% of MF=3 at 14 MeV and 84% at 20 MeV, is the case
    /// that prompted the check.
    pub partial_sum_mismatches: Vec<String>,
    /// MF=40 sections read, one per evaluation and MT, whatever the MT.
    pub mf40_sections: usize,
    /// Covariance blocks written, all of them and the NI ones by `lb`.
    pub mf40_blocks: usize,
    pub mf40_blocks_by_lb: BTreeMap<i64, usize>,
    /// NC blocks written. None of the published libraries carries one in
    /// MF=40, so one appearing is worth seeing here first.
    pub mf40_nc_blocks: usize,
    /// MF=40 product states in an MT with a chain kind that match no MF=9 or
    /// MF=10 state, one line each. Written, with a null `target`.
    pub mf40_unmatched_states: Vec<String>,
    /// MF=40 product states in an MT with no chain kind (TENDL's MT 18 with
    /// IZAP 0). Written, with a null `reaction` and `target`.
    pub mf40_states_without_chain_kind: usize,
    /// MF=40 product states matched to a state whose split MF=9 yields give
    /// rather than an MF=10 partial. Written, with a null `energy` and
    /// `values` since there is no MF=10 partial to give; the covariance is
    /// still of the production cross section, not of the yield.
    pub mf40_on_yield_channels: usize,
    /// Sub-subsections whose MAT1 names the evaluation itself (JEFF-4.0 U235
    /// MT 4 writes its own MAT there). Written as the tape gives it, so a
    /// reader has to compare `mat1` with `mat` rather than with zero.
    pub mf40_mat1_naming_itself: usize,
    /// Blocks correlating two different states (MT1 or XLFS1 other than the
    /// state's own), rather than a state with itself. JEFF-4.0 U235 MT 4,
    /// ground with its 77 eV isomer, is the one published case, and both its
    /// states resolve to U235: a reader has to key on (`mt`, `lfs`, `mt1`,
    /// `xlfs1`), not on (`target`, `target1`), to tell it from a self block.
    pub mf40_cross_state_blocks: usize,
    /// Product states with no sub-subsection and sub-subsections with no
    /// block, one line each. They hold no number and have no row, so they are
    /// the one part of MF=40 the file cannot show; none of the published
    /// libraries has one.
    pub mf40_without_blocks: Vec<String>,
    /// Sub-subsections whose `target1` was decided by the row's own IZAP,
    /// one line each: several products of the MT share the partner's level,
    /// and MF=40 gives no IZAP for the partner, so the converter takes the
    /// row's own. The tape's `mt1` and `xlfs1` are written as given either
    /// way; only the derived `target1` rests on this reading.
    pub mf40_partner_by_own_izap: Vec<String>,
}

/// The value of `t` at `e`, zero outside its tabulated range.
///
/// `Tabulated1D::eval` clamps to the range, which for a partial tabulated from
/// its threshold up would carry its first point down to zero energy.
fn value_or_zero(t: &Tabulated1D, e: f64) -> f64 {
    match (t.x.first(), t.x.last()) {
        (Some(&lo), Some(&hi)) if e >= lo && e <= hi => t.eval(e),
        _ => 0.0,
    }
}

/// How the partials of one reaction compare with the total they should sum to,
/// at the point where the defect matters most.
struct PartialSum {
    /// Which partials were summed.
    partials: &'static str,
    /// What they were compared with.
    total: &'static str,
    /// Summed partials over the total.
    ratio: f64,
    /// The defect as a fraction of the nonelastic cross section.
    share: f64,
    /// That point's incident energy in eV.
    energy: f64,
}

/// Compare a reaction's partials with its total.
///
/// `None` when the comparison is not meaningful: the partials leave out the
/// ground state (a file giving isomer partials alone leaves the ground state
/// as the remainder, by design), mix MF=9 and MF=10, or have no MF=3 for the
/// reaction or for the nonelastic (MT=3, else the total MT=1) to stand
/// against. The points checked, below [`PARTIAL_SUM_TOP_EV`], are the ones
/// every partial that covers them tabulates: there the partials are exact and
/// only the finer MF=3 is interpolated. Checking on the MF=3 grid reads a
/// coarse partial between its points and calls the interpolation a defect
/// (Au197 (n,n') at 270 keV, a level step MF=3 resolves and a 100 keV
/// partial grid does not), and so does a point one partial has and another
/// lacks (Ir191 (n,n') at 172 keV, the second isomer's threshold, reading
/// the ground-state partial between 100 and 200 keV).
/// Where the file gives resonance parameters that MF=3 leaves out (LRP = 1)
/// the check starts above the resonance ranges, since MF=3 there is a
/// background. The point returned is the one with the largest defect
/// relative to the nonelastic cross section.
fn partial_sum(
    material: &Material,
    mt: i32,
    states: &[RadionuclideProduction],
) -> Option<PartialSum> {
    if !states.iter().any(|s| s.lfs == 0) {
        return None;
    }
    let sigma = &material.mf3(mt)?.sigma;
    let nonelastic = &material.mf3(3).or_else(|| material.mf3(1))?.sigma;
    let (partials, total, curves, yields): (&'static str, &'static str, Vec<&Tabulated1D>, bool) =
        if states.iter().all(|s| s.cross_section.is_some()) {
            (
                "MF=10 partial cross sections",
                "the MF=3 cross section",
                states
                    .iter()
                    .filter_map(|s| s.cross_section.as_ref())
                    .collect(),
                false,
            )
        } else if states.iter().all(|s| s.yields.is_some()) {
            (
                "MF=9 yields",
                "one",
                states.iter().filter_map(|s| s.yields.as_ref()).collect(),
                true,
            )
        } else {
            return None;
        };
    let resonance_top = match material.mf1_mt451().map(|h| h.lrp) {
        Some(1) => material.mf2().map_or(0.0, |mf2| {
            mf2.isotopes
                .iter()
                .flat_map(|isotope| isotope.ranges.iter())
                .filter(|range| range.lru != 0)
                .map(|range| range.eh)
                .fold(0.0, f64::max)
        }),
        _ => 0.0,
    };
    let mut grid: Vec<f64> = curves
        .iter()
        .flat_map(|c| c.x.iter().copied())
        .filter(|&e| e > resonance_top && e <= PARTIAL_SUM_TOP_EV)
        .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let covers = |c: &Tabulated1D, e: f64| {
        c.x.first().is_some_and(|&lo| lo <= e) && c.x.last().is_some_and(|&hi| e <= hi)
    };
    let tabulates = |c: &Tabulated1D, e: f64| c.x.binary_search_by(|x| x.total_cmp(&e)).is_ok();
    grid.retain(|&e| curves.iter().all(|c| !covers(c, e) || tabulates(c, e)));
    let mut worst: Option<PartialSum> = None;
    for &energy in &grid {
        let reference = value_or_zero(nonelastic, energy);
        let total_here = value_or_zero(sigma, energy);
        if reference <= 0.0 || total_here <= 0.0 {
            continue;
        }
        let sum = curves.iter().map(|c| value_or_zero(c, energy)).sum::<f64>();
        let (ratio, defect) = if yields {
            (sum, (sum - 1.0) * total_here)
        } else {
            (sum / total_here, sum - total_here)
        };
        let share = defect.abs() / reference;
        if worst.as_ref().is_none_or(|w| share > w.share) {
            worst = Some(PartialSum {
                partials,
                total,
                ratio,
                share,
                energy,
            });
        }
    }
    worst
}

/// Accumulates branching rows, one neutron evaluation at a time.
///
/// The work is per-evaluation once the isomer table is built, so a caller
/// reading a sublibrary off disk can add each file and drop it instead of
/// holding the set. That matters for the same reason it did in
/// `Chain::from_endf` (issue #53): the 552 parents this is usually scoped to
/// are about 7 GB parsed, and the scoping is a convention of the driver rather
/// than anything this enforces, so an unscoped call is the 39 GB that used to
/// be fatal. [`extract_branching`] is this driven over a slice.
pub struct BranchingExtractor {
    mt2type: BTreeMap<i64, String>,
    isomers: endf::radionuclide_production::IsomerTable,
    tol_ev: f64,
    linearize_tol: f64,
    rows: Vec<BranchingRow>,
    covariance: Vec<BranchingCovarianceRow>,
    stats: BranchingStats,
    metastable: std::collections::BTreeSet<String>,
}

/// What one evaluation contributed, before it is merged into an extractor.
///
/// Exists so the per-evaluation work, which is independent of every other
/// evaluation's, can run off to the side and be merged back afterwards. Merging
/// in the order the files were read leaves the result identical to reading them
/// one at a time, which matters because the rows, the flagged levels and the
/// partial-sum lines are all ordered.
#[derive(Debug, Clone, Default)]
pub struct BranchingPartial {
    rows: Vec<BranchingRow>,
    covariance: Vec<BranchingCovarianceRow>,
    stats: BranchingStats,
    metastable: std::collections::BTreeSet<String>,
}

/// What [`BranchingExtractor::finish`] hands back.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    /// `branching.arrow`, duplicate targets merged.
    pub rows: Vec<BranchingRow>,
    /// `branching_covariance.arrow`, one entry per MF=40 sub-subsection.
    pub covariance: Vec<BranchingCovarianceRow>,
    pub stats: BranchingStats,
}

/// One MF=9 or MF=10 state of an evaluation, as the MF=40 match needs it.
struct ProductionState {
    target: String,
    /// The linearized MF=10 partial, `None` for a state MF=9 gives instead.
    curve: Option<(Vec<f64>, Vec<f64>)>,
    excitation: f64,
}

impl BranchingExtractor {
    /// `decay` is read only for the isomer table, so metastable evaluations
    /// suffice.
    pub fn new(decay: &[Material], tol_ev: f64, linearize_tol: f64) -> BranchingExtractor {
        BranchingExtractor {
            mt2type: mt_to_type(),
            isomers: endf::radionuclide_production::isomer_table_from_materials(decay),
            tol_ev,
            linearize_tol,
            rows: Vec::new(),
            covariance: Vec::new(),
            stats: BranchingStats::default(),
            metastable: Default::default(),
        }
    }

    /// Add one neutron evaluation's rows.
    pub fn add(&mut self, material: &Material) {
        let partial = self.extract_one(material);
        self.absorb(partial);
    }

    /// One evaluation's contribution, worked out without touching the
    /// accumulator, so callers can do this for many evaluations at once and
    /// [`Self::absorb`] the results in file order afterwards.
    pub fn extract_one(&self, material: &Material) -> BranchingPartial {
        let (mt2type, isomers, tol_ev, linearize_tol) = (
            &self.mt2type,
            &self.isomers,
            self.tol_ev,
            self.linearize_tol,
        );
        let mut out = BranchingPartial::default();
        let (rows, stats, metastable) = (&mut out.rows, &mut out.stats, &mut out.metastable);
        // Every state's resolved target and partial, for the MF=40 match below,
        // and how many MF=10 states each (kind, target) sums, across every MT
        // of the evaluation, as `merge_duplicates` sums them.
        let mut states_by_mt: BTreeMap<i32, BTreeMap<(i64, i64), ProductionState>> =
            BTreeMap::new();
        let mut summed: BTreeMap<(String, String), usize> = BTreeMap::new();

        // The evaluation names itself in MF=1/451, which is the same route
        // Chain::from_endf takes, so parent names match the reactions
        // subsection rather than a filename convention.
        let Some(meta) = material.mf1_mt451() else {
            return out;
        };
        let parent = endf::gnds_name(
            (meta.za / 1000) as u32,
            (meta.za % 1000) as u32,
            meta.liso as u32,
        );
        stats.parents += 1;
        let production = endf::radionuclide_production::radionuclide_production(material);
        let mut emitted_any = false;

        for (mt, states) in &production {
            let Some(rtype) = mt2type.get(&(*mt as i64)) else {
                continue;
            };
            if let Some(sum) = partial_sum(material, *mt, states) {
                if sum.share > PARTIAL_SUM_TOLERANCE {
                    stats.partial_sum_mismatches.push(format!(
                        "{parent} MT{mt}: {} sum to {:.3} of {} at {:.4e} eV, a defect of {:.1}% of the nonelastic cross section",
                        sum.partials, sum.ratio, sum.total, sum.energy, 100.0 * sum.share
                    ));
                }
            }
            for s in states {
                let z = s.zap / 1000;
                let a = s.zap % 1000;
                let resolved = endf::radionuclide_production::resolve_level(
                    z,
                    a,
                    s.lfs,
                    Some(s.excitation_energy()),
                    isomers,
                    tol_ev,
                );
                let liso = resolved.liso;
                let target = endf::gnds_name(z as u32, a as u32, liso as u32);
                if liso > 0 {
                    metastable.insert(target.clone());
                }
                if s.lfs > 0 {
                    *stats
                        .level_routes
                        .entry(resolved.route.label().to_string())
                        .or_insert(0) += 1;
                    let why = match (resolved.route, resolved.conflicting_liso) {
                        (LevelRoute::Unresolved, _) => {
                            Some("unresolved, taken as ground".to_string())
                        }
                        (LevelRoute::NearEnergy, _) => {
                            Some("matched by energy only within a tenth".to_string())
                        }
                        (_, Some(other)) => Some(format!(
                            "level index points at {}",
                            endf::gnds_name(z as u32, a as u32, other as u32)
                        )),
                        _ => None,
                    };
                    if let Some(why) = why {
                        stats.flagged_levels.push(format!(
                            "{parent} MT{mt} -> {target}: level {} at {:.1} keV, {why}",
                            s.lfs,
                            s.excitation_energy() / 1.0e3
                        ));
                    }
                }
                let mut curve = None;
                for (quantity, tab) in [("yield", &s.yields), ("cross_section", &s.cross_section)] {
                    let Some(tab) = tab else { continue };
                    if !is_linear(tab) {
                        stats.linearized_curves += 1;
                    }
                    let (energy, values) = linearize(tab, linearize_tol);
                    if quantity == "cross_section" {
                        curve = Some((energy.clone(), values.clone()));
                        *summed.entry((rtype.clone(), target.clone())).or_insert(0) += 1;
                    }
                    rows.push(BranchingRow {
                        nuclide: parent.clone(),
                        reaction: rtype.clone(),
                        target: target.clone(),
                        quantity: quantity.to_string(),
                        energy,
                        values,
                    });
                    emitted_any = true;
                }
                states_by_mt.entry(*mt).or_default().insert(
                    (s.zap, s.lfs),
                    ProductionState {
                        target,
                        curve,
                        excitation: s.excitation_energy(),
                    },
                );
            }
        }
        if emitted_any {
            stats.parents_with_data += 1;
        }

        // Every MF=40 section, whatever its MT, since the file is the tape's
        // MF=40 whole. Each state is matched once up front, so a block's
        // partner, which may sit in another MT's section, is found by level.
        let mf40: Vec<(i32, &Mf40)> = material
            .sections()
            .into_iter()
            .filter(|&(mf, _)| mf == 40)
            .filter_map(|(_, mt)| material.mf40(mt).map(|section| (mt, section)))
            .collect();
        let empty = BTreeMap::new();
        let matched: BTreeMap<i32, Vec<Option<&ProductionState>>> = mf40
            .iter()
            .map(|&(mt, section)| {
                let found = if mt2type.contains_key(&(mt as i64)) {
                    let states = states_by_mt.get(&mt).unwrap_or(&empty);
                    section
                        .subsections
                        .iter()
                        .map(|sub| match_mf40_state(sub, mt, meta.za, states, isomers, tol_ev))
                        .collect()
                } else {
                    vec![None; section.subsections.len()]
                };
                (mt, found)
            })
            .collect();
        let own_mat = material.mat as i64;
        for &(mt, section) in &mf40 {
            stats.mf40_sections += 1;
            let rtype = mt2type.get(&(mt as i64));
            for (state_idx, (sub, state)) in
                section.subsections.iter().zip(&matched[&mt]).enumerate()
            {
                let excitation_kev = (sub.qm - sub.qi) / 1.0e3;
                match (rtype, state) {
                    (None, _) => stats.mf40_states_without_chain_kind += 1,
                    (Some(_), None) => stats.mf40_unmatched_states.push(format!(
                        "{parent} MT{mt}: IZAP {} LFS {} at {excitation_kev:.1} keV matches no MF=9 or MF=10 state",
                        sub.izap, sub.lfs
                    )),
                    (Some(_), Some(state)) if state.curve.is_none() => {
                        stats.mf40_on_yield_channels += 1
                    }
                    _ => {}
                }
                if sub.subsubsections.is_empty() {
                    stats.mf40_without_blocks.push(format!(
                        "{parent} MT{mt}: IZAP {} LFS {} at {excitation_kev:.1} keV has no sub-subsection",
                        sub.izap, sub.lfs
                    ));
                }
                // The state's own partial, when `branching.arrow` carries it
                // summed with another state's.
                let own_curve = rtype
                    .zip(*state)
                    .filter(|(rtype, state)| {
                        summed
                            .get(&((*rtype).clone(), state.target.clone()))
                            .is_some_and(|&n| n > 1)
                    })
                    .and_then(|(_, state)| state.curve.clone());
                for (subsection_idx, ss) in sub.subsubsections.iter().enumerate() {
                    if ss.mat1 == own_mat {
                        stats.mf40_mat1_naming_itself += 1;
                    }
                    if ss.nc_subsections.is_empty() && ss.ni_subsections.is_empty() {
                        stats.mf40_without_blocks.push(format!(
                            "{parent} MT{mt}: IZAP {} LFS {} sub-subsection {subsection_idx} has no block",
                            sub.izap, sub.lfs
                        ));
                    }
                    // The partner is an MF=10 partial of this evaluation when
                    // MAT1 is 0 (or, as JEFF-4.0 U235 writes it, this MAT) and
                    // XMF1 is 10. Any other XMF1 is left unresolved rather
                    // than read as MF=10, and so is an MT1 of 0, which the
                    // manual gives no meaning in MF=40.
                    let partner_mt = ss.mt1 as i32;
                    let in_this_material = ss.mat1 == 0 || ss.mat1 == own_mat;
                    let (target1, by_own_izap) =
                        if in_this_material && ss.xmf1 == 10.0 && partner_mt != 0 {
                            partner_target(
                                sub,
                                mt,
                                meta.za,
                                partner_mt,
                                ss.xlfs1,
                                &mf40,
                                &matched,
                                &states_by_mt,
                            )
                        } else {
                            (None, false)
                        };
                    if by_own_izap {
                        stats.mf40_partner_by_own_izap.push(format!(
                            "{parent} MT{mt}: IZAP {} LFS {} sub-subsection {subsection_idx}, partner MT{partner_mt} level {}",
                            sub.izap, sub.lfs, ss.xlfs1
                        ));
                    }
                    // A block between two different states, rather than a
                    // state's covariance with itself. JEFF-4.0 U235 MT 4
                    // correlates its ground (LFS 0) with the 77 eV isomer
                    // (XLFS1 1), both of which resolve to U235.
                    if partner_mt != mt || ss.xlfs1 != sub.lfs as f64 {
                        stats.mf40_cross_state_blocks +=
                            ss.nc_subsections.len() + ss.ni_subsections.len();
                    }
                    stats.mf40_nc_blocks += ss.nc_subsections.len();
                    for ni in &ss.ni_subsections {
                        *stats.mf40_blocks_by_lb.entry(ni.lb).or_insert(0) += 1;
                    }
                    stats.mf40_blocks += ss.nc_subsections.len() + ss.ni_subsections.len();
                    out.covariance.push(BranchingCovarianceRow {
                        nuclide: parent.clone(),
                        reaction: rtype.cloned(),
                        target: state.map(|state| state.target.clone()),
                        target1,
                        energy: own_curve.as_ref().map(|(energy, _)| energy.clone()),
                        values: own_curve.as_ref().map(|(_, values)| values.clone()),
                        mat: material.mat,
                        mt,
                        za: section.za,
                        awr: section.awr,
                        lis: section.lis,
                        state_idx,
                        qm: sub.qm,
                        qi: sub.qi,
                        izap: sub.izap,
                        lfs: sub.lfs,
                        subsection_idx,
                        subsection: ss.clone(),
                    });
                }
            }
        }
        out
    }

    /// Merge one evaluation's contribution.
    ///
    /// Call order is the row order, so a caller that extracted out of order
    /// must absorb in the order the files were read to get the same answer.
    /// `merged_duplicate_groups` and `metastable_targets` are not merged here
    /// because [`Self::finish`] is what sets them.
    pub fn absorb(&mut self, partial: BranchingPartial) {
        self.rows.extend(partial.rows);
        self.covariance.extend(partial.covariance);
        self.metastable.extend(partial.metastable);
        let stats = partial.stats;
        self.stats.parents += stats.parents;
        self.stats.parents_with_data += stats.parents_with_data;
        self.stats.linearized_curves += stats.linearized_curves;
        for (route, n) in stats.level_routes {
            *self.stats.level_routes.entry(route).or_insert(0) += n;
        }
        self.stats.flagged_levels.extend(stats.flagged_levels);
        self.stats
            .partial_sum_mismatches
            .extend(stats.partial_sum_mismatches);
        self.stats.mf40_sections += stats.mf40_sections;
        self.stats.mf40_blocks += stats.mf40_blocks;
        for (lb, n) in stats.mf40_blocks_by_lb {
            *self.stats.mf40_blocks_by_lb.entry(lb).or_insert(0) += n;
        }
        self.stats.mf40_nc_blocks += stats.mf40_nc_blocks;
        self.stats
            .mf40_unmatched_states
            .extend(stats.mf40_unmatched_states);
        self.stats.mf40_states_without_chain_kind += stats.mf40_states_without_chain_kind;
        self.stats.mf40_on_yield_channels += stats.mf40_on_yield_channels;
        self.stats.mf40_mat1_naming_itself += stats.mf40_mat1_naming_itself;
        self.stats.mf40_cross_state_blocks += stats.mf40_cross_state_blocks;
        self.stats
            .mf40_without_blocks
            .extend(stats.mf40_without_blocks);
        self.stats
            .mf40_partner_by_own_izap
            .extend(stats.mf40_partner_by_own_izap);
    }

    /// The rows, the covariance and the statistics, with duplicate target
    /// groups merged.
    pub fn finish(mut self) -> Extracted {
        let (rows, merged) = merge_duplicates(self.rows);
        self.stats.merged_duplicate_groups = merged;
        self.stats.metastable_targets = self.metastable.into_iter().collect();
        Extracted {
            rows,
            covariance: self.covariance,
            stats: self.stats,
        }
    }
}

/// The MF=9 or MF=10 state one MF=40 product state is the covariance of.
///
/// By (IZAP, LFS) first, which is how MF=9 and MF=10 are joined, provided
/// the two states' excitations agree within `tol_ev`. The two files need not
/// number a level alike, though: ENDF/B-VIII.1 Pb204 MT4 gives
/// the 2.186 MeV isomer as LFS=21 in MF=10 and LFS=1 in MF=40, with the same
/// QI. So failing that, the state is resolved through the isomer table from
/// its own QM - QI, as the rows are, and matched by the chain nuclide it lands
/// on; among several states landing there, the one nearest in excitation.
/// That one must also sit within `tol_ev` of the state's own excitation:
/// `resolve_level` can land on a nuclide by level index, or by a lone isomer,
/// with no regard to energy, and a state at another level is not this one's
/// partial. Such a state is left unmatched, so it is counted as such.
/// IZAP=0 on MT 4 is matched as the target itself, since JEFF-4.0 U235 writes
/// it so where its MF=10 says 92235. Only the match reads it that way; the row
/// keeps the tape's 0.
fn match_mf40_state<'a>(
    sub: &Mf40Subsection,
    mt: i32,
    za: i64,
    states: &'a BTreeMap<(i64, i64), ProductionState>,
    isomers: &endf::radionuclide_production::IsomerTable,
    tol_ev: f64,
) -> Option<&'a ProductionState> {
    let izap = mf40_zap(sub, mt, za);
    let excitation = sub.qm - sub.qi;
    let near = |state: &ProductionState| (state.excitation - excitation).abs() <= tol_ev;
    if let Some(state) = states.get(&(izap, sub.lfs)).filter(|state| near(state)) {
        return Some(state);
    }
    let (z, a) = (izap / 1000, izap % 1000);
    let resolved = endf::radionuclide_production::resolve_level(
        z,
        a,
        sub.lfs,
        Some(excitation),
        isomers,
        tol_ev,
    );
    let target = endf::gnds_name(z as u32, a as u32, resolved.liso as u32);
    states
        .iter()
        .filter(|((zap, _), state)| *zap == izap && state.target == target)
        .map(|(_, state)| state)
        .min_by(|x, y| {
            (x.excitation - excitation)
                .abs()
                .total_cmp(&(y.excitation - excitation).abs())
        })
        .filter(|state| near(state))
}

/// The chain target of a block's partner state, level `xlfs1` of `partner_mt`
/// in this evaluation.
///
/// Found among the states of `partner_mt`'s own MF=40 section when it has one,
/// so the partner is matched as its own row's `target` was, and otherwise
/// among that MT's MF=9 and MF=10 states, whose LFS is the one XLFS1 names.
/// MF=40 gives no IZAP for the partner, so where several products of the MT
/// share the level the row's own IZAP is taken when the partner is in the
/// same MT, and the partner is otherwise left unresolved rather than guessed.
/// The second value says the own-IZAP reading was what decided it, so the
/// caller can count it: the tape does not say so, the converter reads it so.
#[allow(clippy::too_many_arguments)]
fn partner_target(
    sub: &Mf40Subsection,
    mt: i32,
    za: i64,
    partner_mt: i32,
    xlfs1: f64,
    mf40: &[(i32, &Mf40)],
    matched: &BTreeMap<i32, Vec<Option<&ProductionState>>>,
    states_by_mt: &BTreeMap<i32, BTreeMap<(i64, i64), ProductionState>>,
) -> (Option<String>, bool) {
    let own_zap = mf40_zap(sub, mt, za);
    let pick = |candidates: Vec<(i64, Option<&ProductionState>)>| {
        let own = candidates
            .iter()
            .find(|(zap, _)| partner_mt == mt && *zap == own_zap);
        match (own, candidates.as_slice()) {
            (None, [(_, state)]) => (state.map(|s| s.target.clone()), false),
            (Some((_, state)), _) => (state.map(|s| s.target.clone()), candidates.len() > 1),
            _ => (None, false),
        }
    };
    if let Some((_, section)) = mf40.iter().find(|(m, _)| *m == partner_mt) {
        return pick(
            section
                .subsections
                .iter()
                .zip(&matched[&partner_mt])
                .filter(|(other, _)| other.lfs as f64 == xlfs1)
                .map(|(other, state)| (mf40_zap(other, partner_mt, za), *state))
                .collect(),
        );
    }
    let Some(states) = states_by_mt.get(&partner_mt) else {
        return (None, false);
    };
    pick(
        states
            .iter()
            .filter(|((_, lfs), _)| *lfs as f64 == xlfs1)
            .map(|((zap, _), state)| (*zap, Some(state)))
            .collect(),
    )
}

/// The product ZA of an MF=40 state, with MT 4's IZAP of 0 (JEFF-4.0 U235)
/// read as the target itself.
fn mf40_zap(sub: &Mf40Subsection, mt: i32, za: i64) -> i64 {
    if sub.izap == 0 && mt == 4 {
        za
    } else {
        sub.izap
    }
}

/// Extract branching rows for each parent's neutron evaluation.
///
/// For a caller that holds the evaluations anyway. One reading them off disk
/// should drive [`BranchingExtractor`] over the files instead, so its peak is
/// one evaluation rather than the sublibrary.
pub fn extract_branching(
    neutron: &[Material],
    decay: &[Material],
    tol_ev: f64,
    linearize_tol: f64,
) -> Result<Extracted, Box<dyn Error>> {
    let mut extractor = BranchingExtractor::new(decay, tol_ev, linearize_tol);
    for material in neutron {
        extractor.add(material);
    }
    Ok(extractor.finish())
}

/// Write the `branching/` subsection.
pub fn write_branching(rows: &[BranchingRow], dir: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let nuclide: Vec<String> = rows.iter().map(|r| r.nuclide.clone()).collect();
    let reaction: Vec<String> = rows.iter().map(|r| r.reaction.clone()).collect();
    let target: Vec<String> = rows.iter().map(|r| r.target.clone()).collect();
    let quantity: Vec<String> = rows.iter().map(|r| r.quantity.clone()).collect();
    let energy: Vec<Vec<f64>> = rows.iter().map(|r| r.energy.clone()).collect();
    let values: Vec<Vec<f64>> = rows.iter().map(|r| r.values.clone()).collect();
    write_section(
        &dir.join("branching.arrow"),
        "branching/branching.arrow",
        vec![
            strings(&nuclide),
            strings(&reaction),
            strings(&target),
            strings(&quantity),
            list_of(&energy),
            list_of(&values),
        ],
    )
}

/// Write `branching/branching_covariance.arrow`, one row per block, and say
/// whether a file was written.
///
/// With no rows there is no file, which is how the section says "no MF=40";
/// no `.absent` marker is written, since that is a download-cache record of a
/// settled 404 rather than anything a conversion produces. A file an earlier
/// run left in `dir` is removed instead: [`write_branching`] replaces
/// `branching.arrow` on every run and the upload copies the directory whole,
/// so a stale covariance would otherwise ship beside a fresh branching curve
/// it no longer describes.
pub fn write_branching_covariance(
    rows: &[BranchingCovarianceRow],
    dir: &Path,
) -> Result<bool, Box<dyn Error>> {
    let path = dir.join("branching_covariance.arrow");
    if rows.is_empty() {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(format!("{}: {e}", path.display()).into()),
        };
    }
    std::fs::create_dir_all(dir)?;

    // The key repeats once per block of its sub-subsection, and the blocks
    // themselves are `covariance.arrow`'s columns, built by the same writer.
    let mut blocks = yamc_convert::covariance::CovarianceRows::default();
    let mut nuclide = Vec::new();
    let mut reaction = Vec::new();
    let mut target = Vec::new();
    let mut target1 = Vec::new();
    let mut energy = Vec::new();
    let mut values = Vec::new();
    let mut mat = Vec::new();
    let mut za = Vec::new();
    let mut awr = Vec::new();
    let mut lis = Vec::new();
    let mut state_idx = Vec::new();
    let mut qm = Vec::new();
    let mut qi = Vec::new();
    let mut izap = Vec::new();
    let mut lfs = Vec::new();
    for row in rows {
        // MF=40 has no lumped-reaction MTL, so none is written.
        let n = blocks.push_subsection(row.mt, row.subsection_idx, None, &row.subsection);
        for _ in 0..n {
            nuclide.push(row.nuclide.clone());
            reaction.push(row.reaction.clone());
            target.push(row.target.clone());
            target1.push(row.target1.clone());
            energy.push(row.energy.clone());
            values.push(row.values.clone());
            mat.push(row.mat);
            za.push(narrow(row.za));
            awr.push(row.awr);
            lis.push(narrow(row.lis));
            state_idx.push(narrow(row.state_idx as i64));
            qm.push(row.qm);
            qi.push(row.qi);
            izap.push(narrow(row.izap));
            lfs.push(narrow(row.lfs));
        }
    }
    let mut columns = vec![
        strings(&nuclide),
        opt_strings(&reaction),
        opt_strings(&target),
        opt_strings(&target1),
        opt_list_of(&energy),
        opt_list_of(&values),
        ints(&mat),
        ints(&za),
        floats(&awr),
        ints(&lis),
        ints(&state_idx),
        floats(&qm),
        floats(&qi),
        ints(&izap),
        ints(&lfs),
    ];
    columns.extend(blocks.columns());
    write_section(&path, "branching/branching_covariance.arrow", columns)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(x: Vec<f64>, y: Vec<f64>, law: i32) -> Tabulated1D {
        let n = x.len() as i32;
        Tabulated1D {
            x,
            y,
            breakpoints: vec![n],
            interpolation: vec![law],
            ..Default::default()
        }
    }

    /// A lin-lin function must come back with its exact points, not resampled.
    #[test]
    fn linear_regions_pass_through_untouched() {
        let t = tab(vec![1.0, 2.0, 4.0], vec![10.0, 20.0, 5.0], 2);
        let (x, y) = linearize(&t, DEFAULT_LINEARIZE_TOL);
        assert_eq!(x, t.x);
        assert_eq!(y, t.y);
    }

    /// Outside its range a partial contributes nothing, where `eval` would
    /// carry its end points outward.
    #[test]
    fn a_partial_is_zero_outside_its_range() {
        let t = tab(vec![1.0, 2.0], vec![3.0, 5.0], 2);
        assert_eq!(value_or_zero(&t, 0.5), 0.0);
        assert_eq!(value_or_zero(&t, 1.5), 4.0);
        assert_eq!(value_or_zero(&t, 2.5), 0.0);
        assert_eq!(t.eval(2.5), 5.0);
    }

    /// Histogram is exact: the step becomes a duplicated breakpoint.
    ///
    /// Resampling it like a smooth law would round the corner off, and a
    /// consumer reading lin-lin pairs would then interpolate up the step
    /// instead of jumping at it.
    #[test]
    fn histogram_becomes_a_duplicated_breakpoint() {
        let t = tab(vec![1.0, 2.0, 3.0], vec![7.0, 9.0, 0.0], 1);
        let (x, y) = linearize(&t, DEFAULT_LINEARIZE_TOL);
        // Held flat to the end of each bin, then jumping.
        assert_eq!(x, vec![1.0, 2.0, 2.0, 3.0, 3.0]);
        assert_eq!(y, vec![7.0, 7.0, 9.0, 9.0, 0.0]);
    }

    /// A log-log region is resampled densely enough that reading the result as
    /// lin-lin reproduces the declared law within the tolerance.
    #[test]
    fn log_log_is_resampled_within_tolerance() {
        // y = x^2 is exactly linear in log-log, and badly wrong read as lin-lin.
        let t = tab(vec![1.0, 100.0], vec![1.0, 10000.0], 5);
        let rel_tol = 1e-3;
        let (x, y) = linearize(&t, rel_tol);
        assert!(x.len() > 2, "nothing was resampled");
        assert!(
            x.windows(2).all(|w| w[1] >= w[0]),
            "the resampled grid is not ascending"
        );
        for i in 0..x.len() - 1 {
            let m = 0.5 * (x[i] + x[i + 1]);
            let exact = m * m;
            let linear = 0.5 * (y[i] + y[i + 1]);
            assert!(
                (linear - exact).abs() <= 4.0 * rel_tol * exact,
                "midpoint of [{}, {}] reads {linear}, the law says {exact}",
                x[i],
                x[i + 1]
            );
        }
    }

    /// A law that cannot be evaluated on an interval leaves the stored pair
    /// alone rather than producing a NaN.
    #[test]
    fn an_undefined_law_falls_back_to_the_stored_pair() {
        // Log-log with a zero ordinate: ln(0) is not a number to resample with.
        let t = tab(vec![1.0, 10.0], vec![0.0, 5.0], 5);
        let (x, y) = linearize(&t, DEFAULT_LINEARIZE_TOL);
        assert_eq!(x, vec![1.0, 10.0]);
        assert_eq!(y, vec![0.0, 5.0]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    fn row(target: &str, energy: Vec<f64>, values: Vec<f64>) -> BranchingRow {
        BranchingRow {
            nuclide: "Ac225".to_string(),
            reaction: "(n,2p)".to_string(),
            target: target.to_string(),
            quantity: "cross_section".to_string(),
            energy,
            values,
        }
    }

    /// Rows for different targets are left alone.
    #[test]
    fn distinct_targets_are_not_merged() {
        let rows = vec![
            row("Fr224", vec![1.0, 2.0], vec![1.0, 1.0]),
            row("Fr224_m1", vec![1.0, 2.0], vec![2.0, 2.0]),
        ];
        let (merged, n) = merge_duplicates(rows);
        assert_eq!(n, 0);
        assert_eq!(merged.len(), 2);
    }

    /// Two levels resolving to one isomeric state sum, on the union grid.
    ///
    /// The property that matters is not the point count but that evaluating the
    /// merged curve anywhere equals the sum of evaluating the originals.
    #[test]
    fn duplicate_targets_sum_on_the_union_grid() {
        let a = row("Fr224", vec![1.0, 3.0], vec![10.0, 30.0]);
        let b = row("Fr224", vec![2.0, 4.0], vec![100.0, 300.0]);
        let (merged, n) = merge_duplicates(vec![a.clone(), b.clone()]);
        assert_eq!(n, 1);
        assert_eq!(merged.len(), 1);
        let m = &merged[0];

        for u in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] {
            let (_, want_a) = eval_left_right(&a.energy, &a.values, u);
            let (_, want_b) = eval_left_right(&b.energy, &b.values, u);
            let (_, got) = eval_left_right(&m.energy, &m.values, u);
            assert!(
                (got - (want_a + want_b)).abs() < 1e-9,
                "at {u}: merged {got}, sum of originals {}",
                want_a + want_b
            );
        }
    }

    /// A step jump in the sum survives the merge as a duplicated breakpoint.
    ///
    /// Without it the merged curve ramps through the discontinuity and the
    /// folded production is wrong on both sides of it.
    #[test]
    fn a_jump_in_the_sum_keeps_its_duplicated_breakpoint() {
        // b starts abruptly at 2.0, so the sum jumps there.
        let a = row("Fr224", vec![1.0, 3.0], vec![10.0, 10.0]);
        let b = row("Fr224", vec![2.0, 2.0, 3.0], vec![0.0, 50.0, 50.0]);
        let (merged, _) = merge_duplicates(vec![a, b]);
        let m = &merged[0];
        let at_two: Vec<usize> = m
            .energy
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == 2.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            at_two.len(),
            2,
            "the jump at 2.0 lost its duplicated breakpoint: {:?}",
            m.energy
        );
        assert!(
            m.values[at_two[0]] < m.values[at_two[1]],
            "the duplicated breakpoint does not carry the jump"
        );
    }
}
