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

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int32Builder, ListBuilder, StringBuilder,
};
use endf::function::Tabulated1D;
use endf::radionuclide_production::{LevelRoute, RadionuclideProduction};
use endf::Material;

use crate::{list_of, opt_strings, strings, write_section};

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
    /// The evaluated production states this row is made from, in the order
    /// they were summed: one, or several when levels (or MTs sharing a
    /// reaction name) map to the same target.
    pub states: Vec<StateFacts>,
    /// The parent evaluation's own account of what it was normalised to, from
    /// its MF=1 description. See [`normalisation_block`].
    pub normalisation: Option<String>,
}

/// What the evaluation states about one production state behind a row.
///
/// Facts recorded for whoever decides what the list means, not used to build
/// the row: nothing here changes `energy` or `values`.
#[derive(Debug, Clone, PartialEq)]
pub struct StateFacts {
    /// The MT the state was listed under.
    pub mt: i32,
    /// The final state's level number (LFS); 0 is the ground state.
    pub lfs: i64,
    /// MF=8's LMF for the state, `None` with no MF=8 subsection for it.
    pub lmf: Option<i64>,
    /// Whether the same MT and file (MF=9 or MF=10) also list the ground state
    /// (LFS = 0) of the same product. A list without it gives isomers only.
    pub list_complete: bool,
    /// How the level was matched to the row's target.
    pub level_route: LevelRoute,
    /// The level's excitation energy in eV, as
    /// [`RadionuclideProduction::excitation_energy`] reads it: MF=8's ELFS,
    /// kept literally, where MF=8 names the state (`lmf` is `Some`, as both
    /// come from the same MF=8 subsection), and QM - QI of the MF=9/10
    /// subsection otherwise, which is computed from the tape rather than read
    /// off it. For an excited level (LFS > 0) a zero means the evaluation did not
    /// state it (JENDL-5's Cd116 MT=107 writes ELFS = 0.0), and a negative
    /// value is a sentinel (TENDL-2017's Pu237 MT=44 writes -2^31).
    pub level_energy: f64,
    /// `level_energy` less the excitation energy of the state the level was
    /// booked to (zero for the ground state), in eV. `None` when that isomer's
    /// energy is unknown to the decay data, or when `level_energy` is not a
    /// stated energy (zero for an excited level, or negative).
    pub level_energy_difference: Option<f64>,
    /// The evaluation's own MF=3 for `mt`, as stated. `None` when the file
    /// has no MF=3 section for that MT, even where it gives the same total
    /// through component MTs. Shared between the states of one MT, since a
    /// TENDL MT=102 section can run to thousands of points.
    pub mf3: Option<Arc<Tabulated1D>>,
}

/// The heading TENDL writes over the normalisation block of its MF=1
/// description.
const NORMALISATION_HEADING: &str = "Normalization to other libraries";

/// The lines of an evaluation's MF=1 description that say what it was
/// normalised to, verbatim, or `None` when it says nothing.
///
/// TENDL-2025 writes the TALYS input it normalised with under a heading of its
/// own, for example Nb93's
/// `library irdff2.0` / `norm mt=16 isom=1 width=0.05 ...`, which says the
/// (n,2n) isomer partial was normalised to IRDFF-II. The block runs from the
/// heading to the banner (a line of `*`) that starts the next part. It is kept
/// whole, commented-out lines included, with only its surrounding blank lines
/// and each line's trailing padding removed; what a line means is left to the
/// reader. 101 TENDL-2025 evaluations carry one, and so do 11 JEFF-4.0
/// evaluations taken from TENDL; TENDL-2017, ENDF/B-VIII.1, JENDL-5 and
/// FENDL-3.2d have none.
pub fn normalisation_block(material: &Material) -> Option<String> {
    let description = &material.mf1_mt451()?.description;
    let start = description
        .iter()
        .position(|line| line.trim() == NORMALISATION_HEADING)?;
    let body: Vec<&str> = description[start + 1..]
        .iter()
        .map(|line| line.trim_end())
        .take_while(|line| !line.starts_with('*'))
        .collect();
    let first = body.iter().position(|line| !line.is_empty())?;
    let last = body.iter().rposition(|line| !line.is_empty())?;
    Some(body[first..=last].join("\n"))
}

/// The tape's own limits of `curve` at `e` from the left and from the right,
/// `None` for a side the tape states nothing about.
///
/// At one of the tape's points these are its stated values, and at a repeated
/// x (a jump) the first and the last of them. Between points they are the
/// region's interpolation law, except that a log law over a zero or negative
/// end states nothing and gives `None`, unless both ends are zero, where the
/// only value it admits is zero. `Tabulated1D::eval` would give NaN there
/// (`0 * exp(ln(0/0))`) and returns one side only at a jump, so it is not used.
fn tape_limits(curve: &Tabulated1D, e: f64) -> (Option<f64>, Option<f64>) {
    let (Some(&lo), Some(&hi)) = (curve.x.first(), curve.x.last()) else {
        return (None, None);
    };
    if !(lo..=hi).contains(&e) {
        return (None, None);
    }
    let first = curve.x.partition_point(|&x| x < e);
    let past = curve.x.partition_point(|&x| x <= e);
    if first < past {
        // A histogram bin holds its start value up to its end, so arriving
        // at the point from the left gives the previous point's value.
        let left = if first > 0 && bin_law(curve, first - 1) == 1 {
            curve.y[first - 1]
        } else {
            curve.y[first]
        };
        return (Some(left), Some(curve.y[past - 1]));
    }
    // Strictly inside the bin from `first - 1` to `first`.
    let bin = first - 1;
    let law = bin_law(curve, bin);
    let (x0, y0, x1, y1) = (curve.x[bin], curve.y[bin], curve.x[first], curve.y[first]);
    let value = if matches!(law, 4 | 5) && y0 == 0.0 && y1 == 0.0 {
        Some(0.0)
    } else if law_defined(law, x0, y0, x1, y1) {
        Some(law_value(law, x0, y0, x1, y1, e)).filter(|v| v.is_finite())
    } else {
        None
    };
    (value, value)
}

/// A level's energy less that of the state it was booked to, `None` where
/// either is unknown.
///
/// An excited level (LFS > 0) at zero, or any level below zero, is an energy
/// the evaluation left unstated (an ELFS of 0.0, or a -2^31 sentinel), so it
/// is not compared: a difference of zero would read as an exact match.
fn level_energy_difference(lfs: i64, level_energy: f64, booked: Option<f64>) -> Option<f64> {
    let stated = level_energy > 0.0 || (lfs == 0 && level_energy == 0.0);
    booked.filter(|_| stated).map(|e| level_energy - e)
}

/// The interpolation law of the bin from point `bin` to the next: that of the
/// first region whose last point lies beyond the bin's start, as in `eval`.
fn bin_law(curve: &Tabulated1D, bin: usize) -> i32 {
    curve
        .breakpoints
        .iter()
        .zip(&curve.interpolation)
        .find(|(&b, _)| (bin as i64) < b as i64 - 1)
        .map_or(*curve.interpolation.last().unwrap_or(&2), |(_, &law)| law)
}

/// A curve sampled on `energy`, `None` where the tape states nothing there.
///
/// Each node takes the tape's own value (see [`tape_limits`]): the stated one
/// wherever the node is one of the tape's points, the region's law between
/// them, and none outside the tabulated range. Where `energy` repeats a node
/// (a jump in the row), the first copy takes the tape's left limit and the
/// second its right, so a jump the two share is kept whole. A single node on a
/// jump of the tape alone gets `None` rather than either side, since one
/// number cannot hold the two values the tape states there.
fn sampled_on(curve: &Tabulated1D, energy: &[f64]) -> Vec<Option<f64>> {
    energy
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let (left, right) = tape_limits(curve, e);
            let repeats_next = energy.get(i + 1) == Some(&e);
            let repeats_previous = i > 0 && energy[i - 1] == e;
            match (repeats_previous, repeats_next) {
                (false, true) => left,
                (true, false) => right,
                _ => left.filter(|_| left == right),
            }
        })
        .collect()
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
        let states: Vec<StateFacts> = group.iter().flat_map(|r| r.states.clone()).collect();
        let mut row = group.into_iter().next().expect("one");
        row.energy = out_x;
        row.values = out_y;
        row.states = states;
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
    /// ground, excited but taken as ground because the decay data has no
    /// isomer for the product, matched only by the looser energy pass, or
    /// matched by energy while the level index pointed at another isomer.
    /// Every excited level that ends up at ground is here.
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
    /// The production states of a transmutation reaction that name no single
    /// product nuclide, one line each, and so give no row. Fission (IZAP =
    /// -1) is one kind; another is a subsection with IZAP = 0 whose level no
    /// single MF=8 subsection names a product for either, where the file does
    /// not say which nuclide it is; the last is any other ZAP whose Z or A is not
    /// positive, reported with its value. None reaches this list from the six
    /// libraries yani builds from: their fission subsections are all MT=18,
    /// which is no transmutation reaction and is passed over before this, and
    /// the one evaluation writing IZAP = 0 elsewhere, FENDL-3.2d's Al27, is
    /// named by its MF=8.
    pub skipped_states: Vec<String>,
    /// One line per production list (a parent's MT, in MF=9 or MF=10) that
    /// gave rows: whether the ground state is listed, whether the file has
    /// the MF=3 the list belongs to, each state's LFS, LMF, target, route and
    /// level energy difference, and the MF=1 normalisation lines that name
    /// the MT. The same facts are stored per row in `branching.arrow`; this
    /// is the copy a build log prints.
    pub list_facts: Vec<String>,
    /// How many lists there are of each kind: `"MF=10 complete"`,
    /// `"MF=10 isomers only"`, the same for MF=9,
    /// `"no MF=3 section for the MT"` and `"normalised"` (a `norm` line of the
    /// MF=1 normalisation block names the MT).
    pub list_counts: BTreeMap<String, usize>,
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
    stats: BranchingStats,
    metastable: std::collections::BTreeSet<String>,
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
        let normalisation = normalisation_block(material);
        let mut emitted_any = false;

        for (mt, states) in &production {
            let Some(rtype) = mt2type.get(&(*mt as i64)) else {
                continue;
            };
            let mf3 = material
                .mf3(*mt)
                .map(|section| Arc::new(section.sigma.clone()));
            // Per file (MF=9 first, then MF=10): the description of each state
            // that gave a row, and whether every one had its ground listed.
            let mut list_lines: [(Vec<String>, bool); 2] = [(Vec::new(), true), (Vec::new(), true)];
            if let Some(sum) = partial_sum(material, *mt, states) {
                if sum.share > PARTIAL_SUM_TOLERANCE {
                    stats.partial_sum_mismatches.push(format!(
                        "{parent} MT{mt}: {} sum to {:.3} of {} at {:.4e} eV, a defect of {:.1}% of the nonelastic cross section",
                        sum.partials, sum.ratio, sum.total, sum.energy, 100.0 * sum.share
                    ));
                }
            }
            for s in states {
                let Some((z, a)) = s.nuclide() else {
                    let why = match s.zap {
                        -1 => "fission, which leaves no single product".to_string(),
                        0 => "no product named: IZAP = 0 in MF=9/10, and MF=8 has no single \
                              subsection for the level naming one (none, several, or ZAP = 0)"
                            .to_string(),
                        zap => format!("ZAP {zap} names no single nuclide"),
                    };
                    stats
                        .skipped_states
                        .push(format!("{parent} MT{mt} level {}: {why}", s.lfs));
                    continue;
                };
                let resolved = endf::radionuclide_production::resolve_level(
                    z,
                    a,
                    s.lfs,
                    s.excitation_energy(),
                    isomers,
                    tol_ev,
                );
                let liso = resolved.liso;
                let target = endf::gnds_name(z as u32, a as u32, liso as u32);
                // The tape's own value, kept as a fact: MF=8's ELFS where it
                // gives one and QM - QI otherwise. excitation_energy() reads
                // it (an unstated excited level is None there), which is not
                // what this column records.
                let level_energy = s.elfs.unwrap_or(s.qm - s.qi);
                let booked_energy = if liso == 0 {
                    Some(0.0)
                } else {
                    isomers
                        .get(&(z, a))
                        .and_then(|table| table.get(&liso))
                        .and_then(|isomer| isomer.e_iso)
                };
                let level_energy_difference =
                    level_energy_difference(s.lfs, level_energy, booked_energy);
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
                        (LevelRoute::NoIsomers, _) => {
                            Some("no isomer in the decay data, taken as ground".to_string())
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
                        let at = match s.excitation_energy() {
                            Some(e) => format!("at {:.1} keV", e / 1.0e3),
                            None => "with no excitation energy".to_string(),
                        };
                        stats.flagged_levels.push(format!(
                            "{parent} MT{mt} -> {target}: level {} {at}, {why}",
                            s.lfs
                        ));
                    }
                }
                for (file, (quantity, tab)) in
                    [("yield", &s.yields), ("cross_section", &s.cross_section)]
                        .into_iter()
                        .enumerate()
                {
                    let Some(tab) = tab else { continue };
                    if !is_linear(tab) {
                        stats.linearized_curves += 1;
                    }
                    let (energy, values) = linearize(tab, linearize_tol);
                    let list_complete = states.iter().any(|g| {
                        g.zap == s.zap
                            && g.lfs == 0
                            && [&g.yields, &g.cross_section][file].is_some()
                    });
                    let lmf = s.lmf.map_or("none".to_string(), |lmf| lmf.to_string());
                    let difference = level_energy_difference
                        .map_or("unknown".to_string(), |d| format!("{:+.3} keV", d / 1.0e3));
                    let (lines, all_complete) = &mut list_lines[file];
                    lines.push(format!(
                        "LFS {} -> {target} (LMF {lmf}, {}, {difference})",
                        s.lfs,
                        resolved.route.label()
                    ));
                    *all_complete &= list_complete;
                    rows.push(BranchingRow {
                        nuclide: parent.clone(),
                        reaction: rtype.clone(),
                        target: target.clone(),
                        quantity: quantity.to_string(),
                        energy,
                        values,
                        states: vec![StateFacts {
                            mt: *mt,
                            lfs: s.lfs,
                            lmf: s.lmf,
                            list_complete,
                            level_route: resolved.route,
                            level_energy,
                            level_energy_difference,
                            mf3: mf3.clone(),
                        }],
                        normalisation: normalisation.clone(),
                    });
                    emitted_any = true;
                }
            }

            // The build log's copy of the facts just stored, one line per list.
            let token = format!("mt={mt}");
            let norm_lines: Vec<&str> = normalisation
                .iter()
                .flat_map(|block| block.lines())
                .filter(|line| {
                    line.starts_with("library ") || line.split_whitespace().any(|t| t == token)
                })
                .collect();
            // Counted only for a live `norm` line; a commented-out one (`#norm`)
            // is still printed, since it is in the file.
            let normalised = norm_lines.iter().any(|line| line.starts_with("norm "));
            for (file, (lines, complete)) in list_lines.into_iter().enumerate() {
                if lines.is_empty() {
                    continue;
                }
                let mf = [9, 10][file];
                let kind = if complete { "complete" } else { "isomers only" };
                *stats
                    .list_counts
                    .entry(format!("MF={mf} {kind}"))
                    .or_insert(0) += 1;
                // Only the section for this MT is looked for. An evaluation
                // can give the same total through component MTs instead (Ag115
                // in ENDF/B-VIII.1 has MT=600-649 and no MT=103), and those are
                // not summed into one here.
                if mf3.is_none() {
                    *stats
                        .list_counts
                        .entry("no MF=3 section for the MT".to_string())
                        .or_insert(0) += 1;
                }
                let mut line = format!(
                    "{parent} MT{mt} {rtype} MF={mf}: {}, {}; {}",
                    if complete {
                        "ground listed"
                    } else {
                        "isomers only"
                    },
                    if mf3.is_some() {
                        format!("MF=3 section for MT={mt}")
                    } else {
                        format!("no MF=3 section for MT={mt}")
                    },
                    lines.join("; ")
                );
                if normalised {
                    *stats
                        .list_counts
                        .entry("normalised".to_string())
                        .or_insert(0) += 1;
                    line.push_str(&format!("; normalised: {}", norm_lines.join(" | ")));
                }
                stats.list_facts.push(line);
            }
        }
        if emitted_any {
            stats.parents_with_data += 1;
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
        self.stats.skipped_states.extend(stats.skipped_states);
        self.stats.list_facts.extend(stats.list_facts);
        for (kind, n) in stats.list_counts {
            *self.stats.list_counts.entry(kind).or_insert(0) += n;
        }
    }

    /// The rows and statistics, with duplicate target groups merged.
    pub fn finish(mut self) -> (Vec<BranchingRow>, BranchingStats) {
        let (rows, merged) = merge_duplicates(self.rows);
        self.stats.merged_duplicate_groups = merged;
        self.stats.metastable_targets = self.metastable.into_iter().collect();
        (rows, self.stats)
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
) -> Result<(Vec<BranchingRow>, BranchingStats), Box<dyn Error>> {
    let mut extractor = BranchingExtractor::new(decay, tol_ev, linearize_tol);
    for material in neutron {
        extractor.add(material);
    }
    Ok(extractor.finish())
}

/// Write the `branching/` subsection.
///
/// The columns after `values` are the per-state facts of [`StateFacts`], one
/// list item per state in the order the row summed them, and the parent's
/// [`normalisation_block`]. `mf3_cross_section` holds, per state, the MF=3
/// for its MT sampled on this row's `energy`, so a reader can set each partial
/// against the evaluation's own total point by point. It is the tape's value
/// on the nodes the two grids share and the tape's own law between its points,
/// not a copy of MF=3: an item is null where MF=3 is not tabulated, where a
/// log law meets a zero, and on a single node where MF=3 alone jumps (see
/// [`sampled_on`]), and the whole entry is null where the file has no MF=3
/// section for the MT. `level_energy` is MF=8's ELFS as the tape gives it
/// where `lmf` is not null, and QM - QI of the MF=9/10 subsection where it is
/// (see [`StateFacts::level_energy`]): for an excited level a zero means the
/// evaluation did not state the energy, and a negative value is a sentinel;
/// `level_energy_difference` is null for both.
pub fn write_branching(rows: &[BranchingRow], dir: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let nuclide: Vec<String> = rows.iter().map(|r| r.nuclide.clone()).collect();
    let reaction: Vec<String> = rows.iter().map(|r| r.reaction.clone()).collect();
    let target: Vec<String> = rows.iter().map(|r| r.target.clone()).collect();
    let quantity: Vec<String> = rows.iter().map(|r| r.quantity.clone()).collect();
    let energy: Vec<Vec<f64>> = rows.iter().map(|r| r.energy.clone()).collect();
    let values: Vec<Vec<f64>> = rows.iter().map(|r| r.values.clone()).collect();
    let normalisation: Vec<Option<String>> = rows.iter().map(|r| r.normalisation.clone()).collect();

    let mut mt = ListBuilder::new(Int32Builder::new());
    let mut lfs = ListBuilder::new(Int32Builder::new());
    let mut lmf = ListBuilder::new(Int32Builder::new());
    let mut list_complete = ListBuilder::new(BooleanBuilder::new());
    let mut level_route = ListBuilder::new(StringBuilder::new());
    let mut level_energy = ListBuilder::new(Float64Builder::new());
    let mut level_energy_difference = ListBuilder::new(Float64Builder::new());
    let mut mf3 = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    for row in rows {
        for state in &row.states {
            mt.values().append_value(state.mt);
            // LFS and LMF are held as the ENDF reader's i64 but stored as
            // Int32 like MT, which the reader's BranchState matches; a value
            // that does not fit fails the write rather than wrapping.
            lfs.values().append_value(i32::try_from(state.lfs)?);
            lmf.values()
                .append_option(state.lmf.map(i32::try_from).transpose()?);
            list_complete.values().append_value(state.list_complete);
            level_route.values().append_value(state.level_route.label());
            level_energy.values().append_value(state.level_energy);
            level_energy_difference
                .values()
                .append_option(state.level_energy_difference);
            match &state.mf3 {
                Some(curve) => {
                    for value in sampled_on(curve, &row.energy) {
                        mf3.values().values().append_option(value);
                    }
                    mf3.values().append(true);
                }
                None => mf3.values().append(false),
            }
        }
        mt.append(true);
        lfs.append(true);
        lmf.append(true);
        list_complete.append(true);
        level_route.append(true);
        level_energy.append(true);
        level_energy_difference.append(true);
        mf3.append(true);
    }

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
            Arc::new(mt.finish()),
            Arc::new(lfs.finish()),
            Arc::new(lmf.finish()),
            Arc::new(list_complete.finish()),
            Arc::new(level_route.finish()),
            Arc::new(level_energy.finish()),
            Arc::new(level_energy_difference.finish()),
            Arc::new(mf3.finish()),
            opt_strings(&normalisation),
        ],
    )
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
            states: Vec::new(),
            normalisation: None,
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

    /// JEFF-4.0's Cs134 MT=102 is one log-log region starting (1e-5, 0),
    /// (308.92, 0), (308.92, 6.01). `Tabulated1D::eval` gives NaN between the
    /// two zeros; the tape admits only zero there, and states nothing where a
    /// log law has one zero end.
    #[test]
    fn a_log_law_over_zeros_is_zero_or_unstated_never_nan() {
        let curve = tab(
            vec![1e-5, 308.92, 308.92, 1000.0, 2000.0],
            vec![0.0, 0.0, 6.01, 3.0, 0.0],
            5,
        );
        assert!(curve.eval(10.0).is_nan(), "the case this guards against");
        let sampled = sampled_on(&curve, &[1e-6, 1e-5, 10.0, 500.0, 1500.0, 2000.0, 3000.0]);
        assert_eq!(sampled[0], None, "below the table");
        assert_eq!(sampled[1], Some(0.0), "a tape point");
        assert_eq!(sampled[2], Some(0.0), "between two zeros");
        let between = sampled[3].expect("both ends positive");
        let expected = law_value(5, 308.92, 6.01, 1000.0, 3.0, 500.0);
        assert_eq!(between, expected);
        assert_eq!(sampled[4], None, "a log law with one zero end");
        assert_eq!(sampled[5], Some(0.0), "a tape point");
        assert_eq!(sampled[6], None, "above the table");
        assert!(sampled.iter().flatten().all(|v| v.is_finite()));
    }

    /// At a jump of the tape, a row that repeats the node gets the tape's left
    /// then right values; a row with a single node there gets neither.
    #[test]
    fn a_jump_in_mf3_keeps_both_sides_or_none() {
        let curve = tab(vec![1.0, 2.0, 2.0, 3.0], vec![1.0, 1.0, 5.0, 5.0], 2);
        assert_eq!(
            sampled_on(&curve, &[1.5, 2.0, 2.0, 3.0]),
            [Some(1.0), Some(1.0), Some(5.0), Some(5.0)]
        );
        assert_eq!(
            sampled_on(&curve, &[1.5, 2.0, 3.0]),
            [Some(1.0), None, Some(5.0)]
        );
        // A row jump where the tape is continuous takes the one value twice.
        assert_eq!(sampled_on(&curve, &[3.0, 3.0]), [Some(5.0), Some(5.0)]);
    }

    /// Under a histogram law every point where the value changes is a jump:
    /// the tape's left limit there is the previous point's value. JENDL-5's
    /// Eu151 MT=107 has such a region.
    #[test]
    fn a_histogram_step_is_a_jump_of_the_tape() {
        let curve = tab(vec![1.0, 2.0, 3.0], vec![4.0, 6.0, 6.0], 1);
        assert_eq!(tape_limits(&curve, 2.0), (Some(4.0), Some(6.0)));
        assert_eq!(
            sampled_on(&curve, &[1.5, 2.0, 2.0, 2.5]),
            [Some(4.0), Some(4.0), Some(6.0), Some(6.0)]
        );
        assert_eq!(sampled_on(&curve, &[2.0, 3.0]), [None, Some(6.0)]);
    }

    /// An excited level the evaluation gives no energy for is not compared
    /// with the state it was booked to. JENDL-5's Cd116 MT=107 LFS=1 writes
    /// ELFS = 0.0 and TENDL-2017's Pu237 MT=44 LFS=1 writes -2^31.
    #[test]
    fn an_unstated_level_energy_has_no_difference() {
        assert_eq!(level_energy_difference(1, 0.0, Some(0.0)), None);
        assert_eq!(level_energy_difference(1, -2.147484e9, Some(0.0)), None);
        assert_eq!(level_energy_difference(0, 0.0, Some(0.0)), Some(0.0));
        assert_eq!(level_energy_difference(1, 100.0, Some(90.0)), Some(10.0));
        assert_eq!(level_energy_difference(1, 100.0, None), None);
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
