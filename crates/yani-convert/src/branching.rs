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

use endf::function::Tabulated1D;
use endf::radionuclide_production::LevelRoute;
use endf::Material;

use crate::{list_of, strings, write_section};

/// Relative tolerance for resampling a non-lin-lin region.
pub const DEFAULT_LINEARIZE_TOL: f64 = 1e-3;

/// Depth cap on the adaptive bisection, so a pathological interval terminates.
const MAX_DEPTH: u32 = 24;

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
fn mt_to_type() -> BTreeMap<i64, String> {
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
}

/// Extract branching rows for each parent's neutron evaluation.
///
/// `neutron` and `decay` are the parsed evaluations; the decay files are read
/// only for the isomer table, so metastable evaluations suffice.
pub fn extract_branching(
    neutron: &[Material],
    decay: &[Material],
    tol_ev: f64,
    linearize_tol: f64,
) -> Result<(Vec<BranchingRow>, BranchingStats), Box<dyn Error>> {
    let mt2type = mt_to_type();
    let isomers = endf::radionuclide_production::isomer_table_from_materials(decay);

    let mut rows = Vec::new();
    let mut stats = BranchingStats::default();
    let mut metastable: std::collections::BTreeSet<String> = Default::default();

    for material in neutron {
        // The evaluation names itself in MF=1/451, which is the same route
        // Chain::from_endf takes, so parent names match the reactions
        // subsection rather than a filename convention.
        let Some(meta) = material.mf1_mt451() else {
            continue;
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
            for s in states {
                let z = s.zap / 1000;
                let a = s.zap % 1000;
                let resolved = endf::radionuclide_production::resolve_level(
                    z,
                    a,
                    s.lfs,
                    Some(s.excitation_energy()),
                    &isomers,
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
                for (quantity, tab) in [("yield", &s.yields), ("cross_section", &s.cross_section)] {
                    let Some(tab) = tab else { continue };
                    if !is_linear(tab) {
                        stats.linearized_curves += 1;
                    }
                    let (energy, values) = linearize(tab, linearize_tol);
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
            }
        }
        if emitted_any {
            stats.parents_with_data += 1;
        }
    }

    let (rows, merged) = merge_duplicates(rows);
    stats.merged_duplicate_groups = merged;
    stats.metastable_targets = metastable.into_iter().collect();
    Ok((rows, stats))
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
