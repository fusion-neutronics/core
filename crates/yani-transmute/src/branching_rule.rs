//! The nominal isomeric-branching rule: the branching evaluation gives the
//! split, the transport cross-section library gives the total.
//!
//! One production formula serves every final state `s` of a parent's reaction
//! MT: `production_s = collapse of p_s(E)`, taken in the same walk, under the
//! same within-group weight and the same shielded flux shape as the transport
//! total `R_MT` (on the coupled path, scored at the same collisions). What
//! `p_s(E)` is depends on how the evaluation represents the list, which the
//! converter records per state (`BranchState::list_complete`) and nothing here
//! infers:
//!
//! * a complete MF=10 list (its ground state listed): a pointwise share,
//!   `p_s = sigma_s(E) / sum_s' sigma_s'(E) * sigma_MT(E)`;
//! * a complete MF=9 list: `p_s = y_s(E) / sum_s' y_s'(E) * sigma_MT(E)`;
//! * an MF=9 list of isomers only: `p_s = y_s(E) * sigma_MT(E)`, the LMF=9
//!   definition, the ground state taking the rest;
//! * an MF=10 list of isomers only, and every `(n,n')` MF=10 list: the partial
//!   itself, `p_s = sigma_s(E)`, an absolute production (ENDF-102 10.3.1), the
//!   ground state taking the rest of `sigma_MT` (for `(n,n')` the ground is
//!   the parent, so nothing is left to take).
//!
//! Beyond a curve's last tabulated point the fraction is held, not the value:
//! a share list's partials are held flat, which holds their ratio; an absolute
//! partial goes on as `sigma_s(E_l) / sigma_MT(E_l) * sigma_MT(E)`. Where a
//! share list's listed values are all zero under a live total (below every
//! threshold of the list, or past a list that ends in zeros) the nearest
//! defined split is held. Either is an extrapolation of the evaluation, and
//! the rate it carries is reported.
//!
//! Where the rule meets a value it cannot represent (absolute partials above
//! the transport total, isomer yields summing above one, a negative partial)
//! the excess is clipped at that energy and its folded rate is reported; see
//! [`BRANCHING_RATE_TOLERANCE`] for when that, or a held fraction, stops a run.

use std::borrow::Cow;
use std::collections::HashMap;

use yamc_nuclide::reaction::Reaction;
use yani::{BranchCurve, BranchQuantity};

/// The largest share of a parent's neutron removal rate that may rest on a
/// value the rule cannot represent, before a solve refuses to run.
///
/// Two things are measured against it, each as a rate over the parent's
/// removal rate in the run's own spectrum: the production clipped because it
/// was impossible (absolute partials above the transport total, isomer yields
/// above one, negative partials), and the production resting on a held
/// fraction outside a list's tabulated range. MT=5's share is reported beside
/// them, not refused yet (see [`measure_unmodelled_mt5`]).
///
/// It is the solver's tolerance for a rate it has no data for, the same
/// 0.1% as [`crate::multigroup::ABOVE_EVALUATION_TOLERANCE`], which refuses a
/// spectrum whose flux above an evaluation's last energy would understate a
/// nuclide's rates by more than that. Below it the rate is carried and
/// reported; above it the run stops and says which parent and how much.
pub const BRANCHING_RATE_TOLERANCE: f64 = 1.0e-3;

/// The reaction kind whose ground state is the parent itself.
pub(crate) const INELASTIC: &str = "(n,n')";
/// The transport MT of `(n,n')`, the total its partials are shares of.
pub(crate) const MT_INELASTIC: i32 = 4;
/// MT=5, `(n,anything)`: a reaction whose products the chain does not model.
pub(crate) const MT_ANYTHING: i32 = 5;

/// What a list's shares are taken over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Denominator {
    /// The sum of the listed values at each energy: a complete list, whose
    /// states are all of the reaction.
    ListedSum,
    /// The transport total: a list of isomers only, whose ground state is the
    /// remainder, and every `(n,n')` partial.
    TransportTotal,
}

/// One parent's list for one reaction kind, as the converter recorded it.
pub(crate) struct ListRule<'a> {
    pub(crate) parent: String,
    pub(crate) kind: String,
    /// The transport MT whose cross section is the list's total.
    pub(crate) mt: Option<i32>,
    /// MF=9 yields rather than MF=10 partial cross sections.
    pub(crate) yields: bool,
    /// The list names the product's ground state.
    pub(crate) complete: bool,
    pub(crate) denominator: Denominator,
    /// The listed curves, in the order the branching table holds them:
    /// borrowed from the table on the multigroup path, owned by the tally.
    pub(crate) curves: Vec<Cow<'a, BranchCurve>>,
    /// Whether each curve makes a product: every curve but the `(n,n')`
    /// ground self row, which leaves the parent as it was.
    pub(crate) produces: Vec<bool>,
    /// MF=9 curves passed over because the same reaction has MF=10 partials.
    pub(crate) passed_over: Vec<Cow<'a, BranchCurve>>,
    /// The union of the curves' grids, ascending and deduplicated.
    pub(crate) nodes: Vec<f64>,
    /// The lowest first energy and the lowest last energy over the curves.
    first: f64,
    last: f64,
    /// Where the listed values most exceed the evaluation's own total
    /// (`mf3_cross_section`), for a list whose values are not normalised:
    /// `(energy [eV], ratio)`. `None` when nowhere above it, or unstated.
    pub(crate) own_total_excess: Option<(f64, f64)>,
}

/// What one energy point of a list gives.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Point {
    /// The factor a share list's values are multiplied by for their
    /// production (see [`Bound::point`]); one for an absolute list.
    pub(crate) factor: f64,
    /// The production removed because it could not be represented [barns].
    pub(crate) clipped: f64,
    /// Whether any value at this energy was held beyond its tabulated range.
    pub(crate) extrapolated: bool,
    /// Whether a share list's split here is the one held from a node, whose
    /// two factors need not meet a neighbour's.
    pub(crate) held: bool,
}

/// Well formed: a grid and values of one length, non-empty. A curve that is
/// not would panic in [`curve_interp`].
pub(crate) fn well_formed(c: &BranchCurve) -> bool {
    !c.energy.is_empty() && c.energy.len() == c.values.len()
}

impl<'a> ListRule<'a> {
    /// The list for `(parent, kind)`, or `None` when it has no usable curve.
    ///
    /// Refuses a list whose curves carry no converter facts (a
    /// `branching.arrow` written before they were stored), since what a
    /// partial means cannot be read from it, and a list whose states disagree
    /// on whether the ground state is listed, which the converter never
    /// writes.
    pub(crate) fn new(
        parent: &str,
        kind: &str,
        curves: &'a [BranchCurve],
    ) -> Result<Option<ListRule<'a>>, String> {
        let partials: Vec<&BranchCurve> = curves
            .iter()
            .filter(|c| well_formed(c) && c.quantity == BranchQuantity::CrossSection)
            .collect();
        let yields: Vec<&BranchCurve> = curves
            .iter()
            .filter(|c| well_formed(c) && c.quantity == BranchQuantity::Yield)
            .collect();
        let (chosen, passed_over, is_yield) = if partials.is_empty() {
            (yields, Vec::new(), true)
        } else {
            (partials, yields, false)
        };
        if chosen.is_empty() {
            return Ok(None);
        }
        let mut complete: Option<bool> = None;
        for c in &chosen {
            if c.states.is_empty() {
                return Err(format!(
                    "the isomeric branching for {parent} {kind} (target {}) carries no list \
                     facts: its branching.arrow was written before the converter stored \
                     whether each list names its ground state. What a partial means (a share \
                     of the reaction or an absolute production) is read from those facts and \
                     is not guessed, so this file cannot be used. Re-convert the branching \
                     subsection with this version (yani.convert_branching), or fetch a \
                     republished one.",
                    c.target
                ));
            }
            for s in c.states.iter() {
                match complete {
                    None => complete = Some(s.list_complete),
                    Some(prev) if prev != s.list_complete => {
                        return Err(format!(
                            "the isomeric branching for {parent} {kind} mixes states whose \
                             list names its ground state with states whose list does not \
                             (at {}, MT={} LFS={}). The converter writes one list per MT and \
                             file, so this file was not written by it.",
                            c.target, s.mt, s.lfs
                        ));
                    }
                    _ => {}
                }
            }
        }
        let complete = complete.unwrap_or(false);
        let inelastic = kind == INELASTIC;
        let mt = if inelastic {
            Some(MT_INELASTIC)
        } else {
            crate::reaction_type_to_mt(kind)
        };
        let denominator = if (inelastic && !is_yield) || !complete {
            Denominator::TransportTotal
        } else {
            Denominator::ListedSum
        };
        let produces: Vec<bool> = chosen
            .iter()
            .map(|c| !(inelastic && c.target == parent))
            .collect();
        // An absolute partial goes on past its last point as a share of the
        // total, which is a step where the total is zero there; one ulp up
        // is where the step lands, so the walk sees it rather than a slope
        // across the next segment.
        let absolute = !is_yield && denominator == Denominator::TransportTotal;
        let mut nodes: Vec<f64> = chosen
            .iter()
            .flat_map(|c| {
                let past = absolute.then(|| c.energy[c.energy.len() - 1].next_up());
                c.energy.iter().copied().chain(past)
            })
            .collect();
        nodes.sort_by(f64::total_cmp);
        nodes.dedup();
        let first = chosen
            .iter()
            .map(|c| c.energy[0])
            .fold(f64::INFINITY, f64::min);
        let last = chosen
            .iter()
            .map(|c| c.energy[c.energy.len() - 1])
            .fold(f64::INFINITY, f64::min);
        let mut rule = ListRule {
            parent: parent.to_string(),
            kind: kind.to_string(),
            mt,
            yields: is_yield,
            complete,
            denominator,
            curves: chosen.into_iter().map(Cow::Borrowed).collect(),
            produces,
            passed_over: passed_over.into_iter().map(Cow::Borrowed).collect(),
            nodes,
            first,
            last,
            own_total_excess: None,
        };
        rule.own_total_excess = rule.excess_over_own_total();
        Ok(Some(rule))
    }

    /// Whether the listed values are absolute productions rather than shares.
    pub(crate) fn is_absolute(&self) -> bool {
        !self.yields && self.denominator == Denominator::TransportTotal
    }

    /// Whether the chain's ground state takes the rest of the reaction.
    pub(crate) fn has_remainder(&self) -> bool {
        self.denominator == Denominator::TransportTotal
    }

    /// Where the values that are not normalised most exceed the evaluation's
    /// own total: the producing partials over `mf3_cross_section`, or the
    /// isomer yields over one. Only for lists whose values stand as they are,
    /// since a normalised list cannot exceed anything.
    fn excess_over_own_total(&self) -> Option<(f64, f64)> {
        if self.denominator != Denominator::TransportTotal {
            return None;
        }
        let mut worst: Option<(f64, f64)> = None;
        let mut consider = |e: f64, ratio: f64| {
            if ratio > 1.0 && worst.is_none_or(|(_, r)| ratio > r) {
                worst = Some((e, ratio));
            }
        };
        if self.yields {
            for &e in &self.nodes {
                let sum: f64 = self
                    .curves
                    .iter()
                    .zip(&self.produces)
                    .filter(|(_, p)| **p)
                    .map(|(c, _)| curve_interp(&c.energy, &c.values, e))
                    .sum();
                consider(e, sum);
            }
            return worst;
        }
        // The own MF=3 is stated on each curve's nodes; all states of one list
        // share the MT, so any producing curve's first state carries it.
        for (c, _) in self.curves.iter().zip(&self.produces).filter(|(_, p)| **p) {
            let Some(mf3) = c.states.first().and_then(|s| s.mf3_cross_section.as_ref()) else {
                continue;
            };
            for (k, &e) in c.energy.iter().enumerate() {
                let Some(Some(total)) = mf3.get(k).copied() else {
                    continue;
                };
                let sum: f64 = self
                    .curves
                    .iter()
                    .zip(&self.produces)
                    .filter(|(_, p)| **p)
                    .map(|(c, _)| curve_interp(&c.energy, &c.values, e))
                    .sum();
                if total > 0.0 {
                    consider(e, sum / total);
                } else if sum > 0.0 {
                    consider(e, f64::INFINITY);
                }
            }
        }
        worst
    }

    /// The same list, owning its curves, for a holder that outlives the
    /// branching table it was read from.
    pub(crate) fn into_owned(self) -> ListRule<'static> {
        let own = |v: Vec<Cow<'a, BranchCurve>>| -> Vec<Cow<'static, BranchCurve>> {
            v.into_iter().map(|c| Cow::Owned(c.into_owned())).collect()
        };
        ListRule {
            parent: self.parent,
            kind: self.kind,
            mt: self.mt,
            yields: self.yields,
            complete: self.complete,
            denominator: self.denominator,
            curves: own(self.curves),
            produces: self.produces,
            passed_over: own(self.passed_over),
            nodes: self.nodes,
            first: self.first,
            last: self.last,
            own_total_excess: self.own_total_excess,
        }
    }

    /// Each curve's share of the transport total at its last point, which an
    /// absolute partial keeps above that point: `None` for a share list, and
    /// where there is no total to share (the partial is then held flat, as a
    /// curve is); zero where the total is zero there, which ends the partial.
    pub(crate) fn tails(&self, total: Option<&Reaction>) -> Vec<Option<f64>> {
        self.curves
            .iter()
            .map(|c| {
                let total = total?;
                if !self.is_absolute() {
                    return None;
                }
                let (&e_last, &v_last) = (c.energy.last()?, c.values.last()?);
                match total.cross_section_at(e_last) {
                    Some(t) if t > 0.0 => Some(v_last / t),
                    _ => Some(0.0),
                }
            })
            .collect()
    }
}

/// A [`ListRule`] with its transport total and its [`ListRule::tails`] for it.
pub(crate) struct Bound<'r, 'a> {
    pub(crate) rule: &'r ListRule<'a>,
    pub(crate) total: Option<&'r Reaction>,
    pub(crate) tail: &'r [Option<f64>],
}

impl Bound<'_, '_> {
    /// The transport total at `e` [barns], zero where there is none.
    pub(crate) fn total_at(&self, e: f64) -> f64 {
        self.total
            .and_then(|r| r.cross_section_at(e))
            .unwrap_or(0.0)
    }

    /// The list's values at `e`, into `out` (one per curve), given the
    /// transport total `t` there.
    ///
    /// For an absolute list `out[s]` is the production itself, in barns. For a
    /// share list the production is `out[s] * factor`, split into the two
    /// factors the evaluation and the library tabulate as linear, so a walk
    /// can integrate their product exactly on each segment: for MF=10 the
    /// partial `sigma_s` and the ratio `sigma_MT / sum sigma`, for MF=9 the
    /// fraction `y_s` (over `sum y` in a complete list) and `sigma_MT`. The
    /// curve that makes no product (the `(n,n')` ground self row) reads zero.
    pub(crate) fn point(&self, e: f64, t: f64, out: &mut [f64]) -> Point {
        let rule = self.rule;
        let mut point = Point {
            factor: 1.0,
            ..Default::default()
        };
        if rule.is_absolute() {
            let mut sum = 0.0;
            for (k, c) in rule.curves.iter().enumerate() {
                if !rule.produces[k] {
                    out[k] = 0.0;
                    continue;
                }
                let e_last = c.energy[c.energy.len() - 1];
                let mut v = if e <= e_last {
                    curve_interp(&c.energy, &c.values, e)
                } else {
                    point.extrapolated = true;
                    match self.tail.get(k).copied().flatten() {
                        Some(share) => share * t,
                        None => c.values[c.values.len() - 1],
                    }
                };
                if v < 0.0 {
                    point.clipped += -v;
                    v = 0.0;
                }
                out[k] = v;
                sum += v;
            }
            // Only a known total bounds the partials: with none there is
            // nothing to be above.
            if self.total.is_some() && sum > t {
                point.clipped += sum - t;
                let scale = if sum > 0.0 { t.max(0.0) / sum } else { 0.0 };
                for v in out.iter_mut() {
                    *v *= scale;
                }
            }
            return point;
        }

        // A yield is a fraction, held at both ends of its range; a partial is
        // zero below its first point, where a share list is covered by the
        // held split below.
        // What is held only matters where there is a total to share out.
        let live = t > 0.0;
        let (mut sum, mut negative) = (0.0, 0.0);
        for (k, c) in rule.curves.iter().enumerate() {
            let mut v = if rule.yields && e < c.energy[0] {
                // A yield starting at zero starts at its threshold, and
                // holding the zero below it says nothing the tape does not.
                point.extrapolated |= live && c.values[0] != 0.0;
                c.values[0]
            } else {
                curve_interp(&c.energy, &c.values, e)
            };
            if e > c.energy[c.energy.len() - 1] {
                point.extrapolated |= live;
            }
            let counted = rule.produces[k] || rule.denominator == Denominator::ListedSum;
            if v < 0.0 {
                if counted {
                    negative -= v;
                }
                v = 0.0;
            }
            out[k] = v;
            if counted {
                sum += v;
            }
        }
        // A negative value is impossible and is taken as zero. What it would
        // have moved is its share of the total: of the listed values' sum for
        // a normalised list, of the whole for a yield standing as it is.
        if negative > 0.0 {
            point.clipped += match rule.denominator {
                Denominator::ListedSum => negative / (sum + negative) * t,
                Denominator::TransportTotal => negative * t,
            };
        }
        match rule.denominator {
            Denominator::ListedSum => {
                if sum > 0.0 {
                    if rule.yields {
                        for v in out.iter_mut() {
                            *v /= sum;
                        }
                        point.factor = t;
                    } else {
                        point.factor = t / sum;
                    }
                } else {
                    // No split here: hold the nearest one. Only where there is
                    // a total to share out is that an extrapolation of the
                    // evaluation; below a common threshold it is only what
                    // the walk needs to integrate up to the next node.
                    point.extrapolated |= live;
                    point.held = true;
                    let (held, node) = self.held_split(e, out);
                    point.factor = if rule.yields {
                        t
                    } else if held <= 0.0 {
                        out.iter_mut().for_each(|v| *v = 0.0);
                        t
                    } else if live {
                        // The held fractions as partials of the sum they were
                        // held from, so the factors meet those of the node.
                        for v in out.iter_mut() {
                            *v *= held;
                        }
                        t / held
                    } else {
                        // The partials are zero here, and the ratio of the
                        // total to them is taken where they are not, which is
                        // its limit at a threshold they share with the total.
                        out.iter_mut().for_each(|v| *v = 0.0);
                        self.total_at(node) / held
                    };
                }
            }
            Denominator::TransportTotal => {
                if sum > 1.0 {
                    point.clipped += (sum - 1.0) * t;
                    for v in out.iter_mut() {
                        *v /= sum;
                    }
                }
                point.factor = t;
            }
        }
        for (k, v) in out.iter_mut().enumerate() {
            if !rule.produces[k] {
                *v = 0.0;
            }
        }
        point
    }

    /// The split at the nearest node below `e` where the listed values sum to
    /// something, else the nearest above, into `out`, returning that sum and
    /// the node; all zero where no node sums to anything.
    fn held_split(&self, e: f64, out: &mut [f64]) -> (f64, f64) {
        let rule = self.rule;
        let nodes = &rule.nodes;
        let at = nodes.partition_point(|&x| x <= e);
        let sum_at = |x: f64, out: &mut [f64]| -> f64 {
            let mut sum = 0.0;
            for (k, c) in rule.curves.iter().enumerate() {
                let v = curve_interp(&c.energy, &c.values, x).max(0.0);
                out[k] = v;
                sum += v;
            }
            sum
        };
        let below = (0..at).rev();
        let above = at..nodes.len();
        for i in below.chain(above) {
            let sum = sum_at(nodes[i], out);
            if sum > 0.0 {
                for v in out.iter_mut() {
                    *v /= sum;
                }
                return (sum, nodes[i]);
            }
        }
        out.iter_mut().for_each(|v| *v = 0.0);
        (0.0, e)
    }
}

/// Linear interpolation of `(energy, values)` at `e`: zero below the first grid
/// point (threshold), flat above the last. Shared with the transmutation tally,
/// which evaluates branching curves at the collision energy (issue #218).
pub(crate) fn curve_interp(energy: &[f64], values: &[f64], e: f64) -> f64 {
    if e <= energy[0] {
        // At/just below threshold the value is the first point only if e ==
        // energy[0]; below the grid the curve is zero.
        return if e < energy[0] { 0.0 } else { values[0] };
    }
    let last = energy.len() - 1;
    if e >= energy[last] {
        return values[last];
    }
    // Binary search for the bracketing interval.
    let idx = match energy.binary_search_by(|x| x.partial_cmp(&e).unwrap()) {
        Ok(i) => return values[i],
        Err(i) => i, // energy[i-1] < e < energy[i]
    };
    let (e0, e1) = (energy[idx - 1], energy[idx]);
    let (v0, v1) = (values[idx - 1], values[idx]);
    if e1 == e0 {
        return v0;
    }
    v0 + (v1 - v0) * (e - e0) / (e1 - e0)
}

/// Every list of a branching table for the parents a solve can drive, built
/// once and refused as a whole on the first list that cannot be read.
pub(crate) type Lists<'a> = HashMap<String, Vec<ListRule<'a>>>;

/// Build the lists of `branch` whose parent is in `chain`, sorted by kind so
/// every walk over them is in one order.
pub(crate) fn build_lists<'a>(
    chain: &HashMap<String, yani::ChainNuclide>,
    branch: &'a yani::BranchTable,
) -> Result<Lists<'a>, String> {
    let mut out: Lists<'a> = HashMap::new();
    for (parent, kinds) in branch {
        if !chain.contains_key(parent) {
            continue;
        }
        let mut names: Vec<&String> = kinds.keys().collect();
        names.sort();
        let mut rules = Vec::new();
        for kind in names {
            if let Some(rule) = ListRule::new(parent, kind, &kinds[kind])? {
                rules.push(rule);
            }
        }
        if !rules.is_empty() {
            out.insert(parent.clone(), rules);
        }
    }
    Ok(out)
}

/// What one list produced under one spectrum. The fields are in the units
/// they were folded in; `to_rate` times any of them is a rate on the footing
/// of the reaction rates (per atom), and a share is the ratio of two.
#[derive(Clone, Debug, Default)]
pub(crate) struct ListRates {
    /// Per curve, in [`ListRule::curves`] order.
    pub(crate) production: Vec<f64>,
    /// The factor from these sums to rates.
    pub(crate) to_rate: f64,
    /// The transport total over the same walk; zero with no total.
    pub(crate) total: f64,
    pub(crate) clipped: f64,
    pub(crate) extrapolated: f64,
    /// Whether the parent had a transport cross section for the MT.
    pub(crate) has_total: bool,
}

/// One state a channel produces, as the report gives it.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchingState {
    /// The chain nuclide the state was booked to.
    pub target: String,
    /// The evaluated levels summed into it (LFS), in the order they were summed.
    pub lfs: Vec<i32>,
    /// How each level was matched to `target` (see `yani::BranchState`).
    pub level_route: Vec<String>,
    /// Each level's energy less the booked state's, in eV, where both are
    /// stated.
    pub level_energy_difference: Vec<Option<f64>>,
    /// Its share of the reaction's rate under this run's spectrum.
    pub share: f64,
}

/// One reaction's isomeric branching as the rule applied it.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchingChannel {
    pub parent: String,
    pub reaction: String,
    /// The MT the evaluation listed the states under.
    pub mt: Option<i32>,
    /// 9 for yields, 10 for partial cross sections.
    pub file: i32,
    /// `"share"` (the values split the transport total) or `"absolute"` (each
    /// value is a production, the ground state taking the rest).
    pub representation: String,
    /// Whether the list names the product's ground state.
    pub complete: bool,
    /// Where `complete` came from: always the converter's `list_complete`,
    /// since nothing is inferred at run time.
    pub completeness_source: String,
    /// What the shares are taken over: `"sum of the listed partials"`, `"sum
    /// of the listed yields"` or `"transport total"`.
    pub denominator: String,
    pub states: Vec<BranchingState>,
    /// The reaction's rate over the parent's neutron removal rate.
    pub removal_share: f64,
    /// Production clipped as impossible, over the parent's removal rate.
    pub clipped_share: f64,
    /// Production resting on a fraction held outside the list's tabulated
    /// range, over the parent's removal rate.
    pub extrapolated_share: f64,
    /// Where the listed values most exceed the evaluation's own total
    /// (partials over its MF=3, isomer yields over one), for a list whose
    /// values are not normalised: `(energy [eV], ratio)`.
    pub own_total_excess: Option<(f64, f64)>,
    /// The parent evaluation's MF=1 normalisation block, when it names one.
    pub normalisation: Option<String>,
}

/// A branching channel, or part of one, that carries no production in the
/// solve, and how much of the parent's removal it would have carried.
#[derive(Clone, Debug, PartialEq)]
pub struct DroppedChannel {
    pub parent: String,
    pub reaction: String,
    /// The state, when only one state of the channel is dropped.
    pub target: Option<String>,
    pub reason: String,
    /// Its folded rate over the parent's removal rate, where that can be
    /// folded (it cannot without the parent's transport cross sections).
    pub removal_share: Option<f64>,
}

/// MT=5's share of one parent's neutron removal rate.
#[derive(Clone, Debug, PartialEq)]
pub struct UnmodelledRate {
    pub nuclide: String,
    /// `R_5 / (R_removal + R_5)`.
    pub share: f64,
}

/// What the isomeric-branching rule did over one step's spectrum, and what it
/// could not do.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BranchingReport {
    pub channels: Vec<BranchingChannel>,
    pub dropped: Vec<DroppedChannel>,
    /// Parents with a non-zero MT=5 rate, largest share first.
    pub unmodelled_mt5: Vec<UnmodelledRate>,
}

impl BranchingReport {
    /// Whether the report says nothing at all.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty() && self.dropped.is_empty() && self.unmodelled_mt5.is_empty()
    }
}

/// A parent's neutron removal rate: every reaction rate the chain drives on
/// it, `(n,n')` to its isomers included.
pub(crate) fn removal_rate(rates: &yani::ReactionRates, parent: &str) -> f64 {
    rates
        .get(parent)
        .map(|kinds| kinds.values().filter(|r| r.is_finite()).sum())
        .unwrap_or(0.0)
}

/// MT=5's share of each parent's neutron removal rate, largest first, for the
/// report.
///
/// MT=5 is `(n,anything)`: its products are given as MF=6 residual yields
/// (the LIP isomer flag with them), MF=10 MT=5 partials and light-particle
/// production, none of which the chain reads, so every atom it removes is
/// missing from the inventory. TENDL starts MT=5 above about 20 MeV, where a
/// D-T or fission spectrum has no flux, but some evaluations lump reactions
/// into it lower down: ENDF/B-VIII.1, JEFF-4.0 and FENDL-3.2d give Fe54's
/// (n,np) and most of its (n,alpha) as MT=5, 0.55 b of 1.39 b nonelastic at
/// 14 MeV.
///
/// Measured and reported, not refused. Refusing above
/// [`BRANCHING_RATE_TOLERANCE`] would refuse every steel on those three
/// libraries under a D-T spectrum (Fe54 38 to 60% of its removal, Fe56 12 to
/// 23%, Mn55 8%, Cr52 2.4%) and under a fission one (Fe56 0.8%, Mn55 1.0%),
/// which is where most runs are; whether to refuse there, or to model MT=5
/// first, is an open decision on issue #140.
///
/// `mt5` is each parent's MT=5 rate on the same footing as `rates`.
pub(crate) fn measure_unmodelled_mt5(
    rates: &yani::ReactionRates,
    mt5: &HashMap<String, f64>,
) -> Vec<UnmodelledRate> {
    // TODO(#140): model MT=5 residual production (MF=6 LIP, MF=10 MT=5 and the
    // light-particle gas) as an (n,X) reaction; until then, decide whether a
    // share above the tolerance refuses the run.
    let mut out: Vec<UnmodelledRate> = mt5
        .iter()
        .filter(|(_, r)| **r > 0.0)
        .map(|(nuclide, &r5)| {
            let removal = removal_rate(rates, nuclide);
            UnmodelledRate {
                nuclide: nuclide.clone(),
                share: r5 / (removal + r5),
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.share
            .total_cmp(&a.share)
            .then_with(|| a.nuclide.cmp(&b.nuclide))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use yani::BranchState;

    #[test]
    fn curve_interp_threshold_and_flat() {
        let e = [1.0e6, 2.0e6, 3.0e6];
        let v = [0.0, 4.0, 8.0];
        assert_eq!(curve_interp(&e, &v, 5.0e5), 0.0); // below threshold
        assert_eq!(curve_interp(&e, &v, 1.0e6), 0.0); // at first point
        assert_eq!(curve_interp(&e, &v, 1.5e6), 2.0); // linear midpoint
        assert_eq!(curve_interp(&e, &v, 3.0e6), 8.0); // last point
        assert_eq!(curve_interp(&e, &v, 9.0e6), 8.0); // flat above range
    }

    fn curve(
        target: &str,
        quantity: BranchQuantity,
        e: &[f64],
        v: &[f64],
        complete: bool,
    ) -> BranchCurve {
        BranchCurve {
            target: target.to_string(),
            quantity,
            energy: e.to_vec(),
            values: v.to_vec(),
            states: std::sync::Arc::from(vec![BranchState {
                mt: 102,
                lfs: if target.contains("_m") { 1 } else { 0 },
                lmf: None,
                list_complete: complete,
                level_route: "energy".to_string(),
                level_energy: 0.0,
                level_energy_difference: None,
                mf3_cross_section: None,
            }]),
            normalisation: None,
        }
    }

    fn point(curves: &[BranchCurve], kind: &str, e: f64, t: f64) -> (Vec<f64>, Point) {
        let rule = ListRule::new("X", kind, curves).unwrap().unwrap();
        let bound = Bound {
            rule: &rule,
            total: None,
            tail: &[],
        };
        let mut out = vec![0.0; curves.len()];
        let p = bound.point(e, t, &mut out);
        (out, p)
    }

    /// Isomer yields summing above one cannot all be made: they are scaled
    /// to one at that energy and the excess, times the total, is clipped.
    #[test]
    fn isomer_yields_above_one_are_clipped_and_measured() {
        let curves = [
            curve(
                "X_m1",
                BranchQuantity::Yield,
                &[1.0, 10.0],
                &[0.8, 0.8],
                false,
            ),
            curve(
                "X_m2",
                BranchQuantity::Yield,
                &[1.0, 10.0],
                &[0.4, 0.4],
                false,
            ),
        ];
        let (out, p) = point(&curves, "(n,gamma)", 5.0, 2.0);
        assert!((out[0] - 0.8 / 1.2).abs() < 1e-15 && (out[1] - 0.4 / 1.2).abs() < 1e-15);
        assert!((p.clipped - 0.2 * 2.0).abs() < 1e-15, "{}", p.clipped);
        assert_eq!(p.factor, 2.0);
        assert!(!p.extrapolated);
    }

    /// A yield that starts at zero starts at its threshold: holding that zero
    /// below it is not an extrapolation, where holding a non-zero first value
    /// is.
    #[test]
    fn a_yield_threshold_is_not_an_extrapolation() {
        let curves = [
            curve(
                "X",
                BranchQuantity::Yield,
                &[1.0e-5, 10.0],
                &[1.0, 1.0],
                true,
            ),
            curve(
                "X_m1",
                BranchQuantity::Yield,
                &[2.0, 10.0],
                &[0.0, 0.5],
                true,
            ),
        ];
        let (_, p) = point(&curves, "(n,gamma)", 1.0, 3.0);
        assert!(!p.extrapolated);
        let (_, p) = point(&curves, "(n,gamma)", 1.0e-6, 3.0);
        assert!(p.extrapolated);
    }

    /// A complete list's split is held from the nearest node where its
    /// values sum to something, and that is flagged where there is a total
    /// to share out, not at a threshold the total shares.
    #[test]
    fn a_complete_list_holds_its_split_where_it_gives_none() {
        let curves = [
            curve(
                "X",
                BranchQuantity::CrossSection,
                &[2.0, 4.0],
                &[1.0, 3.0],
                true,
            ),
            curve(
                "X_m1",
                BranchQuantity::CrossSection,
                &[2.0, 4.0],
                &[3.0, 1.0],
                true,
            ),
        ];
        // Below the list, under a live total: the split at 2 eV, 1:3, as
        // partials of the sum it was held from.
        let (out, p) = point(&curves, "(n,2n)", 1.0, 8.0);
        assert!(p.extrapolated);
        assert_eq!((out[0] * p.factor, out[1] * p.factor), (2.0, 6.0));
        // Below the list with no total there: nothing is held, and the factor
        // is the ratio at the node, so a walk integrates up to it cleanly.
        let (out, p) = point(&curves, "(n,2n)", 1.0, 0.0);
        assert!(!p.extrapolated);
        assert_eq!(out, vec![0.0, 0.0]);
        // Above it, the partials are held flat, which holds their ratio.
        let (out, p) = point(&curves, "(n,2n)", 9.0, 8.0);
        assert!(p.extrapolated);
        assert_eq!((out[0] * p.factor, out[1] * p.factor), (6.0, 2.0));
    }

    /// `(n,n')` partials are absolute whatever the list, and the ground self
    /// row makes nothing; a complete `(n,n')` MF=9 list is normalised over
    /// every listed state, the self row included, which then makes nothing.
    #[test]
    fn inelastic_lists_leave_the_ground_self_row_out() {
        let partials = [
            curve(
                "X",
                BranchQuantity::CrossSection,
                &[1.0, 10.0],
                &[5.0, 5.0],
                true,
            ),
            curve(
                "X_m1",
                BranchQuantity::CrossSection,
                &[1.0, 10.0],
                &[0.5, 0.5],
                true,
            ),
        ];
        let rule = ListRule::new("X", "(n,n')", &partials).unwrap().unwrap();
        assert!(rule.is_absolute());
        assert_eq!(rule.produces, vec![false, true]);
        let (out, _) = point(&partials, "(n,n')", 5.0, 1.0);
        assert_eq!(out, vec![0.0, 0.5]);

        let yields = [
            curve("X", BranchQuantity::Yield, &[1.0, 10.0], &[0.6, 0.6], true),
            curve(
                "X_m1",
                BranchQuantity::Yield,
                &[1.0, 10.0],
                &[0.2, 0.2],
                true,
            ),
        ];
        let (out, p) = point(&yields, "(n,n')", 5.0, 3.0);
        assert!((out[1] - 0.25).abs() < 1e-15 && out[0] == 0.0);
        assert_eq!(p.factor, 3.0);
    }

    /// MT=5 is reported as its share of the parent's removal, the MT=5 rate
    /// counted in that removal.
    #[test]
    fn mt5_is_reported_as_a_share_of_the_removal() {
        let rates: yani::ReactionRates = HashMap::from([(
            "Fe54".to_string(),
            HashMap::from([("(n,p)".to_string(), 1.0)]),
        )]);
        let mt5 = HashMap::from([("Fe54".to_string(), 1.5), ("Fe56".to_string(), 0.0)]);
        let report = measure_unmodelled_mt5(&rates, &mt5);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].nuclide, "Fe54");
        assert!((report[0].share - 0.6).abs() < 1e-15);
    }
}
