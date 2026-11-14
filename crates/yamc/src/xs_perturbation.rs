//! Nuclear data with one replica's cross-section draw applied, for a transport
//! run that sees perturbed cross sections everywhere.
//!
//! A replica's draw is a field over each nuclide's covariance cells
//! ([`yani_transmute::covariance_fold::TransportField`]). Applying it means:
//!
//! 1. every non-redundant reaction the field covers is multiplied by its
//!    cell's multiplier, and shifted by its absolute cell's shift, at each
//!    point of its energy grid. One whose covariance an NC block derives
//!    (ENDF/B-VIII.1 Pb208 elastic above 1.5 MeV is `σ_1 - σ_4 - σ_16 -
//!    σ_102`) also moves by `Σ c_t δσ_t` over the derivation's range, each
//!    `δσ_t` what the draw moves the named reaction's cross section by
//!    ([`TransportField::derived`]), as the activation fold derives it;
//! 2. the redundant rows that other code reads are moved by exactly what their
//!    parts moved: MT 16, 18 and 103 to 107 by the change in the sum of their
//!    components where those are present, then MT 1, 3, 4, 27 and 101 by the
//!    change in [`yamc_nuclide::synthesis::synthesize`], the loader's own sums.
//!    A zero draw therefore leaves every row bit-identical to the nominal. The material's
//!    free-flight total and nuclide selection read the stored MT 1, GPU and
//!    tally scores read MT 4 and 27, and transmutation tallies read MT 16 and
//!    103, so leaving any of them nominal would perturb the reaction split
//!    while the flights stayed nominal;
//! 3. each temperature's [`FastXSGrid`] is rebuilt from the new reactions by
//!    the loader's own builder.
//!
//! Particle-production rows (MT 203 to 207) move by what the reactions
//! emitting the particle moved, each weighted by how many it emits. Heating,
//! local heating and damage energy (MT 301, 901 and 444) move by what the
//! moved reactions deposit, `Σ_mt h_mt Δσ_mt`, with `h_mt` each reaction's
//! deposit per reaction ([`yamc_nuclide::deposit`]); the part of a row its
//! partials do not reproduce (fission's share) stays nominal. What is held
//! at nominal: rows with no covariance (named in the coverage report) and the
//! short-range (`lb = 8`) noise, which averages away along a track.
//!
//! In the unresolved range each probability-table band moves by its smooth
//! channel's relative change. A table of factors on the smooth cross sections
//! does that by construction; a table of absolute cross sections is given the
//! nominal smooth cross sections ([`yamc_nuclide::urr::UrrData::nominal_smooth`])
//! so its bands scale the same way.
//!
//! The multiplier is evaluated at each grid point, so between two points that
//! straddle a covariance cell edge the interpolated cross section blends the
//! two cells' multipliers. That touches one grid interval per cell edge, on
//! grids of tens of thousands of points.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::deposit::{deposits, Response};
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::synthesis::{synthesize, SYNTHETIC_MTS};
use yani_transmute::covariance_fold::{Cell, DerivedTerm, Read, TransportField};
use yani_transmute::covariance_sample::Draw;
use yani_transmute::resonance_rates::TransportResonance;

/// The redundant rows rebuilt from their components before the totals are
/// synthesized, because the loader's synthesis reads them as partials.
const COMPONENT_SUMS: [i32; 7] = [16, 18, 103, 104, 105, 106, 107];

/// One reaction's cells, sorted by energy, with each cell's index in the
/// field.
pub(crate) type Cells = Vec<(f64, f64, usize)>;

pub(crate) fn cells_by_mt(cells: &[Cell]) -> BTreeMap<i32, Cells> {
    let mut out: BTreeMap<i32, Cells> = BTreeMap::new();
    for (k, c) in cells.iter().enumerate() {
        out.entry(c.mt).or_default().push((c.lo, c.hi, k));
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    out
}

/// The field index of the cell of `cells` holding `energy`, if one does. A
/// cell is `[lo, hi)`, except that the last one also holds its top edge.
pub(crate) fn cell_at(cells: &Cells, energy: f64) -> Option<usize> {
    let i = cells.partition_point(|c| c.0 <= energy).checked_sub(1)?;
    let (_, hi, k) = cells[i];
    (energy < hi || (i + 1 == cells.len() && energy == hi)).then_some(k)
}

/// A reaction's cross section on the full grid of `n` points, zero below its
/// threshold.
pub(crate) fn on_full_grid(reaction: &Reaction, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    let xs = reaction.cross_section.as_slice();
    let start = reaction.threshold_idx.min(n);
    let len = xs.len().min(n - start);
    out[start..start + len].copy_from_slice(&xs[..len]);
    out
}

/// How a particle-production row (MT 203 to 207) is shared over the
/// non-redundant reactions of `reactions`: per reaction, the weight `w(E)`
/// on the grid of `n` points with which its cross section enters the row, so
/// that the row moves by `Σ w_mt δσ_mt` when the reactions move by `δσ_mt`.
///
/// A reaction whose definition states the particle
/// ([`endf::reaction::light_particles`]) weighs its count. The rest of the
/// stored row comes from emission the MT does not name. Where a non-redundant
/// MT 5 is non-zero that rest is MT 5's, its only candidate. Elsewhere it is
/// the break-up of a residual (the LR flags of MF 3: Li7's inelastic levels
/// emit a triton and an alpha, Be9's `(n,2n)` leaves Be8, two alphas, C12's
/// levels above 7.7 MeV three alphas). The processed data does not carry
/// those flags, but a break-up emits a fixed number of particles per
/// reaction, so the rest is fitted, by non-negative least squares over the
/// grid, as `Σ c_mt σ_mt` over the reactions that leave a residual nucleus
/// (not elastic, capture or fission), with each `c_mt` constant in energy.
/// The fitted shares are then scaled point by point so that the attributed
/// part is exactly the rest of the row. ENDF/B-VIII.1 C12 has an MT 5 that is
/// zero below 20 MeV, and its alphas there are its `(n,n'3a)` levels'.
pub(crate) fn production_weights<'a>(
    x: i32,
    row: &[f64],
    reactions: impl IntoIterator<Item = (i32, &'a Reaction)>,
    n: usize,
) -> Vec<(i32, Vec<f64>)> {
    debug_assert!((203..=207).contains(&x));
    let particle = (x - 203) as usize;
    let partials: Vec<(i32, &Reaction)> = reactions
        .into_iter()
        .filter(|(_, r)| !r.redundant)
        .collect();
    let mut out = Vec::new();
    let mut explained = vec![0.0; n];
    for &(mt, reaction) in &partials {
        let Some(counts) = endf::reaction::light_particles(mt) else {
            continue;
        };
        let c = counts[particle] as f64;
        if c == 0.0 {
            continue;
        }
        for (e, v) in explained.iter_mut().zip(on_full_grid(reaction, n)) {
            *e += c * v;
        }
        out.push((mt, vec![c; n]));
    }
    let rest: Vec<f64> = row
        .iter()
        .zip(&explained)
        .map(|(r, e)| (r - e).max(0.0))
        .collect();
    // Below this the rest is rounding in the stored row.
    let scale = row.iter().fold(0.0_f64, |m, v| m.max(*v));
    if !rest.iter().any(|v| *v > 1e-9 * scale) {
        return out;
    }
    let mut rest = rest;
    if let Some((_, r5)) = partials.iter().find(|(mt, _)| *mt == 5) {
        let xs5 = on_full_grid(r5, n);
        let share: Vec<f64> = rest
            .iter()
            .zip(&xs5)
            .map(|(r, x)| if *x > 0.0 { r / x } else { 0.0 })
            .collect();
        for (r, x) in rest.iter_mut().zip(&xs5) {
            if *x > 0.0 {
                *r = 0.0;
            }
        }
        out.push((5, share));
        if !rest.iter().any(|v| *v > 1e-9 * scale) {
            return out;
        }
    }
    let candidates: Vec<(i32, Vec<f64>)> = partials
        .iter()
        .filter(|(mt, _)| {
            *mt != 2
                && *mt != 102
                && endf::reaction::light_particles(*mt).is_some()
                && !endf::reaction::FISSION_MTS.contains(mt)
        })
        .map(|(mt, r)| (*mt, on_full_grid(r, n)))
        .filter(|(_, xs)| xs.iter().any(|v| *v > 0.0))
        .collect();
    let columns: Vec<&[f64]> = candidates.iter().map(|(_, xs)| xs.as_slice()).collect();
    let c = nonnegative_least_squares(&columns, &rest);
    let mut fitted = vec![0.0; n];
    for ((_, xs), ci) in candidates.iter().zip(&c) {
        for (f, v) in fitted.iter_mut().zip(xs) {
            *f += ci * v;
        }
    }
    let ratio: Vec<f64> = rest
        .iter()
        .zip(&fitted)
        .map(|(r, f)| if *f > 0.0 { r / f } else { 0.0 })
        .collect();
    for ((mt, _), ci) in candidates.iter().zip(&c) {
        if *ci <= 0.0 {
            continue;
        }
        let w: Vec<f64> = ratio.iter().map(|q| ci * q).collect();
        match out.iter_mut().find(|(m, _)| m == mt) {
            Some((_, have)) => {
                for (h, v) in have.iter_mut().zip(&w) {
                    *h += v;
                }
            }
            None => out.push((*mt, w)),
        }
    }
    out
}

/// The `c >= 0` minimising `|Σ c_j a_j - b|`, by Lawson and Hanson's active
/// set method on the normal equations. The columns are scaled to unit norm
/// for the solve, so cross sections of very different size are fitted alike.
fn nonnegative_least_squares(columns: &[&[f64]], b: &[f64]) -> Vec<f64> {
    let k = columns.len();
    let norms: Vec<f64> = columns
        .iter()
        .map(|a| a.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let mut gram = vec![0.0; k * k];
    let mut rhs = vec![0.0; k];
    for i in 0..k {
        if norms[i] == 0.0 {
            continue;
        }
        rhs[i] = columns[i].iter().zip(b).map(|(a, y)| a * y).sum::<f64>() / norms[i];
        for j in 0..=i {
            if norms[j] == 0.0 {
                continue;
            }
            let g = columns[i]
                .iter()
                .zip(columns[j])
                .map(|(a, c)| a * c)
                .sum::<f64>()
                / (norms[i] * norms[j]);
            gram[i * k + j] = g;
            gram[j * k + i] = g;
        }
    }
    let gradient = |x: &[f64]| -> Vec<f64> {
        (0..k)
            .map(|i| rhs[i] - (0..k).map(|j| gram[i * k + j] * x[j]).sum::<f64>())
            .collect()
    };
    // Solve the normal equations restricted to `set`, zero elsewhere.
    let solve = |set: &[usize]| -> Option<Vec<f64>> {
        let m = set.len();
        let mut a: Vec<f64> = set
            .iter()
            .flat_map(|&i| set.iter().map(move |&j| (i, j)))
            .map(|(i, j)| gram[i * k + j])
            .collect();
        let mut y: Vec<f64> = set.iter().map(|&i| rhs[i]).collect();
        for p in 0..m {
            let pivot =
                (p..m).max_by(|&r, &s| a[r * m + p].abs().total_cmp(&a[s * m + p].abs()))?;
            if a[pivot * m + p].abs() < 1e-12 {
                return None;
            }
            for c in 0..m {
                a.swap(p * m + c, pivot * m + c);
            }
            y.swap(p, pivot);
            for r in p + 1..m {
                let f = a[r * m + p] / a[p * m + p];
                for c in p..m {
                    a[r * m + c] -= f * a[p * m + c];
                }
                y[r] -= f * y[p];
            }
        }
        for p in (0..m).rev() {
            let s: f64 = (p + 1..m).map(|c| a[p * m + c] * y[c]).sum();
            y[p] = (y[p] - s) / a[p * m + p];
        }
        let mut z = vec![0.0; k];
        for (&i, v) in set.iter().zip(y) {
            z[i] = v;
        }
        Some(z)
    };
    let tolerance = 1e-10 * rhs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let mut x = vec![0.0; k];
    let mut passive: Vec<usize> = Vec::new();
    let mut collinear: Vec<usize> = Vec::new();
    for _ in 0..3 * k.max(1) {
        let w = gradient(&x);
        let Some(next) = (0..k)
            .filter(|i| {
                !passive.contains(i)
                    && !collinear.contains(i)
                    && norms[*i] > 0.0
                    && w[*i] > tolerance
            })
            .max_by(|&i, &j| w[i].total_cmp(&w[j]))
        else {
            break;
        };
        passive.push(next);
        loop {
            let Some(z) = solve(&passive) else {
                // A column collinear with those already in: leave it out.
                passive.retain(|&i| i != next);
                collinear.push(next);
                break;
            };
            if passive.iter().all(|&i| z[i] > 0.0) {
                x = z;
                break;
            }
            let alpha = passive
                .iter()
                .filter(|&&i| z[i] <= 0.0)
                .map(|&i| x[i] / (x[i] - z[i]))
                .fold(f64::INFINITY, f64::min);
            for (xi, zi) in x.iter_mut().zip(&z) {
                *xi += alpha * (zi - *xi);
            }
            passive.retain(|&i| x[i] > 1e-14);
            for (i, xi) in x.iter_mut().enumerate() {
                if !passive.contains(&i) {
                    *xi = 0.0;
                }
            }
            if passive.is_empty() {
                break;
            }
        }
    }
    x.iter()
        .zip(&norms)
        .map(|(v, s)| if *s > 0.0 { v / s } else { 0.0 })
        .collect()
}

/// One [`DerivedTerm`] of a partial, on the nuclide's energy grid.
pub(crate) struct GridTerm {
    /// The reaction whose cells the term reads.
    pub(crate) mt: i32,
    /// `c σ_t(E_i)`, in barns, at the points inside the term's range from the
    /// partial's threshold, and zero elsewhere: what a relative change of the
    /// named reaction moves the partial by.
    pub(crate) weight: Vec<f64>,
    /// `σ_t(E_i)`, the named reaction's own cross section, which an absolute
    /// shift on its cells is a shift of.
    pub(crate) xs: Vec<f64>,
}

/// Whether grid energy `e` is inside `range`: `[lo, hi)`, closed at the top
/// of the grid `top` so the last point of a derivation that runs to the end
/// of the grid is inside it, as the last covariance cell holds its top edge.
fn in_range(range: (f64, f64), e: f64, top: f64) -> bool {
    range.0 <= e && (e < range.1 || (e == range.1 && e == top))
}

/// `partial`'s derived terms on the grid `energies`, its cross sections read
/// from `reactions`.
///
/// The replica weights and the reruns both read a partial's NC derivation
/// through this, so the two apply it the same way. A term is zero below the
/// partial's threshold, where the partial has no cross section to move.
pub(crate) fn grid_terms(
    terms: &[DerivedTerm],
    partial: &Reaction,
    reactions: &HashMap<i32, Arc<Reaction>>,
    energies: &[f64],
) -> Vec<GridTerm> {
    let n = energies.len();
    let top = energies.last().copied().unwrap_or(f64::NAN);
    terms
        .iter()
        .map(|t| {
            let mut xs = vec![0.0; n];
            for r in t.cross_sections.iter().filter_map(|mt| reactions.get(mt)) {
                for (o, v) in xs.iter_mut().zip(on_full_grid(r, n)) {
                    *o += v;
                }
            }
            let weight = (0..n)
                .map(|i| {
                    let inside = i >= partial.threshold_idx && in_range(t.range, energies[i], top);
                    if inside {
                        t.coefficient * xs[i]
                    } else {
                        0.0
                    }
                })
                .collect();
            GridTerm {
                mt: t.mt,
                weight,
                xs,
            }
        })
        .collect()
}

/// `reaction` with `values` (on its own grid, from its threshold) in place of
/// its cross section.
fn with_values(reaction: &Reaction, values: Vec<f64>) -> Reaction {
    Reaction {
        cross_section: values.into(),
        ..reaction.clone()
    }
}

/// One replica's change to a nuclide's resonance cross sections, at one
/// temperature: [`TransportResonance::changes`] on that temperature's grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ResonanceChange {
    /// The temperature label whose grid `by_mt` is on.
    pub temperature: String,
    /// Per reaction, the change in barns at every point of the grid.
    pub by_mt: BTreeMap<i32, Vec<f64>>,
}

/// Why `nuclide`'s resonance parameters cannot be applied by a transport at
/// `temperature`, or `None` where they can: every reaction the ranges give
/// must be a partial the transport samples, since a change to a reaction
/// held only as a sum, or not at all, has nowhere to go.
fn unplaceable(nuclide: &Nuclide, temperature: &str, reactions: &[i32]) -> Option<String> {
    let held = nuclide.reactions_for_temp(temperature)?;
    let missing: Vec<String> = reactions
        .iter()
        .filter(|mt| !held.get(mt).is_some_and(|r| !r.redundant))
        .map(|mt| mt.to_string())
        .collect();
    (!missing.is_empty()).then(|| {
        format!(
            "its transport data has no partial reaction (MT {}) for a reaction its \
             resonance ranges give to move",
            missing.join(", ")
        )
    })
}

/// Every nuclide of `material` with resonance parameters (MF=2 and MF=32,
/// `resonance_parameters.arrow`), and whether a transport samples them: the
/// prepared parameters, or why the nuclide keeps the first-order MF=32 rows
/// of its `covariance.arrow` instead. The material's nuclear data must be
/// loaded with its resonance parameters
/// (`Material::ensure_resonance_parameters_loaded`).
///
/// The replica weights and the coverage report both read this, so they agree
/// on which nuclides are sampled.
pub fn resonance_sampling(
    material: &Material,
) -> BTreeMap<String, Result<TransportResonance, String>> {
    let mut out = BTreeMap::new();
    for (name, nuclide) in &material.nuclide_data {
        if !material.nuclides.contains_key(name) {
            continue;
        }
        let Some(data) = nuclide.resonance_parameters.as_ref() else {
            continue;
        };
        let temperature = if material.temperature().is_empty() {
            nuclide
                .loaded_temperatures
                .first()
                .cloned()
                .unwrap_or_default()
        } else {
            material.temperature().to_string()
        };
        let sampled = TransportResonance::new(name, data, &temperature).and_then(|r| {
            match unplaceable(nuclide, &temperature, &r.reactions()) {
                Some(reason) => Err(reason),
                None => Ok(r),
            }
        });
        out.insert(name.clone(), sampled);
    }
    out
}

/// The names [`resonance_sampling`] samples.
pub fn sampled_names(
    sampling: &BTreeMap<String, Result<TransportResonance, String>>,
) -> BTreeSet<String> {
    sampling
        .iter()
        .filter(|(_, r)| r.is_ok())
        .map(|(n, _)| n.clone())
        .collect()
}

/// A copy of `nuclide` with one replica's draw applied, and how many
/// perturbed grid values came out negative and were floored at zero.
///
/// `relative` holds `m_k - 1` and `absolute` the shift in barns per cell of
/// `transport`'s field, as [`Draw::relative`] and [`Draw::absolute`] give
/// them. Every loaded temperature gets the same draw: covariance does not
/// depend on temperature. `resonance`, where the replica samples the
/// nuclide's resonance parameters, is added to the partials it names at its
/// own temperature only, as broadening makes it depend on that; `transport`
/// should then be fielded without the first-order MF=32 rows
/// ([`yani_transmute::covariance_fold::transport_fields`]).
pub fn perturbed_nuclide(
    nuclide: &Nuclide,
    transport: &TransportField,
    relative: &[f64],
    absolute: &[f64],
    resonance: Option<&ResonanceChange>,
) -> Result<(Nuclide, usize), String> {
    if transport.field.is_none() && resonance.is_none() {
        return Ok((nuclide.clone(), 0));
    }
    let (relative_cells, absolute_cells) = match &transport.field {
        Some(field) => (
            cells_by_mt(&field.relative_cells),
            cells_by_mt(&field.absolute_cells),
        ),
        None => (BTreeMap::new(), BTreeMap::new()),
    };
    let name = nuclide
        .name
        .clone()
        .unwrap_or_else(|| "nuclide".to_string());

    let mut out = nuclide.clone();
    let mut floored = 0;
    for (t, temperature) in nuclide.loaded_temperatures.iter().enumerate() {
        let Some(grid) = nuclide.energy.as_ref().and_then(|m| m.get(temperature)) else {
            continue;
        };
        let energies = grid.as_slice();
        let n = energies.len();
        let reactions = &nuclide.reactions[t];
        let mut new: HashMap<i32, Arc<Reaction>> = reactions.clone();

        // The sums whose absolute shifts components share, each expanded onto
        // the grid once.
        let sums_on_grid: HashMap<i32, Vec<f64>> = transport
            .reads
            .values()
            .filter_map(|r| match r {
                Read::Parent(sum) if absolute_cells.contains_key(sum) => Some(*sum),
                _ => None,
            })
            .filter_map(|sum| reactions.get(&sum).map(|r| (sum, on_full_grid(r, n))))
            .collect();

        // 1. The perturbed partials.
        for (mt, read) in &transport.reads {
            let source = match read {
                Read::Own => *mt,
                Read::Parent(sum) => *sum,
                Read::Nominal => continue,
            };
            let Some(reaction) = reactions.get(mt) else {
                continue;
            };
            // An absolute shift stated on a sum is shared among its components
            // in proportion to their cross sections, so the components still
            // add up to the shifted sum.
            let share = |i: usize, xs: f64| -> f64 {
                match read {
                    Read::Parent(sum) => sums_on_grid
                        .get(sum)
                        .map(|s| s[i])
                        .filter(|total| *total > 0.0)
                        .map_or(0.0, |total| xs / total),
                    _ => 1.0,
                }
            };
            // What an NC derivation adds: each named reaction's change at the
            // point, times its coefficient, an absolute shift as a share of
            // the named reaction's own cross section, as the replicas read it.
            let derived = match (read, transport.derived.get(mt)) {
                (Read::Own, Some(terms)) => grid_terms(terms, reaction, reactions, energies),
                _ => Vec::new(),
            };
            let derived_change = |i: usize, e: f64| -> f64 {
                derived
                    .iter()
                    .filter(|t| t.weight.get(i).is_some_and(|w| *w != 0.0))
                    .map(|t| {
                        let dm = relative_cells
                            .get(&t.mt)
                            .and_then(|c| cell_at(c, e))
                            .map_or(0.0, |k| relative[k]);
                        let a = absolute_cells
                            .get(&t.mt)
                            .and_then(|c| cell_at(c, e))
                            .filter(|_| t.xs[i] > 0.0)
                            .map_or(0.0, |k| absolute[k] / t.xs[i]);
                        t.weight[i] * (dm + a)
                    })
                    .sum()
            };
            let start = reaction.threshold_idx;
            let values: Vec<f64> = reaction
                .cross_section
                .as_slice()
                .iter()
                .enumerate()
                .map(|(j, &xs)| {
                    let i = start + j;
                    let e = energies.get(i).copied().unwrap_or(f64::NAN);
                    let m = relative_cells
                        .get(&source)
                        .and_then(|c| cell_at(c, e))
                        .map_or(1.0, |k| 1.0 + relative[k]);
                    let a = absolute_cells
                        .get(&source)
                        .and_then(|c| cell_at(c, e))
                        .map_or(0.0, |k| absolute[k] * share(i, xs));
                    let v = xs * m + a + derived_change(i, e);
                    if v < 0.0 {
                        floored += 1;
                        0.0
                    } else {
                        v
                    }
                })
                .collect();
            new.insert(*mt, Arc::new(with_values(reaction, values)));
        }

        // The resonance parameters' change, onto the partials as perturbed
        // above: the two sources are independent and add.
        if let Some(change) = resonance.filter(|c| c.temperature == *temperature) {
            for (mt, delta) in &change.by_mt {
                let Some(reaction) = new.get(mt).cloned() else {
                    continue;
                };
                if delta.len() != n {
                    return Err(format!(
                        "{name}: a resonance change of {} points for a grid of {n}",
                        delta.len()
                    ));
                }
                let start = reaction.threshold_idx;
                let values: Vec<f64> = reaction
                    .cross_section
                    .as_slice()
                    .iter()
                    .zip(&delta[start.min(n)..])
                    .map(|(&xs, d)| {
                        let v = xs + d;
                        if v < 0.0 {
                            floored += 1;
                            0.0
                        } else {
                            v
                        }
                    })
                    .collect();
                new.insert(*mt, Arc::new(with_values(&reaction, values)));
            }
        }

        // The heating and damage-energy rows, moved by what the moved
        // partials deposit. Not floored: a KERMA may be negative.
        let moved: BTreeSet<i32> = new
            .iter()
            .filter(|(mt, r)| !r.redundant && reactions.get(mt).is_some_and(|b| !Arc::ptr_eq(b, r)))
            .map(|(mt, _)| *mt)
            .collect();
        if !moved.is_empty() {
            for response in Response::ALL {
                let Some(stored) = reactions.get(&response.mt()) else {
                    continue;
                };
                let mut change = vec![0.0; n];
                for (mt, h) in deposits(nuclide, t, response, &moved) {
                    let (b, a) = (on_full_grid(&reactions[&mt], n), on_full_grid(&new[&mt], n));
                    for i in 0..n {
                        change[i] += h[i] * (a[i] - b[i]);
                    }
                }
                let start = stored.threshold_idx.min(n);
                let values = stored
                    .cross_section
                    .as_slice()
                    .iter()
                    .zip(&change[start..])
                    .map(|(v, d)| v + d)
                    .collect();
                new.insert(response.mt(), Arc::new(with_values(stored, values)));
            }
        }

        // 2. The redundant rows other code reads, moved by exactly what their
        // parts moved. Each becomes its stored values plus the change in the
        // sum of its parts, not a fresh sum: an evaluation's stored MT 1 is
        // not exactly the sum of its partials, and replacing it would shift
        // every replica from the nominal even under a zero draw.
        let delta =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        let shifted = |stored: &Reaction, change: &[f64]| -> Reaction {
            let start = stored.threshold_idx.min(n);
            let values = stored
                .cross_section
                .as_slice()
                .iter()
                .zip(&change[start..])
                .map(|(v, d)| (v + d).max(0.0))
                .collect();
            with_values(stored, values)
        };
        for &sum in &COMPONENT_SUMS {
            let Some(stored) = new.get(&sum).cloned() else {
                continue;
            };
            let Some(rule) = endf::data::sum_rule(sum) else {
                continue;
            };
            if !stored.redundant || !rule.iter().any(|c| new.contains_key(c)) {
                continue;
            }
            let mut change = vec![0.0; n];
            for c in rule {
                if let (Some(after), Some(before)) = (new.get(c), reactions.get(c)) {
                    for (o, d) in change
                        .iter_mut()
                        .zip(delta(&on_full_grid(after, n), &on_full_grid(before, n)))
                    {
                        *o += d;
                    }
                }
            }
            new.insert(sum, Arc::new(shifted(&stored, &change)));
        }
        // Particle-production rows (MT 203 to 207) move by what the reactions
        // emitting the particle moved, each weighted by its share of the row
        // (see `production_weights`, which `transport::replica` reads too).
        for x in 203..=207 {
            let Some(stored) = new.get(&x).cloned() else {
                continue;
            };
            let row = on_full_grid(&stored, n);
            let mut change = vec![0.0; n];
            let weights = production_weights(
                x,
                &row,
                reactions.iter().map(|(mt, r)| (*mt, r.as_ref())),
                n,
            );
            for (mt, w) in weights {
                let (b, a) = (on_full_grid(&reactions[&mt], n), on_full_grid(&new[&mt], n));
                for i in 0..n {
                    change[i] += w[i] * (a[i] - b[i]);
                }
            }
            new.insert(x, Arc::new(shifted(&stored, &change)));
        }

        let partials_of = |map: &HashMap<i32, Arc<Reaction>>| -> BTreeMap<i32, Vec<f64>> {
            map.iter()
                .filter(|(mt, _)| !SYNTHETIC_MTS.contains(mt))
                .map(|(mt, r)| (*mt, on_full_grid(r, n)))
                .collect()
        };
        let before = synthesize(&partials_of(reactions), n);
        let after = synthesize(&partials_of(&new), n);
        for (mt, values) in &after {
            if let (Some(stored), Some(nominal)) = (new.get(mt).cloned(), before.get(mt)) {
                new.insert(*mt, Arc::new(shifted(&stored, &delta(values, nominal))));
            }
        }

        // 3. The lookup, by the loader's builder.
        if let Some(fast) = nuclide.fast_xs.get(t)
            && !fast.energy.is_empty()
        {
            out.fast_xs[t] = FastXSGrid::build(
                grid,
                &new,
                nuclide.fission_photon_release.as_ref(),
                &format!("{name} at {temperature} K (perturbed)"),
            )?;
        }
        out.reactions[t] = new;
        if let (Some(Some(urr)), Some(nominal)) = (out.urr_data.get_mut(t), nuclide.fast_xs.get(t))
            && !urr.multiply_smooth
        {
            urr.nominal_smooth = Some(Arc::new(nominal.clone()));
        }
    }
    Ok((out, floored))
}

/// A copy of `held` with each reaction it holds at temperature `label` moved
/// by exactly what a draw moved the same reaction of `full`, which `perturbed`
/// is a copy of with the draw applied, and how many values came out negative
/// and were floored at zero.
///
/// `held` is the same evaluation at the same temperature as `full`, loaded at
/// a narrower scope: a transmutation product at activation scope, which holds
/// the network's reactions and none of the transport sections. Its field over
/// that scope is not `full`'s, so a draw of its own would not be the draw
/// `full` takes, and a product that later enters transport, and is read in
/// full, would change its cross sections mid-run. Moving it by `full`'s
/// change instead keeps one draw per nuclide whatever scope holds it: a
/// reaction `held` holds without the partials it sums (MT 103 without 600 to
/// 649) moves by what those partials moved.
///
/// Only `label` moves; transport and the transmutation tallies read a
/// nuclide at its material's temperature alone. An unmoved draw returns
/// `held` unchanged, bit for bit.
pub fn moved_like(
    held: &Nuclide,
    full: &Nuclide,
    perturbed: &Nuclide,
    label: &str,
) -> Result<(Nuclide, usize), String> {
    let name = held.name.clone().unwrap_or_else(|| "nuclide".to_string());
    let (Some(t), Some(f), Some(p)) = (
        held.get_temp_idx(label),
        full.get_temp_idx(label),
        perturbed.get_temp_idx(label),
    ) else {
        return Err(format!("{name}: no data at {label} K to move"));
    };
    let grid = |n: &Nuclide| n.energy.as_ref().and_then(|m| m.get(label)).cloned();
    let (Some(energies), Some(full_grid)) = (grid(held), grid(full)) else {
        return Err(format!("{name}: no energy grid at {label} K"));
    };
    if energies.as_slice() != full_grid.as_slice() {
        return Err(format!(
            "{name}: the grid it holds at {label} K is not the one its full data is on, so \
             the draw cannot be carried over point by point"
        ));
    }
    let n = energies.len();
    let (before, after) = (&full.reactions[f], &perturbed.reactions[p]);
    let mut new = held.reactions[t].clone();
    let mut floored = 0;
    for (mt, reaction) in &held.reactions[t] {
        let (Some(b), Some(a)) = (before.get(mt), after.get(mt)) else {
            return Err(format!(
                "{name}: MT {mt} is held without a counterpart in its full data"
            ));
        };
        if Arc::ptr_eq(a, b) {
            continue;
        }
        let (b, a) = (on_full_grid(b, n), on_full_grid(a, n));
        let start = reaction.threshold_idx;
        let values = reaction
            .cross_section
            .as_slice()
            .iter()
            .enumerate()
            .map(|(j, &xs)| {
                let v = xs + (a[start + j] - b[start + j]);
                // A KERMA or damage energy may be negative; a cross section
                // may not.
                if *mt < 300 && v < 0.0 {
                    floored += 1;
                    0.0
                } else {
                    v
                }
            })
            .collect();
        new.insert(*mt, Arc::new(with_values(reaction, values)));
    }
    let moved = new
        .iter()
        .any(|(mt, r)| !Arc::ptr_eq(r, &held.reactions[t][mt]));
    if !moved {
        return Ok((held.clone(), 0));
    }
    let mut out = held.clone();
    if let Some(fast) = held.fast_xs.get(t)
        && !fast.energy.is_empty()
    {
        out.fast_xs[t] = FastXSGrid::build(
            &energies,
            &new,
            held.fission_photon_release.as_ref(),
            &format!("{name} at {label} K (perturbed)"),
        )?;
    }
    out.reactions[t] = new;
    if let (Some(Some(urr)), Some(nominal)) = (out.urr_data.get_mut(t), held.fast_xs.get(t))
        && !urr.multiply_smooth
    {
        urr.nominal_smooth = Some(Arc::new(nominal.clone()));
    }
    Ok((out, floored))
}

/// A copy of `material` with every nuclide that has a field in `fields`
/// perturbed by `draw`, and every nuclide in `resonance` by its change, its
/// cross-section tables cleared so the next run rebuilds them from the
/// perturbed data, and how many grid values were floored at zero.
///
/// The material's nuclear data must already be loaded at the run's
/// temperature and with covariance: a lazy load after this would read the
/// nominal data back from disk over the perturbed copy.
pub fn perturbed_material(
    material: &Material,
    fields: &BTreeMap<String, TransportField>,
    draw: &Draw,
    resonance: &BTreeMap<String, ResonanceChange>,
) -> Result<(Material, usize), String> {
    let mut out = material.clone();
    let mut floored = 0;
    let unfielded = TransportField {
        reads: BTreeMap::new(),
        derived: BTreeMap::new(),
        field: None,
    };
    let names: BTreeSet<&String> = fields.keys().chain(resonance.keys()).collect();
    for name in names {
        let Some(nuclide) = material.nuclide_data.get(name) else {
            continue;
        };
        // A nuclide the draw has no field for moves by its resonance change
        // alone, if it has one.
        let drawn = draw.relative(name).zip(draw.absolute(name));
        let change = resonance.get(name);
        let (transport, relative, absolute) = match (fields.get(name), drawn) {
            (Some(t), Some((r, a))) => (t, r, a),
            _ if change.is_some() => (&unfielded, &[][..], &[][..]),
            _ => continue,
        };
        let (perturbed, n) = perturbed_nuclide(nuclide, transport, relative, absolute, change)?;
        floored += n;
        out.nuclide_data.insert(name.clone(), Arc::new(perturbed));
    }
    out.invalidate_xs_cache();
    Ok((out, floored))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonnegative_least_squares_recovers_constant_multiplicities() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        let b: Vec<f64> = (0..50).map(|i| (i as f64 * 0.07).exp() * 0.1).collect();
        let c: Vec<f64> = (0..50).map(|i| if i > 20 { 1.0 } else { 0.0 }).collect();
        // Two of the first and three of the third; the second plays no part.
        let y: Vec<f64> = (0..50).map(|i| 2.0 * a[i] + 3.0 * c[i]).collect();
        let x = nonnegative_least_squares(&[&a, &b, &c], &y);
        assert!((x[0] - 2.0).abs() < 1e-9, "{x:?}");
        assert!(x[1].abs() < 1e-9, "{x:?}");
        assert!((x[2] - 3.0).abs() < 1e-9, "{x:?}");
    }

    #[test]
    fn nonnegative_least_squares_holds_negative_fits_at_zero() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0, 1.0];
        // The unconstrained fit is 1 a - 1 b.
        let y = [0.0, 1.0, 2.0];
        let x = nonnegative_least_squares(&[&a, &b], &y);
        assert!(x.iter().all(|v| *v >= 0.0), "{x:?}");
        assert_eq!(x[1], 0.0, "{x:?}");
    }

    /// The share each reaction of a cached fixture takes of production row
    /// `x`, and the row's value against the shares' sum, at 14.1 MeV.
    type Shares = (Vec<(i32, f64)>, f64, f64);

    fn shares(nuclide: &str, x: i32) -> Option<Shares> {
        let path = yamc_test_cache::transport_nuclide(nuclide)?;
        let n = yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow(
            std::path::Path::new(&path),
            &yamc_nuclide::LoadScope::full(),
        )
        .ok()?;
        let t = n.get_temp_idx("294")?;
        let grid = n.energy.as_ref()?.get("294")?.as_slice().to_vec();
        let len = grid.len();
        let i = grid.partition_point(|e| *e < 14.1e6);
        let reactions = &n.reactions[t];
        let row = on_full_grid(reactions.get(&x)?, len);
        let weights = production_weights(
            x,
            &row,
            reactions.iter().map(|(m, r)| (*m, r.as_ref())),
            len,
        );
        let mut total = 0.0;
        let mut out = Vec::new();
        for (mt, w) in weights {
            let share = w[i] * on_full_grid(&reactions[&mt], len)[i];
            total += share;
            out.push((mt, share / row[i]));
        }
        Some((out, row[i], total))
    }

    /// Li7's tritons, Be9's and C12's alphas come from the break-up of the
    /// residual, which no MT names. Each row is attributed in full to the
    /// reactions that make it, so a data draw moves it.
    #[test]
    fn break_up_production_is_attributed_to_the_reactions_that_make_it() {
        for (nuclide, x) in [("Li7", 205), ("Li7", 207), ("Be9", 207), ("C12", 207)] {
            let Some((shares, row, total)) = shares(nuclide, x) else {
                eprintln!("skipping {nuclide}: fixture not cached");
                continue;
            };
            eprintln!("{nuclide} MT {x}: row {row:e} b, {shares:?}");
            assert!(row > 0.0);
            assert!(
                (total / row - 1.0).abs() < 1e-9,
                "{nuclide} MT {x}: {total} of {row}"
            );
            if nuclide == "Be9" {
                // Be8, left by (n,2n), is two alphas.
                let n2n = shares.iter().find(|(mt, _)| *mt == 16).map_or(0.0, |s| s.1);
                assert!(n2n > 0.95, "Be9 (n,2n) carries {n2n} of the alphas");
            }
        }
    }
}
