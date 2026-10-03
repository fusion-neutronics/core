//! Nuclear data with one replica's cross-section draw applied, for a transport
//! run that sees perturbed cross sections everywhere.
//!
//! A replica's draw is a field over each nuclide's covariance cells
//! ([`yani_transmute::covariance_fold::TransportField`]). Applying it means:
//!
//! 1. every non-redundant reaction the field covers is multiplied by its
//!    cell's multiplier, and shifted by its absolute cell's shift, at each
//!    point of its energy grid;
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
//! emitting the particle moved, each weighted by how many it emits. What is
//! held at nominal: rows with no covariance (named in the coverage report),
//! heating, KERMA and damage rows (absolute values with no per-reaction
//! split), and the short-range (`lb = 8`) noise, which averages away along a
//! track.
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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::synthesis::{synthesize, SYNTHETIC_MTS};
use yani_transmute::covariance_fold::{Cell, Read, TransportField};
use yani_transmute::covariance_sample::Draw;

/// The redundant rows rebuilt from their components before the totals are
/// synthesized, because the loader's synthesis reads them as partials.
const COMPONENT_SUMS: [i32; 7] = [16, 18, 103, 104, 105, 106, 107];

/// One reaction's cells, sorted by energy, with each cell's index in the
/// field.
type Cells = Vec<(f64, f64, usize)>;

fn cells_by_mt(cells: &[Cell]) -> BTreeMap<i32, Cells> {
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
fn cell_at(cells: &Cells, energy: f64) -> Option<usize> {
    let i = cells.partition_point(|c| c.0 <= energy).checked_sub(1)?;
    let (_, hi, k) = cells[i];
    (energy < hi || (i + 1 == cells.len() && energy == hi)).then_some(k)
}

/// A reaction's cross section on the full grid of `n` points, zero below its
/// threshold.
fn on_full_grid(reaction: &Reaction, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    let xs = reaction.cross_section.as_slice();
    let start = reaction.threshold_idx.min(n);
    let len = xs.len().min(n - start);
    out[start..start + len].copy_from_slice(&xs[..len]);
    out
}

/// `reaction` with `values` (on its own grid, from its threshold) in place of
/// its cross section.
fn with_values(reaction: &Reaction, values: Vec<f64>) -> Reaction {
    Reaction {
        cross_section: values.into(),
        ..reaction.clone()
    }
}

/// A copy of `nuclide` with one replica's draw applied, and how many
/// perturbed grid values came out negative and were floored at zero.
///
/// `relative` holds `m_k - 1` and `absolute` the shift in barns per cell of
/// `transport`'s field, as [`Draw::relative`] and [`Draw::absolute`] give
/// them. Every loaded temperature gets the same draw: covariance does not
/// depend on temperature.
pub fn perturbed_nuclide(
    nuclide: &Nuclide,
    transport: &TransportField,
    relative: &[f64],
    absolute: &[f64],
) -> Result<(Nuclide, usize), String> {
    let Some(field) = &transport.field else {
        return Ok((nuclide.clone(), 0));
    };
    let relative_cells = cells_by_mt(&field.relative_cells);
    let absolute_cells = cells_by_mt(&field.absolute_cells);
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
                    let v = xs * m + a;
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
        // emitting the particle moved, each weighted by how many it emits, and
        // the part of the row no stated reaction explains moves with MT 5, its
        // only candidate (see `transport::replica`, which reads the same rule).
        for x in 203..=207 {
            let Some(stored) = new.get(&x).cloned() else {
                continue;
            };
            let particle = (x - 203) as usize;
            let mut change = vec![0.0; n];
            let mut explained = vec![0.0; n];
            for (mt, before) in reactions {
                if before.redundant {
                    continue;
                }
                let Some(counts) = endf::reaction::light_particles(*mt) else {
                    continue;
                };
                let c = counts[particle] as f64;
                if c == 0.0 {
                    continue;
                }
                let (b, a) = (on_full_grid(before, n), on_full_grid(&new[mt], n));
                for i in 0..n {
                    explained[i] += c * b[i];
                    change[i] += c * (a[i] - b[i]);
                }
            }
            if let (Some(b5), Some(a5)) = (reactions.get(&5), new.get(&5)) {
                if !b5.redundant {
                    let (b, a) = (on_full_grid(b5, n), on_full_grid(a5, n));
                    let row = on_full_grid(&stored, n);
                    for i in 0..n {
                        if b[i] > 0.0 {
                            let residual = (row[i] - explained[i]).max(0.0);
                            change[i] += residual * (a[i] / b[i] - 1.0);
                        }
                    }
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
        if let Some(fast) = nuclide.fast_xs.get(t) {
            if !fast.energy.is_empty() {
                out.fast_xs[t] = FastXSGrid::build(
                    grid,
                    &new,
                    nuclide.fission_photon_release.as_ref(),
                    &format!("{name} at {temperature} K (perturbed)"),
                )?;
            }
        }
        out.reactions[t] = new;
        if let (Some(Some(urr)), Some(nominal)) = (out.urr_data.get_mut(t), nuclide.fast_xs.get(t))
        {
            if !urr.multiply_smooth {
                urr.nominal_smooth = Some(Arc::new(nominal.clone()));
            }
        }
    }
    Ok((out, floored))
}

/// A copy of `material` with every nuclide that has a field in `fields`
/// perturbed by `draw`, its cross-section tables cleared so the next run
/// rebuilds them from the perturbed data, and how many grid values were
/// floored at zero.
///
/// The material's nuclear data must already be loaded at the run's
/// temperature and with covariance: a lazy load after this would read the
/// nominal data back from disk over the perturbed copy.
pub fn perturbed_material(
    material: &Material,
    fields: &BTreeMap<String, TransportField>,
    draw: &Draw,
) -> Result<(Material, usize), String> {
    let mut out = material.clone();
    let mut floored = 0;
    for (name, transport) in fields {
        let (Some(nuclide), Some(relative), Some(absolute)) = (
            material.nuclide_data.get(name),
            draw.relative(name),
            draw.absolute(name),
        ) else {
            continue;
        };
        let (perturbed, n) = perturbed_nuclide(nuclide, transport, relative, absolute)?;
        floored += n;
        out.nuclide_data.insert(name.clone(), Arc::new(perturbed));
    }
    out.invalidate_xs_cache();
    Ok((out, floored))
}
