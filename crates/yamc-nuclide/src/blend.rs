//! Building the cross sections for a temperature the library does not carry.
//!
//! A material at 450 K against a library holding 294 K and 600 K used to be a
//! hard failure even though the data brackets the request. It is now served by
//! blending the two neighbours ONCE, at load time, into a temperature that is
//! real from then on: it gets an entry in `loaded_temperatures`, in `reactions`,
//! in `fast_xs`, in `urr_data` and in the energy map, and every consumer
//! downstream sees a nuclide with three temperatures rather than two.
//!
//! That is the whole design. `Nuclide::get_temp_idx` still returns one index,
//! `FastXSGrid::lookup` still reads one grid, the GPU extractors still do an
//! exact label match, and none of them knows interpolation happened. The cost
//! is paid once per nuclide per synthesised temperature, and the hot path is
//! untouched.
//!
//! # Why not sample between the two, the way OpenMC does
//!
//! OpenMC treats the interpolation factor as a Bernoulli probability and picks
//! one of the two bracketing data sets wholesale for the lookup
//! (`src/nuclide.cpp`, `if (f > prn(...)) ++i_temp;`). That is right for
//! OpenMC, which shares one nuclide across every cell and supports runtime
//! temperature changes. It is wrong here, and not merely different: sampling
//! sigma_t and then sampling a flight from it is not the same as sampling a
//! flight from the mean sigma_t. For a two-point pick with half-spread `delta`
//! in the total, uncollided transmission through optical depth `tau` comes out
//! inflated by `cosh(delta * tau)`. At 20 mfp and 5 percent that is a 54
//! percent overestimate of transmitted flux, and it is a BIAS, so more
//! particles converge on the wrong answer. Deep-penetration flux lives at cross
//! section minima, which is exactly where Doppler broadening moves things, so
//! the spread at the energies carrying the signal is not the small number a
//! spectrum average suggests. Fixed-source shielding is the entire use case
//! here, so a deterministic blend is the only defensible choice.
//!
//! A consequence worth stating where a user might read it: this will not
//! reproduce an OpenMC run bit for bit at an interpolated temperature, and will
//! differ from it statistically at the same T. That is a chosen difference.
//!
//! # What is NOT blended
//!
//! URR probability tables. They are conditional distributions of cross-section
//! ratios, and band `k` at 294 K is not the same probability band as band `k`
//! at 600 K, so an elementwise average of the two is not a distribution of
//! anything. The synthesised temperature borrows the nearer neighbour's table
//! whole. Leaving it `None` would be silent: the material path reads a missing
//! table as "no unresolved range" and falls back to the unshielded smooth
//! total, so a 450 K U238 would quietly lose all its self-shielding between 20
//! and 150 keV, differing from BOTH neighbours by far more than any
//! interpolation error, with every existing test still green.
//!
//! Secondary distributions need no blending: they hang off `Reaction.products`
//! and carry no temperature.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::buffer::F64Buffer;
use crate::nuclide::{FastXSGrid, Nuclide};
use crate::reaction::Reaction;
use crate::temperature::{self, TemperatureSource};

/// Bins in the logarithmic lookup index.
///
/// The same 8000 the converter writes (`yamc-convert/src/fast_xs.rs`), fixed
/// rather than scaled to the grid, so a synthesised temperature's accelerator
/// has the same shape as a converted one.
const LOG_BINS: usize = 8000;

/// Relative spacing below which two grid energies are the same point.
///
/// Applied against the last point KEPT, not the last point seen, so a run of
/// nearly-equal values collapses to one rather than walking. 1e-9 relative is
/// far below any genuine spacing in a published library (the closest real
/// neighbours are around 1e-7 apart relatively) and far above the rounding
/// difference between two temperatures' copies of the same nominal energy.
const GRID_EPSILON: f64 = 1e-9;

/// The logarithmic lookup index for a grid, in the form the reader expects.
///
/// Deliberately the same recipe as the converter's, including the last-entry
/// override: `exp(ln(e_max))` need not return `e_max`, and the reader uses this
/// entry as the upper bracket of a search, so one short can exclude the topmost
/// energy while the last index never can.
pub fn build_log_grid_index(energy: &[f64]) -> (f64, f64, Vec<u32>) {
    let log_e_min = energy[0].ln();
    let log_e_max = energy[energy.len() - 1].ln();
    let delta = (log_e_max - log_e_min) / LOG_BINS as f64;
    let inv_log_delta = 1.0 / delta;

    let mut index = Vec::with_capacity(LOG_BINS + 1);
    let mut at = 0usize;
    for bin in 0..=LOG_BINS {
        let e = (log_e_min + bin as f64 * delta).exp();
        while at + 1 < energy.len() && energy[at + 1] <= e {
            at += 1;
        }
        index.push(at as u32);
    }
    if let Some(last) = index.last_mut() {
        *last = (energy.len() - 1) as u32;
    }
    (log_e_min, inv_log_delta, index)
}

/// The sorted union of two sorted grids, deduplicated.
///
/// A union rather than a zip because the two temperatures' grids are different
/// lengths: NJOY thins each one to its own tolerance, so neither is a subset of
/// the other and the blend has to be defined at every point either carries.
/// The endpoints are taken from the merged extremes exactly, so `log_e_min` and
/// the top of the index are derived from real grid values.
pub fn union_grid(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    let push = |out: &mut Vec<f64>, x: f64| match out.last() {
        Some(&kept) if (x - kept).abs() <= GRID_EPSILON * kept.abs() => {}
        _ => out.push(x),
    };
    while i < a.len() || j < b.len() {
        let take_a = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) => x <= y,
            (Some(_), None) => true,
            _ => false,
        };
        if take_a {
            push(&mut out, a[i]);
            i += 1;
        } else {
            push(&mut out, b[j]);
            j += 1;
        }
    }
    out
}

/// A cursor over one sorted grid, for evaluating it at the union's points.
///
/// The union is walked in order and so is each source, so a moving cursor makes
/// the whole blend linear in the grid size where a binary search per point
/// would be `n log n`. The interpolation itself is the same lin-lin form
/// [`crate::interpolation::interpolate_linear`] uses, including its refusal to
/// extrapolate: below the first point or above the last, the endpoint value is
/// returned.
struct GridWalk<'a> {
    x: &'a [f64],
    at: usize,
}

impl<'a> GridWalk<'a> {
    fn new(x: &'a [f64]) -> Self {
        GridWalk { x, at: 0 }
    }

    /// Advance to the interval containing `e` and return `(idx, frac)` such
    /// that a column `y` evaluates to `y[idx] + frac * (y[idx + 1] - y[idx])`,
    /// or `(idx, 0.0)` when `idx` is the last point.
    fn locate(&mut self, e: f64) -> (usize, f64) {
        if self.x.len() < 2 {
            return (0, 0.0);
        }
        if e <= self.x[0] {
            self.at = 0;
            return (0, 0.0);
        }
        let last = self.x.len() - 1;
        if e >= self.x[last] {
            self.at = last;
            return (last, 0.0);
        }
        while self.at + 1 < last && self.x[self.at + 1] <= e {
            self.at += 1;
        }
        while self.at > 0 && self.x[self.at] > e {
            self.at -= 1;
        }
        let (x1, x2) = (self.x[self.at], self.x[self.at + 1]);
        let frac = if x2 > x1 { (e - x1) / (x2 - x1) } else { 0.0 };
        (self.at, frac)
    }
}

/// One source column evaluated at a located point.
#[inline]
fn at(y: &[f64], (idx, frac): (usize, f64)) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let i = idx.min(y.len() - 1);
    if frac == 0.0 || i + 1 >= y.len() {
        return y[i];
    }
    y[i] + frac * (y[i + 1] - y[i])
}

/// `(1 - w) * lo + w * hi`, with both sides evaluated at the same energy.
#[inline]
fn mix(lo: f64, hi: f64, w: f64) -> f64 {
    lo + w * (hi - lo)
}

/// Blend one row-major `[n_energies, n_mts]` matrix onto the union grid.
fn blend_matrix(
    lo: &[f64],
    hi: &[f64],
    n_mts: usize,
    w: f64,
    union: &[f64],
    lo_grid: &[f64],
    hi_grid: &[f64],
) -> F64Buffer {
    if n_mts == 0 {
        return F64Buffer::empty();
    }
    let mut out = Vec::with_capacity(union.len() * n_mts);
    let mut lo_walk = GridWalk::new(lo_grid);
    let mut hi_walk = GridWalk::new(hi_grid);
    for &e in union {
        let li = lo_walk.locate(e);
        let hi_i = hi_walk.locate(e);
        for j in 0..n_mts {
            let lo_v = column_at(lo, n_mts, j, li);
            let hi_v = column_at(hi, n_mts, j, hi_i);
            out.push(mix(lo_v, hi_v, w));
        }
    }
    F64Buffer::from_slice(&out)
}

/// Column `j` of a row-major matrix, interpolated at a located point.
#[inline]
fn column_at(m: &[f64], n_mts: usize, j: usize, (idx, frac): (usize, f64)) -> f64 {
    if m.is_empty() || n_mts == 0 {
        return 0.0;
    }
    let rows = m.len() / n_mts;
    if rows == 0 {
        return 0.0;
    }
    let i = idx.min(rows - 1);
    let v1 = m[i * n_mts + j];
    if frac == 0.0 || i + 1 >= rows {
        return v1;
    }
    let v2 = m[(i + 1) * n_mts + j];
    v1 + frac * (v2 - v1)
}

/// Blend one flat per-energy column onto the union grid.
fn blend_column(
    lo: &[f64],
    hi: &[f64],
    w: f64,
    union: &[f64],
    lo_grid: &[f64],
    hi_grid: &[f64],
) -> F64Buffer {
    if lo.is_empty() && hi.is_empty() {
        return F64Buffer::empty();
    }
    let mut lo_walk = GridWalk::new(lo_grid);
    let mut hi_walk = GridWalk::new(hi_grid);
    let values: Vec<f64> = union
        .iter()
        .map(|&e| {
            let l = at(lo, lo_walk.locate(e));
            let h = at(hi, hi_walk.locate(e));
            mix(l, h, w)
        })
        .collect();
    F64Buffer::from_slice(&values)
}

/// Why a blend could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendError {
    /// One of the two sources has no energy grid, so there is nothing to blend
    /// against.
    EmptyGrid { which: &'static str },
    /// The two temperatures do not carry the same reaction channels in the same
    /// order.
    ///
    /// Refused rather than zero-filled. Zero-filling the absent side would
    /// scale that channel by the blend weight at EVERY energy, not just near a
    /// threshold, which is a wrong cross section that loads and samples without
    /// complaint. No published library has ever been observed to vary its MT
    /// set across temperatures, so this is a guard against a future surprise
    /// rather than a case that must work.
    ChannelsDiffer {
        group: &'static str,
        lower: Vec<i32>,
        upper: Vec<i32>,
    },
}

impl std::fmt::Display for BlendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlendError::EmptyGrid { which } => write!(
                f,
                "the {which} bracketing temperature has no energy grid, so an \
                 intermediate temperature cannot be built from it"
            ),
            BlendError::ChannelsDiffer {
                group,
                lower,
                upper,
            } => write!(
                f,
                "the two bracketing temperatures carry different {group} \
                 channels ({lower:?} against {upper:?}), so blending them would \
                 have to invent a cross section for the channel one of them \
                 does not have"
            ),
        }
    }
}

impl std::error::Error for BlendError {}

fn same_channels(group: &'static str, lower: &[i32], upper: &[i32]) -> Result<(), BlendError> {
    if lower == upper {
        return Ok(());
    }
    Err(BlendError::ChannelsDiffer {
        group,
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    })
}

/// Blend two temperatures' lookup accelerators onto their union grid.
///
/// The MT-number vectors are COPIED from the lower source rather than rebuilt.
/// That is load-bearing: `build_inelastic_walk_order` appends untabled MTs in
/// storage order, so reordering the columns would change which channel a given
/// draw selects and desynchronise the CPU from the GPU. The columns are blended
/// in place, the labels are carried over unchanged.
///
/// The `Arc<Reaction>` arrays are left empty. Only [`synthesise_temperature`]
/// can fill them, because they must point into the blended reaction map rather
/// than at either source's reactions.
pub fn blend_fast_xs(lo: &FastXSGrid, hi: &FastXSGrid, w: f64) -> Result<FastXSGrid, BlendError> {
    let lo_grid = lo.energy.as_slice();
    let hi_grid = hi.energy.as_slice();
    if lo_grid.is_empty() {
        return Err(BlendError::EmptyGrid { which: "lower" });
    }
    if hi_grid.is_empty() {
        return Err(BlendError::EmptyGrid { which: "upper" });
    }

    same_channels("scattering", &lo.scatter_mt_numbers, &hi.scatter_mt_numbers)?;
    same_channels("fission", &lo.fission_mt_numbers, &hi.fission_mt_numbers)?;
    same_channels(
        "photon-producing",
        &lo.photon_rxn_mt_numbers,
        &hi.photon_rxn_mt_numbers,
    )?;
    same_channels(
        "absorption",
        &lo.absorption_mt_numbers,
        &hi.absorption_mt_numbers,
    )?;

    let union = union_grid(lo_grid, hi_grid);
    let energy = F64Buffer::from_slice(&union);

    // The four summed channels, blended together so one walk serves all of
    // them.
    let mut xs = Vec::with_capacity(union.len());
    {
        let mut lo_walk = GridWalk::new(lo_grid);
        let mut hi_walk = GridWalk::new(hi_grid);
        for &e in &union {
            let li = lo_walk.locate(e);
            let hi_i = hi_walk.locate(e);
            let mut row = [0.0f64; 4];
            for (k, slot) in row.iter_mut().enumerate() {
                let l = lo.xs.get(li.0.min(lo.xs.len().saturating_sub(1)));
                let h = hi.xs.get(hi_i.0.min(hi.xs.len().saturating_sub(1)));
                let lv = match (l, li.1) {
                    (Some(r), 0.0) => r[k],
                    (Some(r), frac) => {
                        let next = lo.xs.get(li.0 + 1).unwrap_or(r);
                        r[k] + frac * (next[k] - r[k])
                    }
                    (None, _) => 0.0,
                };
                let hv = match (h, hi_i.1) {
                    (Some(r), 0.0) => r[k],
                    (Some(r), frac) => {
                        let next = hi.xs.get(hi_i.0 + 1).unwrap_or(r);
                        r[k] + frac * (next[k] - r[k])
                    }
                    (None, _) => 0.0,
                };
                *slot = mix(lv, hv, w);
            }
            xs.push(row);
        }
    }

    let (log_e_min, inv_log_delta, log_grid_index) = build_log_grid_index(&union);
    let elastic_idx = lo.scatter_mt_numbers.iter().position(|&mt| mt == 2);

    Ok(FastXSGrid {
        log_grid_index,
        log_e_min,
        inv_log_delta,
        xs,
        scatter_mt_xs: blend_matrix(
            lo.scatter_mt_xs.as_slice(),
            hi.scatter_mt_xs.as_slice(),
            lo.scatter_mt_numbers.len(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        scatter_mt_numbers: lo.scatter_mt_numbers.clone(),
        scatter_mt_reactions: Vec::new(),
        elastic_idx,
        inelastic_walk_order: FastXSGrid::build_inelastic_walk_order(
            &lo.scatter_mt_numbers,
            elastic_idx,
        ),
        reaction_absorption: None,
        fission_mt_xs: blend_matrix(
            lo.fission_mt_xs.as_slice(),
            hi.fission_mt_xs.as_slice(),
            lo.fission_mt_numbers.len(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        fission_mt_numbers: lo.fission_mt_numbers.clone(),
        fission_mt_reactions: Vec::new(),
        has_partial_fission: lo.has_partial_fission,
        xs_ngamma: blend_column(
            lo.xs_ngamma.as_slice(),
            hi.xs_ngamma.as_slice(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        photon_prod: blend_column(
            lo.photon_prod.as_slice(),
            hi.photon_prod.as_slice(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        photon_rxn_xs: blend_matrix(
            lo.photon_rxn_xs.as_slice(),
            hi.photon_rxn_xs.as_slice(),
            lo.photon_rxn_mt_numbers.len(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        photon_rxn_mt_numbers: lo.photon_rxn_mt_numbers.clone(),
        photon_rxn_reactions: Vec::new(),
        absorption_mt_xs: blend_matrix(
            lo.absorption_mt_xs.as_slice(),
            hi.absorption_mt_xs.as_slice(),
            lo.absorption_mt_numbers.len(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        absorption_mt_numbers: lo.absorption_mt_numbers.clone(),
        delayed_photon_scaling: blend_column(
            lo.delayed_photon_scaling.as_slice(),
            hi.delayed_photon_scaling.as_slice(),
            w,
            &union,
            lo_grid,
            hi_grid,
        ),
        energy,
    })
}

/// Blend the per-MT reactions the samplers read.
///
/// Separate from the accelerator because the CPU's inelastic and absorption
/// constituent draws, and the GPU extractor, both go through
/// `reactions[temp_idx]` rather than through `fast_xs`.
///
/// Every reaction is evaluated with [`Reaction::cross_section_at`], which
/// already returns zero below a threshold, so a threshold that MOVES between
/// the two temperatures smears out of the pointwise blend without a special
/// case. The blended threshold is then recovered by trimming the leading exact
/// zeros, which is the same convention the loader uses.
pub fn blend_reactions(
    lo: &HashMap<i32, Arc<Reaction>>,
    hi: &HashMap<i32, Arc<Reaction>>,
    grid: &F64Buffer,
    w: f64,
) -> HashMap<i32, Arc<Reaction>> {
    let mts: BTreeSet<i32> = lo.keys().chain(hi.keys()).copied().collect();
    let energies = grid.as_slice();
    let mut out = HashMap::with_capacity(mts.len());

    for mt in mts {
        // The lower side where it exists, so the metadata a blend cannot
        // interpolate (products, frame, Q) comes from one place rather than
        // being mixed.
        let template = lo.get(&mt).or_else(|| hi.get(&mt));
        let Some(template) = template else { continue };

        let values: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let l = lo
                    .get(&mt)
                    .and_then(|r| r.cross_section_at(e))
                    .unwrap_or(0.0);
                let h = hi
                    .get(&mt)
                    .and_then(|r| r.cross_section_at(e))
                    .unwrap_or(0.0);
                mix(l, h, w)
            })
            .collect();

        let threshold_idx = values.iter().position(|&v| v != 0.0).unwrap_or(0);
        out.insert(
            mt,
            Arc::new(Reaction {
                cross_section: F64Buffer::from_slice(&values[threshold_idx..]),
                threshold_idx,
                // A view of the one grid the whole synthesised temperature
                // shares, not a copy: a copy per reaction is what issue #476
                // removed, and a synthesised temperature must not bring it
                // back.
                energy: grid.tail(threshold_idx),
                mt_number: mt,
                q_value: template.q_value,
                products: template.products.clone(),
                scatter_in_cm: template.scatter_in_cm,
                redundant: template.redundant,
            }),
        );
    }
    out
}

/// Give `nuclide` a temperature it does not carry, by blending its neighbours.
///
/// Does nothing when the label already resolves to a loaded temperature, so a
/// caller need not check first. Errors when the request is outside the range
/// the data covers, carrying the available list in the message.
///
/// Inserts into all five parallel structures at the numerically correct
/// position, because `loaded_temperatures` is ordered and `reactions`,
/// `fast_xs` and `urr_data` are indexed by the same position.
pub fn synthesise_temperature(
    nuclide: &mut Nuclide,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let wanted = temperature::strip_k(label).to_string();
    if nuclide.loaded_temperatures.contains(&wanted) {
        return Ok(());
    }

    // Against what is LOADED, not what the file lists: the two brackets must be
    // in memory to blend. The loader widens the request to the bracketing pair
    // before it gets here.
    let source = temperature::resolve(&wanted, &nuclide.loaded_temperatures)?;
    let (lo_idx, hi_idx, w) = match source {
        // Resolved exactly against a loaded temperature under the snap
        // tolerance, so the label the caller used names data already present
        // and nothing has to be built.
        TemperatureSource::Exact { .. } => return Ok(()),
        TemperatureSource::Blend {
            lo_idx,
            hi_idx,
            weight,
        } => (lo_idx, hi_idx, weight),
    };

    let blended_fast_xs = match (nuclide.fast_xs.get(lo_idx), nuclide.fast_xs.get(hi_idx)) {
        (Some(lo), Some(hi)) if !lo.energy.is_empty() && !hi.energy.is_empty() => {
            Some(blend_fast_xs(lo, hi, w)?)
        }
        // An XsOnly load leaves the accelerator empty on purpose. The blended
        // temperature is then also without one, which is the same shape its
        // neighbours have and is what a reaction-rate collapse reads.
        _ => None,
    };

    // One grid for the whole synthesised temperature. The accelerator's, when
    // there is one, so the reactions and the energy map are views of the same
    // allocation rather than three copies.
    let grid = match &blended_fast_xs {
        Some(f) => f.energy.clone(),
        None => {
            let by_label = |idx: usize| -> F64Buffer {
                nuclide
                    .loaded_temperatures
                    .get(idx)
                    .and_then(|t| nuclide.energy.as_ref().and_then(|m| m.get(t)))
                    .cloned()
                    .unwrap_or_else(F64Buffer::empty)
            };
            let lo = by_label(lo_idx);
            let hi = by_label(hi_idx);
            F64Buffer::from_slice(&union_grid(lo.as_slice(), hi.as_slice()))
        }
    };

    let lo_reactions = nuclide.reactions.get(lo_idx).cloned().unwrap_or_default();
    let hi_reactions = nuclide.reactions.get(hi_idx).cloned().unwrap_or_default();
    let mut blended_reactions = blend_reactions(&lo_reactions, &hi_reactions, &grid, w);

    // Rewire the accelerator's reaction pointers into the blended map, so the
    // samplers that go through `fast_xs` and the ones that go through
    // `reactions` are looking at the same cross sections.
    let fast = blended_fast_xs.map(|mut f| {
        let pick = |mts: &[i32]| -> Vec<Arc<Reaction>> {
            mts.iter()
                .filter_map(|mt| blended_reactions.get(mt).cloned())
                .collect()
        };
        f.scatter_mt_reactions = pick(&f.scatter_mt_numbers);
        f.fission_mt_reactions = pick(&f.fission_mt_numbers);
        f.photon_rxn_reactions = pick(&f.photon_rxn_mt_numbers);
        f.reaction_absorption = blended_reactions.get(&101).cloned();
        f
    });
    blended_reactions.shrink_to_fit();

    // The nearer bracketing table, never None. See the module doc for why a
    // None here is the most expensive silent failure available.
    let urr = if w < 0.5 {
        nuclide.urr_data.get(lo_idx).cloned().flatten()
    } else {
        nuclide.urr_data.get(hi_idx).cloned().flatten()
    };

    let position = nuclide
        .loaded_temperatures
        .iter()
        .position(|t| {
            temperature::label_to_kelvin(t).unwrap_or(f64::INFINITY)
                > temperature::label_to_kelvin(&wanted).unwrap_or(0.0)
        })
        .unwrap_or(nuclide.loaded_temperatures.len());

    nuclide.loaded_temperatures.insert(position, wanted.clone());
    nuclide.reactions.insert(position, blended_reactions);
    if !nuclide.fast_xs.is_empty() || fast.is_some() {
        nuclide.fast_xs.insert(
            position.min(nuclide.fast_xs.len()),
            fast.unwrap_or_default(),
        );
    }
    if !nuclide.urr_data.is_empty() {
        nuclide
            .urr_data
            .insert(position.min(nuclide.urr_data.len()), urr);
    }
    if let Some(map) = nuclide.energy.as_mut() {
        map.insert(wanted, grid);
    }

    Ok(())
}
