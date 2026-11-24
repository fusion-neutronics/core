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
/// Fixed rather than scaled to the grid (the published data used 8000 for a
/// 631-point H1 grid and an 80,222-point U235 one alike), so every
/// temperature's accelerator has the same shape, synthesised or built.
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

/// `(1 - w) * lo + w * hi`, with both sides evaluated at the same energy.
#[inline]
fn mix(lo: f64, hi: f64, w: f64) -> f64 {
    lo + w * (hi - lo)
}

/// Why a blend could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum BlendError {
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

/// Refuse to build a lookup for two bracketing temperatures whose lookups do
/// not carry the same channels in the same order.
///
/// Checked on the two loaded lookups rather than on the blended one, because the
/// blended reactions are the union of both sides' MTs and a channel only one
/// side carries would build a column without complaint.
fn same_lookup_channels(lo: &FastXSGrid, hi: &FastXSGrid) -> Result<(), BlendError> {
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
    )
}

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

/// Blend the per-MT reactions the samplers read.
///
/// The CPU's inelastic and absorption constituent draws and the GPU extractor
/// read these through `reactions[temp_idx]`, and the synthesised temperature's
/// lookup is then built from them by [`FastXSGrid::build`], exactly as a loaded
/// temperature's is built from the reactions read off disk.
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
                // shares, not a copy: a synthesised temperature must not
                // duplicate the grid once per reaction.
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

    // An XsOnly load leaves the lookup empty on purpose. The blended
    // temperature is then also without one, which is the same shape its
    // neighbours have and is what a reaction-rate collapse reads.
    let build_lookup = match (nuclide.fast_xs.get(lo_idx), nuclide.fast_xs.get(hi_idx)) {
        (Some(lo), Some(hi)) if !lo.energy.is_empty() && !hi.energy.is_empty() => {
            same_lookup_channels(lo, hi)?;
            true
        }
        _ => false,
    };

    // One grid for the whole synthesised temperature, shared by the reactions,
    // the lookup and the energy map rather than copied into each.
    let grid = {
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
    };

    let lo_reactions = nuclide.reactions.get(lo_idx).cloned().unwrap_or_default();
    let hi_reactions = nuclide.reactions.get(hi_idx).cloned().unwrap_or_default();
    let mut blended_reactions = blend_reactions(&lo_reactions, &hi_reactions, &grid, w);
    blended_reactions.shrink_to_fit();

    // Built from the blended reactions by the same builder the loader runs on
    // every loaded temperature, so the two cannot disagree. That includes the
    // column order: the builder lays the MT columns out in ascending MT, and
    // `build_inelastic_walk_order` depends on that storage order, so a
    // synthesised temperature walks its inelastic channels in the same order
    // as its neighbours and the CPU stays in step with the GPU.
    let fast = if build_lookup {
        let name = nuclide.name.as_deref().unwrap_or("nuclide");
        Some(FastXSGrid::build(
            &grid,
            &blended_reactions,
            nuclide.fission_photon_release.as_ref(),
            &format!("{name} at {wanted} K (blended)"),
        )?)
    } else {
        None
    };

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
