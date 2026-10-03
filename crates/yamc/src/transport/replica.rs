//! Correlated nuclear-data replica weights.
//!
//! Every history is transported once, with the nominal cross sections, and
//! carries one weight ratio per replica: the likelihood ratio of its path
//! under replica `k`'s cross sections against the nominal ones (Rief, Ann.
//! Nucl. Energy 11 (1984) 455). Replica `k`'s tally is then the nominal
//! histories' scores times those ratios, so every replica sees every history
//! and the nominal run is the replica whose ratios are all one.
//!
//! The ratio changes in two places:
//!
//! - **a flight** of length `d` at energy `E` in a material is a survival to
//!   `d`, whose probability is `exp(-Σ_t d)`; under replica `k` it is
//!   `exp(-Σ'_t,k d)`, so the ratio is multiplied by `exp(-ΔΣ_k d)`;
//! - **a collision** that ends in reaction `mt` on a nuclide had probability
//!   density `Σ_t · (Σ_i / Σ_t) · (σ_mt / σ_i) = N_i σ_mt`, so the ratio is
//!   multiplied by `σ'_mt,k / σ_mt`. Absorption ends the history and needs no
//!   factor; a fission's neutrons inherit `σ'_f,k / σ_f`.
//!
//! A track-length score over a segment is the integral of the likelihood
//! ratio of reaching each point along it, `r_k ∫_0^d exp(-ΔΣ_k s) ds`, which
//! is `r_k d φ(ΔΣ_k d)` with `φ(x) = (1 - e^-x) / x`. A reaction-rate score
//! carries its own cross section too, so it is further multiplied by
//! `Σ'_x,k / Σ_x`.
//!
//! The perturbed cross sections are the same as [`crate::xs_perturbation`]
//! builds for a rerun: each covered partial multiplied by its covariance
//! cell's multiplier and shifted by its absolute cell's shift, a component
//! of a sum taking the sum's. Rows with no covariance, heating and damage
//! rows, and the short-range noise stay nominal; particle production
//! (MT 203 to 207) moves with the reactions emitting the particle.
//!
//! In a nuclide's unresolved range the flight, the reaction and the tallies
//! read a probability-table band instead of the smooth cross sections: each
//! of the lookup's four channels (elastic, non-elastic scattering,
//! disappearance, fission) is the smooth one times the band's factor. A
//! replica moves each band by its smooth channel's relative change, as a
//! factor table does and the reruns do for either kind of table, so a
//! channel's change enters the flight and the band-valued scores times that
//! factor. The collision factor needs no band: within a channel the band
//! cancels from `σ'_mt / σ_mt`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::synthesis::{channel_of, Channel};
use yamc_nuclide::urr::UrrBand;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::welford::ReplicaModes;
use yani_transmute::covariance_fold::{transport_fields, Read, TransportField};
use yani_transmute::covariance_sample::{Modes, Sampler};

/// The share of each nuclide's correlation its kept modes hold. Fewer modes
/// cost less per event; what they miss leaves the estimate unbiased and only
/// puts more of it on the replicas' sampling. On an 8 cm iron sphere at
/// 14 MeV this keeps 123 of Fe56's 1361 modes and its sigma's spread over
/// draws matches that with all of them.
const KEPT_CORRELATION: f64 = 0.999;

/// One covariance source of one nuclide: the reaction whose cells some of
/// its partials read, with what those partials add up to.
struct Source {
    /// `Σ σ_mt(E)` over the partials reading this source, in barns, on the
    /// nuclide's grid.
    sum: Vec<f64>,
    /// `sum` split by the lookup channel each partial is in (see
    /// [`Channel`]), for a probability-table band, which scales each channel
    /// by its own factor.
    channels: [Vec<f64>; 4],
    /// `σ_src(E)`, the source reaction's own cross section, on the grid: what
    /// an absolute shift on its cells is a shift of.
    own: Vec<f64>,
    /// Relative cells `[lo, hi)`, and per cell the `m_k - 1` of each replica,
    /// row-major `cells × R`.
    relative: Vec<(f64, f64)>,
    relative_dm: Vec<f64>,
    /// Absolute cells, and per cell each replica's shift in barns.
    absolute: Vec<(f64, f64)>,
    absolute_shift: Vec<f64>,
    /// Per relative cell, its row of the nuclide's relative mode loadings,
    /// row-major `cells × k`, with `(offset, k)` the nuclide's relative modes
    /// in the run's mode vector.
    relative_loading: Vec<f64>,
    relative_modes: (usize, usize),
    /// The same for the absolute cells.
    absolute_loading: Vec<f64>,
    absolute_modes: (usize, usize),
}

impl Source {
    /// The cell of `cells` holding `e`, the last closed at its top.
    fn cell(cells: &[(f64, f64)], e: f64) -> Option<usize> {
        let i = cells.partition_point(|c| c.0 <= e).checked_sub(1)?;
        (e < cells[i].1 || (i + 1 == cells.len() && e == cells[i].1)).then_some(i)
    }

    /// Each replica's relative change of a partial reading this source at
    /// `e`, `σ'/σ - 1`, added into `out` times `scale`, and the change's
    /// gradient along the run's modes into `modes` (left alone when empty).
    /// `own` is `σ_src(e)`.
    ///
    /// The change is linear in the draw, `Σ_c g_c p_c` over the cells holding
    /// `e`, so a replica's is the gradient dotted with its draw and the
    /// gradient along mode `k` is `Σ_c g_c L_ck`, `L` the mode loadings.
    fn add_change(&self, e: f64, own: f64, scale: f64, out: &mut [f64], modes: &mut [f64]) {
        let r = out.len();
        if let Some(c) = Self::cell(&self.relative, e) {
            for (o, dm) in out.iter_mut().zip(&self.relative_dm[c * r..(c + 1) * r]) {
                *o += scale * dm;
            }
            let (offset, k) = self.relative_modes;
            if k > 0 && !modes.is_empty() {
                let loading = &self.relative_loading[c * k..(c + 1) * k];
                for (o, l) in modes[offset..offset + k].iter_mut().zip(loading) {
                    *o += scale * l;
                }
            }
        }
        if own > 0.0 {
            if let Some(c) = Self::cell(&self.absolute, e) {
                for (o, a) in out.iter_mut().zip(&self.absolute_shift[c * r..(c + 1) * r]) {
                    *o += scale * a / own;
                }
                let (offset, k) = self.absolute_modes;
                if k > 0 && !modes.is_empty() {
                    let loading = &self.absolute_loading[c * k..(c + 1) * k];
                    for (o, l) in modes[offset..offset + k].iter_mut().zip(loading) {
                        *o += scale * l / own;
                    }
                }
            }
        }
    }
}

/// One reaction-rate score's tables for one nuclide: per source, the summed
/// cross section of the score's partials reading it, and the nominal score
/// cross section, on the nuclide's grid.
type ScoreTable = (Vec<(usize, Vec<f64>)>, Vec<f64>);

/// One nuclide of one material, as the replicas read it.
struct ReplicaNuclide {
    /// Atoms per barn-cm.
    density: f64,
    grid: Vec<f64>,
    sources: Vec<Source>,
    /// Each covered partial's source.
    source_of: HashMap<i32, usize>,
    /// The fission partials, each with its source (if covered) and its cross
    /// section on the grid, for the factor a fission's neutrons inherit.
    fission: Vec<(Option<usize>, Vec<f64>)>,
    /// Per tally score MT: per source, the summed cross section of the
    /// score's partials reading it, and the nominal score cross section, on
    /// the grid.
    scores: HashMap<i32, ScoreTable>,
    /// The nuclide and its temperature index, where it has probability
    /// tables to draw a band from.
    urr: Option<(Arc<Nuclide>, usize)>,
}

/// The channels whose bands a tally's score of `mt` reads inside the
/// unresolved range ([`yamc_materials::Material::macro_xs_by_mt`]), or `None`
/// for a score read from the smooth cross sections there too.
fn band_channels(mt: i32) -> Option<&'static [Channel]> {
    match mt {
        1 => Some(&Channel::ALL),
        2 => Some(&[Channel::Elastic]),
        18 => Some(&[Channel::Fission]),
        27 => Some(&[Channel::Capture, Channel::Fission]),
        102 => Some(&[Channel::Capture]),
        _ => None,
    }
}

/// Per channel, the band over the smooth cross section, `band_c / σ_c`: the
/// factor a band puts on its channel, and so on any change to it. A channel
/// the band leaves out (non-elastic, where the table does not carry it)
/// moves nothing.
fn band_factors(band: &UrrBand) -> [f64; 4] {
    let factor = |b: f64, smooth: f64| if smooth > 0.0 { b / smooth } else { 0.0 };
    [
        factor(band.elastic, band.smooth_elastic),
        if band.inelastic_in_table { 1.0 } else { 0.0 },
        factor(band.capture, band.smooth_capture),
        factor(band.fission, band.smooth_fission),
    ]
}

/// The band's value of each channel, in [`Channel::ALL`] order.
fn band_values(band: &UrrBand) -> [f64; 4] {
    [band.elastic, band.inelastic(), band.capture, band.fission]
}

impl ReplicaNuclide {
    /// `(index, fraction)` of `e` on the grid, for linear interpolation.
    fn locate(&self, e: f64) -> Option<(usize, f64)> {
        let g = &self.grid;
        if g.len() < 2 || e < g[0] || e > g[g.len() - 1] {
            return None;
        }
        let i = g
            .partition_point(|&x| x <= e)
            .saturating_sub(1)
            .min(g.len() - 2);
        let f = if g[i + 1] > g[i] {
            (e - g[i]) / (g[i + 1] - g[i])
        } else {
            0.0
        };
        Some((i, f))
    }

    fn at(table: &[f64], (i, f): (usize, f64)) -> f64 {
        table[i] + f * (table[i + 1] - table[i])
    }

    /// The probability-table band at `e` for the per-collision base seed
    /// `urr_random`, the one the flight, the reaction and the tallies read.
    fn band(&self, e: f64, urr_random: Option<f64>) -> Option<UrrBand> {
        let (nuclide, t) = self.urr.as_ref()?;
        nuclide.urr_band(*t, e, urr_random?)
    }

    /// `Σ_c g_c σ_c,s(E)` over `channels`: what a source's partials in
    /// those channels add up to once each carries its band factor `g_c`.
    fn banded(source: &Source, at: (usize, f64), g: &[f64; 4], channels: &[Channel]) -> f64 {
        channels
            .iter()
            .map(|c| g[c.index()] * Self::at(&source.channels[c.index()], at))
            .sum()
    }
}

/// One material's replica tables.
pub(crate) struct ReplicaMaterial {
    nuclides: Vec<ReplicaNuclide>,
    by_name: HashMap<String, usize>,
}

/// The replica weights of one run: the draws, tabulated per material slot.
pub(crate) struct ReplicaContext {
    pub(crate) replicas: usize,
    /// By material slot (`Cell::material_idx`).
    materials: Vec<Option<ReplicaMaterial>>,
    /// The principal modes the control variate is built on, `None` where no
    /// nuclide has any.
    pub(crate) modes: Option<Arc<ReplicaModes>>,
}

/// One nuclide's modes, where they sit in the run's mode vector.
struct RegisteredModes {
    modes: Modes,
    relative_offset: usize,
    absolute_offset: usize,
}

/// Every nuclide's modes across a run's materials, numbered once each, so a
/// nuclide in two materials moves along the same modes in both.
#[derive(Default)]
struct ModeRegistry {
    by_name: HashMap<String, RegisteredModes>,
    variance: Vec<f64>,
    nuclides: Vec<String>,
    /// Per replica, its coordinate on each mode so far.
    coordinates: Vec<Vec<f64>>,
}

impl ModeRegistry {
    /// Number `name`'s `modes` after the ones already registered, with every
    /// replica's coordinates on them from its draw.
    fn register(&mut self, name: &str, modes: Modes, relative: &[&[f64]], absolute: &[&[f64]]) {
        if self.coordinates.is_empty() {
            self.coordinates = vec![Vec::new(); relative.len()];
        }
        let relative_offset = self.variance.len();
        let absolute_offset = relative_offset + modes.n_relative();
        let project = |projection: &[f64], k: usize, draw: &[f64]| -> Vec<f64> {
            (0..k)
                .map(|m| {
                    draw.iter()
                        .enumerate()
                        .map(|(c, p)| p * projection[c * k + m])
                        .sum()
                })
                .collect()
        };
        for (r, coordinates) in self.coordinates.iter_mut().enumerate() {
            coordinates.extend(project(
                &modes.relative_projection,
                modes.n_relative(),
                relative[r],
            ));
            coordinates.extend(project(
                &modes.absolute_projection,
                modes.n_absolute(),
                absolute[r],
            ));
        }
        self.variance.extend(&modes.relative_variance);
        self.variance.extend(&modes.absolute_variance);
        self.nuclides
            .extend(std::iter::repeat_n(name.to_string(), modes.len()));
        self.by_name.insert(
            name.to_string(),
            RegisteredModes {
                modes,
                relative_offset,
                absolute_offset,
            },
        );
    }

    fn finish(self) -> Option<Arc<ReplicaModes>> {
        if self.variance.is_empty() {
            return None;
        }
        Some(Arc::new(ReplicaModes {
            variance: self.variance,
            nuclides: self.nuclides,
            coordinates: self.coordinates.into_iter().flatten().collect(),
        }))
    }
}

/// The index of `temperature` among `nuclide`'s loaded ones, or its only one.
fn temperature_index(nuclide: &Nuclide, temperature: &str) -> Result<usize, String> {
    nuclide
        .loaded_temperatures
        .iter()
        .position(|l| l == temperature)
        .or_else(|| (nuclide.loaded_temperatures.len() == 1).then_some(0))
        .ok_or_else(|| {
            format!(
                "{} is not loaded at {temperature} K for the replica weights",
                nuclide.name.as_deref().unwrap_or("a nuclide")
            )
        })
}

/// A cross section on the full grid of `n` points, zero below threshold.
fn on_grid(reaction: &Reaction, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    let xs = reaction.cross_section.as_slice();
    let start = reaction.threshold_idx.min(n);
    let len = xs.len().min(n - start);
    out[start..start + len].copy_from_slice(&xs[..len]);
    out
}

/// The non-redundant partials a score of `mt` sums, among those the nuclide
/// holds: `mt` itself when it is one, else its components under the ENDF sum
/// rules, recursively.
fn partials_of(mt: i32, held: &BTreeMap<i32, Arc<Reaction>>) -> BTreeSet<i32> {
    if held.get(&mt).is_some_and(|r| !r.redundant) {
        return BTreeSet::from([mt]);
    }
    let mut out = BTreeSet::new();
    if let Some(rule) = endf::data::sum_rule(mt) {
        for &c in rule {
            out.extend(partials_of(c, held));
        }
    }
    out
}

/// Whether `mt` is a fission partial.
fn is_fission(mt: i32) -> bool {
    matches!(mt, 18 | 19 | 20 | 21 | 38)
}

/// The score table of a particle-production score (MT 203 to 207: proton,
/// deuteron, triton, helion and alpha production), or `None` for any other
/// MT.
///
/// A production cross section is `Σ_mt c_mt σ_mt` over the reactions that
/// emit the particle, `c_mt` the count of it each emits by the reaction's
/// definition ([`endf::reaction::light_particles`]). Each covered reaction
/// joins its source's table with that weight. The denominator is the
/// nuclide's stored production row, the one the tally scores. Production the
/// stated reactions do not account for, as an evaluation that carries its
/// charged particles inside MT 5 has, is attributed to MT 5 where MT 5 is
/// covered: the part of the row MT 5 is the only candidate for.
fn production_table(
    x: i32,
    held: &BTreeMap<i32, Arc<Reaction>>,
    source_of: &HashMap<i32, usize>,
    n: usize,
) -> Option<ScoreTable> {
    if !(203..=207).contains(&x) {
        return None;
    }
    let particle = (x - 203) as usize;
    let nominal = match held.get(&x) {
        Some(row) => on_grid(row, n),
        None => return Some((Vec::new(), vec![0.0; n])),
    };
    let mut by_source: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut explained = vec![0.0; n];
    for (mt, reaction) in held {
        if reaction.redundant {
            continue;
        }
        let Some(counts) = endf::reaction::light_particles(*mt) else {
            continue;
        };
        let c = counts[particle] as f64;
        if c == 0.0 {
            continue;
        }
        let xs = on_grid(reaction, n);
        for (e, v) in explained.iter_mut().zip(&xs) {
            *e += c * v;
        }
        if let Some(&s) = source_of.get(mt) {
            let entry = by_source.entry(s).or_insert_with(|| vec![0.0; n]);
            for (o, v) in entry.iter_mut().zip(&xs) {
                *o += c * v;
            }
        }
    }
    if let (Some(&s5), true) = (
        source_of.get(&5),
        held.get(&5).is_some_and(|r| !r.redundant),
    ) {
        let residual: Vec<f64> = nominal
            .iter()
            .zip(&explained)
            .map(|(row, known)| (row - known).max(0.0))
            .collect();
        if residual.iter().any(|v| *v > 0.0) {
            let entry = by_source.entry(s5).or_insert_with(|| vec![0.0; n]);
            for (o, v) in entry.iter_mut().zip(residual) {
                *o += v;
            }
        }
    }
    Some((by_source.into_iter().collect(), nominal))
}

impl ReplicaContext {
    /// Tabulate `replicas` draws, seeded by `seed`, for every material in
    /// `materials` (indexed by slot), for the reaction-rate scores of
    /// `tallies`. The materials' nuclear data must be loaded, with
    /// covariance, at the run's temperature.
    pub(crate) fn new(
        materials: &[Arc<Material>],
        tallies: &[Arc<Tally>],
        replicas: usize,
        seed: u64,
    ) -> Result<Self, String> {
        let score_mts: BTreeSet<i32> = tallies
            .iter()
            .flat_map(|t| t.scores.iter())
            .filter_map(|s| match s {
                Score::ReactionRate(r) => Some(r.mt.as_i32()),
                Score::Production(p) => Some(p.mt.as_i32()),
                _ => None,
            })
            .collect();
        let mut out = Vec::with_capacity(materials.len());
        let mut registry = ModeRegistry::default();
        for material in materials {
            out.push(Self::material(
                material,
                &score_mts,
                replicas,
                seed,
                &mut registry,
            )?);
        }
        // With no covariance anywhere every replica is the nominal run and
        // every sigma would read zero, which looks like an answer. It is not
        // one: the data says nothing about the uncertainty.
        if out.iter().all(Option::is_none) {
            return Err(
                "data_uncertainty: no nuclide in the model's materials carries covariance \
                 data, so every nuclear-data sigma would read zero. Use a library whose \
                 data includes covariance; model.data_uncertainty_coverage() lists what \
                 each nuclide carries"
                    .to_string(),
            );
        }
        Ok(ReplicaContext {
            replicas,
            materials: out,
            modes: registry.finish(),
        })
    }

    /// Modes the particles carry derivatives along.
    pub(crate) fn n_modes(&self) -> usize {
        self.modes.as_ref().map_or(0, |m| m.len())
    }

    /// A source particle's replica state: every ratio one, every derivative
    /// zero.
    pub(crate) fn source_state(&self) -> Box<[f64]> {
        let mut state = vec![1.0; self.replicas];
        state.resize(self.replicas + self.n_modes(), 0.0);
        state.into_boxed_slice()
    }

    fn material(
        material: &Material,
        score_mts: &BTreeSet<i32>,
        replicas: usize,
        seed: u64,
        registry: &mut ModeRegistry,
    ) -> Result<Option<ReplicaMaterial>, String> {
        let (fields, _) = transport_fields(material);
        let cell_fields = fields
            .iter()
            .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
            .collect();
        let sampler = Sampler::new(&cell_fields, &[]);
        let draws: Vec<_> = (0..replicas as u64)
            .map(|k| sampler.draw(seed, k))
            .collect();
        let densities = material
            .get_atoms_per_barn_cm()
            .map_err(|e| format!("atom densities: {e}"))?;

        // Every nuclide with data, covered or not: one with no covariance
        // perturbs nothing, but its cross section is still part of every
        // reaction-rate score's nominal Σ_x, which the score's relative
        // change is divided by.
        let mut names: Vec<&String> = material.nuclides.keys().collect();
        names.sort();
        let mut nuclides = Vec::new();
        let mut by_name = HashMap::new();
        let mut any_covered = false;
        for name in names {
            let Some(nuclide) = material.nuclide_data.get(name) else {
                continue;
            };
            let transport = fields.get(name).filter(|t| t.field.is_some());
            let relative: Vec<&[f64]> = draws
                .iter()
                .map(|d| d.relative(name).unwrap_or(&[]))
                .collect();
            let absolute: Vec<&[f64]> = draws
                .iter()
                .map(|d| d.absolute(name).unwrap_or(&[]))
                .collect();
            if transport.is_some() && !registry.by_name.contains_key(name) {
                if let Some(modes) = sampler.modes(name, KEPT_CORRELATION) {
                    registry.register(name, modes, &relative, &absolute);
                }
            }
            let Some(tab) = Self::nuclide(
                Arc::clone(nuclide),
                material.temperature(),
                transport,
                &relative,
                &absolute,
                score_mts,
                densities.get(name).copied().unwrap_or(0.0),
                registry.by_name.get(name),
            )?
            else {
                continue;
            };
            any_covered |= !tab.sources.is_empty();
            by_name.insert(name.clone(), nuclides.len());
            nuclides.push(tab);
        }
        Ok(any_covered.then_some(ReplicaMaterial { nuclides, by_name }))
    }

    #[allow(clippy::too_many_arguments)]
    fn nuclide(
        nuclide: Arc<Nuclide>,
        temperature: &str,
        transport: Option<&TransportField>,
        relative: &[&[f64]],
        absolute: &[&[f64]],
        score_mts: &BTreeSet<i32>,
        density: f64,
        modes: Option<&RegisteredModes>,
    ) -> Result<Option<ReplicaNuclide>, String> {
        let t = temperature_index(&nuclide, temperature)?;
        let label = &nuclide.loaded_temperatures[t];
        let Some(grid) = nuclide.energy.as_ref().and_then(|m| m.get(label)) else {
            return Ok(None);
        };
        let grid = grid.as_slice().to_vec();
        let n = grid.len();
        let held: BTreeMap<i32, Arc<Reaction>> = nuclide.reactions[t]
            .iter()
            .map(|(mt, r)| (*mt, Arc::clone(r)))
            .collect();
        let r = relative.len();

        let mut sources: Vec<Source> = Vec::new();
        let mut index: HashMap<i32, usize> = HashMap::new();
        let mut source_of: HashMap<i32, usize> = HashMap::new();
        let no_reads = BTreeMap::new();
        let reads = transport.map_or(&no_reads, |t| &t.reads);
        for (mt, read) in reads {
            let Some(field) = transport.and_then(|t| t.field.as_ref()) else {
                break;
            };
            let src = match read {
                Read::Own => *mt,
                Read::Parent(sum) => *sum,
                Read::Nominal => continue,
            };
            let Some(reaction) = held.get(mt) else {
                continue;
            };
            let s = *index.entry(src).or_insert_with(|| {
                // The source's cells, each replica's draw on each, and each
                // cell's row of the nuclide's mode loadings (`k` per cell).
                let cells = |list: &[yani_transmute::covariance_fold::Cell],
                             values: &[&[f64]],
                             loading: &[f64],
                             k: usize|
                 -> (Vec<(f64, f64)>, Vec<f64>, Vec<f64>) {
                    let picked: Vec<usize> = list
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.mt == src)
                        .map(|(k, _)| k)
                        .collect();
                    let bounds = picked.iter().map(|&k| (list[k].lo, list[k].hi)).collect();
                    let mut per = Vec::with_capacity(picked.len() * r);
                    let mut rows = Vec::with_capacity(picked.len() * k);
                    for &c in &picked {
                        for v in values {
                            per.push(v.get(c).copied().unwrap_or(0.0));
                        }
                        rows.extend_from_slice(&loading[c * k..(c + 1) * k]);
                    }
                    (bounds, per, rows)
                };
                let none = Modes::default();
                let m = modes.map_or(&none, |m| &m.modes);
                let (rel, rel_dm, rel_loading) = cells(
                    &field.relative_cells,
                    relative,
                    &m.relative_loading,
                    m.n_relative(),
                );
                let (abs, abs_shift, abs_loading) = cells(
                    &field.absolute_cells,
                    absolute,
                    &m.absolute_loading,
                    m.n_absolute(),
                );
                let own = held
                    .get(&src)
                    .map_or_else(|| vec![0.0; n], |r| on_grid(r, n));
                sources.push(Source {
                    sum: vec![0.0; n],
                    channels: std::array::from_fn(|_| vec![0.0; n]),
                    own,
                    relative: rel,
                    relative_dm: rel_dm,
                    absolute: abs,
                    absolute_shift: abs_shift,
                    relative_loading: rel_loading,
                    relative_modes: (modes.map_or(0, |m| m.relative_offset), m.n_relative()),
                    absolute_loading: abs_loading,
                    absolute_modes: (modes.map_or(0, |m| m.absolute_offset), m.n_absolute()),
                });
                sources.len() - 1
            });
            let xs = on_grid(reaction, n);
            if let Some(c) = channel_of(*mt) {
                for (o, v) in sources[s].channels[c.index()].iter_mut().zip(&xs) {
                    *o += v;
                }
            }
            for (o, v) in sources[s].sum.iter_mut().zip(xs) {
                *o += v;
            }
            source_of.insert(*mt, s);
        }
        // A sum read only by its components has no row of its own to read an
        // absolute shift against when it is absent: the components' sum is
        // that row.
        for s in &mut sources {
            if s.own.iter().all(|v| *v == 0.0) {
                s.own.clone_from(&s.sum);
            }
        }

        let fission = held
            .iter()
            .filter(|(mt, r)| is_fission(**mt) && !r.redundant)
            .map(|(mt, r)| (source_of.get(mt).copied(), on_grid(r, n)))
            .collect();

        let mut scores = HashMap::new();
        for &x in score_mts {
            if let Some(table) = production_table(x, &held, &source_of, n) {
                scores.insert(x, table);
                continue;
            }
            let partials = partials_of(x, &held);
            if partials.is_empty() {
                continue;
            }
            let mut by_source: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for p in &partials {
                if let (Some(&s), Some(reaction)) = (source_of.get(p), held.get(p)) {
                    let e = by_source.entry(s).or_insert_with(|| vec![0.0; n]);
                    for (o, v) in e.iter_mut().zip(on_grid(reaction, n)) {
                        *o += v;
                    }
                }
            }
            // The nominal score cross section the tally multiplies by: the
            // stored row where the nuclide has one, else its partials' sum.
            let nominal = match held.get(&x) {
                Some(row) => on_grid(row, n),
                None => {
                    let mut total = vec![0.0; n];
                    for p in &partials {
                        for (o, v) in total.iter_mut().zip(on_grid(&held[p], n)) {
                            *o += v;
                        }
                    }
                    total
                }
            };
            scores.insert(x, (by_source.into_iter().collect(), nominal));
        }

        // The band index is the material's own temperature lookup, the one
        // its flights and tallies draw bands with.
        let urr = nuclide
            .get_temp_idx(temperature)
            .filter(|_| nuclide.urr_present)
            .map(|t| (Arc::clone(&nuclide), t));

        Ok(Some(ReplicaNuclide {
            density,
            grid,
            sources,
            source_of,
            fission,
            scores,
            urr,
        }))
    }

    fn material_at(&self, slot: Option<u32>) -> Option<&ReplicaMaterial> {
        self.materials.get(slot? as usize)?.as_ref()
    }

    /// `ΔΣ_k(E) = Σ'_t,k - Σ_t` in 1/cm for each replica, into `out`.
    ///
    /// Inside a nuclide's unresolved range the flight saw its band, each
    /// channel the smooth one times the band's factor, and a replica moves
    /// each band by its smooth channel's relative change, so the change in
    /// the band total is each channel's change times that factor.
    /// `urr_random` is the per-collision base seed the bands are drawn with.
    ///
    /// `grad` gets `ΔΣ`'s gradient along the run's modes, in 1/cm.
    pub(crate) fn flight_delta(
        &self,
        slot: Option<u32>,
        e: f64,
        urr_random: Option<f64>,
        out: &mut [f64],
        grad: &mut [f64],
    ) {
        out.fill(0.0);
        grad.fill(0.0);
        let Some(m) = self.material_at(slot) else {
            return;
        };
        for nuc in &m.nuclides {
            let Some(at) = nuc.locate(e) else {
                continue;
            };
            let g = nuc.band(e, urr_random).map(|b| band_factors(&b));
            for s in &nuc.sources {
                let sum = match &g {
                    Some(g) => ReplicaNuclide::banded(s, at, g, &Channel::ALL),
                    None => ReplicaNuclide::at(&s.sum, at),
                };
                if sum == 0.0 {
                    continue;
                }
                let own = ReplicaNuclide::at(&s.own, at);
                s.add_change(e, own, nuc.density * sum, out, grad);
            }
        }
    }

    /// The factor `σ'_mt,k / σ_mt` a collision ending in reaction `mt` on
    /// `nuclide` puts on each replica's ratio, into `out`; all ones where
    /// nothing perturbs it. `grad` gets the factor's gradient along the
    /// run's modes, which is its logarithm's at the nominal.
    pub(crate) fn collision_factor(
        &self,
        slot: Option<u32>,
        nuclide: &str,
        mt: i32,
        e: f64,
        out: &mut [f64],
        grad: &mut [f64],
    ) {
        out.fill(1.0);
        grad.fill(0.0);
        let Some(m) = self.material_at(slot) else {
            return;
        };
        let Some(nuc) = m.by_name.get(nuclide).map(|&i| &m.nuclides[i]) else {
            return;
        };
        let Some(at) = nuc.locate(e) else {
            return;
        };
        if is_fission(mt) && !nuc.fission.is_empty() {
            // The fission arm samples which fission partial too, but its
            // neutrons are what continue, so they carry σ'_f / σ_f.
            let total: f64 = nuc
                .fission
                .iter()
                .map(|(_, xs)| ReplicaNuclide::at(xs, at))
                .sum();
            if total <= 0.0 {
                return;
            }
            for (s, xs) in &nuc.fission {
                if let Some(s) = s {
                    let src = &nuc.sources[*s];
                    let w = ReplicaNuclide::at(xs, at) / total;
                    let own = ReplicaNuclide::at(&src.own, at);
                    src.add_change(e, own, w, out, grad);
                }
            }
            return;
        }
        if let Some(&s) = nuc.source_of.get(&mt) {
            let src = &nuc.sources[s];
            let own = ReplicaNuclide::at(&src.own, at);
            src.add_change(e, own, 1.0, out, grad);
        }
    }

    /// `Σ'_x,k / Σ_x` for a reaction-rate or particle-production score at `e`
    /// in the material, into `out`; all ones for flux, for a score nothing
    /// perturbs, and for heating and damage scores, which stay nominal.
    ///
    /// Inside a nuclide's unresolved range a tally reads the total, elastic,
    /// fission, absorption and capture from its band, so that nuclide's
    /// share of those scores is the band's, and moves by the band's change.
    /// `urr_random` is the per-collision base seed the bands are drawn with.
    pub(crate) fn score_ratio(
        &self,
        slot: Option<u32>,
        score: &Score,
        e: f64,
        urr_random: Option<f64>,
        out: &mut [f64],
        grad: &mut [f64],
    ) {
        out.fill(1.0);
        grad.fill(0.0);
        let x = match score {
            Score::ReactionRate(rr) => rr.mt.as_i32(),
            Score::Production(p) => p.mt.as_i32(),
            _ => return,
        };
        let Some(m) = self.material_at(slot) else {
            return;
        };
        let banded = band_channels(x);
        let mut denominator = 0.0;
        let mut change = vec![0.0; out.len()];
        let mut change_grad = vec![0.0; grad.len()];
        for nuc in &m.nuclides {
            let Some(at) = nuc.locate(e) else {
                continue;
            };
            if let (Some(channels), Some(band)) = (banded, nuc.band(e, urr_random)) {
                let values = band_values(&band);
                denominator +=
                    nuc.density * channels.iter().map(|c| values[c.index()]).sum::<f64>();
                let g = band_factors(&band);
                for src in &nuc.sources {
                    let own = ReplicaNuclide::at(&src.own, at);
                    let sum = ReplicaNuclide::banded(src, at, &g, channels);
                    src.add_change(e, own, nuc.density * sum, &mut change, &mut change_grad);
                }
                continue;
            }
            let Some((by_source, nominal)) = nuc.scores.get(&x) else {
                continue;
            };
            denominator += nuc.density * ReplicaNuclide::at(nominal, at);
            for (s, xs) in by_source {
                let src = &nuc.sources[*s];
                let own = ReplicaNuclide::at(&src.own, at);
                src.add_change(
                    e,
                    own,
                    nuc.density * ReplicaNuclide::at(xs, at),
                    &mut change,
                    &mut change_grad,
                );
            }
        }
        if denominator > 0.0 {
            for (o, c) in out.iter_mut().zip(change) {
                *o += c / denominator;
            }
            for (o, c) in grad.iter_mut().zip(change_grad) {
                *o = c / denominator;
            }
        }
    }
}

/// `(1 - e^-x) / x`, the mean of `e^-s` over `[0, x]`, stable near zero.
pub(crate) fn mean_survival(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0 - 0.5 * x
    } else {
        -(-x).exp_m1() / x
    }
}

/// One flight segment's replica state: what it scores with and what it does
/// to the particle's ratios and derivatives.
pub(crate) struct ReplicaSegment<'a> {
    ctx: &'a ReplicaContext,
    slot: Option<u32>,
    /// The per-collision base seed the segment's probability-table bands are
    /// drawn with.
    urr_random: Option<f64>,
    /// `r_k φ(ΔΣ_k d)`: the ratio integrated along the segment, per unit
    /// length, which a track-length score is multiplied by.
    factors: Vec<f64>,
    /// `exp(-ΔΣ_k d)`: the survival ratio to the segment's end.
    survival: Vec<f64>,
    /// Per mode, the derivative of a track-length score's logarithm along
    /// the segment before its own cross section: the particle's
    /// `S_k = ∂ ln w / ∂ξ_k` less `d/2` times `ΔΣ`'s gradient, `φ'(0) = -1/2`.
    slope: Vec<f64>,
    /// `ΔΣ`'s gradient along the modes, in 1/cm, and the segment's length.
    grad: Vec<f64>,
    dist: f64,
}

impl<'a> ReplicaSegment<'a> {
    /// The segment a particle at its current energy flies for `dist` in the
    /// material of `slot`, or `None` on a particle without replica ratios.
    pub(crate) fn new(
        ctx: &'a ReplicaContext,
        particle: &yamc_particle::particle::Particle,
        slot: Option<u32>,
        dist: f64,
    ) -> Option<Self> {
        let state = particle.replica.as_ref()?;
        let (ratios, sensitivity) = state.split_at(ctx.replicas);
        let urr_random = yamc_particle::particle::urr_to_option(particle.urr_random);
        let mut delta = vec![0.0; ratios.len()];
        let mut grad = vec![0.0; sensitivity.len()];
        ctx.flight_delta(slot, particle.energy, urr_random, &mut delta, &mut grad);
        let factors = ratios
            .iter()
            .zip(&delta)
            .map(|(r, d)| r * mean_survival(d * dist))
            .collect();
        let survival = delta.iter().map(|d| (-d * dist).exp()).collect();
        let slope = sensitivity
            .iter()
            .zip(&grad)
            .map(|(s, g)| s - 0.5 * dist * g)
            .collect();
        Some(Self {
            ctx,
            slot,
            urr_random,
            factors,
            survival,
            slope,
            grad,
            dist,
        })
    }

    /// Replica factors for one score at energy `e`, then the score's
    /// derivative factors along the modes: a contribution `v` adds `v` times
    /// each to the replicas' and the derivatives' sums.
    pub(crate) fn factors_for(&self, score: &Score, e: f64) -> Vec<f64> {
        let mut ratio = vec![1.0; self.factors.len()];
        let mut grad = vec![0.0; self.slope.len()];
        self.ctx
            .score_ratio(self.slot, score, e, self.urr_random, &mut ratio, &mut grad);
        ratio
            .iter()
            .zip(&self.factors)
            .map(|(a, b)| a * b)
            .chain(self.slope.iter().zip(&grad).map(|(s, g)| s + g))
            .collect()
    }

    /// Carry the particle's ratios and derivatives to the segment's end.
    pub(crate) fn apply(&self, particle: &mut yamc_particle::particle::Particle) {
        if let Some(state) = particle.replica.as_mut() {
            let (ratios, sensitivity) = state.split_at_mut(self.ctx.replicas);
            for (x, s) in ratios.iter_mut().zip(&self.survival) {
                *x *= s;
            }
            for (x, g) in sensitivity.iter_mut().zip(&self.grad) {
                *x -= self.dist * g;
            }
        }
    }
}

/// Multiply a particle's ratios by the factor of a collision ending in
/// reaction `mt` on `nuclide`, at the particle's current (incident) energy,
/// and add the factor's gradient to its derivatives. Call it after the
/// reaction is chosen and before any secondary is banked, so secondaries
/// inherit it.
pub(crate) fn apply_collision(
    ctx: Option<&ReplicaContext>,
    particle: &mut yamc_particle::particle::Particle,
    slot: Option<u32>,
    nuclide: &str,
    mt: i32,
) {
    let Some(ctx) = ctx else {
        return;
    };
    let energy = particle.energy;
    let Some(state) = particle.replica.as_mut() else {
        return;
    };
    let (ratios, sensitivity) = state.split_at_mut(ctx.replicas);
    let mut factor = vec![1.0; ratios.len()];
    let mut grad = vec![0.0; sensitivity.len()];
    ctx.collision_factor(slot, nuclide, mt, energy, &mut factor, &mut grad);
    for (x, f) in ratios.iter_mut().zip(factor) {
        *x *= f;
    }
    for (x, g) in sensitivity.iter_mut().zip(grad) {
        *x += g;
    }
}
