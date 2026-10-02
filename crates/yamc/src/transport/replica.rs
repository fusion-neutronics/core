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
//! of a sum taking the sum's. Rows with no covariance, heating and
//! production rows, and the short-range noise stay nominal.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yani_transmute::covariance_fold::{transport_fields, Read, TransportField};
use yani_transmute::covariance_sample::Sampler;

/// One covariance source of one nuclide: the reaction whose cells some of
/// its partials read, with what those partials add up to.
struct Source {
    /// `Σ σ_mt(E)` over the partials reading this source, in barns, on the
    /// nuclide's grid.
    sum: Vec<f64>,
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
}

impl Source {
    /// The cell of `cells` holding `e`, the last closed at its top.
    fn cell(cells: &[(f64, f64)], e: f64) -> Option<usize> {
        let i = cells.partition_point(|c| c.0 <= e).checked_sub(1)?;
        (e < cells[i].1 || (i + 1 == cells.len() && e == cells[i].1)).then_some(i)
    }

    /// Replica `k`'s relative change of a partial reading this source at
    /// `e`, `σ'/σ - 1`, added into `out` times `scale`. `own` is `σ_src(e)`.
    fn add_relative_change(&self, e: f64, own: f64, scale: f64, out: &mut [f64]) {
        let r = out.len();
        if let Some(c) = Self::cell(&self.relative, e) {
            for (o, dm) in out.iter_mut().zip(&self.relative_dm[c * r..(c + 1) * r]) {
                *o += scale * dm;
            }
        }
        if own > 0.0 {
            if let Some(c) = Self::cell(&self.absolute, e) {
                for (o, a) in out.iter_mut().zip(&self.absolute_shift[c * r..(c + 1) * r]) {
                    *o += scale * a / own;
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
                _ => None,
            })
            .collect();
        let mut out = Vec::with_capacity(materials.len());
        for material in materials {
            out.push(Self::material(material, &score_mts, replicas, seed)?);
        }
        Ok(ReplicaContext {
            replicas,
            materials: out,
        })
    }

    fn material(
        material: &Material,
        score_mts: &BTreeSet<i32>,
        replicas: usize,
        seed: u64,
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
            let Some(tab) = Self::nuclide(
                nuclide,
                material.temperature(),
                transport,
                &relative,
                &absolute,
                score_mts,
                densities.get(name).copied().unwrap_or(0.0),
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
        nuclide: &Nuclide,
        temperature: &str,
        transport: Option<&TransportField>,
        relative: &[&[f64]],
        absolute: &[&[f64]],
        score_mts: &BTreeSet<i32>,
        density: f64,
    ) -> Result<Option<ReplicaNuclide>, String> {
        let t = nuclide
            .loaded_temperatures
            .iter()
            .position(|l| l == temperature)
            .or_else(|| (nuclide.loaded_temperatures.len() == 1).then_some(0));
        let Some(t) = t else {
            return Err(format!(
                "{} is not loaded at {temperature} K for the replica weights",
                nuclide.name.as_deref().unwrap_or("a nuclide")
            ));
        };
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
                let cells = |list: &[yani_transmute::covariance_fold::Cell],
                             values: &[&[f64]]|
                 -> (Vec<(f64, f64)>, Vec<f64>) {
                    let picked: Vec<usize> = list
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.mt == src)
                        .map(|(k, _)| k)
                        .collect();
                    let bounds = picked.iter().map(|&k| (list[k].lo, list[k].hi)).collect();
                    let mut per = Vec::with_capacity(picked.len() * r);
                    for &k in &picked {
                        for v in values {
                            per.push(v.get(k).copied().unwrap_or(0.0));
                        }
                    }
                    (bounds, per)
                };
                let (rel, rel_dm) = cells(&field.relative_cells, relative);
                let (abs, abs_shift) = cells(&field.absolute_cells, absolute);
                let own = held
                    .get(&src)
                    .map_or_else(|| vec![0.0; n], |r| on_grid(r, n));
                sources.push(Source {
                    sum: vec![0.0; n],
                    own,
                    relative: rel,
                    relative_dm: rel_dm,
                    absolute: abs,
                    absolute_shift: abs_shift,
                });
                sources.len() - 1
            });
            for (o, v) in sources[s].sum.iter_mut().zip(on_grid(reaction, n)) {
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

        Ok(Some(ReplicaNuclide {
            density,
            grid,
            sources,
            source_of,
            fission,
            scores,
        }))
    }

    fn material_at(&self, slot: Option<u32>) -> Option<&ReplicaMaterial> {
        self.materials.get(slot? as usize)?.as_ref()
    }

    /// `ΔΣ_k(E) = Σ'_t,k - Σ_t` in 1/cm for each replica, into `out`.
    pub(crate) fn flight_delta(&self, slot: Option<u32>, e: f64, out: &mut [f64]) {
        out.fill(0.0);
        let Some(m) = self.material_at(slot) else {
            return;
        };
        for nuc in &m.nuclides {
            let Some(at) = nuc.locate(e) else {
                continue;
            };
            for s in &nuc.sources {
                let sum = ReplicaNuclide::at(&s.sum, at);
                if sum == 0.0 {
                    continue;
                }
                let own = ReplicaNuclide::at(&s.own, at);
                s.add_relative_change(e, own, nuc.density * sum, out);
            }
        }
    }

    /// The factor `σ'_mt,k / σ_mt` a collision ending in reaction `mt` on
    /// `nuclide` puts on each replica's ratio, into `out`; all ones where
    /// nothing perturbs it.
    pub(crate) fn collision_factor(
        &self,
        slot: Option<u32>,
        nuclide: &str,
        mt: i32,
        e: f64,
        out: &mut [f64],
    ) {
        out.fill(1.0);
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
                    src.add_relative_change(e, own, w, out);
                }
            }
            return;
        }
        if let Some(&s) = nuc.source_of.get(&mt) {
            let src = &nuc.sources[s];
            let own = ReplicaNuclide::at(&src.own, at);
            src.add_relative_change(e, own, 1.0, out);
        }
    }

    /// `Σ'_x,k / Σ_x` for a reaction-rate score of `mt` at `e` in the
    /// material, into `out`; all ones for flux, for a score nothing perturbs,
    /// and for heating, production and damage scores, which stay nominal.
    pub(crate) fn score_ratio(&self, slot: Option<u32>, score: &Score, e: f64, out: &mut [f64]) {
        out.fill(1.0);
        let Score::ReactionRate(rr) = score else {
            return;
        };
        let x = rr.mt.as_i32();
        let Some(m) = self.material_at(slot) else {
            return;
        };
        let mut denominator = 0.0;
        let mut change = vec![0.0; out.len()];
        for nuc in &m.nuclides {
            let (Some((by_source, nominal)), Some(at)) = (nuc.scores.get(&x), nuc.locate(e)) else {
                continue;
            };
            denominator += nuc.density * ReplicaNuclide::at(nominal, at);
            for (s, xs) in by_source {
                let src = &nuc.sources[*s];
                let own = ReplicaNuclide::at(&src.own, at);
                src.add_relative_change(
                    e,
                    own,
                    nuc.density * ReplicaNuclide::at(xs, at),
                    &mut change,
                );
            }
        }
        if denominator > 0.0 {
            for (o, c) in out.iter_mut().zip(change) {
                *o += c / denominator;
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
/// to the particle's ratios.
pub(crate) struct ReplicaSegment<'a> {
    ctx: &'a ReplicaContext,
    slot: Option<u32>,
    /// `r_k φ(ΔΣ_k d)`: the ratio integrated along the segment, per unit
    /// length, which a track-length score is multiplied by.
    factors: Vec<f64>,
    /// `exp(-ΔΣ_k d)`: the survival ratio to the segment's end.
    survival: Vec<f64>,
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
        let ratios = particle.replica.as_ref()?;
        let mut delta = vec![0.0; ratios.len()];
        ctx.flight_delta(slot, particle.energy, &mut delta);
        let factors = ratios
            .iter()
            .zip(&delta)
            .map(|(r, d)| r * mean_survival(d * dist))
            .collect();
        let survival = delta.iter().map(|d| (-d * dist).exp()).collect();
        Some(Self {
            ctx,
            slot,
            factors,
            survival,
        })
    }

    /// Replica factors for one score at energy `e`.
    pub(crate) fn factors_for(&self, score: &Score, e: f64) -> Vec<f64> {
        let mut ratio = vec![1.0; self.factors.len()];
        self.ctx.score_ratio(self.slot, score, e, &mut ratio);
        ratio
            .iter()
            .zip(&self.factors)
            .map(|(a, b)| a * b)
            .collect()
    }

    /// Carry the particle's ratios to the segment's end.
    pub(crate) fn apply(&self, particle: &mut yamc_particle::particle::Particle) {
        if let Some(r) = particle.replica.as_mut() {
            for (x, s) in r.iter_mut().zip(&self.survival) {
                *x *= s;
            }
        }
    }
}

/// Multiply a particle's ratios by the factor of a collision ending in
/// reaction `mt` on `nuclide`, at the particle's current (incident) energy.
/// Call it after the reaction is chosen and before any secondary is banked,
/// so secondaries inherit it.
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
    let Some(r) = particle.replica.as_mut() else {
        return;
    };
    let mut factor = vec![1.0; r.len()];
    ctx.collision_factor(slot, nuclide, mt, energy, &mut factor);
    for (x, f) in r.iter_mut().zip(factor) {
        *x *= f;
    }
}
