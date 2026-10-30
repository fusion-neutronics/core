//! Shutdown dose rate over a reactor that changes along its schedule.
//!
//! The single-model D1S run (`use_decay_photons`) activates and transports in
//! one geometry, so it cannot describe a machine that is DD-irradiated in one
//! configuration, DT-irradiated in another, then opened, has components
//! removed or moved and is finally dismantled. This module splits each D1S run
//! in two:
//!
//! 1. **Capture.** For every model a pulse irradiates in, the neutrons are
//!    transported with decay photons on, but each decay photon is recorded at
//!    birth (position, energy, weight, parent radionuclide, birth cell) instead
//!    of being transported.
//! 2. **Replay.** For every (irradiation stage, observation stage) pair, those
//!    births are moved with their cell, dropped if their cell is absent, and
//!    emitted as the photon source of the observation stage's model, with the
//!    user's tallies binned by parent radionuclide.
//!
//! A replay tally is per source neutron of its irradiation stage, exactly what
//! a single-model D1S tally is, so the existing per-radionuclide time
//! correction applies unchanged: the dose at the end of step `s` is the sum
//! over irradiation stages `c` of `TCF_c(s) . R(c, stage(s))`, where `TCF_c`
//! uses only the pulse rates of stage `c`. The replay runs once per stage
//! pair, not once per observation time.
//!
//! Cells are matched across models by name. A stage's `placements` say where a
//! named cell sits in that stage's model relative to its reference position,
//! as a rigid transform; a birth made under placement `T_c` is moved by
//! `T_o . T_c^-1` into the observation stage. Geometry is not moved for the
//! user: the observation model must contain the cell at its new position, and
//! every moved birth is checked to land in the cell of the same name.
//!
//! Standard deviations are the photon transport's, conditional on the captured
//! births. Replays that share a capture are correlated through it, and that
//! correlation is not propagated.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use rand::RngExt;

use yamc_nuclide::config::ChainUse;
use yamc_nuclide::nuclide_registry::NuclideId;
use yamc_particle::particle::{Particle, ParticleType};
use yamc_tallies::filter::Filter;
use yamc_tallies::{ParentNuclideFilter, Tally};

use crate::model::{Model, TransportSettings};

/// A rigid placement: a rotation then a translation, stored as the top three
/// rows of a homogeneous 4x4 matrix (cm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement([[f64; 4]; 3]);

impl Default for Placement {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Placement {
    /// No rotation, no translation.
    pub const IDENTITY: Self = Self([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]);

    /// A pure translation by `t` (cm).
    pub fn translation(t: [f64; 3]) -> Result<Self, String> {
        let mut m = Self::IDENTITY.0;
        for (row, v) in m.iter_mut().zip(t) {
            row[3] = v;
        }
        Self::new([m[0], m[1], m[2], [0.0, 0.0, 0.0, 1.0]])
    }

    /// From a homogeneous 4x4 matrix. Refuses anything that is not a proper
    /// rigid motion (finite, last row `[0, 0, 0, 1]`, orthonormal rotation of
    /// determinant +1): a scale, shear or mirror would change a cell's shape,
    /// which a moved component does not do.
    pub fn new(m: [[f64; 4]; 4]) -> Result<Self, String> {
        const TOL: f64 = 1e-9;
        if m.iter().flatten().any(|v| !v.is_finite()) {
            return Err("placement matrix entries must be finite".into());
        }
        if m[3] != [0.0, 0.0, 0.0, 1.0] {
            return Err(format!(
                "placement matrix last row must be [0, 0, 0, 1], got {:?}",
                m[3]
            ));
        }
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[i][k] * m[j][k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                if (dot - want).abs() > TOL {
                    return Err(
                        "placement rotation must be orthonormal: a move may rotate and \
                         translate a cell, not scale or shear it"
                            .into(),
                    );
                }
            }
        }
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (det - 1.0).abs() > TOL {
            return Err("placement rotation must have determinant +1, not be a mirror".into());
        }
        Ok(Self([m[0], m[1], m[2]]))
    }

    /// The placed position of reference point `p`.
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| {
            self.0[i][0] * p[0] + self.0[i][1] * p[1] + self.0[i][2] * p[2] + self.0[i][3]
        })
    }

    /// The inverse motion: rotation transposed, translation `-R^T t`.
    pub fn inverse(&self) -> Self {
        let r = |i: usize, j: usize| self.0[j][i];
        Self(std::array::from_fn(|i| {
            let t = -(0..3).map(|k| r(i, k) * self.0[k][3]).sum::<f64>();
            [r(i, 0), r(i, 1), r(i, 2), t]
        }))
    }

    /// `self` applied after `first`.
    pub fn after(&self, first: &Self) -> Self {
        Self(std::array::from_fn(|i| {
            let mut row = [0.0; 4];
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| self.0[i][k] * first.0[k][j]).sum::<f64>();
            }
            row[3] += self.0[i][3];
            row
        }))
    }
}

/// One configuration of the machine: a model and where its moved cells sit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stage {
    /// Index into the `models` slice.
    pub model: usize,
    /// Cell name -> placement relative to the cell's reference position. A
    /// cell not listed is at its reference position.
    pub placements: BTreeMap<String, Placement>,
}

/// One schedule step: a source `rate` (n/s, 0 for cooling) held for
/// `duration` (s) in stage `stage`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    pub rate: f64,
    pub duration: f64,
    pub stage: usize,
}

/// The time-corrected dose of one tally, one row per schedule step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DoseTally {
    pub mean: Vec<Vec<f64>>,
    pub std_dev: Vec<Vec<f64>>,
    /// Radionuclide -> its share of `mean`.
    pub by_nuclide: BTreeMap<String, Vec<Vec<f64>>>,
}

/// What [`simulate_shutdown_dose`] returns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShutdownDose {
    /// Elapsed time (s) at the end of each step.
    pub times: Vec<f64>,
    /// One per input tally, in order.
    pub tallies: Vec<DoseTally>,
    /// The parent radionuclides the tallies were binned by.
    pub radionuclides: Vec<String>,
}

/// A tally's flat mean and standard deviation.
type MeanStd = (Vec<f64>, Vec<f64>);

/// One decay photon recorded at birth by a capture run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Birth {
    pub position: [f64; 3],
    pub energy: f64,
    pub weight: f64,
    /// Index into [`Captured::parents`].
    pub parent: u16,
    /// Birth cell index in the capture model's geometry.
    pub cell: u32,
}

/// A raw birth as the transport loop records it, before the run's nuclide ids
/// are turned into names.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawBirth {
    pub position: [f64; 3],
    pub energy: f64,
    pub weight: f64,
    pub parent: NuclideId,
    pub cell: u32,
}

/// Every decay photon a capture run made, per source neutron history.
#[derive(Debug, Clone, Default)]
pub(crate) struct Captured {
    pub births: Vec<Birth>,
    pub parents: Vec<String>,
    pub histories: u64,
}

impl Captured {
    /// Name the raw births' parents from the run's registry.
    pub(crate) fn from_raw(
        raw: Vec<RawBirth>,
        registry: &yamc_nuclide::nuclide_registry::NuclideRegistry,
        histories: u64,
    ) -> Self {
        let mut index: HashMap<NuclideId, u16> = HashMap::new();
        let mut parents = Vec::new();
        let mut births = Vec::with_capacity(raw.len());
        for b in raw {
            let next = index.len();
            let parent = *index.entry(b.parent).or_insert_with(|| {
                parents.push(registry.name(b.parent).to_string());
                next as u16
            });
            births.push(Birth {
                position: b.position,
                energy: b.energy,
                weight: b.weight,
                parent,
                cell: b.cell,
            });
        }
        Self {
            births,
            parents,
            histories,
        }
    }
}

/// The photon source of a replay run: births drawn in proportion to their
/// weight, each launched isotropically with the same weight `weight`, which
/// is the births' total weight per source neutron of the capture. A replay
/// tally is therefore per source neutron, as a single-model D1S tally is.
#[derive(Debug, Clone)]
pub(crate) struct ReplaySource {
    births: Vec<Birth>,
    cumulative: Vec<f64>,
    pub(crate) parents: Vec<String>,
    weight: f64,
}

/// What `Model::run_staged` does with decay photons, beyond a normal run.
pub(crate) enum StagedRun<'a> {
    /// Record each decay photon at birth into the vec instead of tracking it.
    Capture(&'a mut Option<Captured>),
    /// Draw the source particles from these births instead of `sources`.
    Replay(&'a ReplaySource),
}

impl ReplaySource {
    fn new(births: Vec<Birth>, parents: Vec<String>, histories: u64) -> Self {
        let mut running = 0.0;
        let cumulative = births
            .iter()
            .map(|b| {
                running += b.weight;
                running
            })
            .collect();
        Self {
            births,
            cumulative,
            parents,
            weight: running / histories.max(1) as f64,
        }
    }

    fn total(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }

    /// One replay photon; `ids` maps `parents` to the run's nuclide ids.
    pub(crate) fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R, ids: &[NuclideId]) -> Particle {
        let threshold = rng.random::<f64>() * self.total();
        let i = self
            .cumulative
            .partition_point(|&c| c <= threshold)
            .min(self.births.len() - 1);
        let b = &self.births[i];
        let mu = 2.0 * rng.random::<f64>() - 1.0;
        let phi = std::f64::consts::TAU * rng.random::<f64>();
        let s = (1.0 - mu * mu).max(0.0).sqrt();
        let mut p = Particle::new(b.position, [s * phi.cos(), s * phi.sin(), mu], b.energy);
        p.particle_type = ParticleType::Photon;
        p.weight = self.weight;
        p.parent_nuclide = Some(ids[b.parent as usize]);
        p
    }
}

/// Cell name -> index, refusing a name used twice (it could not say which
/// cell a birth belongs to).
fn cell_names(model: &Model) -> Result<HashMap<&str, usize>, String> {
    let mut out = HashMap::new();
    for (i, cell) in model.geometry.cells().iter().enumerate() {
        if let Some(name) = cell.name.as_deref()
            && out.insert(name, i).is_some()
        {
            return Err(format!(
                "two cells are named '{name}'; shutdown dose stages match cells across \
                 models by name, so names must be unique"
            ));
        }
    }
    Ok(out)
}

/// The births of capture `cap` (made in stage `from`) as they sit in stage
/// `to`: dropped where their cell is not in `to`'s model, moved with their
/// cell otherwise, and checked to land in it.
fn transfer(
    cap: &Captured,
    models: &[Model],
    from: &Stage,
    to: &Stage,
) -> Result<Vec<Birth>, String> {
    let (src, dst) = (&models[from.model], &models[to.model]);
    let unmoved = from.model == to.model && from.placements == to.placements;
    if unmoved {
        return Ok(cap.births.clone());
    }
    let dst_names = cell_names(dst)?;
    let src_cells = src.geometry.cells();
    // Per source cell: (destination cell index, motion), or None if removed.
    let mut plan: HashMap<u32, Option<(usize, Placement)>> = HashMap::new();
    let mut out = Vec::with_capacity(cap.births.len());
    for b in &cap.births {
        let step = match plan.get(&b.cell) {
            Some(s) => *s,
            None => {
                let cell = &src_cells[b.cell as usize];
                let name = cell.name.as_deref().ok_or_else(|| {
                    format!(
                        "decay photons are born in an unnamed cell (id {}); name every \
                         activated cell so later stages can find it in their models",
                        cell.cell_id.map_or("None".into(), |i| i.to_string())
                    )
                })?;
                let s = dst_names.get(name).map(|&i| {
                    let at = |st: &Stage| st.placements.get(name).copied().unwrap_or_default();
                    (i, at(to).after(&at(from).inverse()))
                });
                plan.insert(b.cell, s);
                s
            }
        };
        let Some((dst_cell, motion)) = step else {
            continue;
        };
        let position = motion.apply(b.position);
        let landed = dst
            .geometry
            .find_cell_index((position[0], position[1], position[2]));
        if landed != Some(dst_cell) {
            let name = src_cells[b.cell as usize]
                .name
                .as_deref()
                .unwrap_or_default();
            return Err(format!(
                "activated cell '{name}' does not line up with its placement: a decay \
                 photon born in it at {:?} moves to {position:?}, which is outside cell \
                 '{name}' in the later model. Build that model's '{name}' at the position \
                 its `moves` describe",
                b.position
            ));
        }
        out.push(Birth {
            position,
            cell: dst_cell as u32,
            ..*b
        });
    }
    Ok(out)
}

/// `tally` with a parent-nuclide filter over `parents` appended (the bin
/// layout `Tally(parent_nuclides=...)` builds), fresh of any accumulation.
fn binned(tally: &Tally, parents: &[String]) -> Result<Arc<Tally>, String> {
    if tally.get_parent_nuclide_filter().is_some() {
        return Err(format!(
            "tally '{}' already has parent_nuclides; leave them out, the shutdown dose \
             bins by every radionuclide the irradiations make",
            tally.name.as_deref().unwrap_or("unnamed")
        ));
    }
    let mut config = tally.to_serde();
    config
        .filters
        .push(Filter::ParentNuclide(ParentNuclideFilter::new(
            parents.to_vec(),
        )));
    let t = Tally::from(config);
    t.validate()?;
    Ok(Arc::new(t))
}

/// Simulate the shutdown dose rate over `steps`, whose configurations are
/// `stages` over `models`.
///
/// Every step is observed at its end, in its own stage. A model needs a
/// neutron source only if a pulse irradiates in it; tallies are attached to
/// every observation stage's model, binned by parent radionuclide here. The
/// `models` are not modified. See the module docs for the method.
pub fn simulate_shutdown_dose(
    models: &[Model],
    stages: &[Stage],
    steps: &[Step],
    tallies: &[Arc<Tally>],
    settings: &TransportSettings,
) -> Result<ShutdownDose, String> {
    if steps.is_empty() {
        return Err("the schedule has no steps".into());
    }
    if tallies.is_empty() {
        return Err("give at least one tally to score the dose in".into());
    }
    for t in tallies {
        // Checked before any transport rather than when the tallies are binned.
        binned(t, &[])?;
    }
    for (i, s) in steps.iter().enumerate() {
        let stage = stages
            .get(s.stage)
            .ok_or_else(|| format!("step {i} names stage {}, which does not exist", s.stage))?;
        if stage.model >= models.len() {
            return Err(format!(
                "stage {} names model {}, which does not exist",
                s.stage, stage.model
            ));
        }
        if !(s.rate.is_finite() && s.rate >= 0.0 && s.duration.is_finite() && s.duration >= 0.0) {
            return Err(format!(
                "step {i} needs a finite, non-negative rate and duration"
            ));
        }
    }
    // Irradiation stages (a pulse with a rate) and observation stages (all).
    let campaigns: BTreeSet<usize> = steps
        .iter()
        .filter(|s| s.rate > 0.0)
        .map(|s| s.stage)
        .collect();
    let observed: BTreeSet<usize> = steps.iter().map(|s| s.stage).collect();
    if campaigns.is_empty() {
        return Err("the schedule irradiates nothing: no pulse has a positive rate".into());
    }

    // 1. Capture once per irradiated model.
    let mut captures: BTreeMap<usize, Captured> = BTreeMap::new();
    for &c in &campaigns {
        let m = stages[c].model;
        if captures.contains_key(&m) {
            continue;
        }
        let mut model = models[m].clone();
        if !model
            .sources
            .iter()
            .any(|s| s.particle_type() == ParticleType::Neutron)
        {
            return Err(format!(
                "a pulse irradiates in model {m}, which has no neutron source"
            ));
        }
        model.use_decay_photons = true;
        model.transport_secondary_photons = true;
        model.tallies = Vec::new();
        model.convergence_targets = Vec::new();
        let mut captured = None;
        model.run_staged(settings, StagedRun::Capture(&mut captured))?;
        captures.insert(m, captured.ok_or("capture run recorded nothing")?);
    }

    // Every radionuclide any irradiation makes, sorted: the tallies' parent bins.
    let parents: Vec<String> = captures
        .values()
        .flat_map(|c| c.parents.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut times = Vec::with_capacity(steps.len());
    let mut elapsed = 0.0;
    for s in steps {
        elapsed += s.duration;
        times.push(elapsed);
    }
    if parents.is_empty() {
        // Nothing the irradiations make emits a decay photon: zero dose.
        let tallies = tallies
            .iter()
            .map(|t| {
                let w = binned(t, &[])?.num_bins();
                Ok(DoseTally {
                    mean: vec![vec![0.0; w]; steps.len()],
                    std_dev: vec![vec![0.0; w]; steps.len()],
                    by_nuclide: BTreeMap::new(),
                })
            })
            .collect::<Result<_, String>>()?;
        return Ok(ShutdownDose {
            times,
            tallies,
            radionuclides: parents,
        });
    }

    // 2. Replay each (campaign, observation) pair, per source neutron.
    let n_scores: Vec<usize> = tallies.iter().map(|t| t.scores.len()).collect();
    // (campaign, observation) -> per tally (mean, std_dev).
    let mut replays: BTreeMap<(usize, usize), Vec<MeanStd>> = BTreeMap::new();
    for &c in &campaigns {
        let cap = &captures[&stages[c].model];
        for &o in &observed {
            let births = transfer(cap, models, &stages[c], &stages[o])?;
            let fresh: Vec<Arc<Tally>> = tallies
                .iter()
                .map(|t| binned(t, &parents))
                .collect::<Result<_, _>>()?;
            let src = ReplaySource::new(births, cap.parents.clone(), cap.histories);
            let rows = if src.total() > 0.0 {
                let mut model = models[stages[o].model].clone();
                model.sources = Vec::new();
                model.use_decay_photons = false;
                model.transport_secondary_photons = true;
                model.tallies = fresh.clone();
                model.convergence_targets = Vec::new();
                model.run_staged(settings, StagedRun::Replay(&src))?;
                fresh
                    .iter()
                    .map(|t| {
                        let r = t.finalize();
                        (r.mean, r.standard_deviation)
                    })
                    .collect()
            } else {
                // Nothing reaches this stage (all its emitters were removed).
                fresh
                    .iter()
                    .map(|t| (vec![0.0; t.num_bins()], vec![0.0; t.num_bins()]))
                    .collect()
            };
            replays.insert((c, o), rows);
        }
    }

    // 3. Time-correct each pair and sum into the steps it observes.
    let chain = yani_transmute::load_configured_chain(ChainUse::Activation)
        .map_err(|e| format!("Failed to load transmutation chain for decay photons: {e}"))?
        .chain;
    let timesteps: Vec<f64> = steps.iter().map(|s| s.duration).collect();
    let width = |k: usize| replays.values().next().map_or(0, |r| r[k].0.len());
    let mut out: Vec<DoseTally> = (0..tallies.len())
        .map(|k| {
            let w = width(k) / parents.len();
            DoseTally {
                mean: vec![vec![0.0; w]; steps.len()],
                std_dev: vec![vec![0.0; w]; steps.len()],
                by_nuclide: BTreeMap::new(),
            }
        })
        .collect();
    for &c in &campaigns {
        let rates: Vec<f64> = steps
            .iter()
            .map(|s| if s.stage == c { s.rate } else { 0.0 })
            .collect();
        let tcf = yani_decay::time_correction_factors(&parents, &timesteps, &rates, &chain)?;
        for &o in &observed {
            let rows: Vec<usize> = (0..steps.len()).filter(|&i| steps[i].stage == o).collect();
            let idx: Vec<usize> = rows.iter().map(|i| i + 1).collect();
            for (k, (mean, std)) in replays[&(c, o)].iter().enumerate() {
                let (sum_m, sum_s) = yani_decay::apply_time_correction(
                    mean,
                    std,
                    &parents,
                    n_scores[k],
                    &tcf,
                    &idx,
                    true,
                )?;
                let (per_m, _) = yani_decay::apply_time_correction(
                    mean,
                    std,
                    &parents,
                    n_scores[k],
                    &tcf,
                    &idx,
                    false,
                )?;
                let d = &mut out[k];
                for (j, &r) in rows.iter().enumerate() {
                    for (b, (&m, &s)) in sum_m[j].iter().zip(&sum_s[j]).enumerate() {
                        d.mean[r][b] += m;
                        // Independent replays: variances add.
                        d.std_dev[r][b] += s * s;
                    }
                    let stride = per_m[j].len() / parents.len();
                    for (p, name) in parents.iter().enumerate() {
                        let entry = d
                            .by_nuclide
                            .entry(name.clone())
                            .or_insert_with(|| vec![vec![0.0; stride]; steps.len()]);
                        for (b, &v) in per_m[j][p * stride..(p + 1) * stride].iter().enumerate() {
                            entry[r][b] += v;
                        }
                    }
                }
            }
        }
    }
    for d in &mut out {
        d.std_dev.iter_mut().flatten().for_each(|v| *v = v.sqrt());
    }
    Ok(ShutdownDose {
        times,
        tallies: out,
        radionuclides: parents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    #[test]
    fn translation_moves_and_inverse_undoes() {
        let t = Placement::translation([1.0, -2.0, 3.0]).unwrap();
        assert!(close(t.apply([0.0, 0.0, 0.0]), [1.0, -2.0, 3.0]));
        assert!(close(
            t.inverse().apply(t.apply([4.0, 5.0, 6.0])),
            [4.0, 5.0, 6.0]
        ));
    }

    #[test]
    fn rotation_composes_with_translation() {
        // 90 degrees about z, then +10 in x.
        let r = Placement::new([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(close(r.apply([1.0, 0.0, 0.0]), [10.0, 1.0, 0.0]));
        assert!(close(r.inverse().apply([10.0, 1.0, 0.0]), [1.0, 0.0, 0.0]));
        // Moving from placement r to placement t is t . r^-1.
        let t = Placement::translation([0.0, 0.0, 5.0]).unwrap();
        assert!(close(
            t.after(&r.inverse()).apply([10.0, 1.0, 0.0]),
            [1.0, 0.0, 5.0]
        ));
        assert_eq!(
            Placement::IDENTITY.after(&Placement::IDENTITY),
            Placement::IDENTITY
        );
    }

    #[test]
    fn non_rigid_matrices_are_refused() {
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(Placement::new(scale).unwrap_err().contains("orthonormal"));
        let mut mirror = scale;
        mirror[0][0] = -1.0;
        assert!(Placement::new(mirror).unwrap_err().contains("determinant"));
        let mut bad_row = Placement::IDENTITY.0.to_vec();
        bad_row.push([0.0, 0.0, 1.0, 1.0]);
        assert!(Placement::new(bad_row.try_into().unwrap())
            .unwrap_err()
            .contains("last row"));
        assert!(Placement::translation([f64::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn replay_weight_is_total_per_history_and_sampling_follows_weight() {
        use rand::SeedableRng;
        let birth = |x: f64, w: f64| Birth {
            position: [x, 0.0, 0.0],
            energy: 1e6,
            weight: w,
            parent: 0,
            cell: 0,
        };
        let src = ReplaySource::new(
            vec![birth(0.0, 1.0), birth(1.0, 3.0)],
            vec!["Mn56".into()],
            8,
        );
        assert_eq!(src.weight, 0.5);
        let mut reg = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let ids = [reg.intern("Mn56")];
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let n = 20_000;
        let mut ones = 0;
        for _ in 0..n {
            let p = src.sample(&mut rng, &ids);
            assert_eq!(p.particle_type, ParticleType::Photon);
            assert_eq!(p.parent_nuclide, Some(ids[0]));
            assert!((p.direction.iter().map(|d| d * d).sum::<f64>() - 1.0).abs() < 1e-12);
            ones += (p.position[0] == 1.0) as usize;
        }
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.75).abs() < 0.02, "{frac}");
    }
}
