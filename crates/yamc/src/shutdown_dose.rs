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
//!    birth (source history, position, direction, energy, weight, parent
//!    radionuclide, birth cell) instead of being transported.
//! 2. **Replay.** For every observation stage, replay history `i` transports
//!    every decay photon that neutron history `i` made in every irradiated
//!    model, once each and at its own weight, moved with its cell (or dropped
//!    if its cell is gone).
//!
//! The estimator is exact, statistics included. The dose at the end of step `s`
//! is `D(s) = sum_c sum_p TCF_{c,p}(s) . x_{c,p}`, over the irradiation stages
//! `c` (whose time correction factors come from their own pulses) and parent
//! radionuclides `p`, where `x_{c,p}` is the tally of the photons stage `c`'s
//! capture made with parent `p`. Each replay history's own `D_i(s)` is formed
//! at the end of the history (a [`BinFold`] on the per-history scores), and the
//! statistics are those of `D_i(s)` over the histories. The histories are
//! independent and identically distributed, so the standard error is that of
//! the whole calculation: the neutron transport's, the photon transport's,
//! the correlation between parent nuclides scored by one history and between
//! irradiation stages sharing a capture, and between captures of different
//! models whose history `i` shares its random numbers. For a machine that never
//! changes this is the single-model D1S estimator, so the two agree in mean and
//! in standard deviation. Nothing is resampled, so there is nothing to tune.
//!
//! The replay draws its random numbers from a stream independent of the
//! capture's: the photons of history `i` were made from the capture's history
//! `i` stream, and transporting them on it again would correlate the photon
//! walk with the neutron history that made it, which biases the result.
//!
//! Cells are matched across models by name. A stage's `placements` say where a
//! named cell sits in that stage's model relative to its reference position,
//! as a rigid transform; a birth made under placement `T_c` is moved by
//! `T_o . T_c^-1` into the observation stage (its direction by the rotation).
//! Geometry is not moved for the user: the observation model must contain the
//! cell at its new position, and every moved birth is checked to land in the
//! cell of the same name.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_nuclide::config::ChainUse;
use yamc_nuclide::nuclide_registry::NuclideId;
use yamc_particle::particle::{Particle, ParticleType};
use yamc_physics::util::bank::ParticleBank;
use yamc_tallies::filter::Filter;
use yamc_tallies::welford::{BinFold, WelfordTallyStats};
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

    /// The placed direction of reference direction `d`: the rotation alone.
    pub fn rotate(&self, d: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| self.0[i][0] * d[0] + self.0[i][1] * d[1] + self.0[i][2] * d[2])
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

/// One decay photon recorded at birth by a capture run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Birth {
    /// Global index of the neutron history that made it.
    pub history: u64,
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    pub weight: f64,
    /// Index into [`Captured::parents`]; in a [`ReplayPlan`], its class.
    pub parent: u16,
    /// Birth cell index in the capture model's geometry.
    pub cell: u32,
}

/// A raw birth as the transport loop records it, before the run's nuclide ids
/// are turned into names.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawBirth {
    pub history: u64,
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    pub weight: f64,
    pub parent: NuclideId,
    pub cell: u32,
}

/// Every decay photon a capture run made, in history order.
#[derive(Debug, Clone, Default)]
pub(crate) struct Captured {
    pub births: Vec<Birth>,
    pub parents: Vec<String>,
    pub histories: u64,
}

impl Captured {
    /// Name the raw births' parents from the run's registry, in history order
    /// (stable, so a history's births keep the order it made them in, and a
    /// seed gives the same result whatever the thread count).
    pub(crate) fn from_raw(
        mut raw: Vec<RawBirth>,
        registry: &yamc_nuclide::nuclide_registry::NuclideRegistry,
        histories: u64,
    ) -> Self {
        raw.sort_by_key(|b| b.history);
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
                history: b.history,
                position: b.position,
                direction: b.direction,
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

/// What a replay transports and how it scores: the photons of every capture
/// as they sit in the observation stage, in history order and tagged with
/// their class (irradiation stage x parent), the class names the tallies' and
/// photons' parent tags use, and each tally's fold onto the steps.
#[derive(Debug, Clone)]
pub(crate) struct ReplayPlan {
    births: Vec<Birth>,
    pub(crate) classes: Vec<String>,
    pub(crate) folds: Vec<Option<Arc<BinFold>>>,
}

impl ReplayPlan {
    /// Put every photon neutron history `history` made into `bank`, each on a
    /// stream of its own derived from the replay history's `seed`. `ids` are
    /// the run's ids of `classes`.
    pub(crate) fn load(&self, history: u64, bank: &mut ParticleBank, seed: u32, ids: &[NuclideId]) {
        let start = self.births.partition_point(|b| b.history < history);
        let end = start + self.births[start..].partition_point(|b| b.history == history);
        for (j, b) in self.births[start..end].iter().enumerate() {
            let mut p = Particle::new(b.position, b.direction, b.energy);
            p.particle_type = ParticleType::Photon;
            p.weight = b.weight;
            p.parent_nuclide = Some(ids[b.parent as usize]);
            bank.add_source_particle(p, yamc_rng::secondary_seed(seed, j as u32));
        }
    }
}

/// A replay's folded statistics and raw sums, per tally.
pub(crate) struct ReplayStats {
    pub per_tally: Vec<WelfordTallyStats>,
    pub raw_sums: Vec<Option<Vec<f64>>>,
}

/// What `Model::run_staged` does with decay photons, beyond a normal run.
pub(crate) enum StagedRun<'a> {
    /// Record each decay photon at birth into `out` instead of tracking it.
    Capture(&'a mut Option<Captured>),
    /// Make each history the photons `plan` holds for it instead of a source
    /// particle, and hand back the folded statistics in `out`.
    Replay {
        plan: &'a ReplayPlan,
        out: &'a mut Option<ReplayStats>,
    },
}

/// The base seed of the replay runs: a stream independent of the capture's.
fn replay_seed(seed: u64) -> u64 {
    yamc_rng::expand_seed(yamc_rng::fold_base_seed(seed ^ 0xD15C_0DE5_0F5D_1D00))
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
    if from.model == to.model && from.placements == to.placements {
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
            direction: motion.rotate(b.direction),
            cell: dst_cell as u32,
            ..*b
        });
    }
    Ok(out)
}

/// `tally` with a parent-nuclide filter over `classes` appended (the bin
/// layout `Tally(parent_nuclides=...)` builds), fresh of any accumulation.
fn binned(tally: &Tally, classes: &[String]) -> Result<Arc<Tally>, String> {
    let name = tally.name.as_deref().unwrap_or("unnamed");
    if tally.get_parent_nuclide_filter().is_some() {
        return Err(format!(
            "tally '{name}' already has parent_nuclides; leave them out, the shutdown dose \
             bins by every radionuclide the irradiations make"
        ));
    }
    if tally.covariance {
        return Err(format!(
            "tally '{name}' asks for its covariance, which the shutdown dose does not report"
        ));
    }
    let mut config = tally.to_serde();
    config
        .filters
        .push(Filter::ParentNuclide(ParentNuclideFilter::new(
            classes.to_vec(),
        )));
    let t = Tally::from(config);
    t.validate()?;
    Ok(Arc::new(t))
}

/// Simulate the shutdown dose rate over `steps`, whose configurations are
/// `stages` over `models`.
///
/// Every step is observed at its end, in its own stage. A model needs a
/// neutron source only if a pulse irradiates in it; tallies are scored in
/// every observation stage's model. The `models` are not modified. The run
/// needs `settings.total_particles`: every capture transports that many
/// neutron histories, and every replay that many photon histories (one per
/// neutron history). See the module docs for the method.
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
    if settings.total_particles.is_none() || settings.max_runtime.is_some() {
        return Err(
            "the shutdown dose needs a fixed total_particles and no max_runtime: each \
             photon history replays the neutron history of the same index"
                .into(),
        );
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
    let campaigns: Vec<usize> = steps
        .iter()
        .filter(|s| s.rate > 0.0)
        .map(|s| s.stage)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let observed: BTreeSet<usize> = steps.iter().map(|s| s.stage).collect();
    if campaigns.is_empty() {
        return Err("the schedule irradiates nothing: no pulse has a positive rate".into());
    }

    // 1. Capture once per irradiated model, all on the same seed: history `i`
    //    of every capture is replayed together, so whatever their shared
    //    random numbers correlate is inside one replay history.
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
    let histories = captures.values().next().map_or(0, |c| c.histories);
    if captures.values().any(|c| c.histories != histories) {
        return Err("the capture runs transported different numbers of histories".into());
    }

    // Every radionuclide any irradiation makes, sorted.
    let parents: Vec<String> = captures
        .values()
        .flat_map(|c| c.parents.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let n_p = parents.len();
    let mut times = Vec::with_capacity(steps.len());
    let mut elapsed = 0.0;
    for s in steps {
        elapsed += s.duration;
        times.push(elapsed);
    }
    let mut out: Vec<DoseTally> = tallies
        .iter()
        .map(|t| {
            let w = binned(t, &[])?.num_bins();
            Ok(DoseTally {
                mean: vec![vec![0.0; w]; steps.len()],
                std_dev: vec![vec![0.0; w]; steps.len()],
                by_nuclide: parents
                    .iter()
                    .map(|p| (p.clone(), vec![vec![0.0; w]; steps.len()]))
                    .collect(),
            })
        })
        .collect::<Result<_, String>>()?;
    if n_p == 0 {
        // Nothing the irradiations make emits a decay photon: zero dose.
        return Ok(ShutdownDose {
            times,
            tallies: out,
            radionuclides: parents,
        });
    }

    // 2. Each irradiation stage's time correction factors, from its own pulses.
    let chain = yani_transmute::load_configured_chain(ChainUse::Activation)
        .map_err(|e| format!("Failed to load transmutation chain for decay photons: {e}"))?
        .chain;
    let timesteps: Vec<f64> = steps.iter().map(|s| s.duration).collect();
    let tcfs: Vec<HashMap<String, Vec<f64>>> = campaigns
        .iter()
        .map(|&c| {
            let rates: Vec<f64> = steps
                .iter()
                .map(|s| if s.stage == c { s.rate } else { 0.0 })
                .collect();
            yani_decay::time_correction_factors(&parents, &timesteps, &rates, &chain)
        })
        .collect::<Result<_, _>>()?;
    // Class k = (irradiation stage ci, parent pi) = ci * n_p + pi.
    let classes: Vec<String> = (0..campaigns.len())
        .flat_map(|ci| parents.iter().map(move |p| format!("{p}@{ci}")))
        .collect();
    let replay_settings = TransportSettings {
        seed: replay_seed(settings.seed),
        total_particles: Some(histories as usize),
        ..settings.clone()
    };

    // 3. One replay per observation stage, folded onto the steps it observes.
    for &o in &observed {
        let rows: Vec<usize> = (0..steps.len()).filter(|&i| steps[i].stage == o).collect();
        let mut births = Vec::new();
        for (ci, &c) in campaigns.iter().enumerate() {
            let cap = &captures[&stages[c].model];
            let global: Vec<u16> = cap
                .parents
                .iter()
                .map(|p| (ci * n_p + parents.binary_search(p).expect("collected above")) as u16)
                .collect();
            births.extend(
                transfer(cap, models, &stages[c], &stages[o])?
                    .into_iter()
                    .map(|b| Birth {
                        parent: global[b.parent as usize],
                        ..b
                    }),
            );
        }
        if births.is_empty() {
            continue; // every emitter was removed: zero dose, exactly
        }
        births.sort_by_key(|b| b.history);
        // coeffs[k][row]: class k's factor at the end of step rows[row].
        let coeffs: Vec<Vec<f64>> = (0..classes.len())
            .map(|k| {
                let tcf = &tcfs[k / n_p][&parents[k % n_p]];
                rows.iter().map(|&s| tcf[s + 1]).collect()
            })
            .collect();
        let fresh: Vec<Arc<Tally>> = tallies
            .iter()
            .map(|t| binned(t, &classes))
            .collect::<Result<_, _>>()?;
        let folds = fresh
            .iter()
            .map(|t| {
                let inner = t.num_energy_bins() * t.num_mesh_bins();
                BinFold::new(t.num_bins(), inner, rows.len(), &coeffs).map(|f| Some(Arc::new(f)))
            })
            .collect::<Result<_, _>>()?;
        let plan = ReplayPlan {
            births,
            classes: classes.clone(),
            folds,
        };
        let mut model = models[stages[o].model].clone();
        model.sources = Vec::new();
        model.use_decay_photons = false;
        model.transport_secondary_photons = true;
        model.tallies = fresh.clone();
        model.convergence_targets = Vec::new();
        let mut stats = None;
        model.run_staged(
            &replay_settings,
            StagedRun::Replay {
                plan: &plan,
                out: &mut stats,
            },
        )?;
        let stats = stats.ok_or("replay run returned no statistics")?;
        for (k, (d, t)) in out.iter_mut().zip(&fresh).enumerate() {
            let s = &stats.per_tally[k];
            let std = s.std_err();
            let w = d.mean[0].len();
            for (r, &step) in rows.iter().enumerate() {
                d.mean[step].copy_from_slice(&s.mean[r * w..(r + 1) * w]);
                d.std_dev[step].copy_from_slice(&std[r * w..(r + 1) * w]);
            }
            // Per-parent means: linear, so from the raw sums.
            let raw = stats.raw_sums[k]
                .as_ref()
                .ok_or("replay kept no raw sums")?;
            let inner = t.num_energy_bins() * t.num_mesh_bins();
            let n = histories.max(1) as f64;
            for (idx, &x) in raw.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                let i = idx % inner;
                let k = (idx / inner) % classes.len();
                let u = (idx / (inner * classes.len())) * inner + i;
                let rows_of = d
                    .by_nuclide
                    .get_mut(&parents[k % n_p])
                    .expect("every parent");
                for (r, &step) in rows.iter().enumerate() {
                    rows_of[step][u] += coeffs[k][r] * x / n;
                }
            }
        }
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
    fn rotation_turns_directions_and_leaves_them_unit() {
        let r = Placement::new([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(close(r.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        // Translations do not touch directions.
        let t = Placement::translation([5.0, 5.0, 5.0]).unwrap();
        assert!(close(t.rotate([0.0, 0.6, 0.8]), [0.0, 0.6, 0.8]));
    }

    fn birth(history: u64, parent: u16) -> Birth {
        Birth {
            history,
            position: [history as f64, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            energy: 1e6,
            weight: 0.25 * (parent as f64 + 1.0),
            parent,
            cell: 0,
        }
    }

    #[test]
    fn a_replay_history_is_every_photon_of_its_neutron_history() {
        let plan = ReplayPlan {
            births: vec![birth(0, 0), birth(2, 1), birth(2, 0), birth(5, 1)],
            classes: vec!["Mn56@0".into(), "Mn54@0".into()],
            folds: Vec::new(),
        };
        let mut reg = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let ids: Vec<NuclideId> = plan.classes.iter().map(|c| reg.intern(c)).collect();
        let mut bank = ParticleBank::new();
        for (history, expect) in [(0, 1), (1, 0), (2, 2), (3, 0), (5, 1), (9, 0)] {
            bank.clear();
            plan.load(history, &mut bank, 7, &ids);
            let mut got = Vec::new();
            let mut seeds = Vec::new();
            while let Some(p) = bank.pop_particle() {
                assert_eq!(p.particle_type, ParticleType::Photon);
                assert_eq!(p.position[0], history as f64);
                seeds.push(bank.walk_seed());
                got.push(p);
            }
            assert_eq!(got.len(), expect, "history {history}");
            // Each photon at its recorded weight and parent, on its own stream.
            for p in &got {
                let k = ids
                    .iter()
                    .position(|&i| Some(i) == p.parent_nuclide)
                    .unwrap();
                assert_eq!(p.weight, 0.25 * (k as f64 + 1.0));
            }
            seeds.dedup();
            assert_eq!(seeds.len(), expect);
        }
    }

    #[test]
    fn replay_stream_is_not_the_capture_stream() {
        for seed in [0, 1, 2, 12345] {
            let r = replay_seed(seed);
            assert_ne!(r, seed);
            assert_ne!(
                yamc_rng::history_seed(r, 0),
                yamc_rng::history_seed(seed, 0)
            );
        }
    }
}
