//! Cell-based rigorous two-step (R2S) shutdown dose rate.
//!
//! D1S ([`crate::shutdown_dose`]) carries each decay photon from the neutron
//! reaction that made its emitter, so it only sees emitters made in one
//! reaction from the irradiated composition: an activation product that is
//! itself activated (Co59 -> Co60 -> Co61, or the W and Re chains) and burnup
//! of the parents are missed. R2S solves the inventory instead:
//!
//! 1. **Activation.** Every material-filled cell of the irradiation model gets
//!    its own material, the neutrons are transported once, and each cell's
//!    inventory is solved through the whole schedule with its own spectrum
//!    (`Model::transmute`, independent method).
//! 2. **Decay photons.** At the end of every step, each cell emits the decay
//!    photons of its inventory then (lines and continua, photons/s), uniformly
//!    over its volume, in the step's model. A photon transport per step scores
//!    the tallies, binned by the emitting radionuclide.
//!
//! The source is uniform within a cell, so a cell with a steep flux gradient
//! should be split into smaller cells. Cells are matched across models by name,
//! as in D1S; a cell absent from a step's model emits nothing there. A moved
//! cell emits wherever the step's model has it, so `moves` need not be given.
//! The pulses must all irradiate in one model.
//!
//! Standard deviations are the photon transport's; the neutron transport's
//! noise, carried into the inventories, is not propagated.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use rand::RngExt;

use yamc_nuclide::config::ChainUse;
use yamc_nuclide::nuclide_registry::NuclideId;
use yamc_particle::particle::{Particle, ParticleType};
use yamc_physics::photon::decay_photon_production::{decay_photon_spectra, DecayPhotonSpectrum};
use yamc_tallies::Tally;

use crate::geo::BoundingBox;
use crate::geometry::backend::GeometryKind;
use crate::model::{Model, TransportSettings};
use crate::shutdown_dose::{
    binned, cell_names, check_inputs, need_neutron_source, step_times, DoseTally, ShutdownDose,
    Stage, StagedRun, Step,
};

/// atoms per (atom/b-cm x cm^3).
const BARN_PER_CM_SQ: f64 = 1.0e24;
/// Points sampled per cell for a volume the user did not give: enough for a
/// fraction of a percent on a cell that fills half its box, and the volume
/// only enters the inventory through reactions on activation products.
const VOLUME_SAMPLES: u64 = 100_000;
/// Rejected positions before a cell is declared too thin to sample in.
const MAX_REJECTIONS: usize = 1_000_000;

/// A radionuclide's decay photons: one spectrum per continuum and one for its
/// lines, picked by their share of its emission rate per atom.
pub(crate) struct Emitter {
    name: String,
    per_atom: f64,
    spectra: Vec<DecayPhotonSpectrum>,
    cumulative: Vec<f64>,
}

impl Emitter {
    fn new(name: &str, nuclide: &yani::ChainNuclide) -> Result<Option<Self>, String> {
        let (rates, spectra): (Vec<f64>, Vec<_>) = decay_photon_spectra(name, nuclide)?
            .into_iter()
            .filter(|(r, _)| *r > 0.0)
            .unzip();
        let cumulative: Vec<f64> = rates
            .iter()
            .scan(0.0, |s, r| {
                *s += r;
                Some(*s)
            })
            .collect();
        Ok(cumulative.last().copied().map(|per_atom| Self {
            name: name.to_string(),
            per_atom,
            spectra,
            cumulative,
        }))
    }

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        let t = rng.random::<f64>() * self.per_atom;
        let i = self.cumulative.partition_point(|&c| c <= t);
        self.spectra[i.min(self.spectra.len() - 1)].sample(rng)
    }
}

/// The box a cell is sampled in: its region's bounds within the geometry's.
fn cell_box(geometry: &GeometryKind, cell: usize) -> Result<BoundingBox, String> {
    let world = geometry.bounding_box();
    let b = match geometry {
        GeometryKind::Csg(_) => geometry.cells()[cell]
            .region
            .bounding_box()
            .intersection(&world),
        #[cfg(feature = "mesh")]
        GeometryKind::Mesh(_) => world,
    };
    if !b.is_finite() || b.volume() <= 0.0 {
        return Err(format!(
            "cell {} has no finite bounding box to sample decay photons in",
            describe(geometry, cell)
        ));
    }
    Ok(b)
}

fn describe(geometry: &GeometryKind, cell: usize) -> String {
    let c = &geometry.cells()[cell];
    match (&c.name, c.cell_id) {
        (Some(n), _) => format!("'{n}'"),
        (None, Some(id)) => format!("id {id}"),
        (None, None) => format!("index {cell}"),
    }
}

/// Decay photons emitted uniformly over cells, each (cell, emitter) in
/// proportion to its rate. Every photon carries the total rate as its weight,
/// so tallies come out per second.
pub(crate) struct CellDecaySource<'a> {
    emitters: &'a [Emitter],
    boxes: Vec<(usize, BoundingBox)>,
    /// (index into `boxes`, emitter index, cumulative rate).
    entries: Vec<(u32, u16, f64)>,
    parents: Vec<String>,
}

impl<'a> CellDecaySource<'a> {
    fn total(&self) -> f64 {
        self.entries.last().map_or(0.0, |e| e.2)
    }

    /// The emitters, in the order `sample`'s `ids` follow.
    pub(crate) fn parents(&self) -> &[String] {
        &self.parents
    }

    /// One decay photon; `ids` maps `parents` to the run's nuclide ids.
    pub(crate) fn sample<R: rand::Rng>(
        &self,
        rng: &mut R,
        ids: &[NuclideId],
        geometry: &GeometryKind,
    ) -> Particle {
        let t = rng.random::<f64>() * self.total();
        let i = self.entries.partition_point(|e| e.2 <= t);
        let (b, em, _) = self.entries[i.min(self.entries.len() - 1)];
        let energy = self.emitters[em as usize].sample(rng);
        let (cell, bounds) = &self.boxes[b as usize];
        let (lo, w) = (bounds.lower_left, bounds.width());
        let position = (0..MAX_REJECTIONS)
            .map(|_| std::array::from_fn(|k| lo[k] + w[k] * rng.random::<f64>()))
            .find(|p: &[f64; 3]| geometry.find_cell_index((p[0], p[1], p[2])) == Some(*cell))
            .unwrap_or_else(|| {
                panic!(
                    "no point of cell {} found in {MAX_REJECTIONS} draws in its bounding box",
                    describe(geometry, *cell)
                )
            });
        let mu = 2.0 * rng.random::<f64>() - 1.0;
        let phi = std::f64::consts::TAU * rng.random::<f64>();
        let s = (1.0 - mu * mu).max(0.0).sqrt();
        let mut p = Particle::new(position, [s * phi.cos(), s * phi.sin(), mu], energy);
        p.particle_type = ParticleType::Photon;
        p.weight = self.total();
        p.parent_nuclide = Some(ids[em as usize]);
        p
    }
}

/// An activated cell: its index in the irradiation model and its material.
struct Activated {
    cell: usize,
    material_id: u32,
    volume: f64,
}

/// Give every material-filled cell of `model` its own transmutable material
/// with the cell's volume, estimated by sampling where the user gave none.
fn differentiate(model: &mut Model, seed: u64) -> Result<Vec<Activated>, String> {
    let mut out = Vec::new();
    for cell in 0..model.geometry.num_cells() {
        let c = &model.geometry.cells()[cell];
        let Some(material) = model.geometry.material_for(c) else {
            continue;
        };
        let volume = match c.volume {
            Some(v) => v,
            None => {
                let b = cell_box(&model.geometry, cell)?;
                let g = &model.geometry;
                let v = crate::stochastic_volume::calculate_stochastic_volumes(
                    VOLUME_SAMPLES,
                    &b,
                    seed.wrapping_add(cell as u64),
                    1,
                    |p| (g.find_cell_index(p) == Some(cell)).then_some(0),
                );
                if v[0].num_hits == 0 {
                    return Err(format!(
                        "cell {} is too thin to estimate its volume; give it a volume",
                        describe(g, cell)
                    ));
                }
                v[0].volume
            }
        };
        let mut m = (**material).clone();
        m.transmutable = true;
        m.volume = Some(volume);
        let material_id = m.assign_material_id(None)?;
        let slot = model.geometry.materials().len() as u32;
        model.geometry.materials_mut().push(Arc::new(m));
        model.geometry.cells_mut()[cell].material_idx = Some(slot);
        out.push(Activated {
            cell,
            material_id,
            volume,
        });
    }
    if out.is_empty() {
        return Err("the irradiation model has no material-filled cell to activate".into());
    }
    Ok(out)
}

/// Simulate the shutdown dose rate over `steps` by cell-based R2S.
///
/// Inputs and result are those of
/// [`crate::shutdown_dose::simulate_shutdown_dose`]; `total_particles` in
/// `settings` applies to the neutron transport and to each step's photon
/// transport. See the module docs for the method.
pub fn simulate_r2s(
    models: &[Model],
    stages: &[Stage],
    steps: &[Step],
    tallies: &[Arc<Tally>],
    settings: &TransportSettings,
) -> Result<ShutdownDose, String> {
    let (campaigns, _) = check_inputs(models, stages, steps, tallies, settings)?;
    let irradiated: BTreeSet<usize> = campaigns.iter().map(|&c| stages[c].model).collect();
    let &irr = irradiated.first().expect("check_inputs requires a pulse");
    if irradiated.len() > 1 {
        return Err(
            "R2S needs every pulse to irradiate in the same model; use method='d1s' for a \
             schedule that irradiates several"
                .into(),
        );
    }

    // 1. Activation, per cell, over the steps that take time.
    let mut model = models[irr].clone();
    need_neutron_source(&model, irr)?;
    model.tallies = Vec::new();
    model.convergence_targets = Vec::new();
    model.use_decay_photons = false;
    let activated = differentiate(&mut model, settings.seed)?;
    let timed: Vec<yani_transmute::ScheduleStep> = steps
        .iter()
        .filter(|s| s.duration > 0.0)
        .map(|s| yani_transmute::ScheduleStep {
            rate: s.rate,
            dt: s.duration,
            is_pulse: s.rate > 0.0,
        })
        .collect();
    // Composition index at the end of each step (0 is the initial one).
    let at_end: Vec<usize> = steps
        .iter()
        .scan(0, |k, s| {
            *k += (s.duration > 0.0) as usize;
            Some(*k)
        })
        .collect();
    let loaded = yani_transmute::load_configured_chain(ChainUse::transmutation(true))
        .map_err(|e| format!("Failed to load the transmutation chain for R2S: {e}"))?;
    let chain = Arc::clone(&loaded.chain);
    let results = model
        .transmute(
            "independent",
            &yani_transmute::Schedule::new(timed)?,
            loaded.chain,
            loaded.branch,
            loaded.parts,
            settings,
            None,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        )
        .map_err(|e| e.to_string())?;

    // Atoms per cell and step, and the emitters among them.
    let atoms: Vec<Vec<HashMap<String, f64>>> = activated
        .iter()
        .map(|a| {
            results.materials[&a.material_id]
                .iter()
                .map(|m| {
                    m.get_atoms_per_barn_cm().map(|d| {
                        d.into_iter()
                            .map(|(n, v)| (n, v * BARN_PER_CM_SQ * a.volume))
                            .collect()
                    })
                })
                .collect()
        })
        .collect::<Result<_, String>>()?;
    let names: BTreeSet<&String> = atoms.iter().flatten().flat_map(|m| m.keys()).collect();
    let mut emitters = Vec::new();
    for name in names {
        if let Some(n) = chain.get(name.as_str())
            && let Some(e) = Emitter::new(name, n)?
        {
            emitters.push(e);
        }
    }
    let index: HashMap<&str, u16> = emitters
        .iter()
        .enumerate()
        .map(|(i, e)| (e.name.as_str(), i as u16))
        .collect();
    let parents: Vec<String> = emitters.iter().map(|e| e.name.clone()).collect();

    // 2. One photon transport per step, in its own model.
    let fresh = |parents: &[String]| -> Result<Vec<Arc<Tally>>, String> {
        tallies.iter().map(|t| binned(t, parents)).collect()
    };
    let n_scores: Vec<usize> = tallies.iter().map(|t| t.scores.len()).collect();
    let ones: HashMap<String, Vec<f64>> =
        parents.iter().map(|p| (p.clone(), vec![1.0; 2])).collect();
    let mut boxes: HashMap<(usize, usize), BoundingBox> = HashMap::new();
    let mut out: Vec<DoseTally> = vec![DoseTally::default(); tallies.len()];
    for (i, step) in steps.iter().enumerate() {
        let to = &models[stages[step.stage].model];
        let same = stages[step.stage].model == irr;
        let dst_names = if same {
            HashMap::new()
        } else {
            cell_names(to)?
        };
        let mut src = CellDecaySource {
            emitters: &emitters,
            boxes: Vec::new(),
            entries: Vec::new(),
            parents: parents.clone(),
        };
        let mut total = 0.0;
        for (a, cell_atoms) in activated.iter().zip(&atoms) {
            let dst = if same {
                a.cell
            } else {
                let c = &model.geometry.cells()[a.cell];
                let name = c.name.as_deref().ok_or_else(|| {
                    format!(
                        "activated cell {} is unnamed; name every activated cell so later \
                         stages can find it in their models",
                        describe(&model.geometry, a.cell)
                    )
                })?;
                match dst_names.get(name) {
                    Some(&d) => d,
                    None => continue,
                }
            };
            let key = (stages[step.stage].model, dst);
            let bounds = match boxes.get(&key) {
                Some(b) => b.clone(),
                None => {
                    let b = cell_box(&to.geometry, dst)?;
                    boxes.insert(key, b.clone());
                    b
                }
            };
            let b = src.boxes.len() as u32;
            let mut any = false;
            let mut inv: Vec<(&String, &f64)> = cell_atoms[at_end[i]].iter().collect();
            inv.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (name, &n) in inv {
                if let Some(&em) = index.get(name.as_str()) {
                    let rate = n * emitters[em as usize].per_atom;
                    if rate > 0.0 {
                        total += rate;
                        src.entries.push((b, em, total));
                        any = true;
                    }
                }
            }
            if any {
                src.boxes.push((dst, bounds));
            }
        }
        let fresh = fresh(&parents)?;
        let rows: Vec<(Vec<f64>, Vec<f64>)> = if total > 0.0 {
            let mut m = to.clone();
            m.sources = Vec::new();
            m.use_decay_photons = false;
            m.transport_secondary_photons = true;
            m.tallies = fresh.clone();
            m.convergence_targets = Vec::new();
            m.run_staged(settings, StagedRun::Cells(&src))?;
            fresh
                .iter()
                .map(|t| {
                    let r = t.finalize();
                    (r.mean, r.standard_deviation)
                })
                .collect()
        } else {
            fresh
                .iter()
                .map(|t| (vec![0.0; t.num_bins()], vec![0.0; t.num_bins()]))
                .collect()
        };
        for (k, (mean, std)) in rows.iter().enumerate() {
            let d = &mut out[k];
            if parents.is_empty() {
                d.mean.push(mean.clone());
                d.std_dev.push(std.clone());
                continue;
            }
            // A tally binned by parent is laid out as the time correction
            // reads it; correcting by one sums or splits the parents.
            let (m, s) = yani_decay::apply_time_correction(
                mean,
                std,
                &parents,
                n_scores[k],
                &ones,
                &[1],
                true,
            )?;
            let (per, _) = yani_decay::apply_time_correction(
                mean,
                std,
                &parents,
                n_scores[k],
                &ones,
                &[1],
                false,
            )?;
            d.mean.push(m[0].clone());
            d.std_dev.push(s[0].clone());
            let stride = per[0].len() / parents.len();
            for (p, name) in parents.iter().enumerate() {
                d.by_nuclide
                    .entry(name.clone())
                    .or_insert_with(|| vec![Vec::new(); steps.len()])[i] =
                    per[0][p * stride..(p + 1) * stride].to_vec();
            }
        }
    }
    Ok(ShutdownDose {
        times: step_times(steps),
        tallies: out,
        radionuclides: parents,
    })
}
