//! Rigorous two-step (R2S) shutdown dose rate, on cells or cell-under-mesh.
//!
//! D1S ([`crate::shutdown_dose`]) carries each decay photon from the neutron
//! reaction that made its emitter, so it only sees emitters made in one
//! reaction from the irradiated composition: an activation product that is
//! itself activated (Co59 -> Co60 -> Co61, or the W and Re chains) and burnup
//! of the parents are missed. R2S solves the inventory of activated regions
//! instead:
//!
//! 1. **Activation.** The neutrons are transported once in the irradiation
//!    model and every region's inventory is solved through the whole schedule.
//!    - Cells: every material-filled cell is a region with its own material,
//!      solved with its own continuous-energy rates (`Model::transmute`,
//!      independent method).
//!    - Cell-under-mesh: every (voxel, material-filled cell) overlap is a
//!      region with the cell's material and the overlap's volume, estimated by
//!      jittered sampling of the voxel. Its rates are collapsed from the
//!      voxel's neutron spectrum, tallied on [`GROUPS`]
//!      (`yani_transmute::transmute_materials`, which collapses each distinct
//!      spectrum and composition once).
//! 2. **Decay photons.** At the end of every step, each region emits the decay
//!    photons of its inventory then (lines and continua, photons/s), uniformly
//!    over its volume, in the step's model. A photon transport per step scores
//!    the tallies, binned by the emitting radionuclide.
//!
//! The source is uniform within a region, so cells with a steep flux gradient
//! should be split, or a mesh used. Cells are matched across models by name,
//! as in D1S; a cell absent from a step's model emits nothing there. With
//! cells, a moved cell emits wherever the step's model has it. With a mesh, a
//! moved cell's voxel parts move with the stage's placement of it (`moves`),
//! as D1S births do, and are checked to land in it. The pulses must all
//! irradiate in one model (with a mesh, in one stage). Voxels no neutron
//! reaches are not activated.
//!
//! Standard deviations are the photon transport's; the neutron transport's
//! noise, carried into the inventories, is not propagated.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use rand::RngExt;
use rayon::prelude::*;

use yamc_nuclide::config::ChainUse;
use yamc_nuclide::nuclide_registry::NuclideId;
use yamc_particle::particle::{Particle, ParticleType};
use yamc_physics::photon::decay_photon_production::{decay_photon_spectra, DecayPhotonSpectrum};
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::{EnergyFilter, MeshFilter, RegularRectangularMesh, Tally};

use crate::geo::BoundingBox;
use crate::geometry::backend::GeometryKind;
use crate::model::{Model, TransportSettings};
use crate::shutdown_dose::{
    binned, cell_names, check_inputs, need_neutron_source, step_times, DoseTally, Placement,
    ShutdownDose, Stage, StagedRun, Step,
};
use crate::track::NoOpTracker;
use crate::util::fast_rng::FastRng;

/// atoms per (atom/b-cm x cm^3).
const BARN_PER_CM_SQ: f64 = 1.0e24;
/// Points sampled per cell for a volume the user did not give: enough for a
/// fraction of a percent on a cell that fills half its box, and the volume
/// only enters the inventory through reactions on activation products.
const VOLUME_SAMPLES: u64 = 100_000;
/// Jittered points per voxel axis for the cell-under-mesh volumes: 512 per
/// voxel resolve a cell to about 0.2% of the voxel, and a part the points
/// miss is that small.
const VOXEL_SAMPLES_PER_AXIS: usize = 8;
/// The neutron groups a voxel's spectrum is tallied on. 175 groups resolve
/// the thresholds and the resonance region for activation, at a quarter of
/// the per-thread tally memory of 709 on a large mesh.
pub const GROUPS: &str = "VITAMIN-J-175";
/// Rejected positions before a region is declared too thin to sample in.
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

/// A uniform point of `cell` within `bounds`, or `None` after
/// [`MAX_REJECTIONS`] misses.
fn point_in<R: rand::Rng>(
    geometry: &GeometryKind,
    cell: usize,
    bounds: &BoundingBox,
    rng: &mut R,
) -> Option<[f64; 3]> {
    let (lo, w) = (bounds.lower_left, bounds.width());
    (0..MAX_REJECTIONS)
        .map(|_| std::array::from_fn(|k| lo[k] + w[k] * rng.random::<f64>()))
        .find(|p: &[f64; 3]| geometry.find_cell_index((p[0], p[1], p[2])) == Some(cell))
}

/// Where a region's photons are born: uniformly over the points of `cell` (in
/// the source's frame geometry) within `bounds`, then placed by `placement`.
struct Site {
    cell: usize,
    bounds: BoundingBox,
    placement: Placement,
}

/// Decay photons emitted uniformly over regions, each (region, emitter) in
/// proportion to its rate. Every photon carries the total rate as its weight,
/// so tallies come out per second.
pub(crate) struct CellDecaySource<'a> {
    /// The geometry the sites' cells are cells of.
    frame: &'a GeometryKind,
    emitters: &'a [Emitter],
    sites: Vec<Site>,
    /// (index into `sites`, emitter index, cumulative rate).
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
    pub(crate) fn sample<R: rand::Rng>(&self, rng: &mut R, ids: &[NuclideId]) -> Particle {
        let t = rng.random::<f64>() * self.total();
        let i = self.entries.partition_point(|e| e.2 <= t);
        let (s, em, _) = self.entries[i.min(self.entries.len() - 1)];
        let energy = self.emitters[em as usize].sample(rng);
        let site = &self.sites[s as usize];
        let point = point_in(self.frame, site.cell, &site.bounds, rng).unwrap_or_else(|| {
            panic!(
                "no point of cell {} found in {MAX_REJECTIONS} draws in its bounding box",
                describe(self.frame, site.cell)
            )
        });
        let mu = 2.0 * rng.random::<f64>() - 1.0;
        let phi = std::f64::consts::TAU * rng.random::<f64>();
        let st = (1.0 - mu * mu).max(0.0).sqrt();
        let mut p = Particle::new(
            site.placement.apply(point),
            [st * phi.cos(), st * phi.sin(), mu],
            energy,
        );
        p.particle_type = ParticleType::Photon;
        p.weight = self.total();
        p.parent_nuclide = Some(ids[em as usize]);
        p
    }
}

/// An activated region: a cell of the irradiation model, or its part in one
/// voxel, and its atoms at every composition of the schedule.
struct Activated {
    cell: usize,
    voxel: Option<BoundingBox>,
    atoms: Vec<HashMap<String, f64>>,
}

/// Atoms of `m` in `volume` cm^3.
fn atoms_in(
    m: &yamc_materials::material::Material,
    volume: f64,
) -> Result<HashMap<String, f64>, String> {
    Ok(m.get_atoms_per_barn_cm()?
        .into_iter()
        .map(|(n, v)| (n, v * BARN_PER_CM_SQ * volume))
        .collect())
}

/// Every material-filled cell of `model` with its own transmutable material,
/// its volume estimated by sampling where the user gave none, solved through
/// `schedule`.
fn activate_cells(
    model: &mut Model,
    schedule: &yani_transmute::Schedule,
    loaded: yani::LoadedChain,
    settings: &TransportSettings,
) -> Result<Vec<Activated>, String> {
    let mut cells = Vec::new();
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
                    settings.seed.wrapping_add(cell as u64),
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
        cells.push((cell, material_id, volume));
    }
    if cells.is_empty() {
        return Err("the irradiation model has no material-filled cell to activate".into());
    }
    let results = model
        .transmute(
            "independent",
            schedule,
            loaded.chain,
            loaded.branch,
            loaded.parts,
            settings,
            None,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
            None,
        )
        .map_err(|e| e.to_string())?;
    cells
        .into_iter()
        .map(|(cell, id, volume)| {
            Ok(Activated {
                cell,
                voxel: None,
                atoms: results.materials[&id]
                    .iter()
                    .map(|m| atoms_in(m, volume))
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect()
}

/// Voxel `v`'s box.
fn voxel_box(mesh: &RegularRectangularMesh, v: usize) -> BoundingBox {
    let ([nx, ny, _], ll, w) = (mesh.shape(), mesh.lower_left(), mesh.width());
    let i = [v % nx, (v / nx) % ny, v / (nx * ny)];
    let lo: [f64; 3] = std::array::from_fn(|k| ll[k] + w[k] * i[k] as f64);
    BoundingBox::new(lo, std::array::from_fn(|k| lo[k] + w[k]))
}

/// The material-filled cells under voxel `v` and their volumes in it, by
/// jittered sampling: one point in each of the voxel's subcells.
fn cells_under(
    geometry: &GeometryKind,
    mesh: &RegularRectangularMesh,
    v: usize,
    seed: u64,
) -> Vec<(usize, f64)> {
    const N: usize = VOXEL_SAMPLES_PER_AXIS;
    let b = voxel_box(mesh, v);
    let w = b.width();
    let mut rng = FastRng::new(seed ^ (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut hits: Vec<(usize, u32)> = Vec::new();
    for i in 0..N * N * N {
        let s = [i % N, (i / N) % N, i / (N * N)];
        let p: [f64; 3] = std::array::from_fn(|k| {
            b.lower_left[k] + w[k] * (s[k] as f64 + rng.random()) / N as f64
        });
        let Some(c) = geometry.find_cell_index((p[0], p[1], p[2])) else {
            continue;
        };
        if geometry.material_for(&geometry.cells()[c]).is_none() {
            continue;
        }
        match hits.iter_mut().find(|h| h.0 == c) {
            Some(h) => h.1 += 1,
            None => hits.push((c, 1)),
        }
    }
    let per_hit = mesh.get_voxel_volume(v) / (N * N * N) as f64;
    hits.into_iter()
        .map(|(c, n)| (c, n as f64 * per_hit))
        .collect()
}

/// Every (voxel, material-filled cell) overlap of `model` the neutrons reach,
/// with the cell's material, solved through `steps` on the voxel's tallied
/// spectrum.
fn activate_mesh(
    model: &mut Model,
    mesh: &RegularRectangularMesh,
    steps: &[yani_transmute::ScheduleStep],
    loaded: yani::LoadedChain,
    settings: &TransportSettings,
) -> Result<Vec<Activated>, String> {
    let groups = yamc_nuclide::group_structures::get_group_structure(GROUPS)?.to_vec();
    let mut t = Tally::new();
    t.name = Some("R2S voxel neutron spectra".into());
    t.scores = vec![Score::Flux(FluxScore)];
    t.filters = vec![
        Filter::Energy(EnergyFilter::new(groups.clone())),
        Filter::Mesh(MeshFilter::new(mesh.clone())),
    ];
    t.reset_accumulation();
    let t = Arc::new(t);
    model.tallies = vec![Arc::clone(&t)];
    model.run_internal::<NoOpTracker>(settings, None, None, false)?;
    // Track length per source neutron, energy-major then voxel.
    let flux = t.finalize().mean;
    let (nv, ng) = (mesh.num_voxels(), groups.len() - 1);
    let reached: Vec<usize> = (0..nv)
        .filter(|&v| (0..ng).any(|g| flux[g * nv + v] > 0.0))
        .collect();
    let geometry = &model.geometry;
    let regions: Vec<(usize, usize, f64)> = reached
        .par_iter()
        .flat_map_iter(|&v| {
            cells_under(geometry, mesh, v, settings.seed)
                .into_iter()
                .map(move |(c, vol)| (v, c, vol))
        })
        .collect();
    if regions.is_empty() {
        return Err(
            "no material-filled cell under the mesh sees a neutron; the mesh must cover \
             the activated components"
                .into(),
        );
    }

    let mut materials = Vec::with_capacity(regions.len());
    let mut plans = Vec::with_capacity(regions.len());
    for &(v, c, volume) in &regions {
        let masses: Vec<f64> = (0..ng).map(|g| flux[g * nv + v]).collect();
        let total: f64 = masses.iter().sum();
        // n/cm^2 per source neutron.
        let per_source = total / mesh.get_voxel_volume(v);
        let mut m = (**geometry
            .material_for(&geometry.cells()[c])
            .expect("regions are material-filled"))
        .clone();
        m.invalidate_xs_cache();
        m.volume = Some(volume);
        m.assign_material_id(None)?;
        materials.push(m);
        plans.push((
            yani_transmute::MultigroupSpectrum {
                boundaries: groups.clone(),
                masses: masses.iter().map(|x| x / total).collect(),
                flux_error: None,
            },
            steps
                .iter()
                .map(|s| yani_transmute::TransmuteStep {
                    dt: s.dt,
                    irradiation: s.is_pulse.then_some((0, s.rate * per_source)),
                })
                .collect::<Vec<_>>(),
        ));
    }
    let ids: Vec<u32> = materials
        .iter()
        .map(|m| m.get_material_id().expect("assigned above"))
        .collect();
    let cases = materials
        .iter_mut()
        .zip(plans)
        .map(
            |(material, (spectrum, steps))| yani_transmute::TransmuteCase {
                material,
                spectra: vec![spectrum],
                steps,
                shielding: None,
                product_threshold: yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
            },
        )
        .collect();
    let results = yani_transmute::transmute_materials(
        cases,
        loaded.chain,
        &loaded.branch,
        loaded.parts,
        None,
        None,
        None,
    )
    .map_err(|e| format!("cell-under-mesh inventories: {e}"))?;
    regions
        .iter()
        .zip(ids)
        .map(|(&(v, cell, volume), id)| {
            Ok(Activated {
                cell,
                voxel: Some(voxel_box(mesh, v)),
                atoms: results.materials[&id]
                    .iter()
                    .map(|m| atoms_in(m, volume))
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect()
}

/// Simulate the shutdown dose rate over `steps` by R2S, on the cells of the
/// irradiation model or, with `mesh`, on the parts of its cells under each
/// voxel.
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
    mesh: Option<&RegularRectangularMesh>,
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
    let irr_stage = &stages[*campaigns.first().expect("check_inputs requires a pulse")];
    if mesh.is_some() && campaigns.len() > 1 {
        return Err(
            "cell-under-mesh R2S needs every pulse to irradiate with the cells in the same \
             place (one set of moves), so the voxels say where the inventory is"
                .into(),
        );
    }

    // 1. Activation, per region, over the steps that take time.
    let mut model = models[irr].clone();
    need_neutron_source(&model, irr)?;
    model.tallies = Vec::new();
    model.convergence_targets = Vec::new();
    model.use_decay_photons = false;
    model.transport_secondary_photons = false;
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
    let activated = match mesh {
        None => activate_cells(
            &mut model,
            &yani_transmute::Schedule::new(timed)?,
            loaded,
            settings,
        )?,
        Some(mesh) => activate_mesh(&mut model, mesh, &timed, loaded, settings)?,
    };

    // The emitters among the regions' nuclides.
    let names: BTreeSet<&String> = activated
        .iter()
        .flat_map(|a| a.atoms.iter().flat_map(|m| m.keys()))
        .collect();
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
        let stage = &stages[step.stage];
        let to = &models[stage.model];
        let same = stage.model == irr;
        let dst_names = if same {
            HashMap::new()
        } else {
            cell_names(to)?
        };
        // Cells sample in the step's model; voxel parts in the irradiation
        // model, where the voxels say the inventory is, then move.
        let mut src = CellDecaySource {
            frame: if mesh.is_some() {
                &model.geometry
            } else {
                &to.geometry
            },
            emitters: &emitters,
            sites: Vec::new(),
            entries: Vec::new(),
            parents: parents.clone(),
        };
        let mut rng = FastRng::new(settings.seed ^ i as u64);
        let mut total = 0.0;
        for a in &activated {
            let name = || {
                model.geometry.cells()[a.cell]
                    .name
                    .as_deref()
                    .ok_or_else(|| {
                        format!(
                            "activated cell {} is unnamed; name every activated cell so later \
                         stages can find it in their models",
                            describe(&model.geometry, a.cell)
                        )
                    })
            };
            let mut inv: Vec<(&String, &f64)> = a.atoms[at_end[i]].iter().collect();
            inv.sort_unstable_by(|a, b| a.0.cmp(b.0));
            let rates: Vec<(u16, f64)> = inv
                .into_iter()
                .filter_map(|(n, &atoms)| {
                    let &em = index.get(n.as_str())?;
                    let r = atoms * emitters[em as usize].per_atom;
                    (r > 0.0).then_some((em, r))
                })
                .collect();
            if rates.is_empty() {
                continue;
            }
            let site = match &a.voxel {
                None => {
                    let dst = if same {
                        a.cell
                    } else {
                        match dst_names.get(name()?) {
                            Some(&d) => d,
                            None => continue,
                        }
                    };
                    let key = (stage.model, dst);
                    let bounds = match boxes.get(&key) {
                        Some(b) => b.clone(),
                        None => {
                            let b = cell_box(&to.geometry, dst)?;
                            boxes.insert(key, b.clone());
                            b
                        }
                    };
                    Site {
                        cell: dst,
                        bounds,
                        placement: Placement::IDENTITY,
                    }
                }
                Some(voxel) => {
                    let key = (irr, a.cell);
                    let cb = match boxes.get(&key) {
                        Some(b) => b.clone(),
                        None => {
                            let b = cell_box(&model.geometry, a.cell)?;
                            boxes.insert(key, b.clone());
                            b
                        }
                    };
                    let bounds = voxel.intersection(&cb);
                    let named = model.geometry.cells()[a.cell].name.as_deref();
                    let at = |s: &Stage| {
                        named
                            .and_then(|n| s.placements.get(n))
                            .copied()
                            .unwrap_or_default()
                    };
                    let placement = at(stage).after(&at(irr_stage).inverse());
                    if !same || placement != Placement::IDENTITY {
                        let name = name()?;
                        let Some(&dst) = dst_names.get(name).or(same.then_some(&a.cell)) else {
                            continue;
                        };
                        // A voxel part lands in its cell where the stage's
                        // model has the cell where its moves put it.
                        let landed = point_in(&model.geometry, a.cell, &bounds, &mut rng)
                            .map(|p| placement.apply(p))
                            .and_then(|p| to.geometry.find_cell_index((p[0], p[1], p[2])));
                        if landed != Some(dst) {
                            return Err(format!(
                                "cell '{name}' is not in step {i}'s model where the step's \
                                 moves put it; with a mesh, give every moved cell's move"
                            ));
                        }
                    }
                    Site {
                        cell: a.cell,
                        bounds,
                        placement,
                    }
                }
            };
            let s = src.sites.len() as u32;
            src.sites.push(site);
            for (em, r) in rates {
                total += r;
                src.entries.push((s, em, total));
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
