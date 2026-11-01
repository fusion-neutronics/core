//! Each replica's own cross sections, for the uncertainty of a coupled
//! transport-transmutation.
//!
//! A coupled replica is the whole coupled run again (see
//! [`yani_transmute::coupled_uncertainty`]). With the cross-section source on,
//! every transport of replica `k`, in every material of the geometry, reads
//! nuclear data with replica `k`'s draw applied, and the transmutation tallies
//! read the same data, so the draw moves the flux, the rates it gives, the
//! compositions those rates make and every later step's flux through them.
//! Nothing is linearised: the replica is a coupled run on a perturbed library.
//!
//! The draw is the one the replica weights make for independent mode
//! ([`crate::transport::replica`]): each nuclide's MF=33 covariance cells as a
//! field, drawn under `(seed, replica, nuclide)` with the cross-material
//! blocks drawn jointly, and its resolved resonance parameters sampled and
//! the cross sections rebuilt where the transport can place them, applied by
//! [`crate::xs_perturbation`] as a rerun applies it. A nuclide in the
//! composition therefore moves in replica `k` exactly as in replica `k` of
//! independent mode.
//!
//! A nuclide has one draw whatever scope its data is held at. Each is fielded
//! on its data read in full, with every MT and its covariance, and a product
//! held at activation scope moves by what that draw moved the same reactions
//! of the full data ([`crate::xs_perturbation::moved_like`]). When a product
//! grows a density and transport widens it to full scope mid-run, the
//! widened data takes the same draw, so the product's cross sections never
//! change within a replica.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::Nuclide;
use yani_transmute::covariance_fold::{transport_fields, CellField, TransportField};
use yani_transmute::covariance_sample::{JointAdjustment, LognormalLimit, Sampler};
use yani_transmute::cross_material::CrossMaterial;
use yani_transmute::resonance_rates::{ResonanceMethod, TransportResonance};
use yani_transmute::uncertainty::Info;

use crate::xs_perturbation::{
    moved_like, perturbed_nuclide, resonance_sampling, sampled_names, ResonanceChange,
};

/// A nuclide at one temperature: its name and its temperature label.
type Key = (String, String);

/// What every replica draws for one nuclide at one temperature.
struct Entry {
    /// The nuclide read in full, with its covariance: what a draw moves.
    full: Arc<Nuclide>,
    /// The field its MF=33 draw is over, if it has one.
    field: Option<TransportField>,
    /// The sampler drawing that field, linked to its cross-material partners.
    sampler: Option<Sampler>,
    /// Its resonance parameters, where they are sampled.
    resonance: Option<TransportResonance>,
}

/// The nuclear data every replica of a coupled run draws from.
pub struct CoupledCrossSections {
    seed: u64,
    cross: CrossMaterial,
    partners: BTreeMap<String, CellField>,
    /// `None` for a nuclide with nothing to draw.
    entries: HashMap<Key, Option<Arc<Entry>>>,
    /// How each nuclide's resolved resonance range was taken.
    methods: BTreeMap<String, ResonanceMethod>,
    /// Nuclides whose data carries no covariance.
    without: BTreeSet<String>,
    /// Per material id, every nuclide a replica met in it.
    met: BTreeMap<u32, BTreeSet<String>>,
    /// Grid values a draw took below zero, floored, over every replica.
    floored: usize,
}

/// The label `material`'s cross sections are read at.
fn label(material: &Material) -> Result<String, String> {
    if material.temperature().is_empty() {
        material.resolved_temperature(None)
    } else {
        Ok(material.temperature().to_string())
    }
}

/// A material at `temperature` holding `nuclides`, each read in full with its
/// covariance and resonance parameters, widened from the data given.
fn read_in_full(
    nuclides: &BTreeMap<String, Arc<Nuclide>>,
    temperature: &str,
) -> Result<Material, String> {
    let names = nuclides.keys().map(|n| (n.clone(), 1.0)).collect();
    let mut m = Material::new(names, "atom", "g/cm3", Some(1.0))?;
    m.set_temperature(temperature);
    m.nuclide_data = nuclides
        .iter()
        .map(|(n, d)| (n.clone(), Arc::clone(d)))
        .collect();
    fn fail(what: &'static str) -> impl Fn(Box<dyn std::error::Error>) -> String {
        move |e| format!("{what}: {e}")
    }
    m.ensure_nuclides_loaded()
        .map_err(fail("reading transmutation products in full"))?;
    m.ensure_covariance_loaded()
        .map_err(fail("loading covariance"))?;
    m.ensure_resonance_parameters_loaded()
        .map_err(fail("loading resonance parameters"))?;
    Ok(m)
}

impl CoupledCrossSections {
    /// Every nuclide of every material in `materials` read in full with its
    /// covariance, fielded, and factorized once for all the replicas drawn
    /// under `seed`.
    pub fn new(materials: &[Arc<Material>], seed: u64) -> Result<Self, String> {
        let mut by_label: BTreeMap<String, BTreeMap<String, Arc<Nuclide>>> = BTreeMap::new();
        for m in materials.iter().filter(|m| !m.nuclide_data.is_empty()) {
            let at = by_label.entry(label(m)?).or_default();
            for (name, data) in &m.nuclide_data {
                at.entry(name.clone()).or_insert_with(|| Arc::clone(data));
            }
        }
        let full: Vec<(String, Material)> = by_label
            .iter()
            .map(|(t, nuclides)| Ok((t.clone(), read_in_full(nuclides, t)?)))
            .collect::<Result<_, String>>()?;
        // The cross-material covariance links nuclides across the materials,
        // so it is read from all of them, as the replica weights read it.
        let cross = CrossMaterial::new(&full.iter().map(|(_, m)| m).collect::<Vec<_>>());
        let sampled: BTreeSet<String> = full
            .iter()
            .flat_map(|(_, m)| sampled_names(&resonance_sampling(m)))
            .collect();
        let partners = cross.transport_fields(&sampled);
        let mut out = Self {
            seed,
            cross,
            partners,
            entries: HashMap::new(),
            methods: BTreeMap::new(),
            without: BTreeSet::new(),
            met: BTreeMap::new(),
            floored: 0,
        };
        for (t, m) in &full {
            out.add(m, t);
        }
        Ok(out)
    }

    /// The entries of every nuclide of `full`, read in full at `temperature`.
    fn add(&mut self, full: &Material, temperature: &str) {
        let mut sampling = resonance_sampling(full);
        let (mut fields, without) = transport_fields(full, &sampled_names(&sampling));
        self.without.extend(without);
        for (name, data) in &full.nuclide_data {
            let resonance = match sampling.remove(name) {
                Some(Ok(r)) => {
                    self.methods.insert(name.clone(), r.method());
                    Some(r)
                }
                Some(Err(reason)) => {
                    self.methods
                        .insert(name.clone(), ResonanceMethod::FirstOrder { reason });
                    None
                }
                None => None,
            };
            let field = fields.remove(name).filter(|t| t.field.is_some());
            let sampler = field.as_ref().and_then(|t| {
                let cells = BTreeMap::from([(name.clone(), t.field.clone()?)]);
                Some(Sampler::coupled(&cells, &[], &self.cross, &self.partners))
            });
            let entry = (field.is_some() || resonance.is_some()).then(|| {
                Arc::new(Entry {
                    full: Arc::clone(data),
                    field,
                    sampler,
                    resonance,
                })
            });
            self.entries
                .insert((name.clone(), temperature.to_string()), entry);
        }
    }

    /// What every replica draws for `name` in `material`, reading it in full
    /// the first time a replica meets it.
    fn entry(&mut self, name: &str, material: &Material) -> Result<Option<Arc<Entry>>, String> {
        let temperature = label(material)?;
        let key = (name.to_string(), temperature.clone());
        if !self.entries.contains_key(&key) {
            let held =
                BTreeMap::from([(name.to_string(), Arc::clone(&material.nuclide_data[name]))]);
            let full = read_in_full(&held, &temperature)?;
            self.add(&full, &temperature);
        }
        Ok(self.entries[&key].clone())
    }

    /// Replica `replica`'s view of the data.
    pub fn replica(&mut self, replica: u64) -> Replica<'_> {
        Replica {
            library: self,
            replica,
            done: HashMap::new(),
            made: std::collections::HashSet::new(),
            full: HashMap::new(),
        }
    }

    /// Fill in what the cross-section source perturbed in `material_id`'s
    /// report: the nuclides drawn, those without covariance, how each
    /// resolved resonance range was taken, the joint draws and lognormal
    /// limits of the nuclides drawn, and the values floored.
    pub fn report(&self, material_id: u32, info: &mut Info) {
        let names = self.met.get(&material_id).cloned().unwrap_or_default();
        let entries: Vec<(&String, &Entry)> = self
            .entries
            .iter()
            .filter(|((n, _), _)| names.contains(n))
            .filter_map(|((n, _), e)| Some((n, e.as_deref()?)))
            .collect();
        info.perturbed = entries.iter().map(|(n, _)| (*n).clone()).collect();
        info.no_covariance_data = names.intersection(&self.without).cloned().collect();
        info.resonance_parameters = self
            .methods
            .iter()
            .filter(|(n, _)| names.contains(*n))
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect();
        let samplers = entries.iter().filter_map(|(_, e)| e.sampler.as_ref());
        let (mut joint, mut lognormal): (
            BTreeMap<String, JointAdjustment>,
            BTreeMap<String, LognormalLimit>,
        ) = Default::default();
        for s in samplers {
            joint.extend(s.joint_adjustments());
            lognormal.extend(s.lognormal_limits());
        }
        info.cross_material_adjusted = joint;
        info.lognormal_not_carried = lognormal;
        let full: HashMap<&String, &Nuclide> =
            entries.iter().map(|(n, e)| (*n, e.full.as_ref())).collect();
        (info.covariance_source, info.covariance_warnings) =
            yani_transmute::covariance_provenance::provenance(&info.perturbed, |name| {
                let Some(nd) = full.get(&name.to_string()) else {
                    return (None, None);
                };
                let library = nd
                    .data_source
                    .as_deref()
                    .filter(|s| yamc_nuclide::storage::url_cache::is_keyword(s))
                    .map(str::to_string)
                    .or_else(|| nd.library.clone());
                let mat = nd
                    .covariance
                    .as_ref()
                    .and_then(|blocks| blocks.iter().map(|b| b.mat).find(|m| *m > 0));
                (library, mat)
            });
        info.cross_section_points_floored = self.floored;
    }
}

/// One replica's draw, applied to the materials it meets.
pub struct Replica<'a> {
    library: &'a mut CoupledCrossSections,
    replica: u64,
    /// Per nominal nuclide (by address) and temperature, its replacement,
    /// with the nominal held so its address is not reused.
    done: HashMap<(usize, String), (Arc<Nuclide>, Arc<Nuclide>)>,
    /// The addresses of every replacement made, which are not drawn again.
    made: std::collections::HashSet<usize>,
    /// Per nuclide, its full data with this replica's draw applied.
    full: HashMap<Key, Arc<Nuclide>>,
}

impl Replica<'_> {
    /// Put this replica's data in place of every nominal nuclide `material`
    /// holds, and whether anything changed. The material's cross-section
    /// tables are cleared when it did, for the caller to rebuild.
    pub fn apply(&mut self, material: &mut Material) -> Result<bool, String> {
        // A void, or a photon-only material, has no nuclear data to draw.
        if material.nuclide_data.is_empty() {
            return Ok(false);
        }
        let temperature = label(material)?;
        let mut names: Vec<String> = material.nuclide_data.keys().cloned().collect();
        names.sort();
        if let Some(id) = material.get_material_id() {
            self.library
                .met
                .entry(id)
                .or_default()
                .extend(names.iter().cloned());
        }
        let mut changed = false;
        for name in names {
            let nominal = Arc::clone(&material.nuclide_data[&name]);
            let at = Arc::as_ptr(&nominal) as usize;
            if self.made.contains(&at) {
                continue;
            }
            let key = (at, temperature.clone());
            let replacement = match self.done.get(&key) {
                Some((_, r)) => Arc::clone(r),
                None => {
                    let r = self.perturb(&name, &nominal, material, &temperature)?;
                    self.made.insert(Arc::as_ptr(&r) as usize);
                    self.done
                        .insert(key, (Arc::clone(&nominal), Arc::clone(&r)));
                    r
                }
            };
            if !Arc::ptr_eq(&replacement, &nominal) {
                material.nuclide_data.insert(name, replacement);
                changed = true;
            }
        }
        if changed {
            material.invalidate_xs_cache();
        }
        Ok(changed)
    }

    /// `nominal`, `name`'s data in `material`, with this replica's draw.
    fn perturb(
        &mut self,
        name: &str,
        nominal: &Arc<Nuclide>,
        material: &Material,
        temperature: &str,
    ) -> Result<Arc<Nuclide>, String> {
        let Some(entry) = self.library.entry(name, material)? else {
            return Ok(Arc::clone(nominal));
        };
        // Every MT held: the draw applies to it directly, at every
        // temperature it holds, as to the full data.
        if nominal.load_scope.mts.is_none() {
            let (out, floored) = self.draw(&entry, name, nominal, temperature)?;
            self.library.floored += floored;
            return Ok(Arc::new(out));
        }
        let key = (name.to_string(), temperature.to_string());
        let full = match self.full.get(&key) {
            Some(f) => Arc::clone(f),
            None => {
                let (f, floored) = self.draw(&entry, name, &entry.full, temperature)?;
                self.library.floored += floored;
                let f = Arc::new(f);
                self.full.insert(key, Arc::clone(&f));
                f
            }
        };
        let (out, floored) = moved_like(nominal, &entry.full, &full, temperature)
            .map_err(|e| format!("replica {}: {e}", self.replica))?;
        self.library.floored += floored;
        Ok(Arc::new(out))
    }

    /// `data` with this replica's draw of `entry` applied.
    fn draw(
        &self,
        entry: &Entry,
        name: &str,
        data: &Nuclide,
        temperature: &str,
    ) -> Result<(Nuclide, usize), String> {
        let (seed, k) = (self.library.seed, self.replica);
        let draw = entry.sampler.as_ref().map(|s| s.draw(seed, k));
        let (relative, absolute) = draw
            .as_ref()
            .and_then(|d| d.relative(name).zip(d.absolute(name)))
            .unwrap_or((&[], &[]));
        let change = match &entry.resonance {
            Some(r) => {
                let grid = data
                    .energy
                    .as_ref()
                    .and_then(|m| m.get(temperature))
                    .ok_or_else(|| format!("{name}: no energy grid at {temperature} K"))?;
                Some(ResonanceChange {
                    temperature: temperature.to_string(),
                    by_mt: r.changes(seed, k, grid.as_slice())?,
                })
            }
            None => None,
        };
        let unfielded = TransportField {
            reads: BTreeMap::new(),
            derived: BTreeMap::new(),
            field: None,
        };
        let field = entry.field.as_ref().unwrap_or(&unfielded);
        perturbed_nuclide(data, field, relative, absolute, change.as_ref())
            .map_err(|e| format!("replica {k}: {e}"))
    }
}
