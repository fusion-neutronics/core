//! Displacement damage over an irradiation: damage energy per atom and NRT dpa.
//!
//! Asked for with a [`DamageRequest`] on [`crate::transmute_materials`]. Each
//! nuclide's MT=444 damage-energy cross section [eV barn] is folded against
//! the step's spectrum by the same multigroup collapse the reaction rates use
//! (see [`crate::multigroup`]), giving the damage energy it receives per atom
//! per second. Each irradiation step then adds, per element, that rate
//! averaged over the element's own nuclides at the step's composition, times
//! the step's flux magnitude and duration. A cooling step adds nothing.
//!
//! # Per element and in total
//!
//! The dpa of an element `X` is the damage energy deposited per atom of `X`,
//! from `X`'s own nuclides, converted with `X`'s displacement threshold energy
//! by [`yamc_element::displacement::nrt_dpa`]. The material total is the
//! atom-fraction-weighted sum over elements, `sum_X f_X * dpa_X`, so each
//! element's damage energy is converted with its own `E_d` before the elements
//! are combined. This is the elemental superposition of the NRT model, which
//! is defined for a monatomic target (M. J. Norgett, M. T. Robinson,
//! I. M. Torrens, Nucl. Eng. Des. 33 (1975) 50; ASTM E521): each element's
//! recoils are treated as if they slowed down among atoms of their own kind.
//! It reduces to the elemental value for a pure element. What it leaves out
//! is the transfer of recoil energy between elements in a cascade, which the
//! polyatomic displacement functions of D. M. Parkin and C. A. Coulter
//! (J. Nucl. Mater. 101 (1981) 261) describe and a single `E_d` per element
//! cannot. The damage energy total is weighted the same way, so it is the
//! damage energy per atom of the material.
//!
//! # The step's composition
//!
//! A step's composition is taken as the average of its start and end states:
//! the element's rate is `(sum_n N_n D_n)` over its nuclides at both ends
//! divided by its atoms at both ends, which is the trapezoidal rule on the
//! damage deposited and stays defined for an element that appears or burns out
//! within the step. A nuclide that burns out over a step therefore contributes
//! less to it, and nothing to the steps after.
//!
//! # What is not counted, and said so
//!
//! A nuclide whose data carries no MT=444 (or whose data the library does not
//! publish at all, which happens for short-lived products) is not given a
//! damage energy of zero silently: it is listed in
//! [`DisplacementDamage::without_damage_energy`] with the largest atom
//! fraction it reached on an irradiated step. A nuclide of the starting
//! composition without MT=444 is refused instead, because the material being
//! irradiated cannot be assessed without it.
//!
//! Every element of the starting composition needs a displacement threshold
//! energy, from the defaults in [`yamc_element::displacement`] or the request,
//! and is refused otherwise. An element that only appears as a transmutation
//! product (hydrogen and helium from `(n,p)` and `(n,a)`, rhenium from
//! tungsten) and has neither contributes its damage energy to the damage
//! energy totals, but no dpa, and is listed in
//! [`DisplacementDamage::without_displacement_energy`] with the largest atom
//! fraction it reached.
//!
//! # Uncertainty
//!
//! With a `DataUncertainty` on the same call, each replica gets its own
//! damage energy and dpa ([`DisplacementDamage::replicas`]), from three
//! things that move per replica:
//!
//! - **Its composition.** [`accumulate`] runs over the replica's own states,
//!   so a nuclide that burns faster in a replica damages less there.
//! - **Its cross sections.** MT=444 carries no covariance of its own, but it
//!   is the sum `K = Σ_mt σ_mt h_mt` of what each reaction deposits, and the
//!   deposit per reaction `h_mt` is a property of the reaction
//!   ([`yamc_nuclide::deposit`]). A draw moves the damage-energy rate by
//!   `Σ_mt h_mt Δσ_mt` folded against the spectrum. Each nuclide of the
//!   starting composition is loaded whole, its field widened to the partials
//!   the damage energy comes from (elastic and inelastic carry most of it and
//!   are not activation channels), and every replica reads its activation
//!   rates and its damage energy off one draw of that field
//!   ([`crate::covariance_fold::cell_fields_with_damage`]). What the deposits
//!   do not reproduce, fission's share above all, is held at nominal, and
//!   [`crate::uncertainty::Info::damage_energy`] says how much that is.
//! - **Its flux spectrum**, where the spectrum carries a per-bin sigma: the
//!   damage energy's own per-group terms under the replica's flux draw, as
//!   the reaction rates are moved.
//!
//! Held at nominal, and listed in [`crate::uncertainty::Info::not_perturbed`]:
//! the damage energy of transmutation products (a small share of the atoms on
//! any schedule a dpa is asked of), short-range (LB=8) blocks, which average
//! away over a spectrum's groups, and, for a nuclide whose resonance
//! parameters are sampled, the resonance range the parameters would move.
//! The displacement threshold energies are inputs and are not perturbed.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use yamc_element::displacement::{
    check_displacement_energy, default_displacement_energy, nrt_dpa, DisplacementEnergy,
    DisplacementEnergySource,
};
use yamc_element::element::element_symbol_from_nuclide;
use yamc_materials::material::Material;
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::nuclide::get_or_load_nuclide;

use crate::self_shielding::Shielding;
use crate::{MultigroupSpectrum, TransmuteStep};

/// The ENDF MT number of the damage-energy production cross section, in eV
/// barn.
pub const MT_DAMAGE_ENERGY: i32 = 444;

/// A request for displacement damage on a transmutation, with any displacement
/// threshold energies that replace the defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamageRequest {
    displacement_energies: BTreeMap<String, f64>,
}

impl DamageRequest {
    /// Damage with the default displacement threshold energies, replaced per
    /// element by `displacement_energies` (element symbol to `E_d` in eV).
    ///
    /// Errors on a key that is not an element symbol or an energy that is not
    /// positive and finite.
    pub fn new(
        displacement_energies: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Self, String> {
        let mut out = BTreeMap::new();
        for (symbol, energy) in displacement_energies {
            check_displacement_energy(&symbol, energy)?;
            out.insert(symbol, energy);
        }
        Ok(Self {
            displacement_energies: out,
        })
    }

    /// The user's displacement threshold energies, by element symbol [eV].
    pub fn displacement_energies(&self) -> &BTreeMap<String, f64> {
        &self.displacement_energies
    }

    /// The displacement threshold energy for an element: the user's, else the
    /// default, else `None`.
    pub fn displacement_energy(&self, symbol: &str) -> Option<DisplacementEnergy> {
        match self.displacement_energies.get(symbol) {
            Some(&energy_ev) => Some(DisplacementEnergy {
                energy_ev,
                source: DisplacementEnergySource::User,
            }),
            None => default_displacement_energy(symbol),
        }
    }

    /// Refuse a starting composition holding an element with no displacement
    /// threshold energy. Checked before the solve, so the refusal costs
    /// nothing.
    pub(crate) fn check_composition(&self, material: &Material) -> Result<(), String> {
        let mut missing: Vec<String> = elements_of(&material.nuclides)
            .into_iter()
            .filter(|el| self.displacement_energy(el).is_none())
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort();
        let example = missing
            .iter()
            .map(|el| format!("\"{el}\": ..."))
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "no displacement threshold energy is known for {}: neither ASTM E521 nor the \
             OECD-NEA 2015 report (NEA/NSC/DOC(2015)9) gives one, and it is not guessed. \
             Supply it in eV with displacement_energies={{{example}}}.",
            missing.join(", ")
        ))
    }
}

/// Displacement damage for one material over the schedule.
///
/// The series are cumulative and indexed as the material's states: entry 0 is
/// the initial composition and holds zero, and entry `i` is the total after
/// schedule step `i - 1`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DisplacementDamage {
    /// Damage energy per atom of the material [eV].
    pub damage_energy: Vec<f64>,
    /// NRT displacements per atom of the material.
    pub dpa: Vec<f64>,
    /// Per element, damage energy per atom of that element [eV].
    pub element_damage_energy: BTreeMap<String, Vec<f64>>,
    /// Per element with a displacement threshold energy, NRT displacements per
    /// atom of that element.
    pub element_dpa: BTreeMap<String, Vec<f64>>,
    /// The displacement threshold energy each element in `element_dpa` was
    /// converted with, and where it came from.
    pub displacement_energies: BTreeMap<String, DisplacementEnergy>,
    /// Nuclides present on an irradiated step with no MT=444 to fold, each with
    /// the largest atom fraction it reached there. Their damage energy is not
    /// counted.
    pub without_damage_energy: BTreeMap<String, f64>,
    /// Product elements present on an irradiated step with no displacement
    /// threshold energy, each with the largest atom fraction it reached there.
    /// Their damage energy is counted; their dpa is not.
    pub without_displacement_energy: BTreeMap<String, f64>,
    /// Every uncertainty replica's series, when the transmutation also asked
    /// for nuclear-data uncertainty.
    pub replicas: Option<DamageReplicas>,
}

/// The damage series of every uncertainty replica, indexed
/// `[replica][state]` with the states as in [`DisplacementDamage`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamageReplicas {
    /// Damage energy per atom of the material [eV].
    pub damage_energy: Vec<Vec<f64>>,
    /// NRT displacements per atom of the material.
    pub dpa: Vec<Vec<f64>>,
    /// Per element of [`DisplacementDamage::element_damage_energy`].
    pub element_damage_energy: BTreeMap<String, Vec<Vec<f64>>>,
    /// Per element of [`DisplacementDamage::element_dpa`].
    pub element_dpa: BTreeMap<String, Vec<Vec<f64>>>,
}

/// What a material's uncertainty replicas need to move its damage energy.
pub(crate) struct ReplicaDamage<'a> {
    /// Every material's widened nuclides, shared so a nuclide is drawn alike
    /// in each ([`damage_inputs`]).
    pub inputs: &'a crate::covariance_fold::DamageInputs,
    /// Per spectrum of the material, per nuclide of its starting
    /// composition, the damage-energy rate per group at unit flux.
    pub per_group: Vec<HashMap<String, Vec<f64>>>,
}

/// The per-group damage-energy rates [`ReplicaDamage::per_group`] holds, for
/// `material`'s starting composition, whose MT=444 must be loaded.
pub(crate) fn per_group_rates(
    material: &Material,
    spectra: &[MultigroupSpectrum],
    shielding: Option<&Shielding>,
) -> Vec<HashMap<String, Vec<f64>>> {
    let mut names: Vec<String> = material
        .nuclides
        .iter()
        .filter(|&(_, &n)| n > 0.0)
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    spectra
        .iter()
        .map(|s| {
            crate::multigroup::per_group_damage_energy_rates(
                material,
                &s.masses,
                &s.boundaries,
                shielding,
                &names,
            )
            .into_iter()
            .filter_map(|(name, groups)| Some((name, groups?)))
            .collect()
        })
        .collect()
}

/// Every nuclide of every material's starting composition, loaded whole with
/// its covariance at its material's temperature, with the damage energy per
/// reaction of each of its non-redundant partials: what widens its field to
/// its damage energy. Keyed by name and temperature, so a nuclide shared by
/// several materials at one temperature is loaded once and drawn alike.
pub(crate) fn damage_inputs(
    materials: &[&Material],
) -> Result<crate::covariance_fold::DamageInputs, Box<dyn std::error::Error>> {
    use yamc_nuclide::deposit::{deposits, Response};
    let mut wanted: BTreeMap<(String, String), Option<String>> = BTreeMap::new();
    {
        let cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for m in materials {
            let Some(temperature) = crate::temperature_of(m) else {
                continue;
            };
            for (name, &n) in &m.nuclides {
                if n <= 0.0 {
                    continue;
                }
                wanted
                    .entry((name.clone(), temperature.clone()))
                    .or_insert_with(|| {
                        m.nuclide_data
                            .get(name)
                            .and_then(|nd| nd.reload_source())
                            .or_else(|| cfg.get_cross_section(name))
                    });
            }
        }
    }
    let load_one = |((name, temperature), source): ((String, String), Option<String>)| {
        let source =
            source.ok_or_else(|| format!("{name}: no cross section source is configured"))?;
        let scope = LoadScope::full()
            .with_temperatures(Some(HashSet::from([temperature.clone()])))
            .with_covariance(true);
        let nuclide = get_or_load_nuclide(&name, &HashMap::from([(name.clone(), source)]), &scope)
            .map_err(|e| {
                let text = e.to_string();
                format!("{name}: {}", text.lines().next().unwrap_or(&text))
            })?;
        let t = nuclide
            .loaded_temperatures
            .iter()
            .position(|l| *l == temperature)
            .or_else(|| (nuclide.loaded_temperatures.len() == 1).then_some(0));
        let deposits = match t {
            Some(t) => {
                let partials: BTreeSet<i32> = nuclide.reactions[t]
                    .iter()
                    .filter(|(_, r)| !r.redundant)
                    .map(|(mt, _)| *mt)
                    .collect();
                deposits(&nuclide, t, Response::DamageEnergy, &partials)
            }
            None => BTreeMap::new(),
        };
        Ok::<_, String>((
            (name, temperature),
            crate::covariance_fold::DamageDeposits { nuclide, deposits },
        ))
    };
    let loaded: Vec<Result<_, String>> = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            wanted.into_par_iter().map(load_one).collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            wanted.into_iter().map(load_one).collect()
        }
    };
    let mut out = crate::covariance_fold::DamageInputs::new();
    let mut refused = Vec::new();
    for outcome in loaded {
        match outcome {
            Ok((key, d)) => {
                out.insert(key, d);
            }
            Err(message) => refused.push(message),
        }
    }
    if !refused.is_empty() {
        return Err(format!(
            "the cross sections the damage-energy uncertainty draws could not be loaded:\n  {}",
            refused.join("\n  ")
        )
        .into());
    }
    Ok(out)
}

/// The elements of a composition with a positive amount.
fn elements_of(nuclides: &HashMap<String, f64>) -> BTreeSet<String> {
    nuclides
        .iter()
        .filter(|&(_, &n)| n > 0.0)
        .map(|(name, _)| element_symbol_from_nuclide(name))
        .collect()
}

/// The nuclides the damage fold needs data for, loaded into `material` with
/// MT=444 among their reactions.
///
/// Those of the starting composition are required, and an error names any the
/// library cannot supply. A product is folded only if the solve already loaded
/// data for it: one it did not is a daughter the library does not publish, or
/// one with no reactions in the chain, and fetching it here would mean a
/// download per short-lived product for a negligible share of the atoms. It is
/// reported instead.
///
/// A nuclide whose load already covers MT=444 is left as it is. One that does
/// not is reloaded at its own scope plus MT=444, so nothing it holds is
/// dropped and the activation load stays as narrow as before when damage is
/// not asked for.
pub(crate) fn ensure_damage_data_loaded(
    material: &mut Material,
    states: &[Material],
) -> Result<(), Box<dyn std::error::Error>> {
    let composition: HashSet<String> = material
        .nuclides
        .iter()
        .filter(|&(_, &n)| n > 0.0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut present: HashSet<String> = composition.clone();
    for state in states {
        for (name, &n) in &state.nuclides {
            if n > 0.0 && material.nuclide_data.contains_key(name) {
                present.insert(name.clone());
            }
        }
    }
    let temperatures = (!material.temperature().is_empty())
        .then(|| HashSet::from([material.temperature().to_string()]));

    let mut to_load: Vec<(String, LoadScope, Option<String>)> = Vec::new();
    {
        let cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut names: Vec<&String> = present.iter().collect();
        names.sort();
        for name in names {
            let (scope, source) = match material.nuclide_data.get(name) {
                Some(nd) if nd.load_scope.wants_mt(MT_DAMAGE_ENERGY) => continue,
                Some(nd) => {
                    let mut scope = nd.load_scope.clone();
                    if let Some(mts) = scope.mts.as_mut() {
                        mts.insert(MT_DAMAGE_ENERGY);
                    }
                    (
                        scope,
                        nd.reload_source().or_else(|| cfg.get_cross_section(name)),
                    )
                }
                None => (
                    LoadScope::activation(HashSet::from([MT_DAMAGE_ENERGY]))
                        .with_temperatures(temperatures.clone()),
                    cfg.get_cross_section(name),
                ),
            };
            to_load.push((name.clone(), scope, source));
        }
    }

    let load_one = |(name, scope, source): (String, LoadScope, Option<String>)| {
        let Some(source) = source else {
            return Err(format!("{name}: no cross section source is configured"));
        };
        let sources = HashMap::from([(name.clone(), source)]);
        get_or_load_nuclide(&name, &sources, &scope)
            .map(|nd| (name.clone(), nd))
            .map_err(|e| {
                // The first line only; the rest is the library's whole index.
                let text = e.to_string();
                text.lines().next().unwrap_or(&text).to_string()
            })
    };
    let loaded: Vec<_> = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            to_load.into_par_iter().map(load_one).collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            to_load.into_iter().map(load_one).collect()
        }
    };
    let mut refused = Vec::new();
    for outcome in loaded {
        match outcome {
            Ok((name, nd)) => {
                material.nuclide_data.insert(name, nd);
            }
            Err(message) => refused.push(message),
        }
    }
    if !refused.is_empty() {
        refused.sort();
        return Err(format!(
            "damage-energy data could not be loaded for the material's nuclides:\n  {}",
            refused.join("\n  ")
        )
        .into());
    }
    Ok(())
}

/// Displacement damage for one material over its schedule.
///
/// `material` holds the data (with MT=444, see [`ensure_damage_data_loaded`])
/// and the starting composition in atoms/barn-cm, as the collapse reads them;
/// `states` are the solve's states, initial first, one more than `steps`.
/// With `ensemble`, the material's uncertainty replicas, every replica's
/// series too ([`DisplacementDamage::replicas`]).
pub(crate) fn displacement_damage(
    material: &Material,
    states: &[Material],
    spectra: &[MultigroupSpectrum],
    steps: &[TransmuteStep],
    shielding: Option<&Shielding>,
    request: &DamageRequest,
    ensemble: Option<&crate::uncertainty::Ensemble>,
) -> Result<DisplacementDamage, Box<dyn std::error::Error>> {
    // Per spectrum, every nuclide present at either end of a step it drives.
    let mut wanted: Vec<HashSet<String>> = vec![HashSet::new(); spectra.len()];
    for (i, st) in steps.iter().enumerate() {
        if let Some((s, _)) = st.irradiation {
            for state in &states[i..=i + 1] {
                wanted[s].extend(
                    state
                        .nuclides
                        .iter()
                        .filter(|&(_, &n)| n > 0.0)
                        .map(|(name, _)| name.clone()),
                );
            }
        }
    }
    let per_spectrum: Vec<HashMap<String, Option<f64>>> = spectra
        .iter()
        .zip(wanted)
        .map(|(s, names)| {
            let mut names: Vec<String> = names.into_iter().collect();
            names.sort();
            crate::multigroup::damage_energy_rates(
                material,
                &s.masses,
                &s.boundaries,
                1.0,
                shielding,
                &names,
            )
        })
        .collect();

    // A nuclide the material is made of with no MT=444 is a gap in the
    // answer for the very thing being irradiated, not a trace product.
    let mut absent: Vec<&str> = per_spectrum
        .iter()
        .flat_map(|rates| {
            rates
                .iter()
                .filter(|(name, d)| {
                    d.is_none() && material.nuclides.get(*name).is_some_and(|&n| n > 0.0)
                })
                .map(|(name, _)| name.as_str())
        })
        .collect();
    absent.sort_unstable();
    absent.dedup();
    if !absent.is_empty() {
        return Err(format!(
            "the cross section library has no MT=444 damage-energy cross section for {}, \
             which the material is made of, so its displacement damage cannot be \
             computed. Use a library that carries MT=444 for it, or leave it out of the \
             composition.",
            absent.join(", ")
        )
        .into());
    }

    let nominal: Vec<&HashMap<String, f64>> = states.iter().map(|m| &m.nuclides).collect();
    let steps: Vec<(f64, Option<(usize, f64)>)> =
        steps.iter().map(|st| (st.dt, st.irradiation)).collect();
    let mut out = accumulate(&nominal, &steps, &per_spectrum, request);
    if let Some(ensemble) = ensemble.filter(|e| e.replicas() > 0) {
        out.replicas = Some(replicas(
            &out,
            &nominal[0],
            &steps,
            &per_spectrum,
            ensemble,
            request,
        ));
    }
    Ok(out)
}

/// Every replica's series: [`accumulate`] over the replica's own states, with
/// each nuclide's damage-energy rate scaled by the replica's factor for it
/// ([`crate::uncertainty::Ensemble`] keeps them).
fn replicas(
    nominal: &DisplacementDamage,
    initial: &HashMap<String, f64>,
    steps: &[(f64, Option<(usize, f64)>)],
    per_spectrum: &[HashMap<String, Option<f64>>],
    ensemble: &crate::uncertainty::Ensemble,
    request: &DamageRequest,
) -> DamageReplicas {
    let mut out = DamageReplicas {
        element_damage_energy: nominal
            .element_damage_energy
            .keys()
            .map(|el| (el.clone(), Vec::new()))
            .collect(),
        element_dpa: nominal
            .element_dpa
            .keys()
            .map(|el| (el.clone(), Vec::new()))
            .collect(),
        ..Default::default()
    };
    let zeros = vec![0.0; steps.len() + 1];
    for r in 0..ensemble.replicas() {
        let rates: Vec<HashMap<String, Option<f64>>> = per_spectrum
            .iter()
            .enumerate()
            .map(|(s, rates)| {
                rates
                    .iter()
                    .map(|(name, d)| {
                        let f = ensemble.damage_factor(r, s, name);
                        (name.clone(), d.map(|d| d * f))
                    })
                    .collect()
            })
            .collect();
        let mut states = vec![initial];
        states.extend(ensemble.replica_inventories(r));
        let d = accumulate(&states, steps, &rates, request);
        for (el, series) in out.element_damage_energy.iter_mut() {
            series.push(d.element_damage_energy.get(el).unwrap_or(&zeros).clone());
        }
        for (el, series) in out.element_dpa.iter_mut() {
            series.push(d.element_dpa.get(el).unwrap_or(&zeros).clone());
        }
        out.damage_energy.push(d.damage_energy);
        out.dpa.push(d.dpa);
    }
    out
}

/// One state's per-element sums.
#[derive(Default, Clone, Copy)]
struct ElementSums {
    /// Atoms of the element [atoms/barn-cm].
    atoms: f64,
    /// `sum_n N_n D_n` over its nuclides with damage data [eV/s per barn-cm at
    /// unit flux].
    damage: f64,
}

/// The arithmetic of [`displacement_damage`], on plain numbers.
///
/// `states[i]` is the composition (atoms/barn-cm) before step `i` and
/// `states[i + 1]` after it; `steps[i]` is `(dt, Some((spectrum, flux)))` or
/// `(dt, None)` for cooling; `per_spectrum[s][nuclide]` is the damage-energy
/// rate per atom at unit flux, `None` for no MT=444.
pub(crate) fn accumulate(
    states: &[&HashMap<String, f64>],
    steps: &[(f64, Option<(usize, f64)>)],
    per_spectrum: &[HashMap<String, Option<f64>>],
    request: &DamageRequest,
) -> DisplacementDamage {
    let n_states = steps.len() + 1;
    let mut out = DisplacementDamage {
        damage_energy: vec![0.0; n_states],
        dpa: vec![0.0; n_states],
        ..Default::default()
    };

    // The elements of the breakdown: the starting composition's, and any
    // product present on an irradiated step.
    // Every sum below runs in name order. A `HashMap`'s order differs between
    // two maps holding the same entries, so summing in it gave one material
    // a dpa that differed in its last bits from run to run, and alone from
    // beside another material.
    let mut elements: BTreeSet<String> = elements_of(states[0]);
    for (i, (_, irradiation)) in steps.iter().enumerate() {
        if irradiation.is_some() {
            elements.extend(elements_of(states[i]));
            elements.extend(elements_of(states[i + 1]));
        }
    }
    for el in &elements {
        out.element_damage_energy
            .insert(el.clone(), vec![0.0; n_states]);
        match request.displacement_energy(el) {
            Some(ed) => {
                out.displacement_energies.insert(el.clone(), ed);
                out.element_dpa.insert(el.clone(), vec![0.0; n_states]);
            }
            None => {
                out.without_displacement_energy.insert(el.clone(), 0.0);
            }
        }
    }

    // Per state and spectrum, the element sums, noting what had no data.
    let sums = |state: &HashMap<String, f64>,
                rates: &HashMap<String, Option<f64>>,
                missing: &mut BTreeMap<String, f64>|
     -> (HashMap<String, ElementSums>, f64) {
        let mut nuclides: Vec<(&String, f64)> = state.iter().map(|(k, &n)| (k, n)).collect();
        nuclides.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let total: f64 = nuclides.iter().map(|&(_, n)| n.max(0.0)).sum();
        let mut by_element: HashMap<String, ElementSums> = HashMap::new();
        for (name, n) in nuclides {
            if n <= 0.0 {
                continue;
            }
            let entry = by_element
                .entry(element_symbol_from_nuclide(name))
                .or_default();
            entry.atoms += n;
            match rates.get(name).copied().flatten() {
                Some(d) => entry.damage += n * d,
                None => {
                    let fraction = n / total;
                    let worst = missing.entry(name.clone()).or_insert(0.0);
                    *worst = worst.max(fraction);
                }
            }
        }
        (by_element, total)
    };

    let mut missing = BTreeMap::new();
    for (i, &(dt, irradiation)) in steps.iter().enumerate() {
        let (mut step_energy, mut step_dpa) = (0.0, 0.0);
        let mut step_element_energy: HashMap<String, f64> = HashMap::new();
        if let Some((s, flux)) = irradiation {
            let (start, total_start) = sums(states[i], &per_spectrum[s], &mut missing);
            let (end, total_end) = sums(states[i + 1], &per_spectrum[s], &mut missing);
            let total = total_start + total_end;
            for el in &elements {
                let a = start.get(el).copied().unwrap_or_default();
                let b = end.get(el).copied().unwrap_or_default();
                let atoms = a.atoms + b.atoms;
                if atoms <= 0.0 {
                    continue;
                }
                let energy = flux * dt * (a.damage + b.damage) / atoms;
                let fraction = atoms / total;
                step_element_energy.insert(el.clone(), energy);
                step_energy += fraction * energy;
                match out.displacement_energies.get(el) {
                    Some(ed) => step_dpa += fraction * nrt_dpa(energy, ed.energy_ev),
                    None => {
                        let worst = out.without_displacement_energy.get_mut(el).unwrap();
                        let at_end = b.atoms / total_end.max(f64::MIN_POSITIVE);
                        let at_start = a.atoms / total_start.max(f64::MIN_POSITIVE);
                        *worst = worst.max(at_start).max(at_end);
                    }
                }
            }
        }
        out.damage_energy[i + 1] = out.damage_energy[i] + step_energy;
        out.dpa[i + 1] = out.dpa[i] + step_dpa;
        for (el, series) in out.element_damage_energy.iter_mut() {
            series[i + 1] = series[i] + step_element_energy.get(el).copied().unwrap_or(0.0);
        }
        for (el, series) in out.element_dpa.iter_mut() {
            let ed = out.displacement_energies[el].energy_ev;
            let energy = step_element_energy.get(el).copied().unwrap_or(0.0);
            series[i + 1] = series[i] + nrt_dpa(energy, ed);
        }
    }
    out.without_damage_energy = missing;
    out
}

/// Which damage series an uncertainty getter reads.
#[derive(Clone, Copy)]
enum Series {
    Dpa,
    DamageEnergy,
}

impl crate::results::TransmutationResults {
    /// NRT dpa at `step` (0 the initial composition, which has none), with
    /// the nuclear-data spread on it: the material total, or with `element`
    /// that element's own.
    ///
    /// `Ok(None)` when the transmutation did not ask for both displacement
    /// damage and nuclear-data uncertainty. See [`crate::damage`] for what the
    /// replicas move.
    pub fn dpa_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        element: Option<&str>,
    ) -> Result<Option<crate::derived::Estimate>, String> {
        self.damage_estimate(material_id, step, element, Series::Dpa)
    }

    /// Damage energy per atom [eV] at `step`, with the nuclear-data spread on
    /// it, as [`Self::dpa_uncertainty`].
    pub fn damage_energy_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        element: Option<&str>,
    ) -> Result<Option<crate::derived::Estimate>, String> {
        self.damage_estimate(material_id, step, element, Series::DamageEnergy)
    }

    fn damage_estimate(
        &self,
        material_id: u32,
        step: usize,
        element: Option<&str>,
        series: Series,
    ) -> Result<Option<crate::derived::Estimate>, String> {
        let Some(damage) = self.get_displacement_damage(material_id) else {
            return Ok(None);
        };
        let Some(replicas) = &damage.replicas else {
            return Ok(None);
        };
        let (nominal, samples) = match (series, element) {
            (Series::Dpa, None) => (&damage.dpa, &replicas.dpa),
            (Series::DamageEnergy, None) => (&damage.damage_energy, &replicas.damage_energy),
            (Series::Dpa, Some(el)) => (
                damage.element_dpa.get(el).ok_or_else(|| {
                    format!(
                        "no dpa for element {el:?}; elements with dpa: {}",
                        damage
                            .element_dpa
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?,
                &replicas.element_dpa[el],
            ),
            (Series::DamageEnergy, Some(el)) => (
                damage.element_damage_energy.get(el).ok_or_else(|| {
                    format!(
                        "no damage energy for element {el:?}; elements present: {}",
                        damage
                            .element_damage_energy
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?,
                &replicas.element_damage_energy[el],
            ),
        };
        let nominal = *nominal
            .get(step)
            .ok_or_else(|| format!("no step {step}: the schedule has {} states", nominal.len()))?;
        let values: Vec<f64> = samples.iter().map(|r| r[step]).collect();
        Ok(Some(crate::derived::estimate(nominal, &values, None, None)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, f64)]) -> HashMap<String, f64> {
        entries.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn rates(entries: &[(&str, Option<f64>)]) -> HashMap<String, Option<f64>> {
        entries.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1e-300), "{a} vs {b}");
    }

    /// The issue's hand check, as arithmetic: 2.05e-9 eV/s per atom of
    /// tungsten for five minutes at E_d = 90 eV is 2.7e-9 dpa.
    #[test]
    fn pure_element_is_rate_times_time_over_two_e_d() {
        let w = map(&[("W186", 1.0)]);
        let states = [&w, &w, &w];
        let d = rates(&[("W186", Some(2.05e-9))]);
        let out = accumulate(
            &states,
            &[(300.0, Some((0, 1.0))), (600.0, None)],
            &[d],
            &DamageRequest::default(),
        );
        close(out.damage_energy[1], 2.05e-9 * 300.0);
        close(out.dpa[1], 0.8 * 2.05e-9 * 300.0 / 180.0);
        assert!((out.dpa[1] - 2.733e-9).abs() < 1e-12);
        // Cooling adds nothing.
        assert_eq!(out.dpa[2], out.dpa[1]);
        assert_eq!(out.element_dpa["W"], out.dpa);
        assert_eq!(
            out.displacement_energies["W"].source,
            DisplacementEnergySource::AstmE521
        );
        assert!(out.without_damage_energy.is_empty());
    }

    /// Two elements: each converted with its own E_d, the total weighted by
    /// atom fraction.
    #[test]
    fn compound_total_is_the_atom_fraction_weighted_sum() {
        let s = map(&[("Fe56", 3.0), ("Cr52", 1.0)]);
        let states = [&s, &s];
        let d = rates(&[("Fe56", Some(4.0)), ("Cr52", Some(8.0))]);
        let request = DamageRequest::new([("Cr".to_string(), 50.0)]).unwrap();
        let out = accumulate(&states, &[(2.0, Some((0, 10.0)))], &[d], &request);
        close(out.element_damage_energy["Fe"][1], 80.0);
        close(out.element_damage_energy["Cr"][1], 160.0);
        close(out.element_dpa["Fe"][1], 0.8 * 80.0 / 80.0);
        close(out.element_dpa["Cr"][1], 0.8 * 160.0 / 100.0);
        close(out.damage_energy[1], 0.75 * 80.0 + 0.25 * 160.0);
        close(
            out.dpa[1],
            0.75 * out.element_dpa["Fe"][1] + 0.25 * out.element_dpa["Cr"][1],
        );
        assert_eq!(
            out.displacement_energies["Cr"],
            DisplacementEnergy {
                energy_ev: 50.0,
                source: DisplacementEnergySource::User
            }
        );
    }

    /// The same composition gives the same dpa to the bit whichever order
    /// its maps happen to iterate in. Each `HashMap` is seeded on its own, so
    /// fresh maps of one set of entries walk it in different orders, and a sum
    /// taken in that order moved in the last bits between a material
    /// transmuted alone and the same material beside another.
    #[test]
    fn the_sums_do_not_depend_on_map_order() {
        // Magnitudes far apart and many elements, so a reordered sum rounds
        // differently.
        let entries: Vec<(String, f64)> = [
            "Fe56", "Fe54", "Fe57", "Mn55", "Mn53", "Mn56", "Cr52", "Cr53", "Cr51", "Cr50", "V51",
            "V50", "Ti48", "H1", "H2", "H3", "He3", "He4", "Co59", "Ni58",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.to_string(),
                0.1 + 1.0e-3_f64.powi(i as i32 % 7) * (i as f64 + 0.37),
            )
        })
        .collect();
        let damage: Vec<(String, Option<f64>)> = entries
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), Some(1.0 / (3.0 + i as f64))))
            .collect();
        let run = |reverse: bool| {
            let mut order = entries.clone();
            let mut d = damage.clone();
            if reverse {
                order.reverse();
                d.reverse();
            }
            let start: HashMap<String, f64> = order.iter().cloned().collect();
            let end: HashMap<String, f64> =
                order.iter().map(|(k, v)| (k.clone(), v * 0.97)).collect();
            let rates: HashMap<String, Option<f64>> = d.into_iter().collect();
            let out = accumulate(
                &[&start, &end],
                &[(86400.0, Some((0, 1.06e14)))],
                &[rates],
                &DamageRequest::default(),
            );
            (out.dpa[1].to_bits(), out.damage_energy[1].to_bits())
        };
        let first = run(false);
        for i in 0..64 {
            assert_eq!(run(i % 2 == 1), first, "run {i} summed in another order");
        }
    }

    /// A step over which a nuclide burns out contributes less than the one
    /// before it, and the step after it contributes only the survivor's.
    #[test]
    fn burnout_reduces_the_contribution() {
        let start = map(&[("Fe54", 1.0), ("Fe56", 1.0)]);
        let half = map(&[("Fe54", 0.0), ("Fe56", 1.0)]);
        let states = [&start, &start, &half, &half];
        let d = rates(&[("Fe54", Some(10.0)), ("Fe56", Some(2.0))]);
        let out = accumulate(
            &states,
            &[(1.0, Some((0, 1.0))); 3],
            &[d],
            &DamageRequest::default(),
        );
        let e = &out.element_damage_energy["Fe"];
        let steps: Vec<f64> = e.windows(2).map(|w| w[1] - w[0]).collect();
        close(steps[0], 6.0);
        // Trapezoid over (Fe54 + Fe56) at the start and Fe56 alone at the end.
        close(steps[1], (10.0 + 2.0 + 2.0) / 3.0);
        close(steps[2], 2.0);
    }

    /// A product without MT=444 is reported with its atom fraction, and a
    /// product element without E_d is reported rather than refused.
    #[test]
    fn gaps_are_reported() {
        let start = map(&[("W186", 1.0)]);
        let end = map(&[("W186", 0.98), ("W187", 0.01), ("H1", 0.01)]);
        let states = [&start, &end];
        let d = rates(&[("W186", Some(1.0)), ("H1", Some(1.0))]);
        let out = accumulate(
            &states,
            &[(1.0, Some((0, 1.0)))],
            &[d],
            &DamageRequest::default(),
        );
        close(out.without_damage_energy["W187"], 0.01);
        close(out.without_displacement_energy["H"], 0.01);
        assert!(!out.element_dpa.contains_key("H"));
        assert!(out.element_damage_energy.contains_key("H"));
    }

    /// Each replica's series is the nominal arithmetic on its own states with
    /// each nuclide's rate scaled by its factor; a nuclide with no factor is
    /// at nominal.
    #[test]
    fn replicas_scale_by_their_factors_on_their_own_states() {
        let start = map(&[("Fe56", 1.0), ("Cr52", 1.0)]);
        let d = rates(&[("Fe56", Some(4.0)), ("Cr52", Some(8.0))]);
        let steps = [(2.0, Some((0, 10.0)))];
        let request = DamageRequest::default();
        let nominal = accumulate(&[&start, &start], &steps, &[d.clone()], &request);

        let mut ensemble = crate::uncertainty::Ensemble::new(1);
        ensemble.push(vec![start.clone()]);
        ensemble.push(vec![map(&[("Fe56", 1.0), ("Cr52", 0.5)])]);
        ensemble.damage_factors = vec![
            vec![HashMap::from([("Fe56".to_string(), 1.5)])],
            vec![HashMap::new()],
        ];
        let out = replicas(&nominal, &start, &steps, &[d], &ensemble, &request);

        close(out.element_damage_energy["Fe"][0][1], 1.5 * 80.0);
        close(out.element_damage_energy["Cr"][0][1], 160.0);
        // The second replica burned half its chromium by the end of the step.
        let second = accumulate(
            &[&start, &map(&[("Fe56", 1.0), ("Cr52", 0.5)])],
            &steps,
            &[rates(&[("Fe56", Some(4.0)), ("Cr52", Some(8.0))])],
            &request,
        );
        assert_eq!(out.dpa[1], second.dpa);
        assert_eq!(out.damage_energy[1], second.damage_energy);
        assert_eq!(out.element_dpa["Fe"].len(), 2);
    }

    #[test]
    fn a_composition_element_without_e_d_is_refused() {
        let mut m = Material::new(
            HashMap::from([("Li6".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        m.nuclides.insert("Li6".to_string(), 0.05);
        let err = DamageRequest::default().check_composition(&m).unwrap_err();
        assert!(
            err.contains("Li") && err.contains("displacement_energies"),
            "{err}"
        );
        let ok = DamageRequest::new([("Li".to_string(), 25.0)]).unwrap();
        assert!(ok.check_composition(&m).is_ok());
    }

    #[test]
    fn bad_overrides_are_refused() {
        assert!(DamageRequest::new([("Iron".to_string(), 40.0)]).is_err());
        assert!(DamageRequest::new([("Fe".to_string(), -40.0)]).is_err());
    }
}
