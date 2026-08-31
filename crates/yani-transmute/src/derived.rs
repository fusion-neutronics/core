//! Uncertainty on quantities derived from a whole inventory.
//!
//! Issue #557 put a standard deviation on every nuclide density. Activity and
//! decay heat are sums over nuclides, and the sigma of a sum of correlated
//! terms is not the quadrature of their sigmas: every Mn56 atom in an
//! irradiated iron foil came out of an Fe56 atom, so the two densities move
//! against each other and their spreads partly cancel. Adding them in
//! quadrature would double-count a variance that is not there.
//!
//! Nor can the quantity be evaluated once from the mean inventory. Activity
//! and decay heat are linear in N, so that gives the right mean and no spread
//! at all -- an error bar of exactly zero, which reads as a confident result
//! rather than as a missing one.
//!
//! So the ensemble is evaluated once per replica and the spread is taken over
//! the results (issues #520, #558). This module holds only that accumulation.
//! The quantities themselves stay where they are, in `yani-decay`.

use std::collections::{BTreeSet, HashMap, HashSet};

use yani::ChainNuclide;
use yani_decay::DoseQuantity;

use crate::results::TransmutationResults;
use crate::uncertainty::Moments;

/// One derived quantity: the value the run reports, and the ensemble's spread
/// around it.
///
/// `nominal` is always present -- it is the unperturbed run, which happens
/// whether or not uncertainty was asked for. `mean` and `std_dev` are absent
/// below two replicas, because a spread over fewer than two samples is not
/// zero, it is unmeasured, and the two must not be reported the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// The quantity from the unperturbed inventory.
    pub nominal: f64,
    /// The ensemble mean, or `None` below two replicas.
    ///
    /// Worth comparing against `nominal`: for a quantity linear in N the two
    /// agree to within the sampling error, and a gap between them says the
    /// perturbation is biased rather than merely wide.
    pub mean: Option<f64>,
    /// The ensemble's sample standard deviation, or `None` below two replicas.
    pub std_dev: Option<f64>,
    /// How many replicas the ensemble held.
    pub replicas: usize,
}

impl Estimate {
    /// The standard deviation as a fraction of the nominal value.
    ///
    /// `None` when there is no spread to report, or when the nominal value is
    /// zero and a relative figure would mean nothing.
    pub fn relative_std_dev(&self) -> Option<f64> {
        let sigma = self.std_dev?;
        (self.nominal != 0.0).then(|| sigma / self.nominal.abs())
    }
}

/// One photon line: its energy, its emission rate with the ensemble's spread,
/// and how many replicas emitted it at all.
///
/// The set of lines is not the same in every replica. A nuclide that falls
/// below the stepper's density floor in one draw takes its lines out of that
/// draw, so the spectrum is reported over the union and a line missing from a
/// replica is a zero in it -- the same rule the densities follow, and the only
/// one under which two lines' sigmas are computed over the same sample.
///
/// `emitting` is what that rule would otherwise hide. A line present in three
/// replicas of 128 has a mean that is 97% zeros and a sigma that says more
/// about whether the line is there at all than about how bright it is. Without
/// the count the two cases are indistinguishable in the numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineEstimate {
    /// Line energy [eV].
    pub energy: f64,
    /// Emission rate [photons/s], with the ensemble's spread.
    pub estimate: Estimate,
    /// Replicas that emitted this line at a positive rate.
    pub emitting: usize,
}

/// The mean and sample standard deviation of `values`, both absent below two.
///
/// The same [`Moments`] the density ensemble accumulates, so a nuclide's sigma
/// and a decay heat's sigma are one statistic rather than two spellings of it,
/// and identical replicas give exactly zero rather than a few ulps.
fn spread(values: &[f64]) -> (Option<f64>, Option<f64>) {
    let mut moments = Moments::default();
    for &value in values {
        moments.update(value);
    }
    if moments.n < 2 {
        return (None, None);
    }
    (Some(moments.mean), Some(moments.std_dev()))
}

/// Assemble one [`Estimate`] from a nominal value and one value per replica.
fn estimate(nominal: f64, values: &[f64]) -> Estimate {
    let (mean, std_dev) = spread(values);
    Estimate {
        nominal,
        mean,
        std_dev,
        replicas: values.len(),
    }
}

/// One [`Estimate`] per nuclide, over the union of the nuclides that appear.
///
/// A nuclide absent from a replica is a zero in that replica, not a gap: the
/// stepper drops anything at or below its density floor, so a nuclide present
/// in one replica and absent from another differs between them by less than
/// the floor. Folding those in as zeros keeps every nuclide's moments over the
/// same number of samples, which is the same rule the density ensemble follows
/// (`Ensemble::fold_absences`).
///
/// A nuclide the nominal run does not have but some replica does is reported
/// with a zero nominal and a positive sigma. That is the honest reading: the
/// nuclide is produced along a route whose rate is uncertain enough to reach
/// past the floor in some draws.
fn estimate_by_nuclide(
    nominal: &HashMap<String, f64>,
    per_replica: &[HashMap<String, f64>],
) -> HashMap<String, Estimate> {
    let names: HashSet<&String> = nominal
        .keys()
        .chain(per_replica.iter().flat_map(|r| r.keys()))
        .collect();

    let mut values = Vec::with_capacity(per_replica.len());
    names
        .into_iter()
        .map(|name| {
            values.clear();
            values.extend(
                per_replica
                    .iter()
                    .map(|r| r.get(name).copied().unwrap_or(0.0)),
            );
            let nominal = nominal.get(name).copied().unwrap_or(0.0);
            (name.clone(), estimate(nominal, &values))
        })
        .collect()
}

/// One [`LineEstimate`] per line, over the union of the lines that appear.
///
/// Keyed on the energy's bits, exactly as `decay_photon_lines` keys its own
/// merge, so coincident lines land together and a `BTreeSet` walk comes out
/// ascending in energy (line energies are positive, where bit order and
/// numeric order agree).
fn estimate_lines(nominal: &[(f64, f64)], per_replica: &[Vec<(f64, f64)>]) -> Vec<LineEstimate> {
    let mut energies: BTreeSet<u64> = nominal.iter().map(|(e, _)| e.to_bits()).collect();
    for lines in per_replica {
        energies.extend(lines.iter().map(|(e, _)| e.to_bits()));
    }
    let replicas: Vec<HashMap<u64, f64>> = per_replica
        .iter()
        .map(|lines| lines.iter().map(|(e, r)| (e.to_bits(), *r)).collect())
        .collect();
    let nominal: HashMap<u64, f64> = nominal.iter().map(|(e, r)| (e.to_bits(), *r)).collect();

    let mut values = Vec::with_capacity(replicas.len());
    energies
        .into_iter()
        .map(|bits| {
            values.clear();
            values.extend(
                replicas
                    .iter()
                    .map(|r| r.get(&bits).copied().unwrap_or(0.0)),
            );
            LineEstimate {
                energy: f64::from_bits(bits),
                emitting: values.iter().filter(|rate| **rate > 0.0).count(),
                estimate: estimate(nominal.get(&bits).copied().unwrap_or(0.0), &values),
            }
        })
        .collect()
}

/// The volume a quantity expressed per unit of material needs.
///
/// Activity, decay heat and the photon spectrum all count atoms, so all three
/// want it. The contact dose does not: it is a half-space estimate, and a
/// bigger lump of the same material reads the same.
fn require_volume(volume: Option<f64>, quantity: &str) -> Result<f64, String> {
    volume.ok_or_else(|| format!("{quantity} requires material.volume in cm^3"))
}

/// One quantity's per-nuclide breakdown on the nominal inventory, and the same
/// on each replica's.
type Evaluated = (HashMap<String, f64>, Vec<HashMap<String, f64>>);

/// Which per-inventory quantity an accessor is after.
///
/// All three are functions of a whole inventory and all three are folded over
/// the ensemble the same way, so they differ only in which `yani-decay` entry
/// point evaluates them and whether it wants a volume.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Quantity {
    /// Becquerel.
    Activity,
    /// Watts.
    DecayHeat,
    /// Gy/h or Sv/h, depending on `quantity`.
    ///
    /// Unlike the other two this is **not** linear in the atom densities: the
    /// estimate is `(build_up / 2) * (response(E) / mu_material(E)) * S * E`,
    /// and `mu_material` is built from the same densities that supply `S`. The
    /// emitters sit in the numerator and the whole material's attenuation in
    /// the denominator, so a replica that makes more of an emitter also
    /// absorbs more of it. That is the case for evaluating per replica rather
    /// than scaling a nominal answer, not against it.
    ContactDose {
        quantity: DoseQuantity,
        build_up: f64,
    },
}

impl Quantity {
    /// The name this quantity goes by in an error message.
    fn name(self) -> &'static str {
        match self {
            Quantity::Activity => "activity",
            Quantity::DecayHeat => "decay_heat",
            Quantity::ContactDose { .. } => "contact_dose",
        }
    }

    fn by_nuclide(
        self,
        densities: &HashMap<String, f64>,
        volume: Option<f64>,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<HashMap<String, f64>, String> {
        match self {
            Quantity::Activity => Ok(yani_decay::activity_by_nuclide(
                densities,
                require_volume(volume, self.name())?,
                chain,
            )),
            Quantity::DecayHeat => Ok(yani_decay::decay_heat_by_nuclide(
                densities,
                require_volume(volume, self.name())?,
                chain,
            )),
            Quantity::ContactDose { quantity, build_up } => {
                yani_decay::contact_dose_by_nuclide(densities, chain, quantity, build_up)
            }
        }
    }
}

impl TransmutationResults {
    /// Evaluate something on the nominal inventory and on every replica's.
    ///
    /// The volume comes off the stored step material rather than from the
    /// caller, so the nominal value and every replica are scaled by the same
    /// number and their ratio is exactly the density ensemble's. The ensemble
    /// stores densities alone, and asking the caller to supply the volume
    /// separately is an invitation to pass a different one.
    ///
    /// `Ok(None)` when the run carried no uncertainty for this material, which
    /// is the default path.
    fn fold_replicas<T: Clone>(
        &self,
        material_id: u32,
        step: usize,
        evaluate: impl Fn(&HashMap<String, f64>, Option<f64>) -> Result<T, String>,
    ) -> Result<Option<(T, Vec<T>)>, String> {
        // Looked up before the material, so a run that asked for no
        // uncertainty answers "not requested" rather than complaining about a
        // volume it would never have used.
        let Some(ensemble) = self.uncertainty.get(&material_id) else {
            return Ok(None);
        };
        let material = self
            .get_material(material_id, step)
            .ok_or_else(|| format!("no material {material_id} at step {step}"))?;
        let densities = material.get_atoms_per_barn_cm()?;
        let volume = material.volume;
        let nominal = evaluate(&densities, volume)?;

        let per_replica = if step == 0 {
            // Step 0 is the initial composition: an input rather than a
            // result, so every replica starts from it, they all evaluate to
            // the nominal value, and the spread is a measured zero -- not the
            // unmeasured one an empty ensemble would report. The ensemble
            // stores nothing for step 0, so say it here rather than
            // special-casing the statistics downstream.
            vec![nominal.clone(); ensemble.replicas()]
        } else {
            self.uncertainty_inventories(material_id, step)
                .unwrap_or_default()
                .into_iter()
                .map(|inventory| evaluate(inventory, volume))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Some((nominal, per_replica)))
    }

    /// Shared body of the per-nuclide accessors.
    fn derived(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        quantity: Quantity,
    ) -> Result<Option<Evaluated>, String> {
        self.fold_replicas(material_id, step, |densities, volume| {
            quantity.by_nuclide(densities, volume, chain)
        })
    }

    /// The total of a derived quantity, summed within each replica.
    ///
    /// Summing inside the replica and taking the spread of the totals is what
    /// keeps the inter-nuclide correlations; summing the per-nuclide sigmas in
    /// quadrature instead assumes an independence that the resampling exists
    /// to avoid assuming.
    fn derived_total(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        quantity: Quantity,
    ) -> Result<Option<Estimate>, String> {
        let Some((nominal, per_replica)) = self.derived(material_id, step, chain, quantity)? else {
            return Ok(None);
        };
        let totals: Vec<f64> = per_replica.iter().map(yani_decay::total).collect();
        Ok(Some(estimate(yani_decay::total(&nominal), &totals)))
    }

    /// Activity of a material at `step` [Bq], with the ensemble's spread.
    ///
    /// `Ok(None)` when the run carried no uncertainty for this material, which
    /// is the default path. `Err` when the material has no volume, the same
    /// condition `Material::activity` rejects.
    ///
    /// `step` is indexed like [`Self::get_material`]: 0 is the initial
    /// composition, whose spread is zero because it is an input.
    pub fn activity_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Option<Estimate>, String> {
        self.derived_total(material_id, step, chain, Quantity::Activity)
    }

    /// Decay heat of a material at `step` [W], with the ensemble's spread.
    ///
    /// See [`Self::activity_uncertainty`] for the return convention.
    pub fn decay_heat_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Option<Estimate>, String> {
        self.derived_total(material_id, step, chain, Quantity::DecayHeat)
    }

    /// Activity at `step` broken down by nuclide [Bq], each with its spread.
    ///
    /// These do NOT add up to [`Self::activity_uncertainty`] in quadrature, and
    /// are not meant to: see the module documentation.
    pub fn activity_uncertainty_by_nuclide(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Option<HashMap<String, Estimate>>, String> {
        Ok(self
            .derived(material_id, step, chain, Quantity::Activity)?
            .map(|(nominal, per_replica)| estimate_by_nuclide(&nominal, &per_replica)))
    }

    /// Decay heat at `step` broken down by nuclide [W], each with its spread.
    ///
    /// See [`Self::activity_uncertainty_by_nuclide`].
    pub fn decay_heat_uncertainty_by_nuclide(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Option<HashMap<String, Estimate>>, String> {
        Ok(self
            .derived(material_id, step, chain, Quantity::DecayHeat)?
            .map(|(nominal, per_replica)| estimate_by_nuclide(&nominal, &per_replica)))
    }

    /// The decay-photon spectrum at `step`, each line with the ensemble's
    /// spread on its emission rate.
    ///
    /// Ascending in energy, coincident lines summed, exactly as
    /// `Material.decay_photon_spectrum` returns them -- and over the union of
    /// the lines that appear in the nominal run and in any replica. A line a
    /// replica does not emit is a zero in it, and
    /// [`LineEstimate::emitting`](LineEstimate) says how many replicas emitted
    /// it at all, which is the part the zero-fill would otherwise hide.
    ///
    /// See [`Self::activity_uncertainty`] for the return convention.
    pub fn photon_spectrum_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Option<Vec<LineEstimate>>, String> {
        Ok(self
            .fold_replicas(material_id, step, |densities, volume| {
                let volume = require_volume(volume, "decay_photon_spectrum")?;
                Ok(yani_decay::decay_photon_lines(densities, volume, chain))
            })?
            .map(|(nominal, per_replica)| estimate_lines(&nominal, &per_replica)))
    }

    /// Contact dose rate at `step` [Gy/h or Sv/h], with the ensemble's spread.
    ///
    /// The one derived quantity here that is not linear in the atom densities:
    /// the material attenuates its own photons, so the emitters sit in the
    /// numerator and the whole inventory in the denominator. Scaling a nominal
    /// answer by the density spread would miss that entirely; evaluating per
    /// replica gets it for free.
    ///
    /// Needs no volume, unlike the other three. The estimate takes the material
    /// for a half-space, which leaves no distance and no volume in the answer.
    ///
    /// See [`Self::activity_uncertainty`] for the return convention.
    pub fn contact_dose_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        quantity: DoseQuantity,
        build_up: f64,
    ) -> Result<Option<Estimate>, String> {
        self.derived_total(
            material_id,
            step,
            chain,
            Quantity::ContactDose { quantity, build_up },
        )
    }

    /// Contact dose at `step` broken down by nuclide, each with its spread.
    ///
    /// See [`Self::contact_dose_uncertainty`], and
    /// [`Self::activity_uncertainty_by_nuclide`] on why these do not add up to
    /// the total in quadrature.
    pub fn contact_dose_uncertainty_by_nuclide(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        quantity: DoseQuantity,
        build_up: f64,
    ) -> Result<Option<HashMap<String, Estimate>>, String> {
        Ok(self
            .derived(
                material_id,
                step,
                chain,
                Quantity::ContactDose { quantity, build_up },
            )?
            .map(|(nominal, per_replica)| estimate_by_nuclide(&nominal, &per_replica)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty::Ensemble;
    use yamc_materials::material::Material;

    /// Iron with a trace of Co60, which emits two lines and whose element the
    /// attenuation tables know. The intensities are per atom per second, so the
    /// decay constant is already folded in, as the chain format stores them.
    fn cobalt_chain() -> HashMap<String, ChainNuclide> {
        let half_life = 1.663_2e8;
        let lambda = std::f64::consts::LN_2 / half_life;
        let mut chain = HashMap::new();
        chain.insert(
            "Co60".to_string(),
            ChainNuclide {
                name: "Co60".to_string(),
                half_life: Some(half_life),
                half_life_uncertainty: None,
                decay_energy: 2_503_000.0,
                decay_energy_uncertainty: None,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![yani::DecaySource {
                    particle: "photon".to_string(),
                    distribution: yani::DecaySourceDistribution::Discrete {
                        energies: vec![1_173_228.0, 1_332_492.0],
                        intensities: vec![0.9985 * lambda, 0.9998 * lambda],
                    },
                }],
            },
        );
        chain.insert(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                half_life_uncertainty: None,
                decay_energy: 0.0,
                decay_energy_uncertainty: None,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
            },
        );
        chain
    }

    /// Nuclides that all decay identically, so a trade between any two moves
    /// the parts and leaves the total exactly where it was.
    fn chain() -> HashMap<String, ChainNuclide> {
        ["Fe56", "Mn56", "Cr51"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    ChainNuclide {
                        name: name.to_string(),
                        half_life: Some(9284.04),
                        half_life_uncertainty: None,
                        decay_energy: 2_522_640.3,
                        decay_energy_uncertainty: None,
                        reactions: vec![],
                        decays: vec![],
                        fission_yields: None,
                        sources: vec![],
                    },
                )
            })
            .collect()
    }

    fn densities(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
    }

    /// Atom densities straight through, with a volume, which is what the
    /// stepper stores and what a derived quantity needs.
    fn material(pairs: &[(&str, f64)]) -> Material {
        let mut m = Material::new(densities(pairs), "atom", "sum", None).expect("build a material");
        m.volume = Some(2.0);
        m
    }

    /// One material, one step, and `replicas` inventories after it.
    fn results(nominal: &[(&str, f64)], replicas: &[Vec<(&str, f64)>]) -> TransmutationResults {
        let mut results = TransmutationResults::new(vec![1.0], vec![0.0]);
        results.add_initial(7, material(nominal));
        results.add_step(7, material(nominal));
        let mut ensemble = Ensemble::new(1);
        for replica in replicas {
            ensemble.push(vec![densities(replica)]);
        }
        results.uncertainty.insert(7, ensemble);
        results
    }

    /// Six replicas trading atoms between two identically-decaying nuclides at
    /// a fixed sum.
    fn trading() -> TransmutationResults {
        let splits = [1.6, 0.4, 1.3, 0.7, 1.9, 0.1];
        let replicas: Vec<Vec<(&str, f64)>> = splits
            .iter()
            .map(|&x| vec![("Fe56", x), ("Mn56", 2.0 - x)])
            .collect();
        results(&[("Fe56", 1.0), ("Mn56", 1.0)], &replicas)
    }

    /// Four replicas whose inventory genuinely moves, for the checks a
    /// conserved total would pass without saying anything.
    fn varied() -> TransmutationResults {
        let replicas: Vec<Vec<(&str, f64)>> = [0.7, 0.9, 1.1, 1.3]
            .iter()
            .map(|&x| vec![("Fe56", x)])
            .collect();
        results(&[("Fe56", 1.0)], &replicas)
    }

    /// The test the whole module exists to pass.
    ///
    /// The parts move by 71% of themselves and the total does not move at all,
    /// because every atom one nuclide loses the other gains. Quadrature over
    /// the per-nuclide sigmas reports 50% on a quantity whose real spread is
    /// zero, so it fails this by the entire value rather than by a factor.
    ///
    /// The zero is a floating-point zero rather than a bitwise one: the six
    /// totals are each a sum of two rounded products and agree only to the
    /// last bit or so. A part in 1e12 against a quadrature of 0.5 leaves no
    /// room to confuse the two.
    #[test]
    fn a_conserved_total_has_no_spread_however_the_parts_move() {
        let results = trading();
        let chain = chain();

        let total = results
            .activity_uncertainty(7, 1, &chain)
            .expect("a material with a volume")
            .expect("an ensemble");
        assert_eq!(total.replicas, 6);
        assert!(
            total.relative_std_dev().expect("a spread") < 1e-12,
            "the total moved: {total:?}"
        );

        let parts = results
            .activity_uncertainty_by_nuclide(7, 1, &chain)
            .expect("a material with a volume")
            .expect("an ensemble");
        let quadrature: f64 = parts
            .values()
            .map(|e| e.std_dev.expect("a spread").powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            quadrature / total.nominal > 0.3,
            "quadrature should be badly wrong here, and was {quadrature}"
        );
    }

    /// Identical replicas have exactly no spread, to the last bit.
    ///
    /// This is the state a run against cross sections carrying no covariance
    /// lands in, and it is common. A total summed in `HashMap` order would
    /// come out a few ulps apart between two identical inventories and report
    /// a tiny non-zero sigma, which reads as a real one on a log axis.
    #[test]
    fn identical_replicas_have_exactly_no_spread() {
        let one = vec![("Fe56", 1.0), ("Mn56", 0.25), ("Cr51", 0.125)];
        let replicas: Vec<Vec<(&str, f64)>> = (0..6).map(|_| one.clone()).collect();
        let results = results(&one, &replicas);

        let total = results
            .decay_heat_uncertainty(7, 1, &chain())
            .unwrap()
            .unwrap();
        assert_eq!(total.std_dev, Some(0.0));
        assert_eq!(total.mean, Some(total.nominal));
    }

    /// The mean of a linear quantity sits on the nominal value, and the
    /// per-nuclide means do too. A gap would mean the fold is biased.
    #[test]
    fn the_ensemble_mean_of_a_linear_quantity_sits_on_the_nominal() {
        let results = varied();
        let total = results
            .decay_heat_uncertainty(7, 1, &chain())
            .unwrap()
            .unwrap();
        let mean = total.mean.expect("a mean");
        assert!((mean - total.nominal).abs() < 1e-12 * total.nominal);
    }

    /// A nuclide the stepper dropped from one replica is a zero in it, not a
    /// gap, so every nuclide's moments are over the same six samples.
    #[test]
    fn a_nuclide_missing_from_a_replica_is_a_zero_in_it() {
        let mut replicas: Vec<Vec<(&str, f64)>> =
            (0..5).map(|_| vec![("Fe56", 1.0), ("Mn56", 1.0)]).collect();
        replicas.push(vec![("Fe56", 1.0)]); // Mn56 fell below the density floor
        let results = results(&[("Fe56", 1.0), ("Mn56", 1.0)], &replicas);

        let parts = results
            .activity_uncertainty_by_nuclide(7, 1, &chain())
            .unwrap()
            .unwrap();

        let mn = parts["Mn56"];
        assert_eq!(mn.replicas, 6);
        assert!(mn.std_dev.expect("a spread") > 0.0);
        // Five replicas at the nominal value and one at zero: the mean is 5/6
        // of nominal, which it cannot be if the absence were skipped.
        assert!((mn.mean.unwrap() - mn.nominal * 5.0 / 6.0).abs() < 1e-9 * mn.nominal);
        assert_eq!(parts["Fe56"].std_dev, Some(0.0));
    }

    /// A nuclide only some replicas reach still gets an entry: a zero nominal
    /// and a real spread is the honest reading, and dropping it would hide a
    /// product whose production rate is uncertain enough to matter.
    #[test]
    fn a_nuclide_the_nominal_inventory_lacks_reads_a_zero_nominal_and_a_positive_sigma() {
        let replicas = vec![
            vec![("Fe56", 1.0)],
            vec![("Fe56", 1.0)],
            vec![("Fe56", 1.0), ("Mn56", 0.5)],
        ];
        let results = results(&[("Fe56", 1.0)], &replicas);

        let parts = results
            .decay_heat_uncertainty_by_nuclide(7, 1, &chain())
            .unwrap()
            .unwrap();

        let mn = parts["Mn56"];
        assert_eq!(mn.nominal, 0.0);
        assert_eq!(mn.replicas, 3);
        assert!(mn.std_dev.expect("a spread") > 0.0);
        assert_eq!(mn.relative_std_dev(), None, "no relative figure off a zero");
    }

    /// Below two replicas there is no spread to report, and `None` says so.
    /// Reporting zero would read as a quantity known exactly.
    #[test]
    fn below_two_replicas_the_spread_is_absent_not_zero() {
        for count in [0usize, 1] {
            let replicas: Vec<Vec<(&str, f64)>> = (0..count).map(|_| vec![("Fe56", 1.0)]).collect();
            let results = results(&[("Fe56", 1.0)], &replicas);
            let est = results
                .activity_uncertainty(7, 1, &chain())
                .unwrap()
                .unwrap();
            assert_eq!(est.replicas, count);
            assert_eq!(est.mean, None, "{count} replicas");
            assert_eq!(est.std_dev, None, "{count} replicas");
            assert!(est.nominal > 0.0, "the unperturbed run still has a value");
        }
    }

    /// The initial composition is an input, so every replica starts from it and
    /// its spread is zero rather than unmeasured. This is the same answer
    /// `get_nuclide_uncertainty` gives at step 0.
    #[test]
    fn the_initial_composition_has_a_zero_spread_rather_than_no_spread() {
        let results = trading();
        let est = results
            .activity_uncertainty(7, 0, &chain())
            .unwrap()
            .unwrap();
        assert_eq!(est.replicas, 6);
        assert_eq!(est.std_dev, Some(0.0));
        assert_eq!(est.mean, Some(est.nominal));
    }

    /// The default path samples nothing, and an absent ensemble has to be
    /// distinguishable from a measured zero.
    #[test]
    fn a_run_without_uncertainty_reports_none_rather_than_zero() {
        let mut results = TransmutationResults::new(vec![1.0], vec![0.0]);
        results.add_initial(7, material(&[("Fe56", 1.0)]));
        results.add_step(7, material(&[("Fe56", 1.0)]));

        assert_eq!(results.activity_uncertainty(7, 1, &chain()), Ok(None));
        assert_eq!(
            results.decay_heat_uncertainty_by_nuclide(7, 1, &chain()),
            Ok(None)
        );
    }

    /// A derived quantity needs a volume, and saying so beats returning a
    /// number scaled by an assumed one.
    #[test]
    fn a_material_without_a_volume_is_an_error_rather_than_an_assumption() {
        let mut results = trading();
        results.materials.get_mut(&7).unwrap()[1].volume = None;
        let error = results
            .activity_uncertainty(7, 1, &chain())
            .expect_err("no volume");
        assert!(error.contains("volume"), "{error}");
    }

    // --- the photon spectrum -------------------------------------------------

    /// The union rule, and the count that keeps it honest.
    ///
    /// One replica in three drops Co60 below the density floor, so both of its
    /// lines vanish from that draw. They still get an entry, folded with a zero
    /// in the replica that lacks them, and `emitting` reports 2 of 3 -- which
    /// is the difference between a line that is dim and a line that is
    /// sometimes not there.
    #[test]
    fn a_line_only_some_replicas_emit_reports_how_many() {
        let replicas = vec![
            vec![("Fe56", 0.08), ("Co60", 1.0e-9)],
            vec![("Fe56", 0.08), ("Co60", 1.2e-9)],
            vec![("Fe56", 0.08)], // Co60 fell below the floor
        ];
        let results = results(&[("Fe56", 0.08), ("Co60", 1.1e-9)], &replicas);

        let lines = results
            .photon_spectrum_uncertainty(7, 1, &cobalt_chain())
            .unwrap()
            .unwrap();

        assert_eq!(lines.len(), 2, "two Co60 lines, over the union");
        assert!(lines[0].energy < lines[1].energy, "ascending in energy");
        for line in &lines {
            assert_eq!(line.emitting, 2, "one replica emitted neither line");
            assert_eq!(
                line.estimate.replicas, 3,
                "and all three are in the moments"
            );
            assert!(line.estimate.nominal > 0.0);
            assert!(line.estimate.std_dev.expect("a spread") > 0.0);
        }
    }

    /// A line every replica emits at the same rate has no spread, and says so
    /// with a full emitting count rather than a partial one.
    #[test]
    fn a_line_every_replica_emits_has_no_spread() {
        let replicas: Vec<Vec<(&str, f64)>> = (0..4)
            .map(|_| vec![("Fe56", 0.08), ("Co60", 1.0e-9)])
            .collect();
        let results = results(&[("Fe56", 0.08), ("Co60", 1.0e-9)], &replicas);

        let lines = results
            .photon_spectrum_uncertainty(7, 1, &cobalt_chain())
            .unwrap()
            .unwrap();
        for line in &lines {
            assert_eq!(line.emitting, 4);
            assert_eq!(line.estimate.std_dev, Some(0.0));
        }
    }

    // --- the contact dose ----------------------------------------------------

    /// The reason the contact dose has to be evaluated per replica rather than
    /// scaled from a nominal answer.
    ///
    /// Every replica here holds the same material scaled by a different factor.
    /// Activity counts atoms, so it moves with that factor. The contact dose
    /// does not: the material attenuates its own photons, so the emitters are
    /// in the numerator and the whole inventory is in the denominator and the
    /// factor cancels. An implementation that took the density spread and
    /// scaled the nominal dose by it would report the activity's sigma here,
    /// which is wrong by the whole value.
    #[test]
    fn the_contact_dose_does_not_move_when_the_whole_inventory_scales() {
        let replicas: Vec<Vec<(&str, f64)>> = [0.7, 0.9, 1.1, 1.3]
            .iter()
            .map(|f| vec![("Fe56", 0.08 * f), ("Co60", 1.0e-9 * f)])
            .collect();
        let results = results(&[("Fe56", 0.08), ("Co60", 1.0e-9)], &replicas);
        let chain = cobalt_chain();

        let activity = results.activity_uncertainty(7, 1, &chain).unwrap().unwrap();
        let dose = results
            .contact_dose_uncertainty(7, 1, &chain, DoseQuantity::AbsorbedAir, 2.0)
            .unwrap()
            .unwrap();

        assert!(dose.nominal > 0.0, "the fixture has to produce a dose");
        let moved = activity.relative_std_dev().expect("a spread");
        let held = dose.relative_std_dev().expect("a spread");
        // The sample standard deviation of the four scale factors about 1.0.
        let expected = (0.2_f64 / 3.0).sqrt();
        assert!(
            (moved - expected).abs() < 1e-9,
            "activity should carry the density spread, and carried {moved}"
        );
        assert!(
            held < 1e-12,
            "the dose should not have moved, and moved {held}"
        );
    }

    /// The contact dose needs no volume, unlike the other three: it takes the
    /// material for a half-space, so a bigger lump reads the same.
    #[test]
    fn the_contact_dose_needs_no_volume() {
        let replicas: Vec<Vec<(&str, f64)>> = [0.9, 1.1]
            .iter()
            .map(|f| vec![("Fe56", 0.08 * f), ("Co60", 1.0e-9 * f)])
            .collect();
        let mut results = results(&[("Fe56", 0.08), ("Co60", 1.0e-9)], &replicas);
        for material in results.materials.get_mut(&7).unwrap() {
            material.volume = None;
        }
        let chain = cobalt_chain();

        assert!(results
            .contact_dose_uncertainty(7, 1, &chain, DoseQuantity::AbsorbedAir, 2.0)
            .expect("no volume needed")
            .is_some());
        assert!(results
            .activity_uncertainty(7, 1, &chain)
            .expect_err("activity counts atoms")
            .contains("volume"));
        assert!(results
            .photon_spectrum_uncertainty(7, 1, &chain)
            .expect_err("so does the spectrum")
            .contains("volume"));
    }

    /// Activity and decay heat differ by the mean decay energy and the eV->J
    /// factor, and nothing else, so their relative spreads are identical.
    #[test]
    fn activity_and_decay_heat_carry_the_same_relative_spread() {
        // Not `trading()`: its total is conserved, so its only spread is
        // rounding, and two roundings agreeing would say nothing.
        let results = varied();
        let chain = chain();
        let activity = results.activity_uncertainty(7, 1, &chain).unwrap().unwrap();
        let heat = results
            .decay_heat_uncertainty(7, 1, &chain)
            .unwrap()
            .unwrap();
        assert!(heat.nominal > 0.0 && activity.nominal > 0.0);
        let per_decay = 2_522_640.3 * 1.602_176_634e-19;
        assert!(
            (heat.nominal / activity.nominal / per_decay - 1.0).abs() < 1e-12,
            "decay heat is activity times the mean decay energy"
        );
        let (a, h) = (
            activity.relative_std_dev().expect("a spread"),
            heat.relative_std_dev().expect("a spread"),
        );
        assert!((a - h).abs() < 1e-12 * a.max(1e-300), "{a} vs {h}");
    }
}
