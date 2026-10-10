//! Uncertainty on quantities derived from a whole inventory.
//!
//! Every nuclide density carries a standard deviation. Activity and
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
//! the results. This module holds only that accumulation.
//! The quantities themselves stay where they are, in `yani-decay`.

use std::collections::{BTreeSet, HashMap, HashSet};

use yani::ChainNuclide;
use yani_decay::DoseQuantity;

use crate::results::TransmutationResults;
use crate::uncertainty::{Moments, PhotonCorrelation};

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
    ///
    /// The lower end of the range [`Self::std_dev_correlated`] closes: where
    /// the `decay_photon_lines` source is drawn, every correlation the decay
    /// evaluation leaves unstated between a nuclide's photon intensities is
    /// taken as zero here, which is the evaluation read literally.
    pub std_dev: Option<f64>,
    /// The ensemble's sample standard deviation with those correlations taken
    /// as one instead: within each nuclide, the lines of a spectrum, the
    /// spectrum's normalisation and its lines, and its gamma and x-ray
    /// spectra all move together. The upper end of the range, `None` below two
    /// replicas.
    ///
    /// The same inventories as `std_dev`, so the two differ by the line data
    /// alone. Equal to `std_dev` for a quantity no photon intensity enters
    /// (activity), or when the `decay_photon_lines` source is not drawn.
    /// Every non-negative correlation between the two ends gives a spread
    /// between them; negative ones are not considered, since the common part
    /// they leave open is a shared normalisation, which moves every line it
    /// scales the same way.
    pub std_dev_correlated: Option<f64>,
    /// The standard error of `std_dev`, how far another ensemble of the same
    /// size could put it, or `None` below four replicas. It allows for a heavy
    /// tail (see [`crate::uncertainty::std_dev_standard_error`]).
    pub std_dev_standard_error: Option<f64>,
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

    /// [`Self::std_dev_correlated`] as a fraction of the nominal value, under
    /// the same conditions as [`Self::relative_std_dev`].
    pub fn relative_std_dev_correlated(&self) -> Option<f64> {
        let sigma = self.std_dev_correlated?;
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
///
/// With the `decay_photon_lines` source on, a replica draws each line's energy
/// as well as its intensity, so a line is identified across replicas by its
/// nominal energy, and `energy_estimate` carries the spread of the energy it
/// was drawn at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineEstimate {
    /// Nominal line energy [eV], the one the evaluation states.
    pub energy: f64,
    /// Emission rate [photons/s], with the ensemble's spread.
    pub estimate: Estimate,
    /// The energy each replica emitted the line at [eV], with the spread,
    /// over the emitting replicas only: a replica that does not emit a line
    /// gives it no energy. Where coincident lines are summed it is their
    /// rate-weighted mean. `nominal` is `energy`.
    pub energy_estimate: Estimate,
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

/// Assemble one [`Estimate`] from a nominal value and one value per replica,
/// and the same replicas evaluated with the photon intensities fully
/// correlated, where that end differs.
fn estimate(nominal: f64, values: &[f64], correlated: Option<&[f64]>) -> Estimate {
    let (mean, std_dev) = spread(values);
    Estimate {
        nominal,
        mean,
        std_dev,
        std_dev_correlated: correlated.map_or(std_dev, |c| spread(c).1),
        std_dev_standard_error: crate::uncertainty::std_dev_standard_error(values),
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
///
/// `correlated` is the same replicas with the photon intensities fully
/// correlated, where that end differs.
fn estimate_by_nuclide(
    nominal: &HashMap<String, f64>,
    per_replica: &[HashMap<String, f64>],
    correlated: Option<&[HashMap<String, f64>]>,
) -> HashMap<String, Estimate> {
    let names: HashSet<&String> = nominal
        .keys()
        .chain(per_replica.iter().flat_map(|r| r.keys()))
        .chain(correlated.into_iter().flatten().flat_map(|r| r.keys()))
        .collect();

    let column = |replicas: &[HashMap<String, f64>], name: &String| -> Vec<f64> {
        replicas
            .iter()
            .map(|r| r.get(name).copied().unwrap_or(0.0))
            .collect()
    };
    names
        .into_iter()
        .map(|name| {
            let values = column(per_replica, name);
            let correlated = correlated.map(|c| column(c, name));
            let nominal = nominal.get(name).copied().unwrap_or(0.0);
            (
                name.clone(),
                estimate(nominal, &values, correlated.as_deref()),
            )
        })
        .collect()
}

/// One [`LineEstimate`] per line, over the union of the lines that appear.
///
/// Keyed on the nominal energy's bits, exactly as `decay_photon_lines` keys
/// its own merge, so coincident lines land together and a `BTreeSet` walk
/// comes out ascending in energy (line energies are positive, where bit order
/// and numeric order agree).
///
/// `correlated` is the same replicas with the photon intensities fully
/// correlated, where that end differs. A line's energy spread is reported for
/// the independent end only: its energy is drawn from the same deviate at both.
fn estimate_lines(
    nominal: &[TracedLine],
    per_replica: &[Vec<TracedLine>],
    correlated: Option<&[Vec<TracedLine>]>,
) -> Vec<LineEstimate> {
    let mut energies: BTreeSet<u64> = nominal.iter().map(|l| l.energy.to_bits()).collect();
    for lines in per_replica.iter().chain(correlated.into_iter().flatten()) {
        energies.extend(lines.iter().map(|l| l.energy.to_bits()));
    }
    let index = |replicas: &[Vec<TracedLine>]| -> Vec<HashMap<u64, f64>> {
        replicas
            .iter()
            .map(|lines| lines.iter().map(|l| (l.energy.to_bits(), l.rate)).collect())
            .collect()
    };
    let correlated = correlated.map(index);
    let replicas: Vec<HashMap<u64, &TracedLine>> = per_replica
        .iter()
        .map(|lines| lines.iter().map(|l| (l.energy.to_bits(), l)).collect())
        .collect();
    let nominal: HashMap<u64, f64> = nominal
        .iter()
        .map(|l| (l.energy.to_bits(), l.rate))
        .collect();

    let mut rates = Vec::with_capacity(replicas.len());
    let mut drawn = Vec::with_capacity(replicas.len());
    energies
        .into_iter()
        .map(|bits| {
            rates.clear();
            drawn.clear();
            for replica in &replicas {
                match replica.get(&bits) {
                    Some(line) => {
                        rates.push(line.rate);
                        if line.rate > 0.0 {
                            drawn.push(line.drawn_energy);
                        }
                    }
                    None => rates.push(0.0),
                }
            }
            let energy = f64::from_bits(bits);
            let correlated: Option<Vec<f64>> = correlated.as_ref().map(|c| {
                c.iter()
                    .map(|r| r.get(&bits).copied().unwrap_or(0.0))
                    .collect()
            });
            LineEstimate {
                energy,
                emitting: drawn.len(),
                estimate: estimate(
                    nominal.get(&bits).copied().unwrap_or(0.0),
                    &rates,
                    correlated.as_deref(),
                ),
                energy_estimate: estimate(energy, &drawn, None),
            }
        })
        .collect()
}

/// Atoms per barn-cm times cm^3 is 1e24 atoms per cm^3 times cm^3.
const BARN_PER_CM_SQ: f64 = 1.0e24;

/// One line of an inventory's spectrum, identified by its nominal energy.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TracedLine {
    /// The nominal energy [eV], the key a line is matched on across replicas.
    energy: f64,
    /// Photons per second.
    rate: f64,
    /// The rate-weighted energy the line was emitted at in this replica [eV];
    /// `energy` itself when nothing moved it.
    drawn_energy: f64,
}

/// An inventory's decay photon lines, as `yani_decay::decay_photon_lines`
/// gives them, with each line keyed on its energy in `nominal` and carrying
/// the energy `drawn` emits it at.
///
/// `drawn` is a replica's chain, the same sources as `nominal` with values
/// drawn, so the two are walked side by side. With `drawn` equal to `nominal`
/// the rates are those of `decay_photon_lines` to the last bit: the lines of
/// one nuclide are merged in the order `ChainNuclide::photon_lines` merges
/// them, then scaled by its atom count, in nuclide-name order.
fn traced_photon_lines(
    densities: &HashMap<String, f64>,
    volume: f64,
    nominal: &HashMap<String, ChainNuclide>,
    drawn: &HashMap<String, ChainNuclide>,
) -> Vec<TracedLine> {
    let mut names: Vec<&String> = densities.keys().collect();
    names.sort();
    // (rate, rate times the drawn energy's offset from the nominal one), keyed
    // on the nominal energy's bits. The offset rather than the energy, so a
    // line nothing moved comes back at its nominal energy exactly, not a
    // rounding off it.
    let mut lines: std::collections::BTreeMap<u64, (f64, f64)> = Default::default();
    for name in names {
        let atoms = densities[name] * BARN_PER_CM_SQ * volume;
        if atoms <= 0.0 {
            continue;
        }
        let (Some(n), Some(d)) = (nominal.get(name), drawn.get(name)) else {
            continue;
        };
        for (energy, intensity, weighted) in traced_nuclide_lines(n, d) {
            if intensity > 0.0 {
                let entry = lines.entry(energy.to_bits()).or_insert((0.0, 0.0));
                entry.0 += atoms * intensity;
                entry.1 += atoms * weighted;
            }
        }
    }
    lines
        .into_iter()
        .map(|(bits, (rate, weighted))| {
            let energy = f64::from_bits(bits);
            TracedLine {
                energy,
                rate,
                drawn_energy: energy + weighted / rate,
            }
        })
        .collect()
}

/// One nuclide's lines as `(nominal energy, drawn intensity, drawn intensity
/// times the drawn energy's offset from the nominal one)`, ascending in
/// nominal energy, coincident ones summed in source order, which is the order
/// `ChainNuclide::photon_lines` sums in.
fn traced_nuclide_lines(nominal: &ChainNuclide, drawn: &ChainNuclide) -> Vec<(f64, f64, f64)> {
    fn photons(cn: &ChainNuclide) -> impl Iterator<Item = &yani::DecaySource> {
        cn.sources
            .iter()
            .filter(|source| source.particle == "photon")
    }
    let mut lines: Vec<(f64, f64, f64)> =
        photons(nominal)
            .zip(photons(drawn))
            .filter_map(|(n, d)| match (&n.distribution, &d.distribution) {
                (
                    yani::DecaySourceDistribution::Discrete { energies: keys, .. },
                    yani::DecaySourceDistribution::Discrete {
                        energies,
                        intensities,
                    },
                ) => Some(keys.iter().zip(intensities).zip(energies).map(
                    |((key, intensity), energy)| (*key, *intensity, intensity * (energy - key)),
                )),
                _ => None,
            })
            .flatten()
            .collect();
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    lines.dedup_by(|later, kept| {
        let same = later.0 == kept.0;
        if same {
            kept.1 += later.1;
            kept.2 += later.2;
        }
        same
    });
    lines
}

/// The volume a quantity expressed per unit of material needs.
///
/// Activity, decay heat and the photon spectrum all count atoms, so all three
/// want it. The contact dose does not: it is a half-space estimate, and a
/// bigger lump of the same material reads the same.
fn require_volume(volume: Option<f64>, quantity: &str) -> Result<f64, String> {
    volume.ok_or_else(|| format!("{quantity} requires material.volume in cm^3"))
}

/// Something evaluated on the nominal inventory and on every replica's.
struct Folded<T> {
    nominal: T,
    /// Per replica, with each nuclide's photon intensities drawn independently
    /// where the `decay_photon_lines` source is on.
    independent: Vec<T>,
    /// Per replica again, the same inventories with the photon intensities
    /// drawn fully correlated. `None` when that end is the independent one:
    /// the source is off, or the quantity reads no photon intensity.
    correlated: Option<Vec<T>>,
}

/// One quantity's per-nuclide breakdown, folded over the ensemble.
type Evaluated = Folded<HashMap<String, f64>>;

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
    /// Reads the photon lines, so with the `decay_photon_lines` source on it
    /// is evaluated at both ends of the range their unstated correlations
    /// leave.
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
    /// Whether a photon intensity enters the quantity, so the two ends of the
    /// range of their unstated correlations differ. The decay heat does: its
    /// gamma part follows each replica's drawn lines.
    fn reads_photons(self) -> bool {
        !matches!(self, Quantity::Activity)
    }

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
    ///
    /// With the `decay_photon_lines` source on and `reads_photons` set, every
    /// replica is evaluated twice on its one inventory: with each nuclide's
    /// photon intensities drawn independently and fully correlated. Lines never
    /// enter the solve, so the second end costs an evaluation, not a solve.
    fn fold_replicas<T: Clone>(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        reads_photons: bool,
        evaluate: impl Fn(
            &HashMap<String, f64>,
            Option<f64>,
            &HashMap<String, ChainNuclide>,
        ) -> Result<T, String>,
    ) -> Result<Option<Folded<T>>, String> {
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
        let nominal = evaluate(&densities, volume, chain)?;

        // Step 0 is the initial composition: an input rather than a result,
        // so every replica starts from it. The ensemble stores nothing for
        // step 0, so the nominal inventory stands in for every replica's.
        let inventories: Vec<&HashMap<String, f64>> = if step == 0 {
            vec![&densities; ensemble.replicas()]
        } else {
            self.uncertainty_inventories(material_id, step)
                .unwrap_or_default()
        };
        let half_lives = ensemble.half_lives();
        let decay_energy_seed = ensemble.decay_energy_seed;
        let decay_photon_seed = ensemble.decay_photon_seed;
        let no_half_lives = HashMap::new();
        let one = |k: usize, inventory: &HashMap<String, f64>, correlation: PhotonCorrelation| {
            let sampled = half_lives.get(k).filter(|h| !h.is_empty());
            if sampled.is_none() && decay_energy_seed.is_none() && decay_photon_seed.is_none() {
                // Every replica evaluates to exactly what the nominal
                // chain gives, which at step 0 is the nominal value
                // itself: a measured zero spread, not the unmeasured one
                // an empty ensemble would report.
                return evaluate(inventory, volume, chain);
            }
            // A replica solved with perturbed half-lives is evaluated with
            // them too: its activity is lambda_k N_k, never lambda N_k, or
            // the cancellation that makes a saturated activity insensitive
            // to its own half-life is lost. So even step 0 has a spread
            // then, the initial radionuclides' own. Its decay energies and
            // photons are drawn for it here, since they never entered the
            // solve.
            let chain_k = replica_chain(
                chain,
                inventory,
                sampled.unwrap_or(&no_half_lives),
                decay_energy_seed.map(|seed| (seed, k as u64)),
                decay_photon_seed.map(|seed| (seed, k as u64, correlation)),
            );
            evaluate(inventory, volume, &chain_k)
        };
        let independent = inventories
            .iter()
            .enumerate()
            .map(|(k, inventory)| one(k, inventory, PhotonCorrelation::Independent))
            .collect::<Result<Vec<_>, _>>()?;
        let correlated = if reads_photons && decay_photon_seed.is_some() {
            Some(
                inventories
                    .iter()
                    .enumerate()
                    .map(|(k, inventory)| one(k, inventory, PhotonCorrelation::Correlated))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(Some(Folded {
            nominal,
            independent,
            correlated,
        }))
    }

    /// Shared body of the per-nuclide accessors.
    fn derived(
        &self,
        material_id: u32,
        step: usize,
        chain: &HashMap<String, ChainNuclide>,
        quantity: Quantity,
    ) -> Result<Option<Evaluated>, String> {
        self.fold_replicas(
            material_id,
            step,
            chain,
            quantity.reads_photons(),
            |densities, volume, chain| quantity.by_nuclide(densities, volume, chain),
        )
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
        let Some(folded) = self.derived(material_id, step, chain, quantity)? else {
            return Ok(None);
        };
        let totals = |replicas: &[HashMap<String, f64>]| -> Vec<f64> {
            replicas.iter().map(yani_decay::total).collect()
        };
        let correlated = folded.correlated.as_deref().map(totals);
        Ok(Some(estimate(
            yani_decay::total(&folded.nominal),
            &totals(&folded.independent),
            correlated.as_deref(),
        )))
    }

    /// Activity of a material at `step` [Bq], with the ensemble's spread.
    ///
    /// `Ok(None)` when the run carried no uncertainty for this material, which
    /// is the default path. `Err` when the material has no volume, the same
    /// condition `Material::activity` rejects.
    ///
    /// `step` is indexed like [`Self::get_material`]: 0 is the initial
    /// composition. That composition is an input, so every replica starts from
    /// the same inventory, but each replica evaluates it with its own sampled
    /// half-lives, decay energies and decay photon data. Step 0 therefore has
    /// a spread whenever any of those sources is sampled, and is zero only
    /// when none of them is.
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
    /// With the `decay_photon_lines` source on, each replica's gamma decay
    /// energy follows its drawn lines (see `replica_chain`), so the heat has
    /// a range, `std_dev` to `std_dev_correlated`, as the contact dose does.
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
            .map(|f| estimate_by_nuclide(&f.nominal, &f.independent, f.correlated.as_deref())))
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
            .map(|f| estimate_by_nuclide(&f.nominal, &f.independent, f.correlated.as_deref())))
    }

    /// The decay-photon spectrum at `step`, each line with the ensemble's
    /// spread on its emission rate, and on its energy when the
    /// `decay_photon_lines` source draws it.
    ///
    /// Ascending in energy, coincident lines summed, exactly as
    /// `Material.decay_photon_spectrum` returns them -- and over the union of
    /// the lines that appear in the nominal run and in any replica. Lines
    /// only, as there: a continuum is a density per eV, which has no place in
    /// a list of line rates, and it is not reported here. A line a
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
            .fold_replicas(
                material_id,
                step,
                chain,
                true,
                |densities, volume, drawn| {
                    let volume = require_volume(volume, "decay_photon_spectrum")?;
                    Ok(traced_photon_lines(densities, volume, chain, drawn))
                },
            )?
            .map(|f| estimate_lines(&f.nominal, &f.independent, f.correlated.as_deref())))
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
    /// The band is the spread of the replicas' inventories, and of their
    /// decay photon lines and continua when the `decay_photon_lines` source
    /// is on. The photon attenuation (NIST XCOM), the air energy absorption
    /// (NIST SRD 126), the ICRP dose coefficients and the build-up factor are
    /// held at their nominal values in every replica, since none of those
    /// sources publishes a per-value uncertainty.
    ///
    /// With the photons drawn, [`Estimate::std_dev`] takes each nuclide's
    /// photon intensities as independent where the evaluation states no
    /// correlation and [`Estimate::std_dev_correlated`] as fully correlated,
    /// on the same inventories.
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
    /// See [`Self::contact_dose_uncertainty`], including what the band holds
    /// at nominal, and [`Self::activity_uncertainty_by_nuclide`] on why these
    /// do not add up to the total in quadrature.
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
            .map(|f| estimate_by_nuclide(&f.nominal, &f.independent, f.correlated.as_deref())))
    }
}

/// The derived totals the uncertainty driver's stopping rule tracks, for one
/// replica's inventory at one step: activity [Bq], decay heat [W] and the
/// decay photon line rate [photons/s], per cm^3.
///
/// Evaluated as the accessors above evaluate a replica, with its own
/// half-lives, decay energies and decay photon data (the photons at the
/// independent end of their range), so the spread judged is the spread
/// reported. Per cm^3 because the volume scales every replica alike
/// and leaves a relative standard error where it is.
pub(crate) fn tracked_totals(
    inventory: &HashMap<String, f64>,
    chain: &HashMap<String, ChainNuclide>,
    half_lives: &HashMap<String, f64>,
    decay_energy: Option<(u64, u64)>,
    decay_photons: Option<(u64, u64)>,
) -> [f64; 3] {
    let drawn;
    let chain = if half_lives.is_empty() && decay_energy.is_none() && decay_photons.is_none() {
        chain
    } else {
        // The independent end, the one `Estimate::std_dev` reports.
        let photons =
            decay_photons.map(|(seed, replica)| (seed, replica, PhotonCorrelation::Independent));
        drawn = replica_chain(chain, inventory, half_lives, decay_energy, photons);
        &drawn
    };
    let lines: f64 = yani_decay::decay_photon_lines(inventory, 1.0, chain)
        .iter()
        .map(|(_, rate)| rate)
        .sum();
    [
        yani_decay::activity_total(inventory, 1.0, chain),
        yani_decay::decay_heat_total(inventory, 1.0, chain),
        lines,
    ]
}

/// The chain entries an inventory's derived quantities read, with one
/// replica's half-lives substituted and its decay energies and photons drawn,
/// the photon intensities at the given end of the range their unstated
/// correlations leave (see [`PhotonCorrelation`]).
///
/// Only the nuclides in the inventory: activity, decay heat, decay photons and
/// contact dose all evaluate each nuclide from its own entry, so the rest of a
/// 3800-nuclide chain would be copied per replica for nothing.
///
/// With the photons drawn, the electromagnetic decay energy E_EM follows them,
/// so one replica's gamma heat and its contact dose come from the same draw
/// of one evaluation. E_EM is the photon energy per decay, the lines' sum of
/// energy times intensity plus each continuum's energy integral, which in
/// ENDF/B-VIII.1 reproduces E_EM to 1% for 1826 of its 1871 photon emitters;
/// the rest, fission products mostly, state an E_EM their tabulated spectra do
/// not carry in full. So E_EM moves by exactly the drawn change in the photon
/// energy per decay, and the part the spectra do not cover is held at
/// nominal. For a nuclide whose photon data carry a sigma this replaces the
/// `decay_energy` source's own draw of E_EM, which would count the lines'
/// uncertainty a second time; the beta and alpha components keep theirs.
/// Where the decay energy states its sigma on the total alone the gamma part
/// cannot be told apart from the rest, so for such a nuclide the total
/// follows the lines and is not drawn again.
fn replica_chain(
    chain: &HashMap<String, ChainNuclide>,
    inventory: &HashMap<String, f64>,
    half_lives: &HashMap<String, f64>,
    decay_energy: Option<(u64, u64)>,
    decay_photons: Option<(u64, u64, PhotonCorrelation)>,
) -> HashMap<String, ChainNuclide> {
    inventory
        .keys()
        .filter_map(|name| {
            let mut cn = chain.get(name)?.clone();
            if let Some(t) = half_lives.get(name) {
                crate::uncertainty::set_half_life(&mut cn, *t);
            }
            let lines_set_gamma = decay_photons.is_some()
                && cn.half_life.is_some_and(|t| t > 0.0)
                && crate::uncertainty::has_decay_photon_sigma(&cn);
            if let Some((seed, replica)) = decay_energy {
                let total_only = !crate::uncertainty::components_state_sigma(&cn);
                if !(lines_set_gamma && total_only) {
                    if let Some((total, parts)) =
                        crate::uncertainty::sample_decay_energy(&cn, seed, replica)
                    {
                        cn.decay_energy = total;
                        cn.decay_energy_components = parts;
                    }
                }
            }
            // After the half-life, whose rescale keeps each line's sigma in
            // proportion to its intensity.
            if let Some((seed, replica, correlation)) = decay_photons {
                let before = crate::uncertainty::photon_energy_rate(&cn);
                crate::uncertainty::sample_decay_photons(&mut cn, seed, replica, correlation);
                if lines_set_gamma {
                    // Rates are per atom per second, so over the replica's own
                    // decay constant the change is per decay.
                    let lambda = std::f64::consts::LN_2 / cn.half_life.expect("checked above");
                    let moved = (crate::uncertainty::photon_energy_rate(&cn) - before) / lambda;
                    let nominal = chain[name].decay_energy_components[1];
                    match (&mut cn.decay_energy_components[1], nominal) {
                        (Some(gamma), Some(nominal)) => {
                            let em = nominal.energy + moved;
                            cn.decay_energy += em - gamma.energy;
                            gamma.energy = em;
                        }
                        _ => cn.decay_energy += moved,
                    }
                }
            }
            Some((name.clone(), cn))
        })
        .collect()
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
                decay_energy_components: Default::default(),
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![yani::DecaySource {
                    particle: "photon".to_string(),
                    radiation: None,
                    uncertainty: None,
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
                decay_energy_components: Default::default(),
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
                        decay_energy_components: Default::default(),
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
        let mut results = TransmutationResults::new(vec![1.0]);
        results.add_initial(7, material(nominal), vec![0.0]);
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

    /// The initial composition is an input, so every replica starts from it.
    /// With no decay data sampled (this fixture samples none) every replica
    /// evaluates to the nominal value, so the spread is zero rather than
    /// unmeasured, the same answer `get_nuclide_uncertainty` gives at step 0.
    /// Sampled half-lives, decay energies or photon data would give step 0 a
    /// spread.
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
        let mut results = TransmutationResults::new(vec![1.0]);
        results.add_initial(7, material(&[("Fe56", 1.0)]), vec![0.0]);
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

    // --- the gamma decay energy follows the drawn lines ------------------------

    /// A source's stored values, lines or continuum points alike.
    fn values(d: &yani::DecaySourceDistribution) -> &[f64] {
        match d {
            yani::DecaySourceDistribution::Discrete { intensities, .. }
            | yani::DecaySourceDistribution::Tabular { intensities, .. } => intensities,
        }
    }

    /// Co60 with its two lines given a 1% normalisation sigma and 2% and 3%
    /// of their own, and its decay energy split into components with a 2%
    /// beta sigma and a 40% gamma one, far wider than the lines allow, so a
    /// gamma part drawn from it as well as from the lines would show.
    fn cobalt_with_sigmas() -> HashMap<String, ChainNuclide> {
        let mut chain = cobalt_chain();
        let co60 = chain.get_mut("Co60").unwrap();
        let intensities = values(&co60.sources[0].distribution).to_vec();
        co60.sources[0].uncertainty = Some(std::sync::Arc::new(yani::DecaySourceUncertainty {
            normalization: Some(1.0),
            normalization_uncertainty: Some(0.01),
            intensity_uncertainties: Some(vec![0.02 * intensities[0], 0.03 * intensities[1]]),
            energy_uncertainties: Some(vec![30.0, 20.0]),
            covariance: None,
        }));
        let component = |energy: f64, sigma: f64| {
            Some(yani::DecayEnergyComponent {
                energy,
                uncertainty: Some(sigma * energy),
            })
        };
        co60.decay_energy_components =
            [component(96_000.0, 0.02), component(2_503_000.0, 0.4), None];
        co60.decay_energy = 96_000.0 + 2_503_000.0;
        chain
    }

    /// With only the photons drawn, a replica's E_EM moves by exactly the
    /// drawn change in its photon energy per decay, and the total with it.
    #[test]
    fn the_gamma_decay_energy_follows_the_drawn_lines_exactly() {
        let chain = cobalt_with_sigmas();
        let inventory = densities(&[("Co60", 1.0e-9)]);
        let nominal = &chain["Co60"];
        let lambda = std::f64::consts::LN_2 / nominal.half_life.unwrap();
        let before = crate::uncertainty::photon_energy_rate(nominal);
        for correlation in [
            PhotonCorrelation::Independent,
            PhotonCorrelation::Correlated,
        ] {
            for replica in 0..32 {
                let drawn = replica_chain(
                    &chain,
                    &inventory,
                    &HashMap::new(),
                    None,
                    Some((5, replica, correlation)),
                );
                let co60 = &drawn["Co60"];
                let moved = (crate::uncertainty::photon_energy_rate(co60) - before) / lambda;
                assert!(moved != 0.0, "the lines moved");
                let gamma = co60.decay_energy_components[1].unwrap().energy;
                assert!((gamma - (2_503_000.0 + moved)).abs() < 1e-9 * gamma);
                assert!((co60.decay_energy - (96_000.0 + gamma)).abs() < 1e-9 * gamma);
                assert_eq!(
                    co60.decay_energy_components[0],
                    nominal.decay_energy_components[0]
                );
            }
        }
    }

    /// With both sources on, the beta part keeps its own draw and the gamma
    /// part follows the lines only: the 40% gamma sigma is not drawn on top,
    /// so the decay energy's spread is the beta's and the lines' in
    /// quadrature, with nothing counted twice.
    #[test]
    fn decay_energy_and_lines_together_count_the_gamma_part_once() {
        let chain = cobalt_with_sigmas();
        let inventory = densities(&[("Co60", 1.0e-9)]);
        let nominal = &chain["Co60"];
        let lambda = std::f64::consts::LN_2 / nominal.half_life.unwrap();
        let before = crate::uncertainty::photon_energy_rate(nominal);
        let n = 8_000u64;
        let (mut totals, mut lines, mut betas) = (Vec::new(), Vec::new(), Vec::new());
        for replica in 0..n {
            let drawn = replica_chain(
                &chain,
                &inventory,
                &HashMap::new(),
                Some((5, replica)),
                Some((5, replica, PhotonCorrelation::Independent)),
            );
            let co60 = &drawn["Co60"];
            let moved = (crate::uncertainty::photon_energy_rate(co60) - before) / lambda;
            let beta = co60.decay_energy_components[0].unwrap().energy;
            let gamma = co60.decay_energy_components[1].unwrap().energy;
            assert!((gamma - (2_503_000.0 + moved)).abs() < 1e-9 * gamma);
            totals.push(co60.decay_energy);
            lines.push(moved);
            betas.push(beta);
        }
        let sd = |v: &[f64]| spread(v).1.unwrap();
        assert!(sd(&betas) > 0.0, "the beta part is still drawn");
        let want = (sd(&betas).powi(2) + sd(&lines).powi(2)).sqrt();
        assert!(
            (sd(&totals) / want - 1.0).abs() < 0.05,
            "{} against {want}",
            sd(&totals)
        );
        assert!(
            sd(&totals) < 0.05 * 2_503_000.0,
            "the 40% gamma sigma was not drawn"
        );
    }

    /// The decay heat of an inventory whose photons alone are drawn has a
    /// spread, the lines', and a range: Co60's two lines with their own dRI
    /// read independently and fully correlated.
    #[test]
    fn decay_heat_and_contact_dose_carry_the_line_range() {
        let replicas: Vec<Vec<(&str, f64)>> = (0..1_000)
            .map(|_| vec![("Fe56", 0.08), ("Co60", 1.0e-9)])
            .collect();
        let mut results = results(&[("Fe56", 0.08), ("Co60", 1.0e-9)], &replicas);
        results.uncertainty.get_mut(&7).unwrap().decay_photon_seed = Some(3);
        let chain = cobalt_with_sigmas();

        let activity = results.activity_uncertainty(7, 1, &chain).unwrap().unwrap();
        assert_eq!(activity.std_dev, Some(0.0), "no line moves the activity");
        assert_eq!(activity.std_dev_correlated, activity.std_dev);

        let heat = results
            .decay_heat_uncertainty(7, 1, &chain)
            .unwrap()
            .unwrap();
        let dose = results
            .contact_dose_uncertainty(7, 1, &chain, DoseQuantity::AbsorbedAir, 2.0)
            .unwrap()
            .unwrap();
        for estimate in [heat, dose] {
            let low = estimate.relative_std_dev().unwrap();
            let high = estimate.relative_std_dev_correlated().unwrap();
            assert!(low > 0.0 && high > 1.2 * low, "{low} to {high}");
        }
        let lines = results
            .photon_spectrum_uncertainty(7, 1, &chain)
            .unwrap()
            .unwrap();
        for line in &lines {
            let low = line.estimate.relative_std_dev().unwrap();
            let high = line.estimate.relative_std_dev_correlated().unwrap();
            assert!(high > low, "{low} to {high}");
        }
    }
}
