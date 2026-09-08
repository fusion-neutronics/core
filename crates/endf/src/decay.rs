//! Radioactive decay data: MF=8 MT=457, and the fission product yields beside
//! it in MT=454 and MT=459.

use std::collections::BTreeMap;

use crate::data::{gnds_name, ATOMIC_SYMBOL};
use crate::error::{Error, Result};
use crate::material::Material;
use crate::mf::mf8::WithUncertainty;
use crate::univariate::{combine_distributions, Discrete, Interpolation, Tabular, Univariate};

/// Each decay mode's name, and what it does to the mass and atomic numbers.
///
/// `None` for a mode whose products are not a single nuclide: spontaneous
/// fission, and the catch-all for an unknown mode.
pub const DECAY_MODES: [(&str, Option<(i64, i64)>); 11] = [
    ("gamma", Some((0, 0))),
    ("beta-", Some((0, 1))),
    ("ec/beta+", Some((0, -1))),
    ("IT", Some((0, 0))),
    ("alpha", Some((-4, -2))),
    ("n", Some((-1, 0))),
    ("sf", None),
    ("p", Some((-1, -1))),
    ("e-", Some((0, 0))),
    ("xray", Some((0, 0))),
    ("unknown", None),
];

/// Which particle each radiation type emits, named as a source distribution
/// keys them.
///
/// Several radiation types share a particle (gammas and x-rays are both
/// photons, betas and Auger electrons both electrons) and their spectra are
/// combined by [`Decay::sources`].
pub const SOURCE_PARTICLES: [(&str, &str); 11] = [
    ("gamma", "photon"),
    ("xray", "photon"),
    ("beta-", "electron"),
    ("e-", "electron"),
    ("ec/beta+", "positron"),
    ("alpha", "alpha"),
    ("n", "neutron"),
    ("sf", "fragment"),
    ("p", "proton"),
    ("anti-neutrino", "anti-neutrino"),
    ("neutrino", "neutrino"),
];

/// The radiation type each STYP value names.
pub const RADIATION_TYPES: [(i64, &str); 11] = [
    (0, "gamma"),
    (1, "beta-"),
    (2, "ec/beta+"),
    (4, "alpha"),
    (5, "n"),
    (6, "sf"),
    (7, "p"),
    (8, "e-"),
    (9, "xray"),
    (10, "anti-neutrino"),
    (11, "neutrino"),
];

/// The forbiddenness of a discrete transition, as the TYPE field records it.
const DISCRETE_TYPES: [&str; 7] = [
    "allowed",
    "first-forbidden",
    "second-forbidden",
    "third-forbidden",
    "fourth-forbidden",
    "fifth-forbidden",
    "",
];

/// The three average decay energies that make up the recoverable heat.
///
/// `AVERAGE_ENERGY_NAMES` has seventeen entries, and the last fourteen are a
/// FINER decomposition of these three, not additional terms: light is the
/// betas, positrons, Auger and conversion electrons; electromagnetic is the
/// gammas, x-rays, bremsstrahlung and annihilation; heavy is the alphas,
/// recoils, spontaneous-fission fragments, neutrons and protons. Neutrinos are
/// excluded from all of them because that energy leaves. Summing all seventeen
/// roughly doubles the heat, and nothing in the type system stops you, so both
/// the total and the split read this one list.
pub const DECAY_HEAT_ENERGY_NAMES: [&str; 3] = ["light", "electromagnetic", "heavy"];

/// The names of the average decay energies, in the order MT=457 stores them.
///
/// The first three are always given; the rest only when the evaluation writes
/// the long form.
pub const AVERAGE_ENERGY_NAMES: [&str; 17] = [
    "light",
    "electromagnetic",
    "heavy",
    "beta-",
    "beta+",
    "auger",
    "conversion",
    "gamma",
    "xray",
    "bremsstrahlung",
    "annihilation",
    "alpha",
    "recoil",
    "SF",
    "neutron",
    "proton",
    "neutrino",
];

/// The chain of decay modes an RTYP value names, e.g. `1.5` is a beta- decay
/// followed by neutron emission.
pub fn decay_modes(rtyp: f64) -> Vec<&'static str> {
    crate::data::python_float_str(rtyp)
        .trim_matches('0')
        .chars()
        .filter(|c| *c != '.')
        .filter_map(|c| {
            c.to_digit(10)
                .and_then(|d| DECAY_MODES.get(d as usize))
                .map(|&(name, _)| name)
        })
        .collect()
}

/// One decay mode of a nuclide, and how much of the decay goes through it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecayMode {
    /// The decaying nuclide, in GNDS convention.
    pub parent: String,
    /// The chain of modes, e.g. `["beta-", "n"]`.
    pub modes: Vec<&'static str>,
    /// Metastable state of the daughter.
    pub daughter_state: i64,
    /// Total decay energy available, in eV.
    pub energy: WithUncertainty,
    /// The fraction of decays that go this way.
    pub branching_ratio: WithUncertainty,
}

impl DecayMode {
    /// The nuclide this mode leaves behind.
    ///
    /// `None` when the parent's name cannot be read, or when a mode in the
    /// chain has no single daughter; spontaneous fission does not.
    pub fn daughter(&self) -> Option<String> {
        let (symbol, a) = split_nuclide_name(&self.parent)?;
        let mut z = ATOMIC_SYMBOL.iter().position(|&s| s == symbol)? as i64;
        let mut a = a;

        for mode in &self.modes {
            let (_, changes) = DECAY_MODES.iter().find(|(name, _)| name == mode)?;
            // A mode with no single daughter leaves the numbers alone, which
            // is what the Python reader does.
            if let Some((delta_a, delta_z)) = changes {
                a += delta_a;
                z += delta_z;
            }
        }

        let symbol = ATOMIC_SYMBOL.get(z as usize)?;
        Some(if self.daughter_state > 0 {
            format!("{symbol}{a}_m{}", self.daughter_state)
        } else {
            format!("{symbol}{a}")
        })
    }
}

/// Split a GNDS name into its symbol and mass number, e.g. `"Am242_m1"`.
fn split_nuclide_name(name: &str) -> Option<(&str, i64)> {
    let split = name.find(|c: char| c.is_ascii_digit())?;
    let (symbol, rest) = name.split_at(split);
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    Some((symbol, digits.parse().ok()?))
}

/// The nuclide a decay evaluation is about.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecayNuclide {
    /// GNDS name, e.g. `"In116_m1"`.
    pub name: String,
    pub atomic_number: i64,
    pub mass_number: i64,
    /// Isomeric state ordinal.
    pub isomeric_state: i64,
    /// Nuclear level index, which is not the same thing.
    pub excited_state: i64,
    /// Atomic mass in neutron masses.
    pub mass: f64,
    pub stable: bool,
    /// `None` when the evaluation reports the spin as unknown.
    pub spin: Option<f64>,
    pub parity: f64,
}

/// One discrete line of a decay spectrum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiscreteLine {
    /// Energy of the emitted particle, in eV.
    pub energy: WithUncertainty,
    /// The decay chain this line comes from.
    pub from_mode: Vec<&'static str>,
    /// Forbiddenness, where the evaluation states one.
    pub transition_type: Option<&'static str>,
    pub intensity: WithUncertainty,
    /// Positron intensity, for ec/beta+ spectra.
    pub positron_intensity: Option<WithUncertainty>,
    /// Internal pair formation coefficient, for gamma spectra.
    pub internal_pair: Option<WithUncertainty>,
    pub total_internal_conversion: Option<WithUncertainty>,
    pub k_shell_conversion: Option<WithUncertainty>,
    pub l_shell_conversion: Option<WithUncertainty>,
}

/// Whether a spectrum is given as lines, as a continuum, or as both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContinuousFlag {
    #[default]
    Discrete,
    Continuous,
    Both,
}

impl ContinuousFlag {
    pub fn name(self) -> &'static str {
        match self {
            ContinuousFlag::Discrete => "discrete",
            ContinuousFlag::Continuous => "continuous",
            ContinuousFlag::Both => "both",
        }
    }

    fn from_lcon(lcon: i64) -> ContinuousFlag {
        match lcon {
            1 => ContinuousFlag::Continuous,
            2 => ContinuousFlag::Both,
            _ => ContinuousFlag::Discrete,
        }
    }
}

/// The spectrum of one radiation type.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecaySpectrum {
    /// The radiation type, e.g. `"gamma"`.
    pub radiation: &'static str,
    pub continuous_flag: ContinuousFlag,
    /// Normalisation of the discrete lines.
    pub discrete_normalization: WithUncertainty,
    /// Average energy of this radiation type, in eV.
    pub energy_average: WithUncertainty,
    /// Normalisation of the continuum.
    pub continuous_normalization: WithUncertainty,
    pub discrete: Vec<DiscreteLine>,
    /// The continuum, as a probability per eV against energy.
    pub continuous: Option<crate::function::Tabulated1D>,
    /// The decay chain the continuum comes from.
    pub continuous_from_mode: Vec<&'static str>,
}

/// Radioactive decay data for one nuclide.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Decay {
    pub nuclide: DecayNuclide,
    /// Half-life in seconds. `None` for a stable nuclide.
    pub half_life: Option<WithUncertainty>,
    /// Average decay energies by radiation type, in eV.
    pub average_energies: BTreeMap<&'static str, WithUncertainty>,
    pub modes: Vec<DecayMode>,
    /// The spectra, by radiation type. An evaluation gives at most one per
    /// type, so a later one of the same type replaces the earlier.
    pub spectra: BTreeMap<&'static str, DecaySpectrum>,
    /// Whether the average energies are a placeholder rather than an
    /// evaluation. See [`Decay::placeholder_mean_energies`].
    pub mean_energy_placeholder: bool,
}

/// Relative tolerance on "equal" average energies and on the Q/3 check.
const PLACEHOLDER_TOLERANCE: f64 = 1.0e-3;

impl Decay {
    /// Whether the average energies are the Q/3 placeholder rather than an
    /// evaluation.
    ///
    /// A nuclide whose decay scheme nobody has evaluated still gets a decay
    /// file: the NNDC's conversions from the Nuclear Wallet Cards (ENDF/B) and
    /// the NUBASE conversions (JEFF) carry a half-life and the decay modes and,
    /// in place of measured average energies, book a third of each beta or
    /// electron-capture branch's Q to the light particles, a third to the
    /// electromagnetic radiation and the last third to the neutrino. Two things
    /// give it away, and both are required: the light and electromagnetic
    /// averages are equal, and either the record has no spectra at all or
    /// three times the light average is the beta and electron-capture Q
    /// summed over branches. The second condition is what separates a
    /// placeholder from the odd real evaluation whose two averages happen to
    /// coincide (Sn117m, a 315 keV isomeric transition split almost evenly
    /// between conversion electrons and photons), and it also catches JENDL-5.0
    /// records that keep the Q/3 averages while carrying theoretical spectra.
    ///
    /// The placeholder is not a rough estimate. Electron capture hands most of
    /// Q to the neutrino, so for an EC emitter the placeholder can be several
    /// times the recoverable energy; Sn111 is booked at 1.63 MeV per decay
    /// against 0.69 MeV from its decay scheme. A consumer summing decay heat
    /// needs to know which records these are.
    pub fn placeholder_mean_energies(&self) -> bool {
        let (Some(&(light, _)), Some(&(electromagnetic, _))) = (
            self.average_energies.get("light"),
            self.average_energies.get("electromagnetic"),
        ) else {
            return false;
        };
        if light <= 0.0 || (light - electromagnetic).abs() > PLACEHOLDER_TOLERANCE * light {
            return false;
        }
        if self.spectra.is_empty() {
            return true;
        }
        let beta_q: f64 = self
            .modes
            .iter()
            .filter(|mode| matches!(mode.modes.first(), Some(&"beta-") | Some(&"ec/beta+")))
            .map(|mode| mode.branching_ratio.0 * mode.energy.0)
            .sum();
        beta_q > 0.0 && (3.0 * light - beta_q).abs() <= PLACEHOLDER_TOLERANCE * beta_q
    }
}

/// Relative tolerance on the light plus electromagnetic averages of an
/// isomeric transition against its Q.
pub const ISOMERIC_TRANSITION_ENERGY_TOLERANCE: f64 = 0.05;

/// Tolerance on the sum of the branching ratios of the decay modes.
pub const BRANCHING_RATIO_SUM_TOLERANCE: f64 = 0.01;

/// One thing a decay record says that cannot be right.
///
/// None of these stops the record being read: the numbers are the library's
/// and a converter passes them on. They are for the record's provenance, so
/// that a nuclide carrying heat in an inventory can be looked up before its
/// number is believed.
#[derive(Debug, Clone, PartialEq)]
pub enum DecayInconsistency {
    /// Flagged unstable (NST = 0) with a half-life of zero. ENDF/B-VIII.1
    /// writes fourteen of these, Xe136 and W182 among them, for nuclides
    /// whose double-beta decay is predicted and not observed. They are
    /// treated as stable, see [`Decay::decay_constant`].
    ZeroHalfLife,
    /// The branching ratios of the decay modes do not sum to one, so the
    /// daughters are fed more or less than one nuclide per decay.
    BranchingRatioSum { sum: f64 },
    /// An isomeric transition emits no neutrino, so its light and
    /// electromagnetic averages should add up to its Q. ENDF/B-VIII.1's
    /// Hf177_m1 books 1.52 MeV against a 1.32 MeV transition and its
    /// Ir190_m1 17 keV against 26 keV; a decay heat summed from either is
    /// wrong by that much.
    IsomericTransitionEnergy { q: f64, recoverable: f64 },
}

impl DecayInconsistency {
    /// Every kind's label, in the order a record lists them.
    pub const LABELS: [&'static str; 3] = [
        "zero_half_life",
        "branching_ratio_sum",
        "isomeric_transition_energy",
    ];

    /// A stable name for the kind, one of [`Self::LABELS`].
    pub fn label(&self) -> &'static str {
        match self {
            DecayInconsistency::ZeroHalfLife => Self::LABELS[0],
            DecayInconsistency::BranchingRatioSum { .. } => Self::LABELS[1],
            DecayInconsistency::IsomericTransitionEnergy { .. } => Self::LABELS[2],
        }
    }
}

impl Decay {
    /// What this record says that cannot all be true.
    ///
    /// Empty for a consistent record, and for a stable one, which claims
    /// nothing to check.
    pub fn inconsistencies(&self) -> Vec<DecayInconsistency> {
        let mut out = Vec::new();
        if self.nuclide.stable {
            return out;
        }
        if self.half_life.is_some_and(|(t, _)| t <= 0.0) {
            out.push(DecayInconsistency::ZeroHalfLife);
        }
        if self.modes.is_empty() {
            return out;
        }
        let sum: f64 = self.modes.iter().map(|m| m.branching_ratio.0).sum();
        if (sum - 1.0).abs() > BRANCHING_RATIO_SUM_TOLERANCE {
            out.push(DecayInconsistency::BranchingRatioSum { sum });
        }
        if self.modes.iter().all(|m| m.modes == ["IT"]) {
            let q: f64 = self
                .modes
                .iter()
                .map(|m| m.branching_ratio.0 * m.energy.0)
                .sum();
            let average = |name| self.average_energies.get(name).map_or(0.0, |&(v, _)| v);
            let recoverable = average("light") + average("electromagnetic");
            if q > 0.0 && (recoverable - q).abs() > ISOMERIC_TRANSITION_ENERGY_TOLERANCE * q {
                out.push(DecayInconsistency::IsomericTransitionEnergy { q, recoverable });
            }
        }
        out
    }
}

impl Decay {
    /// Read the decay data of a material.
    pub fn from_material(material: &Material) -> Result<Decay> {
        let section = material.mf8_mt457().ok_or(Error::Unsupported {
            what: "an evaluation with no MF=8 MT=457 decay section",
        })?;

        let (z, a) = (section.za / 1000, section.za % 1000);
        let stable = section.nst == 1;
        let mut decay = Decay {
            nuclide: DecayNuclide {
                name: gnds_name(z as u32, a as u32, section.liso as u32),
                atomic_number: z,
                mass_number: a,
                isomeric_state: section.liso,
                excited_state: section.lis,
                mass: section.awr,
                stable,
                // ENDF-102 writes an unknown spin as -77.777.
                spin: (section.spi != -77.777).then_some(section.spi),
                parity: section.par,
            },
            half_life: if stable { None } else { section.half_life },
            ..Default::default()
        };

        for (name, &value) in AVERAGE_ENERGY_NAMES.iter().zip(&section.ex) {
            decay.average_energies.insert(name, value);
        }

        for mode in &section.modes {
            decay.modes.push(DecayMode {
                parent: decay.nuclide.name.clone(),
                modes: decay_modes(mode.rtyp),
                daughter_state: mode.rfs as i64,
                energy: mode.q,
                branching_ratio: mode.br,
            });
        }

        for spectrum in &section.spectra {
            let radiation = RADIATION_TYPES
                .iter()
                .find(|&&(styp, _)| styp == spectrum.styp as i64)
                .map(|&(_, name)| name)
                .ok_or(Error::Unsupported {
                    what: "a decay radiation type the format does not define",
                })?;
            let continuous_flag = ContinuousFlag::from_lcon(spectrum.lcon);

            let discrete = spectrum
                .discrete
                .iter()
                .map(|line| DiscreteLine {
                    energy: line.er,
                    from_mode: decay_modes(line.rtyp),
                    // A TYPE of zero means the evaluation did not say.
                    transition_type: DISCRETE_TYPES
                        .get(line.type_ as usize - usize::from(line.type_ >= 1.0))
                        .filter(|_| line.type_ >= 1.0)
                        .copied(),
                    intensity: line.ri,
                    // The same field is the positron intensity for ec/beta+
                    // and the internal pair coefficient for gammas.
                    positron_intensity: (radiation == "ec/beta+").then_some(line.ris).flatten(),
                    internal_pair: (radiation == "gamma").then_some(line.ris).flatten(),
                    total_internal_conversion: line.ricc,
                    k_shell_conversion: line.rick,
                    l_shell_conversion: line.ricl,
                })
                .collect();

            decay.spectra.insert(
                radiation,
                DecaySpectrum {
                    radiation,
                    continuous_flag,
                    discrete_normalization: spectrum.fd,
                    energy_average: spectrum.er_av,
                    continuous_normalization: spectrum.fc,
                    discrete,
                    continuous: spectrum.continuous.as_ref().map(|c| c.rp.clone()),
                    continuous_from_mode: spectrum
                        .continuous
                        .as_ref()
                        .map_or_else(Vec::new, |c| decay_modes(c.rtyp)),
                },
            );
        }

        decay.mean_energy_placeholder = decay.placeholder_mean_energies();
        Ok(decay)
    }

    /// The decay constant in inverse seconds, with its uncertainty.
    ///
    /// `None` for a stable nuclide, and also for one whose half-life the
    /// evaluation gives as zero. Zero means the half-life was not evaluated,
    /// not that the nuclide decays instantly, ENDF/B-VIII.0's Xe136 is
    /// flagged unstable with a half-life of zero, its real one being some
    /// 10^21 years. `Chain::from_endf` reads it the same way, and so does the
    /// Python property since issue #23 was fixed, it used to divide by the
    /// zero and raise `ZeroDivisionError`.
    pub fn decay_constant(&self) -> Option<WithUncertainty> {
        let (t, sigma) = self.half_life?;
        if t == 0.0 {
            return None;
        }
        let ln2 = std::f64::consts::LN_2;
        Some((ln2 / t, ln2 / (t * t) * sigma))
    }

    /// Average energy per decay available for decay heat, in eV.
    pub fn decay_energy(&self) -> WithUncertainty {
        let parts = self.decay_energy_components();
        let value = parts.iter().map(|p| p.map_or(0.0, |(v, _)| v)).sum();
        // The uncertainties add in quadrature, as the Python package's
        // `uncertainties` does for a sum of independent terms. MT=457
        // publishes no covariance between the three, so quadrature is the only
        // defensible combination and the split must not be recombined by any
        // other rule downstream.
        let variance: f64 = parts.iter().map(|p| p.map_or(0.0, |(_, s)| s * s)).sum();
        (value, variance.sqrt())
    }

    /// The three recoverable-heat components, in `DECAY_HEAT_ENERGY_NAMES`
    /// order, or `None` per component where the evaluation stated nothing.
    ///
    /// `None` and `Some((0.0, 0.0))` are different claims and both occur: a
    /// stable nuclide writes no EX record at all, while plenty of unstable
    /// ones write a component that is genuinely zero. Collapsing the first into
    /// the second is what [`decay_energy`](Self::decay_energy) does on purpose,
    /// because a sum needs a number; a consumer carrying the split into a
    /// column must not, or the column reports "measured to be exact" for data
    /// that was never given.
    pub fn decay_energy_components(&self) -> [Option<WithUncertainty>; 3] {
        let mut out = [None; 3];
        for (slot, name) in out.iter_mut().zip(DECAY_HEAT_ENERGY_NAMES) {
            *slot = self.average_energies.get(name).copied();
        }
        out
    }

    /// The particles this nuclide emits, as distributions in emitted particles
    /// per second.
    ///
    /// The spectra are intensities per decay; multiplying by the decay
    /// constant makes them rates. Lines and continua of the same particle are
    /// combined, so a nuclide that emits both gammas and x-rays gives one
    /// photon distribution.
    pub fn sources(&self) -> Result<BTreeMap<&'static str, Univariate>> {
        let Some((decay_constant, _)) = self.decay_constant() else {
            return Ok(BTreeMap::new());
        };

        let mut by_particle: BTreeMap<&'static str, Vec<Univariate>> = BTreeMap::new();
        for (radiation, spectrum) in &self.spectra {
            let particle = SOURCE_PARTICLES
                .iter()
                .find(|&&(r, _)| r == *radiation)
                .map(|&(_, p)| p)
                .ok_or(Error::Unsupported {
                    what: "a decay radiation type with no source particle",
                })?;
            let dists = by_particle.entry(particle).or_default();

            if spectrum.continuous_flag != ContinuousFlag::Continuous {
                let norm = spectrum.discrete_normalization.0;
                dists.push(Univariate::Discrete(Discrete::new(
                    spectrum.discrete.iter().map(|d| d.energy.0).collect(),
                    spectrum
                        .discrete
                        .iter()
                        .map(|d| decay_constant * norm * d.intensity.0)
                        .collect(),
                )));
            }

            if spectrum.continuous_flag != ContinuousFlag::Discrete {
                let f = spectrum.continuous.as_ref().ok_or(Error::Unsupported {
                    what: "a spectrum that claims a continuum and gives none",
                })?;
                if f.interpolation.len() > 1 {
                    return Err(Error::Unsupported {
                        what: "a continuous decay spectrum with more than one interpolation region",
                    });
                }
                let interpolation =
                    Interpolation::from_endf_code(f.interpolation.first().copied().unwrap_or(2))?;
                let norm = spectrum.continuous_normalization.0;
                dists.push(Univariate::Tabular(Tabular::new(
                    f.x.clone(),
                    f.y.iter().map(|&y| decay_constant * norm * y).collect(),
                    interpolation,
                )));
            }
        }

        let mut sources = BTreeMap::new();
        for (particle, dists) in by_particle {
            let probs = vec![1.0; dists.len()];
            sources.insert(particle, combine_distributions(&dists, &probs)?);
        }
        Ok(sources)
    }
}

/// The yield of one fission product.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProductYield {
    /// GNDS name of the product, e.g. `"Xe135_m1"`.
    pub name: String,
    pub yield_: WithUncertainty,
}

/// The fissioning nuclide, as MF=1 MT=451 describes it.
///
/// Smaller than [`DecayNuclide`]: a yield evaluation has no MF=8 MT=457, so
/// there is no spin, parity or mass to report.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FissioningNuclide {
    /// GNDS name, e.g. `"U235"`.
    pub name: String,
    pub atomic_number: i64,
    pub mass_number: i64,
    /// Isomeric state ordinal.
    pub isomeric_state: i64,
    /// Nuclear level index, which is not the same thing.
    pub excited_state: i64,
}

/// Independent and cumulative fission product yields, from MF=8 MT=454 and
/// MT=459.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FissionProductYields {
    /// The fissioning nuclide.
    pub nuclide: FissioningNuclide,
    /// The incident energies the yields are given at, in eV.
    pub energies: Vec<f64>,
    /// Yields before delayed decay, one map per incident energy.
    pub independent: Vec<Vec<ProductYield>>,
    /// Yields after it.
    pub cumulative: Vec<Vec<ProductYield>>,
}

impl FissionProductYields {
    /// Read the fission product yields of a material.
    pub fn from_material(material: &Material) -> Result<FissionProductYields> {
        let mut out = FissionProductYields::default();

        // The fissioning nuclide comes from MF=1 MT=451, not from the yield
        // sections, which only identify the products.
        if let Some(info) = material.mf1_mt451() {
            let (z, a) = (info.za / 1000, info.za % 1000);
            out.nuclide = FissioningNuclide {
                name: crate::gnds_name(z as u32, a as u32, info.liso as u32),
                atomic_number: z,
                mass_number: a,
                isomeric_state: info.liso,
                excited_state: info.lis,
            };
        }

        for (mt, target) in [(454, true), (459, false)] {
            let Some(section) = material.mf8_mt454(mt) else {
                continue;
            };
            let mut energies = Vec::with_capacity(section.yields.len());
            let mut yields = Vec::with_capacity(section.yields.len());
            for set in &section.yields {
                energies.push(set.energy);
                yields.push(
                    set.products
                        .iter()
                        .map(|p| ProductYield {
                            name: product_name(p.zafp as i64, p.fps as i64),
                            yield_: p.y,
                        })
                        .collect(),
                );
            }

            if out.energies.is_empty() {
                out.energies = energies;
            } else if out.energies != energies {
                return Err(Error::Mismatched {
                    what: "the incident energies of the independent and cumulative yields",
                });
            }
            if target {
                out.independent = yields;
            } else {
                out.cumulative = yields;
            }
        }
        Ok(out)
    }
}

/// The name of a fission product, from its ZA and isomeric state.
fn product_name(za: i64, isomeric_state: i64) -> String {
    let (z, a) = (za / 1000, za % 1000);
    let symbol = ATOMIC_SYMBOL.get(z as usize).copied().unwrap_or("?");
    if isomeric_state > 0 {
        format!("{symbol}{a}_m{isomeric_state}")
    } else {
        format!("{symbol}{a}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN116M1: &[u8] = include_bytes!("../fixtures/dec-049_In_116m1.endf.xz");
    const IN116M2: &[u8] = include_bytes!("../fixtures/dec-049_In_116m2.endf.xz");

    fn decay(raw: &[u8]) -> Decay {
        let m = Material::from_str(&crate::testdata::text(raw)).unwrap();
        Decay::from_material(&m).unwrap()
    }

    #[test]
    fn decodes_a_chain_of_decay_modes_from_one_number() {
        // RTYP packs the chain as the digits of a decimal.
        assert_eq!(decay_modes(1.0), ["beta-"]);
        assert_eq!(decay_modes(4.0), ["alpha"]);
        assert_eq!(decay_modes(1.5), ["beta-", "n"]);
        assert_eq!(decay_modes(2.4), ["ec/beta+", "alpha"]);
        // Stripping the zeros is what the format's encoding amounts to, and
        // it leaves gamma (mode zero) unrepresentable.
        assert!(decay_modes(0.0).is_empty());
        // A ten is two digits, not the tenth mode, which is what the Python
        // reader makes of it too.
        assert_eq!(decay_modes(10.0), ["beta-", "gamma"]);
    }

    #[test]
    fn a_decay_mode_names_its_daughter() {
        let mode = DecayMode {
            parent: "In116_m1".to_string(),
            modes: vec!["beta-"],
            daughter_state: 0,
            ..Default::default()
        };
        // Beta- turns a neutron into a proton: Z rises, A does not.
        assert_eq!(mode.daughter().as_deref(), Some("Sn116"));

        let alpha = DecayMode {
            parent: "Am242".to_string(),
            modes: vec!["alpha"],
            ..Default::default()
        };
        assert_eq!(alpha.daughter().as_deref(), Some("Np238"));

        // A metastable daughter is named as one.
        let it = DecayMode {
            parent: "In116_m2".to_string(),
            modes: vec!["IT"],
            daughter_state: 1,
            ..Default::default()
        };
        assert_eq!(it.daughter().as_deref(), Some("In116_m1"));

        // Spontaneous fission has no single daughter, so the numbers are left
        // alone rather than guessed at.
        let sf = DecayMode {
            parent: "Cf252".to_string(),
            modes: vec!["sf"],
            ..Default::default()
        };
        assert_eq!(sf.daughter().as_deref(), Some("Cf252"));
    }

    #[test]
    fn reads_a_beta_emitter() {
        let d = decay(IN116M1);
        assert_eq!(d.nuclide.name, "In116_m1");
        assert_eq!(d.nuclide.isomeric_state, 1);
        assert!(!d.nuclide.stable);
        assert_eq!(d.nuclide.spin, Some(5.0));

        assert_eq!(d.half_life, Some((3257.4, 10.2)));
        let (lambda, _) = d.decay_constant().unwrap();
        // The decay constant and the half-life are the same fact.
        assert!((lambda * 3257.4 - std::f64::consts::LN_2).abs() < 1e-12);

        // One mode, beta-, to tin.
        assert_eq!(d.modes.len(), 1);
        assert_eq!(d.modes[0].modes, ["beta-"]);
        assert_eq!(d.modes[0].daughter().as_deref(), Some("Sn116"));
        assert_eq!(d.modes[0].branching_ratio.0, 1.0);

        // Four spectra, and the decay energy is the three average energies.
        assert_eq!(
            d.spectra.keys().copied().collect::<Vec<_>>(),
            ["beta-", "e-", "gamma", "xray"]
        );
        let sum: f64 = ["light", "electromagnetic", "heavy"]
            .iter()
            .map(|k| d.average_energies[k].0)
            .sum();
        assert_eq!(d.decay_energy().0, sum);
    }

    #[test]
    fn gammas_and_xrays_become_one_photon_source() {
        let d = decay(IN116M1);
        let sources = d.sources().unwrap();
        // Four radiation types, two particles: gamma and xray are both
        // photons, beta- and e- both electrons.
        assert_eq!(
            sources.keys().copied().collect::<Vec<_>>(),
            ["electron", "photon"]
        );

        let Univariate::Discrete(photons) = &sources["photon"] else {
            panic!("both photon spectra are discrete, so they merge into one");
        };
        // Every line of both spectra is there, and the energies ascend.
        let lines: usize = ["gamma", "xray"]
            .iter()
            .map(|r| d.spectra[r].discrete.len())
            .sum();
        assert_eq!(photons.x.len(), lines);
        assert!(photons.x.windows(2).all(|w| w[1] > w[0]));

        // The intensities are rates, so they scale with the decay constant.
        let (lambda, _) = d.decay_constant().unwrap();
        let norm = d.spectra["gamma"].discrete_normalization.0;
        let first = &d.spectra["gamma"].discrete[0];
        let i = photons.x.iter().position(|&e| e == first.energy.0).unwrap();
        assert_eq!(photons.p[i], lambda * norm * first.intensity.0);
    }

    #[test]
    fn an_isomeric_transition_goes_to_the_state_below() {
        let d = decay(IN116M2);
        assert_eq!(d.nuclide.name, "In116_m2");
        assert_eq!(d.modes.len(), 1);
        assert_eq!(d.modes[0].modes, ["IT"]);
        assert_eq!(d.modes[0].daughter_state, 1);
        assert_eq!(d.modes[0].daughter().as_deref(), Some("In116_m1"));
    }

    /// A half-life of zero means "not evaluated", so there is no decay
    /// constant and no source rates that scale with one. See issue #23.
    #[test]
    fn the_total_is_the_quadrature_of_the_components_it_reports() {
        // The split and the total must be the same fact, or a consumer that
        // carries the components and one that carries the total disagree while
        // both look right. Reading them from one list is what guarantees it.
        let d = decay(IN116M1);
        let (total, sigma) = d.decay_energy();
        let parts = d.decay_energy_components();

        let summed: f64 = parts.iter().map(|p| p.map_or(0.0, |(v, _)| v)).sum();
        let quadrature: f64 = parts
            .iter()
            .map(|p| p.map_or(0.0, |(_, s)| s * s))
            .sum::<f64>()
            .sqrt();
        assert!((total - summed).abs() <= 1e-9 * total.abs().max(1.0));
        assert!((sigma - quadrature).abs() <= 1e-9 * sigma.abs().max(1.0));

        // And they are the three recoverable ones by name, not the first three
        // of whatever order the map happens to iterate in. This is the guard
        // against the real hazard: AVERAGE_ENERGY_NAMES has seventeen entries
        // and the last fourteen decompose the first three, so a consumer that
        // sums the table rather than these names roughly doubles the heat.
        // (In116_m1 writes only the short form, so here the two happen to be
        // equal -- an evaluation writing the long form is where it would bite.)
        assert_eq!(parts.len(), DECAY_HEAT_ENERGY_NAMES.len());
        for (part, name) in parts.iter().zip(DECAY_HEAT_ENERGY_NAMES) {
            assert_eq!(*part, d.average_energies.get(name).copied(), "{name}");
        }
    }

    /// Xe136's zero half-life is the one thing wrong with its record, and an
    /// evaluated record has nothing reported: In116_m1 (beta-) and Sn117_m1,
    /// whose 315 keV transition is split between conversion electrons and
    /// photons and adds back up to its Q.
    #[test]
    fn a_zero_half_life_is_reported_and_an_evaluated_record_is_not() {
        const XE136: &[u8] = include_bytes!("../fixtures/dec-054_Xe_136.endf.xz");
        const SN117M1: &[u8] = include_bytes!("../fixtures/dec-050_Sn_117m1.endf.xz");
        assert_eq!(
            decay(XE136).inconsistencies(),
            vec![DecayInconsistency::ZeroHalfLife]
        );
        assert_eq!(decay(IN116M1).inconsistencies(), vec![]);
        assert_eq!(decay(SN117M1).inconsistencies(), vec![]);
    }

    /// ENDF/B-VIII.1 books 1.52 MeV of electrons and photons to Hf177_m1's
    /// 1.32 MeV isomeric transition, which has no neutrino to make up the
    /// difference.
    #[test]
    fn an_isomeric_transition_paying_out_more_than_its_q_is_reported() {
        const HF177M1: &[u8] = include_bytes!("../fixtures/dec-072_Hf_177m1.endf.xz");
        let found = decay(HF177M1).inconsistencies();
        let [DecayInconsistency::IsomericTransitionEnergy { q, recoverable }] = &found[..] else {
            panic!("{found:?}");
        };
        assert!((q - 1_315_450.0).abs() < 1.0, "{q}");
        assert!((recoverable - 1_518_972.4).abs() < 1.0, "{recoverable}");
        assert_eq!(found[0].label(), "isomeric_transition_energy");
    }

    /// Branching ratios more than a percent off one are reported, and a
    /// stable nuclide claims nothing to check.
    #[test]
    fn branching_ratios_off_one_are_reported() {
        let mut d = decay(IN116M1);
        d.modes[0].branching_ratio.0 -= 0.1;
        let sum: f64 = d.modes.iter().map(|m| m.branching_ratio.0).sum();
        assert_eq!(
            d.inconsistencies(),
            vec![DecayInconsistency::BranchingRatioSum { sum }]
        );
        d.nuclide.stable = true;
        assert!(d.inconsistencies().is_empty());
    }

    #[test]
    fn every_kind_has_a_label_in_the_list() {
        let kinds = [
            DecayInconsistency::ZeroHalfLife,
            DecayInconsistency::BranchingRatioSum { sum: 0.0 },
            DecayInconsistency::IsomericTransitionEnergy {
                q: 1.0,
                recoverable: 1.0,
            },
        ];
        for kind in &kinds {
            assert!(DecayInconsistency::LABELS.contains(&kind.label()));
        }
    }

    #[test]
    fn a_zero_half_life_has_no_decay_constant() {
        const XE136: &[u8] = include_bytes!("../fixtures/dec-054_Xe_136.endf.xz");
        let m = Material::from_str(&crate::testdata::text(XE136)).unwrap();
        let d = Decay::from_material(&m).unwrap();

        assert!(!d.nuclide.stable, "Xe136 is flagged unstable");
        assert_eq!(d.half_life, Some((0.0, 0.0)));
        assert_eq!(d.decay_constant(), None);
        // Every source rate is a multiple of the decay constant, so there is
        // nothing to report rather than a rate of zero.
        assert!(d.sources().unwrap().is_empty());

        // A nuclide with a real half-life is unaffected.
        const CS137: &[u8] = include_bytes!("../fixtures/dec-055_Cs_137.endf.xz");
        let m = Material::from_str(&crate::testdata::text(CS137)).unwrap();
        let d = Decay::from_material(&m).unwrap();
        let (lambda, _) = d.decay_constant().expect("Cs137 has a half-life");
        assert!(lambda > 0.0);
    }

    #[test]
    fn reads_independent_and_cumulative_yields() {
        const NFY: &[u8] = include_bytes!("../fixtures/synthetic-nfy.endf.xz");
        let m = Material::from_str(&crate::testdata::text(NFY)).unwrap();
        let fpy = FissionProductYields::from_material(&m).unwrap();

        assert_eq!(fpy.nuclide.name, "U235");
        assert_eq!(fpy.nuclide.atomic_number, 92);
        assert_eq!(fpy.nuclide.mass_number, 235);
        assert_eq!(fpy.energies, [0.0253, 5.0e5]);

        // The fast energy carries one product more than the thermal one, so a
        // reader that reuses NFP across energies would be caught here.
        assert_eq!(fpy.independent.len(), 2);
        assert_eq!(fpy.independent[0].len(), 3);
        assert_eq!(fpy.independent[1].len(), 4);
        assert_eq!(fpy.independent[0][1].name, "Xe135_m1");
        assert_eq!(fpy.independent[0][1].yield_, (0.0134, 0.0006));

        // Independent and cumulative differ, so returning one for the other
        // would be caught too.
        assert_eq!(fpy.cumulative[0][0].name, "Zr95");
        assert_eq!(fpy.cumulative[0][0].yield_, (0.0605, 0.0018));
        assert_ne!(fpy.independent[0][0].yield_, fpy.cumulative[0][0].yield_);
    }

    #[test]
    fn an_evaluation_with_no_decay_section_is_refused() {
        const AM244: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        assert!(Decay::from_material(&m).is_err());
        // The yields are simply absent rather than an error.
        let fpy = FissionProductYields::from_material(&m).unwrap();
        assert!(fpy.energies.is_empty());
    }

    /// Sn111 in ENDF/B-VIII.1 is a Nuclear Wallet Cards conversion: no
    /// spectra, and Q/3 in each of the light and electromagnetic slots.
    #[test]
    fn a_wallet_cards_conversion_is_a_placeholder() {
        const SN111: &[u8] = include_bytes!("../fixtures/dec-050_Sn_111.endf.xz");
        let m = Material::from_str(&crate::testdata::text(SN111)).unwrap();
        let d = Decay::from_material(&m).unwrap();
        assert!(d.mean_energy_placeholder);
        assert!(d.spectra.is_empty());
        let (light, _) = d.average_energies["light"];
        let (em, _) = d.average_energies["electromagnetic"];
        assert_eq!(light, em);
        // 2451.824 keV of Q, a third each way, and the sum is what a heat
        // calculation would otherwise take at face value.
        assert!((light - 2_451_824.0 / 3.0).abs() < 1.0);
        assert!((d.decay_energy().0 - 1_634_549.4).abs() < 1.0);
    }

    /// The same nuclide in JENDL-5.0 has an evaluated decay scheme.
    #[test]
    fn an_evaluated_decay_scheme_is_not_a_placeholder() {
        const SN111_JENDL: &[u8] = include_bytes!("../fixtures/dec-050-Sn-111.jendl5.endf.xz");
        let m = Material::from_str(&crate::testdata::text(SN111_JENDL)).unwrap();
        let d = Decay::from_material(&m).unwrap();
        assert!(!d.mean_energy_placeholder);
        assert!(!d.spectra.is_empty());
        assert!((d.decay_energy().0 - 693_315.8).abs() < 1.0);

        // In116m1, an ENSDF conversion with spectra, likewise.
        const IN116M1: &[u8] = include_bytes!("../fixtures/dec-049_In_116m1.endf.xz");
        let m = Material::from_str(&crate::testdata::text(IN116M1)).unwrap();
        assert!(!Decay::from_material(&m).unwrap().mean_energy_placeholder);
    }

    /// Sn117m is an evaluated 315 keV isomeric transition whose conversion
    /// electrons and photons carry almost the same energy: equal averages, but
    /// spectra and no beta branch, so not a placeholder.
    #[test]
    fn coincidentally_equal_averages_with_spectra_are_not_a_placeholder() {
        const SN117M1: &[u8] = include_bytes!("../fixtures/dec-050_Sn_117m1.endf.xz");
        let m = Material::from_str(&crate::testdata::text(SN117M1)).unwrap();
        let d = Decay::from_material(&m).unwrap();
        let (light, _) = d.average_energies["light"];
        let (em, _) = d.average_energies["electromagnetic"];
        assert!((light - em).abs() < 1.0e-3 * light, "{light} vs {em}");
        assert!(!d.spectra.is_empty());
        assert!(!d.mean_energy_placeholder);
    }

    /// The rule on its own: equal positive averages, and then either no
    /// spectra or the Q/3 arithmetic.
    #[test]
    fn the_placeholder_rule_needs_equal_energies_and_a_reason() {
        let decay = |light: f64, em: f64, spectra: bool, modes: Vec<(&'static str, f64, f64)>| {
            let mut d = Decay {
                average_energies: BTreeMap::from([
                    ("light", (light, 0.0)),
                    ("electromagnetic", (em, 0.0)),
                ]),
                ..Default::default()
            };
            if spectra {
                d.spectra.insert("gamma", DecaySpectrum::default());
            }
            for (mode, q, br) in modes {
                d.modes.push(DecayMode {
                    modes: vec![mode],
                    energy: (q, 0.0),
                    branching_ratio: (br, 0.0),
                    ..Default::default()
                });
            }
            d
        };
        // Wallet Cards: equal, no spectra.
        assert!(decay(
            817_274.7,
            817_274.7,
            false,
            vec![("ec/beta+", 2_451_824.0, 1.0)]
        )
        .placeholder_mean_energies());
        // Equal within a part in a thousand still counts (Co62m: 1761.1 vs 1761.3 keV).
        assert!(decay(
            1_761_075.0,
            1_761_295.0,
            false,
            vec![("beta-", 5_336_591.0, 0.99), ("IT", 22_000.0, 0.01)]
        )
        .placeholder_mean_energies());
        // JENDL-5.0 style: Q/3 averages beside theoretical spectra.
        assert!(decay(
            814_806.7,
            814_904.2,
            true,
            vec![("ec/beta+", 2_444_420.0, 1.0)]
        )
        .placeholder_mean_energies());
        // Equal by coincidence, with spectra and no beta branch: an evaluation.
        assert!(
            !decay(157_811.6, 157_820.3, true, vec![("IT", 314_580.0, 1.0)])
                .placeholder_mean_energies()
        );
        // Not equal.
        assert!(
            !decay(1000.0, 1002.0, false, vec![("beta-", 3000.0, 1.0)]).placeholder_mean_energies()
        );
        // Nothing stated.
        assert!(!decay(0.0, 0.0, false, vec![]).placeholder_mean_energies());
        assert!(!Decay::default().placeholder_mean_energies());
    }
}
