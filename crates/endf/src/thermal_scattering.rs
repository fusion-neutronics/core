//! Thermal neutron scattering for one bound material, from a thermal
//! scattering law evaluation (NSUB=12).
//!
//! MF=7 gives the scattering law of a nuclide bound in a material (H in
//! water, C in graphite) in two parts: MT=4, the incoherent inelastic law
//! S(alpha, beta) at each temperature, and MT=2, the elastic part, which is
//! closed form (Bragg edges for a crystal, a Debye-Waller integral for an
//! incoherent scatterer, or both). MT=451, when present, names the atoms the
//! law describes. The formulas are those of the ENDF-6 manual, BNL-203218-2018
//! (section 7).
//!
//! This is what the evaluation says and nothing more. The inelastic law is
//! not sampleable as tabulated: turning it into outgoing energy and angle
//! distributions (the alpha-beta integration, the short-collision-time tail)
//! is a processing step done elsewhere.

use crate::data::K_BOLTZMANN;
use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};
use crate::material::Material;
use crate::mf::mf7::{Mf7Mt2, Mf7Mt4, ThermalElement};

/// How a secondary (non-principal) atom of the material scatters, from its
/// analytic-model flag B(6i+1) in MT=4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryModel {
    /// The short-collision-time approximation, which has its own effective
    /// temperature.
    ShortCollisionTime,
    /// Free gas.
    FreeGas,
    /// Diffusive motion.
    Diffusive,
}

/// A secondary atom of the material, described by an analytic law.
#[derive(Debug, Clone, PartialEq)]
pub struct SecondaryAtom {
    pub model: SecondaryModel,
    /// Its free-atom scattering cross section, in barns, for all atoms of
    /// this kind in the molecule together.
    pub free_atom_xs: f64,
    /// Its mass in neutron masses.
    pub awr: f64,
    /// How many atoms of this kind the molecule has.
    pub count: f64,
}

/// The incoherent inelastic law S(alpha, beta, T) of the principal atom, as
/// tabulated.
#[derive(Debug, Clone, PartialEq)]
pub struct InelasticLaw {
    /// The alpha grid, shared by every beta and temperature.
    pub alpha: Vec<f64>,
    /// The beta grid.
    pub beta: Vec<f64>,
    /// Interpolation in alpha: the regions of the first beta's TAB1, which
    /// every beta shares.
    pub alpha_breakpoints: Vec<i32>,
    pub alpha_interpolation: Vec<i32>,
    /// Interpolation in beta, from the TAB2 record.
    pub beta_breakpoints: Vec<i32>,
    pub beta_interpolation: Vec<i32>,
    /// S at each temperature, `s[t][i_beta * alpha.len() + i_alpha]`. When
    /// [`ThermalScattering::lln`] is set these are ln S.
    pub s: Vec<Vec<f64>>,
    /// Interpolation between each temperature and the previous one (the
    /// first is 0, having no previous one).
    pub temperature_interpolation: Vec<i64>,
}

impl InelasticLaw {
    /// S (or ln S) at grid point `(i_alpha, i_beta)` of temperature `t`.
    pub fn s_at(&self, t: usize, i_alpha: usize, i_beta: usize) -> f64 {
        self.s[t][i_beta * self.alpha.len() + i_alpha]
    }
}

/// Coherent elastic scattering, off the Bragg edges of a crystal.
#[derive(Debug, Clone, PartialEq)]
pub struct CoherentElastic {
    /// The temperatures it is given at, in Kelvin.
    pub temperatures: Vec<f64>,
    /// The Bragg edges, in eV.
    pub bragg_edges: Vec<f64>,
    /// The cumulative structure factor at each edge, per temperature, in
    /// eV-barns: `factors[t][i]` is the sum over every edge up to and
    /// including edge `i`.
    pub factors: Vec<Vec<f64>>,
}

impl CoherentElastic {
    /// The cross section in barns at `energy` (eV) for temperature index `t`:
    /// the cumulative factor of the last edge at or below `energy`, over
    /// `energy`. Zero below the first edge.
    pub fn xs(&self, t: usize, energy: f64) -> f64 {
        let n = self.bragg_edges.partition_point(|&e| e <= energy);
        if n == 0 {
            0.0
        } else {
            self.factors[t][n - 1] / energy
        }
    }
}

/// Incoherent elastic scattering, from the Debye-Waller integral.
#[derive(Debug, Clone, PartialEq)]
pub struct IncoherentElastic {
    /// The characteristic bound cross section in barns, as the evaluation
    /// gives it (SB in the format).
    pub bound_xs: f64,
    /// The Debye-Waller integral W'(T) in 1/eV against temperature in Kelvin.
    pub debye_waller: Tabulated1D,
}

impl IncoherentElastic {
    /// The cross section in barns at `energy` (eV) and `temperature` (K):
    /// `(SB / 2) (1 - exp(-4 E W')) / (2 E W')`.
    pub fn xs(&self, temperature: f64, energy: f64) -> f64 {
        let w = self.debye_waller.eval(temperature);
        let x = 2.0 * energy * w;
        if x == 0.0 {
            return self.bound_xs;
        }
        self.bound_xs / 2.0 * (1.0 - (-2.0 * x).exp()) / x
    }
}

/// The elastic part of a thermal scattering law (MF=7 MT=2), by its LTHR.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalElastic {
    Coherent(CoherentElastic),
    Incoherent(IncoherentElastic),
    /// Both, whose cross sections add.
    Mixed {
        coherent: CoherentElastic,
        incoherent: IncoherentElastic,
    },
}

impl ThermalElastic {
    /// The LTHR flag this representation is written with.
    pub fn lthr(&self) -> i64 {
        match self {
            ThermalElastic::Coherent(_) => 1,
            ThermalElastic::Incoherent(_) => 2,
            ThermalElastic::Mixed { .. } => 3,
        }
    }
}

/// One thermal scattering law evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalScattering {
    /// The material number.
    pub mat: i32,
    /// ZA as the evaluation writes it, which names the material, not a
    /// nuclide (134 is CH4).
    pub za: i64,
    /// The ZSYMAM text, such as `"H(H2O)"`.
    pub zsymam: Option<String>,
    /// The principal atom's mass in neutron masses (A0 in the format).
    pub awr: f64,
    /// The principal atom's free-atom scattering cross section in barns, for
    /// all principal atoms of the molecule together. Zero when the principal
    /// atom is described by an analytic law, in which case no S(alpha, beta)
    /// is tabulated.
    pub free_atom_xs: f64,
    /// The energy above which the law is not to be used, in eV.
    pub energy_max: f64,
    /// The value epsilon the evaluation gives for the principal atom, in eV.
    pub epsilon: f64,
    /// How many principal atoms the molecule has (M0).
    pub principal_count: f64,
    /// 1 when alpha and beta are in units of kT at 0.0253 eV, 0 when in kT
    /// at the actual temperature.
    pub lat: i64,
    /// True when S is symmetric in beta (LASYM = 0).
    pub symmetric: bool,
    /// 1 when S is stored as its natural logarithm.
    pub lln: i64,
    pub secondary: Vec<SecondaryAtom>,
    /// The temperatures of the inelastic law, in Kelvin. For a principal atom
    /// with an analytic law, the temperatures of its effective temperature
    /// table.
    pub temperatures: Vec<f64>,
    /// The tabulated inelastic law, absent for an analytic principal atom.
    pub inelastic: Option<InelasticLaw>,
    /// Effective temperatures against material temperature: the principal
    /// atom's first, then each short-collision-time secondary atom's.
    pub effective_temperature: Vec<Tabulated1D>,
    /// The elastic part, absent when the evaluation has no MT=2.
    pub elastic: Option<ThermalElastic>,
    /// The atoms MF=7 MT=451 names, when the evaluation has that section.
    pub atoms: Option<Vec<ThermalElement>>,
}

fn unsupported(what: &'static str) -> Error {
    Error::Unsupported { what }
}

impl ThermalScattering {
    /// Build from a thermal scattering law material.
    pub fn from_endf(material: &Material) -> Result<Self> {
        let mt4 = material.mf7_mt4().ok_or(unsupported(
            "a thermal scattering material with no MF=7 MT=4",
        ))?;
        let b = &mt4.b;
        let get = |i: usize| -> Result<f64> {
            b.get(i).copied().ok_or(unsupported(
                "an MF=7 MT=4 B list shorter than the format requires",
            ))
        };
        let secondary = (0..mt4.ns.max(0) as usize)
            .map(|i| {
                let base = 6 * (i + 1);
                let model = match get(base)? {
                    0.0 => SecondaryModel::ShortCollisionTime,
                    1.0 => SecondaryModel::FreeGas,
                    2.0 => SecondaryModel::Diffusive,
                    _ => {
                        return Err(unsupported(
                            "an MF=7 MT=4 secondary atom model other than 0, 1 or 2",
                        ))
                    }
                };
                Ok(SecondaryAtom {
                    model,
                    free_atom_xs: get(base + 1)?,
                    awr: get(base + 2)?,
                    count: get(base + 5)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let inelastic = inelastic_law(mt4)?;
        let temperatures = match &inelastic {
            Some(_) => {
                let first = &mt4.beta_data[0];
                std::iter::once(first.t)
                    .chain(first.others.iter().map(|o| o.t))
                    .collect()
            }
            None => mt4.teff.first().map(|t| t.x.clone()).unwrap_or_default(),
        };

        let elastic = material.mf7_mt2().map(elastic_part).transpose()?;
        let header = material.mf1_mt451();

        Ok(ThermalScattering {
            mat: material.mat,
            za: mt4.za,
            zsymam: header.and_then(|h| h.zsymam.as_ref().map(|s| s.trim().to_string())),
            awr: get(2)?,
            free_atom_xs: get(0)?,
            energy_max: get(3)?,
            epsilon: get(1)?,
            principal_count: get(5)?,
            lat: mt4.lat,
            symmetric: mt4.lasym == 0,
            lln: mt4.lln,
            secondary,
            temperatures,
            inelastic,
            effective_temperature: mt4.teff.clone(),
            elastic,
            atoms: material.mf7_mt451().map(|s| s.elements.clone()),
        })
    }

    /// kT in eV at each of [`Self::temperatures`].
    pub fn kts(&self) -> Vec<f64> {
        self.temperatures.iter().map(|t| K_BOLTZMANN * t).collect()
    }

    /// The ZA of every atom MF=7 MT=451 names, principal first, or `None`
    /// when the evaluation has no MT=451.
    pub fn atom_zas(&self) -> Option<Vec<i64>> {
        self.atoms.as_ref().map(|elements| {
            elements
                .iter()
                .flat_map(|e| e.zai.iter().map(|&z| z as i64))
                .collect()
        })
    }
}

fn tab2_regions(t: &Tabulated2D) -> (Vec<i32>, Vec<i32>) {
    (t.breakpoints.clone(), t.interpolation.clone())
}

fn inelastic_law(mt4: &Mf7Mt4) -> Result<Option<InelasticLaw>> {
    let Some(first) = mt4.beta_data.first() else {
        return Ok(None);
    };
    let alpha = first.s.x.clone();
    let n_alpha = alpha.len();
    let n_temps = 1 + first.others.len();
    let n_beta = mt4.beta_data.len();
    let mut s = vec![vec![0.0; n_alpha * n_beta]; n_temps];
    let mut beta = Vec::with_capacity(n_beta);
    for (i, law) in mt4.beta_data.iter().enumerate() {
        if law.s.x != alpha || law.others.len() + 1 != n_temps {
            return Err(unsupported(
                "an MF=7 MT=4 law whose betas do not share one alpha grid and one set of temperatures",
            ));
        }
        if law.s.breakpoints != first.s.breakpoints || law.s.interpolation != first.s.interpolation
        {
            return Err(unsupported(
                "an MF=7 MT=4 law whose betas interpolate in alpha differently",
            ));
        }
        beta.push(law.beta);
        s[0][i * n_alpha..(i + 1) * n_alpha].copy_from_slice(&law.s.y);
        for (t, other) in law.others.iter().enumerate() {
            if other.s.len() != n_alpha {
                return Err(unsupported(
                    "an MF=7 MT=4 temperature whose S list is not the length of the alpha grid",
                ));
            }
            s[t + 1][i * n_alpha..(i + 1) * n_alpha].copy_from_slice(&other.s);
        }
    }
    let temperature_interpolation = std::iter::once(0)
        .chain(first.others.iter().map(|o| o.li))
        .collect();
    let (beta_breakpoints, beta_interpolation) =
        mt4.beta_int.as_ref().map(tab2_regions).unwrap_or_default();
    Ok(Some(InelasticLaw {
        alpha,
        beta,
        alpha_breakpoints: first.s.breakpoints.clone(),
        alpha_interpolation: first.s.interpolation.clone(),
        beta_breakpoints,
        beta_interpolation,
        s,
        temperature_interpolation,
    }))
}

fn coherent(c: &crate::mf::mf7::CoherentElastic) -> Result<CoherentElastic> {
    let bragg_edges = c.s.x.clone();
    let mut factors = vec![c.s.y.clone()];
    let mut temperatures = vec![c.t];
    for o in &c.others {
        if o.s.len() != bragg_edges.len() {
            return Err(unsupported(
                "an MF=7 MT=2 coherent temperature whose factors are not one per Bragg edge",
            ));
        }
        temperatures.push(o.t);
        factors.push(o.s.clone());
    }
    Ok(CoherentElastic {
        temperatures,
        bragg_edges,
        factors,
    })
}

fn incoherent(i: &crate::mf::mf7::IncoherentElastic) -> IncoherentElastic {
    IncoherentElastic {
        bound_xs: i.sb,
        debye_waller: i.w.clone(),
    }
}

fn elastic_part(mt2: &Mf7Mt2) -> Result<ThermalElastic> {
    let missing = || unsupported("an MF=7 MT=2 section missing the part its LTHR names");
    match mt2.lthr {
        1 => Ok(ThermalElastic::Coherent(coherent(
            mt2.coherent.as_ref().ok_or_else(missing)?,
        )?)),
        2 => Ok(ThermalElastic::Incoherent(incoherent(
            mt2.incoherent.as_ref().ok_or_else(missing)?,
        ))),
        3 => Ok(ThermalElastic::Mixed {
            coherent: coherent(mt2.coherent.as_ref().ok_or_else(missing)?)?,
            incoherent: incoherent(mt2.incoherent.as_ref().ok_or_else(missing)?),
        }),
        _ => Err(unsupported("an MF=7 MT=2 LTHR other than 1, 2 or 3")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(fixture: &[u8]) -> ThermalScattering {
        let m = Material::from_str(&crate::testdata::text(fixture)).unwrap();
        ThermalScattering::from_endf(&m).unwrap()
    }

    const SIO2: &[u8] = include_bytes!("../fixtures/tsl-SiO2-beta.endf.xz");
    const ZR_ZRH: &[u8] = include_bytes!("../fixtures/tsl-ZrinZrH.endf.xz");
    const CH4: &[u8] = include_bytes!("../fixtures/tsl-s-CH4.endf.xz");

    #[test]
    fn a_coherent_crystal_with_a_short_collision_time_secondary() {
        let t = load(SIO2);
        assert_eq!(t.temperatures.len(), 2);
        assert_eq!(t.secondary.len(), 1);
        assert_eq!(t.secondary[0].model, SecondaryModel::ShortCollisionTime);
        // The principal atom's effective temperature, then the SCT atom's.
        assert_eq!(t.effective_temperature.len(), 2);
        let law = t.inelastic.as_ref().unwrap();
        assert_eq!(law.s.len(), 2);
        assert_eq!(law.s[0].len(), law.alpha.len() * law.beta.len());
        // The second temperature carries its own LI, not the outer LT.
        assert_eq!(law.temperature_interpolation.len(), 2);
        assert_ne!(
            law.temperature_interpolation[1], 1,
            "LT, not LI, was stored"
        );
        let Some(ThermalElastic::Coherent(c)) = &t.elastic else {
            panic!("SiO2 is coherent")
        };
        assert_eq!(c.temperatures, t.temperatures);
        // Zero below the first Bragg edge, factor over E above it.
        let e0 = c.bragg_edges[0];
        assert_eq!(c.xs(0, 0.5 * e0), 0.0);
        let e = 0.5 * (c.bragg_edges[0] + c.bragg_edges[1]);
        assert_eq!(c.xs(0, e), c.factors[0][0] / e);
    }

    #[test]
    fn an_incoherent_scatterer_at_eight_temperatures() {
        let t = load(ZR_ZRH);
        assert_eq!(t.temperatures.len(), 8);
        assert!(t.symmetric);
        let Some(ThermalElastic::Incoherent(i)) = &t.elastic else {
            panic!("Zr in ZrH is incoherent")
        };
        // The cross section tends to SB as E W' goes to zero and falls as
        // 1/E at high energy.
        let lo = i.xs(296.0, 1e-9);
        assert!((lo - i.bound_xs).abs() < 1e-6 * i.bound_xs);
        let w = i.debye_waller.eval(296.0);
        let e = 10.0 / w;
        assert!((i.xs(296.0, e) - i.bound_xs / 2.0 / (2.0 * e * w)).abs() < 1e-9);
    }

    #[test]
    fn mixed_elastic_adds_both_parts() {
        let coherent = CoherentElastic {
            temperatures: vec![296.0],
            bragg_edges: vec![1e-3, 2e-3],
            factors: vec![vec![1e-3, 3e-3]],
        };
        let incoherent = IncoherentElastic {
            bound_xs: 4.0,
            debye_waller: Tabulated1D::new(vec![296.0, 400.0], vec![10.0, 12.0]),
        };
        let mixed = ThermalElastic::Mixed {
            coherent: coherent.clone(),
            incoherent: incoherent.clone(),
        };
        assert_eq!(mixed.lthr(), 3);
        assert_eq!(coherent.xs(0, 2.5e-3), 3e-3 / 2.5e-3);
        assert_eq!(
            coherent.xs(0, 1e-3),
            1.0,
            "an energy on an edge takes that edge"
        );
    }

    #[test]
    fn a_free_gas_secondary_has_no_effective_temperature() {
        let t = load(CH4);
        assert_eq!(t.secondary[0].model, SecondaryModel::FreeGas);
        assert_eq!(t.effective_temperature.len(), 1);
        assert_eq!(t.lat, 0);
        assert!(t.atoms.is_none() || t.atom_zas().is_some());
    }
}
