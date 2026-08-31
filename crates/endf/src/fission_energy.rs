//! Energy released by fission, split into its components.
//!
//! Built from MF=1 MT=458. Each component is either a polynomial in incident
//! energy or a tabulation, and which one it is varies per component within a
//! single evaluation: U235, U238 and Pu239 all pair a tabulated prompt photon
//! term with a polynomial delayed one. That distinction is preserved here
//! rather than collapsed, because it is what the converted data has to carry.

use crate::data::EV_PER_MEV;
use crate::error::{Error, Result};
use crate::function::{Polynomial, Tabulated1D};
use crate::material::Material;
use crate::mf::mf1::{FissionEnergyRelease as Mt458Component, Nu};

/// The components of the fission energy release, in the order MT=458 stores
/// them. `recoverable` and `total` are recomputed from the others rather than
/// read, so they do not appear here.
pub const COMPONENT_NAMES: [&str; 9] = [
    "fragments",
    "prompt_neutrons",
    "delayed_neutrons",
    "prompt_photons",
    "delayed_photons",
    "betas",
    "neutrinos",
    "recoverable",
    "total",
];

/// One component's energy release as a function of incident energy, in eV.
#[derive(Debug, Clone, PartialEq)]
pub enum Component {
    Polynomial(Polynomial),
    Tabulated(Tabulated1D),
}

impl Component {
    pub fn eval(&self, e: f64) -> f64 {
        match self {
            Component::Polynomial(p) => p.eval(e),
            Component::Tabulated(t) => t.eval(e),
        }
    }
}

/// Energy released by fission, by component. Every field is a function of the
/// incident neutron energy in eV, returning eV.
#[derive(Debug, Clone, PartialEq)]
pub struct FissionEnergyRelease {
    /// Kinetic energy of the fission fragments.
    pub fragments: Component,
    /// Kinetic energy of the prompt fission neutrons.
    pub prompt_neutrons: Component,
    /// Kinetic energy of the delayed fission neutrons.
    pub delayed_neutrons: Component,
    /// Energy of the prompt fission photons.
    pub prompt_photons: Component,
    /// Energy of the delayed fission photons.
    pub delayed_photons: Component,
    /// Energy of the delayed beta particles.
    pub betas: Component,
    /// Energy carried off by neutrinos, which is not recoverable.
    pub neutrinos: Component,
}

impl FissionEnergyRelease {
    /// Everything except the neutrinos, which escape.
    pub fn recoverable(&self, e: f64) -> f64 {
        self.fragments.eval(e)
            + self.prompt_neutrons.eval(e)
            + self.delayed_neutrons.eval(e)
            + self.prompt_photons.eval(e)
            + self.delayed_photons.eval(e)
            + self.betas.eval(e)
    }

    /// Every component, neutrinos included.
    pub fn total(&self, e: f64) -> f64 {
        self.recoverable(e) + self.neutrinos.eval(e)
    }

    /// Prompt fission Q value: the prompt release less the incident energy.
    pub fn q_prompt(&self, e: f64) -> f64 {
        self.fragments.eval(e) + self.prompt_neutrons.eval(e) + self.prompt_photons.eval(e) - e
    }

    pub fn q_recoverable(&self, e: f64) -> f64 {
        self.recoverable(e) - e
    }

    pub fn q_total(&self, e: f64) -> f64 {
        self.total(e) - e
    }

    /// Read the fission energy release from an evaluation.
    ///
    /// `nu` is only consulted when the evaluation gives a single coefficient
    /// per component, in which case the prompt neutron term takes its energy
    /// dependence from the Sher-Beck formula, which needs nu-bar. Pass the
    /// prompt or total neutron yield from the fission reaction; the delayed
    /// fraction is small enough that ENDF-102 does not distinguish them.
    pub fn from_material(material: &Material, nu: Option<&Nu>) -> Result<FissionEnergyRelease> {
        let metadata = material.mf1_mt451().ok_or(Error::Unsupported {
            what: "an evaluation with no MF=1 MT=451 section",
        })?;
        if metadata.lfi != 1 {
            return Err(Error::Unsupported {
                what: "fission energy release from a non-fissionable evaluation",
            });
        }
        let section = material.mf1_mt458().ok_or(Error::Unsupported {
            what: "an evaluation with no MF=1 MT=458 section",
        })?;

        let npoly = section.nply;
        let mut functions: Vec<Option<Component>> = vec![None; COMPONENT_NAMES.len()];

        for (i, name) in COMPONENT_NAMES.iter().enumerate() {
            // These two are recomputed from the components above.
            if matches!(*name, "recoverable" | "total") {
                continue;
            }
            let Some(Mt458Component::Polynomial(pairs)) = section.components.get(i) else {
                continue;
            };
            let mut coeffs: Vec<f64> = pairs.iter().map(|&(c, _)| c).collect();

            // ENDF/B-VII.1 left the second-order coefficients in MeV by
            // mistake. A 5 MeV neutron cannot change the release by more than
            // 100 MeV, so a term that large is the units error rather than
            // physics.
            if npoly == 2 {
                if let Some(c2) = coeffs.get_mut(2) {
                    if c2.abs() * (5.0e6f64).powi(2) > 1.0e8 {
                        *c2 /= EV_PER_MEV;
                    }
                }
            }

            if npoly > 0 {
                functions[i] = Some(Component::Polynomial(Polynomial::new(coeffs)));
                continue;
            }

            // A single coefficient, so the energy dependence is Sher-Beck.
            let zeroth = coeffs.first().copied().unwrap_or(0.0);
            let func = match *name {
                "delayed_photons" | "betas" => {
                    Component::Polynomial(Polynomial::new(vec![zeroth, -0.075]))
                }
                "neutrinos" => Component::Polynomial(Polynomial::new(vec![zeroth, -0.105])),
                "prompt_neutrons" => sher_beck_prompt_neutrons(zeroth, nu)?,
                _ => Component::Polynomial(Polynomial::new(coeffs)),
            };
            functions[i] = Some(func);
        }

        // A tabulated component replaces the polynomial form. IFC indexes the
        // component list, one-based.
        for (i, component) in section.components.iter().enumerate() {
            if let Mt458Component::Tabulated { eifc, .. } = component {
                functions[i] = Some(Component::Tabulated(eifc.clone()));
            }
        }

        let take = |i: usize| -> Result<Component> {
            functions[i].clone().ok_or(Error::Unsupported {
                what: "an MF=1 MT=458 section missing a required component",
            })
        };

        Ok(FissionEnergyRelease {
            fragments: take(0)?,
            prompt_neutrons: take(1)?,
            delayed_neutrons: take(2)?,
            prompt_photons: take(3)?,
            delayed_photons: take(4)?,
            betas: take(5)?,
            neutrinos: take(6)?,
        })
    }
}

/// The Sher-Beck energy dependence of the prompt neutron kinetic energy, which
/// is written in terms of nu-bar.
fn sher_beck_prompt_neutrons(zeroth: f64, nu: Option<&Nu>) -> Result<Component> {
    let nu = nu.ok_or(Error::Unsupported {
        what: "the Sher-Beck fission energy release without a nu-bar to build it from",
    })?;
    Ok(match nu {
        Nu::Tabulated(t) => {
            let y0 = t.y.first().copied().unwrap_or(0.0);
            let y =
                t.x.iter()
                    .zip(&t.y)
                    .map(|(&x, &y)| zeroth + 1.307 * x - 8.07e6 * (y - y0))
                    .collect();
            Component::Tabulated(Tabulated1D {
                x: t.x.clone(),
                y,
                breakpoints: t.breakpoints.clone(),
                interpolation: t.interpolation.clone(),
                threshold_idx: None,
            })
        }
        Nu::Polynomial(coef) => {
            let mut out = vec![zeroth];
            if coef.len() <= 1 {
                out.push(1.307);
            } else {
                out.push(1.307 - 8.07e6 * coef[1]);
                out.extend(coef[2..].iter().map(|&c| -8.07e6 * c));
            }
            Component::Polynomial(Polynomial::new(out))
        }
        Nu::Absent => {
            return Err(Error::Unsupported {
                what: "the Sher-Beck fission energy release without a nu-bar to build it from",
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");

    /// Values taken from the Python implementation on the same evaluation, so
    /// this is a parity test and not just a plausibility one. The golden
    /// harness covers the parsed sections; the derived quantities here are
    /// pinned separately because they are computed rather than read.
    #[test]
    fn matches_the_python_fission_energy_release() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let nu = m.mf1_mt452(452).map(|s| &s.nu);
        let fer = FissionEnergyRelease::from_material(&m, nu).unwrap();

        // (incident energy, fragments, prompt neutrons, prompt photons,
        //  delayed photons, neutrinos, total, prompt Q)
        let cases = [
            (
                0.0253,
                180560600.0,
                6651000.01,
                6949000.0,
                5106196.998,
                7058463.997,
                211579126.0,
                194160600.0,
            ),
            (
                1.0e6,
                180015126.4,
                7029000.0,
                6965930.0,
                5031197.0,
                6958464.0,
                211178500.4,
                193010056.4,
            ),
            (
                1.4e7,
                172923969.6,
                11943000.0,
                7186020.0,
                4056197.0,
                5658464.0,
                205970367.6,
                178052989.6,
            ),
        ];

        for (e, frag, pn, pp, dp, nu_, total, q_prompt) in cases {
            let close = |got: f64, want: f64, what: &str| {
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs(),
                    "at E={e}: {what} is {got}, Python gives {want}"
                );
            };
            close(fer.fragments.eval(e), frag, "fragments");
            close(fer.prompt_neutrons.eval(e), pn, "prompt neutrons");
            close(fer.prompt_photons.eval(e), pp, "prompt photons");
            close(fer.delayed_photons.eval(e), dp, "delayed photons");
            close(fer.neutrinos.eval(e), nu_, "neutrinos");
            close(fer.total(e), total, "total");
            close(fer.q_prompt(e), q_prompt, "prompt Q");
        }

        // The neutrinos escape, so the recoverable release is strictly less.
        assert!(fer.recoverable(0.0253) < fer.total(0.0253));
    }

    #[test]
    fn a_non_fissionable_evaluation_is_rejected() {
        const PHOTOAT_H: &[u8] = include_bytes!("../fixtures/photoat-001_H_000.endf.xz");
        let m = Material::from_str(&crate::testdata::text(PHOTOAT_H)).unwrap();
        assert!(FissionEnergyRelease::from_material(&m, None).is_err());
    }
}
