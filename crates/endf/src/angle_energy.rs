//! Joint distributions of secondary particle angle and energy.
//!
//! A reaction product's angle and energy may be described independently
//! ([`UncorrelatedAngleEnergy`]), through the Kalbach-Mann systematics
//! ([`KalbachMann`]), as an angular distribution conditional on the outgoing
//! energy ([`CorrelatedAngleEnergy`]), or by N-body phase space kinematics
//! ([`NBodyPhaseSpace`]).
//!
//! These are the processed forms an ACE table holds. The ENDF equivalents live
//! in [`crate::mf::mf6`], which describes the file rather than interpreting it.

use crate::ace::Table;
use crate::error::{Error, Result};
use crate::function::Tabulated1D;
use crate::mf::mf4::AngleDistribution;
use crate::mf::mf5::{ace_incident_grid, ace_outgoing_energy, EnergyDistribution};
use crate::univariate::{Interpolation, Tabular, Uniform, Univariate};

/// A distribution in secondary angle and energy.
#[derive(Debug, Clone, PartialEq)]
pub enum AngleEnergy {
    Uncorrelated(UncorrelatedAngleEnergy),
    KalbachMann(KalbachMann),
    Correlated(CorrelatedAngleEnergy),
    NBodyPhaseSpace(NBodyPhaseSpace),
}

/// Angle and energy sampled independently of each other.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UncorrelatedAngleEnergy {
    /// The outgoing angle, when one is given here rather than in the AND block.
    pub angle: Option<AngleDistribution>,
    /// The outgoing energy.
    pub energy: Option<EnergyDistribution>,
}

/// Kalbach-Mann systematics.
///
/// The outgoing energy is tabulated, and the angular distribution at each
/// outgoing energy follows from a precompound fraction and a slope.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KalbachMann {
    pub breakpoints: Vec<i32>,
    pub interpolation: Vec<i32>,
    /// Incident energies in eV.
    pub energy: Vec<f64>,
    /// The outgoing energy distribution at each incident energy.
    pub energy_out: Vec<Univariate>,
    /// The precompound fraction `r` against outgoing energy, one per incident
    /// energy.
    pub precompound: Vec<Tabulated1D>,
    /// The Kalbach-Chadwick slope `a`, likewise.
    pub slope: Vec<Tabulated1D>,
}

/// An angular distribution given for each outgoing energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CorrelatedAngleEnergy {
    pub breakpoints: Vec<i32>,
    pub interpolation: Vec<i32>,
    /// Incident energies in eV.
    pub energy: Vec<f64>,
    /// The outgoing energy distribution at each incident energy.
    pub energy_out: Vec<Univariate>,
    /// The scattering cosine for each pair of incident and outgoing energies.
    pub mu: Vec<Vec<Univariate>>,
}

/// N-body phase space kinematics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NBodyPhaseSpace {
    /// Total mass of the product particles.
    pub total_mass: f64,
    pub n_particles: i64,
    /// Of the target nuclide.
    pub atomic_weight_ratio: f64,
    /// The reaction Q value in eV.
    pub q_value: f64,
}

impl AngleEnergy {
    /// Read an angle-energy distribution from an ACE table.
    ///
    /// `location_dist` is the start of the block holding it, e.g. JXS(11) for
    /// DLW; `location_start` is the one-based offset of this reaction's array
    /// within it. `q_reaction` is needed only by law 66.
    pub fn from_ace(
        table: &Table,
        location_dist: i64,
        location_start: i64,
        q_reaction: Option<f64>,
    ) -> Result<AngleEnergy> {
        let xss = &table.xss;
        let at = |i: i64| -> f64 {
            usize::try_from(i)
                .ok()
                .and_then(|i| xss.get(i).copied())
                .unwrap_or(0.0)
        };

        let idx = location_dist + location_start - 1;
        let law = at(idx + 1) as i64;
        let location_data = at(idx + 2) as i64;

        // Where this law's own parameters begin.
        let idx = (location_dist + location_data - 1).max(0) as usize;

        let uncorrelated = |energy: EnergyDistribution| {
            AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
                angle: None,
                energy: Some(energy),
            })
        };

        Ok(match law {
            2 => uncorrelated(EnergyDistribution::discrete_photon_from_ace(table, idx)),
            3 | 33 => uncorrelated(EnergyDistribution::level_inelastic_from_ace(xss, idx)),
            4 => uncorrelated(EnergyDistribution::continuous_tabular_from_ace(
                xss,
                idx,
                location_dist,
            )?),
            // Law 5 is the general evaporation spectrum, which neither this
            // reader nor the Python one nor OpenMC implements. Both of those
            // now raise NotImplementedError; this is the same refusal. See
            // issue #19.
            5 => {
                return Err(Error::Unsupported {
                    what: "ACE law 5, the general evaporation spectrum",
                })
            }
            7 => uncorrelated(EnergyDistribution::maxwell_from_ace(xss, idx)),
            9 => uncorrelated(EnergyDistribution::evaporation_from_ace(xss, idx)),
            11 => uncorrelated(EnergyDistribution::watt_from_ace(xss, idx)),
            44 => AngleEnergy::KalbachMann(KalbachMann::from_ace(xss, idx, location_dist)?),
            61 => {
                AngleEnergy::Correlated(CorrelatedAngleEnergy::from_ace(xss, idx, location_dist)?)
            }
            66 => {
                let q_value = q_reaction.ok_or(Error::Unsupported {
                    what: "ACE law 66 without the reaction it belongs to",
                })?;
                AngleEnergy::NBodyPhaseSpace(NBodyPhaseSpace::from_ace(table, idx, q_value))
            }
            _ => {
                return Err(Error::Unsupported {
                    what: "this ACE secondary energy distribution law",
                })
            }
        })
    }
}

impl KalbachMann {
    /// Read a Kalbach-Mann distribution from an ACE table's XSS array.
    ///
    /// `idx` is where the law's data begins (`LDIS + LOCC - 1`) and `ldis` the
    /// start of the energy distribution block.
    pub fn from_ace(xss: &[f64], idx: usize, ldis: i64) -> Result<KalbachMann> {
        let grid = ace_incident_grid(xss, idx);

        let n = grid.energy.len();
        let mut energy_out = Vec::with_capacity(n);
        let mut precompound = Vec::with_capacity(n);
        let mut slope = Vec::with_capacity(n);
        for &loc in &grid.loc_dist {
            let idx = (ldis + loc - 1).max(0) as usize;
            // Five columns: the usual three, then `r` and `a`.
            let out = ace_outgoing_energy(xss, idx, 5)?;
            precompound.push(Tabulated1D::new(out.data[0].clone(), out.data[3].clone()));
            slope.push(Tabulated1D::new(out.data[0].clone(), out.data[4].clone()));
            energy_out.push(out.distribution);
        }

        Ok(KalbachMann {
            breakpoints: grid.breakpoints,
            interpolation: grid.interpolation,
            energy: grid.energy,
            energy_out,
            precompound,
            slope,
        })
    }
}

impl CorrelatedAngleEnergy {
    /// Read a correlated angle-energy distribution from an ACE table's XSS
    /// array. The arguments are as for [`KalbachMann::from_ace`].
    pub fn from_ace(xss: &[f64], idx: usize, ldis: i64) -> Result<CorrelatedAngleEnergy> {
        let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);
        let grid = ace_incident_grid(xss, idx);

        let n = grid.energy.len();
        let mut energy_out = Vec::with_capacity(n);
        let mut mu = Vec::with_capacity(n);
        for &loc in &grid.loc_dist {
            let idx = (ldis + loc - 1).max(0) as usize;
            // Four columns: the usual three, then a locator per outgoing
            // energy.
            let out = ace_outgoing_energy(xss, idx, 4)?;

            let mut mu_i = Vec::with_capacity(out.data[3].len());
            for &lc in &out.data[3] {
                let lc = lc as i64;
                // Zero, and anything negative, means isotropic.
                if lc <= 0 {
                    mu_i.push(Univariate::Uniform(Uniform::new(-1.0, 1.0)));
                    continue;
                }
                let idx = (ldis + lc.abs() - 1).max(0) as usize;
                let intt = at(idx) as i32;
                let n_cosine = at(idx + 1) as usize;
                let col = |row: usize| -> Vec<f64> {
                    (0..n_cosine)
                        .map(|k| at(idx + 2 + row * n_cosine + k))
                        .collect()
                };
                mu_i.push(Univariate::Tabular(Tabular::with_cdf(
                    col(0),
                    col(1),
                    Interpolation::from_endf_code(intt)?,
                    col(2),
                )));
            }

            energy_out.push(out.distribution);
            mu.push(mu_i);
        }

        Ok(CorrelatedAngleEnergy {
            breakpoints: grid.breakpoints,
            interpolation: grid.interpolation,
            energy: grid.energy,
            energy_out,
            mu,
        })
    }
}

impl NBodyPhaseSpace {
    /// Read an N-body phase space distribution from an ACE table.
    pub fn from_ace(table: &Table, idx: usize, q_value: f64) -> NBodyPhaseSpace {
        let at = |i: usize| table.xss.get(i).copied().unwrap_or(0.0);
        NBodyPhaseSpace {
            n_particles: at(idx) as i64,
            total_mass: at(idx + 1),
            atomic_weight_ratio: table.atomic_weight_ratio,
            q_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ace;

    /// The smallest table that reaches the law dispatch: with
    /// `LDIS + LOCC - 1 = 0`, the law sits at index 1 and its data locator at
    /// index 2.
    fn table_with_law(law: f64) -> ace::Table {
        ace::Table {
            name: "1001.00c".into(),
            atomic_weight_ratio: 1.0,
            kt: 0.0,
            pairs: Vec::new(),
            nxs: vec![0; 17],
            jxs: vec![0; 33],
            xss: vec![0.0, law, 1.0],
        }
    }

    /// Walk the DLW linked list for one reaction, as `Reaction::from_ace`
    /// does: LDLW (JXS(10)) gives the first locator, and each distribution's
    /// own first word points at the next one for the same reaction.
    fn dlw_chain(t: &ace::Table, i_reaction: usize) -> Vec<i64> {
        let (ldlw, dlw) = (t.jxs[10], t.jxs[11]);
        let mut chain = Vec::new();
        let mut lnw = t.xss[ldlw as usize + i_reaction - 1] as i64;
        while lnw > 0 {
            chain.push(lnw);
            lnw = t.xss[(dlw + lnw - 1) as usize] as i64;
        }
        chain
    }

    #[test]
    fn reads_every_distribution_li6_holds() {
        let tables = crate::testdata::ace_tables(crate::testdata::LI6_ACE);
        let t = &tables[0];

        let mut seen = 0;
        for i_reaction in 1..=t.nxs[5] as usize {
            for loc in dlw_chain(t, i_reaction) {
                let dist = AngleEnergy::from_ace(t, t.jxs[11], loc, Some(0.0))
                    .expect("every law in this table is one the reader knows");
                // Whatever the shape, it has to carry something.
                match dist {
                    AngleEnergy::Uncorrelated(u) => assert!(u.energy.is_some()),
                    AngleEnergy::KalbachMann(k) => assert!(!k.energy.is_empty()),
                    AngleEnergy::Correlated(c) => assert!(!c.energy.is_empty()),
                    AngleEnergy::NBodyPhaseSpace(n) => assert!(n.n_particles > 0),
                }
                seen += 1;
            }
        }
        assert!(seen >= t.nxs[5] as usize);
    }

    #[test]
    fn an_unknown_law_is_refused() {
        // A block whose law is 99: with LDIS + LOCC - 1 = 0, the law sits at
        // index 1 and the data locator at index 2.
        assert!(AngleEnergy::from_ace(&table_with_law(99.0), 0, 1, None).is_err());
    }

    #[test]
    fn law_5_reports_the_gap_rather_than_guessing() {
        // The Python reader dies with an AttributeError here; see issue #19.
        assert!(AngleEnergy::from_ace(&table_with_law(5.0), 0, 1, None).is_err());
    }
}
