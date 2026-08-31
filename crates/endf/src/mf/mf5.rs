//! MF=5, energy distributions of secondary particles.

use crate::ace::Table;
use crate::data::EV_PER_MEV;
use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;
use crate::univariate::{Discrete, Interpolation, Mixture, Tabular, Univariate};

/// MF=5: the energy distributions of one reaction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf5 {
    pub za: i64,
    pub awr: f64,
    pub nk: i64,
    pub subsections: Vec<Subsection>,
}

/// One partial distribution, with the fraction of emission it accounts for.
#[derive(Debug, Clone, PartialEq)]
pub struct Subsection {
    /// Which law the distribution uses.
    pub lf: i64,
    /// Applicability: the fraction of the emission this subsection describes,
    /// as a function of incident energy.
    pub p: Tabulated1D,
    pub distribution: EnergyDistribution,
}

/// An energy distribution, in whichever law the evaluation uses.
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyDistribution {
    /// LF=1: an arbitrary tabulated function of outgoing energy.
    ArbitraryTabulated {
        e_int: Tabulated2D,
        /// Incident energies.
        energy: Vec<f64>,
        /// The outgoing distribution at each incident energy.
        g: Vec<Tabulated1D>,
    },
    /// LF=5: general evaporation spectrum.
    GeneralEvaporation {
        u: f64,
        theta: Tabulated1D,
        g: Tabulated1D,
    },
    /// LF=7: simple Maxwellian fission spectrum.
    MaxwellEnergy { u: f64, theta: Tabulated1D },
    /// LF=9: evaporation spectrum.
    Evaporation { u: f64, theta: Tabulated1D },
    /// LF=11: energy-dependent Watt spectrum.
    WattEnergy {
        u: f64,
        a: Tabulated1D,
        b: Tabulated1D,
    },
    /// LF=12: Madland-Nix fission spectrum.
    MadlandNix {
        efl: f64,
        efh: f64,
        t_m: Tabulated1D,
    },
    /// ACE laws 3 and 33: inelastic scattering to a discrete level.
    ///
    /// No ENDF law corresponds; the processed file states the kinematics
    /// directly rather than tabulating the resulting spectrum.
    LevelInelastic {
        /// Laboratory threshold, `(A + 1)/A * |Q|`, in eV.
        threshold: f64,
        /// `(A/(A + 1))^2`.
        mass_ratio: f64,
    },
    /// ACE law 2: a photon of one energy.
    DiscretePhoton {
        /// 1 for a primary photon, 2 for a non-primary one.
        primary_flag: i64,
        /// The photon energy if primary, the binding energy if not, in eV.
        energy: f64,
        /// Of the nuclide that emitted it.
        atomic_weight_ratio: f64,
    },
    /// ACE law 4: an outgoing spectrum tabulated at each incident energy.
    ContinuousTabular {
        breakpoints: Vec<i32>,
        interpolation: Vec<i32>,
        /// Incident energies in eV.
        energy: Vec<f64>,
        /// The outgoing energy distribution at each of them.
        energy_out: Vec<Univariate>,
    },
}

/// Parse an MF=5 section.
pub fn parse_mf5(reader: &mut Reader) -> Result<Mf5> {
    let head = reader.head_record()?;
    let mut data = Mf5 {
        za: head.za,
        awr: head.awr,
        nk: head.n1,
        subsections: Vec::new(),
    };

    for _ in 0..data.nk.max(0) {
        // The applicability record also carries the law and its parameters.
        let applicability = reader.tab1_record()?;
        let lf = applicability.l2;
        let (c1, c2) = (applicability.c1, applicability.c2);

        let distribution = match lf {
            1 => {
                let tab2 = reader.tab2_record()?;
                let n = tab2.cont.n2.max(0);
                let mut energy = Vec::with_capacity(n as usize);
                let mut g = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let tab = reader.tab1_record()?;
                    energy.push(tab.c2);
                    g.push(tab.table);
                }
                EnergyDistribution::ArbitraryTabulated {
                    e_int: tab2.table,
                    energy,
                    g,
                }
            }
            5 => EnergyDistribution::GeneralEvaporation {
                u: c1,
                theta: reader.tab1_record()?.table,
                g: reader.tab1_record()?.table,
            },
            7 => EnergyDistribution::MaxwellEnergy {
                u: c1,
                theta: reader.tab1_record()?.table,
            },
            9 => EnergyDistribution::Evaporation {
                u: c1,
                theta: reader.tab1_record()?.table,
            },
            11 => EnergyDistribution::WattEnergy {
                u: c1,
                a: reader.tab1_record()?.table,
                b: reader.tab1_record()?.table,
            },
            12 => EnergyDistribution::MadlandNix {
                efl: c1,
                efh: c2,
                t_m: reader.tab1_record()?.table,
            },
            // The Python reader leaves `dist` unbound here and raises
            // UnboundLocalError; this reports the same failure with a usable
            // message.
            _ => {
                return Err(Error::Unsupported {
                    what: "an unrecognised MF=5 energy distribution law",
                })
            }
        };

        data.subsections.push(Subsection {
            lf,
            p: applicability.table,
            distribution,
        });
    }

    Ok(data)
}

/// A `theta`-and-`U` law as an ACE table gives it: LF=7 and LF=9 share a
/// layout, differing only in which variant they build.
///
/// Returns the tabulated temperature, already in eV, and the restriction
/// energy.
fn ace_theta_and_u(xss: &[f64], idx: usize) -> (Tabulated1D, f64) {
    let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);

    // The nuclear temperature is stored in MeV against an energy in MeV, so
    // both axes are converted.
    let mut theta = Tabulated1D::from_ace(xss, idx, true);
    for v in &mut theta.y {
        *v *= EV_PER_MEV;
    }

    let nr = at(idx) as usize;
    let ne = at(idx + 1 + 2 * nr) as usize;
    let u = at(idx + 2 + 2 * nr + 2 * ne) * EV_PER_MEV;
    (theta, u)
}

impl EnergyDistribution {
    /// LF=7, a Maxwellian fission spectrum, from an ACE table.
    pub fn maxwell_from_ace(xss: &[f64], idx: usize) -> EnergyDistribution {
        let (theta, u) = ace_theta_and_u(xss, idx);
        EnergyDistribution::MaxwellEnergy { u, theta }
    }

    /// LF=9, an evaporation spectrum, from an ACE table.
    pub fn evaporation_from_ace(xss: &[f64], idx: usize) -> EnergyDistribution {
        let (theta, u) = ace_theta_and_u(xss, idx);
        EnergyDistribution::Evaporation { u, theta }
    }

    /// LF=11, an energy-dependent Watt spectrum, from an ACE table.
    pub fn watt_from_ace(xss: &[f64], idx: usize) -> EnergyDistribution {
        let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);

        // `a` is an energy, stored in MeV.
        let mut a = Tabulated1D::from_ace(xss, idx, true);
        for v in &mut a.y {
            *v *= EV_PER_MEV;
        }
        let nr = at(idx) as usize;
        let ne = at(idx + 1 + 2 * nr) as usize;
        let idx = idx + 2 + 2 * nr + 2 * ne;

        // `b` is an inverse energy, so it converts the other way.
        let mut b = Tabulated1D::from_ace(xss, idx, true);
        for v in &mut b.y {
            *v /= EV_PER_MEV;
        }
        let nr = at(idx) as usize;
        let ne = at(idx + 1 + 2 * nr) as usize;
        let idx = idx + 2 + 2 * nr + 2 * ne;

        EnergyDistribution::WattEnergy {
            u: at(idx) * EV_PER_MEV,
            a,
            b,
        }
    }

    /// ACE laws 3 and 33, inelastic scattering to a discrete level.
    pub fn level_inelastic_from_ace(xss: &[f64], idx: usize) -> EnergyDistribution {
        let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);
        EnergyDistribution::LevelInelastic {
            threshold: at(idx) * EV_PER_MEV,
            mass_ratio: at(idx + 1),
        }
    }

    /// ACE law 2, a photon of one energy.
    ///
    /// Takes the whole table rather than the XSS array: the atomic weight
    /// ratio it records comes from the header.
    pub fn discrete_photon_from_ace(table: &Table, idx: usize) -> EnergyDistribution {
        let at = |i: usize| table.xss.get(i).copied().unwrap_or(0.0);
        EnergyDistribution::DiscretePhoton {
            primary_flag: at(idx) as i64,
            energy: at(idx + 1) * EV_PER_MEV,
            atomic_weight_ratio: table.atomic_weight_ratio,
        }
    }

    /// ACE law 4, an outgoing spectrum tabulated at each incident energy.
    ///
    /// `idx` is where this law's data begins (`LDIS + LOCC - 1`) and `ldis`
    /// the start of the energy distribution block, e.g. JXS(11); the locators
    /// inside are relative to the latter.
    pub fn continuous_tabular_from_ace(
        xss: &[f64],
        idx: usize,
        ldis: i64,
    ) -> Result<EnergyDistribution> {
        let grid = ace_incident_grid(xss, idx);

        let mut energy_out = Vec::with_capacity(grid.energy.len());
        for &loc in &grid.loc_dist {
            let idx = (ldis + loc - 1).max(0) as usize;
            energy_out.push(ace_outgoing_energy(xss, idx, 3)?.distribution);
        }

        Ok(EnergyDistribution::ContinuousTabular {
            breakpoints: grid.breakpoints,
            interpolation: grid.interpolation,
            energy: grid.energy,
            energy_out,
        })
    }
}

/// The incident energy grid shared by ACE laws 4, 44 and 61.
#[derive(Debug, Clone, PartialEq)]
pub struct AceIncidentGrid {
    pub breakpoints: Vec<i32>,
    pub interpolation: Vec<i32>,
    /// Incident energies in eV.
    pub energy: Vec<f64>,
    /// Where each incident energy's outgoing distribution begins, relative to
    /// the start of the energy distribution block.
    pub loc_dist: Vec<i64>,
}

/// Read the incident energy grid shared by ACE laws 4, 44 and 61.
pub fn ace_incident_grid(xss: &[f64], idx: usize) -> AceIncidentGrid {
    let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);
    let slice = |i: usize, n: usize| -> Vec<f64> { (0..n).map(|k| at(i + k)).collect() };

    let n_regions = at(idx) as usize;
    let n_energy_in = at(idx + 1 + 2 * n_regions) as usize;

    let mut idx = idx + 1;
    let (breakpoints, interpolation) = if n_regions > 0 {
        (
            (0..n_regions).map(|i| at(idx + i) as i32).collect(),
            (0..n_regions)
                .map(|i| at(idx + n_regions + i) as i32)
                .collect(),
        )
    } else {
        // Zero regions means one linear-linear region over the whole grid.
        (vec![n_energy_in as i32], vec![2])
    };

    idx += 2 * n_regions + 1;
    let energy: Vec<f64> = slice(idx, n_energy_in)
        .into_iter()
        .map(|e| e * EV_PER_MEV)
        .collect();

    idx += n_energy_in;
    let loc_dist = slice(idx, n_energy_in)
        .into_iter()
        .map(|v| v as i64)
        .collect();

    AceIncidentGrid {
        breakpoints,
        interpolation,
        energy,
        loc_dist,
    }
}

/// One tabulated outgoing energy distribution, and the columns it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct AceOutgoingEnergy {
    /// The outgoing energy distribution.
    pub distribution: Univariate,
    /// The stored columns, `n_cols` rows of `n_energy_out`. Row 0 is the
    /// outgoing energy, already in eV; row 1 the density, still per MeV; row 2
    /// the CDF. What follows depends on the law.
    pub data: Vec<Vec<f64>>,
    /// How many of the points are discrete lines rather than a continuum.
    pub n_discrete_lines: usize,
}

/// Read one tabulated outgoing energy distribution from an ACE table.
///
/// Laws 4, 44 and 61 store the outgoing energy the same way and differ only in
/// how many columns follow: three (energy, density, CDF) for law 4, five for
/// Kalbach-Mann, four for correlated angle-energy.
pub fn ace_outgoing_energy(xss: &[f64], idx: usize, n_cols: usize) -> Result<AceOutgoingEnergy> {
    let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);

    // INTT is the interpolation scheme, 1 histogram or 2 linear-linear. When
    // discrete lines are present the stored value is 10*n_lines + INTT.
    let packed = at(idx) as i64;
    let (n_discrete_lines, intt) = (packed.div_euclid(10), packed.rem_euclid(10));
    // Anything else is read as linear-linear, which is what the Python reader
    // falls back to after warning.
    let interpolation = match intt {
        1 => Interpolation::Histogram,
        _ => Interpolation::LinearLinear,
    };
    let n_discrete_lines = n_discrete_lines.max(0) as usize;

    let n_energy_out = at(idx + 1) as usize;
    let mut data = Vec::with_capacity(n_cols);
    for row in 0..n_cols {
        data.push(
            (0..n_energy_out)
                .map(|k| at(idx + 2 + row * n_energy_out + k))
                .collect::<Vec<f64>>(),
        );
    }
    if data.len() < 3 {
        return Err(Error::BadAceTable {
            what: "an outgoing energy distribution needs at least three columns".into(),
        });
    }
    for v in &mut data[0] {
        *v *= EV_PER_MEV;
    }

    let split = n_discrete_lines.min(n_energy_out);
    let continuous = Tabular::with_cdf(
        data[0][split..].to_vec(),
        // The density is per MeV where the energy is in eV.
        data[1][split..].iter().map(|&v| v / EV_PER_MEV).collect(),
        interpolation,
        data[2][split..].to_vec(),
    );

    let distribution = if split > 0 {
        let mut discrete = Discrete::new(data[0][..split].to_vec(), data[1][..split].to_vec());
        discrete.c = Some(data[2][..split].to_vec());
        if split == n_energy_out {
            Univariate::Discrete(discrete)
        } else {
            let p_discrete = discrete.p.iter().sum::<f64>().min(1.0);
            Univariate::Mixture(Mixture::new(
                vec![p_discrete, 1.0 - p_discrete],
                vec![
                    Univariate::Discrete(discrete),
                    Univariate::Tabular(continuous),
                ],
            ))
        }
    } else {
        Univariate::Tabular(continuous)
    };

    Ok(AceOutgoingEnergy {
        distribution,
        data,
        n_discrete_lines: split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_the_fission_neutron_spectrum() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let d = m.mf5(18).expect("MF=5 MT=18 is present");
        assert_eq!(d.za, 95244);
        assert_eq!(d.subsections.len(), d.nk as usize);
        assert!(!d.subsections.is_empty());

        // The applicability of a single subsection is unity throughout.
        let sub = &d.subsections[0];
        assert!(sub.p.y.iter().all(|&v| v == 1.0));

        // Am244 gives its fission spectrum as a simple Maxwellian.
        assert_eq!(sub.lf, 7);
        match &sub.distribution {
            EnergyDistribution::MaxwellEnergy { u, theta } => {
                // U bounds the outgoing energy by 0 <= E' <= E - U, so a
                // negative U widens that range rather than narrowing it. This
                // evaluation uses -20 MeV.
                assert_eq!(*u, -2.0e7);
                // Theta is the Maxwellian temperature in eV, constant here.
                assert!(!theta.x.is_empty());
                assert!(theta.y.iter().all(|&v| v > 0.0));
            }
            other => panic!("unexpected distribution {other:?}"),
        }
    }
}
