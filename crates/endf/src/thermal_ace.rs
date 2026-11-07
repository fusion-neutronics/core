//! A processed thermal scattering table (ACE, `iopt=2`), one temperature.
//!
//! The layout is the thermal ACE format NJOY's ACER writes (NJOY2016 manual,
//! LA-UR-17-20093, ACER section): ITIE holds the incoherent inelastic cross
//! section, ITXE its outgoing energies and equiprobable cosines, ITCE the
//! elastic cross section, ITCA the incoherent elastic cosines, and for mixed
//! elastic ITCEI and ITCAI the incoherent part after the coherent one.
//!
//! Two things are done to what ACER wrote, both lossless: the equiprobable
//! cosines of each outgoing energy are sorted (ACER does not always write them
//! in order, and being equiprobable their order carries no information), and
//! where an outgoing-energy cdf does not start at zero a point at E' = 0 with
//! pdf 0 and cdf 0, and isotropic cosines, is put in front (the scattering law
//! goes to zero at E' = 0, ENDF-6 manual eq. 7.6). The pdf and cdf are kept
//! exactly as written: they are not normalised and are not renormalised here.

use crate::ace::{Table, TableType};
use crate::data::EV_PER_MEV;
use crate::error::{Error, Result};

/// The outgoing energies and cosines for one incident energy, in the
/// continuous form ACER writes with `iwt=2`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingDistribution {
    /// Outgoing energies, in eV.
    pub energy_out: Vec<f64>,
    /// pdf at each outgoing energy, in 1/eV.
    pub pdf: Vec<f64>,
    pub cdf: Vec<f64>,
    /// The equiprobable cosines, `mu[j * n_mu + k]` for outgoing energy `j`,
    /// sorted within each `j`.
    pub mu: Vec<f64>,
    pub n_mu: usize,
}

/// The incoherent inelastic angle-energy distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum InelasticDistribution {
    /// Continuous outgoing energies (NXS(7) = 2), one per incident energy.
    Continuous(Vec<OutgoingDistribution>),
    /// Equally probable outgoing energies (NXS(7) = 0, or 1 when skewed),
    /// each with `n_mu` cosines.
    Discrete {
        /// `energy_out[i][j]`, in eV.
        energy_out: Vec<Vec<f64>>,
        /// `mu[i][j * n_mu + k]`.
        mu: Vec<Vec<f64>>,
        n_mu: usize,
        skewed: bool,
    },
}

/// The elastic part of a thermal ACE table.
#[derive(Debug, Clone, PartialEq)]
pub enum AceElastic {
    /// Bragg edges in eV and the cumulative factor at each, in eV-barns.
    Coherent {
        bragg_edges: Vec<f64>,
        factors: Vec<f64>,
    },
    /// The cross section on its own grid, and `n_mu` equiprobable cosines per
    /// energy.
    Incoherent {
        energy: Vec<f64>,
        xs: Vec<f64>,
        mu: Vec<f64>,
        n_mu: usize,
    },
    Mixed {
        bragg_edges: Vec<f64>,
        factors: Vec<f64>,
        energy: Vec<f64>,
        xs: Vec<f64>,
        mu: Vec<f64>,
        n_mu: usize,
    },
}

/// One thermal scattering ACE table.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalAce {
    /// The table's full name, such as `"lwtr.10t"`.
    pub name: String,
    pub atomic_weight_ratio: f64,
    /// Temperature in Kelvin.
    pub temperature: f64,
    /// kT in eV.
    pub kt: f64,
    /// The ZA values the table applies to.
    pub zaids: Vec<i64>,
    /// Incident energies of the inelastic cross section, in eV.
    pub inelastic_energy: Vec<f64>,
    /// The inelastic cross section in barns.
    pub inelastic_xs: Vec<f64>,
    pub inelastic: InelasticDistribution,
    pub elastic: Option<AceElastic>,
}

/// An incoherent elastic block as read: energies, cross sections, cosines
/// and the cosine count.
type IncoherentBlocks = (Vec<f64>, Vec<f64>, Vec<f64>, usize);

fn bad(what: String) -> Error {
    Error::BadAceTable { what }
}

/// A slice of XSS, one-based like the format, or an error naming the block.
fn span<'a>(xss: &'a [f64], start: i64, len: usize, block: &str) -> Result<&'a [f64]> {
    let s = usize::try_from(start).map_err(|_| bad(format!("{block} starts at {start}")))?;
    xss.get(s..s + len)
        .ok_or_else(|| bad(format!("{block} runs past the end of XSS ({s} + {len})")))
}

fn count(value: f64, block: &str) -> Result<usize> {
    if value >= 0.0 && value.fract() == 0.0 {
        Ok(value as usize)
    } else {
        Err(bad(format!("{block} has a count of {value}")))
    }
}

impl ThermalAce {
    /// Read a thermal table. Refuses any other kind of ACE table.
    pub fn from_ace(table: &Table) -> Result<Self> {
        if table.data_type()? != TableType::ThermalScattering {
            return Err(bad(format!(
                "{:?} is not a thermal scattering table",
                table.name
            )));
        }
        let (nxs, jxs, xss) = (&table.nxs, &table.jxs, &table.xss);
        let nx = |i: usize| nxs.get(i).copied().unwrap_or(0);
        let jx = |i: usize| jxs.get(i).copied().unwrap_or(0);

        // ITIE: the inelastic cross section.
        let n_energy = count(*span(xss, jx(1), 1, "ITIE")?.first().unwrap(), "ITIE")?;
        let inelastic_energy: Vec<f64> = span(xss, jx(1) + 1, n_energy, "ITIE")?
            .iter()
            .map(|e| e * EV_PER_MEV)
            .collect();
        let inelastic_xs = span(xss, jx(1) + 1 + n_energy as i64, n_energy, "ITIE")?.to_vec();

        // ITXE: the inelastic distribution.
        let inelastic = if nx(7) == 2 {
            let n_mu =
                usize::try_from(nx(3) - 1).map_err(|_| bad(format!("NXS(3) = {}", nx(3))))?;
            if n_mu == 0 {
                return Err(bad("a continuous table with no cosines".into()));
            }
            let locc = span(xss, jx(3), n_energy, "ITXE locators")?.to_vec();
            let n_out = span(xss, jx(3) + n_energy as i64, n_energy, "ITXE counts")?.to_vec();
            let stride = n_mu + 3;
            let mut out = Vec::with_capacity(n_energy);
            for i in 0..n_energy {
                let loc = locc[i] as i64;
                let n = count(n_out[i], "ITXE")?;
                let block = span(xss, loc + 1, n * stride, "ITXE")?;
                let mut energy_out = Vec::with_capacity(n + 1);
                let mut pdf = Vec::with_capacity(n + 1);
                let mut cdf = Vec::with_capacity(n + 1);
                let mut mu = Vec::with_capacity((n + 1) * n_mu);
                for j in 0..n {
                    let row = &block[j * stride..(j + 1) * stride];
                    energy_out.push(row[0] * EV_PER_MEV);
                    pdf.push(row[1] / EV_PER_MEV);
                    cdf.push(row[2]);
                    let mut cosines = row[3..].to_vec();
                    cosines.sort_by(f64::total_cmp);
                    if cosines[0] < -1.0 || cosines[n_mu - 1] > 1.0 {
                        return Err(bad(format!(
                            "{}: a cosine outside [-1, 1] at incident energy {i}, outgoing energy {j}",
                            table.name
                        )));
                    }
                    mu.extend(cosines);
                }
                if cdf.first().is_some_and(|&c| c > 0.0) {
                    energy_out.insert(0, 0.0);
                    pdf.insert(0, 0.0);
                    cdf.insert(0, 0.0);
                    let d = 2.0 / n_mu as f64;
                    let isotropic: Vec<f64> =
                        (0..n_mu).map(|k| -1.0 + d * (k as f64 + 0.5)).collect();
                    mu.splice(0..0, isotropic);
                }
                out.push(OutgoingDistribution {
                    energy_out,
                    pdf,
                    cdf,
                    mu,
                    n_mu,
                });
            }
            InelasticDistribution::Continuous(out)
        } else {
            let n_mu = usize::try_from(nx(3)).map_err(|_| bad(format!("NXS(3) = {}", nx(3))))?;
            let n_out = usize::try_from(nx(4)).map_err(|_| bad(format!("NXS(4) = {}", nx(4))))?;
            let stride = n_mu + 2;
            let block = span(xss, jx(3), n_energy * n_out * stride, "ITXE")?;
            let mut energy_out = Vec::with_capacity(n_energy);
            let mut mu = Vec::with_capacity(n_energy);
            for i in 0..n_energy {
                let mut e = Vec::with_capacity(n_out);
                let mut m = Vec::with_capacity(n_out * (n_mu + 1));
                for j in 0..n_out {
                    let row = &block[(i * n_out + j) * stride..(i * n_out + j + 1) * stride];
                    e.push(row[0] * EV_PER_MEV);
                    m.extend_from_slice(&row[1..]);
                }
                energy_out.push(e);
                mu.push(m);
            }
            InelasticDistribution::Discrete {
                energy_out,
                mu,
                n_mu: n_mu + 1,
                skewed: nx(7) == 1,
            }
        };

        // ITCE: the elastic part.
        let elastic = if jx(4) == 0 {
            None
        } else {
            let read_xs = |at: i64, block: &str| -> Result<(Vec<f64>, Vec<f64>)> {
                let n = count(*span(xss, at, 1, block)?.first().unwrap(), block)?;
                let e = span(xss, at + 1, n, block)?
                    .iter()
                    .map(|e| e * EV_PER_MEV)
                    .collect();
                let v = span(xss, at + 1 + n as i64, n, block)?.to_vec();
                Ok((e, v))
            };
            let incoherent = |xs_at: i64, mu_at: i64, nmu_field: i64| -> Result<IncoherentBlocks> {
                let (energy, xs) = read_xs(xs_at, "ITCE (incoherent)")?;
                let n_mu = usize::try_from(nmu_field + 1)
                    .map_err(|_| bad(format!("incoherent elastic NXS = {nmu_field}")))?;
                if n_mu == 0 {
                    return Err(bad("incoherent elastic with no cosines".into()));
                }
                let mu = span(xss, mu_at, energy.len() * n_mu, "ITCA")?.to_vec();
                Ok((energy, xs, mu, n_mu))
            };
            match nx(5) {
                4 => {
                    let (bragg_edges, p) = read_xs(jx(4), "ITCE (coherent)")?;
                    Some(AceElastic::Coherent {
                        bragg_edges,
                        factors: p.iter().map(|v| v * EV_PER_MEV).collect(),
                    })
                }
                3 => {
                    let (energy, xs, mu, n_mu) = incoherent(jx(4), jx(6), nx(6))?;
                    Some(AceElastic::Incoherent {
                        energy,
                        xs,
                        mu,
                        n_mu,
                    })
                }
                5 => {
                    let (bragg_edges, p) = read_xs(jx(4), "ITCE (coherent)")?;
                    let (energy, xs, mu, n_mu) = incoherent(jx(7), jx(9), nx(8))?;
                    Some(AceElastic::Mixed {
                        bragg_edges,
                        factors: p.iter().map(|v| v * EV_PER_MEV).collect(),
                        energy,
                        xs,
                        mu,
                        n_mu,
                    })
                }
                other => {
                    return Err(bad(format!(
                        "{}: elastic mode NXS(5) = {other}, not 3, 4 or 5",
                        table.name
                    )))
                }
            }
        };

        Ok(ThermalAce {
            name: table.name.clone(),
            atomic_weight_ratio: table.atomic_weight_ratio,
            temperature: table.temperature(),
            kt: table.kt * EV_PER_MEV,
            zaids: table
                .pairs
                .iter()
                .filter(|(z, _)| *z > 0)
                .map(|(z, _)| *z)
                .collect(),
            inelastic_energy,
            inelastic_xs,
            inelastic,
            elastic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC: &[u8] = include_bytes!("../fixtures/synthetic-thermal.ace.xz");

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-12 * y.abs().max(1e-300))
    }

    fn tables() -> Vec<ThermalAce> {
        crate::testdata::ace_tables(SYNTHETIC)
            .iter()
            .map(|t| ThermalAce::from_ace(t).unwrap())
            .collect()
    }

    #[test]
    fn continuous_inelastic_and_incoherent_elastic() {
        let t = &tables()[0];
        assert_eq!(t.zaids, vec![1001, 8016]);
        assert!(close(&t.inelastic_energy, &[1.0e-5, 1.0e-3, 4.0]));
        assert_eq!(t.inelastic_xs, vec![2.0, 3.0, 4.0]);
        let InelasticDistribution::Continuous(d) = &t.inelastic else {
            panic!("continuous")
        };
        // The first incident energy's cdf started above zero: E' = 0 with
        // pdf 0, cdf 0 and isotropic cosines is put in front.
        assert_eq!(d[0].energy_out.len(), 3);
        assert_eq!(
            (d[0].energy_out[0], d[0].pdf[0], d[0].cdf[0]),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(&d[0].mu[..4], &[-0.75, -0.25, 0.25, 0.75]);
        // pdf in 1/eV, energies in eV, and the cdf as written.
        assert!(close(
            &[d[0].energy_out[1], d[0].pdf[1], d[0].cdf[1]],
            &[1.0e-3, 0.1, 0.5]
        ));
        // The others start at zero and are left alone.
        assert_eq!(d[1].energy_out.len(), 3);
        // Cosines written out of order come back sorted.
        let row = &d[1].mu[4..8];
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        let Some(AceElastic::Incoherent {
            energy,
            xs,
            mu,
            n_mu,
        }) = &t.elastic
        else {
            panic!("incoherent")
        };
        assert_eq!((energy.len(), xs.len(), *n_mu, mu.len()), (3, 3, 4, 12));
    }

    #[test]
    fn discrete_skewed_inelastic_and_coherent_elastic() {
        let t = &tables()[1];
        let InelasticDistribution::Discrete {
            energy_out,
            mu,
            n_mu,
            skewed,
        } = &t.inelastic
        else {
            panic!("discrete")
        };
        assert!(*skewed);
        assert_eq!(*n_mu, 3);
        assert!(close(&energy_out[0], &[2.0e-3, 4.0e-3, 6.0e-3]));
        assert_eq!(mu[0].len(), 9);
        let Some(AceElastic::Coherent {
            bragg_edges,
            factors,
        }) = &t.elastic
        else {
            panic!("coherent")
        };
        assert!(close(bragg_edges, &[1.0e-3, 5.0e-3, 2.0e-2]));
        // P is in MeV-barns on the tape and eV-barns here.
        assert!(close(factors, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn mixed_elastic_reads_its_incoherent_part_after_the_coherent() {
        let t = &tables()[2];
        let Some(AceElastic::Mixed {
            bragg_edges,
            xs,
            n_mu,
            ..
        }) = &t.elastic
        else {
            panic!("mixed")
        };
        assert_eq!(bragg_edges.len(), 3);
        assert_eq!(xs, &vec![4.0, 3.0, 2.0]);
        assert_eq!(*n_mu, 4);
    }

    #[test]
    fn refuses_a_table_that_is_not_thermal() {
        const LI6: &[u8] = include_bytes!("../fixtures/Li6.ace.xz");
        let li6 = &crate::testdata::ace_tables(LI6)[0];
        assert!(ThermalAce::from_ace(li6).is_err());
    }
}
