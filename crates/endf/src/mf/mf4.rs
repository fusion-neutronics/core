//! MF=4, angular distributions of secondary particles.

use crate::ace::Table;
use crate::data::EV_PER_MEV;
use crate::error::Result;
use crate::function::{Legendre, Tabulated1D, Tabulated2D};
use crate::records::Reader;
use crate::univariate::{Interpolation, Tabular, Uniform};

/// Angular distributions given as Legendre coefficients (LTT=1).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LegendreAngles {
    /// Interpolation across incident energy.
    pub e_int: Tabulated2D,
    /// Temperature. The format gives one per incident energy; like the Python
    /// reader, only the last is kept.
    pub t: f64,
    /// Test flag, likewise the last one read.
    pub lt: i64,
    /// Incident energies.
    pub energy: Vec<f64>,
    /// Legendre coefficients at each incident energy, `a_1` upward, the
    /// `a_0 = 1` term the format omits is not inserted here.
    pub a_l: Vec<Vec<f64>>,
}

/// Angular distributions given as tabulated probabilities (LTT=2).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TabulatedAngles {
    pub e_int: Tabulated2D,
    pub t: f64,
    pub lt: i64,
    pub energy: Vec<f64>,
    /// The distribution of the scattering cosine at each incident energy.
    pub mu: Vec<Tabulated1D>,
}

/// MF=4: the angular distribution of one reaction's secondary particle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf4 {
    pub za: i64,
    pub awr: f64,
    /// 0 isotropic, 1 Legendre, 2 tabulated, 3 Legendre then tabulated.
    pub ltt: i64,
    /// 1 when the distribution is isotropic at every energy.
    pub li: i64,
    /// 1 laboratory frame, 2 centre-of-mass.
    pub lct: i64,
    /// Present for LTT=1 and LTT=3.
    pub legendre: Option<LegendreAngles>,
    /// Present for LTT=2 and LTT=3.
    pub tabulated: Option<TabulatedAngles>,
    /// True when the obsolete energy transformation matrix was present and
    /// skipped. The Python reader warns here.
    pub had_transformation_matrix: bool,
}

fn parse_legendre(reader: &mut Reader) -> Result<LegendreAngles> {
    let tab2 = reader.tab2_record()?;
    let n_energy = tab2.cont.n2.max(0);
    let mut data = LegendreAngles {
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..n_energy {
        let list = reader.list_record()?;
        // T and LT are per incident energy in the format but only one slot is
        // kept, so the last wins. Mirrors the Python reader.
        data.t = list.cont.c1;
        data.lt = list.cont.l1;
        data.energy.push(list.cont.c2);
        data.a_l.push(list.values);
    }
    Ok(data)
}

fn parse_tabulated(reader: &mut Reader) -> Result<TabulatedAngles> {
    let tab2 = reader.tab2_record()?;
    let n_energy = tab2.cont.n2.max(0);
    let mut data = TabulatedAngles {
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..n_energy {
        let tab = reader.tab1_record()?;
        data.t = tab.c1;
        data.lt = tab.l1;
        data.energy.push(tab.c2);
        data.mu.push(tab.table);
    }
    Ok(data)
}

/// Parse an MF=4 section.
pub fn parse_mf4(reader: &mut Reader) -> Result<Mf4> {
    let head = reader.head_record()?;
    let lvt = head.l1;
    let ltt = head.l2;
    let c = reader.cont_record()?;
    let (li, lct, nk) = (c.l1, c.l2, c.n1);

    let mut data = Mf4 {
        za: head.za,
        awr: head.awr,
        ltt,
        li,
        lct,
        ..Default::default()
    };

    // The energy transformation matrix was removed from the format long ago.
    // Evaluations that still carry one are read past.
    if lvt > 0 {
        data.had_transformation_matrix = true;
        reader.skip_lines((nk.max(0) as usize).div_ceil(6))?;
    }

    match (ltt, li) {
        // Purely isotropic: nothing follows.
        (0, 1) => {}
        (1, 0) => data.legendre = Some(parse_legendre(reader)?),
        (2, 0) => data.tabulated = Some(parse_tabulated(reader)?),
        (3, 0) => {
            data.legendre = Some(parse_legendre(reader)?);
            data.tabulated = Some(parse_tabulated(reader)?);
        }
        _ => {}
    }

    Ok(data)
}

/// The distribution of the scattering cosine at one incident energy.
///
/// Which shape it takes depends on where the data came from: ENDF gives
/// Legendre coefficients or a tabulated density, ACE gives a tabulated density
/// or a bare "isotropic" flag.
#[derive(Debug, Clone, PartialEq)]
pub enum AngleAtEnergy {
    /// Legendre coefficients including the `a_0 = 1` term the format omits.
    Legendre(Legendre),
    /// A density tabulated against the cosine, as ENDF's MF=4 LTT=2 gives it.
    Tabulated(Tabulated1D),
    /// A density tabulated against the cosine, as an ACE table gives it:
    /// carrying the CDF the file supplied.
    Tabular(Tabular),
    /// Isotropic scattering.
    Isotropic(Uniform),
}

/// The angular distribution of a reaction's secondary particle, as a function
/// of incident energy.
///
/// This is the interpreted form of [`Mf4`]: the parsed section says what the
/// file holds, and this says what it means. Build it with
/// [`AngleDistribution::from_mf4`] or [`AngleDistribution::from_ace`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AngleDistribution {
    /// Incident energies in eV at which a distribution is given.
    pub energy: Vec<f64>,
    /// The distribution at each of those energies.
    pub mu: Vec<AngleAtEnergy>,
}

impl AngleDistribution {
    /// Interpret a parsed MF=4 section.
    ///
    /// A purely isotropic section (LTT=0, LI=1) yields empty vectors rather
    /// than a uniform distribution at every energy, matching the Python
    /// reader, the incident energy grid is not given in that case, so there
    /// is nothing to hang a distribution on.
    pub fn from_mf4(data: &Mf4) -> AngleDistribution {
        // The format writes the coefficients from `a_1`; `a_0` is 1 by the
        // normalisation and is restored here.
        let with_a0 = |a_l: &Vec<f64>| {
            let mut coef = Vec::with_capacity(a_l.len() + 1);
            coef.push(1.0);
            coef.extend_from_slice(a_l);
            AngleAtEnergy::Legendre(Legendre::new(coef))
        };

        let mut out = AngleDistribution::default();
        match (data.ltt, data.li) {
            (1, 0) | (3, 0) => {
                if let Some(leg) = &data.legendre {
                    out.energy.extend_from_slice(&leg.energy);
                    out.mu.extend(leg.a_l.iter().map(with_a0));
                }
            }
            _ => {}
        }
        if matches!((data.ltt, data.li), (2, 0) | (3, 0)) {
            if let Some(tab) = &data.tabulated {
                out.energy.extend_from_slice(&tab.energy);
                out.mu
                    .extend(tab.mu.iter().cloned().map(AngleAtEnergy::Tabulated));
            }
        }
        out
    }

    /// Read an angular distribution from an ACE table.
    ///
    /// `location_dist` is the start of the angular distribution block, e.g.
    /// JXS(9); `location_start` is the one-based offset within it of this
    /// reaction's array, as the LOCB list gives it.
    pub fn from_ace(
        table: &Table,
        location_dist: i64,
        location_start: i64,
    ) -> Result<AngleDistribution> {
        let xss = &table.xss;
        let at = |i: i64| -> f64 {
            usize::try_from(i)
                .ok()
                .and_then(|i| xss.get(i).copied())
                .unwrap_or(0.0)
        };
        let slice = |i: i64, n: usize| -> Vec<f64> { (0..n as i64).map(|k| at(i + k)).collect() };

        let mut idx = location_dist + location_start - 1;

        let n_energies = at(idx) as usize;
        idx += 1;

        let energy: Vec<f64> = slice(idx, n_energies)
            .into_iter()
            .map(|e| e * EV_PER_MEV)
            .collect();
        idx += n_energies as i64;

        // Where each energy's distribution sits, and in which of the two
        // encodings: positive is the 32 equiprobable bins of the older format,
        // negative is a tabulated density, zero means isotropic.
        let lc: Vec<i64> = slice(idx, n_energies)
            .into_iter()
            .map(|v| v as i64)
            .collect();

        let mut mu = Vec::with_capacity(n_energies);
        for &loc in &lc {
            mu.push(match loc {
                0 => AngleAtEnergy::Isotropic(Uniform::new(-1.0, 1.0)),
                _ => {
                    let idx = location_dist + loc.abs() - 1;
                    if loc > 0 {
                        // 32 equiprobable bins, given by their 33 boundaries.
                        const N_BINS: usize = 32;
                        let cos = slice(idx, N_BINS + 1);
                        let mut pdf = vec![0.0; N_BINS + 1];
                        for i in 0..N_BINS {
                            pdf[i] = 1.0 / (N_BINS as f64 * (cos[i + 1] - cos[i]));
                        }
                        let cdf: Vec<f64> =
                            (0..=N_BINS).map(|i| i as f64 / N_BINS as f64).collect();
                        AngleAtEnergy::Tabular(Tabular::with_cdf(
                            cos,
                            pdf,
                            Interpolation::Histogram,
                            cdf,
                        ))
                    } else {
                        let intt = at(idx) as i32;
                        let n_points = at(idx + 1) as usize;
                        // Stored as three consecutive rows: values, PDF, CDF.
                        let values = slice(idx + 2, n_points);
                        let pdf = slice(idx + 2 + n_points as i64, n_points);
                        let cdf = slice(idx + 2 + 2 * n_points as i64, n_points);
                        let interp = Interpolation::from_endf_code(intt)?;
                        AngleAtEnergy::Tabular(Tabular::with_cdf(values, pdf, interp, cdf))
                    }
                }
            });
        }

        Ok(AngleDistribution { energy, mu })
    }

    /// Fraction of scattering into the cone `mu >= mu_cutoff`, at each energy.
    ///
    /// Used for the removal cross sections of point-kernel shielding, where a
    /// forward-scattered neutron counts as still in the uncollided beam.
    ///
    /// Every shape is handled, including the two an ACE table produces. The
    /// Python reader fills only the Legendre and tabulated entries and returns
    /// whatever `np.empty` gave it for the rest, so its answer for ACE data is
    /// not merely different but changes between calls; see issue #21. There is
    /// no behaviour to match, so this computes them.
    pub fn forward_fraction(&self, mu_cutoff: f64) -> Vec<f64> {
        let mut fractions = vec![0.0; self.energy.len()];
        for (i, mu_i) in self.mu.iter().enumerate() {
            if i >= fractions.len() {
                break;
            }
            fractions[i] = match mu_i {
                AngleAtEnergy::Legendre(leg) => {
                    // The stored coefficients are the ENDF `a_l`; the density
                    // is p(mu) = sum_l (2l+1)/2 a_l P_l(mu).
                    let pdf_coeffs: Vec<f64> = leg
                        .coefficients
                        .iter()
                        .enumerate()
                        .map(|(l, &a)| (2.0 * l as f64 + 1.0) / 2.0 * a)
                        .collect();
                    let antideriv = Legendre::new(pdf_coeffs).integ();
                    antideriv.eval(1.0) - antideriv.eval(mu_cutoff)
                }
                AngleAtEnergy::Tabulated(f) => {
                    let cdf = f.integral();
                    let cdf_func = Tabulated1D::new(f.x.clone(), cdf.clone());
                    let total = *cdf.last().unwrap_or(&0.0);
                    (total - cdf_func.eval(mu_cutoff)) / total
                }
                AngleAtEnergy::Tabular(t) => {
                    // The same, through the density's own cumulative
                    // distribution rather than through an integral of a TAB1.
                    let cdf = t.cdf();
                    let cdf_func = Tabulated1D::new(t.x.clone(), cdf.clone());
                    let total = *cdf.last().unwrap_or(&0.0);
                    if total > 0.0 {
                        (total - cdf_func.eval(mu_cutoff)) / total
                    } else {
                        0.0
                    }
                }
                AngleAtEnergy::Isotropic(u) => {
                    // The part of [a, b] that lies at or above the cutoff.
                    let width = u.b - u.a;
                    if width > 0.0 {
                        ((u.b - mu_cutoff.max(u.a)) / width).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                }
            };
        }
        fractions
    }
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_legendre_angular_distributions() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let d = m.mf4(2).expect("MF=4 MT=2 is present");
        assert_eq!(d.za, 95244);
        // Elastic scattering is given in the centre-of-mass frame.
        assert_eq!(d.lct, 2);

        let leg = d
            .legendre
            .as_ref()
            .expect("LTT=1 gives Legendre coefficients");
        assert_eq!(leg.energy.len(), leg.a_l.len());
        assert!(!leg.energy.is_empty());
        assert!(leg.energy.windows(2).all(|w| w[1] >= w[0]));
    }
}
