//! Bound thermal scattering: the outgoing energy and lab cosine of a neutron
//! scattered by a nuclide bound in a material, from a thermal scattering
//! table (ENDF-6 manual, BNL-203218-2018, section 7; the inelastic tables as
//! NJOY's THERMR and ACER write them).
//!
//! Every sampler works on flat slices and draws through `yamc_rng::next_xi`
//! on a threaded PCG state, so the GPU kernel can call the same functions.
//! Each returns `(E', mu)` in the lab frame: a thermal law is written for the
//! bound system, so there is no target velocity and no centre-of-mass step.
//!
//! # Draw schedule
//!
//! Per sampler, in order:
//! - [`coherent_elastic`]: one draw (the Bragg edge).
//! - [`incoherent_elastic`]: one draw (the cosine).
//! - [`incoherent_inelastic`]: four draws: the incident grid point, the
//!   outgoing energy, the cosine bin, and the smearing within the bin.
//!
//! The choices above them (bound or free, which table temperature, elastic or
//! inelastic, coherent or incoherent) are drawn by the caller and documented
//! where they are made.

use yamc_rng::next_xi;

/// Coherent elastic scattering off the Bragg edges below `e_in`: an edge is
/// chosen with probability proportional to its own share of the cumulative
/// factor, and the cosine is that edge's, `mu = 1 - 2 E_k / E` (ENDF-6 eq.
/// 7.2). The energy is unchanged. `e_in` must be at or above the first edge
/// (below it the coherent cross section is zero and no caller samples it).
pub fn coherent_elastic(
    e_in: f64,
    bragg_edges: &[f64],
    factors: &[f64],
    pcg: &mut u64,
) -> (f64, f64) {
    let n = bragg_edges.partition_point(|&e| e <= e_in).max(1);
    let target = next_xi(pcg) * factors[n - 1];
    // The first edge whose cumulative factor reaches the target; an edge
    // whose factor does not increase (a zero step) is never chosen.
    let k = factors[..n].partition_point(|&f| f < target).min(n - 1);
    (e_in, 1.0 - 2.0 * bragg_edges[k] / e_in)
}

/// Incoherent elastic scattering with Debye-Waller integral `w` (1/eV): the
/// cosine has density proportional to `exp(-c (1 - mu))`, `c = 2 E W'`
/// (ENDF-6 eq. 7.4), inverted exactly. Written as
/// `1 + ln(e^(-2c) + u (1 - e^(-2c))) / c`, which neither overflows for a
/// large `c` nor loses the low-`c` limit (`mu = 2u - 1`). The energy is
/// unchanged.
pub fn incoherent_elastic(e_in: f64, w: f64, pcg: &mut u64) -> (f64, f64) {
    let u = next_xi(pcg);
    let c = 2.0 * e_in * w;
    if c < 1e-10 {
        return (e_in, 2.0 * u - 1.0);
    }
    let low = (-2.0 * c).exp();
    let mu = 1.0 + (low + u * (-(-2.0 * c).exp_m1())).ln() / c;
    (e_in, mu.clamp(-1.0, 1.0))
}

/// One incident energy's outgoing distribution, as flat slices.
#[derive(Debug, Clone, Copy)]
pub struct Outgoing<'a> {
    pub energy_out: &'a [f64],
    pub pdf: &'a [f64],
    pub cdf: &'a [f64],
    /// `mu[j * n_mu + k]`.
    pub mu: &'a [f32],
    pub n_mu: usize,
}

/// Incoherent inelastic scattering from the tabulated continuous outgoing
/// distributions at incident energies `energy`.
///
/// 1. Of the two grid points around `e_in`, the upper is taken with
///    probability equal to `e_in`'s interpolation fraction between them. (The
///    usual sampler takes the nearer one; choosing in proportion is the same
///    number of draws and leaves no discretisation step at the midpoint.)
/// 2. The outgoing energy is drawn from that point's cdf, inverting the
///    linear-linear pdf in the bin, and moved to `e_in`: scaled by
///    `2 e_in / E_l - 1` below `E_l / 2`, shifted by `e_in - E_l` above.
/// 3. One of the `n_mu` equiprobable cosines is chosen, interpolated between
///    the two outgoing energies bracketing the draw, and smeared uniformly over
///    the smaller of its distances to the neighbouring cosines (to -1 and +1 at
///    the ends).
///
/// `outgoing(l)` returns grid point `l`'s distribution.
pub fn incoherent_inelastic<'a>(
    e_in: f64,
    energy: &[f64],
    outgoing: impl Fn(usize) -> Outgoing<'a>,
    pcg: &mut u64,
) -> (f64, f64) {
    let n = energy.len();
    // Grid point below e_in, clamped, and the fraction of the way to the next.
    let i = energy
        .partition_point(|&e| e <= e_in)
        .clamp(1, n.max(2) - 1)
        - 1;
    let f0 = if n < 2 {
        0.0
    } else {
        ((e_in - energy[i]) / (energy[i + 1] - energy[i])).clamp(0.0, 1.0)
    };
    let l = if n >= 2 && next_xi(pcg) < f0 {
        i + 1
    } else {
        i
    };
    let d = outgoing(l);
    let m = d.energy_out.len();

    let r1 = next_xi(pcg);
    let mut j = d.cdf[1..m].partition_point(|&c| c <= r1).min(m - 2);
    if d.cdf[j + 1] <= d.cdf[j] {
        // A flat cdf bin (no probability) cannot hold the draw; take the
        // first bin that does, as the cdf search would.
        while j + 2 < m && d.cdf[j + 1] <= d.cdf[j] {
            j += 1;
        }
    }
    let (e0, e1) = (d.energy_out[j], d.energy_out[j + 1]);
    let (p0, p1) = (d.pdf[j], d.pdf[j + 1]);
    let (c0, c1) = (d.cdf[j], d.cdf[j + 1]);
    let slope = (p1 - p0) / (e1 - e0);
    let mut e_out = if slope == 0.0 || !slope.is_finite() {
        if p0 > 0.0 {
            e0 + (r1 - c0) / p0
        } else {
            e0 + (e1 - e0) * ((r1 - c0) / (c1 - c0)).clamp(0.0, 1.0)
        }
    } else {
        e0 + ((p0 * p0 + 2.0 * slope * (r1 - c0)).max(0.0).sqrt() - p0) / slope
    };
    let e_l = energy[l];
    if e_out < 0.5 * e_l {
        e_out *= 2.0 * e_in / e_l - 1.0;
    } else {
        e_out += e_in - e_l;
    }

    let f = if c1 > c0 {
        ((r1 - c0) / (c1 - c0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let n_mu = d.n_mu;
    let k = ((next_xi(pcg) * n_mu as f64) as usize).min(n_mu - 1);
    let at = |kk: usize| -> f64 {
        let lo = d.mu[j * n_mu + kk] as f64;
        let hi = d.mu[(j + 1) * n_mu + kk] as f64;
        lo + f * (hi - lo)
    };
    let mu = at(k);
    let left = if k == 0 { -1.0 - (mu + 1.0) } else { at(k - 1) };
    let right = if k == n_mu - 1 {
        1.0 + (1.0 - mu)
    } else {
        at(k + 1)
    };
    let mu = mu + (mu - left).min(right - mu) * (next_xi(pcg) - 0.5);
    (e_out.max(0.0), mu.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bragg_edge_gives_its_own_cosine_and_a_zero_step_is_never_chosen() {
        let edges = [1e-3, 2e-3, 3e-3];
        // The middle edge adds nothing.
        let factors = [1.0, 1.0, 3.0];
        let mut pcg = 12345u64;
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            let (e, mu) = coherent_elastic(5e-3, &edges, &factors, &mut pcg);
            assert_eq!(e, 5e-3);
            let k = edges
                .iter()
                .position(|&ek| (mu - (1.0 - 2.0 * ek / 5e-3)).abs() < 1e-15)
                .unwrap();
            counts[k] += 1;
        }
        assert_eq!(counts[1], 0);
        let share = counts[0] as f64 / 30_000.0;
        assert!((share - 1.0 / 3.0).abs() < 0.01, "{share}");
        // Below the second edge only the first can scatter.
        let (_, mu) = coherent_elastic(1.5e-3, &edges, &factors, &mut pcg);
        assert_eq!(mu, 1.0 - 2.0 * 1e-3 / 1.5e-3);
    }

    /// Chi-squared of the sampled cosines against the closed-form cdf.
    #[test]
    fn incoherent_elastic_cosines_follow_the_debye_waller_density() {
        for &(e, w) in &[(0.01, 10.0), (0.1, 2.0), (2.0, 1.0), (1e-6, 1.0)] {
            let c: f64 = 2.0 * e * w;
            let cdf = |mu: f64| {
                if c < 1e-10 {
                    (mu + 1.0) / 2.0
                } else {
                    ((c * (mu - 1.0)).exp() - (-2.0 * c).exp()) / -(-2.0 * c).exp_m1()
                }
            };
            let bins = 20;
            let n = 200_000;
            let mut counts = vec![0usize; bins];
            let mut pcg = 99u64;
            for _ in 0..n {
                let (_, mu) = incoherent_elastic(e, w, &mut pcg);
                assert!((-1.0..=1.0).contains(&mu));
                counts[(((mu + 1.0) / 2.0 * bins as f64) as usize).min(bins - 1)] += 1;
            }
            let mut chi2 = 0.0;
            let mut used = 0;
            for (b, &obs) in counts.iter().enumerate() {
                let lo = -1.0 + 2.0 * b as f64 / bins as f64;
                let hi = lo + 2.0 / bins as f64;
                let exp = n as f64 * (cdf(hi) - cdf(lo));
                if exp > 5.0 {
                    chi2 += (obs as f64 - exp).powi(2) / exp;
                    used += 1;
                }
            }
            // A generous bound: about 3.5 sigma for the bins used.
            let dof = (used - 1) as f64;
            assert!(
                chi2 < dof + 3.5 * (2.0 * dof).sqrt(),
                "E={e} W={w}: chi2 {chi2} over {dof}"
            );
        }
    }

    #[test]
    fn a_large_debye_waller_exponent_neither_overflows_nor_leaves_forward_scattering() {
        // c = 2 E W' = 500: exp(2c) overflows a float, the stable form does not.
        let mut pcg = 5u64;
        let n = 100_000;
        let mean: f64 = (0..n)
            .map(|_| incoherent_elastic(5.0, 50.0, &mut pcg).1)
            .sum::<f64>()
            / n as f64;
        // The mean cosine of exp(-c(1 - mu)) on [-1, 1] is 1 - 1/c for large c.
        assert!((mean - (1.0 - 1.0 / 500.0)).abs() < 1e-4, "{mean}");
    }

    /// Incident energies, and per incident energy E', pdf, cdf and cosines.
    type Table = (
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f32>>,
    );

    fn table() -> Table {
        // Two incident energies, flat pdf on [0, 2E] (cdf linear), 4 cosines.
        let energy = vec![0.01, 0.02];
        let mut eo = Vec::new();
        let mut pdf = Vec::new();
        let mut cdf = Vec::new();
        let mut mu = Vec::new();
        for &e in &energy {
            eo.push(vec![0.0, e, 2.0 * e]);
            pdf.push(vec![0.5 / e, 0.5 / e, 0.5 / e]);
            cdf.push(vec![0.0, 0.5, 1.0]);
            mu.push([-0.75f32, -0.25, 0.25, 0.75].repeat(3));
        }
        (energy, eo, pdf, cdf, mu)
    }

    #[test]
    fn inelastic_energies_and_cosines_stay_in_range_and_track_the_incident_energy() {
        let (energy, eo, pdf, cdf, mu) = table();
        let get = |l: usize| Outgoing {
            energy_out: &eo[l],
            pdf: &pdf[l],
            cdf: &cdf[l],
            mu: &mu[l],
            n_mu: 4,
        };
        let mut pcg = 7u64;
        let n = 100_000;
        let mut mean_e = 0.0;
        let mut mean_mu = 0.0;
        for _ in 0..n {
            let (e, m) = incoherent_inelastic(0.015, &energy, get, &mut pcg);
            assert!(e >= 0.0 && (-1.0..=1.0).contains(&m));
            mean_e += e;
            mean_mu += m;
        }
        mean_e /= n as f64;
        mean_mu /= n as f64;
        // Flat on [0, 2E_l] moved to E_in = 0.015: the mean outgoing energy is
        // E_in whichever grid point was drawn.
        assert!((mean_e / 0.015 - 1.0).abs() < 0.01, "{mean_e}");
        // Symmetric cosines, smeared symmetrically: mean zero.
        assert!(mean_mu.abs() < 0.01, "{mean_mu}");
    }

    #[test]
    fn the_upper_grid_point_is_drawn_in_proportion_to_the_interpolation_fraction() {
        let (energy, eo, pdf, cdf, mu) = table();
        // Mark grid point 1 by shifting its cosines all to +0.9.
        let mut mu = mu;
        mu[1] = vec![0.9f32; 12];
        let get = |l: usize| Outgoing {
            energy_out: &eo[l],
            pdf: &pdf[l],
            cdf: &cdf[l],
            mu: &mu[l],
            n_mu: 4,
        };
        let mut pcg = 3u64;
        let n = 50_000;
        let upper = (0..n)
            .filter(|_| {
                let mu = incoherent_inelastic(0.0175, &energy, get, &mut pcg).1;
                (mu - f64::from(0.9f32)).abs() < 1e-12
            })
            .count();
        let share = upper as f64 / n as f64;
        assert!((share - 0.75).abs() < 0.01, "{share}");
    }
}
