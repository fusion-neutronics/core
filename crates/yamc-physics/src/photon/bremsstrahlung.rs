// Thick-target bremsstrahlung (TTB) emission: samples photon energies off the
// preprocessed per-material tables and banks the secondaries.
//
// The tables themselves (`Bremsstrahlung`, `BremsstrahlungData`) and the
// preprocessing that builds them from `bremsstrahlung.arrow` live in
// `yamc_element::bremsstrahlung`, next to the data they are read from. Only the
// emission half is here, because only it needs `ParticleBank` and `Particle`.

use crate::util::bank::ParticleBank;
use rand::RngExt;
use yamc_element::bremsstrahlung::Bremsstrahlung;
use yamc_particle::particle::ParticleType;

// ============================================================================
// TTB SAMPLING
// ============================================================================

/// Find the lower bounding index in a sorted slice.
/// Returns the largest index `i` such that `grid[i] <= value`.
/// If `value < grid[0]`, returns 0. If `value >= grid[n-1]`, returns n-2.
///
/// `saturating_sub` guards the degenerate len < 2 case: the TTB sampler calls
/// this on `cdf[i_e][..i_e]`, which has length 1 for the lowest electron-energy
/// bracket (`i_e == 1`, i.e. `j == 0`). There `partition_point` can return 1
/// (value at/above the single CDF point), and a plain `len - 2` would underflow
/// to `usize::MAX` and panic indexing the 200-point energy grid (#issue). Index
/// 0 is the only valid bracket start in that case.
#[inline]
fn lower_bound_index(grid: &[f64], value: f64) -> usize {
    match grid.partition_point(|&v| v <= value) {
        0 => 0,
        p if p >= grid.len() => grid.len().saturating_sub(2),
        p => p - 1,
    }
}

/// Inverse-CDF interpolation mapping a uniform draw `c` (in `[c_l, c_max]`)
/// to a thick-target-bremsstrahlung photon energy (linear eV).
///
/// `w_l_log` / `w_r_log` are the bracketing log-space photon-energy grid
/// points; `p_l` / `p_r` are the PDF values at those points and `c_l` is
/// the CDF value at the lower point, all for the chosen incident-energy
/// row. Factored out of [`thick_target_bremsstrahlung`] so the GPU kernel
/// (`yamc_gpu::photon::ttb_energy::ttb_photon_energy`) is tested against
/// the exact same closed form -- keep the two in lock-step.
pub fn sample_ttb_photon_energy(
    c: f64,
    w_l_log: f64,
    w_r_log: f64,
    p_l: f64,
    p_r: f64,
    c_l: f64,
) -> f64 {
    let a = (p_r / p_l).ln() / (w_r_log - w_l_log) + 1.0;
    // Inverse transform: w = exp(w_l) * (a*(c-c_l)/(exp(w_l)*p_l) + 1)^(1/a)
    w_l_log.exp() * (a * (c - c_l) / (w_l_log.exp() * p_l) + 1.0).powf(1.0 / a)
}

/// Sample bremsstrahlung photons from an electron or positron using the
/// thick-target bremsstrahlung approximation.
///
/// Uses precomputed PDF/CDF tables to sample photon energies and number.
///
/// The TTB energy grid must be in LOG space when this is called.
///
/// # Arguments
/// * `electron_energy` - Kinetic energy of the electron/positron in eV
/// * `is_positron` - Whether the particle is a positron
/// * `ttb_data` - Preprocessed bremsstrahlung data for this material
/// * `position` - Position of the electron (inherited by secondary photons)
/// * `direction` - Direction of the electron (inherited by secondary photons)
/// * `weight` - Statistical weight
/// * `photon_cutoff` - Minimum photon energy to bank (eV)
/// * `bank` - Particle bank for secondary photons
/// * `rng` - Random number generator
///
/// # Returns
/// Total energy lost to bremsstrahlung photons (eV).
#[allow(clippy::too_many_arguments)]
pub fn thick_target_bremsstrahlung<R: rand::Rng + ?Sized>(
    electron_energy: f64,
    is_positron: bool,
    ttb_data: &Bremsstrahlung,
    position: [f64; 3],
    direction: [f64; 3],
    weight: f64,
    photon_cutoff: f64,
    bank: &mut ParticleBank,
    rng: &mut R,
) -> f64 {
    if electron_energy < photon_cutoff {
        return 0.0;
    }

    let mat = if is_positron {
        &ttb_data.positron
    } else {
        &ttb_data.electron
    };

    let ttb_e_grid = yamc_element::photon::ttb_e_grid_log();
    let n_e = ttb_e_grid.len();
    if n_e == 0 {
        return 0.0;
    }

    // The grid is the log-space copy; the electron energy is logged to match.
    let e = electron_energy.ln();

    // Find lower bounding index
    let mut j = lower_bound_index(&ttb_e_grid, e);
    if j >= n_e - 1 {
        j = n_e - 2;
    }

    let e_l = ttb_e_grid[j];
    let e_r = ttb_e_grid[j + 1];
    let y_l = mat.yield_[j];
    let y_r = mat.yield_[j + 1];

    // Interpolation weight
    let f = if (e_r - e_l).abs() > 1e-30 {
        (e - e_l) / (e_r - e_l)
    } else {
        0.0
    };

    // Log-log interpolation of yield
    let y = (y_l + (y_r - y_l) * f).exp();

    // Sample number of photons: integer part + Bernoulli(fractional part)
    let n = (y as usize)
        + if rng.random::<f64>() < (y - (y as usize) as f64) {
            1
        } else {
            0
        };

    if n == 0 {
        return 0.0;
    }

    // Choose which PDF to use (j or j+1)
    let (i_e, c_max) = if rng.random::<f64>() <= f || j == 0 {
        let i_e = j + 1;

        // Interpolate maximum CDF value at particle energy
        let p_l = mat.pdf[i_e][i_e - 1];
        let p_r = mat.pdf[i_e][i_e];
        let c_l = mat.cdf[i_e][i_e - 1];
        let a = (p_r / p_l).ln() / (e_r - e_l) + 1.0;
        let c_max = c_l + e_l.exp() * p_l / a * (a * (e - e_l)).exp_m1();

        (i_e, c_max)
    } else {
        let i_e = j;
        let c_max = mat.cdf[i_e][i_e];
        (i_e, c_max)
    };

    let mut e_lost = 0.0;

    for _ in 0..n {
        let c = rng.random::<f64>() * c_max;

        // Binary search in CDF for this incident energy index
        let i_w = lower_bound_index(&mat.cdf[i_e][..i_e], c);

        // Sample photon energy via inverse CDF (shared with the GPU kernel).
        let w_l = ttb_e_grid[i_w];
        let w_r = ttb_e_grid[i_w + 1];
        let p_l = mat.pdf[i_e][i_w];
        let p_r = mat.pdf[i_e][i_w + 1];
        let c_l = mat.cdf[i_e][i_w];
        let mut w = sample_ttb_photon_energy(c, w_l, w_r, p_l, p_r, c_l);

        if w > photon_cutoff {
            // Ensure total energy doesn't exceed incident energy
            if e_lost + w > electron_energy {
                w = electron_energy - e_lost;
            }

            // Create secondary photon
            let mut photon = yamc_particle::particle::Particle::new(position, direction, w);
            photon.particle_type = ParticleType::Photon;
            photon.weight = weight;
            photon.alive = true;
            bank.bank_secondary(photon);

            #[cfg(feature = "debug_diagnostics")]
            {
                crate::photon_diag::TTB_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if (5000.0..10000.0).contains(&w) {
                    crate::photon_diag::TTB_5_10.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if w >= 7112.0 {
                    crate::photon_diag::TTB_ABOVE_KEDGE
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            e_lost += w;
        }
    }

    e_lost
}

#[cfg(test)]
mod tests {
    use super::lower_bound_index;

    #[test]
    fn lower_bound_index_degenerate_lengths_do_not_underflow() {
        // Length-1 slice (the TTB lowest-bracket case, j == 0). A value at or
        // above the single point must clamp to index 0, never underflow to
        // usize::MAX (which previously panicked indexing the energy grid).
        assert_eq!(lower_bound_index(&[1.0], 0.5), 0);
        assert_eq!(lower_bound_index(&[1.0], 1.0), 0);
        assert_eq!(lower_bound_index(&[1.0], 2.0), 0);
        // Length-0 slice: defensive, returns 0.
        assert_eq!(lower_bound_index(&[], 1.0), 0);
    }

    #[test]
    fn lower_bound_index_normal() {
        let g = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(lower_bound_index(&g, -1.0), 0); // below grid
        assert_eq!(lower_bound_index(&g, 0.5), 0);
        assert_eq!(lower_bound_index(&g, 1.5), 1);
        assert_eq!(lower_bound_index(&g, 2.5), 2);
        assert_eq!(lower_bound_index(&g, 9.0), 2); // above grid -> len-2
    }
}
