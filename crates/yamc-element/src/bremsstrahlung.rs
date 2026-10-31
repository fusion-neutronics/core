// Thick-target bremsstrahlung (TTB) data structures and preprocessing.
//
// The tables are built once per material from the per-element
// `bremsstrahlung.arrow` data this crate already reads (see
// `photon_arrow::read_bremsstrahlung`), which is why the preprocessing half
// lives here. The emission half -- sampling a photon energy off these tables
// and banking the secondary -- needs `ParticleBank` and `Particle`, so it stays
// in `yamc_physics::photon::bremsstrahlung`.

use crate::photon::MASS_ELECTRON_EV;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Planck's constant times c in eV-Angstroms.
const PLANCK_C: f64 = 1.2398419839593942e4;

/// Inverse fine structure constant.
const FINE_STRUCTURE: f64 = 137.035999084;

/// Avogadro's number in 10^24/mol.
const N_AVOGADRO: f64 = 0.602214076;

/// Neutron mass in amu.
const MASS_NEUTRON: f64 = 1.00866491595;

/// Barn per cm^2.
const BARN_PER_CM_SQ: f64 = 1.0e24;

/// cm per Angstrom.
const CM_PER_ANGSTROM: f64 = 1.0e-8;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Bremsstrahlung energy distribution data for one particle type.
/// PDF and CDF are lower-triangular: `pdf[j][i]` and `cdf[j][i]` are valid
/// only for `i <= j`.
#[derive(Debug, Clone)]
pub struct BremsstrahlungData {
    /// Bremsstrahlung energy PDF. `pdf[j][i]` = cumulative integral of the
    /// PDF from photon energy index i to j for incident energy index j.
    pub pdf: Vec<Vec<f64>>,
    /// Bremsstrahlung energy CDF. `cdf[j][i]` = CDF value at photon energy
    /// index i for incident energy index j.
    pub cdf: Vec<Vec<f64>>,
    /// Photon number yield per incident energy (stored in log space).
    pub yield_: Vec<f64>,
}

/// Container for electron and positron bremsstrahlung data.
#[derive(Debug, Clone)]
pub struct Bremsstrahlung {
    pub electron: BremsstrahlungData,
    pub positron: BremsstrahlungData,
}

// ============================================================================
// CUBIC SPLINE HELPERS
// ============================================================================

/// Compute natural cubic spline second derivatives.
/// `x` and `y` are the data points, `z` receives the second derivatives.
/// Uses tridiagonal matrix algorithm with natural boundary conditions (z[0] = z[n-1] = 0).
fn spline(x: &[f64], y: &[f64], z: &mut [f64]) {
    let n = x.len();
    if n < 3 {
        for v in z.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mut c_new = vec![0.0; n - 1];
    z[0] = 0.0;
    z[n - 1] = 0.0;

    // Forward sweep
    for i in 1..n - 1 {
        let a = x[i] - x[i - 1];
        let c = x[i + 1] - x[i];
        let b = 2.0 * (a + c);
        let d = 6.0 * ((y[i + 1] - y[i]) / c - (y[i] - y[i - 1]) / a);

        let denom = b - a * c_new[i - 1];
        c_new[i] = c / denom;
        z[i] = (d - a * z[i - 1]) / denom;
    }

    // Back substitution
    for i in (0..n - 2).rev() {
        z[i] -= c_new[i] * z[i + 1];
    }
}

/// Integrate a cubic spline from `xa` to `xb`.
/// `x`, `y` are the data points, `z` are the second derivatives from `spline()`.
fn spline_integrate(x: &[f64], y: &[f64], z: &[f64], xa: f64, xb: f64) -> f64 {
    let n = x.len();

    // Find lower bounding index of xa
    let mut ia = n - 1;
    while ia > 0 {
        ia -= 1;
        if xa >= x[ia] {
            break;
        }
    }

    // Find lower bounding index of xb
    let mut ib = n - 1;
    while ib > 0 {
        ib -= 1;
        if xb >= x[ib] {
            break;
        }
    }

    let mut s = 0.0;
    for i in ia..=ib {
        let mut h = x[i + 1] - x[i];
        let b = (y[i + 1] - y[i]) / h - (h / 6.0) * (z[i + 1] + 2.0 * z[i]);
        let c = z[i] / 2.0;
        let d = (z[i + 1] - z[i]) / (h * 6.0);

        // Subtract integral from x[ia] to xa
        if i == ia {
            let r = xa - x[ia];
            s -= y[i] * r + b / 2.0 * r * r + c / 3.0 * r * r * r + d / 4.0 * r * r * r * r;
        }

        // In final interval, integrate only to xb
        if i == ib {
            h = xb - x[ib];
        }

        // Accumulate integral
        s += y[i] * h + b / 2.0 * h * h + c / 3.0 * h * h * h + d / 4.0 * h * h * h * h;
    }

    s
}

// ============================================================================
// STERNHEIMER DENSITY EFFECT
// ============================================================================

/// Compute the Sternheimer adjustment factor rho using Newton's method.
/// Uses Newton's method to solve for the adjustment parameter.
fn sternheimer_adjustment(
    f: &[f64],
    e_b_sq: &[f64],
    e_p_sq: f64,
    n_conduction: f64,
    log_i: f64,
) -> f64 {
    let n = f.len();
    let tol = 1.0e-6;
    let max_iter = 100;

    let mut rho = 2.0;
    for _iter in 0..max_iter {
        let rho_0 = rho;

        let mut g = 0.0;
        let mut gp = 0.0;

        for i in 0..n {
            let e_r_sq = e_b_sq[i] * rho * rho + 2.0 / 3.0 * f[i] * e_p_sq;
            g += f[i] * e_r_sq.ln();
            gp += e_b_sq[i] * f[i] * rho / e_r_sq;
        }

        if n_conduction > 0.0 {
            g += n_conduction * (n_conduction * e_p_sq).ln();
        }

        rho -= (g - 2.0 * log_i) / (2.0 * gp);

        if rho < 0.0 {
            rho = rho_0 / 2.0;
        }

        if ((rho - rho_0) / rho_0).abs() < tol {
            return rho;
        }
    }

    // Did not converge
    eprintln!("Warning: Sternheimer adjustment Newton's method did not converge");
    1.0e-6
}

/// Compute the density effect correction delta at a given energy.
/// Computes the density effect correction using the Sternheimer model.
fn density_effect(
    f: &[f64],
    e_b_sq: &[f64],
    e_p_sq: f64,
    n_conduction: f64,
    rho: f64,
    energy: f64,
) -> f64 {
    let n = f.len();
    let tol = 1.0e-6;
    let max_iter = 100;

    let beta_sq = energy * (energy + 2.0 * MASS_ELECTRON_EV)
        / ((energy + MASS_ELECTRON_EV) * (energy + MASS_ELECTRON_EV));

    // For non-metals, delta = 0 for beta < beta_0
    let mut beta_0_sq = 0.0;
    if n_conduction == 0.0 {
        for i in 0..n {
            beta_0_sq += f[i] * e_p_sq / (e_b_sq[i] * rho * rho);
        }
        beta_0_sq = 1.0 / (1.0 + beta_0_sq);
    }
    if beta_sq < beta_0_sq {
        return 0.0;
    }

    // Compute w^2 using Newton's method
    let mut w_sq = energy / MASS_ELECTRON_EV * (energy / MASS_ELECTRON_EV + 2.0);
    for _iter in 0..max_iter {
        let w_sq_0 = w_sq;

        let mut g = 0.0;
        let mut gp = 0.0;

        for i in 0..n {
            let c = e_b_sq[i] * rho * rho / e_p_sq + w_sq;
            g += f[i] / c;
            gp -= f[i] / (c * c);
        }

        g += n_conduction / w_sq;
        gp -= n_conduction / (w_sq * w_sq);

        w_sq -= (g + 1.0 - 1.0 / beta_sq) / gp;

        if w_sq < 0.0 {
            w_sq = w_sq_0 / 2.0;
        }

        if ((w_sq - w_sq_0) / w_sq_0).abs() < tol {
            // Converged -- compute delta
            let mut delta = 0.0;
            for i in 0..n {
                let l_sq = e_b_sq[i] * rho * rho / e_p_sq + 2.0 / 3.0 * f[i];
                delta += f[i] * ((l_sq + w_sq) / l_sq).ln();
            }
            if n_conduction > 0.0 {
                delta += n_conduction * ((n_conduction + w_sq) / n_conduction).ln();
            }
            return delta - w_sq * (1.0 - beta_sq);
        }
    }

    // Did not converge
    eprintln!("Warning: Density effect Newton's method did not converge");
    0.0
}

// ============================================================================
// COLLISION STOPPING POWER
// ============================================================================

/// Compute collision stopping power for a material at each TTB energy grid point.
/// Uses the Bethe formula with shell and density effect corrections.
///
/// The TTB energy grid must be in LINEAR space (eV) when this is called.
#[allow(clippy::type_complexity)]
fn collision_stopping_power(
    ttb_e_grid: &[f64],
    element_data: &[(f64, u32, &[f64], &[f64], f64)], // (atom_density, Z, n_electrons, ionization_energy, I)
    nuclide_awrs: &[f64],                             // AWR for each element/nuclide pair
    positron: bool,
) -> Vec<f64> {
    let n_e = ttb_e_grid.len();

    // Accumulate material properties
    let mut electron_density = 0.0_f64;
    let mut mass_density = 0.0_f64;
    let mut log_i = 0.0_f64;
    let mut n_conduction = 0.0_f64;
    let mut f_vec: Vec<f64> = Vec::new();
    let mut e_b_sq_vec: Vec<f64> = Vec::new();

    for (idx, &(atom_density, z, n_electrons, ionization_energy, mean_i)) in
        element_data.iter().enumerate()
    {
        electron_density += atom_density * z as f64;
        mass_density += atom_density * nuclide_awrs[idx] * MASS_NEUTRON;
        log_i += atom_density * z as f64 * mean_i.ln();

        for j in 0..n_electrons.len() {
            if n_electrons[j] < 0.0 {
                n_conduction -= n_electrons[j] * atom_density;
                continue;
            }
            e_b_sq_vec.push(ionization_energy[j] * ionization_energy[j]);
            f_vec.push(n_electrons[j] * atom_density);
        }
    }

    log_i /= electron_density;
    n_conduction /= electron_density;
    for fi in &mut f_vec {
        *fi /= electron_density;
    }

    // Get density in g/cm^3
    let density = mass_density / N_AVOGADRO;

    // Square of the plasma energy
    let e_p_sq = PLANCK_C * PLANCK_C * PLANCK_C * N_AVOGADRO * electron_density * density
        / (2.0
            * std::f64::consts::PI
            * std::f64::consts::PI
            * FINE_STRUCTURE
            * MASS_ELECTRON_EV
            * mass_density);

    // Sternheimer adjustment factor
    let rho = sternheimer_adjustment(&f_vec, &e_b_sq_vec, e_p_sq, n_conduction, log_i);

    // Classical electron radius in cm
    let r_e = CM_PER_ANGSTROM * PLANCK_C
        / (2.0 * std::f64::consts::PI * FINE_STRUCTURE * MASS_ELECTRON_EV);

    // Constant in the collision stopping power expression
    let c = BARN_PER_CM_SQ
        * 2.0
        * std::f64::consts::PI
        * r_e
        * r_e
        * MASS_ELECTRON_EV
        * electron_density;

    let mut s_col = vec![0.0; n_e];

    for i in 0..n_e {
        let energy = ttb_e_grid[i];

        let delta = density_effect(&f_vec, &e_b_sq_vec, e_p_sq, n_conduction, rho, energy);

        let beta_sq = energy * (energy + 2.0 * MASS_ELECTRON_EV)
            / ((energy + MASS_ELECTRON_EV) * (energy + MASS_ELECTRON_EV));

        let tau = energy / MASS_ELECTRON_EV;

        let f_correction = if positron {
            let t = tau + 2.0;
            4.0_f64.ln() - (beta_sq / 12.0) * (23.0 + 14.0 / t + 10.0 / (t * t) + 4.0 / (t * t * t))
        } else {
            (1.0 - beta_sq) * (1.0 + tau * tau / 8.0 - (2.0 * tau + 1.0) * 2.0_f64.ln())
        };

        s_col[i] = c / beta_sq
            * (2.0 * (energy.ln() - log_i) + (1.0 + tau / 2.0).ln() + f_correction - delta);
    }

    s_col
}

// ============================================================================
// INIT BREMSSTRAHLUNG (preprocessing)
// ============================================================================

/// Initialize bremsstrahlung data for a material.
///
/// Must be called while TTB_E_GRID is still in LINEAR space.
/// Computes PDF, CDF, and photon yield for electron and positron bremsstrahlung.
///
/// # Arguments
/// * `ttb_e_grid` - Electron energy grid in LINEAR space (eV).
/// * `ttb_k_grid` - Reduced photon energy grid (k = E_photon / E_electron).
/// * `element_data` - Per-element: (atom_density, Z, &dcs[n_e][n_k], &stopping_power_radiative[n_e], &n_electrons, &ionization_energy, mean_excitation_energy)
/// * `nuclide_awrs` - AWR for each element/nuclide pair
#[allow(clippy::type_complexity)]
pub fn init_bremsstrahlung(
    ttb_e_grid: &[f64],
    ttb_k_grid: &[f64],
    element_data: &[(f64, u32, &[Vec<f64>], &[f64], &[f64], &[f64], f64)],
    nuclide_awrs: &[f64],
) -> Bremsstrahlung {
    let n_e = ttb_e_grid.len();
    let n_k = ttb_k_grid.len();

    let mut bremsstrahlung = Bremsstrahlung {
        electron: BremsstrahlungData {
            pdf: vec![vec![0.0; n_e]; n_e],
            cdf: vec![vec![0.0; n_e]; n_e],
            yield_: vec![0.0; n_e],
        },
        positron: BremsstrahlungData {
            pdf: vec![vec![0.0; n_e]; n_e],
            cdf: vec![vec![0.0; n_e]; n_e],
            yield_: vec![0.0; n_e],
        },
    };

    for particle in 0..2 {
        let positron = particle == 1;

        // Collect data for collision stopping power
        let csp_data: Vec<(f64, u32, &[f64], &[f64], f64)> = element_data
            .iter()
            .map(|&(ad, z, _dcs, _srad, ne, ie, mei)| (ad, z, ne, ie, mei))
            .collect();

        let stopping_power_collision =
            collision_stopping_power(ttb_e_grid, &csp_data, nuclide_awrs, positron);

        // Aggregate material DCS and radiative stopping power via Bragg additivity
        let mut dcs = vec![vec![0.0; n_k]; n_e];
        let mut stopping_power_radiative = vec![0.0; n_e];
        let mut z_eq_sq = 0.0_f64;
        let mut sum_density = 0.0_f64;

        for &(atom_density, z, elem_dcs, elem_srad, _ne, _ie, _mei) in element_data {
            let z_f = z as f64;
            z_eq_sq += atom_density * z_f * z_f;
            sum_density += atom_density;

            for i in 0..n_e {
                // DCS weighted by atom_density * Z^2
                for j in 0..n_k {
                    dcs[i][j] += atom_density * z_f * z_f * elem_dcs[i][j];
                }
                // Radiative stopping power weighted by atom_density
                stopping_power_radiative[i] += atom_density * elem_srad[i];
            }
        }
        z_eq_sq /= sum_density;

        // Apply PENELOPE positron correction factor
        if positron {
            for i in 0..n_e {
                let t = (1.0 + 1.0e6 * ttb_e_grid[i] / (z_eq_sq * MASS_ELECTRON_EV)).ln();
                let r = 1.0
                    - (-1.2359e-1 * t + 6.1274e-2 * t.powi(2) - 3.1516e-2 * t.powi(3)
                        + 7.7446e-3 * t.powi(4)
                        - 1.0595e-3 * t.powi(5)
                        + 7.0568e-5 * t.powi(6)
                        - 1.808e-6 * t.powi(7))
                    .exp();
                stopping_power_radiative[i] *= r;
                for dcs_val in dcs[i].iter_mut().take(n_k) {
                    *dcs_val *= r;
                }
            }
        }

        // Total stopping power
        let stopping_power: Vec<f64> = stopping_power_collision
            .iter()
            .zip(stopping_power_radiative.iter())
            .map(|(&c, &r)| c + r)
            .collect();

        // Get mutable reference to the right particle's data
        let ttb = if positron {
            &mut bremsstrahlung.positron
        } else {
            &mut bremsstrahlung.electron
        };

        // Compute PDF by cubic spline integration over incident energy
        let mut f_arr = vec![0.0; n_e];
        let mut z_arr = vec![0.0; n_e];

        for i in 0..n_e - 1 {
            let w = ttb_e_grid[i]; // photon energy

            // Compute integrand f(j) for incident energies j >= i
            for j in i..n_e {
                let e = ttb_e_grid[j]; // incident electron energy
                let k = w / e; // reduced photon energy

                // Find lower bounding index in ttb_k_grid
                let i_k = match ttb_k_grid.partition_point(|&v| v <= k) {
                    0 => 0,
                    p => (p - 1).min(n_k - 2),
                };

                let k_l = ttb_k_grid[i_k];
                let k_r = ttb_k_grid[i_k + 1];
                let x_l = dcs[j][i_k];
                let x_r = dcs[j][i_k + 1];

                // Linear interpolation in reduced photon energy
                let x = x_l + (k - k_l) * (x_r - x_l) / (k_r - k_l);

                let beta_sq = e * (e + 2.0 * MASS_ELECTRON_EV)
                    / ((e + MASS_ELECTRON_EV) * (e + MASS_ELECTRON_EV));

                f_arr[j] = x / (beta_sq * stopping_power[j] * w);
            }

            let n_pts = n_e - i;

            if n_pts > 2 {
                // Cubic spline integration
                spline(&ttb_e_grid[i..], &f_arr[i..], &mut z_arr[i..]);

                let mut c = 0.0;
                for j in i..n_e - 1 {
                    c += spline_integrate(
                        &ttb_e_grid[i..],
                        &f_arr[i..],
                        &z_arr[i..],
                        ttb_e_grid[j],
                        ttb_e_grid[j + 1],
                    );
                    ttb.pdf[j + 1][i] = c;
                }
            } else {
                // 2-point trapezoidal rule in log-log space
                let e_l = ttb_e_grid[i].ln();
                let e_r = ttb_e_grid[i + 1].ln();
                let x_l = f_arr[i].ln();
                let x_r = f_arr[i + 1].ln();
                ttb.pdf[i + 1][i] = 0.5 * (e_r - e_l) * ((e_l + x_l).exp() + (e_r + x_r).exp());
            }
        }

        // Compute CDF from PDF using analytical log-log integration
        for j in 1..n_e {
            // Set last element to small non-zero for log-log interpolation
            ttb.pdf[j][j] = (-500.0_f64).exp();

            let mut c = 0.0;
            for i in 0..j {
                let w_l = ttb_e_grid[i].ln();
                let w_r = ttb_e_grid[i + 1].ln();
                let x_l = ttb.pdf[j][i].ln();
                let x_r = ttb.pdf[j][i + 1].ln();
                let beta = (x_r - x_l) / (w_r - w_l);
                let a = beta + 1.0;

                // Analytical integral: exp(w_l + x_l) / a * expm1(a * (w_r - w_l))
                c += (w_l + x_l).exp() / a * (a * (w_r - w_l)).exp_m1();

                ttb.cdf[j][i + 1] = c;
            }

            // Photon number yield
            ttb.yield_[j] = c;
        }

        // Store yield in log space
        for y in &mut ttb.yield_ {
            if *y > 0.0 {
                *y = y.ln();
            } else {
                *y = -500.0;
            }
        }
    }

    bremsstrahlung
}
