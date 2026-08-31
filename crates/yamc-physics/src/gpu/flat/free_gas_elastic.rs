//! Free-gas elastic scattering target velocity sampler.
//!
//! Implements the standard target-velocity rejection
//! sampling for thermal neutron elastic scattering off a Maxwell-
//! Boltzmann gas, followed by a CM→lab frame transform. Mirrors the
//! cubecl kernel's free-gas branch bit-for-bit, including the
//! constant-temperature short-circuit (`do_free_gas` predicate),
//! the 32-iteration rejection cap, and the per-iteration RNG
//! schedule (4 draws when the second branch fires, otherwise 3 +
//! the post-loop draws for `phi_t` and `phi_cm`).
//!
//! Returns `(dx, dy, dz, e_out, did_run)`. `did_run = false` when
//! the predicate falls through (no free-gas treatment for this
//! collision); callers fall back to their own kinematics path.

use yamc_rng::next_xi;

/// Boltzmann constant in eV/K. Matches the kernel literal exactly.
const K_B: f64 = 8.617333e-5;
const SQRT_PI: f64 = 1.7724538509055159;
const TWO_PI: f64 = std::f64::consts::TAU;

/// Sample an outgoing neutron direction and energy from free-gas
/// elastic scattering off a target nucleus of mass `target_mass`
/// (in neutron masses) at temperature `temperature_k`.
///
/// `free_gas_threshold` is the resonance/thermal cutoff multiplier
/// (the model option of the same name, default `400.0`): the free-gas
/// regime boundary is `free_gas_threshold * kT`. The CPU production
/// path, the GPU kernel, and this CPU twin all pass the same value, so
/// the regime boundary cannot drift between backends (issue #102).
///
/// # Returns
/// `(dx, dy, dz, e_out, did_run)` where `did_run = true` indicates
/// the free-gas path was taken and the new direction/energy should
/// be used. `did_run = false` means either:
/// - `temperature_k <= 0` (no temperature data), or
/// - `target_mass > 1` and `energy_in >= free_gas_threshold·kT` (above
///   the free-gas regime -- caller should treat as cold-target elastic).
///
/// # RNG schedule
///
/// Per rejection iteration:
/// - 3 base draws (r1, r2, r3)
/// - 1 extra draw (r4) if the branch `r3 >= alpha_w` fires
/// - 1 final draw (r5) for the candidate-μ acceptance
/// - 1 final draw (r6) for the acceptance test
///
/// After the loop: 1 draw for `phi_t`, 1 for `phi_cm`.
#[allow(clippy::too_many_arguments)]
pub fn sample_free_gas_elastic(
    energy_in: f64,
    dx_in: f64,
    dy_in: f64,
    dz_in: f64,
    target_mass: f64,
    temperature_k: f64,
    free_gas_threshold: f64,
    mu_cm: f64,
    state: &mut u64,
) -> (f64, f64, f64, f64, bool) {
    if temperature_k <= 0.0 {
        return (dx_in, dy_in, dz_in, energy_in, false);
    }
    let kt = K_B * temperature_k;
    let do_free_gas = if target_mass > 1.0 {
        energy_in < free_gas_threshold * kt
    } else {
        true
    };
    if !do_free_gas {
        return (dx_in, dy_in, dz_in, energy_in, false);
    }

    let beta_vn_sq = target_mass * energy_in / kt;
    let beta_vn = beta_vn_sq.sqrt();
    let alpha_w = 1.0 / (1.0 + SQRT_PI * beta_vn * 0.5);
    let mut beta_vt_sq = 0.0_f64;
    let mut mu_t = 0.0_f64;
    let mut accepted = false;
    let mut iter = 0u32;
    while iter < 32 && !accepted {
        let r1 = next_xi(state);
        let r2 = next_xi(state);
        let r3 = next_xi(state);
        let cand: f64 = if r3 < alpha_w {
            -(r1 * r2).ln()
        } else {
            let r4 = next_xi(state);
            let cv = (std::f64::consts::FRAC_PI_2 * r4).cos();
            -(r1.ln()) - r2.ln() * cv * cv
        };

        let r5 = next_xi(state);
        let mu_cand = 2.0 * r5 - 1.0;

        let beta_vt_local = cand.sqrt();
        let v_rel_sq = beta_vn_sq + cand - 2.0 * beta_vn * beta_vt_local * mu_cand;
        let v_rel = if v_rel_sq > 0.0 { v_rel_sq.sqrt() } else { 0.0 };
        let acc = v_rel / (beta_vn + beta_vt_local);

        let r6 = next_xi(state);
        if r6 < acc {
            beta_vt_sq = cand;
            mu_t = mu_cand;
            accepted = true;
        }
        iter += 1;
    }

    let vt_mag = (beta_vt_sq * kt / target_mass).sqrt();

    // Direction of v_t: rotate (dx_in, dy_in, dz_in) by (mu_t, phi_t).
    let xi_phi_t = next_xi(state);
    let phi_t = TWO_PI * xi_phi_t;
    let sin_th_t = (1.0 - mu_t * mu_t).max(0.0).sqrt();
    let cos_phi_t = phi_t.cos();
    let sin_phi_t = phi_t.sin();
    let b_t = (1.0 - dz_in * dz_in).max(0.0).sqrt();
    let mut t_dx = sin_th_t * cos_phi_t;
    let mut t_dy = sin_th_t * sin_phi_t;
    let mut t_dz = mu_t;
    if dz_in < 0.0 {
        t_dy = -t_dy;
        t_dz = -mu_t;
    }
    if b_t > 1e-10 {
        t_dx = mu_t * dx_in + sin_th_t * (dx_in * dz_in * cos_phi_t - dy_in * sin_phi_t) / b_t;
        t_dy = mu_t * dy_in + sin_th_t * (dy_in * dz_in * cos_phi_t + dx_in * sin_phi_t) / b_t;
        t_dz = mu_t * dz_in - sin_th_t * b_t * cos_phi_t;
    }
    let v_tx = vt_mag * t_dx;
    let v_ty = vt_mag * t_dy;
    let v_tz = vt_mag * t_dz;

    let vel_n = energy_in.sqrt();
    let v_nx = dx_in * vel_n;
    let v_ny = dy_in * vel_n;
    let v_nz = dz_in * vel_n;

    let inv_apl1 = 1.0 / (target_mass + 1.0);
    let v_cm_x = (v_nx + target_mass * v_tx) * inv_apl1;
    let v_cm_y = (v_ny + target_mass * v_ty) * inv_apl1;
    let v_cm_z = (v_nz + target_mass * v_tz) * inv_apl1;

    let v_ncm_x = v_nx - v_cm_x;
    let v_ncm_y = v_ny - v_cm_y;
    let v_ncm_z = v_nz - v_cm_z;
    let vel_cm_sq = v_ncm_x * v_ncm_x + v_ncm_y * v_ncm_y + v_ncm_z * v_ncm_z;

    let xi_phi_cm = next_xi(state);
    let phi_cm = TWO_PI * xi_phi_cm;

    if vel_cm_sq < 1e-20 {
        let sin_th = (1.0 - mu_cm * mu_cm).max(0.0).sqrt();
        let cphi = phi_cm.cos();
        let sphi = phi_cm.sin();
        let new_dx = sin_th * cphi;
        let new_dy = sin_th * sphi;
        let new_dz = mu_cm;
        let new_e = v_cm_x * v_cm_x + v_cm_y * v_cm_y + v_cm_z * v_cm_z;
        let e_out = if new_e > 0.0 { new_e } else { energy_in };
        return (new_dx, new_dy, new_dz, e_out, true);
    }

    let vel_cm = vel_cm_sq.sqrt();
    let inv_vel_cm = 1.0 / vel_cm;
    let u_cmx = v_ncm_x * inv_vel_cm;
    let u_cmy = v_ncm_y * inv_vel_cm;
    let u_cmz = v_ncm_z * inv_vel_cm;

    let sin_th_cm = (1.0 - mu_cm * mu_cm).max(0.0).sqrt();
    let cphi_cm = phi_cm.cos();
    let sphi_cm = phi_cm.sin();
    let b_cm = (1.0 - u_cmz * u_cmz).max(0.0).sqrt();
    let mut u_new_x = sin_th_cm * cphi_cm;
    let mut u_new_y = sin_th_cm * sphi_cm;
    let mut u_new_z = mu_cm;
    if u_cmz < 0.0 {
        u_new_y = -u_new_y;
        u_new_z = -mu_cm;
    }
    if b_cm > 1e-10 {
        u_new_x = mu_cm * u_cmx + sin_th_cm * (u_cmx * u_cmz * cphi_cm - u_cmy * sphi_cm) / b_cm;
        u_new_y = mu_cm * u_cmy + sin_th_cm * (u_cmy * u_cmz * cphi_cm + u_cmx * sphi_cm) / b_cm;
        u_new_z = mu_cm * u_cmz - sin_th_cm * b_cm * cphi_cm;
    }
    let v_ncm_new_x = vel_cm * u_new_x;
    let v_ncm_new_y = vel_cm * u_new_y;
    let v_ncm_new_z = vel_cm * u_new_z;
    let v_nlab_x = v_ncm_new_x + v_cm_x;
    let v_nlab_y = v_ncm_new_y + v_cm_y;
    let v_nlab_z = v_ncm_new_z + v_cm_z;
    let new_e = v_nlab_x * v_nlab_x + v_nlab_y * v_nlab_y + v_nlab_z * v_nlab_z;
    if new_e > 0.0 {
        let inv_vlab = 1.0 / new_e.sqrt();
        (
            v_nlab_x * inv_vlab,
            v_nlab_y * inv_vlab,
            v_nlab_z * inv_vlab,
            new_e,
            true,
        )
    } else {
        (dx_in, dy_in, dz_in, energy_in, true)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `temperature_k <= 0` → `did_run = false`, no state advance,
    /// and the input direction/energy are returned unchanged.
    #[test]
    fn free_gas_short_circuits_on_zero_temperature() {
        let mut state: u64 = 0xFEED;
        let before = state;
        let (dx, dy, dz, e, ran) =
            sample_free_gas_elastic(1.0e6, 0.0, 0.0, 1.0, 56.0, 0.0, 400.0, 0.5, &mut state);
        assert!(!ran);
        assert_eq!(state, before, "no temperature must not advance RNG state");
        assert_eq!((dx, dy, dz, e), (0.0, 0.0, 1.0, 1.0e6));
    }

    /// Heavy target above `400·kT` → cold-target path (no free-gas
    /// treatment). `did_run = false`, no RNG advance.
    #[test]
    fn free_gas_short_circuits_above_400_kt() {
        let mut state: u64 = 0xBEEF;
        let before = state;
        // 56·1eV-equivalent: 400·kT at 294 K is ≈ 10.13 eV → above
        // means well over that. Use 1 keV.
        let (_, _, _, _, ran) =
            sample_free_gas_elastic(1.0e3, 0.0, 0.0, 1.0, 56.0, 294.0, 400.0, 0.5, &mut state);
        assert!(!ran);
        assert_eq!(state, before);
    }

    /// Below `400·kT` for a heavy target → free-gas path runs and
    /// produces a unit-vector direction with positive energy.
    #[test]
    fn free_gas_runs_below_400_kt_for_heavy_target() {
        let mut state: u64 = 1;
        // 0.025 eV thermal neutron on Fe-56 at room T -- well within
        // the free-gas regime.
        let (dx, dy, dz, e, ran) =
            sample_free_gas_elastic(0.025, 0.0, 0.0, 1.0, 56.0, 294.0, 400.0, 0.5, &mut state);
        assert!(ran);
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-9,
            "output direction must be unit length, got mag={mag}"
        );
        assert!(e > 0.0, "outgoing energy must be positive, got {e}");
    }

    /// H-1 always takes the free-gas path (target_mass == 1.0
    /// short-circuits the energy check).
    #[test]
    fn free_gas_always_runs_for_hydrogen() {
        let mut state: u64 = 42;
        let (_, _, _, _, ran) =
            sample_free_gas_elastic(1.0e6, 0.0, 0.0, 1.0, 1.0, 294.0, 400.0, 0.5, &mut state);
        assert!(
            ran,
            "H-1 must always run free-gas path regardless of energy"
        );
    }

    /// A non-default `free_gas_threshold` moves the regime boundary
    /// (issue #102). At one fixed energy / temperature, a low threshold
    /// puts the neutron above the boundary (cold-target, `did_run =
    /// false`) while a high threshold puts it below (free gas, `did_run
    /// = true`). Guards against any backend re-hardcoding `400`.
    #[test]
    fn free_gas_threshold_is_honored() {
        // 400·kT at 294 K is ~10.13 eV. Pick 100 eV, which sits above
        // a threshold of 400 but below a threshold of 8000.
        let e = 100.0;
        let mut lo_state: u64 = 0xABCD;
        let (_, _, _, _, ran_lo) =
            sample_free_gas_elastic(e, 0.0, 0.0, 1.0, 56.0, 294.0, 400.0, 0.5, &mut lo_state);
        assert!(!ran_lo, "100 eV is above 400·kT -> cold target");

        let mut hi_state: u64 = 0xABCD;
        let (_, _, _, _, ran_hi) =
            sample_free_gas_elastic(e, 0.0, 0.0, 1.0, 56.0, 294.0, 8000.0, 0.5, &mut hi_state);
        assert!(ran_hi, "100 eV is below 8000·kT -> free gas");
    }
}
