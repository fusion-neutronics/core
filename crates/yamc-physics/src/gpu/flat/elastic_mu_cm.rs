//! Elastic scattering cosine-of-angle sampler in the CM frame.
//!
//! Two-stage tabulated sampling:
//!   1. Stochastic incident-energy bracket pick from a per-material
//!      `(elastic_angle_energy_grid, n_mu, interp)` table.
//!   2. CDF inversion within the chosen bracket using the per-bracket
//!      `(mu, cdf, pdf, interp)` arrays.
//!
//! Falls back to the supplied isotropic μ when the slot is empty or
//! the per-bracket table has fewer than 2 points.

use super::grid::locate_bracket;
use yamc_rng::next_xi;

/// Interpolation flag for the per-bracket μ/CDF table -- `0` selects
/// histogram interpolation (`μ = x_j + (ξ − c_j)/p_j`), `1` selects
/// linear-linear (the quadratic `m = (p_{j+1}−p_j)/(x_{j+1}−x_j)`
/// branch). Mirrored from the kernel's `ANGLE_INTERP_*` u32 codes.
pub const ANGLE_INTERP_HISTOGRAM: u32 = 0;
pub const ANGLE_INTERP_LINLIN: u32 = 1;

/// Sample the cosine of the elastic-scattering angle in the CM frame.
///
/// # Arguments
/// * `energy` -- incident energy.
/// * `xi3_isotropic_fallback` -- pre-sampled `[0,1)` draw used to seed
///   the isotropic fallback (`μ = 1 − 2·ξ₃`) when the slot has no
///   tabulated data. The caller draws this before calling so the RNG
///   schedule stays bit-identical to the kernel even when this
///   function takes the fallback branch.
/// * `energy_grid` -- sorted per-material incident-energy grid, length
///   `n_ae`.
/// * `n_mu_per_e` -- number of (μ, CDF) points at each incident-energy
///   point, length `n_ae`.
/// * `interp_per_e` -- interpolation flag per incident-energy point,
///   length `n_ae` (`ANGLE_INTERP_HISTOGRAM` or `ANGLE_INTERP_LINLIN`).
/// * `mu_table`, `cdf_table`, `pdf_table` -- flat tables packed
///   variable-length: row `i` occupies `mu_offset[i] .. mu_offset[i] +
///   n_mu_per_e[i]`. Rows are stored back-to-back with no padding.
/// * `mu_offset` -- start index of each incident-energy row in the
///   flat `mu_table` / `cdf_table` / `pdf_table`, length `n_ae`.
/// * `state` -- inline PCG RNG state, advanced 2× on the happy path
///   (one for the bracket pick, one for the CDF inversion).
///
/// Returns the sampled `μ` clamped to `[-1, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn sample_elastic_mu_cm(
    energy: f64,
    xi3_isotropic_fallback: f64,
    energy_grid: &[f64],
    n_mu_per_e: &[u32],
    interp_per_e: &[u32],
    mu_table: &[f64],
    cdf_table: &[f64],
    pdf_table: &[f64],
    mu_offset: &[u32],
    state: &mut u64,
) -> f64 {
    let n_ae = energy_grid.len();
    let mut mu_cm = 1.0 - 2.0 * xi3_isotropic_fallback;
    if n_ae == 0 {
        return mu_cm;
    }
    let (i_ae, r_ae) = locate_bracket(energy, energy_grid);
    let xi_eb = next_xi(state);
    let mut bin_e = i_ae;
    if r_ae > xi_eb && bin_e + 1 < n_ae {
        bin_e = i_ae + 1;
    }
    let n_mu = n_mu_per_e[bin_e] as usize;
    if n_mu < 2 {
        return mu_cm;
    }
    let mu_off = mu_offset[bin_e] as usize;
    let xi_mu = next_xi(state);
    let mut j: usize = 0;
    let mut j_found = false;
    for k in 0..n_mu - 1 {
        let c_k1 = cdf_table[mu_off + k + 1];
        if !j_found && xi_mu <= c_k1 {
            j = k;
            j_found = true;
        }
    }
    if !j_found {
        j = n_mu - 2;
    }
    let c_j = cdf_table[mu_off + j];
    let x_j = mu_table[mu_off + j];
    let x_j1 = mu_table[mu_off + j + 1];
    let p_j = pdf_table[mu_off + j];
    let p_j1 = pdf_table[mu_off + j + 1];
    let interp_kind = interp_per_e[bin_e];
    let dx = x_j1 - x_j;
    mu_cm = if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
        let m = (p_j1 - p_j) / dx;
        if m.abs() < 1e-30 {
            if p_j > 0.0 {
                x_j + (xi_mu - c_j) / p_j
            } else {
                x_j
            }
        } else {
            let disc = (p_j * p_j + 2.0 * m * (xi_mu - c_j)).max(0.0);
            x_j + (disc.sqrt() - p_j) / m
        }
    } else if p_j > 0.0 {
        x_j + (xi_mu - c_j) / p_j
    } else {
        x_j
    };
    mu_cm.clamp(-1.0, 1.0)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build a 2-point energy grid with a forward-peaked
    /// CDF at both points (μ ∈ [0,1] only, equiprobable on the upper
    /// half). 4k samples should produce a mean ≈ 0.5 -- catches gross
    /// arithmetic bugs in the CDF inversion + interp branches.
    #[test]
    fn elastic_mu_cm_recovers_forward_peak() {
        const MAX_MU: usize = 8;
        let energy_grid = [1.0e3, 2.0e7];
        let n_mu_per_e = [2u32, 2u32];
        let interp_per_e = [ANGLE_INTERP_LINLIN, ANGLE_INTERP_LINLIN];
        // 2-point μ table: μ_0=0, μ_1=1; CDF 0→1 linearly; PDF flat = 1.
        let mut mu_table = vec![0.0f64; 2 * MAX_MU];
        let mut cdf_table = vec![0.0f64; 2 * MAX_MU];
        let mut pdf_table = vec![0.0f64; 2 * MAX_MU];
        for row in 0..2 {
            mu_table[row * MAX_MU] = 0.0;
            mu_table[row * MAX_MU + 1] = 1.0;
            cdf_table[row * MAX_MU] = 0.0;
            cdf_table[row * MAX_MU + 1] = 1.0;
            pdf_table[row * MAX_MU] = 1.0;
            pdf_table[row * MAX_MU + 1] = 1.0;
        }
        let mut state: u64 = 0xCAFE;
        let mut sum = 0.0;
        let n = 4_000usize;
        for _ in 0..n {
            sum += sample_elastic_mu_cm(
                5.0e6,
                0.5,
                &energy_grid,
                &n_mu_per_e,
                &interp_per_e,
                &mu_table,
                &cdf_table,
                &pdf_table,
                &[0u32, MAX_MU as u32],
                &mut state,
            );
        }
        let mean = sum / n as f64;
        assert!(
            (0.45..=0.55).contains(&mean),
            "elastic μ mean {mean} should be ≈ 0.5"
        );
    }

    /// Empty grid → isotropic fallback μ = 1 − 2·ξ₃, no state advance.
    #[test]
    fn elastic_mu_cm_falls_back_to_isotropic_when_no_data() {
        let empty_f: [f64; 0] = [];
        let empty_u: [u32; 0] = [];
        let mut state: u64 = 0xBEEF;
        let before = state;
        let mu = sample_elastic_mu_cm(
            5.0e6, 0.3, &empty_f, &empty_u, &empty_u, &empty_f, &empty_f, &empty_f, &empty_u,
            &mut state,
        );
        assert_eq!(mu, 1.0 - 2.0 * 0.3);
        assert_eq!(state, before, "empty slot must not advance RNG state");
    }
}
