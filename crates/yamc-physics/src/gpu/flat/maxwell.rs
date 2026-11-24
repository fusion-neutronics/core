//! Maxwell E_out sampler (fission spectrum form
//! `p(E) ∝ √E · exp(-E/θ(E_in))`, capped at `E_in - U`).
//!
//! Three RNG draws per rejection iteration (the third feeds
//! `cos(π/2 · ξ₃)`), up to 32 iterations. Bit-identical to the cubecl
//! kernel's Maxwell branch.

use super::interp::linear_interp_on_grid;
use yamc_rng::next_xi;

/// Sample an outgoing center-of-mass energy from a Maxwell
/// distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid of length `n`.
/// * `theta_grid` -- temperature parameter θ at each grid point, length `n`.
/// * `u` -- restriction energy.
/// * `state` -- inline PCG RNG state, advanced 3× per iteration.
///
/// Returns `Some(e_out)` on acceptance, `None` when the slot is empty,
/// parameters are degenerate, or rejection exhausts.
pub fn sample_maxwell(
    e_in: f64,
    energy_grid: &[f64],
    theta_grid: &[f64],
    u: f64,
    state: &mut u64,
) -> Option<f64> {
    let n = energy_grid.len();
    if n == 0 {
        return None;
    }
    let theta_val = linear_interp_on_grid(e_in, energy_grid, theta_grid);
    if theta_val <= 0.0 || e_in <= u {
        return None;
    }
    let cap_e = e_in - u;
    for _ in 0..32 {
        let xi1 = next_xi(state);
        let xi2 = next_xi(state);
        let xi3 = next_xi(state);
        let cos_xi3 = (std::f64::consts::FRAC_PI_2 * xi3).cos();
        if xi1 > 0.0 && xi2 > 0.0 {
            let e_out = -theta_val * (xi1.ln() + xi2.ln() * cos_xi3 * cos_xi3);
            if e_out > 0.0 && e_out <= cap_e {
                return Some(e_out);
            }
        }
    }
    None
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Maxwell rejection sampler smoke test.
    /// θ = 1 MeV, u = 0 over a 2-point flat grid; 4k samples at
    /// E_in = 14.06 MeV should accept ≥95% and produce an empirical
    /// mean within 5% of the analytic Maxwell mean `<E> = (3/2)·θ`.
    /// Catches gross arithmetic / RNG bugs without needing a nuclide
    /// that carries `EnergyDistribution::Maxwell`.
    #[test]
    fn maxwell_mean_matches_theory() {
        let energy_grid = [1.0e3, 2.0e7];
        let theta_val = 1.0e6;
        let theta_grid = [theta_val, theta_val];
        let u = 0.0;
        let e_in = 14.06e6;
        let mut state: u64 = 0xC0FFEE;
        let n = 4_000usize;
        let mut sum = 0.0;
        let mut ok = 0usize;
        for _ in 0..n {
            if let Some(e_out) = sample_maxwell(e_in, &energy_grid, &theta_grid, u, &mut state) {
                sum += e_out;
                ok += 1;
            }
        }
        assert!(
            ok as f64 / n as f64 > 0.95,
            "Maxwell rejection acceptance {ok} of {n} below 95% -- sampler may have a tightness bug"
        );
        let mean = sum / ok as f64;
        let theory = 1.5 * theta_val;
        let rel_err = ((mean - theory) / theory).abs();
        assert!(
            rel_err < 0.05,
            "Maxwell mean {mean:.4e} differs from theory {theory:.4e} by {rel_err:.3} (>5%)"
        );
    }

    /// Must return `None` when the slot has no Maxwell data (empty
    /// grid) or when E_in ≤ U (no acceptable region).
    #[test]
    fn maxwell_returns_none_for_invalid_inputs() {
        let mut state: u64 = 1;
        // Empty grid → None.
        let empty: [f64; 0] = [];
        assert!(sample_maxwell(14.06e6, &empty, &empty, 0.0, &mut state).is_none());

        // E_in below restriction energy → None.
        let grid = [1.0, 1.0e7];
        let theta = [1.0e6, 1.0e6];
        assert!(sample_maxwell(1.0e6, &grid, &theta, 1.0e7, &mut state).is_none());
    }
}
