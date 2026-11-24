//! Watt-spectrum E_out sampler for fission outgoing energy:
//! `p(E) ∝ exp(-E/a) · sinh(√(b·E))`, capped at `E_in - U`.
//!
//! Four RNG draws per rejection iteration, up to 32 iterations. Bit-
//! identical to the cubecl kernel's Watt branch.

use super::interp::linear_interp_on_grid;
use yamc_rng::next_xi;

/// Sample an outgoing energy from a Watt distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid of length `n`.
/// * `a_grid` -- Watt parameter `a(E_in)`, length `n`.
/// * `b_grid` -- Watt parameter `b(E_in)`, length `n`.
/// * `u` -- restriction energy.
/// * `state` -- inline PCG RNG state, advanced 4× per iteration.
///
/// Returns `Some(e_out)` on acceptance, `None` when the slot is empty,
/// parameters are degenerate, or rejection exhausts.
pub fn sample_watt_inelastic(
    e_in: f64,
    energy_grid: &[f64],
    a_grid: &[f64],
    b_grid: &[f64],
    u: f64,
    state: &mut u64,
) -> Option<f64> {
    let n = energy_grid.len();
    if n == 0 {
        return None;
    }
    let a_val = linear_interp_on_grid(e_in, energy_grid, a_grid);
    let b_val = linear_interp_on_grid(e_in, energy_grid, b_grid);
    if a_val <= 0.0 || e_in <= u {
        return None;
    }
    let cap_e = e_in - u;
    let a2b = a_val * a_val * b_val;
    for _ in 0..32 {
        let xi1 = next_xi(state);
        let xi2 = next_xi(state);
        let xi3 = next_xi(state);
        let xi4 = next_xi(state);
        let cos_xi3 = (std::f64::consts::FRAC_PI_2 * xi3).cos();
        let mut w_m = 0.0_f64;
        if xi1 > 0.0 && xi2 > 0.0 {
            w_m = -a_val * (xi1.ln() + xi2.ln() * cos_xi3 * cos_xi3);
        }
        let u_xi = 2.0 * xi4 - 1.0;
        let prod = a2b * w_m;
        if w_m > 0.0 && prod >= 0.0 {
            let e_out = w_m + 0.25 * a2b + u_xi * prod.sqrt();
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

    /// Watt rejection sampler smoke test.
    /// a = 1 MeV, b = 2.249e-6 / eV, u = 0 over a 2-point flat grid;
    /// 4k samples at E_in = 14.06 MeV should accept ≥90% and produce
    /// an empirical mean within 5% of the analytic Watt mean
    /// `<E> = (3/2)·a + (a²·b)/4`.
    #[test]
    fn watt_mean_matches_theory() {
        let energy_grid = [1.0e3, 2.0e7];
        let a_val = 1.0e6;
        let b_val = 2.249e-6;
        let a_grid = [a_val, a_val];
        let b_grid = [b_val, b_val];
        let u = 0.0;
        let e_in = 14.06e6;
        let mut state: u64 = 0xDEC0DE;
        let n = 4_000usize;
        let mut sum = 0.0;
        let mut ok = 0usize;
        for _ in 0..n {
            if let Some(e_out) =
                sample_watt_inelastic(e_in, &energy_grid, &a_grid, &b_grid, u, &mut state)
            {
                sum += e_out;
                ok += 1;
            }
        }
        assert!(
            ok as f64 / n as f64 > 0.90,
            "Watt rejection acceptance {ok} of {n} below 90% -- sampler may have a tightness bug"
        );
        let mean = sum / ok as f64;
        let theory = 1.5 * a_val + 0.25 * a_val * a_val * b_val;
        let rel_err = ((mean - theory) / theory).abs();
        assert!(
            rel_err < 0.05,
            "Watt mean {mean:.4e} differs from theory {theory:.4e} by {rel_err:.3} (>5%)"
        );
    }

    /// Must return `None` when the slot has no Watt data, when `a`
    /// is non-positive, or when E_in ≤ U.
    #[test]
    fn watt_returns_none_for_invalid_inputs() {
        let mut state: u64 = 1;
        let empty: [f64; 0] = [];
        assert!(sample_watt_inelastic(14.06e6, &empty, &empty, &empty, 0.0, &mut state).is_none());

        let grid = [1.0, 1.0e7];
        let a = [1.0e6, 1.0e6];
        let b = [2.249e-6, 2.249e-6];
        assert!(sample_watt_inelastic(1.0e6, &grid, &a, &b, 1.0e7, &mut state).is_none());
    }
}
