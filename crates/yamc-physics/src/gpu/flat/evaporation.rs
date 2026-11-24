//! Evaporation E_out sampler (fission spectrum form
//! `p(E) ∝ E · exp(-E/θ(E_in))`, capped at `E_in - U`).
//!
//! Two RNG draws per rejection iteration, up to 32 iterations. Bit-
//! identical to the cubecl kernel's Evaporation branch.

use super::interp::linear_interp_on_grid;
use yamc_rng::next_xi;

/// Sample an outgoing center-of-mass energy from an Evaporation
/// distribution.
///
/// # Arguments
/// * `e_in` -- incident energy in the laboratory frame.
/// * `energy_grid` -- sorted per-slot incident-energy grid of length `n`.
/// * `theta_grid` -- temperature parameter θ at each incident-energy
///   point, length `n`.
/// * `u` -- restriction energy (subtracted from `e_in` to cap E_out).
/// * `state` -- inline PCG RNG state, advanced 2× per iteration.
///
/// Returns `Some(e_out)` when the rejection loop accepts within 32
/// iterations, `None` when the slot is empty (`n == 0`), parameters
/// are degenerate (`θ ≤ 0`, `e_in ≤ U`), or rejection exhausts.
pub fn sample_evaporation(
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
    let y = (e_in - u) / theta_val;
    let v_e = 1.0 - (-y).exp();
    for _ in 0..32 {
        let xi1 = next_xi(state);
        let xi2 = next_xi(state);
        let arg_a = 1.0 - v_e * xi1;
        let arg_b = 1.0 - v_e * xi2;
        if arg_a > 0.0 && arg_b > 0.0 {
            let x_e = -arg_a.ln() - arg_b.ln();
            if x_e <= y {
                let e_out = x_e * theta_val;
                if e_out > 0.0 {
                    return Some(e_out);
                }
                return None;
            }
        }
    }
    None
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaporation rejection sampler smoke test.
    /// θ = 1 MeV, u = 0 over a 2-point flat grid; with E_in = 14.06 MeV
    /// the analytic Evaporation mean is `<E> = 2·θ` (ignoring the cap
    /// since cap = E_in − u = 14 θ ≫ 2 θ).
    #[test]
    fn evaporation_mean_matches_theory() {
        let energy_grid = [1.0e3, 2.0e7];
        let theta_val = 1.0e6;
        let theta_grid = [theta_val, theta_val];
        let u = 0.0;
        let e_in = 14.06e6;
        let mut state: u64 = 0xEEEEE;
        let n = 4_000usize;
        let mut sum = 0.0;
        let mut ok = 0usize;
        for _ in 0..n {
            if let Some(e_out) = sample_evaporation(e_in, &energy_grid, &theta_grid, u, &mut state)
            {
                sum += e_out;
                ok += 1;
            }
        }
        assert!(
            ok as f64 / n as f64 > 0.90,
            "Evaporation rejection acceptance {ok} of {n} below 90% -- sampler may have a tightness bug"
        );
        let mean = sum / ok as f64;
        let theory = 2.0 * theta_val;
        let rel_err = ((mean - theory) / theory).abs();
        assert!(
            rel_err < 0.05,
            "Evaporation mean {mean:.4e} differs from theory {theory:.4e} by {rel_err:.3} (>5%)"
        );
    }

    /// Must return `None` when the slot has no Evaporation data or
    /// when E_in ≤ U.
    #[test]
    fn evaporation_returns_none_for_invalid_inputs() {
        let mut state: u64 = 1;
        let empty: [f64; 0] = [];
        assert!(sample_evaporation(14.06e6, &empty, &empty, 0.0, &mut state).is_none());

        let grid = [1.0, 1.0e7];
        let theta = [1.0e6, 1.0e6];
        assert!(sample_evaporation(1.0e6, &grid, &theta, 1.0e7, &mut state).is_none());
    }
}
