//! Tabulated equiprobable E_out sampler.
//!
//! Reads from the same per-slot `eout_*` buffers as the
//! ContinuousTabular path -- the data shape is identical, but the
//! CDF column is zero-padded on this branch and one uniform draw
//! picks an outgoing-energy bin directly. No rejection loop.

use yamc_rng::next_xi;

/// Sample an outgoing center-of-mass energy from a tabulated
/// equiprobable distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid of length
///   `n_e`. Used to pick a nearest-lower index `i`, matching the
///   `find_energy_index` semantics in `yamc-nuclide::reaction_product`.
/// * `n_x_per_i` -- number of equiprobable bins at each incident-energy
///   point, length `n_e`.
/// * `x_table` -- flat table of outgoing energies packed
///   variable-length (issue #104): row `i` occupies `x_offset[i] ..
///   x_offset[i] + n_x_per_i[i]`, stored back to back with no padding.
/// * `x_offset` -- start index of each incident-energy row in the flat
///   `x_table`, length `n_e`.
/// * `state` -- inline PCG RNG state, advanced once.
///
/// Returns `Some(e_out)` when the table has data and the sampled
/// entry is positive; `None` when the slot is empty or the sampled
/// entry is zero (signalling no data at this incident-energy point).
pub fn sample_tabulated_equiprobable(
    e_in: f64,
    energy_grid: &[f64],
    n_x_per_i: &[u32],
    x_table: &[f64],
    x_offset: &[u32],
    state: &mut u64,
) -> Option<f64> {
    let n_e = energy_grid.len();
    if n_e == 0 {
        return None;
    }
    let e_first = energy_grid[0];
    let e_last = energy_grid[n_e - 1];
    let i = if e_in >= e_last {
        n_e - 1
    } else if e_in <= e_first {
        0
    } else {
        let mut idx = 0usize;
        for k in 0..n_e - 1 {
            let e_k = energy_grid[k];
            let e_k1 = energy_grid[k + 1];
            if e_in >= e_k && e_in < e_k1 {
                idx = k;
                break;
            }
        }
        idx
    };
    let n_x = n_x_per_i[i] as usize;
    if n_x == 0 {
        return None;
    }
    let xi = next_xi(state);
    let mut bin = (xi * n_x as f64) as usize;
    if bin >= n_x {
        bin = n_x - 1;
    }
    let e_out = x_table[x_offset[i] as usize + bin];
    if e_out > 0.0 {
        Some(e_out)
    } else {
        None
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: 10 equiprobable bins spanning `[1e6, 10e6]` eV at a
    /// constant incident-energy axis (2 grid points, same bins for
    /// both); 4k samples should produce an empirical mean within
    /// 3% of the midpoint `(1e6 + 10e6) / 2 = 5.5e6`.
    #[test]
    fn tabulated_equiprobable_mean_matches_bin_midpoint() {
        const MAX_X: usize = 16;
        const N_BINS: usize = 10;
        let energy_grid = [1.0e3, 2.0e7];
        let n_x_per_i = [N_BINS as u32, N_BINS as u32];
        // Build the 2-row × MAX_X table; bins 0..10 carry the
        // equiprobable values, the rest are zero (unused).
        let mut x_table = vec![0.0f64; 2 * MAX_X];
        for row in 0..2 {
            for bin in 0..N_BINS {
                let frac = (bin as f64 + 0.5) / N_BINS as f64;
                x_table[row * MAX_X + bin] = 1.0e6 + frac * 9.0e6;
            }
        }
        let e_in = 14.06e6_f64;
        let x_offset = [0u32, MAX_X as u32];
        let mut state: u64 = 0xFACEFEED;
        let n = 4_000usize;
        let mut sum = 0.0;
        let mut ok = 0usize;
        for _ in 0..n {
            if let Some(e_out) = sample_tabulated_equiprobable(
                e_in,
                &energy_grid,
                &n_x_per_i,
                &x_table,
                &x_offset,
                &mut state,
            ) {
                sum += e_out;
                ok += 1;
            }
        }
        assert!(ok as f64 / n as f64 > 0.95, "acceptance {ok}/{n} too low");
        let mean = sum / ok as f64;
        let theory = 5.5e6;
        let rel_err = ((mean - theory) / theory).abs();
        assert!(
            rel_err < 0.03,
            "mean {mean:.4e} differs from theory {theory:.4e} by {rel_err:.3} (>3%)"
        );
    }

    /// Empty energy grid → `None`, no state advance.
    #[test]
    fn tabulated_equiprobable_returns_none_for_empty_slot() {
        let empty_e: [f64; 0] = [];
        let empty_n: [u32; 0] = [];
        let empty_x: [f64; 0] = [];
        let empty_o: [u32; 0] = [];
        let mut state: u64 = 1;
        let before = state;
        let r = sample_tabulated_equiprobable(
            14.06e6, &empty_e, &empty_n, &empty_x, &empty_o, &mut state,
        );
        assert!(r.is_none());
        assert_eq!(state, before, "empty slot must not advance RNG state");
    }
}
