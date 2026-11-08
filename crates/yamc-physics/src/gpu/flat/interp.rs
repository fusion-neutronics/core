//! Shared linear interpolation of a per-slot parameter on a sorted
//! incident-energy grid, used by the Maxwell, Evaporation, and Watt
//! samplers. Clamps at grid endpoints; in interior, scans for the
//! bracket and linearly interpolates. The seemingly-redundant
//! "fallback to last grid value" path mirrors the kernel's
//! initialisation so a degenerate `de == 0` bracket cannot leave the
//! output uninitialised.

/// Linearly interpolate `param_grid` at `e_in` on the matching
/// `energy_grid`. Both slices must be the same non-empty length.
#[inline]
pub(crate) fn linear_interp_on_grid(e_in: f64, energy_grid: &[f64], param_grid: &[f64]) -> f64 {
    let n = energy_grid.len();
    let e_first = energy_grid[0];
    let e_last = energy_grid[n - 1];
    if e_in <= e_first {
        return param_grid[0];
    }
    if e_in >= e_last {
        return param_grid[n - 1];
    }
    let mut out = param_grid[n - 1];
    for k in 0..n - 1 {
        let e_k = energy_grid[k];
        let e_k1 = energy_grid[k + 1];
        if e_in >= e_k && e_in < e_k1 {
            let de = e_k1 - e_k;
            let f = if de > 0.0 { (e_in - e_k) / de } else { 0.0 };
            let p_k = param_grid[k];
            let p_k1 = param_grid[k + 1];
            out = p_k + f * (p_k1 - p_k);
            break;
        }
    }
    out
}
