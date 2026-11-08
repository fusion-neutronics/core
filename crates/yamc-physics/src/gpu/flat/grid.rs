//! Shared helpers for incident-energy grid lookups inside `flat/` samplers.
//!
//! Mirrors the cubecl `#[cube]` neutron kernel's grid-bracket code path:
//! locate the `[grid[i], grid[i+1]]` bracket that contains `e_in`, and
//! return the linear-interpolation fraction `r = (e_in - grid[i]) /
//! (grid[i+1] - grid[i])`. Above the last grid point clamps to
//! `(n-2, 1.0)`; below the first clamps to `(0, 0.0)`. On a strictly
//! monotonic grid (the ENDF invariant) at most one bracket matches, so
//! the kernel's no-`break` scan and this `break` variant produce identical
//! `(i, r)` pairs.

/// Bracket `e_in` on a strictly monotonic incident-energy grid.
///
/// Returns `(i, r)` where `grid[i] <= e_in < grid[i+1]` and
/// `r = (e_in - grid[i]) / (grid[i+1] - grid[i])`, with the clamps
/// described above for out-of-range inputs.
#[inline]
pub(crate) fn locate_bracket(e_in: f64, grid: &[f64]) -> (usize, f64) {
    let n = grid.len();
    let e_first = grid[0];
    let e_last = grid[n - 1];
    if e_in >= e_last {
        return (if n > 1 { n - 2 } else { 0 }, 1.0);
    }
    if e_in <= e_first {
        return (0, 0.0);
    }
    let mut i = 0usize;
    let mut r = 0.0;
    for k in 0..n - 1 {
        let e_k = grid[k];
        let e_k1 = grid[k + 1];
        if e_in >= e_k && e_in < e_k1 {
            i = k;
            let de = e_k1 - e_k;
            if de > 0.0 {
                r = (e_in - e_k) / de;
            }
            break;
        }
    }
    (i, r)
}
