//! Shared stochastic incident-energy bracket pick.
//!
//! Several distribution samplers (the elastic and inelastic angular tables,
//! the correlated angle-energy law, the Kalbach-Mann law, and the fission
//! outgoing-energy law) bracket the incident energy between two tabulated grid
//! points `i` and `i + 1`, with interpolation fraction `r in [0, 1]`, then
//! stochastically pick ONE bracketing distribution: draw a uniform `xi` and
//! take the upper bracket when `r > xi`, otherwise the lower. This mirrors the
//! CPU samplers' `if r_interp > rng.random() { i + 1 } else { i }` (and the
//! ENDF "use the nearer grid point with probability proportional to the
//! interpolation fraction" convention).
//!
//! Extracted verbatim from the neutron transport kernel's five inline
//! occurrences (each was `draw_uniform; if r > xi && bin + 1 < n { bin + 1 }`).
//! The continuous-tabular eout sampler keeps its own copy because it suppresses
//! the pick under an outer histogram_interp flag; this helper is the bare,
//! always-on form. The draw happens INSIDE the helper, matching every call
//! site's RNG draw order. Pure integer-PCG + one f64 compare, so the `#[cube]`
//! kernel matches the [`pick_energy_bracket_cpu`] twin bit-for-bit
//! (`gpu_pick_energy_bracket_matches_cpu`).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use cubecl::prelude::*;

/// Result of [`pick_energy_bracket`]: the chosen grid index (`i` or `i + 1`)
/// and the advanced PCG-32 state.
#[derive(CubeType)]
pub struct BracketPick {
    pub bin: u32,
    pub state: u64,
}

/// Draw one uniform and pick the incident-energy bracket. `i` is the lower
/// bracket index, `n` the number of grid points, `r` the interpolation
/// fraction. Returns `i + 1` when `r > xi` and `i + 1 < n`, else `i`, plus the
/// advanced state.
#[cube]
pub fn pick_energy_bracket(r: f64, i: u32, n: u32, state_in: u64) -> BracketPick {
    let d = crate::common::pcg32::draw_uniform(state_in);
    let mut bin = i;
    if r > d.xi && bin + 1u32 < n {
        bin = i + 1u32;
    }
    BracketPick {
        bin,
        state: d.state,
    }
}

/// CPU twin of [`pick_energy_bracket`] (wrapping 64-bit PCG arithmetic). Returns
/// `(bin, state)`. Bit-for-bit identical to the `#[cube]` form (integer PCG
/// plus one f64 compare; no transcendentals).
pub fn pick_energy_bracket_cpu(r: f64, i: u32, n: u32, state_in: u64) -> (u32, u64) {
    let s = state_in;
    let rand = pcg_out(s);
    let state = s.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    let xi = (rand as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let mut bin = i;
    if r > xi && bin + 1 < n {
        bin = i + 1;
    }
    (bin, state)
}

/// Test/validation kernel: one bracket pick per thread.
#[cube(launch_unchecked)]
fn bracket_test_kernel(
    r: &[f64],
    i: &[u32],
    n: &[u32],
    seeds: &[u32],
    out_bin: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_bin.len() {
        terminate!();
    }
    let p = pick_energy_bracket(
        r[ABSOLUTE_POS],
        i[ABSOLUTE_POS],
        n[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_bin[ABSOLUTE_POS] = p.bin;
    out_state[ABSOLUTE_POS] = p.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    /// GPU bracket pick must match the CPU twin bit-for-bit: the PCG state
    /// advances by exact u64 integer math and the chosen bin is a pure
    /// comparison. Sweeps `r` across `[0, 1]` (incl. endpoints), `i` at the
    /// top boundary (`i + 1 == n`, where the upper pick must be suppressed),
    /// and many seeds.
    #[test]
    fn gpu_pick_energy_bracket_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // (r, i, n): include r=0 (never upper), r=1 (almost always upper),
        // and i+1==n (upper suppressed by the bound).
        let cases: &[(f64, u32, u32)] = &[
            (0.0, 0, 4),
            (0.25, 0, 4),
            (0.5, 1, 4),
            (0.75, 2, 4),
            (1.0, 0, 4),
            (0.9, 3, 4), // i + 1 == n -> never upper
            (0.5, 0, 1), // single grid point -> never upper
        ];

        let mut r = Vec::new();
        let mut iv = Vec::new();
        let mut nv = Vec::new();
        let mut seeds = Vec::new();
        let n_seeds = 300u32;
        for &(rr, ii, nn) in cases {
            for s in 0..n_seeds {
                r.push(rr);
                iv.push(ii);
                nv.push(nn);
                seeds.push((ii.wrapping_mul(7919) + s).wrapping_mul(2_654_435_761));
            }
        }
        let cnt = seeds.len();

        let mut cpu_bin = vec![0u32; cnt];
        let mut cpu_state = vec![0u64; cnt];
        for k in 0..cnt {
            let (b, st) = pick_energy_bracket_cpu(
                r[k],
                iv[k],
                nv[k],
                crate::common::rng::expand_seed(seeds[k]),
            );
            cpu_bin[k] = b;
            cpu_state[k] = st;
        }

        let client = ctx.client();
        let r_h = client.create_from_slice(bytemuck::cast_slice(&r));
        let i_h = client.create_from_slice(bytemuck::cast_slice(&iv));
        let n_h = client.create_from_slice(bytemuck::cast_slice(&nv));
        let seed_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let bin_h = client.empty(cnt * core::mem::size_of::<u32>());
        let st_h = client.empty(cnt * core::mem::size_of::<u64>());

        const WG: u32 = 64;
        let groups = (cnt as u32).div_ceil(WG);
        unsafe {
            bracket_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(r_h, r.len()),
                BufferArg::from_raw_parts(i_h, iv.len()),
                BufferArg::from_raw_parts(n_h, nv.len()),
                BufferArg::from_raw_parts(seed_h, seeds.len()),
                BufferArg::from_raw_parts(bin_h.clone(), cnt),
                BufferArg::from_raw_parts(st_h.clone(), cnt),
            );
        }
        let gpu_bin = bytemuck::cast_slice::<u8, u32>(&client.read_one(bin_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(st_h).unwrap()).to_vec();

        for k in 0..cnt {
            assert_eq!(
                gpu_state[k], cpu_state[k],
                "sample {k}: PCG state diverged (gpu {} cpu {}, r {}, i {}, n {}, seed {})",
                gpu_state[k], cpu_state[k], r[k], iv[k], nv[k], seeds[k]
            );
            assert_eq!(
                gpu_bin[k], cpu_bin[k],
                "sample {k}: bin diverged (gpu {} cpu {}, r {}, i {}, n {}, seed {})",
                gpu_bin[k], cpu_bin[k], r[k], iv[k], nv[k], seeds[k]
            );
            // Upper pick must respect the grid bound.
            assert!(gpu_bin[k] == iv[k] || gpu_bin[k] == iv[k] + 1);
            if iv[k] + 1 >= nv[k] {
                assert_eq!(gpu_bin[k], iv[k], "sample {k}: upper pick at grid boundary");
            }
        }
        println!(
            "pick_energy_bracket: {} samples, GPU==CPU bit-exact (bin + state)",
            cnt
        );
    }
}
