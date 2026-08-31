//! Marsaglia azimuthal-direction sampler.
//!
//! Samples `(cos_phi, sin_phi)` for a uniformly-random azimuthal angle via
//! Marsaglia's rejection method: draw `(a, b)` uniformly in the square
//! `[-1, 1]^2`, accept when they land in the unit disk (`0 < a^2 + b^2 <= 1`),
//! and read off `cos_phi = a / sqrt(s)`, `sin_phi = b / sqrt(s)`. This avoids a
//! `sin`/`cos` pair per azimuthal draw. Up to 8 attempts (16 PCG draws); if all
//! are rejected it falls back to `(1, 0)`.
//!
//! Extracted verbatim from the neutron transport kernel's shared
//! angular-sampling block so it is a single unit-tested primitive (and is
//! reusable by the coupled-photon production path). Value-in / struct-out
//! threads the PCG state through the variable number of draws, preserving the
//! RNG stream exactly. Pure integer-PCG + f64 arithmetic apart from one
//! `sqrt` per accepted attempt, so the `#[cube]` kernel matches the
//! [`marsaglia_cos_sin_phi_cpu`] twin to a few ULPs (`gpu_marsaglia_phi_matches_cpu`).

use crate::common::pcg32::{expand_seed, pcg_next};
use crate::common::polyfills::{cos_f64, sin_f64};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Result of [`marsaglia_cos_sin_phi`]: the sampled azimuthal `(cos, sin)`
/// and the advanced 64-bit PCG state.
#[derive(CubeType)]
pub struct MarsagliaPhi {
    pub cos_phi: f64,
    pub sin_phi: f64,
    pub state: u64,
}

/// Sample `(cos_phi, sin_phi)` by Marsaglia rejection (up to 8 attempts, 2 PCG
/// draws each). Falls back to `(1, 0)` if every attempt is rejected. Returns
/// the result plus the advanced `state`.
#[cube]
pub fn marsaglia_cos_sin_phi(state_in: u64) -> MarsagliaPhi {
    let mut state = state_in;
    let mut accepted = false;
    let mut cos_phi = 1.0;
    let mut sin_phi = 0.0;
    let mut rej = 0u32;
    while rej < 8u32 && !accepted {
        let d_a = pcg_next(state);
        let r_a = d_a.rand;
        state = d_a.state;
        let d_b = pcg_next(state);
        let r_b = d_b.rand;
        state = d_b.state;
        let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let ss = a * a + b * b;
        if ss > 0.0 && ss <= 1.0 {
            let inv_sqrt_s = 1.0 / ss.sqrt();
            cos_phi = a * inv_sqrt_s;
            sin_phi = b * inv_sqrt_s;
            accepted = true;
        }
        rej += 1u32;
    }
    MarsagliaPhi {
        cos_phi,
        sin_phi,
        state,
    }
}

/// Sample `(cos_phi, sin_phi)` for a uniform lab azimuth via ONE PCG draw:
/// `phi = TAU * xi`, then `cos_f64(phi)` / `sin_f64(phi)`. This matches the
/// production CPU's azimuth draw schedule (`scatter.rs` ->
/// `rotate_direction_fast`, a single `TAU * next_xi` uniform), so the
/// per-history matched stream (#40) stays in lockstep past the first collision
/// (issue #136 / #111). It replaces the variable-draw [`marsaglia_cos_sin_phi`]
/// on the transport azimuth. The `#[cube]` kernel calls this (cos/sin via the
/// `cos_f64`/`sin_f64` polyfills); the CPU twin in `transport/shared.rs` inlines
/// std libm `cos`/`sin` so it matches the production CPU bit-for-bit (kernel vs
/// twin then agree within ulps, exactly as ln/exp already do).
#[cube]
pub fn azimuth_cos_sin_phi(state_in: u64) -> MarsagliaPhi {
    let d = pcg_next(state_in);
    let xi = (d.rand as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let phi = std::f64::consts::TAU * xi;
    MarsagliaPhi {
        cos_phi: cos_f64(phi),
        sin_phi: sin_f64(phi),
        state: d.state,
    }
}

/// CPU twin of [`marsaglia_cos_sin_phi`] (same draws, same rejection, same
/// 8-attempt cap and `(1, 0)` fallback). Returns `(cos_phi, sin_phi, state)`.
pub fn marsaglia_cos_sin_phi_cpu(state_in: u64) -> (f64, f64, u64) {
    let mut state = state_in;
    let mut accepted = false;
    let mut cos_phi = 1.0_f64;
    let mut sin_phi = 0.0_f64;
    let mut rej = 0u32;
    while rej < 8 && !accepted {
        let (r_a, s1) = crate::common::pcg32::pcg_next_cpu(state);
        state = s1;
        let (r_b, s2) = crate::common::pcg32::pcg_next_cpu(state);
        state = s2;
        let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let ss = a * a + b * b;
        if ss > 0.0 && ss <= 1.0 {
            let inv_sqrt_s = 1.0 / ss.sqrt();
            cos_phi = a * inv_sqrt_s;
            sin_phi = b * inv_sqrt_s;
            accepted = true;
        }
        rej += 1;
    }
    (cos_phi, sin_phi, state)
}

/// Validation kernel: one Marsaglia draw per thread from its expanded seed.
#[cube(launch_unchecked)]
fn marsaglia_phi_kernel(
    seeds: &[u32],
    out_cos: &mut [f64],
    out_sin: &mut [f64],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let m = marsaglia_cos_sin_phi(expand_seed(seeds[ABSOLUTE_POS]));
    out_cos[ABSOLUTE_POS] = m.cos_phi;
    out_sin[ABSOLUTE_POS] = m.sin_phi;
    out_state[ABSOLUTE_POS] = m.state;
}

/// Run [`marsaglia_cos_sin_phi`] on the GPU for each seed; returns
/// `(cos_phi, sin_phi, advanced_state)` per seed. For test/validation use.
// Output buffers are sized by element count (one per seed), not by the seeds
// slice's byte size, so `size_of_val(seeds)` would be wrong here.
#[allow(clippy::manual_slice_size_calculation)]
pub fn run_marsaglia_phi(ctx: &GpuContext, seeds: &[u32]) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    let client = ctx.client();
    let n = seeds.len();
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let cos_h = client.empty(n * core::mem::size_of::<f64>());
    let sin_h = client.empty(n * core::mem::size_of::<f64>());
    let st_h = client.empty(n * core::mem::size_of::<u64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        marsaglia_phi_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(seed_h, n),
            BufferArg::from_raw_parts(cos_h.clone(), n),
            BufferArg::from_raw_parts(sin_h.clone(), n),
            BufferArg::from_raw_parts(st_h.clone(), n),
        );
    }
    let cos = bytemuck::cast_slice::<u8, f64>(&client.read_one(cos_h).unwrap()).to_vec();
    let sin = bytemuck::cast_slice::<u8, f64>(&client.read_one(sin_h).unwrap()).to_vec();
    let st = bytemuck::cast_slice::<u8, u64>(&client.read_one(st_h).unwrap()).to_vec();
    (cos, sin, st)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// GPU Marsaglia sampling must match the CPU twin: the advanced PCG state
    /// is exact `u64` integer math (the variable draw count must agree), and
    /// `(cos_phi, sin_phi)` agree to a few ULPs (one `sqrt`). Also checks the
    /// result is a unit vector. Sweeps many seeds.
    #[test]
    fn gpu_marsaglia_phi_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let seeds: Vec<u32> = (0..1024)
            .map(|i: u32| i.wrapping_mul(2_654_435_761).wrapping_add(0x1234_5678))
            .chain([0, 1, 7, 42, u32::MAX])
            .collect();

        let (gpu_cos, gpu_sin, gpu_state) = run_marsaglia_phi(&ctx, &seeds);
        assert_eq!(gpu_cos.len(), seeds.len());

        for (i, &s) in seeds.iter().enumerate() {
            // Mirror the kernel: expand the 32-bit seed into the 64-bit PCG
            // state before running the twin.
            let (cpu_cos, cpu_sin, cpu_state) =
                marsaglia_cos_sin_phi_cpu(crate::common::rng::expand_seed(s));
            assert_eq!(
                gpu_state[i], cpu_state,
                "seed {s}: advanced state differs (gpu {} cpu {cpu_state}) -- draw count diverged",
                gpu_state[i]
            );
            assert!(
                (gpu_cos[i] - cpu_cos).abs() <= 1e-12 && (gpu_sin[i] - cpu_sin).abs() <= 1e-12,
                "seed {s}: (cos,sin) differ gpu ({},{}) cpu ({cpu_cos},{cpu_sin})",
                gpu_cos[i],
                gpu_sin[i]
            );
            let norm = gpu_cos[i] * gpu_cos[i] + gpu_sin[i] * gpu_sin[i];
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "seed {s}: (cos,sin) not a unit vector (norm^2 {norm})"
            );
        }
        println!(
            "marsaglia phi: {} seeds, GPU state bit-exact, (cos,sin) within 1e-12",
            seeds.len()
        );
    }
}
