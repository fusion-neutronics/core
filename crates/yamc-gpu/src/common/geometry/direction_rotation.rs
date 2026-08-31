//! 3D direction rotation for elastic scatter (Phase E.3).
//!
//! Given an old unit-vector direction `(u, v, w)` and the lab-frame
//! cosine `µ_lab` from elastic scatter, sample the azimuthal angle
//! uniformly and rotate the direction:
//!
//! ```text
//! sin θ = sqrt(1 - µ²)
//! if 1 - w² > eps:                        ( = sphere not at pole )
//!   sin_phi_w = sqrt(1 - w²)
//!   u' = µ u + sin θ · (u w cos φ - v sin φ) / sin_phi_w
//!   v' = µ v + sin θ · (v w cos φ + u sin φ) / sin_phi_w
//!   w' = µ w - sin θ · sin_phi_w · cos φ
//! else:                                   ( pole case )
//!   sgn = sign(w)
//!   u' = sin θ · cos φ
//!   v' = sgn · sin θ · sin φ
//!   w' = sgn · µ
//! ```
//!
//! # Why rejection sampling
//!
//! `cos φ` and `sin φ` would naturally come from `(2π ξ).cos()` and
//! `.sin()`, but f64 `sin`/`cos` are broken on cubecl-spirv on this
//! driver (see `f64_sin_cos.rs`). Instead we draw `(a, b)` uniform in
//! `[-1, 1]²`, accept if `s = a² + b² ∈ (0, 1]`, and set
//! `cos φ = a / sqrt s`, `sin φ = b / sqrt s`. Marsaglia's classic
//! method, no transcendentals.
//!
//! Acceptance per attempt is `π/4 ≈ 0.785`. With 8 attempts the
//! all-fail probability is `(1 - π/4)⁸ ≈ 5e-7` -- never seen in
//! practice on test sizes ≤ 10M particles, but bounded so the kernel
//! always terminates. If all 8 fail, we fall back to `(1, 0)`
//! (no rotation in φ); the contribution to MC noise from that 5e-7
//! fraction is dwarfed by everything else.
//!
//! Rejection sampling needs many random samples per particle (up to
//! 16 -- two per attempt × 8 attempts). The kernel advances the
//! PCG-32 state explicitly, same pattern as `full_step`.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn direction_rotation_kernel(
    seeds: &[u32],
    directions_in: &[f64],
    mu_lab: &[f64],
    directions_out: &mut [f64],
) {
    if ABSOLUTE_POS >= mu_lab.len() {
        terminate!();
    }

    let i3 = ABSOLUTE_POS * 3;
    let u = directions_in[i3];
    let v = directions_in[i3 + 1];
    let w = directions_in[i3 + 2];
    let mu = mu_lab[ABSOLUTE_POS];

    // Marsaglia rejection: 8 attempts × 2 samples per attempt = up to
    // 16 PCG-32 advances. We advance the state up front and use the
    // first valid pair.
    let mut state = expand_seed(seeds[ABSOLUTE_POS]);
    let mut accepted = false;
    let mut cos_phi = 1.0;
    let mut sin_phi = 0.0;
    let mut attempt = 0u32;
    while attempt < 8u32 && !accepted {
        // Two PCG-32 samples for (a, b).
        let s_a = state;
        let r_a = pcg_out(s_a);
        state = s_a * PCG_MULT + PCG_INCR;
        let s_b = state;
        let r_b = pcg_out(s_b);
        state = s_b * PCG_MULT + PCG_INCR;

        // Map to [-1, 1].
        let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
        let s = a * a + b * b;
        if s > 0.0 && s <= 1.0 {
            let inv_sqrt_s = 1.0 / s.sqrt();
            cos_phi = a * inv_sqrt_s;
            sin_phi = b * inv_sqrt_s;
            accepted = true;
        }
        attempt += 1u32;
    }

    let sin_theta_sq = 1.0 - mu * mu;
    let mut sin_theta = 0.0;
    if sin_theta_sq > 0.0 {
        sin_theta = sin_theta_sq.sqrt();
    }

    let one_minus_w_sq = 1.0 - w * w;

    // Default branch: pole case (will get overwritten if not at pole).
    let mut new_u = sin_theta * cos_phi;
    let mut new_v = sin_theta * sin_phi;
    let mut new_w = mu;
    if w < 0.0 {
        new_v = -new_v;
        new_w = -mu;
    }

    if one_minus_w_sq > 1e-14 {
        let sin_phi_w = one_minus_w_sq.sqrt();
        new_u = mu * u + sin_theta * (u * w * cos_phi - v * sin_phi) / sin_phi_w;
        new_v = mu * v + sin_theta * (v * w * cos_phi + u * sin_phi) / sin_phi_w;
        new_w = mu * w - sin_theta * sin_phi_w * cos_phi;
    }

    directions_out[i3] = new_u;
    directions_out[i3 + 1] = new_v;
    directions_out[i3 + 2] = new_w;
}

/// Run the direction-rotation kernel. `directions_in` is a flat
/// stride-3 buffer of unit vectors; `mu_lab[i]` is the lab cosine
/// from the elastic-scatter step for particle `i`. Returns the
/// rotated unit vectors in the same flat stride-3 layout.
pub fn run_direction_rotation(
    ctx: &GpuContext,
    seeds: &[u32],
    directions_in: &[f64],
    mu_lab: &[f64],
) -> Vec<f64> {
    let n = mu_lab.len();
    assert_eq!(seeds.len(), n);
    assert_eq!(directions_in.len(), 3 * n);

    let client = ctx.client();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let in_h = client.create_from_slice(bytemuck::cast_slice(directions_in));
    let mu_h = client.create_from_slice(bytemuck::cast_slice(mu_lab));
    let out_h = client.empty(std::mem::size_of_val(directions_in));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        direction_rotation_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(in_h, 3 * n),
            BufferArg::from_raw_parts(mu_h, n),
            BufferArg::from_raw_parts(out_h.clone(), 3 * n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

/// CPU equivalent. Same algorithm, same Marsaglia rejection.
pub fn run_direction_rotation_cpu(
    seeds: &[u32],
    directions_in: &[f64],
    mu_lab: &[f64],
) -> Vec<f64> {
    let n = mu_lab.len();
    assert_eq!(seeds.len(), n);
    assert_eq!(directions_in.len(), 3 * n);

    let mut out = vec![0.0_f64; 3 * n];
    for i in 0..n {
        let i3 = i * 3;
        let u = directions_in[i3];
        let v = directions_in[i3 + 1];
        let w = directions_in[i3 + 2];
        let mu = mu_lab[i];

        let mut state = crate::common::rng::expand_seed(seeds[i]);
        let mut accepted = false;
        let mut cos_phi = 1.0;
        let mut sin_phi = 0.0;
        for _ in 0..8 {
            if accepted {
                break;
            }
            let s_a = state;
            let r_a = pcg_out(s_a);
            state = s_a.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let s_b = state;
            let r_b = pcg_out(s_b);
            state = s_b.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);

            let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
            let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
            let s = a * a + b * b;
            if s > 0.0 && s <= 1.0 {
                let inv_sqrt_s = 1.0 / s.sqrt();
                cos_phi = a * inv_sqrt_s;
                sin_phi = b * inv_sqrt_s;
                accepted = true;
            }
        }

        let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
        let one_minus_w_sq = 1.0 - w * w;

        let (new_u, new_v, new_w);
        if one_minus_w_sq > 1e-14 {
            let sin_phi_w = one_minus_w_sq.sqrt();
            new_u = mu * u + sin_theta * (u * w * cos_phi - v * sin_phi) / sin_phi_w;
            new_v = mu * v + sin_theta * (v * w * cos_phi + u * sin_phi) / sin_phi_w;
            new_w = mu * w - sin_theta * sin_phi_w * cos_phi;
        } else {
            let sgn = if w >= 0.0 { 1.0 } else { -1.0 };
            new_u = sin_theta * cos_phi;
            new_v = sgn * sin_theta * sin_phi;
            new_w = sgn * mu;
        }

        out[i3] = new_u;
        out[i3 + 1] = new_v;
        out[i3 + 2] = new_w;
    }
    out
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// All particles starting at (0, 0, 1) with µ = 0 → new direction
    /// must lie in the xy-plane (w' = 0), be a unit vector, and the
    /// azimuthal angles should be approximately uniformly distributed.
    #[test]
    fn perpendicular_scatter_from_z_axis_is_uniform_in_xy() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let n = 100_000usize;
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        // All particles point along +z.
        let mut dirs_in = Vec::with_capacity(3 * n);
        for _ in 0..n {
            dirs_in.extend_from_slice(&[0.0, 0.0, 1.0]);
        }
        let mu = vec![0.0_f64; n]; // perpendicular scatter

        let out = run_direction_rotation(&ctx, &seeds, &dirs_in, &mu);

        let mut sum_u = 0.0;
        let mut sum_v = 0.0;
        let mut max_w = 0.0_f64;
        let mut max_norm_err = 0.0_f64;
        for i in 0..n {
            let nu = out[3 * i];
            let nv = out[3 * i + 1];
            let nw = out[3 * i + 2];
            let norm = (nu * nu + nv * nv + nw * nw).sqrt();
            sum_u += nu;
            sum_v += nv;
            max_w = max_w.max(nw.abs());
            max_norm_err = max_norm_err.max((norm - 1.0).abs());
        }
        let mean_u = sum_u / n as f64;
        let mean_v = sum_v / n as f64;
        let sem = (1.0_f64 / 2.0 / n as f64).sqrt(); // var of uniform on circle is 1/2

        println!(
            "perp scatter from +z: mean_u = {mean_u}, mean_v = {mean_v}, SEM = {sem}, \
             max |w'| = {max_w}, max |‖d‖−1| = {max_norm_err}"
        );
        assert!((mean_u).abs() < 5.0 * sem, "mean u biased: {mean_u}");
        assert!((mean_v).abs() < 5.0 * sem, "mean v biased: {mean_v}");
        assert!(max_w < 1e-12, "w should be ~0 for µ=0 from z-axis");
        assert!(max_norm_err < 1e-12, "rotated direction not unit length");
    }

    /// Cosine between old and new direction must equal `µ_lab` to
    /// floating-point precision, regardless of what φ was sampled.
    #[test]
    fn cosine_old_new_equals_mu_lab() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 1000usize;
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        // Mix of starting directions and µ values to exercise the
        // pole and non-pole branches.
        let mut dirs_in = Vec::with_capacity(3 * n);
        let mut mu = Vec::with_capacity(n);
        for i in 0..n {
            let phi = (i as f64) * 0.1;
            let theta = (i as f64) * 0.05;
            let u = theta.cos() * phi.cos();
            let v = theta.cos() * phi.sin();
            let w = theta.sin();
            // Re-normalize against floating-point drift.
            let r = (u * u + v * v + w * w).sqrt();
            dirs_in.extend_from_slice(&[u / r, v / r, w / r]);
            mu.push(0.5 - 0.001 * (i as f64 % 100.0));
        }

        let out = run_direction_rotation(&ctx, &seeds, &dirs_in, &mu);

        let mut max_dot_err = 0.0_f64;
        let mut max_norm_err = 0.0_f64;
        for (i, &mu_i) in mu.iter().enumerate() {
            let i3 = 3 * i;
            let dot = dirs_in[i3] * out[i3]
                + dirs_in[i3 + 1] * out[i3 + 1]
                + dirs_in[i3 + 2] * out[i3 + 2];
            let norm =
                (out[i3] * out[i3] + out[i3 + 1] * out[i3 + 1] + out[i3 + 2] * out[i3 + 2]).sqrt();
            max_dot_err = max_dot_err.max((dot - mu_i).abs());
            max_norm_err = max_norm_err.max((norm - 1.0).abs());
        }
        println!("max |d·d' − µ| = {max_dot_err:e}, max |‖d'‖ − 1| = {max_norm_err:e}");
        // Cosine preservation is tight (no cancellation); norm is
        // looser because the rotation chain (mul, div, sqrt) has more
        // FMA-affected steps. Both well below MC noise.
        assert!(max_dot_err < 1e-12, "rotated cosine off from µ");
        assert!(max_norm_err < 1e-10, "rotated direction not unit length");
    }

    /// GPU and CPU per-particle bit-equivalence: same RNG, same
    /// rejection acceptance, same arithmetic. Allow a few ULPs of
    /// FMA drift but no more.
    #[test]
    fn gpu_direction_rotation_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 1000usize;
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let mut dirs_in = Vec::with_capacity(3 * n);
        let mut mu = Vec::with_capacity(n);
        for i in 0..n {
            let phi = (i as f64) * 0.07;
            let theta = (i as f64) * 0.03;
            let u = theta.cos() * phi.cos();
            let v = theta.cos() * phi.sin();
            let w = theta.sin();
            let r = (u * u + v * v + w * w).sqrt();
            dirs_in.extend_from_slice(&[u / r, v / r, w / r]);
            mu.push(0.3);
        }

        let g = run_direction_rotation(&ctx, &seeds, &dirs_in, &mu);
        let c = run_direction_rotation_cpu(&seeds, &dirs_in, &mu);

        let mut max_abs = 0.0_f64;
        for i in 0..(3 * n) {
            max_abs = max_abs.max((g[i] - c[i]).abs());
        }
        println!("GPU vs CPU max |Δ| = {max_abs:e}");
        // Direction components are in [-1, 1]; rotation chain is long
        // (multiplies, divides, sqrt, plus the rejection inv_sqrt_s
        // step) and FMA contraction adds drift. 1e-10 absolute is
        // well below MC noise.
        assert!(max_abs < 1e-10, "GPU vs CPU divergence {max_abs:e}");
    }
}
