//! Elastic scatter kinematics on the GPU (Phase E.2).
//!
//! For an elastic collision of a neutron at energy `E` with a target
//! nucleus of mass `A` (units of neutron masses), the standard MC
//! treatment samples the center-of-mass scattering cosine `µ_cm`
//! isotropically in `[-1, 1]` and transforms back to the lab frame:
//!
//! ```text
//! α     = ((A - 1) / (A + 1))²
//! E'    = E * ((1 + α) + (1 - α) * µ_cm) / 2
//! µ_lab = (1 + A * µ_cm) / sqrt(A² + 2A * µ_cm + 1)
//! ```
//!
//! Two outputs per particle: the new energy `E'` and the lab cosine
//! `µ_lab`. The full 3D direction update (rotate the old direction by
//! polar angle `arccos(µ_lab)` and azimuthal angle `2π * ξ₂`) needs
//! `sin`/`cos`, which we haven't yet validated on cubecl-spirv (the
//! same Mesa NIR gap that broke `exp`/`ln` likely affects `fsin`/
//! `fcos` too). That step will land in E.3 once the polyfill is in.
//!
//! # Why this is a useful step on its own
//!
//! - The energy update is what *thermalises* a neutron in a moderator.
//!   It's the dominant physics in shielding/dose calculations.
//! - The lab cosine is what feeds the next-step direction rotation, so
//!   computing it now leaves only the `sin`/`cos`-dependent
//!   3D-rotation step for E.3.
//! - The math chain (square, divide, sqrt, multiply-add) exercises a
//!   different mix of GPU-friendly ops than any of the previous
//!   kernels, sharpening our picture of consumer-iGPU f64 cost.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Per-particle: PCG-32 → uniform `ξ` → `µ_cm` → lab transform.
/// Writes two outputs in parallel arrays: `out_energy[tid] = E'` and
/// `out_mu_lab[tid] = µ_lab`.
#[cube(launch_unchecked)]
fn elastic_scatter_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    target_masses: &[f64],
    out_energies: &mut [f64],
    out_mu_lab: &mut [f64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    // Inline PCG-32 for the CM scattering cosine.
    let oldstate = expand_seed(seeds[ABSOLUTE_POS]);
    let _next_state = oldstate * PCG_MULT + PCG_INCR;
    let r_u32 = pcg_out(oldstate);
    let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let mu_cm = 1.0 - 2.0 * xi;

    let e_in = energies_in[ABSOLUTE_POS];
    let a = target_masses[ABSOLUTE_POS];

    // E' / E for elastic scatter: uniform in [α, 1] when µ_cm uniform
    // in [-1, 1]. Direct form avoids precomputing α.
    let one_plus_a = a + 1.0;
    let one_plus_a_sq = one_plus_a * one_plus_a;
    let denom = one_plus_a_sq;
    let numer = a * a + 2.0 * a * mu_cm + 1.0;
    out_energies[ABSOLUTE_POS] = e_in * numer / denom;

    // Lab cosine µ_lab. denom_root is sqrt of the same numer above
    // (which is always positive for finite `a` and µ_cm in [-1, 1]).
    let denom_root = numer.sqrt();
    out_mu_lab[ABSOLUTE_POS] = (1.0 + a * mu_cm) / denom_root;
}

/// Run the elastic-scatter kernel. Each `seeds[i]`, `energies_in[i]`,
/// `target_masses[i]` make one independent particle. Returns
/// `(new_energies, lab_cosines)` in input order.
pub fn run_elastic_scatter(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    target_masses: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(seeds.len(), energies_in.len());
    assert_eq!(seeds.len(), target_masses.len());

    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_handle = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let masses_handle = client.create_from_slice(bytemuck::cast_slice(target_masses));
    let out_energies_handle = client.empty(std::mem::size_of_val(energies_in));
    let out_mu_handle = client.empty(std::mem::size_of_val(energies_in));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        elastic_scatter_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(energies_handle, n),
            BufferArg::from_raw_parts(masses_handle, n),
            BufferArg::from_raw_parts(out_energies_handle.clone(), n),
            BufferArg::from_raw_parts(out_mu_handle.clone(), n),
        );
    }

    let new_energies: Vec<f64> =
        bytemuck::cast_slice(&client.read_one(out_energies_handle).unwrap()).to_vec();
    let mu_lab: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_mu_handle).unwrap()).to_vec();
    (new_energies, mu_lab)
}

/// CPU equivalent. Same PCG-32 sequence, same arithmetic; libm `sqrt`.
pub fn run_elastic_scatter_cpu(
    seeds: &[u32],
    energies_in: &[f64],
    target_masses: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(seeds.len(), energies_in.len());
    assert_eq!(seeds.len(), target_masses.len());
    let n = seeds.len();
    let mut new_energies = Vec::with_capacity(n);
    let mut mu_lab = Vec::with_capacity(n);
    for i in 0..n {
        let oldstate = crate::common::rng::expand_seed(seeds[i]);
        let _next_state = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let r_u32 = pcg_out(oldstate);
        let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let mu_cm = 1.0 - 2.0 * xi;

        let a = target_masses[i];
        let denom = (a + 1.0) * (a + 1.0);
        let numer = a * a + 2.0 * a * mu_cm + 1.0;
        new_energies.push(energies_in[i] * numer / denom);
        mu_lab.push((1.0 + a * mu_cm) / numer.sqrt());
    }
    (new_energies, mu_lab)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// `E'/E` for elastic scatter on a target of mass `A` is uniform
    /// in `[α, 1]` where `α = ((A-1)/(A+1))²`. Test on carbon
    /// (`A = 12`, `α ≈ 0.716`), 100k particles, all starting at
    /// `E = 1`. Expected mean `(1 + α) / 2 ≈ 0.858`. SEM of mean is
    /// `(1 - α) / sqrt(12 N)` ≈ 0.00026 with `N = 100_000`. A 3-σ
    /// window of 0.0008 is tight enough to catch a real bug.
    #[test]
    fn elastic_scatter_energy_distribution() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let n = 100_000usize;
        let a = 12.0_f64; // carbon
        let alpha = ((a - 1.0) / (a + 1.0)).powi(2);
        let expected_mean_ratio = (1.0 + alpha) / 2.0;
        let expected_var_ratio = (1.0 - alpha).powi(2) / 12.0;
        let sem_mean_ratio = (expected_var_ratio / n as f64).sqrt();

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies_in: Vec<f64> = vec![1.0; n];
        let masses: Vec<f64> = vec![a; n];

        let (e_out, mu_lab) = run_elastic_scatter(&ctx, &seeds, &energies_in, &masses);

        let mean_ratio: f64 = e_out.iter().sum::<f64>() / n as f64;
        println!(
            "elastic scatter on A={a}: mean E'/E = {mean_ratio} (expected {expected_mean_ratio}, SEM {sem_mean_ratio})"
        );
        assert!(
            (mean_ratio - expected_mean_ratio).abs() < 3.0 * sem_mean_ratio,
            "mean E'/E {mean_ratio} outside 3-sigma window of {expected_mean_ratio}"
        );

        // Per-particle ratios must lie in [α, 1] with no leakage.
        for (i, &ep) in e_out.iter().enumerate() {
            assert!(
                ep >= alpha - 1e-12 && ep <= 1.0 + 1e-12,
                "E'/E = {ep} at i={i} outside [{alpha}, 1]"
            );
        }
        // µ_lab must be in [-1, 1].
        for (i, &m) in mu_lab.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&m),
                "µ_lab[{i}] = {m} outside [-1, 1]"
            );
        }
    }

    /// GPU and CPU must produce per-particle agreement.
    ///
    /// **Why µ_lab uses absolute, not ulp, tolerance:** the formula
    /// `(1 + a·µ_cm) / sqrt(a² + 2a·µ_cm + 1)` has a small numerator
    /// when `µ_cm ≈ -1/a`. For `a = 2` that's `µ_cm ≈ -0.5`; for
    /// `a = 200` it's `µ_cm ≈ -0.005`. Most random samples land near
    /// such a region for *some* `a`. At those points the GPU's FMA
    /// contraction of `1.0 + a*µ_cm` keeps more bits than the CPU's
    /// separate-rounding `let t = a*µ_cm; t + 1.0`, so GPU and CPU
    /// disagree by many ulps in the quotient even though both are
    /// physically correct to ~1e-12 absolute. Absolute tolerance on
    /// `µ_lab ∈ [-1, 1]` captures the real precision; ulp counting on
    /// near-zero quotients does not. `E'` doesn't have this problem
    /// (no cancellation) so it stays on the ulp budget.
    #[test]
    fn gpu_elastic_scatter_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 5_000usize;
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        // Mix of energies and target masses to exercise different
        // operand magnitudes. A starts at 2 (deuterium) and ranges to
        // ~232 (covers everything up to actinides).
        let energies_in: Vec<f64> = (0..n).map(|i| (1.0 + (i as f64 % 7.0)) * 1e3).collect();
        let masses: Vec<f64> = (0..n).map(|i| 2.0 + (i as f64 % 230.0)).collect();

        let (gpu_e, gpu_mu) = run_elastic_scatter(&ctx, &seeds, &energies_in, &masses);
        let (cpu_e, cpu_mu) = run_elastic_scatter_cpu(&seeds, &energies_in, &masses);

        let mut max_e_ulps = 0u64;
        let mut max_mu_abs = 0.0;
        for (i, ((&ge, &ce), (&gm, &cm))) in gpu_e
            .iter()
            .zip(cpu_e.iter())
            .zip(gpu_mu.iter().zip(cpu_mu.iter()))
            .enumerate()
        {
            assert!(ge.is_finite() && ce.is_finite(), "non-finite at i={i}");
            assert!(gm.is_finite() && cm.is_finite(), "non-finite at i={i}");
            // Same-sign positive ulp diff for E' (no cancellation).
            let ge_b = ge.to_bits();
            let ce_b = ce.to_bits();
            let de = ge_b.max(ce_b) - ge_b.min(ce_b);
            if de > max_e_ulps {
                max_e_ulps = de;
            }
            // Absolute diff for µ_lab -- see comment above for why ulp
            // counting is the wrong metric on this quantity.
            let dm = (gm - cm).abs();
            if dm > max_mu_abs {
                max_mu_abs = dm;
            }
        }
        println!("elastic_scatter: max E' ULP drift = {max_e_ulps}, max |µ_lab GPU − CPU| = {max_mu_abs:e}");
        assert!(max_e_ulps <= 16, "E' drift {max_e_ulps} > 16 ulps");
        assert!(
            max_mu_abs < 1e-10,
            "µ_lab abs drift {max_mu_abs:e} > 1e-10 (well above MC noise but worth investigating)"
        );
    }
}
