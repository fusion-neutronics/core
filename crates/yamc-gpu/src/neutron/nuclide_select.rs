//! On-device collision-nuclide selection.
//!
//! At a neutron collision in a material the transport must pick *which*
//! nuclide the neutron interacts with, sampling nuclide `i` with
//! probability `Sigma_i(E) / Sum_j Sigma_j(E)` where `Sigma_i` is that
//! nuclide's macroscopic total cross section (atom-density times the
//! micro XS). The GPU kernel cannot do this today because it only carries
//! the per-material density-summed macro XS; this module is the first
//! slice of restoring it (the prerequisite for sampling secondary photons
//! per-nuclide, and for D1S's `parent_nuclide`).
//!
//! This slice proves the **selection walk itself** in isolation: given a
//! material's per-nuclide macroscopic totals at the collision energy
//! (pre-interpolated) and a PCG seed, pick the nuclide. The walk is pure
//! integer-PCG + f64 add/mul (no FMA, no transcendentals), so the GPU
//! kernel and the [`select_nuclide_cpu`] 32-bit-PCG twin agree
//! **bit-for-bit** -- `gpu_nuclide_select_matches_cpu` pins that on real
//! hardware. Faithful to `Material::sample_interacting_nuclide`'s
//! cumulative-sum-with-strict-`<` walk, but driven by the kernel's 32-bit
//! PCG (the parity reference is the 32-bit twin, not the 64-bit
//! production sampler).
//!
//! Deferred to later slices: extracting/interpolating the per-nuclide
//! macro XS into device buffers (P2b), and wiring selection into the
//! production neutron kernel + the descriptor-budget probe (P4). The
//! single-nuclide RNG-draw fast path (production skips the draw) is also a
//! P4 RNG-stream-alignment concern; here both backends always draw.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// CPU twin of the GPU nuclide-selection walk. `xs` is the material's
/// per-nuclide macroscopic total XS at the collision energy; `seed` is the
/// PCG state. Returns the selected nuclide's index into `xs`. Uses the
/// kernel's 32-bit PCG output function (single draw, no state advance) so
/// it matches the `#[cube]` kernel bit-for-bit.
pub fn select_nuclide_cpu(xs: &[f64], seed: u32) -> u32 {
    let n = xs.len() as u32;
    let mut total = 0.0_f64;
    for &v in xs {
        total += v;
    }
    // One PCG-32 draw: uniform in (0, 1], scaled by the total.
    let s = crate::common::rng::expand_seed(seed);
    let r = pcg_out(s);
    let xi = (r as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;
    // chosen = number of nuclides whose cumulative total is <= xi,
    // clamped to the last index (equivalent to the CPU's first-i-with-
    // xi<accum, including the skip of zero-XS nuclides).
    let mut accum = 0.0_f64;
    let mut chosen = 0u32;
    for &v in xs {
        accum += v;
        if xi >= accum {
            chosen += 1;
        }
    }
    if chosen >= n {
        chosen = n - 1;
    }
    chosen
}

/// Test/validation kernel: one selection per thread. `xs_flat` packs each
/// sample's per-nuclide macro XS contiguously; `offsets`/`counts` slice it
/// per sample; `seeds` is the PCG state per sample. Writes the selected
/// (local) nuclide index. The walk is inlined here -- it is the same block
/// production code will inline at the collision site.
#[cube(launch_unchecked)]
fn nuclide_select_kernel(
    xs_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
    out: &mut [u32],
) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let base = offsets[ABSOLUTE_POS] as usize;
    let count = counts[ABSOLUTE_POS];

    let mut total = 0.0_f64;
    let mut i = 0u32;
    while i < count {
        total += xs_flat[base + i as usize];
        i += 1u32;
    }

    // One PCG-32 draw from the seed (output function only; no advance).
    let s = expand_seed(seeds[ABSOLUTE_POS]);
    let r = pcg_out(s);
    let xi = (r as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;

    let mut accum = 0.0_f64;
    let mut chosen = 0u32;
    let mut j = 0u32;
    while j < count {
        accum += xs_flat[base + j as usize];
        if xi >= accum {
            chosen += 1u32;
        }
        j += 1u32;
    }
    if chosen >= count {
        chosen = count - 1u32;
    }
    out[ABSOLUTE_POS] = chosen;
}

/// Run the nuclide-selection kernel for each sample (`offsets[k]`,
/// `counts[k]`, `seeds[k]` slicing `xs_flat`). Returns the selected
/// per-sample nuclide index. For test/validation use.
pub fn run_nuclide_select(
    ctx: &GpuContext,
    xs_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
) -> Vec<u32> {
    let n = seeds.len();
    assert_eq!(offsets.len(), n);
    assert_eq!(counts.len(), n);
    let client = ctx.client();
    let xs_h = client.create_from_slice(bytemuck::cast_slice(xs_flat));
    let off_h = client.create_from_slice(bytemuck::cast_slice(offsets));
    let cnt_h = client.create_from_slice(bytemuck::cast_slice(counts));
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    // One u32 output per sample (== per seed).
    let out_h = client.empty(core::mem::size_of_val(seeds));

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        nuclide_select_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(xs_h, xs_flat.len()),
            BufferArg::from_raw_parts(off_h, offsets.len()),
            BufferArg::from_raw_parts(cnt_h, counts.len()),
            BufferArg::from_raw_parts(seed_h, seeds.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice::<u8, u32>(&client.read_one(out_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// GPU nuclide selection must match the 32-bit-PCG CPU twin
    /// bit-for-bit (the walk is integer-PCG + f64 add/mul, no FMA / no
    /// transcendentals). Sweeps representative per-nuclide XS shapes
    /// (single, equal, skewed, with zero-XS nuclides, wide dynamic range)
    /// across many seeds, and checks selection-frequency sanity.
    #[test]
    fn gpu_nuclide_select_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Representative per-nuclide macroscopic-total XS arrays.
        let arrays: Vec<Vec<f64>> = vec![
            vec![1.0],                                // single nuclide
            vec![1.0, 1.0],                           // two equal
            vec![0.1, 5.0, 0.3],                      // skewed toward #1
            vec![2.0, 0.0, 3.0, 0.0, 1.0],            // interspersed zero-XS
            vec![1.0e-3, 1.0e2, 1.0, 50.0, 0.5, 7.0], // wide dynamic range
        ];

        // Flatten the distinct arrays once; samples reference them by
        // offset, each with its own seed.
        let mut xs_flat: Vec<f64> = Vec::new();
        let mut arr_off: Vec<u32> = Vec::new();
        let mut arr_cnt: Vec<u32> = Vec::new();
        for a in &arrays {
            arr_off.push(xs_flat.len() as u32);
            arr_cnt.push(a.len() as u32);
            xs_flat.extend_from_slice(a);
        }

        let n_seeds = 200u32;
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        let mut seeds = Vec::new();
        for (ai, _) in arrays.iter().enumerate() {
            for s in 0..n_seeds {
                offsets.push(arr_off[ai]);
                counts.push(arr_cnt[ai]);
                // Decorrelate seeds per array+sample with the kernel's
                // standard seed banding.
                seeds.push((ai as u32 * 9_973 + s).wrapping_mul(2_654_435_761));
            }
        }

        let gpu = run_nuclide_select(&ctx, &xs_flat, &offsets, &counts, &seeds);
        assert_eq!(gpu.len(), seeds.len());

        for (k, (&off, &cnt)) in offsets.iter().zip(counts.iter()).enumerate() {
            let a = &xs_flat[off as usize..(off + cnt) as usize];
            let cpu = select_nuclide_cpu(a, seeds[k]);
            assert_eq!(
                gpu[k], cpu,
                "GPU/CPU nuclide pick differs at sample {k}: gpu {} cpu {} (xs {a:?}, seed {})",
                gpu[k], cpu, seeds[k]
            );
            assert!(gpu[k] < cnt, "selected index {} out of range {cnt}", gpu[k]);
            // No zero-XS nuclide may ever be selected.
            assert!(
                a[gpu[k] as usize] > 0.0,
                "selected a zero-XS nuclide at sample {k} (xs {a:?})"
            );
        }

        // Sanity: for the skewed array [0.1, 5.0, 0.3], the dominant
        // nuclide (#1) must be picked most often over the seed sweep.
        let skew_ai = 2usize;
        let mut hist = vec![0u32; arrays[skew_ai].len()];
        for s in 0..n_seeds {
            let seed = (skew_ai as u32 * 9_973 + s).wrapping_mul(2_654_435_761);
            hist[select_nuclide_cpu(&arrays[skew_ai], seed) as usize] += 1;
        }
        assert!(
            hist[1] > hist[0] && hist[1] > hist[2],
            "dominant-XS nuclide should be picked most often, got {hist:?}"
        );
        println!(
            "nuclide select: {} samples, GPU==CPU bit-exact; skewed histogram {hist:?}",
            seeds.len()
        );
    }
}
