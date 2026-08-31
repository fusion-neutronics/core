//! GPU port of the per-nuclide URR probability-table decorrelator
//! ([`yamc_nuclide::urr::urr_nuclide_random`]), expressed as a cubecl
//! `#[cube]` function so it compiles to SPIR-V on the Vulkan backend.
//!
//! In the unresolved resonance range each nuclide's cross section is an
//! independent draw from its own probability table (isotopes' resonance
//! structures are statistically uncorrelated). The transport kernel draws
//! ONE base uniform per collision and derives each in-range URR nuclide's
//! band from it by mixing in the nuclide's `ZA` (issue #204). This is the
//! GPU twin of that mixer; a host test pins it bit-for-bit against the
//! canonical CPU implementation, and a GPU validation kernel pins the SPIR-V
//! lowering against the CPU twin on real hardware.

use crate::common::polyfills::{exp_f64, ln_f64};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// SplitMix64 golden-ratio increment used to mix `ZA` into the base entropy.
const URR_GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64 finalizer multipliers.
const URR_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
const URR_MIX_2: u64 = 0x94D0_49BB_1331_11EB;
/// `2^53`; the top 53 bits of the finalized word map to a uniform in `[0, 1)`.
const URR_TWO_POW_53: f64 = 9_007_199_254_740_992.0;

/// Derive an independent per-nuclide URR probability-table random in `[0, 1)`
/// from a per-collision base seed and the nuclide's `ZA` identifier
/// (`Z * 1000 + A`, or an FNV name-hash fallback). Bit-for-bit twin of
/// [`yamc_nuclide::urr::urr_nuclide_random`].
///
/// SplitMix64 finalizer over `base.to_bits() + za * golden`. The u64 add /
/// multiply wrap (2's complement) on SPIR-V exactly as the CPU
/// `wrapping_add` / `wrapping_mul` do; the f64->u64 bit reinterpretation is
/// the only lowering unique to this helper (`u64::reinterpret`, the same
/// bitcast `ln_f64` / `exp_f64` already use).
#[cube]
pub fn urr_nuclide_random(base_seed: f64, za: u32) -> f64 {
    let za64 = za as u64;
    let mut z = u64::reinterpret(base_seed) + za64 * URR_GOLDEN;
    z = (z ^ (z >> 30u64)) * URR_MIX_1;
    z = (z ^ (z >> 27u64)) * URR_MIX_2;
    z = z ^ (z >> 31u64);
    // Top 53 bits -> uniform [0, 1).
    (z >> 11u64) as f64 * (1.0 / URR_TWO_POW_53)
}

/// CPU twin of [`urr_nuclide_random`] (explicit wrapping `u64` arithmetic).
/// Kept alongside the `#[cube]` function so the host test can compare the
/// exact bit pattern the GPU compiles from against the canonical
/// [`yamc_nuclide::urr::urr_nuclide_random`].
pub fn urr_nuclide_random_cpu(base_seed: f64, za: u32) -> f64 {
    let mut z = base_seed
        .to_bits()
        .wrapping_add((za as u64).wrapping_mul(URR_GOLDEN));
    z = (z ^ (z >> 30)).wrapping_mul(URR_MIX_1);
    z = (z ^ (z >> 27)).wrapping_mul(URR_MIX_2);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / URR_TWO_POW_53)
}

/// Validation kernel: one `(base_seed, za)` pair per thread through
/// [`urr_nuclide_random`], writing the result back for a bit-for-bit host
/// comparison against the CPU twin.
#[cube(launch_unchecked)]
fn urr_nuclide_random_kernel(base_seeds: &[f64], zas: &[u32], out: &mut [f64]) {
    if ABSOLUTE_POS >= base_seeds.len() {
        terminate!();
    }
    out[ABSOLUTE_POS] = urr_nuclide_random(base_seeds[ABSOLUTE_POS], zas[ABSOLUTE_POS]);
}

/// Run [`urr_nuclide_random`] on the GPU for each `(base_seed, za)` pair.
/// For test/validation use.
// The output buffer is sized by element count (one f64 per pair), not by the
// `base_seeds` slice's byte size; they coincide here but the intent is a count.
#[allow(clippy::manual_slice_size_calculation)]
pub fn run_urr_nuclide_random(ctx: &GpuContext, base_seeds: &[f64], zas: &[u32]) -> Vec<f64> {
    assert_eq!(base_seeds.len(), zas.len());
    let client = ctx.client();
    let n = base_seeds.len();
    let base_h = client.create_from_slice(bytemuck::cast_slice(base_seeds));
    let za_h = client.create_from_slice(bytemuck::cast_slice(zas));
    let out_h = client.empty(n * core::mem::size_of::<f64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        urr_nuclide_random_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(base_h, n),
            BufferArg::from_raw_parts(za_h, n),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice::<u8, f64>(&client.read_one(out_h).unwrap()).to_vec()
}

/// One slab's URR-perturbed macroscopic partials.
///
/// `fired == 0` means this slab has no URR table covering the energy, and the
/// caller keeps its smooth values.
#[derive(CubeType)]
pub struct UrrSlabPartials {
    pub elastic: f64,
    pub absorption: f64,
    pub inelastic: f64,
    pub fission: f64,
    pub fired: u32,
}

/// Perturb one slab's smooth macroscopic partials with its URR probability
/// table, at the band the shared per-collision base `r_base` selects for it.
///
/// This is the kernel's single definition of the perturbation. The material
/// aggregate that governs the flight, the per-nuclide weights that govern which
/// nuclide is struck, and the partials that govern the reaction split all call
/// it with the same `r_base`, so all three ride ONE sampled band (issue #347).
/// `urr_nuclide_random` is a pure function of `(r_base, ZA)`, so recomputing it
/// per site cannot drift.
///
/// Tight CSR (issue #104): the URR buffers carry no `MAX_URR_*` padding, so
/// every bracket search here is bounded by the slab's own counts. A fixed
/// comptime bound would read past the slab under `launch_unchecked`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn urr_slab_partials(
    urr_meta: &[u32],
    urr_ae_offset: &[u32],
    urr_cdf_offset: &[u32],
    urr_energy_grid: &[f64],
    urr_cdf: &[f64],
    urr_xs: &[f64],
    urr_atom_density: &[f64],
    slab: u32,
    energy: f64,
    r_base: f64,
    base_e: f64,
    base_a: f64,
    base_i: f64,
    base_f: f64,
) -> UrrSlabPartials {
    let mut out_e = base_e;
    let mut out_a = base_a;
    let mut out_i = base_i;
    let mut out_f = base_f;
    let mut fired = 0u32;

    let mo = slab * 8u32; // URR_META_COLS
    if urr_meta[mo as usize] == 1u32 {
        let n_e = urr_meta[(mo + 1u32) as usize];
        let n_cdf = urr_meta[(mo + 2u32) as usize];
        let interp_kind = urr_meta[(mo + 3u32) as usize];
        let inel_flag = urr_meta[(mo + 4u32) as usize];
        let mult_smooth = urr_meta[(mo + 6u32) as usize];
        let za = urr_meta[(mo + 7u32) as usize];
        let eg_off = urr_ae_offset[slab as usize];
        if n_e >= 2u32 {
            let e_first = urr_energy_grid[eg_off as usize];
            let e_last = urr_energy_grid[(eg_off + n_e - 1u32) as usize];
            if energy > e_first && energy < e_last {
                // Independent per-nuclide band from the shared base (#204).
                let r_urr = urr_nuclide_random(r_base, za);

                // Energy bracket within this slab's tight grid.
                let mut i_e = 0u32;
                let mut k = 0u32;
                while k + 1u32 < n_e {
                    let e_k = urr_energy_grid[(eg_off + k) as usize];
                    let e_k1 = urr_energy_grid[(eg_off + k + 1u32) as usize];
                    if energy >= e_k && energy < e_k1 {
                        i_e = k;
                    }
                    k += 1u32;
                }
                let i_e_next = i_e + 1u32;
                let e_lo = urr_energy_grid[(eg_off + i_e) as usize];
                let e_hi = urr_energy_grid[(eg_off + i_e_next) as usize];
                let mut interp_f = 0.0_f64;
                if e_hi > e_lo {
                    if interp_kind == 1u32 && e_lo > 0.0 {
                        interp_f = ln_f64(energy / e_lo) / ln_f64(e_hi / e_lo);
                    } else {
                        interp_f = (energy - e_lo) / (e_hi - e_lo);
                    }
                }

                // CDF band at both energy points (upper bound, clamped).
                let cdf_base = urr_cdf_offset[slab as usize];
                let cdf_off_lo = cdf_base + i_e * n_cdf;
                let cdf_off_hi = cdf_base + i_e_next * n_cdf;
                let mut j_lo = 0u32;
                let mut j_hi = 0u32;
                let mut kc = 0u32;
                while kc < n_cdf {
                    if urr_cdf[(cdf_off_lo + kc) as usize] <= r_urr {
                        j_lo = kc + 1u32;
                    }
                    if urr_cdf[(cdf_off_hi + kc) as usize] <= r_urr {
                        j_hi = kc + 1u32;
                    }
                    kc += 1u32;
                }
                if j_lo >= n_cdf {
                    j_lo = n_cdf - 1u32;
                }
                if j_hi >= n_cdf {
                    j_hi = n_cdf - 1u32;
                }

                let xs_off_lo = (cdf_off_lo + j_lo) * 4u32; // URR_XS_COLS
                let xs_off_hi = (cdf_off_hi + j_hi) * 4u32;
                let xs_e_lo_t = urr_xs[(xs_off_lo + 1u32) as usize];
                let xs_e_hi_t = urr_xs[(xs_off_hi + 1u32) as usize];
                let xs_f_lo_t = urr_xs[(xs_off_lo + 2u32) as usize];
                let xs_f_hi_t = urr_xs[(xs_off_hi + 2u32) as usize];
                let xs_g_lo_t = urr_xs[(xs_off_lo + 3u32) as usize];
                let xs_g_hi_t = urr_xs[(xs_off_hi + 3u32) as usize];
                let mut urr_micro_e = if interp_kind == 1u32 && xs_e_lo_t > 0.0 && xs_e_hi_t > 0.0 {
                    exp_f64((1.0 - interp_f) * ln_f64(xs_e_lo_t) + interp_f * ln_f64(xs_e_hi_t))
                } else {
                    (1.0 - interp_f) * xs_e_lo_t + interp_f * xs_e_hi_t
                };
                let mut urr_micro_f = if interp_kind == 1u32 && xs_f_lo_t > 0.0 && xs_f_hi_t > 0.0 {
                    exp_f64((1.0 - interp_f) * ln_f64(xs_f_lo_t) + interp_f * ln_f64(xs_f_hi_t))
                } else {
                    (1.0 - interp_f) * xs_f_lo_t + interp_f * xs_f_hi_t
                };
                let mut urr_micro_g = if interp_kind == 1u32 && xs_g_lo_t > 0.0 && xs_g_hi_t > 0.0 {
                    exp_f64((1.0 - interp_f) * ln_f64(xs_g_lo_t) + interp_f * ln_f64(xs_g_hi_t))
                } else {
                    (1.0 - interp_f) * xs_g_lo_t + interp_f * xs_g_hi_t
                };
                if urr_micro_e < 0.0 {
                    urr_micro_e = 0.0;
                }
                if urr_micro_f < 0.0 {
                    urr_micro_f = 0.0;
                }
                if urr_micro_g < 0.0 {
                    urr_micro_g = 0.0;
                }

                // multiply_smooth = 1: table entries are factors on the smooth
                // macroscopic baseline; else absolute micro XS scaled by atom
                // density. The CPU multiplies the capture column by
                // (smooth_absorption - smooth_fission) == `base_a`.
                if mult_smooth == 1u32 {
                    out_e = urr_micro_e * base_e;
                    out_a = urr_micro_g * base_a;
                    out_f = urr_micro_f * base_f;
                } else {
                    let n_dens = urr_atom_density[slab as usize];
                    out_e = n_dens * urr_micro_e;
                    out_a = n_dens * urr_micro_g;
                    out_f = n_dens * urr_micro_f;
                }
                // Smooth-inelastic exclusion (#105): with inelastic_flag <= 0
                // the CPU drops inelastic from the URR-window total.
                if inel_flag == 0u32 {
                    out_i = 0.0;
                }
                fired = 1u32;
            }
        }
    }

    UrrSlabPartials {
        elastic: out_e,
        absorption: out_a,
        inelastic: out_i,
        fission: out_f,
        fired,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// A representative `(base, za)` table: adjacent W ZAs (the natural-W
    /// isotopes the multimaterial-slab regression exercises), a handful of
    /// other isotopes, and an FNV name-hash fallback key (top bit set).
    fn table() -> (Vec<f64>, Vec<u32>) {
        let bases = [
            0.0,
            0.123_456_789,
            0.5,
            0.999_999,
            0.000_001,
            0.314_159_265,
            0.271_828_182,
        ];
        let zas: [u32; 8] = [74182, 74183, 74184, 74186, 26056, 92238, 1001, 0x8000_00FF];
        let mut b = Vec::new();
        let mut z = Vec::new();
        for &base in &bases {
            for &za in &zas {
                b.push(base);
                z.push(za);
            }
        }
        (b, z)
    }

    /// The GPU-ported finalizer's CPU twin must be bit-identical to the
    /// canonical `yamc_nuclide::urr::urr_nuclide_random` across the table,
    /// including the adjacent W ZAs (74182/74183/74184/74186) and the
    /// FNV-fallback key. This runs without a GPU (the twin is the exact
    /// arithmetic the `#[cube]` function compiles from).
    #[test]
    fn urr_nuclide_random_cpu_twin_matches_reference() {
        let (bases, zas) = table();
        for (&base, &za) in bases.iter().zip(zas.iter()) {
            let ours = urr_nuclide_random_cpu(base, za);
            let reference = yamc_nuclide::urr::urr_nuclide_random(base, za);
            assert_eq!(
                ours.to_bits(),
                reference.to_bits(),
                "base={base} za={za}: twin {ours} != reference {reference}"
            );
            assert!((0.0..1.0).contains(&ours), "must be in [0,1): {ours}");
        }
    }

    /// The SPIR-V lowering must match the CPU twin bit-for-bit on real
    /// hardware. Skips when no Vulkan f64 adapter is present (llvmpipe cannot
    /// run the kernel). The one lowering unique to this helper vs the proven
    /// `pcg32` path is the u64 wrapping multiply and the f64->u64 bitcast, so
    /// this is the check that pins them.
    #[test]
    fn gpu_urr_nuclide_random_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let (bases, zas) = table();
        let gpu = run_urr_nuclide_random(&ctx, &bases, &zas);
        assert_eq!(gpu.len(), bases.len());
        for (i, (&base, &za)) in bases.iter().zip(zas.iter()).enumerate() {
            let cpu = urr_nuclide_random_cpu(base, za);
            assert_eq!(
                gpu[i].to_bits(),
                cpu.to_bits(),
                "base={base} za={za}: gpu {} != cpu {cpu}",
                gpu[i]
            );
        }
        println!(
            "urr_nuclide_random: {} (base, za) pairs, GPU==CPU bit-exact",
            bases.len()
        );
    }
}
