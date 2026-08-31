//! On-device coupled neutron->photon production sampling.
//!
//! After a neutron collision a material may emit secondary photons. Two
//! quantities are sampled per collision:
//!
//! 1. **How many** photons (the photon COUNT): the fractional yield
//!    `y_t = photon_prod / total` is split into a guaranteed `floor(y_t)`
//!    photons plus one more with probability `y_t - floor(y_t)` (a single
//!    Bernoulli draw).
//! 2. For each emitted photon, **which (reaction, photon-product)** pair it
//!    comes from -- a cumulative-weight walk over every packed photon product,
//!    selecting product `i` with weight `rxn_xs(E) * prod_yield_grid(E)`, each
//!    factor linearly interpolated in the collision bracket `[idx_lo, idx_hi]`
//!    (factor `frac`) so the weight tracks the exact collision energy, matching
//!    the CPU `photon_rxn_xs_interp` / `product_yield.evaluate` (the per-product
//!    layout S1 packs into
//!    [`crate::neutron::xs::photon_production::GpuPhotonProductionXs`]).
//!
//! This slice (P4 S2) proves both **selection walks** in isolation, the
//! coupled-production analogue of the neutron
//! [`crate::neutron::nuclide_select`] walk and the photon
//! [`crate::photon::discrete_spectrum`] line walk. Each is pure integer-PCG +
//! f64 add/mul (no FMA, no transcendentals), so the GPU kernels and their
//! CPU twins agree **bit-for-bit** -- `gpu_photon_product_select_matches_cpu`
//! and `gpu_photon_count_matches_cpu` pin that on real hardware.
//!
//! Faithful to `yamc_physics::photon::photon_production`:
//! [`sample_photon_product`] mirrors the two-pass `sample_photon_product`
//! (Pass-1 total, Pass-2 `cutoff = xi * total`, strict `prob > cutoff`,
//! fall-back to the last positive-weight product), and [`sample_photon_count`]
//! mirrors `sample_secondary_photons`'s `y_t` floor + Bernoulli draw
//! (lines 46-52). Both draw with the kernel's 32-bit PCG (one draw each);
//! the parity reference is the 32-bit twin, not the 64-bit `rand`-crate
//! production sampler.
//!
//! The per-product weights are interpolated at the live collision energy: the
//! selection walk takes the collision bracket `[idx_lo, idx_hi]` and the
//! linear-E factor `frac`, and both samplers are wired into the production
//! neutron kernel so emitted photons append to the device particle bank (P4).

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

// `draw_uniform` / `pcg_next` live in `crate::common::pcg32`; pull the cube
// helpers in directly so the kernels share the single tested PCG source.
use crate::common::pcg32::{draw_uniform, draw_uniform_cpu, expand_seed};

// ----------------------- product selection --------------------------

/// CPU twin of the GPU photon-product selection walk. The packed table is
/// `rxn_xs` (row-major `n_photon_rxn * n_grid`), `prod_rxn_idx`
/// (`n_product`, the row each product reads), and `prod_yield_grid`
/// (product-major `n_product * n_grid`); the collision energy falls in the
/// bracket `[idx_lo, idx_hi]` with linear-E interpolation factor `frac`, and
/// `seed` is the PCG state. Returns the selected PRODUCT index in
/// `0..n_product`.
///
/// Product `i`'s weight is the rxn-xs * yield product **linearly interpolated
/// in the collision bracket**:
/// `lerp(rxn_xs[row_lo], rxn_xs[row_hi]) * lerp(prod_yield_grid[lo], prod_yield_grid[hi])`
/// (the S1 `prod_scaling` is folded to `1.0`). This mirrors the CPU
/// `sample_photon_product`, which evaluates each per-(reaction, product) term
/// from `photon_rxn_xs_interp(i_grid, f, j)` and `product_yield.evaluate(E)` at
/// the *exact* collision energy -- both linearly interpolated, not floored at
/// `idx_lo`. Pass-1 sums the interpolated weights into `total`; Pass-2 takes
/// `cutoff = xi * total` from a single PCG draw and returns the first product
/// whose running sum is **strictly** greater than `cutoff`, falling back to the
/// last positive-weight product. A non-positive total returns the last
/// positive-weight product (or `0` if none), never a zero-weight slot. Uses the
/// kernel's 32-bit PCG so it matches the `#[cube]` kernel bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub fn sample_photon_product_cpu(
    rxn_xs: &[f64],
    prod_rxn_idx: &[u32],
    prod_yield_grid: &[f64],
    prod_base: u32,
    n_product: u32,
    n_grid: u32,
    idx_lo: u32,
    idx_hi: u32,
    frac: f64,
    seed: u64,
) -> u32 {
    let ng = n_grid as usize;
    let lo = idx_lo as usize;
    let hi = idx_hi as usize;

    // The walk runs over the material's contiguous product sub-table
    // `[prod_base, prod_base + n_product)`. `prod_rxn_idx` is GLOBAL (each
    // entry points at a row in the concatenated `rxn_xs`), and the
    // `prod_yield_grid` index `i * n_grid + g` is likewise global, so the
    // single base offset is all that distinguishes one material's sub-table
    // from another. `prod_base == 0` recovers the single-material behaviour.
    let begin = prod_base;
    let end = prod_base + n_product;

    // Pass 1: total weight, and the last product carrying positive weight
    // (the CPU's `last_product` fall-back target).
    let mut total = 0.0_f64;
    let mut last_pos = begin;
    let mut i = begin;
    while i < end {
        let rrow = prod_rxn_idx[i as usize] as usize * ng;
        let rxn = rxn_xs[rrow + lo] + (rxn_xs[rrow + hi] - rxn_xs[rrow + lo]) * frac;
        let yrow = i as usize * ng;
        let yld = prod_yield_grid[yrow + lo]
            + (prod_yield_grid[yrow + hi] - prod_yield_grid[yrow + lo]) * frac;
        let w = rxn * yld;
        total += w;
        if w > 0.0 {
            last_pos = i;
        }
        i += 1;
    }

    // One PCG-32 draw -> cutoff in (0, total]. (Drawn even when total <= 0 so
    // the RNG stream advances identically to the kernel; the result is then
    // the fall-back product.)
    let (xi, _state) = draw_uniform_cpu(seed);
    let cutoff = xi * total;

    // Pass 2: first product whose running sum strictly exceeds the cutoff;
    // fall back to the last positive-weight product. `accum` only advances at
    // positive-weight products, so the strict crossing can never land on a
    // zero-weight slot. The returned index is GLOBAL (`prod_base + relative`),
    // ready to feed `sample_photon_kinematics`.
    let mut accum = 0.0_f64;
    let mut chosen = last_pos;
    let mut found = false;
    let mut j = begin;
    while j < end {
        let rrow = prod_rxn_idx[j as usize] as usize * ng;
        let rxn = rxn_xs[rrow + lo] + (rxn_xs[rrow + hi] - rxn_xs[rrow + lo]) * frac;
        let yrow = j as usize * ng;
        let yld = prod_yield_grid[yrow + lo]
            + (prod_yield_grid[yrow + hi] - prod_yield_grid[yrow + lo]) * frac;
        accum += rxn * yld;
        if !found && accum > cutoff {
            chosen = j;
            found = true;
        }
        j += 1;
    }
    chosen
}

/// `#[cube]` photon-product selection walk: see [`sample_photon_product_cpu`]
/// for the contract. Inlined at the photon-emission site in production. Takes
/// `seed` by value and makes ONE [`draw_uniform`] draw; returns the selected
/// PRODUCT index. Each product weight is the rxn-xs * yield product linearly
/// interpolated in the collision bracket `[idx_lo, idx_hi]` (factor `frac`),
/// matching the CPU's `photon_rxn_xs_interp` / `product_yield.evaluate` at the
/// exact collision energy.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn sample_photon_product(
    rxn_xs: &[f64],
    prod_rxn_idx: &[u32],
    prod_yield_grid: &[f64],
    prod_base: u32,
    n_product: u32,
    n_grid: u32,
    idx_lo: u32,
    idx_hi: u32,
    frac: f64,
    seed: u64,
) -> u32 {
    let ng = n_grid as usize;
    let lo = idx_lo as usize;
    let hi = idx_hi as usize;

    // Walk the material's contiguous product sub-table `[prod_base,
    // prod_base + n_product)`. `prod_rxn_idx` / `prod_yield_grid` are GLOBAL,
    // so the base offset is the only thing keying one material's table from
    // another; `prod_base == 0` recovers the single-material behaviour.
    let begin = prod_base;
    let end = prod_base + n_product;

    // Pass 1: total weight + last positive-weight product (fall-back target).
    let mut total = 0.0_f64;
    let mut last_pos = begin;
    let mut i = begin;
    while i < end {
        let iu = i as usize;
        let rrow = prod_rxn_idx[iu] as usize * ng;
        let rxn = rxn_xs[rrow + lo] + (rxn_xs[rrow + hi] - rxn_xs[rrow + lo]) * frac;
        let yrow = iu * ng;
        let yld = prod_yield_grid[yrow + lo]
            + (prod_yield_grid[yrow + hi] - prod_yield_grid[yrow + lo]) * frac;
        let w = rxn * yld;
        total += w;
        if w > 0.0 {
            last_pos = i;
        }
        i += 1u32;
    }

    // One PCG-32 draw -> cutoff in (0, total].
    let d = draw_uniform(seed);
    let cutoff = d.xi * total;

    // Pass 2: first product whose running sum strictly exceeds the cutoff;
    // fall back to the last positive-weight product. A `found` flag avoids an
    // early break (cubecl prefers branch-free loops). The returned index is
    // GLOBAL (`prod_base + relative`), ready to feed `sample_photon_kinematics`.
    let mut accum = 0.0_f64;
    let mut chosen = last_pos;
    let mut found = 0u32;
    let mut j = begin;
    while j < end {
        let ju = j as usize;
        let rrow = prod_rxn_idx[ju] as usize * ng;
        let rxn = rxn_xs[rrow + lo] + (rxn_xs[rrow + hi] - rxn_xs[rrow + lo]) * frac;
        let yrow = ju * ng;
        let yld = prod_yield_grid[yrow + lo]
            + (prod_yield_grid[yrow + hi] - prod_yield_grid[yrow + lo]) * frac;
        accum += rxn * yld;
        if found == 0u32 && accum > cutoff {
            chosen = j;
            found = 1u32;
        }
        j += 1u32;
    }
    chosen
}

/// Test/validation kernel: one product selection per thread. The packed
/// `rxn_xs` / `prod_rxn_idx` / `prod_yield_grid` table is shared across all
/// threads (it is one material's table); `i_grids` and `seeds` are per-thread.
/// Writes the selected product index.
#[cube(launch_unchecked)]
fn photon_product_select_kernel(
    rxn_xs: &[f64],
    prod_rxn_idx: &[u32],
    prod_yield_grid: &[f64],
    i_grids: &[u32],
    seeds: &[u32],
    out: &mut [u32],
    #[comptime] n_product: u32,
    #[comptime] n_grid: u32,
) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    // The validation harness probes one grid column at a time; pass
    // `idx_hi == idx_lo` / `frac == 0` so the interpolated weight collapses to
    // the floor-column value the CPU twin computes for the same inputs.
    let ig = i_grids[ABSOLUTE_POS];
    out[ABSOLUTE_POS] = sample_photon_product(
        rxn_xs,
        prod_rxn_idx,
        prod_yield_grid,
        0u32,
        n_product,
        n_grid,
        ig,
        ig,
        0.0,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
}

/// Run the photon-product selection kernel for each sample. The packed table
/// (`rxn_xs`, `prod_rxn_idx`, `prod_yield_grid`, `n_product`, `n_grid`) is one
/// material's; `i_grids[k]` / `seeds[k]` drive sample `k`. Returns the
/// selected per-sample product index. For test/validation use.
#[allow(clippy::too_many_arguments)]
pub fn run_photon_product_select(
    ctx: &GpuContext,
    rxn_xs: &[f64],
    prod_rxn_idx: &[u32],
    prod_yield_grid: &[f64],
    n_product: u32,
    n_grid: u32,
    i_grids: &[u32],
    seeds: &[u32],
) -> Vec<u32> {
    let n = seeds.len();
    assert_eq!(i_grids.len(), n);
    assert_eq!(prod_rxn_idx.len(), n_product as usize);
    assert_eq!(prod_yield_grid.len(), (n_product * n_grid) as usize);
    let client = ctx.client();
    let xs_h = client.create_from_slice(bytemuck::cast_slice(rxn_xs));
    let idx_h = client.create_from_slice(bytemuck::cast_slice(prod_rxn_idx));
    let yld_h = client.create_from_slice(bytemuck::cast_slice(prod_yield_grid));
    let ig_h = client.create_from_slice(bytemuck::cast_slice(i_grids));
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    // One u32 output per sample (== per seed).
    let out_h = client.empty(core::mem::size_of_val(seeds));

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        photon_product_select_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(xs_h, rxn_xs.len()),
            BufferArg::from_raw_parts(idx_h, prod_rxn_idx.len()),
            BufferArg::from_raw_parts(yld_h, prod_yield_grid.len()),
            BufferArg::from_raw_parts(ig_h, i_grids.len()),
            BufferArg::from_raw_parts(seed_h, seeds.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
            n_product,
            n_grid,
        );
    }
    bytemuck::cast_slice::<u8, u32>(&client.read_one(out_h).unwrap()).to_vec()
}

// ------------------------- photon count -----------------------------

/// CPU twin of the GPU photon-count sampler. Mirrors
/// `sample_secondary_photons` lines 46-52: the fractional yield
/// `y_t = photon_prod_g / total_g` is split into `n = floor(y_t)` guaranteed
/// photons plus one more if a single uniform draw `xi < (y_t - n)`. Returns
/// the photon count. A non-positive `total_g` (or `photon_prod_g`) yields `0`
/// after the floor (and the Bernoulli fraction is `<= 0`), matching the CPU
/// guard `photon_prod_xs <= 0 || total <= 0 => 0`. Uses the kernel's 32-bit
/// PCG so it matches the `#[cube]` kernel bit-for-bit.
pub fn sample_photon_count_cpu(photon_prod_g: f64, total_g: f64, seed: u64) -> u32 {
    if total_g <= 0.0 || photon_prod_g <= 0.0 {
        return 0;
    }
    let y_t = photon_prod_g / total_g;
    let n = y_t as u32; // floor (y_t >= 0 here)
    let (xi, _state) = draw_uniform_cpu(seed);
    if xi < (y_t - n as f64) {
        n + 1
    } else {
        n
    }
}

/// `#[cube]` photon-count sampler: see [`sample_photon_count_cpu`]. Takes
/// `seed` by value and makes ONE [`draw_uniform`] draw; returns the photon
/// count.
#[cube]
pub fn sample_photon_count(photon_prod_g: f64, total_g: f64, seed: u64) -> u32 {
    let mut count = 0u32;
    if total_g > 0.0 && photon_prod_g > 0.0 {
        let y_t = photon_prod_g / total_g;
        // Truncate-toward-zero == floor for the non-negative `y_t` here, the
        // same float->int cast the tally bin code uses (see multi_step.rs).
        let n = y_t as u32;
        let d = draw_uniform(seed);
        count = n;
        if d.xi < (y_t - n as f64) {
            count = n + 1u32;
        }
    }
    count
}

/// Test/validation kernel: one photon-count draw per thread from its
/// `(photon_prod, total, seed)`.
#[cube(launch_unchecked)]
fn photon_count_kernel(photon_prod: &[f64], total: &[f64], seeds: &[u32], out: &mut [u32]) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    out[ABSOLUTE_POS] = sample_photon_count(
        photon_prod[ABSOLUTE_POS],
        total[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
}

/// Run the photon-count kernel for each sample (`photon_prod[k]`, `total[k]`,
/// `seeds[k]`). Returns the per-sample photon count. For test/validation use.
pub fn run_photon_count(
    ctx: &GpuContext,
    photon_prod: &[f64],
    total: &[f64],
    seeds: &[u32],
) -> Vec<u32> {
    let n = seeds.len();
    assert_eq!(photon_prod.len(), n);
    assert_eq!(total.len(), n);
    let client = ctx.client();
    let pp_h = client.create_from_slice(bytemuck::cast_slice(photon_prod));
    let tot_h = client.create_from_slice(bytemuck::cast_slice(total));
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let out_h = client.empty(core::mem::size_of_val(seeds));

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        photon_count_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(pp_h, photon_prod.len()),
            BufferArg::from_raw_parts(tot_h, total.len()),
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

    /// CPU per-product weight at `i_grid`, exactly as both walks compute it.
    fn weight_at(
        rxn_xs: &[f64],
        prod_rxn_idx: &[u32],
        prod_yield_grid: &[f64],
        n_grid: u32,
        i: usize,
        i_grid: u32,
    ) -> f64 {
        let row = prod_rxn_idx[i] as usize * n_grid as usize + i_grid as usize;
        rxn_xs[row] * prod_yield_grid[i * n_grid as usize + i_grid as usize]
    }

    /// Build a representative packed photon-production table: a few reactions
    /// (rows), several products, skewed weights, some genuinely zero-weight
    /// products (zero yield or a product reading a zero-xs reaction row), and
    /// multi-product reactions. Returns `(rxn_xs, prod_rxn_idx,
    /// prod_yield_grid, n_product, n_grid)`. The values vary across the grid so
    /// different `i_grid` columns exercise different weight shapes.
    fn synth_table() -> (Vec<f64>, Vec<u32>, Vec<f64>, u32, u32) {
        let n_grid = 4u32;
        // 4 reaction rows. Row 1 is identically zero (a thresholded reaction
        // below threshold) so every product reading it has zero weight.
        let rxn_xs: Vec<f64> = vec![
            // row 0
            5.0, 4.0, 3.0, 2.0, //
            // row 1 (all zero -> dead row)
            0.0, 0.0, 0.0, 0.0, //
            // row 2
            0.1, 0.5, 1.0, 2.0, //
            // row 3
            10.0, 1.0, 0.01, 0.001,
        ];
        // 8 products. Multi-product reactions: row 0 carries products 0,1,2;
        // row 1 carries product 3 (dead); row 2 carries products 4,5; row 3
        // carries products 6,7. Product 5 has zero yield everywhere (a packed
        // but unsamplable slot). The last packed product (7) reads the
        // fast-decaying row 3, so at high i_grid its weight underflows toward
        // zero -- exercising the last-positive-weight fall-back.
        let prod_rxn_idx: Vec<u32> = vec![0, 0, 0, 1, 2, 2, 3, 3];
        let prod_yield_grid: Vec<f64> = vec![
            // p0 (row0): mild
            1.0, 1.0, 1.0, 1.0, //
            // p1 (row0): skewed high
            3.0, 3.0, 3.0, 3.0, //
            // p2 (row0): tiny
            0.01, 0.01, 0.01, 0.01, //
            // p3 (row1, dead row): nonzero yield but zero xs -> zero weight
            2.0, 2.0, 2.0, 2.0, //
            // p4 (row2): grid-varying
            0.5, 1.0, 2.0, 4.0, //
            // p5 (row2): identically zero yield -> always zero weight
            0.0, 0.0, 0.0, 0.0, //
            // p6 (row3): large at low E
            1.0, 1.0, 1.0, 1.0, //
            // p7 (row3): also reads fast-decaying row3
            0.5, 0.5, 0.5, 0.5,
        ];
        let n_product = prod_rxn_idx.len() as u32;
        (rxn_xs, prod_rxn_idx, prod_yield_grid, n_product, n_grid)
    }

    /// GPU photon-product selection must match the CPU twin bit-for-bit. The
    /// walk is integer-PCG + f64 add/mul (the `xi * total` cutoff is one f64
    /// multiply identical on both sides), so the selected index is exactly
    /// equal -- not within a tolerance. Sweeps every grid column across many
    /// seeds and checks: bit-exact GPU==CPU, never a zero-weight product is
    /// returned, and the index is in range.
    #[test]
    fn gpu_photon_product_select_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let (rxn_xs, prod_rxn_idx, prod_yield_grid, n_product, n_grid) = synth_table();

        // Per-sample (i_grid, seed): every grid column x many seeds.
        let n_seeds = 250u32;
        let mut i_grids = Vec::new();
        let mut seeds = Vec::new();
        for g in 0..n_grid {
            for s in 0..n_seeds {
                i_grids.push(g);
                seeds.push((g * 9_973 + s).wrapping_mul(2_654_435_761));
            }
        }

        let gpu = run_photon_product_select(
            &ctx,
            &rxn_xs,
            &prod_rxn_idx,
            &prod_yield_grid,
            n_product,
            n_grid,
            &i_grids,
            &seeds,
        );
        assert_eq!(gpu.len(), seeds.len());

        for k in 0..seeds.len() {
            let cpu = sample_photon_product_cpu(
                &rxn_xs,
                &prod_rxn_idx,
                &prod_yield_grid,
                0,
                n_product,
                n_grid,
                i_grids[k],
                i_grids[k],
                0.0,
                crate::common::rng::expand_seed(seeds[k]),
            );
            assert_eq!(
                gpu[k], cpu,
                "GPU/CPU product pick differs at sample {k}: gpu {} cpu {} (i_grid {}, seed {})",
                gpu[k], cpu, i_grids[k], seeds[k]
            );
            assert!(
                gpu[k] < n_product,
                "selected product {} out of range {n_product}",
                gpu[k]
            );
            // Never a zero-weight product (dead-row or zero-yield slots).
            let w = weight_at(
                &rxn_xs,
                &prod_rxn_idx,
                &prod_yield_grid,
                n_grid,
                gpu[k] as usize,
                i_grids[k],
            );
            assert!(
                w > 0.0,
                "sample {k}: picked zero-weight product {} @ i_grid {} (w {w})",
                gpu[k],
                i_grids[k]
            );
        }

        // Sanity: at i_grid 0, the highest-weight product must be picked most
        // often over the seed sweep. Weights at g=0:
        //   p0=5, p1=15, p2=0.05, p3=0, p4=0.05, p5=0, p6=10, p7=5.
        // p1 (15) dominates.
        let mut hist = vec![0u32; n_product as usize];
        for s in 0..n_seeds {
            // Same seed formula the sweep uses for grid column g == 0.
            let seed = s.wrapping_mul(2_654_435_761);
            let p = sample_photon_product_cpu(
                &rxn_xs,
                &prod_rxn_idx,
                &prod_yield_grid,
                0,
                n_product,
                n_grid,
                0,
                0,
                0.0,
                crate::common::rng::expand_seed(seed),
            );
            hist[p as usize] += 1;
        }
        // Dead/zero-weight products must never be picked.
        assert_eq!(hist[3], 0, "dead-row product 3 picked {} times", hist[3]);
        assert_eq!(hist[5], 0, "zero-yield product 5 picked {} times", hist[5]);
        let dominant = hist.iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0;
        assert_eq!(
            dominant, 1,
            "highest-weight product (1) should dominate at i_grid 0, got hist {hist:?}"
        );
        println!(
            "photon product select: {} samples, GPU==CPU bit-exact; i_grid 0 histogram {hist:?}",
            seeds.len()
        );
    }

    /// GPU photon-count must match the CPU twin bit-for-bit. The count is
    /// `floor(y_t)` plus a single Bernoulli draw on the fraction; `y_t` is one
    /// f64 divide (identical on both backends) and the floor + compare are
    /// exact, so the count is exactly equal. Sweeps representative
    /// `(photon_prod, total)` pairs -- sub-unity yields, integer yields,
    /// fractional > 1 yields, the zero/degenerate guards -- across many seeds,
    /// and checks the mean count tracks `y_t`.
    #[test]
    fn gpu_photon_count_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // (photon_prod, total) pairs spanning the regimes the count sampler
        // sees: y_t < 1 (the common case), y_t exactly integer, y_t > 1 with a
        // fractional part, and the degenerate guards (zero/negative total or
        // prod).
        let pairs: Vec<(f64, f64)> = vec![
            (0.3, 1.0),  // y_t = 0.3
            (0.7, 1.0),  // y_t = 0.7
            (1.0, 1.0),  // y_t = 1.0 exactly
            (2.0, 1.0),  // y_t = 2.0 exactly
            (2.5, 1.0),  // y_t = 2.5
            (3.3, 2.0),  // y_t = 1.65
            (1.0, 3.0),  // y_t = 1/3 (repeating)
            (0.0, 1.0),  // no production
            (1.0, 0.0),  // degenerate total -> 0
            (-1.0, 1.0), // negative prod -> 0
            (5.0, 2.0),  // y_t = 2.5 again at different magnitude
        ];

        let n_seeds = 400u32;
        let mut photon_prod = Vec::new();
        let mut total = Vec::new();
        let mut seeds = Vec::new();
        for (pi, &(pp, t)) in pairs.iter().enumerate() {
            for s in 0..n_seeds {
                photon_prod.push(pp);
                total.push(t);
                seeds.push((pi as u32 * 9_973 + s).wrapping_mul(2_654_435_761));
            }
        }

        let gpu = run_photon_count(&ctx, &photon_prod, &total, &seeds);
        assert_eq!(gpu.len(), seeds.len());

        for k in 0..seeds.len() {
            let cpu = sample_photon_count_cpu(
                photon_prod[k],
                total[k],
                crate::common::rng::expand_seed(seeds[k]),
            );
            assert_eq!(
                gpu[k], cpu,
                "GPU/CPU photon count differs at sample {k}: gpu {} cpu {} (pp {}, total {}, seed {})",
                gpu[k], cpu, photon_prod[k], total[k], seeds[k]
            );
            // Count is floor(y_t) or floor+1.
            if total[k] > 0.0 && photon_prod[k] > 0.0 {
                let y_t = photon_prod[k] / total[k];
                let floor = y_t as u32;
                assert!(
                    gpu[k] == floor || gpu[k] == floor + 1,
                    "sample {k}: count {} not in {{{floor}, {}}} for y_t {y_t}",
                    gpu[k],
                    floor + 1
                );
            } else {
                assert_eq!(gpu[k], 0, "degenerate guard should give 0");
            }
        }

        // Mean count over the sweep must track y_t for the fractional cases.
        for (pi, &(pp, t)) in pairs.iter().enumerate() {
            if t <= 0.0 || pp <= 0.0 {
                continue;
            }
            let y_t = pp / t;
            let mut sum = 0u64;
            for s in 0..n_seeds {
                let seed = (pi as u32 * 9_973 + s).wrapping_mul(2_654_435_761);
                sum += sample_photon_count_cpu(pp, t, crate::common::rng::expand_seed(seed)) as u64;
            }
            let mean = sum as f64 / n_seeds as f64;
            assert!(
                (mean - y_t).abs() < 0.15,
                "mean count {mean} should track y_t {y_t} (pair {pi})"
            );
        }
        println!(
            "photon count: {} samples, GPU==CPU bit-exact across {} (prod,total) regimes",
            seeds.len(),
            pairs.len()
        );
    }
}
