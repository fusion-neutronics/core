//! On-device per-collision element selection for the photon kernel (task #72).
//!
//! At a photon collision in a multi-element material the transport must pick
//! *which* element the photon interacts with, sampling element `i` with
//! probability `Sigma_i(E) / Sum_j Sigma_j(E)` where `Sigma_i` is that
//! element's macroscopic total photon cross section (`atom_density * micro
//! total`) at the collision energy. The chosen element's form-factor /
//! relaxation / pair slab then drives the secondary physics, mirroring the CPU
//! `Material::sample_element` + `handle_photon_collision` chain.
//!
//! This is the photon analog of [`crate::neutron::nuclide_select`]. The
//! `elem_macro_total` table is pre-built per element on the shared log-energy
//! grid; the kernel interpolates it at the collision `(idx_lo, idx_hi, frac)`
//! it already computed for the macroscopic XS slate, then runs the
//! cumulative-sum walk. The walk is pure integer-PCG + f64 add/mul (no FMA, no
//! transcendentals), so the kernel and the [`select_photon_element_cpu`] twin
//! agree bit-for-bit -- `gpu_photon_element_select_matches_cpu` pins that on
//! real hardware.
//!
//! # Single-element bit-identity
//!
//! When a material has `count == 1` the production kernel SKIPS this call
//! entirely (the element is trivially `offset`), so no random is drawn and the
//! RNG stream is byte-identical to the #79 single-element path. Both this
//! `#[cube]` helper and the CPU twin assume `count >= 1` and always draw; the
//! caller is responsible for the skip.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Selection result: the chosen LOCAL element index in `[0, count)` and the
/// advanced PCG state.
#[derive(CubeType)]
pub struct ElementSelect {
    pub local: u32,
    pub state: u64,
}

/// Select the interacting element by macroscopic-total contribution. Draws one
/// PCG-32 uniform from `state`, scales by the material's interpolated total
/// (`sum_i Sigma_i`), and returns the first element whose cumulative total
/// exceeds the cutoff (clamped to the last element). `elem_off` is the
/// material's element-slab base, `count` its element count (>= 1). The macro
/// total of element local `i` is interpolated linearly from
/// `elem_macro_total[(elem_off + i) * n_grid + {idx_lo, idx_hi}]` with `frac`.
///
/// THE single source of truth, called by both the test launcher and the inline
/// selection of `multi_cell_photon_transport`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn select_photon_element(
    state: u64,
    elem_off: u32,
    count: u32,
    n_grid: u32,
    idx_lo: u32,
    idx_hi: u32,
    frac: f64,
    elem_macro_total: &[f64],
) -> ElementSelect {
    // Interpolated macro total per element, summed.
    let mut total = 0.0_f64;
    let mut i = 0u32;
    while i < count {
        let row = (elem_off + i) * n_grid;
        let lo = elem_macro_total[(row + idx_lo) as usize];
        let hi = elem_macro_total[(row + idx_hi) as usize];
        total += lo + (hi - lo) * frac;
        i += 1u32;
    }

    // One PCG-32 draw: uniform in (0, 1], scaled by the total.
    let d = crate::common::pcg32::pcg_next(state);
    let new_state = d.state;
    let xi = (d.rand as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;

    // chosen = number of elements whose cumulative total is <= xi, clamped to
    // the last index (matches the CPU `Material::sample_element` first-element-
    // with-cutoff<prob walk).
    let mut accum = 0.0_f64;
    let mut chosen = 0u32;
    let mut j = 0u32;
    while j < count {
        let row = (elem_off + j) * n_grid;
        let lo = elem_macro_total[(row + idx_lo) as usize];
        let hi = elem_macro_total[(row + idx_hi) as usize];
        accum += lo + (hi - lo) * frac;
        if xi >= accum {
            chosen += 1u32;
        }
        j += 1u32;
    }
    if chosen >= count {
        chosen = count - 1u32;
    }

    ElementSelect {
        local: chosen,
        state: new_state,
    }
}

/// CPU twin of the selection walk. `xs` is the material's per-element
/// interpolated macroscopic total at the collision energy (length `count`);
/// `state` is the 64-bit PCG state (callers expand a per-history seed with
/// `expand_seed` first). Returns the chosen local element index. Uses the
/// kernel's 32-bit PCG output (single draw) so it matches the `#[cube]` helper
/// bit-for-bit.
pub fn select_photon_element_cpu(xs: &[f64], state: u64) -> u32 {
    let n = xs.len() as u32;
    let mut total = 0.0_f64;
    for &v in xs {
        total += v;
    }
    let (rand, _new_state) = crate::common::pcg32::pcg_next_cpu(state);
    let xi = (rand as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;
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
/// sample's per-element macro total contiguously on a 1-point grid;
/// `offsets`/`counts` slice it per sample; `seeds` is the per-history PCG seed
/// (expanded to the 64-bit state in-kernel). Writes the selected local element
/// index.
#[cube(launch_unchecked)]
fn photon_element_select_kernel(
    xs_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
    out: &mut [u32],
) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    // n_grid = 1, idx_lo = idx_hi = 0, frac = 0 -> the interpolation collapses
    // to the stored value, exercising the same walk the production kernel runs.
    let state = crate::common::pcg32::expand_seed(seeds[ABSOLUTE_POS]);
    let sel = select_photon_element(
        state,
        offsets[ABSOLUTE_POS],
        counts[ABSOLUTE_POS],
        1u32,
        0u32,
        0u32,
        0.0_f64,
        xs_flat,
    );
    out[ABSOLUTE_POS] = sel.local;
}

/// Run the element-selection kernel for each sample (`offsets[k]`,
/// `counts[k]`, `seeds[k]` slicing `xs_flat`). Returns the selected per-sample
/// local element index. For test/validation use.
pub fn run_photon_element_select(
    ctx: &GpuContext,
    xs_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
) -> Vec<u32> {
    let client = ctx.client();
    let n = seeds.len();
    let xs_h = client.create_from_slice(bytemuck::cast_slice(xs_flat));
    let off_h = client.create_from_slice(bytemuck::cast_slice(offsets));
    let cnt_h = client.create_from_slice(bytemuck::cast_slice(counts));
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let out_h = client.empty(std::mem::size_of_val(seeds));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        photon_element_select_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(xs_h, xs_flat.len()),
            BufferArg::from_raw_parts(off_h, offsets.len()),
            BufferArg::from_raw_parts(cnt_h, counts.len()),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
