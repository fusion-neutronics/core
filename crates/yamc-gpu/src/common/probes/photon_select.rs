//! Photon interaction-type selection on the GPU -- given a uniform `cutoff`
//! in `[0, sigma_total)` and the partial cross sections, pick the reaction by
//! cumulative comparison (0=coherent, 1=incoherent, 2=photoelectric,
//! 3=pair). Extracted from the inline block in `multi_cell_photon_transport`
//! so the mega kernel calls it and it can be unit-tested against the CPU
//! selection in `handle_photon_collision`.
//!
//! Pure (the caller draws `cutoff = xi * sigma_total`), so byte-identical.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Select the photon reaction type. `cutoff = xi * sigma_total`; returns
/// 0=coherent, 1=incoherent, 2=photoelectric, 3=pair. Pair is the
/// fall-through (so `sigma_pair` isn't needed). THE single source of truth,
/// called by both the test launcher and the inline selection of
/// `multi_cell_photon_transport`.
#[cube]
pub fn select_photon_reaction(
    cutoff: f64,
    sigma_coh: f64,
    sigma_inc: f64,
    sigma_photo: f64,
) -> u32 {
    let mut prob = sigma_coh;
    let mut rxn_kind = 0u32;
    if prob <= cutoff {
        prob += sigma_inc;
        rxn_kind = 1u32;
        if prob <= cutoff {
            prob += sigma_photo;
            rxn_kind = 2u32;
            if prob <= cutoff {
                rxn_kind = 3u32;
            }
        }
    }
    rxn_kind
}

/// Per-thread test launcher: each thread reads `cutoffs[tid]` and the three
/// partial XS (single-element buffers `sc`/`si`/`sp`), writes the selected
/// reaction kind.
#[cube(launch_unchecked)]
fn photon_select_kernel(cutoffs: &[f64], sc: &[f64], si: &[f64], sp: &[f64], out_kind: &mut [u32]) {
    if ABSOLUTE_POS >= cutoffs.len() {
        terminate!();
    }
    out_kind[ABSOLUTE_POS] =
        select_photon_reaction(cutoffs[ABSOLUTE_POS], sc[0usize], si[0usize], sp[0usize]);
}

/// Run the selection for a sweep of `cutoffs` with fixed partial XS. Returns
/// the selected reaction kind per cutoff.
pub fn run_photon_select(
    ctx: &GpuContext,
    cutoffs: &[f64],
    sigma_coh: f64,
    sigma_inc: f64,
    sigma_photo: f64,
) -> Vec<u32> {
    let client = ctx.client();
    let n = cutoffs.len();
    let cut_h = client.create_from_slice(bytemuck::cast_slice(cutoffs));
    let sc_h = client.create_from_slice(bytemuck::cast_slice(&[sigma_coh]));
    let si_h = client.create_from_slice(bytemuck::cast_slice(&[sigma_inc]));
    let sp_h = client.create_from_slice(bytemuck::cast_slice(&[sigma_photo]));
    let out_h = client.empty(n * std::mem::size_of::<u32>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        photon_select_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(cut_h, n),
            BufferArg::from_raw_parts(sc_h, 1),
            BufferArg::from_raw_parts(si_h, 1),
            BufferArg::from_raw_parts(sp_h, 1),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
