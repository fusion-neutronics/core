//! Photoelectron emission direction on the GPU -- sample the non-relativistic
//! Sauter polar cosine by rejection, then build a unit direction from it.
//! Extracted from the photon kernel's photoelectric branch as the single
//! source of truth so it can be unit-tested against the CPU reference
//! `yamc_physics::photon::photoelectron::sample_photoelectron_direction`.
//!
//! `mu` is taken about the **lab +x axis**, not about the incident photon's
//! direction, because that is what the CPU and OpenMC
//! (`sample_photon_reaction`, which assigns `u.x = mu`) both do. The
//! photoelectron is only ever a bremsstrahlung source here, and rotating into
//! the photon frame would diverge from the benchmark rather than match it.
//!
//! The kernel used to give the photoelectron's brem the parent photon's
//! direction instead. That is a Z-biased flux error: the photoelectron carries
//! nearly the whole photon energy, so in a high-Z medium its brem is a large
//! share of the secondary photon flux.

use crate::common::pcg32::{draw_uniform, expand_seed};
use crate::common::polyfills::{cos_f64, sin_f64};
use crate::GpuContext;
use cubecl::prelude::*;

/// Electron rest mass in eV. Mirrors `yamc_element::photon::MASS_ELECTRON_EV`.
pub const MASS_ELECTRON_EV: f64 = 0.510_998_950_00e6;

/// A sampled photoelectron direction plus the advanced PCG-32 state.
#[derive(CubeType)]
pub struct SauterDirection {
    /// Advanced PCG-32 state.
    pub state: u64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Sample a photoelectron direction for kinetic energy `electron_ke` (eV).
/// THE single source of truth, called by the photon kernel's photoelectric
/// branch and by the test launcher below.
///
/// The rejection loop is bounded at 32 trials for SPIR-V. Acceptance of
/// `4 r (1 - r) >= r2` is 2/3 per trial, so exhausting the budget has
/// probability 3^-32; `mu` then keeps its forward default rather than the
/// kernel hanging.
#[cube]
pub fn sample_sauter_direction(state: u64, electron_ke: f64) -> SauterDirection {
    let mut st = state;
    let mut mu = 1.0_f64;
    let mut iter = 0u32;
    let mut done = 0u32;
    while iter < 32u32 && done == 0u32 {
        iter += 1u32;
        let d_r1 = draw_uniform(st);
        st = d_r1.state;
        let r = d_r1.xi;
        let d_r2 = draw_uniform(st);
        st = d_r2.state;
        if 4.0 * (1.0 - r) * r >= d_r2.xi {
            let rel_vel = (electron_ke * (electron_ke + 2.0 * MASS_ELECTRON_EV)).sqrt()
                / (electron_ke + MASS_ELECTRON_EV);
            mu = (2.0 * r + rel_vel - 1.0) / (2.0 * rel_vel * r - rel_vel + 1.0);
            done = 1u32;
        }
    }

    let d_phi = draw_uniform(st);
    st = d_phi.state;
    let phi = d_phi.xi * std::f64::consts::TAU;
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();

    SauterDirection {
        state: st,
        dx: mu,
        dy: sin_theta * cos_f64(phi),
        dz: sin_theta * sin_f64(phi),
    }
}

/// Per-thread test launcher: one direction per seed, at the kinetic energy in
/// `ke_buf[0]`. Writes `(dx, dy, dz)` interleaved into `out`.
#[cube(launch_unchecked)]
fn sauter_direction_kernel(seeds: &[u32], ke_buf: &[f64], out: &mut [f64]) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    // `electron_ke` as a single-element buffer (this cubecl has no scalar
    // launch args; the codebase passes scalars this way).
    let ke = ke_buf[0usize];
    let s = sample_sauter_direction(expand_seed(seeds[ABSOLUTE_POS]), ke);
    out[ABSOLUTE_POS * 3] = s.dx;
    out[ABSOLUTE_POS * 3 + 1] = s.dy;
    out[ABSOLUTE_POS * 3 + 2] = s.dz;
}

/// Run the Sauter direction sampler once per seed at kinetic energy
/// `electron_ke` (eV). Returns one `[dx, dy, dz]` per seed.
pub fn run_sauter_direction(ctx: &GpuContext, seeds: &[u32], electron_ke: f64) -> Vec<[f64; 3]> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let ke_handle = client.create_from_slice(bytemuck::cast_slice(&[electron_ke]));
    let out_handle = client.empty(n * 3 * std::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        sauter_direction_kernel::launch_unchecked(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(ke_handle, 1),
            BufferArg::from_raw_parts(out_handle.clone(), n * 3),
        );
    }

    let flat: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_handle).unwrap()).to_vec();
    flat.as_chunks::<3>().0.to_vec()
}
