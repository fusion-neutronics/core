//! Polar direction rotation by (µ, φ) on the GPU -- given a unit direction
//! and a scattering cosine µ plus azimuthal angle φ (caller supplies
//! cos φ / sin φ), produce the rotated unit direction. This is the formula
//! the photon kernel inlines 4-5x (photon Compton, Compton electron,
//! pair positron, Rayleigh, fluorescence) -- extracted as the single source
//! of truth (issue #309) so it can be unit-tested against the CPU reference
//! `yamc_physics::neutron::interaction::rotate_direction_fast`.
//!
//! General case (direction not at the z-pole):
//! ```text
//! u' = µ·u + sinθ·(u·w·cosφ - v·sinφ) / b      b = sqrt(1 - w²)
//! v' = µ·v + sinθ·(v·w·cosφ + u·sinφ) / b
//! w' = µ·w - sinθ·b·cosφ
//! ```
//! Pole fallback (|w| ≈ 1): `(sinθ·cosφ, sinθ·sinφ, sign(w)·µ)`.
//! The result is renormalized to guard against drift (the inline kernel
//! copies did the same).

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Rotated unit direction.
#[derive(CubeType)]
pub struct RotatedDir {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Rotate the unit direction `(dx, dy, dz)` by polar cosine `mu` and
/// azimuthal angle given as `(cos_phi, sin_phi)`. THE single source of
/// truth, called by the photon kernel's scatter branches and the test
/// launcher.
#[cube]
#[allow(unused_assignments)]
pub fn rotate_mu_phi(dx: f64, dy: f64, dz: f64, mu: f64, cos_phi: f64, sin_phi: f64) -> RotatedDir {
    let sqrt_1_mu2 = (1.0_f64 - mu * mu).max(0.0_f64).sqrt();
    let denom_dz = (1.0_f64 - dz * dz).max(0.0_f64).sqrt();
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    if denom_dz > 1e-12_f64 {
        nx = mu * dx + sqrt_1_mu2 * (dx * dz * cos_phi - dy * sin_phi) / denom_dz;
        ny = mu * dy + sqrt_1_mu2 * (dy * dz * cos_phi + dx * sin_phi) / denom_dz;
        nz = mu * dz - sqrt_1_mu2 * denom_dz * cos_phi;
    } else {
        // Polar-axis fallback.
        nx = sqrt_1_mu2 * cos_phi;
        ny = sqrt_1_mu2 * sin_phi;
        let mut sign_dz = 1.0_f64;
        if dz < 0.0_f64 {
            sign_dz = 0.0_f64 - 1.0_f64;
        }
        nz = mu * sign_dz;
    }
    // Renormalize to guard against drift.
    let nrm = (nx * nx + ny * ny + nz * nz).sqrt();
    if nrm > 0.0_f64 {
        nx /= nrm;
        ny /= nrm;
        nz /= nrm;
    }
    RotatedDir {
        x: nx,
        y: ny,
        z: nz,
    }
}

/// Per-thread test launcher: rotate `dirs[3i..3i+3]` by `(mus[i], phis_cos[i],
/// phis_sin[i])`; write the rotated direction to `out[3i..3i+3]`.
#[cube(launch_unchecked)]
fn rotate_mu_phi_kernel(
    dirs: &[f64],
    mus: &[f64],
    phis_cos: &[f64],
    phis_sin: &[f64],
    out: &mut [f64],
) {
    if ABSOLUTE_POS >= mus.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS;
    let r = rotate_mu_phi(
        dirs[3 * i],
        dirs[3 * i + 1],
        dirs[3 * i + 2],
        mus[i],
        phis_cos[i],
        phis_sin[i],
    );
    out[3 * i] = r.x;
    out[3 * i + 1] = r.y;
    out[3 * i + 2] = r.z;
}

/// Run the rotation for each `(dirs[3i..], mus[i], phis[i])` triple. `phis`
/// are angles in radians (cos/sin computed host-side). Returns the rotated
/// directions flat.
pub fn run_rotate_mu_phi(ctx: &GpuContext, dirs: &[f64], mus: &[f64], phis: &[f64]) -> Vec<f64> {
    assert_eq!(dirs.len(), 3 * mus.len());
    assert_eq!(mus.len(), phis.len());
    let client = ctx.client();
    let n = mus.len();
    let cos_p: Vec<f64> = phis.iter().map(|p| p.cos()).collect();
    let sin_p: Vec<f64> = phis.iter().map(|p| p.sin()).collect();
    let dirs_h = client.create_from_slice(bytemuck::cast_slice(dirs));
    let mus_h = client.create_from_slice(bytemuck::cast_slice(mus));
    let cos_h = client.create_from_slice(bytemuck::cast_slice(&cos_p));
    let sin_h = client.create_from_slice(bytemuck::cast_slice(&sin_p));
    let out_h = client.empty(std::mem::size_of_val(dirs));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        rotate_mu_phi_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(dirs_h, dirs.len()),
            BufferArg::from_raw_parts(mus_h, n),
            BufferArg::from_raw_parts(cos_h, n),
            BufferArg::from_raw_parts(sin_h, n),
            BufferArg::from_raw_parts(out_h.clone(), dirs.len()),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
