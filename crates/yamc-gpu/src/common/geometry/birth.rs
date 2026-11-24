//! The kernels' twin of [`yamc_geo::birth::birth_step`]: how far along its
//! flight a particle with no cell yet is located, so a particle born
//! exactly on a CSG surface starts in the cell it is heading into rather
//! than in neither half-space.
//!
//! Uses only comparisons, `+ - * /` and unary `-`, which are validated on
//! cubecl-spirv (see `common::probes::f64_arith`), and the constants of the
//! CPU function, so both backends locate a birth the same way.

use cubecl::prelude::*;
use yamc_geo::birth::{BIRTH_NUDGE, BIRTH_STEP_ULPS, MAX_BIRTH_STEP};

/// `BIRTH_STEP_ULPS * f64::EPSILON`. Both factors are powers of two, so
/// `BIRTH_STEP_ULPS_EPS * |x|` is bit-identical to the CPU's
/// `BIRTH_STEP_ULPS * |x| * f64::EPSILON`.
const BIRTH_STEP_ULPS_EPS: f64 = BIRTH_STEP_ULPS * f64::EPSILON;

/// Largest finite f64: `|x| <= F64_MAX` is the CPU's `x.is_finite()` (false
/// for infinities and NaN).
const F64_MAX: f64 = f64::MAX;

/// The step one coordinate needs: `BIRTH_STEP_ULPS * |x| * EPSILON / |u|`,
/// or 0 where the direction does not change that coordinate or `x` is not
/// finite, as in the CPU loop.
#[cube]
fn axis_birth_step(x: f64, u: f64) -> f64 {
    let mut ax = x;
    if x < 0.0f64 {
        ax = -x;
    }
    let mut au = u;
    if u < 0.0f64 {
        au = -u;
    }
    // Seeded from a runtime value: a literal-initialised f64 local is
    // comptime in cubecl and cannot take the assignment below.
    let mut step = au * 0.0f64;
    if au > 0.0f64 && ax <= F64_MAX {
        step = BIRTH_STEP_ULPS_EPS * ax / au;
    }
    step
}

/// [`yamc_geo::birth::birth_step`] for the point `(px, py, pz)` and the unit
/// direction `(dx, dy, dz)`.
#[cube]
pub fn birth_step(px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64) -> f64 {
    // Runtime-seeded for the same reason as above; `dx` is a finite
    // direction cosine, so this is exactly BIRTH_NUDGE.
    let mut step = dx * 0.0f64 + BIRTH_NUDGE;
    let sx = axis_birth_step(px, dx);
    let sy = axis_birth_step(py, dy);
    let sz = axis_birth_step(pz, dz);
    if sx > step {
        step = sx;
    }
    if sy > step {
        step = sy;
    }
    if sz > step {
        step = sz;
    }
    if step > MAX_BIRTH_STEP {
        step = dx * 0.0f64 + MAX_BIRTH_STEP;
    }
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// One birth step per (point, direction) row of `pd` (stride 6).
    #[cube(launch)]
    fn birth_step_kernel(pd: &[f64], out: &mut [f64]) {
        if ABSOLUTE_POS >= out.len() {
            terminate!();
        }
        let b = ABSOLUTE_POS * 6;
        out[ABSOLUTE_POS] =
            birth_step(pd[b], pd[b + 1], pd[b + 2], pd[b + 3], pd[b + 4], pd[b + 5]);
    }

    fn run_birth_step(ctx: &GpuContext, rows: &[[f64; 6]]) -> Vec<f64> {
        let client = ctx.client();
        let n = rows.len();
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let in_h = client.create_from_slice(bytemuck::cast_slice(&flat));
        let out_h = client.empty(n * std::mem::size_of::<f64>());
        const WORKGROUP_SIZE: u32 = 64;
        let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
        launch!(
            birth_step_kernel,
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(in_h, flat.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
        bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
    }

    fn unit(u: [f64; 3]) -> [f64; 3] {
        let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [u[0] / n, u[1] / n, u[2] / n]
    }

    /// The kernel's birth step is the CPU's: the plain nudge, the widened
    /// step for a direction grazing a far surface, the cap, negative
    /// coordinates and directions, and a zero direction cosine.
    #[test]
    fn gpu_birth_step_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let cases: Vec<([f64; 3], [f64; 3])> = vec![
            ([1.0, 2.0, -3.0], unit([0.3, -0.5, 0.8])),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([500.0, 500.0, 0.0], unit([0.2, -0.4, 0.9])),
            ([500.0, 500.0, 2200.0], unit([-0.95, 0.31, -1e-4])),
            ([500.0, 500.0, 2200.0], unit([1.0, 0.0, 1e-15])),
            ([-2.0e4, 3.0e3, -600.0], unit([1e-5, -1.0, 3.6e-4])),
            ([1.0e-300, -7.5, 1.0e6], unit([0.0, -1e-3, 1.0])),
        ];
        let rows: Vec<[f64; 6]> = cases
            .iter()
            .map(|(p, u)| [p[0], p[1], p[2], u[0], u[1], u[2]])
            .collect();
        let gpu = run_birth_step(&ctx, &rows);
        for ((p, u), g) in cases.iter().zip(&gpu) {
            let cpu = yamc_geo::birth::birth_step(*p, *u);
            // Exact on a correctly rounded f64 divide; allow one unit in the
            // last place for a driver whose divide is not.
            let ulps = g.to_bits().abs_diff(cpu.to_bits());
            assert!(
                ulps <= 1,
                "birth step at {p:?} along {u:?}: GPU {g:e}, CPU {cpu:e}"
            );
        }
    }
}
