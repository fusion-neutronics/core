//! Distance-to-boundary kernel for multi-cell transport.
//!
//! For a particle at position `P` moving in direction `D`, compute
//! the minimum positive distance to any surface in a flat surface
//! set. All eight `yamc-geo` surface kinds are supported:
//!
//! - **Sphere**: `(P + t·D - C)² = R²`, smaller positive root
//!   (or larger if the particle starts inside).
//! - **Plane**: `n · (P + t·D) = d`, where `n` is the unit normal
//!   and `d` is the offset; `t = (d - n·P) / (n·D)`.
//! - **Cylinder**: general-axis cylinder. With `q = P − O` and
//!   `q⊥ = q − (q·a)·a`, `D⊥ = D − (D·a)·a`,
//!   the intersection is the quadratic
//!   `|D⊥|² t² + 2(D⊥·q⊥) t + (|q⊥|² − r²) = 0`.
//! - **X/Y/ZTorus** (axis-symmetric, possibly elliptical): the
//!   ray-torus intersection is a quartic in `t`, built from
//!   `(major a, axial minor b, radial minor c)` and solved via
//!   `quartic_smallest_positive` (Ferrari + Newton). All three route
//!   through `surface_distance::torus_smallest_positive` with their
//!   transverse/axial coordinates permuted.
//! - **Quadric** (general): `Σ` of the ten quadric terms reduces to a
//!   quadratic in `t`.
//! - **Cone** (arbitrary-axis double cone): also a quadratic in `t`.
//!
//! All reduce to a single positive scalar -- the kernel takes the min
//! across the whole surface set per particle.
//!
//! # Layout
//!
//! - `surface_types[i]` is one of the `SURFACE_*` discriminants below.
//! - `surface_params` is stride-`SURFACE_PARAM_STRIDE` (10): ten `f64`
//!   per surface, padded with zeros past each surface's used slots.
//!   The wide stride exists so the general quadric's ten coefficients
//!   fit; the other surfaces leave the tail unused.
//!   - Sphere: `[cx, cy, cz, r, …]`
//!   - Plane: `[nx, ny, nz, d, …]` (caller normalises `n`)
//!   - Cylinder: `[ox, oy, oz, r, ax, ay, az, …]` (axis must be unit)
//!   - ZTorus: `[x0, y0, z0, a, b, c, …]` (a major, b axial minor,
//!     c radial minor, matches `yamc-geo::SurfaceKind::ZTorus`)
//!   - X/YTorus: `[x0, y0, z0, a, b, c, …]` (same as ZTorus; the
//!     kernel permutes coordinates by axis)
//!   - Quadric: `[a, b, c, d, e, f, g, h, j, k]` (all ten slots)
//!   - Cone: `[apex_x, apex_y, apex_z, ax, ay, az, tan2θ, …]`
//!     (axis must be unit)
//!
//! # First-cut limits
//!
//! - **Single surface set per kernel.** Every particle sees the same
//!   list of surfaces. A real per-cell surface list (different cells
//!   bounded by different surface subsets) would need either an
//!   indexed offset array or per-cell scratch buffers; both are
//!   straightforward extensions once cell-finding is wired into the
//!   transport loop.
//! - **No surface sense.** CSG cells in yamc track which side
//!   ("sense") of each bounding surface they're on. This kernel just
//!   returns the distance to the nearest surface -- it doesn't decide
//!   which side the particle ended up on. The transport loop can
//!   re-find the cell after crossing.
//! - **`MISS_SENTINEL = 1e30`** for "no forward intersection with any
//!   surface in the set."

use crate::common::geometry::surface_distance::{
    cone_smallest_positive, cone_smallest_positive_cpu, quadric_smallest_positive,
    quadric_smallest_positive_cpu, torus_smallest_positive, torus_smallest_positive_cpu,
};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Distance returned when no surface in the set is hit forward.
pub const MISS_SENTINEL: f64 = 1e30;

/// Surface-type discriminants. Match the `surface_types` array values
/// and `yamc-geo::SurfaceKind`. Mirrored in `yamc::gpu::translate`.
pub const SURFACE_SPHERE: u32 = 0;
pub const SURFACE_PLANE: u32 = 1;
pub const SURFACE_CYLINDER: u32 = 2;
pub const SURFACE_ZTORUS: u32 = 3;
pub const SURFACE_XTORUS: u32 = 4;
pub const SURFACE_YTORUS: u32 = 5;
pub const SURFACE_QUADRIC: u32 = 6;
pub const SURFACE_CONE: u32 = 7;

/// Number of `f64` per surface in `surface_params`. Wide enough for
/// the general quadric's ten coefficients; the other surfaces use a
/// prefix and leave the tail as zero padding (sphere/plane use four,
/// cylinder seven, the tori six, the cone seven).
pub const SURFACE_PARAM_STRIDE: usize = 10;

#[cube(launch_unchecked)]
fn boundary_distance_kernel(
    positions: &[f64],
    directions: &[f64],
    surface_types: &[u32],
    surface_params: &[f64],
    distances: &mut [f64],
) {
    if ABSOLUTE_POS >= distances.len() {
        terminate!();
    }

    let i3 = ABSOLUTE_POS * 3;
    let px = positions[i3];
    let py = positions[i3 + 1];
    let pz = positions[i3 + 2];
    let dx = directions[i3];
    let dy = directions[i3 + 1];
    let dz = directions[i3 + 2];

    let n_surfaces = surface_types.len() as u32;
    let mut min_dist = 1e30_f64;
    let mut s = 0u32;
    while s < n_surfaces {
        let stype = surface_types[s as usize];
        let base = (s * 10u32) as usize;
        let p0 = surface_params[base];
        let p1 = surface_params[base + 1];
        let p2 = surface_params[base + 2];
        let p3 = surface_params[base + 3];
        let p4 = surface_params[base + 4];
        let p5 = surface_params[base + 5];
        let p6 = surface_params[base + 6];
        let p7 = surface_params[base + 7];
        let p8 = surface_params[base + 8];
        let p9 = surface_params[base + 9];

        let mut hit = 1e30_f64;

        if stype == SURFACE_SPHERE {
            // p0..p2 = centre, p3 = radius. Quadratic
            // |P + tD - C|² = R²; smallest positive root.
            let qx = px - p0;
            let qy = py - p1;
            let qz = pz - p2;
            let b = 2.0 * (qx * dx + qy * dy + qz * dz);
            let c = qx * qx + qy * qy + qz * qz - p3 * p3;
            let disc = b * b - 4.0 * c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let t1 = (-b - sqrt_disc) * 0.5;
                let t2 = (-b + sqrt_disc) * 0.5;
                if t2 > 1e-12 {
                    hit = t2;
                }
                if t1 > 1e-12 {
                    hit = t1;
                }
            }
        }
        if stype == SURFACE_PLANE {
            // p0..p2 = unit normal, p3 = offset.
            // t = (d - n·P) / (n·D), only if n·D != 0 and t > 0.
            let n_dot_d = p0 * dx + p1 * dy + p2 * dz;
            if n_dot_d != 0.0 {
                let n_dot_p = p0 * px + p1 * py + p2 * pz;
                let t = (p3 - n_dot_p) / n_dot_d;
                if t > 1e-12 {
                    hit = t;
                }
            }
        }
        if stype == SURFACE_ZTORUS {
            // p0..p2 = centre, p3 = a (major), p4 = b (axial
            // minor), p5 = c (radial minor). Axial coordinate is z;
            // transverse pair is (x, y).
            hit = torus_smallest_positive(px - p0, py - p1, pz - p2, dx, dy, dz, p3, p4, p5);
        }
        if stype == SURFACE_XTORUS {
            // Axial coordinate is x; transverse pair is (y, z).
            hit = torus_smallest_positive(py - p1, pz - p2, px - p0, dy, dz, dx, p3, p4, p5);
        }
        if stype == SURFACE_YTORUS {
            // Axial coordinate is y; transverse pair is (x, z).
            hit = torus_smallest_positive(px - p0, pz - p2, py - p1, dx, dz, dy, p3, p4, p5);
        }
        if stype == SURFACE_QUADRIC {
            // p0..p9 = a, b, c, d, e, f, g, h, j, k.
            hit = quadric_smallest_positive(
                px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            );
        }
        if stype == SURFACE_CONE {
            // p0..p2 = apex, p3..p5 = unit axis, p6 = tan²θ.
            hit = cone_smallest_positive(px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6);
        }
        if stype == SURFACE_CYLINDER {
            // p0..p2 = origin O, p3 = radius r, p4..p6 = unit axis a.
            // q = P − O; q⊥ = q − (q·a)a; D⊥ = D − (D·a)a.
            let qx = px - p0;
            let qy = py - p1;
            let qz = pz - p2;
            let q_dot_a = qx * p4 + qy * p5 + qz * p6;
            let mx = qx - q_dot_a * p4;
            let my = qy - q_dot_a * p5;
            let mz = qz - q_dot_a * p6;
            let d_dot_a = dx * p4 + dy * p5 + dz * p6;
            let dxp = dx - d_dot_a * p4;
            let dyp = dy - d_dot_a * p5;
            let dzp = dz - d_dot_a * p6;
            let a_q = dxp * dxp + dyp * dyp + dzp * dzp;
            let b_q = 2.0 * (dxp * mx + dyp * my + dzp * mz);
            let c_q = mx * mx + my * my + mz * mz - p3 * p3;
            // a_q ≈ 0 means the ray is parallel to the cylinder
            // axis: it either runs forever inside or never enters,
            // both of which we report as MISS.
            if a_q > 0.0 {
                let disc = b_q * b_q - 4.0 * a_q * c_q;
                if disc >= 0.0 {
                    let sqrt_disc = disc.sqrt();
                    let inv_2a = 0.5 / a_q;
                    let t1 = (-b_q - sqrt_disc) * inv_2a;
                    let t2 = (-b_q + sqrt_disc) * inv_2a;
                    if t2 > 1e-12 {
                        hit = t2;
                    }
                    if t1 > 1e-12 {
                        hit = t1;
                    }
                }
            }
        }

        if hit < min_dist {
            min_dist = hit;
        }
        s += 1u32;
    }

    distances[ABSOLUTE_POS] = min_dist;
}

/// Run the boundary-distance kernel. Each particle (`positions[i*3..]`,
/// `directions[i*3..]`) gets the min positive distance to any surface
/// in the shared surface set, or `MISS_SENTINEL` if none.
pub fn run_boundary_distance(
    ctx: &GpuContext,
    positions: &[f64],
    directions: &[f64],
    surface_types: &[u32],
    surface_params: &[f64],
) -> Vec<f64> {
    assert_eq!(positions.len(), directions.len());
    assert!(positions.len().is_multiple_of(3));
    assert_eq!(
        surface_types.len() * SURFACE_PARAM_STRIDE,
        surface_params.len()
    );
    let n = positions.len() / 3;

    let client = ctx.client();
    let positions_h = client.create_from_slice(bytemuck::cast_slice(positions));
    let dirs_h = client.create_from_slice(bytemuck::cast_slice(directions));
    let types_h = client.create_from_slice(bytemuck::cast_slice(surface_types));
    let params_h = client.create_from_slice(bytemuck::cast_slice(surface_params));
    let out_h = client.empty(n * core::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        boundary_distance_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(positions_h, positions.len()),
            BufferArg::from_raw_parts(dirs_h, directions.len()),
            BufferArg::from_raw_parts(types_h, surface_types.len()),
            BufferArg::from_raw_parts(params_h, surface_params.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

/// CPU equivalent of `run_boundary_distance`. Same algorithm.
pub fn run_boundary_distance_cpu(
    positions: &[f64],
    directions: &[f64],
    surface_types: &[u32],
    surface_params: &[f64],
) -> Vec<f64> {
    let n = positions.len() / 3;
    let mut out = Vec::with_capacity(n);
    for p in 0..n {
        let i3 = p * 3;
        let px = positions[i3];
        let py = positions[i3 + 1];
        let pz = positions[i3 + 2];
        let dx = directions[i3];
        let dy = directions[i3 + 1];
        let dz = directions[i3 + 2];

        let mut min_dist = MISS_SENTINEL;
        for (s, &stype) in surface_types.iter().enumerate() {
            let base = s * SURFACE_PARAM_STRIDE;
            let p0 = surface_params[base];
            let p1 = surface_params[base + 1];
            let p2 = surface_params[base + 2];
            let p3 = surface_params[base + 3];
            let p4 = surface_params[base + 4];
            let p5 = surface_params[base + 5];
            let p6 = surface_params[base + 6];
            let p7 = surface_params[base + 7];
            let p8 = surface_params[base + 8];
            let p9 = surface_params[base + 9];

            let mut hit = MISS_SENTINEL;
            if stype == SURFACE_SPHERE {
                let qx = px - p0;
                let qy = py - p1;
                let qz = pz - p2;
                let b = 2.0 * (qx * dx + qy * dy + qz * dz);
                let c = qx * qx + qy * qy + qz * qz - p3 * p3;
                let disc = b * b - 4.0 * c;
                if disc >= 0.0 {
                    let sqrt_disc = disc.sqrt();
                    let t1 = (-b - sqrt_disc) * 0.5;
                    let t2 = (-b + sqrt_disc) * 0.5;
                    if t2 > 1e-12 {
                        hit = t2;
                    }
                    if t1 > 1e-12 {
                        hit = t1;
                    }
                }
            } else if stype == SURFACE_PLANE {
                let n_dot_d = p0 * dx + p1 * dy + p2 * dz;
                if n_dot_d != 0.0 {
                    let n_dot_p = p0 * px + p1 * py + p2 * pz;
                    let t = (p3 - n_dot_p) / n_dot_d;
                    if t > 1e-12 {
                        hit = t;
                    }
                }
            } else if stype == SURFACE_CYLINDER {
                let qx = px - p0;
                let qy = py - p1;
                let qz = pz - p2;
                let q_dot_a = qx * p4 + qy * p5 + qz * p6;
                let mx = qx - q_dot_a * p4;
                let my = qy - q_dot_a * p5;
                let mz = qz - q_dot_a * p6;
                let d_dot_a = dx * p4 + dy * p5 + dz * p6;
                let dxp = dx - d_dot_a * p4;
                let dyp = dy - d_dot_a * p5;
                let dzp = dz - d_dot_a * p6;
                let a_q = dxp * dxp + dyp * dyp + dzp * dzp;
                let b_q = 2.0 * (dxp * mx + dyp * my + dzp * mz);
                let c_q = mx * mx + my * my + mz * mz - p3 * p3;
                if a_q > 0.0 {
                    let disc = b_q * b_q - 4.0 * a_q * c_q;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let inv_2a = 0.5 / a_q;
                        let t1 = (-b_q - sqrt_disc) * inv_2a;
                        let t2 = (-b_q + sqrt_disc) * inv_2a;
                        if t2 > 1e-12 {
                            hit = t2;
                        }
                        if t1 > 1e-12 {
                            hit = t1;
                        }
                    }
                }
            } else if stype == SURFACE_ZTORUS {
                hit =
                    torus_smallest_positive_cpu(px - p0, py - p1, pz - p2, dx, dy, dz, p3, p4, p5);
            } else if stype == SURFACE_XTORUS {
                hit =
                    torus_smallest_positive_cpu(py - p1, pz - p2, px - p0, dy, dz, dx, p3, p4, p5);
            } else if stype == SURFACE_YTORUS {
                hit =
                    torus_smallest_positive_cpu(px - p0, pz - p2, py - p1, dx, dz, dy, p3, p4, p5);
            } else if stype == SURFACE_QUADRIC {
                hit = quadric_smallest_positive_cpu(
                    px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                );
            } else if stype == SURFACE_CONE {
                hit =
                    cone_smallest_positive_cpu(px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6);
            }
            if hit < min_dist {
                min_dist = hit;
            }
        }
        out.push(min_dist);
    }
    out
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Repack readable stride-7 surface rows into the kernel's
    /// stride-`SURFACE_PARAM_STRIDE` layout by zero-padding the unused
    /// tail of each surface. Lets the fixtures keep their 7-column rows
    /// while the buffer the kernel reads is the full width.
    fn widen(rows7: &[f64]) -> Vec<f64> {
        assert_eq!(rows7.len() % 7, 0, "fixture rows must be 7 wide");
        rows7
            .as_chunks::<7>()
            .0
            .iter()
            .flat_map(|c| c.iter().copied().chain([0.0; SURFACE_PARAM_STRIDE - 7]))
            .collect()
    }

    /// Single sphere at origin radius 1. Particle at (2,0,0) going
    /// in -x → expected hit at distance 1.
    #[test]
    fn single_sphere_head_on() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let surface_types = vec![SURFACE_SPHERE];
        let surface_params = widen(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let positions = vec![2.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let g = gpu[0];
        assert!((g - 1.0).abs() < 1e-12, "expected 1.0, got {g}");
    }

    /// Single plane: x = 5. Particle at origin going +x → distance 5.
    /// Plane in `n·x = d` form with `n = (1,0,0)`, `d = 5`.
    #[test]
    fn single_plane_head_on() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_PLANE];
        let surface_params = widen(&[1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]);
        let positions = vec![0.0, 0.0, 0.0];
        let directions = vec![1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let g = gpu[0];
        assert!((g - 5.0).abs() < 1e-12, "expected 5.0, got {g}");
    }

    /// Plane behind the particle (going away) → no intersection.
    #[test]
    fn plane_behind_misses() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_PLANE];
        // Plane at x = -3, normal in +x. Particle at origin going +x.
        // (-3 - 0) / (1) = -3 < 0 → miss.
        let surface_params = widen(&[1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0]);
        let positions = vec![0.0, 0.0, 0.0];
        let directions = vec![1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!(gpu[0] >= MISS_SENTINEL);
    }

    /// Mixed sphere + plane; the sphere is closer so it should win.
    #[test]
    fn closer_sphere_beats_plane() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        // Sphere at origin r=1, plane at x = 10. Particle at (2,0,0)
        // going -x. Sphere hit at d=1, plane never hit going -x.
        let surface_types = vec![SURFACE_SPHERE, SURFACE_PLANE];
        let surface_params = widen(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // sphere
            1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, // plane x=10
        ]);
        let positions = vec![2.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!(
            (gpu[0] - 1.0).abs() < 1e-12,
            "sphere should win, got {}",
            gpu[0]
        );
    }

    /// Z-axis cylinder, radius 1 at origin. Particle at (3, 0, 0)
    /// going -x → expected hit at distance 2 (the +x side of the
    /// cylinder). Tests the general-axis cylinder code with the
    /// canonical z-axis case.
    #[test]
    fn z_cylinder_head_on() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_CYLINDER];
        // origin = (0,0,0), r = 1, axis = +z
        let surface_params = widen(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let positions = vec![3.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!((gpu[0] - 2.0).abs() < 1e-12, "expected 2.0, got {}", gpu[0]);
    }

    /// Particle inside a z-axis cylinder of radius 1, at (0.5, 0, 0)
    /// going +x. The cylinder wall is hit at distance 0.5.
    #[test]
    fn cylinder_from_inside() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_CYLINDER];
        let surface_params = widen(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let positions = vec![0.5, 0.0, 0.0];
        let directions = vec![1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!((gpu[0] - 0.5).abs() < 1e-12, "expected 0.5, got {}", gpu[0]);
    }

    /// Ray parallel to the cylinder axis must miss (the kernel
    /// reports MISS when `|D⊥|² == 0`).
    #[test]
    fn cylinder_parallel_axis_misses() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_CYLINDER];
        let surface_params = widen(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        // Particle outside cylinder going parallel to z-axis.
        let positions = vec![3.0, 0.0, 0.0];
        let directions = vec![0.0, 0.0, 1.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!(gpu[0] >= MISS_SENTINEL);
    }

    /// Cylinder with a non-axis-aligned axis (`(1,1,0)/√2`). Ray
    /// from a known offset should match the CPU implementation.
    #[test]
    fn oblique_cylinder_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let surface_types = vec![SURFACE_CYLINDER];
        let surface_params = widen(&[1.0, 2.0, 3.0, 0.5, inv_sqrt2, inv_sqrt2, 0.0]);
        let positions = vec![5.0, 0.0, 0.0, -3.0, 1.5, 4.0, 2.0, 2.5, 1.0];
        let directions = vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // Either both miss or both hit within ulp tolerance.
            if *g >= MISS_SENTINEL || *c >= MISS_SENTINEL {
                assert!(
                    *g >= MISS_SENTINEL && *c >= MISS_SENTINEL,
                    "particle {i}: gpu {g} cpu {c} disagree on miss"
                );
            } else {
                let gi = g.to_bits();
                let ci = c.to_bits();
                let ulps = gi.max(ci) - gi.min(ci);
                assert!(ulps <= 64, "particle {i}: gpu {g} cpu {c} ulps {ulps}");
            }
        }
    }

    /// Circular ZTorus at origin, major radius 3, minor radius 1
    /// (so b = c = 1). Particle at (10, 0, 0) going -x → first hit
    /// of the torus tube on the +x side. Tube center is at radius
    /// 3 from origin, so the +x outer wall is at x = 4 → distance
    /// from x = 10 is 6.
    #[test]
    fn ztorus_head_on_outer_wall() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_ZTORUS];
        let surface_params = widen(&[0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0]);
        let positions = vec![10.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        assert!(
            (gpu[0] - 6.0).abs() < 1e-9,
            "GPU expected 6.0, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 6.0).abs() < 1e-9,
            "CPU expected 6.0, got {}",
            cpu[0]
        );
    }

    /// Elliptical ZTorus: a = 3 (major), b = 1 (axial minor),
    /// c = 0.5 (radial minor). The yamc-geo convention puts the
    /// radial minor on `c`, so the outer ring at z=0 is at
    /// `sqrt(x²+y²) = a + c = 3.5`. Particle at (10,0,0) going −x
    /// → distance 6.5.
    #[test]
    fn ztorus_elliptical_xy_plane_outer() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_ZTORUS];
        let surface_params = widen(&[0.0, 0.0, 0.0, 3.0, 1.0, 0.5, 0.0]);
        let positions = vec![10.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        assert!(
            (gpu[0] - 6.5).abs() < 1e-9,
            "GPU expected 6.5, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 6.5).abs() < 1e-9,
            "CPU expected 6.5, got {}",
            cpu[0]
        );
    }

    /// Particle aimed away from the torus → no hit.
    #[test]
    fn ztorus_miss() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_ZTORUS];
        let surface_params = widen(&[0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0]);
        // Particle far on +x going further in +x -- torus is behind.
        let positions = vec![10.0, 0.0, 5.0];
        let directions = vec![1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        assert!(gpu[0] >= MISS_SENTINEL, "expected miss, got {}", gpu[0]);
    }

    /// 1000 random rays + 4 surfaces (2 spheres + 2 planes); GPU and
    /// CPU must agree to within ~16 ulps (allows for FMA contraction
    /// in the longer arithmetic chains).
    #[test]
    fn gpu_matches_cpu_random_rays() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_SPHERE, SURFACE_SPHERE, SURFACE_PLANE, SURFACE_PLANE];
        let surface_params = widen(&[
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // sphere centred at origin, r=2
            5.0, 5.0, 0.0, 1.5, 0.0, 0.0, 0.0, // sphere centred at (5,5,0), r=1.5
            0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, // plane z=8
            1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, // plane x=10
        ]);

        let n = 1000usize;
        let mut positions = Vec::with_capacity(3 * n);
        let mut directions = Vec::with_capacity(3 * n);
        for i in 0..n {
            // Pseudo-random rays from random points pointing toward
            // a random target -- most rays will hit one of the surfaces.
            let f = i as f64;
            let px = ((f * 0.137) % 12.0) - 6.0;
            let py = ((f * 0.241) % 12.0) - 6.0;
            let pz = ((f * 0.173) % 6.0) - 3.0;
            let tx = ((f * 0.317) % 10.0) - 5.0;
            let ty = ((f * 0.421) % 10.0) - 5.0;
            let tz = ((f * 0.529) % 5.0) - 2.5;
            let mut vx = tx - px;
            let mut vy = ty - py;
            let mut vz = tz - pz;
            let mag = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-9);
            vx /= mag;
            vy /= mag;
            vz /= mag;
            positions.extend_from_slice(&[px, py, pz]);
            directions.extend_from_slice(&[vx, vy, vz]);
        }

        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        let mut max_ulps = 0u64;
        let mut both_miss = 0;
        let mut both_hit = 0;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            if *g >= MISS_SENTINEL && *c >= MISS_SENTINEL {
                both_miss += 1;
                continue;
            }
            // Both should be finite hits; ulp distance for same-sign
            // positive f64.
            let gi = g.to_bits();
            let ci = c.to_bits();
            let d = gi.max(ci) - gi.min(ci);
            if d > max_ulps {
                max_ulps = d;
            }
            both_hit += 1;
        }
        println!("boundary distance: max ULP = {max_ulps}, both-miss = {both_miss}, both-hit = {both_hit}");
        // 128 ulps in f64 is ~3e-14 relative -- still vastly below MC
        // noise. Higher than sphere_distance's 16 because we do the
        // sphere quadratic followed by the plane divide and a min
        // across surfaces, accumulating FMA-induced drift across
        // each step.
        assert!(
            max_ulps <= 128,
            "GPU vs CPU drift > 128 ulps (max {max_ulps})"
        );
    }

    /// Quadric encoding of a sphere `x²+y²+z²-4 = 0` (origin, r=2).
    /// Particle at (5,0,0) going -x hits at distance 3. Exercises the
    /// general-quadric arm and the stride-10 tail slots.
    #[test]
    fn quadric_sphere_head_on() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_QUADRIC];
        let surface_params = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0];
        let positions = vec![5.0, 0.0, 0.0];
        let directions = vec![-1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        assert!(
            (gpu[0] - 3.0).abs() < 1e-9,
            "GPU expected 3.0, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 3.0).abs() < 1e-9,
            "CPU expected 3.0, got {}",
            cpu[0]
        );
    }

    /// 45° double cone about +z at the origin (`tan²θ = 1`). Ray from
    /// (-5, 0, -2) along +x crosses the lower sheet at x = -2 → distance
    /// 3 (the double-sheet case from the yamc-geo cone tests).
    #[test]
    fn cone_double_sheet_head_on() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![SURFACE_CONE];
        // apex (0,0,0), axis +z, tan²θ = 1.
        let surface_params = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let positions = vec![-5.0, 0.0, -2.0];
        let directions = vec![1.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        assert!(
            (gpu[0] - 3.0).abs() < 1e-9,
            "GPU expected 3.0, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 3.0).abs() < 1e-9,
            "CPU expected 3.0, got {}",
            cpu[0]
        );
    }

    /// Circular X- and Y-tori (major 3, minor 1) at the origin. The
    /// outer equator sits at radius a+c = 4 in the torus's transverse
    /// plane, so a head-on ray from 10 units out hits at distance 6. These are
    /// the X/YTorus analogues of `ztorus_head_on_outer_wall`.
    #[test]
    fn xy_torus_head_on_outer_wall() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        // XTorus (axis = x): transverse plane is (y, z). Ray from
        // (0,10,0) along -y hits the outer wall at y = 4 → distance 6.
        let xt_params = vec![0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &[0.0, 10.0, 0.0],
            &[0.0, -1.0, 0.0],
            &[SURFACE_XTORUS],
            &xt_params,
        );
        let cpu = run_boundary_distance_cpu(
            &[0.0, 10.0, 0.0],
            &[0.0, -1.0, 0.0],
            &[SURFACE_XTORUS],
            &xt_params,
        );
        assert!(
            (gpu[0] - 6.0).abs() < 1e-9,
            "XTorus GPU expected 6.0, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 6.0).abs() < 1e-9,
            "XTorus CPU expected 6.0, got {}",
            cpu[0]
        );

        // YTorus (axis = y): transverse plane is (x, z). Ray from
        // (10,0,0) along -x hits the outer wall at x = 4 → distance 6.
        let yt_params = vec![0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let gpu = run_boundary_distance(
            &ctx,
            &[10.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &[SURFACE_YTORUS],
            &yt_params,
        );
        let cpu = run_boundary_distance_cpu(
            &[10.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &[SURFACE_YTORUS],
            &yt_params,
        );
        assert!(
            (gpu[0] - 6.0).abs() < 1e-9,
            "YTorus GPU expected 6.0, got {}",
            gpu[0]
        );
        assert!(
            (cpu[0] - 6.0).abs() < 1e-9,
            "YTorus CPU expected 6.0, got {}",
            cpu[0]
        );
    }

    /// CPU-only end-to-end check of the four new surface arms through
    /// `run_boundary_distance_cpu`: validates the stride-10 indexing
    /// and the dispatch glue without needing a GPU (the analytic
    /// expectations match the GPU tests above).
    #[test]
    fn cpu_mirror_new_surfaces_analytic() {
        // Quadric sphere x²+y²+z²-4=0: (5,0,0) along -x → 3.
        let q = run_boundary_distance_cpu(
            &[5.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &[SURFACE_QUADRIC],
            &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0],
        );
        assert!(
            (q[0] - 3.0).abs() < 1e-9,
            "quadric expected 3.0, got {}",
            q[0]
        );

        // 45° double cone about +z: (-5,0,-2) along +x → 3.
        let c = run_boundary_distance_cpu(
            &[-5.0, 0.0, -2.0],
            &[1.0, 0.0, 0.0],
            &[SURFACE_CONE],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );
        assert!((c[0] - 3.0).abs() < 1e-9, "cone expected 3.0, got {}", c[0]);

        // XTorus (a=3, b=c=1): (0,10,0) along -y → 6.
        let xt = run_boundary_distance_cpu(
            &[0.0, 10.0, 0.0],
            &[0.0, -1.0, 0.0],
            &[SURFACE_XTORUS],
            &[0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert!(
            (xt[0] - 6.0).abs() < 1e-9,
            "xtorus expected 6.0, got {}",
            xt[0]
        );

        // YTorus (a=3, b=c=1): (10,0,0) along -x → 6.
        let yt = run_boundary_distance_cpu(
            &[10.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &[SURFACE_YTORUS],
            &[0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert!(
            (yt[0] - 6.0).abs() < 1e-9,
            "ytorus expected 6.0, got {}",
            yt[0]
        );
    }

    /// 1000 random rays against a surface set containing **every**
    /// surface kind (sphere, plane, cylinder, all three tori, the
    /// general quadric, and a cone). GPU and CPU must agree per ray:
    /// both miss, or both hit within a small relative tolerance. This
    /// is the all-surface analogue of `gpu_matches_cpu_random_rays`
    /// (which covers only sphere + plane) and the per-step boundary
    /// scan the transport kernels run is exactly this routine, so it
    /// pins CPU/GPU distance agreement for the full surface set.
    #[test]
    fn gpu_matches_cpu_random_rays_all_surfaces() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let surface_types = vec![
            SURFACE_SPHERE,
            SURFACE_PLANE,
            SURFACE_CYLINDER,
            SURFACE_ZTORUS,
            SURFACE_XTORUS,
            SURFACE_YTORUS,
            SURFACE_QUADRIC,
            SURFACE_CONE,
        ];
        let surface_params = vec![
            // sphere: centre origin, r = 2
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            // plane: x = 4
            1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            // z-cylinder: origin, r = 1.5, axis +z
            0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            // z-torus: major 3, minor 0.7 (circular)
            0.0, 0.0, 0.0, 3.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, //
            // x-torus: major 3, minor 0.7
            0.0, 0.0, 0.0, 3.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, //
            // y-torus: major 3, minor 0.7
            0.0, 0.0, 0.0, 3.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, //
            // quadric: ellipsoid x²/4 + y² + z² - 1 = 0
            0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, //
            // cone: apex (0,0,5), axis +z, tan²θ = 0.5
            0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, //
        ];

        let n = 1000usize;
        let mut positions = Vec::with_capacity(3 * n);
        let mut directions = Vec::with_capacity(3 * n);
        for i in 0..n {
            let f = i as f64;
            let px = ((f * 0.137) % 12.0) - 6.0;
            let py = ((f * 0.241) % 12.0) - 6.0;
            let pz = ((f * 0.173) % 8.0) - 4.0;
            let tx = ((f * 0.317) % 8.0) - 4.0;
            let ty = ((f * 0.421) % 8.0) - 4.0;
            let tz = ((f * 0.529) % 8.0) - 4.0;
            let mut vx = tx - px;
            let mut vy = ty - py;
            let mut vz = tz - pz;
            let mag = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-9);
            vx /= mag;
            vy /= mag;
            vz /= mag;
            positions.extend_from_slice(&[px, py, pz]);
            directions.extend_from_slice(&[vx, vy, vz]);
        }

        let gpu = run_boundary_distance(
            &ctx,
            &positions,
            &directions,
            &surface_types,
            &surface_params,
        );
        let cpu =
            run_boundary_distance_cpu(&positions, &directions, &surface_types, &surface_params);
        let mut both_hit = 0;
        let mut both_miss = 0;
        let mut max_rel = 0.0_f64;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if *g >= MISS_SENTINEL || *c >= MISS_SENTINEL {
                assert!(
                    *g >= MISS_SENTINEL && *c >= MISS_SENTINEL,
                    "ray {i}: GPU/CPU disagree on miss (gpu {g}, cpu {c})"
                );
                both_miss += 1;
                continue;
            }
            let rel = (g - c).abs() / c.abs().max(1.0);
            max_rel = max_rel.max(rel);
            both_hit += 1;
        }
        println!(
            "all-surface random rays: both_hit={both_hit} both_miss={both_miss} max_rel={max_rel}"
        );
        // Guard against the fixture silently degenerating to near-all-miss
        // (e.g. a stride regression): most of the 1000 rays must hit.
        assert!(
            both_hit > 100,
            "random-ray fixture degenerated: only {both_hit} hits"
        );
        // 1e-9 relative is far below MC noise; the quartic-based torus
        // arms accumulate more FMA drift across the two backends than
        // the sphere/plane quadratics, so this is looser than the
        // sphere-only ULP bound but still a tight bug catch.
        assert!(max_rel < 1e-9, "GPU vs CPU relative drift {max_rel} > 1e-9");
    }
}
