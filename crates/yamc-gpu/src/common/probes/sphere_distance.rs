//! Ray-sphere intersection on the GPU. First geometry kernel of the
//! GPU port (Phase C).
//!
//! For a particle at position `P` moving in unit direction `D`, the
//! distance to the first intersection with a sphere centred at `C`
//! with radius `R` is the smaller positive root of
//!
//! ```text
//! |P + t*D - C|^2 = R^2
//! ```
//!
//! Substituting `Q = P - C` and using `|D| = 1` gives the standard
//! quadratic
//!
//! ```text
//! t^2 + 2 (Q·D) t + (|Q|^2 - R^2) = 0
//! ```
//!
//! with roots `t = -b/2 ± sqrt(b²/4 - c)` where `b = 2 Q·D` and
//! `c = |Q|² - R²`. We return the smallest positive root, or
//! `MISS_SENTINEL` if the ray doesn't hit (negative discriminant) or
//! the sphere is entirely behind the particle (both roots negative).
//!
//! # Layout
//!
//! Particle positions and directions are passed as flat `Array<f64>`
//! buffers with stride 3: `[x0,y0,z0, x1,y1,z1, ...]`. This is the
//! simplest layout that maps cleanly to a Pod buffer; SoA per-component
//! layouts (separate `Array<f64>` for x, y, z) would be friendlier for
//! coalesced memory access on warps and is a follow-up.
//!
//! # Why no `f64::INFINITY` sentinel
//!
//! Cubecl's tracking issue tracelai/cubecl#68 documents that
//! `F64::new(f64::INFINITY)` doesn't expand cleanly. To avoid it, this
//! kernel uses a large finite sentinel `MISS_SENTINEL = 1e30` for
//! "no intersection" cases. Callers check `dist >= MISS_SENTINEL` to
//! detect misses. f64 has plenty of headroom above `1e30` so this
//! doesn't collide with any realistic transport distance.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Sentinel value for "ray misses sphere" or "sphere is behind ray".
/// Caller treats `result >= MISS_SENTINEL` as a miss.
pub const MISS_SENTINEL: f64 = 1e30;

/// Per-thread sphere-distance computation. Each thread reads its
/// particle's position and direction from the flat buffers (stride 3),
/// computes the intersection, writes the distance.
#[cube(launch_unchecked)]
fn sphere_distance_kernel(
    positions: &[f64],
    directions: &[f64],
    params: &[f64],
    distances: &mut [f64],
) {
    if ABSOLUTE_POS >= distances.len() {
        terminate!();
    }

    // Stride-3 unpack of position and direction.
    let i3 = ABSOLUTE_POS * 3;
    let px = positions[i3];
    let py = positions[i3 + 1];
    let pz = positions[i3 + 2];
    let dx = directions[i3];
    let dy = directions[i3 + 1];
    let dz = directions[i3 + 2];

    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let r = params[3];

    let qx = px - cx;
    let qy = py - cy;
    let qz = pz - cz;

    // |D|^2 == 1 by precondition; no normalisation here.
    let b = 2.0 * (qx * dx + qy * dy + qz * dz);
    let c = qx * qx + qy * qy + qz * qz - r * r;

    // Discriminant for `t² + b·t + c = 0`. Standard form discriminant
    // is `b² - 4c` (a = 1).
    let disc = b * b - 4.0 * c;

    // Default to miss; sequential `if`s update if the ray actually hits.
    // The if-else-if shape that would be natural here trips cubecl's
    // macro on type unification, so write it imperatively. Use a plain
    // f64 literal for the sentinel (referencing the `MISS_SENTINEL`
    // const wedges the macro into a NativeExpand<f64> conversion that
    // it can't resolve).
    let mut dist = 1e30_f64;
    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) * 0.5;
        let t2 = (-b + sqrt_disc) * 0.5;
        // We want the smallest positive root. By construction t1 ≤ t2
        // (since sqrt_disc ≥ 0). Try t2 first, then overwrite with t1
        // if it's also positive.
        if t2 > 0.0 {
            dist = t2;
        }
        if t1 > 0.0 {
            dist = t1;
        }
    }
    distances[ABSOLUTE_POS] = dist;
}

/// Run the sphere-distance kernel. `positions` and `directions` are
/// flat stride-3 slices (`[x0,y0,z0, x1,y1,z1, ...]`) with `n_particles
/// * 3` elements each. `params = [cx, cy, cz, r]`. Output is one f64
/// per particle: either the distance to the first hit or
/// `MISS_SENTINEL` for misses.
pub fn run_sphere_distance(
    ctx: &GpuContext,
    positions: &[f64],
    directions: &[f64],
    sphere_params: &[f64; 4],
) -> Vec<f64> {
    assert_eq!(
        positions.len(),
        directions.len(),
        "positions and directions must be the same length"
    );
    assert!(
        positions.len().is_multiple_of(3),
        "positions length must be a multiple of 3 (stride-3 packed)"
    );
    let n = positions.len() / 3;

    let client = ctx.client();
    let positions_handle = client.create_from_slice(bytemuck::cast_slice(positions));
    let directions_handle = client.create_from_slice(bytemuck::cast_slice(directions));
    let params_handle = client.create_from_slice(bytemuck::cast_slice(sphere_params));
    let out_handle = client.empty(n * core::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        sphere_distance_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(positions_handle, positions.len()),
            BufferArg::from_raw_parts(directions_handle, directions.len()),
            BufferArg::from_raw_parts(params_handle, sphere_params.len()),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// CPU equivalent for testing and benchmarking. Same algorithm as the
/// GPU kernel; libm `sqrt` instead of cubecl's `sqrt` (which we
/// validated is bit-exact on this driver, so the difference is purely
/// FMA contraction.)
pub fn run_sphere_distance_cpu(
    positions: &[f64],
    directions: &[f64],
    sphere_params: &[f64; 4],
) -> Vec<f64> {
    let n = positions.len() / 3;
    let cx = sphere_params[0];
    let cy = sphere_params[1];
    let cz = sphere_params[2];
    let r = sphere_params[3];

    (0..n)
        .map(|i| {
            let i3 = i * 3;
            let qx = positions[i3] - cx;
            let qy = positions[i3 + 1] - cy;
            let qz = positions[i3 + 2] - cz;
            let dx = directions[i3];
            let dy = directions[i3 + 1];
            let dz = directions[i3 + 2];
            let b = 2.0 * (qx * dx + qy * dy + qz * dz);
            let c = qx * qx + qy * qy + qz * qz - r * r;
            let disc = b * b - 4.0 * c;
            if disc < 0.0 {
                return MISS_SENTINEL;
            }
            let sqrt_disc = disc.sqrt();
            let t1 = (-b - sqrt_disc) * 0.5;
            let t2 = (-b + sqrt_disc) * 0.5;
            if t1 > 0.0 {
                t1
            } else if t2 > 0.0 {
                t2
            } else {
                MISS_SENTINEL
            }
        })
        .collect()
}

/// Rayon-parallel CPU equivalent.
pub fn run_sphere_distance_cpu_rayon(
    positions: &[f64],
    directions: &[f64],
    sphere_params: &[f64; 4],
) -> Vec<f64> {
    use rayon::prelude::*;
    let n = positions.len() / 3;
    let cx = sphere_params[0];
    let cy = sphere_params[1];
    let cz = sphere_params[2];
    let r = sphere_params[3];

    (0..n)
        .into_par_iter()
        .map(|i| {
            let i3 = i * 3;
            let qx = positions[i3] - cx;
            let qy = positions[i3 + 1] - cy;
            let qz = positions[i3 + 2] - cz;
            let dx = directions[i3];
            let dy = directions[i3 + 1];
            let dz = directions[i3 + 2];
            let b = 2.0 * (qx * dx + qy * dy + qz * dz);
            let c = qx * qx + qy * qy + qz * qz - r * r;
            let disc = b * b - 4.0 * c;
            if disc < 0.0 {
                return MISS_SENTINEL;
            }
            let sqrt_disc = disc.sqrt();
            let t1 = (-b - sqrt_disc) * 0.5;
            let t2 = (-b + sqrt_disc) * 0.5;
            if t1 > 0.0 {
                t1
            } else if t2 > 0.0 {
                t2
            } else {
                MISS_SENTINEL
            }
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Particle outside a unit sphere pointing straight at its centre.
    /// Analytic answer: distance = `|P| - 1.0`. Run a few different
    /// stand-off distances; expect bit-equality (or very close -- sqrt
    /// is exact in cubecl-spirv on this driver, FMA contraction is the
    /// only drift source).
    #[test]
    fn gpu_sphere_distance_head_on_hits() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let r = 1.0_f64;
        // 5 particles, each on the +x axis at increasing distance,
        // pointing in -x direction (straight at the sphere centre).
        let stand_offs = [2.0_f64, 3.0, 5.0, 10.0, 100.0];
        let mut positions = Vec::with_capacity(stand_offs.len() * 3);
        let mut directions = Vec::with_capacity(stand_offs.len() * 3);
        for &d in &stand_offs {
            positions.extend_from_slice(&[d, 0.0, 0.0]);
            directions.extend_from_slice(&[-1.0, 0.0, 0.0]);
        }
        let params = [0.0, 0.0, 0.0, r];

        let gpu = run_sphere_distance(&ctx, &positions, &directions, &params);
        for (i, &d) in stand_offs.iter().enumerate() {
            let expected = d - r;
            let got = gpu[i];
            let ulps = if got.is_finite() && expected.is_finite() {
                let g = got.to_bits();
                let e = expected.to_bits();
                g.max(e) - g.min(e)
            } else {
                u64::MAX
            };
            assert!(
                ulps <= 4,
                "head-on hit at stand-off {d}: GPU = {got}, expected {expected}, ulps = {ulps}"
            );
        }
    }

    /// Particle outside the sphere pointing away. Both roots negative,
    /// so the kernel must return the miss sentinel.
    #[test]
    fn gpu_sphere_distance_facing_away_misses() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let r = 1.0_f64;
        let positions = [2.0, 0.0, 0.0]; // outside on +x
        let directions = [1.0, 0.0, 0.0]; // pointing further away
        let params = [0.0, 0.0, 0.0, r];
        let gpu = run_sphere_distance(&ctx, &positions, &directions, &params);
        assert_eq!(gpu.len(), 1);
        assert!(
            gpu[0] >= MISS_SENTINEL,
            "facing-away ray should miss (got {})",
            gpu[0]
        );
    }

    /// Particle outside the sphere pointing tangent (never crosses
    /// surface). Discriminant negative; should miss.
    #[test]
    fn gpu_sphere_distance_grazing_ray_misses() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let r = 1.0_f64;
        // Particle at (0, 2, 0), pointing in +x. Closest approach to
        // origin is y = 2, well outside the unit sphere.
        let positions = [0.0, 2.0, 0.0];
        let directions = [1.0, 0.0, 0.0];
        let params = [0.0, 0.0, 0.0, r];
        let gpu = run_sphere_distance(&ctx, &positions, &directions, &params);
        assert!(
            gpu[0] >= MISS_SENTINEL,
            "grazing ray should miss (got {})",
            gpu[0]
        );
    }

    /// Particle inside the sphere; should return the distance to the
    /// far side (the positive root). Inside `(0, 0, 0)`, pointing
    /// along +x, the far hit is at `x = r` so distance = r.
    #[test]
    fn gpu_sphere_distance_inside_hits_far_side() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let r = 1.0_f64;
        let positions = [0.0, 0.0, 0.0];
        let directions = [1.0, 0.0, 0.0];
        let params = [0.0, 0.0, 0.0, r];
        let gpu = run_sphere_distance(&ctx, &positions, &directions, &params);
        let ulps = {
            let g = gpu[0].to_bits();
            let e = r.to_bits();
            g.max(e) - g.min(e)
        };
        assert!(
            ulps <= 4,
            "inside-sphere ray should reach far side at distance r = {r}, got {} (ulps {ulps})",
            gpu[0]
        );
    }

    /// 1000 random rays that all hit the sphere; GPU and CPU must
    /// agree to within ~16 ulps. The arithmetic chain is long enough
    /// (`b² - 4c`, sqrt, `(-b ± sqrt)/2`) that FMA contraction adds a
    /// few ulps per step. 16 ulps is comfortable; 16 ulps in f64 is
    /// ~3.5e-15 relative error, vastly below MC noise.
    #[test]
    fn gpu_sphere_distance_matches_cpu_for_random_hits() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };

        let r = 1.0_f64;
        let n = 1000usize;
        // Spawn each particle on a random point on the +x hemisphere
        // at a known stand-off, all pointing towards origin. By
        // construction every ray hits.
        let mut positions = Vec::with_capacity(n * 3);
        let mut directions = Vec::with_capacity(n * 3);
        for i in 0..n {
            // Pseudo-random direction on a sphere via golden-ratio
            // spiral. Doesn't matter that it's not Monte Carlo; we just
            // need diverse rays that all hit.
            let t = (i as f64 + 0.5) / n as f64;
            let phi = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt()) * (i as f64);
            let theta = (1.0 - 2.0 * t).acos();
            let standoff = 2.5_f64 + (i as f64 % 3.0);
            let px = standoff * theta.sin() * phi.cos();
            let py = standoff * theta.sin() * phi.sin();
            let pz = standoff * theta.cos();
            positions.extend_from_slice(&[px, py, pz]);
            // Point at the origin: direction = -P / |P|.
            let mag = (px * px + py * py + pz * pz).sqrt();
            directions.extend_from_slice(&[-px / mag, -py / mag, -pz / mag]);
        }
        let params = [0.0, 0.0, 0.0, r];

        let gpu = run_sphere_distance(&ctx, &positions, &directions, &params);
        let cpu = run_sphere_distance_cpu(&positions, &directions, &params);

        let mut max_ulps = 0u64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(g < MISS_SENTINEL && c < MISS_SENTINEL, "ray {i} missed");
            let gi = g.to_bits();
            let ci = c.to_bits();
            let d = gi.max(ci) - gi.min(ci);
            if d > max_ulps {
                max_ulps = d;
                worst_idx = i;
            }
        }
        println!(
            "sphere_distance: max ULP drift = {max_ulps} (idx {worst_idx}, GPU = {}, CPU = {})",
            gpu[worst_idx], cpu[worst_idx]
        );
        assert!(
            max_ulps <= 16,
            "GPU vs CPU sphere distance drifts > 16 ulps (max {max_ulps})"
        );
    }
}
