//! Shared ray-surface distance helpers for the torus, quadric, and
//! cone surface arms, used by every transport dispatch site (the
//! standalone `boundary_distance` kernel, the neutron GPU kernel and
//! its CPU mirror, and the photon GPU kernel).
//!
//! Each helper returns the smallest positive intersection distance
//! `> 1e-12`, or the `1e30` MISS sentinel if the ray never hits the
//! surface forward. The `1e-12` floor matches `yamc-geo`'s surface
//! solvers so a particle sitting on a surface it just crossed doesn't
//! immediately re-cross it.
//!
//! These are faithful ports of the corresponding `yamc-geo` routines:
//! - [`torus_smallest_positive`] ↔ `yamc_geo::surface::torus::torus_distance`
//! - [`quadric_smallest_positive`] ↔ `yamc_geo::surface::quadric::distance`
//! - [`cone_smallest_positive`] ↔ `yamc_geo::surface::cone::distance`
//!
//! The X/Y/Z tori all route through [`torus_smallest_positive`] with
//! their transverse/axial coordinates permuted by the caller, exactly
//! as the CPU `SurfaceKind::{X,Y,Z}Torus` arms permute into the shared
//! `torus_distance`. Quadric and cone both reduce to a quadratic and
//! share [`quadratic_smallest_positive`] for root selection.
//!
//! Sphere, plane and cylinder are deliberately *not* centralized here:
//! their intersection math is a few lines and stays inlined at each
//! dispatch site (the hottest arms). Only the heavier shapes -- the
//! quartic tori and the quadric/cone quadratic -- are shared, since
//! duplicating their solvers across four kernels is the real cost.
//!
//! Each `#[cube]` function has a hand-mirrored `_cpu` twin (same
//! algorithm in plain Rust) so the rayon CPU transport path produces
//! bit-comparable results, following the established pattern in
//! `poly_solve`.

use crate::common::poly_solve::{
    quadratic_root_max, quadratic_root_max_cpu, quadratic_root_min, quadratic_root_min_cpu,
    quartic_factors, quartic_factors_cpu, NO_ROOT,
};
use cubecl::prelude::*;

/// Distance returned when a surface is not hit forward. Crate-internal;
/// the public dispatch sites expose their own `MISS_SENTINEL` (same value).
pub(crate) const MISS: f64 = 1e30;

/// Smallest positive root `> 1e-12` of `qa·t² + qb·t + qc = 0`, or
/// `MISS` if none. Handles the degenerate linear case (`qa ≈ 0`) and
/// prefers the nearer root, falling back to the farther one when the
/// nearer is behind the particle. Shared by the quadric and cone arms.
#[cube]
pub fn quadratic_smallest_positive(qa: f64, qb: f64, qc: f64) -> f64 {
    let mut result = 1e30_f64;
    let mut abs_qa = qa;
    if abs_qa < 0.0 {
        abs_qa = -qa;
    }
    if abs_qa < 1e-30 {
        // Degenerate to the linear equation qb·t + qc = 0.
        let mut abs_qb = qb;
        if abs_qb < 0.0 {
            abs_qb = -qb;
        }
        if abs_qb >= 1e-30 {
            let t = -qc / qb;
            if t > 1e-12 {
                result = t;
            }
        }
    } else {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            // Divide by 2*qa (a single rounding; 2*qa is exact) rather
            // than multiplying by 0.5/qa, so the GPU path is bit-for-bit
            // identical to the CPU mirror and the yamc-geo oracle.
            let two_a = 2.0 * qa;
            let t1 = (-qb - sqrt_disc) / two_a;
            let t2 = (-qb + sqrt_disc) / two_a;
            let mut lo = t1;
            let mut hi = t2;
            if t2 < t1 {
                lo = t2;
                hi = t1;
            }
            // Prefer the nearer root; fall back to the farther one when
            // the nearer is behind the particle. Set the fallback first
            // so the nearer root overrides it (mirrors the sphere /
            // cylinder root selection in the transport kernels).
            if hi > 1e-12 {
                result = hi;
            }
            if lo > 1e-12 {
                result = lo;
            }
        }
    }
    result
}

/// Crossings closer than this fraction of the torus size to the point
/// where the particle crossed the torus are that crossing. Same value
/// and reasoning as `yamc-geo`'s torus `ON_SURFACE_TOLERANCE`.
pub const TORUS_ON_SURFACE_TOLERANCE: f64 = 1e-9;

/// Smallest positive ray-torus intersection in axis-permuted
/// coordinates: `(t1, t2)` are the transverse displacements from the
/// torus centre, `ax` the axial one, with matching direction
/// components `(dt1, dt2, dax)`. `a` is the major radius, `b` the
/// axial minor, `c` the radial minor (the `yamc-geo` convention).
///
/// Port of `yamc_geo`'s `torus_distance`, which documents the method:
/// the axial coordinate is scaled by `c/b` (elliptical to circular),
/// lengths are divided by `a + c`, the quartic is factorised by
/// [`quartic_factors`] (Algorithm 1010), and roots on the mirror torus
/// (`|q|^2 + a^2 - c^2 < 0` at the hit point) are rejected.
///
/// `on_surface_offset >= 0` says the particle has just crossed this
/// torus and sits that far past the crossing along the ray (the
/// transport nudge); the quartic is then built at the crossing point
/// with its constant term exactly zero, and roots within
/// [`TORUS_ON_SURFACE_TOLERANCE`] of it are dropped. Pass a negative
/// value when the particle is not on the torus.
#[cube]
// `a - a` is a runtime zero (cubecl keeps literal-seeded f64s comptime).
#[allow(clippy::too_many_arguments, clippy::eq_op)]
pub fn torus_smallest_positive(
    t1: f64,
    t2: f64,
    ax: f64,
    dt1: f64,
    dt2: f64,
    dax: f64,
    a: f64,
    b: f64,
    c: f64,
    on_surface_offset: f64,
) -> f64 {
    let on_surface = on_surface_offset >= 0.0;
    // Runtime zero (cubecl keeps literal-seeded `let mut` f64s comptime).
    let rt_zero = a - a;
    let mut offset = rt_zero;
    if on_surface {
        offset = on_surface_offset;
    }
    let size = a + c;
    let inv = 1.0 / size;
    let ax_scale = c / b;
    let x = (t1 - offset * dt1) * inv;
    let y = (t2 - offset * dt2) * inv;
    let z = (ax - offset * dax) * ax_scale * inv;
    let w = dax * ax_scale;
    let ar = a * inv;
    let cr = c * inv;
    let m = dt1 * dt1 + dt2 * dt2 + w * w;
    let k = x * dt1 + y * dt2 + z * w;
    let n = x * x + y * y + z * z;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = x * dt1 + y * dt2;
    let n_t = x * x + y * y;
    let a2 = ar * ar;
    let cr2 = cr * cr;
    let big_s = n + a2 - cr2;
    let c4 = m * m;
    let c3 = 4.0 * m * k;
    let c2 = 4.0 * k * k + 2.0 * m * big_s - 4.0 * a2 * m_t;
    let c1 = 4.0 * k * big_s - 8.0 * a2 * k_t;
    let mut c0 = big_s * big_s - 4.0 * a2 * n_t;
    if on_surface {
        c0 = 0.0;
    }
    let mut floor = (1e-12 + offset) * inv;
    if on_surface && floor < TORUS_ON_SURFACE_TOLERANCE {
        floor = TORUS_ON_SURFACE_TOLERANCE;
    }
    let mut min_t = rt_zero + 1e30;
    // Bounding ball |q| <= a + c (radius 1 here): a ray starting outside
    // that moves away or whose line misses it cannot hit.
    let mut may_hit = c4 != 0.0;
    if n > 1.0 + 1e-9 && (k >= 0.0 || n - k * k / m > 1.0 + 1e-9) {
        may_hit = false;
    }
    if may_hit {
        let f = quartic_factors(c3 / c4, c2 / c4, c1 / c4, c0 / c4);
        if f.real == 1u32 {
            let mut q = 0u32;
            while q < 2u32 {
                let mut alpha = f.alpha1;
                let mut beta = f.beta1;
                if q == 1u32 {
                    alpha = f.alpha2;
                    beta = f.beta2;
                }
                let x_max = quadratic_root_max(alpha, beta);
                if x_max < NO_ROOT {
                    let x_min = quadratic_root_min(beta, x_max);
                    let mut r = 0u32;
                    while r < 2u32 {
                        let mut t = x_max * f.scale;
                        if r == 1u32 {
                            t = x_min * f.scale;
                        }
                        if t > floor && t < min_t {
                            let sx = x + t * dt1;
                            let sy = y + t * dt2;
                            let sz = z + t * w;
                            if sx * sx + sy * sy + sz * sz + a2 - cr2 >= 0.0 {
                                min_t = t;
                            }
                        }
                        r += 1u32;
                    }
                }
                q += 1u32;
            }
        }
    }
    let mut result = rt_zero + MISS;
    if min_t < 1e30_f64 {
        result = min_t * size - offset;
    }
    result
}

/// Smallest positive ray intersection with the general quadric
/// `a x² + b y² + c z² + d xy + e yz + f xz + g x + h y + j z + k = 0`.
/// `(px,py,pz)` is the ray origin (absolute coordinates), `(dx,dy,dz)`
/// the direction. Substituting `P + tD` yields `qa·t² + qb·t + qc = 0`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn quadric_smallest_positive(
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    j: f64,
    k: f64,
) -> f64 {
    let qa = a * dx * dx + b * dy * dy + c * dz * dz + d * dx * dy + e * dy * dz + f * dx * dz;
    let qb = 2.0 * (a * px * dx + b * py * dy + c * pz * dz)
        + d * (px * dy + py * dx)
        + e * (py * dz + pz * dy)
        + f * (px * dz + pz * dx)
        + g * dx
        + h * dy
        + j * dz;
    let qc = a * px * px
        + b * py * py
        + c * pz * pz
        + d * px * py
        + e * py * pz
        + f * px * pz
        + g * px
        + h * py
        + j * pz
        + k;
    quadratic_smallest_positive(qa, qb, qc)
}

/// Smallest positive ray intersection with an arbitrary-axis double
/// cone: points where `|perp(p - apex)|² = tan2θ · ((p - apex)·axis)²`.
/// `axis` must be a unit vector and `(dx,dy,dz)` a unit direction.
/// Substituting `P + tD` yields `qa·t² + qb·t + qc = 0` with
/// `κ = 1 + tan2θ`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cone_smallest_positive(
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    apx: f64,
    apy: f64,
    apz: f64,
    axx: f64,
    axy: f64,
    axz: f64,
    tan2: f64,
) -> f64 {
    let vx = px - apx;
    let vy = py - apy;
    let vz = pz - apz;
    let kappa = 1.0 + tan2;
    let va = vx * axx + vy * axy + vz * axz;
    let da = dx * axx + dy * axy + dz * axz;
    let vd = vx * dx + vy * dy + vz * dz;
    let v2 = vx * vx + vy * vy + vz * vz;
    let qa = 1.0 - kappa * da * da;
    let qb = 2.0 * (vd - kappa * va * da);
    let qc = v2 - kappa * va * va;
    quadratic_smallest_positive(qa, qb, qc)
}

// ----------------------------- CPU mirrors -----------------------------

/// CPU mirror of [`quadratic_smallest_positive`]. Same algorithm.
pub fn quadratic_smallest_positive_cpu(qa: f64, qb: f64, qc: f64) -> f64 {
    if qa.abs() < 1e-30 {
        if qb.abs() < 1e-30 {
            return MISS;
        }
        let t = -qc / qb;
        return if t > 1e-12 { t } else { MISS };
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return MISS;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-qb - sqrt_disc) / (2.0 * qa);
    let t2 = (-qb + sqrt_disc) / (2.0 * qa);
    let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
    if lo > 1e-12 {
        lo
    } else if hi > 1e-12 {
        hi
    } else {
        MISS
    }
}

/// CPU mirror of [`torus_smallest_positive`]. Same algorithm.
#[allow(clippy::too_many_arguments)]
pub fn torus_smallest_positive_cpu(
    t1: f64,
    t2: f64,
    ax: f64,
    dt1: f64,
    dt2: f64,
    dax: f64,
    a: f64,
    b: f64,
    c: f64,
    on_surface_offset: f64,
) -> f64 {
    let on_surface = on_surface_offset >= 0.0;
    let offset = if on_surface { on_surface_offset } else { 0.0 };
    let size = a + c;
    let inv = 1.0 / size;
    let ax_scale = c / b;
    let x = (t1 - offset * dt1) * inv;
    let y = (t2 - offset * dt2) * inv;
    let z = (ax - offset * dax) * ax_scale * inv;
    let w = dax * ax_scale;
    let ar = a * inv;
    let cr = c * inv;
    let m = dt1 * dt1 + dt2 * dt2 + w * w;
    let k = x * dt1 + y * dt2 + z * w;
    let n = x * x + y * y + z * z;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = x * dt1 + y * dt2;
    let n_t = x * x + y * y;
    let a2 = ar * ar;
    let cr2 = cr * cr;
    let big_s = n + a2 - cr2;
    let c4 = m * m;
    let c3 = 4.0 * m * k;
    let c2 = 4.0 * k * k + 2.0 * m * big_s - 4.0 * a2 * m_t;
    let c1 = 4.0 * k * big_s - 8.0 * a2 * k_t;
    let c0 = if on_surface {
        0.0
    } else {
        big_s * big_s - 4.0 * a2 * n_t
    };
    let mut floor = (1e-12 + offset) * inv;
    if on_surface && floor < TORUS_ON_SURFACE_TOLERANCE {
        floor = TORUS_ON_SURFACE_TOLERANCE;
    }
    if c4 == 0.0 || (n > 1.0 + 1e-9 && (k >= 0.0 || n - k * k / m > 1.0 + 1e-9)) {
        return MISS;
    }
    let f = quartic_factors_cpu(c3 / c4, c2 / c4, c1 / c4, c0 / c4);
    if !f.real {
        return MISS;
    }
    let mut min_t = 1e30_f64;
    for (alpha, beta) in [(f.alpha1, f.beta1), (f.alpha2, f.beta2)] {
        let x_max = quadratic_root_max_cpu(alpha, beta);
        if x_max >= NO_ROOT {
            continue;
        }
        let x_min = quadratic_root_min_cpu(beta, x_max);
        for t in [x_max * f.scale, x_min * f.scale] {
            if t > floor && t < min_t {
                let (sx, sy, sz) = (x + t * dt1, y + t * dt2, z + t * w);
                if sx * sx + sy * sy + sz * sz + a2 - cr2 >= 0.0 {
                    min_t = t;
                }
            }
        }
    }
    if min_t < 1e30_f64 {
        min_t * size - offset
    } else {
        MISS
    }
}

/// CPU mirror of [`quadric_smallest_positive`]. Same algorithm.
#[allow(clippy::too_many_arguments)]
pub fn quadric_smallest_positive_cpu(
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    j: f64,
    k: f64,
) -> f64 {
    let qa = a * dx * dx + b * dy * dy + c * dz * dz + d * dx * dy + e * dy * dz + f * dx * dz;
    let qb = 2.0 * (a * px * dx + b * py * dy + c * pz * dz)
        + d * (px * dy + py * dx)
        + e * (py * dz + pz * dy)
        + f * (px * dz + pz * dx)
        + g * dx
        + h * dy
        + j * dz;
    let qc = a * px * px
        + b * py * py
        + c * pz * pz
        + d * px * py
        + e * py * pz
        + f * px * pz
        + g * px
        + h * py
        + j * pz
        + k;
    quadratic_smallest_positive_cpu(qa, qb, qc)
}

/// CPU mirror of [`cone_smallest_positive`]. Same algorithm.
#[allow(clippy::too_many_arguments)]
pub fn cone_smallest_positive_cpu(
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    apx: f64,
    apy: f64,
    apz: f64,
    axx: f64,
    axy: f64,
    axz: f64,
    tan2: f64,
) -> f64 {
    let vx = px - apx;
    let vy = py - apy;
    let vz = pz - apz;
    let kappa = 1.0 + tan2;
    let va = vx * axx + vy * axy + vz * axz;
    let da = dx * axx + dy * axy + dz * axz;
    let vd = vx * dx + vy * dy + vz * dz;
    let v2 = vx * vx + vy * vy + vz * vz;
    let qa = 1.0 - kappa * da * da;
    let qb = 2.0 * (vd - kappa * va * da);
    let qc = v2 - kappa * va * va;
    quadratic_smallest_positive_cpu(qa, qb, qc)
}

/// Test kernel: [`torus_smallest_positive`] per thread. `cases` is
/// stride-10 `[t1, t2, ax, dt1, dt2, dax, a, b, c, on_surface_offset]`.
#[cube(launch_unchecked)]
fn torus_distance_kernel(cases: &[f64], out: &mut [f64]) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS * 10;
    out[ABSOLUTE_POS] = torus_smallest_positive(
        cases[i],
        cases[i + 1],
        cases[i + 2],
        cases[i + 3],
        cases[i + 4],
        cases[i + 5],
        cases[i + 6],
        cases[i + 7],
        cases[i + 8],
        cases[i + 9],
    );
}

/// Run [`torus_smallest_positive`] on the GPU for the parity tests;
/// `cases` as in `torus_distance_kernel`.
pub fn run_torus_distance(ctx: &crate::GpuContext, cases: &[f64]) -> Vec<f64> {
    assert!(cases.len().is_multiple_of(10));
    let n = cases.len() / 10;
    let client = ctx.client();
    let cases_h = client.create_from_slice(bytemuck::cast_slice(cases));
    let out_h = client.empty(n * core::mem::size_of::<f64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        torus_distance_kernel::launch_unchecked(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(cases_h, cases.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------
//
// These validate the CPU mirrors against `yamc-geo`'s surface solvers
// (the oracle). The `#[cube]` twins are validated against these same
// CPU mirrors on real GPU hardware in `boundary_distance.rs`; here we
// pin the math itself, which runs on any host.

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_geo::Surface;

    /// Deterministic direction fan over the unit sphere (same shape as
    /// the `yamc-geo` surface tests).
    fn direction_fan() -> Vec<[f64; 3]> {
        let mut dirs = Vec::new();
        for i in 0..8 {
            for j in 1..8 {
                let phi = std::f64::consts::TAU * i as f64 / 8.0;
                let theta = std::f64::consts::PI * j as f64 / 8.0;
                dirs.push([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        dirs
    }

    /// Compare a helper distance (MISS-sentinel convention) to the
    /// `yamc-geo` oracle (`Option`), with a relative tolerance.
    fn agree(helper: f64, oracle: Option<f64>, rel_tol: f64, ctx: &str) {
        match oracle {
            Some(d) => {
                assert!(helper < MISS, "{ctx}: oracle hit at {d} but helper missed");
                let rel = (helper - d).abs() / d.abs().max(1.0);
                assert!(
                    rel < rel_tol,
                    "{ctx}: helper {helper} vs oracle {d} (rel {rel})"
                );
            }
            None => assert!(
                helper >= MISS,
                "{ctx}: oracle missed but helper hit at {helper}"
            ),
        }
    }

    #[test]
    fn quadric_matches_geo_oracle() {
        // A sphere, a plane, and a z-cylinder, each written as a general
        // quadric. The dedicated geo variants are oracle-tested against
        // each other in yamc-geo, so matching the Quadric variant pins
        // the kernel math.
        let cases: Vec<(Surface, [f64; 10])> = vec![
            // sphere r=3 at (1,-2,0.5): x²+y²+z² -2·1 x +4 y -1 z + (1+4+0.25-9)
            (
                Surface::new_quadric(
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    4.0,
                    -1.0,
                    1.0 + 4.0 + 0.25 - 9.0,
                    None,
                    None,
                ),
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    4.0,
                    -1.0,
                    1.0 + 4.0 + 0.25 - 9.0,
                ],
            ),
            // plane 2x+3y-z-4=0
            (
                Surface::new_quadric(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, -1.0, -4.0, None, None,
                ),
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, -1.0, -4.0],
            ),
            // z-cylinder x²+y²-4=0
            (
                Surface::new_quadric(
                    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, None, None,
                ),
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0],
            ),
        ];
        let points = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-3.0, 1.0, 2.0]];
        for (geo, q) in &cases {
            for p in points {
                for d in direction_fan() {
                    let oracle = geo.distance_to_surface(p, d);
                    let helper = quadric_smallest_positive_cpu(
                        p[0], p[1], p[2], d[0], d[1], d[2], q[0], q[1], q[2], q[3], q[4], q[5],
                        q[6], q[7], q[8], q[9],
                    );
                    agree(helper, oracle, 1e-9, &format!("quadric p={p:?} d={d:?}"));
                }
            }
        }
    }

    #[test]
    fn cone_matches_geo_oracle() {
        // Arbitrary-axis cone and the three axis-aligned constructors.
        let apex = [0.5, -1.0, 2.0];
        let tan2 = 0.25;
        let inv = 1.0 / 3.0_f64.sqrt();
        let cases = [
            (
                Surface::z_cone(apex[0], apex[1], apex[2], tan2, None, None),
                [0.0, 0.0, 1.0],
            ),
            (
                Surface::x_cone(apex[0], apex[1], apex[2], tan2, None, None),
                [1.0, 0.0, 0.0],
            ),
            (
                Surface::y_cone(apex[0], apex[1], apex[2], tan2, None, None),
                [0.0, 1.0, 0.0],
            ),
            (
                Surface::new_cone(apex, [inv, inv, inv], tan2, None, None),
                [inv, inv, inv],
            ),
        ];
        let points = [[0.0, 0.0, 0.0], [3.0, 2.0, -4.0], [0.6, -1.0, 2.1]];
        for (geo, axis) in &cases {
            for p in points {
                for d in direction_fan() {
                    let oracle = geo.distance_to_surface(p, d);
                    let helper = cone_smallest_positive_cpu(
                        p[0], p[1], p[2], d[0], d[1], d[2], apex[0], apex[1], apex[2], axis[0],
                        axis[1], axis[2], tan2,
                    );
                    // 1e-7: the geo cone solver and this one arrange the
                    // quadratic differently (observed drift ~1e-8), so this
                    // still leaves a decade of headroom over the real drift.
                    agree(helper, oracle, 1e-7, &format!("cone p={p:?} d={d:?}"));
                }
            }
        }
    }

    #[test]
    fn torus_matches_geo_oracle() {
        // Elliptical tori about each axis, centred off the origin.
        let (x0, y0, z0, a, b, c) = (1.0, 2.0, 3.0, 3.0, 0.5, 0.8);
        let zt = Surface::new_ztorus(x0, y0, z0, a, b, c, None, None);
        let xt = Surface::new_xtorus(x0, y0, z0, a, b, c, None, None);
        let yt = Surface::new_ytorus(x0, y0, z0, a, b, c, None, None);
        let points = [[5.0, 2.0, 3.0], [1.0, 2.0, 3.0], [-4.0, 5.0, 1.0]];
        for p in points {
            for d in direction_fan() {
                let (px, py, pz) = (p[0], p[1], p[2]);
                let (dx, dy, dz) = (d[0], d[1], d[2]);
                // ZTorus: axial = z, transverse = (x, y).
                let z_helper = torus_smallest_positive_cpu(
                    px - x0,
                    py - y0,
                    pz - z0,
                    dx,
                    dy,
                    dz,
                    a,
                    b,
                    c,
                    -1.0,
                );
                agree(z_helper, zt.distance_to_surface(p, d), 1e-7, "ztorus");
                // XTorus: axial = x, transverse = (y, z).
                let x_helper = torus_smallest_positive_cpu(
                    py - y0,
                    pz - z0,
                    px - x0,
                    dy,
                    dz,
                    dx,
                    a,
                    b,
                    c,
                    -1.0,
                );
                agree(x_helper, xt.distance_to_surface(p, d), 1e-7, "xtorus");
                // YTorus: axial = y, transverse = (x, z).
                let y_helper = torus_smallest_positive_cpu(
                    px - x0,
                    pz - z0,
                    py - y0,
                    dx,
                    dz,
                    dy,
                    a,
                    b,
                    c,
                    -1.0,
                );
                agree(y_helper, yt.distance_to_surface(p, d), 1e-7, "ytorus");
            }
        }
    }

    /// The degenerate-linear branch of the shared quadratic: a quadric
    /// with no quadratic terms (a plane) must hit at the linear root.
    #[test]
    fn quadratic_linear_branch() {
        // 2t + (-4) = 0 -> t = 2.
        assert!((quadratic_smallest_positive_cpu(0.0, 2.0, -4.0) - 2.0).abs() < 1e-12);
        // Linear root behind the particle -> miss.
        assert!(quadratic_smallest_positive_cpu(0.0, 2.0, 4.0) >= MISS);
        // No solution at all (qa = qb = 0) -> miss.
        assert!(quadratic_smallest_positive_cpu(0.0, 0.0, 1.0) >= MISS);
    }

    /// Torus test cases in axis-frame coordinates, stride 10 as the
    /// test kernel takes them, from points on circular and elliptical
    /// tori (unit-sized and at the 907 cm blanket scale) along a
    /// direction fan: flagged exactly on the surface, flagged after the
    /// 1e-8 cm crossing nudge, and unflagged from off-surface points.
    fn torus_cases() -> Vec<[f64; 10]> {
        let mut cases = Vec::new();
        for (a, b, c) in [
            (3.0, 1.0, 1.0),
            (3.0, 0.5, 0.8),
            (907.2, 467.7, 467.7),
            (907.2, 750.0, 467.7),
            (1.0, 1.5, 1.5),
        ] {
            for i in 0..6 {
                let theta = std::f64::consts::TAU * (i as f64 + 0.3) / 6.0;
                let phi = 0.4 + 1.1 * i as f64;
                let rho = a + c * theta.cos();
                let p = [rho * phi.cos(), rho * phi.sin(), b * theta.sin()];
                for d in direction_fan() {
                    cases.push([p[0], p[1], p[2], d[0], d[1], d[2], a, b, c, 0.0]);
                    let q = [p[0] + 1e-8 * d[0], p[1] + 1e-8 * d[1], p[2] + 1e-8 * d[2]];
                    cases.push([q[0], q[1], q[2], d[0], d[1], d[2], a, b, c, 1e-8]);
                    let o = [0.5 * p[0], 0.7 * p[1], 1.3 * p[2]];
                    cases.push([o[0], o[1], o[2], d[0], d[1], d[2], a, b, c, -1.0]);
                }
            }
        }
        cases
    }

    /// The CPU twin agrees with the `yamc-geo` torus distance, on and
    /// off the surface, for all three axes (the twin is called with the
    /// axis permutation each dispatch site applies).
    #[test]
    fn torus_on_surface_matches_geo_oracle() {
        for case in torus_cases() {
            let [p0, p1, p2, d0, d1, d2, a, b, c, off] = case;
            let on = (off >= 0.0).then_some(off);
            let twin = torus_smallest_positive_cpu(p0, p1, p2, d0, d1, d2, a, b, c, off);
            let ctx = format!("{case:?}");
            let zt = Surface::new_ztorus(0.0, 0.0, 0.0, a, b, c, None, None);
            agree(
                twin,
                zt.distance_to_surface_on([p0, p1, p2], [d0, d1, d2], on),
                1e-9,
                &ctx,
            );
            let xt = Surface::new_xtorus(0.0, 0.0, 0.0, a, b, c, None, None);
            let x = xt.distance_to_surface_on([p2, p0, p1], [d2, d0, d1], on);
            agree(twin, x, 1e-9, &ctx);
            let yt = Surface::new_ytorus(0.0, 0.0, 0.0, a, b, c, None, None);
            let y = yt.distance_to_surface_on([p0, p2, p1], [d0, d2, d1], on);
            agree(twin, y, 1e-9, &ctx);
        }
    }

    /// GPU twin parity for the torus: the kernel matches its CPU twin
    /// on every case (identical arithmetic up to the sqrt-based cube
    /// root, so to round-off).
    #[test]
    fn gpu_torus_matches_cpu_twin() {
        let ctx = match crate::GpuContext::new() {
            Ok(c) => c,
            Err(crate::GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let cases = torus_cases();
        let flat: Vec<f64> = cases.iter().flatten().copied().collect();
        let gpu = run_torus_distance(&ctx, &flat);
        for (case, g) in cases.iter().zip(&gpu) {
            let [p0, p1, p2, d0, d1, d2, a, b, c, off] = *case;
            let cpu = torus_smallest_positive_cpu(p0, p1, p2, d0, d1, d2, a, b, c, off);
            let same = *g == cpu || (g - cpu).abs() <= 1e-12 * cpu.abs().max(1.0);
            assert!(same, "{case:?}: gpu {g} cpu {cpu}");
        }
    }
}
