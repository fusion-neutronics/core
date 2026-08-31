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

use crate::common::poly_solve::{quartic_smallest_positive, quartic_smallest_positive_cpu};
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

/// Smallest positive ray-torus intersection in axis-permuted
/// coordinates: `(t1, t2)` are the transverse displacements from the
/// torus centre, `ax` the axial one, with matching direction
/// components `(dt1, dt2, dax)`. `a` is the major radius, `b` the
/// axial minor, `c` the radial minor (the `yamc-geo` convention).
///
/// Scales the axial coordinate by `c/b` to reduce an elliptical torus
/// to a circular one of minor radius `c`, builds the quartic in the
/// ray parameter, and returns its smallest positive root via
/// [`quartic_smallest_positive`].
#[cube]
#[allow(clippy::too_many_arguments)]
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
) -> f64 {
    let mut ax_scale = 1.0_f64;
    let bc = b - c;
    let mut abs_bc = bc;
    if abs_bc < 0.0 {
        abs_bc = -bc;
    }
    if abs_bc > 1e-14 {
        ax_scale = c / b;
    }
    let pax_s = ax * ax_scale;
    let dax_s = dax * ax_scale;
    let mq = dt1 * dt1 + dt2 * dt2 + dax_s * dax_s;
    let kq = t1 * dt1 + t2 * dt2 + pax_s * dax_s;
    let nq = t1 * t1 + t2 * t2 + pax_s * pax_s;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = t1 * dt1 + t2 * dt2;
    let n_t = t1 * t1 + t2 * t2;
    let big_s = nq + a * a - c * c;
    let c4 = mq * mq;
    let c3 = 4.0 * mq * kq;
    let c2 = 4.0 * kq * kq + 2.0 * mq * big_s - 4.0 * a * a * m_t;
    let c1 = 4.0 * kq * big_s - 8.0 * a * a * k_t;
    let c0 = big_s * big_s - 4.0 * a * a * n_t;
    let mut result = 1e30_f64;
    let mut abs_c4 = c4;
    if abs_c4 < 0.0 {
        abs_c4 = -c4;
    }
    if abs_c4 > 1e-30 {
        let bn = c3 / c4;
        let cn = c2 / c4;
        let dn = c1 / c4;
        let en = c0 / c4;
        let t = quartic_smallest_positive(bn, cn, dn, en);
        if t < 1e30_f64 {
            result = t;
        }
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
) -> f64 {
    let ax_scale = if (b - c).abs() > 1e-14 { c / b } else { 1.0 };
    let pax_s = ax * ax_scale;
    let dax_s = dax * ax_scale;
    let mq = dt1 * dt1 + dt2 * dt2 + dax_s * dax_s;
    let kq = t1 * dt1 + t2 * dt2 + pax_s * dax_s;
    let nq = t1 * t1 + t2 * t2 + pax_s * pax_s;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = t1 * dt1 + t2 * dt2;
    let n_t = t1 * t1 + t2 * t2;
    let big_s = nq + a * a - c * c;
    let c4 = mq * mq;
    let c3 = 4.0 * mq * kq;
    let c2 = 4.0 * kq * kq + 2.0 * mq * big_s - 4.0 * a * a * m_t;
    let c1 = 4.0 * kq * big_s - 8.0 * a * a * k_t;
    let c0 = big_s * big_s - 4.0 * a * a * n_t;
    if c4.abs() < 1e-30 {
        return MISS;
    }
    let t = quartic_smallest_positive_cpu(c3 / c4, c2 / c4, c1 / c4, c0 / c4);
    if t < MISS {
        t
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
                let z_helper =
                    torus_smallest_positive_cpu(px - x0, py - y0, pz - z0, dx, dy, dz, a, b, c);
                agree(z_helper, zt.distance_to_surface(p, d), 1e-7, "ztorus");
                // XTorus: axial = x, transverse = (y, z).
                let x_helper =
                    torus_smallest_positive_cpu(py - y0, pz - z0, px - x0, dy, dz, dx, a, b, c);
                agree(x_helper, xt.distance_to_surface(p, d), 1e-7, "xtorus");
                // YTorus: axial = y, transverse = (x, z).
                let y_helper =
                    torus_smallest_positive_cpu(px - x0, pz - z0, py - y0, dx, dz, dy, a, b, c);
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
}
