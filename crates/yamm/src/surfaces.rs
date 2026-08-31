//! Parametric surface evaluation (UV → 3-D) without a CAD kernel.
//!
//! The surface counterpart of [`crate::curves`]: Python serializes each
//! face's surface DEFINITION once and Rust evaluates batches of UV
//! points - replacing the per-point OCC `Surface::Value` loops in
//! face evaluation and optimize projection.
//!
//! Parameterizations mirror OCC's exactly (`gp_Pln`, `gp_Cylinder`,
//! `gp_Cone`, `gp_Sphere`, `gp_Torus`, `Geom_BSplineSurface`,
//! `Geom_SurfaceOfLinearExtrusion`, `Geom_SurfaceOfRevolution`). The
//! frame carries explicit X/Y/Z directions, preserving left-handed
//! `gp_Ax3` placements.

use crate::curves::{find_span, Curve3};

/// A placement frame: origin + explicit axes (handedness preserved).
#[derive(Clone, Debug)]
pub struct Frame {
    pub origin: [f64; 3],
    pub x_dir: [f64; 3],
    pub y_dir: [f64; 3],
    pub z_dir: [f64; 3],
}

impl Frame {
    #[inline]
    fn point(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        [
            self.origin[0] + x * self.x_dir[0] + y * self.y_dir[0] + z * self.z_dir[0],
            self.origin[1] + x * self.x_dir[1] + y * self.y_dir[1] + z * self.z_dir[1],
            self.origin[2] + x * self.x_dir[2] + y * self.y_dir[2] + z * self.z_dir[2],
        ]
    }
}

/// A 3-D parametric surface definition.
#[derive(Clone, Debug)]
pub enum Surface3 {
    /// `P(u, v) = origin + u X + v Y`
    Plane { frame: Frame },
    /// `P(u, v) = origin + r cos(u) X + r sin(u) Y + v Z`
    Cylinder { frame: Frame, radius: f64 },
    /// `P(u, v) = origin + (r + v sin a)(cos u X + sin u Y) + v cos(a) Z`
    Cone {
        frame: Frame,
        ref_radius: f64,
        semi_angle: f64,
    },
    /// `P(u, v) = origin + r cos(v)(cos u X + sin u Y) + r sin(v) Z`
    Sphere { frame: Frame, radius: f64 },
    /// `P(u, v) = origin + (R + r cos v)(cos u X + sin u Y) + r sin(v) Z`
    Torus {
        frame: Frame,
        major_radius: f64,
        minor_radius: f64,
    },
    /// Tensor-product NURBS surface; poles are u-major:
    /// `poles[i * n_v_poles + j]` is pole (i, j).
    BSpline {
        degree_u: usize,
        degree_v: usize,
        n_u_poles: usize,
        n_v_poles: usize,
        /// Flat knot vectors WITH multiplicities expanded.
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        poles: Vec<[f64; 3]>,
        weights: Option<Vec<f64>>,
    },
    /// `P(u, v) = C(u) + v * dir`
    Extrusion { basis: Box<Curve3>, dir: [f64; 3] },
    /// `P(u, v)` = basis point `C(v)` rotated by angle `u` around the
    /// axis through `axis_origin` along `axis_dir`.
    Revolution {
        basis: Box<Curve3>,
        axis_origin: [f64; 3],
        axis_dir: [f64; 3],
    },
}

impl Surface3 {
    /// Evaluate the surface position at `(u, v)`.
    pub fn value(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            Surface3::Plane { frame } => frame.point(u, v, 0.0),
            Surface3::Cylinder { frame, radius } => {
                let (s, c) = u.sin_cos();
                frame.point(radius * c, radius * s, v)
            }
            Surface3::Cone {
                frame,
                ref_radius,
                semi_angle,
            } => {
                let (s, c) = u.sin_cos();
                let r = ref_radius + v * semi_angle.sin();
                frame.point(r * c, r * s, v * semi_angle.cos())
            }
            Surface3::Sphere { frame, radius } => {
                let (su, cu) = u.sin_cos();
                let (sv, cv) = v.sin_cos();
                frame.point(radius * cv * cu, radius * cv * su, radius * sv)
            }
            Surface3::Torus {
                frame,
                major_radius,
                minor_radius,
            } => {
                let (su, cu) = u.sin_cos();
                let (sv, cv) = v.sin_cos();
                let r = major_radius + minor_radius * cv;
                frame.point(r * cu, r * su, minor_radius * sv)
            }
            Surface3::BSpline {
                degree_u,
                degree_v,
                n_u_poles,
                n_v_poles,
                knots_u,
                knots_v,
                poles,
                weights,
            } => eval_bspline_surface(
                *degree_u,
                *degree_v,
                *n_u_poles,
                *n_v_poles,
                knots_u,
                knots_v,
                poles,
                weights.as_deref(),
                u,
                v,
            ),
            Surface3::Extrusion { basis, dir } => {
                let p = basis.value(u);
                [p[0] + v * dir[0], p[1] + v * dir[1], p[2] + v * dir[2]]
            }
            Surface3::Revolution {
                basis,
                axis_origin,
                axis_dir,
            } => {
                // Rodrigues rotation of C(v) around the axis by angle u.
                let p = basis.value(v);
                let d = axis_dir;
                let rel = [
                    p[0] - axis_origin[0],
                    p[1] - axis_origin[1],
                    p[2] - axis_origin[2],
                ];
                let (s, c) = u.sin_cos();
                let dot = d[0] * rel[0] + d[1] * rel[1] + d[2] * rel[2];
                let cross = [
                    d[1] * rel[2] - d[2] * rel[1],
                    d[2] * rel[0] - d[0] * rel[2],
                    d[0] * rel[1] - d[1] * rel[0],
                ];
                [
                    axis_origin[0] + rel[0] * c + cross[0] * s + d[0] * dot * (1.0 - c),
                    axis_origin[1] + rel[1] * c + cross[1] * s + d[1] * dot * (1.0 - c),
                    axis_origin[2] + rel[2] * c + cross[2] * s + d[2] * dot * (1.0 - c),
                ]
            }
        }
    }

    /// Evaluate many UV points (the batch API the Python layer uses).
    pub fn values(&self, uvs: &[[f64; 2]]) -> Vec<[f64; 3]> {
        uvs.iter().map(|&[u, v]| self.value(u, v)).collect()
    }
}

/// Tensor-product (rational) B-spline surface evaluation: homogeneous
/// de Boor in u for the affected v-rows, then de Boor in v.
#[allow(clippy::too_many_arguments)]
fn eval_bspline_surface(
    degree_u: usize,
    degree_v: usize,
    _n_u_poles: usize,
    n_v_poles: usize,
    knots_u: &[f64],
    knots_v: &[f64],
    poles: &[[f64; 3]],
    weights: Option<&[f64]>,
    u: f64,
    v: f64,
) -> [f64; 3] {
    let ku = find_span(degree_u, knots_u, u);
    let kv = find_span(degree_v, knots_v, v);
    let u = u.clamp(knots_u[degree_u], knots_u[knots_u.len() - degree_u - 1]);
    let v = v.clamp(knots_v[degree_v], knots_v[knots_v.len() - degree_v - 1]);

    // de Boor in u for each affected v-column -> (degree_v + 1)
    // intermediate homogeneous points.
    let mut col: Vec<([f64; 3], f64)> = Vec::with_capacity(degree_v + 1);
    for jj in 0..=degree_v {
        let j = jj + kv - degree_v;
        // Homogeneous u-direction working set for this v index.
        let mut d: Vec<([f64; 3], f64)> = (0..=degree_u)
            .map(|ii| {
                let i = ii + ku - degree_u;
                let pi = i * n_v_poles + j;
                let w = weights.map_or(1.0, |ws| ws[pi]);
                let p = poles[pi];
                ([p[0] * w, p[1] * w, p[2] * w], w)
            })
            .collect();
        for r in 1..=degree_u {
            for ii in (r..=degree_u).rev() {
                let i = ii + ku - degree_u;
                let denom = knots_u[i + degree_u - r + 1] - knots_u[i];
                let alpha = if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (u - knots_u[i]) / denom
                };
                let (p1, w1) = d[ii - 1];
                let (p0, w0) = d[ii];
                d[ii] = (
                    [
                        (1.0 - alpha) * p1[0] + alpha * p0[0],
                        (1.0 - alpha) * p1[1] + alpha * p0[1],
                        (1.0 - alpha) * p1[2] + alpha * p0[2],
                    ],
                    (1.0 - alpha) * w1 + alpha * w0,
                );
            }
        }
        col.push(d[degree_u]);
    }

    // de Boor in v over the intermediate points.
    for r in 1..=degree_v {
        for jj in (r..=degree_v).rev() {
            let j = jj + kv - degree_v;
            let denom = knots_v[j + degree_v - r + 1] - knots_v[j];
            let alpha = if denom.abs() < 1e-300 {
                0.0
            } else {
                (v - knots_v[j]) / denom
            };
            let (p1, w1) = col[jj - 1];
            let (p0, w0) = col[jj];
            col[jj] = (
                [
                    (1.0 - alpha) * p1[0] + alpha * p0[0],
                    (1.0 - alpha) * p1[1] + alpha * p0[1],
                    (1.0 - alpha) * p1[2] + alpha * p0[2],
                ],
                (1.0 - alpha) * w1 + alpha * w0,
            );
        }
    }
    let (mut p, w) = col[degree_v];
    if weights.is_some() && w.abs() > 1e-300 {
        p[0] /= w;
        p[1] /= w;
        p[2] /= w;
    }
    p
}

/// Curvature-adaptive size-field arrays sampled on an (nu x nv) UV grid.
///
/// Mirrors the Python `compute_size_field` formulas exactly (elements
/// per 2-pi of curvature, optional chordal cap `sqrt(8 tol / kappa)`,
/// min/max clamps, principal-direction angle in the (du, dv) tangent
/// basis). Derivatives come from central finite differences on the
/// exact evaluator - a sizing HEURISTIC needs ~1% curvature accuracy
/// and FD with span-relative steps delivers ~1e-6, uniformly for every
/// surface kind including rational B-splines.
pub struct SizeFieldArrays {
    pub target_h: Vec<f64>,
    pub metric_e: Vec<f64>,
    pub metric_f: Vec<f64>,
    pub metric_g: Vec<f64>,
    pub target_h1: Vec<f64>,
    pub target_h2: Vec<f64>,
    pub curvature_angle: Vec<f64>,
}

/// Sample a curvature-adaptive size field on an `nu x nv` UV grid.
///
/// Returns per-grid-point target edge lengths (isotropic `target_h`
/// plus the two principal-direction sizes and the direction angle) and
/// the first-fundamental-form metric. Derivatives come from central
/// finite differences on [`Surface3::value`] - a sizing heuristic needs
/// ~1% curvature accuracy, which FD delivers comfortably while staying
/// CAD-kernel-free. See [`SizeFieldArrays`] for the field layout.
#[allow(clippy::too_many_arguments)]
pub fn size_field_from_surface(
    surf: &Surface3,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    nu: usize,
    nv: usize,
    elements_per_2pi: f64,
    min_h: f64,
    max_h: Option<f64>,
    chordal_tolerance: Option<f64>,
) -> SizeFieldArrays {
    let n = nu * nv;
    let mut out = SizeFieldArrays {
        target_h: Vec::with_capacity(n),
        metric_e: Vec::with_capacity(n),
        metric_f: Vec::with_capacity(n),
        metric_g: Vec::with_capacity(n),
        target_h1: Vec::with_capacity(n),
        target_h2: Vec::with_capacity(n),
        curvature_angle: Vec::with_capacity(n),
    };
    let hu = ((u1 - u0).abs() * 1e-5).max(1e-9);
    let hv = ((v1 - v0).abs() * 1e-5).max(1e-9);

    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let scale = |a: [f64; 3], s: f64| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    let (ulo, uhi) = if u0 <= u1 { (u0, u1) } else { (u1, u0) };
    let (vlo, vhi) = if v0 <= v1 { (v0, v1) } else { (v1, v0) };
    // Only B-spline-family surfaces clamp evaluation to the knot range
    // (which would corrupt boundary-row FD stencils); analytic surfaces
    // extrapolate exactly AND have meaningful degeneracies at domain
    // edges (sphere poles report curvature-undefined in OCC -> max_h)
    // that an inward shift would mask.
    let clamp_inward = matches!(
        surf,
        Surface3::BSpline { .. } | Surface3::Extrusion { .. } | Surface3::Revolution { .. }
    );
    for iv in 0..nv {
        let vp = v0 + iv as f64 * (v1 - v0) / (nv.max(2) - 1) as f64;
        for iu in 0..nu {
            let up = u0 + iu as f64 * (u1 - u0) / (nu.max(2) - 1) as f64;

            // Pull the FD center inside the domain for B-spline-family
            // surfaces: their evaluation clamps to the knot range, so a
            // stencil straddling the boundary would see
            // f(u1 + h) == f(u1) and halve the derivative (measured:
            // 34x curvature error on a lofted B-spline's edge rows vs
            // OCC). Curvature varies smoothly at the 1e-5-span scale,
            // so the inward shift is invisible to the sizing heuristic.
            let (up, vp) = if clamp_inward {
                (up.clamp(ulo + hu, uhi - hu), vp.clamp(vlo + hv, vhi - hv))
            } else {
                (up, vp)
            };

            // Central-difference first and second derivatives.
            let p = surf.value(up, vp);
            let pu1 = surf.value(up + hu, vp);
            let pu0 = surf.value(up - hu, vp);
            let pv1 = surf.value(up, vp + hv);
            let pv0 = surf.value(up, vp - hv);
            let du = scale(sub(pu1, pu0), 0.5 / hu);
            let dv = scale(sub(pv1, pv0), 0.5 / hv);
            let duu = scale(
                [
                    pu1[0] - 2.0 * p[0] + pu0[0],
                    pu1[1] - 2.0 * p[1] + pu0[1],
                    pu1[2] - 2.0 * p[2] + pu0[2],
                ],
                1.0 / (hu * hu),
            );
            let dvv = scale(
                [
                    pv1[0] - 2.0 * p[0] + pv0[0],
                    pv1[1] - 2.0 * p[1] + pv0[1],
                    pv1[2] - 2.0 * p[2] + pv0[2],
                ],
                1.0 / (hv * hv),
            );
            let ppp = surf.value(up + hu, vp + hv);
            let ppm = surf.value(up + hu, vp - hv);
            let pmp = surf.value(up - hu, vp + hv);
            let pmm = surf.value(up - hu, vp - hv);
            let duv = scale(
                [
                    ppp[0] - ppm[0] - pmp[0] + pmm[0],
                    ppp[1] - ppm[1] - pmp[1] + pmm[1],
                    ppp[2] - ppm[2] - pmp[2] + pmm[2],
                ],
                0.25 / (hu * hv),
            );

            let e = dot(du, du);
            let f = dot(du, dv);
            let g = dot(dv, dv);
            out.metric_e.push(e);
            out.metric_f.push(f);
            out.metric_g.push(g);

            // Second fundamental form via the unit normal.
            let nvec = cross(du, dv);
            let nlen = dot(nvec, nvec).sqrt();
            let (k1, k2, dir1) = if nlen > 1e-14 {
                let un = scale(nvec, 1.0 / nlen);
                let l = dot(duu, un);
                let m = dot(duv, un);
                let nn = dot(dvv, un);
                let det_i = e * g - f * f;
                if det_i.abs() > 1e-20 {
                    // Shape operator S = I^-1 II (2x2).
                    let s11 = (g * l - f * m) / det_i;
                    let s12 = (g * m - f * nn) / det_i;
                    let s21 = (e * m - f * l) / det_i;
                    let s22 = (e * nn - f * m) / det_i;
                    let tr = s11 + s22;
                    let dt = s11 * s22 - s12 * s21;
                    let disc = (tr * tr / 4.0 - dt).max(0.0).sqrt();
                    let ka = tr / 2.0 + disc;
                    let kb = tr / 2.0 - disc;
                    // Eigenvector of S for ka in UV coords -> 3-D direction.
                    let (eu, ev) = if s12.abs() > 1e-14 {
                        (s12, ka - s11)
                    } else if s21.abs() > 1e-14 {
                        (ka - s22, s21)
                    } else {
                        (1.0, 0.0)
                    };
                    let d3 = [
                        eu * du[0] + ev * dv[0],
                        eu * du[1] + ev * dv[1],
                        eu * du[2] + ev * dv[2],
                    ];
                    (ka, kb, d3)
                } else {
                    (0.0, 0.0, du)
                }
            } else {
                (0.0, 0.0, du)
            };

            let kappa = k1.abs().max(k2.abs());
            let cap = |mut h: f64, k: f64| {
                if let Some(tol) = chordal_tolerance {
                    if k > 1e-12 {
                        h = h.min((8.0 * tol / k).sqrt());
                    }
                }
                h = h.max(min_h);
                if let Some(mh) = max_h {
                    h = h.min(mh);
                }
                h
            };
            let base = |k: f64| {
                if k > 1e-12 {
                    2.0 * std::f64::consts::PI / (k * elements_per_2pi)
                } else {
                    max_h.unwrap_or(1e6)
                }
            };
            out.target_h.push(cap(base(kappa), kappa));
            out.target_h1.push(cap(base(k1.abs()), k1.abs()));
            out.target_h2.push(cap(base(k2.abs()), k2.abs()));

            // Principal-direction angle in the (du, dv) tangent basis,
            // matching the Python convention (per-axis normalization).
            let du_mag = dot(du, du).sqrt();
            let dv_mag = dot(dv, dv).sqrt();
            let angle = if du_mag > 1e-12 && dv_mag > 1e-12 {
                let cos_a = dot(dir1, du) / du_mag;
                let sin_a = dot(dir1, dv) / dv_mag;
                sin_a.atan2(cos_a)
            } else {
                0.0
            };
            out.curvature_angle.push(angle);
        }
    }
    out
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn xyz_frame() -> Frame {
        Frame {
            origin: [1.0, 2.0, 3.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            z_dir: [0.0, 0.0, 1.0],
        }
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12 && (a[2] - b[2]).abs() < 1e-12
    }

    #[test]
    fn plane_eval() {
        let s = Surface3::Plane { frame: xyz_frame() };
        assert!(close(s.value(2.0, -1.0), [3.0, 1.0, 3.0]));
    }

    #[test]
    fn cylinder_eval() {
        let s = Surface3::Cylinder {
            frame: xyz_frame(),
            radius: 2.0,
        };
        assert!(close(
            s.value(std::f64::consts::FRAC_PI_2, 5.0),
            [1.0, 4.0, 8.0]
        ));
    }

    #[test]
    fn cone_eval() {
        // Semi-angle 45deg, ref radius 1: at v = sqrt(2), radius grows
        // by 1 and height by 1.
        let s = Surface3::Cone {
            frame: xyz_frame(),
            ref_radius: 1.0,
            semi_angle: std::f64::consts::FRAC_PI_4,
        };
        let v = std::f64::consts::SQRT_2;
        assert!(close(s.value(0.0, v), [3.0, 2.0, 4.0]));
    }

    #[test]
    fn sphere_eval() {
        let s = Surface3::Sphere {
            frame: xyz_frame(),
            radius: 3.0,
        };
        // North pole at v = pi/2.
        assert!(close(
            s.value(0.7, std::f64::consts::FRAC_PI_2),
            [1.0, 2.0, 6.0]
        ));
        assert!(close(s.value(0.0, 0.0), [4.0, 2.0, 3.0]));
    }

    #[test]
    fn torus_eval() {
        let s = Surface3::Torus {
            frame: xyz_frame(),
            major_radius: 5.0,
            minor_radius: 1.0,
        };
        assert!(close(s.value(0.0, 0.0), [7.0, 2.0, 3.0]));
        assert!(close(
            s.value(0.0, std::f64::consts::FRAC_PI_2),
            [6.0, 2.0, 4.0]
        ));
    }

    #[test]
    fn left_handed_frame_preserved() {
        // gp_Ax3 placements can be left-handed; explicit Y must be used.
        let s = Surface3::Cylinder {
            frame: Frame {
                origin: [0.0; 3],
                x_dir: [1.0, 0.0, 0.0],
                y_dir: [0.0, -1.0, 0.0], // left-handed
                z_dir: [0.0, 0.0, 1.0],
            },
            radius: 1.0,
        };
        assert!(close(
            s.value(std::f64::consts::FRAC_PI_2, 0.0),
            [0.0, -1.0, 0.0]
        ));
    }

    #[test]
    fn bspline_bilinear_patch_interpolates() {
        // Degree-1 x degree-1 patch == bilinear interpolation of corners.
        let s = Surface3::BSpline {
            degree_u: 1,
            degree_v: 1,
            n_u_poles: 2,
            n_v_poles: 2,
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![
                [0.0, 0.0, 0.0], // (u0, v0)
                [0.0, 1.0, 1.0], // (u0, v1)
                [1.0, 0.0, 0.0], // (u1, v0)
                [1.0, 1.0, 3.0], // (u1, v1)
            ],
            weights: None,
        };
        assert!(close(s.value(0.0, 0.0), [0.0, 0.0, 0.0]));
        assert!(close(s.value(1.0, 1.0), [1.0, 1.0, 3.0]));
        assert!(close(s.value(0.5, 0.5), [0.5, 0.5, 1.0]));
    }

    #[test]
    fn rational_bspline_surface_quarter_cylinder() {
        // Quarter circle (rational quadratic) extruded linearly in v as
        // a degree (2, 1) NURBS: every point must sit on radius 1.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let s = Surface3::BSpline {
            degree_u: 2,
            degree_v: 1,
            n_u_poles: 3,
            n_v_poles: 2,
            knots_u: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 5.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 5.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 5.0],
            ],
            weights: Some(vec![1.0, 1.0, w, w, 1.0, 1.0]),
        };
        for i in 0..=8 {
            for j in 0..=4 {
                let p = s.value(i as f64 / 8.0, j as f64 / 4.0);
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!((r - 1.0).abs() < 1e-12, "r={r} at ({i},{j})");
            }
        }
    }

    #[test]
    fn extrusion_and_revolution() {
        let circle = Curve3::Circle {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            radius: 2.0,
        };
        let ex = Surface3::Extrusion {
            basis: Box::new(circle.clone()),
            dir: [0.0, 0.0, 1.0],
        };
        assert!(close(ex.value(0.0, 3.0), [2.0, 0.0, 3.0]));

        // Revolve a line parallel to z at x=2 around the z axis: a cylinder.
        let line = Curve3::Line {
            origin: [2.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
        };
        let rv = Surface3::Revolution {
            basis: Box::new(line),
            axis_origin: [0.0, 0.0, 0.0],
            axis_dir: [0.0, 0.0, 1.0],
        };
        assert!(close(
            rv.value(std::f64::consts::FRAC_PI_2, 4.0),
            [0.0, 2.0, 4.0]
        ));
    }
}
