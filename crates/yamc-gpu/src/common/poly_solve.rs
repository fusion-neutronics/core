//! Newton-based cubic and Ferrari-based quartic solvers for the GPU
//! ray-surface kernels (specifically the ZTorus distance-to-boundary).
//!
//! Why Newton, not Cardano: Cardano's classical formula needs `cbrt`,
//! and the three-real-root branch needs `cos`/`acos` -- none of which
//! cubecl-spirv emits validly for f64 on the AMD/RADV path. Newton
//! from above the largest root needs only `+ - * / sqrt`, which we've
//! validated. For our use case (Ferrari uses just the **largest** real
//! root of the resolvent cubic for numerical stability), returning
//! that single root is sufficient.
//!
//! # Cubic solver
//!
//! `cubic_largest_real(p, q)` returns the largest real root of the
//! depressed cubic `t³ + p·t + q = 0`. By Cauchy's bound all roots
//! satisfy `|r| ≤ 1 + max(|p|, |q|)`, so we start Newton at
//! `t₀ = 1.5 · (1 + max(|p|, |q|))` (conservatively above the bound),
//! where the cubic is convex (`f' > 0`, `f'' > 0`) and Newton converges
//! monotonically downward to the largest root. 60 iterations is far
//! more than needed for f64 precision but costs ~600 flops, which is
//! negligible against the rest of the transport step.
//!
//! # Quartic solver
//!
//! `quartic_smallest_positive(b, c, d, e)` returns the smallest root
//! `> 1e-12` of `t⁴ + b·t³ + c·t² + d·t + e = 0`, or `1e30` if none.
//! Standard Ferrari: depress to `u = t − b/4`, build the resolvent
//! cubic, take its largest root via `cubic_largest_real`, then split
//! into two quadratics. The biquadratic shortcut (`q ≈ 0`) is handled
//! separately.

use cubecl::prelude::*;

/// Largest real root of the depressed cubic `t³ + p·t + q = 0`,
/// computed by Newton starting above Cauchy's bound. Convex
/// monotone descent -- never escapes outward, never overshoots
/// the largest root.
#[cube]
pub fn cubic_largest_real(p: f64, q: f64) -> f64 {
    let mut abs_p = p;
    if abs_p < 0.0 {
        abs_p = -p;
    }
    let mut abs_q = q;
    if abs_q < 0.0 {
        abs_q = -q;
    }
    let mut bound = abs_q;
    if abs_p > abs_q {
        bound = abs_p;
    }
    let mut t = 1.5 * (1.0 + bound);

    let mut iter = 0u32;
    while iter < 60u32 {
        let f = t * t * t + p * t + q;
        let fp = 3.0 * t * t + p;
        // For t starting above Cauchy's bound, f' > 0, so the
        // division is safe. The first 5–10 steps are linear (factor
        // ~2/3 each); after that quadratic, hitting f64 precision
        // in another ~5 steps. 60 is well past convergence.
        t -= f / fp;
        iter += 1u32;
    }
    t
}

/// Smallest real root `> 1e-12` of the monic quartic
/// `t⁴ + b·t³ + c·t² + d·t + e = 0`, or `1e30` if no such root
/// exists. Mirrors `yamc-geo::surface::solve_quartic` but returns
/// only the smallest positive root.
#[cube]
pub fn quartic_smallest_positive(b: f64, c: f64, d: f64, e: f64) -> f64 {
    // Depress: substitute t = u + shift, with shift = -b/4.
    let b2 = b * b;
    let b3 = b2 * b;
    let b4 = b2 * b2;
    let p = c - 3.0 * b2 / 8.0;
    let q = d - b * c / 2.0 + b3 / 8.0;
    let r = e - b * d / 4.0 + b2 * c / 16.0 - 3.0 * b4 / 256.0;
    let shift = -b / 4.0;

    let mut min_t = 1e30_f64;
    let mut abs_q = q;
    if abs_q < 0.0 {
        abs_q = -q;
    }

    if abs_q < 1e-14 {
        // Biquadratic: u⁴ + p·u² + r = 0. Solve as quadratic in v=u².
        let disc_b = p * p - 4.0 * r;
        if disc_b >= -1e-12 {
            let mut sd = disc_b;
            if sd < 0.0 {
                sd = 0.0;
            }
            let sqrt_disc = sd.sqrt();
            let w1 = (-p + sqrt_disc) * 0.5;
            let w2 = (-p - sqrt_disc) * 0.5;
            if w1 >= -1e-12 {
                let mut sw = w1;
                if sw < 0.0 {
                    sw = 0.0;
                }
                let s = sw.sqrt();
                let t_a = s + shift;
                let t_b = -s + shift;
                if t_a > 1e-12 && t_a < min_t {
                    min_t = t_a;
                }
                if t_b > 1e-12 && t_b < min_t {
                    min_t = t_b;
                }
            }
            if w2 >= -1e-12 {
                let mut sw = w2;
                if sw < 0.0 {
                    sw = 0.0;
                }
                let s = sw.sqrt();
                let t_a = s + shift;
                let t_b = -s + shift;
                if t_a > 1e-12 && t_a < min_t {
                    min_t = t_a;
                }
                if t_b > 1e-12 && t_b < min_t {
                    min_t = t_b;
                }
            }
        }
    } else {
        // Ferrari. Resolvent cubic y³ + A·y² + B·y + C = 0 with
        // A = -p/2, B = -r, C = r·p/2 − q²/8. Depress to find
        // the largest real root.
        let cubic_a = -p / 2.0;
        let cubic_b = -r;
        let cubic_c = r * p / 2.0 - q * q / 8.0;
        let cubic_a_cubed = cubic_a * cubic_a * cubic_a;
        let cp = cubic_b - cubic_a * cubic_a / 3.0;
        let cq = cubic_c - cubic_a * cubic_b / 3.0 + 2.0 * cubic_a_cubed / 27.0;
        let cubic_shift = -cubic_a / 3.0;
        let w_largest = cubic_largest_real(cp, cq);
        let y = w_largest + cubic_shift;

        let two_y_minus_p = 2.0 * y - p;
        if two_y_minus_p >= -1e-12 {
            let mut safe = two_y_minus_p;
            if safe < 0.0 {
                safe = 0.0;
            }
            let sq = safe.sqrt();
            let mut abs_sq = sq;
            if abs_sq < 0.0 {
                abs_sq = -sq;
            }
            if abs_sq > 1e-14 {
                let h = q / (2.0 * sq);
                // First quadratic: u² − sq·u + (y + h) = 0.
                let disc1 = sq * sq * 0.25 - (y + h);
                if disc1 >= -1e-12 {
                    let mut sd = disc1;
                    if sd < 0.0 {
                        sd = 0.0;
                    }
                    let s = sd.sqrt();
                    let t_a = sq * 0.5 + s + shift;
                    let t_b = sq * 0.5 - s + shift;
                    if t_a > 1e-12 && t_a < min_t {
                        min_t = t_a;
                    }
                    if t_b > 1e-12 && t_b < min_t {
                        min_t = t_b;
                    }
                }
                // Second quadratic: u² + sq·u + (y − h) = 0.
                let disc2 = sq * sq * 0.25 - (y - h);
                if disc2 >= -1e-12 {
                    let mut sd = disc2;
                    if sd < 0.0 {
                        sd = 0.0;
                    }
                    let s = sd.sqrt();
                    let t_a = -sq * 0.5 + s + shift;
                    let t_b = -sq * 0.5 - s + shift;
                    if t_a > 1e-12 && t_a < min_t {
                        min_t = t_a;
                    }
                    if t_b > 1e-12 && t_b < min_t {
                        min_t = t_b;
                    }
                }
            } else {
                // sq ≈ 0 degenerate: solve u² = w directly.
                let disc = y * y - r;
                if disc >= -1e-12 {
                    let mut sd = disc;
                    if sd < 0.0 {
                        sd = 0.0;
                    }
                    let s = sd.sqrt();
                    let w_a = -y + s;
                    let w_b = -y - s;
                    if w_a >= -1e-12 {
                        let mut sw = w_a;
                        if sw < 0.0 {
                            sw = 0.0;
                        }
                        let v = sw.sqrt();
                        let t_a = v + shift;
                        let t_b = -v + shift;
                        if t_a > 1e-12 && t_a < min_t {
                            min_t = t_a;
                        }
                        if t_b > 1e-12 && t_b < min_t {
                            min_t = t_b;
                        }
                    }
                    if w_b >= -1e-12 {
                        let mut sw = w_b;
                        if sw < 0.0 {
                            sw = 0.0;
                        }
                        let v = sw.sqrt();
                        let t_a = v + shift;
                        let t_b = -v + shift;
                        if t_a > 1e-12 && t_a < min_t {
                            min_t = t_a;
                        }
                        if t_b > 1e-12 && t_b < min_t {
                            min_t = t_b;
                        }
                    }
                }
            }
        }
    }

    min_t
}

/// CPU mirror of `cubic_largest_real`. Same algorithm, regular Rust.
pub fn cubic_largest_real_cpu(p: f64, q: f64) -> f64 {
    let bound = p.abs().max(q.abs());
    let mut t = 1.5 * (1.0 + bound);
    for _ in 0..60 {
        let f = t * t * t + p * t + q;
        let fp = 3.0 * t * t + p;
        t -= f / fp;
    }
    t
}

/// CPU mirror of `quartic_smallest_positive`. Same algorithm.
pub fn quartic_smallest_positive_cpu(b: f64, c: f64, d: f64, e: f64) -> f64 {
    let b2 = b * b;
    let b3 = b2 * b;
    let b4 = b2 * b2;
    let p = c - 3.0 * b2 / 8.0;
    let q = d - b * c / 2.0 + b3 / 8.0;
    let r = e - b * d / 4.0 + b2 * c / 16.0 - 3.0 * b4 / 256.0;
    let shift = -b / 4.0;

    let mut min_t = 1e30_f64;
    let consider = |t: f64, min_t: &mut f64| {
        if t > 1e-12 && t < *min_t {
            *min_t = t;
        }
    };

    if q.abs() < 1e-14 {
        let disc_b = p * p - 4.0 * r;
        if disc_b >= -1e-12 {
            let sqrt_disc = disc_b.max(0.0).sqrt();
            let w1 = (-p + sqrt_disc) * 0.5;
            let w2 = (-p - sqrt_disc) * 0.5;
            if w1 >= -1e-12 {
                let s = w1.max(0.0).sqrt();
                consider(s + shift, &mut min_t);
                consider(-s + shift, &mut min_t);
            }
            if w2 >= -1e-12 {
                let s = w2.max(0.0).sqrt();
                consider(s + shift, &mut min_t);
                consider(-s + shift, &mut min_t);
            }
        }
        return min_t;
    }

    let cubic_a = -p / 2.0;
    let cubic_b = -r;
    let cubic_c = r * p / 2.0 - q * q / 8.0;
    let cp = cubic_b - cubic_a * cubic_a / 3.0;
    let cq = cubic_c - cubic_a * cubic_b / 3.0 + 2.0 * cubic_a.powi(3) / 27.0;
    let cubic_shift = -cubic_a / 3.0;
    let y = cubic_largest_real_cpu(cp, cq) + cubic_shift;

    let two_y_minus_p = 2.0 * y - p;
    if two_y_minus_p < -1e-12 {
        return min_t;
    }
    let sq = two_y_minus_p.max(0.0).sqrt();

    if sq.abs() > 1e-14 {
        let h = q / (2.0 * sq);
        let disc1 = sq * sq * 0.25 - (y + h);
        if disc1 >= -1e-12 {
            let s = disc1.max(0.0).sqrt();
            consider(sq * 0.5 + s + shift, &mut min_t);
            consider(sq * 0.5 - s + shift, &mut min_t);
        }
        let disc2 = sq * sq * 0.25 - (y - h);
        if disc2 >= -1e-12 {
            let s = disc2.max(0.0).sqrt();
            consider(-sq * 0.5 + s + shift, &mut min_t);
            consider(-sq * 0.5 - s + shift, &mut min_t);
        }
    } else {
        let disc = y * y - r;
        if disc >= -1e-12 {
            let s = disc.max(0.0).sqrt();
            for &w in &[-y + s, -y - s] {
                if w >= -1e-12 {
                    let v = w.max(0.0).sqrt();
                    consider(v + shift, &mut min_t);
                    consider(-v + shift, &mut min_t);
                }
            }
        }
    }

    min_t
}

/// Test kernel: per-thread solve of a quartic from input coefficients.
#[cube(launch_unchecked)]
fn quartic_solve_kernel(coeffs: &[f64], roots: &mut [f64]) {
    if ABSOLUTE_POS >= roots.len() {
        terminate!();
    }
    let i4 = ABSOLUTE_POS * 4;
    let b = coeffs[i4];
    let c = coeffs[i4 + 1];
    let d = coeffs[i4 + 2];
    let e = coeffs[i4 + 3];
    roots[ABSOLUTE_POS] = quartic_smallest_positive(b, c, d, e);
}

use crate::{GpuContext, WgpuRuntime};

/// Run `quartic_smallest_positive` on the GPU for test/validation
/// purposes. `coeffs` is stride-4: `[b₀, c₀, d₀, e₀, b₁, c₁, …]`.
pub fn run_quartic_solve(ctx: &GpuContext, coeffs: &[f64]) -> Vec<f64> {
    assert!(coeffs.len().is_multiple_of(4));
    let n = coeffs.len() / 4;
    let client = ctx.client();
    let coeffs_h = client.create_from_slice(bytemuck::cast_slice(coeffs));
    let roots_h = client.empty(n * core::mem::size_of::<f64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        quartic_solve_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(coeffs_h, coeffs.len()),
            BufferArg::from_raw_parts(roots_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(roots_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Cubic with three real roots: `t³ − 7t + 6` has roots 1, 2, −3.
    /// Newton must find the largest, t = 2.
    #[test]
    fn cubic_three_real_roots_returns_largest() {
        let r = cubic_largest_real_cpu(-7.0, 6.0);
        assert!(
            (r - 2.0).abs() < 1e-12,
            "expected largest root 2.0, got {r}"
        );
    }

    /// Cubic with one real root: `t³ + t + 1`, root ≈ −0.6823.
    #[test]
    fn cubic_one_real_root() {
        let r = cubic_largest_real_cpu(1.0, 1.0);
        let expected = -0.682_327_803_828_019_3_f64;
        assert!((r - expected).abs() < 1e-12, "expected {expected}, got {r}");
    }

    /// Quartic `(t − 1)(t − 2)(t − 3)(t − 4) = 0` expanded:
    /// `t⁴ − 10t³ + 35t² − 50t + 24`. Smallest positive root is 1.
    #[test]
    fn quartic_four_real_roots_returns_smallest() {
        let r = quartic_smallest_positive_cpu(-10.0, 35.0, -50.0, 24.0);
        assert!((r - 1.0).abs() < 1e-10, "expected 1.0, got {r}");
    }

    /// Quartic with no positive real roots -- return sentinel.
    /// `(t + 1)(t + 2)(t² + 1)` expanded:
    /// `t⁴ + 3t³ + 3t² + 3t + 2`. Real roots are −1 and −2.
    #[test]
    fn quartic_no_positive_returns_sentinel() {
        let r = quartic_smallest_positive_cpu(3.0, 3.0, 3.0, 2.0);
        assert!(r >= 1e29, "expected sentinel, got {r}");
    }

    /// GPU and CPU solvers agree on a battery of quartics:
    /// integer-rooted, no-real-roots, and a couple ray-torus-shaped
    /// cases (very small leading coefficients after depressing).
    #[test]
    fn gpu_quartic_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        // (b, c, d, e) for a few quartics:
        // 1) (t-1)(t-2)(t-3)(t-4)         smallest > 0 = 1
        // 2) (t-0.5)(t-2)(t-5)(t-10)      smallest > 0 = 0.5
        // 3) (t+1)(t+2)(t²+1)             no positive root → sentinel
        // 4) Biquadratic (t²-1)(t²-4)     = t⁴ - 5t² + 4, smallest > 0 = 1
        // 5) (t-1)²(t-3)(t-4)             repeated root at 1
        let coeffs = vec![
            -10.0, 35.0, -50.0, 24.0, -17.5, 88.5, -141.0, 50.0, 3.0, 3.0, 3.0, 2.0, 0.0, -5.0,
            0.0, 4.0, -9.0, 27.0, -35.0, 12.0,
        ];
        let gpu = run_quartic_solve(&ctx, &coeffs);
        let cpu: Vec<f64> = coeffs
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| quartic_smallest_positive_cpu(c[0], c[1], c[2], c[3]))
            .collect();
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // Sentinel comparison: both should be ≥ 1e29 or both finite.
            if *c >= 1e29 || *g >= 1e29 {
                assert!(*c >= 1e29 && *g >= 1e29, "case {i}: gpu {g} cpu {c}");
                continue;
            }
            let rel = (g - c).abs() / c.abs().max(1e-12);
            assert!(rel < 1e-9, "case {i}: gpu {g} cpu {c} rel err {rel}");
        }
    }
}
