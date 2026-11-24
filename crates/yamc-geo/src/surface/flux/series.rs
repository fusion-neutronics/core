//! The boundary as a double Fourier series, and its evaluation with
//! derivatives.
//!
//! The convention is DESC's product basis:
//!
//! ```text
//! R(theta, zeta) = sum r_mn T_m(theta) F_n(zeta)
//! Z(theta, zeta) = sum z_mn T_m(theta) F_n(zeta)
//! ```
//!
//! with `T_m = cos(m theta)` for `m >= 0` and `sin(|m| theta)` for `m < 0`,
//! `F_n = cos(n N zeta)` for `n >= 0` and `sin(|n| N zeta)` for `n < 0`, `N`
//! the number of field periods and `zeta` the cylindrical toroidal angle, so
//! every `zeta = const` cut of the surface is a planar closed curve.

/// Largest `|m|` and `|n|` accepted; they size the stack tables used while
/// evaluating.
pub(crate) const MAX_M: usize = 64;
pub(crate) const MAX_N: usize = 64;

/// `cos(k angle)` and `sin(k angle)` for `k = 0..=n`, by angle addition.
/// The recurrence loses about one ulp per step, so 1e-14 relative at the
/// largest mode numbers accepted.
#[inline]
pub(super) fn harmonics(angle: f64, n: usize, cos: &mut [f64], sin: &mut [f64]) {
    let (s1, c1) = angle.sin_cos();
    cos[0] = 1.0;
    sin[0] = 0.0;
    for k in 1..=n {
        cos[k] = cos[k - 1] * c1 - sin[k - 1] * s1;
        sin[k] = sin[k - 1] * c1 + cos[k - 1] * s1;
    }
}

/// Sup-norm bounds of `R`, `Z` and their derivatives over the whole
/// surface, from the coefficients alone (each term is bounded by the
/// modulus of its coefficient times its frequency factors).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SeriesBounds {
    pub r: f64,
    pub z: f64,
    pub r_t: f64,
    pub z_t: f64,
    pub r_tt: f64,
    pub z_tt: f64,
    pub r_z: f64,
    pub z_z: f64,
    pub r_tz: f64,
    pub z_tz: f64,
    pub r_zz: f64,
    pub z_zz: f64,
}

impl SeriesBounds {
    /// Bound on `|X_theta theta|` for the Cartesian point
    /// `X = (R cos zeta, R sin zeta, Z)`.
    pub fn x_tt(&self) -> f64 {
        self.r_tt + self.z_tt
    }

    /// Bound on `|X_zeta zeta|`: `X_zz = ((R_zz - R) cos - 2 R_z sin,
    /// (R_zz - R) sin + 2 R_z cos, Z_zz)`.
    pub fn x_zz(&self) -> f64 {
        self.r_zz + self.r + 2.0 * self.r_z + self.z_zz
    }
}

/// The series with its coefficients scattered into a dense
/// `(m, n)` table: `coef[m * (n_max + 1) + n]` holds
/// `[R_cc, R_cs, R_sc, R_ss, Z_cc, Z_cs, Z_sc, Z_ss]`, where the first
/// letter is the poloidal factor (`c` = cos, `s` = sin) and the second the
/// toroidal one.
#[derive(Debug, Clone)]
pub(super) struct Series {
    pub nfp: f64,
    pub m_max: usize,
    pub n_max: usize,
    coef: Vec<[f64; 8]>,
    pub bounds: SeriesBounds,
}

/// The coefficients of `cos(m theta)` and `sin(m theta)` at one `zeta`,
/// with their first and second `zeta` derivatives:
/// `c[m][order] = [R_cos, R_sin, Z_cos, Z_sin]`.
pub(super) struct Collapsed {
    pub m_max: usize,
    pub c: [[[f64; 4]; 3]; MAX_M + 1],
}

/// `R` and `Z` at one `(theta, zeta)` with the derivatives
/// `[value, d_theta, d_theta_theta, d_zeta, d_theta_zeta, d_zeta_zeta]`.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SectionPoint {
    pub r: [f64; 6],
    pub z: [f64; 6],
}

/// A Cartesian surface point and its parameter derivatives.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SurfacePoint {
    pub x: [f64; 3],
    pub x_t: [f64; 3],
    pub x_z: [f64; 3],
    pub x_tt: [f64; 3],
    pub x_tz: [f64; 3],
    pub x_zz: [f64; 3],
}

impl Series {
    /// Scatter signed `(m, n)` modes into the dense table. The caller has
    /// checked the modes (in range, no duplicates).
    pub fn new(
        nfp: u32,
        r_modes: &[[i32; 2]],
        r_mn: &[f64],
        z_modes: &[[i32; 2]],
        z_mn: &[f64],
    ) -> Self {
        let m_max = r_modes
            .iter()
            .chain(z_modes)
            .map(|mode| mode[0].unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        let n_max = r_modes
            .iter()
            .chain(z_modes)
            .map(|mode| mode[1].unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        let mut coef = vec![[0.0; 8]; (m_max + 1) * (n_max + 1)];
        for (offset, modes, values) in [(0, r_modes, r_mn), (4, z_modes, z_mn)] {
            for (&[m, n], &value) in modes.iter().zip(values) {
                let slot = 2 * usize::from(m < 0) + usize::from(n < 0);
                let index = m.unsigned_abs() as usize * (n_max + 1) + n.unsigned_abs() as usize;
                coef[index][offset + slot] = value;
            }
        }
        let nfp = nfp as f64;
        let mut bounds = SeriesBounds::default();
        for m in 0..=m_max {
            for n in 0..=n_max {
                let k = &coef[m * (n_max + 1) + n];
                let rabs = k[..4].iter().map(|v| v.abs()).sum::<f64>();
                let zabs = k[4..].iter().map(|v| v.abs()).sum::<f64>();
                let (mf, w) = (m as f64, n as f64 * nfp);
                bounds.r += rabs;
                bounds.z += zabs;
                bounds.r_t += mf * rabs;
                bounds.z_t += mf * zabs;
                bounds.r_tt += mf * mf * rabs;
                bounds.z_tt += mf * mf * zabs;
                bounds.r_z += w * rabs;
                bounds.z_z += w * zabs;
                bounds.r_tz += mf * w * rabs;
                bounds.z_tz += mf * w * zabs;
                bounds.r_zz += w * w * rabs;
                bounds.z_zz += w * w * zabs;
            }
        }
        Self {
            nfp,
            m_max,
            n_max,
            coef,
            bounds,
        }
    }

    /// Sum the toroidal factors at `zeta`, leaving a trigonometric
    /// polynomial in `theta`. `orders` (1 to 3) is how many `zeta`
    /// derivatives (value, first, second) to form.
    pub fn collapse(&self, zeta: f64, orders: usize) -> Collapsed {
        let mut cos_n = [0.0; MAX_N + 1];
        let mut sin_n = [0.0; MAX_N + 1];
        harmonics(self.nfp * zeta, self.n_max, &mut cos_n, &mut sin_n);
        let mut out = Collapsed {
            m_max: self.m_max,
            c: [[[0.0; 4]; 3]; MAX_M + 1],
        };
        let stride = self.n_max + 1;
        for m in 0..=self.m_max {
            let row = &self.coef[m * stride..(m + 1) * stride];
            let mut acc = [[0.0; 4]; 3];
            for (n, k) in row.iter().enumerate() {
                let (c, s) = (cos_n[n], sin_n[n]);
                acc[0][0] += k[0] * c + k[1] * s;
                acc[0][1] += k[2] * c + k[3] * s;
                acc[0][2] += k[4] * c + k[5] * s;
                acc[0][3] += k[6] * c + k[7] * s;
                if orders > 1 {
                    let w = n as f64 * self.nfp;
                    acc[1][0] += w * (k[1] * c - k[0] * s);
                    acc[1][1] += w * (k[3] * c - k[2] * s);
                    acc[1][2] += w * (k[5] * c - k[4] * s);
                    acc[1][3] += w * (k[7] * c - k[6] * s);
                    if orders > 2 {
                        let w2 = -w * w;
                        acc[2][0] += w2 * (k[0] * c + k[1] * s);
                        acc[2][1] += w2 * (k[2] * c + k[3] * s);
                        acc[2][2] += w2 * (k[4] * c + k[5] * s);
                        acc[2][3] += w2 * (k[6] * c + k[7] * s);
                    }
                }
            }
            out.c[m] = acc;
        }
        out
    }
}

impl Collapsed {
    /// Bound on the second `theta` derivative of the cut `(R, Z)(theta)`
    /// at this `zeta`, for the chord-sag of a polyline through it.
    pub fn curvature_bound(&self) -> f64 {
        (0..=self.m_max)
            .map(|m| {
                let c = &self.c[m][0];
                (m * m) as f64 * (c[0].abs() + c[1].abs() + c[2].abs() + c[3].abs())
            })
            .sum()
    }

    /// The cut `(R, Z)` and its first two `theta` derivatives only.
    #[inline]
    pub fn curve(&self, theta: f64) -> [f64; 6] {
        let mut cos_m = [0.0; MAX_M + 1];
        let mut sin_m = [0.0; MAX_M + 1];
        harmonics(theta, self.m_max, &mut cos_m, &mut sin_m);
        let mut out = [0.0; 6];
        for m in 0..=self.m_max {
            let (c, s, mf) = (cos_m[m], sin_m[m], m as f64);
            let k = &self.c[m][0];
            let (rv, zv) = (k[0] * c + k[1] * s, k[2] * c + k[3] * s);
            out[0] += rv;
            out[1] += zv;
            out[2] += mf * (k[1] * c - k[0] * s);
            out[3] += mf * (k[3] * c - k[2] * s);
            out[4] -= mf * mf * rv;
            out[5] -= mf * mf * zv;
        }
        // [R, Z, R_t, Z_t, R_tt, Z_tt]
        out
    }

    /// The cut `(R, Z)` from precomputed `cos(m theta_k)` / `sin(m theta_k)`
    /// rows (`table[m]`, `table[m_max + 1 + m]`).
    #[inline]
    pub fn curve_from_table(&self, cos_row: &[f64], sin_row: &[f64]) -> (f64, f64) {
        let (mut r, mut z) = (0.0, 0.0);
        for m in 0..=self.m_max {
            let k = &self.c[m][0];
            r += k[0] * cos_row[m] + k[1] * sin_row[m];
            z += k[2] * cos_row[m] + k[3] * sin_row[m];
        }
        (r, z)
    }

    /// `R`, `Z` and all first and second derivatives at `theta` (the
    /// collapse must have been formed with all three orders).
    pub fn section(&self, theta: f64) -> SectionPoint {
        let mut cos_m = [0.0; MAX_M + 1];
        let mut sin_m = [0.0; MAX_M + 1];
        harmonics(theta, self.m_max, &mut cos_m, &mut sin_m);
        let mut p = SectionPoint::default();
        for m in 0..=self.m_max {
            let (c, s, mf) = (cos_m[m], sin_m[m], m as f64);
            let [k0, k1, k2] = &self.c[m];
            for (out, i) in [(&mut p.r, 0), (&mut p.z, 2)] {
                let v0 = k0[i] * c + k0[i + 1] * s;
                out[0] += v0;
                out[1] += mf * (k0[i + 1] * c - k0[i] * s);
                out[2] -= mf * mf * v0;
                out[3] += k1[i] * c + k1[i + 1] * s;
                out[4] += mf * (k1[i + 1] * c - k1[i] * s);
                out[5] += k2[i] * c + k2[i + 1] * s;
            }
        }
        p
    }
}

impl SectionPoint {
    /// The Cartesian point `X = (R cos zeta, R sin zeta, Z)` and its
    /// derivatives.
    pub fn cartesian(&self, zeta: f64) -> SurfacePoint {
        let (s, c) = zeta.sin_cos();
        let [r, r_t, r_tt, r_z, r_tz, r_zz] = self.r;
        let [z, z_t, z_tt, z_z, z_tz, z_zz] = self.z;
        SurfacePoint {
            x: [r * c, r * s, z],
            x_t: [r_t * c, r_t * s, z_t],
            x_z: [r_z * c - r * s, r_z * s + r * c, z_z],
            x_tt: [r_tt * c, r_tt * s, z_tt],
            x_tz: [r_tz * c - r_t * s, r_tz * s + r_t * c, z_tz],
            x_zz: [
                (r_zz - r) * c - 2.0 * r_z * s,
                (r_zz - r) * s + 2.0 * r_z * c,
                z_zz,
            ],
        }
    }
}

impl Series {
    /// The Cartesian surface point and its derivatives at `(theta, zeta)`.
    pub fn point(&self, theta: f64, zeta: f64) -> SurfacePoint {
        self.collapse(zeta, 3).section(theta).cartesian(zeta)
    }
}
