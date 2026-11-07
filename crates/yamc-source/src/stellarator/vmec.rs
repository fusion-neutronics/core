//! Fitting a [`StellaratorShape`] to the flux surfaces of a VMEC equilibrium.
//!
//! A VMEC `wout` file gives the flux surface `s_j = j / (ns - 1)`, `s` the
//! normalised toroidal flux, as
//!
//! ```text
//! R = sum rmnc_j cos(xm theta - xn zeta) + rmns_j sin(xm theta - xn zeta)
//! Z = sum zmns_j sin(xm theta - xn zeta) + zmnc_j cos(xm theta - xn zeta)
//! ```
//!
//! with `xn` a multiple of the number of field periods `N`. Each surface is
//! rewritten exactly in the product basis of the Fourier-Zernike series with
//! `cos(a - b) = cos a cos b + sin a sin b` and
//! `sin(a - b) = sin a cos b - cos a sin b`, and the radial profile of each
//! product term is fitted by least squares with the Zernike radial
//! polynomials `R_l^|m|(rho)`, `rho = sqrt(s)`, which go as `rho^|m|` on the
//! magnetic axis. DESC imports a `wout` file with the same fit, but takes its
//! radial resolution from the poloidal one (`L = M`), then pins the axis and
//! the boundary to VMEC's exactly and reverses theta. yamc keeps VMEC's
//! angles and fits every surface alike, the boundary included, so its
//! `rho = 1` surface is VMEC's boundary to within
//! [`StellaratorShape::fit_error`].

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use super::{jacobi_step, StellaratorShape, MAX_N};
use crate::distribution::check_finite;

/// No fit missing the surfaces by more than this fraction of the minor radius
/// is kept.
const TOLERANCE: f64 = 0.01;
/// A fit within this fraction of the minor radius is exact to rounding, so
/// the search for the radial resolution stops there.
const EXACT: f64 = 1.0e-10;
/// The error for coefficients too large to sum or fit.
const OVERFLOW: &str = "the fit overflows: check the coefficients are in cm";

impl StellaratorShape {
    /// Fit a shape to the flux surfaces of a VMEC equilibrium, given as the
    /// Fourier coefficients of a `wout` file (see the module docs) in cm.
    /// `rmnc[j][k]` is the `R` coefficient of mode `(xm[k], xn[k])` on surface
    /// `s_j = j / (ns - 1)`; `rmns` and `zmnc` are the extra terms of an
    /// equilibrium that is not stellarator symmetric.
    ///
    /// The angles are VMEC's: the shape at `(rho, theta, zeta)` is VMEC's
    /// surface `s = rho^2` at `(theta, zeta)`, so a `wout` file's theta
    /// (VMEC's `signgs = -1`) runs counter-clockwise in the `(R, Z)` plane, as
    /// a [`TokamakShape`](crate::tokamak::TokamakShape)'s does.
    ///
    /// Every product term gets the same number of radial polynomials, trying
    /// up to `sqrt(ns) + 1`, since polynomials of higher degree in `s` can
    /// oscillate between equally spaced surfaces: the fewest whose fit misses
    /// the surfaces by no more than twice what the best fit does, nor by more
    /// than 1% of the minor radius, the mean distance of the boundary from the
    /// magnetic axis. The coefficients are refused if even the best fit misses
    /// by more; [`StellaratorShape::fit_error`] says how far the fit misses.
    pub fn from_vmec(
        num_field_periods: u32,
        xm: &[f64],
        xn: &[f64],
        rmnc: &[Vec<f64>],
        zmns: &[Vec<f64>],
        rmns: Option<&[Vec<f64>]>,
        zmnc: Option<&[Vec<f64>]>,
    ) -> Result<Self, String> {
        if num_field_periods == 0 {
            return Err("num_field_periods must be at least 1".to_string());
        }
        let modes = mode_numbers(num_field_periods, xm, xn)?;
        let ns = rmnc.len();
        if ns < 2 {
            return Err(format!(
                "rmnc must hold at least two surfaces, the magnetic axis and the boundary \
                 (got {ns})"
            ));
        }
        for (name, rows) in [
            ("rmnc", Some(rmnc)),
            ("zmns", Some(zmns)),
            ("rmns", rmns),
            ("zmnc", zmnc),
        ] {
            let Some(rows) = rows else { continue };
            if rows.len() != ns {
                return Err(format!(
                    "{name} has {} surfaces and rmnc {ns}; they must have the same number",
                    rows.len()
                ));
            }
            for row in rows {
                if row.len() != modes.len() {
                    return Err(format!(
                        "{name} has a surface with {} coefficients; it needs one per mode, \
                         {} as in xm",
                        row.len(),
                        modes.len()
                    ));
                }
                for &value in row {
                    check_finite(name, value)?;
                }
            }
        }
        let r = Profiles::new(&modes, Some(rmnc), rmns, ns);
        let z = Profiles::new(&modes, zmnc, Some(zmns), ns);
        if r.terms.is_empty() || z.terms.is_empty() {
            return Err(format!(
                "every {} coefficient is zero on every surface",
                if r.terms.is_empty() { "R" } else { "Z" }
            ));
        }
        let grid = Grid::new(num_field_periods, &r, &z);
        let minor_radius = grid.minor_radius(&r, &z);
        if !minor_radius.is_finite() {
            return Err(OVERFLOW.to_string());
        }
        if minor_radius <= 0.0 {
            return Err(
                "the boundary (the last surface) is the magnetic axis (the first): the \
                 surfaces enclose no volume"
                    .to_string(),
            );
        }

        let rho: Vec<f64> = (0..ns)
            .map(|j| (j as f64 / (ns - 1) as f64).sqrt())
            .collect();
        let most = ((ns as f64).sqrt() as usize + 1).min(ns - 1);
        let mut fits = Vec::with_capacity(most);
        for count in 1..=most {
            let mut solvers = BTreeMap::new();
            let (r_lmn, r_residual) = r.fit(&rho, count, &mut solvers);
            let (z_lmn, z_residual) = z.fit(&rho, count, &mut solvers);
            let miss = grid.miss(&r, &r_residual, &z, &z_residual);
            let exact = miss.distance <= EXACT * minor_radius;
            fits.push((r_lmn, z_lmn, miss));
            if exact {
                break;
            }
        }
        let allowed = TOLERANCE * minor_radius;
        let best = fits
            .iter()
            .map(|fit| &fit.2)
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
            .filter(|best| best.distance.is_finite())
            .ok_or(OVERFLOW)?;
        if best.distance > allowed {
            return Err(format!(
                "the best fit misses the VMEC surfaces by up to {:.4} cm, on surface j = {} \
                 of {ns} (rho = {:.4}) at theta = {:.4}, zeta = {:.4}, more than the \
                 {allowed:.4} cm allowed, 1% of the minor radius: the coefficients are not \
                 smooth in rho, or too few surfaces resolve them",
                best.distance, best.surface, rho[best.surface], best.theta, best.zeta,
            ));
        }
        // The fewest polynomials within both limits; the best fit is within them.
        let limit = allowed.min(2.0 * best.distance);
        let fewest = fits.iter().take_while(|fit| fit.2.distance > limit).count();
        let (r_lmn, z_lmn, miss) = fits.swap_remove(fewest);
        let (r_modes, r_lmn) = r.modes(r_lmn);
        let (z_modes, z_lmn) = z.modes(z_lmn);
        let mut shape = Self::new(num_field_periods, r_modes, r_lmn, z_modes, z_lmn)?;
        shape.fit_error = Some(miss.distance);
        Ok(shape)
    }
}

/// VMEC's `(xm, xn)` as `(m, n)` with `n = xn / N`, checked.
fn mode_numbers(num_field_periods: u32, xm: &[f64], xn: &[f64]) -> Result<Vec<[i32; 2]>, String> {
    if xm.is_empty() || xm.len() != xn.len() {
        return Err(format!(
            "xm and xn must be non-empty and the same length (got {} and {})",
            xm.len(),
            xn.len()
        ));
    }
    let nfp = f64::from(num_field_periods);
    let mut modes = Vec::with_capacity(xm.len());
    for (&m, &toroidal) in xm.iter().zip(xn) {
        if !(m.fract() == 0.0 && (0.0..=f64::from(i32::MAX / 2)).contains(&m)) {
            return Err(format!("xm must hold whole numbers >= 0 (got {m})"));
        }
        let n = toroidal / nfp;
        if n.fract() != 0.0 {
            return Err(format!(
                "xn must hold multiples of num_field_periods = {num_field_periods}, as a wout \
                 file's xn includes the field-period factor (got {toroidal})"
            ));
        }
        if n.abs() > MAX_N as f64 {
            return Err(format!(
                "xn has {toroidal}, {n} times num_field_periods; up to {MAX_N} times is \
                 supported"
            ));
        }
        modes.push([m as i32, n as i32]);
    }
    let mut sorted = modes.clone();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!(
            "xm and xn list the mode ({}, {}) more than once",
            w[0][0],
            f64::from(w[0][1]) * nfp
        ));
    }
    Ok(modes)
}

/// `R` or `Z` on every surface in the product basis.
struct Profiles {
    /// The `(m, n)` terms, sorted, in DESC's convention: `m < 0` for
    /// `sin(|m| theta)`, `n < 0` for `sin(|n| N zeta)`.
    terms: Vec<[i32; 2]>,
    /// `values[t][j]`: term `t` on surface `j`.
    values: Vec<Vec<f64>>,
}

impl Profiles {
    /// Rewrite `sum cos_k cos(m theta - n N zeta) + sin_k sin(m theta - n N zeta)`
    /// in the product basis, dropping the terms that are zero everywhere.
    fn new(
        modes: &[[i32; 2]],
        cos: Option<&[Vec<f64>]>,
        sin: Option<&[Vec<f64>]>,
        ns: usize,
    ) -> Self {
        let mut terms = BTreeMap::<[i32; 2], Vec<f64>>::new();
        let mut add = |term: [i32; 2], sign: f64, rows: &[Vec<f64>], k: usize| {
            let values = terms.entry(term).or_insert_with(|| vec![0.0; ns]);
            for (value, row) in values.iter_mut().zip(rows) {
                *value += sign * row[k];
            }
        };
        for (k, &[m, n]) in modes.iter().enumerate() {
            let (sign_m, sign_n) = (f64::from(m.signum()), f64::from(n.signum()));
            let (m, n) = (m.abs(), n.abs());
            if let Some(rows) = cos {
                add([m, n], 1.0, rows, k);
                if m != 0 && n != 0 {
                    add([-m, -n], sign_m * sign_n, rows, k);
                }
            }
            if let Some(rows) = sin {
                if m != 0 {
                    add([-m, n], sign_m, rows, k);
                }
                if n != 0 {
                    add([m, -n], -sign_n, rows, k);
                }
            }
        }
        terms.retain(|_, values| values.iter().any(|&v| v != 0.0));
        let (terms, values) = terms.into_iter().unzip();
        Self { terms, values }
    }

    /// The least-squares coefficients of `count` radial polynomials for every
    /// term, and the residuals on the surfaces. `solvers` holds the
    /// factorisation for each `|m|`, shared by `R` and `Z`.
    fn fit(
        &self,
        rho: &[f64],
        count: usize,
        solvers: &mut BTreeMap<i32, LeastSquares>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        self.terms
            .iter()
            .zip(&self.values)
            .map(|(&[m, _], values)| {
                let alpha = m.abs();
                let columns = || {
                    (0..count)
                        .map(|k| rho.iter().map(|&r| zernike_radial(alpha, k, r)).collect())
                        .collect()
                };
                solvers
                    .entry(alpha)
                    .or_insert_with(|| LeastSquares::new(columns()))
                    .solve(values)
            })
            .unzip()
    }

    /// The Fourier-Zernike modes and coefficients of a fit.
    fn modes(&self, lmn: Vec<Vec<f64>>) -> (Vec<[i32; 3]>, Vec<f64>) {
        let mut modes = Vec::new();
        let mut coefficients = Vec::new();
        for (&[m, n], radial) in self.terms.iter().zip(lmn) {
            for (k, value) in radial.into_iter().enumerate() {
                modes.push([m.abs() + 2 * k as i32, m, n]);
                coefficients.push(value);
            }
        }
        (modes, coefficients)
    }
}

/// The Zernike radial polynomial `R_{alpha + 2k}^alpha(rho)`.
fn zernike_radial(alpha: i32, k: usize, rho: f64) -> f64 {
    let x = 1.0 - 2.0 * rho * rho;
    let (mut p, mut p_prev) = (1.0, 0.0);
    for degree in 1..=k {
        (p_prev, p, _, _) = jacobi_step(degree as i32, f64::from(alpha), x, p, p_prev, 0.0, 0.0);
    }
    let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
    sign * rho.powi(alpha) * p
}

/// Least squares by Householder QR: one design matrix, many right-hand sides.
struct LeastSquares {
    /// The design matrix by column, kept for the residuals.
    columns: Vec<Vec<f64>>,
    /// The reflector of each column, from the diagonal down, and its
    /// squared length.
    reflectors: Vec<(Vec<f64>, f64)>,
    /// The triangular factor by column: `r[k][i]` for `i <= k`.
    r: Vec<Vec<f64>>,
}

impl LeastSquares {
    fn new(columns: Vec<Vec<f64>>) -> Self {
        let mut work = columns.clone();
        let mut reflectors = Vec::with_capacity(columns.len());
        for k in 0..columns.len() {
            let mut v = work[k][k..].to_vec();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Reflect onto -sign(v_0) |v| so the subtraction does not cancel.
            v[0] += if v[0] < 0.0 { -norm } else { norm };
            let length = v.iter().map(|x| x * x).sum::<f64>();
            for column in &mut work[k..] {
                reflect(&v, length, &mut column[k..]);
            }
            reflectors.push((v, length));
        }
        let r = work
            .iter()
            .enumerate()
            .map(|(k, column)| column[..=k].to_vec())
            .collect();
        Self {
            columns,
            reflectors,
            r,
        }
    }

    /// The coefficients minimising `|A c - b|`, and the residual `b - A c`.
    fn solve(&self, b: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut y = b.to_vec();
        for (k, (v, length)) in self.reflectors.iter().enumerate() {
            reflect(v, *length, &mut y[k..]);
        }
        let mut c = vec![0.0; self.r.len()];
        for k in (0..c.len()).rev() {
            let sum: f64 = (k + 1..c.len()).map(|i| self.r[i][k] * c[i]).sum();
            c[k] = (y[k] - sum) / self.r[k][k];
        }
        let mut residual = b.to_vec();
        for (column, &ck) in self.columns.iter().zip(&c) {
            for (value, a) in residual.iter_mut().zip(column) {
                *value -= ck * a;
            }
        }
        (c, residual)
    }
}

/// Apply the reflector `I - 2 v v^T / |v|^2` to `x`.
fn reflect(v: &[f64], length: f64, x: &mut [f64]) {
    if length == 0.0 {
        return;
    }
    let w = 2.0 * v.iter().zip(&*x).map(|(a, b)| a * b).sum::<f64>() / length;
    for (value, a) in x.iter_mut().zip(v) {
        *value -= w * a;
    }
}

/// DESC's poloidal or toroidal factor: `cos(k x)` for `k >= 0`,
/// `sin(|k| x)` for `k < 0`.
fn harmonic(k: i32, x: f64) -> f64 {
    let (s, c) = (f64::from(k.abs()) * x).sin_cos();
    if k >= 0 {
        c
    } else {
        s
    }
}

/// Where a fit misses the surfaces by most, and by how much (cm).
struct Miss {
    distance: f64,
    surface: usize,
    theta: f64,
    zeta: f64,
}

/// The angular grid a fit is checked on: one field period, eight points or
/// more to the shortest wavelength in each angle.
struct Grid {
    thetas: Vec<f64>,
    zetas: Vec<f64>,
    m_max: i32,
    n_max: i32,
    /// `poloidal[(m + m_max) * thetas.len() + i]`: the poloidal factor of
    /// `m` at `thetas[i]`.
    poloidal: Vec<f64>,
    /// `toroidal[(n + n_max) * zetas.len() + i]`: the toroidal factor of `n`
    /// at `zetas[i]`.
    toroidal: Vec<f64>,
}

impl Grid {
    fn new(num_field_periods: u32, r: &Profiles, z: &Profiles) -> Self {
        let (m_max, n_max) = r
            .terms
            .iter()
            .chain(&z.terms)
            .fold((0, 0), |(m, n), t| (m.max(t[0].abs()), n.max(t[1].abs())));
        let n_theta = (8 * m_max as usize + 16).max(32);
        let n_zeta = (8 * n_max as usize + 16).max(16);
        let thetas: Vec<f64> = (0..n_theta)
            .map(|i| TAU * i as f64 / n_theta as f64)
            .collect();
        let zetas: Vec<f64> = (0..n_zeta)
            .map(|i| TAU / f64::from(num_field_periods) * i as f64 / n_zeta as f64)
            .collect();
        let table = |max: i32, angles: &[f64], scale: f64| -> Vec<f64> {
            (-max..=max)
                .flat_map(|k| angles.iter().map(move |&a| harmonic(k, scale * a)))
                .collect()
        };
        Self {
            poloidal: table(m_max, &thetas, 1.0),
            toroidal: table(n_max, &zetas, f64::from(num_field_periods)),
            thetas,
            zetas,
            m_max,
            n_max,
        }
    }

    fn points(&self) -> usize {
        self.thetas.len() * self.zetas.len()
    }

    /// `sum_t c(t) P_m(theta) T_n(zeta)` over the terms `(m, n)` (sorted by
    /// `m`) at every grid point, into `out[i * zetas.len() + k]`.
    fn values(&self, terms: &[[i32; 2]], c: impl Fn(usize) -> f64, out: &mut [f64]) {
        let (n_theta, n_zeta) = (self.thetas.len(), self.zetas.len());
        out.fill(0.0);
        let mut partial = vec![0.0; n_zeta];
        let mut t = 0;
        while t < terms.len() {
            // One poloidal factor at a time: its toroidal sum first.
            let m = terms[t][0];
            partial.fill(0.0);
            while t < terms.len() && terms[t][0] == m {
                let start = (terms[t][1] + self.n_max) as usize * n_zeta;
                let value = c(t);
                for (p, f) in partial.iter_mut().zip(&self.toroidal[start..]) {
                    *p += value * f;
                }
                t += 1;
            }
            let start = (m + self.m_max) as usize * n_theta;
            for (row, &g) in out
                .chunks_exact_mut(n_zeta)
                .zip(&self.poloidal[start..start + n_theta])
            {
                for (o, p) in row.iter_mut().zip(&partial) {
                    *o += g * p;
                }
            }
        }
    }

    /// The mean distance of the boundary from the magnetic axis.
    fn minor_radius(&self, r: &Profiles, z: &Profiles) -> f64 {
        let (mut dr, mut dz) = (vec![0.0; self.points()], vec![0.0; self.points()]);
        let rise = |p: &Profiles, t: usize| p.values[t][p.values[t].len() - 1] - p.values[t][0];
        self.values(&r.terms, |t| rise(r, t), &mut dr);
        self.values(&z.terms, |t| rise(z, t), &mut dz);
        dr.iter().zip(&dz).map(|(a, b)| a.hypot(*b)).sum::<f64>() / self.points() as f64
    }

    /// Where the residuals of a fit put it furthest from the surfaces.
    /// Surfaces are taken in decreasing order of `sum |residual|`, which
    /// bounds the distance, until no other can be further.
    fn miss(
        &self,
        r: &Profiles,
        r_residual: &[Vec<f64>],
        z: &Profiles,
        z_residual: &[Vec<f64>],
    ) -> Miss {
        let ns = r.values[0].len();
        let total =
            |residual: &[Vec<f64>], j: usize| -> f64 { residual.iter().map(|v| v[j].abs()).sum() };
        let bounds: Vec<f64> = (0..ns)
            .map(|j| f64::hypot(total(r_residual, j), total(z_residual, j)))
            .collect();
        let mut order: Vec<usize> = (0..ns).collect();
        order.sort_by(|&a, &b| bounds[b].total_cmp(&bounds[a]));
        let mut miss = Miss {
            distance: 0.0,
            surface: 0,
            theta: 0.0,
            zeta: 0.0,
        };
        let (mut dr, mut dz) = (vec![0.0; self.points()], vec![0.0; self.points()]);
        for j in order {
            if bounds[j] <= miss.distance {
                break;
            }
            self.values(&r.terms, |t| r_residual[t][j], &mut dr);
            self.values(&z.terms, |t| z_residual[t][j], &mut dz);
            for (p, (a, b)) in dr.iter().zip(&dz).enumerate() {
                let distance = a.hypot(*b);
                if !distance.is_finite() {
                    miss.distance = f64::INFINITY;
                    return miss;
                }
                if distance > miss.distance {
                    miss = Miss {
                        distance,
                        surface: j,
                        theta: self.thetas[p / self.zetas.len()],
                        zeta: self.zetas[p % self.zetas.len()],
                    };
                }
            }
        }
        miss
    }
}
