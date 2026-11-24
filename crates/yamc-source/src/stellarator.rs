//! Stellarator plasma source.
//!
//! The plasma shape is a Fourier-Zernike series in DESC's conventions:
//!
//! ```text
//! R(rho, theta, zeta) = sum R_lmn Z_l^m(rho, theta) F_n(zeta)
//! Z(rho, theta, zeta) = sum Z_lmn Z_l^m(rho, theta) F_n(zeta)
//! ```
//!
//! with `Z_l^m = R_l^|m|(rho) cos(m theta)` for `m >= 0` and
//! `R_l^|m|(rho) sin(|m| theta)` for `m < 0`, and `F_n = cos(n N zeta)` for
//! `n >= 0` and `sin(|n| N zeta)` for `n < 0`, `N` the number of field
//! periods. `R_l^m` is the unnormalised Zernike radial polynomial
//! (`R_l^m(1) = 1`), `rho` the square root of the normalised toroidal flux and
//! `zeta` the cylindrical toroidal angle. DESC's `R_lmn` / `Z_lmn` (times 100,
//! for cm) are these coefficients as they stand.
//!
//! The plasma volume element is `R * tau drho dtheta dzeta` with
//! `tau = R_rho Z_theta - R_theta Z_rho`, evaluated analytically. The plasma
//! is split into radial shells; each becomes one [`Source`] per reaction whose
//! spatial distribution is a [`FluxShell`] sampling
//! `S(rho) R |tau|` by rejection, and whose energy is the Ballabio spectrum at
//! the shell's emission-weighted ion temperature.
//!
//! [`StellaratorShape::from_vmec`] fits a shape to the flux surfaces of a
//! VMEC equilibrium.

mod vmec;

use std::f64::consts::TAU;
use std::sync::Arc;

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::distribution::angular::AngularDistribution;
use crate::distribution::check_finite;
use crate::distribution::energy::fusion_neutron_spectrum;
use crate::source::{Source, SourceEnergyDistribution, SourceSpatialDistribution};
use crate::tokamak::{
    fusion_reactions, neutron_source_density, validate_fuel, FuelIon, PlasmaProfiles,
};

/// Radial shells the plasma is split into. Each costs one source per
/// reaction; 50 resolves the profiles and keeps the source list short.
const SHELLS: usize = 50;
/// Gauss-Legendre points per shell for the shell strength integrals.
const GAUSS: [(f64, f64); 4] = [
    (-0.861_136_311_594_052_6, 0.347_854_845_137_453_9),
    (-0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
    (0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
    (0.861_136_311_594_052_6, 0.347_854_845_137_453_9),
];
/// Allowance on the rejection majorant for peaks between grid points.
const MAJORANT_MARGIN: f64 = 1.15;
/// Largest toroidal mode number |n| accepted, sizing the stack tables used
/// while sampling.
const MAX_N: usize = 64;

/// A stellarator plasma shape as a Fourier-Zernike series (see the module
/// docs for the conventions). Lengths in cm.
///
/// [`StellaratorShape::new`] checks the geometry: `R > 0` everywhere and a
/// volume element of one sign, so the surfaces are nested and do not
/// self-intersect. A model file is read with the structural checks only, as
/// it was checked when built.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ShapeData")]
pub struct StellaratorShape {
    num_field_periods: u32,
    r_modes: Vec<[i32; 3]>,
    r_lmn: Vec<f64>,
    z_modes: Vec<[i32; 3]>,
    z_lmn: Vec<f64>,
    /// How far a shape fitted by [`StellaratorShape::from_vmec`] is from the
    /// surfaces it was fitted to (cm).
    #[serde(skip_serializing_if = "Option::is_none")]
    fit_error: Option<f64>,
    /// Largest |n|, kept so evaluation does not rescan the modes.
    #[serde(skip)]
    n_max: usize,
}

#[derive(Deserialize)]
struct ShapeData {
    num_field_periods: u32,
    r_modes: Vec<[i32; 3]>,
    r_lmn: Vec<f64>,
    z_modes: Vec<[i32; 3]>,
    z_lmn: Vec<f64>,
    #[serde(default)]
    fit_error: Option<f64>,
}

impl TryFrom<ShapeData> for StellaratorShape {
    type Error = String;

    fn try_from(d: ShapeData) -> Result<Self, String> {
        let mut shape =
            Self::structural(d.num_field_periods, d.r_modes, d.r_lmn, d.z_modes, d.z_lmn)?;
        if let Some(error) = d.fit_error {
            check_finite("fit_error", error)?;
            if error < 0.0 {
                return Err(format!("fit_error must not be negative (got {error})"));
            }
        }
        shape.fit_error = d.fit_error;
        Ok(shape)
    }
}

/// Position and poloidal-plane derivatives at one `(rho, theta, zeta)`.
#[derive(Debug, Clone, Copy)]
pub struct ShapePoint {
    pub r: f64,
    pub z: f64,
    pub r_rho: f64,
    pub z_rho: f64,
    pub r_theta: f64,
    pub z_theta: f64,
}

impl ShapePoint {
    /// `R * tau`, the volume element per `drho dtheta dzeta`, signed by the
    /// orientation of the coordinates.
    pub fn jacobian(&self) -> f64 {
        self.r * (self.r_rho * self.z_theta - self.r_theta * self.z_rho)
    }
}

impl StellaratorShape {
    /// Build and check a shape. `r_modes` / `z_modes` are `(l, m, n)` triples
    /// paired with the coefficients `r_lmn` / `z_lmn` (cm).
    pub fn new(
        num_field_periods: u32,
        r_modes: Vec<[i32; 3]>,
        r_lmn: Vec<f64>,
        z_modes: Vec<[i32; 3]>,
        z_lmn: Vec<f64>,
    ) -> Result<Self, String> {
        let shape = Self::structural(num_field_periods, r_modes, r_lmn, z_modes, z_lmn)?;
        shape.check_geometry()?;
        Ok(shape)
    }

    fn structural(
        num_field_periods: u32,
        r_modes: Vec<[i32; 3]>,
        r_lmn: Vec<f64>,
        z_modes: Vec<[i32; 3]>,
        z_lmn: Vec<f64>,
    ) -> Result<Self, String> {
        if num_field_periods == 0 {
            return Err("num_field_periods must be at least 1".to_string());
        }
        for (name, modes, coefficients) in [("r", &r_modes, &r_lmn), ("z", &z_modes, &z_lmn)] {
            if modes.is_empty() || modes.len() != coefficients.len() {
                return Err(format!(
                    "{name}_modes and {name}_lmn must be non-empty and the same length (got {} \
                     and {})",
                    modes.len(),
                    coefficients.len()
                ));
            }
            for &[l, m, n] in modes.iter() {
                if l < m.abs() || (l - m.abs()) % 2 != 0 {
                    return Err(format!(
                        "{name}_modes has (l, m, n) = ({l}, {m}, {n}): a Zernike mode needs \
                         l >= |m| and l - |m| even"
                    ));
                }
                if n.unsigned_abs() as usize > MAX_N {
                    return Err(format!(
                        "{name}_modes has toroidal mode n = {n}; |n| up to {MAX_N} is supported"
                    ));
                }
            }
            let mut sorted = modes.clone();
            sorted.sort_unstable();
            if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
                return Err(format!("{name}_modes lists ({:?}) more than once", w[0]));
            }
            for value in coefficients.iter() {
                check_finite(&format!("{name}_lmn"), *value)?;
            }
        }
        // Sorted by (|m|, l) so evaluation advances one Jacobi recurrence per
        // |m| instead of restarting it for every mode.
        let sort = |modes: Vec<[i32; 3]>, lmn: Vec<f64>| {
            let mut pairs: Vec<_> = modes.into_iter().zip(lmn).collect();
            pairs.sort_by_key(|([l, m, n], _)| (m.abs(), *l, *m, *n));
            pairs.into_iter().unzip()
        };
        let (r_modes, r_lmn) = sort(r_modes, r_lmn);
        let (z_modes, z_lmn) = sort(z_modes, z_lmn);
        let mut shape = Self {
            num_field_periods,
            r_modes,
            r_lmn,
            z_modes,
            z_lmn,
            fit_error: None,
            n_max: 0,
        };
        shape.n_max = shape.resolution().2;
        Ok(shape)
    }

    pub fn num_field_periods(&self) -> u32 {
        self.num_field_periods
    }

    pub fn r_modes(&self) -> &[[i32; 3]] {
        &self.r_modes
    }

    pub fn r_lmn(&self) -> &[f64] {
        &self.r_lmn
    }

    pub fn z_modes(&self) -> &[[i32; 3]] {
        &self.z_modes
    }

    pub fn z_lmn(&self) -> &[f64] {
        &self.z_lmn
    }

    /// For a shape fitted by [`StellaratorShape::from_vmec`], the largest
    /// distance (cm) between it and the VMEC surfaces, over every surface
    /// and an angular grid.
    pub fn fit_error(&self) -> Option<f64> {
        self.fit_error
    }

    /// Largest `l`, `|m|` and `|n|` over both series.
    fn resolution(&self) -> (usize, usize, usize) {
        self.r_modes
            .iter()
            .chain(&self.z_modes)
            .fold((0, 0, 0), |(l, m, n), mode| {
                (
                    l.max(mode[0] as usize),
                    m.max(mode[1].unsigned_abs() as usize),
                    n.max(mode[2].unsigned_abs() as usize),
                )
            })
    }

    /// Evaluate the shape and its `rho` / `theta` derivatives.
    pub fn point(&self, rho: f64, theta: f64, zeta: f64) -> ShapePoint {
        // cos / sin(k N zeta) for k = 0..=MAX_N by angle addition.
        let mut cos_n = [0.0; MAX_N + 1];
        let mut sin_n = [0.0; MAX_N + 1];
        let (s1, c1) = (self.num_field_periods as f64 * zeta).sin_cos();
        cos_n[0] = 1.0;
        for k in 1..=self.n_max {
            cos_n[k] = cos_n[k - 1] * c1 - sin_n[k - 1] * s1;
            sin_n[k] = sin_n[k - 1] * c1 + cos_n[k - 1] * s1;
        }
        let toroidal = |n: i32| {
            let k = n.unsigned_abs() as usize;
            if n >= 0 {
                cos_n[k]
            } else {
                sin_n[k]
            }
        };
        let (r, r_rho, r_theta) = series(&self.r_modes, &self.r_lmn, rho, theta, &toroidal);
        let (z, z_rho, z_theta) = series(&self.z_modes, &self.z_lmn, rho, theta, &toroidal);
        ShapePoint {
            r,
            z,
            r_rho,
            z_rho,
            r_theta,
            z_theta,
        }
    }

    /// The flux surface `rho` as a double Fourier series in `(theta, zeta)`:
    /// `(m, n)` modes and coefficients (cm) for `R` and for `Z`, in the same
    /// convention as the shape (`m < 0` selects `sin(|m| theta)`, `n < 0`
    /// selects `sin(|n| N zeta)`). Each coefficient is the shape's
    /// `sum_l R_lmn R_l^|m|(rho)`, so the series is exactly the surface.
    #[allow(clippy::type_complexity)]
    pub fn surface_modes(&self, rho: f64) -> (Vec<[i32; 2]>, Vec<f64>, Vec<[i32; 2]>, Vec<f64>) {
        let collapse = |modes: &[[i32; 3]], lmn: &[f64]| {
            let mut groups = std::collections::BTreeMap::<(i32, i32), Vec<(i32, f64)>>::new();
            for (&[l, m, n], &c) in modes.iter().zip(lmn) {
                groups.entry((m, n)).or_default().push((l, c));
            }
            groups
                .into_iter()
                .map(|((m, n), mut terms)| {
                    terms.sort_by_key(|&(l, _)| l);
                    ([m, n], radial_sum(m.abs(), &terms, rho).0)
                })
                .unzip::<_, _, Vec<_>, Vec<_>>()
        };
        let (r_modes, r_mn) = collapse(&self.r_modes, &self.r_lmn);
        let (z_modes, z_mn) = collapse(&self.z_modes, &self.z_lmn);
        (r_modes, r_mn, z_modes, z_mn)
    }

    /// Plasma volume (cm^3).
    pub fn volume(&self) -> f64 {
        let grid = AngularGrid::new(self);
        let mut surface = Surface::default();
        let mut volume = 0.0;
        for shell in 0..SHELLS {
            let (lo, hi) = shell_bounds(shell);
            for (x, w) in GAUSS {
                let rho = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x;
                volume += 0.5 * (hi - lo) * w * grid.stats(rho, &mut surface).0;
            }
        }
        volume
    }

    /// Refuse `R <= 0` and a volume element changing sign, which means the
    /// surfaces self-intersect or are not nested.
    fn check_geometry(&self) -> Result<(), String> {
        let (l_max, _, _) = self.resolution();
        let n_rho = (2 * l_max + 8).max(16);
        let grid = AngularGrid::new(self);
        let mut surface = Surface::default();
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 1..=n_rho {
            let rho = i as f64 / n_rho as f64;
            grid.surface(rho, &mut surface);
            for p in 0..grid.points {
                let r = surface.r[p];
                if r <= 0.0 {
                    return Err(format!(
                        "the plasma reaches R = {r} cm <= 0 at rho = {rho}; check the \
                         coefficients (they should be in cm)"
                    ));
                }
                // Normalised by rho, as the volume element vanishes on the
                // axis.
                let f = surface.jacobian(p) / rho;
                low = low.min(f);
                high = high.max(f);
            }
        }
        if low < 0.0 && high > 0.0 && (-low).min(high) > 1.0e-6 * (-low).max(high) {
            return Err(
                "the volume element changes sign inside the plasma: the surfaces \
                 self-intersect or are not nested"
                    .to_string(),
            );
        }
        if low.abs().max(high.abs()) == 0.0 {
            return Err("the shape has zero volume".to_string());
        }
        Ok(())
    }
}

/// A Fourier-Zernike series tabulated on the angular grid: its modes grouped
/// by `(m, n)`, each group's angular factor and theta derivative stored at
/// every grid point, so a surface costs one radial sum per group plus one
/// multiply-add per group and point.
struct TabulatedSeries {
    /// Per `(m, n)` group: `|m|` and the `(l, coefficient)` terms, by `l`.
    columns: Vec<(i32, Vec<(i32, f64)>)>,
    /// `angle[c * points + p]`: the angular factor of group `c` at point `p`.
    angle: Vec<f64>,
    angle_theta: Vec<f64>,
}

impl TabulatedSeries {
    fn new(modes: &[[i32; 3]], lmn: &[f64], nfp: u32, thetas: &[f64], zetas: &[f64]) -> Self {
        let mut groups = std::collections::BTreeMap::<(i32, i32), Vec<(i32, f64)>>::new();
        for (&[l, m, n], &c) in modes.iter().zip(lmn) {
            groups.entry((m, n)).or_default().push((l, c));
        }
        let points = thetas.len() * zetas.len();
        let mut angle = Vec::with_capacity(groups.len() * points);
        let mut angle_theta = Vec::with_capacity(groups.len() * points);
        let mut columns = Vec::with_capacity(groups.len());
        for ((m, n), mut terms) in groups {
            terms.sort_by_key(|t| t.0);
            let a = m.abs() as f64;
            for &theta in thetas {
                let (s, c) = (a * theta).sin_cos();
                let (value, derivative) = if m >= 0 { (c, -a * s) } else { (s, a * c) };
                for &zeta in zetas {
                    let (sz, cz) = (n.unsigned_abs() as f64 * nfp as f64 * zeta).sin_cos();
                    let toroidal = if n >= 0 { cz } else { sz };
                    angle.push(value * toroidal);
                    angle_theta.push(derivative * toroidal);
                }
            }
            columns.push((m.abs(), terms));
        }
        Self {
            columns,
            angle,
            angle_theta,
        }
    }

    /// Accumulate the series and its `rho` / `theta` derivatives at every
    /// grid point of surface `rho`.
    fn surface(&self, rho: f64, value: &mut [f64], d_rho: &mut [f64], d_theta: &mut [f64]) {
        let points = value.len();
        value.fill(0.0);
        d_rho.fill(0.0);
        d_theta.fill(0.0);
        for (c, (alpha, terms)) in self.columns.iter().enumerate() {
            let (a, da) = radial_sum(*alpha, terms, rho);
            let angle = &self.angle[c * points..(c + 1) * points];
            let angle_theta = &self.angle_theta[c * points..(c + 1) * points];
            for p in 0..points {
                value[p] += a * angle[p];
                d_rho[p] += da * angle[p];
                d_theta[p] += a * angle_theta[p];
            }
        }
    }
}

/// `sum c R_l^alpha(rho)` over `terms` (sorted by `l`) and its `rho`
/// derivative, advancing one Jacobi recurrence.
fn radial_sum(alpha: i32, terms: &[(i32, f64)], rho: f64) -> (f64, f64) {
    let x = 1.0 - 2.0 * rho * rho;
    let rho_m = rho.powi(alpha);
    let d_rho_m = if alpha == 0 {
        0.0
    } else {
        alpha as f64 * rho.powi(alpha - 1)
    };
    let (mut k, mut p, mut p_prev, mut dp, mut dp_prev) = (0, 1.0, 0.0, 0.0, 0.0);
    let (mut sum, mut d_sum) = (0.0, 0.0);
    for &(l, c) in terms {
        while k < (l - alpha) / 2 {
            (p_prev, p, dp_prev, dp) = jacobi_step(k + 1, alpha as f64, x, p, p_prev, dp, dp_prev);
            k += 1;
        }
        let sign = if k % 2 == 0 { c } else { -c };
        sum += sign * rho_m * p;
        d_sum += sign * (d_rho_m * p - 4.0 * rho * rho_m * dp);
    }
    (sum, d_sum)
}

/// A shape tabulated on the angular grid over one field period on which the
/// trapezoid rule integrates `R * tau` exactly: it is a trigonometric
/// polynomial of degree at most `3 |m|max` in theta and `3 |n|max` in
/// `N zeta`, and the grid has more points than that.
struct AngularGrid {
    points: usize,
    r: TabulatedSeries,
    z: TabulatedSeries,
}

/// One surface evaluated on the angular grid.
#[derive(Default)]
struct Surface {
    r: Vec<f64>,
    r_rho: Vec<f64>,
    r_theta: Vec<f64>,
    z: Vec<f64>,
    z_rho: Vec<f64>,
    z_theta: Vec<f64>,
}

impl Surface {
    fn jacobian(&self, p: usize) -> f64 {
        self.r[p] * (self.r_rho[p] * self.z_theta[p] - self.r_theta[p] * self.z_rho[p])
    }
}

impl AngularGrid {
    fn new(shape: &StellaratorShape) -> Self {
        let (_, m_max, n_max) = shape.resolution();
        let (n_theta, n_zeta) = ((4 * m_max + 8).max(32), (4 * n_max + 8).max(16));
        let period = TAU / shape.num_field_periods as f64;
        let thetas: Vec<f64> = (0..n_theta)
            .map(|i| TAU * i as f64 / n_theta as f64)
            .collect();
        let zetas: Vec<f64> = (0..n_zeta)
            .map(|j| period * j as f64 / n_zeta as f64)
            .collect();
        let nfp = shape.num_field_periods;
        Self {
            points: n_theta * n_zeta,
            r: TabulatedSeries::new(&shape.r_modes, &shape.r_lmn, nfp, &thetas, &zetas),
            z: TabulatedSeries::new(&shape.z_modes, &shape.z_lmn, nfp, &thetas, &zetas),
        }
    }

    fn surface(&self, rho: f64, s: &mut Surface) {
        for v in [
            &mut s.r,
            &mut s.r_rho,
            &mut s.r_theta,
            &mut s.z,
            &mut s.z_rho,
            &mut s.z_theta,
        ] {
            v.resize(self.points, 0.0);
        }
        self.r.surface(rho, &mut s.r, &mut s.r_rho, &mut s.r_theta);
        self.z.surface(rho, &mut s.z, &mut s.z_rho, &mut s.z_theta);
    }

    /// At `rho`: the volume derivative `dV/drho` (full torus, unsigned) and
    /// the largest `|R * tau| / rho` over the grid.
    fn stats(&self, rho: f64, s: &mut Surface) -> (f64, f64) {
        self.surface(rho, s);
        let (mut sum, mut peak) = (0.0, 0.0_f64);
        for p in 0..self.points {
            let f = s.jacobian(p);
            sum += f;
            peak = peak.max(f.abs());
        }
        ((sum * TAU * TAU / self.points as f64).abs(), peak / rho)
    }
}

/// One Fourier-Zernike series and its `rho` / `theta` derivatives. `modes`
/// must be sorted by `(|m|, l)`, as [`StellaratorShape`] keeps them.
fn series(
    modes: &[[i32; 3]],
    lmn: &[f64],
    rho: f64,
    theta: f64,
    toroidal: &impl Fn(i32) -> f64,
) -> (f64, f64, f64) {
    let x = 1.0 - 2.0 * rho * rho;
    let (mut value, mut d_rho, mut d_theta) = (0.0, 0.0, 0.0);
    // Jacobi P_k^(alpha, 0)(x) recurrence state for the current alpha = |m|.
    let mut alpha = -1;
    let (mut k, mut p, mut p_prev, mut dp, mut dp_prev) = (0, 1.0, 0.0, 0.0, 0.0);
    let (mut rho_m, mut d_rho_m, mut cos_m, mut sin_m) = (1.0, 0.0, 1.0, 0.0);
    for (&[l, m, n], &c) in modes.iter().zip(lmn) {
        let a = m.abs();
        if a != alpha {
            alpha = a;
            (k, p, p_prev, dp, dp_prev) = (0, 1.0, 0.0, 0.0, 0.0);
            rho_m = rho.powi(a);
            d_rho_m = if a == 0 {
                0.0
            } else {
                a as f64 * rho.powi(a - 1)
            };
            (sin_m, cos_m) = (a as f64 * theta).sin_cos();
        }
        let target = (l - a) / 2;
        while k < target {
            (p_prev, p, dp_prev, dp) = jacobi_step(k + 1, a as f64, x, p, p_prev, dp, dp_prev);
            k += 1;
        }
        // R_l^|m| = (-1)^k rho^|m| P_k(1 - 2 rho^2).
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let radial = sign * rho_m * p;
        let radial_d = sign * (d_rho_m * p - 4.0 * rho * rho_m * dp);
        let (angle, angle_d) = if m >= 0 {
            (cos_m, -(a as f64) * sin_m)
        } else {
            (sin_m, a as f64 * cos_m)
        };
        let c = c * toroidal(n);
        value += c * radial * angle;
        d_rho += c * radial_d * angle;
        d_theta += c * radial * angle_d;
    }
    (value, d_rho, d_theta)
}

/// Advance the Jacobi `P^(alpha, 0)` recurrence to degree `k` from degrees
/// `k - 1` (`p`) and `k - 2` (`p_prev`), carrying the derivatives in `x`.
/// Returns `(P_{k-1}, P_k, P'_{k-1}, P'_k)`.
fn jacobi_step(
    k: i32,
    alpha: f64,
    x: f64,
    p: f64,
    p_prev: f64,
    dp: f64,
    dp_prev: f64,
) -> (f64, f64, f64, f64) {
    if k == 1 {
        let next = (alpha + 1.0) + 0.5 * (alpha + 2.0) * (x - 1.0);
        return (p, next, dp, 0.5 * (alpha + 2.0));
    }
    let n = k as f64;
    let s = 2.0 * n + alpha;
    let a = 2.0 * n * (n + alpha) * (s - 2.0);
    let b = (s - 1.0) * (s * (s - 2.0) * x + alpha * alpha);
    let c = 2.0 * (n + alpha - 1.0) * (n - 1.0) * s;
    let next = (b * p - c * p_prev) / a;
    let d_next = (b * dp + (s - 1.0) * s * (s - 2.0) * p - c * dp_prev) / a;
    (p, next, dp, d_next)
}

/// `rho` bounds of shell `index`.
fn shell_bounds(index: usize) -> (f64, f64) {
    (
        index as f64 / SHELLS as f64,
        (index + 1) as f64 / SHELLS as f64,
    )
}

/// The part of a stellarator plasma between two flux surfaces, sampled with
/// density `S(rho) R |tau|`, `S` interpolated linearly between the surfaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FluxShellData")]
pub struct FluxShell {
    pub shape: Arc<StellaratorShape>,
    /// Inner and outer `rho`.
    pub rho: [f64; 2],
    /// Emission density at the inner and outer surface (relative).
    pub emission: [f64; 2],
    /// Upper bound of `S(rho) R |tau| / rho` over the shell.
    pub majorant: f64,
}

#[derive(Deserialize)]
struct FluxShellData {
    shape: Arc<StellaratorShape>,
    rho: [f64; 2],
    emission: [f64; 2],
    majorant: f64,
}

impl TryFrom<FluxShellData> for FluxShell {
    type Error = String;

    fn try_from(d: FluxShellData) -> Result<Self, String> {
        let [lo, hi] = d.rho;
        if !(0.0 <= lo && lo < hi && hi <= 1.0) {
            return Err(format!(
                "FluxShell rho must satisfy 0 <= inner < outer <= 1 (got {lo}, {hi})"
            ));
        }
        for value in d.emission {
            check_finite("FluxShell emission", value)?;
            if value < 0.0 {
                return Err(format!(
                    "FluxShell emission must not be negative (got {value})"
                ));
            }
        }
        check_finite("FluxShell majorant", d.majorant)?;
        if d.majorant <= 0.0 {
            return Err(format!(
                "FluxShell majorant must be positive (got {})",
                d.majorant
            ));
        }
        Ok(Self {
            shape: d.shape,
            rho: d.rho,
            emission: d.emission,
            majorant: d.majorant,
        })
    }
}

impl FluxShell {
    /// Sample a position (cm). `rho` is proposed with density `~ rho` (the
    /// volume element vanishes linearly on the axis), the angles uniformly,
    /// and accepted against the majorant; the field period is then chosen at
    /// random.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        let periods = self.shape.num_field_periods;
        let period = TAU / periods as f64;
        let [lo, hi] = self.rho;
        let (lo2, hi2) = (lo * lo, hi * hi);
        loop {
            let rho = (lo2 + rng.random::<f64>() * (hi2 - lo2)).sqrt();
            let theta = TAU * rng.random::<f64>();
            let zeta = period * rng.random::<f64>();
            let p = self.shape.point(rho, theta, zeta);
            let t = (rho - lo) / (hi - lo);
            let emission = self.emission[0] + t * (self.emission[1] - self.emission[0]);
            // NaN when rho is exactly 0, which the comparison rejects.
            let density = emission * p.jacobian().abs() / rho;
            if rng.random::<f64>() * self.majorant < density {
                let phi = zeta + period * rng.random_range(0..periods) as f64;
                let (s, c) = phi.sin_cos();
                return [p.r * c, p.r * s, p.z];
            }
        }
    }
}

/// A stellarator plasma: a shape, the Fausser profiles in `rho` and a fuel.
#[derive(Debug, Clone, PartialEq)]
pub struct StellaratorPlasma {
    pub shape: Arc<StellaratorShape>,
    pub profiles: PlasmaProfiles,
    /// Fuel species and their atom fractions, which must sum to 1.
    pub fuel: Vec<(FuelIon, f64)>,
}

impl StellaratorPlasma {
    /// One source per (radial shell, reaction), isotropic, with the Ballabio
    /// spectrum at the shell's emission-weighted ion temperature. Strengths
    /// are normalised to sum to 1.
    pub fn sources(&self) -> Result<Vec<Source>, String> {
        self.profiles.validate()?;
        validate_fuel(&self.fuel)?;
        let reactions = fusion_reactions(&self.fuel)?;
        let profiles = &self.profiles;
        let emission = |rho: f64, reactants, pair_factor: f64| {
            let density = profiles.ion_density(rho);
            neutron_source_density(
                pair_factor * density * density,
                profiles.ion_temperature(rho),
                reactants,
            )
        };

        // Angular statistics at each shell edge and Gauss point, shared by
        // every reaction.
        let grid = AngularGrid::new(&self.shape);
        let mut surface = Surface::default();
        let mut shells = Vec::with_capacity(SHELLS);
        for index in 0..SHELLS {
            let (lo, hi) = shell_bounds(index);
            let gauss: Vec<(f64, f64, f64)> = GAUSS
                .iter()
                .map(|(x, w)| {
                    let rho = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x;
                    let (volume_derivative, peak) = grid.stats(rho, &mut surface);
                    (rho, 0.5 * (hi - lo) * w * volume_derivative, peak)
                })
                .collect();
            let peak = [lo.max(1.0e-6), 0.5 * (lo + hi), hi]
                .iter()
                .map(|&rho| grid.stats(rho, &mut surface).1)
                .chain(gauss.iter().map(|g| g.2))
                .fold(0.0, f64::max);
            shells.push((lo, hi, gauss, peak));
        }

        let mut raw = Vec::new();
        for (reactants, pair_factor) in &reactions {
            for (lo, hi, gauss, peak) in &shells {
                let (mut strength, mut weighted_temperature) = (0.0, 0.0);
                for &(rho, volume, _) in gauss {
                    let weight = emission(rho, *reactants, *pair_factor) * volume;
                    strength += weight;
                    weighted_temperature += weight * profiles.ion_temperature(rho);
                }
                if strength <= 0.0 {
                    continue;
                }
                let ends = [
                    emission(*lo, *reactants, *pair_factor),
                    emission(*hi, *reactants, *pair_factor),
                ];
                raw.push((
                    *reactants,
                    strength,
                    weighted_temperature / strength,
                    FluxShell {
                        shape: Arc::clone(&self.shape),
                        rho: [*lo, *hi],
                        emission: ends,
                        majorant: MAJORANT_MARGIN * ends[0].max(ends[1]) * peak,
                    },
                ));
            }
        }
        let total: f64 = raw.iter().map(|r| r.1).sum();
        if total <= 0.0 {
            return Err(
                "total neutron source density is zero: the ion temperatures or densities are \
                 too low to produce fusion reactions"
                    .to_string(),
            );
        }
        raw.into_iter()
            .filter(|r| r.3.majorant > 0.0)
            .map(|(reactants, strength, temperature, shell)| {
                Ok(Source {
                    space: SourceSpatialDistribution::FluxShell(Box::new(shell)),
                    angle: AngularDistribution::Isotropic,
                    energy: SourceEnergyDistribution::Normal(fusion_neutron_spectrum(
                        temperature,
                        reactants,
                    )?),
                    strength: strength / total,
                })
            })
            .collect()
    }
}
