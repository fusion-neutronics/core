//! Tests for fitting a stellarator shape to VMEC Fourier coefficients.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use yamc_source::stellarator::{StellaratorPlasma, StellaratorShape};
use yamc_source::tokamak::{ConfinementMode, FuelIon, PlasmaProfiles, TokamakShape};

const R0: f64 = 600.0;
const A: f64 = 150.0;
const E: f64 = 50.0;

/// VMEC coefficients as a `wout` file holds them: `rmnc[j][k]` is mode
/// `(xm[k], xn[k])` on surface `s_j = j / (ns - 1)`.
struct Vmec {
    nfp: u32,
    xm: Vec<f64>,
    xn: Vec<f64>,
    rmnc: Vec<Vec<f64>>,
    zmns: Vec<Vec<f64>>,
    rmns: Option<Vec<Vec<f64>>>,
    zmnc: Option<Vec<Vec<f64>>>,
}

impl Vmec {
    fn fit(&self) -> Result<StellaratorShape, String> {
        StellaratorShape::from_vmec(
            self.nfp,
            &self.xm,
            &self.xn,
            &self.rmnc,
            &self.zmns,
            self.rmns.as_deref(),
            self.zmnc.as_deref(),
        )
    }

    fn rho(&self, j: usize) -> f64 {
        (j as f64 / (self.rmnc.len() - 1) as f64).sqrt()
    }

    /// VMEC's `(R, Z)` on surface `j`, summed term by term.
    fn point(&self, j: usize, theta: f64, zeta: f64) -> (f64, f64) {
        let (mut r, mut z) = (0.0, 0.0);
        for k in 0..self.xm.len() {
            let (s, c) = (self.xm[k] * theta - self.xn[k] * zeta).sin_cos();
            r += self.rmnc[j][k] * c + self.rmns.as_ref().map_or(0.0, |a| a[j][k]) * s;
            z += self.zmns[j][k] * s + self.zmnc.as_ref().map_or(0.0, |a| a[j][k]) * c;
        }
        (r, z)
    }
}

/// The rotating ellipse `R = R0 + a rho cos(t) + e rho cos(t - N z)`,
/// `Z = z_sign (a rho sin(t) - e rho sin(t - N z))`, as VMEC writes it.
fn rotating_ellipse_vmec(ns: usize, nfp: u32, z_sign: f64) -> Vmec {
    let rho = |j: usize| (j as f64 / (ns - 1) as f64).sqrt();
    Vmec {
        nfp,
        xm: vec![0.0, 1.0, 1.0],
        xn: vec![0.0, 0.0, f64::from(nfp)],
        rmnc: (0..ns).map(|j| vec![R0, A * rho(j), E * rho(j)]).collect(),
        zmns: (0..ns)
            .map(|j| vec![0.0, z_sign * A * rho(j), -z_sign * E * rho(j)])
            .collect(),
        rmns: None,
        zmnc: None,
    }
}

/// The same rotating ellipse in the product basis.
fn rotating_ellipse(nfp: u32) -> StellaratorShape {
    StellaratorShape::new(
        nfp,
        vec![[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, -1, -1]],
        vec![R0, A, E, E],
        vec![[1, -1, 0], [1, -1, 1], [1, 1, -1]],
        vec![A, -E, E],
    )
    .unwrap()
}

/// A shape with radial structure up to `l = 10`, in every product term a
/// stellarator-symmetric shape has, plus, with `asymmetric`, the terms that
/// break the symmetry.
fn rich_shape(asymmetric: bool) -> StellaratorShape {
    let mut r = vec![
        ([0, 0, 0], R0),
        ([1, 1, 0], A),
        ([2, 0, 0], -10.0),
        ([1, 1, 1], 20.0),
        ([1, -1, -1], 20.0),
        ([3, 1, 0], 5.0),
        ([2, 2, 1], 4.0),
        ([2, -2, -1], 3.0),
        ([4, 0, 1], 2.0),
        ([5, 3, 0], 1.5),
        ([6, -2, -2], 1.0),
        ([8, 0, 2], 0.8),
        ([7, 3, 1], 0.5),
        ([10, 4, 0], 0.3),
    ];
    let mut z = vec![
        ([1, -1, 0], A),
        ([1, -1, 1], -20.0),
        ([1, 1, -1], 20.0),
        ([3, -1, 0], 4.0),
        ([2, -2, 1], 3.0),
        ([2, 2, -1], -2.0),
        ([4, 0, -1], 1.5),
        ([5, -3, 0], 1.0),
        ([6, 2, -2], 0.7),
        ([8, -4, 1], 0.4),
        ([9, -1, 2], 0.3),
    ];
    if asymmetric {
        r.extend([([1, -1, 0], 8.0), ([2, 0, -1], 3.0), ([3, 1, -2], 1.0)]);
        z.extend([([0, 0, 0], 12.0), ([2, 0, 1], 3.0), ([3, -1, -1], 2.0)]);
    }
    let (r_modes, r_lmn) = r.into_iter().unzip();
    let (z_modes, z_lmn) = z.into_iter().unzip();
    StellaratorShape::new(3, r_modes, r_lmn, z_modes, z_lmn).unwrap()
}

/// Tabulate `shape` on `ns` surfaces and project each one onto VMEC's
/// `cos(m theta - n zeta)` / `sin(m theta - n zeta)` by discrete Fourier
/// transform, which is exact for a grid finer than twice the highest mode.
fn vmec_of(shape: &StellaratorShape, ns: usize) -> Vmec {
    let nfp = shape.num_field_periods();
    let all = shape.r_modes().iter().chain(shape.z_modes());
    let m_max = all.clone().map(|mode| mode[1].abs()).max().unwrap();
    let n_max = all.map(|mode| mode[2].abs()).max().unwrap();
    let mut modes = vec![];
    for m in 0..=m_max {
        let low = if m == 0 { 0 } else { -n_max };
        modes.extend((low..=n_max).map(|n| (m, n)));
    }
    let (n_theta, n_zeta) = (4 * (m_max + 1) as usize, 4 * (n_max + 1) as usize);
    let period = TAU / f64::from(nfp);
    let mut vmec = Vmec {
        nfp,
        xm: modes.iter().map(|&(m, _)| f64::from(m)).collect(),
        xn: modes
            .iter()
            .map(|&(_, n)| f64::from(n * nfp as i32))
            .collect(),
        rmnc: vec![],
        zmns: vec![],
        rmns: Some(vec![]),
        zmnc: Some(vec![]),
    };
    for j in 0..ns {
        let rho = (j as f64 / (ns - 1) as f64).sqrt();
        let (mut rc, mut rs, mut zc, mut zs) = (
            vec![0.0; modes.len()],
            vec![0.0; modes.len()],
            vec![0.0; modes.len()],
            vec![0.0; modes.len()],
        );
        for i in 0..n_theta {
            let theta = TAU * i as f64 / n_theta as f64;
            for k in 0..n_zeta {
                let zeta = period * k as f64 / n_zeta as f64;
                let p = shape.point(rho, theta, zeta);
                for (t, &(m, n)) in modes.iter().enumerate() {
                    let angle = f64::from(m) * theta - f64::from(n) * f64::from(nfp) * zeta;
                    let (s, c) = angle.sin_cos();
                    rc[t] += p.r * c;
                    rs[t] += p.r * s;
                    zc[t] += p.z * c;
                    zs[t] += p.z * s;
                }
            }
        }
        for (t, &(m, n)) in modes.iter().enumerate() {
            let norm = if (m, n) == (0, 0) { 1.0 } else { 2.0 } / (n_theta * n_zeta) as f64;
            for value in [&mut rc[t], &mut rs[t], &mut zc[t], &mut zs[t]] {
                *value *= norm;
            }
        }
        vmec.rmnc.push(rc);
        vmec.zmns.push(zs);
        vmec.rmns.as_mut().unwrap().push(rs);
        vmec.zmnc.as_mut().unwrap().push(zc);
    }
    vmec
}

/// Assert the shapes agree on `R` and `Z` at random `(rho, theta, zeta)`.
fn assert_same_shape(fitted: &StellaratorShape, original: &StellaratorShape) {
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..500 {
        let (rho, theta, zeta) = (
            rng.random::<f64>(),
            TAU * rng.random::<f64>(),
            TAU * rng.random::<f64>(),
        );
        let (p, q) = (
            fitted.point(rho, theta, zeta),
            original.point(rho, theta, zeta),
        );
        assert!(
            (p.r - q.r).abs() < 1.0e-9 * R0 && (p.z - q.z).abs() < 1.0e-9 * R0,
            "at ({rho}, {theta}, {zeta}): fitted ({}, {}) vs ({}, {})",
            p.r,
            p.z,
            q.r,
            q.z
        );
    }
}

/// Signed area of the `rho = 1` cut at `zeta`, positive when theta runs
/// counter-clockwise in the `(R, Z)` plane.
fn cut_area(point: impl Fn(f64) -> (f64, f64)) -> f64 {
    let n = 256;
    let points: Vec<(f64, f64)> = (0..n).map(|i| point(TAU * i as f64 / n as f64)).collect();
    (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum::<f64>()
        / 2.0
}

#[test]
fn rotating_ellipse_round_trips() {
    let vmec = rotating_ellipse_vmec(21, 5, 1.0);
    let fitted = vmec.fit().unwrap();
    let original = rotating_ellipse(5);
    assert_same_shape(&fitted, &original);
    // One radial polynomial per term is exact, so the modes are DESC's own.
    let sorted = |modes: &[[i32; 3]], lmn: &[f64]| {
        let mut pairs: Vec<_> = modes.iter().copied().zip(lmn.iter().copied()).collect();
        pairs.sort_by_key(|p| p.0);
        pairs
    };
    for (fitted, original) in [
        (
            sorted(fitted.r_modes(), fitted.r_lmn()),
            sorted(original.r_modes(), original.r_lmn()),
        ),
        (
            sorted(fitted.z_modes(), fitted.z_lmn()),
            sorted(original.z_modes(), original.z_lmn()),
        ),
    ] {
        assert_eq!(fitted.len(), original.len(), "{fitted:?} vs {original:?}");
        for ((mode, c), (expected_mode, expected)) in fitted.iter().zip(&original) {
            assert_eq!(mode, expected_mode);
            assert!(
                (c - expected).abs() < 1.0e-9 * R0,
                "{mode:?}: {c} vs {expected}"
            );
        }
    }
    assert!(fitted.fit_error().unwrap() < 1.0e-9 * A);
    let volume = 2.0 * PI * PI * R0 * (A + E) * (A - E);
    assert!((fitted.volume() - volume).abs() < 1.0e-9 * volume);
    assert_eq!(original.fit_error(), None);
}

#[test]
fn rich_shape_round_trips() {
    let original = rich_shape(false);
    let mut vmec = vmec_of(&original, 41);
    // Symmetric: the sine terms of R and cosine terms of Z are rounding.
    for rows in [vmec.rmns.take().unwrap(), vmec.zmnc.take().unwrap()] {
        assert!(rows.iter().flatten().all(|v| v.abs() < 1.0e-9));
    }
    let fitted = vmec.fit().unwrap();
    assert_same_shape(&fitted, &original);
    assert!(fitted.fit_error().unwrap() < 1.0e-9 * A);
    assert!((fitted.volume() - original.volume()).abs() < 1.0e-9 * original.volume());
}

#[test]
fn asymmetric_shape_round_trips() {
    let original = rich_shape(true);
    let vmec = vmec_of(&original, 41);
    let fitted = vmec.fit().unwrap();
    assert_same_shape(&fitted, &original);
    assert!(fitted.fit_error().unwrap() < 1.0e-9 * A);
    // Without its sine terms of R and cosine terms of Z, it is another shape.
    let symmetric = Vmec {
        rmns: None,
        zmnc: None,
        ..vmec
    }
    .fit()
    .unwrap();
    assert!((symmetric.point(1.0, 1.0, 0.0).z - original.point(1.0, 1.0, 0.0).z).abs() > 1.0);
}

#[test]
fn vmec_angles_and_orientation_are_kept() {
    // A wout's theta runs counter-clockwise in (R, Z), as TokamakShape's
    // does; the fit keeps it, and keeps a clockwise one clockwise too.
    let tokamak = TokamakShape::new(R0, A, 1.6, 0.4, 10.0).unwrap();
    assert!(cut_area(|theta| tokamak.point(1.0, theta)) > 0.0);
    for z_sign in [1.0, -1.0] {
        let vmec = rotating_ellipse_vmec(21, 5, z_sign);
        let fitted = vmec.fit().unwrap();
        for j in [1, 7, 20] {
            for (theta, zeta) in [(0.0, 0.0), (0.7, 0.2), (2.0, 1.1), (4.5, 3.0)] {
                let p = fitted.point(vmec.rho(j), theta, zeta);
                let (r, z) = vmec.point(j, theta, zeta);
                assert!(
                    (p.r - r).abs() < 1.0e-9 * R0 && (p.z - z).abs() < 1.0e-9 * R0,
                    "surface {j} at ({theta}, {zeta}): ({}, {}) vs ({r}, {z})",
                    p.r,
                    p.z
                );
            }
        }
        for zeta in [0.0, 0.3, 1.0] {
            let area = cut_area(|theta| {
                let p = fitted.point(1.0, theta, zeta);
                (p.r, p.z)
            });
            assert_eq!(area.signum(), z_sign, "cut at zeta = {zeta}");
            assert_eq!(
                fitted.point(0.5, 1.0, zeta).jacobian().signum(),
                z_sign,
                "volume element at zeta = {zeta}"
            );
        }
    }
}

/// The rotating ellipse with `a` times `step` outside `rho = 0.5`, a jump no
/// polynomial in rho follows.
fn stepped_ellipse(ns: usize, step: f64) -> Vmec {
    let mut vmec = rotating_ellipse_vmec(ns, 5, 1.0);
    for j in 0..ns {
        if vmec.rho(j) > 0.5 {
            for row in [&mut vmec.rmnc[j], &mut vmec.zmns[j]] {
                row[1] *= step;
            }
        }
    }
    vmec
}

/// How far the fit with one radial polynomial per term misses the surfaces
/// of a stepped ellipse. Only `a(rho)` is not fitted exactly, by `c rho`, and
/// its residual moves every point of a surface by that much.
fn single_polynomial_miss(vmec: &Vmec) -> f64 {
    let rho: Vec<f64> = (0..vmec.rmnc.len()).map(|j| vmec.rho(j)).collect();
    let a: Vec<f64> = vmec.rmnc.iter().map(|row| row[1]).collect();
    let c = rho.iter().zip(&a).map(|(r, a)| r * a).sum::<f64>()
        / rho.iter().map(|r| r * r).sum::<f64>();
    rho.iter()
        .zip(&a)
        .map(|(r, a)| (a - c * r).abs())
        .fold(0.0, f64::max)
}

/// The minor radius as `from_vmec` takes it: the mean distance of the
/// boundary from the magnetic axis.
fn minor_radius(vmec: &Vmec) -> f64 {
    let (n, last) = (64, vmec.rmnc.len() - 1);
    let period = TAU / f64::from(vmec.nfp);
    let mut sum = 0.0;
    for i in 0..n {
        for k in 0..n {
            let (theta, zeta) = (TAU * i as f64 / n as f64, period * k as f64 / n as f64);
            let ((r0, z0), (r1, z1)) = (vmec.point(0, theta, zeta), vmec.point(last, theta, zeta));
            sum += (r1 - r0).hypot(z1 - z0);
        }
    }
    sum / (n * n) as f64
}

#[test]
fn surfaces_not_smooth_in_rho_are_refused() {
    // A 20% jump: the error gives the miss of the best fit, which comes
    // closer than one radial polynomial per term does.
    let vmec = stepped_ellipse(41, 1.2);
    let message = vmec.fit().unwrap_err();
    let prefix = "the best fit misses the VMEC surfaces by up to ";
    assert!(
        message.starts_with(prefix)
            && message.contains("cm allowed, 1% of the minor radius")
            && message.contains("not smooth in rho"),
        "{message}"
    );
    let distance: f64 = message[prefix.len()..]
        .split(' ')
        .next()
        .and_then(|d| d.parse().ok())
        .unwrap();
    // The message gives four decimals.
    assert!(
        distance > 0.01 * minor_radius(&vmec) && distance + 1.0e-4 < single_polynomial_miss(&vmec),
        "{message}"
    );
}

#[test]
fn the_fewest_polynomials_within_the_tolerance_are_kept() {
    // A 2.4% jump: one radial polynomial per term misses by 1.67 cm, more
    // than the 1.58 cm allowed but within twice the 0.87 cm of the best fit,
    // three. Two miss by 1.19 cm, the fewest within both.
    let vmec = stepped_ellipse(21, 1.024);
    let allowed = 0.01 * minor_radius(&vmec);
    let single = single_polynomial_miss(&vmec);
    let fitted = vmec.fit().unwrap();
    let error = fitted.fit_error().unwrap();
    assert!(
        error <= allowed && allowed < single,
        "fit error {error}, allowed {allowed}, one polynomial {single}"
    );
    // The miss is the same all round a surface, so fit_error bounds it at
    // any angle.
    let mut rng = StdRng::seed_from_u64(11);
    for j in 0..21 {
        let (theta, zeta) = (TAU * rng.random::<f64>(), TAU * rng.random::<f64>());
        let p = fitted.point(vmec.rho(j), theta, zeta);
        let (r, z) = vmec.point(j, theta, zeta);
        assert!(
            (p.r - r).hypot(p.z - z) <= error * (1.0 + 1.0e-9),
            "surface {j}"
        );
    }
}

#[test]
fn a_noisy_fit_reports_its_error() {
    // Rounding-sized noise of 1e-6 cm on every coefficient: the fit is
    // accepted and fit_error says how far off it is.
    let mut vmec = rotating_ellipse_vmec(51, 5, 1.0);
    let mut rng = StdRng::seed_from_u64(3);
    for row in vmec.rmnc.iter_mut().chain(&mut vmec.zmns) {
        for value in row.iter_mut() {
            *value += 1.0e-6 * (2.0 * rng.random::<f64>() - 1.0);
        }
    }
    let fitted = vmec.fit().unwrap();
    let error = fitted.fit_error().unwrap();
    assert!(error > 1.0e-7 && error < 1.0e-5, "{error}");
    assert_same_shape_within(&fitted, &rotating_ellipse(5), 1.0e-4);
}

fn assert_same_shape_within(fitted: &StellaratorShape, original: &StellaratorShape, cm: f64) {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..200 {
        let (rho, theta, zeta) = (
            rng.random::<f64>(),
            TAU * rng.random::<f64>(),
            TAU * rng.random::<f64>(),
        );
        let (p, q) = (
            fitted.point(rho, theta, zeta),
            original.point(rho, theta, zeta),
        );
        assert!(
            (p.r - q.r).hypot(p.z - q.z) < cm,
            "at ({rho}, {theta}, {zeta})"
        );
    }
}

#[test]
fn bad_coefficients_are_refused() {
    let ellipse = || rotating_ellipse_vmec(11, 5, 1.0);
    let error = |change: &dyn Fn(&mut Vmec)| {
        let mut vmec = ellipse();
        change(&mut vmec);
        vmec.fit().unwrap_err()
    };
    let cases: Vec<(String, &str)> = vec![
        (
            error(&|v| v.nfp = 0),
            "num_field_periods must be at least 1",
        ),
        (
            error(&|v| {
                v.xn.pop();
            }),
            "same length",
        ),
        (error(&|v| v.xm[1] = 1.5), "whole numbers"),
        (error(&|v| v.xm[1] = -1.0), "whole numbers"),
        (error(&|v| v.xn[2] = 3.0), "multiples of num_field_periods"),
        (error(&|v| v.xn[2] = 65.0 * 5.0), "up to 64 times"),
        (error(&|v| v.xn[2] = 0.0), "more than once"),
        (
            error(&|v| v.rmnc.truncate(1)),
            "at least two surfaces, the magnetic axis and the boundary",
        ),
        (error(&|v| v.zmns.truncate(5)), "surfaces and rmnc 11"),
        (error(&|v| v.rmnc[3].push(0.0)), "one per mode"),
        (
            error(&|v| v.rmns = Some(vec![vec![0.0; 3]; 10])),
            "rmns has 10 surfaces",
        ),
        (error(&|v| v.zmns[4][1] = f64::NAN), "zmns must be a finite"),
        (
            error(&|v| v.rmnc.iter_mut().for_each(|row| row.fill(0.0))),
            "every R coefficient is zero",
        ),
        // Every surface the boundary: no axis.
        (
            error(&|v| {
                let (r, z) = (v.rmnc[10].clone(), v.zmns[10].clone());
                v.rmnc.fill(r);
                v.zmns.fill(z);
            }),
            "enclose no volume",
        ),
        // Shapes the fit reproduces exactly: one reaching R <= 0, and one not
        // nested, R - R0 + i Z = a rho e^(i t) + b rho^2 e^(-2 i t) with
        // b = 120 cm, whose volume element changes sign at
        // rho = a / (2 b) = 0.625.
        (
            error(&|v| {
                for row in &mut v.rmnc {
                    row[1] *= 4.0;
                }
            }),
            "cm <= 0 at rho",
        ),
        (
            error(&|v| {
                (v.xm[2], v.xn[2]) = (2.0, 0.0);
                for (j, (r, z)) in v.rmnc.iter_mut().zip(&mut v.zmns).enumerate() {
                    let s = j as f64 / 10.0;
                    (r[2], z[2]) = (120.0 * s, -120.0 * s);
                }
            }),
            "self-intersect or are not nested",
        ),
        // Too large to sum, or to fit.
        (error(&|v| v.rmnc[10][2] = 1.0e308), "overflows"),
        (
            error(&|v| v.rmnc.iter_mut().for_each(|row| row[0] = 1.0e308)),
            "overflows",
        ),
    ];
    for (message, expected) in cases {
        assert!(
            message.contains(expected),
            "{message:?} should mention {expected:?}"
        );
    }
}

#[test]
fn model_file_keeps_the_fit_error() {
    let fitted = rotating_ellipse_vmec(21, 5, 1.0).fit().unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    assert!(json.contains("fit_error"));
    let read: StellaratorShape = serde_json::from_str(&json).unwrap();
    assert_eq!(read, fitted);
    // A shape given as Fourier-Zernike coefficients writes no fit error.
    let json = serde_json::to_string(&rotating_ellipse(5)).unwrap();
    assert!(!json.contains("fit_error"));
    let negative = json.replace('}', ",\"fit_error\":-1.0}");
    let message = serde_json::from_str::<StellaratorShape>(&negative).unwrap_err();
    assert!(
        message.to_string().contains("must not be negative"),
        "{message}"
    );
}

#[test]
fn fitted_shape_drives_the_source() {
    let shape = rotating_ellipse_vmec(21, 5, 1.0).fit().unwrap();
    let plasma = StellaratorPlasma {
        shape: Arc::new(shape),
        profiles: PlasmaProfiles {
            mode: ConfinementMode::H,
            ion_density_centre: 1.09e20,
            ion_density_peaking_factor: 1.0,
            ion_density_pedestal: 1.09e20,
            ion_density_separatrix: 3.0e19,
            ion_temperature_centre: 45.9e3,
            ion_temperature_peaking_factor: 8.06,
            ion_temperature_beta: 6.0,
            ion_temperature_pedestal: 6.09e3,
            ion_temperature_separatrix: 0.1e3,
            pedestal: 0.8,
        },
        fuel: vec![(FuelIon::Deuterium, 0.5), (FuelIon::Tritium, 0.5)],
    };
    let sources = plasma.sources().unwrap();
    let mut rng = StdRng::seed_from_u64(9);
    for source in sources.iter().step_by(7) {
        for _ in 0..100 {
            let [x, y, z] = source.space.sample(&mut rng);
            // In the cut at phi the ellipse has semi-axes a + e and a - e,
            // its major axis turned by N phi / 2.
            let phi = y.atan2(x);
            let (s, c) = (2.5 * phi).sin_cos();
            let (dr, dz) = (x.hypot(y) - R0, z);
            let (u, v) = (c * dr + s * dz, -s * dr + c * dz);
            let radius = (u / (A + E)).hypot(v / (A - E));
            assert!(radius <= 1.0 + 1.0e-9, "{radius}");
        }
    }
}
