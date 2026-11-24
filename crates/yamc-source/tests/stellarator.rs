//! Tests for the Fourier-Zernike stellarator shape and plasma source.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use yamc_source::distribution::energy::FusionReactants;
use yamc_source::source::{Source, SourceEnergyDistribution, SourceSpatialDistribution};
use yamc_source::stellarator::{StellaratorPlasma, StellaratorShape};
use yamc_source::tokamak::{neutron_source_density, ConfinementMode, FuelIon, PlasmaProfiles};

const R0: f64 = 600.0;
const A: f64 = 150.0;

/// Circular torus `R = R0 + a rho cos(theta)`, `Z = a rho sin(theta)`.
fn torus(z_sign: f64) -> StellaratorShape {
    StellaratorShape::new(
        1,
        vec![[0, 0, 0], [1, 1, 0]],
        vec![R0, A],
        vec![[1, -1, 0]],
        vec![z_sign * A],
    )
    .unwrap()
}

/// Rotating ellipse, as in OpenMC's stellarator source tests:
/// `R = R0 + a rho cos(t) + e rho cos(t - N z)`,
/// `Z = a rho sin(t) - e rho sin(t - N z)`, written in the product basis.
fn rotating_ellipse(e: f64, nfp: u32, z_sign: f64) -> StellaratorShape {
    StellaratorShape::new(
        nfp,
        vec![[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, -1, -1]],
        vec![R0, A, e, e],
        vec![[1, -1, 0], [1, -1, 1], [1, 1, -1]],
        vec![z_sign * A, -z_sign * e, z_sign * e],
    )
    .unwrap()
}

/// Flat density and temperature, so the emission density is uniform.
fn flat_profiles() -> PlasmaProfiles {
    PlasmaProfiles {
        mode: ConfinementMode::L,
        ion_density_centre: 1.0e20,
        ion_density_peaking_factor: 0.0,
        ion_density_pedestal: 1.0e20,
        ion_density_separatrix: 1.0e20,
        ion_temperature_centre: 20.0e3,
        ion_temperature_peaking_factor: 0.0,
        ion_temperature_beta: 0.0,
        ion_temperature_pedestal: 20.0e3,
        ion_temperature_separatrix: 20.0e3,
        pedestal: 0.8,
    }
}

fn peaked_profiles() -> PlasmaProfiles {
    PlasmaProfiles {
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
    }
}

fn plasma(shape: StellaratorShape, profiles: PlasmaProfiles) -> StellaratorPlasma {
    StellaratorPlasma {
        shape: Arc::new(shape),
        profiles,
        fuel: vec![(FuelIon::Deuterium, 1.0)],
    }
}

/// Sample `count` positions, picking sources by strength as transport does.
fn sample(sources: &[Source], count: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let cumulative: Vec<f64> = sources
        .iter()
        .scan(0.0, |sum, s| {
            *sum += s.strength;
            Some(*sum)
        })
        .collect();
    (0..count)
        .map(|_| {
            let u = rng.random::<f64>() * cumulative.last().unwrap();
            let index = cumulative.partition_point(|&c| c < u);
            sources[index].space.sample(&mut rng)
        })
        .collect()
}

/// Assert the sample mean of `values` is within 4 standard errors of `expected`.
fn assert_mean(values: &[f64], expected: f64, what: &str) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let error = (variance / n).sqrt();
    assert!(
        (mean - expected).abs() < 4.0 * error + 1.0e-9 * expected.abs(),
        "{what}: mean {mean} vs expected {expected} (standard error {error})"
    );
}

/// Unnormalised Zernike radial polynomial by its explicit sum.
fn zernike_radial(l: i32, m: i32, rho: f64) -> f64 {
    let factorial = |n: i32| (1..=n).map(f64::from).product::<f64>();
    let m = m.abs();
    (0..=(l - m) / 2)
        .map(|s| {
            let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
            sign * factorial(l - s)
                / (factorial(s) * factorial((l + m) / 2 - s) * factorial((l - m) / 2 - s))
                * rho.powi(l - 2 * s)
        })
        .sum()
}

/// DESC's basis function `Z_l^m(rho, theta) F_n(zeta)`.
fn basis(mode: [i32; 3], nfp: u32, rho: f64, theta: f64, zeta: f64) -> f64 {
    let [l, m, n] = mode;
    let poloidal = if m >= 0 {
        (m as f64 * theta).cos()
    } else {
        (-m as f64 * theta).sin()
    };
    let toroidal_angle = n.abs() as f64 * nfp as f64 * zeta;
    let toroidal = if n >= 0 {
        toroidal_angle.cos()
    } else {
        toroidal_angle.sin()
    };
    zernike_radial(l, m, rho) * poloidal * toroidal
}

#[test]
fn evaluation_matches_explicit_desc_basis() {
    // A torus plus small high-order terms of every sign combination, so the
    // recurrence, the sin/cos choices and the mode sorting are all exercised
    // while the surfaces stay nested.
    let nfp = 3;
    let mut r_modes = vec![[0, 0, 0], [1, 1, 0]];
    let mut r_lmn = vec![R0, A];
    let mut z_modes = vec![[1, -1, 0]];
    let mut z_lmn = vec![A];
    for (i, mode) in [
        [2, 0, 1],
        [3, 1, -1],
        [4, -2, 2],
        [5, 3, 0],
        [6, -4, -2],
        [7, 1, 1],
        [8, 0, -3],
        [10, 2, 1],
        [12, -6, 0],
    ]
    .into_iter()
    .enumerate()
    {
        let c = 0.05 + 0.01 * i as f64;
        r_modes.push(mode);
        r_lmn.push(c);
        z_modes.push([mode[0], -mode[1], -mode[2]]);
        z_lmn.push(-c);
    }
    let shape = StellaratorShape::new(
        nfp,
        r_modes.clone(),
        r_lmn.clone(),
        z_modes.clone(),
        z_lmn.clone(),
    )
    .unwrap();

    let reference = |modes: &[[i32; 3]], lmn: &[f64], rho, theta, zeta| -> f64 {
        modes
            .iter()
            .zip(lmn)
            .map(|(mode, c)| c * basis(*mode, nfp, rho, theta, zeta))
            .sum()
    };
    let h = 1.0e-6;
    for (rho, theta, zeta) in [
        (0.3, 0.4, 0.1),
        (0.77, 2.5, 1.9),
        (1.0, 5.0, 0.6),
        (0.05, 1.0, 3.0),
    ] {
        let p = shape.point(rho, theta, zeta);
        let r = reference(&r_modes, &r_lmn, rho, theta, zeta);
        let z = reference(&z_modes, &z_lmn, rho, theta, zeta);
        assert!((p.r - r).abs() < 1.0e-9 * R0, "R at {rho}: {} vs {r}", p.r);
        assert!((p.z - z).abs() < 1.0e-9 * R0, "Z at {rho}: {} vs {z}", p.z);
        let d = |f: &dyn Fn(f64, f64) -> f64| (f(rho + h, theta) - f(rho - h, theta)) / (2.0 * h);
        let dt = |f: &dyn Fn(f64, f64) -> f64| (f(rho, theta + h) - f(rho, theta - h)) / (2.0 * h);
        let rf = |a: f64, b: f64| reference(&r_modes, &r_lmn, a, b, zeta);
        let zf = |a: f64, b: f64| reference(&z_modes, &z_lmn, a, b, zeta);
        for (value, expected, name) in [
            (p.r_rho, d(&rf), "R_rho"),
            (p.z_rho, d(&zf), "Z_rho"),
            (p.r_theta, dt(&rf), "R_theta"),
            (p.z_theta, dt(&zf), "Z_theta"),
        ] {
            assert!(
                (value - expected).abs() < 1.0e-5 * A,
                "{name} at rho {rho}: {value} vs {expected}"
            );
        }
    }
}

#[test]
fn zernike_radial_is_one_on_the_boundary() {
    // R_l^m(1) = 1: a single (l, m) mode on top of the torus shifts R at
    // rho = 1, theta = 0 by exactly its coefficient.
    for (l, m) in [(2, 0), (4, 2), (9, 3), (14, 0), (20, 4)] {
        let shape = StellaratorShape::new(
            1,
            vec![[0, 0, 0], [1, 1, 0], [l, m, 0]],
            vec![R0, A, 0.01],
            vec![[1, -1, 0]],
            vec![A],
        )
        .unwrap();
        let expected = R0 + A + 0.01;
        assert!(
            (shape.point(1.0, 0.0, 0.0).r - expected).abs() < 1.0e-9,
            "l={l} m={m}"
        );
    }
}

#[test]
fn torus_volume_is_analytic() {
    let expected = 2.0 * PI * PI * R0 * A * A;
    for shape in [torus(1.0), torus(-1.0)] {
        let volume = shape.volume();
        assert!(
            (volume - expected).abs() < 1.0e-9 * expected,
            "{volume} vs {expected}"
        );
    }
    // The rotating ellipse has semi-axes a + e and a - e at every angle.
    let e = 50.0;
    let volume = rotating_ellipse(e, 5, 1.0).volume();
    let expected = 2.0 * PI * PI * R0 * (A + e) * (A - e);
    assert!(
        (volume - expected).abs() < 1.0e-9 * expected,
        "{volume} vs {expected}"
    );
}

#[test]
fn uniform_torus_has_analytic_moments() {
    // With uniform emission the density is ~ (R0 + a rho cos t) a^2 rho, so
    // E[rho] = 2/3 and E[R] = R0 + a^2 / (4 R0).
    for z_sign in [1.0, -1.0] {
        let sources = plasma(torus(z_sign), flat_profiles()).sources().unwrap();
        let points = sample(&sources, 40_000, 7);
        let major: Vec<f64> = points.iter().map(|p| p[0].hypot(p[1])).collect();
        let minor: Vec<f64> = points
            .iter()
            .zip(&major)
            .map(|(p, r)| (r - R0).hypot(p[2]) / A)
            .collect();
        assert!(minor.iter().all(|&rho| rho <= 1.0 + 1.0e-12));
        assert_mean(&minor, 2.0 / 3.0, "rho");
        assert_mean(&major, R0 + A * A / (4.0 * R0), "R");
    }
}

#[test]
fn rotating_ellipse_moments_match_quadrature() {
    let (e, nfp) = (50.0, 5);
    let profiles = peaked_profiles();
    for z_sign in [1.0, -1.0] {
        let shape = rotating_ellipse(e, nfp, z_sign);

        // Quadrature of S(rho) R |tau| over one field period; every period
        // contributes the same R and Z distribution.
        let (n_rho, n_theta, n_zeta) = (200, 48, 24);
        let (mut weight, mut sum_r, mut sum_z, mut sum_z2) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n_rho {
            let rho = (i as f64 + 0.5) / n_rho as f64;
            let density = profiles.ion_density(rho);
            let s = neutron_source_density(
                0.5 * density * density,
                profiles.ion_temperature(rho),
                FusionReactants::DD,
            );
            for j in 0..n_theta {
                let theta = TAU * j as f64 / n_theta as f64;
                for k in 0..n_zeta {
                    let zeta = TAU / nfp as f64 * k as f64 / n_zeta as f64;
                    let p = shape.point(rho, theta, zeta);
                    let w = s * p.jacobian().abs();
                    weight += w;
                    sum_r += w * p.r;
                    sum_z += w * p.z;
                    sum_z2 += w * p.z * p.z;
                }
            }
        }

        let sources = plasma(shape, profiles.clone()).sources().unwrap();
        let points = sample(&sources, 40_000, 11);
        let major: Vec<f64> = points.iter().map(|p| p[0].hypot(p[1])).collect();
        let z: Vec<f64> = points.iter().map(|p| p[2]).collect();
        let z2: Vec<f64> = z.iter().map(|v| v * v).collect();
        assert_mean(&major, sum_r / weight, "R");
        assert_mean(&z, sum_z / weight, "Z");
        assert_mean(&z2, sum_z2 / weight, "Z^2");
    }
}

#[test]
fn samples_fill_every_field_period() {
    let sources = plasma(rotating_ellipse(50.0, 5, 1.0), flat_profiles())
        .sources()
        .unwrap();
    let points = sample(&sources, 20_000, 3);
    let mut counts = [0usize; 5];
    for p in &points {
        let phi = p[1].atan2(p[0]).rem_euclid(TAU);
        counts[((phi / (TAU / 5.0)) as usize).min(4)] += 1;
    }
    for count in counts {
        assert!(
            (count as f64 - 4000.0).abs() < 4.0 * 4000.0_f64.sqrt(),
            "{counts:?}"
        );
    }
}

#[test]
fn sources_are_shells_with_normalised_strengths() {
    let mut p = plasma(rotating_ellipse(50.0, 5, 1.0), peaked_profiles());
    p.fuel = vec![(FuelIon::Deuterium, 0.5), (FuelIon::Tritium, 0.5)];
    let sources = p.sources().unwrap();
    assert_eq!(sources.len(), 100, "50 shells for each of D-T and D-D");
    let total: f64 = sources.iter().map(|s| s.strength).sum();
    assert!((total - 1.0).abs() < 1.0e-12);
    for source in &sources {
        assert!(matches!(
            source.space,
            SourceSpatialDistribution::FluxShell(_)
        ));
        assert!(matches!(source.energy, SourceEnergyDistribution::Normal(_)));
    }
    // The core shells are the hottest: the innermost D-T shell has a higher
    // mean energy spread than the outermost.
    let width = |s: &Source| match &s.energy {
        SourceEnergyDistribution::Normal(n) => n.std_dev(),
        _ => unreachable!(),
    };
    assert!(width(&sources[0]) > width(&sources[49]));
}

#[test]
fn model_file_round_trip_samples_identically() {
    let sources = plasma(rotating_ellipse(50.0, 5, 1.0), flat_profiles())
        .sources()
        .unwrap();
    let json = serde_json::to_string(&sources[10]).unwrap();
    let read: Source = serde_json::from_str(&json).unwrap();
    let (mut a, mut b) = (StdRng::seed_from_u64(5), StdRng::seed_from_u64(5));
    for _ in 0..100 {
        assert_eq!(sources[10].space.sample(&mut a), read.space.sample(&mut b));
    }
}

fn error_of(
    nfp: u32,
    r_modes: Vec<[i32; 3]>,
    r_lmn: Vec<f64>,
    z_modes: Vec<[i32; 3]>,
    z_lmn: Vec<f64>,
) -> String {
    StellaratorShape::new(nfp, r_modes, r_lmn, z_modes, z_lmn).unwrap_err()
}

#[test]
fn bad_shapes_are_refused() {
    let r = || (vec![[0, 0, 0], [1, 1, 0]], vec![R0, A]);
    let z = || (vec![[1, -1, 0]], vec![A]);
    let cases: Vec<(String, &str)> = vec![
        (error_of(0, r().0, r().1, z().0, z().1), "num_field_periods"),
        (error_of(1, r().0, vec![R0], z().0, z().1), "same length"),
        (error_of(1, vec![], vec![], z().0, z().1), "non-empty"),
        (
            error_of(1, vec![[0, 0, 0], [1, 3, 0]], r().1, z().0, z().1),
            "l >= |m|",
        ),
        (
            error_of(1, vec![[0, 0, 0], [2, 1, 0]], r().1, z().0, z().1),
            "even",
        ),
        (
            error_of(1, vec![[0, 0, 0], [0, 0, 0]], r().1, z().0, z().1),
            "more than once",
        ),
        (
            error_of(1, vec![[0, 0, 0], [1, 1, 65]], r().1, z().0, z().1),
            "|n| up to",
        ),
        (
            error_of(1, r().0, vec![R0, f64::NAN], z().0, z().1),
            "finite",
        ),
        // Minor radius beyond the major radius reaches R < 0.
        (error_of(1, r().0, vec![100.0, A], z().0, z().1), "R ="),
        // A strong m = 2 term folds the surfaces over.
        (
            error_of(
                1,
                vec![[0, 0, 0], [1, 1, 0], [2, 2, 0]],
                vec![R0, A, 1.5 * A],
                z().0,
                z().1,
            ),
            "changes sign",
        ),
        (
            error_of(1, vec![[0, 0, 0]], vec![R0], vec![[0, 0, 0]], vec![0.0]),
            "zero volume",
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
fn bad_plasmas_are_refused() {
    let mut p = plasma(torus(1.0), peaked_profiles());
    p.profiles.pedestal = 1.0;
    assert!(p.sources().unwrap_err().contains("pedestal_rho"));
    let mut p = plasma(torus(1.0), peaked_profiles());
    p.fuel = vec![(FuelIon::Tritium, 1.0)];
    assert!(p.sources().unwrap_err().contains("deuterium"));
    let mut p = plasma(torus(1.0), peaked_profiles());
    p.profiles.ion_temperature_centre = 0.0;
    p.profiles.ion_temperature_pedestal = 0.0;
    p.profiles.ion_temperature_separatrix = 0.0;
    assert!(p.sources().unwrap_err().contains("zero"));
}

/// The rotating ellipse of the stellarator source example, plus an `l > |m|`
/// mode so the radial sum matters.
fn rotating_ellipse_with_radial_modes() -> StellaratorShape {
    StellaratorShape::new(
        5,
        vec![
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, -1, -1],
            [3, 1, 0],
            [2, 0, 1],
        ],
        vec![600.0, 150.0, 50.0, 50.0, 4.0, -3.0],
        vec![[1, -1, 0], [1, -1, 1], [1, 1, -1], [3, -1, 0]],
        vec![150.0, -50.0, 50.0, 2.5],
    )
    .unwrap()
}

#[test]
fn surface_modes_are_the_flux_surface() {
    let shape = rotating_ellipse_with_radial_modes();
    let nfp = shape.num_field_periods() as f64;
    let series = |modes: &[[i32; 2]], values: &[f64], theta: f64, zeta: f64| {
        modes
            .iter()
            .zip(values)
            .map(|(&[m, n], &c)| {
                let poloidal = if m >= 0 {
                    (m as f64 * theta).cos()
                } else {
                    (-m as f64 * theta).sin()
                };
                let toroidal = if n >= 0 {
                    (n as f64 * nfp * zeta).cos()
                } else {
                    (-n as f64 * nfp * zeta).sin()
                };
                c * poloidal * toroidal
            })
            .sum::<f64>()
    };
    for rho in [0.3, 0.75, 1.0] {
        let (r_modes, r_mn, z_modes, z_mn) = shape.surface_modes(rho);
        for k in 0..40 {
            let (theta, zeta) = (k as f64 * 0.37, k as f64 * 0.21);
            let p = shape.point(rho, theta, zeta);
            let (r, z) = (
                series(&r_modes, &r_mn, theta, zeta),
                series(&z_modes, &z_mn, theta, zeta),
            );
            assert!(
                (r - p.r).abs() < 1e-9 && (z - p.z).abs() < 1e-9,
                "rho {rho}: ({r}, {z}) vs ({}, {})",
                p.r,
                p.z
            );
        }
    }
}
