//! Tests for the parametric tokamak plasma source.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use rand::rngs::StdRng;
use rand::SeedableRng;
use yamc_source::distribution::energy::FusionReactants;
use yamc_source::source::{SourceEnergyDistribution, SourceSpatialDistribution};
use yamc_source::tokamak::{
    neutron_source_density, reactivity, ConfinementMode, FuelIon, TokamakPlasma, TokamakShape,
};

/// The shape of a EU-DEMO-like plasma, the parameter set used in the
/// `openmc-plasma-source` examples, so the two implementations can be
/// compared directly.
fn demo_shape() -> TokamakShape {
    TokamakShape::new(906.0, 292.258, 1.557, 0.270, 0.44789).unwrap()
}

fn demo_plasma() -> TokamakPlasma {
    TokamakPlasma {
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
        pedestal_radius: 0.8 * 292.258,
        // A coarse mesh keeps the test fast; the physics is resolution
        // independent to within the binning.
        mesh_resolution: (20, 20),
        grid_density: 100,
        ..TokamakPlasma::new(demo_shape())
    }
}

fn relative_error(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference
}

#[test]
fn reactivity_matches_bosch_hale_table() {
    // Bosch & Hale, Nucl. Fusion 32 (1992) 611, Table VIII, converted from
    // cm^3/s to m^3/s. The fit is quoted to better than 0.5%, so 1% here
    // catches a transcribed coefficient without tracking round-off.
    for (temperature_kev, reference) in [(2.0, 2.977e-25), (10.0, 1.136e-22), (20.0, 4.330e-22)] {
        let value = reactivity(temperature_kev * 1.0e3, FusionReactants::DT);
        assert!(
            relative_error(value, reference) < 0.01,
            "D-T reactivity at {temperature_kev} keV was {value}, expected {reference}"
        );
    }
    for (temperature_kev, reference) in [(2.0, 3.110e-27), (10.0, 6.023e-25), (20.0, 2.603e-24)] {
        let value = reactivity(temperature_kev * 1.0e3, FusionReactants::DD);
        assert!(
            relative_error(value, reference) < 0.01,
            "D-D reactivity at {temperature_kev} keV was {value}, expected {reference}"
        );
    }
}

#[test]
fn reactivity_is_zero_at_zero_temperature() {
    assert_eq!(reactivity(0.0, FusionReactants::DT), 0.0);
    assert_eq!(reactivity(-1.0, FusionReactants::DT), 0.0);
}

#[test]
fn neutron_source_density_scales_with_pair_density() {
    let single = neutron_source_density(1.0e40, 20.0e3, FusionReactants::DT);
    let double = neutron_source_density(2.0e40, 20.0e3, FusionReactants::DT);
    assert!((double - 2.0 * single).abs() < 1.0e-9 * single);
}

#[test]
fn profiles_hit_their_named_values() {
    let plasma = demo_plasma();

    // On the magnetic axis the H-mode profiles return the centre values.
    assert!(relative_error(plasma.ion_density(0.0).unwrap(), 1.09e20) < 1.0e-12);
    assert!(relative_error(plasma.ion_temperature(0.0).unwrap(), 45.9e3) < 1.0e-12);

    // At the pedestal the two branches meet at the pedestal values.
    let pedestal = plasma.pedestal_radius;
    assert!(relative_error(plasma.ion_density(pedestal).unwrap(), 1.09e20) < 1.0e-9);
    assert!(relative_error(plasma.ion_temperature(pedestal).unwrap(), 6.09e3) < 1.0e-9);

    // And at the separatrix (the plasma edge) the separatrix values.
    let edge = plasma.shape.minor_radius();
    assert!(relative_error(plasma.ion_density(edge).unwrap(), 3.0e19) < 1.0e-9);
    assert!(relative_error(plasma.ion_temperature(edge).unwrap(), 0.1e3) < 1.0e-9);
}

#[test]
fn l_mode_profiles_are_parabolic() {
    let plasma = TokamakPlasma {
        mode: ConfinementMode::L,
        ion_density_peaking_factor: 1.0,
        ion_temperature_peaking_factor: 1.0,
        ..demo_plasma()
    };
    let half = 0.5 * plasma.shape.minor_radius();
    // (1 - (r/a)^2) with r = a/2 is 3/4.
    assert!(relative_error(plasma.ion_density(half).unwrap(), 0.75 * 1.09e20) < 1.0e-9);
    assert!(relative_error(plasma.ion_temperature(half).unwrap(), 0.75 * 45.9e3) < 1.0e-9);
    // L mode has no pedestal: both profiles reach zero at the edge.
    assert!(
        plasma
            .ion_density(plasma.shape.minor_radius())
            .unwrap()
            .abs()
            < 1.0e-6
    );
}

#[test]
fn profiles_reject_positions_outside_the_plasma() {
    let plasma = demo_plasma();
    let error = plasma
        .ion_density(plasma.shape.minor_radius() * 1.1)
        .unwrap_err();
    assert!(error.contains("minor radius position"), "{error}");
    assert!(plasma.ion_temperature(-1.0).is_err());
}

#[test]
fn shape_places_the_flux_surface_extremes() {
    let shape = demo_shape();
    let (r0, a) = (shape.major_radius(), shape.minor_radius());
    // No Shafranov shift on the boundary, so the outboard and inboard
    // midplane points sit a minor radius either side of R0.
    let (r_outboard, z_outboard) = shape.point(1.0, 0.0);
    assert!(relative_error(r_outboard, r0 + a) < 1.0e-12);
    assert!(z_outboard.abs() < 1.0e-9);

    let (r_inboard, _) = shape.point(1.0, PI);
    assert!(relative_error(r_inboard, r0 - a) < 1.0e-12);

    // The top of the plasma is elongation * a above the midplane, pulled in
    // to R0 - a sin(triangularity).
    let (r_top, z_top) = shape.point(1.0, FRAC_PI_2);
    assert!(relative_error(z_top, shape.elongation() * a) < 1.0e-12);
    assert!(relative_error(r_top, r0 - a * shape.triangularity().sin()) < 1.0e-12);

    // The Shafranov shift is at its largest on the magnetic axis.
    let (r_axis, z_axis) = shape.point(0.0, 1.0);
    assert_eq!((r_axis, z_axis), (r0 + shape.shafranov_factor(), 0.0));
}

#[test]
fn shape_matches_the_fausser_parameterisation() {
    let shape = demo_shape();
    for rho in [0.0, 0.25, 0.5, 0.8, 1.0] {
        for theta in [0.0, 0.7, 2.0, PI, 4.1, 5.9] {
            let a = rho * shape.minor_radius();
            let r = shape.major_radius()
                + a * (theta + shape.triangularity() * theta.sin()).cos()
                + shape.shafranov_factor() * (1.0 - rho * rho);
            let z = shape.elongation() * a * theta.sin();
            let (r_shape, z_shape) = shape.point(rho, theta);
            assert!(
                relative_error(r_shape, r) < 1.0e-12,
                "R at ({rho}, {theta})"
            );
            assert!(
                (z_shape - z).abs() < 1.0e-12 * a.max(1.0),
                "Z at ({rho}, {theta})"
            );
        }
    }
}

/// Bessel function `J_n(x)` by its power series, converged for `|x| <= 2`.
fn bessel_j(n: i32, x: f64) -> f64 {
    let factorial = |k: i32| (1..=k).map(f64::from).product::<f64>();
    (0..30)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * (0.5 * x).powi(2 * k + n) / (factorial(k) * factorial(k + n))
        })
        .sum()
}

#[test]
fn shape_volume_is_analytic() {
    // A circular cross-section is a torus, 2 pi^2 R0 a^2, whatever the
    // Shafranov factor: the shift is zero on the boundary.
    for shafranov_factor in [0.0, 40.0, -40.0] {
        let volume = TokamakShape::new(600.0, 150.0, 1.0, 0.0, shafranov_factor)
            .unwrap()
            .volume();
        let expected = 2.0 * PI * PI * 600.0 * 150.0 * 150.0;
        assert!(
            relative_error(volume, expected) < 1.0e-12,
            "{volume} vs {expected}"
        );
    }
    // Elongation stretches it in Z alone.
    let volume = TokamakShape::new(600.0, 150.0, 1.8, 0.0, 0.0)
        .unwrap()
        .volume();
    let expected = 2.0 * PI * PI * 600.0 * 1.8 * 150.0 * 150.0;
    assert!(
        relative_error(volume, expected) < 1.0e-12,
        "{volume} vs {expected}"
    );

    // With triangularity d, expanding R^2 on the boundary with the
    // Jacobi-Anger expansion gives
    // pi^2 k a^2 [2 R0 (J0(d) + J2(d)) - a/2 (J1(2d) + J3(2d))].
    let (r0, a, k) = (906.0, 292.258, 1.557);
    for d in [0.27, -0.6, 1.0, -1.0] {
        let terms = 2.0 * r0 * (bessel_j(0, d) + bessel_j(2, d))
            - 0.5 * a * (bessel_j(1, 2.0 * d) + bessel_j(3, 2.0 * d));
        let expected = PI * PI * k * a * a * terms;
        let volume = TokamakShape::new(r0, a, k, d, 0.44789).unwrap().volume();
        assert!(
            relative_error(volume, expected) < 1.0e-12,
            "{volume} vs {expected}"
        );
    }
}

#[test]
fn shape_volume_matches_a_polygon_through_the_boundary() {
    // Pappus: 2 pi times the first moment of the cross-section about the axis,
    // (1/6) sum (R_i + R_j)(R_i Z_j - R_j Z_i) for a polygon. An inscribed
    // polygon of n sides is short by O(1/n^2).
    let n = 200_000;
    for triangularity in [0.27, -0.6, 1.0] {
        let shape = TokamakShape::new(906.0, 292.258, 1.557, triangularity, 0.44789).unwrap();
        let points: Vec<(f64, f64)> = (0..n)
            .map(|i| shape.point(1.0, TAU * i as f64 / n as f64))
            .collect();
        let moment: f64 = (0..n)
            .map(|i| {
                let ((r_i, z_i), (r_j, z_j)) = (points[i], points[(i + 1) % n]);
                (r_i + r_j) * (r_i * z_j - r_j * z_i)
            })
            .sum::<f64>()
            / 6.0;
        let volume = shape.volume();
        assert!(
            relative_error(volume, TAU * moment) < 1.0e-8,
            "{volume} vs {}",
            TAU * moment
        );
    }
}

#[test]
fn bad_shapes_are_refused() {
    let cases = [
        (
            (906.0, 1000.0, 1.557, 0.27, 0.44789),
            "minor_radius must be less than major_radius (got 1000 >= 906)",
        ),
        (
            (-1.0, 292.258, 1.557, 0.27, 0.44789),
            "major_radius must be positive (got -1)",
        ),
        (
            (906.0, 0.0, 1.557, 0.27, 0.44789),
            "minor_radius must be positive (got 0)",
        ),
        (
            (906.0, 292.258, 0.0, 0.27, 0.44789),
            "elongation must be positive (got 0)",
        ),
        (
            (906.0, 292.258, 1.557, 1.5, 0.44789),
            "triangularity must be between -1 and 1 (got 1.5)",
        ),
        (
            (906.0, 292.258, 1.557, 0.27, -200.0),
            "abs(shafranov_factor) must be less than half the minor radius (got 200 >= 146.129)",
        ),
    ];
    for ((r0, a, k, d, s), expected) in cases {
        assert_eq!(TokamakShape::new(r0, a, k, d, s).unwrap_err(), expected);
    }
    // NaN and infinity are refused first, naming the parameter.
    for (index, name) in [
        "major_radius",
        "minor_radius",
        "elongation",
        "triangularity",
        "shafranov_factor",
    ]
    .into_iter()
    .enumerate()
    {
        for value in [f64::NAN, f64::INFINITY] {
            let mut parameters = [906.0, 292.258, 1.557, 0.27, 0.44789];
            parameters[index] = value;
            let [r0, a, k, d, s] = parameters;
            assert_eq!(
                TokamakShape::new(r0, a, k, d, s).unwrap_err(),
                format!("{name} must be a finite number (got {value})")
            );
        }
    }
}

#[test]
fn a_shape_round_trips_through_serde_and_is_checked_when_read() {
    let shape = demo_shape();
    let json = serde_json::to_string(&shape).unwrap();
    assert_eq!(serde_json::from_str::<TokamakShape>(&json).unwrap(), shape);

    let bad = json.replace("\"minor_radius\":292.258", "\"minor_radius\":1000.0");
    assert_ne!(bad, json);
    let error = serde_json::from_str::<TokamakShape>(&bad)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("minor_radius must be less than major_radius"),
        "{error}"
    );
}

#[test]
fn sources_are_normalised_and_shaped_like_the_plasma() {
    let plasma = demo_plasma();
    let sources = plasma.sources().unwrap();
    assert!(!sources.is_empty());

    let total: f64 = sources.iter().map(|source| source.strength).sum();
    assert!((total - 1.0).abs() < 1.0e-12, "strengths summed to {total}");

    let mut rng = StdRng::seed_from_u64(42);
    let shape = plasma.shape;
    let r_max = shape.major_radius() + shape.minor_radius() + shape.shafranov_factor().abs();
    let r_min = shape.major_radius() - shape.minor_radius() - shape.shafranov_factor().abs();
    let z_max = shape.elongation() * shape.minor_radius();
    for source in &sources {
        for _ in 0..4 {
            let [x, y, z] = source.space.sample(&mut rng);
            let r = (x * x + y * y).sqrt();
            assert!(r >= r_min - 1.0e-9 && r <= r_max + 1.0e-9, "R = {r}");
            assert!(z.abs() <= z_max + 1.0e-9, "Z = {z}");
        }
    }
}

#[test]
fn each_voxel_carries_a_ring_and_a_ballabio_spectrum() {
    let sources = demo_plasma().sources().unwrap();
    let mut dt_seen = false;
    let mut dd_seen = false;
    for source in &sources {
        assert!(matches!(
            source.space,
            SourceSpatialDistribution::CylindricalRing(_)
        ));
        let SourceEnergyDistribution::Normal(energy) = &source.energy else {
            panic!(
                "expected a Ballabio (Normal) spectrum, got {:?}",
                source.energy
            );
        };
        // The two reactions are far enough apart in energy to tell apart.
        if energy.mean_val() > 13.0e6 {
            dt_seen = true;
        } else if energy.mean_val() > 2.0e6 && energy.mean_val() < 3.0e6 {
            dd_seen = true;
        } else {
            panic!("unexpected mean neutron energy {}", energy.mean_val());
        }
    }
    assert!(dt_seen, "no D-T sources built for 50:50 D-T fuel");
    assert!(dd_seen, "no D-D sources built for 50:50 D-T fuel");
}

#[test]
fn dt_carries_almost_all_of_the_yield() {
    let sources = demo_plasma().sources().unwrap();
    let dd_strength: f64 = sources
        .iter()
        .filter(|source| match &source.energy {
            SourceEnergyDistribution::Normal(energy) => energy.mean_val() < 5.0e6,
            _ => false,
        })
        .map(|source| source.strength)
        .sum();
    // At tens of keV the D-D reactivity is a few hundred times below D-T, and
    // only half the D-D branches make a neutron.
    assert!(
        dd_strength > 1.0e-4 && dd_strength < 1.0e-2,
        "D-D share of the yield was {dd_strength}"
    );
}

#[test]
fn deuterium_only_fuel_makes_dd_neutrons_alone() {
    let plasma = TokamakPlasma {
        fuel: vec![(FuelIon::Deuterium, 1.0)],
        ..demo_plasma()
    };
    for source in plasma.sources().unwrap() {
        let SourceEnergyDistribution::Normal(energy) = &source.energy else {
            panic!("expected a Normal spectrum");
        };
        assert!(energy.mean_val() < 3.0e6, "{}", energy.mean_val());
    }
}

#[test]
fn tritium_only_fuel_is_rejected() {
    let plasma = TokamakPlasma {
        fuel: vec![(FuelIon::Tritium, 1.0)],
        ..demo_plasma()
    };
    let error = plasma.sources().unwrap_err();
    assert!(error.contains("T-T fusion is not supported"), "{error}");
}

#[test]
fn a_toroidal_sector_only_emits_inside_itself() {
    let plasma = TokamakPlasma {
        start_angle: 0.5,
        rotation_angle: FRAC_PI_2,
        ..demo_plasma()
    };
    let sources = plasma.sources().unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    for source in &sources {
        for _ in 0..4 {
            let [x, y, _] = source.space.sample(&mut rng);
            let phi = y.atan2(x);
            assert!(
                (0.5 - 1.0e-9..=0.5 + FRAC_PI_2 + 1.0e-9).contains(&phi),
                "phi = {phi} outside the requested sector"
            );
        }
    }
}

#[test]
fn a_negative_rotation_angle_sweeps_the_other_way() {
    let plasma = TokamakPlasma {
        start_angle: 0.0,
        rotation_angle: -FRAC_PI_2,
        ..demo_plasma()
    };
    let sources = plasma.sources().unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    for source in sources.iter().take(50) {
        let [x, y, _] = source.space.sample(&mut rng);
        let phi = y.atan2(x);
        assert!(
            (-FRAC_PI_2 - 1.0e-9..=1.0e-9).contains(&phi),
            "phi = {phi} outside the requested sector"
        );
    }
}

#[test]
fn a_sector_and_a_full_torus_have_the_same_poloidal_emission() {
    // Only the toroidal extent differs, so the per-voxel strengths must be
    // identical: a sector is the same plasma seen through a wedge.
    let full = demo_plasma().sources().unwrap();
    let sector = TokamakPlasma {
        rotation_angle: std::f64::consts::FRAC_PI_4,
        ..demo_plasma()
    }
    .sources()
    .unwrap();
    assert_eq!(full.len(), sector.len());
    for (a, b) in full.iter().zip(sector.iter()) {
        assert!((a.strength - b.strength).abs() < 1.0e-12);
    }
}

#[test]
fn emission_is_concentrated_where_the_plasma_is_hottest() {
    // The peaked temperature profile puts most of the yield near the magnetic
    // axis, not spread evenly over the poloidal cross-section.
    let plasma = demo_plasma();
    let shape = plasma.shape;
    let sources = plasma.sources().unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let mut near_axis = 0.0;
    for source in &sources {
        let [x, y, z] = source.space.sample(&mut rng);
        let r = (x * x + y * y).sqrt();
        let radius = ((r - shape.major_radius()).powi(2) + (z / shape.elongation()).powi(2)).sqrt();
        if radius < 0.5 * shape.minor_radius() {
            near_axis += source.strength;
        }
    }
    // Emission spread evenly over the cross-section would put a quarter of
    // the yield inside the half-radius contour (it is a quarter of the area).
    assert!(
        near_axis > 0.6,
        "only {near_axis} of the yield came from the inner half of the plasma"
    );
}

#[test]
fn resolution_does_not_move_the_emission() {
    // Doubling the mesh must not change where the neutrons come from: compare
    // the mean major radius of emission between two resolutions.
    let mean_radius = |resolution| {
        let plasma = TokamakPlasma {
            mesh_resolution: resolution,
            grid_density: 200,
            ..demo_plasma()
        };
        let sources = plasma.sources().unwrap();
        let mut rng = StdRng::seed_from_u64(19);
        sources
            .iter()
            .map(|source| {
                let [x, y, _] = source.space.sample(&mut rng);
                source.strength * (x * x + y * y).sqrt()
            })
            .sum::<f64>()
    };
    let coarse = mean_radius((20, 20));
    let fine = mean_radius((40, 40));
    assert!(
        relative_error(fine, coarse) < 0.01,
        "mean emission radius moved from {coarse} to {fine}"
    );
}

#[test]
fn invalid_parameters_are_rejected() {
    let cases: Vec<(TokamakPlasma, &str)> = vec![
        (
            TokamakPlasma {
                pedestal_radius: 500.0,
                ..demo_plasma()
            },
            "pedestal_radius must be less than minor_radius (got 500 >= 292.258)",
        ),
        (
            TokamakPlasma {
                rotation_angle: 0.0,
                ..demo_plasma()
            },
            "rotation_angle must be a non-zero value",
        ),
        (
            TokamakPlasma {
                fuel: vec![(FuelIon::Deuterium, 0.3), (FuelIon::Tritium, 0.3)],
                ..demo_plasma()
            },
            "fuel fractions must sum to 1",
        ),
        (
            TokamakPlasma {
                ion_density_centre: 0.0,
                ..demo_plasma()
            },
            "ion_density_centre must be positive",
        ),
    ];
    for (plasma, expected) in cases {
        let error = plasma.sources().unwrap_err();
        assert!(
            error.contains(expected),
            "expected {expected:?}, got {error}"
        );
    }
}

#[test]
fn a_cold_plasma_reports_no_neutrons_rather_than_dividing_by_zero() {
    let plasma = TokamakPlasma {
        ion_temperature_centre: 0.0,
        ion_temperature_pedestal: 0.0,
        ion_temperature_separatrix: 0.0,
        ..demo_plasma()
    };
    let error = plasma.sources().unwrap_err();
    assert!(
        error.contains("total neutron source density is zero"),
        "{error}"
    );
}

#[test]
fn confinement_modes_and_fuel_species_parse() {
    assert_eq!(ConfinementMode::parse("H").unwrap(), ConfinementMode::H);
    assert_eq!(ConfinementMode::parse("L").unwrap(), ConfinementMode::L);
    assert_eq!(ConfinementMode::parse("A").unwrap(), ConfinementMode::A);
    assert!(ConfinementMode::parse("X")
        .unwrap_err()
        .contains("mode must be"));
    assert_eq!(FuelIon::parse("D").unwrap(), FuelIon::Deuterium);
    assert_eq!(FuelIon::parse("T").unwrap(), FuelIon::Tritium);
    assert!(FuelIon::parse("He3").unwrap_err().contains("fuel species"));
}

/// A NaN passes the range checks written as `value <= 0.0` and the like, and
/// the peaking factors and beta were not checked at all, so every float
/// parameter is checked for being finite first, naming itself. The shape's
/// own are refused when it is built (`bad_shapes_are_refused`).
#[test]
fn a_parameter_that_is_not_a_finite_number_is_refused_naming_it() {
    type Field = fn(&mut TokamakPlasma) -> &mut f64;
    let fields: [(&str, Field); 12] = [
        ("ion_density_centre", |p| &mut p.ion_density_centre),
        ("ion_density_peaking_factor", |p| {
            &mut p.ion_density_peaking_factor
        }),
        ("ion_density_pedestal", |p| &mut p.ion_density_pedestal),
        ("ion_density_separatrix", |p| &mut p.ion_density_separatrix),
        ("ion_temperature_centre", |p| &mut p.ion_temperature_centre),
        ("ion_temperature_peaking_factor", |p| {
            &mut p.ion_temperature_peaking_factor
        }),
        ("ion_temperature_beta", |p| &mut p.ion_temperature_beta),
        ("ion_temperature_pedestal", |p| {
            &mut p.ion_temperature_pedestal
        }),
        ("ion_temperature_separatrix", |p| {
            &mut p.ion_temperature_separatrix
        }),
        ("pedestal_radius", |p| &mut p.pedestal_radius),
        ("start_angle", |p| &mut p.start_angle),
        ("rotation_angle", |p| &mut p.rotation_angle),
    ];
    for (name, field) in fields {
        for value in [f64::NAN, f64::INFINITY] {
            let mut plasma = demo_plasma();
            *field(&mut plasma) = value;
            assert_eq!(
                plasma.sources().unwrap_err(),
                format!("{name} must be a finite number (got {value})")
            );
        }
    }

    let mut plasma = demo_plasma();
    plasma.fuel = vec![(FuelIon::Deuterium, f64::NAN), (FuelIon::Tritium, 0.5)];
    let err = plasma.sources().unwrap_err();
    assert!(
        err.contains("fuel fractions must be greater than 0 and at most 1") && err.contains("NaN"),
        "{err}"
    );
}
