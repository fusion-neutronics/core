//! Rust unit tests for dose coefficients and EnergyFunctionFilter.

use yamc_nuclide::data::effective_dose::{
    dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle,
};
use yamc_tallies::EnergyFunctionFilter;

// ==================== Dose Coefficients Tests ====================

#[test]
fn test_icrp116_neutron_dose_coefficients_ap() {
    let (energy, coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::AP,
        DoseDataSource::ICRP116,
    );

    // Should have data points (ICRP-116 has 68 neutron energy points)
    assert!(!energy.is_empty());
    assert_eq!(energy.len(), coeffs.len());
    assert_eq!(energy.len(), 68);

    // Energy should be in eV (converted from MeV)
    // First point is 1e-9 MeV = 1e-3 eV
    assert!(energy[0] > 0.0);
    assert!(energy[0] < 1.0); // Less than 1 eV

    // Last point should be in GeV range (converted to eV)
    assert!(energy.last().unwrap() > &1e9); // Greater than 1 GeV in eV

    // Coefficients should be positive (pSv·cm²)
    for &c in &coeffs {
        assert!(c > 0.0, "Dose coefficient should be positive");
    }

    // Energy should be monotonically increasing
    for i in 1..energy.len() {
        assert!(
            energy[i] > energy[i - 1],
            "Energy grid must be monotonically increasing"
        );
    }
}

#[test]
fn test_icrp74_neutron_dose_coefficients() {
    let (energy, coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::AP,
        DoseDataSource::ICRP74,
    );

    assert!(!energy.is_empty());
    assert_eq!(energy.len(), coeffs.len());

    // ICRP-74 has 47 points
    assert_eq!(energy.len(), 47);

    for &c in &coeffs {
        assert!(c > 0.0);
    }
}

#[test]
fn test_all_geometries_icrp116() {
    let geometries = [
        DoseGeometry::AP,
        DoseGeometry::PA,
        DoseGeometry::LLAT,
        DoseGeometry::RLAT,
        DoseGeometry::ROT,
        DoseGeometry::ISO,
    ];

    for geom in geometries {
        let (energy, coeffs) =
            dose_coefficients(DoseParticle::Neutron, geom, DoseDataSource::ICRP116);
        assert!(!energy.is_empty(), "Geometry {:?} should have data", geom);
        assert_eq!(energy.len(), coeffs.len());

        // All geometries should have same energy grid
        let (ref_energy, _) = dose_coefficients(
            DoseParticle::Neutron,
            DoseGeometry::AP,
            DoseDataSource::ICRP116,
        );
        assert_eq!(
            energy.len(),
            ref_energy.len(),
            "All geometries should have same number of points"
        );
    }
}

#[test]
fn test_ap_vs_pa_different_coefficients() {
    let (_, ap_coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::AP,
        DoseDataSource::ICRP116,
    );
    let (_, pa_coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::PA,
        DoseDataSource::ICRP116,
    );

    // AP and PA should have different coefficients
    let mut different = false;
    for (ap, pa) in ap_coeffs.iter().zip(pa_coeffs.iter()) {
        if (ap - pa).abs() > 1e-10 {
            different = true;
            break;
        }
    }
    assert!(
        different,
        "AP and PA geometries should have different coefficients"
    );
}

// ==================== EnergyFunctionFilter Tests ====================

#[test]
fn test_energy_function_filter_creation() {
    let energy = vec![1.0, 10.0, 100.0, 1000.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let filter = EnergyFunctionFilter::new(energy.clone(), y.clone());

    assert_eq!(filter.num_bins(), 1);
    assert_eq!(filter.energy().len(), 4);
    assert_eq!(filter.y().len(), 4);
}

#[test]
fn test_energy_function_filter_from_dose_coefficients() {
    let (energy, coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::AP,
        DoseDataSource::ICRP116,
    );

    // Should be able to create filter from dose coefficients
    let filter = EnergyFunctionFilter::new(energy.clone(), coeffs.clone());

    assert_eq!(filter.num_bins(), 1);
    assert_eq!(filter.energy().len(), energy.len());
    assert_eq!(filter.y().len(), coeffs.len());
}

#[test]
fn test_cubic_spline_at_data_points() {
    let energy = vec![1.0, 10.0, 100.0, 1000.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let filter = EnergyFunctionFilter::new(energy.clone(), y.clone());

    // At data points, cubic spline should return exact values
    for (i, &e) in energy.iter().enumerate() {
        let weight = filter.get_weight(e).unwrap();
        assert!(
            (weight - y[i]).abs() < 1e-10,
            "At energy {}, expected {}, got {}",
            e,
            y[i],
            weight
        );
    }
}

#[test]
fn test_cubic_spline_interpolation_smooth() {
    // Create data from a smooth function (y = x^0.5)
    let energy = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // sqrt values
    let filter = EnergyFunctionFilter::new(energy, y);

    // Test at midpoint - cubic spline should give smooth result
    let weight = filter.get_weight(6.25).unwrap(); // sqrt(6.25) = 2.5
                                                   // Allow some deviation since cubic spline won't be exact for sqrt
    assert!(
        (weight - 2.5).abs() < 0.3,
        "Interpolated value at 6.25 should be close to 2.5, got {}",
        weight
    );
}

#[test]
fn test_outside_energy_range() {
    let energy = vec![10.0, 100.0, 1000.0, 10000.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let filter = EnergyFunctionFilter::new(energy, y);

    // Below range
    assert_eq!(filter.get_weight(5.0), None);
    assert_eq!(filter.get_weight(9.99), None);

    // Above range
    assert_eq!(filter.get_weight(10001.0), None);
    assert_eq!(filter.get_weight(1e6), None);
}

#[test]
fn test_at_range_boundaries() {
    let energy = vec![10.0, 100.0, 1000.0, 10000.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let filter = EnergyFunctionFilter::new(energy, y);

    // At exact boundaries
    assert!(filter.get_weight(10.0).is_some());
    assert!(filter.get_weight(10000.0).is_some());
}

#[test]
#[should_panic(expected = "monotonically increasing")]
fn test_non_monotonic_energy_panics() {
    let energy = vec![10.0, 5.0, 100.0, 1000.0]; // Not monotonic
    let y = vec![1.0, 2.0, 3.0, 4.0];
    EnergyFunctionFilter::new(energy, y);
}

#[test]
#[should_panic(expected = "same length")]
fn test_mismatched_lengths_panics() {
    let energy = vec![10.0, 100.0, 1000.0, 10000.0];
    let y = vec![1.0, 2.0, 3.0]; // One less
    EnergyFunctionFilter::new(energy, y);
}

#[test]
#[should_panic(expected = "at least 4")]
fn test_too_few_points_panics() {
    let energy = vec![10.0, 100.0, 1000.0]; // Only 3 points
    let y = vec![1.0, 2.0, 3.0];
    EnergyFunctionFilter::new(energy, y);
}

#[test]
fn test_with_realistic_dose_data() {
    // Use actual ICRP-116 data
    let (energy, coeffs) = dose_coefficients(
        DoseParticle::Neutron,
        DoseGeometry::AP,
        DoseDataSource::ICRP116,
    );
    let filter = EnergyFunctionFilter::new(energy.clone(), coeffs.clone());

    // Test at a few known energies
    // 14 MeV = 14e6 eV - typical fusion neutron energy
    if let Some(weight) = filter.get_weight(14e6) {
        assert!(
            weight > 0.0,
            "Dose coefficient at 14 MeV should be positive"
        );
        // ICRP-116 AP geometry at ~14 MeV is around 400-500 pSv·cm²
        assert!(
            weight > 100.0 && weight < 1000.0,
            "Dose coefficient at 14 MeV should be reasonable, got {}",
            weight
        );
    }

    // 1 MeV = 1e6 eV
    if let Some(weight) = filter.get_weight(1e6) {
        assert!(weight > 0.0);
        // ICRP-116 AP geometry at 1 MeV is around 300 pSv·cm²
        assert!(weight > 100.0 && weight < 500.0);
    }

    // Thermal energy ~0.025 eV
    if let Some(weight) = filter.get_weight(0.025) {
        assert!(weight > 0.0);
        // Thermal neutron dose coefficient is much lower
        assert!(weight < 50.0);
    }
}
