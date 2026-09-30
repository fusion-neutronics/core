//! Dose coefficient data and functions for ICRP-74 and ICRP-116.
//!
//! Provides energy-dependent fluence-to-effective-dose conversion coefficients
//! for neutrons and photons in various irradiation geometries, and
//! fluence-to-ambient-dose-equivalent H*(10) coefficients from ICRP-74.

/// Irradiation geometry for dose coefficients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoseGeometry {
    /// Anterior-Posterior (front-facing)
    AP,
    /// Posterior-Anterior (back-facing)
    PA,
    /// Left Lateral
    LLAT,
    /// Right Lateral
    RLAT,
    /// Rotational (360° rotation)
    ROT,
    /// Isotropic
    ISO,
}

/// Data source for dose coefficients
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoseDataSource {
    /// ICRP Publication 74 (1996)
    ICRP74,
    /// ICRP Publication 116 (2010)
    ICRP116,
}

/// Particle the dose coefficients convert the fluence of
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoseParticle {
    /// Neutrons
    Neutron,
    /// Photons
    Photon,
}

// Embed the data files at compile time
const ICRP74_NEUTRONS: &str = include_str!("icrp74/neutrons.txt");
const ICRP116_NEUTRONS: &str = include_str!("icrp116/neutrons.txt");
const ICRP74_PHOTONS: &str = include_str!("icrp74/photons.txt");
const ICRP116_PHOTONS: &str = include_str!("icrp116/photons.txt");
const ICRP74_AMBIENT_NEUTRONS: &str = include_str!("icrp74_ambient/neutrons.txt");
const ICRP74_AMBIENT_PHOTONS: &str = include_str!("icrp74_ambient/photons.txt");

/// Return effective dose conversion coefficients.
///
/// Returns energy-dependent fluence-to-dose conversion factors based on
/// ICRP Publication 74 or 116.
///
/// # Arguments
/// * `particle` - Neutron or photon
/// * `geometry` - Irradiation geometry (AP, PA, LLAT, RLAT, ROT, ISO)
/// * `data_source` - ICRP74 or ICRP116
///
/// # Returns
/// * `(Vec<f64>, Vec<f64>)` - (energy in eV, dose_coefficients in pSv·cm²)
///
/// # Example
/// ```
/// use yamc_nuclide::data::effective_dose::{
///     dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle,
/// };
///
/// let (energy, coeffs) =
///     dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP116);
/// assert!(!energy.is_empty());
/// assert_eq!(energy.len(), coeffs.len());
/// ```
///
/// # References
/// - ICRP Publication 74: <https://doi.org/10.1016/S0146-6453(96)90010-X>
/// - ICRP Publication 116: <https://doi.org/10.1016/j.icrp.2011.10.001>
pub fn dose_coefficients(
    particle: DoseParticle,
    geometry: DoseGeometry,
    data_source: DoseDataSource,
) -> (Vec<f64>, Vec<f64>) {
    let data_str = match (particle, data_source) {
        (DoseParticle::Neutron, DoseDataSource::ICRP74) => ICRP74_NEUTRONS,
        (DoseParticle::Neutron, DoseDataSource::ICRP116) => ICRP116_NEUTRONS,
        (DoseParticle::Photon, DoseDataSource::ICRP74) => ICRP74_PHOTONS,
        (DoseParticle::Photon, DoseDataSource::ICRP116) => ICRP116_PHOTONS,
    };

    let column_index = match geometry {
        DoseGeometry::AP => 1,
        DoseGeometry::PA => 2,
        DoseGeometry::LLAT => 3,
        DoseGeometry::RLAT => 4,
        DoseGeometry::ROT => 5,
        DoseGeometry::ISO => 6,
    };

    parse_dose_data(data_str, column_index)
}

/// Return ambient dose equivalent H*(10) conversion coefficients.
///
/// H*(10) is the operational quantity defined at 10 mm depth in the ICRU
/// sphere in an aligned and expanded field, so it has no irradiation geometry.
/// It is designed to over-estimate effective dose for area monitoring, but
/// under-reads it for high-energy neutrons.
///
/// ICRP-74 tabulates neutrons from 1 meV to 20 MeV and photons from 10 keV to
/// 10 MeV. ICRP-116 does not tabulate H*(10), so `DoseDataSource::ICRP116`
/// returns `None`.
///
/// # Returns
/// * `Some((Vec<f64>, Vec<f64>))` - (energy in eV, dose_coefficients in pSv·cm²)
///
/// # Example
/// ```
/// use yamc_nuclide::data::effective_dose::{
///     ambient_dose_coefficients, DoseDataSource, DoseParticle,
/// };
///
/// let (energy, coeffs) =
///     ambient_dose_coefficients(DoseParticle::Neutron, DoseDataSource::ICRP74).unwrap();
/// assert_eq!(energy.len(), coeffs.len());
/// assert!(ambient_dose_coefficients(DoseParticle::Neutron, DoseDataSource::ICRP116).is_none());
/// ```
///
/// # References
/// - ICRP Publication 74: <https://doi.org/10.1016/S0146-6453(96)90010-X>,
///   Table A.21 (photons) and Table A.42 (neutrons)
pub fn ambient_dose_coefficients(
    particle: DoseParticle,
    data_source: DoseDataSource,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let data_str = match (particle, data_source) {
        (DoseParticle::Neutron, DoseDataSource::ICRP74) => ICRP74_AMBIENT_NEUTRONS,
        (DoseParticle::Photon, DoseDataSource::ICRP74) => ICRP74_AMBIENT_PHOTONS,
        (_, DoseDataSource::ICRP116) => return None,
    };
    Some(parse_dose_data(data_str, 1))
}

/// Parse dose coefficient data from embedded text files.
///
/// The data files have the following format:
/// - Line 1: Title
/// - Line 2: Empty
/// - Line 3: Header (Energy, AP, PA, LLAT, RLAT, ROT, ISO for effective dose,
///   or Energy, H*(10) for ambient dose equivalent)
/// - Lines 4+: Data rows (energy in MeV, then coefficients)
fn parse_dose_data(data: &str, column_index: usize) -> (Vec<f64>, Vec<f64>) {
    let mut energies = Vec::new();
    let mut coefficients = Vec::new();

    for line in data.lines().skip(3) {
        // Skip 3 header lines
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() > column_index {
            if let (Ok(energy_mev), Ok(coeff)) = (
                parse_scientific_notation(parts[0]),
                parse_scientific_notation(parts[column_index]),
            ) {
                energies.push(energy_mev * 1e6); // Convert MeV to eV
                coefficients.push(coeff);
            }
        }
    }

    (energies, coefficients)
}

/// Parse a number that might be in scientific notation (e.g., "1.0E-9" or "1.17E+3")
fn parse_scientific_notation(s: &str) -> Result<f64, std::num::ParseFloatError> {
    s.parse::<f64>()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icrp116_neutron_dose_coefficients() {
        let (energy, coeffs) = dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP116);

        // Should have data points
        assert!(!energy.is_empty());
        assert_eq!(energy.len(), coeffs.len());

        // ICRP-116 has 68 neutron energy points
        assert_eq!(energy.len(), 68);

        // Energy should be in eV (converted from MeV)
        assert!(energy[0] > 0.0);
        assert!(energy[0] < 1.0); // First point is ~1e-9 MeV = 1e-3 eV

        // Coefficients should be positive
        for &c in &coeffs {
            assert!(c > 0.0);
        }
    }

    #[test]
    fn test_icrp74_neutron_dose_coefficients() {
        let (energy, coeffs) = dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP74);

        assert!(!energy.is_empty());
        assert_eq!(energy.len(), coeffs.len());

        // ICRP-74 has 47 neutron energy points
        assert_eq!(energy.len(), 47);
    }

    #[test]
    fn test_icrp116_photon_dose_coefficients() {
        let (energy, coeffs) =
            dose_coefficients(DoseParticle::Photon, DoseGeometry::AP, DoseDataSource::ICRP116);

        // ICRP-116 tabulates photons from 10 keV to 10 GeV on 55 points.
        assert_eq!(energy.len(), 55);
        assert_eq!(energy.len(), coeffs.len());
        assert_eq!(energy[0], 1.0e4);
        assert_eq!(energy[energy.len() - 1], 1.0e10);

        // ICRP-116 Table A.1: 0.0685 pSv cm² at 10 keV, 4.49 at 1 MeV (AP).
        assert!((coeffs[0] - 0.0685).abs() < 1e-12);
        let one_mev = energy.iter().position(|&e| e == 1.0e6).unwrap();
        assert!((coeffs[one_mev] - 4.49).abs() < 1e-12);
    }

    #[test]
    fn test_icrp74_photon_dose_coefficients() {
        let (energy, coeffs) =
            dose_coefficients(DoseParticle::Photon, DoseGeometry::AP, DoseDataSource::ICRP74);

        assert_eq!(energy.len(), 23);
        assert_eq!(energy.len(), coeffs.len());
    }

    #[test]
    fn test_neutron_and_photon_tables_are_distinct() {
        let (neutron_energy, _) =
            dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP116);
        let (photon_energy, _) =
            dose_coefficients(DoseParticle::Photon, DoseGeometry::AP, DoseDataSource::ICRP116);
        assert_ne!(neutron_energy.len(), photon_energy.len());
    }

    #[test]
    fn test_different_geometries() {
        let geometries = [
            DoseGeometry::AP,
            DoseGeometry::PA,
            DoseGeometry::LLAT,
            DoseGeometry::RLAT,
            DoseGeometry::ROT,
            DoseGeometry::ISO,
        ];

        for geom in geometries {
            let (energy, coeffs) = dose_coefficients(DoseParticle::Neutron, geom, DoseDataSource::ICRP116);
            assert!(!energy.is_empty());
            assert_eq!(energy.len(), coeffs.len());
        }
    }

    #[test]
    fn test_energy_monotonically_increasing() {
        let (energy, _) = dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP116);

        for i in 1..energy.len() {
            assert!(
                energy[i] > energy[i - 1],
                "Energy grid must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_icrp74_ambient_neutron_coefficients() {
        let (energy, coeffs) =
            ambient_dose_coefficients(DoseParticle::Neutron, DoseDataSource::ICRP74).unwrap();

        // ICRP-74 Table A.42: 47 points from 1 meV to 20 MeV.
        assert_eq!(energy.len(), 47);
        assert_eq!(energy.len(), coeffs.len());
        assert_eq!(energy[0], 1.0e-3);
        assert_eq!(energy[energy.len() - 1], 2.0e7);

        // 10.6 pSv cm² at thermal (0.0253 eV), 520 at 14 MeV, 600 at 20 MeV.
        let at = |e: f64| coeffs[energy.iter().position(|&x| (x - e).abs() < 1e-9 * e).unwrap()];
        assert_eq!(at(2.53e-2), 10.6);
        assert_eq!(at(1.4e7), 520.0);
        assert_eq!(at(2.0e7), 600.0);
    }

    #[test]
    fn test_icrp74_ambient_photon_coefficients() {
        let (energy, coeffs) =
            ambient_dose_coefficients(DoseParticle::Photon, DoseDataSource::ICRP74).unwrap();

        // ICRP-74 Table A.21: 25 points from 10 keV to 10 MeV.
        assert_eq!(energy.len(), 25);
        assert_eq!(energy.len(), coeffs.len());
        assert_eq!(energy[0], 1.0e4);
        assert_eq!(energy[energy.len() - 1], 1.0e7);

        // 0.061 pSv cm² at 10 keV, 5.20 at 1 MeV, 25.6 at 10 MeV.
        assert_eq!(coeffs[0], 0.061);
        let one_mev = energy.iter().position(|&e| e == 1.0e6).unwrap();
        assert_eq!(coeffs[one_mev], 5.20);
        assert_eq!(coeffs[coeffs.len() - 1], 25.6);
    }

    #[test]
    fn test_icrp116_has_no_ambient_coefficients() {
        assert!(ambient_dose_coefficients(DoseParticle::Neutron, DoseDataSource::ICRP116).is_none());
        assert!(ambient_dose_coefficients(DoseParticle::Photon, DoseDataSource::ICRP116).is_none());
    }

    #[test]
    fn test_ap_vs_pa_different() {
        let (_, ap_coeffs) = dose_coefficients(DoseParticle::Neutron, DoseGeometry::AP, DoseDataSource::ICRP116);
        let (_, pa_coeffs) = dose_coefficients(DoseParticle::Neutron, DoseGeometry::PA, DoseDataSource::ICRP116);

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
}
