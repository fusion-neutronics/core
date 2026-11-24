//! Photon coefficient tables read with log-log interpolation.
//!
//! Both embedded tables are tabulated against photon energy over several
//! decades and vary over more decades still, which is what the log-log rule the
//! evaluators use is for: NIST publishes these tabulations expecting it, and
//! reading them linearly misses by tens of percent between the widely spaced
//! points at low energy.

use std::collections::HashMap;

use once_cell::sync::Lazy;

/// Photon mass attenuation coefficients mu/rho [cm^2/g] per element, from the
/// NIST XCOM database. See the file header for provenance.
const MASS_ATTENUATION_DATA: &str = include_str!("mass_attenuation_xcom.txt");

/// Mass energy-absorption coefficients mu_en/rho [cm^2/g] for air, from NIST
/// SRD 126. See the file header for provenance.
const AIR_MASS_ENERGY_ABSORPTION_DATA: &str =
    include_str!("mass_energy_absorption_air_nist126.txt");

/// A coefficient tabulated against photon energy, evaluated with log-log
/// interpolation.
///
/// Energies are in eV and strictly increasing. An absorption edge arrives from
/// the data files as two rows at the same energy; [`CoefficientTable::new`]
/// moves the lower of the pair down by one ulp so the grid can be searched,
/// which keeps the jump within a single ulp of where the data puts it.
///
/// Evaluating outside the tabulated range returns the nearest end value rather
/// than extrapolating: these tabulations stop where the physics they were
/// fitted to stops, and a log-log extrapolation of the last interval is not a
/// statement anyone made. Callers that must not silently clamp should compare
/// against [`CoefficientTable::min_energy`] and
/// [`CoefficientTable::max_energy`] first.
#[derive(Debug, Clone)]
pub struct CoefficientTable {
    energy: Vec<f64>,
    value: Vec<f64>,
}

impl CoefficientTable {
    /// Build a table from ascending `energy` (eV) and matching `value`.
    ///
    /// Repeated energies -- absorption edges -- are separated by one ulp, the
    /// earlier point moving down, so the grid is strictly increasing.
    pub fn new(mut energy: Vec<f64>, value: Vec<f64>) -> Self {
        assert_eq!(
            energy.len(),
            value.len(),
            "coefficient table needs one value per energy"
        );
        assert!(
            energy.len() >= 2,
            "coefficient table needs at least two points"
        );
        for i in (0..energy.len() - 1).rev() {
            if energy[i] >= energy[i + 1] {
                energy[i] = f64::from_bits(energy[i + 1].to_bits() - 1);
            }
        }
        Self { energy, value }
    }

    /// The tabulated energies [eV], ascending.
    pub fn energy(&self) -> &[f64] {
        &self.energy
    }

    /// The coefficient at each tabulated energy.
    pub fn value(&self) -> &[f64] {
        &self.value
    }

    /// Lowest tabulated energy [eV].
    pub fn min_energy(&self) -> f64 {
        self.energy[0]
    }

    /// Highest tabulated energy [eV].
    pub fn max_energy(&self) -> f64 {
        self.energy[self.energy.len() - 1]
    }

    /// The coefficient at `energy` [eV], log-log interpolated.
    pub fn interpolate(&self, energy: f64) -> f64 {
        if energy <= self.min_energy() {
            return self.value[0];
        }
        if energy >= self.max_energy() {
            return self.value[self.value.len() - 1];
        }
        // partition_point gives the first index above `energy`; the interval is
        // the pair straddling it.
        let upper = self.energy.partition_point(|&e| e <= energy).max(1);
        let (x0, x1) = (self.energy[upper - 1], self.energy[upper]);
        let (y0, y1) = (self.value[upper - 1], self.value[upper]);
        if y0 <= 0.0 || y1 <= 0.0 {
            // A zero or negative coefficient has no logarithm; fall back to
            // linear-linear over that interval rather than returning a NaN.
            return y0 + (y1 - y0) * (energy - x0) / (x1 - x0);
        }
        let log_span = x1.ln() - x0.ln();
        if log_span <= 0.0 {
            // The interval is an absorption edge, whose two energies are one
            // ulp apart and share a logarithm. There is nothing to interpolate
            // across: the jump happens here.
            return if energy <= x0 { y0 } else { y1 };
        }
        let slope = (y1.ln() - y0.ln()) / log_span;
        (y0.ln() + slope * (energy.ln() - x0.ln())).exp()
    }
}

/// Read a table whose data rows are `<energy> <value>`, skipping the prose
/// header. A row is data once its first token parses as a number, which is what
/// separates it from the header lines and needs no line count to be kept in
/// step with the file.
fn parse_pairs(data: &str) -> (Vec<f64>, Vec<f64>) {
    let mut energy = Vec::new();
    let mut value = Vec::new();
    for line in data.lines() {
        let mut tokens = line.split_whitespace();
        let (Some(first), Some(second)) = (tokens.next(), tokens.next()) else {
            continue;
        };
        let (Ok(e), Ok(v)) = (first.parse::<f64>(), second.parse::<f64>()) else {
            continue;
        };
        energy.push(e);
        value.push(v);
    }
    (energy, value)
}

/// mu/rho by atomic number, parsed once from the embedded XCOM table.
///
/// One element per line: `Z <atomic number> <n> <n energies> <n coefficients>`.
/// A malformed line is a corrupted embedded table rather than bad user input, so
/// it panics with the line rather than being skipped into a silently short grid.
static MASS_ATTENUATION: Lazy<HashMap<u32, CoefficientTable>> = Lazy::new(|| {
    let mut tables = HashMap::new();
    for line in MASS_ATTENUATION_DATA.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first() != Some(&"Z") {
            continue;
        }
        let malformed = || panic!("malformed row in the embedded mu/rho table: {line}");

        let (Ok(z), Ok(count)) = (tokens[1].parse::<u32>(), tokens[2].parse::<usize>()) else {
            malformed()
        };
        let body = &tokens[3..];
        if body.len() != 2 * count {
            malformed();
        }
        let read = |tokens: &[&str]| {
            tokens
                .iter()
                .map(|token| token.parse::<f64>().unwrap_or_else(|_| malformed()))
                .collect::<Vec<f64>>()
        };
        let energy = read(&body[..count]);
        let value = read(&body[count..]);
        tables.insert(z, CoefficientTable::new(energy, value));
    }
    tables
});

/// mu_en/rho for air, parsed once from the embedded NIST SRD 126 table.
static AIR_MASS_ENERGY_ABSORPTION: Lazy<CoefficientTable> = Lazy::new(|| {
    let (energy, value) = parse_pairs(AIR_MASS_ENERGY_ABSORPTION_DATA);
    CoefficientTable::new(energy, value)
});

/// The photon mass attenuation coefficient mu/rho [cm^2/g] of element `z`.
///
/// Total attenuation with coherent scattering, 1 keV to 20 MeV, for Z = 1 to
/// 100. `None` for an atomic number outside that range.
///
/// # Example
/// ```
/// use yamc_nuclide::data::photon_attenuation::mass_attenuation_coefficient;
///
/// // Iron at 1 MeV, where Compton scattering dominates.
/// let iron = mass_attenuation_coefficient(26).unwrap();
/// let mu_over_rho = iron.interpolate(1.0e6);
/// assert!((mu_over_rho - 0.05995).abs() < 1e-5);
/// ```
pub fn mass_attenuation_coefficient(z: u32) -> Option<&'static CoefficientTable> {
    MASS_ATTENUATION.get(&z)
}

/// The mass energy-absorption coefficient mu_en/rho [cm^2/g] of dry air near
/// sea level, 1 keV to 20 MeV.
pub fn mass_energy_absorption_air() -> &'static CoefficientTable {
    &AIR_MASS_ENERGY_ABSORPTION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_from_hydrogen_to_fermium_is_tabulated() {
        for z in 1..=100 {
            let table = mass_attenuation_coefficient(z)
                .unwrap_or_else(|| panic!("no mass attenuation data for Z={z}"));
            assert_eq!(table.min_energy(), 1000.0);
            assert_eq!(table.max_energy(), 2.0e7);
        }
        assert!(mass_attenuation_coefficient(0).is_none());
        assert!(mass_attenuation_coefficient(101).is_none());
    }

    #[test]
    fn tabulated_energies_strictly_increase_after_edge_separation() {
        for z in 1..=100 {
            let table = mass_attenuation_coefficient(z).unwrap();
            for pair in table.energy.windows(2) {
                assert!(
                    pair[1] > pair[0],
                    "Z={z} has a non-increasing grid at {} -> {}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    #[test]
    fn tabulated_points_are_returned_unchanged() {
        // The interpolation must reproduce the table at its own grid points.
        let iron = mass_attenuation_coefficient(26).unwrap();
        for (i, &energy) in iron.energy.iter().enumerate() {
            let interpolated = iron.interpolate(energy);
            assert!(
                (interpolated - iron.value[i]).abs() <= iron.value[i] * 1e-12,
                "point {i} at {energy} eV: {interpolated} != {}",
                iron.value[i]
            );
        }
    }

    #[test]
    fn iron_k_edge_jumps_between_its_two_rows() {
        // Fe's K edge sits at 7112 eV, where mu/rho jumps by about 7.7x. The
        // two rows at that energy must both be reachable.
        let iron = mass_attenuation_coefficient(26).unwrap();
        let below = iron.interpolate(7111.0);
        let above = iron.interpolate(7113.0);
        assert!(
            above / below > 7.0,
            "K edge did not jump: {below} -> {above}"
        );
    }

    #[test]
    fn log_log_interpolation_is_geometric_at_the_midpoint() {
        let table = CoefficientTable::new(vec![1.0, 100.0], vec![1.0, 100.0]);
        // On a log-log straight line through (1,1) and (100,100), x = 10 gives
        // y = 10.
        assert!((table.interpolate(10.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn evaluating_outside_the_grid_clamps_to_the_end_values() {
        let air = mass_energy_absorption_air();
        assert_eq!(air.interpolate(1.0), air.interpolate(air.min_energy()));
        assert_eq!(air.interpolate(1.0e12), air.interpolate(air.max_energy()));
    }

    #[test]
    fn air_absorption_matches_the_published_table() {
        let air = mass_energy_absorption_air();
        assert_eq!(air.min_energy(), 1000.0);
        assert_eq!(air.max_energy(), 2.0e7);
        // NIST SRD 126 Table 4: 2.789e-2 cm^2/g at 1 MeV, 2.325e-2 at 100 keV.
        assert!((air.interpolate(1.0e6) - 2.789e-2).abs() < 1e-15);
        assert!((air.interpolate(1.0e5) - 2.325e-2).abs() < 1e-15);
    }
}
