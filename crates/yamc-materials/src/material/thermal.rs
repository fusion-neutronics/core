//! Thermal scattering cross sections on a material's union grid.
//!
//! Below a bound table's energy_max, the elastic channel of each nuclide the
//! table binds is replaced before any collision is sampled, so flight length,
//! the choice of reaction and the scattering kinematics all see the bound
//! cross section:
//!
//! ```text
//! sigma_el  <- f (sigma_el,b + sigma_inel,b) + (1 - f) sigma_el,free
//! sigma_tot <- sigma_tot + f (sigma_el,b + sigma_inel,b - sigma_el,free)
//! ```
//!
//! with f the bound fraction. Between two table temperatures the bound cross
//! sections are blended linearly in temperature with the same
//! [`yamc_nuclide::temperature::blend_weight`] the neutron data uses.

use yamc_nuclide::interpolate_linear;
use yamc_nuclide::temperature::blend_weight;
use yamc_nuclide::thermal_scattering::ThermalTemperature;

use crate::material::BoundThermal;

/// Ratio between successive grid points added below energy_max: linear
/// interpolation of a 1/E cross section across a step of 2% is off by at most
/// (0.02)^2 / 8 = 5e-5 relative.
const GRID_RATIO: f64 = 1.02;

/// How far below a Bragg edge the point marking the step is put, relative.
const EDGE_GAP: f64 = 1e-9;

impl BoundThermal {
    /// The table temperatures in use and the weight of the upper one: the
    /// single table the material sits on (weight 0), or the two it falls
    /// between.
    pub fn bracket(&self) -> (&ThermalTemperature, Option<(&ThermalTemperature, f64)>) {
        let mut temps: Vec<&ThermalTemperature> = self.table.temperatures.values().collect();
        temps.sort_by(|a, b| a.kelvin.total_cmp(&b.kelvin));
        match temps.as_slice() {
            [one] => (one, None),
            [lo, hi, ..] => (
                lo,
                Some((hi, blend_weight(lo.kelvin, hi.kelvin, self.kelvin))),
            ),
            [] => unreachable!("a loaded table holds at least one temperature"),
        }
    }

    fn blended(&self, f: impl Fn(&ThermalTemperature) -> f64) -> f64 {
        match self.bracket() {
            (lo, None) => f(lo),
            (lo, Some((hi, w))) => (1.0 - w) * f(lo) + w * f(hi),
        }
    }

    /// The bound elastic cross section per principal atom at `energy` (eV).
    pub fn elastic_xs(&self, energy: f64) -> f64 {
        self.blended(|t| t.elastic.as_ref().map_or(0.0, |e| e.xs(energy)))
    }

    /// The bound inelastic cross section per principal atom at `energy` (eV),
    /// linear between the table's points.
    pub fn inelastic_xs(&self, energy: f64) -> f64 {
        self.blended(|t| interpolate_linear(&t.inelastic.energy, &t.inelastic.xs, energy))
    }

    /// The energies below energy_max the union grid needs for these cross
    /// sections to interpolate faithfully: every point of the inelastic
    /// tables, every Bragg edge with a point just below it, and a geometric
    /// ladder between `low` and energy_max.
    pub fn grid_points(&self, low: f64) -> Vec<f64> {
        let emax = self.table.energy_max;
        let mut points = Vec::new();
        for t in self.table.temperatures.values() {
            points.extend(t.inelastic.energy.iter().copied().filter(|&e| e < emax));
            if let Some(
                yamc_nuclide::thermal_scattering::Elastic::Coherent { bragg_edges, .. }
                | yamc_nuclide::thermal_scattering::Elastic::Mixed { bragg_edges, .. },
            ) = &t.elastic
            {
                for &e in bragg_edges.iter().filter(|&&e| e < emax) {
                    points.push(e);
                    points.push(e * (1.0 - EDGE_GAP));
                }
            }
        }
        let mut e = low.max(1e-5);
        while e < emax {
            points.push(e);
            e *= GRID_RATIO;
        }
        points.push(emax);
        points.push(emax * (1.0 - EDGE_GAP));
        points
    }
}

/// Replace the elastic and total cross sections of each bound nuclide in
/// `micro` (nuclide -> MT -> values on `grid`) below its table's energy_max.
pub(crate) fn apply_bound_xs(
    thermal: &[BoundThermal],
    grid: &[f64],
    micro: &mut std::collections::HashMap<String, std::collections::HashMap<i32, Vec<f64>>>,
) {
    for bound in thermal {
        let emax = bound.table.energy_max;
        let f = bound.fraction;
        let n_below = grid.partition_point(|&e| e < emax);
        let thermal_xs: Vec<f64> = grid[..n_below]
            .iter()
            .map(|&e| f * (bound.elastic_xs(e) + bound.inelastic_xs(e)))
            .collect();
        for nuclide in &bound.nuclides {
            let Some(reactions) = micro.get_mut(nuclide) else {
                continue;
            };
            let Some(free) = reactions.get(&2).map(|v| v[..n_below].to_vec()) else {
                continue;
            };
            if let Some(el) = reactions.get_mut(&2) {
                for i in 0..n_below {
                    el[i] = thermal_xs[i] + (1.0 - f) * free[i];
                }
            }
            if let Some(total) = reactions.get_mut(&1) {
                for i in 0..n_below {
                    total[i] += thermal_xs[i] - f * free[i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;
    use yamc_nuclide::thermal_scattering::{Elastic, Inelastic, ThermalTable};

    fn temperature(kelvin: f64, xs: f64) -> ThermalTemperature {
        ThermalTemperature {
            kelvin,
            kt: 8.617333262e-5 * kelvin,
            inelastic: Inelastic {
                energy: vec![1e-5, 1.0, 4.0],
                xs: vec![xs, xs, xs],
                distributions: Vec::new(),
                n_mu: 16,
            },
            elastic: Some(Elastic::Coherent {
                bragg_edges: vec![0.002, 0.005],
                factors: vec![0.01, 0.03],
            }),
        }
    }

    fn bound(kelvin: f64, fraction: f64) -> BoundThermal {
        let temperatures = BTreeMap::from([
            ("400K".to_string(), temperature(400.0, 2.0)),
            ("500K".to_string(), temperature(500.0, 4.0)),
        ]);
        BoundThermal {
            table: Arc::new(ThermalTable {
                name: "c_Test".into(),
                bound_nuclides: vec!["C12".into()],
                atomic_weight_ratio: 11.9,
                energy_max: 4.0,
                available: vec![("400K".into(), 400.0), ("500K".into(), 500.0)],
                temperatures,
            }),
            fraction,
            nuclides: vec!["C12".into()],
            kelvin,
        }
    }

    #[test]
    fn between_two_tables_the_cross_sections_blend_linearly_in_temperature() {
        let b = bound(450.0, 1.0);
        assert!((b.inelastic_xs(0.5) - 3.0).abs() < 1e-12);
        let b = bound(475.0, 1.0);
        assert!((b.inelastic_xs(0.5) - 3.5).abs() < 1e-12);
        // Coherent elastic is the same at both, so the blend is that value.
        assert!((b.elastic_xs(0.003) - 0.01 / 0.003).abs() < 1e-12);
    }

    #[test]
    fn the_bound_cross_section_replaces_free_elastic_below_energy_max_only() {
        let b = bound(400.0, 0.6);
        let grid = vec![0.001, 0.003, 1.0, 3.9, 4.0, 10.0];
        let free_el = vec![5.0; 6];
        let total = vec![7.0; 6];
        let mut micro = HashMap::from([(
            "C12".to_string(),
            HashMap::from([(2, free_el.clone()), (1, total.clone())]),
        )]);
        apply_bound_xs(std::slice::from_ref(&b), &grid, &mut micro);
        let r = &micro["C12"];
        for (i, &e) in grid.iter().enumerate() {
            if e < 4.0 {
                let th = 0.6 * (b.elastic_xs(e) + b.inelastic_xs(e));
                assert!((r[&2][i] - (th + 0.4 * 5.0)).abs() < 1e-12, "{e}");
                assert!((r[&1][i] - (7.0 + th - 0.6 * 5.0)).abs() < 1e-12, "{e}");
                // The total moves by exactly what elastic moves by.
                assert!(((r[&1][i] - 7.0) - (r[&2][i] - 5.0)).abs() < 1e-12);
            } else {
                assert_eq!(
                    (r[&2][i], r[&1][i]),
                    (5.0, 7.0),
                    "{e}: at or above energy_max"
                );
            }
        }
    }

    #[test]
    fn the_grid_marks_every_bragg_edge_and_energy_max() {
        let b = bound(400.0, 1.0);
        let points = b.grid_points(1e-5);
        for e in [0.002, 0.005, 4.0] {
            assert!(points.contains(&e));
            assert!(points.contains(&(e * (1.0 - EDGE_GAP))));
        }
        // The geometric ladder is no coarser than GRID_RATIO.
        let mut sorted = points.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        assert!(sorted
            .windows(2)
            .all(|w| w[1] / w[0] <= GRID_RATIO * (1.0 + 1e-12)));
    }
}
