//! A bound nuclide's thermal-table cross sections on the GPU's grids.
//!
//! Below a thermal table's energy_max the CPU replaces the elastic channel
//! of each nuclide the table binds before any collision is sampled
//! (`yamc_materials::material::thermal::apply_bound_xs`):
//!
//! ```text
//! sigma_el  <- f (sigma_el,b + sigma_inel,b) + (1 - f) sigma_el,free
//! sigma_tot <- sigma_tot + f (sigma_el,b + sigma_inel,b - sigma_el,free)
//! ```
//!
//! with f the bound fraction. The GPU extractors apply the same replacement
//! at every point of their grids through [`ThermalOverride`], so the flight
//! length, the choice of nuclide, the reaction split and the elastic and
//! total tally scores all see the bound cross section, as on the CPU. Elastic
//! and total move by the same amount, so the inelastic and absorption a
//! derivation takes from their difference are unchanged.

/// How one nuclide's elastic and total cross sections change below its
/// thermal table's energy_max. Built by the host from the material's bound
/// table, which `yamc-gpu` does not depend on.
pub struct ThermalOverride<'a> {
    /// The fraction of the nuclide that is bound, in (0, 1].
    pub fraction: f64,
    /// Below this energy (eV) the table applies.
    pub energy_max: f64,
    /// The table's own energies the grid needs for its cross sections to
    /// interpolate faithfully (every inelastic point, both sides of every
    /// Bragg edge, a geometric ladder up to energy_max), merged into the
    /// material's union grid by [`super::union_energy_grid`].
    pub grid_points: Vec<f64>,
    /// `sigma_el,b(E) + sigma_inel,b(E)` (barns), blended between table
    /// temperatures as the CPU blends them.
    pub bound_xs: Box<dyn Fn(f64) -> f64 + Sync + 'a>,
}

impl ThermalOverride<'_> {
    /// The elastic and total at `energy` given the free-atom elastic
    /// `free_elastic` and total `total` there (any common scale: microscopic,
    /// or both already multiplied by the nuclide's density with
    /// `density` passed so the bound part is scaled the same way).
    pub fn apply(&self, energy: f64, free_elastic: f64, total: f64, density: f64) -> (f64, f64) {
        if energy >= self.energy_max {
            return (free_elastic, total);
        }
        let f = self.fraction;
        let thermal = density * f * (self.bound_xs)(energy);
        (
            thermal + (1.0 - f) * free_elastic,
            total + thermal - f * free_elastic,
        )
    }
}

/// The override for nuclide `i` of a material, if one binds it. `thermal` is
/// either empty (no nuclide is bound) or aligned with the material's
/// nuclides.
pub fn thermal_of<'b, 'a>(
    thermal: &'b [Option<ThermalOverride<'a>>],
    i: usize,
) -> Option<&'b ThermalOverride<'a>> {
    thermal.get(i).and_then(Option::as_ref)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_override_replaces_elastic_and_moves_total_by_the_same_amount() {
        let bound = |e: f64| 2.0 + e;
        let o = ThermalOverride {
            fraction: 0.6,
            energy_max: 4.0,
            grid_points: Vec::new(),
            bound_xs: Box::new(bound),
        };
        let (el, tot) = o.apply(1.0, 5.0, 7.0, 1.0);
        let th = 0.6 * 3.0;
        assert!((el - (th + 0.4 * 5.0)).abs() < 1e-12);
        assert!((tot - (7.0 + th - 0.6 * 5.0)).abs() < 1e-12);
        assert!(((tot - 7.0) - (el - 5.0)).abs() < 1e-12);
        // Density-weighted inputs scale the bound part by the same density.
        let (el_m, tot_m) = o.apply(1.0, 2.0 * 5.0, 2.0 * 7.0, 2.0);
        assert!((el_m - 2.0 * el).abs() < 1e-12);
        assert!((tot_m - 2.0 * tot).abs() < 1e-12);
        // At and above energy_max nothing changes.
        assert_eq!(o.apply(4.0, 5.0, 7.0, 1.0), (5.0, 7.0));
    }
}
