use crate::buffer::F64Buffer;
use crate::reaction_product::ReactionProduct;
use serde::{Deserialize, Serialize};

/// Represents a single reaction channel (identified by ENDF/MT number) for a
/// specific nuclide at a given temperature.
///
/// The reaction may either provide its own truncated energy grid or rely on
/// the parent nuclide's top‑level temperature grid (offset by `threshold_idx`).
/// `cross_section` values correspond 1‑to‑1 with the reaction's effective
/// energy grid (either its own `energy` or a slice of the parent grid).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Reaction {
    /// Cross section values in barns for the reaction energy grid.
    pub cross_section: F64Buffer,
    /// Index into the parent (top‑level) energy grid where this reaction becomes active.
    pub threshold_idx: usize,
    /// Reaction‑specific energy grid (may be empty until synthesized).
    ///
    /// Normally a zero-copy view of the parent nuclide's grid for this
    /// temperature, offset by `threshold_idx`, rather than a copy of it. A copy
    /// per reaction was the single largest allocation in a load: ENDF/B-VIII.1
    /// Fe56 carries ~42k grid points and 90 reactions per temperature, so the
    /// duplicated grid came to 38 MiB against 2.3 MiB of actual grids
    /// (issue #476).
    #[serde(skip, default)]
    pub energy: F64Buffer, // Reaction-specific energy grid
    /// ENDF/MT reaction identifier.
    pub mt_number: i32, // The MT number for this reaction
    /// Q-value of the reaction in eV (energy released/absorbed in the reaction)
    #[serde(default)]
    pub q_value: f64,
    /// Products emitted by this reaction (e.g., neutrons, photons, fragments)
    #[serde(default)]
    pub products: Vec<ReactionProduct>,
    /// Whether scattering is in center-of-mass frame (requires CM to LAB conversion)
    #[serde(default)]
    pub scatter_in_cm: bool,
    /// Whether this reaction is redundant (cross section is sum of constituent reactions).
    /// Redundant reactions should be skipped for transport physics but can still be tallied.
    #[serde(default)]
    pub redundant: bool,
}

impl Reaction {
    /// Returns the cross section value for a given neutron energy using linear interpolation.
    /// For threshold reactions (threshold_idx > 0), returns 0 below the threshold.
    /// For non-threshold reactions (threshold_idx == 0), returns the first value below grid.
    /// If above the grid, returns the last value.
    /// Otherwise, performs linear interpolation between grid points.
    ///
    /// Indexes `cross_section` with a position found in `energy`, which is sound
    /// because the two are the same length: the reader establishes that when it
    /// attaches the grid, so a file where they disagree fails to load rather
    /// than panicking here on the first lookup (issue #507).
    #[inline]
    pub fn cross_section_at(&self, energy: f64) -> Option<f64> {
        if self.energy.is_empty() || self.cross_section.is_empty() {
            return None;
        }
        // A NaN belongs to no interval, and the boundary cases below do not
        // catch it: every comparison against it is false, so it would reach the
        // search and sort past the end of the grid.
        if energy.is_nan() {
            return None;
        }

        let n = self.energy.len();

        // Handle boundary cases
        if energy < self.energy[0] {
            // For threshold reactions (threshold_idx > 0), return 0 below threshold
            // For non-threshold reactions (threshold_idx == 0), return first value
            // (e.g., capture reactions like n,γ continue below the grid)
            if self.threshold_idx > 0 {
                return Some(0.0);
            } else {
                return Some(self.cross_section[0]);
            }
        }
        // At or above maximum energy: return last value
        if energy >= self.energy[n - 1] {
            return Some(self.cross_section[n - 1]);
        }

        // Binary search for the interval. `total_cmp` rather than `partial_cmp`
        // with an unwrap: the two agree on every ordinary grid point, and a NaN
        // that reached the grid from a corrupt file orders instead of panicking.
        match self.energy.binary_search_by(|e| e.total_cmp(&energy)) {
            Ok(idx) => Some(self.cross_section[idx]),
            Err(idx) => {
                // idx is the insertion point, so energy is between [idx-1] and [idx].
                // The boundary cases above leave `energy[0] < energy < energy[n-1]`,
                // which puts the insertion point in `1..n` for a sorted grid. An
                // unsorted one can land outside that, and `idx - 1` would wrap on
                // a `usize`, so report no value rather than indexing on it.
                if idx == 0 || idx >= n {
                    return None;
                }
                let i = idx - 1;
                let e0 = self.energy[i];
                let e1 = self.energy[idx];
                let xs0 = self.cross_section[i];
                let xs1 = self.cross_section[idx];

                // Linear interpolation
                let t = (energy - e0) / (e1 - e0);
                Some(xs0 + t * (xs1 - xs0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::F64Buffer;

    /// A non-threshold reaction on a three-point grid.
    fn reaction(energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
        Reaction {
            cross_section: F64Buffer::from(cross_section),
            threshold_idx: 0,
            energy: F64Buffer::from(energy),
            mt_number: 16,
            q_value: 0.0,
            products: Vec::new(),
            scatter_in_cm: false,
            redundant: false,
        }
    }

    #[test]
    fn interpolates_between_grid_points() {
        let rxn = reaction(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
        assert_eq!(rxn.cross_section_at(1.5), Some(15.0));
        assert_eq!(rxn.cross_section_at(2.0), Some(20.0));
    }

    #[test]
    fn below_and_above_the_grid_clamp() {
        let rxn = reaction(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
        assert_eq!(rxn.cross_section_at(0.5), Some(10.0));
        assert_eq!(rxn.cross_section_at(9.0), Some(30.0));
    }

    #[test]
    fn a_threshold_reaction_is_zero_below_its_grid() {
        let mut rxn = reaction(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
        rxn.threshold_idx = 7;
        assert_eq!(rxn.cross_section_at(0.5), Some(0.0));
    }

    /// Issue #507: a NaN query used to reach `partial_cmp(..).unwrap()`.
    #[test]
    fn a_nan_query_has_no_cross_section() {
        let rxn = reaction(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
        assert_eq!(rxn.cross_section_at(f64::NAN), None);
    }

    /// Issue #507: a NaN on the grid itself is a corrupt file, not a panic.
    #[test]
    fn a_nan_on_the_grid_does_not_panic() {
        let rxn = reaction(vec![1.0, f64::NAN, 3.0], vec![10.0, 20.0, 30.0]);
        rxn.cross_section_at(1.5);
        rxn.cross_section_at(2.5);
    }

    /// Issue #507: an unsorted grid can put the insertion point at 0, where
    /// `idx - 1` would wrap on a `usize`.
    #[test]
    fn an_unsorted_grid_does_not_underflow() {
        let rxn = reaction(vec![5.0, 1.0, 9.0], vec![10.0, 20.0, 30.0]);
        rxn.cross_section_at(2.0);
        rxn.cross_section_at(6.0);
    }

    #[test]
    fn an_empty_reaction_has_no_cross_section() {
        let rxn = reaction(Vec::new(), Vec::new());
        assert_eq!(rxn.cross_section_at(1.0), None);
    }
}
