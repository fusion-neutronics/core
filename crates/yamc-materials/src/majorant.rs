//! Majorant total cross section for Woodcock (delta) tracking.
//!
//! A majorant `Σ_maj(E)` is an energy-dependent upper bound on the local
//! total cross section: `Σ_maj(E) ≥ Σ_t(r, E)` for every position `r` in
//! the geometry that the majorant covers. Woodcock tracking samples free
//! flight against `Σ_maj` (which is the same everywhere within a covered
//! region), then runs a rejection loop at each sampled "collision" to
//! decide whether the event is real (probability `Σ_t(r, E) / Σ_maj(E)`)
//! or virtual (no physics happens, the particle just continues).
//!
//! # Phase 1 (this file): global majorant
//!
//! `Σ_maj(E)` is constant over space -- the per-energy maximum of `Σ_t`
//! across every material in the model. Cheap to compute at simulation
//! start, no per-region partitioning, no per-step boundary checks
//! against region maps. The downside is the rejection rate: in fusion
//! geometries with vacuum + tungsten coexisting, the majorant tracks
//! tungsten and tracking through vacuum becomes ~all virtual events.
//! Phase 1.5 (cost-derived fallback) and Phase 2 (per-material local
//! majorants) address this.
//!
//! # URR handling (Phase 2c)
//!
//! Materials with URR (unresolved resonance range) probability-table
//! data are fully supported. [`GlobalMajorant::new`] queries
//! [`Material::total_xs_majorant`] at each unified energy point; for
//! energies inside any URR nuclide's URR range, that method returns
//! the worst-case URR-sampled `Σ_t` (max over the probability-table
//! CDF), not the smooth value.
//!
//! The result: `Σ_maj ≥ Σ_t(any URR draw)` everywhere, so the
//! Woodcock rejection probability `p_real = Σ_t / Σ_maj` stays in
//! `[0, 1]` regardless of which URR sample lands. Phase 2a–2b put
//! the per-material plumbing in place; this caveat (which used to
//! say "validate-reject in Phase 1") was lifted in Phase 2c.

use crate::{interpolate_linear, Material};

/// Energy-dependent upper bound on the local total cross section.
///
/// Implementations:
/// - [`GlobalMajorant`] -- constant over space (Phase 1). `material_id`
///   is ignored; the same per-energy bound applies everywhere.
/// - [`LocalMajorant`] -- one majorant per material (Phase 2). Uses
///   `material_id` to select the right per-material bound. Vacuum
///   (no material at this position, `material_id = None`) returns 0,
///   which the Woodcock loop interprets as "infinite free flight";
///   the particle moves to the next geometry boundary without
///   sampling a collision.
///
/// The trait is shaped so the *caller* identifies the current material
/// (via the cell lookup it already does), not the impl. This keeps
/// yamc-materials free of any geometry dependency.
pub trait Majorant: Send + Sync {
    /// Returns `Σ_maj(material, E)` -- a value guaranteed `≥ Σ_t(material, E)`
    /// when the particle is in that material. Used by Woodcock
    /// free-flight sampling (`-ln(ξ) / Σ_maj`) and the rejection-loop
    /// probability check (`p_real = Σ_t / Σ_maj`).
    ///
    /// `material_id = None` means "no material at this position"
    /// (vacuum): implementations must return `0.0`, which the transport
    /// loop interprets as infinite free flight.
    fn sigma_max(&self, material_id: Option<u32>, energy: f64) -> f64;
}

/// Phase 1 majorant: per-energy maximum `Σ_t` across every material in
/// the model. Independent of position.
///
/// Stored as a sorted energy grid plus parallel majorant values. Lookup
/// is binary-search + linear interpolation between adjacent grid points.
#[derive(Debug, Clone)]
pub struct GlobalMajorant {
    /// Unified energy grid (sorted, ascending) of all material grids.
    energies: Vec<f64>,
    /// Per-energy majorant: `Σ_maj[i] = max over materials of Σ_t(energies[i])`.
    sigma_max: Vec<f64>,
}

impl GlobalMajorant {
    /// Build a global majorant by taking, at every unique energy in any
    /// material's grid, the maximum of every material's smooth `Σ_t` at
    /// that energy.
    ///
    /// Materials must already have their macroscopic neutron cross
    /// sections prepared (i.e. `calculate_macroscopic_xs` has been called
    /// and `unified_energy_grid_neutron` is populated).
    pub fn new(materials: &[&Material]) -> Self {
        // Union of all material energy grids. BTreeSet keeps it sorted
        // and dedupes via the `to_bits` trick so f64 NaNs (which there
        // shouldn't be) don't blow up Ord.
        use std::collections::BTreeSet;
        let mut energy_bits: BTreeSet<u64> = BTreeSet::new();
        for m in materials {
            for &e in &m.unified_energy_grid_neutron {
                energy_bits.insert(e.to_bits());
            }
        }
        let energies: Vec<f64> = energy_bits.iter().map(|&b| f64::from_bits(b)).collect();

        // At every unified energy, take the per-material max of the
        // URR-aware Σ_t majorant -- equals smooth Σ_t outside any URR
        // range and the worst-case URR-sampled Σ_t inside one. This
        // is what lets Woodcock run correctly on URR-bearing materials
        // (Phase 2c); Phase 1's `lookup_xs_by_mt(1, e)` bounded only
        // smooth Σ_t and biased the rejection loop on URR samples.
        let sigma_max: Vec<f64> = energies
            .iter()
            .map(|&e| {
                materials
                    .iter()
                    .map(|m| m.total_xs_majorant(e))
                    .fold(0.0_f64, f64::max)
            })
            .collect();

        Self {
            energies,
            sigma_max,
        }
    }

    /// Linear-interpolate the majorant at the requested energy. Empty
    /// grid returns `0.0` (a vacuum-like "no bound" rather than the
    /// `NaN` that the shared helper would produce).
    fn interp(&self, energy: f64) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        interpolate_linear(&self.energies, &self.sigma_max, energy)
    }

    /// Number of energy points in the unified grid; exposed for tests.
    pub fn num_energy_points(&self) -> usize {
        self.energies.len()
    }
}

impl Majorant for GlobalMajorant {
    /// `material_id` is ignored: the global majorant is uniform over
    /// space. Vacuum (None) still returns the global bound -- safe
    /// (over-bounds), and consistent with Phase 1 behaviour where the
    /// rejection loop runs in every cell with material.
    #[inline]
    fn sigma_max(&self, _material_id: Option<u32>, energy: f64) -> f64 {
        self.interp(energy)
    }
}

/// Global majorant for **photon** total cross section, used by Woodcock
/// tracking when photons are transported (photon sources, coupled
/// neutron->photon production, or D1S decay photons).
///
/// Built exactly like [`GlobalMajorant`] but over the photon physics:
/// the energy grid is the union of every element's photon grid (stored
/// natively as `ln(E)`; converted to linear `E` here), and the bound at
/// each grid energy is the per-material maximum of the macroscopic
/// photon total cross section [`Material::calculate_photon_xs`]. The
/// result is constant over space, so `material_id` is ignored at lookup.
///
/// Photon total `Σ_t` is smooth between absorption edges and the union
/// grid carries those edges as grid points, so linear interpolation of
/// the per-grid-point maxima bounds `Σ_t` to the same accuracy the
/// neutron [`GlobalMajorant`] achieves. The Woodcock rejection step
/// clamps `p_real = Σ_t / Σ_maj` to `[0, 1]`, matching the neutron path.
#[derive(Debug, Clone)]
pub struct GlobalPhotonMajorant {
    energies: Vec<f64>,
    sigma_max: Vec<f64>,
}

impl GlobalPhotonMajorant {
    /// Build a global photon majorant. Materials must already have their
    /// photon data initialised (`init_photon_data`) so `calculate_photon_xs`
    /// returns non-zero cross sections; materials without photon data
    /// contribute nothing (their `cached_elements` is empty).
    pub fn new(materials: &[&Material]) -> Self {
        // Union of every element's photon energy grid. The grid is stored
        // as ln(E); convert to linear E before deduping so the majorant
        // lookup (which takes linear energy) interpolates correctly.
        use std::collections::BTreeSet;
        let mut energy_bits: BTreeSet<u64> = BTreeSet::new();
        for m in materials {
            for (_name, element) in &m.cached_elements {
                for &ln_e in &element.energy {
                    energy_bits.insert(ln_e.exp().to_bits());
                }
            }
        }
        let energies: Vec<f64> = energy_bits.iter().map(|&b| f64::from_bits(b)).collect();

        let sigma_max: Vec<f64> = energies
            .iter()
            .map(|&e| {
                materials
                    .iter()
                    .map(|m| m.calculate_photon_xs(e).total)
                    .fold(0.0_f64, f64::max)
            })
            .collect();

        Self {
            energies,
            sigma_max,
        }
    }

    /// Linear-interpolate the majorant at the requested energy. Empty
    /// grid (no photon data on any material) returns `0.0`.
    fn interp(&self, energy: f64) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        interpolate_linear(&self.energies, &self.sigma_max, energy)
    }

    /// Number of energy points in the unified photon grid; exposed for tests.
    pub fn num_energy_points(&self) -> usize {
        self.energies.len()
    }
}

impl Majorant for GlobalPhotonMajorant {
    /// `material_id` is ignored: the global photon majorant is uniform
    /// over space, just like [`GlobalMajorant`].
    #[inline]
    fn sigma_max(&self, _material_id: Option<u32>, energy: f64) -> f64 {
        self.interp(energy)
    }
}

/// Phase 2 majorant: one bound per material, looked up by `material_id`.
///
/// Each entry is itself a [`GlobalMajorant`] built over a single
/// material -- so its "global max" is the per-energy maximum of that
/// one material's `Σ_t`. That's the tightest single-material bound,
/// and per-material lookup means a vacuum (no material) or a sparse
/// region (very small `Σ_t`) doesn't pay the rejection-loop cost of a
/// dense material's majorant.
///
/// Vacuum is represented by `material_id = None`, which returns 0 from
/// [`Self::sigma_max`]. The Woodcock loop interprets `Σ_maj = 0` as
/// "infinite free flight": the particle moves to the next boundary
/// without sampling a collision -- exactly the right behaviour in
/// vacuum.
///
/// Unknown `material_id` (a value that was not in the materials slice
/// at construction) also returns 0. This is defensive: a caller that
/// passes a stale material_id gets a no-rejection step rather than a
/// panic or biased result.
#[derive(Debug, Clone, Default)]
pub struct LocalMajorant {
    per_material: std::collections::HashMap<u32, GlobalMajorant>,
}

impl LocalMajorant {
    /// Build a per-material majorant from the materials in the model.
    /// Materials without a `material_id` are skipped (they get the
    /// vacuum behaviour at lookup time -- conservative but safe).
    /// Materials must already have their macroscopic neutron cross
    /// sections prepared.
    pub fn new(materials: &[&Material]) -> Self {
        let mut per_material = std::collections::HashMap::new();
        for m in materials {
            let Some(id) = m.get_material_id() else {
                continue;
            };
            // A single-material `GlobalMajorant` *is* a per-material
            // majorant. Reuse the existing construction.
            let single = GlobalMajorant::new(std::slice::from_ref(m));
            per_material.insert(id, single);
        }
        Self { per_material }
    }

    /// Number of distinct materials covered. Exposed for tests.
    pub fn num_materials(&self) -> usize {
        self.per_material.len()
    }
}

impl Majorant for LocalMajorant {
    #[inline]
    fn sigma_max(&self, material_id: Option<u32>, energy: f64) -> f64 {
        match material_id {
            Some(id) => self
                .per_material
                .get(&id)
                .map(|m| m.interp(energy))
                .unwrap_or(0.0),
            None => 0.0,
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic majorant from a small hand-built grid. Verifies the
    /// interpolation + max semantics without needing nuclear data.
    #[test]
    fn interp_picks_max_at_grid_points() {
        let m = GlobalMajorant {
            energies: vec![1.0e6, 2.0e6, 3.0e6],
            sigma_max: vec![0.5, 0.7, 0.3],
        };
        assert!((m.sigma_max(None, 1.0e6) - 0.5).abs() < 1e-12);
        assert!((m.sigma_max(None, 2.0e6) - 0.7).abs() < 1e-12);
        assert!((m.sigma_max(None, 3.0e6) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn interp_linear_between_grid_points() {
        let m = GlobalMajorant {
            energies: vec![1.0, 2.0],
            sigma_max: vec![0.0, 1.0],
        };
        assert!((m.sigma_max(None, 1.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn interp_clamps_outside_grid() {
        let m = GlobalMajorant {
            energies: vec![1.0, 2.0],
            sigma_max: vec![0.3, 0.7],
        };
        assert_eq!(m.sigma_max(None, 0.1), 0.3);
        assert_eq!(m.sigma_max(None, 99.0), 0.7);
    }

    #[test]
    fn empty_majorant_returns_zero() {
        let m = GlobalMajorant {
            energies: vec![],
            sigma_max: vec![],
        };
        assert_eq!(m.sigma_max(None, 1.0e6), 0.0);
    }

    /// Build a `LocalMajorant` from hand-written per-material data without
    /// going through `Material` (which requires nuclear data on disk).
    /// Lets the structural tests stay self-contained.
    fn local_with_two_synthetic_materials() -> LocalMajorant {
        let mut per_material = std::collections::HashMap::new();
        per_material.insert(
            1,
            GlobalMajorant {
                energies: vec![1.0e6, 2.0e6],
                sigma_max: vec![0.1, 0.2],
            },
        );
        per_material.insert(
            2,
            GlobalMajorant {
                energies: vec![1.0e6, 2.0e6],
                sigma_max: vec![0.7, 0.9],
            },
        );
        LocalMajorant { per_material }
    }

    #[test]
    fn local_majorant_returns_per_material_value() {
        let m = local_with_two_synthetic_materials();
        // Material 1 has 0.1 at 1e6; material 2 has 0.7.
        assert!((m.sigma_max(Some(1), 1.0e6) - 0.1).abs() < 1e-12);
        assert!((m.sigma_max(Some(2), 1.0e6) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn local_majorant_interpolates_per_material() {
        let m = local_with_two_synthetic_materials();
        // Mid-grid of material 1: linear between 0.1 and 0.2 → 0.15.
        assert!((m.sigma_max(Some(1), 1.5e6) - 0.15).abs() < 1e-12);
        // Mid-grid of material 2: linear between 0.7 and 0.9 → 0.8.
        assert!((m.sigma_max(Some(2), 1.5e6) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn local_majorant_vacuum_returns_zero() {
        let m = local_with_two_synthetic_materials();
        // material_id = None means vacuum (no material at this position).
        // Woodcock interprets sigma_max = 0 as infinite free flight, which
        // is exactly what we want in vacuum.
        assert_eq!(m.sigma_max(None, 1.0e6), 0.0);
        assert_eq!(m.sigma_max(None, 1.5e6), 0.0);
    }

    #[test]
    fn local_majorant_unknown_material_returns_zero() {
        let m = local_with_two_synthetic_materials();
        // Defensive: a caller passing a stale material_id (e.g. one not
        // present at construction) gets 0 rather than a panic. The
        // Woodcock loop then takes a no-rejection step.
        assert_eq!(m.sigma_max(Some(99), 1.0e6), 0.0);
    }

    #[test]
    fn local_majorant_strictly_below_or_equal_to_global() {
        // The conceptual contract: each per-material entry's max at
        // any energy is ≤ the global max across all materials at that
        // energy. So a global majorant *over-bounds* a local one.
        let local = local_with_two_synthetic_materials();
        let combined_global = GlobalMajorant {
            energies: vec![1.0e6, 2.0e6],
            sigma_max: vec![0.7, 0.9], // material 2 dominates
        };
        for e in [1.0e6, 1.3e6, 1.7e6, 2.0e6] {
            for id in [1u32, 2u32] {
                let local_v = local.sigma_max(Some(id), e);
                let global_v = combined_global.sigma_max(None, e);
                assert!(
                    local_v <= global_v + 1e-12,
                    "local Σ_maj at material {id}, E={e:e} ({local_v:e}) \
                     should not exceed global ({global_v:e})"
                );
            }
        }
    }

    #[test]
    fn local_majorant_num_materials_counts_entries() {
        let m = local_with_two_synthetic_materials();
        assert_eq!(m.num_materials(), 2);
        let empty = LocalMajorant::default();
        assert_eq!(empty.num_materials(), 0);
    }
}
