//! GPU-side pair-production element constants.
//!
//! The CPU's `PhotonInteraction::pair_production` sampler is closed-
//! form (no per-element interpolation tables) -- it needs only the
//! atomic number Z of the absorbing atom and one Z-indexed scalar,
//! the reduced screening radius `r_z` from PENELOPE-2011.
//!
//! Three of the four per-element scalars in the sampler depend only
//! on Z (not on the incident photon energy), so the host precomputes
//! them:
//!   - `r_z`   -- reduced screening radius, looked up from
//!     `yamc_element::photon::REDUCED_SCREENING_RADII`
//!   - `a`     -- `Z / FINE_STRUCTURE`, the Born parameter
//!   - `c`     -- Coulomb correction, a closed-form polynomial in `a²`
//!
//! These get packed ONE slab per element (task #72), concatenated
//! material-major -- same per-element slab convention as the Rayleigh,
//! Doppler, IFF, and atomic-relaxation extractors. The kernel's
//! per-collision element selection indexes whichever element the photon
//! struck (mirrors CPU `Material::sample_element`).

use std::sync::Arc;

use yamc_element::photon::{PhotonInteraction, REDUCED_SCREENING_RADII};

/// Inverse fine-structure constant (dimensionless). Matches
/// `yamc_element::photon` and the kernel-side `FINE_STRUCTURE`
/// constant in `multi_cell_photon_transport.rs`.
pub const FINE_STRUCTURE: f64 = 137.035_999_084;

/// Per-ELEMENT pair-production scalars ready for the GPU kernel (task #72).
/// Slab dimension = total elements across materials, element-major within a
/// material and concatenated material-major, so the kernel indexes whichever
/// element the per-collision selection struck.
///
/// `has_data[slab] = 0` flags element slabs with no usable Z (no atoms, or
/// `Z >= 99` outside the screening table). The kernel falls back to dropping
/// the photon when this flag is clear.
#[derive(Clone, Debug)]
pub struct GpuPairProduction {
    /// Per-element flag: `1` if the element Z is in [1, 98] and
    /// pair-production data is meaningful. Length `n_slab`.
    pub has_data: Vec<u32>,
    /// Per-element reduced screening radius (PENELOPE-2011), eV-
    /// independent. Length `n_slab`.
    pub r_z: Vec<f64>,
    /// Per-element Born parameter `a = Z / FINE_STRUCTURE`. Length `n_slab`.
    pub a: Vec<f64>,
    /// Per-element Coulomb correction, closed-form polynomial in
    /// `a²` from the CPU sampler. Length `n_slab`.
    pub c: Vec<f64>,
}

impl GpuPairProduction {
    pub fn empty_for_slabs(n_slab: usize) -> Self {
        let n_slab = n_slab.max(1);
        Self {
            has_data: vec![0u32; n_slab],
            r_z: vec![0.0_f64; n_slab],
            a: vec![0.0_f64; n_slab],
            c: vec![0.0_f64; n_slab],
        }
    }
}

/// Closed-form Coulomb correction `c(a)` from the CPU
/// `pair_production` sampler. Polynomial in `a²`. Shared by the
/// host-side extractor (one source of truth).
fn coulomb_correction(a: f64) -> f64 {
    let a2 = a * a;
    a2 * (1.0 / (1.0 + a2)
        + 0.202_059
        + a2 * (-0.036_93
            + a2 * (0.008_35
                + a2 * (-0.002_01 + a2 * (0.000_49 + a2 * (-0.000_12 + a2 * 0.000_03))))))
}

/// Pack per-ELEMENT pair-production constants (task #72). `materials` is the
/// same per-material `(name, element, density)` triples shape the other photon
/// extractors take. Emits one slab per element (concatenated material-major);
/// an element with no atoms or Z out of range leaves `has_data = 0` for its
/// slab.
pub fn extract_pair_production_for_gpu(
    materials: &[Vec<(String, Arc<PhotonInteraction>, f64)>],
) -> GpuPairProduction {
    let n_slab: usize = materials.iter().map(|m| m.len()).sum();
    let mut pack = GpuPairProduction::empty_for_slabs(n_slab);

    // One slab per element (task #72): pair physics for whichever element the
    // per-collision selection struck (mirrors CPU `Material::sample_element`).
    let mut slab = 0usize;
    for mat_elements in materials {
        for (_, el, _) in mat_elements {
            let m = slab;
            slab += 1;
            let z_num = el.atomic_number;
            let z = z_num as usize;
            if z == 0 || z >= REDUCED_SCREENING_RADII.len() {
                continue;
            }
            let a = z_num as f64 / FINE_STRUCTURE;
            pack.r_z[m] = REDUCED_SCREENING_RADII[z];
            pack.a[m] = a;
            pack.c[m] = coulomb_correction(a);
            pack.has_data[m] = 1u32;
        }
    }

    pack
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pack_flags_no_data() {
        let pack = GpuPairProduction::empty_for_slabs(3);
        assert_eq!(pack.has_data, vec![0u32; 3]);
        assert_eq!(pack.r_z.len(), 3);
    }

    #[test]
    fn extract_empty_inputs() {
        let pack = extract_pair_production_for_gpu(&[]);
        assert_eq!(pack.has_data, vec![0u32]);
    }

    #[test]
    fn coulomb_correction_matches_cpu_for_fe() {
        // Cross-check the polynomial against the CPU pair_production
        // sampler's inline formulation. Fe: Z=26, a = 26/137.036.
        let a = 26.0 / FINE_STRUCTURE;
        let c = coulomb_correction(a);
        // Hand-evaluated reference from the polynomial: dominated by
        // the first two terms; for Fe a ≈ 0.1898 so a² ≈ 0.036, and
        // c ≈ 0.036 × (1/1.036 + 0.202 + small) ≈ 0.043. Just sanity-
        // check the order of magnitude rather than asserting bits.
        assert!(c > 0.03 && c < 0.06, "Fe Coulomb correction sanity: {c}");
    }
}
