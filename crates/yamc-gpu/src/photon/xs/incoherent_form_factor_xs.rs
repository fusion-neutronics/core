//! GPU-side incoherent (bound-electron Compton) scattering-function
//! table.
//!
//! Plumbing for the bound-electron form-factor rejection sampled in
//! CPU `compton_scatter`:
//!
//! ```text
//! sample (α_out, μ) from free-electron Klein-Nishina
//! x = (m_e/hc) · α · sqrt((1-μ)/2)              (Hubbell momentum)
//! accept if random < S(x, Z) / S(x_max, Z)
//! else resample
//! ```
//!
//! The kernel reads `S(x)` from a per-material tabulated function
//! (Tabulated1D-style). One dominant-element pack per material,
//! matching the Rayleigh / Doppler extractor convention.
//!
//! Scope: data plumbing only. Kernel-side rejection sampler is
//! added in the same slice.

use std::sync::Arc;

use yamc_element::photon::PhotonInteraction;
use yamc_nuclide::reaction_product::Tabulated1D;

/// Maximum number of (x, S(x)) tabulation points stored on GPU.
/// ENDF/B-VIII.0+ data typically uses 30-50 momentum points;
/// 64 is the next power of two with headroom.
pub const MAX_INCOHERENT_FF: usize = 64;

/// Per-ELEMENT incoherent scattering-function table ready for upload to the
/// GPU photon kernel. Slab dimension = total elements across materials,
/// element-major within a material and concatenated material-major (the order
/// `PhotonElementSelectInputs::mat_elem_meta` indexes), so the kernel reads
/// the per-collision-SELECTED element's `S(x, Z)` (task #72), not a single
/// dominant element's (#79).
#[derive(Clone, Debug)]
pub struct GpuIncoherentFormFactor {
    /// Per-element `x` (momentum transfer) grid, flat
    /// `[n_slab × MAX_INCOHERENT_FF]`. Zero-padded past `n_points[slab]`.
    pub x: Vec<f64>,
    /// Per-element `S(x, Z)` values, same shape as `x`.
    pub s: Vec<f64>,
    /// Per-element count of valid `(x, S)` points.
    pub n_points: Vec<u32>,
    /// Per-element flag: `1` when tables are populated, `0` when the
    /// kernel should skip form-factor rejection (fall back to pure
    /// free-electron Kahn).
    pub has_data: Vec<u32>,
}

impl GpuIncoherentFormFactor {
    /// Degenerate single-slab pack for when form-factor data is
    /// unavailable. `n_slab` slabs (buffers stay non-zero -- cubecl rejects
    /// empty arrays); the kernel reads `has_data[slab]` to decide whether
    /// to apply the rejection.
    pub fn empty_for_slabs(n_slab: usize) -> Self {
        let n_slab = n_slab.max(1);
        Self {
            x: vec![0.0; n_slab * MAX_INCOHERENT_FF],
            s: vec![0.0; n_slab * MAX_INCOHERENT_FF],
            n_points: vec![0u32; n_slab],
            has_data: vec![0u32; n_slab],
        }
    }
}

/// Pack per-ELEMENT incoherent form-factor tables for the GPU photon kernel.
///
/// One slab per element (task #72): copy each element's
/// `incoherent_form_factor` `(x, y)` Tabulated1D into its slab. If the source
/// table has more than `MAX_INCOHERENT_FF` points, subsample uniformly in
/// index space. The per-collision element selection then indexes the slab of
/// whichever element the photon struck (mirrors CPU
/// `Material::sample_element`).
pub fn extract_incoherent_form_factor_for_gpu(
    materials: &[Vec<(String, Arc<PhotonInteraction>, f64)>],
) -> GpuIncoherentFormFactor {
    if materials.is_empty() {
        return GpuIncoherentFormFactor::empty_for_slabs(0);
    }

    let n_slab: usize = materials.iter().map(|m| m.len()).sum();
    let mut pack = GpuIncoherentFormFactor::empty_for_slabs(n_slab);
    let mut slab = 0usize;

    for mat_elements in materials {
        for (_, el, _) in mat_elements {
            let cur = slab;
            slab += 1;
            let Tabulated1D::Tabulated1D { x, y, .. } = &el.incoherent_form_factor;
            let n_src = x.len().min(y.len());
            if n_src < 2 {
                continue;
            }
            let n = n_src.min(MAX_INCOHERENT_FF);
            let stride = if n_src > MAX_INCOHERENT_FF {
                (n_src - 1) as f64 / (MAX_INCOHERENT_FF - 1) as f64
            } else {
                1.0
            };
            let off = cur * MAX_INCOHERENT_FF;
            for j in 0..n {
                let src_j = if n_src > MAX_INCOHERENT_FF {
                    (j as f64 * stride).round() as usize
                } else {
                    j
                };
                pack.x[off + j] = x[src_j.min(n_src - 1)];
                pack.s[off + j] = y[src_j.min(n_src - 1)];
            }
            pack.n_points[cur] = n as u32;
            pack.has_data[cur] = 1u32;
        }
    }

    pack
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pack_for_zero_slabs() {
        let pack = GpuIncoherentFormFactor::empty_for_slabs(0);
        assert_eq!(pack.has_data, vec![0u32]);
        assert_eq!(pack.n_points, vec![0u32]);
        assert_eq!(pack.x.len(), MAX_INCOHERENT_FF);
    }

    #[test]
    fn extract_empty_inputs() {
        let pack = extract_incoherent_form_factor_for_gpu(&[]);
        assert_eq!(pack.has_data.len(), 1);
        assert_eq!(pack.has_data[0], 0);
    }
}
