//! GPU-side Compton Doppler-broadening tables.
//!
//! Plumbing: packs per-element Compton profile data (J(p_z) curves
//! per atomic shell, plus shell binding energies and occupancy
//! weights) into flat-buffer arrays the photon kernel can index.
//! Mirror of `yamc-element/src/photon.rs`'s `PhotonInteraction`
//! fields:
//!   - `binding_energy: Vec<f64>` (per-shell binding energy, eV)
//!   - `electron_pdf:   Vec<f64>` (per-shell occupancy weight, sums
//!     to ≈ 1.0; used to sample which shell the photon interacts with)
//!   - `profile_pdf: Vec<Vec<f64>>` (J(p_z) PDF per shell, indexed by
//!     the shared `compton_profile_pz` grid)
//!   - `profile_cdf: Vec<Vec<f64>>` (corresponding trapezoidal CDF)
//!
//! Scope of this module: data plumbing + dominant-element extraction
//! (parallel pattern to the Rayleigh form-factor in `photon_xs.rs`).
//! The kernel-side Doppler sampler that consumes this data is a
//! follow-up commit in the same slice.

use std::sync::Arc;

use yamc_element::photon::PhotonInteraction;

/// Maximum number of Compton shells per material stored on GPU.
/// Heavy elements (W, U, Pu, …) max out around 20-25 subshells; 32
/// is generous headroom with uniform-in-index subsampling if any
/// element exceeds it.
pub const MAX_COMPTON_SHELLS: usize = 32;

/// Maximum number of `p_z` grid points stored on GPU. The shared
/// `compton_profile_pz` grid is ~31 points across all elements in
/// ENDF/B-VIII.0+ data; 64 is the next power of two with headroom.
pub const MAX_COMPTON_PZ: usize = 64;

/// Maximum number of atomic-relaxation subshells a single Compton (n, l) shell
/// maps to. A Compton shell splits into at most its j = l +/- 1/2 doublet, so 2.
pub const MAX_COMPTON_RELAX: usize = 2;

/// Per-ELEMENT Compton Doppler-broadening tables ready for upload to the GPU
/// photon kernel (task #72). Slab dimension = total elements across materials,
/// element-major within a material and concatenated material-major.
///
/// Layout: per element-slab `e`, the per-shell scalar fields sit at
/// `slot[e * MAX_COMPTON_SHELLS + s]`, and the per-(shell, pz)
/// tables sit at `slot[(e * MAX_COMPTON_SHELLS + s) * MAX_COMPTON_PZ + i]`.
/// `n_shells[e]` is the count of valid shells; padding past that
/// holds zeros. `has_data[e] = 0` flags slabs with no Doppler
/// tables -- the kernel will fall back to free-electron Klein-Nishina
/// for those.
#[derive(Clone, Debug)]
pub struct GpuComptonDoppler {
    /// Shared `p_z` grid, length `n_pz`. Same units as
    /// `yamc-element`'s `compton_profile_pz()` (atomic units / m_e c).
    pub pz_grid: Vec<f64>,
    pub n_pz: u32,

    /// Per-material per-shell occupancy weight, flat
    /// `[n_materials × MAX_COMPTON_SHELLS]`. Used for shell sampling
    /// (cumulative comparison on `xi`).
    pub electron_pdf: Vec<f64>,
    /// Per-material per-shell binding energy in eV, flat
    /// `[n_materials × MAX_COMPTON_SHELLS]`.
    pub binding_energy: Vec<f64>,

    /// Per-material per-shell Compton profile `J(p_z)`, flat
    /// `[n_materials × MAX_COMPTON_SHELLS × MAX_COMPTON_PZ]`.
    pub profile_pdf: Vec<f64>,
    /// Per-material per-shell trapezoidal CDF of the profile, flat
    /// `[n_materials × MAX_COMPTON_SHELLS × MAX_COMPTON_PZ]`.
    pub profile_cdf: Vec<f64>,

    /// Number of valid Compton shells per material, length
    /// `n_materials`. Zero means "no Doppler data" -- combined with
    /// `has_data == 0` for explicit fall-back.
    pub n_shells: Vec<u32>,
    /// Per-material flag: `1` when Doppler tables are populated,
    /// `0` when the kernel should skip Doppler and use free
    /// Klein-Nishina E_out.
    pub has_data: Vec<u32>,

    /// Compton-profile shell -> constituent atomic-relaxation subshell indices,
    /// flat `[n_slab × MAX_COMPTON_SHELLS × MAX_COMPTON_RELAX]`. A Compton (n, l)
    /// shell maps to one or two (n, l, j) subshells; the kernel picks one weighted
    /// by `subshell_w0` and relaxes it (banking the fluorescence). Unused slots
    /// hold `u32::MAX`, consistent with the `trans_primary` sentinel in
    /// `atomic_relaxation_xs.rs`. Mirror of the CPU `compton_relax_map`.
    pub subshell_idx: Vec<u32>,
    /// Occupancy weight of constituent 0 per Compton shell, flat
    /// `[n_slab × MAX_COMPTON_SHELLS]`. The kernel picks slot 0 when its draw is
    /// below this, slot 1 otherwise (single-constituent shells store 1.0).
    pub subshell_w0: Vec<f64>,
    /// Number of constituent relaxation subshells per Compton shell (0, 1, or 2),
    /// flat `[n_slab × MAX_COMPTON_SHELLS]`. 0 = no relaxation counterpart.
    pub subshell_cnt: Vec<u32>,
}

impl GpuComptonDoppler {
    /// A degenerate single-shell pack for when Doppler is disabled or
    /// no element data is available. Same shape as the populated case
    /// (uniform argument shape on the kernel launcher) but
    /// `has_data[slab] = 0` so the kernel skips sampling. `n_slab` is the
    /// element-slab count (task #72).
    pub fn empty_for_slabs(n_slab: usize) -> Self {
        let n_slab = n_slab.max(1);
        let n_pz = 1u32;
        Self {
            pz_grid: vec![0.0],
            n_pz,
            electron_pdf: vec![0.0; n_slab * MAX_COMPTON_SHELLS],
            binding_energy: vec![0.0; n_slab * MAX_COMPTON_SHELLS],
            profile_pdf: vec![0.0; n_slab * MAX_COMPTON_SHELLS * MAX_COMPTON_PZ],
            profile_cdf: vec![0.0; n_slab * MAX_COMPTON_SHELLS * MAX_COMPTON_PZ],
            n_shells: vec![0u32; n_slab],
            has_data: vec![0u32; n_slab],
            subshell_idx: vec![u32::MAX; n_slab * MAX_COMPTON_SHELLS * MAX_COMPTON_RELAX],
            subshell_w0: vec![0.0; n_slab * MAX_COMPTON_SHELLS],
            subshell_cnt: vec![0u32; n_slab * MAX_COMPTON_SHELLS],
        }
    }
}

/// Pack per-material Compton Doppler tables into a `GpuComptonDoppler`.
///
/// `materials` is the per-material triples list (same shape as
/// `extract_photon_material_xs`'s input). Picks the element dominating
/// the incoherent macroscopic XS per material (the Doppler profile is
/// the energy-broadening companion of the incoherent channel; see
/// `dominant_element_for_reaction`) and copies its profile tables into
/// the per-material slab. Future improvement: weighted aggregation
/// across elements, but for the single-element materials in our
/// verification suite, dominant-only is exact.
pub fn extract_compton_doppler_for_gpu(
    materials: &[Vec<(String, Arc<PhotonInteraction>, f64)>],
    pz_grid: &[f64],
) -> GpuComptonDoppler {
    let n_slab: usize = materials.iter().map(|m| m.len()).sum();
    if pz_grid.is_empty() || materials.is_empty() {
        return GpuComptonDoppler::empty_for_slabs(n_slab);
    }

    let n_pz_src = pz_grid.len();
    let n_pz = n_pz_src.min(MAX_COMPTON_PZ);

    let mut electron_pdf = vec![0.0_f64; n_slab.max(1) * MAX_COMPTON_SHELLS];
    let mut binding_energy = vec![0.0_f64; n_slab.max(1) * MAX_COMPTON_SHELLS];
    let mut profile_pdf = vec![0.0_f64; n_slab.max(1) * MAX_COMPTON_SHELLS * MAX_COMPTON_PZ];
    let mut profile_cdf = vec![0.0_f64; n_slab.max(1) * MAX_COMPTON_SHELLS * MAX_COMPTON_PZ];
    let mut n_shells_v = vec![0u32; n_slab.max(1)];
    let mut has_data = vec![0u32; n_slab.max(1)];
    let mut subshell_idx = vec![u32::MAX; n_slab.max(1) * MAX_COMPTON_SHELLS * MAX_COMPTON_RELAX];
    let mut subshell_w0 = vec![0.0_f64; n_slab.max(1) * MAX_COMPTON_SHELLS];
    let mut subshell_cnt = vec![0u32; n_slab.max(1) * MAX_COMPTON_SHELLS];

    // One slab per element (task #72): the per-collision element selection
    // indexes whichever element the photon struck, mirroring CPU
    // `Material::sample_element` -> that element's Doppler profile.
    let mut slab = 0usize;
    for mat_elements in materials {
        for (_, el, _) in mat_elements {
            let cur = slab;
            slab += 1;
            if el.electron_pdf.is_empty() || el.binding_energy.is_empty() {
                continue;
            }

            let n_shells_src = el
                .electron_pdf
                .len()
                .min(el.binding_energy.len())
                .min(el.profile_pdf.len())
                .min(el.profile_cdf.len());
            let n_shells = n_shells_src.min(MAX_COMPTON_SHELLS);
            if n_shells == 0 {
                continue;
            }

            let shell_off = cur * MAX_COMPTON_SHELLS;
            for s in 0..n_shells {
                electron_pdf[shell_off + s] = el.electron_pdf[s];
                binding_energy[shell_off + s] = el.binding_energy[s];
                // Compton-profile shell `s` -> its constituent relaxation subshells
                // (occupancy weighted). CPU `compton_relax_map[s]` holds 0, 1, or 2
                // targets; an empty list means no relaxation counterpart so the
                // kernel skips relaxation (subshell_cnt stays 0).
                if let Some(targets) = el.compton_relax_map.get(s) {
                    let cnt = targets.len().min(MAX_COMPTON_RELAX);
                    subshell_cnt[shell_off + s] = cnt as u32;
                    let relax_off = (shell_off + s) * MAX_COMPTON_RELAX;
                    for (k, t) in targets.iter().take(cnt).enumerate() {
                        subshell_idx[relax_off + k] = t.shell_index as u32;
                    }
                    subshell_w0[shell_off + s] = targets.first().map(|t| t.weight).unwrap_or(0.0);
                }

                let prof_pdf = &el.profile_pdf[s];
                let prof_cdf = &el.profile_cdf[s];
                let n_pts = prof_pdf.len().min(prof_cdf.len()).min(n_pz);
                let table_off = (shell_off + s) * MAX_COMPTON_PZ;
                profile_pdf[table_off..table_off + n_pts].copy_from_slice(&prof_pdf[..n_pts]);
                profile_cdf[table_off..table_off + n_pts].copy_from_slice(&prof_cdf[..n_pts]);
            }
            n_shells_v[cur] = n_shells as u32;
            has_data[cur] = 1u32;
        }
    }

    let mut pz_out = vec![0.0_f64; n_pz];
    pz_out[..n_pz].copy_from_slice(&pz_grid[..n_pz]);

    GpuComptonDoppler {
        pz_grid: pz_out,
        n_pz: n_pz as u32,
        electron_pdf,
        binding_energy,
        profile_pdf,
        profile_cdf,
        n_shells: n_shells_v,
        has_data,
        subshell_idx,
        subshell_w0,
        subshell_cnt,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pack_skips_sampling() {
        let pack = GpuComptonDoppler::empty_for_slabs(2);
        assert_eq!(pack.has_data, vec![0u32, 0u32]);
        assert_eq!(pack.n_shells, vec![0u32, 0u32]);
        assert_eq!(pack.n_pz, 1);
        assert_eq!(pack.electron_pdf.len(), 2 * MAX_COMPTON_SHELLS);
    }

    #[test]
    fn extract_empty_inputs_returns_empty_pack() {
        let pack = extract_compton_doppler_for_gpu(&[], &[1.0, 2.0]);
        assert_eq!(pack.n_pz, 1);
        assert_eq!(pack.has_data.len(), 1);
        assert_eq!(pack.has_data[0], 0);

        let pack = extract_compton_doppler_for_gpu(&[vec![]], &[]);
        assert_eq!(pack.n_pz, 1);
    }
}
