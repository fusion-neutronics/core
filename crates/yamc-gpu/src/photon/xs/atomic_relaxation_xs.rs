//! GPU-side atomic relaxation tables.
//!
//! Plumbing for the photon kernel's photoelectric path: packs
//! per-element subshell PE cross sections (interpolated onto the
//! common log-energy grid) and atomic-transition tables
//! (fluorescence + Auger / Coster-Kronig) into flat-buffer arrays
//! the kernel can index without dynamic allocation.
//!
//! Mirrors the CPU's `PhotonInteraction::sample_photoelectric_subshell`
//! and `atomic_relaxation` paths in `yamc-element/src/photon.rs`.
//!
//! Scope: ONE slab per element (task #72), concatenated material-major --
//! the kernel's per-collision element selection indexes whichever element
//! the photon struck, mirroring CPU `Material::sample_element`. (Same
//! per-element slab convention as the Rayleigh / Doppler / IFF / pair
//! extractors.)

use std::sync::Arc;

use yamc_element::photon::PhotonInteraction;

/// Maximum number of atomic-relaxation subshells per material stored
/// on GPU. Fe has 7, U has ~15, so 16 is comfortable headroom.
pub const MAX_AR_SHELLS: usize = 16;

/// Maximum number of atomic transitions per subshell. Typical inner
/// shells of heavy elements have 20-30 transitions; 32 covers them.
pub const MAX_AR_TRANS: usize = 32;

/// Per-ELEMENT atomic-relaxation pack ready for upload to the GPU photon
/// kernel (task #72). Slab dimension = total elements across materials,
/// element-major within a material and concatenated material-major.
///
/// Layout: per element-slab `e`, scalar fields sit at slot `e`. Per-shell
/// fields at `e * MAX_AR_SHELLS + s`. Per-(shell, transition) fields
/// at `(e * MAX_AR_SHELLS + s) * MAX_AR_TRANS + t`. Per-(grid, shell)
/// PE XS at `(e * n_grid + i_grid) * MAX_AR_SHELLS + s`. `n_shells[e]`
/// is the count of valid shells; padding past that holds zeros.
/// `has_data[e] = 0` flags element slabs with no atomic-relaxation tables
/// -- the kernel falls back to K-shell-guess photoelectric with no
/// cascade for those.
#[derive(Clone, Debug)]
pub struct GpuAtomicRelaxation {
    /// Per-material flag: `1` if atomic-relaxation data is populated.
    pub has_data: Vec<u32>,
    /// Per-material count of atomic-relaxation subshells.
    pub n_shells: Vec<u32>,
    /// Per-material per-shell binding energy in eV, flat
    /// `[n_materials × MAX_AR_SHELLS]`. Used for the photoelectron
    /// KE calculation after subshell sampling.
    pub binding_energy: Vec<f64>,
    /// Per-material per-(grid, shell) PE cross section in log-form,
    /// interpolated onto the common log-energy grid:
    /// `[n_materials × n_grid × MAX_AR_SHELLS]`. Index by
    /// `(m * n_grid + i_grid) * MAX_AR_SHELLS + s`. Stores `-900.0`
    /// when below the shell's threshold (matches the CPU
    /// `ElectronSubshell::threshold` semantics).
    pub pe_subshell_xs_log: Vec<f64>,
    /// Per-material per-shell transition count, flat
    /// `[n_materials × MAX_AR_SHELLS]`.
    pub n_trans: Vec<u32>,
    /// Per-(material, shell, transition) primary-vacancy subshell
    /// index (into the AR shell list), or `u32::MAX` (`0xFFFFFFFF`)
    /// if untracked. `[n_materials × MAX_AR_SHELLS × MAX_AR_TRANS]`.
    /// Encoded as `u32` rather than `i32` because the kernel only
    /// declares unsigned-int storage buffers; the CPU's `-1` sentinel
    /// maps to `u32::MAX`.
    pub trans_primary: Vec<u32>,
    /// Per-(material, shell, transition) secondary-vacancy subshell
    /// index, or `u32::MAX` for radiative transitions (fluorescence --
    /// no secondary vacancy). `[n_materials × MAX_AR_SHELLS × MAX_AR_TRANS]`.
    pub trans_secondary: Vec<u32>,
    /// Per-(material, shell, transition) transition energy in eV.
    /// `[n_materials × MAX_AR_SHELLS × MAX_AR_TRANS]`.
    pub trans_energy: Vec<f64>,
    /// Per-(material, shell, transition) cumulative probability (CDF).
    /// `[n_materials × MAX_AR_SHELLS × MAX_AR_TRANS]`.
    pub trans_cum_prob: Vec<f64>,
}

impl GpuAtomicRelaxation {
    /// A degenerate empty pack -- same shape as the populated case so
    /// the kernel launcher can pass uniform arguments -- but
    /// `has_data[slab] = 0` everywhere so the kernel falls back to the
    /// no-cascade photoelectric path. `n_slab` is the element-slab count
    /// (task #72).
    pub fn empty_for_slabs(n_slab: usize, n_grid: usize) -> Self {
        let n_slab = n_slab.max(1);
        let n_grid = n_grid.max(1);
        Self {
            has_data: vec![0u32; n_slab],
            n_shells: vec![0u32; n_slab],
            binding_energy: vec![0.0_f64; n_slab * MAX_AR_SHELLS],
            pe_subshell_xs_log: vec![-900.0_f64; n_slab * n_grid * MAX_AR_SHELLS],
            n_trans: vec![0u32; n_slab * MAX_AR_SHELLS],
            trans_primary: vec![u32::MAX; n_slab * MAX_AR_SHELLS * MAX_AR_TRANS],
            trans_secondary: vec![u32::MAX; n_slab * MAX_AR_SHELLS * MAX_AR_TRANS],
            trans_energy: vec![0.0_f64; n_slab * MAX_AR_SHELLS * MAX_AR_TRANS],
            trans_cum_prob: vec![0.0_f64; n_slab * MAX_AR_SHELLS * MAX_AR_TRANS],
        }
    }
}

/// Pack per-ELEMENT atomic-relaxation tables (task #72).
///
/// `materials` is the per-material `(name, element, density)` triples
/// list (same shape as the other photon extractors). `log_energy_grid`
/// is the common ln-energy grid the kernel uses for total / partial
/// photon XS lookups -- per-subshell PE XS is interpolated onto it so
/// the kernel can reuse the same `(idx_lo, idx_hi, frac)` it already
/// computes for the macroscopic XS slate.
///
/// Emits one slab per element (concatenated material-major); the kernel's
/// per-collision element selection indexes whichever element the photon
/// struck, mirroring CPU `Material::sample_element` -> that element's
/// subshells / relaxation cascade.
pub fn extract_atomic_relaxation_for_gpu(
    materials: &[Vec<(String, Arc<PhotonInteraction>, f64)>],
    log_energy_grid: &[f64],
) -> GpuAtomicRelaxation {
    let n_grid = log_energy_grid.len();
    let n_slab: usize = materials.iter().map(|m| m.len()).sum();
    if n_slab == 0 || n_grid == 0 {
        return GpuAtomicRelaxation::empty_for_slabs(n_slab, n_grid);
    }

    let mut pack = GpuAtomicRelaxation::empty_for_slabs(n_slab, n_grid);

    // One slab per element (task #72): the per-collision element selection
    // indexes whichever element the photon struck (mirrors CPU
    // `Material::sample_element` -> that element's subshells / cascade).
    let mut slab = 0usize;
    for mat_elements in materials {
        for (_, el, _) in mat_elements {
            let m = slab;
            slab += 1;
            if el.shells.is_empty() {
                continue;
            }

            let n_shells = el.shells.len().min(MAX_AR_SHELLS);
            let shell_off = m * MAX_AR_SHELLS;

            // Per-shell scalar + transition tables.
            for s in 0..n_shells {
                let shell = &el.shells[s];
                pack.binding_energy[shell_off + s] = shell.binding_energy;

                let n_t = shell.transitions.len().min(MAX_AR_TRANS);
                pack.n_trans[shell_off + s] = n_t as u32;
                let t_off = (shell_off + s) * MAX_AR_TRANS;
                for t in 0..n_t {
                    let tr = &shell.transitions[t];
                    pack.trans_primary[t_off + t] = if tr.primary_subshell < 0 {
                        u32::MAX
                    } else {
                        tr.primary_subshell as u32
                    };
                    pack.trans_secondary[t_off + t] = if tr.secondary_subshell < 0 {
                        u32::MAX
                    } else {
                        tr.secondary_subshell as u32
                    };
                    pack.trans_energy[t_off + t] = tr.energy;
                    pack.trans_cum_prob[t_off + t] = tr.probability;
                }
            }

            // Per-(grid, shell) PE XS. `el.cross_sections[i_grid][i_shell]`
            // is the CPU's table -- values are `ln(σ)` at the element's
            // own energy grid, or 0.0 below the shell's threshold. The
            // element's grid aligns with the common `log_energy_grid` (the
            // master grid lifted by `extract_photon_material_xs`); mismatched
            // grids fall back to nearest-index lookup.
            let el_grid = &el.energy;
            let el_xs_2d = &el.cross_sections;
            if el_grid.is_empty() || el_xs_2d.is_empty() {
                continue;
            }
            for (i_grid, &log_e) in log_energy_grid.iter().enumerate().take(n_grid) {
                // Locate `log_e` on the element's own grid.
                let mut lo: usize = 0;
                let mut hi: usize = el_grid.len();
                while lo + 1 < hi {
                    let mid = (lo + hi) / 2;
                    if el_grid[mid] <= log_e {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let i_el = lo.min(el_grid.len() - 1);
                let i_el_next = (i_el + 1).min(el_grid.len() - 1);
                let denom = if i_el_next > i_el {
                    el_grid[i_el_next] - el_grid[i_el]
                } else {
                    0.0
                };
                let f = if denom > 1e-30 {
                    ((log_e - el_grid[i_el]) / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let row_lo = &el_xs_2d[i_el];
                let row_hi = if i_el_next < el_xs_2d.len() {
                    &el_xs_2d[i_el_next]
                } else {
                    row_lo
                };
                let row_off = (m * n_grid + i_grid) * MAX_AR_SHELLS;
                for s in 0..n_shells {
                    // CPU convention: `cross_sections[i_grid][i_shell] = 0.0`
                    // below the shell's threshold. Preserve that sentinel
                    // so the kernel can detect "below threshold" with a
                    // direct == 0.0 check (mirrors the CPU sampler).
                    let xs_lo = row_lo.get(s).copied().unwrap_or(0.0);
                    let xs_hi = row_hi.get(s).copied().unwrap_or(0.0);
                    let value = if xs_lo == 0.0 && xs_hi == 0.0 {
                        0.0
                    } else if xs_lo == 0.0 {
                        xs_hi
                    } else if xs_hi == 0.0 {
                        xs_lo
                    } else {
                        xs_lo + (xs_hi - xs_lo) * f
                    };
                    pack.pe_subshell_xs_log[row_off + s] = value;
                }
            }

            pack.n_shells[m] = n_shells as u32;
            pack.has_data[m] = if el.has_atomic_relaxation { 1u32 } else { 0u32 };
        }
    }

    pack
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pack_skips_sampling() {
        let pack = GpuAtomicRelaxation::empty_for_slabs(2, 4);
        assert_eq!(pack.has_data, vec![0u32, 0u32]);
        assert_eq!(pack.n_shells, vec![0u32, 0u32]);
        assert_eq!(pack.binding_energy.len(), 2 * MAX_AR_SHELLS);
        assert_eq!(pack.pe_subshell_xs_log.len(), 2 * 4 * MAX_AR_SHELLS);
        assert_eq!(pack.trans_primary.len(), 2 * MAX_AR_SHELLS * MAX_AR_TRANS);
    }

    #[test]
    fn extract_empty_inputs_returns_empty_pack() {
        let pack = extract_atomic_relaxation_for_gpu(&[], &[1.0, 2.0]);
        assert_eq!(pack.has_data, vec![0u32]);
        let pack = extract_atomic_relaxation_for_gpu(&[vec![]], &[]);
        assert_eq!(pack.has_data, vec![0u32]);
    }
}
