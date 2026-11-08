//! GPU-side thick-target bremsstrahlung (TTB) tables.
//!
//! Plumbing: takes the per-material `Bremsstrahlung` tables that
//! `yamc_element::bremsstrahlung::init_bremsstrahlung` pre-computes on
//! the CPU and packs them into flat-buffer arrays the photon kernel can
//! index.
//!
//! Why this lives here (yamc-gpu) and not in yamc: the kernel needs to
//! read these tables on every Compton / photoelectric / pair event,
//! same shape as `GpuPhotonXs`. Keeping the GPU representation next to
//! the GPU kernel keeps shape changes local; the dispatch layer in
//! yamc translates from the CPU representation into the flat-buffer
//! form on every launch.
//!
//! Scope of this module: data plumbing only. No sampling logic yet --
//! the kernel-side TTB sampler lands in a follow-up that adds the
//! actual secondary-photon production + per-thread queue.

/// Flat-buffer per-material bremsstrahlung tables ready for upload to
/// the GPU photon kernel.
///
/// Layout for the lower-triangular PDF and CDF: for `n_e` grid points,
/// each material owns a `n_e * n_e` slab in row-major order, indexed
/// as `slab[j * n_e + i]` where `j` is the incident-electron-energy
/// index (row) and `i` is the photon-energy index (column). The CPU's
/// `BremsstrahlungData::pdf` is the same shape (`Vec<Vec<f64>>`) just
/// nested rather than flat. Lower-triangular means entries with
/// `i > j` are zero.
#[derive(Clone, Debug)]
pub struct GpuBremsstrahlung {
    /// Log-space TTB electron-energy grid (`ln(E)` per point), shared
    /// across all materials and both particle types. Length `n_e`.
    pub e_grid_log: Vec<f64>,
    /// Number of grid points (`= e_grid_log.len()`). Cached so the
    /// kernel can index without a `len()` call.
    pub n_e: u32,

    /// Per-material electron bremsstrahlung PDF, flat
    /// `[n_materials × n_e × n_e]` row-major (per slab, row = incident
    /// energy index, column = photon energy index).
    pub electron_pdf: Vec<f64>,
    /// Per-material electron bremsstrahlung CDF, same shape as
    /// `electron_pdf`.
    pub electron_cdf: Vec<f64>,
    /// Per-material electron photon-yield, flat `[n_materials × n_e]`.
    /// Stored in log-space (matches the CPU's `BremsstrahlungData::yield_`).
    pub electron_yield: Vec<f64>,

    /// Per-material positron bremsstrahlung PDF / CDF / yield (same
    /// shape as the electron arrays). Used by the kernel's pair-
    /// production secondaries -- distinct from electron tables because
    /// of the PENELOPE positron correction applied during CPU prep.
    pub positron_pdf: Vec<f64>,
    pub positron_cdf: Vec<f64>,
    pub positron_yield: Vec<f64>,

    /// Per-material flag: `1` when the material has populated TTB
    /// tables, `0` when init_bremsstrahlung was skipped (e.g. missing
    /// element data). The kernel will skip TTB sampling on materials
    /// with `has_data[m] == 0` rather than reading zeros and producing
    /// degenerate samples. Length `n_materials`.
    pub has_data: Vec<u32>,
}

impl GpuBremsstrahlung {
    /// A zero-material empty pack. Returned by the extractor when
    /// TTB is disabled on the model so the kernel launcher can still
    /// take a uniform argument shape without conditional binding.
    /// Caller must hand this to the kernel with `n_e = 1, n_materials = 1`
    /// padding (the kernel reads `has_data[m]` and skips sampling).
    pub fn empty_for_materials(n_materials: usize) -> Self {
        let n_materials = n_materials.max(1);
        // Single-point grid with a sentinel ln(E) -- the kernel never
        // reads it because `has_data[m] == 0` for every material. We
        // still need at least one grid point + one PDF/CDF slot each
        // so the GPU buffer length stays non-zero (cubecl rejects
        // zero-length arrays).
        Self {
            e_grid_log: vec![0.0],
            n_e: 1,
            electron_pdf: vec![0.0; n_materials],
            electron_cdf: vec![0.0; n_materials],
            electron_yield: vec![0.0; n_materials],
            positron_pdf: vec![0.0; n_materials],
            positron_cdf: vec![0.0; n_materials],
            positron_yield: vec![0.0; n_materials],
            has_data: vec![0u32; n_materials],
        }
    }
}

/// Per-material input to the bremsstrahlung extractor: the CPU's
/// already-preprocessed `Bremsstrahlung` (electron + positron sub-
/// tables) for one material, or `None` for materials that didn't run
/// `init_bremsstrahlung` (e.g. TTB disabled, missing data).
pub struct MaterialTtb<'a> {
    /// Lower-triangular PDF for electrons, shape `[n_e][n_e]`. Row j
    /// is the PDF at incident electron energy index j.
    pub electron_pdf: &'a [Vec<f64>],
    /// Lower-triangular CDF for electrons, shape `[n_e][n_e]`.
    pub electron_cdf: &'a [Vec<f64>],
    /// Per-incident-energy electron photon yield, log-space, length `n_e`.
    pub electron_yield: &'a [f64],
    /// Lower-triangular PDF for positrons, shape `[n_e][n_e]`.
    pub positron_pdf: &'a [Vec<f64>],
    /// Lower-triangular CDF for positrons, shape `[n_e][n_e]`.
    pub positron_cdf: &'a [Vec<f64>],
    /// Per-incident-energy positron photon yield, log-space, length `n_e`.
    pub positron_yield: &'a [f64],
}

/// Pack per-material bremsstrahlung tables into a `GpuBremsstrahlung`.
///
/// `materials` is the per-material TTB data (`None` for materials
/// without preprocessed tables -- those materials get `has_data[m] = 0`
/// and zero-padded slots). `e_grid_log` is the global TTB energy grid
/// in log space (same grid the CPU uses after `ensure_ttb_e_grid_log`
/// runs at the end of `Model::ensure_photon_data_for_gpu`).
///
/// Empty `e_grid_log` (TTB not enabled / not initialised) returns a
/// degenerate single-grid-point pack via `GpuBremsstrahlung::empty_for_materials`.
pub fn extract_bremsstrahlung_for_gpu(
    materials: &[Option<MaterialTtb<'_>>],
    e_grid_log: &[f64],
) -> GpuBremsstrahlung {
    if e_grid_log.is_empty() || materials.is_empty() {
        return GpuBremsstrahlung::empty_for_materials(materials.len());
    }

    let n_e = e_grid_log.len();
    let n_e_u32 = n_e as u32;
    let n_mat = materials.len();

    let slab_len = n_e * n_e;
    let mut electron_pdf = vec![0.0_f64; n_mat * slab_len];
    let mut electron_cdf = vec![0.0_f64; n_mat * slab_len];
    let mut electron_yield = vec![0.0_f64; n_mat * n_e];
    let mut positron_pdf = vec![0.0_f64; n_mat * slab_len];
    let mut positron_cdf = vec![0.0_f64; n_mat * slab_len];
    let mut positron_yield = vec![0.0_f64; n_mat * n_e];
    let mut has_data = vec![0u32; n_mat];

    for (m, ttb_opt) in materials.iter().enumerate() {
        let Some(ttb) = ttb_opt else { continue };

        // Defensive shape check -- if any sub-table doesn't match the
        // expected `n_e × n_e` shape, skip this material rather than
        // copy garbage into the kernel's slab.
        if ttb.electron_yield.len() != n_e
            || ttb.positron_yield.len() != n_e
            || ttb.electron_pdf.len() != n_e
            || ttb.electron_cdf.len() != n_e
            || ttb.positron_pdf.len() != n_e
            || ttb.positron_cdf.len() != n_e
        {
            continue;
        }

        let slab_off = m * slab_len;
        let yield_off = m * n_e;

        // Yields are already log-space scalars -- copy 1:1.
        electron_yield[yield_off..yield_off + n_e].copy_from_slice(ttb.electron_yield);
        positron_yield[yield_off..yield_off + n_e].copy_from_slice(ttb.positron_yield);

        for j in 0..n_e {
            let row_e_pdf = &ttb.electron_pdf[j];
            let row_e_cdf = &ttb.electron_cdf[j];
            let row_p_pdf = &ttb.positron_pdf[j];
            let row_p_cdf = &ttb.positron_cdf[j];
            // Row length is n_e (CPU stores square matrices with zero
            // above the diagonal). Skip rows that disagree on shape.
            if row_e_pdf.len() != n_e || row_e_cdf.len() != n_e {
                continue;
            }
            if row_p_pdf.len() != n_e || row_p_cdf.len() != n_e {
                continue;
            }
            let row_off = slab_off + j * n_e;
            electron_pdf[row_off..row_off + n_e].copy_from_slice(row_e_pdf);
            electron_cdf[row_off..row_off + n_e].copy_from_slice(row_e_cdf);
            positron_pdf[row_off..row_off + n_e].copy_from_slice(row_p_pdf);
            positron_cdf[row_off..row_off + n_e].copy_from_slice(row_p_cdf);
        }
        has_data[m] = 1u32;
    }

    GpuBremsstrahlung {
        e_grid_log: e_grid_log.to_vec(),
        n_e: n_e_u32,
        electron_pdf,
        electron_cdf,
        electron_yield,
        positron_pdf,
        positron_cdf,
        positron_yield,
        has_data,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pack_for_zero_materials() {
        let pack = GpuBremsstrahlung::empty_for_materials(0);
        // n_materials clamped to 1 so buffers stay non-zero.
        assert_eq!(pack.n_e, 1);
        assert_eq!(pack.has_data.len(), 1);
        assert_eq!(pack.has_data[0], 0);
        assert_eq!(pack.electron_pdf.len(), 1);
    }

    #[test]
    fn empty_pack_for_three_materials() {
        let pack = GpuBremsstrahlung::empty_for_materials(3);
        assert_eq!(pack.has_data, vec![0u32; 3]);
        assert_eq!(pack.electron_yield.len(), 3);
    }

    #[test]
    fn extract_with_empty_grid_returns_empty_pack() {
        let pack = extract_bremsstrahlung_for_gpu(&[None, None], &[]);
        assert_eq!(pack.n_e, 1);
        assert_eq!(pack.has_data, vec![0u32; 2]);
    }

    #[test]
    fn extract_with_none_materials_zero_pads() {
        let grid: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let pack = extract_bremsstrahlung_for_gpu(&[None, None], &grid);
        assert_eq!(pack.n_e, 4);
        assert_eq!(pack.e_grid_log, grid);
        assert_eq!(pack.has_data, vec![0u32, 0u32]);
        // 2 materials × 16 slab + 0 = all zero
        assert!(pack.electron_pdf.iter().all(|&v| v == 0.0));
        assert!(pack.electron_cdf.iter().all(|&v| v == 0.0));
        assert!(pack.electron_yield.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn extract_with_one_material_fills_slab() {
        let n_e = 3;
        let grid: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
        // Lower-triangular PDF / CDF: pdf[j][i] = 1.0 for i <= j else 0.0.
        let pdf: Vec<Vec<f64>> = (0..n_e)
            .map(|j| (0..n_e).map(|i| if i <= j { 1.0 } else { 0.0 }).collect())
            .collect();
        let cdf = pdf.clone();
        let yield_: Vec<f64> = (0..n_e).map(|j| (j + 1) as f64).collect();

        let ttb = MaterialTtb {
            electron_pdf: &pdf,
            electron_cdf: &cdf,
            electron_yield: &yield_,
            positron_pdf: &pdf,
            positron_cdf: &cdf,
            positron_yield: &yield_,
        };
        let pack = extract_bremsstrahlung_for_gpu(&[Some(ttb)], &grid);
        assert_eq!(pack.has_data, vec![1u32]);
        assert_eq!(pack.n_e, n_e as u32);
        assert_eq!(pack.electron_yield, yield_);

        // Spot-check the flat PDF layout: slab[j * n_e + i]
        for j in 0..n_e {
            for i in 0..n_e {
                let expect = if i <= j { 1.0 } else { 0.0 };
                let got = pack.electron_pdf[j * n_e + i];
                assert_eq!(got, expect, "pdf[{}][{}]", j, i);
            }
        }
    }

    #[test]
    fn extract_skips_material_with_shape_mismatch() {
        let n_e = 3;
        let grid: Vec<f64> = (0..n_e).map(|i| i as f64).collect();
        // Bad shape: yield is too short.
        let pdf = vec![vec![0.0; n_e]; n_e];
        let bad_yield = vec![1.0; n_e - 1];
        let ttb = MaterialTtb {
            electron_pdf: &pdf,
            electron_cdf: &pdf,
            electron_yield: &bad_yield,
            positron_pdf: &pdf,
            positron_cdf: &pdf,
            positron_yield: &bad_yield,
        };
        let pack = extract_bremsstrahlung_for_gpu(&[Some(ttb)], &grid);
        assert_eq!(
            pack.has_data,
            vec![0u32],
            "shape-mismatched material skipped"
        );
    }
}
