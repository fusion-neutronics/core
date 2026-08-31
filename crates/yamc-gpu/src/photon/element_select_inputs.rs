//! Per-collision element-selection inputs for the photon transport kernel
//! (task #72).
//!
//! At a photon collision in a multi-element material the transport must pick
//! *which* element the photon interacts with, proportional to that element's
//! macroscopic total photon cross section at the collision energy
//! (`atom_density * micro.total`), and then run THAT element's coherent /
//! incoherent / photoelectric / pair physics (form factors + relaxation
//! cascade). This mirrors the CPU `Material::sample_element` +
//! `handle_photon_collision` chain exactly.
//!
//! Before #72 the GPU carried only the single dominant element's
//! form-factor / relaxation tables per material (#79), so a material with two
//! comparable-Z elements (Pb+W, Pb+Bi, ...) modelled only ONE element's
//! secondary / fluorescence spectrum. This struct restores true per-collision
//! selection: the form-factor / relaxation / pair packs are stored one slab
//! per ELEMENT (concatenated material-major) and this table both weights the
//! selection draw and maps a material to its element slab range.
//!
//! - [`Self::elem_macro_total`] -- per-(element, energy) macroscopic total xs
//!   on the shared grid, flat `[n_slab x n_grid]` where
//!   `n_slab = sum_mat n_elements(mat)`. This is the per-collision selection
//!   weight table (cumulative-sum walk against the collision-energy
//!   interpolant), element-major within a material and concatenated
//!   material-major.
//! - [`Self::mat_elem_meta`] -- stride-2 `[offset, count]` per material:
//!   `mat_elem_meta[mat*2]` is the element-slab base of the material's first
//!   element row, `mat_elem_meta[mat*2 + 1]` is its element count.
//!
//! # Single-element bit-identity
//!
//! When a material has exactly one element (`count == 1`) the kernel and the
//! CPU twin SKIP the selection draw entirely -- the element is trivially the
//! only one, so no extra random is consumed and the RNG stream is
//! byte-identical to the #79 single-dominant-element behaviour. The element
//! slab index then equals `offset`, which is that material's only element, so
//! the form-factor / relaxation lookups are unchanged. Every single-element
//! material in the verification suite is therefore bit-identical.

/// Packed per-collision element-selection inputs (see module docs). Built once
/// per launch and passed by reference into the host launcher and the CPU twin.
///
/// This struct lives OUTSIDE the macOS-gated `transport` module (like
/// `neutron::nuclide_select_inputs`) because the always-compiled
/// `translate_photon.rs` / `dispatch.rs` reference it. A plain input struct
/// inside the gated kernel module would break the macOS wheel / MPI builds.
#[derive(Debug, Clone, Default)]
pub struct PhotonElementSelectInputs {
    /// Per-(element, energy) macroscopic total photon xs on the shared grid,
    /// flat `[n_slab x n_grid]`, element-major within a material and
    /// concatenated material-major. Row `slab` starts at `slab * n_grid`.
    pub elem_macro_total: Vec<f64>,
    /// Stride-2 `[offset, count]` per material: element-slab base + count.
    pub mat_elem_meta: Vec<u32>,
}

impl PhotonElementSelectInputs {
    /// Degenerate single-element-per-material layout: one slab row per
    /// material, `count == 1`, so the kernel / CPU twin never draws the
    /// selection random and stays byte-identical to the #79 single-element
    /// behaviour. The macro-total row is a single `n_grid`-wide block of zeros
    /// (never read when `count == 1`). Used for fixtures / void slots.
    pub fn single_element(n_materials: usize, n_grid: usize) -> Self {
        let n_materials = n_materials.max(1);
        let n_grid = n_grid.max(1);
        let mut mat_elem_meta = Vec::with_capacity(n_materials * 2);
        for slab in 0..n_materials {
            mat_elem_meta.push(slab as u32); // offset = one row per material
            mat_elem_meta.push(1u32); // count = 1
        }
        Self {
            elem_macro_total: vec![0.0; n_materials * n_grid],
            mat_elem_meta,
        }
    }

    /// Build from a flat per-(element, energy) macro-total table plus the
    /// per-material element counts (as produced by `extract_photon_material_xs`,
    /// which already concatenates the element slabs material-major). This avoids
    /// re-slicing the flat table into per-material blocks. `elem_counts` is
    /// length `n_materials` (a void slot has count 0).
    pub fn from_flat(elem_macro_total: Vec<f64>, elem_counts: &[u32], n_grid: usize) -> Self {
        let mut mat_elem_meta = Vec::with_capacity(elem_counts.len() * 2);
        let mut slab_base: u32 = 0;
        for &count in elem_counts {
            mat_elem_meta.push(slab_base);
            mat_elem_meta.push(count);
            slab_base += count;
        }
        let mut elem_macro_total = elem_macro_total;
        if elem_macro_total.is_empty() {
            elem_macro_total = vec![0.0; n_grid.max(1)];
        }
        if mat_elem_meta.is_empty() {
            mat_elem_meta = vec![0, 1];
        }
        Self {
            elem_macro_total,
            mat_elem_meta,
        }
    }

    /// Build from per-material element macro-total rows: `materials[m]` is that
    /// material's `[n_elements x n_grid]` flat block (element-major, in the
    /// SAME element order the form-factor / relaxation packs slab-concatenate).
    /// Materials are concatenated in the given order, building the element-slab
    /// offset table as it goes.
    ///
    /// A material with zero elements (a void slot) gets `count == 0` and an
    /// empty slab range; the kernel never reaches the selection block for it
    /// (`sigma_t == 0` streams the photon through). `n_grid` is the shared grid
    /// length (columns per row).
    pub fn from_materials(materials: &[Vec<f64>], n_grid: usize) -> Self {
        let mut elem_macro_total = Vec::new();
        let mut mat_elem_meta = Vec::with_capacity(materials.len() * 2);
        let mut slab_base: u32 = 0;
        for rows in materials {
            let count = rows.len().checked_div(n_grid).unwrap_or(0);
            debug_assert_eq!(
                rows.len(),
                count * n_grid,
                "macro-total block must be [n_elements x n_grid]"
            );
            mat_elem_meta.push(slab_base);
            mat_elem_meta.push(count as u32);
            elem_macro_total.extend_from_slice(rows);
            slab_base += count as u32;
        }
        // cubecl rejects zero-length buffers; pad an empty model to one slot.
        if elem_macro_total.is_empty() {
            elem_macro_total = vec![0.0; n_grid.max(1)];
        }
        if mat_elem_meta.is_empty() {
            mat_elem_meta = vec![0, 1];
        }
        Self {
            elem_macro_total,
            mat_elem_meta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_element_is_degenerate() {
        let inp = PhotonElementSelectInputs::single_element(3, 4);
        assert_eq!(inp.mat_elem_meta, vec![0, 1, 1, 1, 2, 1]);
        assert_eq!(inp.elem_macro_total.len(), 3 * 4);
    }

    #[test]
    fn from_materials_builds_offsets() {
        // Material 0: 2 elements, material 1: 1 element. n_grid = 2.
        let m0 = vec![1.0, 2.0, 3.0, 4.0]; // 2 elements x 2 grid
        let m1 = vec![5.0, 6.0]; // 1 element x 2 grid
        let inp = PhotonElementSelectInputs::from_materials(&[m0, m1], 2);
        // offsets: mat0 base 0 count 2; mat1 base 2 count 1.
        assert_eq!(inp.mat_elem_meta, vec![0, 2, 2, 1]);
        assert_eq!(inp.elem_macro_total, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn empty_model_pads() {
        let inp = PhotonElementSelectInputs::from_materials(&[], 4);
        assert_eq!(inp.elem_macro_total.len(), 4);
        assert_eq!(inp.mat_elem_meta, vec![0, 1]);
    }

    #[test]
    fn void_material_has_zero_count() {
        // A void material (empty row block) -> count 0.
        let inp = PhotonElementSelectInputs::from_materials(&[vec![1.0, 2.0], vec![]], 2);
        assert_eq!(inp.mat_elem_meta, vec![0, 1, 1, 0]);
    }
}
