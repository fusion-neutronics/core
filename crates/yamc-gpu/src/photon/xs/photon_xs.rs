//! GPU-side photon cross-section extraction.
//!
//! Mirrors `nuclide_xs.rs` for the photon side: aggregate the
//! material's per-element micro photon XS into per-material macro
//! arrays on a unified energy grid, plus a per-material Rayleigh
//! coherent form-factor table (single-dominant-element approximation,
//! the element chosen by largest coherent macroscopic XS contribution --
//! see `dominant_element_for_reaction`).
//!
//! ## What's modelled
//!
//! - Total photon XS, partitioned into:
//!   - Coherent (Rayleigh)
//!   - Incoherent (Compton)
//!   - Photoelectric (sum over subshells)
//!   - Pair production (nuclear + electron)
//! - Macroscopic photon heating / KERMA XS, aggregated the SAME way
//!   the CPU `Material::calculate_photon_xs` does (density-weighted sum
//!   of each element's `ElementMicroXS.heating`). The kernel scores
//!   photon `Score::Heating` / `Score::HeatingLocal` tallies (MT 301 /
//!   901) by interpolating this array linearly, exactly mirroring the
//!   CPU track-length photon-heating estimate (issue #356).

use std::sync::Arc;

use yamc_element::photon::PhotonInteraction;
use yamc_nuclide::reaction_product::Tabulated1D;

/// Maximum incident-energy points stored on the GPU per-material
/// Rayleigh form-factor table. Coherent form factors are typically
/// tabulated on ~50 momentum-transfer points; 64 is generous with
/// uniform-in-index subsampling if a particular element's table
/// exceeds this.
pub const MAX_RAYLEIGH_FF: usize = 64;

/// Per-material photon cross-section data ready for upload to the
/// GPU photon transport kernel.
#[derive(Clone, Debug)]
pub struct GpuPhotonXs {
    /// `ln(energy[i])` for each grid point. Same length as the
    /// `xs_*` arrays below.
    pub log_energy_grid: Vec<f64>,
    /// Macroscopic total photon XS per material at each energy grid
    /// point, flat `[n_materials × n_grid]`.
    pub xs_total: Vec<f64>,
    /// Coherent (Rayleigh) macroscopic XS.
    pub xs_coherent: Vec<f64>,
    /// Incoherent (Compton) macroscopic XS.
    pub xs_incoherent: Vec<f64>,
    /// Photoelectric macroscopic XS (sum of all subshells).
    pub xs_photoelectric: Vec<f64>,
    /// Pair-production macroscopic XS (nuclear + electron pair).
    pub xs_pair: Vec<f64>,
    /// Macroscopic photon heating / KERMA XS (eV·barn × atom density,
    /// i.e. eV/cm per unit flux). Density-weighted sum of each
    /// element's `ElementMicroXS.heating` -- identical aggregation and
    /// interpolation form to the CPU `Material::calculate_photon_xs`
    /// `.heating` field that a `Score::Heating` photon tally scores.
    /// Same flat `[n_materials × n_grid]` layout as the `xs_*` arrays;
    /// the kernel interpolates it linearly just like the other
    /// components.
    pub xs_heating: Vec<f64>,
    /// Per-ELEMENT Rayleigh coherent integrated form-factor table -- the
    /// `x²` axis. Layout: `[n_slab × MAX_RAYLEIGH_FF]` where
    /// `n_slab = sum_mat n_elements(mat)`, element-major within a material
    /// and concatenated material-major (the order
    /// `PhotonElementSelectInputs::mat_elem_meta` indexes). Each element gets
    /// its own slab from its `coherent_int_form_factor` (Tabulated1D), so the
    /// kernel reads the per-collision-SELECTED element's form factor (task
    /// #72), not a single dominant element's (#79). Zero-padded past
    /// `rayleigh_n_points[slab]`.
    pub rayleigh_x2: Vec<f64>,
    /// Per-element integrated form-factor CDF values -- the `y` axis of
    /// `coherent_int_form_factor`. Same shape as `rayleigh_x2`.
    pub rayleigh_cdf: Vec<f64>,
    /// Number of valid points in each element's Rayleigh table
    /// (0 when no element-data is available), length `n_slab`.
    pub rayleigh_n_points: Vec<u32>,
    /// Per-(element, energy) macroscopic total photon xs on the master grid,
    /// flat `[n_slab × n_grid]`, in the SAME element-slab order as
    /// `rayleigh_*`. This is the per-collision element-selection weight table
    /// (`atom_density × micro.total`), consumed by
    /// `PhotonElementSelectInputs::from_materials`.
    pub elem_macro_total: Vec<f64>,
    /// Per-material element count, length `n_materials` (excludes void slots
    /// the caller appends as empty triple lists -> count 0). Used to build the
    /// `mat_elem_meta` offset/count table.
    pub elem_counts: Vec<u32>,
}

/// Build photon-side flat-buffer inputs for a list of materials,
/// each carrying `(element_name, element_arc, atom_density)` triples.
///
/// All materials are resampled onto the first material's element's
/// energy grid (same convention as the neutron path's master grid).
pub fn extract_photon_material_xs(
    materials: &[Vec<(String, Arc<PhotonInteraction>, f64)>],
) -> GpuPhotonXs {
    if materials.is_empty() {
        return GpuPhotonXs {
            log_energy_grid: Vec::new(),
            xs_total: Vec::new(),
            xs_coherent: Vec::new(),
            xs_incoherent: Vec::new(),
            xs_photoelectric: Vec::new(),
            xs_pair: Vec::new(),
            xs_heating: Vec::new(),
            rayleigh_x2: Vec::new(),
            rayleigh_cdf: Vec::new(),
            rayleigh_n_points: Vec::new(),
            elem_macro_total: Vec::new(),
            elem_counts: Vec::new(),
        };
    }

    // Master energy grid: pull from the first material's first
    // element. `PhotonInteraction::energy` is already in `ln(E)`.
    let (_, first_element, _) = materials
        .iter()
        .find_map(|nucs| nucs.first().cloned())
        .expect("at least one material with at least one element");
    let log_energy_grid: Vec<f64> = first_element.energy.clone();
    let n_grid = log_energy_grid.len();

    let n_mat = materials.len();
    let n_slab: usize = materials.iter().map(|m| m.len()).sum();
    let mut xs_total = vec![0.0_f64; n_mat * n_grid];
    let mut xs_coherent = vec![0.0_f64; n_mat * n_grid];
    let mut xs_incoherent = vec![0.0_f64; n_mat * n_grid];
    let mut xs_photoelectric = vec![0.0_f64; n_mat * n_grid];
    let mut xs_pair = vec![0.0_f64; n_mat * n_grid];
    let mut xs_heating = vec![0.0_f64; n_mat * n_grid];
    // Rayleigh form factor and the selection-weight macro-total are now keyed
    // by ELEMENT slab (one slab per element, concatenated material-major), so
    // the kernel can read the per-collision-selected element (task #72).
    let mut rayleigh_x2 = vec![0.0_f64; n_slab * MAX_RAYLEIGH_FF];
    let mut rayleigh_cdf = vec![0.0_f64; n_slab * MAX_RAYLEIGH_FF];
    let mut rayleigh_n_points = vec![0u32; n_slab];
    let mut elem_macro_total = vec![0.0_f64; n_slab * n_grid];
    let mut elem_counts = vec![0u32; n_mat];
    let mut slab_base: usize = 0;

    for (m, mat_elements) in materials.iter().enumerate() {
        elem_counts[m] = mat_elements.len() as u32;

        // Aggregate macroscopic XS at each energy grid point. The
        // `PhotonInteraction.calculate_xs` returns micro XS (barns);
        // multiply by atom density to get macro and sum across
        // elements. The first element's ln-energy grid is the
        // master -- re-evaluate each element's micro XS at every
        // master-grid energy by exponentiating ln(E) and calling
        // `calculate_xs`. Avoids any axis-mismatch concerns when
        // a material has elements with slightly different native
        // grids.
        let mat_off = m * n_grid;
        for (i, &log_e) in log_energy_grid.iter().enumerate() {
            let energy = log_e.exp();
            let mut t = 0.0_f64;
            let mut coh = 0.0_f64;
            let mut inc = 0.0_f64;
            let mut photo = 0.0_f64;
            let mut pair = 0.0_f64;
            let mut heat = 0.0_f64;
            for (e, (_, el, density)) in mat_elements.iter().enumerate() {
                let micro = el.calculate_xs(energy);
                let elem_total = density * micro.total;
                t += elem_total;
                coh += density * micro.coherent;
                inc += density * micro.incoherent;
                photo += density * micro.photoelectric;
                pair += density * micro.pair_production;
                // Density-weighted heating, mirroring CPU
                // `Material::calculate_photon_xs`: `micro.heating` is
                // the element's KERMA value (log-linear-interp of the
                // tabulated `heating_xs` then `.exp()`, with the same
                // physics fallback when the table is absent).
                heat += density * micro.heating;
                // Per-element macro total -- the selection weight the kernel
                // samples the interacting element from (task #72), mirroring
                // CPU `Material::sample_element`'s `atom_density × micro.total`.
                elem_macro_total[(slab_base + e) * n_grid + i] = elem_total;
            }
            xs_total[mat_off + i] = t;
            xs_coherent[mat_off + i] = coh;
            xs_incoherent[mat_off + i] = inc;
            xs_photoelectric[mat_off + i] = photo;
            xs_pair[mat_off + i] = pair;
            xs_heating[mat_off + i] = heat;
        }

        // Rayleigh form factor: dump EACH element's integrated form factor
        // `F(x², Z)` onto its own slab (task #72 -- the kernel reads the
        // per-collision-selected element's slab, not a single dominant
        // element's). The kernel samples `x²` by inverse-CDF on (x², F(x²)),
        // so it needs both axes side by side.
        for (e, (_, el, _)) in mat_elements.iter().enumerate() {
            let slab = slab_base + e;
            let Tabulated1D::Tabulated1D { x, y, .. } = &el.coherent_int_form_factor;
            let n_src = x.len().min(y.len());
            let n = n_src.min(MAX_RAYLEIGH_FF);
            let stride = if n_src > MAX_RAYLEIGH_FF {
                (n_src - 1) as f64 / (MAX_RAYLEIGH_FF - 1) as f64
            } else {
                1.0
            };
            let off = slab * MAX_RAYLEIGH_FF;
            for j in 0..n {
                let src_j = if n_src > MAX_RAYLEIGH_FF {
                    (j as f64 * stride).round() as usize
                } else {
                    j
                };
                rayleigh_x2[off + j] = x[src_j.min(n_src - 1)];
                rayleigh_cdf[off + j] = y[src_j.min(n_src - 1)];
            }
            rayleigh_n_points[slab] = n as u32;
        }
        slab_base += mat_elements.len();
    }

    GpuPhotonXs {
        log_energy_grid,
        xs_total,
        xs_coherent,
        xs_incoherent,
        xs_photoelectric,
        xs_pair,
        xs_heating,
        rayleigh_x2,
        rayleigh_cdf,
        rayleigh_n_points,
        elem_macro_total,
        elem_counts,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_empty_materials() {
        let xs = extract_photon_material_xs(&[]);
        assert!(xs.log_energy_grid.is_empty());
        assert!(xs.xs_total.is_empty());
        assert_eq!(xs.rayleigh_n_points.len(), 0);
    }
}
