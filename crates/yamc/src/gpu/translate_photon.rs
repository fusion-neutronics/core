//! `Model` → photon-kernel flat-buffer inputs.
//!
//! Mirrors `translate.rs` for photon transport: same geometry /
//! surface / initial-particle translation, but the XS slate is
//! photon-specific (per-material macro photoatomic XS partitioned
//! into coherent / incoherent / photoelectric / pair, plus a
//! dominant-element Rayleigh form-factor table).

use std::sync::Arc;

use yamc_gpu::photon::element_select_inputs::PhotonElementSelectInputs;
use yamc_gpu::photon::xs::atomic_relaxation_xs::{
    extract_atomic_relaxation_for_gpu, GpuAtomicRelaxation,
};
use yamc_gpu::photon::xs::bremsstrahlung_xs::{
    extract_bremsstrahlung_for_gpu, GpuBremsstrahlung, MaterialTtb,
};
use yamc_gpu::photon::xs::compton_doppler_xs::{
    extract_compton_doppler_for_gpu, GpuComptonDoppler,
};
use yamc_gpu::photon::xs::incoherent_form_factor_xs::{
    extract_incoherent_form_factor_for_gpu, GpuIncoherentFormFactor,
};
use yamc_gpu::photon::xs::pair_production_xs::{
    extract_pair_production_for_gpu, GpuPairProduction,
};
use yamc_gpu::photon::xs::photon_xs::{extract_photon_material_xs, GpuPhotonXs};

use super::error::GpuTranslateError;
use super::translate::{
    sample_initial_particles_for_batch, translate_cells, translate_region_program,
    translate_surfaces,
};
use crate::geometry::backend::GeometryKind;
use crate::model::Model;
use yamc_materials::material::Material;

/// Photon-side companion to `GpuTransportInputs`. Most of the
/// geometry / source-particle buffers are identical to the neutron
/// path's; only the XS slate is photon-specific.
///
/// `ttb` is the thick-target-bremsstrahlung pack used by the kernel
/// to generate secondary photons from electrons produced in
/// Compton / photoelectric / pair events. Populated only when
/// `model.electron_treatment` is `Ttb` and `init_bremsstrahlung`
/// has been run on every material; otherwise carries a degenerate
/// `has_data = [0; n_materials]` pack that the kernel skips. Future
/// kernel changes (slice 2) read this; current kernel ignores it.
#[derive(Debug, Clone)]
pub struct GpuPhotonTransportInputs {
    /// `Model::photon_cutoff_energy` in eV (issue #286). Carried here so every
    /// photon launch (primary, coupled sub-pass, mixed, D1S) applies the model's
    /// cutoff instead of the kernel's old hardcoded 1 keV default.
    pub photon_cutoff_energy: f64,
    pub seeds: Vec<u32>,
    pub energies: Vec<f64>,
    pub positions: Vec<f64>,
    pub directions: Vec<f64>,
    pub cell_aabbs: Vec<f64>,
    pub cell_to_material: Vec<u32>,
    pub surface_types: Vec<u32>,
    pub surface_params: Vec<f64>,
    pub surface_boundaries: Vec<u32>,
    pub region_program: Vec<u32>,
    pub log_energy_grid: Vec<f64>,
    pub xs_total: Vec<f64>,
    pub xs_coherent: Vec<f64>,
    pub xs_incoherent: Vec<f64>,
    pub xs_photoelectric: Vec<f64>,
    pub xs_pair: Vec<f64>,
    /// Per-material macroscopic photon heating / KERMA XS (flat
    /// `[n_materials × n_grid]`). Same density-weighted aggregation and
    /// linear-interp form as the component XS above; the kernel reads
    /// it for `Score::Heating` / `Score::HeatingLocal` (MT 301 / 901)
    /// photon tallies, mirroring the CPU track-length estimate.
    pub xs_heating: Vec<f64>,
    pub rayleigh_x2: Vec<f64>,
    pub rayleigh_cdf: Vec<f64>,
    pub rayleigh_n_points: Vec<u32>,
    /// Per-collision element-selection inputs (task #72): the per-(element,
    /// energy) macroscopic-total weight table and the per-material element-slab
    /// `[offset, count]` meta. The kernel samples the interacting element by
    /// macro-XS contribution (mirroring CPU `Material::sample_element`), then
    /// indexes that element's form-factor / relaxation / pair slab in
    /// `rayleigh_*`, `doppler`, `iff`, `atomic_relaxation`, and `pair` -- all
    /// now keyed by element slab rather than material.
    pub element_select: PhotonElementSelectInputs,
    pub ttb: GpuBremsstrahlung,
    /// Compton Doppler-broadening tables (per-element shell binding
    /// energies, occupancies, and `J(p_z)` profiles). Used by the
    /// kernel's Compton path to replace free-electron Klein-Nishina
    /// E_out with the bound-electron Doppler-shifted value -- matches
    /// CPU's `compton_doppler` sampler. Empty pack (has_data = 0)
    /// when `electron_treatment` is `Local` or no element has
    /// Compton profile data loaded.
    pub doppler: GpuComptonDoppler,
    /// Incoherent (bound-electron Compton) form-factor table. Used
    /// by the kernel's Kahn rejection loop to filter free-electron
    /// Klein-Nishina samples against `S(x)/S(x_max)` -- matches CPU's
    /// `compton_scatter` form-factor rejection step. Empty pack
    /// (has_data = 0) when no element data is available; the kernel
    /// then falls back to pure free-electron Kahn.
    pub iff: GpuIncoherentFormFactor,
    /// Atomic-relaxation pack (per-subshell PE cross sections +
    /// fluorescence / Auger transition tables). Used by the kernel's
    /// photoelectric branch to sample which subshell absorbs the
    /// photon and to emit fluorescent X-rays / Auger electrons --
    /// matches CPU's `sample_photoelectric_subshell` +
    /// `atomic_relaxation` chain. Empty pack (has_data = 0) when
    /// `electron_treatment` is `Local` or no element data is
    /// available; the kernel then falls back to the K-shell guess.
    pub atomic_relaxation: GpuAtomicRelaxation,
    /// Per-material pair-production constants (dominant-element
    /// reduced screening radius `r_z`, Born parameter `a`, Coulomb
    /// correction `c`). The kernel's pair branch uses these to
    /// sample the electron / positron energies and angles, then
    /// runs TTB on each and emits two 511 keV annihilation photons.
    /// Empty pack (has_data = 0) when the dominant Z is out of range
    /// -- the kernel falls back to simple absorption.
    pub pair: GpuPairProduction,
}

/// Build flat-buffer inputs for the GPU photon kernel from a yamc
/// `Model`. Validates that every source emits photons; rejects
/// mesh geometry; otherwise translates geometry / surfaces / sources
/// using the same helpers as the neutron path and aggregates per-
/// material photon XS via `yamc_gpu::photon::xs::photon_xs::extract_photon_material_xs`.
pub fn translate_photon_for_gpu(
    model: &Model,
    n_particles: usize,
    base_seed: u64,
) -> Result<GpuPhotonTransportInputs, GpuTranslateError> {
    if n_particles == 0 {
        return Err(GpuTranslateError::NoParticles);
    }
    if model.sources.is_empty() {
        return Err(GpuTranslateError::NoSources);
    }

    #[cfg(feature = "mesh")]
    let geometry = match &model.geometry {
        GeometryKind::Csg(g) => g,
        GeometryKind::Mesh(_) => return Err(GpuTranslateError::MeshGeometryUnsupported),
    };
    #[cfg(not(feature = "mesh"))]
    let GeometryKind::Csg(geometry) = &model.geometry;

    let (cell_aabbs, cell_to_material, has_void) =
        translate_cells(&geometry.cells, geometry.materials.len())?;
    let (surface_types, surface_params, surface_boundaries) = translate_surfaces(&geometry.cells)?;
    // Per-cell CSG region program: the photon kernel needs the exact region test
    // (not AABB-only) to resolve nested / overlapping cells, same as the neutron
    // kernel. Built from the same shared translator.
    let region_program = translate_region_program(&geometry.cells);
    // Void (material-less) cells map to a synthetic material slot at
    // index `n_materials`. Every photon pack gets one extra all-zero /
    // `has_data = 0` row for that slot so the kernel's `mat_idx =
    // n_materials` lookup lands on a zero XS row -- `sigma_t = 0`, the
    // particle streams to the next surface with no interaction. (The
    // photon kernel guards the collision block behind `sigma_t > 0.0`,
    // so the void row is never sampled.)
    let n_void = usize::from(has_void);
    let photon_xs = build_photon_xs(&geometry.materials, n_void);
    // Total element-slab count (task #72): every per-element pack (rayleigh,
    // Doppler, IFF, AR, pair) is keyed by this slab count, and the TTB-off
    // empty packs must match so the kernel's `elem_slab` indexing is in bounds.
    // Void slots contribute zero elements, so this equals the sum of real
    // per-material element counts.
    let n_slab: usize = photon_xs.elem_counts.iter().map(|&c| c as usize).sum();
    let ttb = build_ttb(model, &geometry.materials, n_void);
    let doppler = build_doppler(&geometry.materials, n_void, n_slab);
    let iff = build_iff(&geometry.materials, n_void);
    let atomic_relaxation =
        build_atomic_relaxation(&geometry.materials, &photon_xs.log_energy_grid, n_void);
    let pair = build_pair_production(&geometry.materials, n_void);
    // Per-collision element-selection inputs (task #72). The macro-total weight
    // table and per-material element counts come from `build_photon_xs`; void
    // slots have count 0 (the kernel never reaches selection for them).
    let element_select = PhotonElementSelectInputs::from_flat(
        photon_xs.elem_macro_total,
        &photon_xs.elem_counts,
        photon_xs.log_energy_grid.len(),
    );
    let (seeds, energies, positions, directions) =
        sample_initial_particles_for_batch(model, n_particles, 0, base_seed);

    Ok(GpuPhotonTransportInputs {
        photon_cutoff_energy: model.photon_cutoff_energy,
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        region_program,
        log_energy_grid: photon_xs.log_energy_grid,
        xs_total: photon_xs.xs_total,
        xs_coherent: photon_xs.xs_coherent,
        xs_incoherent: photon_xs.xs_incoherent,
        xs_photoelectric: photon_xs.xs_photoelectric,
        xs_pair: photon_xs.xs_pair,
        xs_heating: photon_xs.xs_heating,
        rayleigh_x2: photon_xs.rayleigh_x2,
        rayleigh_cdf: photon_xs.rayleigh_cdf,
        rayleigh_n_points: photon_xs.rayleigh_n_points,
        element_select,
        ttb,
        doppler,
        iff,
        atomic_relaxation,
        pair,
    })
}

/// `(name, element_arc, atom_density)` triples for one material, in
/// the shape the photon-side extractors consume.
type MaterialTriples = Vec<(String, Arc<yamc_element::photon::PhotonInteraction>, f64)>;

/// Walk each material's cached photon elements + atom densities into
/// the per-material `(name, element_arc, density)` triple lists the
/// photon-side extractors consume. When the material's
/// `cached_element_atom_densities` haven't been populated (e.g. the
/// model never ran `calculate_macroscopic_xs`), fall back to computing
/// them via `get_atoms_per_barn_cm`.
///
/// `n_void` synthetic void slots are appended as empty triple lists.
/// Every photon extractor treats a material with no elements as a void
/// row (all-zero XS, `has_data = 0`), so this is how the void material
/// slot flows uniformly into each pack without per-pack field padding.
fn material_element_triples(materials: &[Arc<Material>], n_void: usize) -> Vec<MaterialTriples> {
    let mut per_mat: Vec<MaterialTriples> = Vec::with_capacity(materials.len() + n_void);
    for m in materials {
        let elements = m.cached_elements.clone();
        let densities = if m.cached_element_atom_densities.len() == elements.len() {
            m.cached_element_atom_densities.clone()
        } else if !elements.is_empty() {
            let atoms_per_bcm = m.get_atoms_per_barn_cm().unwrap_or_else(|e| panic!("{e}"));
            elements
                .iter()
                .map(|(name, _)| atoms_per_bcm.get(name).copied().unwrap_or(0.0))
                .collect()
        } else {
            Vec::new()
        };
        let mut triples = Vec::with_capacity(elements.len());
        for ((name, el), density) in elements.into_iter().zip(densities) {
            triples.push((name, el, density));
        }
        per_mat.push(triples);
    }
    // Synthetic void slots: empty element lists -> all-zero void rows.
    for _ in 0..n_void {
        per_mat.push(Vec::new());
    }
    per_mat
}

/// Pack per-material pair-production constants (dominant-element
/// reduced screening radius, Born parameter, Coulomb correction).
/// Mirrors the dominant-element pattern of the other photon
/// extractors. Empty pack returned when no usable Z is found.
fn build_pair_production(materials: &[Arc<Material>], n_void: usize) -> GpuPairProduction {
    let per_mat = material_element_triples(materials, n_void);
    extract_pair_production_for_gpu(&per_mat)
}

/// Pack per-material atomic-relaxation (fluorescence) tables for the kernel.
/// Mirrors `build_doppler` / `build_iff`'s dominant-element pattern. Atomic
/// relaxation is emitted regardless of the electron treatment: LED vs TTB
/// governs only how the electron's energy is handled (local deposition vs
/// bremsstrahlung), not whether characteristic X-rays are produced. So the pack
/// is built for both, matching the CPU path, which always relaxes an ionized
/// shell. (Previously this returned an empty pack under `Local`, which silently
/// dropped all fluorescence on the GPU under LED -- most visibly the K/L lines
/// of high-Z elements such as tungsten and lead.)
fn build_atomic_relaxation(
    materials: &[Arc<Material>],
    log_energy_grid: &[f64],
    n_void: usize,
) -> GpuAtomicRelaxation {
    let per_mat = material_element_triples(materials, n_void);
    extract_atomic_relaxation_for_gpu(&per_mat, log_energy_grid)
}

/// Pack per-material incoherent form-factor tables. Reads
/// `cached_elements` + atom densities to build the
/// `(name, element_arc, density)` triples
/// `extract_incoherent_form_factor_for_gpu` consumes.
fn build_iff(materials: &[Arc<Material>], n_void: usize) -> GpuIncoherentFormFactor {
    let per_mat = material_element_triples(materials, n_void);
    extract_incoherent_form_factor_for_gpu(&per_mat)
}

/// Pack per-material Compton Doppler tables for the kernel. Returns a degenerate
/// empty pack only when the global Compton-profile pz grid isn't set up (no
/// element loaded photon data), where the kernel falls back to free-electron
/// Klein-Nishina. Doppler broadening -- and the Compton-ionization fluorescence
/// it drives via `dop_subshell_map` -- is independent of the electron treatment,
/// so it is built under LED as well as TTB (previously it was gated on TTB,
/// which dropped Compton fluorescence on the GPU under LED).
fn build_doppler(materials: &[Arc<Material>], n_void: usize, n_slab: usize) -> GpuComptonDoppler {
    let pz_grid = yamc_element::photon::compton_profile_pz();
    if pz_grid.is_empty() {
        return GpuComptonDoppler::empty_for_slabs(n_slab);
    }
    let per_mat = material_element_triples(materials, n_void);
    extract_compton_doppler_for_gpu(&per_mat, &pz_grid)
}

/// Pack the per-material bremsstrahlung tables (`material.ttb`) into
/// the flat-buffer `GpuBremsstrahlung` shape the photon kernel reads.
///
/// Empty pack returned when `model.electron_treatment` is `Local`,
/// or no material has been through `init_bremsstrahlung` yet (the GPU
/// dispatch normally does this via
/// `Model::ensure_photon_data_for_gpu` before reaching this point).
fn build_ttb(model: &Model, materials: &[Arc<Material>], n_void: usize) -> GpuBremsstrahlung {
    if !model.electron_treatment.ttb() {
        return GpuBremsstrahlung::empty_for_materials(materials.len() + n_void);
    }

    // Global TTB log-energy grid (populated by `Model::ensure_photon_data_for_gpu`
    // → `ensure_ttb_e_grid_log`). Empty when no material has
    // initialised photon data yet; in that case fall back to a
    // degenerate empty pack.
    let e_grid_log = yamc_element::photon::ttb_e_grid_log();
    if e_grid_log.is_empty() {
        return GpuBremsstrahlung::empty_for_materials(materials.len() + n_void);
    }

    let mut mat_views: Vec<Option<MaterialTtb<'_>>> = materials
        .iter()
        .map(|m| {
            m.ttb.as_ref().map(|t| MaterialTtb {
                electron_pdf: t.electron.pdf.as_slice(),
                electron_cdf: t.electron.cdf.as_slice(),
                electron_yield: t.electron.yield_.as_slice(),
                positron_pdf: t.positron.pdf.as_slice(),
                positron_cdf: t.positron.cdf.as_slice(),
                positron_yield: t.positron.yield_.as_slice(),
            })
        })
        .collect();
    // Void slots have no TTB tables: `None` -> zero / `has_data = 0` row.
    for _ in 0..n_void {
        mat_views.push(None);
    }
    extract_bremsstrahlung_for_gpu(&mat_views, &e_grid_log)
}

fn build_photon_xs(materials: &[Arc<Material>], n_void: usize) -> GpuPhotonXs {
    let per_mat = material_element_triples(materials, n_void);
    extract_photon_material_xs(&per_mat)
}
