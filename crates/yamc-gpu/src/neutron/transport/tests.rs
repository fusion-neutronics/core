//! Integration tests for the multi-cell neutron transport
//! kernel and its CPU mirrors. Exercises the full pipeline
//! through `run_multi_cell_transport*`, plus a handful of
//! unit-level tests on the shared helpers (`log_edges_bin`,
//! `MT_SLOTS` layout, etc.). The yamc-physics flat samplers
//! are tested in their own `yamc-physics::flat::*::tests`
//! modules.

use super::*;
use crate::common::tallies::TallyVarianceMode;
// Issue #104: the per-MT distribution and elastic-angle fixtures now use the
// tight variable-length CSR layout (per-(material,MT) count / CSR-offset arrays
// stay `[n_slab * MT_INELASTIC_COUNT]` or `[n_slab]`, all per-row / per-point
// data arrays are sized exactly to the rows/points present). The old fixed
// per-family subsampling caps are gone; no-distribution fixtures have zero
// rows/points, so those data arrays are simply empty.
use crate::common::geometry::boundary_distance::{
    SURFACE_CONE, SURFACE_PARAM_STRIDE, SURFACE_QUADRIC, SURFACE_SPHERE, SURFACE_XTORUS,
    SURFACE_YTORUS,
};
use crate::common::geometry::region_eval::{
    REGION_OP_AND, REGION_OP_BELOW, REGION_OP_NOT, REGION_OP_SHIFT,
};
use crate::neutron::xs::{ANGLE_INTERP_HISTOGRAM, EOUT_KIND_MAXWELL, EOUT_KIND_WATT};
use crate::{GpuContext, GpuInitError};

/// Repack readable stride-7 surface rows (the fixture layout used
/// throughout this file) into the kernel's `SURFACE_PARAM_STRIDE`
/// layout by zero-padding each surface's unused tail. Keeps the
/// fixtures legible while the buffer the kernel reads is the full
/// width the general quadric needs.
fn pad_surface_params(rows7: &[f64]) -> Vec<f64> {
    assert_eq!(rows7.len() % 7, 0, "fixture rows must be 7 wide");
    rows7
        .as_chunks::<7>()
        .0
        .iter()
        .flat_map(|c| c.iter().copied().chain([0.0; SURFACE_PARAM_STRIDE - 7]))
        .collect()
}

/// Sanity: `KERNEL_STORAGE_BUFFER_COUNT` must match the actual
/// number of `&Array<...>` parameters in the
/// `multi_cell_transport_kernel` `#[cube]` signature. Drift
/// silently when someone adds or drops a buffer would defeat
/// the descriptor-budget guard at the launcher boundary.
/// We grep the source itself; the macro hides the function
/// from `inspect`-style introspection so source counting is
/// the cleanest available check.
#[test]
fn kernel_storage_buffer_count_matches_signature() {
    let src = include_str!("kernel.rs");
    let mut in_kernel = false;
    let mut count = 0u32;
    for line in src.lines() {
        if line.trim_start().starts_with("#[cube(launch_unchecked)]") {
            in_kernel = true;
            continue;
        }
        if in_kernel {
            if line.starts_with(") {") || line.starts_with(")\u{a0}{") {
                break;
            }
            let trimmed = line.trim_start();
            // Skip doc / comment lines -- only `&[` slice-parameter text
            // inside an actual parameter declaration counts. (cubecl 0.11
            // kernels take `&[T]`/`&mut [T]` slices; immutable `&[` params are
            // the read storage buffers. `&mut [` does not contain `&[`.)
            if trimmed.starts_with("//") || trimmed.starts_with("///") {
                continue;
            }
            if line.contains("&[") {
                count += 1;
            }
        }
    }
    assert_eq!(
        count, KERNEL_STORAGE_BUFFER_COUNT,
        "found {count} `&[...]` slice parameters in the kernel \
         signature, but KERNEL_STORAGE_BUFFER_COUNT is set to \
         {KERNEL_STORAGE_BUFFER_COUNT}. Update the constant in \
         lockstep with kernel-signature changes -- it gates the \
         GPU descriptor-budget guard at the launcher boundary."
    );
}

/// Build a `coarse_meta` (issue #212) for the single-nuclide, single-shared-
/// coarse-grid test layout: every material uses the same coarse grid at base 0
/// (`grid_offset = 0`, `coarse_n = coarse_len`), and since each material has one
/// nuclide its first-slab base into `permt_meta` ROWS is `m * MT_INELASTIC_COUNT`
/// (the sparse per-MT storage, issue #212). This reproduces the global
/// `slab * MT_INELASTIC_COUNT + slot` per-MT indexing exactly.
fn single_nuclide_coarse_meta(target_mass_per_material: &[f64], coarse_len: usize) -> Vec<u32> {
    let mut meta = Vec::with_capacity(target_mass_per_material.len() * 3);
    for m in 0..target_mass_per_material.len() {
        meta.push(0u32); // COL_COARSE_GRID_OFFSET
        meta.push(coarse_len as u32); // COL_COARSE_N
        meta.push((m * MT_INELASTIC_COUNT) as u32); // COL_COARSE_MT_BASE (permt row base)
    }
    meta
}

/// Build a `fine_log_energy_grid` (issue #212) for the single-nuclide,
/// shared-grid test layout: every material's fine grid IS `log_grid`, so the
/// per-material concatenation is `log_grid` repeated `n_mat` times. Material
/// `m`'s row then starts at `m * log_grid.len()`, matching the `[n_mat × n_grid]`
/// aggregate-XS buffers the fixtures build.
fn single_nuclide_fine_grid(log_grid: &[f64], n_mat: usize) -> Vec<f64> {
    let mut g = Vec::with_capacity(log_grid.len() * n_mat.max(1));
    for _ in 0..n_mat.max(1) {
        g.extend_from_slice(log_grid);
    }
    g
}

/// Build a `fine_meta` (issue #212) matching [`single_nuclide_fine_grid`]:
/// material `m` has grid/aggregate-XS base `m * n_grid`, fine length `n_grid`,
/// and (single nuclide per material) nuc first-slab base `m * n_grid`. Reproduces
/// the pre-#212 shared-grid indexing exactly.
fn single_nuclide_fine_meta(target_mass_per_material: &[f64], n_grid: usize) -> Vec<u32> {
    let mut meta = Vec::with_capacity(target_mass_per_material.len() * 3);
    for m in 0..target_mass_per_material.len() {
        meta.push((m * n_grid) as u32); // COL_FINE_GRID_OFFSET
        meta.push(n_grid as u32); // COL_FINE_N
        meta.push((m * n_grid) as u32); // COL_FINE_NUC_BASE (count 1 per material)
    }
    meta
}

fn constant_xs_grid(sigma_e: f64, sigma_a: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let xs_e: Vec<f64> = vec![sigma_e; n_grid];
    let xs_a: Vec<f64> = vec![sigma_a; n_grid];
    (log_grid, xs_e, xs_a)
}

/// All-zero per-MT angular distribution buffers padded to the
/// kernel's flat shape `[n_materials × MT_INELASTIC_COUNT × ...]`.
/// `angle_n_energies` stays zero, so the kernel's slice-B angular
/// branch is skipped entirely and the inelastic mu falls back to
/// isotropic-in-CM. Lets every test that doesn't care about
/// angular kinematics keep its existing semantics.
pub(super) struct DefaultAngleBuffers {
    pub n_energies: Vec<u32>,
    pub energy_grid: Vec<f64>,
    pub n_mu: Vec<u32>,
    pub mu: Vec<f64>,
    pub cdf: Vec<f64>,
    pub pdf: Vec<f64>,
    pub interp: Vec<u32>,
    // Slice C eout buffers, all zero-padded so every MT slot
    // defaults to LevelInelastic + closed-form energy.
    pub eout_kind: Vec<u32>,
    pub eout_n_energies: Vec<u32>,
    pub eout_ae_offset: Vec<u32>,
    pub eout_histogram_interp: Vec<u32>,
    pub eout_energy_grid: Vec<f64>,
    pub eout_n_x: Vec<u32>,
    pub eout_x_offset: Vec<u32>,
    pub eout_x: Vec<f64>,
    pub eout_p: Vec<f64>,
    pub eout_cdf: Vec<f64>,
    pub eout_interp: Vec<u32>,
    pub eout_n_discrete: Vec<u32>,
    // Slice D corr buffers, all zero-padded.
    pub corr_n_energies: Vec<u32>,
    pub corr_n_components: Vec<u32>,
    pub corr_ae_offset: Vec<u32>,
    pub corr_energy_grid: Vec<f64>,
    pub corr_n_x: Vec<u32>,
    pub corr_x_offset: Vec<u32>,
    pub corr_x: Vec<f64>,
    pub corr_cdf: Vec<f64>,
    pub corr_p: Vec<f64>,
    pub corr_interp: Vec<u32>,
    pub corr_n_discrete: Vec<u32>,
    pub corr_n_mu: Vec<u32>,
    pub corr_mu_offset: Vec<u32>,
    pub corr_mu: Vec<f64>,
    pub corr_mu_cdf: Vec<f64>,
    pub corr_mu_pdf: Vec<f64>,
    pub corr_mu_interp: Vec<u32>,
    pub scatter_in_cm: Vec<u32>,
    // Slice E km buffers, all zero-padded.
    pub km_n_energies: Vec<u32>,
    pub km_ae_offset: Vec<u32>,
    pub km_energy_grid: Vec<f64>,
    pub km_interp: Vec<u32>,
    pub km_n_discrete: Vec<u32>,
    pub km_n_x: Vec<u32>,
    pub km_x_offset: Vec<u32>,
    pub km_x: Vec<f64>,
    pub km_p: Vec<f64>,
    pub km_c: Vec<f64>,
    pub km_r: Vec<f64>,
    pub km_a: Vec<f64>,
    pub evap_n_energies: Vec<u32>,
    pub evap_n_components: Vec<u32>,
    pub evap_ae_offset: Vec<u32>,
    pub evap_theta_offset: Vec<u32>,
    pub evap_energy_grid: Vec<f64>,
    pub evap_theta: Vec<f64>,
    pub evap_u: Vec<f64>,
    pub nbps_n_bodies: Vec<u32>,
    pub nbps_total_mass: Vec<f64>,
    pub maxwell_n_energies: Vec<u32>,
    pub maxwell_ae_offset: Vec<u32>,
    pub maxwell_energy_grid: Vec<f64>,
    pub maxwell_theta: Vec<f64>,
    pub maxwell_u: Vec<f64>,
    pub watt_n_energies: Vec<u32>,
    pub watt_ae_offset: Vec<u32>,
    pub watt_energy_grid: Vec<f64>,
    pub watt_a: Vec<f64>,
    pub watt_b: Vec<f64>,
    pub watt_u: Vec<f64>,
}

pub(super) fn default_angle_buffers(n_materials: usize) -> DefaultAngleBuffers {
    // Tight CSR (issue #104): a no-distribution fixture has zero ae-rows and
    // zero points across every per-MT family, so the per-(material,MT) count /
    // CSR-offset arrays stay `[n_materials * MT_INELASTIC_COUNT]` (all zero)
    // and every per-row / per-point data array is empty. Mirrors
    // `GpuNuclideXs::void` and the `build_per_mt_*` tight builders.
    let n_slots = n_materials * MT_INELASTIC_COUNT;
    DefaultAngleBuffers {
        n_energies: vec![0u32; n_slots],
        energy_grid: Vec::new(),
        n_mu: Vec::new(),
        mu: Vec::new(),
        cdf: Vec::new(),
        pdf: Vec::new(),
        interp: Vec::new(),
        eout_kind: vec![0u32; n_slots],
        eout_n_energies: vec![0u32; n_slots],
        eout_ae_offset: vec![0u32; n_slots],
        eout_histogram_interp: vec![0u32; n_slots],
        eout_energy_grid: Vec::new(),
        eout_n_x: Vec::new(),
        eout_x_offset: Vec::new(),
        eout_x: Vec::new(),
        eout_p: Vec::new(),
        eout_cdf: Vec::new(),
        eout_interp: Vec::new(),
        eout_n_discrete: Vec::new(),
        corr_n_energies: vec![0u32; n_slots],
        corr_n_components: vec![0u32; n_slots],
        corr_ae_offset: vec![0u32; n_slots],
        corr_energy_grid: Vec::new(),
        corr_n_x: Vec::new(),
        corr_x_offset: Vec::new(),
        corr_x: Vec::new(),
        corr_cdf: Vec::new(),
        corr_p: Vec::new(),
        corr_interp: Vec::new(),
        corr_n_discrete: Vec::new(),
        corr_n_mu: Vec::new(),
        corr_mu_offset: Vec::new(),
        corr_mu: Vec::new(),
        corr_mu_cdf: Vec::new(),
        corr_mu_pdf: Vec::new(),
        corr_mu_interp: Vec::new(),
        scatter_in_cm: vec![0u32; n_slots],
        km_n_energies: vec![0u32; n_slots],
        km_ae_offset: vec![0u32; n_slots],
        km_energy_grid: Vec::new(),
        km_interp: Vec::new(),
        km_n_discrete: Vec::new(),
        km_n_x: Vec::new(),
        km_x_offset: Vec::new(),
        km_x: Vec::new(),
        km_p: Vec::new(),
        km_c: Vec::new(),
        km_r: Vec::new(),
        km_a: Vec::new(),
        evap_n_energies: vec![0u32; n_slots],
        evap_n_components: vec![0u32; n_slots],
        evap_ae_offset: vec![0u32; n_slots],
        evap_theta_offset: vec![0u32; n_slots],
        evap_energy_grid: Vec::new(),
        evap_theta: Vec::new(),
        evap_u: Vec::new(),
        nbps_n_bodies: vec![0u32; n_slots],
        nbps_total_mass: vec![0.0_f64; n_slots],
        maxwell_n_energies: vec![0u32; n_slots],
        maxwell_ae_offset: vec![0u32; n_slots],
        maxwell_energy_grid: Vec::new(),
        maxwell_theta: Vec::new(),
        maxwell_u: vec![0.0_f64; n_slots],
        watt_n_energies: vec![0u32; n_slots],
        watt_ae_offset: vec![0u32; n_slots],
        watt_energy_grid: Vec::new(),
        watt_a: Vec::new(),
        watt_b: Vec::new(),
        watt_u: vec![0.0_f64; n_slots],
    }
}

/// Build a log-uniform `tally_log_edges` array equivalent to the
/// pre-slice-F `(log_e_min, log_e_max, n_bins)` triple. Returns
/// `n_bins + 1` strictly increasing log-energy values.
pub(super) fn log_uniform_edges(log_e_min: f64, log_e_max: f64, n_bins: usize) -> Vec<f64> {
    assert!(n_bins >= 1);
    let step = (log_e_max - log_e_min) / n_bins as f64;
    (0..=n_bins)
        .map(|i| log_e_min + (i as f64) * step)
        .collect()
}

/// Test-fixture helper: build the standard 2-tally `Flux` +
/// `Absorption` pack covering every cell in `cell_aabbs` (cells
/// derived as `cell_aabbs.len() / 6`). Replaces the pre-multi-tally
/// `&log_uniform_edges(...)` call sites -- same energy bins, same
/// per-cell layout, but now expressed as a `TalliesPack`.
pub(super) fn flux_abs_pack_for(cell_aabbs: &[f64], log_edges: &[f64]) -> TalliesPack {
    TalliesPack::flux_abs_pack((cell_aabbs.len() / 6) as u32, log_edges)
}

/// Issue #234: the CPU-twin mesh-tally accumulator (`accumulate_tallies`)
/// fans one track-length step across the voxels it crosses, weighting each by
/// its in-voxel length fraction and writing into `out_off + (cell*n_bins +
/// energy) * n_mesh + voxel` (mesh innermost). This pins the flat-index
/// composition, the `d * length_fraction * score * weight` weighting, the
/// mesh-kind routing, and the fixed-point rounding against an independent
/// walk of the same segment (the voxel geometry itself is validated against
/// the real `RegularRectangularMesh` in yamc's `gpu_mesh_dda_twin` test).
#[test]
fn mesh_track_length_accumulates_per_voxel() {
    use crate::common::tallies::{
        rect_mesh_crossings, round_fixed_point_bits, DEFAULT_FIXED_POINT_SCALE, MESH_RECT_ROWMAJOR,
        SCORE_FLUX,
    };

    // 4x4x4 mesh over [0,4]^3 (unit voxels), packed as build_tallies_pack does.
    let mut params: Vec<f64> = Vec::new();
    params.extend_from_slice(&[0.0, 0.0, 0.0]); // lower_left
    params.extend_from_slice(&[4.0, 4.0, 4.0]); // upper_right
    params.extend_from_slice(&[1.0, 1.0, 1.0]); // inv_width
    params.extend_from_slice(&[1.0, 1.0, 1.0]); // width
    params.extend_from_slice(&[4.0, 4.0, 4.0]); // shape
    let n_mesh = 64u32;

    let n_geom_cells = 1usize;
    let pack = TalliesPack {
        score_kinds: vec![SCORE_FLUX],
        cell_to_bin: vec![0u32; n_geom_cells],
        n_cells_per_tally: vec![1],
        edges_offsets: vec![0],
        n_bins_per_tally: vec![1],
        log_edges: vec![f64::NEG_INFINITY, f64::INFINITY],
        out_offsets: vec![0, n_mesh],
        score_data: vec![0],
        score_mt: vec![0],
        fixed_point_scales: vec![DEFAULT_FIXED_POINT_SCALE],
        is_collision: vec![0],
        n_parent_per_tally: vec![1],
        parent_offsets: vec![0, 0],
        parent_ids: Vec::new(),
        n_mesh_per_tally: vec![n_mesh],
        mesh_kind: vec![MESH_RECT_ROWMAJOR],
        mesh_params_offsets: vec![0, params.len() as u32],
        mesh_params: params.clone(),
        efunc_offsets: vec![0, 0],
        efunc_params: Vec::new(),
    };

    let r0 = [0.3_f64, 0.4, 0.5];
    let r1 = [3.6_f64, 3.2, 2.9];
    let d = ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();
    let dir = [
        (r1[0] - r0[0]) / d,
        (r1[1] - r0[1]) / d,
        (r1[2] - r0[2]) / d,
    ];
    let weight = 0.75;

    let mut acc = vec![0u64; n_mesh as usize];
    accumulate_tallies(
        &pack,
        0,
        n_geom_cells,
        0.0,
        // Linear energy; this fixture's tally carries no energy-function
        // filter, so the value is never read.
        1.0,
        d,
        weight,
        1.0,
        0.0,
        0.0,
        0,
        1,
        0,
        0,
        0.0,
        &[],
        0,
        super::UrrScore::default(),
        r0,
        r1,
        dir,
        &mut acc,
    );

    // Independent expectation: SCORE_FLUX factor is 1.0, so each voxel gets
    // `round(d * length_fraction * weight * scale)` summed.
    let scale = DEFAULT_FIXED_POINT_SCALE;
    let mut expected = vec![0i64; n_mesh as usize];
    rect_mesh_crossings(&params, r0, r1, dir, |voxel, lf| {
        let contrib = d * lf * weight;
        expected[voxel as usize] += round_fixed_point_bits(contrib, scale) as i64;
    });
    for v in 0..n_mesh as usize {
        assert_eq!(acc[v] as i64, expected[v], "voxel {v} contribution");
    }

    // Sanity: the segment lies wholly inside the mesh, so the total scored
    // track length recovers `d * weight` to fixed-point precision.
    let total: f64 = acc.iter().map(|&b| (b as i64 as f64) / scale).sum();
    assert!(
        (total - d * weight).abs() < 1e-6,
        "total track length {total} != d*weight {}",
        d * weight
    );
}

/// Two adjacent unit cubes along the x-axis: cell 0 at [0,1]³,
/// cell 1 at [1,2]×[0,1]×[0,1]. 7 surfaces bound the union: 3
/// planes at x=0, x=1, x=2 plus 4 faces at y=0, y=1, z=0, z=1.
/// Particles start in cell 0 at (0.5, 0.5, 0.5) going +x;
/// constant XS keeps things deterministic.
#[test]
fn two_cube_geometry_runs_cleanly() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let cell_aabbs: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0: [0,1]^3
        1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1: [1,2]x[0,1]x[0,1]
    ];
    // Plane equations in n·x = d form. For the +x walls of cell 0
    // and +x walls of cell 1 we use the same plane on the shared
    // face.
    let surface_types = vec![1u32; 7];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0 (-x face of cell 0)
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1 (shared face)
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2 (+x face of cell 1)
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ];
    let (log_grid, xs_e, xs_a) = constant_xs_grid(2.0, 1.0);

    let n = 1000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    // Single-material setup: both cells map to material 0,
    // single XS curve in the per-material array.
    let cell_to_material: Vec<u32> = vec![0, 0];
    let target_mass_per_material: Vec<f64> = vec![12.0];
    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e,
        &xs_a,
        &vec![0.0_f64; xs_a.len()],
        &vec![0.0_f64; xs_a.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                        // fission_eout_energy_grid (no fission -> empty tight)
        &[],                        // fission_eout_n_x
        &[],                        // fission_eout_x_offset
        &[],                        // fission_eout_x
        &[],                        // fission_eout_cdf
        &[],                        // fission_eout_p
        &[],                        // fission_eout_interp
        &[],                        // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges((1e-3_f64).ln(), (1e3_f64).ln(), 1),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        200,
        400.0,
        TallyVarianceMode::PerStep,
    );

    // Sanity checks: the kernel ran, outputs are well-typed, and
    // each particle has a defined fate.
    let absorption_per_cell = r.absorption_per_cell();
    assert_eq!(r.alive.len(), n);
    assert_eq!(r.n_steps.len(), n);
    assert_eq!(r.final_energies.len(), n);
    assert_eq!(absorption_per_cell.len(), 2);

    let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
    let dead_count = n - alive_count;
    let mean_steps = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;

    println!(
        "multi-cell transport: alive at end = {alive_count} of {n}, mean steps = {mean_steps}, \
         tally cell 0 = {}, tally cell 1 = {}",
        absorption_per_cell[0], absorption_per_cell[1]
    );

    // With Σ_a/Σ_t = 1/3, particles eventually absorb or escape.
    // 200 steps is plenty.
    assert!(
        r.n_steps.iter().all(|&s| s <= 200),
        "n_steps must respect the cap"
    );
    assert!(
        dead_count > 0,
        "at least some particles should have terminated"
    );
    assert!(
        absorption_per_cell.iter().all(|&t| t >= 0.0),
        "all per-cell tallies must be non-negative"
    );
    let total_tally: f64 = absorption_per_cell.iter().sum();
    assert!(total_tally > 0.0, "total tally must be positive");
}

/// Same 2-cube geometry, but each cell gets a *different* material.
/// Cell 0 is a strong absorber (Σ_a dominates); cell 1 is a strong
/// scatterer (mostly elastic, tiny Σ_a). Particles start in cell 0
/// going +x. Expectation:
/// - Most particles absorb in cell 0 (~92% based on `Σ_t·d = 5·0.5
///   = 2.5`, so survival ≈ exp(−2.5) ≈ 0.08).
/// - Of those that reach cell 1, they scatter many times. Some
///   eventually escape, some hit the step cap.
/// - Tally per cell reflects each cell's own Σ_a -- proves the
///   per-cell material lookup is wired correctly. If the kernel
///   wrongly used material 0 in cell 1, cell 1's tally would be
///   much larger.
#[test]
fn two_distinct_materials() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let cell_aabbs: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0: absorber
        1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1: scatterer
    ];
    let surface_types = vec![1u32; 7];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1 (shared)
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ];

    // Build flat per-material XS arrays. Energy grid shared.
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();

    // Material 0: absorber. σ_e = 0.0, σ_a = 5.0.
    // Material 1: scatterer. σ_e = 5.0, σ_a = 0.1.
    let mut xs_e_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend(std::iter::repeat_n(0.0, n_grid));
    xs_e_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    let mut xs_a_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    xs_a_per_mat.extend(std::iter::repeat_n(0.1, n_grid));
    let target_mass_per_material: Vec<f64> = vec![12.0, 12.0];

    let cell_to_material: Vec<u32> = vec![0, 1];

    let n = 2000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges((1e-3_f64).ln(), (1e3_f64).ln(), 1),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        500,
        400.0,
        TallyVarianceMode::PerStep,
    );

    let abs_per_cell = r.absorption_per_cell();
    let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
    let absorbed_count = r
        .alive
        .iter()
        .zip(r.n_steps.iter())
        .filter(|(&a, &s)| a == 0 && s < 500)
        .count();
    let mean_steps = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;

    println!(
        "two materials: alive at cap = {alive_count}/{n}, absorbed = {absorbed_count}/{n}, \
         mean steps = {mean_steps}, tally cell 0 = {}, tally cell 1 = {}",
        abs_per_cell[0], abs_per_cell[1]
    );

    // The dominant pathway is absorption in cell 0. The expected
    // fraction is ~exp(−Σ_t × 0.5) = exp(−2.5) ≈ 0.08, so
    // ~92% absorb here. Loose bound: at least 70% absorbed.
    assert!(
        absorbed_count > (n * 7) / 10,
        "expected most particles to absorb (got {absorbed_count}/{n})"
    );

    // Cell 0 tally is `~N × Σ_a × <track length per particle in cell 0>`.
    // With Σ_a = 5 and mean entrance distance < 0.5, this is several
    // hundred to a few thousand. The exact value depends on physics
    // detail; we just check it's positive and large.
    assert!(
        abs_per_cell[0] > 100.0,
        "cell 0 absorber tally should be substantial, got {}",
        abs_per_cell[0]
    );

    // Cell 1 tally with Σ_a = 0.1 and the small fraction of
    // particles that reach it should be much smaller than cell 0.
    // The crucial check: it's NOT equal to what you'd get with
    // material 0's Σ_a = 5 in cell 1. With ~8% of particles
    // reaching cell 1 and bouncing, expected tally ~0.1 × tracks
    // ≈ low tens. Assert it's strictly smaller than cell 0 tally.
    assert!(
        abs_per_cell[1] < abs_per_cell[0],
        "cell 1 scatterer tally should be smaller than cell 0 absorber, got {} vs {}",
        abs_per_cell[1],
        abs_per_cell[0]
    );
}

/// Bit-exact CPU/GPU validation: run identical inputs through
/// both implementations and verify alive flags, n_steps, and the
/// fixed-point per-cell tally agree exactly. Final energies are
/// allowed to drift by a few ulps because GPU FMA contraction
/// can reassociate expressions like `mu_lab * dx + ...` and
/// because boundary scans accumulate rounding differently across
/// many steps.
/// Run identical inputs through the GPU kernel and the sequential CPU
/// mirror, returning both results. Shared by the CPU/GPU equivalence
/// tests. `surface_params` must already be at `SURFACE_PARAM_STRIDE`.
#[allow(clippy::too_many_arguments)]
fn run_equiv(
    ctx: &GpuContext,
    seeds: Vec<u32>,
    energies: Vec<f64>,
    positions: Vec<f64>,
    directions: Vec<f64>,
    cell_aabbs: Vec<f64>,
    cell_to_material: Vec<u32>,
    surface_types: Vec<u32>,
    surface_params: Vec<f64>,
    surface_boundaries: Vec<u32>,
    region_program: Vec<u32>,
    log_grid: Vec<f64>,
    xs_e_per_mat: Vec<f64>,
    xs_a_per_mat: Vec<f64>,
    target_mass_per_material: Vec<f64>,
    n_energy_bins: usize,
    max_steps: u32,
    survival: &SurvivalBiasingInputs,
) -> (MultiCellResult, MultiCellResult) {
    let log_min = log_grid[0];
    let log_max = *log_grid.last().unwrap();
    let gpu = run_multi_cell_transport(
        ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &surface_params,
        &surface_boundaries,
        &region_program,
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges(log_min, log_max, n_energy_bins),
        ),
        &[],
        survival,
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        max_steps,
        400.0,
        TallyVarianceMode::PerStep,
    );
    let (cpu, _cpu_trace) = run_multi_cell_transport_cpu(
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &surface_params,
        &surface_boundaries,
        &region_program,
        &log_grid,
        // #88 coarse grid: equal to the fine grid for this single-material fixture.
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges(log_min, log_max, n_energy_bins),
        ),
        &[],
        survival,
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        max_steps,
        400.0,
        false,
        PendDrain::Fifo,
    );

    (gpu, cpu)
}

/// Bit-exact CPU/GPU comparison: alive flags and fixed-point tallies
/// must match exactly, n_steps within 2%, final energies within 1024
/// ulps. Valid only for surface sets whose distance math is bit-exact
/// across the two backends (planes/spheres/cylinders).
fn assert_gpu_cpu_equiv(gpu: &MultiCellResult, cpu: &MultiCellResult, n: usize) {
    // Per-particle alive and n_steps should be byte-identical:
    // termination is driven by integer comparisons on PCG state
    // and well-separated probability thresholds. Any drift here
    // points at an algorithmic divergence, not floating-point.
    assert_eq!(gpu.alive, cpu.alive, "alive flags must match exactly");

    // Cell × energy fixed-point tally: integer sum, order-independent
    // because addition is associative on integers. Atomic adds
    // on the GPU and sequential `wrapping_add` on the CPU
    // produce the same final value at every (cell, bin).
    let gpu_tally_bits: Vec<i64> = gpu
        .absorption_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    let cpu_tally_bits: Vec<i64> = cpu
        .absorption_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    assert_eq!(
        gpu_tally_bits,
        cpu_tally_bits,
        "fixed-point per-(cell × bin) absorption tally must match exactly: gpu {:?} vs cpu {:?}",
        gpu.absorption_per_cell_energy(),
        cpu.absorption_per_cell_energy()
    );
    // Flux (raw track length) is also order-independent integer
    // accumulation; must match exactly too.
    let gpu_flux_bits: Vec<i64> = gpu
        .flux_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    let cpu_flux_bits: Vec<i64> = cpu
        .flux_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    assert_eq!(
        gpu_flux_bits, cpu_flux_bits,
        "fixed-point per-(cell × bin) flux tally must match exactly"
    );

    // n_steps may differ by at most 1 in rare cases where ulp
    // drift in d_xs vs d_boundary flips which event wins on a
    // borderline step. Count divergences; require the vast
    // majority to match exactly.
    let n_steps_mismatches = gpu
        .n_steps
        .iter()
        .zip(cpu.n_steps.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        n_steps_mismatches <= n / 50,
        "{n_steps_mismatches}/{n} n_steps mismatches exceeds 2% tolerance"
    );

    // Final energies: allow modest ulp drift accumulated across
    // many scatter events. With ~3 collisions per particle and a
    // few f64 ops per collision, a handful of ulps is expected.
    let mut max_ulp = 0u64;
    for (g, c) in gpu.final_energies.iter().zip(cpu.final_energies.iter()) {
        let gb = g.to_bits() as i64;
        let cb = c.to_bits() as i64;
        let ulp = (gb - cb).unsigned_abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    assert!(
        max_ulp < 1024,
        "final energy max ulp drift {max_ulp} too large"
    );
    println!(
        "cpu/gpu equivalence: tally exact, alive exact, \
         n_steps mismatches {n_steps_mismatches}/{n}, \
         max final-energy ulp drift {max_ulp}"
    );
}

#[test]
fn cpu_gpu_equivalence_two_materials() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let cell_aabbs: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0
        1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1
    ];
    let surface_types = vec![1u32; 7];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ];

    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let mut xs_e_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend(std::iter::repeat_n(0.0, n_grid));
    xs_e_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    let mut xs_a_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    xs_a_per_mat.extend(std::iter::repeat_n(0.1, n_grid));
    let target_mass_per_material: Vec<f64> = vec![12.0, 12.0];
    let cell_to_material: Vec<u32> = vec![0, 1];

    let n = 500usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let max_steps: u32 = 200;
    // Use 4 energy bins so the test exercises the per-cell-per-bin
    // indexing path (n_energy_bins=1 would collapse to the old
    // single-bin behaviour and miss any bin-axis bugs).
    let n_energy_bins = 4;
    let n_surf = surface_types.len();
    let n_cell = cell_aabbs.len() / 6;
    let (gpu, cpu) = run_equiv(
        &ctx,
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        pad_surface_params(&surface_params),
        vec![0u32; n_surf],     // all transmission boundaries
        vec![0u32; n_cell + 1], // identity region (AABB-only fixture)
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        target_mass_per_material,
        n_energy_bins,
        max_steps,
        &SurvivalBiasingInputs::off(),
    );
    assert_gpu_cpu_equiv(&gpu, &cpu, n);
}

/// End-to-end CPU/GPU equivalence with survival biasing ON, in a regime
/// where weight-cutoff Russian roulette actually fires. A scatter-dominated
/// cube (Sigma_e = 20, Sigma_a = 0.5) means implicit capture multiplies
/// weight by ~0.976 per collision and a particle collides many times before
/// escaping, so its weight crosses the `weight_cutoff = 0.25` trigger and the
/// roulette draw runs. The roulette decision is integer-PCG + f64
/// compare/divide/select on plane surfaces (bit-exact across backends), so
/// the GPU and CPU twins must agree bit-for-bit -- the same bar as the analog
/// equivalence tests. A survival-OFF baseline run confirms the VR path
/// actually changed the histories (so the roulette branch was exercised, not
/// dead).
#[test]
fn cpu_gpu_equivalence_survival_roulette() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // One cube cell [0,2]^3 bounded by six planes; scatter-dominated so
    // implicit capture decays the weight across many collisions.
    let cell_aabbs: Vec<f64> = vec![0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let surface_types = vec![1u32; 6];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, // y = 2
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, // z = 2
    ];

    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    // High scatter, modest capture -> implicit capture keeps the particle
    // alive while its weight bleeds off toward the cutoff.
    let xs_e_per_mat: Vec<f64> = std::iter::repeat_n(20.0, n_grid).collect();
    let xs_a_per_mat: Vec<f64> = std::iter::repeat_n(0.5, n_grid).collect();
    let target_mass_per_material: Vec<f64> = vec![12.0];
    let cell_to_material: Vec<u32> = vec![0];

    let n = 500usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for i in 0..n {
        positions.extend_from_slice(&[1.0, 1.0, 1.0]); // centre
        let f = i as f64;
        let z = 2.0 * (f + 0.5) / n as f64 - 1.0;
        let r = (1.0 - z * z).sqrt();
        let phi = std::f64::consts::TAU * f * 0.618_033_988_75;
        directions.extend_from_slice(&[r * phi.cos(), r * phi.sin(), z]);
    }

    let max_steps: u32 = 500;
    let n_energy_bins = 4;
    let n_surf = surface_types.len();
    let n_cell = cell_aabbs.len() / 6;

    // Default survival params: weight_cutoff 0.25, weight_survive 1.0.
    let survival = SurvivalBiasingInputs::on(0.25, 1.0);
    let (gpu, cpu) = run_equiv(
        &ctx,
        seeds.clone(),
        energies.clone(),
        positions.clone(),
        directions.clone(),
        cell_aabbs.clone(),
        cell_to_material.clone(),
        surface_types.clone(),
        pad_surface_params(&surface_params),
        vec![0u32; n_surf],     // transmission boundaries; escape via complement
        vec![0u32; n_cell + 1], // identity region (AABB-only fixture)
        log_grid.clone(),
        xs_e_per_mat.clone(),
        xs_a_per_mat.clone(),
        target_mass_per_material.clone(),
        n_energy_bins,
        max_steps,
        &survival,
    );
    // The roulette decision is bit-exact across backends on planes.
    assert_gpu_cpu_equiv(&gpu, &cpu, n);

    // Isolate the roulette: re-run with survival biasing STILL ON (implicit
    // capture unchanged) but a sub-threshold `weight_cutoff = 1e-300`, so the
    // `weight < weight_cutoff` gate never fires and no roulette draw happens.
    // The two runs differ ONLY in whether the roulette executes, so any
    // difference in the dead-particle count is attributable purely to the
    // roulette killing low-weight survivors. (A survival-OFF baseline would
    // conflate the roulette with the implicit-capture branch change.)
    let (gpu_norr, _cpu_norr) = run_equiv(
        &ctx,
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        pad_surface_params(&surface_params),
        vec![0u32; n_surf],
        vec![0u32; n_cell + 1],
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        target_mass_per_material,
        n_energy_bins,
        max_steps,
        &SurvivalBiasingInputs::on(1e-300, 1.0),
    );
    let dead_rr = gpu.alive.iter().filter(|&&a| a == 0).count();
    let dead_norr = gpu_norr.alive.iter().filter(|&&a| a == 0).count();
    assert!(
        dead_rr > dead_norr,
        "roulette must kill additional low-weight survivors: dead with roulette \
         {dead_rr}, dead without {dead_norr} (the roulette branch did not fire)"
    );
    println!(
        "survival roulette equivalence: GPU==CPU bit-exact; roulette killed \
         {} of {n} extra particles ({dead_rr} dead vs {dead_norr} without roulette)",
        dead_rr - dead_norr
    );
}

/// End-to-end CPU/GPU transport through curved bounding surfaces
/// (quadric, cone, X-torus, Y-torus) -- the PR's new dispatch arms in
/// kernel.rs / shared.rs that the plane-only equivalence test never
/// touches. Unlike planes, the quartic torus and quadric/cone quadratic
/// are not bit-exact across backends (~1e-11 FMA drift), so a few
/// grazing trajectories diverge; we assert the aggregate tallies and
/// survivor count agree closely rather than bit-for-bit, plus a
/// positive-tally smoke check that the new arms actually executed.
#[test]
fn cpu_gpu_equivalence_curved_surfaces() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };
    // One cell spanning the surface set; a scattering material so
    // particles cross the curved boundaries many times per history.
    let cell_aabbs: Vec<f64> = vec![-5.0, -5.0, -5.0, 5.0, 5.0, 5.0];
    let cell_to_material: Vec<u32> = vec![0];
    let surface_types = vec![
        SURFACE_QUADRIC,
        SURFACE_CONE,
        SURFACE_XTORUS,
        SURFACE_YTORUS,
    ];
    // Built directly at stride-10 (the general quadric needs all ten slots,
    // so this cannot go through the stride-7 pad_surface_params fixture).
    let surface_params = vec![
        // quadric: sphere x^2 + y^2 + z^2 - 9 = 0 (r = 3)
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.0, //
        // cone: apex (0,0,-4), axis +z, tan^2 = 0.5
        0.0, 0.0, -4.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, //
        // x-torus: major 3, minor 0.8
        0.0, 0.0, 0.0, 3.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, //
        // y-torus: major 3, minor 0.8
        0.0, 0.0, 0.0, 3.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, //
    ];
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    // Long mean free path (1/0.35 ~ 2.9) comparable to the surface radii
    // (~2.2-3), so particles frequently reach and cross the curved
    // boundaries before colliding -- exercising the d_boundary-wins path,
    // not just the per-step distance computation.
    let xs_e_per_mat = vec![0.3; n_grid]; // scattering
    let xs_a_per_mat = vec![0.05; n_grid]; // light absorption
    let target_mass_per_material = vec![12.0];

    let n = 500usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for i in 0..n {
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        // Fibonacci sphere: deterministic, near-uniform unit directions so
        // rays reach all four curved surfaces in the set.
        let f = i as f64;
        let z = 2.0 * (f + 0.5) / n as f64 - 1.0;
        let r = (1.0 - z * z).sqrt();
        let phi = std::f64::consts::TAU * f * 0.618_033_988_75;
        directions.extend_from_slice(&[r * phi.cos(), r * phi.sin(), z]);
    }

    let n_surf = surface_types.len();
    let n_cell = cell_aabbs.len() / 6;
    let (gpu, cpu) = run_equiv(
        &ctx,
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        vec![0u32; n_surf],     // all transmission boundaries
        vec![0u32; n_cell + 1], // identity region (AABB-only fixture)
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        target_mass_per_material,
        4,
        200,
        &SurvivalBiasingInputs::off(),
    );

    // Smoke: the curved arms ran and produced finite, positive physics.
    let gpu_abs: f64 = gpu.absorption_per_cell_energy().iter().sum();
    let cpu_abs: f64 = cpu.absorption_per_cell_energy().iter().sum();
    assert!(
        gpu_abs.is_finite() && gpu_abs > 0.0,
        "GPU absorption {gpu_abs}"
    );
    assert!(
        cpu_abs.is_finite() && cpu_abs > 0.0,
        "CPU absorption {cpu_abs}"
    );

    // Tolerant agreement: a permutation/stride mistake in one backend's
    // curved-surface dispatch would blow the aggregate far past this; the
    // ~1e-11 per-step drift only perturbs a few grazing histories.
    let rel = (gpu_abs - cpu_abs).abs() / cpu_abs;
    assert!(
        rel < 0.05,
        "GPU/CPU aggregate absorption disagree by {rel} (gpu {gpu_abs}, cpu {cpu_abs})"
    );
    let gpu_alive = gpu.alive.iter().filter(|&&a| a == 1).count() as i64;
    let cpu_alive = cpu.alive.iter().filter(|&&a| a == 1).count() as i64;
    assert!(
        (gpu_alive - cpu_alive).abs() <= (n as i64) / 20,
        "GPU/CPU survivor counts diverge: {gpu_alive} vs {cpu_alive}"
    );
    println!(
        "curved-surface equivalence: gpu_abs={gpu_abs} cpu_abs={cpu_abs} rel={rel} alive {gpu_alive}/{cpu_alive}"
    );
}

/// D10: GPU vs CPU across an INTERNAL transmission boundary in a real
/// CSG multi-cell geometry (not the AABB-only identity fixture every
/// other equivalence test uses). Two concentric spheres of DIFFERENT
/// materials sharing one internal transmission surface, with a vacuum
/// outer boundary and an isotropic point source at the origin:
///   surface 0: sphere r = 2  (internal transmission boundary)
///   surface 1: sphere r = 5  (vacuum outer boundary)
///   cell 0 (inner ball):  below s0                  -> material 0
///   cell 1 (outer shell): above s0 AND below s1     -> material 1
/// Every history streams out from r = 0 and MUST cross the internal
/// r = 2 surface, handing off cell 0 -> cell 1. The per-cell flux
/// (track-length) and absorption (track-length) tallies are compared
/// bit-for-bit: a double-count, lost track length, or a mis-identified
/// next cell at the crossing would shift a cell's integer tally and
/// fail. Spheres are bit-exact across backends (quadratic, no FMA
/// drift), so exact parity is the right bar here.
#[test]
fn cpu_gpu_equivalence_concentric_transmission_boundary() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // RPN region-program word builders (see region_eval). "inside a
    // sphere" is `below` its surface; the shell is the complement of the
    // inner ball intersected with the outer ball.
    let below = |idx: u32| (REGION_OP_BELOW << REGION_OP_SHIFT) | idx;
    let not = || REGION_OP_NOT << REGION_OP_SHIFT;
    let and = || REGION_OP_AND << REGION_OP_SHIFT;

    // Both cells' AABBs straddle the origin (the inner ball's bbox and
    // the shell's full bounding box both contain r = 0), so the cheap
    // AABB pre-filter cannot separate them -- the CSG region test must.
    let cell_aabbs: Vec<f64> = vec![
        -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, // cell 0: inner ball bbox
        -5.0, -5.0, -5.0, 5.0, 5.0, 5.0, // cell 1: outer shell bbox
    ];
    let cell_to_material: Vec<u32> = vec![0, 1];

    let surface_types = vec![SURFACE_SPHERE, SURFACE_SPHERE];
    // Sphere params: (cx, cy, cz, r) then zero-pad to stride 10.
    let mut surface_params: Vec<f64> = Vec::new();
    for &r in &[2.0_f64, 5.0_f64] {
        surface_params.extend_from_slice(&[0.0, 0.0, 0.0, r]);
        surface_params.extend(std::iter::repeat_n(0.0, SURFACE_PARAM_STRIDE - 4));
    }
    // s0 = internal transmission boundary (0), s1 = vacuum outer (1).
    let surface_boundaries = vec![0u32, BOUNDARY_VACUUM];

    // region_program: 2-cell header (offsets) + ops.
    //   cell 0 = below s0                       -> ops [below 0]
    //   cell 1 = (NOT below s0) AND (below s1)   -> ops [below 0, not, below 1, and]
    let ops = vec![below(0), below(0), not(), below(1), and()];
    let offsets = vec![0u32, 1, 5]; // cell0 ops [0,1), cell1 ops [1,5)
    let mut region_program = offsets;
    region_program.extend_from_slice(&ops);

    // Two materials with clearly different cross-sections so a handoff
    // error (a segment scored in the wrong cell) is unmistakable. Mat 0
    // (inner) is a strong absorber; mat 1 (shell) is a strong scatterer.
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let mut xs_e_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend(std::iter::repeat_n(0.3, n_grid)); // mat 0 scatter
    xs_e_per_mat.extend(std::iter::repeat_n(1.0, n_grid)); // mat 1 scatter
    let mut xs_a_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend(std::iter::repeat_n(0.8, n_grid)); // mat 0 absorb
    xs_a_per_mat.extend(std::iter::repeat_n(0.2, n_grid)); // mat 1 absorb
    let target_mass_per_material: Vec<f64> = vec![56.0, 12.0]; // Fe-ish / C-ish

    let n = 2000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for i in 0..n {
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        // Fibonacci-sphere isotropic point source.
        let f = i as f64;
        let z = 2.0 * (f + 0.5) / n as f64 - 1.0;
        let r = (1.0 - z * z).sqrt();
        let phi = std::f64::consts::TAU * f * 0.618_033_988_75;
        directions.extend_from_slice(&[r * phi.cos(), r * phi.sin(), z]);
    }

    let n_energy_bins = 4;
    let max_steps: u32 = 400;
    let (gpu, cpu) = run_equiv(
        &ctx,
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
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        target_mass_per_material,
        n_energy_bins,
        max_steps,
        &SurvivalBiasingInputs::off(),
    );

    // Bit-exact per-(cell x bin) flux and absorption across the internal
    // transmission boundary, plus exact alive flags and bounded n_steps /
    // final-energy drift (spheres are bit-exact, so this is the strong bar).
    assert_gpu_cpu_equiv(&gpu, &cpu, n);

    // Both cells must see real, positive flux: the inner ball is crossed by
    // every history and the shell is reached by every survivor -- a handoff
    // that lost particles at r = 2 would zero out the shell tally.
    let gpu_flux = gpu.flux_per_cell();
    assert_eq!(gpu_flux.len(), 2, "two cells");
    assert!(
        gpu_flux[0] > 0.0 && gpu_flux[1] > 0.0,
        "both cells must accumulate flux: inner={} shell={}",
        gpu_flux[0],
        gpu_flux[1]
    );
    println!(
        "concentric transmission-boundary equivalence: inner flux {:.4}, shell flux {:.4}",
        gpu_flux[0], gpu_flux[1]
    );
}

/// Energy-binned tally lands contributions in the right bin and
/// doesn't bleed into neighbours. Single cell, constant XS, all
/// particles start at a known energy → with no scatter, every
/// score falls into one bin (the first one's). With scatter
/// enabled, energy decreases monotonically so contributions
/// fall into bins ≤ the starting bin.
#[test]
fn energy_bin_assignment_is_correct() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => return,
    };

    // Single big cell so every track scores into one cell.
    let cell_aabbs: Vec<f64> = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
    let cell_to_material: Vec<u32> = vec![0];
    let surface_types = vec![1u32; 6];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, // x = 10
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, // y = 10
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, // z = 10
    ];

    // Pure absorber so particles die on first collision; no
    // scatter means every track scores into the starting energy
    // bin. σ_e = 0, σ_a = 1.
    let (log_grid, xs_e, xs_a) = constant_xs_grid(0.0, 1.0);
    let target_mass_per_material = vec![12.0];

    // Energy 1.0 (log_e = 0) sits EXACTLY on the bin-3/4 edge of an
    // 8-bin grid spanning [ln(1e-3), ln(1e3)] = [-6.9, 6.9] (the
    // midpoint). Under the `(lo, hi]` upper-edge-inclusive convention
    // (matching the CPU `EnergyFilter::get_bin`), a value exactly at an
    // interior edge scores into the LOWER bin -> bin 3.
    let log_min = (1e-3_f64).ln();
    let log_max = (1e3_f64).ln();
    let n_bins = 8usize;
    let starting_log_e = (1.0_f64).ln(); // = 0.0, exactly on an edge
    let expected_bin =
        ((starting_log_e - log_min) / (log_max - log_min) * n_bins as f64) as usize - 1; // edge value -> lower bin

    let n = 200usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[5.0, 5.0, 5.0]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e,
        &xs_a,
        &vec![0.0_f64; xs_a.len()],
        &vec![0.0_f64; xs_a.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                        // fission_eout_energy_grid (no fission -> empty tight)
        &[],                        // fission_eout_n_x
        &[],                        // fission_eout_x_offset
        &[],                        // fission_eout_x
        &[],                        // fission_eout_cdf
        &[],                        // fission_eout_p
        &[],                        // fission_eout_interp
        &[],                        // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(&cell_aabbs, &log_uniform_edges(log_min, log_max, n_bins)),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        50,
        400.0,
        TallyVarianceMode::PerStep,
    );

    // With σ_e = 0, every particle absorbs at its first
    // collision, taking exactly one step at the starting energy.
    // All tally goes into `expected_bin`; every other bin is 0.
    assert_eq!(r.absorption_per_cell_energy().len(), n_bins);
    for (b, &v) in r.absorption_per_cell_energy().iter().enumerate() {
        if b == expected_bin {
            assert!(v > 0.0, "bin {b} (expected) is empty");
        } else {
            assert_eq!(v, 0.0, "bin {b} got {v}, expected 0");
        }
    }
}

/// Flux tally consistency: flux × σ_a should equal absorption.
/// With σ_a constant this is a per-step identity, so bin-summed
/// `absorption_per_cell ≈ σ_a × flux_per_cell`. Loose tolerance
/// because both are integer-rounded at slightly different scales.
#[test]
fn flux_times_sigma_a_matches_absorption() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => return,
    };
    let cell_aabbs: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let cell_to_material: Vec<u32> = vec![0];
    let surface_types = vec![1u32; 6];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ];

    // σ_a = 0.5 constant so absorption == 0.5 × flux per step.
    let (log_grid, xs_e, xs_a) = constant_xs_grid(2.0, 0.5);
    let target_mass_per_material = vec![12.0];

    let n = 500usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e,
        &xs_a,
        &vec![0.0_f64; xs_a.len()],
        &vec![0.0_f64; xs_a.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                        // fission_eout_energy_grid (no fission -> empty tight)
        &[],                        // fission_eout_n_x
        &[],                        // fission_eout_x_offset
        &[],                        // fission_eout_x
        &[],                        // fission_eout_cdf
        &[],                        // fission_eout_p
        &[],                        // fission_eout_interp
        &[],                        // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges((1e-3_f64).ln(), (1e3_f64).ln(), 1),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        200,
        400.0,
        TallyVarianceMode::PerStep,
    );
    let abs = r.absorption_per_cell();
    let flux = r.flux_per_cell();
    assert!(flux[0] > 0.0, "flux must be positive");
    let ratio = abs[0] / flux[0];
    assert!(
        (ratio - 0.5).abs() < 1e-6,
        "absorption / flux should equal σ_a = 0.5, got {ratio}"
    );
}

/// Rayon CPU mirror must produce identical fixed-point tallies
/// to the sequential CPU mirror -- addition is associative on
/// integers, so per-thread accumulation + reduction is byte-
/// equal to sequential `wrapping_add`.
#[test]
fn cpu_rayon_matches_cpu_seq() {
    let cell_aabbs: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
    let cell_to_material: Vec<u32> = vec![0, 1];
    let surface_types = vec![1u32; 7];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let mut xs_e_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend(std::iter::repeat_n(2.0, n_grid));
    xs_e_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    let mut xs_a_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend(std::iter::repeat_n(1.0, n_grid));
    xs_a_per_mat.extend(std::iter::repeat_n(0.1, n_grid));
    let target_mass = vec![12.0, 12.0];

    let n = 500usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let (seq, _seq_trace) = run_multi_cell_transport_cpu(
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        // #88 coarse grid: equal to the fine grid for this single-material fixture.
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass.len()),
        &single_nuclide_fine_meta(&target_mass, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass,
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],
        &[],
        &[], // eout_x_offset
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                 // eout_p
        &[],                                                 // eout_interp
        &[],                                                 // eout_n_discrete
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],
        &[],
        &[], // corr_x_offset
        &[],
        &[],
        &[], // corr_p
        &[], // corr_interp
        &[], // corr_n_discrete
        &[],
        &[], // corr_mu_offset
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len()],
        &vec![0u32; target_mass.len()],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len()],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],
        &[],
        &[],
        &[],
        &[], // km_x_offset
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * URR_META_COLS],
        &vec![0u32; target_mass.len()], // urr_ae_offset
        &vec![0u32; target_mass.len()], // urr_cdf_offset
        &Vec::<f64>::new(),             // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),             // urr_cdf
        &Vec::<f64>::new(),             // urr_xs
        &vec![0.0_f64; target_mass.len()],
        &flux_abs_pack_for(&cell_aabbs, &log_uniform_edges(log_e_min, log_e_max, 8)),
        &[],
        &SurvivalBiasingInputs::off(),
        &NuclideSelectInputs::single_nuclide(&target_mass, log_grid.len()),
        &FissionBankInputs::off(),
        200,
        400.0,
        false,
        PendDrain::Fifo,
    );
    let par = run_multi_cell_transport_cpu_rayon(
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        // #88 coarse grid: equal to the fine grid for this single-material fixture.
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass.len()),
        &single_nuclide_fine_meta(&target_mass, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass,
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],
        &[],
        &[], // eout_x_offset
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                 // eout_p
        &[],                                                 // eout_interp
        &[],                                                 // eout_n_discrete
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],
        &[],
        &[], // corr_x_offset
        &[],
        &[],
        &[], // corr_p
        &[], // corr_interp
        &[], // corr_n_discrete
        &[],
        &[], // corr_mu_offset
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len()],
        &vec![0u32; target_mass.len()],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len()],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],
        &[],
        &[],
        &[],
        &[], // km_x_offset
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * MT_INELASTIC_COUNT],
        &[],
        &[],
        &[],
        &vec![0.0_f64; target_mass.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass.len() * URR_META_COLS],
        &vec![0u32; target_mass.len()], // urr_ae_offset
        &vec![0u32; target_mass.len()], // urr_cdf_offset
        &Vec::<f64>::new(),             // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),             // urr_cdf
        &Vec::<f64>::new(),             // urr_xs
        &vec![0.0_f64; target_mass.len()],
        &flux_abs_pack_for(&cell_aabbs, &log_uniform_edges(log_e_min, log_e_max, 8)),
        &[],
        &SurvivalBiasingInputs::off(),
        &NuclideSelectInputs::single_nuclide(&target_mass, log_grid.len()),
        &FissionBankInputs::off(),
        200,
        400.0,
    );

    assert_eq!(seq.alive, par.alive);
    assert_eq!(seq.n_steps, par.n_steps);
    let seq_bits: Vec<i64> = seq
        .absorption_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    let par_bits: Vec<i64> = par
        .absorption_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    assert_eq!(seq_bits, par_bits, "rayon tally must match seq exactly");
    let seq_flux: Vec<i64> = seq
        .flux_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    let par_flux: Vec<i64> = par
        .flux_per_cell_energy()
        .iter()
        .map(|&v| (v * 1_073_741_824.0).round() as i64)
        .collect();
    assert_eq!(seq_flux, par_flux, "rayon flux must match seq exactly");
    for (s, p) in seq.final_energies.iter().zip(par.final_energies.iter()) {
        assert_eq!(s.to_bits(), p.to_bits());
    }
}

/// Toroidal cell: a single AABB cube with a circular ZTorus
/// embedded at the origin (a=2, b=c=0.5). Particles start at
/// (0, 0, 0) (inside the AABB but outside the torus tube)
/// going +x. Smoke test that the kernel handles surface_type=3
/// without crashing and produces a positive tally.
#[test]
fn ztorus_in_multi_cell_geometry() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => return,
    };
    let cell_aabbs: Vec<f64> = vec![-3.0, -3.0, -3.0, 3.0, 3.0, 3.0];
    let cell_to_material: Vec<u32> = vec![0];
    let surface_types = vec![1u32, 1, 1, 1, 1, 1, 3];
    let surface_params = vec![
        -1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, // x = -3
        1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, // x = 3
        0.0, -1.0, 0.0, -3.0, 0.0, 0.0, 0.0, // y = -3
        0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, // y = 3
        0.0, 0.0, -1.0, -3.0, 0.0, 0.0, 0.0, // z = -3
        0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, // z = 3
        0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, // torus a=2, b=c=0.5
    ];
    let (log_grid, xs_e, xs_a) = constant_xs_grid(2.0, 1.0);
    let target_mass_per_material = vec![12.0];

    let n = 100usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        // Start at origin (centre of torus hole) going +x.
        // First surface hit going +x is the inner torus wall
        // at x = a − c = 1.5.
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e,
        &xs_a,
        &vec![0.0_f64; xs_a.len()],
        &vec![0.0_f64; xs_a.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                        // fission_eout_energy_grid (no fission -> empty tight)
        &[],                        // fission_eout_n_x
        &[],                        // fission_eout_x_offset
        &[],                        // fission_eout_x
        &[],                        // fission_eout_cdf
        &[],                        // fission_eout_p
        &[],                        // fission_eout_interp
        &[],                        // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges((1e-3_f64).ln(), (1e3_f64).ln(), 1),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        200,
        400.0,
        TallyVarianceMode::PerStep,
    );
    let abs_per_cell = r.absorption_per_cell();
    assert!(
        abs_per_cell[0] > 0.0,
        "expected positive tally with torus surface, got {}",
        abs_per_cell[0]
    );
    let alive = r.alive.iter().filter(|&&a| a == 1).count();
    assert!(alive < n, "some particles must terminate");
}

/// Cylindrical cell: a single AABB containing a z-axis cylinder
/// of radius 0.4 plus six bounding planes for the AABB. Particles
/// start at (0, 0, 0) heading +x; with constant XS they should
/// hit either the cylinder wall (at distance 0.4) or scatter
/// before. Smoke test that the kernel handles surface_type=2
/// without crashing and produces a positive tally.
#[test]
fn cylinder_in_multi_cell_geometry() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => return,
    };

    let cell_aabbs: Vec<f64> = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let cell_to_material: Vec<u32> = vec![0];
    // Six AABB planes plus a z-axis cylinder of radius 0.4
    // centred on the z-axis. The kernel reports min distance to
    // any surface, so the cylinder wall will dominate boundary
    // distance for outward-going particles.
    let surface_types = vec![1u32, 1, 1, 1, 1, 1, 2];
    let surface_params = vec![
        -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, // x = -1
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1
        0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, // y = -1
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, // z = -1
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
        0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 1.0, // cyl r=0.4 axis +z
    ];

    let (log_grid, xs_e, xs_a) = constant_xs_grid(2.0, 1.0);
    let target_mass_per_material = vec![12.0];

    let n = 200usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e,
        &xs_a,
        &vec![0.0_f64; xs_a.len()],
        &vec![0.0_f64; xs_a.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                        // fission_eout_energy_grid (no fission -> empty tight)
        &[],                        // fission_eout_n_x
        &[],                        // fission_eout_x_offset
        &[],                        // fission_eout_x
        &[],                        // fission_eout_cdf
        &[],                        // fission_eout_p
        &[],                        // fission_eout_interp
        &[],                        // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges((1e-3_f64).ln(), (1e3_f64).ln(), 1),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        200,
        400.0,
        TallyVarianceMode::PerStep,
    );

    let abs_per_cell = r.absorption_per_cell();
    // The kernel ran without faulting and produced a positive
    // tally -- the cylinder surface is being processed alongside
    // the planes.
    assert!(
        abs_per_cell[0] > 0.0,
        "expected positive tally with cylinder surface, got {}",
        abs_per_cell[0]
    );
    // Mean step distance must be < cylinder radius for at least
    // some particles (otherwise the cylinder is being ignored
    // and they'd be hitting the +x AABB face at d=1.0 instead).
    // With Σ_t = 3 the mean free path is ~0.33 < 0.4, so most
    // collisions happen before reaching the cylinder wall.
    let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
    assert!(
        alive_count < n,
        "some particles must terminate (collide or escape)"
    );
}

/// Real ENDF data: Be9 (pure scatterer) and Pb208 (heavy
/// scatterer with modest capture). Loads both from the
/// committed Arrow fixtures in `crates/yamc/tests/`, resamples
/// Pb208 onto Be9's energy grid, and runs the multi-cell
/// kernel on a two-cube geometry. The point is to prove the
/// kernel handles real (non-constant, threshold-bounded,
/// resonance-shaped) cross sections -- not to validate exact
/// rates against another code.
#[test]
fn real_endf_two_material_geometry() {
    use crate::neutron::xs::{MT_CAPTURE, MT_ELASTIC};
    use yamc_nuclide::nuclide_loader::load_nuclide;

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let manifest = env!("CARGO_MANIFEST_DIR");
    let be9_path = format!("{manifest}/../yamc/tests/Be9.arrow");
    let pb_path = format!("{manifest}/../yamc/tests/Pb208.arrow");
    let be9 = match load_nuclide(&be9_path, &yamc_nuclide::LoadScope::full()) {
        Ok(n) => n,
        Err(e) => {
            println!("Be9.arrow load failed ({e}); skipping");
            return;
        }
    };
    let pb208 = match load_nuclide(&pb_path, &yamc_nuclide::LoadScope::full()) {
        Ok(n) => n,
        Err(e) => {
            println!("Pb208.arrow load failed ({e}); skipping");
            return;
        }
    };

    // Pick a temperature both nuclides have. The fixtures all
    // carry "294" (room temperature), so this is the safe pick.
    let temp = "294";
    let be9_temp_idx = be9.get_temp_idx(temp).expect("Be9 missing 294");
    let pb_temp_idx = pb208.get_temp_idx(temp).expect("Pb208 missing 294");
    let be9_grid = be9
        .energy
        .as_ref()
        .and_then(|e| e.get(temp))
        .expect("Be9 missing energy grid")
        .clone();
    let log_grid: Vec<f64> = be9_grid.iter().map(|e| e.ln()).collect();
    let n_grid = be9_grid.len();

    // Resample each nuclide's elastic + capture onto Be9's grid
    // via Reaction::cross_section_at -- same path extract_material_xs
    // uses internally.
    let sample = |nuclide: &yamc_nuclide::nuclide::Nuclide, idx: usize| {
        let reactions = &nuclide.reactions[idx];
        let elastic = reactions.get(&MT_ELASTIC).expect("MT=2 missing");
        let capture = reactions.get(&MT_CAPTURE).expect("MT=102 missing");
        let xs_e: Vec<f64> = be9_grid
            .iter()
            .map(|&e| elastic.cross_section_at(e).unwrap_or(0.0))
            .collect();
        let xs_a: Vec<f64> = be9_grid
            .iter()
            .map(|&e| capture.cross_section_at(e).unwrap_or(0.0))
            .collect();
        let mass = nuclide.atomic_weight_ratio.expect("missing AWR");
        (xs_e, xs_a, mass)
    };
    let (be9_xs_e, be9_xs_a, be9_mass) = sample(&be9, be9_temp_idx);
    let (pb_xs_e, pb_xs_a, pb_mass) = sample(&pb208, pb_temp_idx);

    // Both arrays must be the same length -- they were sampled
    // on the same grid, so this is a tautology, but cheap to
    // assert as documentation.
    assert_eq!(be9_xs_e.len(), n_grid);
    assert_eq!(pb_xs_e.len(), n_grid);

    // Per-cell-material flat buffers, material-major.
    let mut xs_e_per_mat = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend_from_slice(&be9_xs_e);
    xs_e_per_mat.extend_from_slice(&pb_xs_e);
    let mut xs_a_per_mat = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend_from_slice(&be9_xs_a);
    xs_a_per_mat.extend_from_slice(&pb_xs_a);
    let target_mass_per_material = vec![be9_mass, pb_mass];

    // Two-cube geometry: cell 0 = Be9, cell 1 = Pb208. Same
    // surface set as the constant-XS multi-material test.
    let cell_aabbs: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0: Be9
        1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1: Pb208
    ];
    let cell_to_material: Vec<u32> = vec![0, 1];
    let surface_types = vec![1u32; 7];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ];

    // Source: 1000 particles at 1 MeV (= 1e6 eV) starting at
    // (0.5, 0.5, 0.5) inside cell 0, going +x. ENDF grids are
    // in eV so this falls comfortably inside the loaded range.
    let n = 1000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0e6; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    // Bracket the ENDF energy range so all bins are populated.
    let tally_log_min = *log_grid.first().expect("empty grid");
    let tally_log_max = *log_grid.last().expect("empty grid");
    let r = run_multi_cell_transport(
        &ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; surface_types.len()],
        &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &vec![0.0_f64; xs_a_per_mat.len()],
        &vec![0.0_f64; xs_a_per_mat.len()], // xs_fission_per_material (no fission)
        &vec![0.0_f64; xs_a_per_mat.len()], // nu_bar_per_material
        &vec![0.0_f64; xs_a_per_mat.len()], // beta_delayed_per_material
        &vec![0.988e6_f64; target_mass_per_material.len()], // fission_a (Watt default)
        &vec![2.249e-6_f64; target_mass_per_material.len()], // fission_b (Watt default)
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
        &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
        &[],                                // fission_eout_energy_grid (no fission -> empty tight)
        &[],                                // fission_eout_n_x
        &[],                                // fission_eout_x_offset
        &[],                                // fission_eout_x
        &[],                                // fission_eout_cdf
        &[],                                // fission_eout_p
        &[],                                // fission_eout_interp
        &[],                                // xs_inelastic_per_mt_sparse (no inelastic)
        &target_mass_per_material,
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &[], // yield_per_mt_sparse (no inelastic)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize], // permt_meta
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-row / per-point data arrays are empty and only the
        // per-(material,MT) count / offset arrays carry length.
        &[], // angle_energy_grid
        &[], // angle_n_mu
        &[], // angle_mu_offset
        &[], // angle_mu
        &[], // angle_cdf
        &[], // angle_pdf
        &[], // angle_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
        &[],                                                              // eout_energy_grid
        &[],                                                              // eout_n_x
        &[],                                                              // eout_x_offset
        &[],                                                              // eout_x
        &[],                                                              // eout_cdf
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_histogram_interp
        &[],                                                              // eout_p
        &[],                                                              // eout_interp
        &[],                                                              // eout_n_discrete
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
        &[],                                                              // corr_energy_grid
        &[],                                                              // corr_n_x
        &[],                                                              // corr_x_offset
        &[],                                                              // corr_x
        &[],                                                              // corr_cdf
        &[],                                                              // corr_p
        &[],                                                              // corr_interp
        &[],                                                              // corr_n_discrete
        &[],                                                              // corr_n_mu
        &[],                                                              // corr_mu_offset
        &[],                                                              // corr_mu
        &[],                                                              // corr_mu_cdf
        &[],                                                              // corr_mu_pdf
        &[],                                                              // corr_mu_interp
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len()], // elastic_angle_ae_offset
        &[],                                         // elastic_angle_energy_grid
        &[],                                         // elastic_angle_n_mu
        &[],                                         // elastic_angle_mu_offset
        &[],                                         // elastic_angle_mu
        &[],                                         // elastic_angle_cdf
        &[],                                         // elastic_angle_pdf
        &[],                                         // elastic_angle_interp
        &vec![0.0_f64; target_mass_per_material.len()],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
        &[],                                                              // km_energy_grid
        &[],                                                              // km_interp
        &[],                                                              // km_n_discrete
        &[],                                                              // km_n_x
        &[],                                                              // km_x_offset
        &[],                                                              // km_x
        &[],                                                              // km_p
        &[],                                                              // km_c
        &[],                                                              // km_r
        &[],                                                              // km_a
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_ae_offset
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // evap_theta_offset
        &[],                                                              // evap_energy_grid
        &[],                                                              // evap_theta
        &[],                                                              // evap_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_ae_offset
        &[],                                                              // maxwell_energy_grid
        &[],                                                              // maxwell_theta
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // maxwell_u
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
        &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_ae_offset
        &[],                                                              // watt_energy_grid
        &[],                                                              // watt_a
        &[],                                                              // watt_b
        &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT], // watt_u
        &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
        &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
        &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
        &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(),                          // urr_cdf
        &Vec::<f64>::new(),                          // urr_xs
        &vec![0.0_f64; target_mass_per_material.len()],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges(tally_log_min, tally_log_max, 8),
        ),
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
        &FissionBankInputs::off(),
        1,
        500,
        400.0,
        TallyVarianceMode::PerStep,
    );

    let abs_per_cell = r.absorption_per_cell();
    // Shape checks.
    assert_eq!(r.alive.len(), n);
    assert_eq!(r.n_steps.len(), n);
    assert_eq!(r.final_energies.len(), n);
    assert_eq!(abs_per_cell.len(), 2);
    assert_eq!(r.absorption_per_cell_energy().len(), 2 * 8);

    // Physics-sane: every particle takes at least one step and
    // never exceeds the cap.
    for &s in &r.n_steps {
        assert!((1..=500).contains(&s), "n_steps {s} out of range");
    }

    // Tallies are non-negative and at least one cell scored
    // (Be9 has a tiny but non-zero σ_a above ~keV).
    for &t in &abs_per_cell {
        assert!(t >= 0.0, "tally negative: {t}");
    }
    let total_tally: f64 = abs_per_cell.iter().sum();
    assert!(
        total_tally > 0.0,
        "total tally must be positive (got {total_tally}); means no particle scored -- XS extraction broken"
    );

    // Energy must not increase in elastic scattering. Final
    // energies of absorbed/escaped particles should all be
    // ≤ starting energy; with ~MeV neutrons in Be9 this is a
    // strong check that the kinematics are wired right.
    let max_final = r.final_energies.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_final <= 1.0e6 + 1e-3,
        "no neutron should gain energy in elastic scatter; got max final {max_final}"
    );

    let alive = r.alive.iter().filter(|&&a| a == 1).count();
    let absorbed = r.alive.iter().filter(|&&a| a == 0).count();
    let mean_steps: f64 = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    println!(
        "real ENDF: Be9 + Pb208 at 1 MeV, 1000 particles -- \
         alive (escaped or at cap) = {alive}, terminated = {absorbed}, \
         mean steps = {mean_steps}, \
         tally cell 0 (Be9) = {}, tally cell 1 (Pb208) = {}",
        abs_per_cell[0], abs_per_cell[1]
    );
}

/// Sanity-check the angle helper's `n_energies == 0` fallback --
/// when no tabulated data is loaded for an MT slot, the helper
/// must fall back to isotropic-in-`xi3` (matches pre-slice-B
/// behaviour bit for bit). The returned mu is `1 − 2·xi3` and
/// the energy is the closed-form `e_cm` value (since
/// `scatter_in_cm == 0` keeps the lab interpretation of the
/// closed-form energy).
#[test]
fn angle_helper_falls_back_to_isotropic_when_no_data() {
    let bufs = default_angle_buffers(1);
    let mut state = crate::common::rng::expand_seed(12345u32);
    let state_before = state;
    let xi3 = 0.25;
    let (mu, e_out, alive) = super::dispatch::sample_inelastic_angle(
        10e6,
        56.0,
        5.0e6,
        xi3,
        0,
        10,
        &mut state,
        &bufs.n_energies,
        &vec![0u32; bufs.n_energies.len()],
        &bufs.energy_grid,
        &bufs.n_mu,
        &vec![0u32; bufs.n_mu.len()],
        &bufs.mu,
        &bufs.cdf,
        &bufs.pdf,
        &bufs.interp,
        &bufs.eout_kind,
        &bufs.eout_n_energies,
        &bufs.eout_ae_offset,
        &bufs.eout_energy_grid,
        &bufs.eout_n_x,
        &bufs.eout_x_offset,
        &bufs.eout_x,
        &bufs.eout_cdf,
        &bufs.eout_histogram_interp,
        &bufs.eout_p,
        &bufs.eout_interp,
        &bufs.eout_n_discrete,
        &bufs.corr_n_energies,
        &bufs.corr_n_components,
        &bufs.corr_ae_offset,
        &bufs.corr_energy_grid,
        &bufs.corr_n_x,
        &bufs.corr_x_offset,
        &bufs.corr_x,
        &bufs.corr_cdf,
        &bufs.corr_p,
        &bufs.corr_interp,
        &bufs.corr_n_discrete,
        &bufs.corr_n_mu,
        &bufs.corr_mu_offset,
        &bufs.corr_mu,
        &bufs.corr_mu_cdf,
        &bufs.corr_mu_pdf,
        &bufs.corr_mu_interp,
        &bufs.scatter_in_cm,
        &bufs.km_n_energies,
        &bufs.km_ae_offset,
        &bufs.km_energy_grid,
        &bufs.km_interp,
        &bufs.km_n_discrete,
        &bufs.km_n_x,
        &bufs.km_x_offset,
        &bufs.km_x,
        &bufs.km_p,
        &bufs.km_c,
        &bufs.km_r,
        &bufs.km_a,
        &bufs.evap_n_energies,
        &bufs.evap_n_components,
        &bufs.evap_ae_offset,
        &bufs.evap_theta_offset,
        &bufs.evap_energy_grid,
        &bufs.evap_theta,
        &bufs.evap_u,
        &bufs.nbps_n_bodies,
        &bufs.nbps_total_mass,
        &bufs.maxwell_n_energies,
        &bufs.maxwell_ae_offset,
        &bufs.maxwell_energy_grid,
        &bufs.maxwell_theta,
        &bufs.maxwell_u,
        &bufs.watt_n_energies,
        &bufs.watt_ae_offset,
        &bufs.watt_energy_grid,
        &bufs.watt_a,
        &bufs.watt_b,
        &bufs.watt_u,
        0.0_f64,
    );
    assert!(alive);
    assert_eq!(state, state_before, "state must not advance when n_ae==0");
    assert!((mu - (1.0 - 2.0 * xi3)).abs() < 1e-12);
    assert!((e_out - 5.0e6).abs() < 1e-9);
}

/// When a slot's angular table is narrowly peaked, sampling
/// should produce mean(mu) close to the peak. Build a 2-point
/// CDF concentrated near mu = 0.95 on a single incident-energy
/// slice (so any reasonable energy sample picks that bracket),
/// drive the helper many times under varying RNG state, and
/// verify the mean lab cosine is well above zero.
#[test]
fn angle_helper_recovers_forward_peak_from_cdf() {
    let mut bufs = default_angle_buffers(1);
    let slot: usize = 0;
    let mat_slot = slot; // single material × slot 0
    bufs.n_energies[mat_slot] = 1; // one incident-energy slice
                                   // Tight CSR (#104): slot 0 -> ae-row 0 -> mu-points 0,1.
    bufs.energy_grid = vec![1.0]; // 1 ae-row: dummy E_in
    bufs.n_mu = vec![2]; // 1 ae-row
    bufs.interp = vec![ANGLE_INTERP_HISTOGRAM]; // 1 ae-row
    bufs.mu = vec![0.94, 0.96]; // 2 mu-points
    bufs.cdf = vec![0.0, 1.0];
    bufs.pdf = vec![0.0, 0.0];
    bufs.scatter_in_cm[mat_slot] = 0; // keep mu in lab

    let n = 5000usize;
    let mut sum = 0.0f64;
    for i in 0..n {
        let mut state =
            crate::common::rng::expand_seed((i as u32).wrapping_mul(2_654_435_761) ^ 0xdead_beef);
        let xi3 = 0.0; // would map to mu=+1.0 if we hit isotropic -- trying to tease out the difference
        let (mu, _e, alive) = super::dispatch::sample_inelastic_angle(
            10e6,
            56.0,
            5.0e6,
            xi3,
            0,
            slot,
            &mut state,
            &bufs.n_energies,
            &vec![0u32; bufs.n_energies.len()],
            &bufs.energy_grid,
            &bufs.n_mu,
            &vec![0u32; bufs.n_mu.len()],
            &bufs.mu,
            &bufs.cdf,
            &bufs.pdf,
            &bufs.interp,
            &bufs.eout_kind,
            &bufs.eout_n_energies,
            &bufs.eout_ae_offset,
            &bufs.eout_energy_grid,
            &bufs.eout_n_x,
            &bufs.eout_x_offset,
            &bufs.eout_x,
            &bufs.eout_cdf,
            &bufs.eout_histogram_interp,
            &bufs.eout_p,
            &bufs.eout_interp,
            &bufs.eout_n_discrete,
            &bufs.corr_n_energies,
            &bufs.corr_n_components,
            &bufs.corr_ae_offset,
            &bufs.corr_energy_grid,
            &bufs.corr_n_x,
            &bufs.corr_x_offset,
            &bufs.corr_x,
            &bufs.corr_cdf,
            &bufs.corr_p,
            &bufs.corr_interp,
            &bufs.corr_n_discrete,
            &bufs.corr_n_mu,
            &bufs.corr_mu_offset,
            &bufs.corr_mu,
            &bufs.corr_mu_cdf,
            &bufs.corr_mu_pdf,
            &bufs.corr_mu_interp,
            &bufs.scatter_in_cm,
            &bufs.km_n_energies,
            &bufs.km_ae_offset,
            &bufs.km_energy_grid,
            &bufs.km_interp,
            &bufs.km_n_discrete,
            &bufs.km_n_x,
            &bufs.km_x_offset,
            &bufs.km_x,
            &bufs.km_p,
            &bufs.km_c,
            &bufs.km_r,
            &bufs.km_a,
            &bufs.evap_n_energies,
            &bufs.evap_n_components,
            &bufs.evap_ae_offset,
            &bufs.evap_theta_offset,
            &bufs.evap_energy_grid,
            &bufs.evap_theta,
            &bufs.evap_u,
            &bufs.nbps_n_bodies,
            &bufs.nbps_total_mass,
            &bufs.maxwell_n_energies,
            &bufs.maxwell_ae_offset,
            &bufs.maxwell_energy_grid,
            &bufs.maxwell_theta,
            &bufs.maxwell_u,
            &bufs.watt_n_energies,
            &bufs.watt_ae_offset,
            &bufs.watt_energy_grid,
            &bufs.watt_a,
            &bufs.watt_b,
            &bufs.watt_u,
            0.0_f64,
        );
        assert!(alive);
        sum += mu;
    }
    let mean = sum / n as f64;
    // The table is uniform on [0.94, 0.96], so the analytic mean is 0.95.
    assert!(
        mean > 0.93 && mean < 0.97,
        "expected mean(mu) ≈ 0.95 from forward-peaked table, got {mean}"
    );
}

/// CM→lab two-body conversion path -- when `scatter_in_cm == 1`,
/// the helper must apply the closed-form CM→lab transform. With
/// a narrow forward-peaked CM-frame mu ≈ 1 the lab cosine is
/// also ≈ 1 (forward kinematics) and `e_lab > e_cm` by the
/// `(E_in + 2·μ·(A+1)·sqrt(E_in·E_cm)) / (A+1)²` correction.
#[test]
fn angle_helper_cm_to_lab_forward_peak() {
    let mut bufs = default_angle_buffers(1);
    let slot: usize = 0;
    bufs.n_energies[slot] = 1;
    // Tight CSR (#104): slot 0 -> ae-row 0 -> mu-points 0,1.
    bufs.energy_grid = vec![1.0];
    bufs.n_mu = vec![2];
    bufs.interp = vec![ANGLE_INTERP_HISTOGRAM];
    bufs.mu = vec![0.99, 1.0];
    bufs.cdf = vec![0.0, 1.0];
    bufs.pdf = vec![0.0, 0.0];
    bufs.scatter_in_cm[slot] = 1;

    let mut state = crate::common::rng::expand_seed(7u32);
    let e_in = 14.0e6;
    let e_cm = 5.0e6;
    let target_mass = 56.0;
    let (mu_lab, e_lab, alive) = super::dispatch::sample_inelastic_angle(
        e_in,
        target_mass,
        e_cm,
        0.5, // xi3 fallback unused (n_ae > 0)
        0,
        slot,
        &mut state,
        &bufs.n_energies,
        &vec![0u32; bufs.n_energies.len()],
        &bufs.energy_grid,
        &bufs.n_mu,
        &vec![0u32; bufs.n_mu.len()],
        &bufs.mu,
        &bufs.cdf,
        &bufs.pdf,
        &bufs.interp,
        &bufs.eout_kind,
        &bufs.eout_n_energies,
        &bufs.eout_ae_offset,
        &bufs.eout_energy_grid,
        &bufs.eout_n_x,
        &bufs.eout_x_offset,
        &bufs.eout_x,
        &bufs.eout_cdf,
        &bufs.eout_histogram_interp,
        &bufs.eout_p,
        &bufs.eout_interp,
        &bufs.eout_n_discrete,
        &bufs.corr_n_energies,
        &bufs.corr_n_components,
        &bufs.corr_ae_offset,
        &bufs.corr_energy_grid,
        &bufs.corr_n_x,
        &bufs.corr_x_offset,
        &bufs.corr_x,
        &bufs.corr_cdf,
        &bufs.corr_p,
        &bufs.corr_interp,
        &bufs.corr_n_discrete,
        &bufs.corr_n_mu,
        &bufs.corr_mu_offset,
        &bufs.corr_mu,
        &bufs.corr_mu_cdf,
        &bufs.corr_mu_pdf,
        &bufs.corr_mu_interp,
        &bufs.scatter_in_cm,
        &bufs.km_n_energies,
        &bufs.km_ae_offset,
        &bufs.km_energy_grid,
        &bufs.km_interp,
        &bufs.km_n_discrete,
        &bufs.km_n_x,
        &bufs.km_x_offset,
        &bufs.km_x,
        &bufs.km_p,
        &bufs.km_c,
        &bufs.km_r,
        &bufs.km_a,
        &bufs.evap_n_energies,
        &bufs.evap_n_components,
        &bufs.evap_ae_offset,
        &bufs.evap_theta_offset,
        &bufs.evap_energy_grid,
        &bufs.evap_theta,
        &bufs.evap_u,
        &bufs.nbps_n_bodies,
        &bufs.nbps_total_mass,
        &bufs.maxwell_n_energies,
        &bufs.maxwell_ae_offset,
        &bufs.maxwell_energy_grid,
        &bufs.maxwell_theta,
        &bufs.maxwell_u,
        &bufs.watt_n_energies,
        &bufs.watt_ae_offset,
        &bufs.watt_energy_grid,
        &bufs.watt_a,
        &bufs.watt_b,
        &bufs.watt_u,
        0.0_f64,
    );
    assert!(alive);
    // Forward CM → forward lab.
    assert!(mu_lab > 0.99, "expected mu_lab close to 1, got {mu_lab}");
    // CM→lab boost: e_lab > e_cm because the forward μ adds the
    // full kinematic cross term.
    assert!(
        e_lab > e_cm,
        "expected e_lab > e_cm under forward CM scattering, got e_lab={e_lab}, e_cm={e_cm}"
    );
}

/// Slice C: when an MT slot has `eout_kind = ContinuousTabular`
/// and a tabulated CDF, the kernel must override the closed-form
/// `e_cm` with a sample drawn from the table. Build a slot whose
/// CDF is concentrated near `E_out = 7 MeV` and check that the
/// returned outgoing energy clusters around that value, well
/// below what the closed-form `mass_ratio * (E_in - threshold)`
/// would produce.
#[test]
fn eout_helper_recovers_tabulated_e_out() {
    let mut bufs = default_angle_buffers(1);
    let slot: usize = 0;
    // Configure slot 0 for ContinuousTabular sampling at one
    // incident-energy slice with a 2-point CDF that maps any ξ
    // into the narrow [6.9, 7.1] MeV window.
    bufs.eout_kind[slot] = 1; // EOUT_KIND_CONTINUOUS_TABULAR
    bufs.eout_n_energies[slot] = 1;
    // Tight CSR (#104): slot 0 -> ae-row 0 -> x-points 0,1.
    bufs.eout_energy_grid = vec![1.0]; // 1 ae-row: dummy E_in
    bufs.eout_n_x = vec![2]; // 1 ae-row
    bufs.eout_x_offset = vec![0u32]; // 1 ae-row -> x-point 0
    bufs.eout_interp = vec![0u32]; // 1 ae-row
    bufs.eout_n_discrete = vec![0u32]; // 1 ae-row
    bufs.eout_x = vec![6.9e6, 7.1e6]; // 2 x-points
    bufs.eout_cdf = vec![0.0, 1.0];
    bufs.eout_p = vec![0.0, 0.0];
    bufs.scatter_in_cm[slot] = 0; // E_out is lab -- no CM boost

    let n = 5000usize;
    let mut sum = 0.0f64;
    for i in 0..n {
        let mut state =
            crate::common::rng::expand_seed((i as u32).wrapping_mul(2_654_435_761) ^ 0xc0ffee);
        // Closed-form would yield 14 MeV * 0.85 ≈ 11.9 MeV -- way
        // off the tabulated peak -- so any non-zero CDF effect
        // pulls the mean down.
        let (_mu, e_out, alive) = super::dispatch::sample_inelastic_angle(
            14e6,
            56.0,
            11.9e6,
            0.5,
            0,
            slot,
            &mut state,
            &bufs.n_energies,
            &vec![0u32; bufs.n_energies.len()],
            &bufs.energy_grid,
            &bufs.n_mu,
            &vec![0u32; bufs.n_mu.len()],
            &bufs.mu,
            &bufs.cdf,
            &bufs.pdf,
            &bufs.interp,
            &bufs.eout_kind,
            &bufs.eout_n_energies,
            &bufs.eout_ae_offset,
            &bufs.eout_energy_grid,
            &bufs.eout_n_x,
            &bufs.eout_x_offset,
            &bufs.eout_x,
            &bufs.eout_cdf,
            &bufs.eout_histogram_interp,
            &bufs.eout_p,
            &bufs.eout_interp,
            &bufs.eout_n_discrete,
            &bufs.corr_n_energies,
            &bufs.corr_n_components,
            &bufs.corr_ae_offset,
            &bufs.corr_energy_grid,
            &bufs.corr_n_x,
            &bufs.corr_x_offset,
            &bufs.corr_x,
            &bufs.corr_cdf,
            &bufs.corr_p,
            &bufs.corr_interp,
            &bufs.corr_n_discrete,
            &bufs.corr_n_mu,
            &bufs.corr_mu_offset,
            &bufs.corr_mu,
            &bufs.corr_mu_cdf,
            &bufs.corr_mu_pdf,
            &bufs.corr_mu_interp,
            &bufs.scatter_in_cm,
            &bufs.km_n_energies,
            &bufs.km_ae_offset,
            &bufs.km_energy_grid,
            &bufs.km_interp,
            &bufs.km_n_discrete,
            &bufs.km_n_x,
            &bufs.km_x_offset,
            &bufs.km_x,
            &bufs.km_p,
            &bufs.km_c,
            &bufs.km_r,
            &bufs.km_a,
            &bufs.evap_n_energies,
            &bufs.evap_n_components,
            &bufs.evap_ae_offset,
            &bufs.evap_theta_offset,
            &bufs.evap_energy_grid,
            &bufs.evap_theta,
            &bufs.evap_u,
            &bufs.nbps_n_bodies,
            &bufs.nbps_total_mass,
            &bufs.maxwell_n_energies,
            &bufs.maxwell_ae_offset,
            &bufs.maxwell_energy_grid,
            &bufs.maxwell_theta,
            &bufs.maxwell_u,
            &bufs.watt_n_energies,
            &bufs.watt_ae_offset,
            &bufs.watt_energy_grid,
            &bufs.watt_a,
            &bufs.watt_b,
            &bufs.watt_u,
            0.0_f64,
        );
        assert!(alive);
        sum += e_out;
    }
    let mean = sum / n as f64;
    // Uniform on [6.9, 7.1] MeV → analytic mean 7.0 MeV.
    assert!(
        (6.85e6..=7.15e6).contains(&mean),
        "expected mean(e_out) ≈ 7 MeV from tabulated CDF, got {mean}"
    );
}

/// When `eout_kind = LevelInelastic` (the default), the helper
/// must keep the closed-form `e_cm` unchanged regardless of how
/// large the tabulated buffers are. Mirrors the pre-slice-C
/// behaviour exactly so existing CPU/GPU bit-equivalence tests
/// hold under empty-eout buffers.
#[test]
fn eout_helper_falls_back_to_closed_form_for_level_inelastic() {
    let bufs = default_angle_buffers(1);
    let mut state = crate::common::rng::expand_seed(7u32);
    let state_before = state;
    let (_mu, e_out, alive) = super::dispatch::sample_inelastic_angle(
        14e6,
        56.0,
        11.9e6,
        0.25,
        0,
        0,
        &mut state,
        &bufs.n_energies,
        &vec![0u32; bufs.n_energies.len()],
        &bufs.energy_grid,
        &bufs.n_mu,
        &vec![0u32; bufs.n_mu.len()],
        &bufs.mu,
        &bufs.cdf,
        &bufs.pdf,
        &bufs.interp,
        &bufs.eout_kind,
        &bufs.eout_n_energies,
        &bufs.eout_ae_offset,
        &bufs.eout_energy_grid,
        &bufs.eout_n_x,
        &bufs.eout_x_offset,
        &bufs.eout_x,
        &bufs.eout_cdf,
        &bufs.eout_histogram_interp,
        &bufs.eout_p,
        &bufs.eout_interp,
        &bufs.eout_n_discrete,
        &bufs.corr_n_energies,
        &bufs.corr_n_components,
        &bufs.corr_ae_offset,
        &bufs.corr_energy_grid,
        &bufs.corr_n_x,
        &bufs.corr_x_offset,
        &bufs.corr_x,
        &bufs.corr_cdf,
        &bufs.corr_p,
        &bufs.corr_interp,
        &bufs.corr_n_discrete,
        &bufs.corr_n_mu,
        &bufs.corr_mu_offset,
        &bufs.corr_mu,
        &bufs.corr_mu_cdf,
        &bufs.corr_mu_pdf,
        &bufs.corr_mu_interp,
        &bufs.scatter_in_cm,
        &bufs.km_n_energies,
        &bufs.km_ae_offset,
        &bufs.km_energy_grid,
        &bufs.km_interp,
        &bufs.km_n_discrete,
        &bufs.km_n_x,
        &bufs.km_x_offset,
        &bufs.km_x,
        &bufs.km_p,
        &bufs.km_c,
        &bufs.km_r,
        &bufs.km_a,
        &bufs.evap_n_energies,
        &bufs.evap_n_components,
        &bufs.evap_ae_offset,
        &bufs.evap_theta_offset,
        &bufs.evap_energy_grid,
        &bufs.evap_theta,
        &bufs.evap_u,
        &bufs.nbps_n_bodies,
        &bufs.nbps_total_mass,
        &bufs.maxwell_n_energies,
        &bufs.maxwell_ae_offset,
        &bufs.maxwell_energy_grid,
        &bufs.maxwell_theta,
        &bufs.maxwell_u,
        &bufs.watt_n_energies,
        &bufs.watt_ae_offset,
        &bufs.watt_energy_grid,
        &bufs.watt_a,
        &bufs.watt_b,
        &bufs.watt_u,
        0.0_f64,
    );
    assert!(alive);
    // No CDF data → kernel uses the supplied closed-form e_cm.
    // No state advance for the eout draws either.
    assert!((e_out - 11.9e6).abs() < 1e-6);
    assert_eq!(
        state, state_before,
        "state must not advance when no continuum data is present"
    );
}

/// Slice D: a correlated slot with a narrow E_out CDF and a
/// narrow per-(E_in, E_out) mu CDF should produce both
/// `e_out` and `mu` in tight ranges around the encoded peaks.
#[test]
fn corr_helper_recovers_tabulated_e_out_and_mu() {
    let mut bufs = default_angle_buffers(1);
    let slot: usize = 0;
    // Mark slot 0 as correlated; populate corr_* with a single
    // incident-energy slice and a 2-point E_out CDF concentrated
    // around 4 MeV. The angular sub-table at the chosen
    // (E_in_idx, E_out_idx) bin is concentrated near μ = 0.5.
    bufs.eout_kind[slot] = 2; // EOUT_KIND_CORRELATED
    bufs.corr_n_energies[slot] = 1;
    bufs.corr_n_components[slot] = 1; // single component: no selector draw
                                      // Tight CSR (#104): slot 0 -> ae-row 0 -> x-points 0,1 -> mu-points 0,1.
    bufs.corr_energy_grid = vec![1.0]; // 1 ae-row
    bufs.corr_n_x = vec![2]; // 1 ae-row -> 2 x-points
    bufs.corr_interp = vec![0u32]; // 1 ae-row
    bufs.corr_n_discrete = vec![0u32]; // 1 ae-row
    bufs.corr_x_offset = vec![0u32]; // 1 ae-row -> x-point 0
    bufs.corr_x = vec![3.9e6, 4.1e6]; // 2 x-points
    bufs.corr_cdf = vec![0.0, 1.0];
    bufs.corr_p = vec![0.0, 0.0];
    // Angular sub-table for (E_in_idx=0, E_out_idx=0): two points
    // around μ = 0.5. Per-x-point arrays have length 2 (one per x-point);
    // only x-point 0 carries a μ sub-table.
    bufs.corr_n_mu = vec![2u32, 0u32]; // per-x-point
    bufs.corr_mu_offset = vec![0u32, 0u32]; // per-x-point -> mu-point 0
    bufs.corr_mu_interp = vec![0u32, 0u32]; // per-x-point
    bufs.corr_mu = vec![0.45, 0.55]; // 2 mu-points
    bufs.corr_mu_cdf = vec![0.0, 1.0];
    bufs.corr_mu_pdf = vec![0.0, 0.0];
    bufs.scatter_in_cm[slot] = 0; // both E_out and mu in lab frame

    let n = 5000usize;
    let mut sum_e = 0.0f64;
    let mut sum_mu = 0.0f64;
    for i in 0..n {
        let mut state =
            crate::common::rng::expand_seed((i as u32).wrapping_mul(2_654_435_761) ^ 0xfeed_face);
        let (mu, e_out, alive) = super::dispatch::sample_inelastic_angle(
            14e6,
            56.0,
            11.9e6, // closed-form fallback the helper should ignore
            0.5,
            0,
            slot,
            &mut state,
            &bufs.n_energies,
            &vec![0u32; bufs.n_energies.len()],
            &bufs.energy_grid,
            &bufs.n_mu,
            &vec![0u32; bufs.n_mu.len()],
            &bufs.mu,
            &bufs.cdf,
            &bufs.pdf,
            &bufs.interp,
            &bufs.eout_kind,
            &bufs.eout_n_energies,
            &bufs.eout_ae_offset,
            &bufs.eout_energy_grid,
            &bufs.eout_n_x,
            &bufs.eout_x_offset,
            &bufs.eout_x,
            &bufs.eout_cdf,
            &bufs.eout_histogram_interp,
            &bufs.eout_p,
            &bufs.eout_interp,
            &bufs.eout_n_discrete,
            &bufs.corr_n_energies,
            &bufs.corr_n_components,
            &bufs.corr_ae_offset,
            &bufs.corr_energy_grid,
            &bufs.corr_n_x,
            &bufs.corr_x_offset,
            &bufs.corr_x,
            &bufs.corr_cdf,
            &bufs.corr_p,
            &bufs.corr_interp,
            &bufs.corr_n_discrete,
            &bufs.corr_n_mu,
            &bufs.corr_mu_offset,
            &bufs.corr_mu,
            &bufs.corr_mu_cdf,
            &bufs.corr_mu_pdf,
            &bufs.corr_mu_interp,
            &bufs.scatter_in_cm,
            &bufs.km_n_energies,
            &bufs.km_ae_offset,
            &bufs.km_energy_grid,
            &bufs.km_interp,
            &bufs.km_n_discrete,
            &bufs.km_n_x,
            &bufs.km_x_offset,
            &bufs.km_x,
            &bufs.km_p,
            &bufs.km_c,
            &bufs.km_r,
            &bufs.km_a,
            &bufs.evap_n_energies,
            &bufs.evap_n_components,
            &bufs.evap_ae_offset,
            &bufs.evap_theta_offset,
            &bufs.evap_energy_grid,
            &bufs.evap_theta,
            &bufs.evap_u,
            &bufs.nbps_n_bodies,
            &bufs.nbps_total_mass,
            &bufs.maxwell_n_energies,
            &bufs.maxwell_ae_offset,
            &bufs.maxwell_energy_grid,
            &bufs.maxwell_theta,
            &bufs.maxwell_u,
            &bufs.watt_n_energies,
            &bufs.watt_ae_offset,
            &bufs.watt_energy_grid,
            &bufs.watt_a,
            &bufs.watt_b,
            &bufs.watt_u,
            0.0_f64,
        );
        assert!(alive);
        sum_e += e_out;
        sum_mu += mu;
    }
    let mean_e = sum_e / n as f64;
    let mean_mu = sum_mu / n as f64;
    assert!(
        (3.85e6..=4.15e6).contains(&mean_e),
        "expected mean(e_out) ≈ 4 MeV from correlated CDF, got {mean_e}"
    );
    assert!(
        (0.45..=0.55).contains(&mean_mu),
        "expected mean(mu) ≈ 0.5 from correlated angular sub-table, got {mean_mu}"
    );
}

/// Slice E + F + coverage-closure: `MT_SLOTS` and `MT_YIELDS` must be
/// parallel arrays with the right multi-neutron-out wiring. Slot 41 =
/// MT 16 (yield 2), slot 42 = MT 17 (yield 3); slots 0..=40
/// (MT 51..=91) and slots 43..=47 (slice-F MT 22/28/32/33/34)
/// stay single-neutron-out (yield 1). Slots 48..=55 are the
/// coverage-closure neutron-emitting MTs (5/23/24/25/37/41/44/45).
#[test]
fn mt_slots_and_yields_layout() {
    use crate::neutron::xs::{MT_SLOTS, MT_YIELDS};
    assert_eq!(MT_SLOTS.len(), 62);
    assert_eq!(MT_YIELDS.len(), 62);
    // Slots 0..=40 are MT 51..=91.
    for k in 0..=40usize {
        assert_eq!(MT_SLOTS[k], 51 + k as i32);
        assert_eq!(MT_YIELDS[k], 1, "slot {k} should be single-neutron-out");
    }
    // Slot 41 = MT 16 (n,2n), slot 42 = MT 17 (n,3n).
    assert_eq!(MT_SLOTS[41], 16);
    assert_eq!(MT_YIELDS[41], 2);
    assert_eq!(MT_SLOTS[42], 17);
    assert_eq!(MT_YIELDS[42], 3);
    // Slice F: charged-particle-out + neutron MTs in slots
    // 43..=47, all yield 1.
    let slice_f: [(usize, i32); 5] = [
        (43, 22), // (n,n'α)
        (44, 28), // (n,n'p)
        (45, 32), // (n,n'd)
        (46, 33), // (n,n't)
        (47, 34), // (n,n'³He)
    ];
    for (slot, mt) in slice_f {
        assert_eq!(MT_SLOTS[slot], mt, "slot {slot} should hold MT {mt}");
        assert_eq!(
            MT_YIELDS[slot], 1,
            "slot {slot} (MT {mt}) should be single-neutron-out"
        );
    }
    // Coverage-closure MTs in slots 48..=55, with nominal multiplicity
    // (runtime yield comes from `yield_per_mt`).
    let closure: [(usize, i32, u32); 8] = [
        (48, 5, 2),  // (n,misc)
        (49, 23, 1), // (n,n'3α)
        (50, 24, 2), // (n,2nα)
        (51, 25, 3), // (n,3nα)
        (52, 37, 4), // (n,4n)
        (53, 41, 2), // (n,2np)
        (54, 44, 1), // (n,n'2p)
        (55, 45, 1), // (n,n'pα)
    ];
    for (slot, mt, y) in closure {
        assert_eq!(MT_SLOTS[slot], mt, "slot {slot} should hold MT {mt}");
        assert_eq!(MT_YIELDS[slot], y, "slot {slot} (MT {mt}) yield");
    }
    // Breakup channels in slots 56..=61 (issue #106), appended in ascending
    // MT order so the walk order they had as untabled MTs is preserved.
    let breakup: [(usize, i32, u32); 6] = [
        (56, 11, 2), // (n,2nd)
        (57, 29, 1), // (n,n'3α)
        (58, 30, 2), // (n,2n2α)
        (59, 35, 1), // (n,n'd2α)
        (60, 36, 1), // (n,n't2α)
        (61, 42, 3), // (n,3np)
    ];
    for (slot, mt, y) in breakup {
        assert_eq!(MT_SLOTS[slot], mt, "slot {slot} should hold MT {mt}");
        assert_eq!(MT_YIELDS[slot], y, "slot {slot} (MT {mt}) yield");
    }
}

// Maxwell / Evaporation / Watt theoretical-mean and return-None
// tests live in `yamc-physics/src/flat/{maxwell,evaporation,watt}.rs`
// -- the physics is owned there, and the slice-based API doesn't
// need any of yamc-gpu's flat-buffer layout constants to exercise.

/// Slice F: `log_edges_bin` must correctly route a log-energy
/// into the bin defined by an arbitrary monotonic edge set
/// (an irregular grid here, mimicking the kind VITAMIN-J-175
/// produces). Verifies the CPU-mirror's bin lookup, which the
/// GPU kernel mirrors via the same binary-search logic.
#[test]
fn log_edges_bin_handles_non_uniform_edges() {
    // Irregular edges in ln(E) -- values precomputed below to
    // avoid `.ln()` (cubecl's prelude shadows the std method
    // outside `#[cube]` functions).
    // For E (in eV) = [1e4, 1e5, 5e5, 1e6, 5e6, 1.4e7]:
    let edges: Vec<f64> = vec![
        9.210340371976182,  // ln(1e4)
        11.512925464970229, // ln(1e5)
        13.122363377404328, // ln(5e5)
        13.815510557964274, // ln(1e6)
        15.424948470398373, // ln(5e6)
        16.45072677228497,  // ln(1.4e7)
    ];
    // log of 1e3 = 6.907755...; below first edge -> dropped (None),
    // matching CPU EnergyFilter::get_bin (energy < bins[0] -> None).
    assert_eq!(super::log_edges_bin(&edges, 6.907755), None);
    // Far below first edge → still dropped.
    assert_eq!(super::log_edges_bin(&edges, -1e10_f64), None);
    // Exactly on the first edge -> bin 0 (lower edge inclusive).
    assert_eq!(super::log_edges_bin(&edges, 9.210340371976182), Some(0));
    // log of 5e4 = 10.819778; first bin (1e4..1e5).
    assert_eq!(super::log_edges_bin(&edges, 10.819778), Some(0));
    // Exactly on edge[1]: upper-edge-inclusive `(lo, hi]` convention
    // (matches CPU EnergyFilter::get_bin) -> end of bin 0.
    assert_eq!(super::log_edges_bin(&edges, 11.512925464970229), Some(0));
    // Middle of bin 1: log of 3e5 ≈ 12.611538.
    assert_eq!(super::log_edges_bin(&edges, 12.611538), Some(1));
    // Bin 3: log of 3e6 ≈ 14.914123.
    assert_eq!(super::log_edges_bin(&edges, 14.914123), Some(3));
    // Bin 4: log of 10e6 ≈ 16.118096.
    assert_eq!(super::log_edges_bin(&edges, 16.118096), Some(4));
    // At last edge → upper-edge-inclusive -> bin 4 (last bin).
    assert_eq!(super::log_edges_bin(&edges, 16.45072677228497), Some(4));
    // Above last edge -> dropped (None).
    assert_eq!(super::log_edges_bin(&edges, 18.42068_f64), None);
}

/// Coupled neutron->photon production (S4b): run a collision-heavy
/// single-Fe56-material sphere through the neutron kernel with coupled
/// emission ON, read the device particle bank, and assert:
///   (a) `bank_overflow == 0` (no silent drops),
///   (b) the banked photon count is plausible vs `sum_collisions y_t`,
///   (c) every banked photon is `PTYPE_PHOTON` with positive `e_out`, a
///       unit-ish direction, finite weight, and a valid cell,
///   (d) photon energies fall in / cluster at the Fe56 discrete gamma
///       lines + continuous range,
///   (e) the NEUTRON tallies / alive / n_steps EQUAL the coupled-OFF run
///       (the child-seed isolation invariant -- photon draws never perturb
///       the neutron stream).
///
/// Skips gracefully if `Fe56.arrow` is absent or there's no f64 GPU adapter.
#[test]
fn coupled_photon_emission_fe56() {
    use crate::neutron::xs::photon_production::{
        extract_photon_production_xs, PHOTON_EOUT_KIND_DISCRETE,
    };
    use crate::neutron::xs::{MT_CAPTURE, MT_ELASTIC};
    use yamc_nuclide::nuclide_loader::load_nuclide;

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let manifest = env!("CARGO_MANIFEST_DIR");
    let fe56_path = format!("{manifest}/../yamc/tests/Fe56.arrow");
    let fe56 = match load_nuclide(&fe56_path, &yamc_nuclide::LoadScope::full()) {
        Ok(n) => n,
        Err(e) => {
            println!("Fe56.arrow load failed ({e}); skipping");
            return;
        }
    };

    let temp = "294";
    let temp_idx = fe56.get_temp_idx(temp).expect("Fe56 missing 294");
    let grid = fe56
        .energy
        .as_ref()
        .and_then(|e| e.get(temp))
        .expect("Fe56 missing energy grid")
        .clone();
    let log_grid: Vec<f64> = grid.iter().map(|e| e.ln()).collect();
    let n_grid = grid.len();

    // Elastic + capture macro xs on Fe56's own grid (unit density). To make
    // the run collision-heavy and bank a healthy photon population, scale the
    // absorption (capture is the dominant photon producer via MT 102) up by a
    // density factor -- this only changes how often collisions happen, not the
    // photon physics keyed off `photon_prod / sigma_t`.
    let density = 0.5_f64;
    let reactions = &fe56.reactions[temp_idx];
    let elastic = reactions.get(&MT_ELASTIC).expect("Fe56 MT=2 missing");
    let capture = reactions.get(&MT_CAPTURE).expect("Fe56 MT=102 missing");
    let xs_e: Vec<f64> = grid
        .iter()
        .map(|&e| density * elastic.cross_section_at(e).unwrap_or(0.0))
        .collect();
    let xs_a: Vec<f64> = grid
        .iter()
        .map(|&e| density * capture.cross_section_at(e).unwrap_or(0.0))
        .collect();
    let mass = fe56.atomic_weight_ratio.expect("Fe56 missing AWR");
    let target_mass_per_material = vec![mass];

    // Per-material photon-production table at the same density / temperature /
    // shared log grid the kernel uses, then concatenate (one material).
    let pp_table = extract_photon_production_xs(&[(&fe56, density)], temp, &log_grid)
        .expect("extract Fe56 photon production");
    let n_photon_products = pp_table.n_product;
    // Collect the discrete gamma-line energies for the clustering check (d).
    let discrete_lines: Vec<f64> = pp_table
        .prod_eout_kind
        .iter()
        .zip(pp_table.prod_line_energy.iter())
        .filter_map(|(&k, &e)| (k == PHOTON_EOUT_KIND_DISCRETE && e > 0.0).then_some(e))
        .collect();
    let coupled_on = CoupledPhotonInputs::from_materials(std::slice::from_ref(&pp_table));

    // Single large sphere of Fe56, radius 50, centred at origin (1 cell, 1
    // surface). 1 MeV source at the centre going +x; many collisions before
    // escape at this density / size.
    let cell_aabbs: Vec<f64> = vec![-50.0, -50.0, -50.0, 50.0, 50.0, 50.0];
    let cell_to_material: Vec<u32> = vec![0];
    let surface_types = vec![0u32]; // sphere
    let surface_params = vec![0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0];

    let n = 2000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| {
            (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(0x1234_5678)
        })
        .collect();
    let energies: Vec<f64> = vec![1.0e6; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }
    let tally_log_min = *log_grid.first().expect("empty grid");
    let tally_log_max = *log_grid.last().expect("empty grid");
    let max_steps = 400u32;

    // Closure threading the (largely zero) neutron buffers; only the coupled
    // inputs + bank capacity vary between the OFF and ON runs.
    let run = |coupled: &CoupledPhotonInputs, cap: usize| -> MultiCellResult {
        run_multi_cell_transport(
            &ctx,
            &seeds,
            &energies,
            &positions,
            &directions,
            &cell_aabbs,
            &cell_to_material,
            &surface_types,
            &pad_surface_params(&surface_params),
            &vec![0u32; surface_types.len()],
            &vec![0u32; (cell_aabbs.len() / 6) + 1], // region_program: zero header = identity region (AABB-only fixture)
            &log_grid,
            &log_grid,
            &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
            &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
            &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
            &xs_e,
            &xs_a,
            &vec![0.0_f64; xs_a.len()],
            &vec![0.0_f64; xs_a.len()],
            &vec![0.0_f64; xs_a.len()],
            &vec![0.0_f64; xs_a.len()], // beta_delayed_per_material
            &vec![0.988e6_f64; target_mass_per_material.len()],
            &vec![2.249e-6_f64; target_mass_per_material.len()],
            // Two chi rows per material (issue #364): prompt then delayed.
            &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_kind
            &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_n_energies
            &vec![0u32; 2 * target_mass_per_material.len()], // fission_eout_ae_offset (tight CSR, #104)
            &[], // fission_eout_energy_grid (no fission -> empty tight)
            &[], // fission_eout_n_x
            &[], // fission_eout_x_offset
            &[], // fission_eout_x
            &[], // fission_eout_cdf
            &[], // fission_eout_p
            &[], // fission_eout_interp
            &[], // xs_inelastic_per_mt_sparse (no inelastic)
            &target_mass_per_material,
            &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[], // yield_per_mt_sparse (no inelastic)
            &vec![
                0u32;
                target_mass_per_material.len() * MT_INELASTIC_COUNT * PERMT_META_COLS as usize
            ], // permt_meta
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // eout_ae_offset
            &[],
            &[],
            &[], // eout_x_offset
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[],
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_n_components (issue #111)
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // corr_ae_offset
            &[],
            &[],
            &[], // corr_x_offset
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[], // corr_mu_offset
            &[],
            &[],
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len()],
            &vec![0u32; target_mass_per_material.len()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &vec![0.0_f64; target_mass_per_material.len()],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT], // km_ae_offset
            &[],
            &[],
            &[],
            &[],
            &[], // km_x_offset
            &[],
            &[],
            &[],
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[],
            &[],
            &[],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[],
            &[],
            &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &[],
            &[],
            &[],
            &vec![0.0_f64; target_mass_per_material.len() * MT_INELASTIC_COUNT],
            &vec![0u32; target_mass_per_material.len() * URR_META_COLS],
            &vec![0u32; target_mass_per_material.len()], // urr_ae_offset
            &vec![0u32; target_mass_per_material.len()], // urr_cdf_offset
            &Vec::<f64>::new(),                          // urr_energy_grid (tight, no URR present)
            &Vec::<f64>::new(),                          // urr_cdf
            &Vec::<f64>::new(),                          // urr_xs
            &vec![0.0_f64; target_mass_per_material.len()],
            &flux_abs_pack_for(
                &cell_aabbs,
                &log_uniform_edges(tally_log_min, tally_log_max, 8),
            ),
            &[],
            &SurvivalBiasingInputs::off(),
            coupled,
            // This test exercises prompt coupling only -- decay stays off.
            &DecayPhotonInputs::decay_off(1, n_grid),
            &NuclideSelectInputs::single_nuclide(&target_mass_per_material, log_grid.len()),
            &FissionBankInputs::off(),
            cap,
            max_steps,
            400.0,
            TallyVarianceMode::PerStep,
        )
    };

    // (e) Coupled-OFF baseline + coupled-ON run with the same seeds.
    let off = run(&CoupledPhotonInputs::coupled_off(1, n_grid), 1);
    // Capacity sized generously: at most one photon per collision is the norm
    // (y_t < 1 for Fe56), and total collisions <= n * max_steps; a healthy
    // over-estimate avoids overflow while staying modest.
    let cap = n * 4;
    let on = run(&coupled_on, cap);

    // (a) No overflow.
    assert_eq!(
        on.photon_bank.overflow, 0,
        "coupled-photon bank overflowed (count {})",
        on.photon_bank.count
    );

    // (e) Child-seed isolation: the neutron transport is byte-identical to the
    // coupled-off run -- alive flags, step counts, and the fixed-point neutron
    // tallies must be EXACTLY equal.
    assert_eq!(on.alive, off.alive, "alive flags diverged with coupling on");
    assert_eq!(on.n_steps, off.n_steps, "n_steps diverged with coupling on");
    for (t, (a, b)) in on
        .tally_outputs
        .iter()
        .zip(off.tally_outputs.iter())
        .enumerate()
    {
        assert_eq!(a, b, "neutron tally {t} diverged with coupling on");
    }

    // (b) Plausible photon count. With the off run we know each particle's step
    // count; collisions <= steps. A loose lower bound: at least some photons
    // banked (capture dominates and emits a discrete cascade); upper bound:
    // never more than the total collisions, and well under capacity.
    let banked = on.photon_bank.count;
    assert!(banked > 0, "expected some banked photons, got 0");
    let total_steps: u64 = off.n_steps.iter().map(|&s| s as u64).sum();
    assert!(
        banked <= total_steps,
        "banked photons {banked} exceeds total neutron steps {total_steps} (>1 photon/collision \
         implies a count bug)"
    );
    assert!(
        (banked as usize) < cap,
        "banked photons {banked} hit capacity {cap} -- size the bank larger"
    );

    // (c) + (d): every written photon record is physical.
    let written = banked.min(cap as u64) as usize;
    let max_line = discrete_lines.iter().cloned().fold(0.0_f64, f64::max);
    // Continuous photons can exceed the highest discrete line; bound by a
    // generous multiple of the incident energy.
    let e_ceiling = (max_line.max(1.0e6)) * 4.0 + 1.0e7;
    let mut near_line = 0usize;
    for s in 0..written {
        let f = s * crate::common::particle_bank::BANK_F64_STRIDE;
        let u = s * crate::common::particle_bank::BANK_U32_STRIDE;
        let e_out = on.photon_bank.bank_f64[f];
        let dx = on.photon_bank.bank_f64[f + 4];
        let dy = on.photon_bank.bank_f64[f + 5];
        let dz = on.photon_bank.bank_f64[f + 6];
        let wt = on.photon_bank.bank_f64[f + 7];
        let ptype = on.photon_bank.bank_u32[u];
        let cell = on.photon_bank.bank_u32[u + 1];

        assert_eq!(
            ptype,
            crate::common::particle_bank::PTYPE_PHOTON,
            "bank slot {s} not a photon"
        );
        assert!(
            e_out > 0.0 && e_out < e_ceiling,
            "bank slot {s} e_out {e_out} out of (0, {e_ceiling})"
        );
        assert!(wt.is_finite() && wt > 0.0, "bank slot {s} weight {wt}");
        assert_eq!(cell, 0, "bank slot {s} cell {cell} (only cell 0 exists)");
        let norm = dx * dx + dy * dy + dz * dz;
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "bank slot {s} direction not unit (norm^2 {norm})"
        );

        // (d) cluster near a discrete gamma line (within 1%)?
        if discrete_lines
            .iter()
            .any(|&line| (e_out - line).abs() <= 0.01 * line)
        {
            near_line += 1;
        }
    }

    // (d) Fe56's photon production is dominated by discrete capture gammas, so
    // a large fraction of banked photons should sit on a discrete line; the
    // rest are the continuous-tabular continuum.
    let frac_on_line = near_line as f64 / written as f64;
    assert!(
        frac_on_line > 0.5,
        "expected most photons on a discrete Fe56 line, got {frac_on_line:.3} ({near_line}/{written})"
    );

    println!(
        "coupled Fe56: {n} neutrons, {n_photon_products} photon products, {banked} photons banked \
         (cap {cap}, overflow 0); {near_line}/{written} on a discrete line; neutron tallies \
         identical to coupled-off (child-seed isolation holds)"
    );
}

/// Hand-built per-(slab,MT) inelastic secondary-energy configuration for the
/// synthetic Maxwell/Watt CSR regression below. Single material / single slab;
/// `MT_INELASTIC_COUNT` slots. The non-`maxwell_*` / `watt_*` distribution
/// families stay empty (all-zero counts and offsets) so only the closed-form,
/// Maxwell, and Watt branches are reachable.
struct InelasticEoutConfig {
    /// `[MT_INELASTIC_COUNT * n_grid]` coarse-grid per-MT inelastic xs. Picks
    /// which slot the per-MT cumulative-xs walk lands on.
    xs_inelastic_per_mt: Vec<f64>,
    /// `[MT_INELASTIC_COUNT]` per-slot outgoing-energy kind.
    eout_kind: Vec<u32>,
    /// `[MT_INELASTIC_COUNT]` Maxwell θ(E_in) row counts (tight CSR).
    maxwell_n_energies: Vec<u32>,
    /// Tight `[Σ maxwell_n_energies]` incident-energy grid.
    maxwell_energy_grid: Vec<f64>,
    /// Tight `[Σ maxwell_n_energies]` θ values, parallel to the grid.
    maxwell_theta: Vec<f64>,
    /// `[MT_INELASTIC_COUNT]` Maxwell restriction energy u.
    maxwell_u: Vec<f64>,
    /// `[MT_INELASTIC_COUNT]` Watt a/b row counts (tight CSR).
    watt_n_energies: Vec<u32>,
    /// Tight `[Σ watt_n_energies]` incident-energy grid.
    watt_energy_grid: Vec<f64>,
    /// Tight `[Σ watt_n_energies]` a values, parallel to the grid.
    watt_a: Vec<f64>,
    /// Tight `[Σ watt_n_energies]` b values, parallel to the grid.
    watt_b: Vec<f64>,
    /// `[MT_INELASTIC_COUNT]` Watt restriction energy u.
    watt_u: Vec<f64>,
}

/// Prefix-sum the per-(slab,MT) row counts into the tight CSR base offsets,
/// exactly as `translate.rs::push_{maxwell,watt}_csr_offsets` do: slot `k`'s
/// rows start at `sum(n_energies[..k])`.
fn csr_offsets_from_counts(n_energies: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(n_energies.len());
    let mut acc = 0u32;
    for &n in n_energies {
        offsets.push(acc);
        acc += n;
    }
    offsets
}

/// Compress DENSE per-MT inelastic xs / yield `[n_slab × MT_INELASTIC_COUNT ×
/// n_grid]` buffers into the SPARSE layout the kernel now consumes (issue #212):
/// `(xs_sparse, yield_sparse, permt_meta)`. Bit-for-bit mirror of the
/// `extract.rs` compression + `translate.rs::push_permt_meta` assembly: each
/// (slab, MT slot) stores only its nonzero-xs range `[i_start, i_end)`, and
/// `permt_meta` records `[value_offset, i_start, n_stored]` per row.
fn dense_per_mt_to_sparse(
    xs_dense: &[f64],
    yield_dense: &[f64],
    n_slab: usize,
    n_grid: usize,
) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let mt = MT_INELASTIC_COUNT;
    let mut xs_sparse = Vec::new();
    let mut yield_sparse = Vec::new();
    let mut permt_meta = Vec::with_capacity(n_slab * mt * PERMT_META_COLS as usize);
    for slab in 0..n_slab {
        for slot in 0..mt {
            let base = (slab * mt + slot) * n_grid;
            let mut i_start = 0usize;
            let mut i_end = 0usize;
            let mut found = false;
            for i in 0..n_grid {
                if xs_dense[base + i] != 0.0 {
                    if !found {
                        i_start = i;
                        found = true;
                    }
                    i_end = i + 1;
                }
            }
            permt_meta.push(xs_sparse.len() as u32);
            if found {
                permt_meta.push(i_start as u32);
                permt_meta.push((i_end - i_start) as u32);
                xs_sparse.extend_from_slice(&xs_dense[base + i_start..base + i_end]);
                yield_sparse.extend_from_slice(&yield_dense[base + i_start..base + i_end]);
            } else {
                permt_meta.push(0);
                permt_meta.push(0);
            }
        }
    }
    (xs_sparse, yield_sparse, permt_meta)
}

/// Run a single cube cell of one inelastic-scattering material through the GPU
/// kernel and the CPU twin, threading `cfg`'s per-MT inelastic secondary-energy
/// buffers into BOTH backends (the same buffers). Everything else is empty /
/// default, mirroring `run_equiv`. Source neutrons start monodirectional from
/// the cube centre at `source_energy`. Returns `(gpu, cpu)`.
#[allow(clippy::too_many_arguments)]
fn run_inelastic_eout_equiv(
    ctx: &GpuContext,
    cfg: &InelasticEoutConfig,
    source_energy: f64,
) -> (MultiCellResult, MultiCellResult) {
    // One cube cell [0,2]^3 bounded by six planes (bit-exact distance math
    // across backends, like the analog equivalence tests).
    let cell_aabbs: Vec<f64> = vec![0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let surface_types = vec![1u32; 6];
    let surface_params = vec![
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, // y = 2
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, // z = 2
    ];

    // Energy grid spanning eV up to ~20 MeV so a multi-MeV source neutron and
    // its down-scattered secondaries all stay inside the grid.
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (2.0e7_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();

    // No elastic, light absorption, strong inelastic: collisions overwhelmingly
    // route into the inelastic branch and from there into the per-MT walk.
    let xs_e_per_mat: Vec<f64> = vec![0.0; n_grid];
    let xs_a_per_mat: Vec<f64> = vec![0.5; n_grid];
    let xs_inelastic_per_mat: Vec<f64> = vec![5.0; n_grid];
    let target_mass_per_material: Vec<f64> = vec![12.0];
    let cell_to_material: Vec<u32> = vec![0];

    let n = 800usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![source_energy; n];
    let mut positions = Vec::with_capacity(3 * n);
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        positions.extend_from_slice(&[1.0, 1.0, 1.0]); // cube centre
        directions.extend_from_slice(&[1.0, 0.0, 0.0]); // monodirectional +x
    }

    let n_mat = target_mass_per_material.len();
    let mt = MT_INELASTIC_COUNT;
    assert_eq!(cfg.xs_inelastic_per_mt.len(), n_mat * mt * n_grid);
    assert_eq!(cfg.eout_kind.len(), n_mat * mt);
    assert_eq!(cfg.maxwell_n_energies.len(), n_mat * mt);
    assert_eq!(cfg.watt_n_energies.len(), n_mat * mt);

    // Tight CSR bases over the per-(slab,MT) row counts (issue #104).
    let maxwell_ae_offset = csr_offsets_from_counts(&cfg.maxwell_n_energies);
    let watt_ae_offset = csr_offsets_from_counts(&cfg.watt_n_energies);
    assert_eq!(cfg.maxwell_theta.len(), cfg.maxwell_energy_grid.len());
    assert_eq!(cfg.watt_a.len(), cfg.watt_energy_grid.len());
    assert_eq!(cfg.watt_b.len(), cfg.watt_energy_grid.len());

    // q = 0 and single-neutron yield for every slot: the closed-form fallback
    // energy is `mass_ratio * E_in` (threshold 0), which the Maxwell/Watt
    // branches override when their slot is selected.
    let q_inelastic_per_mt = vec![0.0_f64; n_mat * mt];
    let yield_per_mt = vec![1.0_f64; n_mat * mt * n_grid];
    // Sparse per-MT storage (issue #212): compress the dense xs / yield into the
    // tight (value, i_start, n_stored) layout both backends now consume.
    let (xs_i_mt_sparse, yield_mt_sparse, permt_meta) =
        dense_per_mt_to_sparse(&cfg.xs_inelastic_per_mt, &yield_per_mt, n_mat, n_grid);
    let scatter_in_cm_per_mt = vec![0u32; n_mat * mt]; // E_out already lab-frame

    let max_steps: u32 = 200;
    let n_energy_bins = 8;
    let n_surf = surface_types.len();
    let n_cell = cell_aabbs.len() / 6;
    let log_min = log_grid[0];
    let log_max = *log_grid.last().unwrap();
    let survival = SurvivalBiasingInputs::off();

    let gpu = run_multi_cell_transport(
        ctx,
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; n_surf],
        &vec![0u32; n_cell + 1], // identity region (AABB-only fixture)
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &xs_inelastic_per_mat,
        &vec![0.0_f64; n_grid], // xs_fission_per_material (no fission)
        &vec![0.0_f64; n_grid], // nu_bar_per_material
        &vec![0.0_f64; n_grid], // beta_delayed_per_material
        &vec![0.988e6_f64; n_mat],
        &vec![2.249e-6_f64; n_mat],
        &vec![0u32; 2 * n_mat], // fission_eout_kind
        &vec![0u32; 2 * n_mat], // fission_eout_n_energies
        &vec![0u32; 2 * n_mat], // fission_eout_ae_offset (tight CSR, #104)
        &[],                    // fission_eout_energy_grid (no fission -> empty tight)
        &[],                    // fission_eout_n_x
        &[],                    // fission_eout_x_offset
        &[],                    // fission_eout_x
        &[],                    // fission_eout_cdf
        &[],                    // fission_eout_p
        &[],                    // fission_eout_interp
        &xs_i_mt_sparse,
        &target_mass_per_material,
        &q_inelastic_per_mt,
        &yield_mt_sparse,
        &permt_meta,
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &cfg.eout_kind,
        &vec![0u32; n_mat * mt], // eout_n_energies
        &vec![0u32; n_mat * mt], // eout_ae_offset
        &[],
        &[],
        &[], // eout_x_offset
        &[],
        &[],
        &vec![0u32; n_mat * mt], // eout_histogram_interp
        &[],                     // eout_p
        &[],                     // eout_interp
        &[],                     // eout_n_discrete
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt], // corr_n_components (issue #111)
        &vec![0u32; n_mat * mt], // corr_ae_offset
        &[],
        &[],
        &[], // corr_x_offset
        &[],
        &[],
        &[], // corr_p
        &[], // corr_interp
        &[], // corr_n_discrete
        &[],
        &[], // corr_mu_offset
        &[],
        &[],
        &[],
        &[],
        &scatter_in_cm_per_mt,
        &vec![0u32; n_mat],
        &vec![0u32; n_mat],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0.0_f64; n_mat],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt], // km_ae_offset
        &[],
        &[],
        &[],
        &[],
        &[], // km_x_offset
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &vec![0.0_f64; n_mat * mt],
        &cfg.maxwell_n_energies,
        &maxwell_ae_offset,
        &cfg.maxwell_energy_grid,
        &cfg.maxwell_theta,
        &cfg.maxwell_u,
        &cfg.watt_n_energies,
        &watt_ae_offset,
        &cfg.watt_energy_grid,
        &cfg.watt_a,
        &cfg.watt_b,
        &cfg.watt_u,
        &vec![0u32; n_mat * URR_META_COLS],
        &vec![0u32; n_mat], // urr_ae_offset
        &vec![0u32; n_mat], // urr_cdf_offset
        &Vec::<f64>::new(), // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(), // urr_cdf
        &Vec::<f64>::new(), // urr_xs
        &vec![0.0_f64; n_mat],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges(log_min, log_max, n_energy_bins),
        ),
        &[],
        &survival,
        &CoupledPhotonInputs::coupled_off(1, 1),
        &DecayPhotonInputs::decay_off(1, 1),
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, n_grid),
        &FissionBankInputs::off(),
        1,
        max_steps,
        400.0,
        TallyVarianceMode::PerStep,
    );

    let (cpu, _cpu_trace) = run_multi_cell_transport_cpu(
        &seeds,
        &energies,
        &positions,
        &directions,
        &cell_aabbs,
        &cell_to_material,
        &surface_types,
        &pad_surface_params(&surface_params),
        &vec![0u32; n_surf],
        &vec![0u32; n_cell + 1],
        &log_grid,
        &log_grid,
        &single_nuclide_coarse_meta(&target_mass_per_material, log_grid.len()),
        &single_nuclide_fine_grid(&log_grid, target_mass_per_material.len()),
        &single_nuclide_fine_meta(&target_mass_per_material, log_grid.len()),
        &xs_e_per_mat,
        &xs_a_per_mat,
        &xs_inelastic_per_mat,
        &vec![0.0_f64; n_grid],
        &vec![0.0_f64; n_grid],
        &vec![0.0_f64; n_grid], // beta_delayed_per_material
        &vec![0.988e6_f64; n_mat],
        &vec![2.249e-6_f64; n_mat],
        // Two chi rows per material (issue #364): prompt then delayed.
        &vec![0u32; 2 * n_mat], // fission_eout_kind
        &vec![0u32; 2 * n_mat], // fission_eout_n_energies
        &vec![0u32; 2 * n_mat], // fission_eout_ae_offset (tight CSR, #104)
        &[],                    // fission_eout_energy_grid (no fission -> empty tight)
        &[],                    // fission_eout_n_x
        &[],                    // fission_eout_x_offset
        &[],                    // fission_eout_x
        &[],                    // fission_eout_cdf
        &[],                    // fission_eout_p
        &[],                    // fission_eout_interp
        &xs_i_mt_sparse,
        &target_mass_per_material,
        &q_inelastic_per_mt,
        &yield_mt_sparse,
        &permt_meta,
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &cfg.eout_kind,
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt], // corr_n_components (issue #111)
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &scatter_in_cm_per_mt,
        &vec![0u32; n_mat],
        &vec![0u32; n_mat],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0.0_f64; n_mat],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &vec![0u32; n_mat * mt],
        &[],
        &[],
        &[],
        &vec![0u32; n_mat * mt],
        &vec![0.0_f64; n_mat * mt],
        &cfg.maxwell_n_energies,
        &maxwell_ae_offset,
        &cfg.maxwell_energy_grid,
        &cfg.maxwell_theta,
        &cfg.maxwell_u,
        &cfg.watt_n_energies,
        &watt_ae_offset,
        &cfg.watt_energy_grid,
        &cfg.watt_a,
        &cfg.watt_b,
        &cfg.watt_u,
        &vec![0u32; n_mat * URR_META_COLS],
        &vec![0u32; n_mat], // urr_ae_offset
        &vec![0u32; n_mat], // urr_cdf_offset
        &Vec::<f64>::new(), // urr_energy_grid (tight, no URR present)
        &Vec::<f64>::new(), // urr_cdf
        &Vec::<f64>::new(), // urr_xs
        &vec![0.0_f64; n_mat],
        &flux_abs_pack_for(
            &cell_aabbs,
            &log_uniform_edges(log_min, log_max, n_energy_bins),
        ),
        &[],
        &survival,
        &NuclideSelectInputs::single_nuclide(&target_mass_per_material, n_grid),
        &FissionBankInputs::off(),
        max_steps,
        400.0,
        false,
        PendDrain::Fifo,
    );

    (gpu, cpu)
}

/// Belt-and-braces regression for issue #104's inelastic Maxwell / Watt CSR
/// migration. No ENDF/B-VIII.1 nuclide ships an inelastic Maxwell or Watt
/// distribution (both are fission-spectrum laws), so the kernel's inelastic
/// Maxwell/Watt branch is unexercised by any real-data fixture. This builds a
/// synthetic single-material cube whose per-MT inelastic walk lands on a
/// `EOUT_KIND_MAXWELL` slot (slot 1) and a `EOUT_KIND_WATT` slot (slot 3), with
/// a multi-point θ(E_in) / a,b(E_in) tabulation (`n_energies = 4 >= 3`) and a
/// DECOY row-set placed before each active slot so its CSR base offset is
/// genuinely non-zero (`maxwell_ae_offset[1] = 4`, `watt_ae_offset[3] = 4`); a
/// wrong base would mis-index into the decoy rows. The GPU kernel and the CPU
/// twin read the SAME buffers, and the run asserts they are byte-identical.
///
/// Differential guard: a CONTROL run flips the two active slots to
/// `EOUT_KIND_LEVEL_INELASTIC` (the closed-form Q-value path) while keeping
/// every other input identical. The Maxwell/Watt sampler draws E_out far below
/// the closed-form `mass_ratio * E_in`, so the two runs' final-energy spectra
/// must differ -- proving the sampler executed rather than falling through.
#[test]
fn cpu_gpu_equivalence_inelastic_maxwell_watt_csr() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let mt = MT_INELASTIC_COUNT;
    let n_grid = 100usize; // must match run_inelastic_eout_equiv's grid
    let source_energy = 1.0e7_f64; // 10 MeV, well inside the grid

    // Active inelastic slots: slot 1 = Maxwell, slot 3 = Watt. Decoy slots 0
    // (Maxwell) and 2 (Watt) carry rows but ZERO selection xs, so they are
    // never picked yet push the active slots' CSR base offsets off zero.
    const MAXWELL_DECOY_SLOT: usize = 0;
    const MAXWELL_SLOT: usize = 1;
    const WATT_DECOY_SLOT: usize = 2;
    const WATT_SLOT: usize = 3;

    // Per-MT inelastic xs (coarse grid == fine grid here): only the two active
    // slots carry xs, each half of the 5.0 macroscopic inelastic. The walk
    // lands on Maxwell for the lower half of xi and Watt for the upper half.
    let mut xs_inelastic_per_mt = vec![0.0_f64; mt * n_grid];
    for g in 0..n_grid {
        xs_inelastic_per_mt[MAXWELL_SLOT * n_grid + g] = 2.5;
        xs_inelastic_per_mt[WATT_SLOT * n_grid + g] = 2.5;
    }

    // Multi-point (4-point) tabulations bracketing the 10 MeV source so the
    // interpolation walk runs (a wrong base reads the decoy rows instead).
    let maxwell_grid_pts = vec![1.0e5_f64, 1.0e6, 5.0e6, 1.5e7];
    // Decoy θ is deliberately distinct from the active θ.
    let maxwell_decoy_theta = vec![9.0e6_f64, 9.0e6, 9.0e6, 9.0e6];
    let maxwell_active_theta = vec![2.0e5_f64, 5.0e5, 1.0e6, 1.5e6];

    let watt_grid_pts = vec![1.0e5_f64, 1.0e6, 5.0e6, 1.5e7];
    let watt_decoy_a = vec![9.0e6_f64, 9.0e6, 9.0e6, 9.0e6];
    let watt_decoy_b = vec![1.0e-7_f64, 1.0e-7, 1.0e-7, 1.0e-7];
    let watt_active_a = vec![3.0e5_f64, 6.0e5, 9.0e5, 1.2e6];
    let watt_active_b = vec![3.0e-6_f64, 3.0e-6, 3.0e-6, 3.0e-6];

    // Maxwell CSR: decoy slot 0 then active slot 1, both 4 rows -> active base
    // offset = 4 (non-zero). All other slots empty.
    let mut maxwell_n_energies = vec![0u32; mt];
    maxwell_n_energies[MAXWELL_DECOY_SLOT] = 4;
    maxwell_n_energies[MAXWELL_SLOT] = 4;
    let mut maxwell_energy_grid = Vec::new();
    let mut maxwell_theta = Vec::new();
    // Slot order matters: the tight arrays are concatenated in slot order, so
    // slot 0's rows precede slot 1's. Push decoy first, then active.
    maxwell_energy_grid.extend_from_slice(&maxwell_grid_pts);
    maxwell_theta.extend_from_slice(&maxwell_decoy_theta);
    maxwell_energy_grid.extend_from_slice(&maxwell_grid_pts);
    maxwell_theta.extend_from_slice(&maxwell_active_theta);
    let maxwell_u = vec![0.0_f64; mt];

    // Watt CSR: decoy slot 2 then active slot 3, both 4 rows -> active base
    // offset = 4 (non-zero).
    let mut watt_n_energies = vec![0u32; mt];
    watt_n_energies[WATT_DECOY_SLOT] = 4;
    watt_n_energies[WATT_SLOT] = 4;
    let mut watt_energy_grid = Vec::new();
    let mut watt_a = Vec::new();
    let mut watt_b = Vec::new();
    watt_energy_grid.extend_from_slice(&watt_grid_pts);
    watt_a.extend_from_slice(&watt_decoy_a);
    watt_b.extend_from_slice(&watt_decoy_b);
    watt_energy_grid.extend_from_slice(&watt_grid_pts);
    watt_a.extend_from_slice(&watt_active_a);
    watt_b.extend_from_slice(&watt_active_b);
    let watt_u = vec![0.0_f64; mt];

    // Sanity: the active slots' CSR bases are non-zero, so the offset is
    // genuinely exercised (a 0 base would read the decoy rows).
    let maxwell_ae_offset = csr_offsets_from_counts(&maxwell_n_energies);
    let watt_ae_offset = csr_offsets_from_counts(&watt_n_energies);
    assert_eq!(maxwell_ae_offset[MAXWELL_SLOT], 4);
    assert_eq!(watt_ae_offset[WATT_SLOT], 4);

    // Maxwell/Watt run: the two active slots sample from the fission-spectrum
    // laws.
    let mut eout_kind = vec![0u32; mt]; // EOUT_KIND_LEVEL_INELASTIC elsewhere
    eout_kind[MAXWELL_SLOT] = EOUT_KIND_MAXWELL;
    eout_kind[WATT_SLOT] = EOUT_KIND_WATT;
    let mw_cfg = InelasticEoutConfig {
        xs_inelastic_per_mt: xs_inelastic_per_mt.clone(),
        eout_kind,
        maxwell_n_energies: maxwell_n_energies.clone(),
        maxwell_energy_grid: maxwell_energy_grid.clone(),
        maxwell_theta: maxwell_theta.clone(),
        maxwell_u: maxwell_u.clone(),
        watt_n_energies: watt_n_energies.clone(),
        watt_energy_grid: watt_energy_grid.clone(),
        watt_a: watt_a.clone(),
        watt_b: watt_b.clone(),
        watt_u: watt_u.clone(),
    };
    let (gpu_mw, cpu_mw) = run_inelastic_eout_equiv(&ctx, &mw_cfg, source_energy);

    // BYTE-IDENTITY: GPU kernel and CPU twin read the new CSR layout the same.
    assert_gpu_cpu_equiv(&gpu_mw, &cpu_mw, 800);

    // CONTROL run: same buffers, but the active slots fall through to the
    // closed-form level-inelastic energy (no sampler). All other inputs (incl.
    // the maxwell/watt CSR buffers) are byte-for-byte identical, so the ONLY
    // change is which eout branch the active slots take.
    let ctrl_cfg = InelasticEoutConfig {
        xs_inelastic_per_mt,
        eout_kind: vec![0u32; mt], // every slot EOUT_KIND_LEVEL_INELASTIC
        maxwell_n_energies,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
    };
    let (gpu_ctrl, cpu_ctrl) = run_inelastic_eout_equiv(&ctx, &ctrl_cfg, source_energy);
    // The control is itself a byte-identity check on the closed-form path.
    assert_gpu_cpu_equiv(&gpu_ctrl, &cpu_ctrl, 800);

    // Differential guard 1: the Maxwell/Watt sampler down-scatters far below
    // the closed-form `mass_ratio * E_in`, so the final-energy spectra differ.
    let mean = |r: &MultiCellResult| -> f64 {
        let s: f64 = r.final_energies.iter().copied().sum();
        s / r.final_energies.len() as f64
    };
    let mean_mw = mean(&gpu_mw);
    let mean_ctrl = mean(&gpu_ctrl);
    let mass_ratio = (12.0_f64 / 13.0).powi(2);
    let closed_form_e = mass_ratio * source_energy;
    assert!(
        (mean_mw - mean_ctrl).abs() > 0.05 * mean_ctrl,
        "Maxwell/Watt sampler did not change the spectrum vs closed form: \
         mean_mw {mean_mw:.4e} vs mean_ctrl {mean_ctrl:.4e} (closed form {closed_form_e:.4e})"
    );
    // The sampler pushes energy DOWN (Maxwell/Watt secondaries are softer than
    // the near-elastic closed-form result), so the Maxwell/Watt mean must be
    // strictly lower.
    assert!(
        mean_mw < mean_ctrl,
        "expected Maxwell/Watt mean below the closed-form control: \
         {mean_mw:.4e} vs {mean_ctrl:.4e}"
    );

    // Differential guard 2: the branch demonstrably ran -- particles took
    // multiple inelastic steps and down-scattered below the source energy.
    let mean_steps =
        gpu_mw.n_steps.iter().map(|&s| s as f64).sum::<f64>() / gpu_mw.n_steps.len() as f64;
    assert!(
        mean_steps > 1.0,
        "expected >1 mean steps (multiple inelastic collisions), got {mean_steps:.3}"
    );
    let downscattered = gpu_mw
        .final_energies
        .iter()
        .filter(|&&e| e < source_energy)
        .count();
    assert!(
        downscattered > gpu_mw.final_energies.len() / 2,
        "expected most particles down-scattered below the source energy, got \
         {downscattered}/{}",
        gpu_mw.final_energies.len()
    );

    println!(
        "inelastic Maxwell/Watt CSR: GPU==CPU byte-identical (Maxwell+Watt run and control); \
         mean final E Maxwell/Watt {mean_mw:.4e} vs closed-form control {mean_ctrl:.4e} \
         (closed form {closed_form_e:.4e}); mean steps {mean_steps:.2}, \
         {downscattered}/{} down-scattered",
        gpu_mw.final_energies.len()
    );
}
