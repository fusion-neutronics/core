//! Histories/sec benchmark for the multi-cell multi-step transport
//! kernel -- the full geometry + per-cell-material composition. Each
//! particle runs up to `MAX_STEPS` events (collisions or surface
//! crossings) until it absorbs, escapes, or hits the cap.
//!
//! Two scenarios:
//!
//! - `2cells`: two adjacent unit cubes along x with two distinct
//!   materials. Particles start at (0.5, 0.5, 0.5) heading +x.
//!   This is the workload the kernel was originally written
//!   against; cell-finding is trivially fast (BVH or linear).
//!
//! - `64cells`: 8×8×1 grid of unit cubes spanning [0,8]×[0,8]×[0,1].
//!   Particles seeded at random positions inside the grid. With
//!   linear cell-finding this is O(64) per step; with BVH it's
//!   O(log 64) ≈ 6. Designed to surface the BVH benefit.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench multi_cell_transport
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::common::geometry::boundary_distance::SURFACE_PARAM_STRIDE;
use yamc_gpu::common::tallies::{TalliesPack, TallyVarianceMode};
use yamc_gpu::neutron::transport::{
    run_multi_cell_transport, run_multi_cell_transport_cpu, run_multi_cell_transport_cpu_rayon,
    CoupledPhotonInputs, DecayPhotonInputs, FissionBankInputs, NuclideSelectInputs, PendDrain,
    SurvivalBiasingInputs, PERMT_META_COLS,
};
use yamc_gpu::neutron::xs::MT_INELASTIC_COUNT;
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;
const MAX_STEPS: u32 = 50;
const N_ENERGY_BINS: usize = 8;

struct Workload {
    seeds: Vec<u32>,
    energies: Vec<f64>,
    positions: Vec<f64>,
    directions: Vec<f64>,
    cell_aabbs: Vec<f64>,
    cell_to_material: Vec<u32>,
    surface_types: Vec<u32>,
    surface_params: Vec<f64>,
    surface_boundaries: Vec<u32>,
    log_grid: Vec<f64>,
    xs_e_per_mat: Vec<f64>,
    xs_a_per_mat: Vec<f64>,
    xs_i_per_mat: Vec<f64>,
    xs_i_per_mt_sparse: Vec<f64>,
    permt_meta: Vec<u32>,
    target_mass: Vec<f64>,
    q_per_mt: Vec<f64>,
    yield_per_mt_sparse: Vec<f64>,
    angle_n_energies: Vec<u32>,
    angle_energy_grid: Vec<f64>,
    angle_n_mu: Vec<u32>,
    angle_mu: Vec<f64>,
    angle_cdf: Vec<f64>,
    angle_interp: Vec<u32>,
    eout_kind: Vec<u32>,
    eout_n_energies: Vec<u32>,
    eout_energy_grid: Vec<f64>,
    eout_n_x: Vec<u32>,
    eout_x: Vec<f64>,
    eout_cdf: Vec<f64>,
    corr_n_energies: Vec<u32>,
    corr_n_components: Vec<u32>,
    corr_energy_grid: Vec<f64>,
    corr_n_x: Vec<u32>,
    corr_x: Vec<f64>,
    corr_cdf: Vec<f64>,
    corr_n_mu: Vec<u32>,
    corr_mu: Vec<f64>,
    corr_mu_cdf: Vec<f64>,
    scatter_in_cm: Vec<u32>,
    log_e_min: f64,
    log_e_max: f64,
}

/// Allocate all the per-MT inelastic, angle, and eout buffers for
/// `n_materials` materials with no real data -- every slot is zero.
/// Mirrors the "no real ENDF data" path the existing benches use;
/// particles elastic-scatter and absorb only.
struct EmptyBuffers {
    xs_i_per_mat: Vec<f64>,
    xs_i_per_mt_sparse: Vec<f64>,
    permt_meta: Vec<u32>,
    q_per_mt: Vec<f64>,
    yield_per_mt_sparse: Vec<f64>,
    angle_n_energies: Vec<u32>,
    angle_energy_grid: Vec<f64>,
    angle_n_mu: Vec<u32>,
    angle_mu: Vec<f64>,
    angle_cdf: Vec<f64>,
    angle_interp: Vec<u32>,
    eout_kind: Vec<u32>,
    eout_n_energies: Vec<u32>,
    eout_energy_grid: Vec<f64>,
    eout_n_x: Vec<u32>,
    eout_x: Vec<f64>,
    eout_cdf: Vec<f64>,
    corr_n_energies: Vec<u32>,
    corr_n_components: Vec<u32>,
    corr_energy_grid: Vec<f64>,
    corr_n_x: Vec<u32>,
    corr_x: Vec<f64>,
    corr_cdf: Vec<f64>,
    corr_n_mu: Vec<u32>,
    corr_mu: Vec<f64>,
    corr_mu_cdf: Vec<f64>,
    scatter_in_cm: Vec<u32>,
}

fn empty_inelastic_and_angle(n_materials: usize, n_grid: usize) -> EmptyBuffers {
    EmptyBuffers {
        xs_i_per_mat: vec![0.0_f64; n_materials * n_grid],
        // Sparse per-MT storage (issue #212): no inelastic data => empty value
        // buffers and all-zero (n_stored == 0) `permt_meta` rows.
        xs_i_per_mt_sparse: Vec::new(),
        permt_meta: vec![0u32; n_materials * MT_INELASTIC_COUNT * PERMT_META_COLS as usize],
        q_per_mt: vec![0.0_f64; n_materials * MT_INELASTIC_COUNT],
        yield_per_mt_sparse: Vec::new(),
        // Tight CSR (issue #104): no inelastic distributions -> zero ae-rows /
        // points, so the per-(material,MT) count arrays carry length and every
        // per-row / per-point data array is empty.
        angle_n_energies: vec![0u32; n_materials * MT_INELASTIC_COUNT],
        angle_energy_grid: Vec::new(),
        angle_n_mu: Vec::new(),
        angle_mu: Vec::new(),
        angle_cdf: Vec::new(),
        angle_interp: Vec::new(),
        eout_kind: vec![0u32; n_materials * MT_INELASTIC_COUNT],
        eout_n_energies: vec![0u32; n_materials * MT_INELASTIC_COUNT],
        eout_energy_grid: Vec::new(),
        eout_n_x: Vec::new(),
        eout_x: Vec::new(),
        eout_cdf: Vec::new(),
        corr_n_energies: vec![0u32; n_materials * MT_INELASTIC_COUNT],
        corr_n_components: vec![0u32; n_materials * MT_INELASTIC_COUNT],
        corr_energy_grid: Vec::new(),
        corr_n_x: Vec::new(),
        corr_x: Vec::new(),
        corr_cdf: Vec::new(),
        corr_n_mu: Vec::new(),
        corr_mu: Vec::new(),
        corr_mu_cdf: Vec::new(),
        scatter_in_cm: vec![0u32; n_materials * MT_INELASTIC_COUNT],
    }
}

/// Build a `coarse_meta` (issue #212) for the single-nuclide, single-shared-
/// coarse-grid bench layout: every material uses the same coarse grid at base 0,
/// with one slab each, so the first-slab base into `permt_meta` ROWS is
/// `m * MT_INELASTIC_COUNT` (the sparse per-MT storage, issue #212).
fn single_nuclide_coarse_meta(target_mass: &[f64], coarse_len: usize) -> Vec<u32> {
    let mut meta = Vec::with_capacity(target_mass.len() * 3);
    for m in 0..target_mass.len() {
        meta.push(0u32);
        meta.push(coarse_len as u32);
        meta.push((m * MT_INELASTIC_COUNT) as u32);
    }
    meta
}

/// Build a `fine_log_energy_grid` (issue #212) for the single-nuclide,
/// shared-grid bench layout: `log_grid` repeated `n_mat` times, so material
/// `m`'s aggregate-XS row starts at `m * log_grid.len()`.
fn single_nuclide_fine_grid(log_grid: &[f64], n_mat: usize) -> Vec<f64> {
    let mut g = Vec::with_capacity(log_grid.len() * n_mat.max(1));
    for _ in 0..n_mat.max(1) {
        g.extend_from_slice(log_grid);
    }
    g
}

/// Build a `fine_meta` (issue #212) matching [`single_nuclide_fine_grid`]:
/// material `m` has grid/aggregate-XS base `m * n_grid`, fine length `n_grid`,
/// and nuc first-slab base `m * n_grid` (one nuclide per material).
fn single_nuclide_fine_meta(target_mass: &[f64], n_grid: usize) -> Vec<u32> {
    let mut meta = Vec::with_capacity(target_mass.len() * 3);
    for m in 0..target_mass.len() {
        meta.push((m * n_grid) as u32);
        meta.push(n_grid as u32);
        meta.push((m * n_grid) as u32);
    }
    meta
}

/// Build a log-uniform edge array for the kernel's tally bins.
fn log_uniform_edges(log_e_min: f64, log_e_max: f64, n_bins: usize) -> Vec<f64> {
    let step = (log_e_max - log_e_min) / n_bins as f64;
    (0..=n_bins)
        .map(|i| log_e_min + (i as f64) * step)
        .collect()
}

fn xs_setup() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    // Material 0: σ_e = 2, σ_a = 1 (Σ_a/Σ_t = 1/3)
    // Material 1: σ_e = 5, σ_a = 0.1 (mostly elastic)
    let mut xs_e_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_e_per_mat.extend(std::iter::repeat_n(2.0, n_grid));
    xs_e_per_mat.extend(std::iter::repeat_n(5.0, n_grid));
    let mut xs_a_per_mat: Vec<f64> = Vec::with_capacity(2 * n_grid);
    xs_a_per_mat.extend(std::iter::repeat_n(1.0, n_grid));
    xs_a_per_mat.extend(std::iter::repeat_n(0.1, n_grid));
    let target_mass: Vec<f64> = vec![12.0, 12.0];
    (
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        target_mass,
        log_e_min,
        log_e_max,
    )
}

/// Repack stride-7 surface rows into the kernel's `SURFACE_PARAM_STRIDE`
/// layout by zero-padding each surface's unused tail.
fn pad_surface_params(rows7: &[f64]) -> Vec<f64> {
    assert_eq!(rows7.len() % 7, 0, "fixture rows must be 7 wide");
    rows7
        .chunks_exact(7)
        .flat_map(|c| c.iter().copied().chain([0.0; SURFACE_PARAM_STRIDE - 7]))
        .collect()
}

fn build_workload_2cells() -> Workload {
    let cell_aabbs: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0
        1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1
    ];
    let cell_to_material: Vec<u32> = vec![0, 1];
    let surface_types = vec![1u32; 7];
    let surface_params = pad_surface_params(&[
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x = 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // x = 1
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, // x = 2
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // y = 0
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, // y = 1
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, // z = 0
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // z = 1
    ]);

    let (log_grid, xs_e_per_mat, xs_a_per_mat, target_mass, log_e_min, log_e_max) = xs_setup();
    let n_grid = log_grid.len();
    let n_materials = target_mass.len();
    let surface_boundaries = vec![0u32; surface_types.len()];
    let EmptyBuffers {
        xs_i_per_mat,
        xs_i_per_mt_sparse,
        permt_meta,
        q_per_mt,
        yield_per_mt_sparse,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_cdf,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        scatter_in_cm,
    } = empty_inelastic_and_angle(n_materials, n_grid);

    let seeds: Vec<u32> = (0..N)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; N];
    let mut positions = Vec::with_capacity(3 * N);
    let mut directions = Vec::with_capacity(3 * N);
    for _ in 0..N {
        positions.extend_from_slice(&[0.5, 0.5, 0.5]);
        directions.extend_from_slice(&[1.0, 0.0, 0.0]);
    }

    Workload {
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        xs_i_per_mat,
        xs_i_per_mt_sparse,
        permt_meta,
        target_mass,
        q_per_mt,
        yield_per_mt_sparse,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_cdf,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        scatter_in_cm,
        log_e_min,
        log_e_max,
    }
}

/// 8×8×1 grid of unit cubes (64 cells). Cells alternate between
/// material 0 and material 1 in a checkerboard pattern. Particles
/// start at random positions inside the box.
fn build_workload_64cells() -> Workload {
    const G: usize = 8;
    let n_cells = G * G;
    let mut cell_aabbs: Vec<f64> = Vec::with_capacity(6 * n_cells);
    let mut cell_to_material: Vec<u32> = Vec::with_capacity(n_cells);
    for ix in 0..G {
        for iy in 0..G {
            let x0 = ix as f64;
            let y0 = iy as f64;
            cell_aabbs.extend_from_slice(&[x0, y0, 0.0, x0 + 1.0, y0 + 1.0, 1.0]);
            cell_to_material.push(((ix + iy) % 2) as u32);
        }
    }

    // Bounding planes: x = 0..G, y = 0..G, z = 0, z = 1.
    let mut surface_types: Vec<u32> = Vec::new();
    let mut surface_params: Vec<f64> = Vec::new();
    for ix in 0..=G {
        let sign = if ix == 0 { -1.0 } else { 1.0 };
        let d = if ix == 0 { 0.0 } else { ix as f64 };
        surface_types.push(1);
        surface_params.extend_from_slice(&[sign, 0.0, 0.0, d, 0.0, 0.0, 0.0]);
    }
    for iy in 0..=G {
        let sign = if iy == 0 { -1.0 } else { 1.0 };
        let d = if iy == 0 { 0.0 } else { iy as f64 };
        surface_types.push(1);
        surface_params.extend_from_slice(&[0.0, sign, 0.0, d, 0.0, 0.0, 0.0]);
    }
    surface_types.push(1);
    surface_params.extend_from_slice(&[0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
    surface_types.push(1);
    surface_params.extend_from_slice(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    // Widen the stride-7 rows built above to the kernel's stride-10.
    let surface_params = pad_surface_params(&surface_params);

    let (log_grid, xs_e_per_mat, xs_a_per_mat, target_mass, log_e_min, log_e_max) = xs_setup();
    let n_grid = log_grid.len();
    let n_materials = target_mass.len();
    let surface_boundaries = vec![0u32; surface_types.len()];
    let EmptyBuffers {
        xs_i_per_mat,
        xs_i_per_mt_sparse,
        permt_meta,
        q_per_mt,
        yield_per_mt_sparse,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_cdf,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        scatter_in_cm,
    } = empty_inelastic_and_angle(n_materials, n_grid);

    let seeds: Vec<u32> = (0..N)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; N];
    let mut positions = Vec::with_capacity(3 * N);
    let mut directions = Vec::with_capacity(3 * N);
    for i in 0..N {
        // Pseudo-random starting point inside the 8×8×1 box.
        let f = i as f64;
        let px = (f * 0.137).rem_euclid(8.0);
        let py = (f * 0.241).rem_euclid(8.0);
        let pz = (f * 0.317).rem_euclid(1.0);
        positions.extend_from_slice(&[px, py, pz]);
        // Direction roughly +x with a small jitter so particles
        // don't all march in lockstep.
        let jx = (f * 0.097).sin();
        let jy = (f * 0.199).sin();
        let mag = (1.0 + jx * jx + jy * jy).sqrt();
        directions.extend_from_slice(&[1.0 / mag, jx / mag, jy / mag]);
    }

    Workload {
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        log_grid,
        xs_e_per_mat,
        xs_a_per_mat,
        xs_i_per_mat,
        xs_i_per_mt_sparse,
        permt_meta,
        target_mass,
        q_per_mt,
        yield_per_mt_sparse,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_cdf,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        scatter_in_cm,
        log_e_min,
        log_e_max,
    }
}

fn bench_one(c: &mut Criterion, name: &str, w: &Workload) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);

    // Build the multi-tally pack once outside the timed loop. This
    // mirrors how the dispatch layer hands a pre-built pack to the
    // kernel and keeps the per-iteration work focused on transport.
    let tallies = TalliesPack::flux_abs_pack(
        (w.cell_aabbs.len() / 6) as u32,
        &log_uniform_edges(w.log_e_min, w.log_e_max, N_ENERGY_BINS),
    );

    let n_mat = w.target_mass.len();
    let n_grid = w.log_grid.len();
    let mt = yamc_gpu::neutron::xs::MT_INELASTIC_COUNT;
    let urr_meta_cols = yamc_gpu::neutron::xs::URR_META_COLS;

    // Tight CSR (issue #104): the fake material carries no per-MT or elastic
    // distributions, so every per-ae-row / per-point data array is empty; only
    // the per-(material,MT) count / CSR-offset arrays keep their fixed length.
    let angle_pdf: Vec<f64> = Vec::new();
    // Two chi rows per material (issue #364): prompt then delayed.
    let fission_eout_kind = vec![0u32; 2 * n_mat];
    let fission_eout_n_e = vec![0u32; 2 * n_mat];
    // Tight CSR (issue #104): no fission -> zero ae-rows, zero points, so the
    // data arrays are empty and the per-chi-row ae-offset is all zero.
    let fission_eout_ae_offset = vec![0u32; 2 * n_mat];
    let fission_eout_eg: Vec<f64> = Vec::new();
    let fission_eout_n_x: Vec<u32> = Vec::new();
    let fission_eout_x_offset: Vec<u32> = Vec::new();
    let fission_eout_x: Vec<f64> = Vec::new();
    let fission_eout_cdf: Vec<f64> = Vec::new();
    let fission_eout_p: Vec<f64> = Vec::new();
    let fission_eout_interp: Vec<u32> = Vec::new();
    let eout_p: Vec<f64> = Vec::new();
    let eout_interp: Vec<u32> = Vec::new();
    let eout_n_discrete: Vec<u32> = Vec::new();
    let eout_hist_interp = vec![0u32; n_mat * mt];
    let corr_p: Vec<f64> = Vec::new();
    let corr_interp: Vec<u32> = Vec::new();
    let corr_n_discrete: Vec<u32> = Vec::new();
    let corr_mu_pdf = vec![0.0_f64; w.corr_mu.len()];
    let corr_mu_interp: Vec<u32> = Vec::new();
    let el_n_e = vec![0u32; n_mat];
    let el_eg: Vec<f64> = Vec::new();
    let el_n_mu: Vec<u32> = Vec::new();
    let el_mu: Vec<f64> = Vec::new();
    let el_cdf: Vec<f64> = Vec::new();
    let el_pdf: Vec<f64> = Vec::new();
    let el_interp: Vec<u32> = Vec::new();
    let temp_k = vec![294.0_f64; n_mat];
    let km_n_e = vec![0u32; n_mat * mt];
    let km_eg: Vec<f64> = Vec::new();
    let km_interp_b: Vec<u32> = Vec::new();
    let km_n_d: Vec<u32> = Vec::new();
    let km_n_x: Vec<u32> = Vec::new();
    let km_x: Vec<f64> = Vec::new();
    let km_p: Vec<f64> = Vec::new();
    let km_c: Vec<f64> = Vec::new();
    let km_r: Vec<f64> = Vec::new();
    let km_a: Vec<f64> = Vec::new();
    let evap_n_e = vec![0u32; n_mat * mt];
    let evap_eg: Vec<f64> = Vec::new();
    let evap_theta: Vec<f64> = Vec::new();
    let evap_u: Vec<f64> = Vec::new();
    let nbps_n = vec![0u32; n_mat * mt];
    let nbps_tm = vec![0.0_f64; n_mat * mt];
    let maxwell_n_e = vec![0u32; n_mat * mt];
    let maxwell_eg: Vec<f64> = Vec::new();
    let maxwell_theta: Vec<f64> = Vec::new();
    let maxwell_u = vec![0.0_f64; n_mat * mt];
    let watt_n_e = vec![0u32; n_mat * mt];
    let watt_eg: Vec<f64> = Vec::new();
    let watt_a: Vec<f64> = Vec::new();
    let watt_b: Vec<f64> = Vec::new();
    let watt_u = vec![0.0_f64; n_mat * mt];
    let urr_meta = vec![0u32; n_mat * urr_meta_cols];
    // Tight CSR (issue #104): no URR-bearing nuclide in the fake material, so
    // the energy / cdf / xs arrays are empty and the per-material bases zero.
    let urr_ae_offset = vec![0u32; n_mat];
    let urr_cdf_offset = vec![0u32; n_mat];
    let urr_eg: Vec<f64> = Vec::new();
    let urr_cdf: Vec<f64> = Vec::new();
    let urr_xs: Vec<f64> = Vec::new();
    let urr_atom_density = vec![0.0_f64; n_mat];

    // CSR row-offset arrays (one per parallel count array). For all-zero fake
    // data the values are irrelevant; only type and length matter so the cpu
    // twin's validate_transport_inputs doesn't trip.
    let angle_ae_offset = vec![0u32; n_mat * mt];
    let angle_mu_offset: Vec<u32> = Vec::new();
    let eout_ae_offset = vec![0u32; n_mat * mt];
    let eout_x_offset: Vec<u32> = Vec::new();
    let corr_ae_offset = vec![0u32; n_mat * mt];
    let corr_x_offset: Vec<u32> = Vec::new();
    let corr_mu_offset: Vec<u32> = Vec::new();
    let el_ae_offset = vec![0u32; n_mat];
    let el_mu_offset: Vec<u32> = Vec::new();
    let km_ae_offset = vec![0u32; n_mat * mt];
    let km_x_offset: Vec<u32> = Vec::new();
    let evap_n_components = vec![0u32; n_mat * mt];
    let evap_ae_offset = vec![0u32; n_mat * mt];
    let evap_theta_offset = vec![0u32; n_mat * mt];
    let maxwell_ae_offset = vec![0u32; n_mat * mt];
    let watt_ae_offset = vec![0u32; n_mat * mt];

    // Inline-vec args hoisted out of the timed closures so they are built once
    // and reused across cpu_seq / cpu_rayon / gpu (clippy::useless_vec).
    let region_program = vec![0u32; (w.cell_aabbs.len() / 6) + 1]; // identity region
    let xs_fission_per_mat = vec![0.0_f64; w.xs_a_per_mat.len()];
    let nu_bar_per_mat = vec![0.0_f64; w.xs_a_per_mat.len()];
    let beta_delayed_per_mat = vec![0.0_f64; w.xs_a_per_mat.len()];
    let fission_a = vec![0.988e6_f64; w.target_mass.len()]; // Watt default
    let fission_b = vec![2.249e-6_f64; w.target_mass.len()]; // Watt default

    let _ = (n_grid,);

    group.bench_function("cpu_seq", |b| {
        b.iter(|| {
            run_multi_cell_transport_cpu(
                &w.seeds,
                &w.energies,
                &w.positions,
                &w.directions,
                &w.cell_aabbs,
                &w.cell_to_material,
                &w.surface_types,
                &w.surface_params,
                &w.surface_boundaries,
                &region_program,
                &w.log_grid,
                &w.log_grid, // coarse_log_energy_grid == fine (single-nuclide)
                &single_nuclide_coarse_meta(&w.target_mass, w.log_grid.len()),
                &single_nuclide_fine_grid(&w.log_grid, w.target_mass.len()),
                &single_nuclide_fine_meta(&w.target_mass, w.log_grid.len()),
                &w.xs_e_per_mat,
                &w.xs_a_per_mat,
                &w.xs_i_per_mat,
                &xs_fission_per_mat,
                &nu_bar_per_mat,
                &beta_delayed_per_mat,
                &fission_a,
                &fission_b,
                &fission_eout_kind,
                &fission_eout_n_e,
                &fission_eout_ae_offset,
                &fission_eout_eg,
                &fission_eout_n_x,
                &fission_eout_x_offset,
                &fission_eout_x,
                &fission_eout_cdf,
                &fission_eout_p,
                &fission_eout_interp,
                &w.xs_i_per_mt_sparse,
                &w.target_mass,
                &w.q_per_mt,
                &w.yield_per_mt_sparse,
                &w.permt_meta,
                &w.angle_n_energies,
                &angle_ae_offset,
                &w.angle_energy_grid,
                &w.angle_n_mu,
                &angle_mu_offset,
                &w.angle_mu,
                &w.angle_cdf,
                &angle_pdf,
                &w.angle_interp,
                &w.eout_kind,
                &w.eout_n_energies,
                &eout_ae_offset,
                &w.eout_energy_grid,
                &w.eout_n_x,
                &eout_x_offset,
                &w.eout_x,
                &w.eout_cdf,
                &eout_hist_interp,
                &eout_p,
                &eout_interp,
                &eout_n_discrete,
                &w.corr_n_energies,
                &w.corr_n_components,
                &corr_ae_offset,
                &w.corr_energy_grid,
                &w.corr_n_x,
                &corr_x_offset,
                &w.corr_x,
                &w.corr_cdf,
                &corr_p,
                &corr_interp,
                &corr_n_discrete,
                &w.corr_n_mu,
                &corr_mu_offset,
                &w.corr_mu,
                &w.corr_mu_cdf,
                &corr_mu_pdf,
                &corr_mu_interp,
                &w.scatter_in_cm,
                &el_n_e,
                &el_ae_offset,
                &el_eg,
                &el_n_mu,
                &el_mu_offset,
                &el_mu,
                &el_cdf,
                &el_pdf,
                &el_interp,
                &temp_k,
                &km_n_e,
                &km_ae_offset,
                &km_eg,
                &km_interp_b,
                &km_n_d,
                &km_n_x,
                &km_x_offset,
                &km_x,
                &km_p,
                &km_c,
                &km_r,
                &km_a,
                &evap_n_e,
                &evap_n_components,
                &evap_ae_offset,
                &evap_theta_offset,
                &evap_eg,
                &evap_theta,
                &evap_u,
                &nbps_n,
                &nbps_tm,
                &maxwell_n_e,
                &maxwell_ae_offset,
                &maxwell_eg,
                &maxwell_theta,
                &maxwell_u,
                &watt_n_e,
                &watt_ae_offset,
                &watt_eg,
                &watt_a,
                &watt_b,
                &watt_u,
                &urr_meta,
                &urr_ae_offset,
                &urr_cdf_offset,
                &urr_eg,
                &urr_cdf,
                &urr_xs,
                &urr_atom_density,
                &tallies,
                &[],
                &SurvivalBiasingInputs::off(),
                &NuclideSelectInputs::single_nuclide(&w.target_mass, w.log_grid.len()),
                &FissionBankInputs::off(),
                MAX_STEPS,
                400.0,
                false,
                PendDrain::Fifo,
            )
        });
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| {
            run_multi_cell_transport_cpu_rayon(
                &w.seeds,
                &w.energies,
                &w.positions,
                &w.directions,
                &w.cell_aabbs,
                &w.cell_to_material,
                &w.surface_types,
                &w.surface_params,
                &w.surface_boundaries,
                &region_program,
                &w.log_grid,
                &w.log_grid, // coarse_log_energy_grid == fine (single-nuclide)
                &single_nuclide_coarse_meta(&w.target_mass, w.log_grid.len()),
                &single_nuclide_fine_grid(&w.log_grid, w.target_mass.len()),
                &single_nuclide_fine_meta(&w.target_mass, w.log_grid.len()),
                &w.xs_e_per_mat,
                &w.xs_a_per_mat,
                &w.xs_i_per_mat,
                &xs_fission_per_mat,
                &nu_bar_per_mat,
                &beta_delayed_per_mat,
                &fission_a,
                &fission_b,
                &fission_eout_kind,
                &fission_eout_n_e,
                &fission_eout_ae_offset,
                &fission_eout_eg,
                &fission_eout_n_x,
                &fission_eout_x_offset,
                &fission_eout_x,
                &fission_eout_cdf,
                &fission_eout_p,
                &fission_eout_interp,
                &w.xs_i_per_mt_sparse,
                &w.target_mass,
                &w.q_per_mt,
                &w.yield_per_mt_sparse,
                &w.permt_meta,
                &w.angle_n_energies,
                &angle_ae_offset,
                &w.angle_energy_grid,
                &w.angle_n_mu,
                &angle_mu_offset,
                &w.angle_mu,
                &w.angle_cdf,
                &angle_pdf,
                &w.angle_interp,
                &w.eout_kind,
                &w.eout_n_energies,
                &eout_ae_offset,
                &w.eout_energy_grid,
                &w.eout_n_x,
                &eout_x_offset,
                &w.eout_x,
                &w.eout_cdf,
                &eout_hist_interp,
                &eout_p,
                &eout_interp,
                &eout_n_discrete,
                &w.corr_n_energies,
                &w.corr_n_components,
                &corr_ae_offset,
                &w.corr_energy_grid,
                &w.corr_n_x,
                &corr_x_offset,
                &w.corr_x,
                &w.corr_cdf,
                &corr_p,
                &corr_interp,
                &corr_n_discrete,
                &w.corr_n_mu,
                &corr_mu_offset,
                &w.corr_mu,
                &w.corr_mu_cdf,
                &corr_mu_pdf,
                &corr_mu_interp,
                &w.scatter_in_cm,
                &el_n_e,
                &el_ae_offset,
                &el_eg,
                &el_n_mu,
                &el_mu_offset,
                &el_mu,
                &el_cdf,
                &el_pdf,
                &el_interp,
                &temp_k,
                &km_n_e,
                &km_ae_offset,
                &km_eg,
                &km_interp_b,
                &km_n_d,
                &km_n_x,
                &km_x_offset,
                &km_x,
                &km_p,
                &km_c,
                &km_r,
                &km_a,
                &evap_n_e,
                &evap_n_components,
                &evap_ae_offset,
                &evap_theta_offset,
                &evap_eg,
                &evap_theta,
                &evap_u,
                &nbps_n,
                &nbps_tm,
                &maxwell_n_e,
                &maxwell_ae_offset,
                &maxwell_eg,
                &maxwell_theta,
                &maxwell_u,
                &watt_n_e,
                &watt_ae_offset,
                &watt_eg,
                &watt_a,
                &watt_b,
                &watt_u,
                &urr_meta,
                &urr_ae_offset,
                &urr_cdf_offset,
                &urr_eg,
                &urr_cdf,
                &urr_xs,
                &urr_atom_density,
                &tallies,
                &[],
                &SurvivalBiasingInputs::off(),
                &NuclideSelectInputs::single_nuclide(&w.target_mass, w.log_grid.len()),
                &FissionBankInputs::off(),
                MAX_STEPS,
                400.0,
            )
        });
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| {
                    run_multi_cell_transport(
                        &ctx,
                        &w.seeds,
                        &w.energies,
                        &w.positions,
                        &w.directions,
                        &w.cell_aabbs,
                        &w.cell_to_material,
                        &w.surface_types,
                        &w.surface_params,
                        &w.surface_boundaries,
                        &region_program,
                        &w.log_grid,
                        &w.log_grid, // coarse_log_energy_grid == fine (single-nuclide)
                        &single_nuclide_coarse_meta(&w.target_mass, w.log_grid.len()),
                        &single_nuclide_fine_grid(&w.log_grid, w.target_mass.len()),
                        &single_nuclide_fine_meta(&w.target_mass, w.log_grid.len()),
                        &w.xs_e_per_mat,
                        &w.xs_a_per_mat,
                        &w.xs_i_per_mat,
                        &vec![0.0_f64; w.xs_a_per_mat.len()], // xs_fission_per_material
                        &vec![0.0_f64; w.xs_a_per_mat.len()], // nu_bar_per_material
                        &vec![0.0_f64; w.xs_a_per_mat.len()], // beta_delayed_per_material
                        &vec![0.988e6_f64; w.target_mass.len()], // fission_a (Watt default)
                        &vec![2.249e-6_f64; w.target_mass.len()], // fission_b (Watt default)
                        &fission_eout_kind,
                        &fission_eout_n_e,
                        &fission_eout_ae_offset,
                        &fission_eout_eg,
                        &fission_eout_n_x,
                        &fission_eout_x_offset,
                        &fission_eout_x,
                        &fission_eout_cdf,
                        &fission_eout_p,
                        &fission_eout_interp,
                        &w.xs_i_per_mt_sparse,
                        &w.target_mass,
                        &w.q_per_mt,
                        &w.yield_per_mt_sparse,
                        &w.permt_meta,
                        &w.angle_n_energies,
                        &angle_ae_offset,
                        &w.angle_energy_grid,
                        &w.angle_n_mu,
                        &angle_mu_offset,
                        &w.angle_mu,
                        &w.angle_cdf,
                        &angle_pdf,
                        &w.angle_interp,
                        &w.eout_kind,
                        &w.eout_n_energies,
                        &eout_ae_offset,
                        &w.eout_energy_grid,
                        &w.eout_n_x,
                        &eout_x_offset,
                        &w.eout_x,
                        &w.eout_cdf,
                        &eout_hist_interp,
                        &eout_p,
                        &eout_interp,
                        &eout_n_discrete,
                        &w.corr_n_energies,
                        &w.corr_n_components,
                        &corr_ae_offset,
                        &w.corr_energy_grid,
                        &w.corr_n_x,
                        &corr_x_offset,
                        &w.corr_x,
                        &w.corr_cdf,
                        &corr_p,
                        &corr_interp,
                        &corr_n_discrete,
                        &w.corr_n_mu,
                        &corr_mu_offset,
                        &w.corr_mu,
                        &w.corr_mu_cdf,
                        &corr_mu_pdf,
                        &corr_mu_interp,
                        &w.scatter_in_cm,
                        &el_n_e,
                        &el_ae_offset,
                        &el_eg,
                        &el_n_mu,
                        &el_mu_offset,
                        &el_mu,
                        &el_cdf,
                        &el_pdf,
                        &el_interp,
                        &temp_k,
                        &km_n_e,
                        &km_ae_offset,
                        &km_eg,
                        &km_interp_b,
                        &km_n_d,
                        &km_n_x,
                        &km_x_offset,
                        &km_x,
                        &km_p,
                        &km_c,
                        &km_r,
                        &km_a,
                        &evap_n_e,
                        &evap_n_components,
                        &evap_ae_offset,
                        &evap_theta_offset,
                        &evap_eg,
                        &evap_theta,
                        &evap_u,
                        &nbps_n,
                        &nbps_tm,
                        &maxwell_n_e,
                        &maxwell_ae_offset,
                        &maxwell_eg,
                        &maxwell_theta,
                        &maxwell_u,
                        &watt_n_e,
                        &watt_ae_offset,
                        &watt_eg,
                        &watt_a,
                        &watt_b,
                        &watt_u,
                        &urr_meta,
                        &urr_ae_offset,
                        &urr_cdf_offset,
                        &urr_eg,
                        &urr_cdf,
                        &urr_xs,
                        &urr_atom_density,
                        &tallies,
                        &[],
                        &SurvivalBiasingInputs::off(),
                        &CoupledPhotonInputs::coupled_off(1, 1),
                        &DecayPhotonInputs::decay_off(1, 1),
                        &NuclideSelectInputs::single_nuclide(&w.target_mass, w.log_grid.len()),
                        &FissionBankInputs::off(),
                        1,
                        MAX_STEPS,
                        400.0,
                        TallyVarianceMode::PerStep,
                    )
                });
            });
        }
        Err(GpuInitError::NoF64Adapter) => {
            eprintln!("skipping GPU bench: no Vulkan f64 adapter on this host");
        }
    }
    group.finish();
}

fn bench_multi_cell(c: &mut Criterion) {
    let w2 = build_workload_2cells();
    bench_one(c, "multi_cell_transport_2cells_1M", &w2);
    let w64 = build_workload_64cells();
    bench_one(c, "multi_cell_transport_64cells_1M", &w64);
}

criterion_group!(benches, bench_multi_cell);
criterion_main!(benches);
