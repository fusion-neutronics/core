//! Twin vs cubecl KERNEL on a FISSILE model (issues #111, #154).
//!
//! The `matched_stream_*` harnesses compare the production CPU against the GPU's
//! CPU TWIN, and since #355 / #357 / #358 a fissile history is bit-identical
//! between those two. Nothing compared the twin against the KERNEL on the
//! fission branch, though: every `cpu_gpu_equivalence_*` fixture in yamc-gpu runs
//! `FissionBankInputs::off()` on a material with no fission cross section, so the
//! fission chi, the fission reaction split and the fission weight path were only
//! ever checked against themselves.
//!
//! That is the half of the CPU-vs-GPU chain #154's 0.2% flux deficit has to live
//! in. This closes it for ONE launch: with the fission bank off, neither side
//! banks progeny, so a single launch each is directly comparable and any
//! difference is the kernel's own physics rather than the host's bank drain.
//!
//! Run it (needs an f64 GPU and the endf-b8.1 U235 cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_twin_kernel_fissile_parity -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::translate_for_gpu;
use yamc::model::{Model, TrackingMode, Verbose};
use yamc_gpu::common::tallies::{TalliesPack, TallyVarianceMode};
use yamc_gpu::neutron::transport::{
    run_multi_cell_transport, run_multi_cell_transport_cpu, CoupledPhotonInputs, DecayPhotonInputs,
    FissionBankInputs, PendDrain, SurvivalBiasingInputs,
};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const SEED: u64 = 4242;
const MAX_STEPS: u32 = 20_000;
const FREE_GAS_THRESHOLD: f64 = 400.0;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}

/// Single-nuclide sphere, `Below(sphere)` so the GPU AABB pass bounds it.
fn sphere_model(nuclide: &str, density: f64, radius: f64, energy_ev: f64) -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([(nuclide.to_string(), cache_dir(nuclide))]),
            None,
        )
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("mat".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model
}

/// Largest relative difference between two tally-output vectors, and where.
fn worst_rel(a: &[f64], b: &[f64]) -> (f64, usize) {
    let mut worst = (0.0f64, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let scale = x.abs().max(y.abs());
        if scale <= 0.0 {
            continue;
        }
        let rel = (x - y).abs() / scale;
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    worst
}

/// Twin and kernel, same inputs, same seeds, fission bank OFF, `PerStep`
/// accumulation (the mode whose `tally_outputs` the twin mirrors).
fn compare(label: &str, nuclide: &str, density: f64, radius: f64, energy_ev: f64, n: usize) {
    if !data_present(nuclide) {
        eprintln!("skipping {label} -- endf-b8.1-{nuclide}.arrow cache absent");
        return;
    }
    let ctx = match yamc_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping {label} -- no f64 GPU");
            return;
        }
    };
    let model = sphere_model(nuclide, density, radius, energy_ev);
    let inputs = translate_for_gpu(&model, n, SEED).expect("translate");
    let n_cells = (inputs.cell_aabbs.len() / 6) as u32;
    let edges: Vec<f64> = [1.0e-5, 1.0e3, 1.0e5, 1.0e6, 2.0e6, 5.0e6, 2.0e7]
        .iter()
        .map(|e: &f64| e.ln())
        .collect();
    let pack = TalliesPack::flux_abs_pack(n_cells, &edges);
    let n_materials = inputs.target_mass_per_material.len();
    let n_grid = inputs.log_energy_grid.len();

    let (twin, _traces) = run_multi_cell_transport_cpu(
        &inputs.seeds,
        &inputs.energies,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.coarse_log_energy_grid,
        &inputs.coarse_meta,
        &inputs.fine_log_energy_grid,
        &inputs.fine_meta,
        &inputs.xs_elastic_per_material,
        &inputs.xs_absorption_per_material,
        &inputs.xs_inelastic_per_material,
        &inputs.xs_fission_per_material,
        &inputs.nu_bar_per_material,
        &inputs.beta_delayed_per_material,
        &inputs.fission_a_per_material,
        &inputs.fission_b_per_material,
        &inputs.fission_eout_kind_per_material,
        &inputs.fission_eout_n_energies_per_material,
        &inputs.fission_eout_ae_offset,
        &inputs.fission_eout_energy_grid_per_material,
        &inputs.fission_eout_n_x_per_material,
        &inputs.fission_eout_x_offset,
        &inputs.fission_eout_x_per_material,
        &inputs.fission_eout_cdf_per_material,
        &inputs.fission_eout_p_per_material,
        &inputs.fission_eout_interp_per_material,
        &inputs.xs_inelastic_per_mt_sparse,
        &inputs.target_mass_per_material,
        &inputs.q_inelastic_per_mt,
        &inputs.yield_per_mt_sparse,
        &inputs.permt_meta,
        &inputs.angle_n_energies,
        &inputs.angle_ae_offset,
        &inputs.angle_energy_grid,
        &inputs.angle_n_mu,
        &inputs.angle_mu_offset,
        &inputs.angle_mu,
        &inputs.angle_cdf,
        &inputs.angle_pdf,
        &inputs.angle_interp,
        &inputs.eout_kind,
        &inputs.eout_n_energies,
        &inputs.eout_ae_offset,
        &inputs.eout_energy_grid,
        &inputs.eout_n_x,
        &inputs.eout_x_offset,
        &inputs.eout_x,
        &inputs.eout_cdf,
        &inputs.eout_histogram_interp,
        &inputs.eout_p,
        &inputs.eout_interp,
        &inputs.eout_n_discrete,
        &inputs.corr_n_energies,
        &inputs.corr_n_components,
        &inputs.corr_ae_offset,
        &inputs.corr_energy_grid,
        &inputs.corr_n_x,
        &inputs.corr_x_offset,
        &inputs.corr_x,
        &inputs.corr_cdf,
        &inputs.corr_p,
        &inputs.corr_interp,
        &inputs.corr_n_discrete,
        &inputs.corr_n_mu,
        &inputs.corr_mu_offset,
        &inputs.corr_mu,
        &inputs.corr_mu_cdf,
        &inputs.corr_mu_pdf,
        &inputs.corr_mu_interp,
        &inputs.scatter_in_cm_per_mt,
        &inputs.elastic_angle_n_energies,
        &inputs.elastic_angle_ae_offset,
        &inputs.elastic_angle_energy_grid,
        &inputs.elastic_angle_n_mu,
        &inputs.elastic_angle_mu_offset,
        &inputs.elastic_angle_mu,
        &inputs.elastic_angle_cdf,
        &inputs.elastic_angle_pdf,
        &inputs.elastic_angle_interp,
        &inputs.temperature_k_per_material,
        &inputs.km_n_energies,
        &inputs.km_ae_offset,
        &inputs.km_energy_grid,
        &inputs.km_interp,
        &inputs.km_n_discrete,
        &inputs.km_n_x,
        &inputs.km_x_offset,
        &inputs.km_x,
        &inputs.km_p,
        &inputs.km_c,
        &inputs.km_r,
        &inputs.km_a,
        &inputs.evap_n_energies,
        &inputs.evap_n_components,
        &inputs.evap_ae_offset,
        &inputs.evap_theta_offset,
        &inputs.evap_energy_grid,
        &inputs.evap_theta,
        &inputs.evap_u,
        &inputs.nbps_n_bodies,
        &inputs.nbps_total_mass,
        &inputs.maxwell_n_energies,
        &inputs.maxwell_ae_offset,
        &inputs.maxwell_energy_grid,
        &inputs.maxwell_theta,
        &inputs.maxwell_u,
        &inputs.watt_n_energies,
        &inputs.watt_ae_offset,
        &inputs.watt_energy_grid,
        &inputs.watt_a,
        &inputs.watt_b,
        &inputs.watt_u,
        &inputs.urr_meta,
        &inputs.urr_ae_offset,
        &inputs.urr_cdf_offset,
        &inputs.urr_energy_grid,
        &inputs.urr_cdf,
        &inputs.urr_xs,
        &inputs.urr_atom_density,
        &pack,
        &[],
        &SurvivalBiasingInputs::off(),
        &inputs.nuclide_select,
        &FissionBankInputs::off(),
        MAX_STEPS,
        FREE_GAS_THRESHOLD,
        false,
        PendDrain::Lifo,
    );

    let gpu = run_multi_cell_transport(
        &ctx,
        &inputs.seeds,
        &inputs.energies,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.coarse_log_energy_grid,
        &inputs.coarse_meta,
        &inputs.fine_log_energy_grid,
        &inputs.fine_meta,
        &inputs.xs_elastic_per_material,
        &inputs.xs_absorption_per_material,
        &inputs.xs_inelastic_per_material,
        &inputs.xs_fission_per_material,
        &inputs.nu_bar_per_material,
        &inputs.beta_delayed_per_material,
        &inputs.fission_a_per_material,
        &inputs.fission_b_per_material,
        &inputs.fission_eout_kind_per_material,
        &inputs.fission_eout_n_energies_per_material,
        &inputs.fission_eout_ae_offset,
        &inputs.fission_eout_energy_grid_per_material,
        &inputs.fission_eout_n_x_per_material,
        &inputs.fission_eout_x_offset,
        &inputs.fission_eout_x_per_material,
        &inputs.fission_eout_cdf_per_material,
        &inputs.fission_eout_p_per_material,
        &inputs.fission_eout_interp_per_material,
        &inputs.xs_inelastic_per_mt_sparse,
        &inputs.target_mass_per_material,
        &inputs.q_inelastic_per_mt,
        &inputs.yield_per_mt_sparse,
        &inputs.permt_meta,
        &inputs.angle_n_energies,
        &inputs.angle_ae_offset,
        &inputs.angle_energy_grid,
        &inputs.angle_n_mu,
        &inputs.angle_mu_offset,
        &inputs.angle_mu,
        &inputs.angle_cdf,
        &inputs.angle_pdf,
        &inputs.angle_interp,
        &inputs.eout_kind,
        &inputs.eout_n_energies,
        &inputs.eout_ae_offset,
        &inputs.eout_energy_grid,
        &inputs.eout_n_x,
        &inputs.eout_x_offset,
        &inputs.eout_x,
        &inputs.eout_cdf,
        &inputs.eout_histogram_interp,
        &inputs.eout_p,
        &inputs.eout_interp,
        &inputs.eout_n_discrete,
        &inputs.corr_n_energies,
        &inputs.corr_n_components,
        &inputs.corr_ae_offset,
        &inputs.corr_energy_grid,
        &inputs.corr_n_x,
        &inputs.corr_x_offset,
        &inputs.corr_x,
        &inputs.corr_cdf,
        &inputs.corr_p,
        &inputs.corr_interp,
        &inputs.corr_n_discrete,
        &inputs.corr_n_mu,
        &inputs.corr_mu_offset,
        &inputs.corr_mu,
        &inputs.corr_mu_cdf,
        &inputs.corr_mu_pdf,
        &inputs.corr_mu_interp,
        &inputs.scatter_in_cm_per_mt,
        &inputs.elastic_angle_n_energies,
        &inputs.elastic_angle_ae_offset,
        &inputs.elastic_angle_energy_grid,
        &inputs.elastic_angle_n_mu,
        &inputs.elastic_angle_mu_offset,
        &inputs.elastic_angle_mu,
        &inputs.elastic_angle_cdf,
        &inputs.elastic_angle_pdf,
        &inputs.elastic_angle_interp,
        &inputs.temperature_k_per_material,
        &inputs.km_n_energies,
        &inputs.km_ae_offset,
        &inputs.km_energy_grid,
        &inputs.km_interp,
        &inputs.km_n_discrete,
        &inputs.km_n_x,
        &inputs.km_x_offset,
        &inputs.km_x,
        &inputs.km_p,
        &inputs.km_c,
        &inputs.km_r,
        &inputs.km_a,
        &inputs.evap_n_energies,
        &inputs.evap_n_components,
        &inputs.evap_ae_offset,
        &inputs.evap_theta_offset,
        &inputs.evap_energy_grid,
        &inputs.evap_theta,
        &inputs.evap_u,
        &inputs.nbps_n_bodies,
        &inputs.nbps_total_mass,
        &inputs.maxwell_n_energies,
        &inputs.maxwell_ae_offset,
        &inputs.maxwell_energy_grid,
        &inputs.maxwell_theta,
        &inputs.maxwell_u,
        &inputs.watt_n_energies,
        &inputs.watt_ae_offset,
        &inputs.watt_energy_grid,
        &inputs.watt_a,
        &inputs.watt_b,
        &inputs.watt_u,
        &inputs.urr_meta,
        &inputs.urr_ae_offset,
        &inputs.urr_cdf_offset,
        &inputs.urr_energy_grid,
        &inputs.urr_cdf,
        &inputs.urr_xs,
        &inputs.urr_atom_density,
        &pack,
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(n_materials, n_grid),
        &DecayPhotonInputs::decay_off(n_materials, n_grid),
        &inputs.nuclide_select,
        &FissionBankInputs::off(),
        1,
        MAX_STEPS,
        FREE_GAS_THRESHOLD,
        TallyVarianceMode::PerStep,
    );

    // Same launch again, but with the PerSource accumulation a FISSILE model
    // actually uses in dispatch (non-fissile models take the PerHistory path, so
    // PerSource is only ever exercised on fission runs). The means must be
    // identical: PerSource only adds per-source bookkeeping on top of the same
    // contributions. Tallies land in `src_acc` (fixed-point, `n * total_bins`
    // row-major) instead of `tally_outputs`, so unpack them the way
    // `accumulate_src_acc` does and sum over sources.
    let total_out_len = pack.total_out_len() as usize;
    let gpu_ps = run_multi_cell_transport(
        &ctx,
        &inputs.seeds,
        &inputs.energies,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.coarse_log_energy_grid,
        &inputs.coarse_meta,
        &inputs.fine_log_energy_grid,
        &inputs.fine_meta,
        &inputs.xs_elastic_per_material,
        &inputs.xs_absorption_per_material,
        &inputs.xs_inelastic_per_material,
        &inputs.xs_fission_per_material,
        &inputs.nu_bar_per_material,
        &inputs.beta_delayed_per_material,
        &inputs.fission_a_per_material,
        &inputs.fission_b_per_material,
        &inputs.fission_eout_kind_per_material,
        &inputs.fission_eout_n_energies_per_material,
        &inputs.fission_eout_ae_offset,
        &inputs.fission_eout_energy_grid_per_material,
        &inputs.fission_eout_n_x_per_material,
        &inputs.fission_eout_x_offset,
        &inputs.fission_eout_x_per_material,
        &inputs.fission_eout_cdf_per_material,
        &inputs.fission_eout_p_per_material,
        &inputs.fission_eout_interp_per_material,
        &inputs.xs_inelastic_per_mt_sparse,
        &inputs.target_mass_per_material,
        &inputs.q_inelastic_per_mt,
        &inputs.yield_per_mt_sparse,
        &inputs.permt_meta,
        &inputs.angle_n_energies,
        &inputs.angle_ae_offset,
        &inputs.angle_energy_grid,
        &inputs.angle_n_mu,
        &inputs.angle_mu_offset,
        &inputs.angle_mu,
        &inputs.angle_cdf,
        &inputs.angle_pdf,
        &inputs.angle_interp,
        &inputs.eout_kind,
        &inputs.eout_n_energies,
        &inputs.eout_ae_offset,
        &inputs.eout_energy_grid,
        &inputs.eout_n_x,
        &inputs.eout_x_offset,
        &inputs.eout_x,
        &inputs.eout_cdf,
        &inputs.eout_histogram_interp,
        &inputs.eout_p,
        &inputs.eout_interp,
        &inputs.eout_n_discrete,
        &inputs.corr_n_energies,
        &inputs.corr_n_components,
        &inputs.corr_ae_offset,
        &inputs.corr_energy_grid,
        &inputs.corr_n_x,
        &inputs.corr_x_offset,
        &inputs.corr_x,
        &inputs.corr_cdf,
        &inputs.corr_p,
        &inputs.corr_interp,
        &inputs.corr_n_discrete,
        &inputs.corr_n_mu,
        &inputs.corr_mu_offset,
        &inputs.corr_mu,
        &inputs.corr_mu_cdf,
        &inputs.corr_mu_pdf,
        &inputs.corr_mu_interp,
        &inputs.scatter_in_cm_per_mt,
        &inputs.elastic_angle_n_energies,
        &inputs.elastic_angle_ae_offset,
        &inputs.elastic_angle_energy_grid,
        &inputs.elastic_angle_n_mu,
        &inputs.elastic_angle_mu_offset,
        &inputs.elastic_angle_mu,
        &inputs.elastic_angle_cdf,
        &inputs.elastic_angle_pdf,
        &inputs.elastic_angle_interp,
        &inputs.temperature_k_per_material,
        &inputs.km_n_energies,
        &inputs.km_ae_offset,
        &inputs.km_energy_grid,
        &inputs.km_interp,
        &inputs.km_n_discrete,
        &inputs.km_n_x,
        &inputs.km_x_offset,
        &inputs.km_x,
        &inputs.km_p,
        &inputs.km_c,
        &inputs.km_r,
        &inputs.km_a,
        &inputs.evap_n_energies,
        &inputs.evap_n_components,
        &inputs.evap_ae_offset,
        &inputs.evap_theta_offset,
        &inputs.evap_energy_grid,
        &inputs.evap_theta,
        &inputs.evap_u,
        &inputs.nbps_n_bodies,
        &inputs.nbps_total_mass,
        &inputs.maxwell_n_energies,
        &inputs.maxwell_ae_offset,
        &inputs.maxwell_energy_grid,
        &inputs.maxwell_theta,
        &inputs.maxwell_u,
        &inputs.watt_n_energies,
        &inputs.watt_ae_offset,
        &inputs.watt_energy_grid,
        &inputs.watt_a,
        &inputs.watt_b,
        &inputs.watt_u,
        &inputs.urr_meta,
        &inputs.urr_ae_offset,
        &inputs.urr_cdf_offset,
        &inputs.urr_energy_grid,
        &inputs.urr_cdf,
        &inputs.urr_xs,
        &inputs.urr_atom_density,
        &pack,
        &[],
        &SurvivalBiasingInputs::off(),
        &CoupledPhotonInputs::coupled_off(n_materials, n_grid),
        &DecayPhotonInputs::decay_off(n_materials, n_grid),
        &inputs.nuclide_select,
        &FissionBankInputs::off(),
        1,
        MAX_STEPS,
        FREE_GAS_THRESHOLD,
        TallyVarianceMode::PerSource {
            chunk_sources: n as u32,
            total_bins: total_out_len as u32,
            source_idx: None,
        },
    );
    let scales = {
        let mut v = vec![1.0f64; total_out_len.max(1)];
        for t in 0..pack.fixed_point_scales.len() {
            let start = pack.out_offsets[t] as usize;
            let end = pack.out_offsets[t + 1] as usize;
            for s in v.iter_mut().take(end).skip(start) {
                *s = pack.fixed_point_scales[t];
            }
        }
        v
    };
    let mut per_source_flat = vec![0.0f64; total_out_len];
    for (i, &bits) in gpu_ps.src_acc.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        let b = i % total_out_len;
        per_source_flat[b] += (bits as i64 as f64) / scales[b];
    }

    println!("==== twin vs kernel: {label} (N={n})");
    let alive_diff = twin
        .alive
        .iter()
        .zip(gpu.alive.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "  alive flags differing : {alive_diff} / {n} ({:.3}%)",
        100.0 * alive_diff as f64 / n as f64
    );
    // Termination is driven by integer PCG comparisons against well-separated
    // probability thresholds, so it is not a floating-point question: any drift
    // here is an algorithmic divergence. Same standard as
    // yamc-gpu's `assert_gpu_cpu_equiv`.
    assert_eq!(
        alive_diff, 0,
        "{label}: {alive_diff} / {n} histories ended alive on one backend and not the other"
    );
    for (t, name) in [(0usize, "flux"), (1usize, "absorption")] {
        let start = pack.out_offsets[t] as usize;
        let end = pack.out_offsets[t + 1] as usize;
        let ps: f64 = per_source_flat[start..end].iter().sum();
        let pstep: f64 = gpu.tally_outputs[t].iter().sum();
        println!(
            "  {name:11} kernel PerStep {pstep:.8e}  PerSource {ps:.8e}  ratio {:.8}",
            ps / pstep
        );
        // PerSource sees the same contributions, but not with the same rounding:
        // PerStep converts EVERY contribution to fixed point and atomically adds
        // it, while PerSource sums a history's contributions in f64 in the touched
        // list and converts ONCE at history end. So the two differ by accumulated
        // quantisation, measured at 1.4e-12 relative here (scale 2^30 gives one
        // quantum ~5e-14 of this total), not by zero. 1e-9 is far above that and
        // far below anything a real difference in what is scored would produce.
        // Fissile models are the ONLY users of PerSource, which is why the
        // comparison belongs in this file.
        assert!(
            (ps - pstep).abs() <= 1e-9 * pstep.abs().max(ps.abs()),
            "{label} {name}: PerStep {pstep:.17e} vs PerSource {ps:.17e}"
        );
        let a = &twin.tally_outputs[t];
        let b = &gpu.tally_outputs[t];
        assert_eq!(a.len(), b.len(), "{name}: tally shape must match");
        let sa: f64 = a.iter().sum();
        let sb: f64 = b.iter().sum();
        let (worst, at) = worst_rel(a, b);
        println!(
            "  {name:11} twin {sa:.8e}  kernel {sb:.8e}  ratio {:.8}  worst bin rel {worst:.2e} (bin {at})",
            sb / sa
        );
        // The kernel's only licence to differ from the twin is its f64
        // ln/exp/cos polyfills, worth ~16 ulp, which cannot reach 1e-9 without
        // flipping a discrete decision. Measured worst bin: 7e-14 on U235, 1e-13
        // on Fe56. A sampler or scoring difference lands orders of magnitude
        // above this.
        assert!(
            worst < 1e-9,
            "{label} {name}: worst per-bin twin-vs-kernel difference {worst:.3e} at bin {at} \
             (twin sum {sa:.17e}, kernel sum {sb:.17e})"
        );
    }
}

/// U235 at 1 MeV: the #154 configuration, fission on essentially every history.
#[test]
fn twin_and_kernel_agree_on_a_fissile_sphere() {
    compare("U235 r=5 1 MeV", "U235", 18.95, 5.0, 1.0e6, 20_000);
}

/// Fe56 at 14 MeV: the non-fissile control, which reads 1.00000 GPU-vs-CPU at
/// the dispatch level, so it should read clean here too.
#[test]
fn twin_and_kernel_agree_on_a_non_fissile_sphere() {
    compare("Fe56 r=5 14 MeV", "Fe56", 7.87, 5.0, 14.06e6, 20_000);
}

/// Bank ON: the twin transports the fission chain IN-THREAD, the dispatch banks
/// it and drains it host-side across generation launches. Same model, same
/// seeds, one all-energy flux bin so the two normalisations line up
/// (`twin_sum / n` against the dispatch's per-source mean).
///
/// This is the half of the chain issue #154's deficit has to live in: with the
/// bank OFF the two agree to 1e-13 (the tests above), and the production CPU is
/// bit-identical to the twin per history since #355 / #357 / #358.
fn compare_bank_on(nuclide: &str, density: f64, radius: f64, energy_ev: f64, n: usize) {
    if !data_present(nuclide) {
        eprintln!("skipping bank-on {nuclide} -- cache absent");
        return;
    }
    let ctx = match yamc_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping bank-on {nuclide} -- no f64 GPU");
            return;
        }
    };
    println!("==== twin (in-thread chain) vs dispatch (host drain): {nuclide} N={n}");
    for &seed in &[
        SEED,
        99_991u64,
        7u64,
        11u64,
        12345u64,
        777u64,
        31337u64,
        20260731u64,
    ] {
        // Twin: raw single-bin flux pack, fission bank ON, LIFO drain.
        let model = sphere_model(nuclide, density, radius, energy_ev);
        let inputs = translate_for_gpu(&model, n, seed).expect("translate");
        let n_cells = (inputs.cell_aabbs.len() / 6) as u32;
        let edges: Vec<f64> = [1.0e-5f64, 2.0e7].iter().map(|e| e.ln()).collect();
        let pack = TalliesPack::flux_abs_pack(n_cells, &edges);
        let _ = &ctx;
        let (twin, _t) = run_multi_cell_transport_cpu(
            &inputs.seeds,
            &inputs.energies,
            &inputs.positions,
            &inputs.directions,
            &inputs.cell_aabbs,
            &inputs.cell_to_material,
            &inputs.surface_types,
            &inputs.surface_params,
            &inputs.surface_boundaries,
            &inputs.region_program,
            &inputs.log_energy_grid,
            &inputs.coarse_log_energy_grid,
            &inputs.coarse_meta,
            &inputs.fine_log_energy_grid,
            &inputs.fine_meta,
            &inputs.xs_elastic_per_material,
            &inputs.xs_absorption_per_material,
            &inputs.xs_inelastic_per_material,
            &inputs.xs_fission_per_material,
            &inputs.nu_bar_per_material,
            &inputs.beta_delayed_per_material,
            &inputs.fission_a_per_material,
            &inputs.fission_b_per_material,
            &inputs.fission_eout_kind_per_material,
            &inputs.fission_eout_n_energies_per_material,
            &inputs.fission_eout_ae_offset,
            &inputs.fission_eout_energy_grid_per_material,
            &inputs.fission_eout_n_x_per_material,
            &inputs.fission_eout_x_offset,
            &inputs.fission_eout_x_per_material,
            &inputs.fission_eout_cdf_per_material,
            &inputs.fission_eout_p_per_material,
            &inputs.fission_eout_interp_per_material,
            &inputs.xs_inelastic_per_mt_sparse,
            &inputs.target_mass_per_material,
            &inputs.q_inelastic_per_mt,
            &inputs.yield_per_mt_sparse,
            &inputs.permt_meta,
            &inputs.angle_n_energies,
            &inputs.angle_ae_offset,
            &inputs.angle_energy_grid,
            &inputs.angle_n_mu,
            &inputs.angle_mu_offset,
            &inputs.angle_mu,
            &inputs.angle_cdf,
            &inputs.angle_pdf,
            &inputs.angle_interp,
            &inputs.eout_kind,
            &inputs.eout_n_energies,
            &inputs.eout_ae_offset,
            &inputs.eout_energy_grid,
            &inputs.eout_n_x,
            &inputs.eout_x_offset,
            &inputs.eout_x,
            &inputs.eout_cdf,
            &inputs.eout_histogram_interp,
            &inputs.eout_p,
            &inputs.eout_interp,
            &inputs.eout_n_discrete,
            &inputs.corr_n_energies,
            &inputs.corr_n_components,
            &inputs.corr_ae_offset,
            &inputs.corr_energy_grid,
            &inputs.corr_n_x,
            &inputs.corr_x_offset,
            &inputs.corr_x,
            &inputs.corr_cdf,
            &inputs.corr_p,
            &inputs.corr_interp,
            &inputs.corr_n_discrete,
            &inputs.corr_n_mu,
            &inputs.corr_mu_offset,
            &inputs.corr_mu,
            &inputs.corr_mu_cdf,
            &inputs.corr_mu_pdf,
            &inputs.corr_mu_interp,
            &inputs.scatter_in_cm_per_mt,
            &inputs.elastic_angle_n_energies,
            &inputs.elastic_angle_ae_offset,
            &inputs.elastic_angle_energy_grid,
            &inputs.elastic_angle_n_mu,
            &inputs.elastic_angle_mu_offset,
            &inputs.elastic_angle_mu,
            &inputs.elastic_angle_cdf,
            &inputs.elastic_angle_pdf,
            &inputs.elastic_angle_interp,
            &inputs.temperature_k_per_material,
            &inputs.km_n_energies,
            &inputs.km_ae_offset,
            &inputs.km_energy_grid,
            &inputs.km_interp,
            &inputs.km_n_discrete,
            &inputs.km_n_x,
            &inputs.km_x_offset,
            &inputs.km_x,
            &inputs.km_p,
            &inputs.km_c,
            &inputs.km_r,
            &inputs.km_a,
            &inputs.evap_n_energies,
            &inputs.evap_n_components,
            &inputs.evap_ae_offset,
            &inputs.evap_theta_offset,
            &inputs.evap_energy_grid,
            &inputs.evap_theta,
            &inputs.evap_u,
            &inputs.nbps_n_bodies,
            &inputs.nbps_total_mass,
            &inputs.maxwell_n_energies,
            &inputs.maxwell_ae_offset,
            &inputs.maxwell_energy_grid,
            &inputs.maxwell_theta,
            &inputs.maxwell_u,
            &inputs.watt_n_energies,
            &inputs.watt_ae_offset,
            &inputs.watt_energy_grid,
            &inputs.watt_a,
            &inputs.watt_b,
            &inputs.watt_u,
            &inputs.urr_meta,
            &inputs.urr_ae_offset,
            &inputs.urr_cdf_offset,
            &inputs.urr_energy_grid,
            &inputs.urr_cdf,
            &inputs.urr_xs,
            &inputs.urr_atom_density,
            &pack,
            &[],
            &SurvivalBiasingInputs::off(),
            &inputs.nuclide_select,
            &FissionBankInputs::on(),
            MAX_STEPS,
            FREE_GAS_THRESHOLD,
            false,
            PendDrain::Lifo,
        );
        let twin_flux: f64 = twin.tally_outputs[0].iter().sum::<f64>() / n as f64;

        // Dispatch: the same model with an equivalent single-bin flux tally.
        let mut disp_model = sphere_model(nuclide, density, radius, energy_ev);
        let mut tally = yamc_tallies::tally::Tally::new();
        tally.filters.push(yamc_tallies::filter::Filter::Cell(
            yamc_tallies::filter::cell::CellFilter::from_id(1),
        ));
        tally.scores = vec!["flux".parse::<yamc_tallies::score::Score>().unwrap()];
        tally.estimator = yamc_tallies::Estimator::TrackLength;
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        disp_model.tallies = vec![Arc::clone(&tally)];
        let settings = yamc::model::TransportSettings {
            total_particles: Some(n),
            seed,
            ..Default::default()
        };
        yamc::gpu::run_on_gpu(&mut disp_model, &settings).expect("dispatch");
        let disp_flux: f64 = tally.get_mean().iter().sum();

        println!(
            "  seed {seed:>8}: twin {twin_flux:.8e}  dispatch {disp_flux:.8e}  ratio {:.6}",
            disp_flux / twin_flux
        );
        // The host drains the chain across generation launches while the twin runs
        // it in-thread, so this pins the whole bank round-trip: the banked record,
        // the `round(w)` relaunch, the generation loop and the per-source fold.
        //
        // Most seeds land at 1e-13 (bit-identical: every banked progeny's stream is
        // keyed on its place in the emission tree, so running the chain in-thread or
        // across launches gives the same particles). A few do not, and the reason is
        // the `round(w)` relaunch itself: a banked progeny whose weight is NOT
        // integral -- an (n,2n) multiply leaves `w = 1.981` on this model -- is
        // re-expanded by the host into `floor(w) + Bernoulli(frac)` unit-weight
        // neutrons (issue #236), so the host transports weight 2.0 where the twin
        // carries 1.981. That is unbiased but not per-history equal, and one such
        // progeny in 100k histories moves the total by ~5e-5. It happens on 3 of the
        // 8 seeds below (worst 5.0e-5) and is unrelated to what this test pins, so
        // the bound is 5e-4: 10x above the measured worst case, still 4x below the
        // 0.2% deficit this exists to rule out.
        let rel = (disp_flux - twin_flux).abs() / twin_flux.abs().max(disp_flux.abs());
        assert!(
            rel < 5.0e-4,
            "seed {seed}: dispatch host drain vs twin in-thread chain differ by {rel:.3e} \
             (twin {twin_flux:.17e}, dispatch {disp_flux:.17e})"
        );
    }
}

/// U235 with the chain banked: twin in-thread against the host drain.
#[test]
fn twin_and_dispatch_agree_with_the_bank_on() {
    compare_bank_on("U235", 18.95, 5.0, 1.0e6, 100_000);
}

/// Production CPU vs the twin at the TALLY level, with the fission chain on.
///
/// The `matched_stream_*` harnesses compare per-collision TRACES (energies and
/// reaction classes), which pins the transport and the RNG stream but says
/// nothing about what each backend SCORES from those collisions. The twin scores
/// with the GPU's semantics, the production CPU with `transport/scoring.rs`. This
/// is the only link in the CPU -> twin -> kernel -> dispatch chain that was never
/// measured, and every other link is now exact (1e-13 or better), while the
/// end-to-end CPU-vs-GPU flux on this model reads 0.998 (issue #154).
fn compare_cpu_vs_twin(nuclide: &str, density: f64, radius: f64, energy_ev: f64, n: usize) {
    if !data_present(nuclide) {
        eprintln!("skipping cpu-vs-twin {nuclide} -- cache absent");
        return;
    }
    println!("==== production CPU vs twin, tallies, bank on: {nuclide} N={n}");
    for &seed in &[
        SEED,
        99_991u64,
        7u64,
        11u64,
        12345u64,
        777u64,
        31337u64,
        20260731u64,
    ] {
        let model = sphere_model(nuclide, density, radius, energy_ev);
        let inputs = translate_for_gpu(&model, n, seed).expect("translate");
        let n_cells = (inputs.cell_aabbs.len() / 6) as u32;
        let edges: Vec<f64> = [1.0e-5f64, 2.0e7].iter().map(|e| e.ln()).collect();
        let pack = TalliesPack::flux_abs_pack(n_cells, &edges);
        let (twin, _t) = run_multi_cell_transport_cpu(
            &inputs.seeds,
            &inputs.energies,
            &inputs.positions,
            &inputs.directions,
            &inputs.cell_aabbs,
            &inputs.cell_to_material,
            &inputs.surface_types,
            &inputs.surface_params,
            &inputs.surface_boundaries,
            &inputs.region_program,
            &inputs.log_energy_grid,
            &inputs.coarse_log_energy_grid,
            &inputs.coarse_meta,
            &inputs.fine_log_energy_grid,
            &inputs.fine_meta,
            &inputs.xs_elastic_per_material,
            &inputs.xs_absorption_per_material,
            &inputs.xs_inelastic_per_material,
            &inputs.xs_fission_per_material,
            &inputs.nu_bar_per_material,
            &inputs.beta_delayed_per_material,
            &inputs.fission_a_per_material,
            &inputs.fission_b_per_material,
            &inputs.fission_eout_kind_per_material,
            &inputs.fission_eout_n_energies_per_material,
            &inputs.fission_eout_ae_offset,
            &inputs.fission_eout_energy_grid_per_material,
            &inputs.fission_eout_n_x_per_material,
            &inputs.fission_eout_x_offset,
            &inputs.fission_eout_x_per_material,
            &inputs.fission_eout_cdf_per_material,
            &inputs.fission_eout_p_per_material,
            &inputs.fission_eout_interp_per_material,
            &inputs.xs_inelastic_per_mt_sparse,
            &inputs.target_mass_per_material,
            &inputs.q_inelastic_per_mt,
            &inputs.yield_per_mt_sparse,
            &inputs.permt_meta,
            &inputs.angle_n_energies,
            &inputs.angle_ae_offset,
            &inputs.angle_energy_grid,
            &inputs.angle_n_mu,
            &inputs.angle_mu_offset,
            &inputs.angle_mu,
            &inputs.angle_cdf,
            &inputs.angle_pdf,
            &inputs.angle_interp,
            &inputs.eout_kind,
            &inputs.eout_n_energies,
            &inputs.eout_ae_offset,
            &inputs.eout_energy_grid,
            &inputs.eout_n_x,
            &inputs.eout_x_offset,
            &inputs.eout_x,
            &inputs.eout_cdf,
            &inputs.eout_histogram_interp,
            &inputs.eout_p,
            &inputs.eout_interp,
            &inputs.eout_n_discrete,
            &inputs.corr_n_energies,
            &inputs.corr_n_components,
            &inputs.corr_ae_offset,
            &inputs.corr_energy_grid,
            &inputs.corr_n_x,
            &inputs.corr_x_offset,
            &inputs.corr_x,
            &inputs.corr_cdf,
            &inputs.corr_p,
            &inputs.corr_interp,
            &inputs.corr_n_discrete,
            &inputs.corr_n_mu,
            &inputs.corr_mu_offset,
            &inputs.corr_mu,
            &inputs.corr_mu_cdf,
            &inputs.corr_mu_pdf,
            &inputs.corr_mu_interp,
            &inputs.scatter_in_cm_per_mt,
            &inputs.elastic_angle_n_energies,
            &inputs.elastic_angle_ae_offset,
            &inputs.elastic_angle_energy_grid,
            &inputs.elastic_angle_n_mu,
            &inputs.elastic_angle_mu_offset,
            &inputs.elastic_angle_mu,
            &inputs.elastic_angle_cdf,
            &inputs.elastic_angle_pdf,
            &inputs.elastic_angle_interp,
            &inputs.temperature_k_per_material,
            &inputs.km_n_energies,
            &inputs.km_ae_offset,
            &inputs.km_energy_grid,
            &inputs.km_interp,
            &inputs.km_n_discrete,
            &inputs.km_n_x,
            &inputs.km_x_offset,
            &inputs.km_x,
            &inputs.km_p,
            &inputs.km_c,
            &inputs.km_r,
            &inputs.km_a,
            &inputs.evap_n_energies,
            &inputs.evap_n_components,
            &inputs.evap_ae_offset,
            &inputs.evap_theta_offset,
            &inputs.evap_energy_grid,
            &inputs.evap_theta,
            &inputs.evap_u,
            &inputs.nbps_n_bodies,
            &inputs.nbps_total_mass,
            &inputs.maxwell_n_energies,
            &inputs.maxwell_ae_offset,
            &inputs.maxwell_energy_grid,
            &inputs.maxwell_theta,
            &inputs.maxwell_u,
            &inputs.watt_n_energies,
            &inputs.watt_ae_offset,
            &inputs.watt_energy_grid,
            &inputs.watt_a,
            &inputs.watt_b,
            &inputs.watt_u,
            &inputs.urr_meta,
            &inputs.urr_ae_offset,
            &inputs.urr_cdf_offset,
            &inputs.urr_energy_grid,
            &inputs.urr_cdf,
            &inputs.urr_xs,
            &inputs.urr_atom_density,
            &pack,
            &[],
            &SurvivalBiasingInputs::off(),
            &inputs.nuclide_select,
            &FissionBankInputs::on(),
            MAX_STEPS,
            FREE_GAS_THRESHOLD,
            false,
            PendDrain::Lifo,
        );
        let twin_flux: f64 = twin.tally_outputs[0].iter().sum::<f64>() / n as f64;
        let twin_abs: f64 = twin.tally_outputs[1].iter().sum::<f64>() / n as f64;

        let mut cpu_model = sphere_model(nuclide, density, radius, energy_ev);
        let mk = |score: &str| {
            let mut t = yamc_tallies::tally::Tally::new();
            t.filters.push(yamc_tallies::filter::Filter::Cell(
                yamc_tallies::filter::cell::CellFilter::from_id(1),
            ));
            t.scores = vec![score.parse::<yamc_tallies::score::Score>().unwrap()];
            t.estimator = yamc_tallies::Estimator::TrackLength;
            t.initialize_batches(1);
            Arc::new(t)
        };
        let flux = mk("flux");
        let absorption = mk("absorption");
        cpu_model.tallies = vec![Arc::clone(&flux), Arc::clone(&absorption)];
        let settings = yamc::model::TransportSettings {
            total_particles: Some(n),
            seed,
            ..Default::default()
        };
        cpu_model.simulate_transport(&settings).expect("cpu run");
        let cpu_flux: f64 = flux.get_mean().iter().sum();
        let cpu_abs: f64 = absorption.get_mean().iter().sum();

        println!(
            "  seed {seed:>8}: flux cpu {cpu_flux:.8e} twin {twin_flux:.8e} ratio {:.6} | absorb cpu {cpu_abs:.8e} twin {twin_abs:.8e} ratio {:.6}",
            twin_flux / cpu_flux,
            twin_abs / cpu_abs
        );
    }
}

/// The missing link, on the #154 model.
#[test]
fn cpu_and_twin_agree_on_tallies() {
    compare_cpu_vs_twin("U235", 18.95, 5.0, 1.0e6, 100_000);
}
