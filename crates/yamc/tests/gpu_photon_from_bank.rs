//! GPU coupled neutron->photon drain (slice S5) integration tests.
//!
//! Exercises the photon kernel's new per-particle initial weight and the
//! `from_bank` drain adapter against the real Fe-sphere photon transport
//! path (the same geometry / cross-section setup as the photon
//! determinism + Co60 tests):
//!
//! - **Bank-source equivalence**: a synthetic device bank of N photons,
//!   drained and transported via `run_photon_transport_from_bank`, yields
//!   tallies byte-identical to transporting the SAME N photons (same
//!   seeds/energies/positions/directions, all weight 1.0) directly via
//!   `run_multi_cell_photon_transport`.
//! - **Weight linearity**: a banked photon at weight 2.0 contributes
//!   exactly 2x the tally of an otherwise-identical weight-1.0 photon,
//!   pinning that Part A applies weight at every score site.
//!
//! N is kept small and the photons monoenergetic. That was originally
//! so the cubecl-spirv 0.10 thread-private-memory race (see
//! `gpu_photon_determinism.rs`, fixed upstream in 0.11.0-pre.3) could not
//! perturb the byte-equality assertions; it also keeps the test fast.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::translate_photon::{translate_photon_for_gpu, GpuPhotonTransportInputs};
use yamc::model::Model;
use yamc_gpu::common::particle_bank::{
    BANK_F64_STRIDE, BANK_U32_STRIDE, PTYPE_NEUTRON, PTYPE_PHOTON,
};
use yamc_gpu::common::tallies::TalliesPack;
use yamc_gpu::photon::from_bank::run_photon_transport_from_bank;
use yamc_gpu::photon::transport::{run_multi_cell_photon_transport, PhotonMultiCellResult};
use yamc_gpu::GpuContext;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const MAX_STEPS: u32 = 5_000;

/// 1 cm Fe sphere with a monoenergetic point photon source -- the shared
/// fixture for both tests (mirrors `gpu_photon_determinism::build_model`).
fn fe_sphere_model(source_e: f64) -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_e], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.transport_secondary_photons = true;
    model.electron_treatment = yamc::model::ElectronTreatment::Ttb;
    let _ = model.ensure_photon_data_for_gpu();
    model
}

/// Hand-pack the photon source held in `inputs` (a populated GPU source)
/// into a fresh device-bank-format pair of stride-packed arrays, tagging
/// every record `PTYPE_PHOTON` with the given per-record weights. Slots
/// `n..capacity` stay zero (unwritten), exactly as a real drained bank.
fn pack_bank(
    inputs: &GpuPhotonTransportInputs,
    weights: &[f64],
    capacity: usize,
) -> (Vec<f64>, Vec<u32>) {
    let n = inputs.seeds.len();
    assert!(n <= capacity);
    assert_eq!(weights.len(), n);
    let mut bf = vec![0.0_f64; capacity * BANK_F64_STRIDE];
    let mut bu = vec![0u32; capacity * BANK_U32_STRIDE];
    // `s` indexes four parallel source arrays at different strides (1, 3,
    // 3, 1), so a single range loop is clearer than zipping iterators.
    #[allow(clippy::needless_range_loop)]
    for s in 0..n {
        let f = s * BANK_F64_STRIDE;
        bf[f] = inputs.energies[s];
        bf[f + 1] = inputs.positions[s * 3];
        bf[f + 2] = inputs.positions[s * 3 + 1];
        bf[f + 3] = inputs.positions[s * 3 + 2];
        bf[f + 4] = inputs.directions[s * 3];
        bf[f + 5] = inputs.directions[s * 3 + 1];
        bf[f + 6] = inputs.directions[s * 3 + 2];
        bf[f + 7] = weights[s];
        let u = s * BANK_U32_STRIDE;
        bu[u] = PTYPE_PHOTON;
        bu[u + 1] = 12345; // cell -- must be IGNORED (re-derived via BVH)
        bu[u + 2] = inputs.seeds[s];
        bu[u + 3] = 0; // gen
    }
    (bf, bu)
}

/// Transport `inputs` directly through the photon launcher with the given
/// per-particle weights.
fn run_direct(
    ctx: &GpuContext,
    inputs: &GpuPhotonTransportInputs,
    weights: &[f64],
    pack: &TalliesPack,
) -> PhotonMultiCellResult {
    let parent_ids = vec![0u32; inputs.seeds.len()];
    run_multi_cell_photon_transport(
        ctx,
        &inputs.seeds,
        &inputs.energies,
        weights,
        &parent_ids,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.xs_total,
        &inputs.xs_coherent,
        &inputs.xs_incoherent,
        &inputs.xs_photoelectric,
        &inputs.xs_pair,
        &inputs.xs_heating,
        &inputs.rayleigh_x2,
        &inputs.rayleigh_cdf,
        &inputs.rayleigh_n_points,
        &inputs.ttb.e_grid_log,
        &inputs.ttb.electron_pdf,
        &inputs.ttb.electron_cdf,
        &inputs.ttb.electron_yield,
        &inputs.ttb.has_data,
        &inputs.doppler.pz_grid,
        &inputs.doppler.electron_pdf,
        &inputs.doppler.binding_energy,
        &inputs.doppler.profile_pdf,
        &inputs.doppler.profile_cdf,
        &inputs.doppler.n_shells,
        &inputs.doppler.has_data,
        &inputs.doppler.subshell_idx,
        &inputs.doppler.subshell_w0,
        &inputs.doppler.subshell_cnt,
        &inputs.iff.x,
        &inputs.iff.s,
        &inputs.iff.n_points,
        &inputs.iff.has_data,
        &inputs.atomic_relaxation.has_data,
        &inputs.atomic_relaxation.n_shells,
        &inputs.atomic_relaxation.binding_energy,
        &inputs.atomic_relaxation.pe_subshell_xs_log,
        &inputs.atomic_relaxation.n_trans,
        &inputs.atomic_relaxation.trans_primary,
        &inputs.atomic_relaxation.trans_secondary,
        &inputs.atomic_relaxation.trans_energy,
        &inputs.atomic_relaxation.trans_cum_prob,
        &inputs.ttb.positron_pdf,
        &inputs.ttb.positron_cdf,
        &inputs.ttb.positron_yield,
        &inputs.pair.has_data,
        &inputs.pair.r_z,
        &inputs.pair.a,
        &inputs.pair.c,
        &inputs.element_select.elem_macro_total,
        &inputs.element_select.mat_elem_meta,
        pack,
        MAX_STEPS,
        1000.0, // photon_cutoff_energy (default, issue #286)
        yamc_gpu::common::tallies::TallyVarianceMode::PerStep,
    )
}

/// Transport a populated bank through the drain adapter, forwarding the
/// same geometry / cross-section arrays held in `inputs`.
fn run_from_bank(
    ctx: &GpuContext,
    bank_f64: &[f64],
    bank_u32: &[u32],
    count: usize,
    inputs: &GpuPhotonTransportInputs,
    pack: &TalliesPack,
) -> PhotonMultiCellResult {
    run_photon_transport_from_bank(
        ctx,
        bank_f64,
        bank_u32,
        count,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.xs_total,
        &inputs.xs_coherent,
        &inputs.xs_incoherent,
        &inputs.xs_photoelectric,
        &inputs.xs_pair,
        &inputs.xs_heating,
        &inputs.rayleigh_x2,
        &inputs.rayleigh_cdf,
        &inputs.rayleigh_n_points,
        &inputs.ttb.e_grid_log,
        &inputs.ttb.electron_pdf,
        &inputs.ttb.electron_cdf,
        &inputs.ttb.electron_yield,
        &inputs.ttb.has_data,
        &inputs.doppler.pz_grid,
        &inputs.doppler.electron_pdf,
        &inputs.doppler.binding_energy,
        &inputs.doppler.profile_pdf,
        &inputs.doppler.profile_cdf,
        &inputs.doppler.n_shells,
        &inputs.doppler.has_data,
        &inputs.doppler.subshell_idx,
        &inputs.doppler.subshell_w0,
        &inputs.doppler.subshell_cnt,
        &inputs.iff.x,
        &inputs.iff.s,
        &inputs.iff.n_points,
        &inputs.iff.has_data,
        &inputs.atomic_relaxation.has_data,
        &inputs.atomic_relaxation.n_shells,
        &inputs.atomic_relaxation.binding_energy,
        &inputs.atomic_relaxation.pe_subshell_xs_log,
        &inputs.atomic_relaxation.n_trans,
        &inputs.atomic_relaxation.trans_primary,
        &inputs.atomic_relaxation.trans_secondary,
        &inputs.atomic_relaxation.trans_energy,
        &inputs.atomic_relaxation.trans_cum_prob,
        &inputs.ttb.positron_pdf,
        &inputs.ttb.positron_cdf,
        &inputs.ttb.positron_yield,
        &inputs.pair.has_data,
        &inputs.pair.r_z,
        &inputs.pair.a,
        &inputs.pair.c,
        &inputs.element_select.elem_macro_total,
        &inputs.element_select.mat_elem_meta,
        pack,
        MAX_STEPS,
        1000.0, // photon_cutoff_energy (default, issue #286)
        yamc_gpu::common::tallies::TallyVarianceMode::PerStep,
    )
}

/// Bank-source equivalence: draining a synthetic bank of N photons and
/// transporting them must produce byte-identical tallies to transporting
/// the SAME N photons directly with all-1.0 weights. This pins that the
/// drain reconstructs the SoA source faithfully (energy/pos/dir/seed/
/// weight) and that the kernel re-derives the cell from position (the
/// bank's bogus `cell = 12345` is ignored).
#[test]
fn from_bank_matches_direct_transport() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }
    };

    let n = 256usize;
    let model = fe_sphere_model(2.0e6);
    let inputs = translate_photon_for_gpu(&model, n, 7).expect("translate");
    let n_cells = inputs.cell_aabbs.len() / 6;
    let pack = TalliesPack::dummy_single_bin(n_cells as u32);

    // Synthetic bank: all weight 1.0, with extra unwritten capacity.
    let weights = vec![1.0_f64; n];
    let capacity = n + 64;
    let (bf, bu) = pack_bank(&inputs, &weights, capacity);

    let r_bank = run_from_bank(&ctx, &bf, &bu, n, &inputs, &pack);
    let r_direct = run_direct(&ctx, &inputs, &weights, &pack);

    assert_eq!(
        r_bank.tally_outputs, r_direct.tally_outputs,
        "drained-bank tally must equal direct transport of the same photons"
    );
    // The flux tally must actually accumulate something (guards against a
    // vacuously-equal all-zero comparison).
    let total: f64 = r_direct.tally_outputs[0].iter().sum();
    assert!(total > 0.0, "flux tally is zero -- fixture is degenerate");
}

/// Weight linearity: a single banked photon at weight 2.0 contributes
/// exactly twice the (track-length flux) tally of the identical photon at
/// weight 1.0. Same seed/energy/position/direction so the transport path
/// is bit-identical and only the weight factor differs -- pins that Part
/// A multiplies weight in at every score site.
#[test]
fn banked_weight_scales_tally_linearly() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }
    };

    // A handful of photons; weight is the only difference between runs.
    let n = 16usize;
    let model = fe_sphere_model(1.0e6);
    let inputs = translate_photon_for_gpu(&model, n, 13).expect("translate");
    let n_cells = inputs.cell_aabbs.len() / 6;
    let pack = TalliesPack::dummy_single_bin(n_cells as u32);

    let capacity = n;
    let w1 = vec![1.0_f64; n];
    let w2 = vec![2.0_f64; n];
    let (bf1, bu1) = pack_bank(&inputs, &w1, capacity);
    let (bf2, bu2) = pack_bank(&inputs, &w2, capacity);

    let r1 = run_from_bank(&ctx, &bf1, &bu1, n, &inputs, &pack);
    let r2 = run_from_bank(&ctx, &bf2, &bu2, n, &inputs, &pack);

    let t1: f64 = r1.tally_outputs[0].iter().sum();
    let t2: f64 = r2.tally_outputs[0].iter().sum();
    assert!(
        t1 > 0.0,
        "weight-1.0 flux tally is zero -- fixture degenerate"
    );
    // Fixed-point accumulation can round each photon's per-step
    // contribution by up to 0.5 ULP of the fixed-point scale; with the
    // identical transport path, weight 2.0 doubles each score exactly in
    // real arithmetic, so the ratio is 2.0 to well within rounding.
    let ratio = t2 / t1;
    assert!(
        (ratio - 2.0).abs() < 1e-9,
        "weight-2.0 tally {t2} should be ~2x weight-1.0 tally {t1} (ratio {ratio})"
    );
}

/// A bank with a neutron record interleaved among photons drains only the
/// photon records: the neutron's weight does NOT leak into the photon
/// transport. Run through the full drain+transport path with a neutron
/// slot spliced in, and confirm the tally equals the photon-only direct
/// transport.
#[test]
fn from_bank_skips_interleaved_neutron() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }
    };

    let n = 64usize;
    let model = fe_sphere_model(1.5e6);
    let inputs = translate_photon_for_gpu(&model, n, 21).expect("translate");
    let n_cells = inputs.cell_aabbs.len() / 6;
    let pack = TalliesPack::dummy_single_bin(n_cells as u32);

    let weights = vec![1.0_f64; n];
    // Pack the n photons, then splice a NEUTRON record into a fresh slot
    // after them. The drain must skip it; `count` covers the neutron slot.
    let capacity = n + 4;
    let (mut bf, mut bu) = pack_bank(&inputs, &weights, capacity);
    let neu = n; // first free slot
    let f = neu * BANK_F64_STRIDE;
    bf[f] = 14.0e6; // neutron energy -- must be ignored
    bf[f + 7] = 5.0; // neutron weight -- must NOT leak into photon flux
    let u = neu * BANK_U32_STRIDE;
    bu[u] = PTYPE_NEUTRON;
    let count = n + 1; // drain spans the neutron slot too

    let r_bank = run_from_bank(&ctx, &bf, &bu, count, &inputs, &pack);
    let r_direct = run_direct(&ctx, &inputs, &weights, &pack);

    assert_eq!(
        r_bank.tally_outputs, r_direct.tally_outputs,
        "interleaved neutron must be skipped -- photon-only tally must match"
    );
}
