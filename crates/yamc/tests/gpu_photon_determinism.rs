//! GPU photon kernel reproducibility regression test.
//!
//! Same model + same seed must produce bit-identical tally output on
//! GPU. Asserts:
//!   - n=1k single-batch tally is bit-identical across two runs.
//!   - n=50k single-batch tally is bit-identical across two runs.
//!   - Two identical-input kernel launches produce bit-identical
//!     per-particle outputs and tally outputs.
//!
//! Status: all three run unconditionally since cubecl 0.11.0-pre.3.
//!
//! On cubecl-spirv 0.10 + Vulkan/RADV the two larger tests failed and
//! were `#[ignore]`d, against an upstream bug rather than a yamc one
//! (<https://github.com/tracel-ai/cubecl/issues/1336>): above ~544
//! bytes of total thread-private state, cubecl-spirv silently demoted
//! some `Function`-storage variables to a location that raced across
//! threads, regardless of whether the cascade stack was held as
//! `Array::new(16usize)`, two `Array::new(8usize)` halves, sixteen
//! `CascadeSlot` struct locals, or 128 individual scalars. Only
//! dropping below the threshold (`PHOTON_CASCADE_STACK_CAP = 8`)
//! restored determinism, and CAP=8 is unacceptable for high-Z + high-E
//! pair cascades, so the kernel kept CAP=16 and lived with the race.
//!
//! The pliron rewrite of the SPIR-V backend in cubecl 0.11.0-pre.3
//! (zero-initialised arrays, SROA pass) removed it: re-tested
//! 2026-09-13 on RADV STRIX_HALO / Mesa 26.0.3, the 50k-history run
//! and the repeated identical-input launch are bit-identical across
//! repeated runs. These tests are the regression guard for it.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, PhotonComponent, PhotonXSScore, Score};
use yamc_tallies::tally::Tally;

fn make_tally(cell: &Cell, score: Score, n_batches: usize) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
    t.scores = vec![score];
    t.initialize_batches(n_batches);
    Arc::new(t)
}

fn build_model(n: usize, seed: u64) -> (Model, Vec<Arc<Tally>>, TransportSettings) {
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
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![5.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let n_batches = 1;
    let tallies: Vec<Arc<Tally>> = vec![
        make_tally(&cell, Score::Flux(FluxScore), n_batches),
        make_tally(
            &cell,
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Coherent,
            }),
            n_batches,
        ),
        make_tally(
            &cell,
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Incoherent,
            }),
            n_batches,
        ),
        make_tally(
            &cell,
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Photoelectric,
            }),
            n_batches,
        ),
        make_tally(
            &cell,
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::PairProduction,
            }),
            n_batches,
        ),
    ];
    let tally_refs: Vec<Arc<Tally>> = tallies.iter().map(Arc::clone).collect();
    let mut model = Model::new(geometry, vec![source], tally_refs);
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.electron_treatment = yamc::model::ElectronTreatment::Ttb;
    let _ = model.ensure_photon_data_for_gpu();
    let settings = TransportSettings {
        total_particles: Some(n),
        seed,
        ..Default::default()
    };
    (model, tallies, settings)
}

fn run_gpu(n: usize) -> Vec<Vec<f64>> {
    let (mut model, tallies, settings) = build_model(n, 1);
    yamc::gpu::run_on_gpu(&mut model, &settings).expect("GPU dispatch");
    tallies.iter().map(|t| t.get_mean().to_vec()).collect()
}

#[test]
fn gpu_photon_kernel_is_reproducible_small() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping - no GPU with f64 compute available");
        return;
    }
    let a = run_gpu(1_000);
    let b = run_gpu(1_000);
    assert_eq!(a, b, "n=1000 GPU photon tally differs between runs");
}

#[test]
fn gpu_photon_kernel_is_reproducible_large() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping - no GPU with f64 compute available");
        return;
    }
    let a = run_gpu(50_000);
    let b = run_gpu(50_000);
    eprintln!("run A: {a:?}");
    eprintln!("run B: {b:?}");
    assert_eq!(
        a, b,
        "n=50000 GPU photon tally differs between runs (race in kernel)"
    );
}

#[test]
fn gpu_photon_kernel_repeated_launch_with_identical_inputs() {
    // Builds the GPU inputs ONCE, then calls the kernel twice with
    // the SAME inputs. If outputs differ → kernel is internally racy.
    // If outputs match → the variance comes from dispatch translation
    // or input regeneration, not the kernel.
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping - no GPU with f64 compute available");
        return;
    }
    use yamc::gpu::translate_photon::translate_photon_for_gpu;

    let (model, _tallies, settings) = build_model(50_000, 1);
    let inputs = translate_photon_for_gpu(&model, 50_000, settings.seed).expect("translate");
    let ctx = yamc_gpu::GpuContext::new().expect("ctx");

    let pack = build_pack_for_test(&model, &inputs);

    let r1 = launch(&ctx, &inputs, &pack);
    let r2 = launch(&ctx, &inputs, &pack);

    assert_eq!(
        r1.alive, r2.alive,
        "per-particle alive flags differ across identical-input launches",
    );
    assert_eq!(
        r1.n_steps, r2.n_steps,
        "per-particle n_steps differ across identical-input launches",
    );
    let mut first_diff = None;
    for (i, (a, b)) in r1.final_energies.iter().zip(&r2.final_energies).enumerate() {
        if a != b {
            first_diff = Some((i, *a, *b));
            break;
        }
    }
    assert!(
        first_diff.is_none(),
        "final_energy differs at particle {first_diff:?}",
    );
    assert_eq!(
        r1.tally_outputs, r2.tally_outputs,
        "tally outputs differ across identical-input launches",
    );
}

fn build_pack_for_test(
    model: &Model,
    _inputs: &yamc::gpu::translate_photon::GpuPhotonTransportInputs,
) -> yamc_gpu::common::tallies::TalliesPack {
    use yamc_gpu::common::tallies::TalliesPack;
    let n_cells = model.geometry.num_cells() as u32;
    TalliesPack::dummy_single_bin(n_cells)
}

fn launch(
    ctx: &yamc_gpu::GpuContext,
    inputs: &yamc::gpu::translate_photon::GpuPhotonTransportInputs,
    pack: &yamc_gpu::common::tallies::TalliesPack,
) -> yamc_gpu::photon::transport::PhotonMultiCellResult {
    let weights = vec![1.0_f64; inputs.seeds.len()];
    let parent_ids = vec![0u32; inputs.seeds.len()];
    yamc_gpu::photon::transport::run_multi_cell_photon_transport(
        ctx,
        &inputs.seeds,
        &inputs.energies,
        &weights,
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
        5_000,
        1000.0, // photon_cutoff_energy (default, issue #286)
        yamc_gpu::common::tallies::TallyVarianceMode::PerStep,
    )
}
