//! GPU support for several scores on one tally (issue #271), on-hardware.
//!
//! `score` is the OUTERMOST dimension of the CPU's 7D bin layout, with a stride
//! equal to the whole remaining block (`Tally::get_bin_index_7d`). So the
//! dispatch expands a multi-score tally into one single-score PACK ENTRY per
//! score and concatenates their blocks in score order at writeback; the kernel
//! never learns about multi-score tallies at all. This test therefore checks the
//! BIN LAYOUT and the per-score kernel routing, not the physics.
//!
//! The multi-score tally deliberately mixes three scores that take three
//! DIFFERENT kernel paths and two different fixed-point accumulator scales:
//!
//!   * `flux`      -- the kernel's flux fast path, default 2^30 scale
//!   * `heating`   -- KERMA-shape, routed through SCORE_PER_MT with the KERMA
//!     scale of 1.0 (eV-magnitude contributions)
//!   * `(n,gamma)` -- SCORE_PER_MT with a per-MT scale sized from MT 102
//!
//! A per-TALLY scale (rather than per-score) would silently wreck at least one
//! of them, which is exactly the failure this mix is chosen to catch.
//!
//! Every assertion is an exact arithmetic identity, not a statistical
//! comparison: the multi-score tally and the three single-score tallies run in
//! the SAME launch over the SAME histories, so a correct layout reproduces them
//! bin for bin. A separate pair of runs then pins GPU/CPU agreement.
//!
//! Needs a real f64 GPU adapter; self-skips otherwise. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_multi_score_tally -- --nocapture --test-threads=1

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const N_PARTICLES: usize = 40_000;
const N_BATCHES: usize = 8;
const SEED: u64 = 20260803;

/// The three scores the multi-score tally carries, in order. Chosen to span
/// three kernel routes and two fixed-point scales (see the module docs).
const SCORES: [&str; 3] = ["flux", "heating", "(n,gamma)"];

/// Energy edges for the score x energy variant: three bins from the 14 MeV
/// source down to thermal.
const E_EDGES: [f64; 4] = [1e-3, 1e5, 1e6, 2e7];

/// Cells the tallies bin over.
const CELL_IDS: [u32; 3] = [1, 2, 3];

fn sphere(id: usize, r: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        r,
        Some(id),
        Some(boundary),
    ))
}

fn iron(density_g_cm3: f64) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density_g_cm3),
    )
    .unwrap();
    m.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

/// Three nested iron shells of differing density, so the three cell bins carry
/// genuinely different magnitudes and a collapsed spatial dimension would show.
fn build_geometry() -> Geometry {
    let s1 = sphere(1, 3.0, BoundaryType::Transmission);
    let s2 = sphere(2, 7.0, BoundaryType::Transmission);
    let s3 = sphere(3, 12.0, BoundaryType::Vacuum);

    let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s1)));
    let shell = |inner: &Arc<Surface>, outer: &Arc<Surface>| {
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(inner))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(outer))),
        )
    };

    let cells = vec![
        Cell::new(Some(1), core, Some("core".into()), Some(0)),
        Cell::new(Some(2), shell(&s1, &s2), Some("mid".into()), Some(1)),
        Cell::new(Some(3), shell(&s2, &s3), Some("outer".into()), Some(2)),
    ];
    Geometry::new(cells, vec![iron(2.0), iron(5.0), iron(7.87)]).unwrap()
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn tally(name: &str, scores: &[&str], filters: Vec<Filter>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters = filters;
    t.scores = scores.iter().map(|s| s.parse::<Score>().unwrap()).collect();
    t.estimator = Estimator::TrackLength;
    t.name = Some(name.to_string());
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn cell_filter() -> Filter {
    Filter::Cell(CellFilter {
        cell_ids: CELL_IDS.to_vec(),
    })
}

/// Tally order is fixed:
///   0        `[flux, heating, (n,gamma)]` over 3 cells         -- 9 bins
///   1,2,3    each score alone over the same 3 cells            -- 3 bins each
///   4        `[flux, heating, (n,gamma)]` over cells x energy   -- 27 bins
///   5,6,7    each score alone over cells x energy               -- 9 bins each
fn build_tallies() -> Vec<Arc<Tally>> {
    let mut v = vec![tally("multi", &SCORES, vec![cell_filter()])];
    for s in SCORES {
        v.push(tally(&format!("single_{s}"), &[s], vec![cell_filter()]));
    }
    let with_energy = || {
        vec![
            cell_filter(),
            Filter::Energy(EnergyFilter::new(E_EDGES.to_vec())),
        ]
    };
    v.push(tally("multi_energy", &SCORES, with_energy()));
    for s in SCORES {
        v.push(tally(&format!("single_energy_{s}"), &[s], with_energy()));
    }
    v
}

fn model(tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut m = Model::new(build_geometry(), vec![neutron_source()], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_PARTICLES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

fn means(tallies: &[Arc<Tally>]) -> Vec<Vec<f64>> {
    tallies.iter().map(|t| t.get_mean().to_vec()).collect()
}

fn assert_close(label: &str, got: f64, want: f64, rtol: f64) {
    let denom = want.abs().max(f64::MIN_POSITIVE);
    let rel = (got - want).abs() / denom;
    assert!(
        rel <= rtol,
        "{label}: got {got:.12e}, want {want:.12e} (relative {rel:.3e} > {rtol:.1e})"
    );
}

/// A multi-score tally's score-k block must equal the same score run as its own
/// tally in the same launch, bin for bin. Exact, because both score the same
/// histories through the same kernel route.
#[test]
fn gpu_multi_score_blocks_match_single_score_tallies() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let tallies = build_tallies();
    let (mut m, settings) = model(tallies.clone());
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");
    let out = means(&tallies);

    let n_cells = CELL_IDS.len();
    let n_e = E_EDGES.len() - 1;
    assert_eq!(out[0].len(), SCORES.len() * n_cells, "multi bins");
    assert_eq!(
        out[4].len(),
        SCORES.len() * n_cells * n_e,
        "multi x energy bins"
    );

    // Every score must actually register, else the block comparisons below
    // could pass on all-zero data.
    for (k, s) in SCORES.iter().enumerate() {
        let block = &out[0][k * n_cells..(k + 1) * n_cells];
        assert!(
            block.iter().all(|v| *v > 0.0),
            "score {s} scored a zero cell bin: {block:?}"
        );
    }

    // Block k of the multi-score tally vs the standalone tally for score k.
    for (k, s) in SCORES.iter().enumerate() {
        let multi = &out[0][k * n_cells..(k + 1) * n_cells];
        let single = &out[1 + k];
        eprintln!("  {s:11} multi {multi:?}");
        eprintln!("  {s:11} single {single:?}");
        for c in 0..n_cells {
            assert_close(
                &format!("score {s} cell bin {c}"),
                multi[c],
                single[c],
                1e-9,
            );
        }
    }

    // Same again with an energy dimension inside the score dimension. This is
    // what pins the STRIDE: score outermost, then cell, energy innermost. A
    // transposed layout would interleave the three scores' spectra and fail.
    let block_e = n_cells * n_e;
    for (k, s) in SCORES.iter().enumerate() {
        let multi = &out[4][k * block_e..(k + 1) * block_e];
        let single = &out[5 + k];
        for b in 0..block_e {
            assert_close(
                &format!("score {s} cell x energy bin {b}"),
                multi[b],
                single[b],
                1e-9,
            );
        }
    }

    // The scores must be mutually distinguishable, otherwise a layout that
    // wrote the same block three times would satisfy everything above.
    for k in 1..SCORES.len() {
        let a = out[0][0];
        let b = out[0][k * n_cells];
        let ratio = (a / b).abs();
        assert!(
            !(0.999..=1.001).contains(&ratio),
            "scores {} and {} are indistinguishable in cell 0 ({a:.6e} vs {b:.6e}); \
             the block comparison is not discriminating",
            SCORES[0],
            SCORES[k]
        );
    }

    // Folding a score's energy bins recovers its unbinned value: an independent
    // cross-check between the binned and unbinned tallies.
    //
    // NOT exact for every score, and not because of the layout. The KERMA-shape
    // accumulator uses a fixed-point scale of 1.0 (heating contributions are
    // eV-magnitude, so a larger scale would overflow the u64 atomic), which
    // rounds every contribution to an integer count. Folding three energy bins
    // therefore carries three roundings against the unbinned tally's one, worth
    // ~0.012 absolute on ~1.1e5 here. Since that belongs to the accumulator
    // rather than the score dimension, the STANDALONE single-score tallies must
    // show the identical discrepancy -- which is what makes this a property
    // check rather than a loosened tolerance.
    for (k, s) in SCORES.iter().enumerate() {
        for c in 0..n_cells {
            let fold =
                |t: &[f64], base: usize| -> f64 { (0..n_e).map(|e| t[base + c * n_e + e]).sum() };
            let folded_multi = fold(&out[4], k * block_e);
            let folded_single = fold(&out[5 + k], 0);
            assert_close(
                &format!("score {s} cell {c} energy fold, multi vs single tally"),
                folded_multi,
                folded_single,
                1e-12,
            );
            assert_close(
                &format!("score {s} cell {c} folded over energy"),
                folded_multi,
                out[0][k * n_cells + c],
                1e-6,
            );
            // And the residual must be the same on both, i.e. attributable to
            // the accumulator and not to the multi-score block layout.
            let unbinned_single = out[1 + k][c];
            assert_close(
                &format!("score {s} cell {c} fold residual, multi vs single"),
                folded_multi - out[0][k * n_cells + c],
                folded_single - unbinned_single,
                1e-12,
            );
        }
    }
}

/// The GPU's multi-score tally must agree with the CPU's, which walks
/// `Tally::score_track_length`'s per-score loop instead of one pack entry per
/// score. Independent MC estimates, so a band rather than an identity.
#[test]
fn gpu_multi_score_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let cpu_tallies = build_tallies();
    let (mut cpu_model, cpu_settings) = model(cpu_tallies.clone());
    cpu_model
        .simulate_transport(&cpu_settings)
        .expect("CPU run");
    let cpu = means(&cpu_tallies);

    let gpu_tallies = build_tallies();
    let (mut gpu_model, gpu_settings) = model(gpu_tallies.clone());
    yamc::gpu::run_on_gpu(&mut gpu_model, &gpu_settings).expect("GPU run");
    let gpu = means(&gpu_tallies);

    let n_cells = CELL_IDS.len();
    for (k, s) in SCORES.iter().enumerate() {
        for c in 0..n_cells {
            let i = k * n_cells + c;
            let (cv, gv) = (cpu[0][i], gpu[0][i]);
            assert!(cv > 0.0, "CPU score {s} cell {c} must be > 0, got {cv}");
            let ratio = gv / cv;
            eprintln!("  {s:11} cell {c}: CPU {cv:.5e}  GPU {gv:.5e}  ratio {ratio:.4}");
            assert!(
                (0.95..=1.05).contains(&ratio),
                "score {s} cell {c}: GPU/CPU ratio {ratio:.4} outside [0.95, 1.05] \
                 (CPU {cv:.5e}, GPU {gv:.5e})"
            );
        }
    }

    // The CPU must itself agree that a multi-score tally's block k equals the
    // standalone score-k tally, so the identity the GPU test asserts is the
    // right one to assert.
    for (k, s) in SCORES.iter().enumerate() {
        for c in 0..n_cells {
            assert_close(
                &format!("CPU score {s} cell {c} multi vs single"),
                cpu[0][k * n_cells + c],
                cpu[1 + k][c],
                1e-9,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Coupled (neutron + photon) runs
// ---------------------------------------------------------------------------
//
// The coupled dispatch routes each tally to the neutron pass, the photon pass,
// or BOTH ("dual", summed), and the pack for each pass is the pass's own
// tallies followed by the dual ones. Expanding per score moves those boundaries
// from tally counts to score counts, which is what these tests pin. All three
// routes are exercised in one launch.

/// A coupled Fe sphere with photon transport on, plus a 14 MeV neutron source.
fn coupled_geometry_and_source() -> (Geometry, ParticleSource) {
    let s = sphere(1, 10.0, BoundaryType::Vacuum);
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s)));

    let mut material = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, neutron_source())
}

/// Both of these scores want the photon pass on an unfiltered tally, so an
/// unfiltered `[flux, heating]` routes to `dual` as a unit.
const COUPLED_SCORES: [&str; 2] = ["flux", "heating"];

fn coupled_model(tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let (geometry, source) = coupled_geometry_and_source();
    let mut m = Model::new(geometry, vec![source], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.max_steps_per_particle = 5_000;
    m.transport_secondary_photons = true;
    m.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(N_PARTICLES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

fn particle_tally(name: &str, scores: &[&str], pt: Option<ParticleType>) -> Arc<Tally> {
    let mut filters = vec![Filter::Cell(CellFilter { cell_ids: vec![1] })];
    if let Some(pt) = pt {
        filters.push(Filter::ParticleType(ParticleTypeFilter::new(pt)));
    }
    tally(name, scores, filters)
}

/// In a coupled run, a multi-score tally's per-score blocks must match the
/// standalone single-score tallies on every one of the three routes: the
/// neutron pass, the photon pass, and the dual (unfiltered) pass whose two
/// contributions are summed. Exact -- same launch, same histories.
#[test]
fn gpu_coupled_multi_score_blocks_match_single_score_tallies() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    if !std::path::Path::new("tests/Fe.arrow").exists() {
        eprintln!("skipping -- photon fixture tests/Fe.arrow missing");
        return;
    }

    // [multi, single_0, single_1] per route, routes in order: neutron, photon, dual.
    let routes = [
        ("neutron", Some(ParticleType::Neutron)),
        ("photon", Some(ParticleType::Photon)),
        ("dual", None),
    ];
    let mut tallies: Vec<Arc<Tally>> = Vec::new();
    for (label, pt) in routes {
        tallies.push(particle_tally(
            &format!("{label}_multi"),
            &COUPLED_SCORES,
            pt,
        ));
        for s in COUPLED_SCORES {
            tallies.push(particle_tally(&format!("{label}_single_{s}"), &[s], pt));
        }
    }

    let (mut m, settings) = coupled_model(tallies.clone());
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("coupled GPU run");
    let out = means(&tallies);

    let per_route = 1 + COUPLED_SCORES.len();
    for (r, (label, _)) in routes.iter().enumerate() {
        let multi = &out[r * per_route];
        assert_eq!(
            multi.len(),
            COUPLED_SCORES.len(),
            "{label}: multi-score bin count"
        );
        for (k, s) in COUPLED_SCORES.iter().enumerate() {
            let single = &out[r * per_route + 1 + k][0];
            eprintln!(
                "  {label:8} {s:8} multi {:.6e}  single {single:.6e}",
                multi[k]
            );
            assert_close(&format!("{label} score {s}"), multi[k], *single, 1e-9);
        }
        // A route that scored nothing would satisfy the identity trivially.
        assert!(
            multi.iter().all(|v| *v > 0.0),
            "{label}: a score came back zero ({multi:?}), identity is vacuous"
        );
    }

    // The dual route must exceed the neutron-only one for flux: it adds the
    // photon pass. Otherwise the dual summation is not actually happening and
    // the three routes would be indistinguishable.
    let neutron_flux = out[0][0];
    let dual_flux = out[2 * per_route][0];
    eprintln!("  neutron flux {neutron_flux:.6e}  dual flux {dual_flux:.6e}");
    assert!(
        dual_flux > neutron_flux * 1.000_001,
        "dual flux {dual_flux:.6e} did not exceed neutron-only {neutron_flux:.6e}; \
         the photon pass is not being summed in"
    );
}

/// An unfiltered multi-score tally whose scores disagree about whether the
/// photon pass contributes has no single correct routing, so the coupled
/// dispatch must reject it rather than silently drop the photon half of the
/// flux or score a neutron MT on the photon pass.
#[test]
fn gpu_coupled_rejects_all_particle_tally_mixing_photon_and_neutron_only_scores() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    if !std::path::Path::new("tests/Fe.arrow").exists() {
        eprintln!("skipping -- photon fixture tests/Fe.arrow missing");
        return;
    }

    // flux wants the photon pass; (n,gamma) does not.
    let mixed = particle_tally("mixed", &["flux", "(n,gamma)"], None);
    let (mut m, settings) = coupled_model(vec![mixed]);
    let err = yamc::gpu::run_on_gpu(&mut m, &settings)
        .expect_err("mixed-duality all-particle tally must be rejected");
    let msg = err.to_string();
    eprintln!("  {msg}");
    assert!(
        msg.contains("must agree on whether the photon pass contributes"),
        "unexpected rejection reason: {msg}"
    );
    // The message has to say what to do about it.
    assert!(
        msg.contains("particle=") && msg.contains("split"),
        "rejection does not offer the two ways out: {msg}"
    );

    // The same tally with an explicit particle filter is fine -- the filter
    // resolves the ambiguity, which is what the message tells the user.
    let filtered = particle_tally(
        "filtered",
        &["flux", "(n,gamma)"],
        Some(ParticleType::Neutron),
    );
    let (mut m2, settings2) = coupled_model(vec![filtered.clone()]);
    yamc::gpu::run_on_gpu(&mut m2, &settings2).expect("particle-filtered mixed scores must run");
    let got = filtered.get_mean().to_vec();
    assert_eq!(got.len(), 2, "two scores, one cell bin each");
    assert!(
        got.iter().all(|v| *v > 0.0),
        "particle-filtered mixed-score tally scored zero: {got:?}"
    );
}
