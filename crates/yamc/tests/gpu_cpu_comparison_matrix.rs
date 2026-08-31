//! Comprehensive CPU-vs-GPU numeric comparison harness.
//!
//! Runs EVERY supported combination of particle/physics mode, estimator,
//! capture (analog / survival biasing), and score on both the CPU
//! (`Model::simulate_transport`) and the GPU (`yamc::gpu::run_on_gpu`), then
//! prints markdown tables of CPU value | GPU value | GPU/CPU ratio. The modes
//! covered: neutron, primary photon, coupled secondary photons, D1S decay
//! photons, a mixed neutron+photon primary source (#58 / PR #98), the fissile
//! fission-bank path, and a multi-nuclide material. Survival biasing (implicit
//! capture), multi-nuclide materials, the fission chain, and D1S decay photons
//! now all run on the GPU and are compared like analog. Where
//! the GPU dispatch still rejects a combination (mesh geometry, or a score on a
//! path the kernel cannot estimate), the GPU cell records `rejected: <message>`
//! instead of a number -- the dispatch `Err` is caught and its `Display`
//! printed, never a panic. Transient `BufferAsyncError`s (the audit shares the
//! single GPU) are retried, not reported as rejections.
//!
//! Geometry / material: ONE Fe56 sphere (CSG, vacuum boundary) carrying BOTH
//! neutron data (`tests/Fe56.arrow`) and photon data (`tests/Fe.arrow`), built
//! exactly as `gpu_coupled_photon.rs` does. The `Below(sphere)` region keeps it
//! GPU-boundable (the AABB pass rejects an unbounded `Complement`).
//!
//! Sources: a 14 MeV isotropic neutron point source for the neutron and coupled
//! modes; a 1.25 MeV (Co-60-mean) photon point source for the primary-photon
//! mode.
//!
//! Tally normalisation is identical on both backends: each cell reads
//! `get_mean().iter().sum::<f64>()` (the per-source-particle mean), the same
//! convention `gpu_coupled_photon.rs` uses.
//!
//! Run it (single-threaded -- the shared cubecl client serialises launches):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_cpu_comparison_matrix -- --nocapture --test-threads=1
//! It self-skips cleanly if the Arrow data is absent or no f64 GPU adapter is
//! present.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 4242;
const RADIUS: f64 = 5.0;
const N_PER_BATCH: usize = 25_000;
const N_BATCHES: usize = 4; // 100k histories total.
const MAX_STEPS: u32 = 5_000;

/// Locally-cached fendl-3.2d U235 neutron Arrow directory, if present. A
/// fissile nuclide exercises the survival-biasing fission path (the GPU
/// banks fission progeny via `weight *= nu_bar` from the pre-discount
/// weight). The matrix self-skips the fissile table when this is absent so
/// the test stays portable.
const LOCAL_U235_NEUTRON: &str =
    "/home/jon/yamc-org/cross_section_data_fendl_3.2d_arrow/fendl-3.2d-arrow/neutron/U235.arrow";

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Fe56 sphere (r=RADIUS, `Below` -> GPU-boundable) with neutron + photon data.
/// Returns `(geometry, cell_id)`.
fn fe_sphere(cell_id: u32) -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
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
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(cell_id), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_id)
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.25e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

/// One tally: a CellFilter, an optional ParticleType filter, one score, one
/// estimator. `initialize_batches(N_BATCHES)` so the GPU/CPU per-batch Welford
/// fold matches.
fn make_tally(
    cell_id: u32,
    particle: Option<ParticleType>,
    score: Score,
    estimator: Estimator,
) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    if let Some(p) = particle {
        t.filters
            .push(Filter::ParticleType(ParticleTypeFilter::new(p)));
    }
    t.scores = vec![score];
    t.estimator = estimator;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

/// Assemble a fresh model for the given source + tallies, with the standard
/// seed / step budget. `survival` toggles implicit-capture survival biasing;
/// `secondary` toggles coupled secondary-photon transport.
fn build_model(
    geometry: Geometry,
    source: ParticleSource,
    tallies: Vec<Arc<Tally>>,
    survival: bool,
    secondary: bool,
) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    if survival {
        model.variance_reduction = vec![VarianceReduction::SurvivalBiasing(
            SurvivalBiasing::default(),
        )];
    }
    if secondary {
        model.transport_secondary_photons = true;
        model.photon_cutoff_energy = 1000.0;
    }
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

/// Sum of a tally's per-bin mean (the per-source-particle quantity both
/// backends report).
fn tally_sum(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum::<f64>()
}

/// Dispatch the model to the GPU, retrying a few times on a transient
/// `BufferAsyncError` (the verification audit shares the single adapter, so a
/// launch can hiccup under contention). A genuine dispatch rejection (mesh,
/// unsupported score, ...) is returned as-is so the caller can record it; only
/// the async/contention errors are retried. The result is `Ok(())` on a
/// successful run (tallies are written in place) or `Err(message)` on a real
/// rejection.
fn run_on_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<(), String> {
    let mut last = String::new();
    for _ in 0..6 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(_) => return Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("BufferAsyncError") || msg.contains("Async") {
                    last = msg;
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    continue;
                }
                return Err(msg);
            }
        }
    }
    panic!("GPU run kept hitting a transient error after retries: {last}");
}

/// A cell of the table: a CPU number plus either a GPU number or a rejection
/// message.
enum Gpu {
    Value(f64),
    Rejected(String),
}

struct Row {
    score: String,
    estimator: Estimator,
    capture: &'static str,
    cpu: f64,
    gpu: Gpu,
}

fn est_name(e: Estimator) -> &'static str {
    match e {
        Estimator::TrackLength => "track-length",
        Estimator::Collision => "collision",
    }
}

/// Render rows as a markdown table and report the worst-case |ratio-1| over the
/// GPU-supported (non-rejected, non-negligible) cells.
fn render(title: &str, rows: &[Row]) {
    println!("\n## {title}\n");
    println!("| score | estimator | capture | CPU | GPU | GPU/CPU |");
    println!("|---|---|---|---|---|---|");
    let mut worst: Option<(f64, String)> = None;
    for r in rows {
        let (gpu_str, ratio_str) = match &r.gpu {
            Gpu::Value(g) => {
                let gpu_str = format!("{g:.4e}");
                if r.cpu.abs() > 0.0 {
                    let ratio = g / r.cpu;
                    // Track worst |ratio-1| only where CPU is a meaningful
                    // (non-trace) signal, so a noisy near-zero channel does not
                    // dominate the summary.
                    if r.cpu.abs() > 1e-6 {
                        let dev = (ratio - 1.0).abs();
                        if worst.as_ref().map(|(w, _)| dev > *w).unwrap_or(true) {
                            worst = Some((
                                dev,
                                format!("{} / {} / {}", r.score, est_name(r.estimator), r.capture),
                            ));
                        }
                    }
                    (gpu_str, format!("{ratio:.3}"))
                } else {
                    (gpu_str, "n/a (CPU 0)".to_string())
                }
            }
            Gpu::Rejected(msg) => ("rejected".to_string(), format!("rejected: {msg}")),
        };
        println!(
            "| {} | {} | {} | {:.4e} | {} | {} |",
            r.score,
            est_name(r.estimator),
            r.capture,
            r.cpu,
            gpu_str,
            ratio_str,
        );
    }
    match worst {
        Some((dev, cell)) => println!(
            "\n_Worst-case |ratio-1| among GPU-supported cells: {:.1}% ({cell})._",
            dev * 100.0
        ),
        None => println!("\n_No GPU-supported numeric cells in this table._"),
    }
}

// --------------------------------------------------------------------------
// Table 1 + 2: single-particle modes (neutron, primary photon)
// --------------------------------------------------------------------------

/// One CPU run scoring the whole set, returning per-score sums in `scores`
/// order. The model is rebuilt fresh per call so each (estimator, capture) cell
/// is independent.
fn run_cpu(
    source_fn: fn() -> ParticleSource,
    particle: Option<ParticleType>,
    scores: &[(&str, Score)],
    estimator: Estimator,
    survival: bool,
    secondary: bool,
) -> Vec<f64> {
    let (geometry, cell_id) = fe_sphere(1);
    let tallies: Vec<Arc<Tally>> = scores
        .iter()
        .map(|(_, s)| make_tally(cell_id, particle, s.clone(), estimator))
        .collect();
    let (mut model, settings) =
        build_model(geometry, source_fn(), tallies.clone(), survival, secondary);
    model
        .simulate_transport(&settings)
        .unwrap_or_else(|e| panic!("CPU run failed ({estimator:?}, survival={survival}): {e}"));
    tallies.iter().map(tally_sum).collect()
}

/// Per-score GPU run so a single rejected score (e.g. PhotonXS on the neutron
/// path) is reported on its own row rather than poisoning the whole table.
fn run_gpu_per_score(
    source_fn: fn() -> ParticleSource,
    particle: Option<ParticleType>,
    score: &Score,
    estimator: Estimator,
    survival: bool,
    secondary: bool,
) -> Gpu {
    let (geometry, cell_id) = fe_sphere(1);
    let t = make_tally(cell_id, particle, score.clone(), estimator);
    let (mut model, settings) = build_model(
        geometry,
        source_fn(),
        vec![Arc::clone(&t)],
        survival,
        secondary,
    );
    match run_on_gpu_retry(&mut model, &settings) {
        Ok(()) => Gpu::Value(tally_sum(&t)),
        Err(e) => Gpu::Rejected(e),
    }
}

/// Build a single-particle mode table (neutron or primary photon): every score
/// x {track-length, collision} x {analog, survival}. Both captures now RUN on
/// the GPU (survival biasing = implicit capture, which is unbiased, so the
/// GPU-survival mean tracks the CPU within MC error). The GPU runs per-score so
/// a per-path-rejected score (e.g. a photon-XS score on the neutron path) shows
/// its own message rather than poisoning the whole table.
fn single_particle_table(
    title: &str,
    source_fn: fn() -> ParticleSource,
    particle: Option<ParticleType>,
    scores: &[(&str, Score)],
    secondary: bool,
) {
    let mut rows: Vec<Row> = Vec::new();
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        for survival in [false, true] {
            let cpu = run_cpu(source_fn, particle, scores, estimator, survival, secondary);
            for (i, (name, score)) in scores.iter().enumerate() {
                // The GPU now supports survival biasing (implicit capture); run
                // it the same as analog. Implicit capture is unbiased, so the
                // GPU-survival flux must match the CPU within MC error.
                let gpu =
                    run_gpu_per_score(source_fn, particle, score, estimator, survival, secondary);
                rows.push(Row {
                    score: (*name).to_string(),
                    estimator,
                    capture: if survival { "survival" } else { "analog" },
                    cpu: cpu[i],
                    gpu,
                });
            }
        }
    }
    render(title, &rows);
}

// --------------------------------------------------------------------------
// Table 5: fissile U235 sphere (survival-biasing fission-bank path)
// --------------------------------------------------------------------------

/// Neutron-only U235 sphere (`Below` -> GPU-boundable), if the local
/// fendl-3.2d U235 neutron data resolves. `None` -> the fissile table skips.
fn u235_sphere(cell_id: u32) -> Option<(Geometry, u32)> {
    if !std::path::Path::new(LOCAL_U235_NEUTRON).exists() {
        return None;
    }
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("U235".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(18.95),
    )
    .ok()?;
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("U235".to_string(), LOCAL_U235_NEUTRON.to_string())]);
    material.read_nuclear_data(&nm, None).ok()?;

    let cell = Cell::new(Some(cell_id), region, Some("u235".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).ok()?;
    Some((geometry, cell_id))
}

/// Fissile U235 sphere: CPU vs GPU over flux / total / absorption / fission,
/// each estimator, analog and survival. Implicit capture is unbiased, so the
/// GPU-survival means must track CPU within MC error -- this exercises the
/// fission-bank-from-pre-discount-weight path (the GPU's `weight *= nu_bar`
/// fission branch under survival biasing).
fn fissile_table() {
    let scores: Vec<(&str, Score)> = vec![
        ("flux", "flux".parse().unwrap()),
        ("total", "total".parse().unwrap()),
        ("absorption", "absorption".parse().unwrap()),
        ("fission", "fission".parse().unwrap()),
    ];

    let build = |survival: bool, estimator: Estimator| -> Option<(Vec<f64>, Vec<Gpu>)> {
        let (geometry, cell_id) = u235_sphere(1)?;
        // CPU: all scores in one run.
        let cpu_tallies: Vec<Arc<Tally>> = scores
            .iter()
            .map(|(_, s)| make_tally(cell_id, Some(ParticleType::Neutron), s.clone(), estimator))
            .collect();
        let (mut cpu_model, settings) = build_model(
            geometry,
            neutron_source(),
            cpu_tallies.clone(),
            survival,
            false,
        );
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU U235 run");
        let cpu: Vec<f64> = cpu_tallies.iter().map(tally_sum).collect();

        // GPU: per-score so a rejected score reports on its own row.
        let gpu: Vec<Gpu> = scores
            .iter()
            .map(|(_, score)| {
                let (g, c2) = u235_sphere(1).expect("U235 data was present above");
                let t = make_tally(c2, Some(ParticleType::Neutron), score.clone(), estimator);
                let (mut m, settings) =
                    build_model(g, neutron_source(), vec![Arc::clone(&t)], survival, false);
                match run_on_gpu_retry(&mut m, &settings) {
                    Ok(()) => Gpu::Value(tally_sum(&t)),
                    Err(e) => Gpu::Rejected(e),
                }
            })
            .collect();
        Some((cpu, gpu))
    };

    let mut rows: Vec<Row> = Vec::new();
    let mut any = false;
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        for survival in [false, true] {
            if let Some((cpu, gpu)) = build(survival, estimator) {
                any = true;
                for (i, (name, _)) in scores.iter().enumerate() {
                    rows.push(Row {
                        score: (*name).to_string(),
                        estimator,
                        capture: if survival { "survival" } else { "analog" },
                        cpu: cpu[i],
                        gpu: match &gpu[i] {
                            Gpu::Value(v) => Gpu::Value(*v),
                            Gpu::Rejected(m) => Gpu::Rejected(m.clone()),
                        },
                    });
                }
            }
        }
    }
    if any {
        render(
            "5. Fissile U235 sphere (14 MeV neutron source, fendl-3.2d U235)",
            &rows,
        );
    } else {
        println!("\n## 5. Fissile U235 sphere\n\n_skipped -- local fendl-3.2d U235 data absent._");
    }
}

#[test]
fn gpu_cpu_comparison_matrix() {
    if !data_present() {
        eprintln!(
            "skipping gpu_cpu_comparison_matrix -- tests/Fe56.arrow or tests/Fe.arrow not found"
        );
        return;
    }
    if !gpu_available() {
        eprintln!("skipping gpu_cpu_comparison_matrix -- no GPU with f64 compute available");
        return;
    }

    println!("\n# CPU vs GPU numeric comparison matrix");
    println!(
        "\nFe56 sphere r={RADIUS} cm (CSG, vacuum), neutron+photon data; \
         seed {SEED}; {} histories ({N_PER_BATCH} x {N_BATCHES} batches).",
        N_PER_BATCH * N_BATCHES
    );

    // ---- Table 1: neutron mode ----
    let neutron_scores: Vec<(&str, Score)> = vec![
        ("flux", "flux".parse().unwrap()),
        ("total", "total".parse().unwrap()),
        ("absorption", "absorption".parse().unwrap()),
        ("elastic", "elastic".parse().unwrap()),
        ("(n,gamma)/MT102", "(n,gamma)".parse().unwrap()),
        ("heating", "heating".parse().unwrap()),
        ("damage-energy", "damage-energy".parse().unwrap()),
        ("He4-production", "He4-production".parse().unwrap()),
    ];
    single_particle_table(
        "1. Neutron mode (14 MeV source)",
        neutron_source,
        Some(ParticleType::Neutron),
        &neutron_scores,
        false,
    );

    // ---- Table 2: primary photon mode ----
    let photon_scores: Vec<(&str, Score)> = vec![
        ("flux", "flux".parse().unwrap()),
        ("total", "total".parse().unwrap()),
        ("heating", "heating".parse().unwrap()),
        ("coherent(502)", "coherent-scatter".parse().unwrap()),
        ("incoherent(504)", "incoherent-scatter".parse().unwrap()),
        ("photoelectric(522)", "photoelectric".parse().unwrap()),
        ("pair(516)", "pair-production".parse().unwrap()),
    ];
    single_particle_table(
        "2. Primary photon mode (1.25 MeV source)",
        photon_source,
        Some(ParticleType::Photon),
        &photon_scores,
        false,
    );

    // ---- Table 3: coupled secondary photons ----
    coupled_table();

    // ---- Table 4: CPU-only / GPU-rejected options ----
    cpu_only_table();

    // ---- Table 5: fissile U235 sphere (survival-biasing fission-bank path) ----
    fissile_table();

    // ---- Table 6: multi-nuclide material (per-collision nuclide selection) ----
    multinuclide_table();

    // ---- Table 7: mixed neutron+photon primary source (#58 / PR #98) ----
    mixed_source_table();
}

// --------------------------------------------------------------------------
// Table 6: multi-nuclide material (H2 + C12 moderator)
// --------------------------------------------------------------------------

/// A 2:1 H2:C12 (CH2-like) moderator sphere exercises per-collision nuclide
/// selection on the GPU (#74): each collision picks WHICH nuclide is struck
/// (proportional to its macroscopic total xs) and uses that nuclide's own AWR
/// and elastic angular table. A material-averaged kernel under-moderates badly
/// here (the light H1 is averaged against the heavy C12), so this is the
/// sharpest single-material multi-nuclide check. CPU vs GPU over flux / total /
/// elastic, each estimator, analog. Skips if the H2/C12 fixtures are absent.
fn multinuclide_table() {
    if !std::path::Path::new("tests/H2.arrow").exists()
        || !std::path::Path::new("tests/C12.arrow").exists()
    {
        println!("\n## 6. Multi-nuclide material (H2:C12)\n\n_skipped -- tests/H2.arrow or tests/C12.arrow absent._");
        return;
    }

    let build = |cell_id: u32| -> (Geometry, u32) {
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: RADIUS,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        let mut material = Material::new(
            HashMap::from([("H2".into(), 2.0), ("C12".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.94),
        )
        .unwrap();
        material.set_material_id(1);
        material.set_temperature("294");
        let nm = HashMap::from([
            ("H2".to_string(), "tests/H2.arrow".to_string()),
            ("C12".to_string(), "tests/C12.arrow".to_string()),
        ]);
        material.read_nuclear_data(&nm, None).unwrap();
        let cell = Cell::new(Some(cell_id), region, Some("ch2".into()), Some(0));
        let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
        (geometry, cell_id)
    };

    let scores: Vec<(&str, Score)> = vec![
        ("flux", "flux".parse().unwrap()),
        ("total", "total".parse().unwrap()),
        ("elastic", "elastic".parse().unwrap()),
    ];

    let mut rows: Vec<Row> = Vec::new();
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        let (geometry, cell_id) = build(1);
        let cpu_tallies: Vec<Arc<Tally>> = scores
            .iter()
            .map(|(_, s)| make_tally(cell_id, Some(ParticleType::Neutron), s.clone(), estimator))
            .collect();
        let (mut cpu_model, settings) = build_model(
            geometry,
            neutron_source(),
            cpu_tallies.clone(),
            false,
            false,
        );
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU H2:C12 run");
        let cpu: Vec<f64> = cpu_tallies.iter().map(tally_sum).collect();

        for (i, (name, score)) in scores.iter().enumerate() {
            let (g, c2) = build(1);
            let t = make_tally(c2, Some(ParticleType::Neutron), score.clone(), estimator);
            let (mut m, settings) =
                build_model(g, neutron_source(), vec![Arc::clone(&t)], false, false);
            let gpu = match run_on_gpu_retry(&mut m, &settings) {
                Ok(()) => Gpu::Value(tally_sum(&t)),
                Err(e) => Gpu::Rejected(e),
            };
            rows.push(Row {
                score: (*name).to_string(),
                estimator,
                capture: "analog",
                cpu: cpu[i],
                gpu,
            });
        }
    }
    render(
        "6. Multi-nuclide material: 2:1 H2:C12 moderator (per-collision nuclide selection)",
        &rows,
    );
}

// --------------------------------------------------------------------------
// Table 3: coupled (neutron source + transport_secondary_photons)
// --------------------------------------------------------------------------

/// Coupled mode: a 14 MeV neutron source with `transport_secondary_photons`.
/// For flux and heating, and each estimator, report the neutron-filtered,
/// photon-filtered, and unfiltered (all-particle) split on both backends. Both
/// the unfiltered flux and the unfiltered heating are the neutron+photon sum
/// (dual-pass on GPU): the neutron pass scores neutron heating, the photon
/// sub-pass scores photon heating, and `classify_coupled_tallies` sums them.
fn coupled_table() {
    // (label, particle filter, score)
    struct Spec {
        label: &'static str,
        particle: Option<ParticleType>,
        score: Score,
    }
    let specs = || -> Vec<Spec> {
        vec![
            Spec {
                label: "flux (neutron)",
                particle: Some(ParticleType::Neutron),
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "flux (photon)",
                particle: Some(ParticleType::Photon),
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "flux (unfiltered/all)",
                particle: None,
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "heating (neutron)",
                particle: Some(ParticleType::Neutron),
                score: "heating".parse().unwrap(),
            },
            Spec {
                label: "heating (photon)",
                particle: Some(ParticleType::Photon),
                score: "heating".parse().unwrap(),
            },
            Spec {
                label: "heating (unfiltered/all)",
                particle: None,
                score: "heating".parse().unwrap(),
            },
        ]
    };

    let mut rows: Vec<Row> = Vec::new();
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        let s = specs();
        // CPU: all six tallies in one run so the split is consistent.
        let (geometry, cell_id) = fe_sphere(1);
        let cpu_tallies: Vec<Arc<Tally>> = s
            .iter()
            .map(|sp| make_tally(cell_id, sp.particle, sp.score.clone(), estimator))
            .collect();
        let (mut cpu_model, settings) =
            build_model(geometry, neutron_source(), cpu_tallies.clone(), false, true);
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU coupled run");
        let cpu: Vec<f64> = cpu_tallies.iter().map(tally_sum).collect();

        // GPU: run each spec individually so a rejected one reports its own
        // message and the others still produce numbers. A coupled GPU run
        // needs the secondary-photon flag set.
        for (i, sp) in s.iter().enumerate() {
            let (g, c2) = fe_sphere(1);
            let t = make_tally(c2, sp.particle, sp.score.clone(), estimator);
            let (mut m, settings) =
                build_model(g, neutron_source(), vec![Arc::clone(&t)], false, true);
            let gpu = match run_on_gpu_retry(&mut m, &settings) {
                Ok(()) => Gpu::Value(tally_sum(&t)),
                Err(e) => Gpu::Rejected(e),
            };
            rows.push(Row {
                score: sp.label.to_string(),
                estimator,
                capture: "analog",
                cpu: cpu[i],
                gpu,
            });
        }
    }
    render(
        "3. Coupled secondary photons (14 MeV neutron source, transport_secondary_photons=true)",
        &rows,
    );
}

// --------------------------------------------------------------------------
// Table 7: mixed neutron+photon primary source (#58 / PR #98)
// --------------------------------------------------------------------------

/// Build a mixed-source model on the shared Fe sphere: an equal-strength 14 MeV
/// neutron source AND a 1.25 MeV photon source, with the given tallies. Each
/// history is one source particle drawn by strength, so roughly half are
/// neutrons and half photons. The neutron share runs the coupled kernel (so it
/// emits secondary photons), the photon share runs the photon kernel; both fold
/// per source particle. Mirrors the validated `gpu_mixed_source.rs` setup.
fn build_mixed_model(tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let (geometry, _) = fe_sphere(1);
    let mut model = Model::new(geometry, vec![neutron_source(), photon_source()], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

/// Mixed primary source mode: a 14 MeV neutron source and a 1.25 MeV photon
/// source together (#58, added on the GPU in PR #98). For flux and heating, and
/// each estimator, report the neutron-filtered (neutron pass only),
/// photon-filtered (primary photons + neutron-induced secondaries), and
/// unfiltered (all-particle SUM = dual path) split on both backends. The GPU
/// splits the run by source strength: the neutron share runs the coupled kernel,
/// the photon share the photon kernel, folded per source particle. (D1S on top
/// of a mixed source is the one rejected combination -- see the kernel's
/// `MixedSourceWithSecondariesUnsupported`; it is not exercised here because the
/// matrix runs analog/secondary mixed sources only.)
fn mixed_source_table() {
    struct Spec {
        label: &'static str,
        particle: Option<ParticleType>,
        score: Score,
    }
    let specs = || -> Vec<Spec> {
        vec![
            Spec {
                label: "flux (neutron)",
                particle: Some(ParticleType::Neutron),
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "flux (photon)",
                particle: Some(ParticleType::Photon),
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "flux (unfiltered/all)",
                particle: None,
                score: "flux".parse().unwrap(),
            },
            Spec {
                label: "heating (neutron)",
                particle: Some(ParticleType::Neutron),
                score: "heating".parse().unwrap(),
            },
            Spec {
                label: "heating (photon)",
                particle: Some(ParticleType::Photon),
                score: "heating".parse().unwrap(),
            },
            Spec {
                label: "heating (unfiltered/all)",
                particle: None,
                score: "heating".parse().unwrap(),
            },
        ]
    };

    let mut rows: Vec<Row> = Vec::new();
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        let s = specs();
        // CPU: all six tallies in one run so the neutron/photon/total split is
        // self-consistent (the mixed source "just works" on the CPU --
        // `sample_source` picks a source by strength, then transports that
        // particle with its own physics).
        let cell_id = 1;
        let cpu_tallies: Vec<Arc<Tally>> = s
            .iter()
            .map(|sp| make_tally(cell_id, sp.particle, sp.score.clone(), estimator))
            .collect();
        let (mut cpu_model, settings) = build_mixed_model(cpu_tallies.clone());
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU mixed-source run");
        let cpu: Vec<f64> = cpu_tallies.iter().map(tally_sum).collect();

        // GPU: run each spec individually so a rejected one reports its own
        // message and the others still produce numbers. The mixed dispatch
        // (`run_on_gpu_mixed`) splits by source strength.
        for (i, sp) in s.iter().enumerate() {
            let t = make_tally(cell_id, sp.particle, sp.score.clone(), estimator);
            let (mut m, settings) = build_mixed_model(vec![Arc::clone(&t)]);
            let gpu = match run_on_gpu_retry(&mut m, &settings) {
                Ok(()) => Gpu::Value(tally_sum(&t)),
                Err(e) => Gpu::Rejected(e),
            };
            rows.push(Row {
                score: sp.label.to_string(),
                estimator,
                capture: "analog",
                cpu: cpu[i],
                gpu,
            });
        }
    }
    render(
        "7. Mixed neutron+photon primary source (14 MeV neutron + 1.25 MeV photon, equal strength)",
        &rows,
    );
}

// --------------------------------------------------------------------------
// Table 4: CPU-only / GPU-rejected options
// --------------------------------------------------------------------------

/// Options that exercise the GPU's tracking-mode handling and the few remaining
/// rejections. Most rows now RUN on the GPU: survival biasing (implicit
/// capture), D1S decay photons (coupled decay-photon bank), and Woodcock /
/// Hybrid tracking (the kernel ignores `tracking_mode`, warns, and surface-
/// tracks). Only mesh geometry is still rejected at dispatch. Each row runs the
/// CPU to get a real number, then runs (or short-circuits) the GPU and prints
/// either the GPU number or the exact rejection message.
fn cpu_only_table() {
    println!("\n## 4. Tracking-mode handling + remaining GPU-rejected options\n");
    println!("| option | CPU flux | GPU result |");
    println!("|---|---|---|");

    let neutron_flux = || -> (Geometry, Arc<Tally>) {
        let (g, cid) = fe_sphere(1);
        let t = make_tally(
            cid,
            Some(ParticleType::Neutron),
            "flux".parse().unwrap(),
            Estimator::TrackLength,
        );
        (g, t)
    };

    // -- Survival biasing (variance reduction) -- now SUPPORTED on the GPU
    // (implicit capture). It is unbiased, so the GPU survival flux must track
    // CPU survival within MC error; report the ratio rather than a rejection.
    {
        let (g, t) = neutron_flux();
        let (mut cpu, settings) =
            build_model(g, neutron_source(), vec![Arc::clone(&t)], true, false);
        cpu.simulate_transport(&settings).expect("CPU survival run");
        let cpu_flux = tally_sum(&t);

        let (g2, t2) = neutron_flux();
        let (mut gpu, settings) =
            build_model(g2, neutron_source(), vec![Arc::clone(&t2)], true, false);
        let msg = match run_on_gpu_retry(&mut gpu, &settings) {
            Ok(()) => {
                let g = tally_sum(&t2);
                let ratio = if cpu_flux != 0.0 {
                    g / cpu_flux
                } else {
                    f64::NAN
                };
                format!("{g:.4e} (GPU/CPU {ratio:.3})")
            }
            Err(e) => format!("UNEXPECTED rejection: {e}"),
        };
        println!("| survival biasing (implicit capture) | {cpu_flux:.4e} | {msg} |");
    }

    // -- Energy function / dose coefficients -- now SUPPORTED on the GPU
    // (issue #271). The kernel evaluates the cubic spline the CPU already
    // solved, so this is a weighted flux, not a rejection. Uses the real
    // ICRP-116 AP dose curve, i.e. what `dose_coefficients=('neutron','AP')`
    // lowers to.
    {
        let (energy, coeffs) = yamc_nuclide::data::effective_dose::dose_coefficients(
            yamc_nuclide::data::effective_dose::DoseParticle::Neutron,
            yamc_nuclide::data::effective_dose::DoseGeometry::AP,
            yamc_nuclide::data::effective_dose::DoseDataSource::ICRP116,
        );
        let dose_filter = || {
            Filter::EnergyFunction(yamc_tallies::EnergyFunctionFilter::new(
                energy.clone(),
                coeffs.clone(),
            ))
        };

        // `Tally` is not Clone, so build a fresh dose-weighted tally per run.
        let dose_tally = |cell_id: u32| -> Arc<Tally> {
            let mut t = Tally::new();
            t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
            t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
                ParticleType::Neutron,
            )));
            t.filters.push(dose_filter());
            t.scores = vec!["flux".parse().unwrap()];
            t.estimator = Estimator::TrackLength;
            t.initialize_batches(N_BATCHES);
            Arc::new(t)
        };

        let (g, cid) = fe_sphere(1);
        let t_cpu = dose_tally(cid);
        let (mut cpu, settings) =
            build_model(g, neutron_source(), vec![Arc::clone(&t_cpu)], false, false);
        cpu.simulate_transport(&settings).expect("CPU dose run");
        let cpu_dose = tally_sum(&t_cpu);

        let (g2, cid2) = fe_sphere(1);
        let t_gpu = dose_tally(cid2);
        let (mut gpu, settings) =
            build_model(g2, neutron_source(), vec![Arc::clone(&t_gpu)], false, false);
        let msg = match run_on_gpu_retry(&mut gpu, &settings) {
            Ok(()) => {
                let g = tally_sum(&t_gpu);
                let ratio = if cpu_dose != 0.0 {
                    g / cpu_dose
                } else {
                    f64::NAN
                };
                format!("{g:.4e} (GPU/CPU {ratio:.3})")
            }
            Err(e) => format!("UNEXPECTED rejection: {e}"),
        };
        println!("| energy_function / dose_coefficients (ICRP-116 AP) | {cpu_dose:.4e} | {msg} |");
    }

    // -- Mesh geometry (feature-gated) --
    mesh_row();

    // -- Woodcock tracking --
    tracking_row("Woodcock tracking", TrackingMode::Woodcock);

    // -- Hybrid tracking --
    tracking_row("Hybrid tracking", TrackingMode::Hybrid);

    // -- D1S decay photons -- now SUPPORTED on the GPU (coupled decay-photon
    // bank). D1S needs the transmutation chain; it is set process-globally and
    // only read when `use_decay_photons=true`, so it does not perturb the other
    // (non-D1S) rows. The photon flux is binned by `parent_nuclides=["Mn56"]`
    // (the Fe56(n,p) activation product) on both backends.
    d1s_row();

    println!(
        "\n_Note: the GPU neutron/photon kernels always surface-track and ignore \
         `tracking_mode`. A non-Surface request is not rejected: the dispatch emits a \
         WARNING and surface-tracks, so Woodcock and Hybrid reproduce the surface-tracking \
         flux (unbiased) rather than the CPU Woodcock/Hybrid number._"
    );
}

/// Path to the shipped transmutation chain used for the D1S decay-photon row.
fn chain_path() -> String {
    format!(
        "{}/tests/transmutation-endf-b8.1-sfr.arrow",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// D1S decay-photon row: a 14 MeV neutron source with `use_decay_photons` on a
/// Fe56 sphere, photon flux binned by `parent_nuclides=["Mn56"]`. Both backends
/// run the coupled decay-photon path (GPU: the neutron kernel banks decay
/// photons tagged by parent radionuclide, the photon sub-pass transports them).
/// Skips if the transmutation chain fixture is absent.
fn d1s_row() {
    use yamc_tallies::filter::parent_nuclide::ParentNuclideFilter;

    if !std::path::Path::new(&chain_path()).exists() {
        println!(
            "| D1S decay photons (use_decay_photons) | (transmutation chain absent -- skipped) | n/a |"
        );
        return;
    }
    // D1S requires the transmutation chain via the global config. Set it once;
    // only D1S rows read it, so non-D1S rows are unaffected.
    {
        // Per-subsection transmutation config (the single-file setter was
        // removed in the transmutation split); point all three parts at the
        // v2 fixture directory, which resolve_subsection splits into
        // decay/reactions/fission_yields subdirs.
        let mut cfg = yamc_nuclide::config::Config::global();
        cfg.transmutation_decay_data = Some(chain_path());
        cfg.transmutation_reactions = yamc_nuclide::config::SubsectionSource::Set(chain_path());
        cfg.transmutation_fission_yields =
            yamc_nuclide::config::SubsectionSource::Set(chain_path());
    }

    // A photon flux tally binned by the Mn56 parent (the Fe56(n,p) product).
    let d1s_photon_tally = |cell_id: u32| -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
        t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Photon,
        )));
        t.filters
            .push(Filter::ParentNuclide(ParentNuclideFilter::new(vec![
                "Mn56".to_string(),
            ])));
        t.scores = vec!["flux".parse().unwrap()];
        t.estimator = Estimator::TrackLength;
        t.initialize_batches(N_BATCHES);
        Arc::new(t)
    };

    let (g, cid) = fe_sphere(1);
    let t = d1s_photon_tally(cid);
    let (mut cpu, settings) = build_model(g, neutron_source(), vec![Arc::clone(&t)], false, true);
    cpu.use_decay_photons = true;
    let cpu_flux = match cpu.simulate_transport(&settings) {
        Ok(_) => format!("{:.4e}", tally_sum(&t)),
        Err(e) => format!("CPU error: {e}"),
    };

    let (g2, cid2) = fe_sphere(1);
    let t2 = d1s_photon_tally(cid2);
    let (mut gpu, settings) = build_model(g2, neutron_source(), vec![Arc::clone(&t2)], false, true);
    gpu.use_decay_photons = true;
    let msg = match run_on_gpu_retry(&mut gpu, &settings) {
        Ok(()) => format!("{:.4e} (Mn56 decay-photon flux)", tally_sum(&t2)),
        Err(e) => format!("UNEXPECTED rejection: {e}"),
    };
    println!("| D1S decay photons (use_decay_photons, parent=Mn56) | {cpu_flux} | {msg} |");
}

/// CPU Woodcock/Hybrid run vs the GPU (which ignores tracking_mode and runs
/// surface). Prints the CPU flux for that tracking mode and the GPU outcome.
fn tracking_row(label: &str, mode: TrackingMode) {
    let (g, cid) = fe_sphere(1);
    let t = make_tally(
        cid,
        Some(ParticleType::Neutron),
        "flux".parse().unwrap(),
        Estimator::TrackLength,
    );
    let (mut cpu, settings) = build_model(g, neutron_source(), vec![Arc::clone(&t)], false, false);
    cpu.tracking_mode = mode;
    let cpu_flux = match cpu.simulate_transport(&settings) {
        Ok(_) => format!("{:.4e}", tally_sum(&t)),
        Err(e) => format!("CPU error: {e}"),
    };

    let (g2, cid2) = fe_sphere(1);
    let t2 = make_tally(
        cid2,
        Some(ParticleType::Neutron),
        "flux".parse().unwrap(),
        Estimator::TrackLength,
    );
    let (mut gpu, settings) =
        build_model(g2, neutron_source(), vec![Arc::clone(&t2)], false, false);
    gpu.tracking_mode = mode;
    let msg = match run_on_gpu_retry(&mut gpu, &settings) {
        Ok(()) => format!(
            "warns + surface-tracks (ignores tracking_mode): {:.4e}",
            tally_sum(&t2)
        ),
        Err(e) => format!("rejected: {e}"),
    };
    println!("| {label} | {cpu_flux} | {msg} |");
}

#[cfg(feature = "mesh")]
fn mesh_row() {
    use yamc::geometry::mesh::MeshGeometry;
    let path = std::path::Path::new("../yamt/tests/data/two_region.arrow");
    if !path.exists() {
        println!("| mesh geometry | (mesh data file absent -- skipped) | n/a |");
        return;
    }
    // Build a simple Fe-only material map for the two named mesh regions.
    let mut fe = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    fe.set_material_id(1);
    fe.set_temperature("294");
    fe.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]),
        None,
    )
    .unwrap();
    let fe = Arc::new(fe);
    let materials: HashMap<String, Arc<Material>> = HashMap::from([
        ("fuel".to_string(), Arc::clone(&fe)),
        ("moderator".to_string(), fe),
    ]);

    let cpu_flux = {
        let mesh = MeshGeometry::from_arrow(path, &materials).expect("load two_region.arrow");
        let mut t = Tally::new();
        t.scores = vec!["flux".parse::<Score>().unwrap()];
        t.initialize_batches(N_BATCHES);
        let t = Arc::new(t);
        let mut m = Model::new_with_mesh(mesh, vec![neutron_source()], vec![Arc::clone(&t)]);
        m.verbose = Verbose::silent();
        m.max_steps_per_particle = MAX_STEPS;
        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: SEED,
            threads: Some(1),
            ..Default::default()
        };
        match m.simulate_transport(&settings) {
            Ok(_) => format!("{:.4e}", tally_sum(&t)),
            Err(e) => format!("CPU error: {e}"),
        }
    };

    let gpu_msg = {
        let mesh = MeshGeometry::from_arrow(path, &materials).expect("load two_region.arrow");
        let mut t = Tally::new();
        t.scores = vec!["flux".parse::<Score>().unwrap()];
        t.initialize_batches(N_BATCHES);
        let mut m = Model::new_with_mesh(mesh, vec![neutron_source()], vec![Arc::new(t)]);
        m.verbose = Verbose::silent();
        m.max_steps_per_particle = MAX_STEPS;
        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: SEED,
            ..Default::default()
        };
        match yamc::gpu::run_on_gpu(&mut m, &settings) {
            Ok(_) => "UNEXPECTED: GPU ran".to_string(),
            Err(e) => format!("rejected: {e}"),
        }
    };
    println!("| mesh geometry (two_region.arrow) | {cpu_flux} | {gpu_msg} |");
}

#[cfg(not(feature = "mesh"))]
fn mesh_row() {
    println!("| mesh geometry | (mesh feature off -- not built) | n/a |");
}
