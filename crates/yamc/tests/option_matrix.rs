//! Transport-option compatibility matrix.
//!
//! Answers "does every option work in every combination?" by sweeping the
//! cross-product of the user-facing transport knobs and asserting each
//! combination either produces sane results or is rejected with a clear,
//! specific error. This is the breadth guard that complements the existing
//! precision tests (e.g. `test_all_scores_collision_parity.py` pins
//! track-length vs collision agreement to 0.1%; here we only check that
//! every combo *runs and is physically sane*).
//!
//! Axes:
//! - geometry: CSG · mesh
//! - tracking: surface · woodcock (delta) · hybrid
//! - estimator: track-length · collision
//! - capture: analog · survival-biasing (implicit)
//! - score: full neutron score set (flux, heating, reaction rates, production, damage) - every score in every cell
//! - device: CPU (here) · GPU (both halves of the matrix, in `mod gpu_guards`):
//!   the GPU "✗" cells are the rejection guards (`gpu_rejects_*`) - they
//!   short-circuit before `GpuContext::new()` so they run on any host, CI
//!   included; the GPU "✓" cells are swept GPU-vs-CPU by
//!   `gpu_neutron_full_score_cross` and `gpu_photon_full_score_cross`, which
//!   need a real f64 adapter and so self-skip on CI (run them single-threaded
//!   via `cargo test-gpu`). The GPU "✓" cells score the full neutron / photon
//!   sets; #388 (GPU returned all-zero tallies) is fixed and closed (#412).
//!   Two reaction-removal channels still diverge from CPU and are flagged
//!   against #415: neutron `absorption`, photon `photoelectric`.
//!
//! Particle/physics modes (neutron · primary photon · secondary/coupled
//! photons · D1S decay photons): the CPU behaviour of the photon, coupled,
//! and hybrid-photon modes across tracking modes is already covered by
//! `test_woodcock_validation.rs` (`woodcock_photon_source_*`,
//! `woodcock_coupled_neutron_photon_produces_photons`,
//! `hybrid_photon_low_density_matches_surface`) and the decay-photon mode by
//! `pytests/.../test_woodcock.py::test_woodcock_decay_photons_match_surface`.
//! Here we add the CPU neutron full-score cross, the GPU neutron and primary-
//! photon full-score crosses (GPU-vs-CPU), and the GPU per-mode rejections.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

/// Full neutron score set - every score gets a tally in every structural cell.
const NEUTRON_SCORES: &[&str] = &[
    "flux",
    "heating",
    "heating-local",
    "total",
    "absorption",
    "elastic",
    "(n,gamma)",
    "H1-production",
    "H2-production",
    "H3-production",
    "He3-production",
    "He4-production",
    "damage-energy",
];

const N_PER_BATCH: usize = 2_500;
const N_BATCHES: usize = 8;

fn li6_fe56_material() -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Li6".into(), 0.3), ("Fe56".into(), 0.7)]),
        "atom",
        "g/cm3",
        Some(5.0),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let nuclide_map = HashMap::from([
        ("Li6".to_string(), "tests/Li6.arrow".to_string()),
        ("Fe56".to_string(), "tests/Fe56.arrow".to_string()),
    ]);
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

/// Li6+Fe56 sphere (r=10), vacuum beyond. One cell, id 1.
fn build_csg() -> Geometry {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            sphere,
        )))),
    };
    let cell = Cell::new(Some(1), region, Some("ball".to_string()), Some(0));
    Geometry::new(vec![cell], vec![li6_fe56_material()]).unwrap()
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

fn make_tally(score: &str, estimator: Estimator, cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec![score.parse::<Score>().unwrap()];
    t.estimator = estimator;
    t.name = Some(score.to_string());
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

/// Run one CPU config and return per-score summed means in `NEUTRON_SCORES` order.
fn run_cpu_config(tracking: TrackingMode, estimator: Estimator, survival: bool) -> Vec<f64> {
    let tallies: Vec<Arc<Tally>> = NEUTRON_SCORES
        .iter()
        .map(|s| make_tally(s, estimator, 1))
        .collect();
    let mut model = Model::new(build_csg(), vec![neutron_source()], tallies.clone());
    model.verbose = Verbose::silent();
    model.tracking_mode = tracking;
    model.max_steps_per_particle = 10_000;
    if survival {
        model.variance_reduction = vec![VarianceReduction::SurvivalBiasing(
            SurvivalBiasing::default(),
        )];
    }
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        })
        .unwrap_or_else(|e| {
            panic!("CPU run failed for {tracking:?}/{estimator:?}/sb={survival}: {e}")
        });
    tallies.iter().map(|t| t.get_mean().iter().sum()).collect()
}

/// CPU: every (tracking × estimator × capture) combination runs, every score
/// is finite and non-negative, flux is strictly positive, and every value
/// stays within a factor of 3 of the surface/track-length/analog baseline
/// (rules out a silently-broken combo producing zeros or garbage).
#[test]
fn cpu_neutron_full_score_cross() {
    let trackings = [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ];
    let estimators = [Estimator::TrackLength, Estimator::Collision];
    let captures = [false, true];

    // Baseline: surface / track-length / analog.
    let baseline = run_cpu_config(TrackingMode::Surface, Estimator::TrackLength, false);

    // Markdown rows for the human-facing table (printed with --nocapture).
    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "| tracking | estimator | capture | all-scores finite | flux>0 | within 3x baseline |"
            .into(),
    );
    rows.push("|---|---|---|---|---|---|".into());

    for &tracking in &trackings {
        for &estimator in &estimators {
            for &survival in &captures {
                let means = run_cpu_config(tracking, estimator, survival);
                let mut all_finite = true;
                let mut within = true;
                for (i, score) in NEUTRON_SCORES.iter().enumerate() {
                    let v = means[i];
                    assert!(
                        v.is_finite() && v >= 0.0,
                        "{tracking:?}/{estimator:?}/sb={survival} score {score}: \
                         non-finite or negative ({v})"
                    );
                    if !(v.is_finite() && v >= 0.0) {
                        all_finite = false;
                    }
                    let b = baseline[i];
                    if b > 1e-12 {
                        // nonzero score: must track the baseline within 3x
                        let ratio = v / b;
                        if !(ratio > 0.3 && ratio < 3.0) {
                            within = false;
                        }
                        assert!(
                            ratio > 0.3 && ratio < 3.0,
                            "{tracking:?}/{estimator:?}/sb={survival} score {score}: \
                             {v:.4e} vs baseline {b:.4e} (ratio {ratio:.3}) outside [0.3,3.0]"
                        );
                    } else {
                        // baseline ~0: this combo must also be ~0 (scaled to flux)
                        assert!(
                            v <= 1e-9 * baseline[0].max(1.0),
                            "{tracking:?}/{estimator:?}/sb={survival} score {score}: \
                             expected ~0 (baseline {b:.4e}) got {v:.4e}"
                        );
                    }
                }
                let flux = means[0];
                assert!(
                    flux > 0.0,
                    "{tracking:?}/{estimator:?}/sb={survival}: flux must be > 0 (got {flux})"
                );
                rows.push(format!(
                    "| {tracking:?} | {estimator:?} | {} | {} | {} | {} |",
                    if survival { "survival" } else { "analog" },
                    if all_finite { "✓" } else { "✗" },
                    if flux > 0.0 { "✓" } else { "✗" },
                    if within { "✓" } else { "✗" },
                ));
            }
        }
    }

    eprintln!("\n### CPU neutron matrix (CSG, full score cross)\n");
    for r in &rows {
        eprintln!("{r}");
    }
}

/// Mesh-geometry transport runs in every tracking/estimator combination
/// (the mesh axis of the matrix). Uses the two-region Arrow fixture
/// (fuel + moderator unit cubes).
#[cfg(feature = "mesh")]
#[test]
fn cpu_neutron_mesh_geometry_runs_all_modes() {
    use yamc::geometry::mesh::MeshGeometry;

    let li6 = li6_fe56_material();
    let mut fe = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.8),
    )
    .unwrap();
    fe.set_material_id(2);
    fe.set_temperature("294");
    fe.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]),
        None,
    )
    .unwrap();

    let materials: HashMap<String, Arc<Material>> = HashMap::from([
        ("fuel".to_string(), li6),
        ("moderator".to_string(), Arc::new(fe)),
    ]);
    let path = std::path::Path::new("../yamt/tests/data/two_region.arrow");

    for tracking in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        for estimator in [Estimator::TrackLength, Estimator::Collision] {
            let mesh = MeshGeometry::from_arrow(path, &materials).expect("load two_region.arrow");
            // No cell filter: score flux over the whole mesh (synthetic mesh
            // cell ids are not the CSG ids, so a filter-free tally is robust).
            let mut t = Tally::new();
            t.scores = vec!["flux".parse::<Score>().unwrap()];
            t.estimator = estimator;
            t.name = Some("mesh_flux".to_string());
            t.initialize_batches(N_BATCHES);
            let t = Arc::new(t);

            let mut model = Model::new_with_mesh(mesh, vec![neutron_source()], vec![t.clone()]);
            model.verbose = Verbose::silent();
            model.tracking_mode = tracking;
            model.max_steps_per_particle = 10_000;
            model
                .simulate_transport(&TransportSettings {
                    total_particles: Some(N_PER_BATCH * N_BATCHES),
                    seed: 42,
                    threads: Some(1),
                    ..Default::default()
                })
                .unwrap_or_else(|e| panic!("mesh CPU run failed {tracking:?}/{estimator:?}: {e}"));
            let flux: f64 = t.get_mean().iter().sum();
            assert!(
                flux.is_finite() && flux > 0.0,
                "mesh {tracking:?}/{estimator:?}: flux must be finite and > 0 (got {flux})"
            );
        }
    }
}

// --- GPU rejection guards (no working kernel needed: these short-circuit
//     before GpuContext::new(), so they verify the GPU "✗" cells of the
//     matrix even on hosts without a usable adapter). ---

#[cfg(feature = "gpu")]
mod gpu_guards {
    use super::*;

    fn neutron_csg_model() -> Model {
        let t = make_tally("flux", Estimator::TrackLength, 1);
        let mut m = Model::new(build_csg(), vec![neutron_source()], vec![t]);
        m.verbose = Verbose::silent();
        m
    }

    /// Same Li6+Fe56 sphere as `build_csg`, but with the "inside the sphere"
    /// region expressed as `Below(sphere)` instead of `Complement(Above(..))`.
    /// They are geometrically identical, but the GPU AABB pass can bound a
    /// `Below` halfspace and rejects a `Complement` ("unbounded extent"). The
    /// CPU accepts either; we run CPU and GPU on THIS geometry so the
    /// comparison is apples-to-apples.
    fn build_csg_gpu() -> Geometry {
        let sphere = Arc::new(Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 10.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        });
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let cell = Cell::new(Some(1), region, Some("ball".to_string()), Some(0));
        Geometry::new(vec![cell], vec![li6_fe56_material()]).unwrap()
    }

    fn err_string(m: &mut Model) -> String {
        // Rejection guards short-circuit before dispatch; the count/seed here
        // match what the guard models previously carried (N_PER_BATCH, 42).
        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH),
            seed: 42,
            ..Default::default()
        };
        match yamc::gpu::run_on_gpu(m, &settings) {
            Ok(_) => panic!("expected GPU dispatch to be REJECTED, but it ran"),
            Err(e) => e.to_string(),
        }
    }

    // NOTE: there is intentionally no `gpu_rejects_survival_biasing` guard.
    // Survival biasing (implicit capture + weight-cutoff Russian roulette) is
    // now SUPPORTED on the GPU neutron kernel (PR #71/#81); the dispatch
    // accepts a `VarianceReduction::SurvivalBiasing` entry rather than
    // rejecting it (see `run_on_gpu_with_device`). The GPU-vs-CPU agreement of
    // the survival-biasing path is covered positively by
    // `gpu_cpu_comparison_matrix` (its `survival: bool` sweep), so the old
    // rejection guard was removed as obsolete rather than left asserting a
    // behaviour the kernel no longer has.

    /// Coupled neutron->photon transport IS supported on GPU now (the two-pass
    /// kernel + device bank; end-to-end GPU-vs-CPU coverage lives in
    /// `gpu_coupled_photon.rs`). It does require the material to carry photon
    /// data: a coupled run on a neutron-only model (no photon data) fails fast
    /// at photon-data preparation, BEFORE any GPU context is created -- so this
    /// guard runs on any host (it is not a "coupled is rejected" check; coupled
    /// is accepted, the data is what is missing here).
    #[test]
    fn gpu_coupled_requires_photon_data() {
        let mut m = neutron_csg_model();
        m.transport_secondary_photons = true;
        let s = err_string(&mut m);
        assert!(
            s.contains("photon"),
            "coupled run without photon data should fail at photon-data prep, got: {s}"
        );
    }

    /// D1S decay-photon transport IS supported on GPU now (it routes through the
    /// same coupled two-pass path; end-to-end coverage lives in
    /// `gpu_d1s_decay_photon.rs`). Like the coupled case it requires data: a D1S
    /// run whose transmutation chain cannot be assembled (here the default
    /// `endf-b8.1` decay library is unresolvable because the test build has no
    /// `download` feature and no local chain path) fails fast at decay-data
    /// preparation, BEFORE any GPU context -- so this guard runs on any host. It
    /// is not a "decay photons are rejected" check; they are accepted, the decay
    /// data is what is missing here.
    #[test]
    fn gpu_decay_photons_require_decay_data() {
        let mut m = neutron_csg_model();
        m.use_decay_photons = true;
        let s = err_string(&mut m);
        assert!(
            s.contains("decay photon") || s.contains("transmutation chain"),
            "D1S run without decay data should fail at decay-data prep, got: {s}"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn gpu_rejects_mesh_geometry() {
        use yamc::geometry::mesh::MeshGeometry;
        let li6 = li6_fe56_material();
        let materials: HashMap<String, Arc<Material>> = HashMap::from([
            ("fuel".to_string(), li6.clone()),
            ("moderator".to_string(), li6),
        ]);
        let mesh = MeshGeometry::from_arrow(
            std::path::Path::new("../yamt/tests/data/two_region.arrow"),
            &materials,
        )
        .unwrap();
        let t = make_tally("flux", Estimator::TrackLength, 1);
        let mut m = Model::new_with_mesh(mesh, vec![neutron_source()], vec![t]);
        m.verbose = Verbose::silent();
        // A mesh-geometry model can NEVER run on the GPU kernel (CSG-only);
        // assert it is rejected (mesh guard, or an earlier one - either way ✗).
        let s = err_string(&mut m);
        eprintln!("mesh GPU rejection: {s}");
    }

    /// A CSG geometry whose cell is filled by a mesh body (issue #232)
    /// passes the GeometryKind::Mesh rejection (it is the Csg variant),
    /// so it needs its own guard in translate_for_gpu.
    #[cfg(feature = "mesh")]
    #[test]
    fn gpu_rejects_mesh_filled_cells() {
        use yamc::geometry::fill::CellFillSpec;
        use yamc::geometry::mesh::MeshGeometry;
        let li6 = li6_fe56_material();
        let materials: HashMap<String, Arc<Material>> = HashMap::from([
            ("fuel".to_string(), li6.clone()),
            ("moderator".to_string(), li6.clone()),
        ]);
        let mesh = MeshGeometry::from_arrow(
            std::path::Path::new("../yamt/tests/data/two_region.arrow"),
            &materials,
        )
        .unwrap();
        let sphere = Arc::new(Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.5,
                y0: 0.5,
                z0: 0.5,
                radius: 10.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        });
        let host = Cell::new(
            Some(1),
            Region::new_from_halfspace(HalfspaceType::Below(sphere)),
            Some("host".to_string()),
            Some(0),
        );
        let geometry = Geometry::new_with_fills(
            vec![host],
            vec![li6],
            vec![CellFillSpec {
                host_cell_index: 0,
                mesh_geometry: mesh,
                translation: [0.0; 3],
                rotation_degrees: [0.0; 3],
                allow_clipping: false,
            }],
        )
        .unwrap();
        let t = make_tally("flux", Estimator::TrackLength, 1);
        let mut m = Model::new(geometry, vec![neutron_source()], vec![t]);
        m.verbose = Verbose::silent();
        let s = err_string(&mut m);
        assert!(
            s.contains("mesh-filled"),
            "hybrid model must hit the mesh-fill guard, got: {s}"
        );
    }

    /// Per-channel GPU-vs-CPU matching check shared by the neutron and photon
    /// sweeps. Builds the markdown agreement table (printed with `--nocapture`)
    /// and panics listing every channel that fails to match.
    ///
    /// The matrix has two jobs: does the combination *run* (the `gpu_rejects_*`
    /// guards) and does it *match* the CPU. This is the "match" half, so the
    /// band is tight enough to mean something:
    /// - flux (index 0) must be strictly positive on both sides.
    /// - A channel whose CPU value is negligible vs flux (`< 1e-6 * flux`, e.g.
    ///   a closed reaction channel or a ~1e-10 trace) is noise-dominated: we
    ///   only require the GPU not to invent a signal there.
    /// - Every other channel must MATCH within **[0.80, 1.25]**. That budgets
    ///   the known ~2–3% XS-interpolation/URR offset plus the GPU's ~few-%
    ///   run-to-run non-determinism (the tally readback is not bit-reproducible
    ///   across launches even at a fixed seed) plus low-statistics noise on the
    ///   smaller channels, while still catching a real regression (zero, 2×).
    ///   EXCEPT the known reaction-removal divergences in `known_divergent`
    ///   (issue #415: neutron `absorption` ~30–35% high, photon `photoelectric`
    ///   ~60% low), which get a wide **[0.2, 5.0]** guard (still catches a
    ///   zero/blow-up) until #415 is fixed and the band can be tightened.
    fn compare_gpu_cpu(
        label: &str,
        names: &[&str],
        cpu: &[f64],
        gpu: &[f64],
        known_divergent: &[&str],
    ) {
        assert!(
            cpu[0] > 0.0 && gpu[0] > 0.0,
            "{label}: flux must be > 0 on both (CPU {}, GPU {})",
            cpu[0],
            gpu[0]
        );
        let flux = cpu[0];
        let mut rows: Vec<String> = vec![
            "| score | CPU | GPU | ratio | check |".into(),
            "|---|---|---|---|---|".into(),
        ];
        let mut bad: Vec<String> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let (c, g) = (cpu[i], gpu[i]);
            if !(g.is_finite() && g >= 0.0) {
                bad.push(format!("{name}: GPU value not finite/non-negative ({g})"));
            }
            let (ratio_str, ok) = if c < 1e-6 * flux {
                // Negligible / closed channel -> only require the GPU not to
                // invent a signal (ratios here are pure noise).
                let ok = g < 1e-3 * flux;
                if !ok {
                    bad.push(format!(
                        "{name}: CPU negligible ({c:.4e}) but GPU = {g:.4e} (flux {flux:.4e})"
                    ));
                }
                ("n/a (CPU negligible)".to_string(), ok)
            } else {
                let r = g / c;
                let divergent = known_divergent.contains(name);
                let (lo, hi) = if divergent { (0.2, 5.0) } else { (0.80, 1.25) };
                let ok = r >= lo && r <= hi;
                if !ok {
                    bad.push(format!(
                        "{name}: GPU/CPU ratio {r:.3} out of [{lo}, {hi}] (CPU {c:.4e}, GPU {g:.4e})"
                    ));
                }
                let tag = if divergent {
                    format!("{r:.3} (#415)")
                } else {
                    format!("{r:.3}")
                };
                (tag, ok)
            };
            rows.push(format!(
                "| {name} | {c:.4e} | {g:.4e} | {ratio_str} | {} |",
                if ok { "ok" } else { "BAD" }
            ));
        }
        eprintln!("\nGPU supported-slice matrix ({label}):");
        for r in &rows {
            eprintln!("{r}");
        }
        assert!(
            bad.is_empty(),
            "GPU {label} scores disagree with CPU:\n  {}",
            bad.join("\n  ")
        );
    }

    /// GPU "supported" cells - the GPU analogue of `cpu_neutron_full_score_cross`.
    ///
    /// The GPU kernel accepts only one slice of the option space: neutron
    /// transport on CSG geometry, track-length estimator, analog capture
    /// (it ignores `tracking_mode` and ray-traces its own boundaries). Within
    /// that slice this sweeps the **full neutron score set** on the GPU and
    /// asserts every score (a) comes back finite and non-negative, and
    /// (b) agrees with the CPU reference within a breadth band wherever the
    /// CPU sees a non-zero channel. This is the guard that the GPU "✓" cells
    /// actually produce CPU-consistent tallies - the all-zero failure
    /// (#388) is fixed in #412.
    ///
    /// Self-skips when no f64 GPU adapter is present, so a plain `cargo test`
    /// stays green on CI (which has no such adapter). On real hardware it must
    /// run single-threaded: `cargo test-gpu` (the shared cubecl client
    /// serializes launches).
    ///
    /// Precision is NOT the point here (the GPU has a known heavy-Z physics
    /// offset and this runs at low statistics) - exact agreement is pinned by
    /// the per-physics `gpu_*.rs` suite. The band only rules out a silently
    /// broken / zeroed score.
    #[test]
    fn gpu_neutron_full_score_cross() {
        if yamc_gpu::GpuContext::new().is_err() {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }

        // CPU reference in the GPU-matching configuration: surface tracking
        // (GPU ignores the mode), track-length, analog, on the GPU-boundable
        // geometry.
        let cpu_tallies: Vec<Arc<Tally>> = NEUTRON_SCORES
            .iter()
            .map(|s| make_tally(s, Estimator::TrackLength, 1))
            .collect();
        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        };
        let mut cpu_model =
            Model::new(build_csg_gpu(), vec![neutron_source()], cpu_tallies.clone());
        cpu_model.verbose = Verbose::silent();
        cpu_model.tracking_mode = TrackingMode::Surface;
        cpu_model.max_steps_per_particle = 10_000;
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU reference run");
        let cpu: Vec<f64> = cpu_tallies
            .iter()
            .map(|t| t.get_mean().iter().sum())
            .collect();

        // GPU run: same geometry + seed + score set.
        let gpu_tallies: Vec<Arc<Tally>> = NEUTRON_SCORES
            .iter()
            .map(|s| make_tally(s, Estimator::TrackLength, 1))
            .collect();
        let mut m = Model::new(build_csg_gpu(), vec![neutron_source()], gpu_tallies.clone());
        m.verbose = Verbose::silent();
        m.max_steps_per_particle = 10_000;
        yamc::gpu::run_on_gpu(&mut m, &settings)
            .expect("GPU dispatch for the supported CSG/neutron/track-length/analog slice");
        let gpu: Vec<f64> = gpu_tallies
            .iter()
            .map(|t| t.get_mean().iter().sum())
            .collect();

        // Verify the match. Every neutron channel now matches within the
        // band (absorption was fixed in #415 by routing it through the
        // tabulated MT-27 score path instead of the derived σ_a).
        compare_gpu_cpu(
            "CSG / neutron / track-length / analog",
            NEUTRON_SCORES,
            &cpu,
            &gpu,
            &[],
        );
    }

    /// A non-Surface `tracking_mode` is warned about, NOT rejected (#66).
    /// The GPU always surface-tracks, but surface tracking and Woodcock /
    /// Hybrid are all unbiased estimators of the same flux, so a
    /// `tracking_mode = Woodcock` neutron model must still dispatch `Ok` and
    /// return a sane (strictly positive) flux. Refusing would drop a working,
    /// numerically-correct capability; the warn-not-reject contract is what
    /// this pins. (Whether the warning *text* fires is a verbose-stream
    /// detail; the must-have is Ok + sane result.) Self-skips without an f64
    /// adapter; run single-threaded via `cargo test-gpu`.
    #[test]
    fn gpu_woodcock_tracking_warns_not_rejected() {
        if yamc_gpu::GpuContext::new().is_err() {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }

        let flux = make_tally("flux", Estimator::TrackLength, 1);
        let mut m = Model::new(build_csg_gpu(), vec![neutron_source()], vec![flux.clone()]);
        m.verbose = Verbose::silent();
        m.max_steps_per_particle = 10_000;
        // The request the GPU cannot honor: must warn-and-proceed, not reject.
        m.tracking_mode = TrackingMode::Woodcock;

        yamc::gpu::run_on_gpu(
            &mut m,
            &TransportSettings {
                total_particles: Some(N_PER_BATCH * N_BATCHES),
                seed: 42,
                ..Default::default()
            },
        )
        .expect("GPU dispatch with tracking_mode=Woodcock must succeed (warn-not-reject, #66)");
        let total: f64 = flux.get_mean().iter().sum();
        assert!(
            total > 0.0,
            "GPU flux under Woodcock request must be sane (positive), got {total}"
        );
    }

    /// GPU "supported" cells - primary-photon companion to
    /// `gpu_neutron_full_score_cross`. The GPU's other ✓ cell is standalone
    /// primary-photon transport (a pure `PhotonSource`); this sweeps the
    /// photon score set (flux + the four photon-XS components) on that slice
    /// and asserts each agrees with the CPU reference. Mirrors the proven
    /// `gpu_photon_components` setup: 1 cm Fe sphere, Co60-mean (1.25 MeV)
    /// point source. (`transport_secondary_photons=true` is accepted here -
    /// the GPU only rejects it for a *neutron* source, i.e. coupled n→γ.)
    /// Self-skips without an f64 adapter; run via `cargo test-gpu`.
    ///
    /// At 1.25 MeV Compton dominates, so flux / coherent / incoherent / pair
    /// (pair is just above the 1.022 MeV threshold, so small but nonzero) all
    /// match the CPU within the agreement band; `photoelectric` is the known
    /// reaction-removal divergence (issue #415) and gets the wide guard.
    #[test]
    fn gpu_photon_full_score_cross() {
        use yamc_tallies::filter::particle_type::ParticleTypeFilter;
        use yamc_tallies::score::{FluxScore, PhotonComponent, PhotonXSScore};

        if yamc_gpu::GpuContext::new().is_err() {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }

        let names = [
            "flux",
            "coherent-scatter",
            "incoherent-scatter",
            "photoelectric",
            "pair-production",
        ];

        // Fe sphere (r=1, Below -> GPU-boundable), Co60-mean photon point
        // source, full photon score set. Built fresh for the CPU and GPU runs.
        let build = || -> (Model, Vec<Arc<Tally>>) {
            let sphere = Arc::new(Surface {
                surface_id: Some(1),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: 1.0,
                },
                boundary: BoundaryType::Vacuum,
                name: None,
            });
            let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
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
            let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
            let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
            let source = ParticleSource::Photon(Source {
                space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
                angle: AngularDistribution::Isotropic,
                energy: SourceEnergyDistribution::Discrete(
                    Discrete::new(vec![1.25e6], vec![1.0]).unwrap(),
                ),
                strength: 1.0,
            });
            let mk = |score: Score| {
                let mut t = Tally::new();
                t.filters.push(Filter::Cell(CellFilter::from_id(1)));
                t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
                    yamc_particle::particle::ParticleType::Photon,
                )));
                t.scores = vec![score];
                t.initialize_batches(N_BATCHES);
                Arc::new(t)
            };
            let tallies = vec![
                mk(Score::Flux(FluxScore)),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Coherent,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Incoherent,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Photoelectric,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::PairProduction,
                })),
            ];
            let mut model = Model::new(geometry, vec![source], tallies.clone());
            model.verbose = Verbose::silent();
            model.max_steps_per_particle = 5_000;
            model.transport_secondary_photons = true;
            (model, tallies)
        };

        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        };
        let (mut cpu_model, cpu_t) = build();
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU reference run");
        let cpu: Vec<f64> = cpu_t.iter().map(|t| t.get_mean().iter().sum()).collect();

        let (mut gpu_model, gpu_t) = build();
        yamc::gpu::run_on_gpu(&mut gpu_model, &settings)
            .expect("GPU dispatch for the supported CSG/primary-photon/track-length/analog slice");
        let gpu: Vec<f64> = gpu_t.iter().map(|t| t.get_mean().iter().sum()).collect();

        // Every photon channel is enforced: the photoelectric ~0.38x deficit
        // (#415) was the GPU running with empty TTB tables -- `run_on_gpu`
        // now self-prepares photon data (TTB / Doppler / relaxation), so the
        // bremsstrahlung photon source matches the CPU and photoelectric
        // sits in-band (~1.1).
        compare_gpu_cpu(
            "CSG / primary photon / track-length / analog",
            &names,
            &cpu,
            &gpu,
            &[],
        );
    }

    /// GPU-vs-CPU comparison for collision-estimator tallies. Same
    /// shape as `compare_gpu_cpu`, but with a WIDE per-bin band of
    /// **[0.5, 2.0]**: the collision (collision-density) estimator
    /// scores once per real collision (`weight × score_xs / Σ_t`)
    /// rather than integrating a path length each step, so its
    /// per-history variance is much larger than track-length's. At the
    /// modest statistics this matrix runs (20k histories) the GPU and
    /// CPU collision tallies are genuinely noisier than their
    /// track-length counterparts, and the GPU tally readback is not
    /// bit-reproducible across launches even at a fixed seed (the
    /// atomic-add order varies). The band is therefore loosened from
    /// the track-length [0.80, 1.25] to [0.5, 2.0] -- still tight
    /// enough to catch a missing `1/Σ_t`, a double-score, or a zeroed
    /// channel (the failure modes this guards), but loose enough not to
    /// flake on collision-estimator noise.
    fn compare_gpu_cpu_collision(label: &str, names: &[&str], cpu: &[f64], gpu: &[f64]) {
        assert!(
            cpu[0] > 0.0 && gpu[0] > 0.0,
            "{label}: flux must be > 0 on both (CPU {}, GPU {})",
            cpu[0],
            gpu[0]
        );
        let flux = cpu[0];
        let mut rows: Vec<String> = vec![
            "| score | CPU (collision) | GPU (collision) | ratio | check |".into(),
            "|---|---|---|---|---|".into(),
        ];
        let mut bad: Vec<String> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let (c, g) = (cpu[i], gpu[i]);
            if !(g.is_finite() && g >= 0.0) {
                bad.push(format!("{name}: GPU value not finite/non-negative ({g})"));
            }
            let (ratio_str, ok) = if c < 1e-6 * flux {
                // Negligible / closed channel -> only require the GPU not to
                // invent a signal.
                let ok = g < 1e-3 * flux;
                if !ok {
                    bad.push(format!(
                        "{name}: CPU negligible ({c:.4e}) but GPU = {g:.4e} (flux {flux:.4e})"
                    ));
                }
                ("n/a (CPU negligible)".to_string(), ok)
            } else {
                let r = g / c;
                let (lo, hi) = (0.5, 2.0);
                let ok = r >= lo && r <= hi;
                if !ok {
                    bad.push(format!(
                        "{name}: GPU/CPU ratio {r:.3} out of [{lo}, {hi}] (CPU {c:.4e}, GPU {g:.4e})"
                    ));
                }
                (format!("{r:.3}"), ok)
            };
            rows.push(format!(
                "| {name} | {c:.4e} | {g:.4e} | {ratio_str} | {} |",
                if ok { "ok" } else { "BAD" }
            ));
        }
        eprintln!("\nGPU collision-estimator matrix ({label}):");
        for r in &rows {
            eprintln!("{r}");
        }
        assert!(
            bad.is_empty(),
            "GPU {label} collision scores disagree with CPU collision estimator:\n  {}",
            bad.join("\n  ")
        );
    }

    /// GPU collision-estimator cell (neutron). Companion to
    /// `gpu_neutron_full_score_cross`: same CSG / neutron / analog
    /// slice, but every tally uses `Estimator::Collision`. The GPU
    /// kernel now scores `weight × score_xs / Σ_t` once per real
    /// collision (the per-tally `is_collision` flag in the tally pack
    /// gates the per-step track-length block off). This compares the
    /// GPU collision tallies against the CPU collision estimator (NOT
    /// track-length) within the wide collision band [0.5, 2.0].
    ///
    /// Self-skips without an f64 adapter; run via `cargo test-gpu`.
    #[test]
    fn gpu_neutron_collision_estimator_cross() {
        if yamc_gpu::GpuContext::new().is_err() {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }

        // CPU reference: collision estimator, same geometry/seed/scores.
        let cpu_tallies: Vec<Arc<Tally>> = NEUTRON_SCORES
            .iter()
            .map(|s| make_tally(s, Estimator::Collision, 1))
            .collect();
        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        };
        let mut cpu_model =
            Model::new(build_csg_gpu(), vec![neutron_source()], cpu_tallies.clone());
        cpu_model.verbose = Verbose::silent();
        cpu_model.tracking_mode = TrackingMode::Surface;
        cpu_model.max_steps_per_particle = 10_000;
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU collision reference run");
        let cpu: Vec<f64> = cpu_tallies
            .iter()
            .map(|t| t.get_mean().iter().sum())
            .collect();

        // GPU run: collision estimator on the same slice.
        let gpu_tallies: Vec<Arc<Tally>> = NEUTRON_SCORES
            .iter()
            .map(|s| make_tally(s, Estimator::Collision, 1))
            .collect();
        let mut m = Model::new(build_csg_gpu(), vec![neutron_source()], gpu_tallies.clone());
        m.verbose = Verbose::silent();
        m.max_steps_per_particle = 10_000;
        yamc::gpu::run_on_gpu(&mut m, &settings)
            .expect("GPU dispatch for CSG/neutron/collision/analog slice");
        let gpu: Vec<f64> = gpu_tallies
            .iter()
            .map(|t| t.get_mean().iter().sum())
            .collect();

        compare_gpu_cpu_collision(
            "CSG / neutron / collision / analog",
            NEUTRON_SCORES,
            &cpu,
            &gpu,
        );
    }

    /// GPU collision-estimator cell (primary photon). Companion to
    /// `gpu_photon_full_score_cross`: same Fe-sphere / Co60-mean photon
    /// slice, but every tally uses `Estimator::Collision`. Compares GPU
    /// collision tallies against the CPU collision estimator within the
    /// wide collision band [0.5, 2.0]. Self-skips without an f64
    /// adapter; run via `cargo test-gpu`.
    #[test]
    fn gpu_photon_collision_estimator_cross() {
        use yamc_tallies::filter::particle_type::ParticleTypeFilter;
        use yamc_tallies::score::{FluxScore, PhotonComponent, PhotonXSScore};

        if yamc_gpu::GpuContext::new().is_err() {
            eprintln!("skipping -- no GPU with f64 compute available");
            return;
        }

        let names = [
            "flux",
            "coherent-scatter",
            "incoherent-scatter",
            "photoelectric",
            "pair-production",
        ];

        let build = || -> (Model, Vec<Arc<Tally>>) {
            let sphere = Arc::new(Surface {
                surface_id: Some(1),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: 1.0,
                },
                boundary: BoundaryType::Vacuum,
                name: None,
            });
            let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
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
            let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
            let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
            let source = ParticleSource::Photon(Source {
                space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
                angle: AngularDistribution::Isotropic,
                energy: SourceEnergyDistribution::Discrete(
                    Discrete::new(vec![1.25e6], vec![1.0]).unwrap(),
                ),
                strength: 1.0,
            });
            let mk = |score: Score| {
                let mut t = Tally::new();
                t.filters.push(Filter::Cell(CellFilter::from_id(1)));
                t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
                    yamc_particle::particle::ParticleType::Photon,
                )));
                t.scores = vec![score];
                t.estimator = Estimator::Collision;
                t.initialize_batches(N_BATCHES);
                Arc::new(t)
            };
            let tallies = vec![
                mk(Score::Flux(FluxScore)),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Coherent,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Incoherent,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Photoelectric,
                })),
                mk(Score::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::PairProduction,
                })),
            ];
            let mut model = Model::new(geometry, vec![source], tallies.clone());
            model.verbose = Verbose::silent();
            model.max_steps_per_particle = 5_000;
            model.transport_secondary_photons = true;
            (model, tallies)
        };

        let settings = TransportSettings {
            total_particles: Some(N_PER_BATCH * N_BATCHES),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        };
        let (mut cpu_model, cpu_t) = build();
        cpu_model
            .simulate_transport(&settings)
            .expect("CPU collision reference run");
        let cpu: Vec<f64> = cpu_t.iter().map(|t| t.get_mean().iter().sum()).collect();

        let (mut gpu_model, gpu_t) = build();
        yamc::gpu::run_on_gpu(&mut gpu_model, &settings)
            .expect("GPU dispatch for CSG/primary-photon/collision/analog slice");
        let gpu: Vec<f64> = gpu_t.iter().map(|t| t.get_mean().iter().sum()).collect();

        compare_gpu_cpu_collision(
            "CSG / primary photon / collision / analog",
            &names,
            &cpu,
            &gpu,
        );
    }
}
