//! GPU void-cell (material-less region) parity with the CPU.
//!
//! The GPU used to reject any material-less cell (`VoidCellsUnsupported`).
//! Void regions (gaps, vacuum vessel, plasma chamber) are common in fusion
//! geometry, so this closes a real CPU-vs-GPU parity gap. A void cell maps to
//! a synthetic all-zero "void material" slot: `sigma_t = 0` makes the
//! distance-to-collision infinite, so the particle streams to the next
//! surface with no interaction. Track-length flux still accrues over the
//! traversed void segment; reaction-rate / total / heating tallies score 0
//! (no material to react with).
//!
//! These tests need a real f64 GPU adapter and self-skip on CI (run
//! single-threaded via `cargo test-gpu`). They assert the GPU track-length
//! flux matches the CPU in both the void cell and the surrounding material
//! shell, and that the void cell flux is nonzero (the source neutron / photon
//! genuinely streams through it).

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
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

const N_PER_BATCH: usize = 4_000;
const N_BATCHES: usize = 8;

fn plane(id: usize, a: f64, b: f64, c: f64, d: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Plane { a, b, c, d },
        boundary,
        name: None,
    })
}

/// `AND` a list of (surface, above?) halfspaces into one intersection region.
/// `above == true` keeps `a·r > d` (the `Above` side), `false` the `Below`
/// side. The result is a fully-bounded box-slab with a finite AABB the GPU
/// can accept.
fn box_region(parts: &[(Arc<Surface>, bool)]) -> Region {
    let mut iter = parts.iter().map(|(s, above)| {
        let hs = if *above {
            HalfspaceType::Above(Arc::clone(s))
        } else {
            HalfspaceType::Below(Arc::clone(s))
        };
        RegionExpr::Halfspace(hs)
    });
    let first = iter.next().expect("at least one halfspace");
    let expr = iter.fold(first, |acc, e| {
        RegionExpr::Intersection(Box::new(acc), Box::new(e))
    });
    Region { expr }
}

fn fe56_material() -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(5.0),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

/// Three stacked z-slabs in a finite box, sharing one set of surfaces (so a
/// surface_id is never duplicated): bottom (-15 < z < -5), middle (-5 < z < 5),
/// top (5 < z < 15), all within x,y in [-2, 2]. `middle_void` makes the middle
/// slab a void cell (`material_idx = None`); otherwise it is Fe56. The slabs
/// have DISJOINT AABBs (no nesting), which the GPU's AABB cell-finding
/// requires. Cell ids are 1 (bottom) / 2 (middle) / 3 (top).
///
/// The cells are stored DOWNSTREAM-FIRST (`[top, middle, bottom]`). The GPU's
/// AABB cell-finding (shared by both backends at the kernel level) is a point-
/// in-AABB test, so when a +z particle lands EXACTLY on a shared slab face
/// (e.g. z = -5) both neighbouring AABBs match and the BVH returns the lower-
/// index cell. Ordering the downstream slab first makes that tie resolve into
/// the cell the beam is entering, which is what the CPU's region-membership
/// handoff does -- without it the boundary segment is mis-scored one slab back.
/// (This is a property of the AABB tie-break, not of void cells: the non-void
/// baseline uses the identical ordering.)
fn build_slab_geometry(middle_void: bool) -> Geometry {
    // Shared box surfaces -- created once, cloned into every slab.
    let x_lo = plane(1, 1.0, 0.0, 0.0, -2.0, BoundaryType::Vacuum);
    let x_hi = plane(2, 1.0, 0.0, 0.0, 2.0, BoundaryType::Vacuum);
    let y_lo = plane(3, 0.0, 1.0, 0.0, -2.0, BoundaryType::Vacuum);
    let y_hi = plane(4, 0.0, 1.0, 0.0, 2.0, BoundaryType::Vacuum);
    let z_bot = plane(5, 0.0, 0.0, 1.0, -15.0, BoundaryType::Vacuum);
    let z_top = plane(6, 0.0, 0.0, 1.0, 15.0, BoundaryType::Vacuum);
    let z_mid_lo = plane(7, 0.0, 0.0, 1.0, -5.0, BoundaryType::Transmission);
    let z_mid_hi = plane(8, 0.0, 0.0, 1.0, 5.0, BoundaryType::Transmission);

    let lateral = |extra: &[(Arc<Surface>, bool)]| -> Region {
        let mut parts = vec![
            (Arc::clone(&x_lo), true),
            (Arc::clone(&x_hi), false),
            (Arc::clone(&y_lo), true),
            (Arc::clone(&y_hi), false),
        ];
        parts.extend(extra.iter().cloned());
        box_region(&parts)
    };

    let bottom = lateral(&[(Arc::clone(&z_bot), true), (Arc::clone(&z_mid_lo), false)]);
    let middle = lateral(&[
        (Arc::clone(&z_mid_lo), true),
        (Arc::clone(&z_mid_hi), false),
    ]);
    let top = lateral(&[(Arc::clone(&z_mid_hi), true), (Arc::clone(&z_top), false)]);

    let cell_bottom = Cell::new(Some(1), bottom, Some("mat_lo".into()), Some(0));
    let mid_mat = if middle_void { None } else { Some(0) };
    let mid_name = if middle_void { "void" } else { "mat_mid" };
    let cell_mid = Cell::new(Some(2), middle, Some(mid_name.into()), mid_mat);
    let cell_top = Cell::new(Some(3), top, Some("mat_hi".into()), Some(0));

    // Downstream-first (see the doc comment): +z beam, so top before bottom.
    Geometry::new(vec![cell_top, cell_mid, cell_bottom], vec![fe56_material()]).unwrap()
}

/// Mat / VOID / mat: the middle slab is a void cell.
fn build_void_geometry() -> Geometry {
    build_slab_geometry(true)
}

/// Non-void baseline: the SAME box / slabs, but the middle slab is Fe56 too
/// (no void). The GPU sees no void slot at all -- this pins that void support
/// does not perturb the all-material path. Cell ids match `build_void_geometry`.
fn build_solid_geometry() -> Geometry {
    build_slab_geometry(false)
}

/// Monodirectional +z 14 MeV beam from z = -10 (inside the bottom material
/// slab). It crosses into the void slab at z = -5 and the top slab at z = 5.
fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, -10.0])),
        angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn make_flux_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec!["flux".parse::<Score>().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.name = Some(format!("flux_cell_{cell_id}"));
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn model(geometry: Geometry, tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut m = Model::new(geometry, vec![neutron_source()], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: 42,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

/// Summed-mean flux per tally after a CPU run.
fn run_cpu(geometry: Geometry, cell_ids: &[u32]) -> Vec<f64> {
    let tallies: Vec<Arc<Tally>> = cell_ids.iter().map(|&id| make_flux_tally(id)).collect();
    let (mut m, settings) = model(geometry, tallies.clone());
    m.simulate_transport(&settings).expect("CPU void run");
    tallies.iter().map(|t| t.get_mean().iter().sum()).collect()
}

/// Summed-mean flux per tally after a GPU run.
fn run_gpu(geometry: Geometry, cell_ids: &[u32]) -> Vec<f64> {
    let tallies: Vec<Arc<Tally>> = cell_ids.iter().map(|&id| make_flux_tally(id)).collect();
    let (mut m, settings) = model(geometry, tallies.clone());
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU void run");
    tallies.iter().map(|t| t.get_mean().iter().sum()).collect()
}

/// Assert two flux vectors agree within a breadth band (the GPU tally
/// readback is not bit-reproducible across launches even at a fixed seed,
/// and the heavy-Z XS interpolation carries a known few-% offset; the band
/// is wide enough for that but still catches a zero / 2x regression).
fn assert_match(label: &str, names: &[&str], cpu: &[f64], gpu: &[f64]) {
    let mut bad = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let (c, g) = (cpu[i], gpu[i]);
        assert!(
            g.is_finite() && g >= 0.0,
            "{label} {name}: GPU value not finite/non-negative ({g})"
        );
        let r = g / c;
        if !(0.85..=1.18).contains(&r) {
            bad.push(format!(
                "{name}: GPU/CPU ratio {r:.3} out of [0.85, 1.18] (CPU {c:.5e}, GPU {g:.5e})"
            ));
        }
        eprintln!("  {label} {name}: CPU {c:.5e}  GPU {g:.5e}  ratio {r:.3}");
    }
    assert!(
        bad.is_empty(),
        "{label} GPU disagrees with CPU:\n  {}",
        bad.join("\n  ")
    );
}

/// Void cell + material shell: the GPU streams the source neutron through the
/// inner void (no collision) and accrues track-length flux there, matching
/// the CPU. The shell flux must also match.
#[test]
fn gpu_void_neutron_flux_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // Cells: 1 = bottom material, 2 = VOID, 3 = top material.
    let cell_ids = [1u32, 2u32, 3u32];
    let cpu = run_cpu(build_void_geometry(), &cell_ids);
    let gpu = run_gpu(build_void_geometry(), &cell_ids);

    // The void cell (index 1) must see real (nonzero) traversal flux on BOTH
    // backends: the +z beam streams straight through the 10 cm void slab, so
    // the void track-length per source neutron is the full slab thickness.
    assert!(
        cpu[1] > 0.0,
        "CPU void-cell flux must be nonzero (the beam streams through it), got {}",
        cpu[1]
    );
    assert!(
        gpu[1] > 0.0,
        "GPU void-cell flux must be nonzero (the beam streams through it), got {}",
        gpu[1]
    );

    assert_match(
        "mat/void/mat",
        &["mat_lo_flux", "void_flux", "mat_hi_flux"],
        &cpu,
        &gpu,
    );
}

/// Non-void baseline: a single solid Fe56 sphere (no void slot) must still
/// match CPU exactly within the band -- the void support must not perturb the
/// material-only path.
#[test]
fn gpu_solid_baseline_unchanged() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let cell_ids = [1u32, 2u32, 3u32];
    let cpu = run_cpu(build_solid_geometry(), &cell_ids);
    let gpu = run_gpu(build_solid_geometry(), &cell_ids);

    assert!(
        cpu.iter().all(|&v| v > 0.0) && gpu.iter().all(|&v| v > 0.0),
        "solid baseline flux must be > 0 in every slab"
    );
    assert_match(
        "solid baseline",
        &["mat_lo_flux", "mat_mid_flux", "mat_hi_flux"],
        &cpu,
        &gpu,
    );
}
