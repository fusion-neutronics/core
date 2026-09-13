//! GPU `MaterialFilter` tally support (issue #271), on-hardware.
//!
//! A cell's material never changes during a run, so a material bin is a pure
//! function of the cell index -- exactly like a cell bin. The dispatch
//! therefore folds both into the kernel's single spatial dimension as
//! `cell_bin * n_material_bins + material_bin` and divides the product back
//! out at writeback. No kernel change is involved, which is precisely why this
//! test has to check the BIN MAPPING rather than the physics.
//!
//! The geometry is four nested shells over two materials in ALTERNATING order
//! (A, B, A, B), so a material bin is not a relabelled cell bin: getting the
//! mapping wrong cannot be hidden by a coincidental one-to-one match. Four
//! tallies run in the same GPU launch over the same histories:
//!
//!   1. `materials=[A, B]`                -- 2 bins
//!   2. `cells=[1, 2, 3, 4]`              -- 4 bins (the reference partition)
//!   3. `materials=[A, B]` + energy bins  -- 6 bins (pins the stride order)
//!   4. `materials=[A]`                   -- 1 bin (pins the exclusion gate)
//!
//! Because they score the same events, the invariants between them are exact
//! arithmetic identities, not statistical ones: material A's flux IS cell 1 +
//! cell 3. That makes the check independent of the CPU and of the fixture's
//! physics. A separate CPU run then pins GPU/CPU agreement on the same
//! material-binned tally.
//!
//! Needs a real f64 GPU adapter; self-skips otherwise. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_material_filter_tally -- --nocapture --test-threads=1

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::{Estimator, MaterialFilter};

const N_PARTICLES: usize = 40_000;
const N_BATCHES: usize = 8;
const SEED: u64 = 8675309;

/// Material ids the tallies filter on. Cells 1 and 3 hold `MAT_A`, cells 2
/// and 4 hold `MAT_B`.
const MAT_A: u32 = 11;
const MAT_B: u32 = 22;

/// Energy bin edges for the material x energy tally: three bins spanning the
/// 14 MeV source down to thermal.
const E_EDGES: [f64; 4] = [1e-3, 1e5, 1e6, 2e7];

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

/// Fe56 at a given density. One nuclide across both materials keeps a single
/// shared energy grid; the differing density gives each material a distinct
/// macroscopic cross section, so the two material bins carry genuinely
/// different flux rather than two halves of the same number.
fn material(id: u32, density_g_cm3: f64) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density_g_cm3),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

/// Four nested shells, materials alternating A, B, A, B so that neither
/// material maps onto a contiguous run of cells.
fn build_geometry() -> Geometry {
    let s1 = sphere(1, 3.0, BoundaryType::Transmission);
    let s2 = sphere(2, 6.0, BoundaryType::Transmission);
    let s3 = sphere(3, 9.0, BoundaryType::Transmission);
    let s4 = sphere(4, 12.0, BoundaryType::Vacuum);

    let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s1)));
    let shell = |inner: &Arc<Surface>, outer: &Arc<Surface>| {
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(inner))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(outer))),
        )
    };

    // material_idx 0 = MAT_A, 1 = MAT_B.
    let cells = vec![
        Cell::new(Some(1), core, Some("core_a".into()), Some(0)),
        Cell::new(Some(2), shell(&s1, &s2), Some("shell_b".into()), Some(1)),
        Cell::new(Some(3), shell(&s2, &s3), Some("shell_a".into()), Some(0)),
        Cell::new(Some(4), shell(&s3, &s4), Some("shell_b2".into()), Some(1)),
    ];
    Geometry::new(cells, vec![material(MAT_A, 2.0), material(MAT_B, 7.0)]).unwrap()
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

fn flux_tally(name: &str, filters: Vec<Filter>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters = filters;
    t.scores = vec!["flux".parse::<Score>().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.name = Some(name.to_string());
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

/// The four tallies described in the module docs, in a fixed order:
/// `[by_material, by_cell, by_material_energy, material_a_only]`.
fn build_tallies() -> Vec<Arc<Tally>> {
    vec![
        flux_tally(
            "by_material",
            vec![Filter::Material(MaterialFilter {
                material_ids: vec![MAT_A, MAT_B],
            })],
        ),
        flux_tally(
            "by_cell",
            vec![Filter::Cell(CellFilter {
                cell_ids: vec![1, 2, 3, 4],
            })],
        ),
        flux_tally(
            "by_material_energy",
            vec![
                Filter::Material(MaterialFilter {
                    material_ids: vec![MAT_A, MAT_B],
                }),
                Filter::Energy(EnergyFilter::new(E_EDGES.to_vec())),
            ],
        ),
        flux_tally(
            "material_a_only",
            vec![Filter::Material(MaterialFilter {
                material_ids: vec![MAT_A],
            })],
        ),
    ]
}

fn model(tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut m = Model::new(build_geometry(), vec![neutron_source()], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.gpu_max_steps_per_particle = 10_000;
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

/// The GPU's material bins must partition the same flux its cell bins do,
/// with the material dimension outside the energy dimension. Every identity
/// here is over ONE launch's histories, so it is exact arithmetic rather
/// than a statistical comparison.
#[test]
fn gpu_material_bins_partition_the_cell_bins() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let tallies = build_tallies();
    let (mut m, settings) = model(tallies.clone());
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");
    let out = means(&tallies);
    let (by_material, by_cell, by_material_energy, material_a_only) =
        (&out[0], &out[1], &out[2], &out[3]);

    assert_eq!(by_material.len(), 2, "material bins");
    assert_eq!(by_cell.len(), 4, "cell bins");
    assert_eq!(by_material_energy.len(), 6, "material x energy bins");
    assert_eq!(material_a_only.len(), 1, "single-material bin");
    for (i, v) in by_cell.iter().enumerate() {
        assert!(*v > 0.0, "cell {} scored no flux ({v})", i + 1);
    }

    // Materials alternate across the shells, so each material bin is the sum
    // of two NON-adjacent cell bins. A mapping that quietly indexed by cell
    // (or reversed the two) fails here.
    eprintln!(
        "  material A: {:.6e}  (cells 1 + 3 = {:.6e})",
        by_material[0],
        by_cell[0] + by_cell[2]
    );
    eprintln!(
        "  material B: {:.6e}  (cells 2 + 4 = {:.6e})",
        by_material[1],
        by_cell[1] + by_cell[3]
    );
    assert_close(
        "material A vs cells 1+3",
        by_material[0],
        by_cell[0] + by_cell[2],
        1e-9,
    );
    assert_close(
        "material B vs cells 2+4",
        by_material[1],
        by_cell[1] + by_cell[3],
        1e-9,
    );

    // The two materials must actually differ -- otherwise the identities
    // above would hold for a degenerate all-in-one-bin mapping too.
    let split = by_material[0] / (by_material[0] + by_material[1]);
    assert!(
        (0.05..=0.95).contains(&split),
        "material A holds {split:.3} of the flux; the two bins are not distinguishable"
    );

    // Material is the OUTER dimension and energy the inner one (the CPU's 7D
    // stride order). Summing each material's three energy bins must recover
    // its unbinned total; a transposed stride would mix the two materials'
    // spectra and break this.
    for material_bin in 0..2 {
        let folded: f64 = by_material_energy[material_bin * 3..material_bin * 3 + 3]
            .iter()
            .sum();
        assert_close(
            &format!("material {material_bin} folded over energy"),
            folded,
            by_material[material_bin],
            1e-9,
        );
    }
    // Each material must populate more than one energy bin, so the fold above
    // is a real test of the stride rather than a single non-zero entry.
    for material_bin in 0..2 {
        let occupied = by_material_energy[material_bin * 3..material_bin * 3 + 3]
            .iter()
            .filter(|v| **v > 0.0)
            .count();
        assert!(
            occupied >= 2,
            "material {material_bin} populated only {occupied} energy bin(s); \
             the energy fold is not exercising the stride"
        );
    }

    // A filter listing one material must gate the others out entirely, not
    // fold them into bin 0.
    assert_close(
        "materials=[A] vs bin 0 of materials=[A, B]",
        material_a_only[0],
        by_material[0],
        1e-9,
    );
}

/// The GPU's material-binned flux must agree with the CPU's, which reaches
/// the same bins through `Tally::score_track_length`'s material gate rather
/// than through a per-cell bin map.
#[test]
fn gpu_material_filter_flux_matches_cpu() {
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

    // Independent MC estimates of the same quantity: compare within a band,
    // not exactly.
    for (material_bin, name) in ["A", "B"].iter().enumerate() {
        let (c, g) = (cpu[0][material_bin], gpu[0][material_bin]);
        assert!(c > 0.0, "CPU material {name} flux must be > 0, got {c}");
        let ratio = g / c;
        eprintln!("  material {name}: CPU {c:.5e}  GPU {g:.5e}  ratio {ratio:.4}");
        assert!(
            (0.95..=1.05).contains(&ratio),
            "material {name}: GPU/CPU flux ratio {ratio:.4} outside [0.95, 1.05] \
             (CPU {c:.5e}, GPU {g:.5e})"
        );
    }

    // And the material-binned totals must match the cell-binned totals on the
    // CPU too, so the identity the GPU test asserts is the right one.
    assert_close(
        "CPU material A vs cells 1+3",
        cpu[0][0],
        cpu[1][0] + cpu[1][2],
        1e-9,
    );
}
