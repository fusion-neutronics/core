//! CPU/GPU parity across the tally SCORE matrix (issue #111, migration step 6).
//!
//! The two backends resolve a score to a cross section by different means: the
//! CPU looks the MT up live against the material's nuclide tables
//! (`Material::macro_xs_by_mt`), while the GPU reads `xs_score_per_mt`, a
//! per-(material, MT, energy) table flattened at translate time. They cannot
//! share one implementation the way the collision samplers do, because the
//! GPU's version has to be resident on the device -- so what keeps them
//! together is this test rather than shared code.
//!
//! It covers aggregates (total, elastic, absorption, capture, (n,2n), H1/He4
//! production, heating, damage) and the discrete charged-particle and inelastic
//! levels (MT 600-602/649, 800-801, 51-53, 91), on a single-nuclide material, a
//! multi-nuclide one, and a URR one -- the three axes where GPU scoring has
//! historically drifted (#212 per-material grids, #307 per-MT scaling, #347
//! multi-nuclide URR).
//!
//! Self-skips without an f64 GPU adapter or the cached data.
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
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
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

const HISTORIES: usize = 200_000;
/// Generous next to the ~0.3% Monte-Carlo spread these tallies show at 200k
/// histories, and still far inside the defects it guards: the per-material
/// grid smearing of #212 was 40%, the multi-nuclide URR mis-weighting of #347
/// was 5%.
const TOL: f64 = 0.03;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn build(
    comp: &[(&str, f64)],
    density: f64,
    src: f64,
    scores: &[Score],
) -> Option<(Model, Vec<Arc<Tally>>, TransportSettings)> {
    if comp
        .iter()
        .any(|(x, _)| !std::path::Path::new(&cache(x)).exists())
    {
        return None;
    }
    let surface = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 20.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let composition: HashMap<String, f64> = comp.iter().map(|(x, f)| (x.to_string(), *f)).collect();
    let data: HashMap<String, String> = comp
        .iter()
        .map(|(x, _)| (x.to_string(), cache(x)))
        .collect();
    let mut m = Material::new(composition, "atom", "g/cm3", Some(density)).ok()?;
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(&data, None).ok()?;
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&surface))),
        Some("c".into()),
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).ok()?;
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![src], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let tallies: Vec<Arc<Tally>> = scores
        .iter()
        .map(|sc| {
            let mut t = Tally::new();
            t.filters.push(Filter::Cell(CellFilter::from_id(1)));
            t.scores = vec![sc.clone()];
            t.initialize_batches(1);
            Arc::new(t)
        })
        .collect();
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(HISTORIES),
        seed: 4242,
        threads: Some(8),
        ..Default::default()
    };
    Some((model, tallies, settings))
}

/// Compare every score in the matrix for one material, returning the failures.
fn compare(label: &str, comp: &[(&str, f64)], density: f64, src: f64) -> Vec<String> {
    let mts: &[(&str, i32)] = &[
        ("total", 1),
        ("elastic", 2),
        ("absorption", 27),
        ("capture", 102),
        ("n2n", 16),
        ("H1prod", 203),
        ("He4prod", 207),
        ("heating", 301),
        ("damage", 444),
        ("np0", 600),
        ("np1", 601),
        ("np2", 602),
        ("npcont", 649),
        ("na0", 800),
        ("na1", 801),
        ("inel1", 51),
        ("inel2", 52),
        ("inel3", 53),
        ("inelcont", 91),
    ];
    let mut scores: Vec<Score> = vec![Score::Flux(FluxScore)];
    let mut names: Vec<String> = vec!["flux".into()];
    for (n, mt) in mts {
        if let Ok(sc) = Score::from_mt_number(*mt) {
            scores.push(sc);
            names.push(format!("{n}(MT{mt})"));
        }
    }
    let Some((mut cpu, ct, settings)) = build(comp, density, src, &scores) else {
        eprintln!("{label}: skipping, cached data absent");
        return Vec::new();
    };
    cpu.simulate_transport(&settings).unwrap();
    let cvals: Vec<f64> = ct.iter().map(|t| t.get_mean().iter().sum()).collect();
    let (mut gpu, gt, settings) = build(comp, density, src, &scores).unwrap();
    yamc::gpu::run_on_gpu(&mut gpu, &settings).expect("GPU run");
    let gvals: Vec<f64> = gt.iter().map(|t| t.get_mean().iter().sum()).collect();

    let mut fails = Vec::new();
    eprintln!("=== {label} ===");
    for i in 0..names.len() {
        let (c, g) = (cvals[i], gvals[i]);
        // A channel that is closed at this energy scores zero on both sides;
        // that agreement is real but carries no information.
        if c == 0.0 && g == 0.0 {
            continue;
        }
        if c == 0.0 || g == 0.0 {
            fails.push(format!(
                "{label} {}: cpu {c:.5e} gpu {g:.5e} (one side zero)",
                names[i]
            ));
            continue;
        }
        let r = g / c;
        eprintln!(
            "  {:>16}: cpu {c:>12.5e}  gpu {g:>12.5e}  gpu/cpu {r:.4}",
            names[i]
        );
        if (r - 1.0).abs() > TOL {
            fails.push(format!("{label} {}: gpu/cpu {r:.4}", names[i]));
        }
    }
    fails
}

#[test]
fn tally_scores_match_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping gpu_tally_score_parity -- no f64 GPU adapter");
        return;
    }
    let mut fails = Vec::new();
    fails.extend(compare("Fe56 14MeV", &[("Fe56", 1.0)], 7.874, 14.06e6));
    fails.extend(compare(
        "natFe 14MeV",
        &[
            ("Fe54", 0.05845),
            ("Fe56", 0.91754),
            ("Fe57", 0.02119),
            ("Fe58", 0.00282),
        ],
        7.874,
        14.06e6,
    ));
    fails.extend(compare("W184 URR", &[("W184", 1.0)], 19.3, 5.0e4));
    assert!(
        fails.is_empty(),
        "tally scores diverge:\n  {}",
        fails.join("\n  ")
    );
}
