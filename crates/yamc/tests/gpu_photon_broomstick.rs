//! GPU-vs-CPU photon broomstick regression test -- the anisotropic-geometry,
//! energy-binned guard that the sphere-based GPU tests structurally cannot
//! provide.
//!
//! Three #415-family GPU bugs passed every sphere-based test and were only
//! visible in this configuration:
//! - empty TTB tables (no bremsstrahlung source -> deep-tail bins zero),
//! - energy-bin edge convention (discrete 1 MeV line on a group boundary
//!   landed one bin high: top-bin ratio 0.011),
//! - azimuth collapse from broken driver f64 sin/cos (every scattered photon
//!   in the x-z plane: invisible under azimuthal symmetry, but scattered
//!   flux in a thin cylinder fell to 0.66-0.90 of CPU).
//!
//! A long thin cylinder breaks the azimuthal/spherical symmetry, and the
//! log-spaced spectrum with the source line exactly on a bin edge pins the
//! binning convention. Per-bin GPU/CPU bands here would have failed on all
//! three bugs.
//!
//! Self-skips without an f64 adapter (CI stays green); run on a GPU box
//! single-threaded via `cargo test-gpu`.

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

const SOURCE_E: f64 = 1.0e6;

/// Log-spaced bin edges from 1 keV to the source energy, so the discrete
/// source line sits EXACTLY on the top interior edge boundary's bin end --
/// pinning the (lo, hi] edge convention -- plus one bin above.
fn bins() -> Vec<f64> {
    let (lo, n) = (1.0e3_f64, 20usize);
    let ln_lo = lo.ln();
    let ln_hi = SOURCE_E.ln();
    let mut edges: Vec<f64> = (0..=n)
        .map(|i| (ln_lo + (ln_hi - ln_lo) * (i as f64) / (n as f64)).exp())
        .collect();
    // Force the top interior edge to be exactly the source energy, then add
    // a final bin above it: an exactly-at-edge line must score into the bin
    // BELOW the edge (CPU EnergyFilter convention).
    let last = edges.len() - 1;
    edges[last] = SOURCE_E;
    edges.push(SOURCE_E * 2.0);
    edges
}

fn build() -> (Model, Arc<Tally>, TransportSettings) {
    // Thin long Fe broomstick: radius 1 cm, half-height 50 cm.
    let cylinder = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Cylinder {
            origin: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            radius: 1.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let z_bot = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Plane {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: -50.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let z_top = Arc::new(Surface {
        surface_id: Some(3),
        kind: SurfaceKind::Plane {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: 50.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(cylinder))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(z_bot)))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(z_top)));

    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.0),
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

    let cell = Cell::new(Some(1), region, Some("broomstick".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.filters.push(Filter::Energy(EnergyFilter::new(bins())));
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(8);
    let t = Arc::new(t);

    let mut model = Model::new(geometry, vec![source], vec![t.clone()]);
    model.max_steps_per_particle = 5_000;
    let settings = TransportSettings {
        total_particles: Some(125_000 * 8),
        seed: 42,
        ..Default::default()
    };
    (model, t, settings)
}

#[test]
fn gpu_photon_broomstick_spectrum_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let (mut cpu_m, cpu_t, settings) = build();
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU run");
    let cpu = cpu_t.get_mean();

    let (mut gpu_m, gpu_t, settings) = build();
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    let edges = bins();
    let n_bins = cpu.len();
    let cpu_max = cpu.iter().cloned().fold(0.0_f64, f64::max);
    assert!(cpu_max > 0.0, "CPU spectrum empty -- test setup wrong");

    // The source line sits exactly on the top interior edge: it must score
    // into the bin BELOW that edge ((lo, hi] convention) on BOTH sides, and
    // the bin ABOVE the source energy must be empty.
    let line_bin = n_bins - 2;
    let above_bin = n_bins - 1;
    assert!(
        cpu[line_bin] == cpu_max && cpu[above_bin] == 0.0,
        "CPU line-bin placement unexpected (convention drift?)"
    );
    assert!(
        gpu[above_bin] == 0.0,
        "GPU put flux ABOVE the source energy: bin-edge convention regressed \
         (exactly-at-edge line must go to the lower bin)"
    );

    let mut rows = vec![
        "| bin | E_lo | E_hi | CPU | GPU | ratio | check |".to_string(),
        "|---|---|---|---|---|---|---|".to_string(),
    ];
    let mut bad: Vec<String> = Vec::new();
    for i in 0..n_bins {
        let (c, g) = (cpu[i], gpu[i]);
        // Only enforce bins that carry meaningful statistics: anything at
        // least 1e-4 of the spectrum max (the azimuth-collapse deficit was
        // 30%+ in bins ~1e-2..1e-3 of max, far above this floor).
        let meaningful = c >= 1e-4 * cpu_max;
        let (ratio_str, ok) = if meaningful {
            let r = g / c;
            // [0.75, 1.30]: tight enough to fail on all three #415-family
            // bugs (deep-tail 0.0, line-bin 0.011, scattered 0.66-0.73);
            // loose enough for MC noise + the known few-% GPU offset at
            // these statistics.
            let ok = (0.75..=1.30).contains(&r);
            if !ok {
                bad.push(format!(
                    "bin {i} [{:.3e},{:.3e}]: GPU/CPU {r:.3} out of [0.75,1.30] (CPU {c:.3e}, GPU {g:.3e})",
                    edges[i], edges[i + 1]
                ));
            }
            (format!("{r:.3}"), ok)
        } else {
            ("n/a".to_string(), true)
        };
        rows.push(format!(
            "| {i} | {:.3e} | {:.3e} | {c:.3e} | {g:.3e} | {ratio_str} | {} |",
            edges[i],
            edges[i + 1],
            if ok { "ok" } else { "BAD" }
        ));
    }
    eprintln!("\nGPU photon broomstick spectrum (anisotropic-geometry guard):");
    for r in &rows {
        eprintln!("{r}");
    }
    assert!(
        bad.is_empty(),
        "GPU broomstick spectrum disagrees with CPU:\n  {}",
        bad.join("\n  ")
    );
}
