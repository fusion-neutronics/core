//! Baseline diagnostic for #88 (GPU multi-isotope epithermal residual on alloys).
//!
//! Prints per-bin GPU/CPU flux ratios across the spectrum for natFe and SS316,
//! and asserts the INTEGRATED flux matches. The dual-grid build (union grid for
//! the collision / nuclide-selection cross sections, coarse grid for the per-MT
//! inelastic buffers) is now applied, so the GPU cross sections are bit-exact to
//! the CPU's union-grid lookups (and to OpenMC). A SEPARATE, pre-existing GPU
//! transport bug still leaves the deep-epithermal per-bin ratios high (SS316
//! 1-300 eV ~1.45x, reproduces single-isotope), so this test asserts only the
//! integrated flux for now; that per-bin residual is tracked by the CPU/GPU
//! transport-parity issues (#101-#107). Self-skips without an f64 GPU adapter or
//! the cached isotope data.
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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn build(comp: &[(&str, f64)], density: f64) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if comp
        .iter()
        .any(|(n, _)| !std::path::Path::new(&cache(n)).exists())
    {
        return None;
    }
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 20.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let composition: HashMap<String, f64> = comp.iter().map(|(n, f)| (n.to_string(), *f)).collect();
    let data: HashMap<String, String> = comp
        .iter()
        .map(|(n, _)| (n.to_string(), cache(n)))
        .collect();
    let mut material = Material::new(composition, "atom", "g/cm3", Some(density)).ok()?;
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(&data, None).ok()?;
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    // Fine log bins spanning thermal -> fast, with dense epithermal coverage.
    let mut edges = vec![1.0e-5];
    let mut e = 1.0e-1;
    while e < 2.0e7 {
        edges.push(e);
        e *= 10f64.powf(0.5); // ~half-decade bins
    }
    edges.push(2.0e7);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.filters.push(Filter::Energy(EnergyFilter::new(edges)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(400_000),
        seed: 77_88,
        threads: Some(1),
        ..Default::default()
    };
    Some((model, tally, settings))
}

fn report(name: &str, comp: &[(&str, f64)], density: f64) {
    let Some((mut cpu, ct, settings)) = build(comp, density) else {
        eprintln!("{name}: skip (data missing)");
        return;
    };
    cpu.simulate_transport(&settings).unwrap();
    let (mut gpu, gt, settings) = build(comp, density).unwrap();
    yamc::gpu::run_on_gpu(&mut gpu, &settings).unwrap();
    let c = ct.get_mean();
    let g = gt.get_mean();
    let edges = if let Filter::Energy(ef) = ct
        .filters
        .iter()
        .find(|f| matches!(f, Filter::Energy(_)))
        .unwrap()
    {
        ef.bins.clone()
    } else {
        vec![]
    };
    let total_c: f64 = c.iter().sum();
    let total_ratio = g.iter().sum::<f64>() / total_c;
    eprintln!("=== {name} (sum CPU={total_c:.4e}, GPU/CPU total={total_ratio:.3}) ===");
    eprintln!("  E_lo(eV)      CPU          GPU        GPU/CPU   %oftot");
    for i in 0..c.len() {
        if c[i] <= 0.0 {
            continue;
        }
        let frac = c[i] / total_c;
        if frac < 1e-4 {
            continue;
        }
        eprintln!(
            "  {:>10.3e}  {:>10.4e}  {:>10.4e}   {:>6.3}   {:>5.2}%",
            edges.get(i).copied().unwrap_or(0.0),
            c[i],
            g[i],
            g[i] / c[i],
            frac * 100.0
        );
    }
    // The integrated flux matches. The dual-grid build fixed the cross-section
    // (finest-vs-union) smearing; the remaining per-bin deep-epithermal residual
    // is a separate GPU transport bug (#101-#107), so we assert only the total.
    assert!(
        (0.92..=1.08).contains(&total_ratio),
        "{name}: integrated GPU/CPU flux {total_ratio:.3} outside [0.92, 1.08]"
    );
}

#[test]
fn alloy_epithermal_baseline() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skip: no gpu");
        return;
    }
    // natural Fe (4 isotopes)
    report(
        "natFe",
        &[
            ("Fe54", 0.05845),
            ("Fe56", 0.91754),
            ("Fe57", 0.02119),
            ("Fe58", 0.00282),
        ],
        7.874,
    );
    // SS316-ish (Fe, Cr, Ni, Mo, Mn isotopic expansion)
    report(
        "SS316",
        &[
            ("Fe54", 0.03812),
            ("Fe56", 0.59850),
            ("Fe57", 0.01382),
            ("Fe58", 0.00184),
            ("Cr50", 0.00712),
            ("Cr52", 0.13724),
            ("Cr53", 0.01556),
            ("Cr54", 0.00387),
            ("Ni58", 0.07721),
            ("Ni60", 0.02974),
            ("Ni61", 0.00129),
            ("Ni62", 0.00412),
            ("Ni64", 0.00105),
            ("Mo92", 0.00373),
            ("Mo94", 0.00233),
            ("Mo95", 0.00401),
            ("Mo96", 0.00420),
            ("Mo97", 0.00241),
            ("Mo98", 0.00609),
            ("Mo100", 0.00243),
            ("Mn55", 0.02000),
        ],
        7.99,
    );
}
