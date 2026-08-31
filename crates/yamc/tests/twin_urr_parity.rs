//! The GPU kernel's CPU twin must reproduce the production CPU's URR
//! self-shielding (issue #342).
//!
//! The twin used to discard every URR buffer it was handed
//! (`transport/cpu.rs`: "CPU mirror doesn't currently sample URR"), so a URR
//! material was invisible to the twin-vs-kernel equivalence tests AND to the
//! issue-#111 matched-stream harness. That is why the kernel could redraw a
//! neutron's probability-table band on every transport step, over-predicting
//! flux ~30% in any multi-material model, without a single test noticing.
//!
//! The twin needs no GPU, so unlike `gpu_urr_band_hold` this runs in CI.
//!
//! Two things are asserted, on W184 (URR over 1e4 .. 1e5 eV) with a source
//! inside that band:
//!
//! 1. **URR is sampled at all.** Against a URR-blind twin the flux is far off,
//!    so agreement with the production CPU is itself the evidence.
//! 2. **The band is held per energy, not per step.** A single material has no
//!    interior crossings and so cannot tell the two rules apart; three
//!    materials can, and the same physical problem must give the same answer
//!    either way.
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::gpu_twin::run_twin;
use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_gpu::common::tallies::TalliesPack;
use yamc_gpu::neutron::transport::PendDrain;
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

/// Single-isotope URR: the #342 case.
const ISOTOPE: &str = "W184";
/// Natural tungsten: four URR isotopes sharing ONE material, so the struck
/// nuclide is selected among them at every collision. This is the #347 case,
/// which a subdivided single isotope cannot reach.
const NAT_W: &[(&str, f64)] = &[
    ("W182", 0.2650),
    ("W183", 0.1431),
    ("W184", 0.3064),
    ("W186", 0.2843),
];
const DENSITY: f64 = 19.3;
/// Inside W184's URR band, so every collision samples the probability table.
const SOURCE_ENERGY: f64 = 5.0e4;
const RADIUS: f64 = 30.0;
const HISTORIES: usize = 50_000;
const MAX_STEPS: u32 = 20_000;
const SEED: u64 = 4242;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

/// One solid W184 sphere cut into `n_mats` concentric shells, each its own
/// `Material`. Flux is tallied in the outermost shell.
fn build(comp: &[(&str, f64)], n_mats: usize) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if comp
        .iter()
        .any(|(n, _)| !std::path::Path::new(&cache(n)).exists())
    {
        return None;
    }
    let mk_mat = |id: u32| -> Option<Arc<Material>> {
        let composition: HashMap<String, f64> =
            comp.iter().map(|(n, f)| (n.to_string(), *f)).collect();
        let data: HashMap<String, String> = comp
            .iter()
            .map(|(n, _)| (n.to_string(), cache(n)))
            .collect();
        let mut m = Material::new(composition, "atom", "g/cm3", Some(DENSITY)).ok()?;
        m.set_material_id(id);
        m.set_temperature("294");
        m.read_nuclear_data(&data, None).ok()?;
        Some(Arc::new(m))
    };
    let mats: Vec<Arc<Material>> = (0..n_mats).map(|i| mk_mat(i as u32 + 1).unwrap()).collect();

    let mut cells = Vec::new();
    let mut prev: Option<Arc<Surface>> = None;
    for i in 0..n_mats {
        let surface = Arc::new(Surface {
            surface_id: Some(i + 1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: RADIUS * (i as f64 + 1.0) / n_mats as f64,
            },
            boundary: if i + 1 == n_mats {
                BoundaryType::Vacuum
            } else {
                BoundaryType::Transmission
            },
            name: None,
        });
        let inside = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&surface)));
        let region = match prev {
            None => inside,
            Some(ref p) => Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(p)))
                .intersection(&inside),
        };
        cells.push(Cell::new(
            Some(i as u32 + 1),
            region,
            Some(format!("shell{i}")),
            Some(i as u32),
        ));
        prev = Some(surface);
    }
    let geometry = Geometry::new(cells, mats).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_ENERGY], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(n_mats as u32)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);

    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(HISTORIES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    Some((model, tally, settings))
}

/// Outer-shell flux from the production CPU and from the twin.
fn cpu_and_twin_flux(comp: &[(&str, f64)], n_mats: usize) -> Option<(f64, f64)> {
    let (mut cpu_model, tally, settings) = build(comp, n_mats)?;
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu: f64 = tally.get_mean().iter().sum();

    let (twin_model, _t, settings) = build(comp, n_mats).unwrap();
    let inputs = yamc::gpu::translate::translate_for_gpu(&twin_model, HISTORIES, settings.seed)
        .expect("translate");
    // One wide energy bin over every cell; tally 0 is flux, laid out
    // cell-major, so the outermost shell is the last cell's entry.
    let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
    let (result, _trace) = run_twin(&inputs, &pack, MAX_STEPS, 400.0, false, PendDrain::Fifo);
    let twin = result.tally_outputs[0][n_mats - 1] / HISTORIES as f64;
    Some((cpu, twin))
}

#[test]
fn twin_reproduces_cpu_urr_self_shielding() {
    let Some((cpu_one, twin_one)) = cpu_and_twin_flux(&[(ISOTOPE, 1.0)], 1) else {
        eprintln!("skipping twin_urr_parity -- no cached {ISOTOPE} data");
        return;
    };
    let (cpu_three, twin_three) = cpu_and_twin_flux(&[(ISOTOPE, 1.0)], 3).unwrap();
    let r_one = twin_one / cpu_one;
    let r_three = twin_three / cpu_three;
    eprintln!(
        "twin/CPU outer-shell flux: 1 material {r_one:.4} (cpu {cpu_one:.5e}), \
         3 materials {r_three:.4} (cpu {cpu_three:.5e})"
    );

    // A twin that ignores URR entirely, or one that redraws the band per step,
    // lands tens of percent away; the bound only has to exclude that, not pin
    // Monte-Carlo noise at 20k histories.
    assert!(
        (0.94..=1.06).contains(&r_one),
        "single-material twin/CPU flux {r_one:.4} outside [0.94, 1.06] -- \
         is the twin sampling URR at all?"
    );
    assert!(
        (0.94..=1.06).contains(&r_three),
        "three-material twin/CPU flux {r_three:.4} outside [0.94, 1.06] -- \
         is the twin holding the URR band per energy (#342)?"
    );
}

/// Natural tungsten in ONE material (issue #347). A nuclide's share of the
/// collision density must follow the cross section that actually governed the
/// flight, so an in-range URR nuclide is selected -- and its reaction split --
/// on its PERTURBED total, not the table average. Selecting on smooth totals
/// read +5.7% against the production CPU.
#[test]
fn twin_reproduces_cpu_urr_nuclide_selection() {
    let Some((cpu, twin)) = cpu_and_twin_flux(NAT_W, 1) else {
        eprintln!("skipping twin_urr_parity (natural W) -- cached data absent");
        return;
    };
    let ratio = twin / cpu;
    eprintln!("natural W twin/CPU flux: {ratio:.4} (cpu {cpu:.5e})");
    assert!(
        (0.96..=1.04).contains(&ratio),
        "natural W twin/CPU flux {ratio:.4} outside [0.96, 1.04] -- is the \
         struck nuclide being selected or split on SMOOTH per-nuclide values \
         while the flight used the URR-perturbed total (#347)?"
    );
}
