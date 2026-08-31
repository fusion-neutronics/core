//! D1S (Direct-1-Step) decay-photon GPU dispatch WITH a void cell.
//!
//! The D1S decay-photon GPU path builds per-material decay photon-production
//! tables. A void (material-less) cell maps to a synthetic "void material" slot
//! the neutron translation appends, so the decay tables must include a matching
//! non-emitting void slot or the void cell's `cell_to_material` index reads past
//! the decay tables (out of bounds / silently aliased). The plain
//! `gpu_d1s_decay_photon` test has no void cell and the `gpu_void_cells` test
//! has no D1S; only the two TOGETHER exercise that slot.
//!
//! Geometry: a +z 14 MeV beam through a Fe56 / VOID / Fe56 slab stack inside a
//! finite vacuum box. Fe56 (n,p) -> Mn56 is an activation product whose decay
//! emits gamma lines; the neutron kernel emits one decay photon per collision in
//! the material slabs (tagged with Mn56) into the device bank, and the photon
//! sub-pass transports it. The middle slab is a void: it produces no decay
//! photons but the source neutron streams through it.
//!
//! Claims:
//!   (a) the run does not panic / index out of bounds (the void decay slot keeps
//!       the per-material tables in-bounds for the void index);
//!   (b) the GPU per-parent photon flux agrees with the CPU within the same
//!       model-fidelity band the void / coupled / D1S tests already use;
//!   (c) the parent-nuclide-binned photon flux is non-zero (Mn56 decay photons
//!       are produced in the material slabs, transported, and binned).

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, RegionExpr, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
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
use yamc_tallies::filter::parent_nuclide::ParentNuclideFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn chain_path() -> String {
    format!(
        "{}/tests/transmutation-endf-b8.1-sfr.arrow",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
        && std::path::Path::new(&chain_path()).exists()
}

fn plane(id: usize, a: f64, b: f64, c: f64, d: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Plane { a, b, c, d },
        boundary,
        name: None,
    })
}

/// `AND` a list of (surface, above?) halfspaces into one intersection region.
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
        Some(7.874),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    m.read_nuclear_data(&nuclide_map, Some(&photon_paths))
        .unwrap();
    m.init_photon_data(&photon_paths).unwrap();
    Arc::new(m)
}

/// Three stacked z-slabs in a finite vacuum box: bottom (-15 < z < -5) Fe56,
/// middle (-5 < z < 5) VOID, top (5 < z < 15) Fe56, all within x,y in [-2, 2].
/// Cell ids 1 (bottom) / 2 (middle void) / 3 (top). Downstream-first storage
/// (`[top, middle, bottom]`) for the +z AABB tie-break (see `gpu_void_cells`).
fn build_void_d1s_geometry() -> Geometry {
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

    let cell_bottom = Cell::new(Some(1), bottom, Some("fe_lo".into()), Some(0));
    // Middle slab is a void cell (material_idx = None).
    let cell_mid = Cell::new(Some(2), middle, Some("void".into()), None);
    let cell_top = Cell::new(Some(3), top, Some("fe_hi".into()), Some(0));

    Geometry::new(vec![cell_top, cell_mid, cell_bottom], vec![fe56_material()]).unwrap()
}

/// Monodirectional +z 14 MeV beam from z = -10 (inside the bottom Fe56 slab),
/// crossing into the void at z = -5 and the top slab at z = 5.
fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, -10.0])),
        angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// D1S model over the Fe56/void/Fe56 stack: a photon flux tally binned by
/// `parent_nuclides=["Mn56"]` with a coarse energy filter, over the whole box
/// (cells 1 + 3 are the only emitters; the void scores zero photon flux).
fn d1s_void_model(
    seed: u64,
    n_particles: usize,
    n_batches: usize,
) -> (Model, Arc<Tally>, TransportSettings) {
    let geometry = build_void_d1s_geometry();

    let mut photon_tally = Tally::new();
    // Span all three cells (incl. the void) so the void cell is part of the
    // scored geometry; only the Fe56 slabs (1, 3) actually emit decay photons.
    photon_tally
        .filters
        .push(Filter::Cell(CellFilter::from_ids(vec![1, 2, 3])));
    photon_tally
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Photon,
        )));
    photon_tally
        .filters
        .push(Filter::Energy(EnergyFilter::new(vec![
            1.0e3, 1.0e6, 2.0e6, 1.0e7,
        ])));
    photon_tally
        .filters
        .push(Filter::ParentNuclide(ParentNuclideFilter::new(vec![
            "Mn56".to_string(),
        ])));
    photon_tally.scores = vec![Score::Flux(FluxScore)];
    photon_tally.initialize_batches(n_batches);
    let photon_tally = Arc::new(photon_tally);

    let mut model = Model::new(
        geometry,
        vec![neutron_source()],
        vec![Arc::clone(&photon_tally)],
    );
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.use_decay_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, photon_tally, settings)
}

/// D1S decay-photon GPU run that ALSO contains a void cell: the void decay slot
/// keeps the per-material tables in-bounds, the run does not panic, and the GPU
/// per-parent photon flux matches the CPU within the model-fidelity band.
#[test]
fn gpu_d1s_decay_photon_void_matches_cpu() {
    if !data_present() {
        eprintln!("skipping gpu_d1s_decay_photon_void_matches_cpu -- test data / chain not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // D1S requires the transmutation chain via the global config.
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

    let seed = 7;
    let n_particles = 4_000;
    let n_batches = 1;

    // --- CPU reference ---
    let (mut cpu_m, cpu_photon, settings) = d1s_void_model(seed, n_particles, n_batches);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU D1S+void transport");
    let cpu_photon_flux = cpu_photon.get_mean();

    // --- GPU D1S+void (must not panic / OOB on the void decay slot) ---
    let (mut gpu_m, gpu_photon, settings) = d1s_void_model(seed, n_particles, n_batches);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU D1S+void dispatch");
    let gpu_photon_flux = gpu_photon.get_mean();

    let cpu_p: f64 = cpu_photon_flux.iter().sum();
    let gpu_p: f64 = gpu_photon_flux.iter().sum();
    println!(
        "D1S+void Mn56 photon flux: CPU {cpu_p:.6e}  GPU {gpu_p:.6e}  ratio {:.3}",
        gpu_p / cpu_p
    );
    println!(
        "per-(parent,energy) photon bins: CPU {:?}  GPU {:?}",
        cpu_photon_flux, gpu_photon_flux
    );

    // (c) Decay photons were produced (in the material slabs), transported, binned.
    assert!(
        gpu_p > 0.0,
        "GPU produced zero parent-nuclide-binned decay photon flux"
    );
    assert!(
        cpu_p > 0.0,
        "CPU produced zero decay photon flux (chain/data issue)"
    );

    // (b) GPU per-parent photon flux agrees with CPU within the model-fidelity
    // band (same approximate GPU photon kernel as the coupled / D1S tests).
    let ratio = gpu_p / cpu_p;
    assert!(
        (0.80..=1.25).contains(&ratio),
        "D1S+void photon flux GPU/CPU ratio {ratio:.3} outside [0.80, 1.25] (CPU {cpu_p:.6e}, GPU {gpu_p:.6e})"
    );
}
