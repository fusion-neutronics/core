//! D1S decay photons drawn from a continuum on the GPU (issue #163).
//!
//! The Fe56 sphere of `gpu_d1s_decay_photon.rs`, with Mn56's photon lines
//! swapped for two continua: one linear-linear and one histogram, on grids
//! chosen so their shapes put different shares in each energy bin. The CPU
//! draws them with `sample_continuum_energy`; the GPU kernel must draw the
//! same spectrum, which before issue #163 it could not (the dispatch refused
//! every continuum channel). A photon flux tally binned in energy by parent
//! compares the two per bin, so a kernel that dropped a continuum, read its
//! points as lines or inverted its integral wrongly moves photons between
//! bins and fails the band.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
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

/// Photon energy bin edges [eV]. The histogram continuum sits in the lowest
/// three bins, the linear-linear one spreads over all of them.
const EDGES: [f64; 6] = [1.0e3, 3.0e5, 6.0e5, 1.0e6, 2.0e6, 1.0e7];

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

/// The fixture chain with Mn56's photon lines replaced by two continua,
/// written to a scratch directory whose path is returned.
fn continuum_chain() -> std::path::PathBuf {
    let src = std::path::PathBuf::from(chain_path());
    let (mut chain, _) = yani::parse_chain_parts(
        &src.join("decay"),
        Some(&src.join("reactions")),
        Some(&src.join("fission_yields")),
        None,
    )
    .expect("parse the fixture chain");
    let mn56 = chain.get_mut("Mn56").expect("the fixture chain has Mn56");
    mn56.sources.retain(|s| s.particle != "photon");
    // Densities are emission rates per atom per second per eV, of the order
    // of Mn56's decay constant (7.5e-5 /s) spread over an MeV.
    mn56.sources.push(yani::DecaySource {
        particle: "photon".to_string(),
        distribution: yani::DecaySourceDistribution::Tabular {
            energies: vec![1.0e5, 1.0e6, 3.0e6],
            intensities: vec![0.0, 4.0e-11, 1.0e-11],
            interpolation: Some(yani::Interpolation::LinearLinear),
        },
    });
    mn56.sources.push(yani::DecaySource {
        particle: "photon".to_string(),
        distribution: yani::DecaySourceDistribution::Tabular {
            energies: vec![2.0e5, 5.0e5, 8.0e5],
            intensities: vec![5.0e-11, 1.0e-10, 0.0],
            interpolation: Some(yani::Interpolation::Histogram),
        },
    });
    let dir = std::env::temp_dir().join(format!("yamc-d1s-continuum-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani::export_chain_parts(&chain, &dir, Some("endf-b8.1")).expect("export the chain");
    dir
}

fn fe_sphere_model(seed: u64, n_particles: usize) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
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
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut photon_tally = Tally::new();
    photon_tally.filters = vec![
        Filter::Cell(CellFilter::from_id(cell_id)),
        Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon)),
        Filter::Energy(EnergyFilter::new(EDGES.to_vec())),
        Filter::ParentNuclide(ParentNuclideFilter::new(vec!["Mn56".to_string()])),
    ];
    photon_tally.scores = vec![Score::Flux(FluxScore)];
    photon_tally.initialize_batches(1);
    let photon_tally = Arc::new(photon_tally);

    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&photon_tally)]);
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    model.use_decay_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(n_particles),
        seed,
        ..Default::default()
    };
    (model, photon_tally, settings)
}

#[test]
fn gpu_d1s_draws_a_decay_photon_continuum_as_the_cpu_does() {
    if !data_present() {
        eprintln!("skipping -- test data / chain not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let chain = continuum_chain();
    {
        let path = chain.to_string_lossy().into_owned();
        let mut cfg = yamc_nuclide::config::Config::global();
        cfg.transmutation_decay_data = Some(path.clone());
        cfg.transmutation_reactions = yamc_nuclide::config::SubsectionSource::Set(path.clone());
        cfg.transmutation_fission_yields = yamc_nuclide::config::SubsectionSource::Set(path);
    }

    let n_particles = 20_000;
    let (mut cpu_m, cpu_tally, settings) = fe_sphere_model(11, n_particles);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU D1S transport");
    let (mut gpu_m, gpu_tally, settings) = fe_sphere_model(11, n_particles);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU D1S draws a continuum");
    let _ = std::fs::remove_dir_all(&chain);

    let cpu = cpu_tally.get_mean();
    let gpu = gpu_tally.get_mean();
    println!("Mn56 continuum photon flux per bin: CPU {cpu:?}  GPU {gpu:?}");
    let cpu_total: f64 = cpu.iter().sum();
    let gpu_total: f64 = gpu.iter().sum();
    assert!(cpu_total > 0.0 && gpu_total > 0.0, "no continuum photons");
    // The same band as the line test: the GPU photon kernel is the same
    // approximate one, so totals agree statistically, not bit for bit.
    let ratio = gpu_total / cpu_total;
    assert!(
        (0.80..=1.25).contains(&ratio),
        "GPU/CPU continuum photon flux {ratio:.3} outside [0.80, 1.25]"
    );
    // Each bin holds the same share of the total on both sides. The shares
    // cancel the kernel's approximation in the totals, so the band is tighter:
    // inverting the linear-linear continuum as a histogram moves half of the
    // top bin's share into the bins below and fails it.
    for (bin, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let share = (g / gpu_total) / (c / cpu_total);
        assert!(
            (0.90..=1.10).contains(&share),
            "bin {bin} [{:.0e}, {:.0e}] eV: GPU/CPU share {share:.3} outside [0.90, 1.10]",
            EDGES[bin],
            EDGES[bin + 1]
        );
    }
}
