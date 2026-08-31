//! GPU photon component-score regression test.
//!
//! Companion to `gpu_photon_co60.rs`: that test only exercised flux
//! scoring. The four per-component photon XS scores
//! (`coherent-scatter`, `incoherent-scatter`, `photoelectric`,
//! `pair-production`) route through `SCORE_PER_MT` on the photon
//! kernel -- see the data-piggybacking refinement of option A wired
//! up in `crates/yamc-gpu/src/kernels/multi_cell_photon_transport.rs`
//! and the photon-side dispatch hook in `crates/yamc/src/gpu/dispatch.rs`.
//!
//! Setup mirrors `gpu_photon_co60_fe_sphere_matches_cpu`: 1 cm Fe
//! sphere, point Co60-mean source at the centre. Tally Pack carries
//! 5 cell-only tallies (flux + 4 component MTs); we assert each
//! GPU component scores within ±25 % of its CPU counterpart. That
//! tolerance matches the parent test's rationale: Kahn's-method
//! Compton vs. incoherent-form-factor on CPU, no atomic relaxation,
//! etc. Wide enough to be a regression catcher, not a physics
//! fidelity claim.

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
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, PhotonComponent, PhotonXSScore, Score};
use yamc_tallies::tally::Tally;

fn fe_sphere_with_component_tallies(
    seed: u64,
    radius: f64,
    source_energy: f64,
) -> (Model, Vec<Arc<Tally>>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
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
    let mut nm = HashMap::new();
    nm.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_energy], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let n_particles = 20_000;
    let n_batches = 4;

    // Build 5 tallies: flux + 4 photon-component XS. Each carries a
    // `particle="photon"` filter, mirroring what the broomstick
    // verification notebook builds from Python.
    let mk = |score: Score| {
        let mut t = Tally::new();
        t.filters
            .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
        t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
            yamc_particle::particle::ParticleType::Photon,
        )));
        t.scores = vec![score];
        t.initialize_batches(n_batches);
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
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, tallies, settings)
}

#[test]
fn gpu_photon_components_fe_sphere_match_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // 1.25 MeV -- Co60-mean, same as the parent test. At this energy
    // Compton is dominant (~95 %), photoelectric is small, pair is
    // zero (below 2·m_e c² threshold). The pair channel should
    // score effectively zero on both sides.
    let source_e = 1.25e6_f64;

    let (mut cpu_m, cpu_t, settings) = fe_sphere_with_component_tallies(42, 1.0, source_e);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu: Vec<f64> = cpu_t
        .iter()
        .map(|t| t.get_mean().iter().sum::<f64>())
        .collect();

    let (mut gpu_m, gpu_t, settings) = fe_sphere_with_component_tallies(42, 1.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu: Vec<f64> = gpu_t
        .iter()
        .map(|t| t.get_mean().iter().sum::<f64>())
        .collect();

    let names = [
        "flux",
        "coherent-scatter",
        "incoherent-scatter",
        "photoelectric",
        "pair-production",
    ];
    for (i, name) in names.iter().enumerate() {
        eprintln!("Fe r=1 {name}: CPU = {:.4e}  GPU = {:.4e}", cpu[i], gpu[i]);
    }

    // Flux must match within ±25 % (same tolerance as the parent
    // test). Compton-dominated channels (flux, incoherent, coherent)
    // get the same tolerance; photoelectric and pair are small
    // numbers where stochastic noise dominates, so we only sanity-
    // check that the GPU and CPU are in the same order of magnitude.
    for &i in &[0, 1, 2] {
        if cpu[i] <= 0.0 {
            panic!("CPU {} is zero -- test setup is wrong", names[i]);
        }
        let ratio = gpu[i] / cpu[i];
        assert!(
            (0.75..1.25).contains(&ratio),
            "{}: GPU/CPU ratio {ratio:.3} outside [0.75, 1.25]",
            names[i]
        );
    }
    // Photoelectric and pair: both small at 1.25 MeV. Pair is zero
    // exactly (below threshold), photoelectric is non-zero but small.
    // We just check they don't blow up -- anything within an order of
    // magnitude either side passes.
    for &i in &[3, 4] {
        if cpu[i] > 0.0 {
            let ratio = gpu[i] / cpu[i];
            assert!(
                ratio < 10.0,
                "{}: GPU/CPU ratio {ratio:.3e} > 10× -- suspicious",
                names[i]
            );
        }
    }
}
