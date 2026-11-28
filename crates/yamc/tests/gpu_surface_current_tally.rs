//! Surface current tallies on the GPU, on-hardware (yamc-and-yani#104).
//!
//! `current-above`, `current-below` and `current-net` are scored at each
//! surface crossing, before a vacuum surface kills the particle, with the
//! weight (signed for net) by the sign of the direction's cosine with the
//! surface normal, as the CPU scores them. Two kinds of check:
//!
//! * identities within one GPU launch over the same histories, which a wrong
//!   direction, bin or sign breaks exactly: net is above minus below in every
//!   bin, nothing comes back in through the vacuum surface, and an unfiltered
//!   current in a coupled run is the neutron current plus the photon current;
//! * agreement with the CPU, independent Monte Carlo estimates, within five
//!   combined standard deviations.
//!
//! Needs a real f64 GPU adapter and the fixtures (`scripts/fetch_test_fixtures.py`);
//! self-skips otherwise. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_surface_current_tally -- --nocapture --test-threads=1

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::{Estimator, SurfaceFilter};

const N_PARTICLES: usize = 40_000;
const SEED: u64 = 20261128;

/// Energy edges: three bins from the 14 MeV source down to thermal.
const E_EDGES: [f64; 4] = [1e-3, 1e5, 1e6, 2e7];

/// Shell surfaces, inner to outer; the outermost is the vacuum boundary.
const SURFACE_IDS: [usize; 3] = [1, 2, 3];

fn ready(needs_photons: bool) -> bool {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return false;
    }
    if !std::path::Path::new("tests/Fe56.arrow").exists()
        || (needs_photons && !std::path::Path::new("tests/Fe.arrow").exists())
    {
        eprintln!("skipping -- fixtures missing: run scripts/fetch_test_fixtures.py");
        return false;
    }
    true
}

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

fn iron(density_g_cm3: f64, with_photons: bool) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density_g_cm3),
    )
    .unwrap();
    m.set_temperature("294");
    let nuclides = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    if with_photons {
        let photons = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
        m.read_nuclear_data(&nuclides, Some(&photons)).unwrap();
        m.init_photon_data(&photons).unwrap();
    } else {
        m.read_nuclear_data(&nuclides, None).unwrap();
    }
    Arc::new(m)
}

/// Three nested iron shells (radii 3, 7 and 12 cm), so particles cross the two
/// inner surfaces both ways and leave through the outer, vacuum one.
fn shells(with_photons: bool) -> Geometry {
    let s1 = sphere(1, 3.0, BoundaryType::Transmission);
    let s2 = sphere(2, 7.0, BoundaryType::Transmission);
    let s3 = sphere(3, 12.0, BoundaryType::Vacuum);
    let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s1)));
    let shell = |inner: &Arc<Surface>, outer: &Arc<Surface>| {
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(inner))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(outer))),
        )
    };
    let cells = vec![
        Cell::new(Some(1), core, Some("core".into()), Some(0)),
        Cell::new(Some(2), shell(&s1, &s2), Some("mid".into()), Some(1)),
        Cell::new(Some(3), shell(&s2, &s3), Some("outer".into()), Some(2)),
    ];
    Geometry::new(
        cells,
        vec![
            iron(2.0, with_photons),
            iron(5.0, with_photons),
            iron(7.87, with_photons),
        ],
    )
    .unwrap()
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

fn current(score: &str, particle: Option<ParticleType>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters = vec![
        Filter::Surface(SurfaceFilter::from_ids(SURFACE_IDS.to_vec())),
        Filter::Energy(EnergyFilter::new(E_EDGES.to_vec())),
    ];
    if let Some(p) = particle {
        t.filters
            .push(Filter::ParticleType(ParticleTypeFilter::new(p)));
    }
    t.scores = vec![score.parse::<Score>().unwrap()];
    t.estimator = Estimator::Surface;
    t.name = Some(score.to_string());
    t.reset_accumulation();
    Arc::new(t)
}

fn model(tallies: Vec<Arc<Tally>>, coupled: bool) -> (Model, TransportSettings) {
    let mut m = Model::new(shells(coupled), vec![neutron_source()], tallies);
    m.verbose = Verbose::silent();
    m.tracking_mode = TrackingMode::Surface;
    m.gpu_max_steps_per_particle = 10_000;
    if coupled {
        m.transport_secondary_photons = true;
        m.photon_cutoff_energy = 1000.0;
    }
    let settings = TransportSettings {
        total_particles: Some(N_PARTICLES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (m, settings)
}

/// `[surface][energy]` bin of a current tally's flat mean array.
fn bin(surface: usize, energy: usize) -> usize {
    surface * (E_EDGES.len() - 1) + energy
}

#[test]
fn gpu_surface_currents_obey_their_identities() {
    if !ready(false) {
        return;
    }
    let tallies = vec![
        current("current-above", None),
        current("current-below", None),
        current("current-net", None),
    ];
    let (mut m, settings) = model(tallies.clone(), false);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");
    let [above, below, net] = [0, 1, 2].map(|i| tallies[i].get_mean().to_vec());
    let n_e = E_EDGES.len() - 1;
    assert_eq!(above.len(), SURFACE_IDS.len() * n_e);

    for (s, id) in SURFACE_IDS.iter().enumerate() {
        let out: f64 = (0..n_e).map(|e| above[bin(s, e)]).sum();
        let back: f64 = (0..n_e).map(|e| below[bin(s, e)]).sum();
        eprintln!("surface {id}: outward {out:.5}, inward {back:.5}");
        assert!(out > 0.1, "surface {id}: outward current {out}");
        for e in 0..n_e {
            let i = bin(s, e);
            assert!(
                (net[i] - (above[i] - below[i])).abs() <= 1e-8,
                "surface {id} energy bin {e}: net {} vs above - below {}",
                net[i],
                above[i] - below[i]
            );
        }
    }
    // Nearly every source neutron leaves the thin core at least once (a few
    // are absorbed first; some leave, return and leave again).
    let out_of_core: f64 = (0..n_e).map(|e| above[bin(0, e)]).sum();
    assert!(
        out_of_core >= 0.95,
        "outward current from the core {out_of_core}"
    );
    // Scattering brings neutrons back in through the inner surfaces, but
    // nothing returns through the vacuum boundary.
    assert!((0..n_e).map(|e| below[bin(0, e)]).sum::<f64>() > 0.0);
    for e in 0..n_e {
        assert_eq!(
            below[bin(2, e)],
            0.0,
            "nothing re-enters through the vacuum surface"
        );
    }
}

#[test]
fn gpu_surface_currents_match_cpu() {
    if !ready(false) {
        return;
    }
    let scores = ["current-above", "current-below", "current-net"];
    let run = |gpu: bool| {
        let tallies: Vec<Arc<Tally>> = scores.iter().map(|s| current(s, None)).collect();
        let (mut m, settings) = model(tallies.clone(), false);
        if gpu {
            yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU run");
        } else {
            m.simulate_transport(&settings).expect("CPU run");
        }
        tallies
            .iter()
            .map(|t| (t.get_mean().to_vec(), t.get_std_dev().to_vec()))
            .collect::<Vec<_>>()
    };
    let cpu = run(false);
    let gpu = run(true);
    for (k, score) in scores.iter().enumerate() {
        let ((cm, cs), (gm, gs)) = (&cpu[k], &gpu[k]);
        for i in 0..cm.len() {
            let sigma = (cs[i] * cs[i] + gs[i] * gs[i]).sqrt();
            eprintln!(
                "{score} bin {i}: cpu {:.5} +- {:.5}, gpu {:.5} +- {:.5}",
                cm[i], cs[i], gm[i], gs[i]
            );
            assert!(
                (cm[i] - gm[i]).abs() <= 5.0 * sigma + 1e-12,
                "{score} bin {i}: cpu {} gpu {} (5 sigma = {})",
                cm[i],
                gm[i],
                5.0 * sigma
            );
        }
    }
}

#[test]
fn gpu_coupled_surface_current_counts_both_particles() {
    if !ready(true) {
        return;
    }
    let tallies = vec![
        current("current-above", None),
        current("current-above", Some(ParticleType::Neutron)),
        current("current-above", Some(ParticleType::Photon)),
    ];
    let (mut m, settings) = model(tallies.clone(), true);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("coupled GPU run");
    let [all, neutrons, photons] = [0, 1, 2].map(|i| tallies[i].get_mean().to_vec());
    let leaking_photons: f64 = (0..E_EDGES.len() - 1).map(|e| photons[bin(2, e)]).sum();
    eprintln!("photons leaving per source neutron: {leaking_photons:.5}");
    assert!(
        leaking_photons > 0.0,
        "the photon pass scores the photon current"
    );
    for i in 0..all.len() {
        assert!(
            (all[i] - (neutrons[i] + photons[i])).abs() <= 1e-8,
            "bin {i}: unfiltered {} vs neutron + photon {}",
            all[i],
            neutrons[i] + photons[i]
        );
    }
}

#[test]
fn gpu_photon_surface_current_matches_cpu() {
    if !ready(true) {
        return;
    }
    let run = |gpu: bool| {
        let tally = current("current-net", Some(ParticleType::Photon));
        let (mut m, settings) = model(vec![Arc::clone(&tally)], true);
        if gpu {
            yamc::gpu::run_on_gpu(&mut m, &settings).expect("coupled GPU run");
        } else {
            m.simulate_transport(&settings).expect("coupled CPU run");
        }
        (tally.get_mean().to_vec(), tally.get_std_dev().to_vec())
    };
    let (cm, cs) = run(false);
    let (gm, gs) = run(true);
    for i in 0..cm.len() {
        let sigma = (cs[i] * cs[i] + gs[i] * gs[i]).sqrt();
        eprintln!(
            "photon current-net bin {i}: cpu {:.5} +- {:.5}, gpu {:.5} +- {:.5}",
            cm[i], cs[i], gm[i], gs[i]
        );
        assert!(
            (cm[i] - gm[i]).abs() <= 5.0 * sigma + 1e-12,
            "photon current-net bin {i}: cpu {} gpu {} (5 sigma = {})",
            cm[i],
            gm[i],
            5.0 * sigma
        );
    }
}
