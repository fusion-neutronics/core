//! A launch dispatched to the GPU in slices gives exactly the result of one
//! dispatch.
//!
//! The launch chunk is no longer bounded by run time; instead each launch goes
//! to the GPU in slices short enough for the driver's watchdog, sized from
//! measured wall time. That is only sound if the slicing cannot change the
//! answer, so this runs the same neutron and photon models in one dispatch per
//! launch and in 97-history slices (`YAMC_GPU_DISPATCH_SLICE`, set on a child
//! process) and requires every tally mean, standard deviation and per-history
//! output to agree bit for bit.
//!
//! Self-skips without the Fe56 / Fe fixtures or an f64 GPU. Run it:
//!   cargo test -p yamc --features gpu --release --test gpu_dispatch_slices \
//!       -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
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
use yamc_tallies::tally::Tally;

const SLICE_ENV: &str = "YAMC_GPU_DISPATCH_SLICE";
const SLICE: &str = "97";
const HISTORIES: usize = 3_000;

fn iron(photon: bool) -> Option<Material> {
    let nuclide = yamc_test_cache::nuclide("Fe56")?;
    let element = yamc_test_cache::element("Fe")?;
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .ok()?;
    m.set_material_id(1);
    m.set_temperature("294");
    let photons = HashMap::from([("Fe".to_string(), element)]);
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), nuclide)]),
        photon.then_some(&photons),
    )
    .ok()?;
    if photon {
        m.init_photon_data(&photons).ok()?;
    }
    Some(m)
}

/// A 20 cm iron sphere with a point source of neutrons (14 MeV) or photons
/// (1.25 MeV), and an energy-binned flux tally.
fn model(photon: bool) -> Option<(Model, Arc<Tally>)> {
    let sphere = Arc::new(Surface {
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
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(sphere)),
        None,
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(iron(photon)?)]).ok()?;
    let source = Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![if photon { 1.25e6 } else { 14.06e6 }], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    };
    let source = if photon {
        ParticleSource::Photon(source)
    } else {
        ParticleSource::Neutron(source)
    };
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.filters.push(Filter::Energy(EnergyFilter::new(vec![
        1e-5, 1e3, 1e5, 1e6, 2e7,
    ])));
    tally.scores = vec!["flux".parse().unwrap()];
    tally.reset_accumulation();
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    Some((model, tally))
}

/// A digest of everything the run returned: each tally bin's mean and
/// standard deviation, and each history's final energy, step count and fate.
fn digest(photon: bool) -> Option<u64> {
    let (mut model, tally) = model(photon)?;
    let settings = TransportSettings {
        total_particles: Some(HISTORIES),
        seed: 21,
        ..Default::default()
    };
    let run = yamc::gpu::run_on_gpu(&mut model, &settings).expect("GPU transport");
    let mut h = DefaultHasher::new();
    for v in tally.get_mean().iter().chain(tally.get_std_dev().iter()) {
        v.to_bits().hash(&mut h);
    }
    for e in &run.final_energies {
        e.to_bits().hash(&mut h);
    }
    run.n_steps.hash(&mut h);
    run.alive.hash(&mut h);
    Some(h.finish())
}

#[test]
fn slicing_a_launch_does_not_change_its_result() {
    if std::env::var(SLICE_ENV).as_deref() == Ok(SLICE) {
        // The child: report the sliced digests.
        let (Some(n), Some(p)) = (digest(false), digest(true)) else {
            panic!("the parent checked the fixtures");
        };
        println!("DIGESTS {n} {p}");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() || model(true).is_none() {
        eprintln!("skipping: no f64 GPU or no Fe56 / Fe fixtures");
        return;
    }
    let whole = (digest(false).unwrap(), digest(true).unwrap());

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "slicing_a_launch_does_not_change_its_result",
            "--exact",
            "--nocapture",
        ])
        .env(SLICE_ENV, SLICE)
        .output()
        .expect("re-run the test binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the sliced child failed:\n{stdout}"
    );
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("DIGESTS "))
        .expect("the child reports its digests");
    let mut parts = line.split_whitespace().map(|v| v.parse::<u64>().unwrap());
    let sliced = (parts.next().unwrap(), parts.next().unwrap());
    eprintln!("one dispatch {whole:?}, {SLICE}-history slices {sliced:?}");
    assert_eq!(
        whole.0, sliced.0,
        "the neutron result depends on how its launch was sliced"
    );
    assert_eq!(
        whole.1, sliced.1,
        "the photon result depends on how its launch was sliced"
    );
}
