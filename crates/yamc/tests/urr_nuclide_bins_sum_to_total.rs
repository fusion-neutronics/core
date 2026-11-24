//! Inside a probability-table band a nuclide's reaction-rate bin scores that
//! nuclide's cross section at the band the flight sampled, the way the
//! material bin does, so the nuclide bins add up to the material bin. Before,
//! they scored the smooth cross section and Fe58's in-band elastic and capture
//! bins missed the band's self-shielding.
//!
//! Each run has one tally per bin on the same histories, so the sums agree to
//! rounding rather than statistically: the tallies do not touch the stream.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{NuclideBin, Tally};
use yamc_tallies::Estimator;

/// Inside Fe58's band [350 keV, 3 MeV]; Be9 has no tables.
const IN_BAND: f64 = 1.0e6;
const BINS: [Option<&str>; 3] = [None, Some("Fe58"), Some("Be9")];

fn build(paths: &HashMap<String, String>, estimator: Estimator) -> (Model, Vec<Arc<Tally>>) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Fe58".to_string(), 0.5), ("Be9".to_string(), 0.5)]),
        "atom",
        "g/cm3",
        Some(4.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(paths, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("m".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![IN_BAND], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let tallies: Vec<Arc<Tally>> = BINS
        .iter()
        .map(|bin| {
            let mut t = Tally::new();
            // The GPU scores per cell, so its tallies carry a cell filter.
            t.filters.push(Filter::Cell(CellFilter::from_ids(vec![1])));
            t.scores = vec!["2".parse().unwrap(), "102".parse().unwrap()];
            t.nuclides = vec![bin.map_or(NuclideBin::Total, |n| NuclideBin::Specific(n.into()))];
            t.estimator = estimator;
            t.reset_accumulation();
            Arc::new(t)
        })
        .collect();
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 10_000;
    (model, tallies)
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(20_000),
        seed: 20261117,
        threads: Some(0),
        ..Default::default()
    }
}

/// Per tally, the score means `[elastic, capture]`.
fn means(tallies: &[Arc<Tally>]) -> Vec<Vec<f64>> {
    tallies.iter().map(|t| t.get_mean()).collect()
}

fn assert_bins_sum(label: &str, m: &[Vec<f64>]) {
    for (s, name) in ["elastic", "capture"].iter().enumerate() {
        let total = m[0][s];
        let sum = m[1][s] + m[2][s];
        eprintln!("{label} {name}: material {total:.8e}, Fe58 + Be9 {sum:.8e}");
        assert!(total > 0.0, "{label} {name}: nothing scored");
        assert!(
            (sum - total).abs() <= 1e-6 * total,
            "{label} {name}: nuclide bins {sum:.8e} do not add up to the material {total:.8e}"
        );
    }
}

#[test]
fn nuclide_bins_add_up_to_the_material_bin_in_band() {
    let Some(paths) = ["Fe58", "Be9"]
        .into_iter()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.to_string(), p)))
        .collect::<Option<HashMap<_, _>>>()
    else {
        eprintln!("skipping: Fe58 or Be9 is not in the test cache");
        return;
    };
    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        let (mut model, tallies) = build(&paths, estimator);
        model.simulate_transport(&settings()).expect("CPU run");
        assert_bins_sum(&format!("CPU {estimator:?}"), &means(&tallies));

        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if yamc_gpu::GpuContext::new().is_ok() {
            let (mut model, tallies) = build(&paths, estimator);
            yamc::gpu::run_on_gpu(&mut model, &settings()).expect("GPU run");
            assert_bins_sum(&format!("GPU {estimator:?}"), &means(&tallies));
        }
    }
}
