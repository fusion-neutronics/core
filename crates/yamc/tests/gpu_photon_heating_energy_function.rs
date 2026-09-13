//! Issues #378 / #382: the GPU must weight the analog photon-heat deposit by the
//! energy function, exactly as the CPU does.
//!
//! The analog photon-heat block in `crates/yamc-gpu/src/photon/transport.rs`
//! (search `ef_in_range_h`) used to apply an energy function as a GATE only,
//! deliberately mirroring the CPU asymmetry so the two backends agreed on the
//! wrong answer. Both now apply it as a weight, and this pins that they still
//! agree -- on the right answer.
//!
//! The check is a scaling identity rather than a cross-backend value comparison:
//! a flat response of `c` must multiply the tally by `c`. Applying the weight
//! consumes no random numbers, so the histories are bit-identical between the
//! weighted and unweighted runs and the relation is deterministic rather than
//! statistical -- it holds to the accumulator's precision (~3e-9 relative here,
//! set by the fixed-point atomic, not to the last bit). That makes this test
//! immune to the collision-vs-track-length convention difference (#356 / #357)
//! and to GPU/CPU transport noise.
//!
//! Which accumulation path this covers: a tallied model with no mesh selects
//! `TallyVarianceMode::PerHistory` (`crates/yamc/src/gpu/dispatch.rs:1992`), so
//! this exercises the per-history branch, which accumulates the raw f64 deposit
//! rather than the fixed-point `bits_h`. That is the path most likely to be
//! missed, because weighting only the fixed-point value would leave it silently
//! unweighted. The kernel applies the weight once, to `heat_scored_h`, which both
//! branches then consume, so the atomic mesh path is weighted by construction.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
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
use yamc_tallies::filter::energy_function::EnergyFunctionFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{HeatingScore, Score};
use yamc_tallies::tally::Tally;

/// Constant response. Deliberately not 1.0, and not a power of two, so a dropped
/// or doubled application cannot coincide with the right answer.
const FLAT: f64 = 7.0;
/// Spans the 1.25 MeV source and everything it downscatters to, so the gate never
/// fires and the whole difference is the weighting.
const GRID: [f64; 4] = [1.0e2, 1.0e4, 1.0e6, 2.0e6];

fn fe_sphere(energy_function: Option<&[f64]>) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
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
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.25e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let n_particles = 20_000;
    let n_batches = 4;

    let mut t = Tally::new();
    t.filters
        .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    if let Some(values) = energy_function {
        t.filters
            .push(Filter::EnergyFunction(EnergyFunctionFilter::new(
                GRID.to_vec(),
                values.to_vec(),
            )));
    }
    t.scores = vec![Score::Heating(HeatingScore)];
    // The analog eV deposit is the collision-estimator arm.
    t.estimator = yamc_tallies::Estimator::Collision;
    t.initialize_batches(n_batches);
    let tally = Arc::new(t);

    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.gpu_max_steps_per_particle = 5_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed: 42,
        ..Default::default()
    };
    (model, tally, settings)
}

fn run_on_gpu(energy_function: Option<&[f64]>) -> f64 {
    let (mut model, tally, settings) = fe_sphere(energy_function);
    yamc::gpu::run_on_gpu(&mut model, &settings).expect("GPU dispatch");
    tally.get_mean().iter().sum::<f64>()
}

#[test]
fn gpu_analog_photon_heating_takes_the_energy_function() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let plain = run_on_gpu(None);
    assert!(
        plain > 0.0,
        "the model must actually deposit photon heat on the GPU for this to test \
         anything, got {plain}"
    );

    let weighted = run_on_gpu(Some(&[FLAT; 4]));
    let expected = plain * FLAT;
    // Not bit-exact, unlike the CPU: the kernel accumulates through a
    // fixed-point atomic, `(value * scale + 0.5) as i64`, so weighting each
    // deposit before quantisation rounds differently from scaling the quantised
    // total afterwards. Measured residual is ~3e-9 relative. The tolerance stays
    // far below any real defect -- a missing weight would show as a factor of 7.
    assert!(
        (weighted - expected).abs() <= 1.0e-6 * expected,
        "a flat energy function of {FLAT} must scale the GPU's analog photon-heat \
         tally by exactly {FLAT}: expected {expected}, got {weighted}. A ratio of 1 \
         means the kernel is gating on the table without weighting by it, which is \
         the #378 defect; any other ratio means it is applied the wrong number of \
         times (#382)"
    );
}
