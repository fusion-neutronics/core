//! The production CPU and the GPU kernel's CPU twin must transport a material at
//! the temperature its label says (issue #478).
//!
//! `Material::temperature_k` was set once in `Material::new` and never updated by
//! `set_temperature`, so the production CPU handed 294 K to
//! `sample_free_gas_elastic` for every material. The GPU side parsed the label
//! itself (`yamc_gpu::neutron::xs::extract::parse_temperature_k`) and got it
//! right, so the two backends ran different physics on any material away from
//! 294 K. Nothing caught it because every parity test in the tree runs at 294 K,
//! where the stale value happens to be correct.
//!
//! The twin needs no GPU, so this runs in CI.
//!
//! Two assertions, and the second is what gives the first teeth:
//!
//! 1. **Parity at 900 K.** Production CPU and twin must agree, as they already
//!    do at 294 K.
//! 2. **The tally is temperature-sensitive.** A hot sphere and a cold one must
//!    give measurably different flux. Without this, assertion 1 would pass just
//!    as happily if the temperature reached neither backend.
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

/// H1 is the case that matters: `awr < 1`, so the free-gas threshold never
/// applies and target motion is sampled at every energy. It also publishes both
/// temperatures this test needs.
const ISOTOPE: &str = "H1";
/// Hydrogenous and dense enough for a source neutron to thermalise well inside
/// the sphere, so the flux reflects the local Maxwellian rather than the source.
const DENSITY: f64 = 0.1;
/// A few kT at either temperature (kT is 0.0253 eV at 294 K and 0.0776 eV at
/// 900 K), so slowing down lands in the range where target motion dominates.
const SOURCE_ENERGY: f64 = 1.0;
const RADIUS: f64 = 30.0;
const HISTORIES: usize = 50_000;
const MAX_STEPS: u32 = 20_000;
const SEED: u64 = 478;

const COLD: &str = "294";
const HOT: &str = "900";

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

/// One H1 sphere at `temperature`, with flux tallied over the whole of it.
fn build(temperature: &str) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if !std::path::Path::new(&cache(ISOTOPE)).exists() {
        return None;
    }
    let composition = HashMap::from([(ISOTOPE.to_string(), 1.0)]);
    let data = HashMap::from([(ISOTOPE.to_string(), cache(ISOTOPE))]);
    let mut material = Material::new(composition, "atom", "g/cm3", Some(DENSITY)).ok()?;
    material.set_material_id(1);
    material.set_temperature(temperature);
    material.read_nuclear_data(&data, None).ok()?;
    let material = Arc::new(material);

    let surface = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(surface));
    let cells = vec![Cell::new(Some(1), region, Some("sphere".into()), Some(0))];
    let geometry = Geometry::new(cells, vec![material]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_ENERGY], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
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

/// Sphere flux from the production CPU and from the twin, at one temperature.
fn cpu_and_twin_flux(temperature: &str) -> Option<(f64, f64)> {
    let (mut cpu_model, tally, settings) = build(temperature)?;
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu: f64 = tally.get_mean().iter().sum();

    let (twin_model, _t, settings) = build(temperature).unwrap();
    let inputs = yamc::gpu::translate::translate_for_gpu(&twin_model, HISTORIES, settings.seed)
        .expect("translate");
    let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
    let (result, _trace) = run_twin(&inputs, &pack, MAX_STEPS, 400.0, false, PendDrain::Fifo);
    let twin = result.tally_outputs[0][0] / HISTORIES as f64;
    Some((cpu, twin))
}

#[test]
fn twin_and_cpu_agree_on_a_hot_material() {
    let Some((cpu_hot, twin_hot)) = cpu_and_twin_flux(HOT) else {
        eprintln!("skipping twin_temperature_parity -- no cached {ISOTOPE} data");
        return;
    };
    let (cpu_cold, twin_cold) = cpu_and_twin_flux(COLD).unwrap();

    let r_hot = twin_hot / cpu_hot;
    let r_cold = twin_cold / cpu_cold;
    let hot_vs_cold = cpu_hot / cpu_cold;
    eprintln!(
        "twin/CPU sphere flux: {HOT}K {r_hot:.4} (cpu {cpu_hot:.5e}), \
         {COLD}K {r_cold:.4} (cpu {cpu_cold:.5e}); hot/cold on the CPU {hot_vs_cold:.4}"
    );

    // The bound only has to exclude a backend running the wrong temperature. With
    // #478 present the CPU sampled targets at 294 K while the twin sampled at
    // 900 K, which is a 3.1x error in kT, far outside this.
    assert!(
        (0.96..=1.04).contains(&r_cold),
        "at {COLD} K twin/CPU flux is {r_cold:.4}, outside [0.96, 1.04]"
    );
    assert!(
        (0.96..=1.04).contains(&r_hot),
        "at {HOT} K twin/CPU flux is {r_hot:.4}, outside [0.96, 1.04]. The two \
         backends parse the material temperature separately; is the CPU still \
         transporting this material at 294 K (#478)?"
    );

    // Teeth for the assertions above: if the temperature reached neither
    // backend they would agree perfectly and prove nothing.
    assert!(
        (hot_vs_cold - 1.0).abs() > 0.02,
        "hot and cold spheres gave the same flux to within {:.3}%, so this test \
         cannot see the temperature at all and the parity bounds above are \
         vacuous",
        100.0 * (hot_vs_cold - 1.0).abs()
    );
}
