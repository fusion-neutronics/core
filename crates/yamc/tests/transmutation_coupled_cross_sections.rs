//! Cross-section uncertainty on a coupled transport-transmutation: each
//! replica the whole coupled run again, every transport on that replica's own
//! nuclear data.
//!
//! The fixture is the cached ENDF/B-VIII.1 W186 directory with the
//! `covariance.arrow` and `resonance_parameters.arrow` the real converter
//! writes from the committed MF=2 and MF=32 evaluation, in a pure tungsten
//! sphere driven at 1 keV, where capture is self-shielded and its
//! uncertainty moves the flux it is tallied in. Self-skips when the
//! nuclear-data fixtures are missing.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::coupled_xs::CoupledCrossSections;
use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::xs_perturbation::{
    perturbed_nuclide, resonance_sampling, sampled_names, ResonanceChange,
};
use yamc_materials::Material;
use yamc_nuclide::LoadScope;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yani_transmute::coupled_uncertainty::CoupledReplicas;
use yani_transmute::covariance_fold::{transport_fields, TransportField};
use yani_transmute::covariance_sample::Sampler;
use yani_transmute::resonance_rates::ResonanceMethod;
use yani_transmute::uncertainty::{DataUncertainty, FluxResponseReport, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep, TransmutationResults};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const LI7_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_007_mf33.endf.xz");
const SEED: u64 = 31;
const BASE_SEED: u64 = 5;
const RADIUS: f64 = 5.0;

type Chain = Arc<HashMap<String, yani::ChainNuclide>>;

/// A copy of the cached `name` directory with the covariance, and the
/// resonance parameters where the evaluation `endf` states them, that the
/// real converter writes from it, or `None` when the fixture is missing.
fn with_covariance(tmp: &Path, name: &str, endf: &[u8]) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide(name)?);
    let dir = tmp.join(format!("{name}.arrow"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached fixture") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &endf[..], &mut raw).expect("fixture decompresses");
    let text = String::from_utf8(raw).expect("ENDF is text");
    let evaluation = endf::Material::from_str(&text).expect("evaluation parses");
    assert!(yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance"));
    yamc_convert::resonance_parameters::write_resonance_parameters(&evaluation, &dir)
        .expect("resonance parameters");
    Some(dir)
}

/// W186 with its MF=2 and MF=32: the resonance parameters are sampled.
fn w186(tmp: &Path) -> Option<PathBuf> {
    with_covariance(tmp, "W186", W186_ENDF)
}

fn chain() -> Option<Chain> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    path.exists()
        .then(|| Arc::new(yani::parse_chain_arrow(&path).expect("chain")))
}

/// Pure W186 read from `dir`, transmutable, filling the sphere.
fn tungsten(dir: &Path) -> Material {
    pure("W186", dir, 19.3)
}

/// Pure `name` read from `dir` at `density`, transmutable, filling the
/// sphere.
fn pure(name: &str, dir: &Path, density: f64) -> Material {
    let mut m = Material::new(
        HashMap::from([(name.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.volume = Some(4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3));
    m.transmutable = true;
    m.read_nuclear_data(
        &HashMap::from([(name.to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read nuclear data");
    m
}

/// A sphere of `material` around a 1 keV point source.
fn model(material: Material) -> Model {
    model_at(material, 1.0e3)
}

/// A sphere of `material` around a point source at `energy` eV.
fn model_at(material: Material, energy: f64) -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).expect("geometry");
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
}

fn settings(seed: u64, particles: usize) -> TransportSettings {
    TransportSettings {
        total_particles: Some(particles),
        seed,
        threads: Some(1),
        ..Default::default()
    }
}

fn request(sources: Vec<UqSource>, samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: SEED,
        samples: Some(samples),
        sources,
        ..Default::default()
    }
}

/// `pulses` pulses that each capture about a part in 10^4 of the W186, then
/// a day's cooldown.
fn schedule(pulses: usize) -> Schedule {
    let pulse = ScheduleStep {
        rate: 1.0e16,
        dt: 1.0e4,
        is_pulse: true,
    };
    let mut steps = vec![pulse; pulses];
    steps.push(ScheduleStep {
        rate: 0.0,
        dt: 8.64e4,
        is_pulse: false,
    });
    Schedule::new(steps).expect("schedule")
}

#[allow(clippy::too_many_arguments)]
fn transmute(
    model: &mut Model,
    method: &str,
    schedule: &Schedule,
    chain: &Chain,
    seed: u64,
    particles: usize,
    request: Option<&DataUncertainty>,
) -> TransmutationResults {
    model
        .transmute(
            method,
            schedule,
            Arc::clone(chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &settings(seed, particles),
            request,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        )
        .expect("transmute")
}

/// Every step's inventory of material 1.
fn inventories(results: &TransmutationResults, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.get_material(1, i).expect("step").nuclides.clone())
        .collect()
}

/// Replica `k`'s every-step inventory, as the ensemble holds it.
fn replica(results: &TransmutationResults, k: usize, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.uncertainty_inventories(1, i).expect("replicas")[k].clone())
        .collect()
}

/// W186 with replica `k`'s draw applied, built here from the public pieces
/// the replica weights draw with: its MF=33 field without the first-order
/// MF=32 rows, drawn under `(SEED, k)`, and its resonance parameters' change
/// under the same. The fixture's covariance is the MF=32 rows alone, so with
/// the parameters sampled the field is empty and the resonance change is the
/// whole draw.
fn perturbed_w186(dir: &Path, k: u64) -> Material {
    let mut m = tungsten(dir);
    m.ensure_covariance_loaded().expect("covariance");
    m.ensure_resonance_parameters_loaded()
        .expect("resonance parameters");
    let sampling = resonance_sampling(&m);
    let (fields, _) = transport_fields(&m, &sampled_names(&sampling));
    let unfielded = TransportField {
        reads: BTreeMap::new(),
        derived: BTreeMap::new(),
        field: None,
    };
    let field = fields
        .get("W186")
        .filter(|t| t.field.is_some())
        .unwrap_or(&unfielded);
    let draw = field
        .field
        .clone()
        .map(|f| Sampler::new(&BTreeMap::from([("W186".to_string(), f)]), &[]).draw(SEED, k));
    let (relative, absolute) = draw
        .as_ref()
        .and_then(|d| d.relative("W186").zip(d.absolute("W186")))
        .unwrap_or((&[], &[]));
    let nominal = Arc::clone(&m.nuclide_data["W186"]);
    let grid = nominal.energy.as_ref().expect("grid")["294"].as_slice();
    let change = ResonanceChange {
        temperature: "294".to_string(),
        by_mt: sampling["W186"]
            .as_ref()
            .expect("sampled")
            .changes(SEED, k, grid)
            .expect("changes"),
    };
    let (w, _) =
        perturbed_nuclide(&nominal, field, relative, absolute, Some(&change)).expect("perturb");
    m.nuclide_data.insert("W186".to_string(), Arc::new(w));
    m.invalidate_xs_cache();
    m
}

/// The nominal results and the model left behind are a plain coupled run's,
/// bit for bit, and replica `k` is bit for bit the plain coupled run on
/// W186 with replica `k`'s draw, transported with replica `k`'s seed: every
/// step's transport, the rates tallied in it and the inventories they make
/// read that replica's own data. The report says what was drawn.
#[test]
fn each_replica_is_the_coupled_run_on_its_own_data() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    const PARTICLES: usize = 2000;
    let schedule = schedule(2);
    let steps = 3;
    let request = request(vec![UqSource::Statistical, UqSource::CrossSections], 2);

    let mut plain_model = model(tungsten(&dir));
    let plain = transmute(
        &mut plain_model,
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        PARTICLES,
        None,
    );
    let mut uq_model = model(tungsten(&dir));
    let t = std::time::Instant::now();
    let uq = transmute(
        &mut uq_model,
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        PARTICLES,
        Some(&request),
    );
    eprintln!("coupled run with 2 replicas: {:.2?}", t.elapsed());
    assert_eq!(inventories(&plain, steps), inventories(&uq, steps));
    let left = |m: &Model| m.geometry.materials()[0].nuclides.clone();
    assert_eq!(left(&plain_model), left(&uq_model));
    let capture = |m: &Model| {
        m.geometry.materials()[0].nuclide_data["W186"]
            .reactions_for_temp("294")
            .expect("294")[&102]
            .cross_section
            .as_slice()
            .to_vec()
    };
    assert_eq!(
        capture(&plain_model),
        capture(&uq_model),
        "the model is left with the nominal data"
    );

    let info = &uq.uncertainty_info[&1];
    assert_eq!(info.flux_response, FluxResponseReport::CoupledReplicas);
    assert!(info.perturbed.contains("W186"), "{:?}", info.perturbed);
    assert!(
        matches!(
            info.resonance_parameters.get("W186"),
            Some(ResonanceMethod::Sampled { .. })
        ),
        "{:?}",
        info.resonance_parameters
    );
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s == "activation cross section (MF=33)"));

    let replicas =
        CoupledReplicas::new(&request, &chain, &yani::BranchTable::new()).expect("carried");
    for k in 0..2 {
        let seed = replicas.transport_seed(BASE_SEED, k);
        let rerun = transmute(
            &mut model(perturbed_w186(&dir, k)),
            "coupled",
            &schedule,
            &chain,
            seed,
            PARTICLES,
            None,
        );
        let ours = replica(&uq, k as usize, steps);
        let theirs = inventories(&rerun, steps);
        assert_eq!(
            ours, theirs,
            "replica {k} is the coupled run on its own data"
        );
        let w187 = |i: &[HashMap<String, f64>]| i[0]["W187"];
        eprintln!(
            "replica {k}: W187 after the first pulse {:+.4e} from the nominal",
            w187(&ours) / w187(&inventories(&plain, steps)) - 1.0
        );
    }
}

/// A product held at activation scope moves by exactly what the draw moved
/// the same reactions of its full data, so when a step widens it to full
/// scope mid-run its cross sections stay those of the same draw. W186 moves
/// by its sampled resonance parameters, Li7 by its MF=33 field.
#[test]
fn a_product_keeps_its_draw_whatever_scope_holds_it() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let cases = [
        ("W186", w186(tmp.path()), 19.3, vec![16, 102, 103, 107]),
        (
            "Li7",
            with_covariance(tmp.path(), "Li7", LI7_ENDF),
            0.53,
            vec![16, 102, 103, 104, 105],
        ),
    ];
    for (name, dir, density, mts) in cases {
        let Some(dir) = dir else {
            eprintln!("skipping {name}: fixture not fetched");
            continue;
        };
        let full = pure(name, &dir, density);
        let narrow = yamc_nuclide::nuclide_loader::load_nuclide(
            &dir,
            &LoadScope::activation(mts.into_iter().collect())
                .with_temperatures(Some(HashSet::from(["294".to_string()]))),
        )
        .expect("activation-scope load");
        assert!(narrow.load_scope.mts.is_some());
        // The same material holding the nuclide as a transmutation product
        // is held: the network's reactions and none of the transport
        // sections.
        let mut product = full.clone();
        product
            .nuclide_data
            .insert(name.to_string(), Arc::new(narrow));

        let mut library =
            CoupledCrossSections::new(&[Arc::new(full.clone())], SEED).expect("library");
        let nominal = full.nuclide_data[name].reactions_for_temp("294").unwrap();
        for k in 0..2 {
            let mut replica = library.replica(k);
            let (mut a, mut b) = (full.clone(), product.clone());
            assert!(replica.apply(&mut a).expect("full"));
            assert!(replica.apply(&mut b).expect("activation"));
            let wide = a.nuclide_data[name].reactions_for_temp("294").unwrap();
            let held = b.nuclide_data[name].reactions_for_temp("294").unwrap();
            let mut moved = false;
            for (mt, r) in held {
                let (x, y) = (
                    r.cross_section.as_slice(),
                    wide[mt].cross_section.as_slice(),
                );
                assert_eq!(x.len(), y.len(), "{name} MT {mt}");
                for (u, v) in x.iter().zip(y) {
                    assert!(
                        (u - v).abs() <= 1e-12 * v.abs(),
                        "{name} replica {k}, MT {mt}: {u} held against {v} in full"
                    );
                }
                moved |= x != nominal[mt].cross_section.as_slice();
            }
            assert!(moved, "{name} replica {k} moves the held reactions");
        }
    }
}

/// A product that grows into transport mid-run is drawn there too: a B10
/// sphere burned enough that the Li7 it makes is widened to full scope
/// after the first pulse, with Li7's MF=33 covariance read for it. Li7 is
/// drawn in every replica (as a product at activation scope, then in full
/// with the same draw, which `a_product_keeps_its_draw_whatever_scope_holds_it`
/// pins), the nominal is still the plain run's, and the replicas move.
#[test]
fn a_product_that_enters_transport_is_drawn_there() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(b10), Some(li7), Some(chain)) = (
        yamc_test_cache::nuclide("B10"),
        with_covariance(tmp.path(), "Li7", LI7_ENDF),
        chain(),
    ) else {
        eprintln!("skipping: B10, Li7 or chain fixture not fetched");
        return;
    };
    yamc_nuclide::config::Config::global()
        .set_cross_section("Li7", Some(li7.to_string_lossy().as_ref()));
    let boron = || pure("B10", Path::new(&b10), 2.3);
    let schedule = Schedule::new(vec![
        ScheduleStep {
            rate: 1.0e20,
            dt: 1.0e4,
            is_pulse: true,
        };
        2
    ])
    .expect("schedule");
    let request = request(vec![UqSource::Statistical, UqSource::CrossSections], 3);
    let plain = transmute(
        &mut model(boron()),
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        2000,
        None,
    );
    let uq = transmute(
        &mut model(boron()),
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        2000,
        Some(&request),
    );
    assert_eq!(inventories(&plain, 2), inventories(&uq, 2));
    let li7 = plain.get_nuclide_density(1, "Li7", 1).expect("Li7");
    let b10 = plain.get_nuclide_density(1, "B10", 1).expect("B10");
    assert!(
        li7 > 1e-3 * b10,
        "Li7 grows into transport: {li7:e} beside {b10:e}"
    );
    let info = &uq.uncertainty_info[&1];
    assert!(info.perturbed.contains("Li7"), "{:?}", info.perturbed);
    assert!(
        info.no_covariance_data.contains("B10"),
        "{:?}",
        info.no_covariance_data
    );
    let spread = uq.uncertainty_inventories(1, 2).expect("replicas");
    assert!(spread.iter().any(|r| r["Li7"] != spread[0]["Li7"]));
}

/// Each replica's `product` over the nominal after the first step, minus
/// one.
fn shifts(results: &TransmutationResults, product: &str) -> Vec<f64> {
    let nominal = results.get_nuclide_density(1, product, 1).expect("product");
    results
        .uncertainty_inventories(1, 1)
        .expect("replicas")
        .iter()
        .map(|i| i.get(product).copied().unwrap_or(0.0) / nominal - 1.0)
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// On one pulse the coupled method's transport is the independent method's,
/// and replica `k` of either draws the same cross sections, so each coupled
/// replica's inventory moves as the independent replica's does, to within the
/// Monte Carlo noise of its own transport. The independent replica reads the
/// draw off the nominal histories through the replica weights, the coupled
/// one transports on it with its own seed: two routes to one number.
///
/// The case is a Li7 sphere at 14 MeV, whose `(n,2n)` to Li6 carries
/// ENDF/B-VIII.1's MF=33: the MF=33 path, without the resonance
/// reconstruction a sampled range costs every replica (that path is pinned
/// bit for bit by `each_replica_is_the_coupled_run_on_its_own_data`).
///
/// Every coupled shift is taken from the one nominal run, whose own noise
/// moves them all alike; the independent replicas share the nominal's
/// histories and carry none of it. So the differences are compared about
/// their mean, which is that one noise draw and is checked on its own. The
/// draws spread Li6 by 9.7% against 0.14% of noise, and the differences
/// spread by the noise alone (0.14%), so the agreement is replica by replica
/// and not only in the spread: draws taken apart from the independent ones
/// would leave differences of about 14%.
#[test]
fn on_one_pulse_each_replica_moves_as_the_independent_one() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (with_covariance(tmp.path(), "Li7", LI7_ENDF), chain()) else {
        eprintln!("skipping: Li7 or chain fixture not fetched");
        return;
    };
    const PARTICLES: usize = 20_000;
    const REPLICAS: usize = 12;
    const PRODUCT: &str = "Li6";
    let schedule = schedule(1);
    let run = |method: &str, sources: Vec<UqSource>, samples: usize| {
        let t = std::time::Instant::now();
        let r = transmute(
            &mut model_at(pure("Li7", &dir, 0.53), 1.406e7),
            method,
            &schedule,
            &chain,
            BASE_SEED,
            PARTICLES,
            Some(&request(sources, samples)),
        );
        eprintln!("{method}, {samples} replicas: {:.2?}", t.elapsed());
        r
    };
    let independent = shifts(
        &run("independent", vec![UqSource::CrossSections], REPLICAS),
        PRODUCT,
    );
    let noise = run("independent", vec![UqSource::Statistical], 256);
    let nominal = noise.get_nuclide_density(1, PRODUCT, 1).expect("product");
    let noise = noise.get_nuclide_uncertainty(1, PRODUCT, 1).expect("sigma") / nominal;
    let coupled = shifts(
        &run(
            "coupled",
            vec![UqSource::Statistical, UqSource::CrossSections],
            REPLICAS,
        ),
        PRODUCT,
    );
    let difference: Vec<f64> = coupled
        .iter()
        .zip(&independent)
        .map(|(c, i)| c - i)
        .collect();
    for k in 0..REPLICAS {
        eprintln!(
            "replica {k:2}: coupled {:+.4e}, independent {:+.4e}",
            coupled[k], independent[k]
        );
    }
    eprintln!(
        "{PRODUCT}: the draws spread it by {:.3e}, the transport's noise is {noise:.3e}; \
         coupled less independent is {:+.3e} on average, {:.3e} about that",
        sd(&independent),
        mean(&difference),
        sd(&difference)
    );
    // Draws taken apart from the independent ones would leave the
    // differences spread by both sets of draws as well as the noise.
    assert!(
        sd(&independent) > 10.0 * noise,
        "the draws must move {PRODUCT} well past the noise for this to test anything"
    );
    // Twelve differences of variance `noise^2` have a standard deviation
    // under 1.6 of it with probability over 0.995 (chi-square, 11 degrees).
    assert!(
        sd(&difference) < 1.6 * noise,
        "coupled replicas differ from the independent ones by {:.3e} about the mean, more \
         than the {noise:.3e} the transport's noise explains",
        sd(&difference)
    );
    assert!(
        mean(&difference).abs() < 4.0 * noise,
        "the common offset {:+.3e} is the nominal run's own noise, {noise:.3e}",
        mean(&difference)
    );
}

/// With attribution, the cross sections' entry is the variance with them
/// and the statistical source less the variance with the statistical source
/// alone, the two ensembles on the same transport seeds, to the bit; and on
/// Li7 at 14 MeV, where the draws move Li6 by far more than the noise, it is
/// nearly all of the total.
#[test]
fn attribution_measures_the_cross_sections_beside_the_noise() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (with_covariance(tmp.path(), "Li7", LI7_ENDF), chain()) else {
        eprintln!("skipping: Li7 or chain fixture not fetched");
        return;
    };
    let schedule = schedule(1);
    let run = |sources: Vec<UqSource>, attribution: bool| {
        transmute(
            &mut model_at(pure("Li7", &dir, 0.53), 1.406e7),
            "coupled",
            &schedule,
            &chain,
            BASE_SEED,
            4000,
            Some(&DataUncertainty {
                attribution,
                ..request(sources, 6)
            }),
        )
    };
    let whole = run(vec![UqSource::Statistical, UqSource::CrossSections], true);
    let both = run(vec![UqSource::Statistical, UqSource::CrossSections], false);
    let stat = run(vec![UqSource::Statistical], false);
    let attribution = whole.uncertainty[&1]
        .attribution
        .as_ref()
        .expect("attribution");
    let variance = |r: &TransmutationResults| r.uncertainty[&1].std_dev_at(0)["Li6"].powi(2);
    let xs = attribution.by_source["cross_sections"][0]["Li6"];
    assert_eq!(xs, variance(&both) - variance(&stat));
    assert_eq!(
        attribution.by_source["statistical"][0]["Li6"],
        variance(&stat)
    );
    assert_eq!(variance(&whole), variance(&both));
    assert!(
        xs > 0.9 * variance(&whole),
        "the cross sections are most of Li6's variance: {xs:e} of {:e}",
        variance(&whole)
    );
}

/// What a coupled replica costs beside the nominal run, on this sphere:
/// reading the data in full once, applying one replica's draw, and the
/// coupled run with and without replicas. Prints, asserts nothing.
#[test]
#[ignore = "timings"]
fn cost_of_a_coupled_replica() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    let particles: usize = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let replicas = 8;
    let material = Arc::new(tungsten(&dir));
    let t = std::time::Instant::now();
    let mut library = CoupledCrossSections::new(&[Arc::clone(&material)], SEED).expect("library");
    eprintln!("read in full and fielded once: {:.2?}", t.elapsed());
    let t = std::time::Instant::now();
    for k in 0..replicas {
        let mut m = (*material).clone();
        library.replica(k).apply(&mut m).expect("apply");
    }
    eprintln!(
        "one replica's draw applied: {:.2?}",
        t.elapsed() / replicas as u32
    );
    for pulses in [1, 3] {
        let schedule = schedule(pulses);
        let t = std::time::Instant::now();
        transmute(
            &mut model(tungsten(&dir)),
            "coupled",
            &schedule,
            &chain,
            BASE_SEED,
            particles,
            None,
        );
        let plain = t.elapsed();
        for sources in [
            vec![UqSource::Statistical],
            vec![UqSource::Statistical, UqSource::CrossSections],
        ] {
            let t = std::time::Instant::now();
            transmute(
                &mut model(tungsten(&dir)),
                "coupled",
                &schedule,
                &chain,
                BASE_SEED,
                particles,
                Some(&request(sources.clone(), replicas as usize)),
            );
            let with = t.elapsed();
            eprintln!(
                "{pulses} pulse(s), {particles} particles: plain {plain:.2?}; with {sources:?} \
                 and {replicas} replicas {with:.2?}, {:.2?} per replica",
                with.saturating_sub(plain) / replicas as u32
            );
        }
    }
}
