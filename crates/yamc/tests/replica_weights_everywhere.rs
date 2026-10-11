//! Correlated nuclear-data replica weights under delta tracking, survival
//! biasing, weight windows and coupled photon transport.
//!
//! Every configuration estimates the same per-replica expectation as
//! analog surface tracking, which `replica_weights.rs` checks against
//! reruns, so each replica's relative shift from the nominal tally is
//! compared with analog surface tracking's over independent seeds (the
//! shifts are well resolved, because each run's replicas share its
//! histories). Photon tallies have no analog surface counterpart without
//! photons, so they are compared with reruns of the same draws.
//!
//! On ENDF/B-VIII.1 Fe56, whose covariance covers elastic, capture and the
//! total inelastic; self-skips where the fixture carries no covariance or
//! the Fe photon data is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction, WeightWindowBounds};
use yamc::xs_perturbation::perturbed_material;
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::covariance_fold::transport_fields;
use yani_transmute::covariance_sample::Sampler;

const SEED: u64 = 11;
const REPLICAS: usize = 4;
const RADIUS: f64 = 8.0;
const ENERGY: f64 = 14.1e6;

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Fe56 with its covariance, and Fe photon data when `photons`.
fn iron(photons: bool) -> Option<Material> {
    let fe56 = tests_dir().join("Fe56.arrow");
    let fe = tests_dir().join("Fe.arrow");
    if !fe56.join("covariance.arrow").is_file() || (photons && !fe.is_dir()) {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    let neutron = HashMap::from([("Fe56".to_string(), fe56.to_string_lossy().into_owned())]);
    let photon = HashMap::from([("Fe".to_string(), fe.to_string_lossy().into_owned())]);
    m.read_nuclear_data(&neutron, photons.then_some(&photon))
        .expect("read Fe56");
    if photons {
        m.init_photon_data(&photon).expect("Fe photon data");
    }
    m.ensure_covariance_loaded().expect("read covariance");
    Some(m)
}

/// How a run transports.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Analog,
    Woodcock,
    Hybrid,
    SurvivalBiasing,
    WeightWindows,
    /// Delta tracking with survival biasing and weight windows at once.
    Everything,
    Photons,
    WoodcockPhotons,
}

impl Mode {
    fn photons(self) -> bool {
        matches!(self, Mode::Photons | Mode::WoodcockPhotons)
    }
}

/// Weight windows over the sphere that roulette every collision in the
/// central voxels up to a weight of 6 and split it again outside, so both
/// games are played on most histories.
fn weight_windows() -> WeightWindowBounds {
    let n = 4;
    let mesh = RegularRectangularMesh::new([-RADIUS; 3], [RADIUS; 3], [n, n, n]);
    let mut lower = Vec::with_capacity(n * n * n);
    let mut upper = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let inner = [i, j, k].iter().all(|&x| x == 1 || x == 2);
                lower.push(if inner { 2.0 } else { 0.1 });
                upper.push(if inner { 10.0 } else { 0.5 });
            }
        }
    }
    WeightWindowBounds {
        mesh,
        particle: ParticleType::Neutron,
        energy_bins: None,
        lower_bounds: lower,
        upper_bounds: upper,
        survival_factor: 3.0,
        max_split: 10,
        weight_floor: 1e-38,
    }
}

fn tally(scores: &[&str], particle: ParticleType) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(particle)));
    t.scores = scores
        .iter()
        .map(|s| s.parse::<Score>().expect("score"))
        .collect();
    t.estimator = Estimator::TrackLength;
    t.reset_accumulation();
    Arc::new(t)
}

/// One run of an iron sphere around a 14.1 MeV point source: a neutron
/// tally of flux and capture, and with photons a photon tally of flux and
/// heating.
fn run(
    material: Material,
    mode: Mode,
    particles: usize,
    seed: u64,
    data: Option<TransportDataUncertainty>,
) -> Vec<Arc<Tally>> {
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
    let mut tallies = vec![tally(&["flux", "(n,gamma)"], ParticleType::Neutron)];
    if mode.photons() {
        tallies.push(tally(&["flux", "heating"], ParticleType::Photon));
    }
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![ENERGY], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(
        geometry,
        vec![source],
        tallies.iter().map(Arc::clone).collect(),
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = match mode {
        Mode::Woodcock | Mode::Everything | Mode::WoodcockPhotons => TrackingMode::Woodcock,
        Mode::Hybrid => TrackingMode::Hybrid,
        _ => TrackingMode::Surface,
    };
    if matches!(mode, Mode::SurvivalBiasing | Mode::Everything) {
        model
            .variance_reduction
            .push(VarianceReduction::SurvivalBiasing(SurvivalBiasing {
                weight_cutoff: 0.25,
                weight_survive: 1.0,
            }));
    }
    if matches!(mode, Mode::WeightWindows | Mode::Everything) {
        model
            .variance_reduction
            .push(VarianceReduction::WeightWindowBounds(Box::new(
                weight_windows(),
            )));
    }
    model.transport_secondary_photons = mode.photons();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed,
            threads: Some(1),
            data_uncertainty: data,
            ..Default::default()
        })
        .expect("transport");
    tallies
}

fn uq() -> Option<TransportDataUncertainty> {
    Some(TransportDataUncertainty {
        seed: SEED,
        replicas: REPLICAS,
    })
}

/// Per score of a tally, the nominal mean and each replica's mean.
fn means(t: &Arc<Tally>) -> Vec<(f64, Vec<f64>)> {
    let sums = t.get_replica_sums().expect("replica sums");
    let h = t.finalize().n_histories as f64;
    let nominal = t.get_mean();
    let width = sums.sums.len() / nominal.len();
    nominal
        .iter()
        .enumerate()
        .map(|(b, &y)| {
            let row = &sums.sums[b * width..];
            (y, (0..sums.replicas).map(|k| row[k] / h).collect())
        })
        .collect()
}

/// Mean and standard error over seeds of each replica's relative shift
/// `x_k / y - 1` in every (tally, score), flattened.
fn shifts(material: &Material, mode: Mode, particles: usize, seeds: &[u64]) -> Vec<(f64, f64)> {
    let mut per_seed: Vec<Vec<f64>> = Vec::new();
    for &seed in seeds {
        let tallies = run(material.clone(), mode, particles, seed, uq());
        let mut row = Vec::new();
        for t in &tallies {
            for (y, xs) in means(t) {
                row.extend(xs.iter().map(|x| x / y - 1.0));
            }
        }
        per_seed.push(row);
    }
    let n = per_seed.len() as f64;
    (0..per_seed[0].len())
        .map(|i| {
            let v: Vec<f64> = per_seed.iter().map(|r| r[i]).collect();
            let mean = v.iter().sum::<f64>() / n;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, (var / n).sqrt())
        })
        .collect()
}

/// Each replica's shift under `mode` agrees with analog surface tracking's,
/// and at least one replica moves the tallies by many standard errors, so
/// the comparison has teeth.
fn agrees_with_analog(mode: Mode, particles: usize) {
    let Some(m) = iron(false) else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let seeds: Vec<u64> = (1..=6).collect();
    let reference = shifts(&m, Mode::Analog, particles, &seeds);
    let tested = shifts(&m, mode, particles, &seeds);
    let mut resolved = 0.0_f64;
    for (i, ((a, sa), (b, sb))) in reference.iter().zip(&tested).enumerate() {
        let z = (b - a) / (sa * sa + sb * sb).sqrt();
        eprintln!(
            "{mode:?} [{i}]: shift {b:+.4e} ± {sb:.1e}, analog {a:+.4e} ± {sa:.1e}, z = {z:+.2}"
        );
        assert!(z.abs() < 4.0, "{mode:?} [{i}]: z = {z:.2}");
        resolved = resolved.max(a.abs() / sa);
    }
    assert!(
        resolved > 5.0,
        "no replica shift is resolved ({resolved:.1} standard errors)"
    );
}

#[test]
fn delta_tracking_carries_the_replicas() {
    agrees_with_analog(Mode::Woodcock, 3_000);
}

#[test]
fn hybrid_tracking_carries_the_replicas() {
    agrees_with_analog(Mode::Hybrid, 3_000);
}

#[test]
fn survival_biasing_carries_the_replicas() {
    agrees_with_analog(Mode::SurvivalBiasing, 3_000);
}

#[test]
fn weight_windows_carry_the_replicas() {
    agrees_with_analog(Mode::WeightWindows, 1_500);
}

#[test]
fn every_technique_at_once_carries_the_replicas() {
    agrees_with_analog(Mode::Everything, 1_500);
}

/// The nominal tallies of a surface-tracked run are bit-identical with the
/// replicas on, survival biasing, weight windows and photons included: the
/// weights only read the transport.
#[test]
fn the_nominal_tallies_are_unchanged_by_the_replicas() {
    let Some(m) = iron(true) else {
        eprintln!("skipping: Fe56 covariance or Fe photon data missing");
        return;
    };
    for mode in [Mode::SurvivalBiasing, Mode::WeightWindows, Mode::Photons] {
        let plain = run(m.clone(), mode, 500, 3, None);
        let with = run(m.clone(), mode, 500, 3, uq());
        for (a, b) in plain.iter().zip(&with) {
            assert_eq!(
                a.get_mean(),
                b.get_mean(),
                "{mode:?}: the nominal mean moved"
            );
            assert_eq!(a.get_std_dev(), b.get_std_dev(), "{mode:?}");
        }
    }
}

/// The material with replica `k`'s draw applied to its data.
fn rerun_material(m: &Material, k: u64) -> Material {
    let (fields, _) = transport_fields(m, &Default::default());
    let cells = fields
        .iter()
        .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
        .collect();
    let draw = Sampler::new(&cells, &[]).draw(SEED, k);
    perturbed_material(m, &fields, &draw, &Default::default())
        .expect("perturb")
        .0
}

/// Photons carry their neutron's ratios times their own reaction's
/// `σ'_mt / σ_mt`, and keep them through their flights and collisions:
/// each replica's photon flux and photon heating match a rerun of the same
/// draw, as do the neutron tallies, under surface and delta tracking.
fn photons_match_their_reruns(mode: Mode) {
    let Some(m) = iron(true) else {
        eprintln!("skipping: Fe56 covariance or Fe photon data missing");
        return;
    };
    let particles = 4_000;
    let weighted = run(m.clone(), mode, particles, 21, uq());
    let h = weighted[0].finalize().n_histories as f64;
    for k in 0..REPLICAS {
        let rerun = run(
            rerun_material(&m, k as u64),
            mode,
            particles,
            500 + k as u64,
            None,
        );
        for (t, (w, r)) in weighted.iter().zip(&rerun).enumerate() {
            let sums = w.get_replica_sums().expect("replica sums");
            let width = sums.sums.len() / w.get_mean().len();
            for (b, (y, sy)) in r.get_mean().iter().zip(r.get_std_dev()).enumerate() {
                let row = &sums.sums[b * width..];
                let (s1, s2) = (row[k], row[REPLICAS + k]);
                let x = s1 / h;
                let sx = ((s2 - s1 * s1 / h) / (h - 1.0) / h).sqrt();
                let z = (x - y) / (sx * sx + sy * sy).sqrt();
                eprintln!("{mode:?} replica {k} tally {t} score {b}: {x:e} ± {sx:e}, rerun {y:e} ± {sy:e}, z = {z:+.2}");
                assert!(
                    z.abs() < 4.0,
                    "{mode:?} replica {k} tally {t} score {b}: z = {z:.2}"
                );
            }
        }
    }
}

#[test]
fn photons_match_their_reruns_under_surface_tracking() {
    photons_match_their_reruns(Mode::Photons);
}

#[test]
fn photons_match_their_reruns_under_delta_tracking() {
    photons_match_their_reruns(Mode::WoodcockPhotons);
}
