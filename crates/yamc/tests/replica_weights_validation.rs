//! Replica weights against brute-force perturbed reruns, replica by
//! replica, in every transport configuration they are carried through.
//!
//! Ignored: it runs millions of histories and prints the tables the pull
//! request reports. Point `YAMC_VALIDATION_DATA` at a release tree
//! (`<root>/neutron/<Nuclide>.arrow` with covariance, `<root>/photon/<El>.arrow`)
//! and run
//!
//! ```text
//! YAMC_VALIDATION_DATA=... cargo test --release -p yamc --test replica_weights_validation -- --ignored --nocapture
//! ```
//!
//! `YAMC_VALIDATION_REPLICAS` (16), `YAMC_VALIDATION_PARTICLES` (100000, the
//! replica-weight run) and `YAMC_VALIDATION_RERUN_PARTICLES` (200000, each
//! rerun) size it, and `YAMC_VALIDATION_CASES` picks cases by name.
//!
//! Each replica `k`'s relative shift from the replica weights,
//! `x_k / y - 1` from one run, is compared with the reruns', `Y_k / Y_0 - 1`
//! from independent runs with replica `k`'s draw applied to the data and
//! with none. The rerun shifts carry the Monte Carlo noise of two
//! independent runs, `n_k`; the replica-weight shifts are resolved far
//! better, because each replica shares the run's histories. Agreement is
//! `rms_k(shift_RW - shift_RR)` against `rms_k(n_k)`, and the spread of the
//! shifts over replicas (the nuclear-data sigma) from each method, the
//! reruns' with their noise deconvolved.

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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::covariance_fold::transport_fields;
use yani_transmute::covariance_sample::Sampler;

const SEED: u64 = 17;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn root() -> Option<PathBuf> {
    std::env::var("YAMC_VALIDATION_DATA")
        .ok()
        .map(PathBuf::from)
}

/// A material of `composition` (atom fractions) at `density` g/cm3, with
/// photon data when `photons`.
fn material(id: u32, composition: &[(&str, f64)], density: f64, photons: bool) -> Material {
    let root = root().expect("YAMC_VALIDATION_DATA");
    let mut m = Material::new(
        composition
            .iter()
            .map(|(n, f)| (n.to_string(), *f))
            .collect(),
        "atom",
        "g/cm3",
        Some(density),
    )
    .expect("material");
    m.set_material_id(id);
    m.set_temperature("294");
    let neutron: HashMap<String, String> = composition
        .iter()
        .map(|(n, _)| {
            (
                n.to_string(),
                root.join("neutron")
                    .join(format!("{n}.arrow"))
                    .to_string_lossy()
                    .into_owned(),
            )
        })
        .collect();
    let elements: HashMap<String, String> = composition
        .iter()
        .map(|(n, _)| {
            let el: String = n.chars().take_while(|c| c.is_alphabetic()).collect();
            let path = root.join("photon").join(format!("{el}.arrow"));
            (el, path.to_string_lossy().into_owned())
        })
        .collect();
    m.read_nuclear_data(&neutron, photons.then_some(&elements))
        .expect("read nuclear data");
    if photons {
        m.init_photon_data(&elements).expect("photon data");
    }
    m.ensure_covariance_loaded().expect("covariance");
    m
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Analog,
    Woodcock,
    Hybrid,
    SurvivalBiasing,
    WeightWindows,
    Everything,
    Photons,
    WoodcockPhotons,
    D1s,
}

/// Concentric spheres: per shell its outer radius and material (`None`
/// void), a 14.1 MeV point source at the centre, and the tallies: per
/// tally its cell, scores and particle.
struct Setup {
    name: &'static str,
    shells: Vec<(f64, Option<Material>)>,
    tallies: Vec<(u32, Vec<&'static str>, ParticleType)>,
    modes: Vec<Mode>,
}

fn setups(photons: bool) -> Vec<Setup> {
    let mut out = vec![
        Setup {
            name: "iron sphere, R 8 cm",
            shells: vec![(8.0, Some(material(1, &[("Fe56", 1.0)], 7.87, false)))],
            tallies: vec![(1, vec!["flux", "(n,gamma)", "(n,p)"], ParticleType::Neutron)],
            modes: vec![
                Mode::Analog,
                Mode::Woodcock,
                Mode::Hybrid,
                Mode::SurvivalBiasing,
                Mode::WeightWindows,
                Mode::Everything,
            ],
        },
        Setup {
            name: "iron core, void gap, water shell",
            shells: vec![
                (4.0, Some(material(1, &[("Fe56", 1.0)], 7.87, false))),
                (6.0, None),
                (
                    20.0,
                    Some(material(2, &[("H1", 2.0), ("O16", 1.0)], 1.0, false)),
                ),
            ],
            tallies: vec![
                (1, vec!["flux", "(n,p)"], ParticleType::Neutron),
                (3, vec!["flux", "(n,gamma)"], ParticleType::Neutron),
            ],
            modes: vec![
                Mode::Analog,
                Mode::Woodcock,
                Mode::Hybrid,
                Mode::SurvivalBiasing,
                Mode::WeightWindows,
            ],
        },
        Setup {
            name: "thorium sphere, R 5 cm (fission)",
            shells: vec![(5.0, Some(material(1, &[("Th232", 1.0)], 11.7, false)))],
            tallies: vec![(
                1,
                vec!["flux", "fission", "(n,gamma)"],
                ParticleType::Neutron,
            )],
            modes: vec![Mode::Analog, Mode::SurvivalBiasing, Mode::Woodcock],
        },
    ];
    if photons {
        out.push(Setup {
            name: "iron sphere with photons, R 8 cm",
            shells: vec![(8.0, Some(material(1, &[("Fe56", 1.0)], 7.87, true)))],
            tallies: vec![
                (1, vec!["flux", "(n,p)"], ParticleType::Neutron),
                (1, vec!["flux", "heating"], ParticleType::Photon),
            ],
            modes: vec![Mode::Photons, Mode::WoodcockPhotons, Mode::D1s],
        });
    }
    out
}

fn weight_windows(radius: f64) -> WeightWindowBounds {
    let n = 4;
    let mesh = RegularRectangularMesh::new([-radius; 3], [radius; 3], [n, n, n]);
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
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

/// One run; per tally, per score, the nominal mean, its standard
/// deviation, and each replica's mean when `replicas` is set.
fn run(
    setup: &Setup,
    materials: &[Material],
    mode: Mode,
    particles: usize,
    seed: u64,
    replicas: Option<usize>,
) -> Vec<(f64, f64, Vec<f64>)> {
    let mut cells = Vec::new();
    let mut inner: Option<Arc<Surface>> = None;
    let mut slot = 0;
    for (i, (radius, m)) in setup.shells.iter().enumerate() {
        let last = i + 1 == setup.shells.len();
        let surface = Arc::new(Surface {
            surface_id: Some(i + 1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: *radius,
            },
            boundary: if last {
                BoundaryType::Vacuum
            } else {
                BoundaryType::Transmission
            },
            name: None,
        });
        let mut region = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&surface)));
        if let Some(s) = &inner {
            region = region.intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                Arc::clone(s),
            )));
        }
        let material_idx = m.as_ref().map(|_| {
            slot += 1;
            slot - 1
        });
        cells.push(Cell::new(Some(i as u32 + 1), region, None, material_idx));
        inner = Some(surface);
    }
    let geometry = Geometry::new(
        cells,
        materials.iter().map(|m| Arc::new(m.clone())).collect(),
    )
    .expect("geometry");
    let tallies: Vec<Arc<Tally>> = setup
        .tallies
        .iter()
        .map(|(cell, scores, particle)| {
            let mut t = Tally::new();
            t.filters.push(Filter::Cell(CellFilter::from_id(*cell)));
            t.filters
                .push(Filter::ParticleType(ParticleTypeFilter::new(*particle)));
            t.scores = scores.iter().map(|s| s.parse().expect("score")).collect();
            t.estimator = Estimator::TrackLength;
            t.reset_accumulation();
            Arc::new(t)
        })
        .collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.verbose = Verbose::silent();
    model.tracking_mode = match mode {
        Mode::Woodcock | Mode::Everything | Mode::WoodcockPhotons => TrackingMode::Woodcock,
        Mode::Hybrid => TrackingMode::Hybrid,
        _ => TrackingMode::Surface,
    };
    if matches!(mode, Mode::SurvivalBiasing | Mode::Everything) {
        model
            .variance_reduction
            .push(VarianceReduction::SurvivalBiasing(
                SurvivalBiasing::default(),
            ));
    }
    if matches!(mode, Mode::WeightWindows | Mode::Everything) {
        let radius = setup.shells.last().unwrap().0;
        model
            .variance_reduction
            .push(VarianceReduction::WeightWindowBounds(Box::new(
                weight_windows(radius),
            )));
    }
    model.transport_secondary_photons =
        matches!(mode, Mode::Photons | Mode::WoodcockPhotons | Mode::D1s);
    model.use_decay_photons = mode == Mode::D1s;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed,
            data_uncertainty: replicas.map(|replicas| TransportDataUncertainty {
                seed: SEED,
                replicas,
            }),
            ..Default::default()
        })
        .expect("transport");
    let mut out = Vec::new();
    for t in &tallies {
        let mean = t.get_mean();
        let sd = t.get_std_dev();
        let sums = t.get_replica_sums();
        let h = t.finalize().n_histories as f64;
        for b in 0..mean.len() {
            let reps = sums.as_ref().map_or(Vec::new(), |s| {
                let width = s.sums.len() / mean.len();
                (0..s.replicas).map(|k| s.sums[b * width + k] / h).collect()
            });
            out.push((mean[b], sd[b], reps));
        }
    }
    out
}

/// Every material of the setup with replica `k`'s draw applied.
fn perturbed(materials: &[Material], k: u64) -> Vec<Material> {
    materials
        .iter()
        .map(|m| {
            let (fields, _) = transport_fields(m, &Default::default());
            let cells = fields
                .iter()
                .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
                .collect();
            let draw = Sampler::new(&cells, &[]).draw(SEED, k);
            perturbed_material(m, &fields, &draw, &Default::default())
                .expect("perturb")
                .0
        })
        .collect()
}

fn sd(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

#[test]
#[ignore = "validation tables: minutes of transport against a local release tree"]
fn replica_weights_against_reruns() {
    if root().is_none() {
        eprintln!("skipping: set YAMC_VALIDATION_DATA to a release tree");
        return;
    }
    let replicas = env_usize("YAMC_VALIDATION_REPLICAS", 16);
    let particles = env_usize("YAMC_VALIDATION_PARTICLES", 100_000);
    let rerun_particles = env_usize("YAMC_VALIDATION_RERUN_PARTICLES", 200_000);
    let picked = std::env::var("YAMC_VALIDATION_CASES").ok();
    let chain = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/transmutation-endf-b8.1-sfr.arrow")
        .to_string_lossy()
        .into_owned();
    {
        let mut cfg = yamc_nuclide::config::Config::global();
        cfg.transmutation_decay_data = Some(chain.clone());
        cfg.transmutation_reactions = yamc_nuclide::config::SubsectionSource::Set(chain.clone());
        cfg.transmutation_fission_yields = yamc_nuclide::config::SubsectionSource::Set(chain);
    }
    println!(
        "| case | mode | tally | sigma RW | sigma reruns deconvolved (raw) | rms(RW - rerun) | rerun noise |"
    );
    println!("|---|---|---|---|---|---|---|");
    for setup in setups(true) {
        if picked.as_ref().is_some_and(|p| !p.contains(setup.name)) {
            continue;
        }
        let materials: Vec<Material> = setup.shells.iter().filter_map(|(_, m)| m.clone()).collect();
        let labels: Vec<String> = setup
            .tallies
            .iter()
            .flat_map(|(cell, scores, particle)| {
                scores
                    .iter()
                    .map(move |s| format!("{particle:?} {s} (cell {cell})"))
            })
            .collect();
        let started = std::time::Instant::now();
        let draws: Vec<Vec<Material>> = (0..replicas as u64)
            .map(|k| perturbed(&materials, k))
            .collect();
        eprintln!(
            "{}: draws {:.0} s",
            setup.name,
            started.elapsed().as_secs_f64()
        );
        for &mode in &setup.modes {
            let started = std::time::Instant::now();
            let rw = run(&setup, &materials, mode, particles, 1, Some(replicas));
            let nominal = run(&setup, &materials, mode, 2 * rerun_particles, 2, None);
            let reruns: Vec<_> = draws
                .iter()
                .enumerate()
                .map(|(k, m)| run(&setup, m, mode, rerun_particles, 100 + k as u64, None))
                .collect();
            eprintln!(
                "{} {mode:?}: {:.0} s",
                setup.name,
                started.elapsed().as_secs_f64()
            );
            for (i, label) in labels.iter().enumerate() {
                let (y, _, xs) = &rw[i];
                let shift_rw: Vec<f64> = xs.iter().map(|x| x / y - 1.0).collect();
                let (y0, s0, _) = nominal[i];
                let mut shift_rr = Vec::new();
                let mut noise = Vec::new();
                for r in &reruns {
                    let (yk, sk, _) = r[i];
                    shift_rr.push(yk / y0 - 1.0);
                    noise.push((yk / y0) * ((sk / yk).powi(2) + (s0 / y0).powi(2)).sqrt());
                }
                let diff: Vec<f64> = shift_rw.iter().zip(&shift_rr).map(|(a, b)| a - b).collect();
                let rr_sd = sd(&shift_rr);
                let rr_noise = rms(&noise);
                let deconvolved = (rr_sd * rr_sd - rr_noise * rr_noise).max(0.0).sqrt();
                println!(
                    "| {} | {mode:?} | {label} | {:.3}% | {:.3}% ({:.3}%) | {:.3}% | {:.3}% |",
                    setup.name,
                    100.0 * sd(&shift_rw),
                    100.0 * deconvolved,
                    100.0 * rr_sd,
                    100.0 * rms(&diff),
                    100.0 * rr_noise,
                );
            }
        }
    }
}
