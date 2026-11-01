//! Replica weights that carry each replica's own composition as well as its
//! own cross sections: replica `k` of one transport on the nominal
//! composition against a rerun on replica `k`'s.
//!
//! A replica with atom densities `N'_k` and cross sections `σ'_k` has, on a
//! flight, `Σ'_t,k = Σ_i N'_k,i σ'_k,i`, and on a collision with nuclide `i`
//! the density `N'_k,i σ'_k,mt`; the ratios the nominal histories carry are
//! these over the nominal ones, so every replica tally is unbiased for the
//! run on its own composition. That is what a coupled run's replicas need
//! of a step's transport once their compositions have drifted apart.
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc::xs_perturbation::{
    perturbed_nuclide, resonance_sampling, sampled_names, ResonanceChange,
};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::covariance_fold::{transport_fields, TransportField};
use yani_transmute::covariance_sample::Sampler;

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 31;
const RADIUS: f64 = 5.0;
const SCORES: [&str; 2] = ["flux", "(n,gamma)"];

/// Per material id, per replica, each nuclide's atom density.
type Densities = HashMap<u32, Vec<HashMap<String, f64>>>;

/// The cached W186 directory with the converter's covariance and resonance
/// parameters, or `None` when the fixture is missing.
fn w186(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("W186")?);
    let dir = tmp.join("W186.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached W186") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    let evaluation =
        endf::Material::from_str(&String::from_utf8(raw).expect("text")).expect("parses");
    yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance");
    yamc_convert::resonance_parameters::write_resonance_parameters(&evaluation, &dir)
        .expect("resonance parameters");
    Some(dir)
}

/// W186 at `scale` times 19.3 g/cm3, its covariance and resonance
/// parameters loaded.
fn tungsten(dir: &Path, scale: f64) -> Material {
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3 * scale),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read W186");
    m.ensure_covariance_loaded().expect("covariance");
    m.ensure_resonance_parameters_loaded()
        .expect("resonance parameters");
    m
}

/// `m` with replica `k`'s draw applied to W186, as the replica weights draw
/// it: its MF=33 field less the first-order MF=32 rows, and its resonance
/// parameters' change.
fn perturbed(m: &Material, k: u64) -> Material {
    let sampling = resonance_sampling(m);
    let (fields, _) = transport_fields(m, &sampled_names(&sampling));
    let unfielded = TransportField {
        reads: Default::default(),
        derived: Default::default(),
        field: None,
    };
    let field = fields
        .get("W186")
        .filter(|t| t.field.is_some())
        .unwrap_or(&unfielded);
    let draw = field.field.clone().map(|f| {
        Sampler::new(
            &std::collections::BTreeMap::from([("W186".to_string(), f)]),
            &[],
        )
        .draw(SEED, k)
    });
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
    let mut out = m.clone();
    out.nuclide_data.insert("W186".to_string(), Arc::new(w));
    out.invalidate_xs_cache();
    out
}

/// One transport of a sphere of `material` around a 1 keV point source,
/// each score tallied in the cell by track length.
fn run(
    material: Material,
    particles: usize,
    seed: u64,
    data: Option<(usize, Densities, bool)>,
    tracking: TrackingMode,
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
    let tallies: Vec<Arc<Tally>> = SCORES
        .iter()
        .map(|s| {
            let mut t = Tally::new();
            t.filters.push(Filter::Cell(CellFilter::from_id(1)));
            t.scores = vec![s.parse::<Score>().expect("score")];
            t.estimator = Estimator::TrackLength;
            t.reset_accumulation();
            Arc::new(t)
        })
        .collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e3], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.verbose = Verbose::silent();
    model.tracking_mode = tracking;
    let replicas = data.as_ref().map(|(r, _, _)| *r);
    model.replica_compositions = data.map(|(_, densities, cross_sections)| {
        Arc::new(yamc::model::ReplicaCompositions {
            densities,
            cross_sections,
            ..Default::default()
        })
    });
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed,
            threads: Some(1),
            data_uncertainty: replicas.map(|replicas| TransportDataUncertainty {
                seed: SEED,
                replicas,
            }),
            ..Default::default()
        })
        .expect("transport");
    tallies
}

/// Replica `k`'s mean and its Monte Carlo standard error.
fn replica(t: &Arc<Tally>, k: usize) -> (f64, f64) {
    let sums = t.get_replica_sums().expect("replica sums");
    let h = t.finalize().n_histories as f64;
    let (s1, s2) = (sums.sums[k], sums.sums[sums.replicas + k]);
    let var = (s2 - s1 * s1 / h) / (h - 1.0);
    (s1 / h, (var / h).sqrt())
}

/// Each replica of one transport on the nominal sphere, carrying its own
/// density of W186 as well as its own draw, scores what a rerun on that
/// density and draw scores, to within the two runs' Monte Carlo errors; and
/// a replica at the nominal density is the plain replica weight, to the
/// rounding of summing its changes in another order, whatever the others
/// carry.
#[test]
fn each_replica_is_the_run_on_its_own_composition() {
    on_its_own_composition(TrackingMode::Surface, true);
}

/// The same with the replicas' compositions alone, every replica on the
/// nominal cross sections: what the decay data's replicas carry.
#[test]
fn each_replica_is_the_run_on_its_own_composition_alone() {
    on_its_own_composition(TrackingMode::Surface, false);
}

/// The same under delta tracking, where a flight samples against a
/// majorant raised to bound every replica's total, its density included,
/// and a virtual collision carries `(Σ_maj - Σ'_t,k) / (Σ_maj - Σ_t)`.
#[test]
fn each_replica_is_the_run_on_its_own_composition_under_delta_tracking() {
    on_its_own_composition(TrackingMode::Woodcock, true);
}

fn on_its_own_composition(tracking: TrackingMode, draws: bool) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path()) else {
        eprintln!("skipping: W186 fixture not fetched");
        return;
    };
    const PARTICLES: usize = 10_000;
    let scales = [1.0, 0.8, 1.5];
    let nominal = tungsten(&dir, 1.0);
    let n = nominal.get_atoms_per_barn_cm().expect("densities")["W186"];
    let compositions = HashMap::from([(
        1u32,
        scales
            .iter()
            .map(|s| HashMap::from([("W186".to_string(), s * n)]))
            .collect(),
    )]);
    let carried = run(
        nominal.clone(),
        PARTICLES,
        5,
        Some((scales.len(), compositions, draws)),
        tracking,
    );
    // The replica weights alone, where there are draws for them to carry.
    let plain = draws.then(|| {
        run(
            nominal.clone(),
            PARTICLES,
            5,
            Some((scales.len(), HashMap::new(), draws)),
            tracking,
        )
    });
    // Under delta tracking the majorant, raised for the replicas'
    // densities, is part of the random walk, so the histories differ and
    // replica 0 is compared with its rerun like the others.
    if let Some(plain) = plain.as_ref().filter(|_| tracking == TrackingMode::Surface) {
        for (a, b) in carried.iter().zip(plain) {
            let ((x, _), (y, _)) = (replica(a, 0), replica(b, 0));
            assert!(
                (x - y).abs() <= 1e-12 * y.abs(),
                "replica 0 is at the nominal density: {x:e} against {y:e}"
            );
        }
    }
    for (k, scale) in scales.iter().enumerate() {
        let material = tungsten(&dir, *scale);
        let rerun = run(
            if draws {
                perturbed(&material, k as u64)
            } else {
                material
            },
            PARTICLES,
            7,
            None,
            tracking,
        );
        for ((t, r), score) in carried.iter().zip(&rerun).zip(SCORES) {
            let (x, sx) = replica(t, k);
            let (y, sy) = (r.get_mean()[0], r.get_std_dev()[0]);
            let z = (x - y) / (sx * sx + sy * sy).sqrt();
            eprintln!(
                "{tracking:?}: replica {k} at {scale} times the density, {score}: weights \
                 {x:.5e} ± {sx:.1e}, rerun {y:.5e} ± {sy:.1e}, z = {z:+.2}"
            );
            assert!(z.abs() < 4.0, "replica {k} {score}: z = {z}");
        }
        // The density moves the tallies by far more than their errors, so
        // agreeing is not agreeing with the nominal.
        if k > 0 {
            let moved = (rerun[1].get_mean()[0] / carried[1].get_mean()[0] - 1.0).abs();
            assert!(moved > 0.02, "replica {k} moves capture by {moved:e}");
        }
    }
}
