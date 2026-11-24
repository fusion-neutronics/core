//! How long a nuclide's nuclear-data uncertainty setup takes: its MF=33
//! sampler (the field over every reaction transport perturbs, repaired,
//! transformed and factorized), and its sampled resonance parameters' change
//! per replica, as a transport run tabulates them.
//!
//! Run as:
//!
//! ```text
//! cargo run --release -p yani-transmute --example covariance_setup_perf -- <dir>...
//! RESONANCE_REPLICAS=256 cargo run --release -p yani-transmute --example covariance_setup_perf -- <dir>
//! ```
//!
//! where each directory is a release's `neutron/<name>.arrow`, carrying
//! `covariance.arrow` (and `resonance_parameters.arrow` for the second form).
//! With `CHECK_DRAWS=1` the first form also compares each nuclide's sampled
//! covariance with the empirical covariance of 4000 of its draws, and with
//! `JOINT=1` it puts every nuclide in one material and builds one sampler
//! over all of them, drawing the ones a cross-material block links jointly.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use yamc_materials::Material;
use yani_transmute::covariance_fold::{transport_fields, CellField};
use yani_transmute::covariance_sample::Sampler;
use yani_transmute::cross_material::CrossMaterial;
use yani_transmute::resonance_rates::TransportResonance;

/// `dir`'s nuclide name and a material holding one atom per barn-cm of it.
fn material(dir: &str) -> Option<(String, Material)> {
    let name = Path::new(dir)
        .file_name()?
        .to_string_lossy()
        .trim_end_matches(".arrow")
        .to_string();
    let mut m = Material::new(HashMap::from([(name.clone(), 1.0)]), "atom", "sum", None).ok()?;
    m.density = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([(name.clone(), dir.to_string())]), None)
        .ok()?;
    Some((name, m))
}

fn field(dir: &str) -> Option<(String, CellField)> {
    let (name, mut m) = material(dir)?;
    m.ensure_covariance_loaded().ok()?;
    let (fields, _) = transport_fields(&m, &BTreeSet::new());
    let cells = fields.into_values().next()?.field?;
    Some((name, cells))
}

/// The largest difference between the covariance of `name`'s draws and the
/// empirical one of 4000 of them, in correlation units, and the largest
/// relative difference of a variance. Taken on `ln m`, which is Gaussian with
/// covariance `ln(1 + C)` for `C` the sampled covariance of the multipliers
/// `m`, so the statistical error is `1/sqrt(4000)` per entry rather than
/// what a lognormal's heavy tail gives a wide channel.
fn check_draws(sampler: &Sampler, name: &str, n: usize) {
    let (sampled, _) = sampler.sampled_covariance(name).expect("field");
    let relative: Vec<f64> = sampled.iter().map(|v| v.ln_1p()).collect();
    let draws = 4000;
    let mut x = faer::Mat::<f64>::zeros(n, draws);
    for k in 0..draws {
        let d = sampler.draw(17, k as u64);
        for (i, v) in d.relative(name).expect("draw").iter().enumerate() {
            x[(i, k)] = v.ln_1p() + 0.5 * relative[i * n + i];
        }
    }
    let mut c = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(
        c.as_mut(),
        faer::Accum::Replace,
        x.as_ref(),
        x.transpose(),
        1.0 / draws as f64,
        faer::Par::Seq,
    );
    let mut worst_correlation = 0.0_f64;
    let mut worst_variance = 0.0_f64;
    for i in 0..n {
        let si = relative[i * n + i];
        if si <= 0.0 {
            continue;
        }
        worst_variance = worst_variance.max((c[(i, i)] / si - 1.0).abs());
        for j in 0..n {
            let sj = relative[j * n + j];
            if sj > 0.0 {
                let d = (c[(i, j)] - relative[i * n + j]).abs() / (si * sj).sqrt();
                worst_correlation = worst_correlation.max(d);
            }
        }
    }
    println!(
        "{name}: {draws} draws, largest |empirical - sampled| {worst_correlation:.4} in \
         correlation units, largest relative variance error {worst_variance:.4}"
    );
}

/// `replicas` draws of `dir`'s sampled resonance ranges on its own grid.
fn resonance(dir: &str, replicas: u64) {
    use rayon::prelude::*;
    let (name, mut m) = material(dir).expect("nuclide");
    m.ensure_resonance_parameters_loaded().expect("parameters");
    let nd = &m.nuclide_data[&name];
    let start = Instant::now();
    let r = TransportResonance::new(
        &name,
        nd.resonance_parameters.as_ref().expect("parameters"),
        "294",
    )
    .expect("prepared");
    println!("{name}: prepared in {:.3} s", start.elapsed().as_secs_f64());
    let grid = nd.energy.as_ref().expect("grid")["294"].as_slice().to_vec();
    let start = Instant::now();
    let all: Vec<_> = (0..replicas)
        .into_par_iter()
        .map(|k| r.changes(1, k, &grid).expect("changes"))
        .collect();
    println!(
        "{name}: {replicas} replicas on {} points in {:.3} s",
        grid.len(),
        start.elapsed().as_secs_f64()
    );
    let checksum: f64 = all
        .iter()
        .flat_map(|d| d.values().flatten())
        .map(|v| v.abs())
        .sum();
    println!("{name}: checksum {checksum:.17e}");
}

/// Every nuclide of `dirs` in one material, and one sampler over all of them
/// with the cross-material blocks between them applied.
fn joint(dirs: &[String]) {
    let mut names = HashMap::new();
    let mut paths = HashMap::new();
    for dir in dirs {
        let (name, _) = material(dir).expect("nuclide");
        names.insert(name.clone(), 1.0);
        paths.insert(name, dir.clone());
    }
    let mut m = Material::new(names, "atom", "sum", None).expect("material");
    m.density = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&paths, None).expect("read");
    m.ensure_covariance_loaded().expect("covariance");
    let start = Instant::now();
    let (fields, _) = transport_fields(&m, &BTreeSet::new());
    let cells: BTreeMap<String, CellField> = fields
        .into_iter()
        .filter_map(|(name, t)| Some((name, t.field?)))
        .collect();
    let cross = CrossMaterial::new(&[&m]);
    let partners = cross.transport_fields(&BTreeSet::new());
    let sampler = Sampler::coupled(&cells, &[], &cross, &partners);
    println!(
        "{} nuclides, {} cross-material links, built in {:.3} s",
        cells.len(),
        if cross.is_empty() { "no" } else { "with" },
        start.elapsed().as_secs_f64()
    );
    for (group, adjustment) in sampler.joint_adjustments() {
        println!("{group}: {adjustment:?}");
    }
}

fn main() {
    if std::env::var_os("JOINT").is_some() {
        joint(&std::env::args().skip(1).collect::<Vec<_>>());
        return;
    }
    if let Ok(replicas) = std::env::var("RESONANCE_REPLICAS") {
        for dir in std::env::args().skip(1) {
            resonance(&dir, replicas.parse().expect("replicas"));
        }
        return;
    }
    for dir in std::env::args().skip(1) {
        let Some((name, cells)) = field(&dir) else {
            eprintln!("{dir}: no covariance field");
            continue;
        };
        let n = cells.relative_cells.len();
        let start = Instant::now();
        let sampler = Sampler::new(&BTreeMap::from([(name.clone(), cells)]), &[]);
        println!(
            "{name}: {n} relative cells, built in {:.3} s",
            start.elapsed().as_secs_f64()
        );
        if std::env::var_os("CHECK_DRAWS").is_some() {
            check_draws(&sampler, &name, n);
        }
    }
}
