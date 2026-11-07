//! Converting every identified evaluation into the folders of a release:
//! `<path>/neutron/<Name>.arrow/`, `<path>/photon/<Element>.arrow/` and
//! `<path>/photonuclear/<Name>.arrow/`.
//!
//! One pool runs in this process. Evaluations are converted largest first,
//! and each one reserves an estimate of its working set against the memory
//! that is free, so a run never starts more NJOY jobs than fit. Running out of
//! memory is not a slowdown: the kernel kills the build mid-evaluation.
//!
//! A folder is reused, not converted again, when its `version.json` says it
//! was finished for this library and release and the tables a reader needs
//! are there. Anything else under that name is removed and converted again,
//! so a run killed partway through is resumed by running it again.
//!
//! An evaluation that fails, by an error or a panic, does not stop the others.
//! Its reason is returned and written to `<path>/.build/failed/`, and the next
//! run tries it again. Deciding which failures a library may have is not done
//! here: that belongs to the gates that check the whole build.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use yamc_convert::entry::{self, PhotonTabulations, Provenance};
use yamc_nuclide::storage::release;

use crate::identify::Material;
use crate::recipe::Sublibrary;

/// A particle folder of a release, and what is converted into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Particle {
    /// The neutron sublibrary, through NJOY, as full transport data with
    /// every covariance section the evaluation has.
    Neutron,
    /// The photo-atomic sublibrary, with the atomic relaxation of the same
    /// element where the library has it.
    Photon,
    /// The photonuclear sublibrary, kept as evaluated (no NJOY).
    Photonuclear,
}

impl Particle {
    /// The folder under the release root.
    pub fn dir(self) -> &'static str {
        match self {
            Particle::Neutron => "neutron",
            Particle::Photon => "photon",
            Particle::Photonuclear => "photonuclear",
        }
    }

    /// The tables every finished folder of this particle has. A table an
    /// evaluation may lack (`urr.arrow`, `total_nu.arrow`, `subshells.arrow`,
    /// the covariance sections) proves nothing by its absence, so it is not
    /// here.
    fn required_tables(self) -> &'static [&'static str] {
        match self {
            Particle::Neutron => &[
                "nuclide.arrow",
                "energy.arrow",
                "reactions.arrow",
                "products.arrow",
                "distributions.arrow",
            ],
            Particle::Photon => &["element.arrow"],
            Particle::Photonuclear => &["photonuclear.arrow", "photonuclear_sections.arrow"],
        }
    }

    /// The range indexes `version.json` must carry. The converter writes them
    /// best effort, but a published neutron folder without them makes every
    /// activation reader fetch whole files, so a build requires them.
    fn required_indexes(self) -> &'static [&'static str] {
        match self {
            Particle::Neutron => &["reaction_ranges", "energy_ranges"],
            Particle::Photon | Particle::Photonuclear => &[],
        }
    }
}

/// How a conversion run is set up.
pub struct Options<'a> {
    /// The library stamp every folder records, which is its keyword.
    pub library: &'a str,
    /// The release stamp every folder records.
    pub release: &'a str,
    /// The NJOY to run for neutron evaluations.
    pub njoy_exec: &'a str,
    /// Temperatures in Kelvin for the neutron data.
    pub temperatures: &'a [f64],
    /// The particle folders to convert.
    pub particles: &'a [Particle],
    /// Only these nuclides and elements, or all when `None`. A name is a
    /// nuclide (`"Fe56"`, `"Am242_m1"`) or an element symbol, which means
    /// every evaluation of that element in every particle (`"Fe"`).
    pub nuclides: Option<&'a [String]>,
    /// The Compton profiles, density effect and bremsstrahlung tables a
    /// photon folder carries beside its evaluation. Required when photons are
    /// converted, so a library is never published without them.
    pub tabulations: Option<PhotonTabulations<'a>>,
    /// The memory the pool may reserve, in bytes. `None` measures what is
    /// free when the run starts.
    pub memory_budget: Option<u64>,
    /// The most conversions to run at once. `None` is the core count.
    pub workers: Option<usize>,
    /// Set from another thread (the binding's Ctrl-C check) to start nothing
    /// more. Conversions already running finish.
    pub stop: Option<&'a AtomicBool>,
}

/// What a run did, by `"<particle>/<Name>"`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Converted by this run.
    pub converted: BTreeSet<String>,
    /// Already finished, from an earlier run.
    pub reused: BTreeSet<String>,
    /// Not converted, with the reason.
    pub failed: BTreeMap<String, String>,
    /// Not started because the run was stopped.
    pub not_started: BTreeSet<String>,
}

/// One conversion: an input file and the folders it writes.
#[derive(Debug, Clone)]
struct Job {
    particle: Particle,
    input: PathBuf,
    /// The atomic relaxation file, for a photon job.
    relaxation: Option<PathBuf>,
    /// The names of the folders it writes. One, except for a photo-atomic
    /// file of several elements (FENDL bundles them).
    names: Vec<String>,
    /// Estimated working set, in bytes.
    cost: u64,
}

impl Job {
    fn keys(&self) -> impl Iterator<Item = String> + '_ {
        self.names
            .iter()
            .map(|n| format!("{}/{n}", self.particle.dir()))
    }
}

/// Convert every material of `materials` that `options` asks for into the
/// release folder `path`.
///
/// Refuses before converting anything when a name in `options.nuclides`
/// matches no material, when photons are asked for without the tabulations,
/// or when a photo-atomic file's elements are split across more than one
/// atomic relaxation file.
pub fn convert(materials: &[Material], path: &Path, options: &Options) -> Result<Outcome, String> {
    let particles: BTreeSet<Particle> = options.particles.iter().copied().collect();
    if particles.contains(&Particle::Photon) && options.tabulations.is_none() {
        return Err(
            "photon data needs the Compton profile, density effect and bremsstrahlung \
             tabulations, and none were given"
                .to_string(),
        );
    }
    let selected = select(materials, options.nuclides)?;
    let jobs = jobs(&selected, &particles)?;

    let mut outcome = Outcome::default();
    let mut todo = Vec::new();
    for job in jobs {
        let out = path.join(job.particle.dir());
        let finished = job
            .names
            .iter()
            .all(|n| is_finished(&out.join(format!("{n}.arrow")), job.particle, options));
        if finished {
            outcome.reused.extend(job.keys());
        } else {
            todo.push(job);
        }
    }

    let failed_dir = path.join(".build").join("failed");
    for job in &todo {
        let out = path.join(job.particle.dir());
        std::fs::create_dir_all(&out)
            .map_err(|e| format!("could not create {}: {e}", out.display()))?;
        for name in &job.names {
            // A folder that is not finished is what an interrupted run left;
            // the converter writes into it, so it goes first.
            let dir = out.join(format!("{name}.arrow"));
            if dir.exists() {
                std::fs::remove_dir_all(&dir)
                    .map_err(|e| format!("could not remove {}: {e}", dir.display()))?;
            }
            let record = failed_dir.join(failure_file(job.particle, name));
            if record.exists() {
                std::fs::remove_file(&record)
                    .map_err(|e| format!("could not remove {}: {e}", record.display()))?;
            }
        }
    }

    // Largest first: the evaluations that take longest start first, so the
    // run does not end with one of them going alone.
    todo.sort_by(|a, b| b.cost.cmp(&a.cost).then(a.input.cmp(&b.input)));
    let budget = match options.memory_budget {
        Some(b) => b,
        None => available_memory().saturating_mul(SAFETY_PERCENT) / 100,
    };
    let workers = options
        .workers
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);
    let results = run_pool(&todo, workers, budget, options.stop, |job| {
        run_job(job, path, options)
    });

    for (job, result) in todo.iter().zip(results) {
        match result {
            None => outcome.not_started.extend(job.keys()),
            Some(Ok(())) => outcome.converted.extend(job.keys()),
            Some(Err(reason)) => {
                std::fs::create_dir_all(&failed_dir)
                    .map_err(|e| format!("could not create {}: {e}", failed_dir.display()))?;
                for (name, key) in job.names.iter().zip(job.keys()) {
                    // What a failed conversion wrote before it stopped is not
                    // a folder of this release, so nothing after this stage
                    // may find it.
                    let dir = path.join(job.particle.dir()).join(format!("{name}.arrow"));
                    if dir.exists() {
                        std::fs::remove_dir_all(&dir)
                            .map_err(|e| format!("could not remove {}: {e}", dir.display()))?;
                    }
                    let record = failed_dir.join(failure_file(job.particle, name));
                    let text = format!("{}\n\n{reason}\n", job.input.display());
                    std::fs::write(&record, text)
                        .map_err(|e| format!("could not write {}: {e}", record.display()))?;
                    outcome.failed.insert(key, reason.clone());
                }
            }
        }
    }
    Ok(outcome)
}

/// The file in `.build/failed/` that records why one folder was not
/// converted.
fn failure_file(particle: Particle, name: &str) -> String {
    format!("{}-{name}.txt", particle.dir())
}

/// The materials `nuclides` names, or every one. Refuses a name that matches
/// nothing, since a typo would otherwise build a library short of it.
fn select<'m>(
    materials: &'m [Material],
    nuclides: Option<&[String]>,
) -> Result<Vec<&'m Material>, String> {
    let Some(wanted) = nuclides else {
        return Ok(materials.iter().collect());
    };
    let mut matched = BTreeSet::new();
    let mut out = Vec::new();
    for m in materials {
        let element = element_of(m);
        let hit = wanted
            .iter()
            .find(|w| **w == m.name || Some(w.as_str()) == element);
        if let Some(w) = hit {
            matched.insert(w.as_str());
            out.push(m);
        }
    }
    let missing: Vec<&str> = wanted
        .iter()
        .map(String::as_str)
        .filter(|w| !matched.contains(w))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "no evaluation in this library is named {}",
            missing.join(", ")
        ));
    }
    Ok(out)
}

/// The element symbol of a material.
fn element_of(m: &Material) -> Option<&'static str> {
    usize::try_from(m.header.za / 1000)
        .ok()
        .and_then(|z| endf::data::ATOMIC_SYMBOL.get(z).copied())
}

/// The conversions the selected materials make for `particles`.
fn jobs(materials: &[&Material], particles: &BTreeSet<Particle>) -> Result<Vec<Job>, String> {
    let mut jobs = Vec::new();
    // A photo-atomic file may hold several elements, and is converted whole,
    // so its materials are gathered by file. The relaxation files are found
    // by element.
    let mut photo_files: BTreeMap<&Path, Vec<&Material>> = BTreeMap::new();
    let mut relaxation: BTreeMap<&str, &Path> = BTreeMap::new();
    for m in materials {
        match m.sublibrary {
            Sublibrary::Neutron if particles.contains(&Particle::Neutron) => {
                // The free neutron (ZA 1, in ENDF/B's neutron sublibrary) has
                // no bound cross sections to process.
                if m.header.za < 1000 {
                    continue;
                }
                jobs.push(Job {
                    particle: Particle::Neutron,
                    input: m.path.clone(),
                    relaxation: None,
                    names: vec![m.name.clone()],
                    cost: cost(&m.path),
                });
            }
            Sublibrary::Photonuclear if particles.contains(&Particle::Photonuclear) => {
                jobs.push(Job {
                    particle: Particle::Photonuclear,
                    input: m.path.clone(),
                    relaxation: None,
                    names: vec![m.name.clone()],
                    cost: cost(&m.path),
                });
            }
            Sublibrary::Photoatomic if particles.contains(&Particle::Photon) => {
                photo_files.entry(&m.path).or_default().push(m);
            }
            Sublibrary::AtomicRelaxation => {
                relaxation.insert(&m.name, &m.path);
            }
            _ => {}
        }
    }
    for (file, elements) in photo_files {
        let paired: BTreeSet<&Path> = elements
            .iter()
            .filter_map(|m| relaxation.get(m.name.as_str()).copied())
            .collect();
        if paired.len() > 1 {
            return Err(format!(
                "{} holds elements whose atomic relaxation is in {} files; one \
                 photo-atomic file is converted with one relaxation file",
                file.display(),
                paired.len()
            ));
        }
        jobs.push(Job {
            particle: Particle::Photon,
            input: file.to_path_buf(),
            relaxation: paired.into_iter().next().map(Path::to_path_buf),
            names: elements.iter().map(|m| m.name.clone()).collect(),
            cost: cost(file),
        });
    }
    Ok(jobs)
}

/// Whether `dir` is a folder a resume may keep: `version.json` is there and
/// says it was converted in this format for this library and release, with
/// the range indexes a build requires, and the required tables are there.
fn is_finished(dir: &Path, particle: Particle, options: &Options) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("version.json")) else {
        return false;
    };
    let Ok(marker) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    marker["format_version"] == nuclear_data_schema::FORMAT_VERSION
        && marker["library"] == options.library
        && marker["data_version"] == options.release
        && particle
            .required_indexes()
            .iter()
            .all(|k| !marker[*k].is_null())
        && particle
            .required_tables()
            .iter()
            .all(|t| dir.join(t).is_file())
}

/// Convert one job, catching a panic as a failure of that job alone.
fn run_job(job: &Job, path: &Path, options: &Options) -> Result<(), String> {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        convert_one(job, path, options)
    }));
    match caught {
        Ok(result) => result,
        Err(panic) => Err(format!("the converter panicked: {}", panic_message(&panic))),
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "no message".to_string()
    }
}

fn convert_one(job: &Job, path: &Path, options: &Options) -> Result<(), String> {
    let out = path.join(job.particle.dir());
    let provenance = Provenance {
        library: options.library.to_string(),
        data_version: options.release.to_string(),
        created_utc: release::utc_timestamp(std::time::SystemTime::now())?,
    };
    let written: Vec<PathBuf> = match job.particle {
        Particle::Neutron => {
            let source = entry::Source::Endf {
                path: &job.input,
                njoy_exec: options.njoy_exec,
                temperatures: options.temperatures.to_vec(),
            };
            // Covariance always: downloads are per section, so a reader that
            // never asks for uncertainty never fetches it, and one that does
            // cannot use data that was never converted.
            vec![
                entry::convert_neutron_transport(&source, &out, &provenance, true)
                    .map_err(|e| e.to_string())?,
            ]
        }
        Particle::Photon => entry::convert_photon(
            &job.input,
            job.relaxation.as_deref(),
            options.tabulations.as_ref(),
            &out,
            &provenance,
        )
        .map_err(|e| e.to_string())?,
        Particle::Photonuclear => {
            vec![
                yamc_convert::photonuclear::convert_photonuclear(&job.input, &out, &provenance)
                    .map_err(|e| e.to_string())?,
            ]
        }
    };

    // The converter names its folder from the evaluation it read. A folder
    // under another name than the header gave would be one the resume and the
    // gates never look at, so it is a failure, not a silent extra.
    let expected: BTreeSet<PathBuf> = job
        .names
        .iter()
        .map(|n| out.join(format!("{n}.arrow")))
        .collect();
    let got: BTreeSet<PathBuf> = written.into_iter().collect();
    if got != expected {
        let names = |s: &BTreeSet<PathBuf>| {
            s.iter()
                .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(format!(
            "the converter wrote {} where the headers name {}",
            names(&got),
            names(&expected)
        ));
    }
    for dir in &expected {
        if !is_finished(dir, job.particle, options) {
            return Err(format!(
                "{} was written without its version.json, a required table or a \
                 range index",
                dir.display()
            ));
        }
    }
    Ok(())
}

// What a conversion is assumed to hold at its peak, from the size of its
// input file. Fitted by nuclear_data_generation_scripts to two measured
// points: about 0.7 GB resident on TENDL-2017's typical 4.4 MB files, and
// about 4 GB at the peak of an ENDF/B-VIII.1 build whose largest file is
// 40.5 MB. The constant is what a worker holds before it reads anything; the
// slope is the evaluation and NJOY's work on it.
const BASE_BYTES: u64 = 200 * 1024 * 1024;
const BYTES_PER_SOURCE_BYTE: f64 = 0.115 * 1024.0;

/// Of the free memory, the share the pool reserves. The rest is headroom for
/// the estimate being low, which is the direction that kills a build.
const SAFETY_PERCENT: u64 = 80;

/// The estimated peak working set of converting `input`.
fn cost(input: &Path) -> u64 {
    let size = std::fs::metadata(input).map_or(0, |m| m.len());
    BASE_BYTES + (size as f64 * BYTES_PER_SOURCE_BYTE) as u64
}

/// Memory free for the build to use, in bytes: what the system reports as
/// available, not the installed total, since the build is not alone on the
/// machine.
fn available_memory() -> u64 {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.available_memory()
}

/// Run `work` on every job with at most `workers` at once, starting them in
/// order and only while their costs fit `budget` together. A job larger than
/// the whole budget runs when nothing else is running, so it is never
/// skipped. Returns each job's result in order, or `None` for a job not
/// started because `stop` was set.
fn run_pool<J: HasCost + Sync, R: Send>(
    jobs: &[J],
    workers: usize,
    budget: u64,
    stop: Option<&AtomicBool>,
    work: impl Fn(&J) -> R + Sync,
) -> Vec<Option<R>> {
    struct State {
        next: usize,
        reserved: u64,
        running: usize,
    }
    let state = Mutex::new(State {
        next: 0,
        reserved: 0,
        running: 0,
    });
    let freed = Condvar::new();
    let results: Vec<Mutex<Option<R>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    let stopped = || stop.is_some_and(|s| s.load(Ordering::Relaxed));

    std::thread::scope(|scope| {
        for _ in 0..workers.min(jobs.len()) {
            scope.spawn(|| loop {
                let (index, cost) = {
                    let mut s = state.lock().unwrap();
                    let cost = loop {
                        if s.next >= jobs.len() || stopped() {
                            return;
                        }
                        let cost = jobs[s.next].cost();
                        if s.running == 0 || s.reserved.saturating_add(cost) <= budget {
                            break cost;
                        }
                        s = freed.wait(s).unwrap();
                    };
                    let index = s.next;
                    s.next += 1;
                    s.running += 1;
                    s.reserved += cost;
                    (index, cost)
                };
                let result = work(&jobs[index]);
                *results[index].lock().unwrap() = Some(result);
                let mut s = state.lock().unwrap();
                s.running -= 1;
                s.reserved -= cost;
                freed.notify_all();
            });
        }
    });
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap())
        .collect()
}

/// What the pool reserves for a job.
trait HasCost {
    fn cost(&self) -> u64;
}

impl HasCost for Job {
    fn cost(&self) -> u64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Cost(u64);
    impl HasCost for Cost {
        fn cost(&self) -> u64 {
            self.0
        }
    }

    /// Runs the jobs and returns the most that ran at once.
    fn peak(costs: &[u64], workers: usize, budget: u64) -> usize {
        let jobs: Vec<Cost> = costs.iter().map(|&c| Cost(c)).collect();
        let now = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let results = run_pool(&jobs, workers, budget, None, |_| {
            let n = now.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(n, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(30));
            now.fetch_sub(1, Ordering::SeqCst);
        });
        assert!(results.iter().all(Option::is_some), "every job ran");
        most.into_inner()
    }

    #[test]
    fn the_pool_runs_no_more_than_the_budget_fits() {
        assert_eq!(peak(&[10; 8], 8, 30), 3);
    }

    #[test]
    fn the_pool_runs_no_more_than_its_workers() {
        assert_eq!(peak(&[1; 8], 2, 1000), 2);
    }

    #[test]
    fn a_job_larger_than_the_budget_still_runs_alone() {
        assert_eq!(peak(&[500, 10, 10], 4, 100), 2);
        assert_eq!(peak(&[500], 4, 100), 1);
    }

    #[test]
    fn a_stopped_pool_starts_nothing_more() {
        let jobs: Vec<Cost> = (0..6).map(|_| Cost(1)).collect();
        let stop = AtomicBool::new(false);
        let results = run_pool(&jobs, 1, 100, Some(&stop), |_| {
            stop.store(true, Ordering::Relaxed);
        });
        assert!(results[0].is_some());
        assert!(results[1..].iter().all(Option::is_none));
    }

    #[test]
    fn a_bigger_file_costs_more() {
        let dir = std::env::temp_dir().join(format!("ndb-cost-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let small = dir.join("small");
        let big = dir.join("big");
        std::fs::write(&small, vec![b' '; 1024]).unwrap();
        std::fs::write(&big, vec![b' '; 1024 * 1024]).unwrap();
        assert!(cost(&big) > cost(&small));
        assert!(cost(&small) >= BASE_BYTES);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
