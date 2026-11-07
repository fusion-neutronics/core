//! The checks a build passes before it is given a manifest. They are about
//! the build, never about the physics of one nuclide: the evaluations are
//! published as they are.
//!
//! - Completeness: every evaluation the build was asked for converted, or is
//!   one its recipe declares NJOY cannot process. A declared one that
//!   converted after all is reported, so the list does not go stale.
//! - Read back: every folder reads through the reader a simulation uses, and
//!   holds the evaluation its name says.
//! - The chain: every subsection's tables decode, and when the decay data is
//!   there the subsections load together as a chain.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use yamc_nuclide::load_scope::{LoadScope, SectionScope};

use crate::chain::subsection_dir;
use crate::convert::{self, Outcome, Particle};
use crate::recipe::{Part, Recipe};

/// How many problems are listed before the rest are counted.
const SHOWN: usize = 20;

/// Check a conversion run against the recipe, returning the declared
/// failures that converted after all. `full` is whether the run was asked for
/// every evaluation, which is the only case where an absent declared failure
/// means anything.
pub fn completeness(
    recipe: &Recipe,
    outcome: &Outcome,
    root: &Path,
    full: bool,
) -> Result<Vec<String>, String> {
    if !outcome.not_started.is_empty() {
        return Err(format!(
            "the build was stopped with {} evaluations not converted; run it again to \
             resume where it stopped",
            outcome.not_started.len()
        ));
    }
    let declared: BTreeSet<&str> = recipe.unconvertible.iter().copied().collect();
    let unexpected: Vec<(&String, &String)> = outcome
        .failed
        .iter()
        .filter(|(key, _)| !declared.contains(key.as_str()))
        .collect();
    if !unexpected.is_empty() {
        let mut msg = format!(
            "{} of {}'s evaluations did not convert, and its recipe does not declare \
             {}:\n",
            unexpected.len(),
            recipe.keyword,
            if unexpected.len() == 1 { "it" } else { "them" }
        );
        for (key, reason) in unexpected.iter().take(SHOWN) {
            let first = reason.lines().next().unwrap_or("");
            msg.push_str(&format!("  - {key}: {first}\n"));
        }
        if unexpected.len() > SHOWN {
            msg.push_str(&format!("  and {} more\n", unexpected.len() - SHOWN));
        }
        msg.push_str(&format!(
            "Each one's full reason is in {}. Running the build again retries only \
             these.",
            root.join(".build").join("failed").display()
        ));
        return Err(msg);
    }
    if !full {
        return Ok(Vec::new());
    }
    let done: BTreeSet<&String> = outcome.converted.iter().chain(&outcome.reused).collect();
    Ok(recipe
        .unconvertible
        .iter()
        .filter(|key| done.contains(&key.to_string()))
        .map(|key| key.to_string())
        .collect())
}

/// Read back every folder of `particles` under `root` through the readers a
/// simulation uses, at most `workers` at once and within `memory_budget`
/// bytes (a full neutron read holds every table of a nuclide).
pub fn read_back(
    root: &Path,
    particles: &[Particle],
    workers: usize,
    memory_budget: u64,
) -> Result<usize, String> {
    let mut jobs = Vec::new();
    for particle in particles {
        let dir = root.join(particle.dir());
        if !dir.is_dir() {
            continue;
        }
        for entry in
            fs::read_dir(&dir).map_err(|e| format!("could not list {}: {e}", dir.display()))?
        {
            let path = entry
                .map_err(|e| format!("could not list {}: {e}", dir.display()))?
                .path();
            if path.is_dir() && path.extension().is_some_and(|x| x == "arrow") {
                let cost = convert::BASE_BYTES + 3 * folder_bytes(&path);
                jobs.push(ReadBack {
                    particle: *particle,
                    path,
                    cost,
                });
            }
        }
    }
    jobs.sort_by(|a, b| b.cost.cmp(&a.cost).then(a.path.cmp(&b.path)));
    let results = convert::run_pool(&jobs, workers, memory_budget, None, |job| {
        std::panic::catch_unwind(|| check_folder(job.particle, &job.path))
            .unwrap_or_else(|_| Err("the reader panicked".to_string()))
    });
    let problems: Vec<String> = jobs
        .iter()
        .zip(results)
        .filter_map(|(job, r)| match r {
            Some(Ok(())) => None,
            Some(Err(e)) => Some(format!("{}: {e}", job.path.display())),
            None => Some(format!("{}: not read", job.path.display())),
        })
        .collect();
    if problems.is_empty() {
        return Ok(jobs.len());
    }
    Err(listed(
        &format!("{} folders do not read back", problems.len()),
        &problems,
    ))
}

struct ReadBack {
    particle: Particle,
    path: PathBuf,
    cost: u64,
}

impl convert::HasCost for ReadBack {
    fn cost(&self) -> u64 {
        self.cost
    }
}

fn folder_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// The folder's stem, which is the name the release lists it under.
fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn check_folder(particle: Particle, path: &Path) -> Result<(), String> {
    let expected = stem(path);
    match particle {
        Particle::Neutron => {
            let nuclide =
                yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(path, &LoadScope::full())
                    .map_err(|e| e.to_string())?;
            if nuclide.name.as_deref() != Some(expected.as_str()) {
                return Err(format!("the data declares {:?}", nuclide.name));
            }
            if nuclide.load_scope.sections != SectionScope::Full {
                return Err(
                    "only the cross-section tables are there, so no simulation can \
                            transport through it"
                        .to_string(),
                );
            }
            if nuclide.available_temperatures.is_empty() {
                return Err("no temperatures".to_string());
            }
            if nuclide.reaction_mts().is_none_or(|m| m.is_empty()) {
                return Err("no reactions".to_string());
            }
        }
        Particle::Photon => {
            let element = yamc_element::photon_arrow::inspect_photon_interaction_from_arrow(path)
                .map_err(|e| e.to_string())?;
            if element.name != expected {
                return Err(format!("the data declares {:?}", element.name));
            }
            if element.energy.is_empty() {
                return Err("an empty element table".to_string());
            }
        }
        Particle::Photonuclear => {
            let summary = yamc_convert::photonuclear::inspect_photonuclear(path)
                .map_err(|e| e.to_string())?;
            if summary.name != expected {
                return Err(format!("the data declares {:?}", summary.name));
            }
            if summary.energy_points.is_empty() {
                return Err("no reactions".to_string());
            }
        }
    }
    Ok(())
}

/// Check the chain subsections among `parts` under `root`: each holds its
/// tables and every table decodes, and when the decay subsection is there
/// the subsections load together as the chain a run reads.
pub fn chain(root: &Path, parts: &[Part]) -> Result<(), String> {
    let wanted: Vec<Part> = parts.iter().copied().filter(|p| p.is_chain()).collect();
    let mut problems = Vec::new();
    for part in &wanted {
        let dir = subsection_dir(root, *part);
        let primary = match part {
            Part::Decay => "nuclides.arrow",
            Part::Reactions => "reactions.arrow",
            Part::FissionYields => "fission_yields.arrow",
            Part::Branching => "branching.arrow",
            _ => unreachable!("not a chain part"),
        };
        if !dir.join(primary).is_file() {
            problems.push(format!("{}: no {primary}", dir.display()));
            continue;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                problems.push(format!("could not list {}: {e}", dir.display()));
                continue;
            }
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().is_some_and(|x| x == "arrow")
                && let Err(e) = decode(&path)
            {
                problems.push(format!("{}: {e}", path.display()));
            }
        }
    }
    if problems.is_empty() && wanted.contains(&Part::Decay) {
        let present = |p: Part| {
            let dir = subsection_dir(root, p);
            wanted.contains(&p).then_some(dir)
        };
        let reactions = present(Part::Reactions);
        let yields = present(Part::FissionYields);
        let branching = present(Part::Branching);
        let overlay: Vec<(&str, &Path)> = branching
            .as_deref()
            .map(|d| vec![("this build", d)])
            .unwrap_or_default();
        if let Err(e) = yani::chain_arrow::parse_chain_parts(
            &subsection_dir(root, Part::Decay),
            reactions.as_deref(),
            yields.as_deref(),
            &overlay,
        ) {
            problems.push(format!("the chain does not load: {e}"));
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(listed("the chain does not read back", &problems))
}

/// Read every record batch of an Arrow IPC file.
fn decode(path: &Path) -> Result<(), String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let reader = arrow_ipc::reader::FileReader::try_new(file, None).map_err(|e| e.to_string())?;
    for batch in reader {
        batch.map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn listed(head: &str, problems: &[String]) -> String {
    let mut msg = format!("{head}:\n");
    for p in problems.iter().take(SHOWN) {
        msg.push_str(&format!("  - {p}\n"));
    }
    if problems.len() > SHOWN {
        msg.push_str(&format!("  and {} more\n", problems.len() - SHOWN));
    }
    msg.trim_end().to_string()
}
