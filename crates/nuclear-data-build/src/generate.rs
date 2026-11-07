//! A whole build, from a library keyword to a release folder with its
//! manifest: the one call the Python `generate_library` makes.
//!
//! In order, every check that can fail fast runs before anything is
//! downloaded:
//!
//! 1. the arguments, the folder (not inside the download cache, not someone
//!    else's folder), and an exclusive lock on `<path>/.build/lock`;
//! 2. the build's state in `<path>/.build/state.json`: the release stamp
//!    (the UTC date the first run started, kept on resume) and the builder
//!    that wrote the folder, since a rerun by another version would mix two
//!    converters' output under one manifest;
//! 3. the NJOY the library needs, when neutron data is built, and the free
//!    disk a full build needs.
//!
//! Then the sources are fetched, unpacked and identified, the evaluations are
//! converted, the chain is built, the gates run, each particle gets its
//! `index.json`, and the manifest is written last. A folder with a manifest
//! is a complete release: the manifest of an earlier run is removed when a
//! run starts, and none is written when a gate fails.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};
use yamc_convert::entry::PhotonTabulations;
use yamc_nuclide::storage::release;

use crate::convert::{self, Particle};
use crate::recipe::{self, Part, Recipe, Sublibrary};
use crate::sources::{self, Search};
use crate::{chain, extract, gates, identify, index};

/// The temperatures neutron data is processed at by default, in Kelvin. The
/// 0 K grid is kept beside them.
pub const TEMPERATURES: [f64; 6] = [250.0, 293.6, 600.0, 900.0, 1200.0, 2500.0];

/// What to build.
pub struct Request<'a> {
    /// One of the library keywords with a recipe.
    pub keyword: &'a str,
    /// The release folder, created if absent. Running again resumes.
    pub path: &'a Path,
    /// The parts to build, or `None` for every part the published release
    /// carries.
    pub parts: Option<&'a [Part]>,
    /// Only these nuclides and elements of the per-evaluation parts, or all
    /// when `None`. The chain is always built whole.
    pub nuclides: Option<&'a [String]>,
    /// Neutron temperatures in Kelvin, or `None` for [`TEMPERATURES`].
    pub temperatures: Option<&'a [f64]>,
    /// The NJOY to run.
    pub njoy_exec: &'a str,
    /// A folder searched first for source files fetched by hand.
    pub downloads: Option<&'a Path>,
    /// What is building, recorded in the folder and the manifest, such as
    /// `"yani-python 0.22.1"`. A rerun by another builder is refused.
    pub builder: &'a str,
    /// Set from another thread to start nothing more.
    pub stop: Option<&'a AtomicBool>,
    /// Told what the build is doing, one line at a time.
    pub progress: Option<&'a (dyn Fn(&str) + Sync)>,
}

/// What a finished build holds.
#[derive(Debug, Clone)]
pub struct Report {
    pub keyword: String,
    pub release: String,
    pub path: PathBuf,
    pub parts: Vec<Part>,
    /// `"<particle>/<Name>"` converted by this run.
    pub converted: BTreeSet<String>,
    /// Already finished by an earlier run.
    pub reused: BTreeSet<String>,
    /// The declared failures, with their reasons.
    pub not_converted: BTreeMap<String, String>,
    /// Declared failures that converted after all: the recipe's list has
    /// gone stale.
    pub stale_declarations: Vec<String>,
    /// The NJOY version the neutron data was processed with.
    pub njoy: Option<String>,
    pub chain_nuclides: usize,
    pub manifest_sha256: String,
    pub file_count: usize,
    pub total_bytes: u64,
}

/// The builder's record in `<path>/.build/state.json`.
const STATE: &str = "state.json";

/// Build the library `request` names. See the module documentation for the
/// order things happen in.
pub fn generate(request: &Request) -> Result<Report, String> {
    let say = |line: &str| {
        if let Some(progress) = request.progress {
            progress(line);
        }
    };
    let recipe = recipe::recipe(request.keyword)?;
    let parts = parts(recipe, request.parts)?;
    let temperatures = request.temperatures.unwrap_or(&TEMPERATURES);
    check_temperatures(temperatures)?;
    let root = request.path;
    check_folder(root)?;

    let build = root.join(".build");
    fs::create_dir_all(&build).map_err(|e| format!("could not create {}: {e}", build.display()))?;
    let _lock = lock(&build)?;
    let release = state(&build, recipe.keyword, request.builder)?;
    let manifest = root.join("manifest.json");
    if manifest.exists() {
        fs::remove_file(&manifest)
            .map_err(|e| format!("could not remove {}: {e}", manifest.display()))?;
    }

    let neutron = parts.contains(&Part::Neutron);
    let njoy = if neutron {
        let found =
            endf::njoy::require(request.njoy_exec, recipe.njoy).map_err(|e| e.to_string())?;
        Some(found.to_string())
    } else {
        None
    };
    if neutron && request.nuclides.is_none() {
        check_disk(root, recipe.disk_bytes)?;
    }

    say(&format!("{}: fetching the sources", recipe.keyword));
    let wanted = sublibraries(recipe, &parts);
    let selected: Vec<recipe::Source> = recipe
        .sources
        .iter()
        .filter(|s| wanted.contains(&s.sublibrary))
        .copied()
        .collect();
    let scoped = Recipe {
        sources: &selected,
        ..*recipe
    };
    let fetched = sources::fetch(&scoped, root, &Search::new(request.downloads))?;
    say(&format!("{}: unpacking the sources", recipe.keyword));
    let folders = extract::extract(&scoped, &fetched, root)?;
    let materials = identify::identify(&folders)?;

    let particles: Vec<Particle> = parts.iter().filter_map(|p| particle(*p)).collect();
    let tabulations_dir = build.join("tabulations");
    let tabulations = if particles.contains(&Particle::Photon) {
        Some(write_tabulations(&tabulations_dir)?)
    } else {
        None
    };
    if !particles.is_empty() {
        let names: Vec<&str> = particles.iter().map(|p| p.dir()).collect();
        say(&format!(
            "{}: converting ({})",
            recipe.keyword,
            names.join(", ")
        ));
    }
    let options = convert::Options {
        library: recipe.keyword,
        release: &release,
        njoy_exec: request.njoy_exec,
        temperatures,
        particles: &particles,
        nuclides: request.nuclides,
        tabulations: tabulations.as_ref().map(|t| t.borrowed()),
        memory_budget: None,
        workers: None,
        stop: request.stop,
        progress: request.progress,
    };
    let outcome = convert::convert(&materials, root, &options)?;
    let stale = gates::completeness(recipe, &outcome, root, request.nuclides.is_none())?;

    let created = std::time::SystemTime::now();
    let stamp = release::utc_timestamp(created)?;
    if parts.iter().any(|p| p.is_chain()) {
        say(&format!(
            "{}: building the transmutation chain",
            recipe.keyword
        ));
    }
    let chain = chain::build(&materials, root, recipe, &parts, &release, &stamp)?;

    say(&format!("{}: reading every folder back", recipe.keyword));
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let budget = convert::available_memory().saturating_mul(convert::SAFETY_PERCENT) / 100;
    gates::read_back(root, &particles, workers, budget)?;
    gates::chain(root, &parts)?;

    for particle in &particles {
        index::write(root, *particle, recipe.keyword)?;
    }

    let provenance = provenance(request, &parts, temperatures, &fetched, njoy.as_deref());
    let written = release::write_manifest(
        root,
        recipe.keyword,
        &release,
        nuclear_data_schema::FORMAT_VERSION,
        created,
        provenance,
    )?;
    say(&format!(
        "{}: release {release} written to {}",
        recipe.keyword,
        root.display()
    ));
    let files = &written.manifest.files;
    Ok(Report {
        keyword: recipe.keyword.to_string(),
        release,
        path: root.to_path_buf(),
        parts,
        converted: outcome.converted,
        reused: outcome.reused,
        not_converted: outcome.failed,
        stale_declarations: stale,
        njoy,
        chain_nuclides: chain.nuclides,
        manifest_sha256: written.manifest_sha256,
        file_count: files.len(),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
    })
}

/// The parts to build: those asked for, each a part the library's release
/// carries, or all of those.
fn parts(recipe: &Recipe, asked: Option<&[Part]>) -> Result<Vec<Part>, String> {
    if recipe.parts.is_empty() {
        return Err(format!(
            "{} has nothing that can be built from its own evaluations yet: its \
             release is isomeric branching, and naming a product level's isomer \
             takes decay data, which {} does not have. The published release is \
             unchanged and stays available",
            recipe.keyword, recipe.keyword
        ));
    }
    let Some(asked) = asked else {
        return Ok(recipe.parts.to_vec());
    };
    if asked.is_empty() {
        return Err("parts is empty, so there is nothing to build".to_string());
    }
    let unknown: Vec<&str> = asked
        .iter()
        .filter(|p| !recipe.parts.contains(p))
        .map(|p| p.name())
        .collect();
    if !unknown.is_empty() {
        let has: Vec<&str> = recipe.parts.iter().map(|p| p.name()).collect();
        return Err(format!(
            "{} has no {} part: its parts are {}",
            recipe.keyword,
            unknown.join(" or "),
            has.join(", ")
        ));
    }
    let mut out: Vec<Part> = asked.to_vec();
    out.sort();
    out.dedup();
    Ok(out)
}

fn check_temperatures(temperatures: &[f64]) -> Result<(), String> {
    if temperatures.is_empty() {
        return Err("temperatures is empty".to_string());
    }
    if let Some(t) = temperatures.iter().find(|t| !t.is_finite() || **t <= 0.0) {
        return Err(format!(
            "temperature {t} K is not a temperature to process at; the 0 K grid is \
             always kept, so it is not asked for"
        ));
    }
    Ok(())
}

/// The particle folder a part is, if it is one.
fn particle(part: Part) -> Option<Particle> {
    match part {
        Part::Neutron => Some(Particle::Neutron),
        Part::Photon => Some(Particle::Photon),
        Part::Photonuclear => Some(Particle::Photonuclear),
        _ => None,
    }
}

/// The sublibraries `parts` are built from.
fn sublibraries(recipe: &Recipe, parts: &[Part]) -> BTreeSet<Sublibrary> {
    let has_yields = recipe
        .sources
        .iter()
        .any(|s| s.sublibrary == Sublibrary::FissionYields);
    let mut out = BTreeSet::new();
    for part in parts {
        match part {
            Part::Neutron => {
                out.insert(Sublibrary::Neutron);
            }
            Part::Photon => {
                out.extend([Sublibrary::Photoatomic, Sublibrary::AtomicRelaxation]);
            }
            Part::Photonuclear => {
                out.insert(Sublibrary::Photonuclear);
            }
            Part::Decay | Part::Reactions | Part::FissionYields => {
                out.extend([
                    Sublibrary::Neutron,
                    Sublibrary::Decay,
                    Sublibrary::FissionYields,
                ]);
            }
            Part::Branching => {
                out.extend([Sublibrary::Neutron, Sublibrary::Decay]);
                // Scoped to the network's parents, when there is a network.
                if has_yields {
                    out.insert(Sublibrary::FissionYields);
                }
            }
        }
    }
    out
}

/// Refuse a folder a build must not write into: one inside the download
/// cache, where a local build could be taken for a published release, and a
/// folder that holds something other than a build.
///
/// A build's folder is one with `.build/state.json`, an empty one, or one
/// holding nothing but `.build/`.
fn check_folder(root: &Path) -> Result<(), String> {
    if root.exists() && !root.is_dir() {
        return Err(format!("{} is not a folder", root.display()));
    }
    let absolute = std::path::absolute(root).map_err(|e| format!("{}: {e}", root.display()))?;
    if let Some(cache) = yamc_nuclide::url_cache::cache_root() {
        let real = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        let within = absolute.starts_with(&cache)
            || existing_ancestor(&absolute).is_some_and(|a| real(&a).starts_with(real(&cache)));
        if within {
            return Err(format!(
                "{} is inside the nuclear data cache ({}), where a local build could \
                 be read as a published release; build somewhere else",
                root.display(),
                cache.display()
            ));
        }
    }
    // A folder holding only `.build/` is a new build that keeps the source
    // archives an earlier one downloaded: its release folders were removed to
    // start a new release.
    if root.is_dir() && !root.join(".build").join(STATE).is_file() {
        let entries =
            fs::read_dir(root).map_err(|e| format!("could not list {}: {e}", root.display()))?;
        let other = entries
            .filter_map(Result::ok)
            .any(|e| e.file_name() != ".build");
        if other {
            return Err(format!(
                "{} already holds files and is not a library build; build into an empty \
                 or new folder",
                root.display()
            ));
        }
    }
    Ok(())
}

/// The nearest ancestor of `path` that exists, `path` itself included.
fn existing_ancestor(path: &Path) -> Option<PathBuf> {
    path.ancestors().find(|a| a.exists()).map(Path::to_path_buf)
}

/// Hold `<build>/lock` for as long as the returned file lives.
fn lock(build: &Path) -> Result<fs::File, String> {
    let path = build.join("lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| format!("could not open {}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => Err(format!(
            "another build is running in {}; wait for it to finish",
            build.parent().unwrap_or(build).display()
        )),
        Err(fs::TryLockError::Error(e)) => Err(format!("could not lock {}: {e}", path.display())),
    }
}

/// The release stamp of the build in this folder, recording the builder on
/// the first run and refusing any other builder or library after it.
fn state(build: &Path, keyword: &str, builder: &str) -> Result<String, String> {
    let path = build.join(STATE);
    if let Ok(text) = fs::read_to_string(&path) {
        let state: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let field = |k: &str| state[k].as_str().unwrap_or("").to_string();
        let (was_keyword, was_builder, release) =
            (field("keyword"), field("builder"), field("release"));
        if was_keyword != keyword {
            return Err(format!(
                "{} holds a build of {was_keyword}, not {keyword}; build {keyword} in \
                 another folder",
                build.parent().unwrap_or(build).display()
            ));
        }
        if was_builder != builder {
            return Err(format!(
                "{} was started by {was_builder}, and this is {builder}. A folder holds \
                 the output of one builder, so resume it with {was_builder}, or build \
                 into a new folder",
                build.parent().unwrap_or(build).display()
            ));
        }
        release::check_release_id(&release)?;
        return Ok(release);
    }
    let now = release::utc_timestamp(std::time::SystemTime::now())?;
    let release = now[..10].to_string();
    let body = json!({"builder": builder, "keyword": keyword, "release": release, "started": now});
    let part = build.join(format!("{STATE}.part"));
    fs::write(&part, body.to_string() + "\n")
        .and_then(|()| fs::rename(&part, &path))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(release)
}

/// Refuse to start a full build on a disk without room for it.
fn check_disk(root: &Path, needed: u64) -> Result<(), String> {
    let Some(at) = existing_ancestor(&std::path::absolute(root).unwrap_or(root.to_path_buf()))
    else {
        return Ok(());
    };
    let at = fs::canonicalize(&at).unwrap_or(at);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let Some(disk) = disks
        .list()
        .iter()
        .filter(|d| at.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
    else {
        return Ok(());
    };
    let used = tree_bytes(root);
    let free = disk.available_space();
    if free.saturating_add(used) < needed {
        let gb = |b: u64| b as f64 / 1e9;
        return Err(format!(
            "{} has {:.1} GB free, and a full build needs about {:.0} GB at its peak \
             ({:.1} GB of it already there); free some space or build on a larger disk",
            disk.mount_point().display(),
            gb(free),
            gb(needed),
            gb(used)
        ));
    }
    Ok(())
}

/// The bytes of every file under `dir`.
fn tree_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => tree_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// The manifest's provenance: what built it from what, and how it differs
/// from the default build.
fn provenance(
    request: &Request,
    parts: &[Part],
    temperatures: &[f64],
    fetched: &[sources::Fetched],
    njoy: Option<&str>,
) -> serde_json::Map<String, Value> {
    let mut p = serde_json::Map::new();
    p.insert("builder".into(), json!(request.builder));
    p.insert("njoy".into(), json!(njoy));
    let sources: Vec<Value> = fetched
        .iter()
        .map(|f| json!({"name": f.source.name, "bytes": f.source.pin.bytes, "sha256": f.source.pin.sha256}))
        .collect();
    p.insert("sources".into(), json!(sources));
    if request.parts.is_some() {
        p.insert(
            "parts".into(),
            json!(parts.iter().map(|p| p.name()).collect::<Vec<_>>()),
        );
    }
    if let Some(nuclides) = request.nuclides {
        p.insert("nuclides".into(), json!(nuclides));
    }
    if temperatures != TEMPERATURES {
        p.insert("temperatures".into(), json!(temperatures));
    }
    p
}

/// The tabulations a photon folder carries beside its evaluation, which are
/// in no ENDF sublibrary: Compton profiles (Biggs et al. 1975), the density
/// effect (Sternheimer et al. 1982) and bremsstrahlung (Seltzer and Berger
/// 1986). Compiled in, so every wheel builds photon data the same way.
const COMPTON: &[u8] =
    include_bytes!("../../../packages/yamc-core/python/yamc/data/compton_profiles_biggs1975.txt");
const DENSITY: &[u8] = include_bytes!(
    "../../../packages/yamc-core/python/yamc/data/density_effect_sternheimer1982.txt"
);
const BREMSSTRAHLUNG: &[u8] = include_bytes!(
    "../../../packages/yamc-core/python/yamc/data/bremsstrahlung_seltzer_berger1986.txt"
);

/// The written tabulation files.
struct Tabulations {
    compton: PathBuf,
    density: PathBuf,
    bremsstrahlung: PathBuf,
}

impl Tabulations {
    fn borrowed(&self) -> PhotonTabulations<'_> {
        PhotonTabulations {
            compton_profiles: &self.compton,
            density_effect: &self.density,
            bremsstrahlung: &self.bremsstrahlung,
        }
    }
}

fn write_tabulations(dir: &Path) -> Result<Tabulations, String> {
    fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let put = |name: &str, body: &[u8]| -> Result<PathBuf, String> {
        let path = dir.join(name);
        if fs::read(&path).is_ok_and(|b| b == body) {
            return Ok(path);
        }
        fs::write(&path, body).map_err(|e| format!("could not write {}: {e}", path.display()))?;
        Ok(path)
    };
    Ok(Tabulations {
        compton: put("compton_profiles_biggs1975.txt", COMPTON)?,
        density: put("density_effect_sternheimer1982.txt", DENSITY)?,
        bremsstrahlung: put("bremsstrahlung_seltzer_berger1986.txt", BREMSSTRAHLUNG)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_default_to_the_release_and_refuse_what_it_does_not_carry() {
        let tendl = recipe::recipe("tendl-2025").unwrap();
        assert_eq!(parts(tendl, None).unwrap(), tendl.parts);
        // TENDL has no decay data, so it carries no chain part.
        let err = parts(tendl, Some(&[Part::Reactions])).unwrap_err();
        assert!(err.contains("no reactions part"), "{err}");
        let got = parts(
            tendl,
            Some(&[Part::Photonuclear, Part::Neutron, Part::Neutron]),
        )
        .unwrap();
        assert_eq!(got, [Part::Neutron, Part::Photonuclear]);
        assert!(parts(tendl, Some(&[])).is_err());
        let irdff = recipe::recipe("irdff-ii").unwrap();
        let err = parts(irdff, None).unwrap_err();
        assert!(err.contains("takes decay data"), "{err}");
    }

    #[test]
    fn chain_parts_fetch_what_the_chain_is_assembled_from() {
        let endf = recipe::recipe("endf-b8.1").unwrap();
        let got: Vec<_> = sublibraries(endf, &[Part::Branching]).into_iter().collect();
        assert_eq!(
            got,
            [
                Sublibrary::Neutron,
                Sublibrary::Decay,
                Sublibrary::FissionYields
            ]
        );
        let tendl = recipe::recipe("tendl-2025").unwrap();
        let got: Vec<_> = sublibraries(tendl, tendl.parts).into_iter().collect();
        assert_eq!(got, [Sublibrary::Neutron, Sublibrary::Photonuclear]);
    }

    #[test]
    fn temperatures_must_be_positive() {
        assert!(check_temperatures(&TEMPERATURES).is_ok());
        assert!(check_temperatures(&[0.0]).is_err());
        assert!(check_temperatures(&[]).is_err());
        assert!(check_temperatures(&[f64::NAN]).is_err());
    }

    #[test]
    fn the_tabulations_are_the_files_yamc_ships() {
        let data =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/yamc-core/python/yamc/data");
        assert_eq!(
            fs::read(data.join("compton_profiles_biggs1975.txt")).unwrap(),
            COMPTON
        );
        assert!(DENSITY.len() > 1000 && BREMSSTRAHLUNG.len() > 1_000_000);
    }
}
