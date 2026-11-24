//! A whole build, from a library keyword to a release folder with its
//! manifest: the one call the Python `generate_library` makes.
//!
//! In order, every check that can fail fast runs before anything is
//! downloaded:
//!
//! 1. the arguments and the folder (not inside the download cache, not
//!    someone else's folder);
//! 2. a builder compiled from uncommitted changes, refused unless the build
//!    is partial (some parts, nuclides or temperatures), which is never
//!    published: no commit describes what it would write;
//! 3. an exclusive lock on `<path>/.build/lock`, and the build's state in
//!    `<path>/.build/state.json`: the release stamp (the UTC date the first
//!    run started, kept on resume) and the builder that wrote the folder
//!    (package, version, commit and dirty flag), since a rerun by another
//!    build would mix two converters' output under one manifest;
//! 4. the NJOY the library needs, when neutron data is built, and the free
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

use nuclear_data_schema::BuildIdentity;
use serde_json::{json, Value};
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
    /// What is building: recorded in the folder's state, in the manifest's
    /// provenance as `builder`, `builder_commit` and `builder_dirty`, and as
    /// the `converter` stamp of every folder converted. A rerun by another
    /// build is refused, and a dirty one builds only a partial library.
    pub builder: BuildIdentity<'a>,
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
    check_clean(&request.builder, partial(request, temperatures))?;

    let build = root.join(".build");
    fs::create_dir_all(&build).map_err(|e| format!("could not create {}: {e}", build.display()))?;
    let _lock = lock(&build)?;
    let release = state(&build, recipe.keyword, &request.builder)?;
    let manifest = root.join("manifest.json");
    if manifest.exists() {
        fs::remove_file(&manifest)
            .map_err(|e| format!("could not remove {}: {e}", manifest.display()))?;
    }

    let neutron = parts.contains(&Part::Neutron);
    let njoy = if neutron || parts.contains(&Part::Thermal) {
        let found =
            endf::njoy::require(request.njoy_exec, recipe.njoy).map_err(|e| e.to_string())?;
        Some(found.to_string())
    } else {
        None
    };
    if (neutron || parts.contains(&Part::Thermal)) && request.nuclides.is_none() {
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
    if !particles.is_empty() {
        let names: Vec<&str> = particles.iter().map(|p| p.dir()).collect();
        say(&format!(
            "{}: converting ({})",
            recipe.keyword,
            names.join(", ")
        ));
    }
    let converter = request.builder.stamp();
    let options = convert::Options {
        library: recipe.keyword,
        release: &release,
        converter: &converter,
        njoy_exec: request.njoy_exec,
        temperatures,
        particles: &particles,
        nuclides: request.nuclides,
        tabulations: None,
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
    let chain = chain::build(
        &materials, root, recipe, &parts, &release, &stamp, &converter,
    )?;

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
    if unknown == ["thermal"] {
        return Err(yamc_nuclide::url_cache::no_thermal_tables(recipe.keyword));
    }
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
        Part::Thermal => Some(Particle::Thermal),
        _ => None,
    }
}

/// The sublibraries `parts` are built from: each part from its own.
fn sublibraries(_recipe: &Recipe, parts: &[Part]) -> BTreeSet<Sublibrary> {
    let mut out = BTreeSet::new();
    for part in parts {
        match part {
            Part::Neutron | Part::Reactions | Part::Branching => {
                out.insert(Sublibrary::Neutron);
            }
            Part::Photon => {
                out.extend([Sublibrary::Photoatomic, Sublibrary::AtomicRelaxation]);
            }
            Part::Photonuclear => {
                out.insert(Sublibrary::Photonuclear);
            }
            Part::Decay => {
                out.insert(Sublibrary::Decay);
            }
            Part::FissionYields => {
                out.insert(Sublibrary::FissionYields);
            }
            // THERMR reads the principal atom's neutron evaluation.
            Part::Thermal => {
                out.extend([Sublibrary::ThermalScattering, Sublibrary::Neutron]);
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

/// Whether the build is partial: some parts, some nuclides or other
/// temperatures. Exactly the builds whose manifest provenance records
/// `parts`, `nuclides` or `temperatures`, which publishing refuses.
fn partial(request: &Request, temperatures: &[f64]) -> bool {
    request.parts.is_some() || request.nuclides.is_some() || temperatures != TEMPERATURES
}

/// Refuse a builder compiled from uncommitted changes, unless the build is
/// `partial` and so never published. A builder whose commit is unknown is
/// not refused: its manifest records a null commit.
fn check_clean(builder: &BuildIdentity, partial: bool) -> Result<(), String> {
    if !builder.is_dirty() || partial {
        return Ok(());
    }
    Err(format!(
        "{} was compiled from core {} with uncommitted changes, so no commit \
         describes the library it would build. Commit the changes and rebuild it \
         (maturin develop --release) to build a whole library. A partial build \
         (parts, nuclides or temperatures), which is never published, may use it",
        builder.package,
        builder.commit.unwrap_or_default()
    ))
}

/// The release stamp of the build in this folder, recording the builder on
/// the first run and refusing any other builder or library after it. The
/// builder is its package, version, commit and dirty flag: two builds from
/// source both report version 0.0.0.dev0, and only the commit tells them
/// apart.
fn state(build: &Path, keyword: &str, builder: &BuildIdentity) -> Result<String, String> {
    let path = build.join(STATE);
    if let Ok(text) = fs::read_to_string(&path) {
        let state: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let field = |k: &str| state[k].as_str().unwrap_or("").to_string();
        let (was_keyword, release) = (field("keyword"), field("release"));
        if was_keyword != keyword {
            return Err(format!(
                "{} holds a build of {was_keyword}, not {keyword}; build {keyword} in \
                 another folder",
                build.parent().unwrap_or(build).display()
            ));
        }
        let was_package = field("builder");
        let was = BuildIdentity {
            package: &was_package,
            commit: state["builder_commit"].as_str(),
            dirty: state["builder_dirty"].as_bool(),
        };
        if was != *builder {
            let (was, now) = (was.stamp(), builder.stamp());
            return Err(format!(
                "{} was started by {was}, and this is {now}. A folder holds the output \
                 of one build, so resume it with {was}, or build into a new folder",
                build.parent().unwrap_or(build).display()
            ));
        }
        release::check_release_id(&release)?;
        return Ok(release);
    }
    let now = release::utc_timestamp(std::time::SystemTime::now())?;
    let release = now[..10].to_string();
    let body = json!({
        "builder": builder.package,
        "builder_commit": builder.commit,
        "builder_dirty": builder.dirty,
        "keyword": keyword,
        "release": release,
        "started": now,
    });
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
    p.insert("builder".into(), json!(request.builder.package));
    p.insert("builder_commit".into(), json!(request.builder.commit));
    p.insert("builder_dirty".into(), json!(request.builder.dirty));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_default_to_the_release_and_refuse_what_it_does_not_carry() {
        let tendl = recipe::recipe("tendl-2025").unwrap();
        assert_eq!(parts(tendl, None).unwrap(), tendl.parts);
        // TENDL has no decay data of its own, so it carries no decay part.
        let err = parts(tendl, Some(&[Part::Decay])).unwrap_err();
        assert!(err.contains("no decay part"), "{err}");
        let got = parts(
            tendl,
            Some(&[Part::Branching, Part::Neutron, Part::Neutron]),
        )
        .unwrap();
        assert_eq!(got, [Part::Neutron, Part::Branching]);
        assert!(parts(tendl, Some(&[])).is_err());
    }

    #[test]
    fn each_part_fetches_its_own_sublibrary() {
        let endf = recipe::recipe("endf-b8.1").unwrap();
        let got: Vec<_> = sublibraries(endf, &[Part::Branching]).into_iter().collect();
        assert_eq!(got, [Sublibrary::Neutron]);
        let got: Vec<_> = sublibraries(endf, &[Part::Decay]).into_iter().collect();
        assert_eq!(got, [Sublibrary::Decay]);
        let irdff = recipe::recipe("irdff-ii").unwrap();
        let got: Vec<_> = sublibraries(irdff, irdff.parts).into_iter().collect();
        assert_eq!(got, [Sublibrary::Neutron]);
    }

    const COMMIT: &str = "250c439e0000000000000000000000000000abcd";

    fn identity(commit: Option<&'static str>, dirty: Option<bool>) -> BuildIdentity<'static> {
        BuildIdentity {
            package: "yani-python 0.0.0.dev0",
            commit,
            dirty,
        }
    }

    fn request<'a>(path: &'a Path, builder: BuildIdentity<'a>) -> Request<'a> {
        Request {
            keyword: "irdff-ii",
            path,
            parts: None,
            nuclides: None,
            temperatures: None,
            njoy_exec: "njoy",
            downloads: None,
            builder,
            stop: None,
            progress: None,
        }
    }

    /// A fresh folder under the system temporary directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "nuclear-data-build-generate-{}-{name}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_dirty_builder_builds_only_a_partial_library() {
        let dirty = identity(Some(COMMIT), Some(true));
        let err = check_clean(&dirty, false).unwrap_err();
        assert!(err.contains("uncommitted changes"), "{err}");
        assert!(err.contains(COMMIT), "{err}");
        assert!(check_clean(&dirty, true).is_ok());
        assert!(check_clean(&identity(Some(COMMIT), Some(false)), false).is_ok());
        // Built outside git: nothing to be dirty against, recorded as null.
        assert!(check_clean(&identity(None, None), false).is_ok());
    }

    #[test]
    fn partial_is_what_the_manifest_records_as_a_subset() {
        let path = Path::new("unused");
        let builder = identity(None, None);
        assert!(!partial(&request(path, builder), &TEMPERATURES));
        assert!(partial(&request(path, builder), &[293.6]));
        let parts = [Part::Branching];
        let r = Request {
            parts: Some(&parts),
            ..request(path, builder)
        };
        assert!(partial(&r, &TEMPERATURES));
        let nuclides = ["Fe56".to_string()];
        let r = Request {
            nuclides: Some(&nuclides),
            ..request(path, builder)
        };
        assert!(partial(&r, &TEMPERATURES));
    }

    /// The refusal comes before anything is written: the folder is not even
    /// created.
    #[test]
    fn generate_refuses_a_whole_library_from_a_dirty_builder_before_writing() {
        let scratch = Scratch::new("dirty");
        let root = scratch.0.join("irdff-ii");
        let err = generate(&request(&root, identity(Some(COMMIT), Some(true)))).unwrap_err();
        assert!(err.contains("uncommitted changes"), "{err}");
        assert!(!root.exists());
    }

    #[test]
    fn the_state_records_the_build_and_refuses_another_commit() {
        let scratch = Scratch::new("state");
        let clean = identity(Some(COMMIT), Some(false));
        let release = state(&scratch.0, "irdff-ii", &clean).unwrap();
        let saved: Value =
            serde_json::from_str(&fs::read_to_string(scratch.0.join(STATE)).unwrap()).unwrap();
        assert_eq!(saved["builder"], "yani-python 0.0.0.dev0");
        assert_eq!(saved["builder_commit"], COMMIT);
        assert_eq!(saved["builder_dirty"], false);
        assert_eq!(state(&scratch.0, "irdff-ii", &clean).unwrap(), release);

        // The 0.0.0.dev0 case: the same version from another commit.
        let other = identity(Some("ffff"), Some(false));
        let err = state(&scratch.0, "irdff-ii", &other).unwrap_err();
        assert!(err.contains(&format!("(core {COMMIT})")), "{err}");
        assert!(err.contains("(core ffff)"), "{err}");
        let dirty = identity(Some(COMMIT), Some(true));
        assert!(state(&scratch.0, "irdff-ii", &dirty).is_err());
        assert!(state(&scratch.0, "irdff-ii", &identity(None, None)).is_err());
    }

    #[test]
    fn an_unknown_commit_is_recorded_as_null() {
        let scratch = Scratch::new("unknown");
        let unknown = identity(None, None);
        state(&scratch.0, "irdff-ii", &unknown).unwrap();
        let saved: Value =
            serde_json::from_str(&fs::read_to_string(scratch.0.join(STATE)).unwrap()).unwrap();
        assert!(saved["builder_commit"].is_null() && saved["builder_dirty"].is_null());
        assert!(state(&scratch.0, "irdff-ii", &unknown).is_ok());

        let p = provenance(
            &request(&scratch.0, unknown),
            &[Part::Branching],
            &TEMPERATURES,
            &[],
            None,
        );
        assert_eq!(p["builder"], "yani-python 0.0.0.dev0");
        assert!(p["builder_commit"].is_null() && p["builder_dirty"].is_null());
    }

    #[test]
    fn the_provenance_records_the_builder_commit_and_dirty_flag() {
        let dirty = identity(Some(COMMIT), Some(true));
        let path = Path::new("unused");
        let parts = [Part::Branching];
        let r = Request {
            parts: Some(&parts),
            ..request(path, dirty)
        };
        let p = provenance(&r, &parts, &TEMPERATURES, &[], Some("2016.80"));
        assert_eq!(p["builder"], "yani-python 0.0.0.dev0");
        assert_eq!(p["builder_commit"], COMMIT);
        assert_eq!(p["builder_dirty"], true);
        assert_eq!(p["njoy"], "2016.80");
        assert_eq!(p["parts"], json!(["branching"]));
    }

    #[test]
    fn temperatures_must_be_positive() {
        assert!(check_temperatures(&TEMPERATURES).is_ok());
        assert!(check_temperatures(&[0.0]).is_err());
        assert!(check_temperatures(&[]).is_err());
        assert!(check_temperatures(&[f64::NAN]).is_err());
    }
}
