//! Building the transmutation chain of a release:
//! `<path>/transmutation/{decay,reactions,fission_yields,branching}.arrow/`.
//!
//! The subsections are written into `<path>/.build/chain/` and each is then
//! renamed into place, so a reader never finds half of one. The chain's own
//! `manifest.json`, which the converters keep beside their subsections,
//! stays in `.build/` and is never published: the release has one manifest.
//!
//! The chain is always built whole, whatever nuclides the cross sections
//! were limited to. It needs no NJOY and takes minutes, so a subset build
//! never publishes a chain cut down to its subset.
//!
//! Decay, reactions and fission yields come from one call, each subsection
//! from the library's own sublibrary alone: decay from its decay
//! evaluations, reactions from its neutron evaluations, fission yields from
//! its yields. Nothing of another library is used.
//!
//! Branching comes from a second call, scoped to the parents of the
//! reactions subsection so its curves line up with the network. A library
//! with no decay data of its own writes its excited product levels as it
//! gives them, and a run names their isomers against the decay data it is
//! given.
//!
//! The scalar isomeric ratios some chains apply to (n,gamma) from an assumed
//! reactor spectrum are not applied: a channel's branching is the
//! evaluation's own, or reported as absent.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use yani_convert::Provenance;

use crate::identify::Material;
use crate::recipe::{Part, Recipe, Sublibrary};

/// The tolerance in eV within which an excited level of a neutron
/// evaluation is matched to an isomer of the decay data.
const LEVEL_TOLERANCE_EV: f64 = 3000.0;

/// The relative tolerance non-linear branching curves are resampled to.
const LINEARIZE_TOLERANCE: f64 = 1e-3;

/// What the chain stage wrote.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Chain {
    /// The subsections now in `<path>/transmutation/`.
    pub subsections: Vec<Part>,
    /// The nuclides of the network, when decay, reactions or fission yields
    /// were built.
    pub nuclides: usize,
    /// The parents the branching was extracted for, when it was built.
    pub branching_parents: usize,
}

/// `<path>/.build/chain/`.
pub fn staging_dir(root: &Path) -> PathBuf {
    root.join(".build").join("chain")
}

/// `<path>/transmutation/<part>.arrow/`.
pub fn subsection_dir(root: &Path, part: Part) -> PathBuf {
    root.join(release_dir())
        .join(format!("{}.arrow", part.name()))
}

fn release_dir() -> &'static str {
    "transmutation"
}

/// Build the chain subsections among `parts` from `materials`, the whole of
/// the library's identified sources, into the release folder `root`.
///
/// `release` and `created_utc` are the stamps the subsections record.
pub fn build(
    materials: &[Material],
    root: &Path,
    recipe: &Recipe,
    parts: &[Part],
    release: &str,
    created_utc: &str,
) -> Result<Chain, String> {
    let wanted: Vec<Part> = parts.iter().copied().filter(|p| p.is_chain()).collect();
    if wanted.is_empty() {
        return Ok(Chain::default());
    }
    let staging = staging_dir(root);
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .map_err(|e| format!("could not clear {}: {e}", staging.display()))?;
    }
    fs::create_dir_all(&staging)
        .map_err(|e| format!("could not create {}: {e}", staging.display()))?;

    let provenance = Provenance {
        library: recipe.keyword.to_string(),
        decay_library: recipe.keyword.to_string(),
        data_version: release.to_string(),
        created_utc: created_utc.to_string(),
    };
    let neutron = files_of(materials, Sublibrary::Neutron);
    let decay = files_of(materials, Sublibrary::Decay);
    let yields = files_of(materials, Sublibrary::FissionYields);
    let mut chain = Chain::default();

    // The network's subsections, each from its own sublibrary alone, and the
    // reactions also when only the branching is wanted: their parents are
    // what the branching is scoped to.
    let mut network: Vec<String> = wanted
        .iter()
        .filter(|p| matches!(p, Part::Decay | Part::Reactions | Part::FissionYields))
        .map(|p| p.name().to_string())
        .collect();
    if wanted.contains(&Part::Branching) && !network.iter().any(|n| n == Part::Reactions.name()) {
        network.push(Part::Reactions.name().to_string());
    }
    for (part, files, what) in [
        (Part::Decay, &decay, "decay"),
        (Part::Reactions, &neutron, "neutron"),
        (Part::FissionYields, &yields, "fission yield"),
    ] {
        if network.iter().any(|n| n == part.name()) && files.is_empty() {
            return Err(format!(
                "{}: the {} subsection is built from {what} evaluations, and the sources \
                 hold none",
                recipe.keyword,
                part.name()
            ));
        }
    }
    if !network.is_empty() {
        chain.nuclides = yani_convert::convert_transmutation_files(
            &strings(&decay),
            &strings(&yields),
            &strings(&neutron),
            &[],
            "",
            None,
            None,
            Some(&network),
            &staging,
            &provenance,
        )
        .map_err(|e| format!("{}: building the chain: {e}", recipe.keyword))?;
    }

    if wanted.contains(&Part::Branching) {
        // With no decay data of its own, the library's excited levels are
        // written as it gives them, and a run names their isomers against the
        // decay data it is given.
        let reactions = staging.join("reactions").join("reactions.arrow");
        let parents = reaction_parents(&reactions)?;
        let inputs = branching_inputs(materials, &parents, &staging.join("inputs"))?;
        chain.branching_parents = parents.len();
        yani_convert::convert_branching_files(
            &strings(&inputs),
            &strings(&decay),
            &staging,
            &provenance,
            LEVEL_TOLERANCE_EV,
            LINEARIZE_TOLERANCE,
        )
        .map_err(|e| format!("{}: extracting the isomeric branching: {e}", recipe.keyword))?;
    }

    let published = root.join(release_dir());
    fs::create_dir_all(&published)
        .map_err(|e| format!("could not create {}: {e}", published.display()))?;
    for part in &wanted {
        let from = staging.join(part.name());
        if !from.is_dir() {
            return Err(format!(
                "{}: the chain converters wrote no {} subsection",
                recipe.keyword,
                part.name()
            ));
        }
        let to = subsection_dir(root, *part);
        if to.exists() {
            fs::remove_dir_all(&to)
                .map_err(|e| format!("could not remove {}: {e}", to.display()))?;
        }
        fs::rename(&from, &to)
            .map_err(|e| format!("could not move {} to {}: {e}", from.display(), to.display()))?;
        chain.subsections.push(*part);
    }
    Ok(chain)
}

/// Every file holding a material of `sub`, sorted, each once.
fn files_of(materials: &[Material], sub: Sublibrary) -> Vec<PathBuf> {
    let files: BTreeSet<&Path> = materials
        .iter()
        .filter(|m| m.sublibrary == sub)
        .map(|m| m.path.as_path())
        .collect();
    files.into_iter().map(Path::to_path_buf).collect()
}

fn strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// The distinct parents of a reactions subsection.
fn reaction_parents(reactions: &Path) -> Result<BTreeSet<String>, String> {
    use arrow_array::{Array, StringArray};
    let what = |e: String| format!("{}: {e}", reactions.display());
    let file = fs::File::open(reactions).map_err(|e| what(e.to_string()))?;
    let reader =
        arrow_ipc::reader::FileReader::try_new(file, None).map_err(|e| what(e.to_string()))?;
    let mut parents = BTreeSet::new();
    for batch in reader {
        let batch = batch.map_err(|e| what(e.to_string()))?;
        let column = batch
            .column_by_name("nuclide")
            .ok_or_else(|| what("has no nuclide column".to_string()))?;
        let names = column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| what("its nuclide column is not text".to_string()))?;
        for i in 0..names.len() {
            if names.is_valid(i) {
                parents.insert(names.value(i).to_string());
            }
        }
    }
    Ok(parents)
}

/// The neutron files holding the evaluations of `parents`. A file is used
/// as it is when every material in it is a parent; otherwise the parents'
/// materials are written to a file of their own under `dir`, since the
/// converter reads every material of a file it is given (IRDFF-II is one
/// tape of isotopic and element evaluations).
fn branching_inputs(
    materials: &[Material],
    parents: &BTreeSet<String>,
    dir: &Path,
) -> Result<Vec<PathBuf>, String> {
    // Per file: the parents' MATs, the first parent's name, and how many
    // materials the file holds.
    let mut by_file: BTreeMap<&Path, (Vec<i32>, Option<&str>, usize)> = BTreeMap::new();
    for m in materials
        .iter()
        .filter(|m| m.sublibrary == Sublibrary::Neutron)
    {
        let entry = by_file.entry(m.path.as_path()).or_default();
        entry.2 += 1;
        if parents.contains(&m.name) {
            entry.0.push(m.header.mat);
            entry.1 = Some(entry.1.map_or(m.name.as_str(), |n| n.min(m.name.as_str())));
        }
    }
    // In parent order, so the subsection's rows come out by parent name
    // whatever the library calls its files.
    let mut files: Vec<(&str, &Path, Vec<i32>, usize)> = by_file
        .into_iter()
        .filter_map(|(file, (mats, first, total))| first.map(|n| (n, file, mats, total)))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for (_, file, mats, total) in files {
        if mats.len() == total {
            out.push(file.to_path_buf());
            continue;
        }
        fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
        let stem = file
            .file_stem()
            .map_or_else(|| "tape".into(), |s| s.to_string_lossy());
        let to = dir.join(format!("{stem}-parents.endf"));
        write_materials(file, &mats, &to)?;
        out.push(to);
    }
    Ok(out)
}

/// Copy the materials numbered `mats` out of the ENDF-6 tape `from` into a
/// tape of their own at `to`: a TPID line, each material's lines through its
/// MEND record, and a TEND record.
fn write_materials(from: &Path, mats: &[i32], to: &Path) -> Result<(), String> {
    let bytes = fs::read(from).map_err(|e| format!("could not read {}: {e}", from.display()))?;
    let wanted: BTreeSet<i32> = mats.iter().copied().collect();
    let mut out = Vec::with_capacity(bytes.len());
    out.extend_from_slice(
        format!(
            "{:<66}{:>4}{:>2}{:>3}\n",
            "materials selected by the library builder", 1, 0, 0
        )
        .as_bytes(),
    );
    let mut current: Option<i32> = None;
    for line in bytes.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(mat) = mat_of(line) else {
            continue;
        };
        if mat > 0 {
            current = Some(mat);
        }
        if current.is_some_and(|m| wanted.contains(&m)) && mat >= 0 {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
        if mat == 0 {
            // MEND closes the material.
            current = None;
        }
    }
    out.extend_from_slice(format!("{:<66}{:>4}{:>2}{:>3}\n", "", -1, 0, 0).as_bytes());
    fs::write(to, out).map_err(|e| format!("could not write {}: {e}", to.display()))
}

/// The MAT number in columns 67 to 70 of an ENDF-6 line.
fn mat_of(line: &[u8]) -> Option<i32> {
    let field = line.get(66..70)?;
    std::str::from_utf8(field).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, mat: i32, mf: i32, mt: i32) -> String {
        format!("{text:<66}{mat:>4}{mf:>2}{mt:>3}")
    }

    #[test]
    fn only_the_named_materials_are_copied_off_a_tape() {
        let dir = std::env::temp_dir().join(format!("ndb-chain-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tape = [
            line("tape id", 1, 0, 0),
            line("Li nat", 300, 1, 451),
            line("", 300, 0, 0),
            line("", 0, 0, 0),
            line("Li6", 325, 1, 451),
            line("", 325, 0, 0),
            line("", 0, 0, 0),
            line("", -1, 0, 0),
        ]
        .join("\n");
        let from = dir.join("tape.endf");
        fs::write(&from, tape).unwrap();
        let to = dir.join("out.endf");
        write_materials(&from, &[325], &to).unwrap();
        let text = fs::read_to_string(&to).unwrap();
        let mats: Vec<i32> = text.lines().filter_map(|l| mat_of(l.as_bytes())).collect();
        assert_eq!(mats, [1, 325, 325, 0, -1], "{text}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
