//! Unpacking the source files into one flat folder per sublibrary, in Rust
//! (zip and gzipped tar), applying each sublibrary's archives in recipe order
//! so a later update archive replaces what an earlier one supplied.
//!
//! Members are written under their file name alone, never their folder, so a
//! member named `../x` cannot write outside the folder. Two members with one
//! file name in the same archive are refused, since one would silently hide
//! the other. Across archives that is how an update replaces a file.
//!
//! A sublibrary's folder records the sources it was unpacked from
//! (`.sources`, written last). A rerun with the same sources leaves the folder
//! alone; with any other, or after an interrupted run, the folder is unpacked
//! again from scratch, so a folder never mixes two sets of sources.

use std::collections::HashSet;
use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use crate::recipe::{Format, Recipe, Source, Sublibrary};
use crate::sources::Fetched;

/// The file in each sublibrary folder that lists what it was unpacked from.
pub const MARKER: &str = ".sources";

/// `<path>/.build/sources/`.
pub fn sources_dir(root: &Path) -> PathBuf {
    root.join(".build").join("sources")
}

/// Unpack every fetched source of `recipe` into
/// `<path>/.build/sources/<sublibrary>/`, returning each sublibrary's folder.
pub fn extract(
    recipe: &Recipe,
    fetched: &[Fetched],
    root: &Path,
) -> Result<Vec<(Sublibrary, PathBuf)>, String> {
    let mut out = Vec::new();
    for sub in recipe.sublibraries() {
        let dir = sources_dir(root).join(sub.dir());
        let parts: Vec<&Fetched> = fetched
            .iter()
            .filter(|f| f.source.sublibrary == sub)
            .collect();
        let expected: String = parts
            .iter()
            .map(|f| {
                let pin = f.source.pin.expect("a fetched source is pinned");
                format!("{} {}\n", pin.sha256, f.source.name)
            })
            .collect();
        let marker = dir.join(MARKER);
        if fs::read_to_string(&marker).is_ok_and(|m| m == expected) {
            out.push((sub, dir));
            continue;
        }
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .map_err(|e| format!("could not clear {}: {e}", dir.display()))?;
        }
        fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
        for f in parts {
            let n = unpack(&f.source, &f.path, &dir)?;
            if n == 0 {
                return Err(format!(
                    "{} holds no {} evaluations: is it the right file?",
                    f.path.display(),
                    sub.dir()
                ));
            }
        }
        let part = dir.join(format!("{MARKER}.part"));
        fs::write(&part, &expected)
            .and_then(|()| fs::rename(&part, &marker))
            .map_err(|e| format!("could not write {}: {e}", marker.display()))?;
        out.push((sub, dir));
    }
    Ok(out)
}

/// Unpack the members `source.keep` accepts from `path` into `dir`,
/// returning how many were written.
fn unpack(source: &Source, path: &Path, dir: &Path) -> Result<usize, String> {
    let what = |e: String| format!("{}: {e}", path.display());
    match source.format() {
        Format::Plain => {
            let to = dir.join(source.name);
            fs::copy(path, &to)
                .map(|_| 1)
                .map_err(|e| format!("could not copy {} to {}: {e}", path.display(), to.display()))
        }
        Format::TarGz => unpack_tar_gz(source.keep, path, dir).map_err(what),
        Format::Zip => unpack_zip(source.keep, path, dir).map_err(what),
    }
}

/// The file name of an archive member, or `None` for a member that names a
/// folder or nothing usable.
fn base_name(member: &str) -> Option<&str> {
    let name = member.rsplit(['/', '\\']).next()?;
    (!name.is_empty() && name != "." && name != "..").then_some(name)
}

/// Write one member, refusing a second member with the same file name.
fn write_member(
    seen: &mut HashSet<String>,
    dir: &Path,
    name: &str,
    body: &mut impl io::Read,
) -> Result<(), String> {
    if !seen.insert(name.to_string()) {
        return Err(format!(
            "holds two files named {name}, and only one of them could be used"
        ));
    }
    let to = dir.join(name);
    let mut file =
        fs::File::create(&to).map_err(|e| format!("could not create {}: {e}", to.display()))?;
    io::copy(body, &mut file).map_err(|e| format!("unpacking {name}: {e}"))?;
    Ok(())
}

fn unpack_tar_gz(keep: fn(&str) -> bool, path: &Path, dir: &Path) -> Result<usize, String> {
    let file = fs::File::open(path).map_err(|e| format!("could not open: {e}"))?;
    let gz = flate2::read::MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file));
    let mut archive = tar::Archive::new(gz);
    let mut seen = HashSet::new();
    let entries = archive
        .entries()
        .map_err(|e| format!("not a gzipped tar archive: {e}"))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("not a readable gzipped tar archive: {e}"))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let member = entry
            .path()
            .map_err(|e| format!("a member name is unreadable: {e}"))?
            .to_string_lossy()
            .into_owned();
        let Some(name) = base_name(&member) else {
            continue;
        };
        if keep(name) {
            let name = name.to_string();
            write_member(&mut seen, dir, &name, &mut entry)?;
        }
    }
    Ok(seen.len())
}

fn unpack_zip(keep: fn(&str) -> bool, path: &Path, dir: &Path) -> Result<usize, String> {
    let file = fs::File::open(path).map_err(|e| format!("could not open: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(BufReader::new(file)).map_err(|e| format!("not a zip archive: {e}"))?;
    let mut seen = HashSet::new();
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("member {i} is unreadable: {e}"))?;
        if !entry.is_file() {
            continue;
        }
        let member = entry
            .name()
            .map_err(|e| format!("member {i} has an unreadable name: {e}"))?
            .into_owned();
        let Some(name) = base_name(&member) else {
            continue;
        };
        if keep(name) {
            let name = name.to_string();
            write_member(&mut seen, dir, &name, &mut entry)?;
        }
    }
    Ok(seen.len())
}
