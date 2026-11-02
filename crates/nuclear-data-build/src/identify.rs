//! Naming every material in the unpacked sources from its own MF=1 MT=451
//! header, rather than from file names, which each library spells its own
//! way (`n-026_Fe_056.endf`, `n_26-Fe-056g.jeff`, `n_026-Fe-056.dat`).
//!
//! A tape of many materials (JEFF's decay data, IRDFF-II) gives one entry per
//! material. Every material must declare the sublibrary (NSUB) its folder is
//! for, and no name may come from two materials of one sublibrary; every
//! problem found is reported together.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use endf::Header;

use crate::extract::MARKER;
use crate::recipe::Sublibrary;

/// One material of the unpacked sources.
#[derive(Debug, Clone)]
pub struct Material {
    pub sublibrary: Sublibrary,
    /// The converters' name for it: `"Fe56"`, `"Am242_m1"`, or the element
    /// for per-element sublibraries, `"Fe"`.
    pub name: String,
    /// The file it is in.
    pub path: PathBuf,
    pub header: Header,
}

/// How many problems are listed before the rest are counted.
const SHOWN: usize = 20;

/// Every material in each sublibrary folder, sorted by sublibrary and name.
pub fn identify(folders: &[(Sublibrary, PathBuf)]) -> Result<Vec<Material>, String> {
    let mut found: BTreeMap<(Sublibrary, String), Material> = BTreeMap::new();
    let mut problems = Vec::new();
    for (sub, dir) in folders {
        for path in files(dir)? {
            let headers = match endf::get_headers(&path) {
                Ok(h) => h,
                Err(e) => {
                    problems.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            for header in headers {
                if header.nsub != sub.nsub() {
                    problems.push(format!(
                        "{}: MAT {} is NSUB {}, not the {} sublibrary (NSUB {})",
                        path.display(),
                        header.mat,
                        header.nsub,
                        sub.dir(),
                        sub.nsub()
                    ));
                    continue;
                }
                let Some(name) = header.name() else {
                    problems.push(format!(
                        "{}: MAT {} (ZA {}) has no name",
                        path.display(),
                        header.mat,
                        header.za
                    ));
                    continue;
                };
                let key = (*sub, name.clone());
                if let Some(first) = found.get(&key) {
                    problems.push(format!(
                        "{name} ({}) is in both {} and {}",
                        sub.dir(),
                        first.path.display(),
                        path.display()
                    ));
                    continue;
                }
                found.insert(
                    key,
                    Material {
                        sublibrary: *sub,
                        name,
                        path: path.clone(),
                        header,
                    },
                );
            }
        }
    }
    if problems.is_empty() {
        return Ok(found.into_values().collect());
    }
    let mut msg = format!(
        "{} problem{} in the unpacked sources:\n",
        problems.len(),
        if problems.len() == 1 { "" } else { "s" }
    );
    for p in problems.iter().take(SHOWN) {
        msg.push_str(&format!("  - {p}\n"));
    }
    if problems.len() > SHOWN {
        msg.push_str(&format!("  and {} more\n", problems.len() - SHOWN));
    }
    Err(msg.trim_end().to_string())
}

/// The files of a sublibrary folder, sorted, without its marker.
fn files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| format!("could not list {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("could not list {}: {e}", dir.display()))?;
        if entry.file_name() == MARKER || !entry.path().is_file() {
            continue;
        }
        out.push(entry.path());
    }
    out.sort();
    Ok(out)
}
