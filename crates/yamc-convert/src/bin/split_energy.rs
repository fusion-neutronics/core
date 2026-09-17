//! Move the union energy grids of already-converted nuclide folders out of
//! `nuclide.arrow` into `energy.arrow`, and reindex them.
//!
//! A library published with the grids inside `nuclide.arrow` makes a client
//! that wants one temperature read all six, and makes a client that wants only
//! to know what a nuclide IS read 6.33 MB of U238 grids to find out
//! (fusion-neutronics/core#100). This rewrites each folder into the version 2
//! layout, copying the grids rather than recomputing them, so NJOY does not run
//! again and every value is what it was. `version.json` is then set to
//! `format_version: 2` and given an `energy_ranges` index.
//!
//! Each file is staged beside its target and renamed into place, and
//! `energy.arrow` is written before `nuclide.arrow` loses its columns, so an
//! interrupted run leaves a folder with its grids in one file or the other,
//! never in neither. The marker is written last: a folder whose marker still
//! says 1 is re-migrated correctly on a rerun.
//!
//! Publishing the result needs a data-version bump, and every client has to be
//! on a build that reads version 2, since that build does not read version 1
//! and vice versa. Pair it with `split_reactions` so users take one cache
//! eviction rather than two.
//!
//! ```text
//! cargo run --release -p yamc-convert --bin split_energy -- DIR [DIR ...]
//! ```
//!
//! Each DIR either is a `{Name}.arrow` folder or holds them. A folder with no
//! `nuclide.arrow` is skipped, which covers a photon element (whose layout this
//! does not change) and a folder already migrated.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yamc_convert::energy_ranges::write_energy_ranges;
use yamc_convert::nuclide::migrate_energy_out_of_nuclide;

/// The `*.arrow` folders under `root`, or `root` itself when it is one.
fn folders(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    if root.join("version.json").exists() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.extension().is_some_and(|e| e == "arrow"))
        .collect();
    out.sort();
    Ok(out)
}

/// Set `format_version` to 2 in a folder's marker, staged and renamed.
fn stamp_version(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let path = dir.join("version.json");
    let mut version: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    version
        .as_object_mut()
        .ok_or("version.json is not a JSON object")?
        .insert("format_version".to_string(), serde_json::json!(2));
    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&version)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Migrate one folder. `Ok(false)` when it has nothing to move.
fn migrate(dir: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    if !migrate_energy_out_of_nuclide(dir)? {
        return Ok(false);
    }
    write_energy_ranges(dir)?;
    stamp_version(dir)?;
    Ok(true)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: split_energy DIR [DIR ...]");
        eprintln!("  DIR is a {{Name}}.arrow folder, or a directory holding them");
        return ExitCode::from(2);
    }

    let (mut moved, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    for arg in &args {
        let root = Path::new(arg);
        let dirs = match folders(root) {
            Ok(d) if d.is_empty() => {
                eprintln!("no *.arrow folders under {}", root.display());
                failed += 1;
                continue;
            }
            Ok(d) => d,
            Err(e) => {
                eprintln!("FAIL {}: {e}", root.display());
                failed += 1;
                continue;
            }
        };
        for dir in dirs {
            let name = dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            match migrate(&dir) {
                Ok(false) => skipped += 1,
                Ok(true) => {
                    moved += 1;
                    println!("ok   {name}");
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL {name}: {e}");
                }
            }
        }
    }

    println!("{moved} migrated, {skipped} skipped, {failed} failed");
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
