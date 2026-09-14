//! Rewrite already-converted nuclide folders one record batch per (MT,
//! temperature), and reindex them.
//!
//! A library published one batch per MT makes a client that wants one
//! temperature of one reaction download all six (fusion-neutronics/core#100).
//! This rewrites each `{Name}.arrow/reactions.arrow` so every (MT, temperature)
//! is its own batch, copying the cross sections rather than recomputing them,
//! so NJOY does not run again and every value is what it was. `version.json`
//! is then reindexed, since every batch offset moves.
//!
//! The rewrite is staged beside the file and renamed into place, so an
//! interrupted run leaves either the old file or the new one, never a torn
//! one; `version.json` is rewritten the same way. Between the two renames the
//! folder holds a split `reactions.arrow` beside a `version.json` indexing the
//! file it replaced; rerunning this over the folder repairs it, since a file
//! already in this shape rewrites to the same rows and is reindexed after.
//!
//! Publishing the result needs a data-version bump, unlike a plain reindex:
//! every offset moves, so a client still holding the old `version.json` would
//! range-fetch the new object at the old boundaries and splice a stream of the
//! wrong bytes. The bump is what makes a cache evict both objects together.
//!
//! ```text
//! cargo run --release -p yamc-convert --bin split_reactions -- DIR [DIR ...]
//! ```
//!
//! Each DIR either is a `{Name}.arrow` folder or holds them. A folder with no
//! `reactions.arrow` (a photon element) is skipped.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yamc_convert::reaction_ranges::write_reaction_ranges;
use yamc_convert::reactions::rewrite_per_temperature;

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

/// Rewrite one folder's `reactions.arrow` and reindex it. `Ok(false)` when
/// the folder has no reactions table.
fn split(dir: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let reactions = dir.join("reactions.arrow");
    if !reactions.exists() {
        return Ok(false);
    }
    let staged = dir.join("reactions.arrow.tmp");
    rewrite_per_temperature(&reactions, &staged)?;
    std::fs::rename(&staged, &reactions)?;
    write_reaction_ranges(dir)?;
    Ok(true)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: split_reactions DIR [DIR ...]");
        eprintln!("  DIR is a {{Name}}.arrow folder, or a directory holding them");
        return ExitCode::from(2);
    }

    let (mut split_count, mut skipped, mut failed) = (0usize, 0usize, 0usize);
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
            match split(&dir) {
                Ok(false) => skipped += 1,
                Ok(true) => {
                    split_count += 1;
                    println!("ok   {name}");
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL {name}: {e}");
                }
            }
        }
    }

    println!("{split_count} split, {skipped} skipped, {failed} failed");
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
