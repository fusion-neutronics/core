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
//!
//! `--data-version VERSION` additionally stamps `data_version` on every folder
//! walked, skipped ones included. That is the field a cache compares against
//! the origin's to decide whether to refetch, so a release wants one value
//! across the whole tree: a folder left on the previous stamp is one no client
//! will refetch. Photon elements need it as much as the nuclides do, which is
//! why the stamp is not tied to whether there was anything to split.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yamc_convert::marker;
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

/// Command line: the folders, and the release stamp to write.
struct Args {
    dirs: Vec<String>,
    /// `--data-version VERSION`, the stamp every folder walked is given.
    data_version: Option<String>,
}

/// Parse `argv`, or `None` to print usage and exit 2.
fn parse(argv: Vec<String>) -> Option<Args> {
    let mut out = Args {
        dirs: Vec::new(),
        data_version: None,
    };
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return None,
            "--data-version" => out.data_version = Some(it.next()?),
            rest if rest.starts_with("--data-version=") => {
                out.data_version = Some(rest["--data-version=".len()..].to_string());
            }
            rest if rest.starts_with('-') => return None,
            rest => out.dirs.push(rest.to_string()),
        }
    }
    (!out.dirs.is_empty()).then_some(out)
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
    let Some(args) = parse(std::env::args().skip(1).collect()) else {
        eprintln!("usage: split_reactions [--data-version VERSION] DIR [DIR ...]");
        eprintln!("  DIR is a {{Name}}.arrow folder, or a directory holding them");
        eprintln!("  --data-version  stamp every folder walked with this release stamp");
        return ExitCode::from(2);
    };

    let (mut split_count, mut skipped, mut stamped, mut failed) = (0usize, 0usize, 0usize, 0usize);
    for arg in &args.dirs {
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
                    // Not stamped: a folder whose split failed must not claim
                    // to be part of the release.
                    continue;
                }
            }
            if let Some(version) = &args.data_version {
                match marker::stamp_data_version(&dir, version) {
                    Ok(true) => stamped += 1,
                    Ok(false) => {}
                    Err(e) => {
                        failed += 1;
                        eprintln!("FAIL {name}: stamping data_version: {e}");
                    }
                }
            }
        }
    }

    match &args.data_version {
        Some(v) => {
            println!(
                "{split_count} split, {skipped} skipped, {stamped} stamped {v}, {failed} failed"
            )
        }
        None => println!("{split_count} split, {skipped} skipped, {failed} failed"),
    }
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
