//! Add MT byte ranges to already-converted nuclide folders.
//!
//! Rewrites `version.json` in each `{Name}.arrow/` to carry the byte range of
//! every MT's record batch in `reactions.arrow`, so an activation reader can
//! range-request just the channels its chain names (8.2x less on Fe56) instead
//! of pulling the full-grid transport MTs it never looks at.
//!
//! The cross-section files themselves are never opened for writing and stay
//! byte-identical, which is what makes this a reindex rather than a rebuild: a
//! published library gains the index without NJOY running again, and only the
//! few-kB `version.json` objects need reuploading.
//!
//! ```text
//! cargo run --release -p yamc-convert --bin index_reactions -- DIR [DIR ...]
//! ```
//!
//! Each DIR either is a `{Name}.arrow` folder or holds them.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yamc_convert::reaction_ranges::write_reaction_ranges;

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: index_reactions DIR [DIR ...]");
        eprintln!("  DIR is a {{Name}}.arrow folder, or a directory holding them");
        return ExitCode::from(2);
    }

    let (mut indexed, mut skipped, mut failed) = (0usize, 0usize, 0usize);
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
            match write_reaction_ranges(&dir) {
                // A photon element, or a conversion with no reactions table.
                Ok(false) => skipped += 1,
                Ok(true) => {
                    indexed += 1;
                    println!("ok   {name}");
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL {name}: {e}");
                }
            }
        }
    }

    println!("{indexed} indexed, {skipped} skipped, {failed} failed");
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
