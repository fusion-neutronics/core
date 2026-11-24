//! Record the git commit this extension is built from.
//!
//! Sets two compile-time variables, read by `register_classes` and exposed on
//! both wheels as `__commit__` and `__commit_dirty__`:
//!
//! * `YANI_BUILD_COMMIT`: the full hash of `HEAD`.
//! * `YANI_BUILD_DIRTY`: `true` if a tracked file differs from `HEAD`.
//!   Untracked files are ignored, because CI drops built blobs (the wasm
//!   viewers, the licence report) into the tree before building the wheel and
//!   they are not part of the source.
//!
//! Neither is set when the build is not in a git checkout of this repository
//! (a source archive, no `git` on the path), and the Python side then reports
//! `None` rather than a guess. The checkout must be this repository's own:
//! a crate vendored into some other repository would otherwise report that
//! repository's commit.
//!
//! The value is as of the last time this script ran. Cargo reruns it when
//! `HEAD`, the branch it points at, the index, or anything under `crates/`
//! changes. An edit to the Python sources alone does not recompile the
//! extension, so it does not refresh the dirty flag either.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        // The manylinux build runs in a container as a different user from
        // the checkout's owner, and git refuses such a repository unless told
        // it is safe.
        .args(["-c", "safe.directory=*", "-C"])
        .arg(root)
        .args(args)
        // `git status` otherwise refreshes the index as it reads it, which
        // would touch a file this script watches and rerun it every build.
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/yani-python -> the workspace root.
    let root = manifest_dir.join("../..");

    // Rerun on any source change, whether or not the build would otherwise
    // see it, so the dirty flag follows the tree.
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.lock").display()
    );

    let Some(toplevel) = git(&root, &["rev-parse", "--show-toplevel"]) else {
        return;
    };
    let same_repo = match (Path::new(&toplevel).canonicalize(), root.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_repo {
        return;
    }

    // `--git-path` resolves inside a linked worktree too, where `.git` is a
    // file and the branch refs live in the main checkout.
    let mut watched = vec![
        "HEAD".to_owned(),
        "index".to_owned(),
        "packed-refs".to_owned(),
    ];
    if let Some(branch) = git(&root, &["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for path in &watched {
        if let Some(resolved) = git(&root, &["rev-parse", "--git-path", path]) {
            let resolved = Path::new(&resolved);
            let resolved = if resolved.is_absolute() {
                resolved.to_path_buf()
            } else {
                root.join(resolved)
            };
            println!("cargo:rerun-if-changed={}", resolved.display());
        }
    }

    let Some(commit) = git(&root, &["rev-parse", "HEAD"]) else {
        return;
    };
    let Some(status) = git(&root, &["status", "--porcelain", "--untracked-files=no"]) else {
        return;
    };
    println!("cargo:rustc-env=YANI_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=YANI_BUILD_DIRTY={}", !status.is_empty());
}
