//! Where NJOY works during an ENDF conversion, and what it leaves behind.
//!
//! One evaluation's NJOY tapes can run to hundreds of MB, so they go in a
//! hidden directory inside the output directory, not in the system temporary
//! directory, which on many build machines is a small tmpfs or under a
//! per-user quota. So do its scratch units, which the Fortran runtime makes in
//! TMPDIR. The conversion removes that directory however it ends.
//!
//! The converter runs a stand-in `njoy` here: a shell script that records the
//! directory it was started in and writes the tapes the deck asks for. So these
//! check something on a machine without NJOY, CI included. The last test puts
//! the real NJOY behind the same script when `njoy` is on PATH.

#![cfg(unix)]

use std::io::Write;
use std::path::{Path, PathBuf};

use yamc_convert::entry::{convert_neutron_xs, Provenance, Source};

const LI6_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");
/// What the stand-in hands back as ACER's output.
const LI6_ACE: &[u8] = include_bytes!("../../endf/fixtures/Li6.ace.xz");

/// Sets `$4` and `$5` to the ACE and xsdir units, from the line after ACER's
/// card in the deck (`nendf npendf 0 nace ndir`).
const ACER_UNITS: &str = "set -- $(awk 'acer { print; exit } /^acer/ { acer = 1 }' deck)";

fn unpack(compressed: &[u8], to: &Path) -> PathBuf {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    std::fs::write(to, out).expect("fixture is written");
    to.to_path_buf()
}

/// An `njoy` for the converter to run. It writes the directory it was started
/// in to `ran_in` beside itself and the one its `TMPDIR` names to `tmpdir`,
/// saves its deck as `deck` where it runs, then runs `body` there.
///
/// Written by `sh`, not by this process. A process another test starts
/// inherits, for a moment, every file this one has open, and running the
/// script while a copy of it is open for writing fails with "Text file busy".
fn njoy(dir: &Path, body: &str) -> PathBuf {
    let script = dir.join("njoy");
    let text = format!(
        "#!/bin/sh\npwd -P > '{}'\n(cd \"${{TMPDIR:-/tmp}}\" && pwd -P) > '{}'\n\
         cat > deck\n{body}\n",
        dir.join("ran_in").display(),
        dir.join("tmpdir").display()
    );
    let mut sh = std::process::Command::new("sh")
        .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("sh starts");
    sh.stdin
        .take()
        .expect("a pipe to sh")
        .write_all(text.as_bytes())
        .expect("the stand-in is written");
    assert!(sh.wait().expect("sh finishes").success());
    script
}

/// Convert the Li6 evaluation into `root/neutron` with `njoy`.
fn convert(root: &Path, njoy: &Path) -> (PathBuf, Result<PathBuf, String>) {
    let evaluation = unpack(LI6_ENDF, &root.join("n-003_Li_006.endf"));
    let out = root.join("neutron");
    let result = convert_neutron_xs(
        &Source::Endf {
            path: &evaluation,
            njoy_exec: njoy.to_str().expect("a UTF-8 path"),
            temperatures: vec![293.6],
        },
        &out,
        &Provenance::default(),
        false,
    )
    .map_err(|e| e.to_string());
    (out, result)
}

/// What a directory holds, sorted.
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("the output directory exists")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// NJOY ran inside `out`, in a hidden directory, so a listing or an `*.arrow`
/// glob of the output skips it while it exists, and its scratch units, which
/// the Fortran runtime puts in TMPDIR, went there too.
fn assert_ran_inside(root: &Path, out: &Path) {
    let ran_in = std::fs::read_to_string(root.join("ran_in")).expect("njoy was run");
    let ran_in = Path::new(ran_in.trim_end());
    let tmpdir = std::fs::read_to_string(root.join("tmpdir")).expect("njoy was run");
    assert_eq!(
        Path::new(tmpdir.trim_end()),
        ran_in,
        "NJOY's TMPDIR is not the directory it ran in"
    );
    // Canonical, as `pwd -P` is, and through the parent, which exists whether
    // or not the conversion got as far as making `out`.
    let parent = std::fs::canonicalize(out.parent().expect("a parent")).expect("parent exists");
    let out = parent.join(out.file_name().expect("a name"));
    let inside = ran_in.strip_prefix(&out).unwrap_or_else(|_| {
        panic!(
            "NJOY ran in {}, outside the output directory {}",
            ran_in.display(),
            out.display()
        )
    });
    let first = inside.components().next().expect("a directory of its own");
    let first = first.as_os_str().to_string_lossy();
    assert!(first.starts_with(".njoy-"), "NJOY ran in {first}");
}

#[test]
fn njoy_works_inside_the_output_directory_and_only_the_nuclide_is_left() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ace = unpack(LI6_ACE, &tmp.path().join("Li6.ace"));
    let njoy = njoy(
        tmp.path(),
        &format!(
            "{ACER_UNITS}\ncp '{}' \"tape$4\"\necho xsdir > \"tape$5\"",
            ace.display()
        ),
    );

    let (out, result) = convert(tmp.path(), &njoy);

    let dir = result.expect("the stand-in's ACE table converts");
    assert_eq!(dir, out.join("Li6.arrow"));
    assert_ran_inside(tmp.path(), &out);
    assert_eq!(entries(&out), ["Li6.arrow"]);
}

#[test]
fn a_failed_njoy_run_leaves_the_output_directory_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let njoy = njoy(tmp.path(), "echo partial > tape21\nexit 1");

    let (out, result) = convert(tmp.path(), &njoy);

    let err = result.expect_err("NJOY failed, so the conversion does");
    assert!(err.contains("NJOY failed on"), "{err}");
    assert_ran_inside(tmp.path(), &out);
    assert_eq!(entries(&out), Vec::<String>::new());
}

/// NJOY finished, with every tape moved into the working directory, and then
/// its ACE table could not be read. With the scratch in /tmp this left the
/// working directory and its tapes behind.
#[test]
fn a_conversion_that_fails_after_njoy_leaves_the_output_directory_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let njoy = njoy(
        tmp.path(),
        &format!(
            "{ACER_UNITS}\necho pointwise > tape21\necho 'not an ACE table' > \"tape$4\"\n\
             echo xsdir > \"tape$5\""
        ),
    );

    let (out, result) = convert(tmp.path(), &njoy);

    result.expect_err("an unreadable ACE table is an error");
    assert_ran_inside(tmp.path(), &out);
    assert_eq!(entries(&out), Vec::<String>::new());
}

/// The same with the real NJOY behind the script, which writes every tape and
/// its own listing where it runs. Announces a loud skip without one, as the
/// other NJOY-gated tests do.
#[test]
fn real_njoy_works_inside_the_output_directory_too() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Run where its listing cannot land in this crate's directory.
    if std::process::Command::new("njoy")
        .arg("--version")
        .current_dir(tmp.path())
        .output()
        .is_err()
    {
        eprintln!("SKIP: no njoy on PATH; this test checked nothing");
        return;
    }
    let njoy = njoy(tmp.path(), "exec njoy < deck");

    let (out, result) = convert(tmp.path(), &njoy);

    let dir = result.expect("NJOY converts the evaluation");
    assert_eq!(dir, out.join("Li6.arrow"));
    assert_ran_inside(tmp.path(), &out);
    assert_eq!(entries(&out), ["Li6.arrow"]);
}
