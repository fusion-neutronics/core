//! The committed manifest must match what the emitter produces.
//!
//! The Python converter builds its pyarrow schemas from that file, so a change
//! to the declarations that is not regenerated splits the two languages apart
//! silently. Compares the whole document, not just field names, so a type,
//! nullability or metadata change is caught too.

use std::process::Command;

#[test]
fn committed_manifest_matches_the_emitter() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate sits two levels below the repo root");
    let path = repo_root
        .join("packages/nuclear_data_to_arrow/src/nuclear_data_to_arrow/schema/manifest.json");

    let emitted = Command::new(env!("CARGO"))
        .args([
            "run",
            "-q",
            "-p",
            "nuclear-data-schema",
            "--bin",
            "emit-schema-manifest",
        ])
        .current_dir(repo_root)
        .output()
        .expect("the emitter runs");
    assert!(
        emitted.status.success(),
        "emitter failed: {}",
        String::from_utf8_lossy(&emitted.stderr)
    );
    let emitted = String::from_utf8(emitted.stdout).expect("emitter output is UTF-8");

    // Windows checks the file out as CRLF under the default core.autocrlf,
    // while the emitter always writes LF.
    let committed = std::fs::read_to_string(&path)
        .expect("manifest is readable")
        .replace("\r\n", "\n");

    assert_eq!(
        committed,
        emitted,
        "\n{} is stale. Regenerate it:\n  cargo run -p nuclear-data-schema \
         --bin emit-schema-manifest > {}\n",
        path.display(),
        path.display()
    );
}
