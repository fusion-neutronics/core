//! Edits to a converted folder's `version.json`.
//!
//! The marker is what a cache compares to decide whether it holds the current
//! release, so the two rewriting binaries (`split_energy`, `split_reactions`)
//! both have to touch it, and both have to touch it the same way: read, set,
//! write to a temporary, rename. A torn marker is worse than a stale one, since
//! it is the file every client reads first.

use std::error::Error;
use std::path::Path;

/// Apply `edits` to `dir/version.json`, staged and renamed.
///
/// `Ok(false)` when the folder has no marker, which is not an error: the
/// caller is walking a tree and a directory without one is not a converted
/// nuclide. Keys the edits do not name are left exactly as they were, so a
/// stamp does not disturb an index written beside it.
pub fn update(dir: &Path, edits: &[(&str, serde_json::Value)]) -> Result<bool, Box<dyn Error>> {
    let path = dir.join("version.json");
    if !path.exists() {
        return Ok(false);
    }
    let mut marker: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let object = marker
        .as_object_mut()
        .ok_or("version.json is not a JSON object")?;
    for (key, value) in edits {
        object.insert((*key).to_string(), value.clone());
    }
    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&marker)?)?;
    std::fs::rename(tmp, path)?;
    Ok(true)
}

/// Set `data_version`, the stamp a cache compares against the origin's.
///
/// Every folder of a release carries the same one, including the photon
/// elements and the nuclides a rewrite had nothing to do to: a folder left on
/// the previous stamp is one a client will not refetch, which is the drift the
/// flag exists to prevent.
pub fn stamp_data_version(dir: &Path, version: &str) -> Result<bool, Box<dyn Error>> {
    update(dir, &[("data_version", serde_json::json!(version))])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(marker: serde_json::Value) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("version.json"),
            serde_json::to_string_pretty(&marker).unwrap(),
        )
        .unwrap();
        tmp
    }

    fn read(dir: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join("version.json")).unwrap()).unwrap()
    }

    #[test]
    fn a_stamp_leaves_every_other_key_alone() {
        // Including the two indexes, which are the expensive thing in the file
        // and have nothing to do with the stamp.
        let dir = folder(serde_json::json!({
            "format_version": 2,
            "data_version": "2026-09-08",
            "library": "endf-b8.1",
            "reaction_ranges": {"schema": [0, 10], "mts": {"102": {"294K": [10, 20]}}},
            "energy_ranges": {"schema": [0, 10], "temperatures": {"294K": [10, 20]}},
        }));
        assert!(stamp_data_version(dir.path(), "2026-09-17").unwrap());
        let after = read(dir.path());
        assert_eq!(after["data_version"], "2026-09-17");
        assert_eq!(after["format_version"], 2);
        assert_eq!(after["library"], "endf-b8.1");
        assert_eq!(after["reaction_ranges"]["mts"]["102"]["294K"][1], 20);
        assert_eq!(after["energy_ranges"]["temperatures"]["294K"][0], 10);
    }

    #[test]
    fn several_edits_land_together() {
        let dir = folder(serde_json::json!({"format_version": 1, "data_version": "old"}));
        assert!(update(
            dir.path(),
            &[
                ("format_version", serde_json::json!(2)),
                ("data_version", serde_json::json!("new")),
            ],
        )
        .unwrap());
        let after = read(dir.path());
        assert_eq!(after["format_version"], 2);
        assert_eq!(after["data_version"], "new");
    }

    #[test]
    fn a_folder_with_no_marker_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!stamp_data_version(tmp.path(), "2026-09-17").unwrap());
    }
}
