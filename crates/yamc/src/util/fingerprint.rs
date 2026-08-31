//! Physics-identity fingerprint of a [`Model`].
//!
//! [`Model::fingerprint`] produces a stable hash of everything that
//! determines the sampled physics: geometry, materials, sources,
//! physics settings (tracking mode, variance reduction, cutoffs, ...)
//! and the per-isotope/per-element nuclear-data provenance. It excludes
//! the run-varying `verbose` knob and the skipped diagnostics (the other
//! run parameters -- seed, particle count, threads, time budget -- are
//! per-call `TransportSettings` and never on the Model at all), plus the
//! `tallies` subtree (tallies are
//! observation-only and may legitimately differ between combinable
//! runs).
//!
//! `combine_results` refuses to pool results whose fingerprints differ:
//! two runs only estimate the same quantities if they sampled the same
//! game.
//!
//! ## Stability
//!
//! The hash input is the `serde_json::Value` form of the model.
//! `serde_json`'s map type is a `BTreeMap` (the `preserve_order`
//! feature is not enabled in this workspace), so object keys serialize
//! in sorted order regardless of `HashMap` iteration order in the
//! source structs -- the canonicalization is inherent, not incidental.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::model::Model;

/// Model fields excluded from the fingerprint: the run-varying `verbose`
/// knob and the observation-only tallies subtree. The `last_*` diagnostics
/// are `#[serde(skip)]` and never appear in the serialized form.
const EXCLUDED_FIELDS: [&str; 2] = ["verbose", "tallies"];

impl Model {
    /// Per-isotope / per-element nuclear-data provenance map, e.g.
    /// `"n:Co58" -> "endf-b8.1"`, `"p:Fe" -> "endf-b8.1/photon"`.
    ///
    /// Resolution precedence per nuclide: the `Nuclide.library` tag when
    /// populated, else a shortened form of the data path it was loaded
    /// from (last two path components, so identity survives across
    /// machines with different data roots). Photon element data uses the
    /// shortened configured path.
    pub fn data_libraries(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for material in self.geometry.materials() {
            for (name, nuclide) in material.nuclide_data.iter() {
                let lib = nuclide
                    .library
                    .clone()
                    .filter(|l| !l.is_empty())
                    .or_else(|| nuclide.data_path.as_deref().map(short_data_path))
                    .unwrap_or_else(|| "unknown".to_string());
                map.insert(format!("n:{name}"), lib);
            }
            for (element, path) in material.photon_data_paths.iter() {
                map.insert(format!("p:{element}"), short_data_path(path));
            }
        }
        map
    }

    /// Stable hash (SHA-256 hex) of the model's physics identity. See
    /// the module docs for what is included and excluded.
    ///
    /// Returns `Err` only if the model fails to serialize. Note that
    /// non-finite floats do NOT error: `serde_json` maps NaN/infinity to
    /// `null`, so configurations differing only in a non-finite value
    /// fingerprint identically (such configurations are invalid input
    /// regardless and rejected elsewhere).
    pub fn fingerprint(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self)
            .map_err(|e| format!("model fingerprint serialization failed: {e}"))?;
        if let serde_json::Value::Object(map) = &mut value {
            for field in EXCLUDED_FIELDS {
                map.remove(field);
            }
        }
        let combined = serde_json::json!({
            "data": self.data_libraries(),
            "model": value,
        });
        let canonical = serde_json::to_string(&combined)
            .map_err(|e| format!("model fingerprint serialization failed: {e}"))?;
        let digest = Sha256::digest(canonical.as_bytes());
        Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
    }
}

/// Last two components of a data path: `"/home/x/endf-b8.1/Co58.arrow"`
/// becomes `"endf-b8.1/Co58.arrow"`. Keeps the identity stable across
/// machines with different data roots while still distinguishing
/// libraries laid out as `<library>/<nuclide>.arrow`.
fn short_data_path(path: &str) -> String {
    let parts: Vec<&str> = path
        .trim_end_matches('/')
        .split(['/', '\\'])
        .filter(|p| !p.is_empty())
        .collect();
    match parts.len() {
        0 => "unknown".to_string(),
        1 => parts[0].to_string(),
        n => format!("{}/{}", parts[n - 2], parts[n - 1]),
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::short_data_path;

    #[test]
    fn short_data_path_keeps_last_two_components() {
        assert_eq!(
            short_data_path("/home/user/data/endf-b8.1/Co58.arrow"),
            "endf-b8.1/Co58.arrow"
        );
        assert_eq!(short_data_path("tests/Co58.arrow"), "tests/Co58.arrow");
        assert_eq!(short_data_path("Co58.arrow"), "Co58.arrow");
        assert_eq!(short_data_path(""), "unknown");
    }
}
