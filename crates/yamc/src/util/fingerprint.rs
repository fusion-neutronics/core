//! Physics-identity fingerprint of a [`Model`].
//!
//! [`Model::fingerprint`] produces a stable hash of everything that
//! determines the sampled physics: geometry, materials, sources,
//! physics settings (tracking mode, variance reduction, cutoffs, ...)
//! and the per-isotope/per-element nuclear-data provenance. It excludes
//! the run-varying `verbose` knob and the skipped diagnostics (the other
//! run parameters -- seed, particle count, threads, time budget -- are
//! per-call `TransportSettings` and never on the Model at all), the
//! `tallies` subtree (tallies are
//! observation-only and may legitimately differ between combinable
//! runs), and the values of the material ids. An id labels a material
//! rather than changing what is sampled, and an automatic one depends on
//! what else the process built, so each is replaced by the first slot of
//! the material store holding it: which slots share a material stays part
//! of the identity. `combine_results` matches material filters the same
//! way, through [`Model::material_slots`].
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

use std::collections::{BTreeMap, HashMap};

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

    /// The published release of each library keyword this model's data came
    /// from: release identifier, manifest sha256, and whether the origin was
    /// unreachable so a cached release was used. Keyed by keyword.
    ///
    /// A library is listed when a nuclide was loaded from that release's cache
    /// folder (`<cache root>/<keyword>/<release>/...`) or a material names the
    /// keyword as its data source. One release per library per process (see
    /// `yamc_nuclide::url_cache::data_releases`), so this is exact for a run.
    /// Empty for data loaded from local directories.
    pub fn data_releases(&self) -> BTreeMap<String, yamc_nuclide::storage::release::DataRelease> {
        let in_use = yamc_nuclide::url_cache::data_releases();
        if in_use.is_empty() {
            return BTreeMap::new();
        }
        let libraries = self.data_libraries();
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        for material in self.geometry.materials() {
            for nuclide in material.nuclide_data.values() {
                if let Some(p) = &nuclide.data_path {
                    paths.push(std::path::PathBuf::from(p));
                }
            }
            for p in material.photon_data_paths.values() {
                paths.push(std::path::PathBuf::from(p));
            }
        }
        in_use
            .into_iter()
            .filter(|(keyword, release)| {
                let named = libraries.values().any(|lib| {
                    yamc_nuclide::url_cache::library_keyword(lib) == keyword
                        || lib.split('/').next() == Some(keyword.as_str())
                });
                let loaded = paths.iter().any(|p| {
                    p.ancestors().any(|a| {
                        a.file_name().and_then(|n| n.to_str()) == Some(release.release.as_str())
                            && a.parent()
                                .and_then(|q| q.file_name())
                                .and_then(|n| n.to_str())
                                == Some(keyword.as_str())
                    })
                });
                named || loaded
            })
            .collect()
    }

    /// Each geometry material's id, mapped to the first slot of the material
    /// store holding it: the run provenance `combine_results` matches
    /// material filters through, by the material each id names.
    pub fn material_slots(&self) -> BTreeMap<u32, u32> {
        let mut slots = BTreeMap::new();
        for (slot, material) in self.geometry.materials().iter().enumerate() {
            if let Some(id) = material.get_material_id() {
                slots.entry(id).or_insert(slot as u32);
            }
        }
        slots
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
            if let Some(geometry) = map.get_mut("geometry") {
                label_materials_by_slot(geometry);
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

/// Replace each id in the serialized geometry's material store by the first
/// slot holding a material with that id.
///
/// A cell reaches its material by slot, so two builds of one model that number
/// their materials differently sample the same game and must still combine;
/// the slots that share an id (a mesh fill's per-volume copies) stay part of
/// the identity.
fn label_materials_by_slot(geometry: &mut serde_json::Value) {
    // `{"Csg": {"materials": [...], ...}}`, keyed by the geometry's kind. A
    // mesh geometry serializes no materials.
    let Some(kinds) = geometry.as_object_mut() else {
        return;
    };
    for kind in kinds.values_mut() {
        let Some(materials) = kind
            .get_mut("materials")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        let mut first_slot: HashMap<u64, usize> = HashMap::new();
        for (slot, material) in materials.iter_mut().enumerate() {
            if let Some(entry) = material.get_mut("material_id")
                && let Some(id) = entry.as_u64()
            {
                *entry = (*first_slot.entry(id).or_insert(slot)).into();
            }
        }
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
    use crate::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
    use crate::geometry::cell::Cell;
    use crate::geometry::csg::Geometry;
    use crate::model::Model;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;
    use yamc_materials::material::Material;

    /// Two nested spheres, one material each: `ids[i]` is the id of the
    /// material in slot `i` (`None` leaves it to the geometry), `densities[i]`
    /// its density.
    fn shells_model(ids: [Option<u32>; 2], densities: [f64; 2]) -> Model {
        let sphere = |surface_id, radius, boundary| {
            Arc::new(Surface {
                surface_id: Some(surface_id),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius,
                },
                boundary,
                name: None,
            })
        };
        let inner = sphere(1, 5.0, BoundaryType::Transmission);
        let outer = sphere(2, 10.0, BoundaryType::Vacuum);
        let materials = ids
            .iter()
            .zip(densities)
            .map(|(id, density)| {
                let mut material = Material::new(
                    HashMap::from([("Li6".to_string(), 1.0)]),
                    "atom",
                    "g/cm3",
                    Some(density),
                )
                .unwrap();
                if let Some(id) = id {
                    material.set_material_id(*id);
                }
                Arc::new(material)
            })
            .collect();
        let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&inner)));
        let shell = Region::new_from_halfspace(HalfspaceType::Above(inner))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(outer)));
        let cells = vec![
            Cell::new(Some(1), core, None, Some(0)),
            Cell::new(Some(2), shell, None, Some(1)),
        ];
        Model::new(Geometry::new(cells, materials).unwrap(), vec![], vec![])
    }

    #[test]
    fn material_ids_are_not_part_of_the_fingerprint() {
        let reference = shells_model([Some(1), Some(2)], [2.0, 3.0])
            .fingerprint()
            .unwrap();
        for ids in [[Some(7), Some(3)], [None, None], [Some(2), Some(1)]] {
            let relabelled = shells_model(ids, [2.0, 3.0]).fingerprint().unwrap();
            assert_eq!(relabelled, reference, "{ids:?}");
        }
        // The materials, and which slot holds which, still are.
        let swapped = shells_model([Some(1), Some(2)], [3.0, 2.0]);
        assert_ne!(swapped.fingerprint().unwrap(), reference);
    }

    #[test]
    fn material_slots_map_each_id_to_its_slot() {
        let model = shells_model([Some(7), Some(3)], [2.0, 3.0]);
        assert_eq!(model.material_slots(), BTreeMap::from([(7, 0), (3, 1)]));
    }

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
