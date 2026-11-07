//! `<particle>/index.json`: what each `<Name>.arrow/` folder of a particle
//! publishes, so a browser can list a library and what is in it without a
//! server. Nothing else on the host says which nuclides exist.
//!
//! The shapes are the ones the published releases carry (format 1), written
//! compact with sorted keys and one trailing newline.
//!
//! neutron, from each folder's `version.json` range indexes:
//!
//! ```text
//! {"data_version":"2026-09-18","format":1,"full_mask":63,"library":"endf-b8.1",
//!  "nuclides":{"Fe56":{"masks":{"2":127},"mts":[1,2,3]}},"particle":"neutron",
//!  "temperatures":["250K","294K","600K","900K","1200K","2500K","0K"]}
//! ```
//!
//! Bit `i` of a mask is `temperatures[i]`. `full_mask` is the mask most MTs
//! carry, and `masks` names only the MTs whose set differs from it (MT 2,
//! the one reaction with a 0 K table), which keeps TENDL's index small.
//!
//! photon, from each element's tables:
//!
//! ```text
//! {"data_version":"2026-09-18","elements":{"Fe":[502,504,515,517,522]},"format":1,
//!  "library":"endf-b8.1","mts":[502,504,515,517,522],"mts_verified":true,
//!  "particle":"photon"}
//! ```
//!
//! photonuclear, from each folder's tables:
//!
//! ```text
//! {"format":1,"library":"tendl-2025","nuclides":{"Fe56":{"energy_points":{"5":120},
//!  "mts":[5]}},"particle":"photonuclear","scope":"evaluated"}
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::convert::Particle;

/// The version of the index shapes.
pub const FORMAT: u32 = 1;

/// The photoatomic MTs an element folder can carry, by what the reader
/// gives for each. MT 501, the total, is not written.
const PHOTON_MTS: [u32; 5] = [502, 504, 515, 517, 522];

/// Write `<root>/<particle>/index.json` for `particle`, returning how many
/// folders it lists. A particle folder with no `<Name>.arrow/` folders gets
/// no index.
pub fn write(root: &Path, particle: Particle, library: &str) -> Result<usize, String> {
    let dir = root.join(particle.dir());
    let folders = folders(&dir)?;
    if folders.is_empty() {
        return Ok(0);
    }
    let (body, count) = match particle {
        Particle::Neutron => neutron(&dir, &folders, library)?,
        Particle::Photon => photon(&dir, &folders, library)?,
        Particle::Photonuclear => photonuclear(&dir, &folders, library)?,
        Particle::Thermal => thermal(&dir, &folders, library)?,
    };
    let path = dir.join("index.json");
    let text = serde_json::to_string(&body).map_err(|e| e.to_string())? + "\n";
    let part = dir.join("index.json.part");
    fs::write(&part, text)
        .and_then(|()| fs::rename(&part, &path))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(count)
}

/// The names of the `<Name>.arrow/` folders in `dir`, sorted.
fn folders(dir: &Path) -> Result<Vec<String>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("could not list {}: {e}", dir.display()))? {
        let path = entry
            .map_err(|e| format!("could not list {}: {e}", dir.display()))?
            .path();
        if !path.is_dir() {
            continue;
        }
        if let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".arrow"))
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}

fn read_marker(path: &Path) -> Result<Value, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The most common value, the first seen winning a tie.
fn most_common<T: Ord + Clone>(seen: &[T]) -> Option<T> {
    let mut counts: BTreeMap<&T, (usize, usize)> = BTreeMap::new();
    for (i, v) in seen.iter().enumerate() {
        let c = counts.entry(v).or_insert((0, i));
        c.0 += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
        .map(|(v, _)| v.clone())
}

/// A temperature label as the tables spell it: `294K`, `0K`, `293.6K`.
fn kelvin(label: &str) -> Option<f64> {
    let number = label.strip_suffix('K')?;
    let ok = !number.is_empty()
        && number.split_once('.').map_or_else(
            || number.bytes().all(|b| b.is_ascii_digit()),
            |(a, b)| {
                !a.is_empty()
                    && !b.is_empty()
                    && a.bytes().all(|c| c.is_ascii_digit())
                    && b.bytes().all(|c| c.is_ascii_digit())
            },
        );
    ok.then(|| number.parse().ok()).flatten()
}

/// An MT and the temperature labels it has a table at.
type MtTemperatures = (i64, BTreeSet<String>);

fn neutron(dir: &Path, folders: &[String], library: &str) -> Result<(Value, usize), String> {
    // name -> mt -> temperature labels, in the order the markers give them.
    let mut per_nuclide: Vec<(String, Vec<MtTemperatures>)> = Vec::new();
    let mut labels: BTreeSet<String> = BTreeSet::new();
    let mut data_versions = Vec::new();
    for name in folders {
        let path = dir.join(format!("{name}.arrow")).join("version.json");
        let marker = read_marker(&path)?;
        let reactions = marker["reaction_ranges"]["mts"]
            .as_object()
            .filter(|m| !m.is_empty());
        let energy = marker["energy_ranges"]["temperatures"]
            .as_object()
            .filter(|m| !m.is_empty());
        let (Some(reactions), Some(energy)) = (reactions, energy) else {
            return Err(format!(
                "{}: no byte-range index, so a browser could only fetch it whole",
                path.display()
            ));
        };
        let check = |label: &str| {
            kelvin(label).map(|_| ()).ok_or_else(|| {
                format!("{}: unexpected temperature label {label:?}", path.display())
            })
        };
        let mut mts = Vec::new();
        for (mt, by_temperature) in reactions {
            let mt: i64 = mt
                .parse()
                .map_err(|_| format!("{}: MT {mt:?} is not a number", path.display()))?;
            let by_temperature = by_temperature.as_object().ok_or_else(|| {
                format!(
                    "{}: reaction_ranges.mts[{mt}] is not keyed by temperature",
                    path.display()
                )
            })?;
            let mut set = BTreeSet::new();
            for label in by_temperature.keys() {
                check(label)?;
                labels.insert(label.clone());
                set.insert(label.clone());
            }
            mts.push((mt, set));
        }
        for label in energy.keys() {
            check(label)?;
            labels.insert(label.clone());
        }
        mts.sort_by_key(|(mt, _)| *mt);
        per_nuclide.push((name.clone(), mts));
        data_versions.push(marker["data_version"].as_str().unwrap_or("").to_string());
    }

    // Numeric order, 0 K last: it is the unbroadened grid, not one of the
    // processed temperatures.
    let mut temperatures: Vec<String> = labels.into_iter().collect();
    temperatures.sort_by(|a, b| {
        let (ka, kb) = (kelvin(a).unwrap(), kelvin(b).unwrap());
        (ka == 0.0, ka).partial_cmp(&(kb == 0.0, kb)).unwrap()
    });
    let bit: BTreeMap<&str, u64> = temperatures
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), 1u64 << i))
        .collect();
    let mask_of = |set: &BTreeSet<String>| set.iter().map(|l| bit[l.as_str()]).sum::<u64>();
    let masks: Vec<u64> = per_nuclide
        .iter()
        .flat_map(|(_, mts)| mts.iter().map(|(_, set)| mask_of(set)))
        .collect();
    let full_mask = most_common(&masks).unwrap_or(0);

    let mut nuclides = serde_json::Map::new();
    for (name, mts) in &per_nuclide {
        let mut entry = serde_json::Map::new();
        entry.insert(
            "mts".into(),
            json!(mts.iter().map(|(mt, _)| *mt).collect::<Vec<_>>()),
        );
        let differing: serde_json::Map<String, Value> = mts
            .iter()
            .filter(|(_, set)| mask_of(set) != full_mask)
            .map(|(mt, set)| (mt.to_string(), json!(mask_of(set))))
            .collect();
        if !differing.is_empty() {
            entry.insert("masks".into(), Value::Object(differing));
        }
        nuclides.insert(name.clone(), Value::Object(entry));
    }
    let body = json!({
        "format": FORMAT,
        "library": library,
        "particle": "neutron",
        "data_version": most_common(&data_versions).unwrap_or_default(),
        "temperatures": temperatures,
        "full_mask": full_mask,
        "nuclides": nuclides,
    });
    Ok((body, per_nuclide.len()))
}

fn photon(dir: &Path, folders: &[String], library: &str) -> Result<(Value, usize), String> {
    let mut elements = serde_json::Map::new();
    let mut data_versions = Vec::new();
    for name in folders {
        let folder = dir.join(format!("{name}.arrow"));
        let marker = folder.join("version.json");
        if marker.is_file() {
            data_versions.push(
                read_marker(&marker)?["data_version"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
            );
        }
        let element = yamc_element::photon_arrow::inspect_photon_interaction_from_arrow(&folder)
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        let columns = [
            &element.coherent_xs,
            &element.incoherent_xs,
            &element.pair_production_electron_xs,
            &element.pair_production_nuclear_xs,
            &element.photoelectric_total_xs,
        ];
        let mts: Vec<u32> = PHOTON_MTS
            .into_iter()
            .zip(columns)
            .filter(|(_, xs)| !xs.is_empty())
            .map(|(mt, _)| mt)
            .collect();
        elements.insert(name.clone(), json!(mts));
    }
    let count = elements.len();
    let body = json!({
        "format": FORMAT,
        "library": library,
        "particle": "photon",
        "data_version": most_common(&data_versions).unwrap_or_default(),
        "mts": PHOTON_MTS,
        "mts_verified": true,
        "elements": elements,
    });
    Ok((body, count))
}

/// thermal, from each table's header: what it binds to, its temperatures and
/// the energy below which it applies.
fn thermal(dir: &Path, folders: &[String], library: &str) -> Result<(Value, usize), String> {
    let mut tables = serde_json::Map::new();
    for name in folders {
        let folder = dir.join(format!("{name}.arrow"));
        let s = yamc_convert::thermal::inspect_thermal(&folder)
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        if s.name != *name {
            return Err(format!(
                "{}: the data declares {}",
                folder.display(),
                s.name
            ));
        }
        tables.insert(
            name.clone(),
            json!({
                "bound_nuclides": s.bound_nuclides,
                "temperatures": s.temperatures,
                "energy_max": s.energy_max,
                "elastic": s.elastic_kind,
            }),
        );
    }
    let count = tables.len();
    let body = json!({
        "format": FORMAT,
        "library": library,
        "particle": "thermal",
        "tables": tables,
    });
    Ok((body, count))
}

fn photonuclear(dir: &Path, folders: &[String], library: &str) -> Result<(Value, usize), String> {
    let mut nuclides = serde_json::Map::new();
    for name in folders {
        let folder = dir.join(format!("{name}.arrow"));
        let summary = yamc_convert::photonuclear::inspect_photonuclear(&folder)
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        if summary.name != *name {
            return Err(format!(
                "{}: the data declares {}",
                folder.display(),
                summary.name
            ));
        }
        let points: serde_json::Map<String, Value> = summary
            .energy_points
            .iter()
            .map(|(mt, n)| (mt.to_string(), json!(n)))
            .collect();
        nuclides.insert(
            name.clone(),
            json!({
                "mts": summary.energy_points.keys().collect::<Vec<_>>(),
                "energy_points": points,
            }),
        );
    }
    let count = nuclides.len();
    let body = json!({
        "format": FORMAT,
        "library": library,
        "particle": "photonuclear",
        "scope": "evaluated",
        "nuclides": nuclides,
    });
    Ok((body, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_labels_are_numbers_in_kelvin() {
        assert_eq!(kelvin("294K"), Some(294.0));
        assert_eq!(kelvin("293.6K"), Some(293.6));
        assert_eq!(kelvin("0K"), Some(0.0));
        for bad in ["K", "294", "2.K", ".5K", "-1K", "1e3K"] {
            assert_eq!(kelvin(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_most_common_value_wins_and_a_tie_goes_to_the_first_seen() {
        assert_eq!(most_common(&[3, 1, 1, 3, 2]), Some(3));
        assert_eq!(most_common(&[5, 4, 4]), Some(4));
        assert_eq!(most_common::<u8>(&[]), None);
    }

    #[test]
    fn a_neutron_index_lists_the_mts_and_the_masks_that_differ() {
        let dir = std::env::temp_dir().join(format!("ndb-index-{}", std::process::id()));
        let neutron = dir.join("neutron");
        for (name, mts) in [
            (
                "Fe56",
                json!({"1": {"294K": {}, "600K": {}}, "2": {"294K": {}, "600K": {}, "0K": {}}}),
            ),
            ("Li6", json!({"1": {"294K": {}, "600K": {}}})),
        ] {
            let folder = neutron.join(format!("{name}.arrow"));
            fs::create_dir_all(&folder).unwrap();
            let marker = json!({
                "data_version": "2026-11-07",
                "reaction_ranges": {"mts": mts},
                "energy_ranges": {"temperatures": {"294K": {}, "600K": {}, "0K": {}}},
            });
            fs::write(folder.join("version.json"), marker.to_string()).unwrap();
        }
        assert_eq!(write(&dir, Particle::Neutron, "endf-b8.1").unwrap(), 2);
        let text = fs::read_to_string(neutron.join("index.json")).unwrap();
        assert_eq!(
            text,
            "{\"data_version\":\"2026-11-07\",\"format\":1,\"full_mask\":3,\
             \"library\":\"endf-b8.1\",\"nuclides\":{\"Fe56\":{\"masks\":{\"2\":7},\
             \"mts\":[1,2]},\"Li6\":{\"mts\":[1]}},\"particle\":\"neutron\",\
             \"temperatures\":[\"294K\",\"600K\",\"0K\"]}\n"
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
