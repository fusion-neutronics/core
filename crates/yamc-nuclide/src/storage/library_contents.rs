//! What each hosted nuclear data library provides, read from its release
//! manifest.
//!
//! A release's `manifest.json` lists every file it publishes, so what the
//! library holds (which nuclides have cross sections, which of them carry
//! covariances, which thermal tables, photon elements, photonuclear targets
//! and transmutation subsections) is read off the file list without
//! downloading any of the data. The keyword resolves to the release a load
//! would read: the current one, or offline the newest cached one.

use std::collections::BTreeSet;

use super::release::Manifest;

/// The neutron files that carry each kind of covariance, with the name a
/// report gives the kind: `(file in neutron/<nuclide>.arrow/, kind)`.
pub const COVARIANCES: [(&str, &str); 5] = [
    // MF=33.
    ("covariance.arrow", "cross_section"),
    // MF=32, read with MF=2 when resonance parameters are sampled.
    ("resonance_parameters.arrow", "resonance_parameters"),
    // MF=34.
    ("angular_covariance.arrow", "angular_distribution"),
    // MF=31.
    ("nubar_covariance.arrow", "nubar"),
    // MF=35.
    ("spectrum_covariance.arrow", "fission_spectrum"),
];

/// The transmutation subsections, by the folder each one is published in:
/// `(transmutation/<folder>.arrow, name)`.
pub const SUBSECTIONS: [(&str, &str); 4] = [
    ("decay", "decay"),
    ("reactions", "reactions"),
    ("fission_yields", "fission_yields"),
    ("branching", "isomeric_branching"),
];

/// One upstream file a release was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceArchive {
    pub name: String,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
}

/// What one release of a library holds. Every list is complete, so an empty
/// one means the library has none of that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryContents {
    pub keyword: String,
    pub release: String,
    /// The origin was unreachable, so this is the newest cached release
    /// rather than the one the origin names now.
    pub offline: bool,
    /// Nuclides with neutron cross sections, sorted.
    pub neutron: Vec<String>,
    /// Per kind of [`COVARIANCES`], in that order, the nuclides that carry it.
    pub covariances: Vec<(&'static str, Vec<String>)>,
    /// Thermal scattering tables.
    pub thermal_scattering: Vec<String>,
    /// Elements with photo-atomic (photon transport) data.
    pub photon: Vec<String>,
    /// Targets with photonuclear data.
    pub photonuclear: Vec<String>,
    /// Per subsection of [`SUBSECTIONS`], in that order, whether the release
    /// publishes it.
    pub transmutation: Vec<(&'static str, bool)>,
    /// The upstream files the release was built from, as its provenance
    /// lists them.
    pub sources: Vec<SourceArchive>,
}

impl LibraryContents {
    /// What `manifest`'s release holds.
    pub fn from_manifest(manifest: &Manifest, offline: bool) -> Self {
        // `<folder>/<name>.arrow/<file>`: the particle or subsection folder,
        // the entry's name and the file in it.
        let entries = manifest.files.iter().filter_map(|f| {
            let mut parts = f.path.splitn(3, '/');
            let folder = parts.next()?;
            let name = parts.next()?.strip_suffix(".arrow")?;
            let file = parts.next()?;
            Some((folder, name, file))
        });
        let mut neutron = BTreeSet::new();
        let mut covariances = vec![BTreeSet::new(); COVARIANCES.len()];
        let mut thermal = BTreeSet::new();
        let mut photon = BTreeSet::new();
        let mut photonuclear = BTreeSet::new();
        let mut subsections = BTreeSet::new();
        for (folder, name, file) in entries {
            match folder {
                "neutron" => {
                    neutron.insert(name);
                    if let Some(k) = COVARIANCES.iter().position(|(f, _)| *f == file) {
                        covariances[k].insert(name);
                    }
                }
                "thermal" => {
                    thermal.insert(name);
                }
                "photon" => {
                    photon.insert(name);
                }
                "photonuclear" => {
                    photonuclear.insert(name);
                }
                "transmutation" => {
                    subsections.insert(name);
                }
                _ => {}
            }
        }
        let owned = |set: BTreeSet<&str>| set.into_iter().map(str::to_string).collect();
        let sources = manifest
            .provenance
            .get("sources")
            .and_then(|s| s.as_array())
            .map(|list| {
                list.iter()
                    .filter_map(|s| {
                        Some(SourceArchive {
                            name: s.get("name")?.as_str()?.to_string(),
                            bytes: s.get("bytes").and_then(|b| b.as_u64()),
                            sha256: s.get("sha256").and_then(|h| h.as_str()).map(str::to_string),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            keyword: manifest.keyword.clone(),
            release: manifest.release.clone(),
            offline,
            neutron: owned(neutron),
            covariances: COVARIANCES
                .iter()
                .zip(covariances)
                .map(|((_, kind), set)| (*kind, owned(set)))
                .collect(),
            thermal_scattering: owned(thermal),
            photon: owned(photon),
            photonuclear: owned(photonuclear),
            transmutation: SUBSECTIONS
                .iter()
                .map(|(folder, name)| (*name, subsections.contains(folder)))
                .collect(),
            sources,
        }
    }
}

/// What the release `keyword` resolves to holds. Reads the release pointer
/// and manifest (cached after the first call), never the data.
pub fn library_contents(keyword: &str) -> Result<LibraryContents, String> {
    if !super::url_cache::KEYWORDS.contains(&keyword) {
        return Err(format!(
            "{keyword:?} is not a hosted library; the libraries are {}",
            super::url_cache::KEYWORDS.join(", ")
        ));
    }
    resolve(keyword)
}

#[cfg(feature = "download")]
fn resolve(keyword: &str) -> Result<LibraryContents, String> {
    let (manifest, offline) = super::release_cache::resolved_manifest(keyword)?;
    Ok(LibraryContents::from_manifest(&manifest, offline))
}

#[cfg(not(feature = "download"))]
fn resolve(keyword: &str) -> Result<LibraryContents, String> {
    Err(format!(
        "'{keyword}' cannot be looked up: this build has no data downloads"
    ))
}

/// Every hosted library keyword, each with what its release holds or why it
/// could not be resolved.
pub fn available_libraries() -> Vec<(&'static str, Result<LibraryContents, String>)> {
    super::url_cache::KEYWORDS
        .iter()
        .map(|keyword| (*keyword, resolve(keyword)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::release::ManifestFile;

    fn manifest(paths: &[&str]) -> Manifest {
        let json = serde_json::json!({
            "keyword": "lib", "release": "2026-01-01", "format_version": 2,
            "provenance": {"sources": [
                {"name": "a.zip", "bytes": 3, "sha256": "ab"},
                {"name": "b.tar.gz"},
            ]},
            "files": [],
        });
        let mut m: Manifest = serde_json::from_value(json).unwrap();
        m.files = paths
            .iter()
            .map(|p| ManifestFile {
                path: p.to_string(),
                bytes: 1,
                sha256: String::new(),
            })
            .collect();
        m
    }

    #[test]
    fn reads_what_a_release_holds_off_its_file_list() {
        let c = LibraryContents::from_manifest(
            &manifest(&[
                "neutron/index.json",
                "neutron/Fe56.arrow/reactions.arrow",
                "neutron/Fe56.arrow/covariance.arrow",
                "neutron/U235.arrow/reactions.arrow",
                "neutron/U235.arrow/nubar_covariance.arrow",
                "neutron/U235.arrow/covariance.arrow",
                "thermal/c_H_in_H2O.arrow/sab.arrow",
                "photon/Fe.arrow/element.arrow",
                "photonuclear/Fe56.arrow/photonuclear.arrow",
                "transmutation/decay.arrow/nuclides.arrow",
                "transmutation/branching.arrow/branching.arrow",
            ]),
            false,
        );
        assert_eq!(c.neutron, ["Fe56", "U235"]);
        assert_eq!(
            c.covariances[0],
            ("cross_section", vec!["Fe56".into(), "U235".into()])
        );
        assert_eq!(c.covariances[3], ("nubar", vec!["U235".into()]));
        assert_eq!(c.covariances[2], ("angular_distribution", vec![]));
        assert_eq!(c.thermal_scattering, ["c_H_in_H2O"]);
        assert_eq!(c.photon, ["Fe"]);
        assert_eq!(c.photonuclear, ["Fe56"]);
        assert_eq!(
            c.transmutation,
            [
                ("decay", true),
                ("reactions", false),
                ("fission_yields", false),
                ("isomeric_branching", true)
            ]
        );
        assert_eq!(c.sources.len(), 2);
        assert_eq!(c.sources[0].bytes, Some(3));
        assert_eq!(c.sources[1].sha256, None);
    }

    #[test]
    fn a_library_with_nothing_of_a_kind_says_so() {
        let c = LibraryContents::from_manifest(
            &manifest(&["transmutation/branching.arrow/branching.arrow"]),
            true,
        );
        assert!(c.neutron.is_empty() && c.photon.is_empty() && c.photonuclear.is_empty());
        assert!(c.covariances.iter().all(|(_, n)| n.is_empty()));
        assert_eq!(c.covariances.len(), COVARIANCES.len());
        assert!(c.offline);
    }

    #[test]
    fn an_unknown_keyword_is_refused_naming_the_libraries() {
        let e = library_contents("jeff-9").unwrap_err();
        assert!(e.contains("tendl-2025"), "{e}");
    }
}
