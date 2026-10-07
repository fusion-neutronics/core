//! Where each nuclide's covariance came from, and the problems its library's
//! own documentation states about it.
//!
//! The covariance is read from whatever library the cross sections are, and
//! some libraries say in their release papers that parts of their covariance
//! should not be relied on. Those statements are reported per nuclide, never
//! acted on: the evaluation is used as it is, and the reader is told what its
//! authors said about it.

use std::collections::{BTreeMap, BTreeSet};

/// One documented problem with a library's covariance.
struct KnownProblem {
    /// Library keyword prefix it applies to (`"fendl-3.2"` matches every
    /// FENDL-3.2 revision).
    library: &'static str,
    /// The nuclides it names, or every nuclide when empty.
    nuclides: &'static [&'static str],
    /// What the library's documentation says, with the source.
    problem: &'static str,
}

const NOBRE: &str = "Nobre et al., ENDF/B-VIII.1, arXiv:2511.03564, section IV";

const KNOWN_PROBLEMS: &[KnownProblem] = &[
    KnownProblem {
        library: "fendl-3.2",
        nuclides: &[],
        problem: "FENDL-3.2 states that its covariances should not be used: the files \
                  were assembled from different libraries and the cross sections and \
                  angular distributions were changed without updating the covariances \
                  (Schnabel et al., FENDL-3.2, arXiv:2311.10063)",
    },
    KnownProblem {
        library: "endf-b8.1",
        nuclides: &["Fe54", "Fe56", "Fe57", "Cr50", "Cr52", "Cr53"],
        problem: "ENDF/B-VIII.1 reuses the ENDF/B-VIII.0 covariance on revised mean values",
    },
    KnownProblem {
        library: "endf-b8.1",
        nuclides: &["Cu63", "Cu65"],
        problem: "ENDF/B-VIII.1 states no fast-range covariance for this evaluation",
    },
    KnownProblem {
        library: "endf-b8.1",
        nuclides: &["Ta181", "W182", "W183", "Pb206", "Pb207", "Pb208"],
        problem: "ENDF/B-VIII.1 flags this evaluation's resolved-resonance covariance as \
                  unrealistically low",
    },
    KnownProblem {
        library: "endf-b8.1",
        nuclides: &["Ta181"],
        problem: "ENDF/B-VIII.1 states this evaluation's unresolved-resonance covariance \
                  was overwritten in error",
    },
];

/// The library keyword for a library name, which a data folder may spell
/// differently: the converter stamps ENDF/B-VIII.1 as `endfb-8.1` in
/// `version.json`, where the keyword is `endf-b8.1`.
pub fn library_keyword(library: &str) -> &str {
    match library {
        "endfb-8.1" => "endf-b8.1",
        other => other,
    }
}

/// The documented problems with `nuclide`'s covariance in `library`, each
/// with its source.
pub fn known_problems(library: &str, nuclide: &str) -> Vec<String> {
    let library = library_keyword(library);
    KNOWN_PROBLEMS
        .iter()
        .filter(|p| library.starts_with(p.library))
        .filter(|p| p.nuclides.is_empty() || p.nuclides.contains(&nuclide))
        .map(|p| {
            if p.library == "endf-b8.1" {
                format!("{} ({NOBRE})", p.problem)
            } else {
                p.problem.to_string()
            }
        })
        .collect()
}

/// Where each of `nuclides`' covariance came from, and the documented problems
/// with it. `source_of` gives a nuclide's library and the evaluation's MAT,
/// either `None` when unknown.
///
/// The source reads `"<library>, MAT <n>"`, with `"unknown"` for a library a
/// folder did not record and the MAT left out when no block names one.
pub fn provenance(
    nuclides: &BTreeSet<String>,
    source_of: impl Fn(&str) -> (Option<String>, Option<i32>),
) -> (BTreeMap<String, String>, BTreeMap<String, Vec<String>>) {
    let mut sources = BTreeMap::new();
    let mut warnings = BTreeMap::new();
    for nuclide in nuclides {
        let (library, mat) = source_of(nuclide);
        let library = library.map(|l| library_keyword(&l).to_string());
        let name = library.clone().unwrap_or_else(|| "unknown".to_string());
        sources.insert(
            nuclide.clone(),
            match mat {
                Some(mat) if mat > 0 => format!("{name}, MAT {mat}"),
                _ => name,
            },
        );
        if let Some(library) = library {
            let problems = known_problems(&library, nuclide);
            if !problems.is_empty() {
                warnings.insert(nuclide.clone(), problems);
            }
        }
    }
    (sources, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fendl_32_covariance_is_warned_about() {
        assert_eq!(known_problems("fendl-3.2d", "Ni58").len(), 1);
        assert_eq!(known_problems("fendl-3.2b", "Fe56").len(), 1);
    }

    #[test]
    fn endf_b81_warnings_name_their_nuclides_only() {
        assert_eq!(known_problems("endf-b8.1", "Fe56").len(), 1);
        assert!(known_problems("endf-b8.1", "Fe56")[0].contains("arXiv:2511.03564"));
        assert!(known_problems("endf-b8.1", "Ni58").is_empty());
        // Ta181 has two: low resolved-resonance covariance, and the
        // unresolved one overwritten.
        assert_eq!(known_problems("endf-b8.1", "Ta181").len(), 2);
    }

    #[test]
    fn the_converters_spelling_of_endf_b81_is_recognised() {
        assert_eq!(library_keyword("endfb-8.1"), "endf-b8.1");
        assert_eq!(known_problems("endfb-8.1", "Fe56").len(), 1);
    }

    #[test]
    fn other_libraries_are_not_warned_about() {
        assert!(known_problems("jeff-4.0", "Fe56").is_empty());
        assert!(known_problems("tendl-2025", "Ta181").is_empty());
    }

    #[test]
    fn provenance_names_the_library_and_mat_and_collects_warnings() {
        let nuclides: BTreeSet<String> = ["Fe56", "Ni58", "Co59"].map(String::from).into();
        let (sources, warnings) = provenance(&nuclides, |n| match n {
            "Fe56" => (Some("endf-b8.1".to_string()), Some(2631)),
            "Ni58" => (Some("endf-b8.1".to_string()), None),
            _ => (None, None),
        });
        assert_eq!(sources["Fe56"], "endf-b8.1, MAT 2631");
        assert_eq!(sources["Ni58"], "endf-b8.1");
        assert_eq!(sources["Co59"], "unknown");
        assert_eq!(warnings.keys().collect::<Vec<_>>(), vec!["Fe56"]);
    }
}
