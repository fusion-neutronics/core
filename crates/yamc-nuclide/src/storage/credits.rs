//! Who made each library a keyword names, what to cite for it, and the terms
//! its data is redistributed under.
//!
//! Every result rests on an evaluated library, and its evaluators ask to be
//! cited wherever their data is used. This table is where that is written
//! down once: a run's `data_releases` records carry each library's
//! references, a library build writes them into the release's `NOTICE.md`,
//! and the docs' citation section (`docs/shared/cite-nuclear-data.md`) lists
//! the same DOIs, which a test below holds equal.
//!
//! Each reference is the one the library's developers ask for, from their
//! own citation page: TENDL's reference pages, the NEA's JEFF-4.0 page (which
//! asks for the data sets until the JEFF-4.0 paper is published), JAEA's
//! JENDL-5 terms (which require the JENDL-5 paper), and each library's own
//! paper otherwise.

use super::release::Citation;

/// A work to cite, and its DOI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference {
    /// Authors, title, journal or publisher, volume, pages and year.
    pub text: &'static str,
    /// The DOI, without the `https://doi.org/` prefix.
    pub doi: &'static str,
}

/// The credit due for the library a keyword names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Credit {
    pub keyword: &'static str,
    /// The library's own name, as its makers write it.
    pub library: &'static str,
    /// Who made it.
    pub made_by: &'static str,
    /// Where its makers publish it.
    pub page: &'static str,
    /// The terms its data is redistributed under, as far as its makers state
    /// them.
    pub terms: &'static str,
    /// What to cite, in the order its makers list it.
    pub references: &'static [Reference],
}

const CC_BY_4_0: &str = "Creative Commons Attribution 4.0 International (CC BY 4.0), \
                         https://creativecommons.org/licenses/by/4.0/";

const TENDL: &[Reference] = &[
    Reference {
        text: "A.J. Koning, D. Rochman, \"Modern Nuclear Data Evaluation with the TALYS \
               Code System\", Nuclear Data Sheets 113 (2012) 2841-2934",
        doi: "10.1016/j.nds.2012.11.002",
    },
    Reference {
        text: "A.J. Koning, D. Rochman, J.-Ch. Sublet, N. Dzysiuk, M. Fleming, S. van der \
               Marck, \"TENDL: Complete Nuclear Data Library for Innovative Nuclear Science \
               and Technology\", Nuclear Data Sheets 155 (2019) 1-55",
        doi: "10.1016/j.nds.2019.01.002",
    },
];

const TENDL_MADE_BY: &str = "A.J. Koning and D. Rochman, who founded TENDL in 2008 and \
                             make it with the TALYS code system, with processing and \
                             benchmarking by J.-Ch. Sublet";

const TENDL_TERMS: &str = "Publicly available from the TENDL project.";

/// One entry per library keyword.
pub const CREDITS: &[Credit] = &[
    Credit {
        keyword: "endf-b8.1",
        library: "ENDF/B-VIII.1",
        made_by: "the Cross Section Evaluation Working Group (CSEWG), released by the \
                  National Nuclear Data Center, Brookhaven National Laboratory",
        page: "https://www.nndc.bnl.gov/endf-library/B-VIII.1/summary/",
        terms: "Publicly available from the National Nuclear Data Center; the US \
                Government disclaimer applies.",
        references: &[
            Reference {
                text: "G.P.A. Nobre, R. Capote, M.T. Pigni et al., \"ENDF/B-VIII.1: Updated \
                       Nuclear Reaction Data Library for Science and Applications\", Nuclear \
                       Data Sheets 210 (2026) 1-224",
                doi: "10.1016/j.nds.2026.04.001",
            },
            Reference {
                text: "ENDF/B-VIII.1, Brookhaven National Laboratory (2024), the library itself",
                doi: "10.11578/endf/2571019",
            },
        ],
    },
    Credit {
        keyword: "jeff-4.0",
        library: "JEFF-4.0",
        made_by: "the Joint Evaluated Fission and Fusion (JEFF) project, coordinated by the \
                  OECD Nuclear Energy Agency Data Bank",
        page: "https://databank.io.oecd-nea.org/data/jeff/40/",
        terms: CC_BY_4_0,
        references: &[
            Reference {
                text: "Joint Evaluated Fission and Fusion Project, \"JEFF-4.0 Evaluated Data: \
                       neutron data\", OECD Nuclear Energy Agency (2025)",
                doi: "10.82555/e9ajn-a3p20",
            },
            Reference {
                text: "Joint Evaluated Fission and Fusion Project, \"JEFF-4.0 Evaluated Data: \
                       radioactive decay data\", OECD Nuclear Energy Agency (2025)",
                doi: "10.82555/tw0c6-t1386",
            },
            Reference {
                text: "Joint Evaluated Fission and Fusion Project, \"JEFF-4.0 Evaluated Data: \
                       neutron induced fission products yields\", OECD Nuclear Energy Agency \
                       (2025)",
                doi: "10.82555/trtwt-k2828",
            },
            Reference {
                text: "Joint Evaluated Fission and Fusion Project, \"JEFF-4.0 Evaluated Data: \
                       thermal scattering law data\", OECD Nuclear Energy Agency (2025)",
                doi: "10.82555/scmva-nqh68",
            },
        ],
    },
    Credit {
        keyword: "jendl-5.0",
        library: "JENDL-5",
        made_by: "the Nuclear Data Center of the Japan Atomic Energy Agency (JAEA)",
        page: "https://wwwndc.jaea.go.jp/jendl/j5/j5.html",
        terms: "Creative Commons Attribution 4.0 International (CC BY 4.0), \
                https://creativecommons.org/licenses/by/4.0/. JAEA requires the reference \
                below to be cited wherever JENDL-5 is used.",
        references: &[Reference {
            text: "O. Iwamoto, N. Iwamoto, S. Kunieda, F. Minato et al., \"Japanese evaluated \
                   nuclear data library version 5: JENDL-5\", Journal of Nuclear Science and \
                   Technology 60 (2023) 1-60",
            doi: "10.1080/00223131.2022.2141903",
        }],
    },
    Credit {
        keyword: "tendl-2025",
        library: "TENDL-2025",
        made_by: TENDL_MADE_BY,
        page: "https://tendl.imperial.ac.uk/tendl_2025/reference.html",
        terms: TENDL_TERMS,
        references: TENDL,
    },
    Credit {
        keyword: "tendl-2017",
        library: "TENDL-2017",
        made_by: TENDL_MADE_BY,
        page: "https://tendl.imperial.ac.uk/tendl_2017/reference.html",
        terms: TENDL_TERMS,
        references: TENDL,
    },
    Credit {
        keyword: "fendl-3.2d",
        library: "FENDL-3.2d",
        made_by: "the IAEA Nuclear Data Section, which coordinates the Fusion Evaluated \
                  Nuclear Data Library",
        page: "https://nds.iaea.org/fendl/",
        terms: "Publicly available from the IAEA Nuclear Data Section.",
        references: &[Reference {
            text: "G. Schnabel, D.L. Aldama, T. Bohm, U. Fischer, S. Kunieda, A. Trkov et al., \
                   \"FENDL: A library for fusion research and applications\", Nuclear Data \
                   Sheets 193 (2024) 1-78",
            doi: "10.1016/j.nds.2024.01.001",
        }],
    },
    Credit {
        keyword: "irdff-ii",
        library: "IRDFF-II",
        made_by: "the IAEA Nuclear Data Section, which coordinates the International Reactor \
                  Dosimetry and Fusion File",
        page: "https://www-nds.iaea.org/IRDFF/",
        terms: "Publicly available from the IAEA Nuclear Data Section.",
        references: &[Reference {
            text: "A. Trkov, P.J. Griffin, S.P. Simakov, L.R. Greenwood, K.I. Zolotarev, \
                   R. Capote et al., \"IRDFF-II: A New Neutron Metrology Library\", Nuclear \
                   Data Sheets 163 (2020) 1-108",
            doi: "10.1016/j.nds.2019.12.001",
        }],
    },
];

/// The credit due for the library `keyword` names, if it is one this table
/// knows.
pub fn credit(keyword: &str) -> Option<&'static Credit> {
    CREDITS.iter().find(|c| c.keyword == keyword)
}

/// What to cite for the library `keyword` names, as a run's results record
/// it. Empty for a name this table does not know, such as a library a user
/// converted themselves under a name of their own.
pub fn citations(keyword: &str) -> Vec<Citation> {
    credit(keyword)
        .map(|c| {
            c.references
                .iter()
                .map(|r| Citation {
                    reference: r.text.to_string(),
                    doi: r.doi.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The terms the library `keyword` names is redistributed under, or empty
/// for a name this table does not know.
pub fn terms(keyword: &str) -> String {
    credit(keyword)
        .map(|c| c.terms.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn every_keyword_has_one_credit() {
        let mut keywords: Vec<&str> = crate::url_cache::KEYWORDS.to_vec();
        let mut credited: Vec<&str> = CREDITS.iter().map(|c| c.keyword).collect();
        keywords.sort_unstable();
        credited.sort_unstable();
        assert_eq!(keywords, credited);
    }

    #[test]
    fn every_credit_has_a_reference_with_a_doi() {
        for c in CREDITS {
            assert!(!c.references.is_empty(), "{}", c.keyword);
            for r in c.references {
                assert!(r.doi.starts_with("10.") && r.doi.contains('/'), "{}", r.doi);
                assert!(!r.text.contains("  "), "{}: {}", c.keyword, r.text);
            }
        }
    }

    /// The docs' citation section and its BibTeX list exactly the DOIs this
    /// table does, so a reference changed in one place is changed in all.
    #[test]
    fn the_docs_cite_the_same_dois() {
        let shared = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/shared");
        let mut ours: Vec<String> = CREDITS
            .iter()
            .flat_map(|c| c.references.iter().map(|r| r.doi.to_lowercase()))
            .collect();
        ours.sort_unstable();
        ours.dedup();
        for file in ["cite-nuclear-data.md", "nuclear-data.bib"] {
            let text = std::fs::read_to_string(shared.join(file)).unwrap();
            let mut theirs: Vec<String> = text
                .match_indices("10.")
                .filter_map(|(i, _)| {
                    let doi: String = text[i..]
                        .chars()
                        .take_while(|ch| !ch.is_whitespace() && !"}])\"".contains(*ch))
                        .collect();
                    let doi = doi.trim_end_matches(['.', ',']).to_lowercase();
                    doi.contains('/').then_some(doi)
                })
                .collect();
            theirs.sort_unstable();
            theirs.dedup();
            assert_eq!(ours, theirs, "docs/shared/{file}");
        }
    }
}
