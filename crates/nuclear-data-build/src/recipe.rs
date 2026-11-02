//! What each library keyword is built from: its source files, where they are
//! published, the size and sha256 each is pinned to, and which sublibrary
//! each one feeds.
//!
//! A pin is the size and sha256 of the file as its host served it when the
//! recipe was written. Hosts have replaced files in place under the same name
//! before (the IAEA's IRDFF-II zip in December 2020), so the pin is what says
//! a file is still the one this recipe was written against, and a replaced
//! file fails the build rather than building something else.
//!
//! The order of a sublibrary's sources is the order they are applied in. That
//! matters for JENDL-5, which publishes corrections as update archives on top
//! of the base archive rather than re-cutting it: a file in a later archive
//! replaces the file of the same name from an earlier one.

/// The part of a library a source file feeds, which is also the ENDF
/// sublibrary its materials declare (NSUB in MF=1 MT=451).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sublibrary {
    Neutron,
    Photoatomic,
    AtomicRelaxation,
    Photonuclear,
    Decay,
    FissionYields,
}

impl Sublibrary {
    /// The NSUB every material of this sublibrary declares.
    pub fn nsub(self) -> i64 {
        match self {
            Sublibrary::Photonuclear => 0,
            Sublibrary::Photoatomic => 3,
            Sublibrary::Decay => 4,
            Sublibrary::AtomicRelaxation => 6,
            Sublibrary::Neutron => 10,
            Sublibrary::FissionYields => 11,
        }
    }

    /// The folder its extracted files go in, under `.build/sources/`.
    pub fn dir(self) -> &'static str {
        match self {
            Sublibrary::Neutron => "neutron",
            Sublibrary::Photoatomic => "photoatomic",
            Sublibrary::AtomicRelaxation => "atomic_relaxation",
            Sublibrary::Photonuclear => "photonuclear",
            Sublibrary::Decay => "decay",
            Sublibrary::FissionYields => "fission_yields",
        }
    }
}

/// The size and sha256 a source file is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pin {
    pub bytes: u64,
    /// Lowercase hex.
    pub sha256: &'static str,
}

/// One file a library is built from.
#[derive(Debug, Clone, Copy)]
pub struct Source {
    /// The file name, which is what it is saved as. Some URLs do not end in
    /// it (the NEA data bank's end in `/content`).
    pub name: &'static str,
    pub url: &'static str,
    /// `None` while the file has not been pinned yet: a recipe with an
    /// unpinned source is refused rather than built from whatever the host
    /// serves.
    pub pin: Option<Pin>,
    pub sublibrary: Sublibrary,
    /// Which members of an archive are evaluations of this sublibrary, by
    /// file name (without its folder). A plain file (a tape of several
    /// materials, as JEFF ships its decay data) is taken whole.
    pub keep: fn(&str) -> bool,
}

impl Source {
    /// How the file is unpacked, from its name.
    pub fn format(&self) -> Format {
        if self.name.ends_with(".zip") {
            Format::Zip
        } else if self.name.ends_with(".tar.gz") || self.name.ends_with(".tgz") {
            Format::TarGz
        } else {
            Format::Plain
        }
    }
}

/// How a source file is unpacked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Zip,
    TarGz,
    /// Used as it is.
    Plain,
}

/// A library keyword's sources.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub keyword: &'static str,
    /// In the order they are applied, per sublibrary.
    pub sources: &'static [Source],
}

impl Recipe {
    /// The sublibraries this recipe builds, in a fixed order.
    pub fn sublibraries(&self) -> Vec<Sublibrary> {
        let mut subs: Vec<Sublibrary> = self.sources.iter().map(|s| s.sublibrary).collect();
        subs.sort();
        subs.dedup();
        subs
    }

    /// The sources of one sublibrary, in the order they are applied.
    pub fn sources_of(&self, sublibrary: Sublibrary) -> impl Iterator<Item = &Source> {
        self.sources.iter().filter(move |s| s.sublibrary == sublibrary)
    }
}

/// The recipe for a library keyword.
pub fn recipe(keyword: &str) -> Result<&'static Recipe, String> {
    RECIPES.iter().find(|r| r.keyword == keyword).ok_or_else(|| {
        let known: Vec<&str> = RECIPES.iter().map(|r| r.keyword).collect();
        format!(
            "{keyword:?} is not a library this can build; the libraries are {}",
            known.join(", ")
        )
    })
}

/// Every keyword with a recipe.
pub fn keywords() -> impl Iterator<Item = &'static str> {
    RECIPES.iter().map(|r| r.keyword)
}

fn named(name: &str, prefix: &str, suffix: &str) -> bool {
    name.len() > prefix.len() + suffix.len() && name.starts_with(prefix) && name.ends_with(suffix)
}

/// `"n_026-Fe-056.dat"`, `"n_095-Am-242m1.dat"`; not the `_0K`, `_300K`,
/// `_act` and `_u20` variants some JENDL update archives carry beside the
/// main files, which belong to other sublibraries.
fn jendl_stem(name: &str, prefix: &str, suffix: &str) -> bool {
    named(name, prefix, suffix)
        && !name[prefix.len()..name.len() - suffix.len()].contains('_')
}

/// The sources tables. `pin: None` marks a file not pinned yet.
static RECIPES: &[Recipe] = &[
    Recipe {
        keyword: "endf-b8.1",
        sources: ENDF_B8_1,
    },
    Recipe {
        keyword: "jeff-4.0",
        sources: JEFF_4_0,
    },
    Recipe {
        keyword: "jendl-5.0",
        sources: JENDL_5_0,
    },
    Recipe {
        keyword: "tendl-2025",
        sources: TENDL_2025,
    },
    Recipe {
        keyword: "tendl-2017",
        sources: TENDL_2017,
    },
    Recipe {
        keyword: "fendl-3.2d",
        sources: FENDL_3_2D,
    },
    Recipe {
        keyword: "irdff-ii",
        sources: IRDFF_II,
    },
];

/// ENDF/B-VIII.1, released August 2024 by the CSEWG through the NNDC.
static ENDF_B8_1: &[Source] = &[
    Source {
        name: "neutrons-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/neutrons/neutrons-version.VIII.1.tar.gz",
        pin: Some(Pin {
            bytes: 343_724_780,
            sha256: "decff90016bfb5c25c8d4c7fb8d81f94ff2f104853165f0fe6a21bf61c4164e4",
        }),
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".endf"),
    },
    Source {
        name: "photoat-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/photoat/photoat-version.VIII.1.tar.gz",
        pin: Some(Pin {
            bytes: 34_304_129,
            sha256: "d39dfc8708b62dd8647bd7ff0af36741f629260d70724f71be13c949c62aa79f",
        }),
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| named(n, "photoat-", ".endf"),
    },
    Source {
        name: "atomic_relax-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/atomic_relax/atomic_relax-version.VIII.1.tar.gz",
        pin: Some(Pin {
            bytes: 1_345_840,
            sha256: "95845646a25982aa84d7e4c3459bd51332ee35f345f8d84a9d62181a448ccd23",
        }),
        sublibrary: Sublibrary::AtomicRelaxation,
        keep: |n| named(n, "atom-", ".endf"),
    },
    Source {
        name: "decay-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/decay/decay-version.VIII.1.tar.gz",
        pin: Some(Pin {
            bytes: 10_615_950,
            sha256: "8639581a4882b35855f7e98e14e537470c6b4b2a6d0673b8ac982baaa50a9d79",
        }),
        sublibrary: Sublibrary::Decay,
        keep: |n| named(n, "dec-", ".endf"),
    },
    Source {
        name: "nfy-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/nfy/nfy-version.VIII.1.tar.gz",
        pin: Some(Pin {
            bytes: 1_491_577,
            sha256: "2cf9b251c66fb0d8da96d03ad017fc3893cd3387a1d5f8cec37223c76319716a",
        }),
        sublibrary: Sublibrary::FissionYields,
        keep: |n| named(n, "nfy-", ".endf"),
    },
];

/// JEFF-4.0, released 2025 by the OECD NEA data bank, one DOI-minted record
/// per sublibrary. The download URL is the record's file endpoint with
/// `/content` appended. The neutron set is the 593-nuclide general-purpose
/// one. Decay and the fission yields are each one tape of every material.
/// Spontaneous fission yields (record tccek-e9s26) are left out: Cm244 is in
/// both yield tapes, and the chain keys yields by nuclide.
///
/// JEFF-4.0 has no photo-atomic sublibrary; its photon data is photonuclear.
static JEFF_4_0: &[Source] = &[
    Source {
        name: "JEFF40-Evaluations-Neutron-593.zip",
        url: "https://data.oecd-nea.org/api/records/e9ajn-a3p20/files/JEFF40-Evaluations-Neutron-593.zip/content",
        pin: Some(Pin {
            bytes: 608_170_633,
            sha256: "202c43e8bbea9ed530b6368cc7ef9fc919776e87bc79db5d4bbde937a999472f",
        }),
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n_", ".jeff"),
    },
    Source {
        name: "Radioactive_Decay_Data_JEFF-40.txt",
        url: "https://data.oecd-nea.org/api/records/tw0c6-t1386/files/Radioactive_Decay_Data_JEFF-40.txt/content",
        pin: Some(Pin {
            bytes: 45_166_329,
            sha256: "551f94f28c655aadf938a238103006698cbaa75c695e1b31b1c6c146d4f416e0",
        }),
        sublibrary: Sublibrary::Decay,
        keep: |_| true,
    },
    Source {
        name: "nf_Fission_Yields_JEFF-40.txt",
        url: "https://data.oecd-nea.org/api/records/trtwt-k2828/files/nf_Fission_Yields_JEFF-40.txt/content",
        pin: Some(Pin {
            bytes: 4_373_109,
            sha256: "bf0687d2e480ff67267dbaa9c988599306af8cfdb4ca9e3fa878fb2d55fd3aa8",
        }),
        sublibrary: Sublibrary::FissionYields,
        keep: |_| true,
    },
];

/// JENDL-5, released December 2021 by the JAEA Nuclear Data Center, with
/// its update archives up to neutron update 17 (March 2026).
///
/// JAEA serves these only to a web browser (a script gets a small HTML page
/// in place of the file), so they are fetched by hand once and pinned from
/// those copies. Not pinned yet.
///
/// The update numbering is one sequence across every sublibrary, and each
/// list is the base archive followed by its own updates in ascending order,
/// the order they are applied in. What each fixes is listed at
/// <https://wwwndc.jaea.go.jp/jendl/j5/JENDL-5_Errata.html>. The base
/// archives are not re-cut with the corrections (jendl5-n.tar.gz still
/// carries H-2 dated 2021-12-25, upd-7 carries it dated 2023-01-26), so the
/// updates are needed. Spontaneous fission yields (`_sf`) are left out, as
/// for JEFF.
static JENDL_5_0: &[Source] = &[
    jendl_neutron("jendl5-n.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n.tar.gz"),
    jendl_neutron("jendl5-n_upd1.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd1.tar.gz"),
    jendl_neutron("jendl5-n_upd6.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd6.tar.gz"),
    jendl_neutron("jendl5-n_upd7.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd7.tar.gz"),
    jendl_neutron("jendl5-n_upd10.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd10.tar.gz"),
    jendl_neutron("jendl5-n_upd11.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd11.tar.gz"),
    jendl_neutron("jendl5-n_upd12.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd12.tar.gz"),
    jendl_neutron("jendl5-n_upd14.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd14.tar.gz"),
    jendl_neutron("jendl5-n_upd17.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd17.tar.gz"),
    Source {
        name: "jendl5-pa.tar.gz",
        url: "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-pa.tar.gz",
        pin: None,
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| jendl_stem(n, "pa_", ".dat"),
    },
    Source {
        name: "jendl5-ar.tar.gz",
        url: "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-ar.tar.gz",
        pin: None,
        sublibrary: Sublibrary::AtomicRelaxation,
        keep: |n| jendl_stem(n, "ar_", ".dat"),
    },
    jendl_decay("jendl5-dec.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec.tar.gz"),
    jendl_decay("jendl5-dec_upd3.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec_upd3.tar.gz"),
    jendl_decay("jendl5-dec_upd5.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec_upd5.tar.gz"),
    jendl_decay("jendl5-dec_upd15.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec_upd15.tar.gz"),
    jendl_yields("jendl5-fpy.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-fpy.tar.gz"),
    jendl_yields("jendl5-fpy_upd2.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-fpy_upd2.tar.gz"),
    jendl_yields("jendl5-fpy_upd8.tar.gz", "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-fpy_upd8.tar.gz"),
];

const fn jendl_neutron(name: &'static str, url: &'static str) -> Source {
    Source {
        name,
        url,
        pin: None,
        sublibrary: Sublibrary::Neutron,
        keep: |n| jendl_stem(n, "n_", ".dat"),
    }
}

const fn jendl_decay(name: &'static str, url: &'static str) -> Source {
    Source {
        name,
        url,
        pin: None,
        sublibrary: Sublibrary::Decay,
        keep: |n| jendl_stem(n, "dec-", ".dat"),
    }
}

const fn jendl_yields(name: &'static str, url: &'static str) -> Source {
    Source {
        name,
        url,
        pin: None,
        sublibrary: Sublibrary::FissionYields,
        keep: |n| named(n, "fpy_", "_nf.dat") && !n["fpy_".len()..n.len() - 7].contains('_'),
    }
}

/// TENDL-2025, released July 2025 (TALYS-based, Imperial College London).
/// Neutron and photonuclear; TENDL has no decay or fission yield data.
static TENDL_2025: &[Source] = &[
    Source {
        name: "TENDL-n.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2025/tar_files/TENDL-n.tgz",
        pin: Some(Pin {
            bytes: 3_517_450_425,
            sha256: "e547527688506cbe09813364dcefa2aed11f474139bfa129d7cd4ca24fae21fa",
        }),
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".tendl"),
    },
    Source {
        name: "TENDL-g.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2025/tar_files/TENDL-g.tgz",
        pin: Some(Pin {
            bytes: 1_562_708_119,
            sha256: "01ae9f05142aaa201641604970f424fcfda6554b293e95dd9ff53a9119bbc378",
        }),
        sublibrary: Sublibrary::Photonuclear,
        keep: |n| named(n, "g-", ".tendl"),
    },
];

/// TENDL-2017 (TALYS-based, PSI), neutron and photonuclear.
static TENDL_2017: &[Source] = &[
    Source {
        name: "TENDL-n.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2017/tar_files/TENDL-n.tgz",
        pin: Some(Pin {
            bytes: 3_285_263_365,
            sha256: "e186fea3638666ed76612997a5d10839f251af4728a6c57fea1f5aba31f394a5",
        }),
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".tendl"),
    },
    Source {
        name: "TENDL-g.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2017/tar_files/TENDL-g.tgz",
        pin: Some(Pin {
            bytes: 1_357_569_101,
            sha256: "dfa0df7f4a6c7bb467bcc905a4192a42fbfca154fea8cbe60f3c86db813ed23a",
        }),
        sublibrary: Sublibrary::Photonuclear,
        keep: |n| named(n, "g-", ".tendl"),
    },
];

/// FENDL-3.2d, the IAEA Fusion Evaluated Nuclear Data Library, published
/// under the per-release `fendl_library/websites` tree with a `-T1-` infix.
///
/// The IAEA serves these only to a web browser (a Cloudflare bot check), so
/// they are fetched by hand once and pinned from those copies. Not pinned yet.
/// FENDL's atom archive carries photo-atomic data only, with no atomic
/// relaxation (MF=28).
static FENDL_3_2D: &[Source] = &[
    Source {
        name: "fendl-3.2d-T1-neutron-endf.zip",
        url: "https://nds.iaea.org/fendl_library/websites/fendl32d-T1/data/neutron/fendl-3.2d-T1-neutron-endf.zip",
        pin: None,
        sublibrary: Sublibrary::Neutron,
        keep: |n| n.ends_with(".endf"),
    },
    Source {
        name: "fendl-3.2d-T1-atom-endf.zip",
        url: "https://nds.iaea.org/fendl_library/websites/fendl32d-T1/data/atom/fendl-3.2d-T1-atom-endf.zip",
        pin: None,
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| n.ends_with(".endf"),
    },
];

/// IRDFF-II, the IAEA International Reactor Dosimetry and Fusion File,
/// released January 2020 with the files last updated on 9 December 2020
/// (A. Trkov et al., Nucl. Data Sheets 163 (2020) 1, doi:10.1016/j.nds.2019.12.001).
/// One zip holding one 70-material tape. The IAEA serves it only to a web
/// browser; the pin is of the zip as served on 2026-09-27, with a
/// Last-Modified of 2020-12-08.
static IRDFF_II: &[Source] = &[Source {
    name: "IRDFF-II_ENDF.zip",
    url: "https://www-nds.iaea.org/IRDFF/IRDFF-II_ENDF.zip",
    pin: Some(Pin {
        bytes: 33_372_016,
        sha256: "225b25787f6d9b54a9c28ebf075ccc715f69358be181fed27d5cc315cf8744db",
    }),
    sublibrary: Sublibrary::Neutron,
    keep: |n| n == "IRDFF-II.endf",
}];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hosted_keyword_has_a_recipe() {
        let got: Vec<&str> = keywords().collect();
        assert_eq!(
            got,
            [
                "endf-b8.1",
                "jeff-4.0",
                "jendl-5.0",
                "tendl-2025",
                "tendl-2017",
                "fendl-3.2d",
                "irdff-ii"
            ]
        );
        let err = recipe("endf-b8.0").unwrap_err();
        assert!(err.contains("endf-b8.1, jeff-4.0"), "{err}");
    }

    #[test]
    fn pins_are_sizes_and_lowercase_sha256s() {
        for r in RECIPES {
            let mut names = std::collections::HashSet::new();
            for s in r.sources {
                assert!(names.insert(s.name), "{} names {} twice", r.keyword, s.name);
                assert!(s.url.starts_with("https://"), "{}", s.url);
                assert!(!s.name.contains(['/', '\\']), "{}", s.name);
                if let Some(pin) = s.pin {
                    assert!(pin.bytes > 0, "{}", s.name);
                    assert_eq!(pin.sha256.len(), 64, "{} {}", r.keyword, s.name);
                    assert!(
                        pin.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                        "{}",
                        s.name
                    );
                }
            }
        }
    }

    #[test]
    fn jendl_updates_follow_their_base_in_ascending_order() {
        let r = recipe("jendl-5.0").unwrap();
        for sub in r.sublibraries() {
            let names: Vec<&str> = r.sources_of(sub).map(|s| s.name).collect();
            let numbers: Vec<u32> = names
                .iter()
                .skip(1)
                .map(|n| {
                    let n = n.split("_upd").nth(1).unwrap();
                    n.trim_end_matches(".tar.gz").parse().unwrap()
                })
                .collect();
            assert!(!names[0].contains("_upd"), "{names:?}");
            assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{names:?}");
        }
        let neutron: Vec<&str> = r.sources_of(Sublibrary::Neutron).map(|s| s.name).collect();
        assert_eq!(neutron.len(), 9);
        assert_eq!(*neutron.last().unwrap(), "jendl5-n_upd17.tar.gz");
    }

    #[test]
    fn keep_filters() {
        let r = recipe("jendl-5.0").unwrap();
        let keep = |sub, n| r.sources_of(sub).next().map(|s| (s.keep)(n)).unwrap();
        assert!(keep(Sublibrary::Neutron, "n_026-Fe-056.dat"));
        assert!(keep(Sublibrary::Neutron, "n_095-Am-242m1.dat"));
        assert!(!keep(Sublibrary::Neutron, "n_026-Fe-056_300K.dat"));
        assert!(!keep(Sublibrary::Neutron, "n_026-Fe-056_act.dat"));
        assert!(keep(Sublibrary::FissionYields, "fpy_092-U-235_nf.dat"));
        assert!(!keep(Sublibrary::FissionYields, "fpy_092-U-238_sf.dat"));
        assert!(keep(Sublibrary::Decay, "dec-027-Co-060m.dat"));
        let b = recipe("endf-b8.1").unwrap();
        let keep = |sub, n| b.sources_of(sub).next().map(|s| (s.keep)(n)).unwrap();
        assert!(keep(Sublibrary::Neutron, "n-026_Fe_056.endf"));
        assert!(!keep(Sublibrary::Neutron, "README.md"));
        assert!(!keep(Sublibrary::Neutron, "n-.endf"));
    }

    #[test]
    fn formats_follow_the_name() {
        let format = |k, n: &str| {
            recipe(k)
                .unwrap()
                .sources
                .iter()
                .find(|s| s.name == n)
                .unwrap()
                .format()
        };
        assert_eq!(format("jeff-4.0", "JEFF40-Evaluations-Neutron-593.zip"), Format::Zip);
        assert_eq!(format("jeff-4.0", "Radioactive_Decay_Data_JEFF-40.txt"), Format::Plain);
        assert_eq!(format("tendl-2025", "TENDL-n.tgz"), Format::TarGz);
        assert_eq!(format("endf-b8.1", "decay-version.VIII.1.tar.gz"), Format::TarGz);
    }
}
