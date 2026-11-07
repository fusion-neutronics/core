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

use endf::njoy::Flavour;

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

/// A part of a library: a particle folder, or a subsection of the
/// transmutation chain. A build makes the parts it is asked for, by default
/// every part the published release of that library carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Part {
    Neutron,
    Photon,
    Photonuclear,
    Decay,
    Reactions,
    FissionYields,
    Branching,
}

impl Part {
    /// Every part, in the order they are listed.
    pub const ALL: [Part; 7] = [
        Part::Neutron,
        Part::Photon,
        Part::Photonuclear,
        Part::Decay,
        Part::Reactions,
        Part::FissionYields,
        Part::Branching,
    ];

    /// The name a caller gives it, which is also the name of its folder
    /// (`neutron/`) or chain subsection (`transmutation/decay.arrow/`).
    pub fn name(self) -> &'static str {
        match self {
            Part::Neutron => "neutron",
            Part::Photon => "photon",
            Part::Photonuclear => "photonuclear",
            Part::Decay => "decay",
            Part::Reactions => "reactions",
            Part::FissionYields => "fission_yields",
            Part::Branching => "branching",
        }
    }

    /// The part a name names.
    pub fn parse(name: &str) -> Result<Part, String> {
        Part::ALL
            .into_iter()
            .find(|p| p.name() == name)
            .ok_or_else(|| {
                let names: Vec<&str> = Part::ALL.iter().map(|p| p.name()).collect();
                format!(
                    "{name:?} is not a part of a library; the parts are {}",
                    names.join(", ")
                )
            })
    }

    /// Whether it is a subsection of the transmutation chain.
    pub fn is_chain(self) -> bool {
        matches!(
            self,
            Part::Decay | Part::Reactions | Part::FissionYields | Part::Branching
        )
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
    /// What the file must be: a copy of any other size or sha256 is refused,
    /// whoever served it.
    pub pin: Pin,
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

/// A library keyword's sources, and what its published release holds.
#[derive(Debug, Clone, Copy)]
pub struct Recipe<'a> {
    pub keyword: &'static str,
    /// In the order they are applied, per sublibrary.
    pub sources: &'a [Source],
    /// The parts the published release carries, which is what a build makes
    /// unless asked for others.
    pub parts: &'static [Part],
    /// The NJOY its neutron evaluations are processed with.
    pub njoy: Flavour,
    /// The library whose decay and fission yield evaluations its chain is
    /// assembled with, when it has none of its own (TENDL, FENDL and IRDFF
    /// use ENDF/B-VIII.1's). Those sources are in `sources` like any other,
    /// and the chain records this as its `decay_library`, so it never reads
    /// as though that half were this library's. `None` for a library that
    /// evaluates its own.
    pub decay_library: Option<&'static str>,
    /// The evaluations NJOY cannot process, as `"<particle>/<Name>"`. A full
    /// build without them is complete; one that converts any of them says
    /// the list has gone stale. Each recipe says why next to its list.
    pub unconvertible: &'static [&'static str],
    /// The free disk a full build needs at its peak, in bytes: the source
    /// archives, what they unpack to, NJOY's scratch and the release. Rounded
    /// up from the measured builds.
    pub disk_bytes: u64,
}

impl Recipe<'_> {
    /// The sublibraries this recipe builds, in a fixed order.
    pub fn sublibraries(&self) -> Vec<Sublibrary> {
        let mut subs: Vec<Sublibrary> = self.sources.iter().map(|s| s.sublibrary).collect();
        subs.sort();
        subs.dedup();
        subs
    }

    /// The sources of one sublibrary, in the order they are applied.
    pub fn sources_of(&self, sublibrary: Sublibrary) -> impl Iterator<Item = &Source> {
        self.sources
            .iter()
            .filter(move |s| s.sublibrary == sublibrary)
    }
}

/// The recipe for a library keyword.
pub fn recipe(keyword: &str) -> Result<&'static Recipe<'static>, String> {
    RECIPES
        .iter()
        .find(|r| r.keyword == keyword)
        .ok_or_else(|| {
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
    named(name, prefix, suffix) && !name[prefix.len()..name.len() - suffix.len()].contains('_')
}

const GB: u64 = 1_000_000_000;

/// Every part of a library with its own complete chain and photon data.
const FULL: &[Part] = &[
    Part::Neutron,
    Part::Photon,
    Part::Decay,
    Part::Reactions,
    Part::FissionYields,
    Part::Branching,
];

/// The sources tables.
static RECIPES: &[Recipe<'static>] = &[
    Recipe {
        keyword: "endf-b8.1",
        sources: ENDF_B8_1,
        parts: FULL,
        njoy: Flavour::Upstream,
        decay_library: None,
        unconvertible: &[],
        disk_bytes: 15 * GB,
    },
    Recipe {
        keyword: "jeff-4.0",
        sources: JEFF_4_0,
        // No photo-atomic sublibrary.
        parts: &[
            Part::Neutron,
            Part::Decay,
            Part::Reactions,
            Part::FissionYields,
            Part::Branching,
        ],
        njoy: Flavour::Upstream,
        decay_library: None,
        unconvertible: &[],
        disk_bytes: 15 * GB,
    },
    Recipe {
        keyword: "jendl-5.0",
        sources: JENDL_5_0,
        parts: FULL,
        njoy: Flavour::Upstream,
        decay_library: None,
        // Every one of its 795 neutron evaluations converts.
        unconvertible: &[],
        disk_bytes: 50 * GB,
    },
    Recipe {
        keyword: "tendl-2025",
        sources: TENDL_2025,
        parts: TENDL_PARTS,
        njoy: Flavour::Upstream,
        decay_library: Some("endf-b8.1"),
        // PURR stops on both building the unresolved range's probability
        // tables ("bad value for nres or emin>emax, increase dmin", STOP
        // 77): their unresolved range has no resonances to average over, or
        // a lower bound above its upper. Reproduced converting each alone on
        // 2026-08-08. Both are short-lived and far from stability. Ne24 also
        // fails in TENDL-2017, for another reason (see there).
        unconvertible: &["neutron/Ne24", "neutron/S42"],
        disk_bytes: 60 * GB,
    },
    Recipe {
        keyword: "tendl-2017",
        sources: TENDL_2017,
        parts: TENDL_PARTS,
        njoy: Flavour::Upstream,
        decay_library: Some("endf-b8.1"),
        // HEATR stops on all twelve ("awp is zero for mt 16 particle 4006",
        // STOP 77). Eleven declare their MF=6 MT=16 (n,2n) products with
        // AWP=0 against a ZAP=0 placeholder, which sixbar cannot average;
        // B10 stops in the same module for a reason not yet run down. The
        // defect is in the evaluations: each reproduces converted alone, and
        // no published TENDL-2017 has carried them.
        unconvertible: &[
            "neutron/B10",
            "neutron/Be7",
            "neutron/Be10",
            "neutron/Be11",
            "neutron/C10",
            "neutron/C11",
            "neutron/C14",
            "neutron/C15",
            "neutron/N13",
            "neutron/Ne24",
            "neutron/O14",
            "neutron/O15",
        ],
        disk_bytes: 60 * GB,
    },
    Recipe {
        keyword: "fendl-3.2d",
        sources: FENDL_3_2D,
        // No decay or fission yield evaluations of its own: the chain is
        // its reactions and their isomeric branching.
        parts: &[
            Part::Neutron,
            Part::Photon,
            Part::Reactions,
            Part::Branching,
        ],
        njoy: Flavour::IaeaFork,
        decay_library: Some("endf-b8.1"),
        unconvertible: &[],
        disk_bytes: 8 * GB,
    },
    Recipe {
        keyword: "irdff-ii",
        sources: IRDFF_II,
        // A dosimetry file: its release is the isomeric branching of its
        // isotopic evaluations, read against ENDF/B-VIII.1's decay data.
        parts: &[Part::Branching],
        njoy: Flavour::Upstream,
        decay_library: Some("endf-b8.1"),
        unconvertible: &[],
        disk_bytes: GB,
    },
];

/// TENDL's parts: neutron and photonuclear data, and a chain of reactions
/// and their isomeric branching (TENDL has no decay or fission yield data).
const TENDL_PARTS: &[Part] = &[
    Part::Neutron,
    Part::Photonuclear,
    Part::Reactions,
    Part::Branching,
];

/// ENDF/B-VIII.1, released August 2024 by the CSEWG through the NNDC.
static ENDF_B8_1: &[Source] = &[
    Source {
        name: "neutrons-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/neutrons/neutrons-version.VIII.1.tar.gz",
        pin: Pin {
            bytes: 343_724_780,
            sha256: "decff90016bfb5c25c8d4c7fb8d81f94ff2f104853165f0fe6a21bf61c4164e4",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".endf"),
    },
    Source {
        name: "photoat-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/photoat/photoat-version.VIII.1.tar.gz",
        pin: Pin {
            bytes: 34_304_129,
            sha256: "d39dfc8708b62dd8647bd7ff0af36741f629260d70724f71be13c949c62aa79f",
        },
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| named(n, "photoat-", ".endf"),
    },
    Source {
        name: "atomic_relax-version.VIII.1.tar.gz",
        url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/atomic_relax/atomic_relax-version.VIII.1.tar.gz",
        pin: Pin {
            bytes: 1_345_840,
            sha256: "95845646a25982aa84d7e4c3459bd51332ee35f345f8d84a9d62181a448ccd23",
        },
        sublibrary: Sublibrary::AtomicRelaxation,
        keep: |n| named(n, "atom-", ".endf"),
    },
    ENDF_B8_1_DECAY,
    ENDF_B8_1_YIELDS,
];

/// ENDF/B-VIII.1's decay sublibrary, which the libraries without decay data
/// of their own assemble their chain with.
const ENDF_B8_1_DECAY: Source = Source {
    name: "decay-version.VIII.1.tar.gz",
    url:
        "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/decay/decay-version.VIII.1.tar.gz",
    pin: Pin {
        bytes: 10_615_950,
        sha256: "8639581a4882b35855f7e98e14e537470c6b4b2a6d0673b8ac982baaa50a9d79",
    },
    sublibrary: Sublibrary::Decay,
    keep: |n| named(n, "dec-", ".endf"),
};

/// ENDF/B-VIII.1's neutron-induced fission yields, likewise.
const ENDF_B8_1_YIELDS: Source = Source {
    name: "nfy-version.VIII.1.tar.gz",
    url: "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/nfy/nfy-version.VIII.1.tar.gz",
    pin: Pin {
        bytes: 1_491_577,
        sha256: "2cf9b251c66fb0d8da96d03ad017fc3893cd3387a1d5f8cec37223c76319716a",
    },
    sublibrary: Sublibrary::FissionYields,
    keep: |n| named(n, "nfy-", ".endf"),
};

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
        pin: Pin {
            bytes: 608_170_633,
            sha256: "202c43e8bbea9ed530b6368cc7ef9fc919776e87bc79db5d4bbde937a999472f",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n_", ".jeff"),
    },
    Source {
        name: "Radioactive_Decay_Data_JEFF-40.txt",
        url: "https://data.oecd-nea.org/api/records/tw0c6-t1386/files/Radioactive_Decay_Data_JEFF-40.txt/content",
        pin: Pin {
            bytes: 45_166_329,
            sha256: "551f94f28c655aadf938a238103006698cbaa75c695e1b31b1c6c146d4f416e0",
        },
        sublibrary: Sublibrary::Decay,
        keep: |_| true,
    },
    Source {
        name: "nf_Fission_Yields_JEFF-40.txt",
        url: "https://data.oecd-nea.org/api/records/trtwt-k2828/files/nf_Fission_Yields_JEFF-40.txt/content",
        pin: Pin {
            bytes: 4_373_109,
            sha256: "bf0687d2e480ff67267dbaa9c988599306af8cfdb4ca9e3fa878fb2d55fd3aa8",
        },
        sublibrary: Sublibrary::FissionYields,
        keep: |_| true,
    },
];

/// JENDL-5, released December 2021 by the JAEA Nuclear Data Center, with
/// its update archives up to neutron update 17 (March 2026).
///
/// JAEA serves these only to a web browser (a script gets a small HTML page
/// in place of the file), so they are fetched by hand. The pins are of the
/// copies downloaded on 2026-11-03.
///
/// The update numbering is one sequence across every sublibrary, and each
/// list is the base archive followed by its own updates in ascending order,
/// the order they are applied in. Decay update 5 and fission-yield update 8
/// re-issue their whole sublibrary (every file of the base and of the
/// earlier updates, checked on the pinned copies), so those lists start
/// there: the archives before them would be overwritten file for file. What each fixes is listed at
/// <https://wwwndc.jaea.go.jp/jendl/j5/JENDL-5_Errata.html>. The base
/// archives are not re-cut with the corrections (jendl5-n.tar.gz still
/// carries H-2 dated 2021-12-25, upd-7 carries it dated 2023-01-26), so the
/// updates are needed. Spontaneous fission yields (`_sf`) are left out, as
/// for JEFF.
static JENDL_5_0: &[Source] = &[
    jendl_neutron(
        "jendl5-n.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n.tar.gz",
        Pin {
            bytes: 4_407_579_521,
            sha256: "5cb3b095207b61a7d29d0a8ceb15baa1fc88a5c479cac9aa3d6903fefb893906",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd1.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd1.tar.gz",
        Pin {
            bytes: 2_517_653,
            sha256: "602ab4ac462f39614467bf66a6e5a455330c89d4bf3e1f59820746de3e0a27f4",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd6.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd6.tar.gz",
        Pin {
            bytes: 10_952_213,
            sha256: "377e4a68bfac7044ee88ce28adb2145d28be11617cee5eba93b27e4ddbcd0f75",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd7.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd7.tar.gz",
        Pin {
            bytes: 251_078,
            sha256: "05b344a9b07cb0fcabb6e94af66104dbe5eba5b2c9cd426b9701b6029100f257",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd10.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd10.tar.gz",
        Pin {
            bytes: 278_562_947,
            sha256: "88c9452573702f343d0bc9d6c81ee3d1b425b589f4cc15b7614b53146278cb34",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd11.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd11.tar.gz",
        Pin {
            bytes: 488_208_183,
            sha256: "33196a9d20168161f471402dbe0874d16036697b4a3d0a707e98086ee8640370",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd12.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd12.tar.gz",
        Pin {
            bytes: 8_015_454,
            sha256: "f9dd19f3400adf8ef9e41247d1eab90644a7fa490117cc3e672d0c8c5fcc0c74",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd14.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd14.tar.gz",
        Pin {
            bytes: 11_327_034,
            sha256: "f18e884699b6bc6ff433f3283222771d54ff55e592b49ff34c82838a6f235456",
        },
    ),
    jendl_neutron(
        "jendl5-n_upd17.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-n_upd17.tar.gz",
        Pin {
            bytes: 107_845_633,
            sha256: "f6dce831bcf6b55a94a25d8a5315aea30ae9699f849aee5a4550e00a9354e93c",
        },
    ),
    Source {
        name: "jendl5-pa.tar.gz",
        url: "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-pa.tar.gz",
        pin: Pin {
            bytes: 31_892_241,
            sha256: "235926e32b92af0a4c7ffc1301befed2ddd01c4e2fe2b38d39a2b2cf0b0ea4d8",
        },
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| jendl_stem(n, "pa_", ".dat"),
    },
    Source {
        name: "jendl5-ar.tar.gz",
        url: "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-ar.tar.gz",
        pin: Pin {
            bytes: 1_618_497,
            sha256: "693170b7a42a8bbc47569ee504879bb3e32234664ea9761cd99e7c5831948842",
        },
        sublibrary: Sublibrary::AtomicRelaxation,
        keep: |n| jendl_stem(n, "ar_", ".dat"),
    },
    jendl_decay(
        "jendl5-dec_upd5.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec_upd5.tar.gz",
        Pin {
            bytes: 29_041_187,
            sha256: "5472ea8696366308ff5bad0338cbcfd617268da9c0ea6f43dcf3aeb9d1380610",
        },
    ),
    jendl_decay(
        "jendl5-dec_upd15.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-dec_upd15.tar.gz",
        Pin {
            bytes: 1_475_461,
            sha256: "4d076dd1bc7297d40794db69090a0663831428b1450a77b3ae30342d789b7b34",
        },
    ),
    jendl_yields(
        "jendl5-fpy_upd8.tar.gz",
        "https://wwwndc.jaea.go.jp/ftpnd/ftp/JENDL/jendl5-fpy_upd8.tar.gz",
        Pin {
            bytes: 2_330_384,
            sha256: "9a0e33da49958aa7766b013a5dd19da7d8eb6918526b1687dfb769fae6ebb553",
        },
    ),
];

const fn jendl_neutron(name: &'static str, url: &'static str, pin: Pin) -> Source {
    Source {
        name,
        url,
        pin,
        sublibrary: Sublibrary::Neutron,
        keep: |n| jendl_stem(n, "n_", ".dat"),
    }
}

const fn jendl_decay(name: &'static str, url: &'static str, pin: Pin) -> Source {
    Source {
        name,
        url,
        pin,
        sublibrary: Sublibrary::Decay,
        keep: |n| jendl_stem(n, "dec-", ".dat"),
    }
}

const fn jendl_yields(name: &'static str, url: &'static str, pin: Pin) -> Source {
    Source {
        name,
        url,
        pin,
        sublibrary: Sublibrary::FissionYields,
        keep: |n| named(n, "fpy_", "_nf.dat") && !n["fpy_".len()..n.len() - 7].contains('_'),
    }
}

/// TENDL-2025, released July 2025 (TALYS-based, Imperial College London).
/// Neutron and photonuclear; TENDL has no decay or fission yield data, so
/// its chain is assembled with ENDF/B-VIII.1's.
static TENDL_2025: &[Source] = &[
    Source {
        name: "TENDL-n.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2025/tar_files/TENDL-n.tgz",
        pin: Pin {
            bytes: 3_517_450_425,
            sha256: "e547527688506cbe09813364dcefa2aed11f474139bfa129d7cd4ca24fae21fa",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".tendl"),
    },
    Source {
        name: "TENDL-g.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2025/tar_files/TENDL-g.tgz",
        pin: Pin {
            bytes: 1_562_708_119,
            sha256: "01ae9f05142aaa201641604970f424fcfda6554b293e95dd9ff53a9119bbc378",
        },
        sublibrary: Sublibrary::Photonuclear,
        keep: |n| named(n, "g-", ".tendl"),
    },
    ENDF_B8_1_DECAY,
    ENDF_B8_1_YIELDS,
];

/// TENDL-2017 (TALYS-based, PSI), neutron and photonuclear, with
/// ENDF/B-VIII.1's decay and fission yields for its chain.
static TENDL_2017: &[Source] = &[
    Source {
        name: "TENDL-n.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2017/tar_files/TENDL-n.tgz",
        pin: Pin {
            bytes: 3_285_263_365,
            sha256: "e186fea3638666ed76612997a5d10839f251af4728a6c57fea1f5aba31f394a5",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| named(n, "n-", ".tendl"),
    },
    Source {
        name: "TENDL-g.tgz",
        url: "https://tendl.imperial.ac.uk/tendl_2017/tar_files/TENDL-g.tgz",
        pin: Pin {
            bytes: 1_357_569_101,
            sha256: "dfa0df7f4a6c7bb467bcc905a4192a42fbfca154fea8cbe60f3c86db813ed23a",
        },
        sublibrary: Sublibrary::Photonuclear,
        keep: |n| named(n, "g-", ".tendl"),
    },
    ENDF_B8_1_DECAY,
    ENDF_B8_1_YIELDS,
];

/// FENDL-3.2d, the IAEA Fusion Evaluated Nuclear Data Library, published
/// under the per-release `fendl_library/websites` tree with a `-T1-` infix.
///
/// The IAEA serves these only to a web browser (a Cloudflare bot check), so
/// they are fetched by hand. The pins are of the copies downloaded on
/// 2026-11-03.
/// FENDL's atom archive carries photo-atomic data only, with no atomic
/// relaxation (MF=28). FENDL has no decay or fission yield data, so its
/// chain is assembled with ENDF/B-VIII.1's.
static FENDL_3_2D: &[Source] = &[
    Source {
        name: "fendl-3.2d-T1-neutron-endf.zip",
        url: "https://nds.iaea.org/fendl_library/websites/fendl32d-T1/data/neutron/fendl-3.2d-T1-neutron-endf.zip",
        pin: Pin {
            bytes: 265_466_831,
            sha256: "e43e5d885d5052d4ebb309f348a33e0e3afff03d7357dd2772381206e91e292c",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| n.ends_with(".endf"),
    },
    Source {
        name: "fendl-3.2d-T1-atom-endf.zip",
        url: "https://nds.iaea.org/fendl_library/websites/fendl32d-T1/data/atom/fendl-3.2d-T1-atom-endf.zip",
        pin: Pin {
            bytes: 4_254_660,
            sha256: "74420c0ab33612370b343119042fa4130ca442ec88e84956518334293d4b4900",
        },
        sublibrary: Sublibrary::Photoatomic,
        keep: |n| n.ends_with(".endf"),
    },
    ENDF_B8_1_DECAY,
    ENDF_B8_1_YIELDS,
];

/// IRDFF-II, the IAEA International Reactor Dosimetry and Fusion File,
/// released January 2020 with the files last updated on 9 December 2020
/// (A. Trkov et al., Nucl. Data Sheets 163 (2020) 1, doi:10.1016/j.nds.2019.12.001).
/// One zip holding one 70-material tape. The IAEA serves it only to a web
/// browser; the pin is of the zip as served on 2026-09-27, with a
/// Last-Modified of 2020-12-08. Its branching is read against
/// ENDF/B-VIII.1's decay data, which it has none of.
static IRDFF_II: &[Source] = &[
    Source {
        name: "IRDFF-II_ENDF.zip",
        url: "https://www-nds.iaea.org/IRDFF/IRDFF-II_ENDF.zip",
        pin: Pin {
            bytes: 33_372_016,
            sha256: "225b25787f6d9b54a9c28ebf075ccc715f69358be181fed27d5cc315cf8744db",
        },
        sublibrary: Sublibrary::Neutron,
        keep: |n| n == "IRDFF-II.endf",
    },
    ENDF_B8_1_DECAY,
];

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
                let pin = s.pin;
                assert!(pin.bytes > 0, "{}", s.name);
                assert_eq!(pin.sha256.len(), 64, "{} {}", r.keyword, s.name);
                assert!(
                    pin.sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{}",
                    s.name
                );
            }
        }
    }

    #[test]
    fn every_chain_part_has_the_evaluations_it_is_built_from() {
        for r in RECIPES {
            let subs = r.sublibraries();
            let has = |s| subs.contains(&s);
            for part in r.parts {
                let needs: &[Sublibrary] = match part {
                    Part::Neutron => &[Sublibrary::Neutron],
                    Part::Photon => &[Sublibrary::Photoatomic],
                    Part::Photonuclear => &[Sublibrary::Photonuclear],
                    Part::Decay | Part::Reactions | Part::FissionYields => &[
                        Sublibrary::Neutron,
                        Sublibrary::Decay,
                        Sublibrary::FissionYields,
                    ],
                    Part::Branching => &[Sublibrary::Neutron, Sublibrary::Decay],
                };
                for s in needs {
                    assert!(has(*s), "{} {} needs {}", r.keyword, part.name(), s.dir());
                }
            }
            // A library that publishes its own decay data does not borrow
            // another's, and one that borrows publishes none.
            let own_decay = r.parts.contains(&Part::Decay);
            assert_eq!(own_decay, r.decay_library.is_none(), "{}", r.keyword);
            if let Some(lib) = r.decay_library {
                let donor = recipe(lib).unwrap();
                for s in r.sources.iter().filter(|s| {
                    matches!(s.sublibrary, Sublibrary::Decay | Sublibrary::FissionYields)
                }) {
                    assert!(
                        donor.sources.iter().any(|d| d.pin == s.pin),
                        "{} takes {} from somewhere other than {lib}",
                        r.keyword,
                        s.name
                    );
                }
            }
        }
    }

    #[test]
    fn declared_failures_name_a_folder_of_a_part_the_library_converts() {
        for r in RECIPES {
            for key in r.unconvertible {
                let (particle, name) = key.split_once('/').unwrap();
                let part = Part::parse(particle).unwrap();
                assert!(r.parts.contains(&part) && !part.is_chain(), "{key}");
                assert!(!name.is_empty() && !name.contains(['/', '.']), "{key}");
            }
        }
    }

    #[test]
    fn parts_parse_from_their_names() {
        for p in Part::ALL {
            assert_eq!(Part::parse(p.name()).unwrap(), p);
        }
        let err = Part::parse("photons").unwrap_err();
        assert!(err.contains("photon, photonuclear"), "{err}");
    }

    #[test]
    fn jendl_updates_follow_their_base_in_ascending_order() {
        let r = recipe("jendl-5.0").unwrap();
        for sub in r.sublibraries() {
            let names: Vec<&str> = r.sources_of(sub).map(|s| s.name).collect();
            // A base archive, if listed, comes first; updates follow in
            // ascending order.
            let numbers: Vec<u32> = names
                .iter()
                .map(|n| {
                    n.split("_upd")
                        .nth(1)
                        .map_or(0, |n| n.trim_end_matches(".tar.gz").parse().unwrap())
                })
                .collect();
            assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{names:?}");
        }
        let neutron: Vec<&str> = r.sources_of(Sublibrary::Neutron).map(|s| s.name).collect();
        assert_eq!(neutron.len(), 9);
        assert_eq!(*neutron.last().unwrap(), "jendl5-n_upd17.tar.gz");
        let first = |sub| r.sources_of(sub).next().unwrap().name;
        assert_eq!(first(Sublibrary::Decay), "jendl5-dec_upd5.tar.gz");
        assert_eq!(first(Sublibrary::FissionYields), "jendl5-fpy_upd8.tar.gz");
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
        assert_eq!(
            format("jeff-4.0", "JEFF40-Evaluations-Neutron-593.zip"),
            Format::Zip
        );
        assert_eq!(
            format("jeff-4.0", "Radioactive_Decay_Data_JEFF-40.txt"),
            Format::Plain
        );
        assert_eq!(format("tendl-2025", "TENDL-n.tgz"), Format::TarGz);
        assert_eq!(
            format("endf-b8.1", "decay-version.VIII.1.tar.gz"),
            Format::TarGz
        );
    }
}
