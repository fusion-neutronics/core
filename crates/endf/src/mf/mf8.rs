//! MF=8, radioactive decay, fission product yields and nuclide production.
//!
//! These are what a transmutation network is built from: MT=457 gives decay
//! modes and spectra, MT=454/459 the fission yields, and the general MF=8
//! sections the radioactive products of a reaction.

use crate::error::Result;
use crate::function::Tabulated1D;
use crate::records::Reader;

/// A value with its uncertainty, as MF=8 stores most quantities.
pub type WithUncertainty = (f64, f64);

/// MF=8 for a reaction: the radioactive nuclides it produces.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf8 {
    pub za: i64,
    pub awr: f64,
    pub lis: i64,
    pub liso: i64,
    pub ns: i64,
    /// 0 when the decay chain is given here, 1 when it is in MT=457.
    pub no: i64,
    pub subsections: Vec<ProductionSubsection>,
}

/// One radioactive product of the reaction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProductionSubsection {
    pub zap: f64,
    pub elfs: f64,
    /// Which file the production cross section is in.
    pub lmf: i64,
    pub lfs: i64,
    /// Number of decay chain entries. `None` when NO=1 and the chain is
    /// elsewhere.
    pub nd: Option<i64>,
    pub hl: Vec<f64>,
    pub rtyp: Vec<f64>,
    pub zan: Vec<f64>,
    pub br: Vec<f64>,
    pub end: Vec<f64>,
    pub ct: Vec<f64>,
}

/// MF=8 MT=454 or MT=459: fission product yields, independent or cumulative.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf8Mt454 {
    pub za: i64,
    pub awr: f64,
    /// One less than the number of incident energies.
    pub le: i64,
    pub yields: Vec<FissionYields>,
}

/// The yields at one incident energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FissionYields {
    pub energy: f64,
    pub nn: i64,
    pub nfp: i64,
    /// LE+1 for the first energy, the interpolation scheme for the rest. The
    /// format overloads the same field, and so does the Python reader, which
    /// keys it `LE` on the first entry and `I` on the others.
    pub le_or_interpolation: i64,
    pub products: Vec<FissionProduct>,
}

/// One fission product and its yield.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FissionProduct {
    pub zafp: f64,
    /// Isomeric state of the product.
    pub fps: f64,
    pub y: WithUncertainty,
}

/// MF=8 MT=457: radioactive decay data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf8Mt457 {
    pub za: i64,
    pub awr: f64,
    pub lis: i64,
    pub liso: i64,
    /// 1 when the nuclide is stable, in which case only the spin and parity
    /// follow.
    pub nst: i64,
    pub nsp: i64,
    pub spi: f64,
    pub par: f64,
    /// Half-life in seconds. `None` for a stable nuclide.
    pub half_life: Option<WithUncertainty>,
    pub nc: i64,
    /// Average decay energies by radiation type.
    pub ex: Vec<WithUncertainty>,
    pub ndk: i64,
    pub modes: Vec<DecayMode>,
    pub spectra: Vec<Spectrum>,
}

/// One decay mode and its branching ratio.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecayMode {
    /// Decay type: 1 beta-, 2 beta+/EC, 3 IT, 4 alpha, and so on. A value like
    /// 1.5 means a chain of two modes.
    pub rtyp: f64,
    /// Isomeric state of the daughter.
    pub rfs: f64,
    pub q: WithUncertainty,
    pub br: WithUncertainty,
}

/// The spectrum of one radiation type emitted in decay.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Spectrum {
    /// Radiation type: 0 gamma, 1 beta-, 2 beta+, 4 alpha, and so on. Stored
    /// as written, a float, because the format puts it in a C field.
    pub styp: f64,
    /// 0 discrete only, 1 continuous only, 2 both.
    pub lcon: i64,
    pub lcov: i64,
    pub ner: i64,
    /// Discrete normalisation factor.
    pub fd: WithUncertainty,
    /// Average decay energy of this radiation type.
    pub er_av: WithUncertainty,
    /// Continuum normalisation factor.
    pub fc: WithUncertainty,
    pub discrete: Vec<DiscreteRadiation>,
    pub continuous: Option<ContinuousSpectrum>,
    pub continuous_covariance: Option<ContinuousCovariance>,
    pub discrete_covariance: Option<DiscreteCovariance>,
}

/// One discrete line.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiscreteRadiation {
    pub er: WithUncertainty,
    pub rtyp: f64,
    pub type_: f64,
    /// Intensity.
    pub ri: WithUncertainty,
    /// Internal pair formation coefficient, for STYP 0 and 2.
    pub ris: Option<WithUncertainty>,
    /// Total internal conversion coefficient, STYP 0 only.
    pub ricc: Option<WithUncertainty>,
    /// K-shell internal conversion coefficient.
    pub rick: Option<WithUncertainty>,
    /// L-shell internal conversion coefficient.
    pub ricl: Option<WithUncertainty>,
}

/// A continuous spectrum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContinuousSpectrum {
    pub rtyp: f64,
    pub rp: Tabulated1D,
}

/// Covariance of a continuous spectrum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContinuousCovariance {
    pub lb: i64,
    pub ek: Vec<f64>,
    pub fk: Vec<f64>,
}

/// Covariance of the discrete lines.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiscreteCovariance {
    pub ls: i64,
    pub lb: i64,
    pub ne: i64,
    pub nerp: i64,
    pub ek: Vec<f64>,
    /// Packed upper-triangular covariance; the format's packing order is not
    /// unpacked here, matching the Python reader.
    pub fkk: Vec<f64>,
}

fn pair(values: &[f64], i: usize) -> WithUncertainty {
    (
        values.get(i).copied().unwrap_or(0.0),
        values.get(i + 1).copied().unwrap_or(0.0),
    )
}

/// The same, but `None` when the record is too short to hold the pair.
///
/// MF=8 MT=457 discrete records are written with NT = 6 or NT = 12 depending
/// on whether the internal conversion coefficients were evaluated, and the two
/// lengths appear within a single spectrum. The Python reader slices, so a
/// missing coefficient comes back as an empty tuple rather than as a zero;
/// this keeps that distinction.
fn opt_pair(values: &[f64], i: usize) -> Option<WithUncertainty> {
    (values.len() >= i + 2).then(|| pair(values, i))
}

fn column(values: &[f64], offset: usize, stride: usize) -> Vec<f64> {
    values
        .iter()
        .skip(offset)
        .step_by(stride)
        .copied()
        .collect()
}

/// Parse a general MF=8 section.
pub fn parse_mf8(reader: &mut Reader) -> Result<Mf8> {
    let head = reader.head_record()?;
    let no = head.n2;
    let mut data = Mf8 {
        za: head.za,
        awr: head.awr,
        lis: head.l1,
        liso: head.l2,
        ns: head.n1,
        no,
        subsections: Vec::new(),
    };

    for _ in 0..data.ns.max(0) {
        let sub = if no == 0 {
            let list = reader.list_record()?;
            let v = &list.values;
            ProductionSubsection {
                zap: list.cont.c1,
                elfs: list.cont.c2,
                lmf: list.cont.l1,
                lfs: list.cont.l2,
                nd: Some((v.len() / 6) as i64),
                hl: column(v, 0, 6),
                rtyp: column(v, 1, 6),
                zan: column(v, 2, 6),
                br: column(v, 3, 6),
                end: column(v, 4, 6),
                ct: column(v, 5, 6),
            }
        } else {
            let c = reader.cont_record()?;
            ProductionSubsection {
                zap: c.c1,
                elfs: c.c2,
                lmf: c.l1,
                lfs: c.l2,
                ..Default::default()
            }
        };
        data.subsections.push(sub);
    }

    Ok(data)
}

/// Parse MF=8 MT=454 or MT=459.
pub fn parse_mf8_mt454(reader: &mut Reader) -> Result<Mf8Mt454> {
    let head = reader.head_record()?;
    let le = head.l1 - 1;
    let mut data = Mf8Mt454 {
        za: head.za,
        awr: head.awr,
        le,
        yields: Vec::new(),
    };

    for _ in 0..(le + 1).max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        let nfp = list.cont.n2;
        let mut set = FissionYields {
            energy: list.cont.c1,
            nn: list.cont.n1,
            nfp,
            le_or_interpolation: list.cont.l1,
            products: Vec::with_capacity(nfp.max(0) as usize),
        };
        for j in 0..nfp.max(0) as usize {
            set.products.push(FissionProduct {
                zafp: v.get(4 * j).copied().unwrap_or(0.0),
                fps: v.get(4 * j + 1).copied().unwrap_or(0.0),
                y: pair(v, 4 * j + 2),
            });
        }
        data.yields.push(set);
    }

    Ok(data)
}

/// Parse MF=8 MT=457.
pub fn parse_mf8_mt457(reader: &mut Reader) -> Result<Mf8Mt457> {
    let head = reader.head_record()?;
    let mut data = Mf8Mt457 {
        za: head.za,
        awr: head.awr,
        lis: head.l1,
        liso: head.l2,
        nst: head.n1,
        nsp: head.n2,
        ..Default::default()
    };

    // A stable nuclide carries only its spin and parity.
    if data.nst == 1 {
        reader.list_record()?;
        let list = reader.list_record()?;
        data.spi = list.cont.c1;
        data.par = list.cont.c2;
        return Ok(data);
    }

    // Half-life and the average decay energies.
    let list = reader.list_record()?;
    data.half_life = Some((list.cont.c1, list.cont.c2));
    data.nc = list.cont.n1 / 2;
    data.ex = list
        .values
        .chunks(2)
        .map(|c| (c[0], c.get(1).copied().unwrap_or(0.0)))
        .collect();

    // Spin, parity and the decay modes.
    let list = reader.list_record()?;
    data.spi = list.cont.c1;
    data.par = list.cont.c2;
    data.ndk = list.cont.n2;
    let v = &list.values;
    for i in 0..data.ndk.max(0) as usize {
        data.modes.push(DecayMode {
            rtyp: v.get(6 * i).copied().unwrap_or(0.0),
            rfs: v.get(6 * i + 1).copied().unwrap_or(0.0),
            q: pair(v, 6 * i + 2),
            br: pair(v, 6 * i + 4),
        });
    }

    for _ in 0..data.nsp.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        let (styp, lcon, lcov, ner) = (list.cont.c2, list.cont.l1, list.cont.l2, list.cont.n2);
        let mut spectrum = Spectrum {
            styp,
            lcon,
            lcov,
            ner,
            fd: pair(v, 0),
            er_av: pair(v, 2),
            fc: pair(v, 4),
            ..Default::default()
        };

        if lcon != 1 {
            for _ in 0..ner.max(0) {
                let list = reader.list_record()?;
                let v = &list.values;
                spectrum.discrete.push(DiscreteRadiation {
                    er: (list.cont.c1, list.cont.c2),
                    rtyp: v.first().copied().unwrap_or(0.0),
                    type_: v.get(1).copied().unwrap_or(0.0),
                    ri: pair(v, 2),
                    ris: (styp == 0.0 || styp == 2.0)
                        .then(|| opt_pair(v, 4))
                        .flatten(),
                    ricc: (styp == 0.0).then(|| opt_pair(v, 6)).flatten(),
                    rick: (styp == 0.0).then(|| opt_pair(v, 8)).flatten(),
                    ricl: (styp == 0.0).then(|| opt_pair(v, 10)).flatten(),
                });
            }
        }

        if lcon != 0 {
            let tab = reader.tab1_record()?;
            spectrum.continuous = Some(ContinuousSpectrum {
                rtyp: tab.c1,
                rp: tab.table,
            });
        }

        if !matches!(lcov, 0 | 2) && lcon != 0 {
            let list = reader.list_record()?;
            spectrum.continuous_covariance = Some(ContinuousCovariance {
                lb: list.cont.l2,
                ek: column(&list.values, 0, 2),
                fk: column(&list.values, 1, 2),
            });
        }

        if !matches!(lcov, 0 | 1) {
            let list = reader.list_record()?;
            let nerp = list.cont.n2.max(0) as usize;
            spectrum.discrete_covariance = Some(DiscreteCovariance {
                ls: list.cont.l1,
                lb: list.cont.l2,
                ne: list.cont.n1,
                nerp: list.cont.n2,
                ek: list.values.iter().take(nerp).copied().collect(),
                fkk: list.values.iter().skip(nerp).copied().collect(),
            });
        }

        data.spectra.push(spectrum);
    }

    Ok(data)
}

/// MF=9 (multiplicities) and MF=10 (production cross sections) for the
/// isomeric states a reaction produces.
///
/// These are where isomeric branching comes from.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf9Mf10 {
    pub za: i64,
    pub awr: f64,
    pub lis: i64,
    pub ns: i64,
    /// 9 or 10, which says whether `func` is a multiplicity or a cross section.
    pub mf: i64,
    pub levels: Vec<IsomerLevel>,
}

/// One isomeric state of the product.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IsomerLevel {
    pub qm: f64,
    pub qi: f64,
    /// ZA of the product.
    pub izap: i64,
    /// Isomeric state number.
    pub lfs: i64,
    /// The multiplicity for MF=9, the cross section for MF=10.
    pub func: Tabulated1D,
}

/// Parse an MF=9 or MF=10 section.
pub fn parse_mf9_mf10(reader: &mut Reader, mf: i64) -> Result<Mf9Mf10> {
    let head = reader.head_record()?;
    let mut data = Mf9Mf10 {
        za: head.za,
        awr: head.awr,
        lis: head.l1,
        ns: head.n1,
        mf,
        levels: Vec::new(),
    };
    for _ in 0..data.ns.max(0) {
        let tab = reader.tab1_record()?;
        data.levels.push(IsomerLevel {
            qm: tab.c1,
            qi: tab.c2,
            izap: tab.l1,
            lfs: tab.l2,
            func: tab.table,
        });
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const IN115: &[u8] = include_bytes!("../../fixtures/n-049_In-115_trimmed.endf.xz");

    #[test]
    fn reads_isomer_production() {
        let m = Material::from_str(&crate::testdata::text(IN115)).unwrap();
        // In-115 (n,gamma) populating the In-116 first isomeric state. This is
        // where an isomeric branching ratio comes from.
        let mf9 = m.mf9(102).expect("MF=9 MT=102 is present");
        assert_eq!(mf9.mf, 9);
        assert_eq!(mf9.levels.len(), mf9.ns as usize);

        let level = &mf9.levels[0];
        assert_eq!(level.izap, 49116, "the product is In-116");
        assert_eq!(level.lfs, 1, "the first isomeric state");
        // MF=9 gives a multiplicity, so it is bounded by one.
        assert!(!level.func.x.is_empty());
        assert!(level.func.y.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn reads_isomer_production_cross_sections() {
        let m = Material::from_str(&crate::testdata::text(IN115)).unwrap();
        // MF=10 carries a cross section for the same kind of product.
        let mf10 = m.mf10(16).expect("MF=10 MT=16 is present");
        assert_eq!(mf10.mf, 10);
        let level = &mf10.levels[0];
        assert_eq!(level.izap, 49114, "(n,2n) leaves In-114");
        assert!(level.func.y.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn reads_radioactive_production() {
        let m = Material::from_str(&crate::testdata::text(IN115)).unwrap();
        let mf8 = m.mf8(102).expect("MF=8 MT=102 is present");
        assert_eq!(mf8.subsections.len(), mf8.ns as usize);
        assert!(!mf8.subsections.is_empty());
        // NO=1 means the decay chain lives in MT=457 rather than here.
        assert_eq!(mf8.no, 1);
        assert_eq!(mf8.subsections[0].nd, None);
    }
}
