//! MF=6, energy-angle distributions of reaction products.
//!
//! These are the four distribution shapes a transport code has to sample, and
//! the ones an Arrow projection of this data carries columns for.

use crate::error::Result;
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;

/// MF=6: the products of one reaction and their distributions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf6 {
    pub za: i64,
    pub awr: f64,
    /// Particle-production flag.
    pub jp: i64,
    /// 1 laboratory frame, 2 centre-of-mass.
    pub lct: i64,
    pub nk: i64,
    pub products: Vec<Product>,
}

/// One product, its yield, and its distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Product {
    /// ZA of the product, 1000*Z + A. 0 is a photon, 1 a neutron.
    pub zap: i64,
    pub awp: f64,
    /// Product modifier flag.
    pub lip: i64,
    /// Which distribution law the product uses.
    pub law: i64,
    /// Multiplicity as a function of incident energy.
    pub yield_: Tabulated1D,
    pub distribution: Distribution,
}

/// The distribution of a product, in whichever law it uses.
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// LAW<0 (given elsewhere), LAW=0 (none), LAW=3 (isotropic discrete) and
    /// LAW=4 (discrete two-body recoil) carry no data of their own.
    None,
    /// LAW=1.
    ContinuumEnergyAngle(ContinuumEnergyAngle),
    /// LAW=2.
    DiscreteTwoBody(DiscreteTwoBody),
    /// LAW=5.
    ChargedParticleElastic(ChargedParticleElastic),
    /// LAW=6.
    NBodyPhaseSpace { apsx: f64, npsx: i64 },
    /// LAW=7.
    LaboratoryAngleEnergy(LaboratoryAngleEnergy),
}

/// LAW=1: continuum energy-angle distribution.
///
/// `lang` selects the angular representation: 1 Legendre, 2 Kalbach-Mann,
/// 11 to 15 tabulated with the given interpolation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContinuumEnergyAngle {
    pub lang: i64,
    /// Interpolation scheme for secondary energy.
    pub lep: i64,
    pub nr: i64,
    pub ne: i64,
    pub e_int: Tabulated2D,
    /// Incident energies.
    pub energy: Vec<f64>,
    pub distribution: Vec<ContinuumSubsection>,
}

/// The outgoing distribution at one incident energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContinuumSubsection {
    /// Number of discrete outgoing energies leading the list.
    pub nd: i64,
    /// Number of angular parameters per outgoing energy.
    pub na: i64,
    pub nw: i64,
    /// Number of outgoing energies.
    pub nep: i64,
    /// Outgoing energies.
    pub e_out: Vec<f64>,
    /// The angular parameters at each outgoing energy: `nep` rows of `na + 1`.
    /// The first column is the probability; the rest depend on `lang`.
    pub b: Vec<Vec<f64>>,
}

/// LAW=2: discrete two-body scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiscreteTwoBody {
    pub nr: i64,
    pub ne: i64,
    pub e_int: Tabulated2D,
    pub energy: Vec<f64>,
    pub distribution: Vec<DiscreteTwoBodySubsection>,
}

/// The angular distribution at one incident energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiscreteTwoBodySubsection {
    /// 0 Legendre coefficients, 12 or 14 tabulated.
    pub lang: i64,
    pub nw: i64,
    pub nl: i64,
    pub a_l: Vec<f64>,
}

/// LAW=5: charged-particle elastic scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChargedParticleElastic {
    pub spi: f64,
    /// Identical-particle flag.
    pub lidp: i64,
    pub ne: i64,
    pub e_int: Tabulated2D,
    pub distribution: Vec<ChargedParticleSubsection>,
}

/// The distribution at one incident energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChargedParticleSubsection {
    pub energy: f64,
    /// 1 nuclear amplitude, 2 nuclear plus interference, 12 to 15 tabulated.
    pub ltp: i64,
    pub nw: i64,
    pub nl: i64,
    pub a: Vec<f64>,
}

/// LAW=7: laboratory energy-angle distribution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LaboratoryAngleEnergy {
    pub ne: i64,
    pub e_int: Tabulated2D,
    pub distribution: Vec<LaboratorySubsection>,
}

/// The distribution at one incident energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LaboratorySubsection {
    pub energy: f64,
    pub nrm: i64,
    pub nmu: i64,
    /// Interpolation across the outgoing cosine.
    pub mu_int: Tabulated2D,
    pub mu: Vec<AngleEntry>,
}

/// The outgoing energy distribution at one cosine.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AngleEntry {
    pub mu: f64,
    pub f: Tabulated1D,
}

/// LAW=1. Public because MF=26 shares this law.
pub fn parse_continuum_energy_angle(reader: &mut Reader) -> Result<ContinuumEnergyAngle> {
    let tab2 = reader.tab2_record()?;
    let mut data = ContinuumEnergyAngle {
        lang: tab2.cont.l1,
        lep: tab2.cont.l2,
        nr: tab2.cont.n1,
        ne: tab2.cont.n2,
        e_int: tab2.table,
        ..Default::default()
    };

    for _ in 0..data.ne.max(0) {
        let list = reader.list_record()?;
        let (nd, na, nw, nep) = (list.cont.l1, list.cont.l2, list.cont.n1, list.cont.n2);
        data.energy.push(list.cont.c2);

        // The values are `nep` rows of `na + 2`: an outgoing energy, then the
        // probability and `na` angular parameters.
        let stride = (na.max(0) as usize) + 2;
        let mut sub = ContinuumSubsection {
            nd,
            na,
            nw,
            nep,
            e_out: Vec::with_capacity(nep.max(0) as usize),
            b: Vec::with_capacity(nep.max(0) as usize),
        };
        for row in list.values.chunks(stride).take(nep.max(0) as usize) {
            sub.e_out.push(row.first().copied().unwrap_or(0.0));
            sub.b.push(row.get(1..).unwrap_or(&[]).to_vec());
        }
        data.distribution.push(sub);
    }

    Ok(data)
}

/// LAW=2. Public because MF=26 shares this law.
pub fn parse_discrete_two_body(reader: &mut Reader) -> Result<DiscreteTwoBody> {
    let tab2 = reader.tab2_record()?;
    let mut data = DiscreteTwoBody {
        nr: tab2.cont.n1,
        ne: tab2.cont.n2,
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..data.ne.max(0) {
        let list = reader.list_record()?;
        data.energy.push(list.cont.c2);
        data.distribution.push(DiscreteTwoBodySubsection {
            lang: list.cont.l1,
            nw: list.cont.n1,
            nl: list.cont.n2,
            a_l: list.values,
        });
    }
    Ok(data)
}

fn parse_charged_particle(reader: &mut Reader) -> Result<ChargedParticleElastic> {
    let tab2 = reader.tab2_record()?;
    let mut data = ChargedParticleElastic {
        spi: tab2.cont.c1,
        lidp: tab2.cont.l1,
        ne: tab2.cont.n2,
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..data.ne.max(0) {
        let list = reader.list_record()?;
        data.distribution.push(ChargedParticleSubsection {
            energy: list.cont.c2,
            ltp: list.cont.l1,
            nw: list.cont.n1,
            nl: list.cont.n2,
            a: list.values,
        });
    }
    Ok(data)
}

fn parse_laboratory(reader: &mut Reader) -> Result<LaboratoryAngleEnergy> {
    let tab2 = reader.tab2_record()?;
    let mut data = LaboratoryAngleEnergy {
        ne: tab2.cont.n2,
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..data.ne.max(0) {
        let tab2 = reader.tab2_record()?;
        let mut sub = LaboratorySubsection {
            energy: tab2.cont.c2,
            nrm: tab2.cont.n1,
            nmu: tab2.cont.n2,
            mu_int: tab2.table,
            mu: Vec::new(),
        };
        for _ in 0..sub.nmu.max(0) {
            let tab = reader.tab1_record()?;
            sub.mu.push(AngleEntry {
                mu: tab.c2,
                f: tab.table,
            });
        }
        data.distribution.push(sub);
    }
    Ok(data)
}

/// Parse an MF=6 section.
pub fn parse_mf6(reader: &mut Reader) -> Result<Mf6> {
    let head = reader.head_record()?;
    let mut data = Mf6 {
        za: head.za,
        awr: head.awr,
        jp: head.l1,
        lct: head.l2,
        nk: head.n1,
        products: Vec::new(),
    };

    for _ in 0..data.nk.max(0) {
        let tab = reader.tab1_record()?;
        let law = tab.l2;
        let distribution = match law {
            1 => Distribution::ContinuumEnergyAngle(parse_continuum_energy_angle(reader)?),
            2 => Distribution::DiscreteTwoBody(parse_discrete_two_body(reader)?),
            5 => Distribution::ChargedParticleElastic(parse_charged_particle(reader)?),
            6 => {
                let c = reader.cont_record()?;
                Distribution::NBodyPhaseSpace {
                    apsx: c.c1,
                    npsx: c.n2,
                }
            }
            7 => Distribution::LaboratoryAngleEnergy(parse_laboratory(reader)?),
            // LAW<0, 0, 3 and 4 carry nothing of their own.
            _ => Distribution::None,
        };

        data.products.push(Product {
            zap: tab.c1 as i64,
            awp: tab.c2,
            lip: tab.l1,
            law,
            yield_: tab.table,
            distribution,
        });
    }

    Ok(data)
}
