//! MF=12 to MF=15, photon production.
//!
//! Grouped into one module because the four files describe one thing between
//! them: how many photons a reaction makes (MF=12), the cross section for
//! making them (MF=13), where they go (MF=14) and with what energy (MF=15).

use crate::error::Result;
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;

// -------------------------------------------------------------------------
// MF=12
// -------------------------------------------------------------------------

/// One discrete photon's multiplicity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Multiplicity {
    /// Photon energy, or the binding energy when LP=2.
    pub eg: f64,
    /// Energy of the level the photon originates from.
    pub es: f64,
    /// Primary photon flag.
    pub lp: i64,
    /// 1 when the photon energy is discrete, 2 when it is a distribution.
    pub lf: i64,
    pub y: Tabulated1D,
}

/// One entry in a transition probability array.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transition {
    /// Energy of the level transitioned to.
    pub es: f64,
    /// Probability of the transition.
    pub tp: f64,
    /// Conditional photon emission probability. Only present for LG=2.
    pub gp: Option<f64>,
}

/// MF=12: photon production multiplicities or transition probabilities.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf12 {
    pub za: i64,
    pub awr: f64,
    /// 1 multiplicities, 2 transition probabilities.
    pub lo: i64,
    pub nk: i64,
    /// LO=1 with more than one photon: the total yield.
    pub total_yield: Option<Tabulated1D>,
    /// LO=1.
    pub multiplicities: Vec<Multiplicity>,
    /// LO=2: 1 simple, 2 complex.
    pub lg: Option<i64>,
    /// LO=2: energy of the highest level.
    pub es_ns: f64,
    pub lp: i64,
    pub nt: i64,
    pub transitions: Vec<Transition>,
    /// True when LO was neither 1 nor 2, which the Python reader warns about
    /// and otherwise ignores.
    pub unrecognised_lo: bool,
}

/// Parse an MF=12 section.
pub fn parse_mf12(reader: &mut Reader) -> Result<Mf12> {
    let head = reader.head_record()?;
    let (lo, lg, nk) = (head.l1, head.l2, head.n1);
    let mut data = Mf12 {
        za: head.za,
        awr: head.awr,
        lo,
        nk,
        ..Default::default()
    };

    if lo == 1 {
        // The total yield is only written when there is more than one photon
        // to total.
        if nk > 1 {
            data.total_yield = Some(reader.tab1_record()?.table);
        }
        for _ in 0..nk.max(0) {
            let tab = reader.tab1_record()?;
            data.multiplicities.push(Multiplicity {
                eg: tab.c1,
                es: tab.c2,
                lp: tab.l1,
                lf: tab.l2,
                y: tab.table,
            });
        }
    } else if lo == 2 {
        data.lg = Some(lg);
        let list = reader.list_record()?;
        data.es_ns = list.cont.c1;
        data.lp = list.cont.l1;
        data.nt = list.cont.n2;
        let v = &list.values;
        let stride = if lg == 2 { 3 } else { 2 };
        for i in 0..data.nt.max(0) as usize {
            data.transitions.push(Transition {
                es: v.get(stride * i).copied().unwrap_or(0.0),
                tp: v.get(stride * i + 1).copied().unwrap_or(0.0),
                gp: (lg == 2).then(|| v.get(stride * i + 2).copied().unwrap_or(0.0)),
            });
        }
    } else {
        data.unrecognised_lo = true;
    }

    Ok(data)
}

// -------------------------------------------------------------------------
// MF=13
// -------------------------------------------------------------------------

/// One discrete photon's production cross section.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhotonCrossSection {
    pub eg: f64,
    pub es: f64,
    pub lp: i64,
    pub lf: i64,
    pub sigma: Tabulated1D,
}

/// MF=13: photon production cross sections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf13 {
    pub za: i64,
    pub awr: f64,
    pub nk: i64,
    /// Written only when there is more than one photon.
    pub sigma_total: Option<Tabulated1D>,
    pub photons: Vec<PhotonCrossSection>,
}

/// Parse an MF=13 section.
pub fn parse_mf13(reader: &mut Reader) -> Result<Mf13> {
    let head = reader.head_record()?;
    let nk = head.n1;
    let mut data = Mf13 {
        za: head.za,
        awr: head.awr,
        nk,
        ..Default::default()
    };
    if nk > 1 {
        data.sigma_total = Some(reader.tab1_record()?.table);
    }
    for _ in 0..nk.max(0) {
        let tab = reader.tab1_record()?;
        data.photons.push(PhotonCrossSection {
            eg: tab.c1,
            es: tab.c2,
            lp: tab.l1,
            lf: tab.l2,
            sigma: tab.table,
        });
    }
    Ok(data)
}

// -------------------------------------------------------------------------
// MF=14
// -------------------------------------------------------------------------

/// The angular distribution of one photon.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhotonAngle {
    pub eg: f64,
    pub es: f64,
    /// Set for the anisotropic subsections only; an isotropic one carries
    /// nothing beyond its two energies.
    pub isotropic: bool,
    pub ne: i64,
    pub e_int: Option<Tabulated2D>,
    /// Incident energies.
    pub energy: Vec<f64>,
    /// LTT=1: the number of Legendre coefficients at each incident energy.
    /// Held as floats because the Python reader stores them in a float array.
    pub nl: Vec<f64>,
    /// LTT=1: the Legendre coefficients at each incident energy.
    pub a_lk: Vec<Vec<f64>>,
    /// LTT=2: the tabulated distribution at each incident energy.
    pub p_k: Vec<Tabulated1D>,
}

/// MF=14: photon angular distributions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf14 {
    pub za: i64,
    pub awr: f64,
    /// 1 when every photon is isotropic, in which case nothing follows.
    pub li: i64,
    pub nk: i64,
    pub ltt: Option<i64>,
    /// Number of isotropic photons, which come first.
    pub ni: Option<i64>,
    pub subsections: Vec<PhotonAngle>,
}

/// Parse an MF=14 section.
pub fn parse_mf14(reader: &mut Reader) -> Result<Mf14> {
    let head = reader.head_record()?;
    let (li, ltt, nk, ni) = (head.l1, head.l2, head.n1, head.n2);
    let mut data = Mf14 {
        za: head.za,
        awr: head.awr,
        li,
        nk,
        ..Default::default()
    };

    // Every photon isotropic: the section ends here.
    if li == 1 {
        return Ok(data);
    }
    data.ltt = Some(ltt);
    data.ni = Some(ni);

    for _ in 0..ni.max(0) {
        let c = reader.cont_record()?;
        data.subsections.push(PhotonAngle {
            eg: c.c1,
            es: c.c2,
            isotropic: true,
            ..Default::default()
        });
    }

    for _ in ni.max(0)..nk.max(0) {
        let tab2 = reader.tab2_record()?;
        let ne = tab2.cont.n2;
        let mut sub = PhotonAngle {
            eg: tab2.cont.c1,
            es: tab2.cont.c2,
            isotropic: false,
            ne,
            e_int: Some(tab2.table),
            ..Default::default()
        };
        if ltt == 1 {
            for _ in 0..ne.max(0) {
                let list = reader.list_record()?;
                sub.energy.push(list.cont.c2);
                sub.nl.push(list.cont.n1 as f64);
                sub.a_lk.push(list.values);
            }
        } else if ltt == 2 {
            for _ in 0..ne.max(0) {
                let tab = reader.tab1_record()?;
                sub.energy.push(tab.c2);
                sub.p_k.push(tab.table);
            }
        }
        data.subsections.push(sub);
    }

    Ok(data)
}

// -------------------------------------------------------------------------
// MF=15
// -------------------------------------------------------------------------

/// One partial photon energy distribution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhotonSpectrum {
    /// Only LF=1 is defined for MF=15.
    pub lf: i64,
    /// Fraction of the emission this subsection describes.
    pub p: Tabulated1D,
    pub e_int: Tabulated2D,
    pub ne: i64,
    /// Incident energies.
    pub energy: Vec<f64>,
    /// Normalised outgoing spectrum at each incident energy.
    pub g: Vec<Tabulated1D>,
}

/// MF=15: continuous photon energy spectra.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf15 {
    pub za: i64,
    pub awr: f64,
    pub nc: i64,
    pub subsections: Vec<PhotonSpectrum>,
}

/// Parse an MF=15 section.
pub fn parse_mf15(reader: &mut Reader) -> Result<Mf15> {
    let head = reader.head_record()?;
    let nc = head.n1;
    let mut data = Mf15 {
        za: head.za,
        awr: head.awr,
        nc,
        subsections: Vec::new(),
    };

    for _ in 0..nc.max(0) {
        let p = reader.tab1_record()?;
        let tab2 = reader.tab2_record()?;
        let ne = tab2.cont.n2;
        let mut sub = PhotonSpectrum {
            lf: p.l2,
            p: p.table,
            e_int: tab2.table,
            ne,
            ..Default::default()
        };
        for _ in 0..ne.max(0) {
            let tab = reader.tab1_record()?;
            sub.energy.push(tab.c2);
            sub.g.push(tab.table);
        }
        data.subsections.push(sub);
    }

    Ok(data)
}
