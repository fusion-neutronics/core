//! MF=7, thermal neutron scattering law data.

use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;

/// Coherent elastic scattering: the structure factor against energy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoherentElastic {
    pub t: f64,
    /// Number of temperatures beyond the first.
    pub lt: i64,
    pub s: Tabulated1D,
    /// The structure factor at each subsequent temperature. These are LIST
    /// records sharing the first temperature's energy grid.
    pub others: Vec<CoherentElasticTemperature>,
}

/// The structure factor at one of the subsequent temperatures.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoherentElasticTemperature {
    pub t: f64,
    /// Interpolation between this temperature and the previous one.
    pub li: i64,
    pub s: Vec<f64>,
}

/// Incoherent elastic scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IncoherentElastic {
    /// Bound cross section.
    pub sb: f64,
    /// Debye-Waller integral against temperature.
    pub w: Tabulated1D,
}

/// MF=7 MT=2: elastic thermal scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf7Mt2 {
    pub za: i64,
    pub awr: f64,
    /// 1 coherent, 2 incoherent, 3 both.
    pub lthr: i64,
    pub coherent: Option<CoherentElastic>,
    pub incoherent: Option<IncoherentElastic>,
}

/// S(alpha, beta) at one beta, over temperature.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScatteringLaw {
    pub t: f64,
    pub beta: f64,
    pub lt: i64,
    pub s: Tabulated1D,
    pub others: Vec<ScatteringLawTemperature>,
}

/// S(alpha, beta) at one further temperature.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScatteringLawTemperature {
    pub t: f64,
    pub beta: f64,
    /// The Python reader stores the outer LT here rather than the LI it reads
    /// from the record, so the interpolation flag is lost. Reproduced so the
    /// two agree; see
    /// <https://github.com/shimwell/endf-python/issues/16>.
    pub lt: i64,
    pub s: Vec<f64>,
}

/// MF=7 MT=4: incoherent inelastic thermal scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf7Mt4 {
    pub za: i64,
    pub awr: f64,
    /// 1 when alpha and beta are given in units of kT at 0.0253 eV.
    pub lat: i64,
    /// 1 when S is asymmetric in beta.
    pub lasym: i64,
    /// 1 when S is stored as its natural logarithm.
    pub lln: i64,
    pub ni: i64,
    pub ns: i64,
    /// The B parameters describing the principal and secondary atoms.
    pub b: Vec<f64>,
    pub beta_int: Option<Tabulated2D>,
    pub nb: i64,
    pub beta_data: Vec<ScatteringLaw>,
    /// Effective temperature for the principal atom and any secondary atom
    /// whose analytic-model flag is zero.
    pub teff: Vec<Tabulated1D>,
}

/// One element of a thermal scattering material.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ThermalElement {
    pub nas: i64,
    pub ni: i64,
    pub zai: Vec<f64>,
    pub lisi: Vec<f64>,
    pub afi: Vec<f64>,
    pub awri: Vec<f64>,
    pub sfi: Vec<f64>,
}

/// MF=7 MT=451: the generalised information file for thermal scattering.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf7Mt451 {
    pub za: i64,
    pub awr: f64,
    pub na: i64,
    pub elements: Vec<ThermalElement>,
}

fn column(values: &[f64], offset: usize, stride: usize) -> Vec<f64> {
    values
        .iter()
        .skip(offset)
        .step_by(stride)
        .copied()
        .collect()
}

fn parse_coherent(reader: &mut Reader) -> Result<CoherentElastic> {
    let tab = reader.tab1_record()?;
    let lt = tab.l1;
    let mut data = CoherentElastic {
        t: tab.c1,
        lt,
        s: tab.table,
        others: Vec::new(),
    };
    for _ in 0..lt.max(0) {
        let list = reader.list_record()?;
        data.others.push(CoherentElasticTemperature {
            t: list.cont.c1,
            li: list.cont.l1,
            s: list.values,
        });
    }
    Ok(data)
}

fn parse_incoherent(reader: &mut Reader) -> Result<IncoherentElastic> {
    let tab = reader.tab1_record()?;
    Ok(IncoherentElastic {
        sb: tab.c1,
        w: tab.table,
    })
}

/// Parse MF=7 MT=2.
pub fn parse_mf7_mt2(reader: &mut Reader) -> Result<Mf7Mt2> {
    let head = reader.head_record()?;
    let lthr = head.l1;
    let mut data = Mf7Mt2 {
        za: head.za,
        awr: head.awr,
        lthr,
        ..Default::default()
    };
    match lthr {
        1 => data.coherent = Some(parse_coherent(reader)?),
        2 => data.incoherent = Some(parse_incoherent(reader)?),
        3 => {
            data.coherent = Some(parse_coherent(reader)?);
            data.incoherent = Some(parse_incoherent(reader)?);
        }
        _ => {}
    }
    Ok(data)
}

/// Parse MF=7 MT=4.
pub fn parse_mf7_mt4(reader: &mut Reader) -> Result<Mf7Mt4> {
    let head = reader.head_record()?;
    let mut data = Mf7Mt4 {
        za: head.za,
        awr: head.awr,
        lat: head.l2,
        lasym: head.n1,
        ..Default::default()
    };

    let list = reader.list_record()?;
    data.lln = list.cont.l1;
    data.ni = list.cont.n1;
    data.ns = list.cont.n2;
    data.b = list.values;

    // B(1) is zero when the principal atom uses an analytic law, in which case
    // no tabulated S(alpha, beta) follows.
    if data.b.first().copied().unwrap_or(0.0) > 0.0 {
        let tab2 = reader.tab2_record()?;
        data.nb = tab2.cont.n2;
        data.beta_int = Some(tab2.table);
        for _ in 0..data.nb.max(0) {
            let tab = reader.tab1_record()?;
            let lt = tab.l1;
            let mut law = ScatteringLaw {
                t: tab.c1,
                beta: tab.c2,
                lt,
                s: tab.table,
                others: Vec::new(),
            };
            for _ in 0..lt.max(0) {
                let list = reader.list_record()?;
                law.others.push(ScatteringLawTemperature {
                    t: list.cont.c1,
                    beta: list.cont.c2,
                    // The record's own LI is read and discarded upstream; the
                    // outer LT is stored instead. See issue #16.
                    lt,
                    s: list.values,
                });
            }
            data.beta_data.push(law);
        }
    }

    // The principal atom always has an effective temperature; a secondary atom
    // has one when its analytic-model flag B(6*i+1) is zero.
    data.teff.push(reader.tab1_record()?.table);
    for i in 0..data.ns.max(0) as usize {
        let idx = 6 * (i + 1);
        let flag = data.b.get(idx).ok_or(Error::Unsupported {
            what: "an MF=7 MT=4 section whose B list is shorter than NS implies",
        })?;
        if *flag == 0.0 {
            data.teff.push(reader.tab1_record()?.table);
        }
    }

    Ok(data)
}

/// Parse MF=7 MT=451.
pub fn parse_mf7_mt451(reader: &mut Reader) -> Result<Mf7Mt451> {
    let head = reader.head_record()?;
    let mut data = Mf7Mt451 {
        za: head.za,
        awr: head.awr,
        na: head.l1,
        elements: Vec::new(),
    };
    for _ in 0..data.na.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        data.elements.push(ThermalElement {
            nas: list.cont.l1,
            ni: list.cont.n2,
            zai: column(v, 0, 6),
            lisi: column(v, 1, 6),
            afi: column(v, 2, 6),
            awri: column(v, 3, 6),
            sfi: column(v, 4, 6),
        });
    }
    Ok(data)
}
