//! MF=33, 34 and 40: covariances.
//!
//! Grouped into one module because MF=40 reuses MF=33's subsection format
//! verbatim, and MF=34 is the same idea applied to angular distributions.

use crate::error::{Error, Result};
use crate::records::Reader;

// -------------------------------------------------------------------------
// MF=33
// -------------------------------------------------------------------------

/// An NC-type sub-subsection: a covariance derived from other reactions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NcSubsection {
    pub lty: i64,
    pub e1: f64,
    pub e2: f64,
    /// LTY=0.
    pub nci: i64,
    pub ci: Vec<f64>,
    pub xmti: Vec<f64>,
    /// LTY/=0.
    pub mats: i64,
    pub mts: i64,
    pub nei: i64,
    pub xmfs: f64,
    pub xlfss: f64,
    pub ei: Vec<f64>,
    pub wei: Vec<f64>,
}

/// An NI-type sub-subsection: a covariance given explicitly.
///
/// `lb` selects the layout, and which fields are populated follows from it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NiSubsection {
    pub lt: i64,
    pub ls: i64,
    pub lb: i64,
    pub nt: i64,
    pub np: i64,
    pub ne: i64,
    pub ner: i64,
    pub nec: i64,
    /// LB 0 to 4, and LB 8 or 9.
    pub ek: Vec<f64>,
    pub fk: Vec<f64>,
    /// LB 0 to 4 only: the second (E, F) table.
    pub el: Vec<f64>,
    pub fl: Vec<f64>,
    /// LB=5: the covariance matrix, in the format's packed order.
    pub fkk: Vec<f64>,
    /// LB=6.
    pub er: Vec<f64>,
    pub ec: Vec<f64>,
    pub fkl: Vec<f64>,
}

/// One MF=33 subsection: the covariance of this reaction with another.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf33Subsection {
    pub xmf1: f64,
    pub xlfs1: f64,
    pub mat1: i64,
    pub mt1: i64,
    pub nc: i64,
    pub ni: i64,
    pub nc_subsections: Vec<NcSubsection>,
    pub ni_subsections: Vec<NiSubsection>,
}

/// MF=33: covariances of neutron cross sections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf33 {
    pub za: i64,
    pub awr: f64,
    /// MT of the reaction this one is lumped into, if any.
    pub mtl: i64,
    pub nl: i64,
    pub subsections: Vec<Mf33Subsection>,
}

fn column(values: &[f64], offset: usize, stride: usize) -> Vec<f64> {
    values
        .iter()
        .skip(offset)
        .step_by(stride)
        .copied()
        .collect()
}

/// Parse one MF=33 subsection. Shared with MF=40, which uses the same format.
pub fn parse_mf33_subsection(reader: &mut Reader) -> Result<Mf33Subsection> {
    let c = reader.cont_record()?;
    let mut sub = Mf33Subsection {
        xmf1: c.c1,
        xlfs1: c.c2,
        mat1: c.l1,
        mt1: c.l2,
        nc: c.n1,
        ni: c.n2,
        ..Default::default()
    };

    for _ in 0..sub.nc.max(0) {
        let lty = reader.cont_record()?.l2;
        let list = reader.list_record()?;
        let v = &list.values;
        let subsub = if lty == 0 {
            NcSubsection {
                lty,
                e1: list.cont.c1,
                e2: list.cont.c2,
                nci: list.cont.n2,
                ci: column(v, 0, 2),
                xmti: column(v, 1, 2),
                ..Default::default()
            }
        } else {
            NcSubsection {
                lty,
                e1: list.cont.c1,
                e2: list.cont.c2,
                mats: list.cont.l1,
                mts: list.cont.l2,
                nei: list.cont.n2,
                xmfs: v.first().copied().unwrap_or(0.0),
                xlfss: v.get(1).copied().unwrap_or(0.0),
                ei: column(&v[2.min(v.len())..], 0, 2),
                wei: column(&v[2.min(v.len())..], 1, 2),
                ..Default::default()
            }
        };
        // One entry per subsection, whatever LTY is. Both readers used to
        // append the LTY=0 case twice; see
        // <https://github.com/shimwell/endf-python/issues/12>.
        sub.nc_subsections.push(subsub);
    }

    for _ in 0..sub.ni.max(0) {
        // The layout depends on LB, which is in the record about to be read,
        // so peek at it first.
        let lb = reader.peek_cont_record()?.l2;
        let list = reader.list_record()?;
        let v = &list.values;
        let mut subsub = NiSubsection {
            lb,
            nt: list.cont.n1,
            ..Default::default()
        };
        match lb {
            0..=4 => {
                subsub.lt = list.cont.l1;
                subsub.np = list.cont.n2;
                let split = (subsub.nt - subsub.np).clamp(0, v.len() as i64) as usize;
                let (k, l) = v.split_at(split);
                subsub.ek = column(k, 0, 2);
                subsub.fk = column(k, 1, 2);
                subsub.el = column(l, 0, 2);
                subsub.fl = column(l, 1, 2);
            }
            5 => {
                subsub.ls = list.cont.l1;
                subsub.ne = list.cont.n2;
                let ne = subsub.ne.clamp(0, v.len() as i64) as usize;
                subsub.ek = v[..ne].to_vec();
                // Left in the format's packed order, as upstream does.
                subsub.fkk = v[ne..].to_vec();
            }
            6 => {
                subsub.ner = list.cont.n2;
                let ner = subsub.ner.max(0) as usize;
                subsub.nec = if ner > 0 {
                    (subsub.nt - 1) / subsub.ner
                } else {
                    0
                };
                let nec = subsub.nec.max(0) as usize;
                subsub.er = v[..ner.min(v.len())].to_vec();
                subsub.ec = v[ner.min(v.len())..(ner + nec).min(v.len())].to_vec();
                subsub.fkl = v[(ner + nec).min(v.len())..].to_vec();
            }
            8 | 9 => {
                subsub.lt = list.cont.l1;
                subsub.np = list.cont.n2;
                subsub.ek = column(v, 0, 2);
                subsub.fk = column(v, 1, 2);
            }
            _ => {
                return Err(Error::Unsupported {
                    what: "an unrecognised MF=33 LB value",
                })
            }
        }
        sub.ni_subsections.push(subsub);
    }

    Ok(sub)
}

/// Parse an MF=33 section.
pub fn parse_mf33(reader: &mut Reader) -> Result<Mf33> {
    let head = reader.head_record()?;
    let mut data = Mf33 {
        za: head.za,
        awr: head.awr,
        mtl: head.l2,
        nl: head.n2,
        subsections: Vec::new(),
    };
    for _ in 0..data.nl.max(0) {
        data.subsections.push(parse_mf33_subsection(reader)?);
    }
    Ok(data)
}

// -------------------------------------------------------------------------
// MF=34
// -------------------------------------------------------------------------

/// The covariance blocks of one (L, L1) pair.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf34SubSubsection {
    /// The symmetry flag of each covariance block.
    pub ls: Vec<f64>,
    /// The covariance matrix type of each block. Both readers used to fill
    /// this with LS; see
    /// <https://github.com/shimwell/endf-python/issues/18>.
    pub lb: Vec<f64>,
    pub nt: Vec<f64>,
    pub ne: Vec<f64>,
    pub data: Vec<Vec<f64>>,
}

/// One MF=34 subsection.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf34Subsection {
    pub mat1: i64,
    pub mt1: i64,
    pub nl: i64,
    pub nss: i64,
    pub lct: i64,
    /// Legendre order of each sub-subsection. Floats, as upstream stores them.
    pub l: Vec<f64>,
    pub l1: Vec<f64>,
    pub ni: Vec<f64>,
    pub subsubsections: Vec<Mf34SubSubsection>,
}

/// MF=34: covariances of angular distributions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf34 {
    pub za: i64,
    pub awr: f64,
    pub ltt: i64,
    pub nmt1: i64,
    /// One per (MAT1, MT1) pair the section covers.
    ///
    /// Both readers used to build these and then drop them on the floor, so
    /// this was always empty; see
    /// <https://github.com/shimwell/endf-python/issues/18>.
    pub subsections: Vec<Mf34Subsection>,
}

/// Parse an MF=34 section. `mt` is the reaction the section belongs to, which
/// the subsection count depends on.
pub fn parse_mf34(reader: &mut Reader, mt: i64) -> Result<Mf34> {
    let head = reader.head_record()?;
    let mut data = Mf34 {
        za: head.za,
        awr: head.awr,
        ltt: head.l2,
        nmt1: head.n2,
        subsections: Vec::new(),
    };

    for _ in 0..data.nmt1.max(0) {
        let c = reader.cont_record()?;
        let (mat1, mt1, nl, nl1) = (c.l1, c.l2, c.n1, c.n2);
        // A reaction's covariance with itself is symmetric, so only the upper
        // triangle is stored.
        let nss = if mt1 == 0 || mt == mt1 {
            nl * (nl + 1) / 2
        } else {
            nl * nl1
        };

        let mut sub = Mf34Subsection {
            mat1,
            mt1,
            nl,
            nss,
            ..Default::default()
        };

        for n in 0..nss.max(0) {
            let c = reader.cont_record()?;
            let ni = c.n2;
            sub.l.push(c.l1 as f64);
            sub.l1.push(c.l2 as f64);
            sub.ni.push(ni as f64);
            if n == 0 {
                sub.lct = c.n1;
            }

            let mut subsub = Mf34SubSubsection::default();
            for _ in 0..ni.max(0) {
                let list = reader.list_record()?;
                subsub.ls.push(list.cont.l1 as f64);
                subsub.lb.push(list.cont.l2 as f64);
                subsub.nt.push(list.cont.n1 as f64);
                subsub.ne.push(list.cont.n2 as f64);
                subsub.data.push(list.values);
            }
            sub.subsubsections.push(subsub);
        }

        data.subsections.push(sub);
    }

    Ok(data)
}

// -------------------------------------------------------------------------
// MF=40
// -------------------------------------------------------------------------

/// One MF=40 subsection: the covariance for one reaction product.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf40Subsection {
    pub qm: f64,
    pub qi: f64,
    pub izap: i64,
    pub lfs: i64,
    pub nl: i64,
    /// Each has the same format as an MF=33 subsection.
    pub subsubsections: Vec<Mf33Subsection>,
}

/// MF=40: covariances of radionuclide production cross sections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf40 {
    pub za: i64,
    pub awr: f64,
    pub lis: i64,
    pub ns: i64,
    pub subsections: Vec<Mf40Subsection>,
}

/// Parse an MF=40 section.
pub fn parse_mf40(reader: &mut Reader) -> Result<Mf40> {
    let head = reader.head_record()?;
    let mut data = Mf40 {
        za: head.za,
        awr: head.awr,
        lis: head.l1,
        ns: head.n1,
        subsections: Vec::new(),
    };

    for _ in 0..data.ns.max(0) {
        let c = reader.cont_record()?;
        let mut sub = Mf40Subsection {
            qm: c.c1,
            qi: c.c2,
            izap: c.l1,
            lfs: c.l2,
            nl: c.n2,
            subsubsections: Vec::new(),
        };
        for _ in 0..sub.nl.max(0) {
            sub.subsubsections.push(parse_mf33_subsection(reader)?);
        }
        data.subsections.push(sub);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: f64) -> String {
        format!("{v:>11}")
    }
    fn i(v: i64) -> String {
        format!("{v:>11}")
    }
    fn line(fields: [String; 6]) -> String {
        format!("{:<66}9999341251\n", fields.concat())
    }

    /// MF=34 keeps its subsections, and LB holds LB rather than a copy of LS.
    /// Both were broken; see
    /// <https://github.com/shimwell/endf-python/issues/18>.
    #[test]
    fn mf34_keeps_its_subsections_and_reads_lb() {
        // NMT1=1; one (L, L1) pair; one NI block with LS=7 and LB=5, chosen
        // so that copying one into the other is unmistakable.
        let text = line([f(26000.0), f(55.365), i(0), i(1), i(0), i(1)])
            + &line([f(0.0), f(0.0), i(0), i(0), i(1), i(1)])
            + &line([f(0.0), f(0.0), i(1), i(1), i(1), i(1)])
            + &line([f(0.0), f(0.0), i(7), i(5), i(2), i(1)])
            + &line([f(1.0), f(2.0), f(0.0), f(0.0), f(0.0), f(0.0)]);

        let d = parse_mf34(&mut Reader::new(&text), 251).unwrap();
        assert_eq!(d.nmt1, 1);
        assert_eq!(d.subsections.len(), 1, "the parsed subsection is kept");

        let sub = &d.subsections[0];
        assert_eq!(sub.nl, 1);
        assert_eq!(sub.subsubsections.len(), 1);

        let subsub = &sub.subsubsections[0];
        assert_eq!(subsub.ls, [7.0], "LS is the symmetry flag");
        assert_eq!(subsub.lb, [5.0], "LB is the matrix type, not a copy of LS");
        assert_eq!(subsub.nt, [2.0]);
        assert_eq!(subsub.data[0], [1.0, 2.0]);
    }

    #[test]
    fn mf33_reads_an_ni_subsection() {
        // NC=0, NI=1, LB=5: a covariance matrix with its energy grid.
        let text = line([f(0.0), f(0.0), i(0), i(2), i(0), i(1)])
            + &line([f(0.0), f(0.0), i(1), i(5), i(5), i(3)])
            + &line([f(1.0), f(2.0), f(3.0), f(10.0), f(20.0), f(0.0)]);

        let sub = parse_mf33_subsection(&mut Reader::new(&text)).unwrap();
        assert_eq!(sub.mt1, 2);
        assert_eq!(sub.ni_subsections.len(), 1);
        let ni = &sub.ni_subsections[0];
        assert_eq!(ni.lb, 5);
        assert_eq!(ni.ne, 3);
        assert_eq!(ni.ek, vec![1.0, 2.0, 3.0]);
        assert_eq!(ni.fkk, vec![10.0, 20.0]);
    }
}
