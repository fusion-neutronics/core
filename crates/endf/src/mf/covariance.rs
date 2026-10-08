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
    /// LB 0 to 4, and LB 8 or 9. For LB 0 to 4 this is the first (E, F)
    /// table, NP - LT pairs of it.
    pub ek: Vec<f64>,
    pub fk: Vec<f64>,
    /// LB 0 to 4 only: the second (E, F) table, LT pairs of it. The format
    /// gives one only to LB 3 and 4; LB 0 to 2 have LT=0 and leave it empty.
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
        // One entry per subsection, whatever LTY is (the LTY=0 case must not
        // be appended a second time).
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
                // The first table holds NP - LT pairs and the second LT
                // (ENDF-102 section 33.2.2.2). Splitting at NT - NP instead is
                // right only when LT = NP / 2: with LT = 0 it would move the
                // upper half of the only table into `el`/`fl`.
                let split = (2 * (subsub.np - subsub.lt)).clamp(0, v.len() as i64) as usize;
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
    /// LCT of this (L, L1) pair: the frame its Legendre coefficients are in
    /// (1 laboratory, 2 centre of mass, 0 the same as MF=4's).
    pub lct: i64,
    /// The symmetry flag of each covariance block.
    pub ls: Vec<f64>,
    /// The covariance matrix type of each block, read from its own field
    /// rather than copied from LS.
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
    /// NL1, the number of Legendre orders of reaction MT1. Equal to NL for a
    /// reaction's covariance with itself.
    pub nl1: i64,
    pub nss: i64,
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
            nl1,
            nss,
            ..Default::default()
        };

        for _ in 0..nss.max(0) {
            let c = reader.cont_record()?;
            let ni = c.n2;
            sub.l.push(c.l1 as f64);
            sub.l1.push(c.l2 as f64);
            sub.ni.push(ni as f64);

            let mut subsub = Mf34SubSubsection {
                lct: c.n1,
                ..Default::default()
            };
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

/// Split one MF=34 covariance block, a LIST record's header fields and
/// values, into the [`NiSubsection`] shape MF=33 blocks take (ENDF-102
/// section 34.2).
///
/// - LB 0 to 2: `NE` pairs of (E, F), all in the first table (`ek`, `fk`).
///   MF=34 states no second table for them, so `lt` is zero.
/// - LB=5: `NE` energies, then the matrix in the format's packed order
///   (`fkk`); `ls` says whether it is symmetric.
/// - LB=6: `NER` row energies (the header's `N2`), `NEC = (NT - 1) / NER`
///   column energies, then the matrix (`fkl`).
///
/// Any other LB is refused: ENDF-102 allows only these in MF=34.
pub fn split_mf34_block(
    ls: i64,
    lb: i64,
    nt: i64,
    ne: i64,
    values: &[f64],
) -> Result<NiSubsection> {
    let v = values;
    let mut block = NiSubsection {
        ls,
        lb,
        nt,
        ..Default::default()
    };
    match lb {
        0..=2 => {
            block.ne = ne;
            let split = (2 * ne).clamp(0, v.len() as i64) as usize;
            block.ek = column(&v[..split], 0, 2);
            block.fk = column(&v[..split], 1, 2);
        }
        5 => {
            block.ne = ne;
            let ne = ne.clamp(0, v.len() as i64) as usize;
            block.ek = v[..ne].to_vec();
            block.fkk = v[ne..].to_vec();
        }
        6 => {
            block.ner = ne;
            let ner = ne.max(0) as usize;
            block.nec = if ner > 0 { (nt - 1) / ne } else { 0 };
            let nec = block.nec.max(0) as usize;
            block.er = v[..ner.min(v.len())].to_vec();
            block.ec = v[ner.min(v.len())..(ner + nec).min(v.len())].to_vec();
            block.fkl = v[(ner + nec).min(v.len())..].to_vec();
        }
        _ => {
            return Err(Error::Unsupported {
                what: "an MF=34 LB value other than 0, 1, 2, 5 or 6",
            })
        }
    }
    Ok(block)
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

    #[test]
    fn an_mf34_lb1_block_is_one_table_of_pairs() {
        let b = split_mf34_block(0, 1, 6, 3, &[1.0, 0.1, 2.0, 0.2, 3.0, 0.3]).unwrap();
        assert_eq!(b.ek, vec![1.0, 2.0, 3.0]);
        assert_eq!(b.fk, vec![0.1, 0.2, 0.3]);
        assert!(b.el.is_empty() && b.fl.is_empty());
        assert_eq!((b.lb, b.ne, b.lt), (1, 3, 0));
    }

    #[test]
    fn an_mf34_lb5_block_is_energies_then_the_packed_matrix() {
        // Symmetric (LS=1), NE=3 energies, so two intervals and three packed
        // values.
        let b = split_mf34_block(1, 5, 6, 3, &[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
        assert_eq!(b.ek, vec![1.0, 2.0, 3.0]);
        assert_eq!(b.fkk, vec![0.1, 0.2, 0.3]);
        assert_eq!((b.ls, b.lb, b.ne), (1, 5, 3));
    }

    #[test]
    fn an_mf34_lb6_block_splits_rows_columns_and_matrix() {
        // NER=2 rows, NEC=(NT-1)/NER=3 columns: 2 + 3 + 2 values, NT=7.
        let v = [1.0, 2.0, 10.0, 20.0, 30.0, 0.5, 0.6];
        let b = split_mf34_block(0, 6, 7, 2, &v).unwrap();
        assert_eq!(b.er, vec![1.0, 2.0]);
        assert_eq!(b.ec, vec![10.0, 20.0, 30.0]);
        assert_eq!(b.fkl, vec![0.5, 0.6]);
        assert_eq!((b.ner, b.nec), (2, 3));
    }

    #[test]
    fn an_mf34_block_with_an_unknown_lb_is_refused() {
        assert!(split_mf34_block(0, 8, 2, 1, &[1.0, 2.0]).is_err());
    }

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

    /// MF=40 wraps MF=33 sub-subsections in a per-product-state CONT, so what
    /// has to come out is the state's own keys (QM, QI, IZAP, LFS) and the
    /// block beneath them, with the partner state in XMF1 and XLFS1.
    #[test]
    fn mf40_reads_a_product_state_and_its_blocks() {
        // HEAD with NS=1; the Nb92_m1 state of (n,2n), LFS=1, NL=1; one
        // MF=33-format subsection correlating it with itself (XMF1=10,
        // XLFS1=1, MT1=16) holding one LB=5 LS=1 block on three energies.
        let text = line([f(41093.0), f(92.10827), i(0), i(0), i(1), i(0)])
            + &line([f(-8830870.0), f(-8966370.0), i(41092), i(1), i(0), i(1)])
            + &line([f(10.0), f(1.0), i(0), i(16), i(0), i(1)])
            + &line([f(0.0), f(0.0), i(1), i(5), i(6), i(3)])
            + &line([f(1.0e-5), f(1.0e7), f(2.0e7), f(0.04), f(0.01), f(0.09)]);

        let d = parse_mf40(&mut Reader::new(&text)).unwrap();
        assert_eq!(d.za, 41093);
        assert_eq!(d.ns, 1);
        assert_eq!(d.subsections.len(), 1);

        let state = &d.subsections[0];
        assert_eq!(state.qm, -8830870.0);
        assert_eq!(state.qi, -8966370.0);
        assert_eq!(state.izap, 41092);
        assert_eq!(state.lfs, 1);
        assert_eq!(state.nl, 1);
        assert_eq!(state.subsubsections.len(), 1);

        let sub = &state.subsubsections[0];
        assert_eq!(sub.xmf1, 10.0);
        assert_eq!(sub.xlfs1, 1.0);
        assert_eq!(sub.mat1, 0);
        assert_eq!(sub.mt1, 16);
        assert!(sub.nc_subsections.is_empty());
        assert_eq!(sub.ni_subsections.len(), 1);
        let ni = &sub.ni_subsections[0];
        assert_eq!((ni.lb, ni.ls, ni.nt, ni.ne), (5, 1, 6, 3));
        assert_eq!(ni.ek, vec![1.0e-5, 1.0e7, 2.0e7]);
        assert_eq!(ni.fkk, vec![0.04, 0.01, 0.09]);
    }

    /// LB 0 to 4 split at 2*(NP - LT) values. Splitting at NT - NP agrees only
    /// when LT = NP / 2.
    #[test]
    fn mf33_lb1_keeps_its_one_table_whole() {
        // LT=0, LB=1, NT=6, NP=3: one table. NP is odd, so the old split also
        // started the second array on an F value.
        let text = line([f(0.0), f(0.0), i(0), i(0), i(0), i(1)])
            + &line([f(0.0), f(0.0), i(0), i(1), i(6), i(3)])
            + &line([f(1.0e-5), f(0.0), f(8.0e5), f(0.01125), f(2.0e7), f(0.0)]);

        let sub = parse_mf33_subsection(&mut Reader::new(&text)).unwrap();
        let ni = &sub.ni_subsections[0];
        assert_eq!((ni.lb, ni.lt, ni.np), (1, 0, 3));
        assert_eq!(ni.ek, vec![1.0e-5, 8.0e5, 2.0e7]);
        assert_eq!(ni.fk, vec![0.0, 0.01125, 0.0]);
        assert!(
            ni.el.is_empty() && ni.fl.is_empty(),
            "LT=0 has no second table"
        );
    }

    #[test]
    fn mf33_lb3_splits_at_the_second_tables_own_length() {
        // LT=2, LB=3, NT=10, NP=5: three pairs, then two. NT - NP would have
        // split two and a half pairs in.
        let text = line([f(0.0), f(0.0), i(0), i(0), i(0), i(1)])
            + &line([f(0.0), f(0.0), i(2), i(3), i(10), i(5)])
            + &line([f(1.0), f(0.1), f(2.0), f(0.2), f(3.0), f(0.0)])
            + &line([f(1.0), f(0.3), f(3.0), f(0.0), f(0.0), f(0.0)]);

        let sub = parse_mf33_subsection(&mut Reader::new(&text)).unwrap();
        let ni = &sub.ni_subsections[0];
        assert_eq!(ni.ek, vec![1.0, 2.0, 3.0]);
        assert_eq!(ni.fk, vec![0.1, 0.2, 0.0]);
        assert_eq!(ni.el, vec![1.0, 3.0]);
        assert_eq!(ni.fl, vec![0.3, 0.0]);
    }
}
