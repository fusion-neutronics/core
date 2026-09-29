//! MF=32 MT=151, covariances of resonance parameters.
//!
//! The file parallels MF=2: the same isotope and energy range records, each
//! range followed by the covariance of the parameters MF=2 gives for it. What
//! follows the range record depends on LRU, LRF and the compatibility flag
//! LCOMP, and every combination an evaluated library actually writes is read
//! here (ENDF-102 section 32.2):
//!
//! | Range | LCOMP | Read into |
//! |---|---|---|
//! | resolved, LRF=2 | 0 | [`Covariance::Compatible`] |
//! | resolved, LRF=2 or 3 | 1 | [`Covariance::General`] |
//! | resolved, LRF=7 | 1 | [`Covariance::GeneralRMatrix`] |
//! | resolved, LRF=2 or 3 | 2 | [`Covariance::Compact`] |
//! | resolved, LRF=7 | 2 | [`Covariance::CompactRMatrix`] |
//! | unresolved | | [`Covariance::Unresolved`] |
//!
//! Everything is kept as the file writes it: the parameters and their
//! uncertainties, the covariance triangles in their packed order, and each
//! INTG line of a compact correlation matrix field by field. Nothing is
//! expanded, rescaled or repaired on the way in. [`PackedCovariance::get`] and
//! [`CompactCorrelation::entries`] expand on request.
//!
//! The representations the format defines but no library uses are refused by
//! name rather than guessed at: an energy-dependent scattering radius
//! (NRO/=0), long-range covariances (NLRS>0), single-level Breit-Wigner
//! (LRF=1), Adler-Adler (LRF=4), and a scattering radius uncertainty in the
//! compatible format (LCOMP=0 with ISR>0).
//!
//! Two kinds of defect do occur on real tapes, and both are recorded in
//! [`Mf32::defects`] with the data left untouched: an INTG line whose row
//! index lies outside the matrix, and a negative variance on the diagonal of a
//! covariance block. Whether a matrix is positive semi-definite is a question
//! for whatever samples from it, not for the reader.

use crate::error::{Error, Result};
use crate::records::{IntgLine, ListRecord, Reader};

/// MF=32 MT=151.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf32 {
    pub za: i64,
    pub awr: f64,
    pub nis: i64,
    pub isotopes: Vec<Isotope>,
    /// What was found wrong with the data, in the order it was read. Empty
    /// for a clean evaluation.
    pub defects: Vec<Defect>,
}

/// One isotope of the material, with its energy ranges.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Isotope {
    pub zai: f64,
    pub abn: f64,
    pub lfw: i64,
    pub ner: i64,
    pub ranges: Vec<Range>,
}

/// One energy range, and the covariance given over it.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub el: f64,
    pub eh: f64,
    /// 1 resolved, 2 unresolved.
    pub lru: i64,
    /// The MF=2 representation the parameters belong to.
    pub lrf: i64,
    pub nro: i64,
    pub naps: i64,
    pub covariance: Covariance,
}

/// The covariance of a range, in whichever format it uses.
#[derive(Debug, Clone, PartialEq)]
pub enum Covariance {
    /// LCOMP=0, Breit-Wigner: the ENDF/B-V compatible format.
    Compatible(Box<Compatible>),
    /// LCOMP=1, LRF=2 or 3: explicit covariance blocks.
    General(Box<General>),
    /// LCOMP=1, LRF=7.
    GeneralRMatrix(Box<GeneralRMatrix>),
    /// LCOMP=2, LRF=2 or 3: uncertainties and a packed correlation matrix.
    Compact(Box<Compact>),
    /// LCOMP=2, LRF=7.
    CompactRMatrix(Box<CompactRMatrix>),
    /// LRU=2.
    Unresolved(Box<Unresolved>),
}

/// The uncertainty on the scattering radius, present when ISR=1.
#[derive(Debug, Clone, PartialEq)]
pub enum ScatteringRadiusUncertainty {
    /// LRF=2: a CONT record carrying DAP alone.
    Cont { dap: f64 },
    /// LRF=3 or 7: a LIST record. For LRF=3 the values are DAP followed by
    /// the per-L DAPi (MLS of them in all); for LRF=7, one DAP per channel of
    /// each spin group.
    List { values: Vec<f64> },
}

/// The upper triangle of a symmetric matrix, by rows, as the format packs it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PackedCovariance {
    pub order: usize,
    /// `order * (order + 1) / 2` values: row 0 from the diagonal out, then
    /// row 1 from its diagonal, and so on.
    pub values: Vec<f64>,
}

impl PackedCovariance {
    /// Where element `(i, j)`, with `i <= j`, sits in [`Self::values`].
    fn index(&self, i: usize, j: usize) -> usize {
        i * self.order - i * i.saturating_sub(1) / 2 + (j - i)
    }

    /// Element `(i, j)` of the full symmetric matrix, 0-based.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        self.values[self.index(i, j)]
    }

    /// The diagonal, in order.
    pub fn diagonal(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.order).map(move |i| self.values[self.index(i, i)])
    }
}

// -------------------------------------------------------------------------
// LCOMP=0
// -------------------------------------------------------------------------

/// LCOMP=0: variances and covariances within each resonance only.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Compatible {
    pub spi: f64,
    pub ap: f64,
    pub nls: i64,
    pub sections: Vec<CompatibleSection>,
}

/// The resonances of one orbital angular momentum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompatibleSection {
    pub awri: f64,
    pub l: i64,
    pub nrs: i64,
    /// Eighteen values per resonance: ER, AJ, GT, GN, GG, GF, then DE2, DN2,
    /// DNDG, DG2, DNDF, DGDF, DF2, DJDN, DJDG, DJDF, DJ2, and one unused.
    pub resonances: Vec<[f64; 18]>,
}

// -------------------------------------------------------------------------
// LCOMP=1
// -------------------------------------------------------------------------

/// LCOMP=1 for Breit-Wigner or Reich-Moore parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct General {
    pub spi: f64,
    pub ap: f64,
    /// LAD for LRF=3, zero for LRF=2.
    pub lad: i64,
    pub nls: i64,
    pub isr: i64,
    pub dap: Option<ScatteringRadiusUncertainty>,
    pub awri: f64,
    pub nsrs: i64,
    pub nlrs: i64,
    /// The NSRS short-range blocks.
    pub blocks: Vec<CovarianceBlock>,
}

/// One short-range block: a set of resonances and the covariance among them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CovarianceBlock {
    /// Parameters per resonance in the matrix: ER, GN, GG, GF, GX for LRF=2,
    /// ER, GN, GG, GFA, GFB for LRF=3, the first MPAR of them.
    pub mpar: i64,
    pub nrb: i64,
    /// The six MF=2 parameters of each resonance, for identification: ER,
    /// AJ, GT, GN, GG, GF for LRF=2 and ER, AJ, GN, GG, GFA, GFB for LRF=3.
    pub resonances: Vec<[f64; 6]>,
    /// Order `mpar * nrb`; parameter `j` of resonance `k` is row
    /// `k * mpar + j`.
    pub covariance: PackedCovariance,
}

/// LCOMP=1 for R-matrix limited parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GeneralRMatrix {
    pub ifg: i64,
    pub njs: i64,
    pub isr: i64,
    pub dap: Option<ScatteringRadiusUncertainty>,
    pub awri: f64,
    pub nsrs: i64,
    pub nlrs: i64,
    pub blocks: Vec<RMatrixBlock>,
}

/// One short-range block of an R-matrix limited evaluation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RMatrixBlock {
    pub njsx: i64,
    pub spin_groups: Vec<RMatrixBlockSpinGroup>,
    /// Order NPARB, the sum over the spin groups of `(nch + 1) * nrb`, in the
    /// order the parameters are listed.
    pub covariance: PackedCovariance,
}

/// The resonances one spin group contributes to an R-matrix block.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RMatrixBlockSpinGroup {
    pub nch: i64,
    pub nrb: i64,
    pub nx: i64,
    /// The LIST values as written: each resonance is ER and its `nch` widths,
    /// padded to a whole number of lines.
    pub values: Vec<f64>,
}

impl RMatrixBlockSpinGroup {
    /// ER then the `nch` widths of resonance `k`.
    pub fn resonance(&self, k: usize) -> &[f64] {
        let stride = self.values.len() / self.nrb.max(1) as usize;
        &self.values[k * stride..k * stride + 1 + self.nch as usize]
    }
}

// -------------------------------------------------------------------------
// LCOMP=2
// -------------------------------------------------------------------------

/// LCOMP=2 for Breit-Wigner or Reich-Moore parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Compact {
    pub spi: f64,
    pub ap: f64,
    /// LAD for LRF=3, zero for LRF=2.
    pub lad: i64,
    pub isr: i64,
    pub dap: Option<ScatteringRadiusUncertainty>,
    pub awri: f64,
    /// APL for LRF=3; LRF=2 writes QX in the same place.
    pub apl: f64,
    /// LRX for LRF=2, zero for LRF=3.
    pub lrx: i64,
    pub nrsa: i64,
    pub resonances: Vec<CompactResonance>,
    pub correlation: CompactCorrelation,
}

/// One resonance of the compact format: its MF=2 parameters and their
/// uncertainties, position for position.
///
/// For LRF=2 the parameters are ER, AJ, GT, GN, GG, GF and the uncertainties
/// DER, 0, 0, DGN, DGG, DGF; for LRF=3, ER, AJ, GN, GG, GFA, GFB and DER, 0,
/// DGN, DGG, DGFA, DGFB.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CompactResonance {
    pub parameters: [f64; 6],
    pub uncertainties: [f64; 6],
}

/// LCOMP=2 for R-matrix limited parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompactRMatrix {
    pub ifg: i64,
    pub njs: i64,
    pub isr: i64,
    pub dap: Option<ScatteringRadiusUncertainty>,
    pub npp: i64,
    pub njsx: i64,
    /// Twelve values per pair: MA, MB, ZA, ZB, IA, IB, Q, PNT, SHF, MT, PA,
    /// PB.
    pub particle_pairs: Vec<[f64; 12]>,
    pub spin_groups: Vec<CompactSpinGroup>,
    pub correlation: CompactCorrelation,
}

/// One spin group of the compact R-matrix format.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompactSpinGroup {
    pub aj: f64,
    pub pj: f64,
    pub nch: i64,
    /// Six values per channel: PPI, L, SCH, BND, APE, APT.
    pub channels: Vec<[f64; 6]>,
    pub nrsa: i64,
    pub nx: i64,
    /// The LIST values as written: for each resonance, ER and its widths
    /// padded to whole lines, then DER and the width uncertainties padded the
    /// same way.
    pub values: Vec<f64>,
}

impl CompactSpinGroup {
    fn stride(&self) -> usize {
        self.values.len() / self.nrsa.max(1) as usize
    }

    /// ER then the `nch` widths of resonance `k`.
    pub fn parameters(&self, k: usize) -> &[f64] {
        let s = self.stride();
        &self.values[k * s..k * s + 1 + self.nch as usize]
    }

    /// DER then the `nch` width uncertainties of resonance `k`.
    pub fn uncertainties(&self, k: usize) -> &[f64] {
        let s = self.stride();
        let o = k * s + s / 2;
        &self.values[o..o + 1 + self.nch as usize]
    }
}

/// A correlation matrix in the compact integer encoding.
///
/// Stored line by line as the file writes it. [`Self::entries`] decodes it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompactCorrelation {
    pub ndigit: i64,
    /// The order of the matrix.
    pub nnn: i64,
    pub rows: Vec<IntgLine>,
}

impl CompactCorrelation {
    /// The correlation a packed integer stands for: the centre of the
    /// interval it was rounded from, so 87 at NDIGIT=2 is 0.875.
    pub fn decode(&self, k: i64) -> f64 {
        let factor = 10f64.powi(self.ndigit as i32);
        if k > 0 {
            (k as f64 + 0.5) / factor
        } else if k < 0 {
            (k as f64 - 0.5) / factor
        } else {
            0.0
        }
    }

    /// Whether a line addresses a row inside the matrix. Lines that do not are
    /// reported in [`Mf32::defects`] and skipped by [`Self::entries`].
    pub fn in_matrix(&self, row: &IntgLine) -> bool {
        row.ii >= 1 && row.ii <= self.nnn && row.jj >= 1
    }

    /// Every non-zero off-diagonal correlation below the diagonal, as
    /// `(i, j, c)` with `i > j`, 0-based. The diagonal is 1 and the upper
    /// triangle is the mirror image; neither is listed.
    ///
    /// The field at position `n` of a line is column `jj + n`. The FORTRAN
    /// sample in ENDF-102 writes every field of a line to column `jj`; that is
    /// an error in the manual, which the prose beside it contradicts.
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.rows
            .iter()
            .filter(|row| self.in_matrix(row))
            .flat_map(move |row| {
                row.kij
                    .iter()
                    .enumerate()
                    .map(move |(n, &k)| (row.jj + n as i64, k))
                    .take_while(move |&(col, _)| col < row.ii)
                    .filter(|&(_, k)| k != 0)
                    .map(move |(col, k)| {
                        ((row.ii - 1) as usize, (col - 1) as usize, self.decode(k))
                    })
            })
    }
}

// -------------------------------------------------------------------------
// LRU=2
// -------------------------------------------------------------------------

/// LRU=2: the relative covariance of the average unresolved parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Unresolved {
    pub spi: f64,
    pub ap: f64,
    pub lssf: i64,
    pub nls: i64,
    pub l_values: Vec<UnresolvedL>,
    /// Parameters per (L, J): D, GNO, GG, GF, GX, the first MPAR of them.
    /// With LFW=0 and MPAR=4 the four are D, GNO, GG and GX.
    pub mpar: i64,
    /// Order `mpar` times the number of (L, J) pairs, in listing order.
    pub relative_covariance: PackedCovariance,
}

/// The average parameters of one orbital angular momentum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnresolvedL {
    pub awri: f64,
    pub l: i64,
    pub njs: i64,
    /// Six values per J: D, AJ, GNO, GG, GF, GX.
    pub parameters: Vec<[f64; 6]>,
}

// -------------------------------------------------------------------------
// Defects
// -------------------------------------------------------------------------

/// A defect in the data, recorded rather than repaired.
///
/// `isotope` and `range` index [`Mf32::isotopes`] and [`Isotope::ranges`].
#[derive(Debug, Clone, PartialEq)]
pub enum Defect {
    /// An INTG line addresses a row outside the NNN by NNN matrix (or has a
    /// row or column index below 1). The line is kept in
    /// [`CompactCorrelation::rows`], at index `line`, and
    /// [`CompactCorrelation::entries`] skips it.
    CorrelationRowOutsideMatrix {
        isotope: usize,
        range: usize,
        line: usize,
        ii: i64,
        jj: i64,
        nnn: i64,
    },
    /// A negative value on the diagonal of a covariance matrix: element
    /// `index` of block `block` (always 0 for an unresolved range). JEFF-4.0
    /// writes -1e6 as the variance of an absent fission width, which is what
    /// this was added to catch.
    NegativeVariance {
        isotope: usize,
        range: usize,
        block: usize,
        index: usize,
        value: f64,
    },
}

/// Where in the section a range sits, for [`Defect`].
#[derive(Clone, Copy)]
struct Site {
    isotope: usize,
    range: usize,
}

// -------------------------------------------------------------------------
// Parsing
// -------------------------------------------------------------------------

/// Split a LIST's values into fixed-width rows.
fn rows<const N: usize>(values: &[f64]) -> Vec<[f64; N]> {
    values
        .chunks_exact(N)
        .map(|c| c.try_into().expect("chunks_exact yields N values"))
        .collect()
}

/// A LIST whose length the format fixes from its own counts.
fn expect_len(list: &ListRecord, n: i64, what: &'static str) -> Result<()> {
    if n < 0 || list.values.len() != n as usize {
        return Err(Error::Mismatched { what });
    }
    Ok(())
}

/// The upper triangle of an `order` by `order` matrix, from a LIST.
fn packed(values: Vec<f64>, order: i64, what: &'static str) -> Result<PackedCovariance> {
    let order = order.max(0) as usize;
    if values.len() != order * (order + 1) / 2 {
        return Err(Error::Mismatched { what });
    }
    Ok(PackedCovariance { order, values })
}

fn flag_negative_variances(
    cov: &PackedCovariance,
    site: Site,
    block: usize,
    defects: &mut Vec<Defect>,
) {
    for (index, value) in cov.diagonal().enumerate() {
        if value < 0.0 {
            defects.push(Defect::NegativeVariance {
                isotope: site.isotope,
                range: site.range,
                block,
                index,
                value,
            });
        }
    }
}

/// The scattering radius uncertainty ISR=1 announces, in the record type LRF
/// calls for.
fn parse_dap(
    reader: &mut Reader,
    lrf: i64,
    isr: i64,
) -> Result<Option<ScatteringRadiusUncertainty>> {
    match isr {
        0 => Ok(None),
        1 if lrf == 2 => Ok(Some(ScatteringRadiusUncertainty::Cont {
            dap: reader.cont_record()?.c2,
        })),
        1 => Ok(Some(ScatteringRadiusUncertainty::List {
            values: reader.list_record()?.values,
        })),
        _ => Err(Error::Unsupported {
            what: "an MF=32 ISR flag other than 0 or 1",
        }),
    }
}

fn parse_correlation(
    reader: &mut Reader,
    site: Site,
    defects: &mut Vec<Defect>,
) -> Result<CompactCorrelation> {
    let c = reader.cont_record()?;
    let mut corr = CompactCorrelation {
        ndigit: c.l1,
        nnn: c.l2,
        rows: Vec::with_capacity(c.n1.max(0) as usize),
    };
    for line in 0..c.n1.max(0) as usize {
        let row = reader.intg_line(corr.ndigit)?;
        if !corr.in_matrix(&row) {
            defects.push(Defect::CorrelationRowOutsideMatrix {
                isotope: site.isotope,
                range: site.range,
                line,
                ii: row.ii,
                jj: row.jj,
                nnn: corr.nnn,
            });
        }
        corr.rows.push(row);
    }
    Ok(corr)
}

/// A resolved Breit-Wigner (LRF=2) or Reich-Moore (LRF=3) range.
fn parse_resolved(
    reader: &mut Reader,
    lrf: i64,
    site: Site,
    defects: &mut Vec<Defect>,
) -> Result<Covariance> {
    let c = reader.cont_record()?;
    let (spi, ap, lad, lcomp, nls, isr) = (c.c1, c.c2, c.l1, c.l2, c.n1, c.n2);
    match lcomp {
        0 => {
            if lrf != 2 {
                return Err(Error::Unsupported {
                    what: "the MF=32 compatible format (LCOMP=0) for anything but \
                           Breit-Wigner parameters",
                });
            }
            if isr != 0 {
                return Err(Error::Unsupported {
                    what: "a scattering radius uncertainty in the MF=32 compatible \
                           format (LCOMP=0 with ISR>0)",
                });
            }
            let mut data = Compatible {
                spi,
                ap,
                nls,
                sections: Vec::new(),
            };
            for _ in 0..nls.max(0) {
                let list = reader.list_record()?;
                let nrs = list.cont.n2;
                expect_len(&list, 18 * nrs, "an MF=32 LCOMP=0 LIST's length and 18*NRS")?;
                data.sections.push(CompatibleSection {
                    awri: list.cont.c1,
                    l: list.cont.l1,
                    nrs,
                    resonances: rows(&list.values),
                });
            }
            Ok(Covariance::Compatible(Box::new(data)))
        }
        1 => {
            let dap = parse_dap(reader, lrf, isr)?;
            let c = reader.cont_record()?;
            if c.n2 != 0 {
                return Err(Error::Unsupported {
                    what: "MF=32 long-range covariance subsections (NLRS>0)",
                });
            }
            let mut data = General {
                spi,
                ap,
                lad,
                nls,
                isr,
                dap,
                awri: c.c1,
                nsrs: c.n1,
                nlrs: c.n2,
                blocks: Vec::new(),
            };
            for block in 0..data.nsrs.max(0) as usize {
                let mut list = reader.list_record()?;
                let (mpar, nrb) = (list.cont.l1, list.cont.n2);
                let n_res = 6 * nrb.max(0) as usize;
                if list.values.len() < n_res {
                    return Err(Error::Mismatched {
                        what: "an MF=32 LCOMP=1 block's length and its NRB resonances",
                    });
                }
                let covariance = list.values.split_off(n_res);
                let covariance = packed(
                    covariance,
                    mpar * nrb,
                    "an MF=32 LCOMP=1 block's NVS and MPAR*NRB",
                )?;
                flag_negative_variances(&covariance, site, block, defects);
                data.blocks.push(CovarianceBlock {
                    mpar,
                    nrb,
                    resonances: rows(&list.values),
                    covariance,
                });
            }
            Ok(Covariance::General(Box::new(data)))
        }
        2 => {
            let dap = parse_dap(reader, lrf, isr)?;
            let list = reader.list_record()?;
            let nrsa = list.cont.n2;
            expect_len(
                &list,
                12 * nrsa,
                "an MF=32 LCOMP=2 LIST's length and 12*NRSA",
            )?;
            let resonances = list
                .values
                .chunks_exact(12)
                .map(|c| CompactResonance {
                    parameters: c[..6].try_into().expect("six values"),
                    uncertainties: c[6..].try_into().expect("six values"),
                })
                .collect();
            Ok(Covariance::Compact(Box::new(Compact {
                spi,
                ap,
                lad,
                isr,
                dap,
                awri: list.cont.c1,
                apl: list.cont.c2,
                lrx: list.cont.l2,
                nrsa,
                resonances,
                correlation: parse_correlation(reader, site, defects)?,
            })))
        }
        _ => Err(Error::Unsupported {
            what: "an MF=32 LCOMP other than 0, 1 or 2",
        }),
    }
}

/// A resolved R-matrix limited (LRF=7) range.
fn parse_r_matrix(
    reader: &mut Reader,
    site: Site,
    defects: &mut Vec<Defect>,
) -> Result<Covariance> {
    let c = reader.cont_record()?;
    let (ifg, lcomp, njs, isr) = (c.l1, c.l2, c.n1, c.n2);
    match lcomp {
        1 => {
            let dap = parse_dap(reader, 7, isr)?;
            let c = reader.cont_record()?;
            if c.n2 != 0 {
                return Err(Error::Unsupported {
                    what: "MF=32 long-range covariance subsections (NLRS>0)",
                });
            }
            let mut data = GeneralRMatrix {
                ifg,
                njs,
                isr,
                dap,
                awri: c.c1,
                nsrs: c.n1,
                nlrs: c.n2,
                blocks: Vec::new(),
            };
            for block in 0..data.nsrs.max(0) as usize {
                let njsx = reader.cont_record()?.l1;
                let mut spin_groups = Vec::new();
                let mut nparb = 0i64;
                for _ in 0..njsx.max(0) {
                    let list = reader.list_record()?;
                    let (nch, nrb) = (list.cont.l1, list.cont.l2);
                    let fits = if nrb > 0 {
                        let len = list.values.len();
                        len % nrb as usize == 0 && len / nrb as usize > nch.max(0) as usize
                    } else {
                        list.values.is_empty()
                    };
                    if nch < 0 || nrb < 0 || !fits {
                        return Err(Error::Mismatched {
                            what: "an MF=32 LRF=7 LCOMP=1 spin group's length and NCH and NRB",
                        });
                    }
                    nparb += (nch + 1) * nrb;
                    spin_groups.push(RMatrixBlockSpinGroup {
                        nch,
                        nrb,
                        nx: list.cont.n2,
                        values: list.values,
                    });
                }
                let list = reader.list_record()?;
                if list.cont.n2 != nparb {
                    return Err(Error::Mismatched {
                        what: "an MF=32 LRF=7 LCOMP=1 block's NPARB and its spin groups",
                    });
                }
                let covariance = packed(
                    list.values,
                    nparb,
                    "an MF=32 LRF=7 LCOMP=1 block's length and NPARB",
                )?;
                flag_negative_variances(&covariance, site, block, defects);
                data.blocks.push(RMatrixBlock {
                    njsx,
                    spin_groups,
                    covariance,
                });
            }
            Ok(Covariance::GeneralRMatrix(Box::new(data)))
        }
        2 => {
            let dap = parse_dap(reader, 7, isr)?;
            let list = reader.list_record()?;
            let npp = list.cont.l1;
            expect_len(
                &list,
                12 * npp,
                "an MF=32 LRF=7 particle pair LIST's length and 12*NPP",
            )?;
            let mut data = CompactRMatrix {
                ifg,
                njs,
                isr,
                dap,
                npp,
                njsx: list.cont.l2,
                particle_pairs: rows(&list.values),
                ..Default::default()
            };
            for _ in 0..njs.max(0) {
                let list = reader.list_record()?;
                let nch = list.cont.n2;
                expect_len(
                    &list,
                    6 * nch,
                    "an MF=32 LRF=7 channel LIST's length and 6*NCH",
                )?;
                let mut group = CompactSpinGroup {
                    aj: list.cont.c1,
                    pj: list.cont.c2,
                    nch,
                    channels: rows(&list.values),
                    ..Default::default()
                };
                let list = reader.list_record()?;
                group.nrsa = list.cont.l2;
                group.nx = list.cont.n2;
                let fits = if group.nrsa > 0 {
                    let len = list.values.len();
                    let stride = len / group.nrsa as usize;
                    len % group.nrsa as usize == 0
                        && stride % 2 == 0
                        && stride / 2 > nch.max(0) as usize
                } else {
                    list.values.is_empty()
                };
                if group.nrsa < 0 || !fits {
                    return Err(Error::Mismatched {
                        what: "an MF=32 LRF=7 LCOMP=2 resonance LIST's length and NCH and NRSA",
                    });
                }
                group.values = list.values;
                data.spin_groups.push(group);
            }
            data.correlation = parse_correlation(reader, site, defects)?;
            Ok(Covariance::CompactRMatrix(Box::new(data)))
        }
        _ => Err(Error::Unsupported {
            what: "an MF=32 R-matrix limited range with LCOMP other than 1 or 2",
        }),
    }
}

/// An unresolved range.
fn parse_unresolved(
    reader: &mut Reader,
    site: Site,
    defects: &mut Vec<Defect>,
) -> Result<Covariance> {
    let c = reader.cont_record()?;
    let mut data = Unresolved {
        spi: c.c1,
        ap: c.c2,
        lssf: c.l1,
        nls: c.n1,
        ..Default::default()
    };
    let mut n_lj = 0i64;
    for _ in 0..data.nls.max(0) {
        let list = reader.list_record()?;
        let njs = list.cont.n2;
        expect_len(
            &list,
            6 * njs,
            "an MF=32 unresolved LIST's length and 6*NJS",
        )?;
        n_lj += njs;
        data.l_values.push(UnresolvedL {
            awri: list.cont.c1,
            l: list.cont.l1,
            njs,
            parameters: rows(&list.values),
        });
    }
    let list = reader.list_record()?;
    data.mpar = list.cont.l1;
    let npar = list.cont.n2;
    if npar != data.mpar * n_lj {
        return Err(Error::Mismatched {
            what: "an MF=32 unresolved NPAR and MPAR times the number of (L, J) pairs",
        });
    }
    data.relative_covariance = packed(
        list.values,
        npar,
        "an MF=32 unresolved covariance's length and NPAR",
    )?;
    flag_negative_variances(&data.relative_covariance, site, 0, defects);
    Ok(Covariance::Unresolved(Box::new(data)))
}

/// Parse MF=32 MT=151.
pub fn parse_mf32(reader: &mut Reader) -> Result<Mf32> {
    let head = reader.head_record()?;
    let mut data = Mf32 {
        za: head.za,
        awr: head.awr,
        nis: head.n1,
        ..Default::default()
    };

    for i in 0..data.nis.max(0) as usize {
        let c = reader.cont_record()?;
        let mut iso = Isotope {
            zai: c.c1,
            abn: c.c2,
            lfw: c.l2,
            ner: c.n1,
            ranges: Vec::new(),
        };

        for r in 0..iso.ner.max(0) as usize {
            let c = reader.cont_record()?;
            let (lru, lrf, nro) = (c.l1, c.l2, c.n1);
            let site = Site {
                isotope: i,
                range: r,
            };
            if nro != 0 {
                return Err(Error::Unsupported {
                    what: "an energy-dependent scattering radius covariance in MF=32 (NRO/=0)",
                });
            }
            let covariance = match (lru, lrf) {
                (1, 2) | (1, 3) => parse_resolved(reader, lrf, site, &mut data.defects)?,
                (1, 7) => parse_r_matrix(reader, site, &mut data.defects)?,
                (1, 1) => {
                    return Err(Error::Unsupported {
                        what: "single-level Breit-Wigner covariances in MF=32 (LRF=1)",
                    })
                }
                (1, 4) => {
                    return Err(Error::Unsupported {
                        what: "Adler-Adler covariances in MF=32 (LRF=4)",
                    })
                }
                (2, 1) => parse_unresolved(reader, site, &mut data.defects)?,
                (2, _) => {
                    return Err(Error::Unsupported {
                        what: "an MF=32 unresolved range with LRF other than 1",
                    })
                }
                _ => {
                    return Err(Error::Unsupported {
                        what: "an MF=32 range with an unrecognised LRU or LRF",
                    })
                }
            };
            iso.ranges.push(Range {
                el: c.c1,
                eh: c.c2,
                lru,
                lrf,
                nro,
                naps: c.n2,
                covariance,
            });
        }

        data.isotopes.push(iso);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{Material, Section};

    /// Writes MF=32 records in the format's fixed columns. Values are plain
    /// decimals, right-justified in 11 columns, which `float_endf` reads as
    /// readily as the e-less exponential form.
    #[derive(Default)]
    struct Tape {
        text: String,
    }

    impl Tape {
        fn line(&mut self, body: &str) {
            self.text += &format!("{body:<66}9999321511\n");
        }

        fn cont(&mut self, c1: f64, c2: f64, l1: i64, l2: i64, n1: i64, n2: i64) -> &mut Self {
            self.line(&format!("{c1:>11}{c2:>11}{l1:>11}{l2:>11}{n1:>11}{n2:>11}"));
            self
        }

        /// A LIST record; NPL is the number of values given.
        fn list(&mut self, c1: f64, c2: f64, l1: i64, l2: i64, n2: i64, v: &[f64]) -> &mut Self {
            self.cont(c1, c2, l1, l2, v.len() as i64, n2);
            for chunk in v.chunks(6) {
                let body: String = chunk.iter().map(|x| format!("{x:>11}")).collect();
                self.line(&body);
            }
            self
        }

        /// One INTG line, laid out as ENDF-102 gives it for `ndigit`.
        fn intg(&mut self, ndigit: i64, ii: i64, jj: i64, k: &[i64]) -> &mut Self {
            let (gap, width) = if ndigit == 6 {
                (0, 7)
            } else {
                (1, ndigit as usize + 1)
            };
            let fields: String = k.iter().map(|k| format!("{k:>width$}")).collect();
            self.line(&format!("{ii:>5}{jj:>5}{}{fields}", " ".repeat(gap)));
            self
        }

        /// The HEAD and isotope records for one isotope with `ner` ranges.
        fn head(ner: i64) -> Tape {
            let mut t = Tape::default();
            t.cont(26056.0, 55.454, 0, 0, 1, 0);
            t.cont(26056.0, 1.0, 0, 0, ner, 0);
            t
        }

        fn parse(&self) -> Result<Mf32> {
            let mut r = Reader::new(&self.text);
            let d = parse_mf32(&mut r)?;
            assert!(r.is_empty(), "{} lines left unread", r.remaining());
            Ok(d)
        }
    }

    fn only_range(d: &Mf32) -> &Covariance {
        &d.isotopes[0].ranges[0].covariance
    }

    /// LCOMP=0 followed by an unresolved range: both read, and the second
    /// starts where the first ends.
    #[test]
    fn reads_the_compatible_format_then_an_unresolved_range() {
        let mut t = Tape::head(2);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 0, 1, 0);
        let mut res = [0.0; 18];
        res[..6].copy_from_slice(&[10.0, 0.5, 1.2, 1.0, 0.2, 0.0]);
        res[6..13].copy_from_slice(&[0.01, 0.04, -0.001, 0.0009, 0.0, 0.0, 0.0]);
        t.list(55.454, 0.0, 0, 0, 1, &res);
        // The unresolved range: two (L, J) pairs, MPAR=2, so order 4.
        t.cont(1000.0, 100000.0, 2, 1, 0, 0);
        t.cont(0.0, 0.54, 0, 0, 1, 0);
        t.list(
            55.454,
            0.0,
            0,
            0,
            2,
            &[
                20.0, 0.5, 0.002, 1.0, 0.0, 0.0, //
                22.0, 1.5, 0.003, 1.0, 0.0, 0.0,
            ],
        );
        let tri = [0.01, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.02, 0.05];
        t.list(0.0, 0.0, 2, 0, 4, &tri);

        let d = t.parse().unwrap();
        assert!(d.defects.is_empty());
        let ranges = &d.isotopes[0].ranges;
        match &ranges[0].covariance {
            Covariance::Compatible(c) => {
                assert_eq!(c.sections[0].nrs, 1);
                assert_eq!(c.sections[0].resonances[0], res);
            }
            other => panic!("expected LCOMP=0, got {other:?}"),
        }
        match &ranges[1].covariance {
            Covariance::Unresolved(u) => {
                assert_eq!(u.mpar, 2);
                assert_eq!(u.l_values[0].parameters[1][0], 22.0);
                assert_eq!(u.relative_covariance.order, 4);
                assert_eq!(u.relative_covariance.get(1, 1), 0.04);
                assert_eq!(u.relative_covariance.get(3, 2), 0.02);
                assert_eq!(u.relative_covariance.get(2, 3), 0.02);
                let diag: Vec<f64> = u.relative_covariance.diagonal().collect();
                assert_eq!(diag, [0.01, 0.04, 0.0, 0.05]);
            }
            other => panic!("expected an unresolved range, got {other:?}"),
        }
    }

    /// LCOMP=1 Breit-Wigner, with ISR's CONT record, and a negative variance
    /// that is kept and flagged.
    #[test]
    fn reads_general_breit_wigner_blocks_and_flags_a_negative_variance() {
        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 1, 0, 1);
        t.cont(0.0, 0.02, 0, 0, 0, 0);
        t.cont(55.454, 0.0, 0, 0, 1, 0);
        // MPAR=3, two resonances: order 6, 21 values.
        let mut v = vec![
            10.0, 0.5, 1.2, 1.0, 0.2, 0.0, //
            20.0, 1.5, 2.2, 2.0, 0.2, 0.0,
        ];
        let tri: Vec<f64> = (0..21).map(|i| i as f64).collect();
        v.extend(&tri);
        // Row 1 starts at 6, so its diagonal is value 6: make it negative.
        v[12 + 6] = -1000000.0;
        t.list(0.0, 0.0, 3, 0, 2, &v);

        let d = t.parse().unwrap();
        let Covariance::General(g) = only_range(&d) else {
            panic!("expected LCOMP=1");
        };
        assert_eq!(g.dap, Some(ScatteringRadiusUncertainty::Cont { dap: 0.02 }));
        let b = &g.blocks[0];
        assert_eq!((b.mpar, b.nrb), (3, 2));
        assert_eq!(b.resonances[1][0], 20.0);
        assert_eq!(b.covariance.order, 6);
        assert_eq!(b.covariance.get(0, 5), 5.0);
        assert_eq!(b.covariance.get(1, 1), -1000000.0);
        assert_eq!(b.covariance.get(5, 5), 20.0);
        assert_eq!(
            d.defects,
            [Defect::NegativeVariance {
                isotope: 0,
                range: 0,
                block: 0,
                index: 1,
                value: -1000000.0
            }]
        );
    }

    /// LCOMP=1 Reich-Moore with ISR's LIST record and more than one block,
    /// then a range with no blocks at all (NSRS=0, as FENDL writes).
    #[test]
    fn reads_general_reich_moore_with_several_blocks() {
        let mut t = Tape::head(2);
        t.cont(0.00001, 1000.0, 1, 3, 0, 0);
        t.cont(0.0, 0.54, 0, 1, 0, 1);
        t.list(0.0, 0.0, 0, 0, 1, &[0.01, 0.02, 0.03]);
        t.cont(55.454, 0.0, 0, 0, 3, 0);
        for k in 0..3 {
            let mut v = vec![10.0 * (k + 1) as f64, 0.5, 1.0, 0.03, 0.0, 0.0];
            v.extend([1.0, 0.1, 2.0]); // MPAR=2, one resonance.
            t.list(0.0, 0.0, 2, 0, 1, &v);
        }
        t.cont(1000.0, 2000.0, 1, 3, 0, 0);
        t.cont(0.0, 0.54, 0, 1, 0, 0);
        t.cont(55.454, 0.0, 0, 0, 0, 0);

        let d = t.parse().unwrap();
        let ranges = &d.isotopes[0].ranges;
        let Covariance::General(g) = &ranges[0].covariance else {
            panic!("expected LCOMP=1");
        };
        assert_eq!(
            g.dap,
            Some(ScatteringRadiusUncertainty::List {
                values: vec![0.01, 0.02, 0.03]
            })
        );
        assert_eq!(g.blocks.len(), 3);
        assert_eq!(g.blocks[2].resonances[0][0], 30.0);
        assert_eq!(g.blocks[2].covariance.get(0, 1), 0.1);
        let Covariance::General(g) = &ranges[1].covariance else {
            panic!("expected LCOMP=1");
        };
        assert_eq!((g.nsrs, g.blocks.len()), (0, 0));
    }

    /// LCOMP=1 R-matrix limited: two spin groups with different channel
    /// counts in one block.
    #[test]
    fn reads_general_r_matrix_blocks() {
        let mut t = Tape::head(1);
        t.cont(0.00001, 1500000.0, 1, 7, 0, 0);
        t.cont(0.0, 0.0, 0, 1, 0, 0);
        t.cont(0.0, 0.0, 0, 0, 1, 0);
        t.cont(0.0, 0.0, 2, 0, 0, 0);
        // NCH=3, two resonances, one line each.
        t.list(
            0.0,
            0.0,
            3,
            2,
            2,
            &[
                -458668.7, 1.0, 980.9, 0.0022, 0.0, 0.0, //
                88892.81, 0.8, 159.3, 0.001, 0.0, 0.0,
            ],
        );
        // NCH=6, one resonance: seven values, padded to two lines.
        let mut v = vec![132677.0, 2.6, 2510.3, 0.001, 1.0, 2.0, 3.0];
        v.resize(12, 0.0);
        t.list(0.0, 0.0, 6, 1, 2, &v);
        // NPARB = 4*2 + 7*1 = 15.
        let tri: Vec<f64> = (1..=120).map(|i| i as f64).collect();
        t.list(0.0, 0.0, 0, 0, 15, &tri);

        let d = t.parse().unwrap();
        let Covariance::GeneralRMatrix(g) = only_range(&d) else {
            panic!("expected LRF=7 LCOMP=1");
        };
        let b = &g.blocks[0];
        assert_eq!(b.njsx, 2);
        assert_eq!(b.spin_groups[0].resonance(1), [88892.81, 0.8, 159.3, 0.001]);
        assert_eq!(
            b.spin_groups[1].resonance(0),
            [132677.0, 2.6, 2510.3, 0.001, 1.0, 2.0, 3.0]
        );
        assert_eq!(b.covariance.order, 15);
        assert_eq!(b.covariance.get(14, 14), 120.0);
    }

    /// LCOMP=2 Breit-Wigner at NDIGIT=2, with a correlation line whose row
    /// lies beyond NNN, as some JEFF-4.0, TENDL-2025 and FENDL-3.2d files
    /// have: the line is kept, flagged, and left out of the expansion.
    #[test]
    fn reads_compact_breit_wigner_and_flags_a_row_outside_the_matrix() {
        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 2, 0, 1);
        t.cont(0.0, 0.03, 0, 0, 0, 0);
        t.list(
            55.454,
            0.0,
            0,
            0,
            2,
            &[
                10.0, 0.5, 1.2, 1.0, 0.2, 0.0, 0.001, 0.0, 0.0, 0.1, 0.02, 0.0, //
                20.0, 1.5, 2.2, 2.0, 0.2, 0.0, 0.002, 0.0, 0.0, 0.2, 0.02, 0.0,
            ],
        );
        // NNN=6 (three parameters per resonance), three lines.
        t.cont(0.0, 0.0, 2, 6, 3, 0);
        t.intg(2, 2, 1, &[87]);
        t.intg(2, 6, 3, &[-12, 0, 5, 9]);
        t.intg(2, 7, 1, &[50]);

        let d = t.parse().unwrap();
        let Covariance::Compact(c) = only_range(&d) else {
            panic!("expected LCOMP=2");
        };
        assert_eq!(c.dap, Some(ScatteringRadiusUncertainty::Cont { dap: 0.03 }));
        assert_eq!(c.nrsa, 2);
        assert_eq!(c.resonances[1].parameters[0], 20.0);
        assert_eq!(c.resonances[1].uncertainties[3], 0.2);
        let corr = &c.correlation;
        assert_eq!((corr.ndigit, corr.nnn, corr.rows.len()), (2, 6, 3));
        // Kept as written, the padding past the diagonal included.
        assert_eq!(corr.rows[1].kij[..4], [-12, 0, 5, 9]);
        assert_eq!(corr.rows[2].ii, 7);

        // Row 6 from column 3: -12 at (6, 3), 5 at (6, 5), and the 9 would be
        // (6, 6), the diagonal, which the format says to ignore.
        let e: Vec<(usize, usize, f64)> = corr.entries().collect();
        assert_eq!(e, [(1, 0, 0.875), (5, 2, -0.125), (5, 4, 0.055)]);

        assert_eq!(
            d.defects,
            [Defect::CorrelationRowOutsideMatrix {
                isotope: 0,
                range: 0,
                line: 2,
                ii: 7,
                jj: 1,
                nnn: 6
            }]
        );
    }

    /// LCOMP=2 Reich-Moore at every other NDIGIT the libraries use, with ISR's
    /// LIST record.
    #[test]
    fn reads_compact_reich_moore_at_each_ndigit() {
        for ndigit in [3, 4, 5, 6] {
            let mut t = Tape::head(1);
            t.cont(0.00001, 1000.0, 1, 3, 0, 0);
            t.cont(0.0, 0.54, 0, 2, 0, 1);
            t.list(0.0, 0.0, 0, 0, 1, &[0.005]);
            t.list(
                55.454,
                0.6,
                0,
                0,
                1,
                &[
                    10.0, 0.5, 1.0, 0.03, 0.1, 0.2, 0.001, 0.0, 0.1, 0.002, 0.01, 0.02,
                ],
            );
            // NNN=5: ER, GN, GG, GFA, GFB.
            t.cont(0.0, 0.0, ndigit, 5, 1, 0);
            let k = 10i64.pow(ndigit as u32) / 2;
            t.intg(ndigit, 5, 1, &[k, 0, 0, -k]);

            let d = t.parse().unwrap();
            let Covariance::Compact(c) = only_range(&d) else {
                panic!("expected LCOMP=2");
            };
            assert_eq!(c.apl, 0.6);
            assert_eq!(c.resonances[0].uncertainties[5], 0.02);
            let e: Vec<(usize, usize, f64)> = c.correlation.entries().collect();
            let half = 0.5 / 10f64.powi(ndigit as i32);
            assert_eq!(
                e,
                [(4, 0, 0.5 + half), (4, 3, -0.5 - half)],
                "NDIGIT={ndigit}"
            );
        }
    }

    /// LCOMP=2 R-matrix limited, with a charged-particle channel (MT=600) as
    /// Cl35 has, and spin groups whose channel counts differ.
    #[test]
    fn reads_compact_r_matrix_with_a_charged_channel() {
        let mut t = Tape::head(1);
        t.cont(0.00001, 1200000.0, 1, 7, 0, 0);
        t.cont(0.0, 0.0, 0, 2, 2, 0);
        t.list(
            0.0,
            0.0,
            3,
            2,
            6,
            &[
                0.0, 35.7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 102.0, 0.0, 0.0, //
                1.0, 34.7, 0.0, 17.0, 0.5, 1.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, //
                1.0, 34.7, 1.0, 16.0, 0.5, 1.5, 615220.0, 1.0, 0.0, 600.0, 0.0, 0.0,
            ],
        );
        // Spin group 1: three channels, one resonance per line pair.
        t.list(
            1.0,
            0.0,
            0,
            0,
            3,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                2.0, 0.0, 1.0, 0.0, 0.37, 0.48, //
                3.0, 0.0, 1.0, 0.0, 0.37, 0.48,
            ],
        );
        t.list(
            0.0,
            0.0,
            0,
            2,
            2,
            &[
                54932.0, 0.367, 46.4, 0.0, 0.0, 0.0, 2.58, 0.18, 6.1, 0.0, 0.0, 0.0, //
                68236.0, 0.393, 217.9, 0.00001, 0.0, 0.0, 5.04, 0.2, 22.8, 0.015, 0.0, 0.0,
            ],
        );
        // Spin group 2: two channels, one resonance.
        t.list(
            2.0,
            0.0,
            0,
            0,
            2,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                2.0, 1.0, 0.5, 0.0, 0.37, 0.48,
            ],
        );
        t.list(
            0.0,
            0.0,
            0,
            1,
            1,
            &[
                115098.0, 0.739, 4.3, 0.0, 0.0, 0.0, 14.8, 0.13, 1.7, 0.0, 0.0, 0.0,
            ],
        );
        // NNN = 2*4 + 1*3 = 11, at NDIGIT=3.
        t.cont(0.0, 0.0, 3, 11, 1, 0);
        t.intg(3, 11, 1, &[500, -500]);

        let d = t.parse().unwrap();
        assert!(d.defects.is_empty());
        let Covariance::CompactRMatrix(c) = only_range(&d) else {
            panic!("expected LRF=7 LCOMP=2");
        };
        assert_eq!((c.npp, c.njs, c.njsx), (3, 2, 2));
        assert_eq!(c.particle_pairs[2][9], 600.0);
        let g = &c.spin_groups[0];
        assert_eq!((g.nch, g.nrsa), (3, 2));
        assert_eq!(g.channels[1][4], 0.37);
        assert_eq!(g.parameters(1), [68236.0, 0.393, 217.9, 0.00001]);
        assert_eq!(g.uncertainties(1), [5.04, 0.2, 22.8, 0.015]);
        let g = &c.spin_groups[1];
        assert_eq!(g.parameters(0), [115098.0, 0.739, 4.3]);
        assert_eq!(g.uncertainties(0), [14.8, 0.13, 1.7]);
        let e: Vec<(usize, usize, f64)> = c.correlation.entries().collect();
        assert_eq!(e, [(10, 0, 0.5005), (10, 1, -0.5005)]);
    }

    /// Every MPAR the unresolved format allows.
    #[test]
    fn reads_unresolved_ranges_for_each_mpar() {
        for mpar in 1..=5i64 {
            let mut t = Tape::head(1);
            t.cont(1000.0, 100000.0, 2, 1, 0, 0);
            t.cont(0.0, 0.54, 0, 0, 1, 0);
            t.list(55.454, 0.0, 1, 0, 1, &[20.0, 0.5, 0.002, 1.0, 0.1, 0.3]);
            let n = mpar as usize;
            let tri: Vec<f64> = (0..n * (n + 1) / 2).map(|i| 0.01 * i as f64).collect();
            t.list(0.0, 0.0, mpar, 0, mpar, &tri);

            let d = t.parse().unwrap();
            let Covariance::Unresolved(u) = only_range(&d) else {
                panic!("expected an unresolved range");
            };
            assert_eq!(u.mpar, mpar);
            assert_eq!(u.l_values[0].l, 1);
            assert_eq!(u.relative_covariance.values, tri);
        }
    }

    /// A count the records disagree on is an error, not a guess.
    #[test]
    fn inconsistent_counts_are_an_error() {
        // MPAR=3 and NRB=1 need six covariance values; five are given.
        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 1, 0, 0);
        t.cont(55.454, 0.0, 0, 0, 1, 0);
        t.list(
            0.0,
            0.0,
            3,
            0,
            1,
            &[10.0, 0.5, 1.2, 1.0, 0.2, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        );
        assert!(matches!(t.parse(), Err(Error::Mismatched { .. })));
    }

    /// The formats no library uses are refused by name.
    #[test]
    fn formats_no_library_uses_are_refused() {
        let unsupported = |t: &Tape, expect: &str| match t.parse() {
            Err(Error::Unsupported { what }) => {
                assert!(
                    what.contains(expect),
                    "{what:?} does not mention {expect:?}"
                )
            }
            other => panic!("expected {expect} to be refused, got {other:?}"),
        };

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 1, 0);
        unsupported(&t, "NRO");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 1, 0, 0);
        t.cont(55.454, 0.0, 0, 0, 0, 1);
        unsupported(&t, "NLRS");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 7, 0, 0);
        t.cont(0.0, 0.0, 0, 1, 0, 0);
        t.cont(55.454, 0.0, 0, 0, 0, 1);
        unsupported(&t, "NLRS");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 1, 0, 0);
        unsupported(&t, "LRF=1");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 4, 0, 0);
        unsupported(&t, "LRF=4");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 2, 0, 0);
        t.cont(0.0, 0.54, 0, 0, 1, 1);
        unsupported(&t, "ISR>0");

        let mut t = Tape::head(1);
        t.cont(0.00001, 1000.0, 1, 3, 0, 0);
        t.cont(0.0, 0.54, 0, 0, 1, 0);
        unsupported(&t, "LCOMP=0");
    }

    /// Through the material: MF=32 MT=151 is its own section now, no longer
    /// left unparsed.
    #[test]
    fn a_material_dispatches_mf32() {
        let mut t = Tape::head(1);
        t.cont(1000.0, 100000.0, 2, 1, 0, 0);
        t.cont(0.0, 0.54, 0, 0, 1, 0);
        t.list(55.454, 0.0, 0, 0, 1, &[20.0, 0.5, 0.002, 1.0, 0.0, 0.0]);
        t.list(0.0, 0.0, 1, 0, 1, &[0.01]);
        let send = format!("{:<66}9999{:>2}{:>3}\n", "", 32, 0);
        let mend = format!("{:<66}{:>4}{:>2}{:>3}\n", "", 0, 0, 0);
        let text =
            format!("{:<66}{:>4}{:>2}{:>3}\n", " tape id", 1, 0, 0) + &t.text + &send + &mend;

        let m = Material::from_str(&text).unwrap();
        assert!(matches!(m.get(32, 151), Some(Section::Mf32(_))));
        let d = m.mf32().expect("MF=32 MT=151 is present");
        assert_eq!(d.za, 26056);
        assert_eq!(d.isotopes[0].ranges[0].lru, 2);
    }
}
