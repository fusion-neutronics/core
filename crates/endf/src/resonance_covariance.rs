//! The covariance of a range's resonance parameters, as one matrix over named
//! parameters.
//!
//! An unresolved range (LRU=2) gives the relative covariance of its average
//! parameters per `(l, J)`; those become unit-mean multipliers on MF=2's
//! parameter tables ([`Location::Unresolved`]), the parameters the range's
//! average cross sections come from. NJOY's ERRORR perturbs MF=32's own
//! energy-independent copies of them instead.
//!
//! MF=32 writes a resolved range's covariance in one of five layouts
//! ([`crate::mf::mf32`]), each listing the parameters its own way: per
//! resonance in fixed 4 by 4 blocks (LCOMP=0), in short-range blocks over a
//! subset of resonances (LCOMP=1), or as uncertainties and an integer-coded
//! correlation matrix (LCOMP=2). [`range_covariances`] reads every one into
//! the same form: a list of [`Parameter`]s, each tied to the MF=2 resonance and
//! quantity it perturbs, and their covariance as a dense matrix in the
//! parameters' own units (eV and eV², the scattering radius in 1e-12 cm). That
//! is what a reconstruction's derivatives are taken against.
//!
//! # How the compact correlation matrix counts its parameters
//!
//! For Breit-Wigner and Reich-Moore (LCOMP=2) the matrix has `MPAR`
//! parameters per resonance, `NNN = NRSA * MPAR`: ER, GN and GG, then GF (or
//! GFA and GFB) where the evaluation has fission, in that order, whether or
//! not a given uncertainty is zero. Reading it instead as "the parameters with
//! non-zero uncertainty", as OpenMC 0.15 does, misplaces every correlation
//! after the first zero: across ENDF/B-VIII.1 and JEFF-4.0 that is 4 and 174
//! compact ranges whose order is not their count of non-zero uncertainties,
//! where every one of the 558 has `NNN` a whole multiple of `NRSA`. For
//! R-matrix limited parameters it is ER and every channel width of every
//! resonance, spin group by spin group, zeros included (all 20 ranges).
//!
//! # Matching MF=32 to MF=2
//!
//! A covariance range need not span its MF=2 range exactly (ENDF/B-VIII.1
//! Si32 stops at 761.1 keV, its MF=2 range at 764.9 keV), so it is matched to
//! the MF=2 resolved range it overlaps most. A resonance is matched to the
//! nearest MF=2 resonance of the same spin within [`ENERGY_TOLERANCE`]: MF=32
//! repeats the MF=2 parameters, but not always to the digit (JEFF-4.0 Xe135
//! writes 0.085107 eV for MF=2's 0.0851068) or even from the same parameter
//! set (JEFF-4.0 U236 differs by up to 0.07%). Matches that are not exact are
//! counted in [`RangeCovariance::approximate`]. A resonance MF=2 does not
//! have at all (JEFF-4.0 Sm151 at -0.08 eV) is left out of the matrix, which
//! keeps the rest of it exact (a marginal of a covariance is its sub-block),
//! and listed in [`RangeCovariance::unmatched`]. R-matrix spin groups are
//! matched in order by spin, channel count and resonance count, so an MF=2
//! group MF=32
//! leaves out (ENDF/B-VIII.1 W183's fifth) is skipped.
//!
//! # Scattering radius
//!
//! Where ISR=1 a Breit-Wigner or Reich-Moore range's radius uncertainty is one
//! parameter, uncorrelated with the resonance parameters (the format gives no
//! such correlation), of unit variance: one standard deviation of it moves
//! every section's radius at once, each by its own step
//! ([`RangeCovariance::radius_steps`]). The steps follow NJOY's ERRORR, the
//! reference reading of the format: the LIST's first value, DAP, is AP's;
//! further values (MLS of them in all) are the first MLS-1 sections' own; a
//! section the list does not reach takes DAP even where it carries an APL of
//! its own, and a section without an APL moves with AP, by DAP. ENDF/B-VIII.1
//! Pb208 lists `[0.027, 0, 0.0027]` over four sections: its s-wave radius is
//! certain, its p-wave radius moves by 0.0027 and its d- and f-wave radii,
//! which repeat AP, by 0.027. For R-matrix limited ranges each channel of each
//! spin group has a radius parameter of its own.

use crate::error::{Error, Result};

/// How far, relative to the energy, an MF=32 resonance may sit from the MF=2
/// resonance it is matched to.
pub const ENERGY_TOLERANCE: f64 = 1e-3;
use crate::mf::mf2::{Mf2, ResonanceParameters, ResonanceRange};
use crate::mf::mf32::{Covariance, Mf32, PackedCovariance, Range, ScatteringRadiusUncertainty};

/// Where a parameter sits in MF=2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// Resonance `index` of the MF=2 section `section` (the section of one
    /// orbital angular momentum), for Breit-Wigner and Reich-Moore.
    Orbital { section: usize, index: usize },
    /// Resonance `index` of spin group `group`, for R-matrix limited.
    SpinGroup { group: usize, index: usize },
    /// The range as a whole: its radius parameter, which moves every
    /// section's radius by its step in [`RangeCovariance::radius_steps`].
    Range,
    /// The radius of channel `channel` of spin group `group`, for R-matrix
    /// limited.
    Channel { group: usize, channel: usize },
    /// The `spin`-th J of the `orbital`-th section of an unresolved range:
    /// its parameter is a unit-mean multiplier on the whole energy table.
    Unresolved { orbital: usize, spin: usize },
}

/// What a parameter is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantity {
    Energy,
    NeutronWidth,
    CaptureWidth,
    /// GF for Breit-Wigner, GFA for Reich-Moore.
    FissionWidth,
    /// GFB for Reich-Moore.
    SecondFissionWidth,
    /// GX, the competitive width, for Breit-Wigner: GT less the others; for
    /// an unresolved range, its average competitive width.
    CompetitiveWidth,
    /// An unresolved range's average level spacing D.
    LevelSpacing,
    /// An unresolved range's average reduced neutron width GN0.
    ReducedNeutronWidth,
    /// The width (or, with IFG=1, the reduced width amplitude) of channel
    /// `c` of an R-matrix limited spin group.
    ChannelWidth(usize),
    ScatteringRadius,
}

/// One parameter of the covariance, and its value in MF=32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameter {
    pub location: Location,
    pub quantity: Quantity,
    pub value: f64,
}

/// One resolved range's parameter covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeCovariance {
    /// Index of the isotope in MF=2 and MF=32, of the range within MF=32's
    /// isotope, and of the MF=2 range it was matched to.
    pub isotope: usize,
    pub range: usize,
    pub mf2_range: usize,
    pub el: f64,
    pub eh: f64,
    /// 1 resolved, 2 unresolved.
    pub lru: i64,
    /// The MF=2 representation: for a resolved range 1 or 2 Breit-Wigner, 3
    /// Reich-Moore, 7 R-matrix limited.
    pub lrf: i64,
    pub parameters: Vec<Parameter>,
    /// Row-major `n × n`, `n` the number of parameters.
    pub covariance: Vec<f64>,
    /// Resonances matched to an MF=2 resonance whose energy differs, within
    /// [`ENERGY_TOLERANCE`].
    pub approximate: usize,
    /// Energies of MF=32 resonances MF=2 does not list, left out of the
    /// matrix. For an unresolved range, the J of each MF=32 spin left out.
    pub unmatched: Vec<f64>,
    /// Per MF=2 section, how far one standard deviation of the range's radius
    /// parameter moves the section's radius (1e-12 cm). Empty where the
    /// range has no radius uncertainty.
    pub radius_steps: Vec<f64>,
}

impl RangeCovariance {
    /// Parameters in the matrix.
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Element `(i, j)` of the covariance.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.covariance[i * self.len() + j]
    }
}

/// Every resolved range's parameter covariance, isotope by isotope, each
/// matched to its MF=2 range. Unresolved ranges are left out.
///
/// Errors where MF=32 does not match MF=2 (a resonance or spin group MF=2 does
/// not have, or ranges that do not line up), or where a matrix's order is not
/// what its parameters need.
pub fn range_covariances(mf2: &Mf2, mf32: &Mf32) -> Result<Vec<RangeCovariance>> {
    let mut out = Vec::new();
    for (i, isotope) in mf32.isotopes.iter().enumerate() {
        let Some(parameters) = mf2.isotopes.get(i) else {
            return Err(Error::Mismatched {
                what: "MF=32 and MF=2 isotopes",
            });
        };
        for (r, range) in isotope.ranges.iter().enumerate() {
            if range.lru != 1 && range.lru != 2 {
                continue;
            }
            let overlap = |el: f64, eh: f64| (eh.min(range.eh) - el.max(range.el)).max(0.0);
            let (mf2_range, resonances) = parameters
                .ranges
                .iter()
                .enumerate()
                .filter(|(_, p)| p.lru == range.lru && overlap(p.el, p.eh) > 0.0)
                .max_by(|a, b| overlap(a.1.el, a.1.eh).total_cmp(&overlap(b.1.el, b.1.eh)))
                .ok_or(Error::Mismatched {
                    what: "an MF=32 range and the MF=2 ranges",
                })?;
            let mut builder = Builder::default();
            if range.lru == 2 {
                read_unresolved(range, &resonances.parameters, isotope.lfw, &mut builder)?;
            } else {
                read_range(range, &resonances.parameters, &mut builder)?;
            }
            let covariance = builder.dense();
            out.push(RangeCovariance {
                isotope: i,
                range: r,
                mf2_range,
                el: range.el,
                eh: range.eh,
                lru: range.lru,
                lrf: range.lrf,
                parameters: builder.parameters,
                covariance,
                approximate: builder.approximate,
                unmatched: builder.unmatched,
                radius_steps: builder.radius_steps,
            });
        }
    }
    Ok(out)
}

/// Equal to the precision ENDF writes numbers to.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1e-30)
}

/// Parameters gathered in order, with their covariance as sparse blocks.
#[derive(Default)]
struct Builder {
    parameters: Vec<Parameter>,
    entries: Vec<(usize, usize, f64)>,
    approximate: usize,
    unmatched: Vec<f64>,
    radius_steps: Vec<f64>,
    /// The MF=2 resonances already given to an MF=32 one.
    claimed: std::collections::HashSet<Location>,
}

impl Builder {
    fn push(&mut self, location: Location, quantity: Quantity, value: f64) -> usize {
        self.parameters.push(Parameter {
            location,
            quantity,
            value,
        });
        self.parameters.len() - 1
    }

    /// [`Self::push`] where the resonance was matched, nothing otherwise: the
    /// row a matrix entry for this parameter lands in.
    fn row(&mut self, location: Option<Location>, quantity: Quantity, value: f64) -> Option<usize> {
        location.map(|l| self.push(l, quantity, value))
    }

    /// The MF=2 location of a resonance from [`orbital`], counting an
    /// approximate match and listing a missing one. Each MF=2 resonance is
    /// given to one MF=32 resonance at most: a second whose nearest match is
    /// already taken is listed as missing rather than doubling the first's
    /// parameters in the matrix.
    fn matched(&mut self, found: Option<(Location, bool)>, er: f64) -> Option<Location> {
        match found {
            Some((location, exact)) if self.claimed.insert(location) => {
                if !exact {
                    self.approximate += 1;
                }
                Some(location)
            }
            _ => {
                self.unmatched.push(er);
                None
            }
        }
    }

    /// `C_ij` (and `C_ji`).
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.entries.push((i, j, value));
        if i != j {
            self.entries.push((j, i, value));
        }
    }

    /// A packed block over the parameters `rows`, in order; a row of `None`
    /// is a parameter left out.
    fn packed(&mut self, rows: &[Option<usize>], packed: &PackedCovariance) -> Result<()> {
        if packed.order != rows.len() {
            return Err(Error::Mismatched {
                what: "an MF=32 covariance block and its parameters",
            });
        }
        for (a, i) in rows.iter().enumerate() {
            for (b, j) in rows.iter().enumerate().skip(a) {
                if let (Some(i), Some(j)) = (i, j) {
                    self.set(*i, *j, packed.get(a, b));
                }
            }
        }
        Ok(())
    }

    /// Uncertainties `u` over the parameters `rows` and a compact correlation
    /// matrix over them: `C_ij = ρ_ij u_i u_j`.
    fn compact(
        &mut self,
        rows: &[Option<usize>],
        u: &[f64],
        correlation: &crate::mf::mf32::CompactCorrelation,
    ) -> Result<()> {
        if correlation.nnn as usize != rows.len() {
            return Err(Error::Mismatched {
                what: "an MF=32 compact correlation matrix and its parameters",
            });
        }
        for (a, i) in rows.iter().enumerate() {
            if let Some(i) = i {
                self.set(*i, *i, u[a] * u[a]);
            }
        }
        for (a, b, rho) in correlation.entries() {
            if let (Some(i), Some(j)) = (rows[a], rows[b]) {
                self.set(i, j, rho * u[a] * u[b]);
            }
        }
        Ok(())
    }

    fn dense(&self) -> Vec<f64> {
        let n = self.parameters.len();
        let mut out = vec![0.0; n * n];
        for &(i, j, v) in &self.entries {
            out[i * n + j] = v;
        }
        out
    }
}

/// How well an MF=2 candidate fits an MF=32 resonance, best first: nearest in
/// energy; among equally near ones, one not yet claimed; then the closest
/// neutron and capture widths. MF=2 can hold resonances of one spin at the
/// same energy (in different orbital sections, or adjacent in one:
/// ENDF/B-VIII.1 Pb208 has three such pairs at 8 MeV, Ne22 one at -1.47 MeV),
/// and only the widths tell them apart.
fn fit(distance: f64, claimed: bool, widths: (f64, f64), mf32: (f64, f64)) -> (f64, bool, f64) {
    let off = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-30);
    (
        distance,
        claimed,
        off(widths.0, mf32.0) + off(widths.1, mf32.1),
    )
}

/// Whether fit `a` is strictly better than `b` (see [`fit`]).
fn better(a: (f64, bool, f64), b: (f64, bool, f64)) -> bool {
    a.0.total_cmp(&b.0)
        .then(a.1.cmp(&b.1))
        .then(a.2.total_cmp(&b.2))
        .is_lt()
}

/// The MF=2 section and index of the Breit-Wigner or Reich-Moore resonance
/// that best fits one MF=32 lists at `er` with spin `aj` and neutron and
/// capture widths `mf32` (see [`fit`]), within [`ENERGY_TOLERANCE`], and
/// whether the energies agree to the digits ENDF writes. `None` where MF=2 has
/// none.
fn orbital(
    parameters: &ResonanceParameters,
    er: f64,
    aj: f64,
    mf32: (f64, f64),
    claimed: &std::collections::HashSet<Location>,
) -> Result<Option<(Location, bool)>> {
    type Columns<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [f64]);
    let sections: Vec<Columns> = match parameters {
        ResonanceParameters::BreitWigner(bw) => bw
            .sections
            .iter()
            .map(|s| {
                (
                    s.er.as_slice(),
                    s.aj.as_slice(),
                    s.gn.as_slice(),
                    s.gg.as_slice(),
                )
            })
            .collect(),
        ResonanceParameters::ReichMoore(rm) => rm
            .sections
            .iter()
            .map(|s| {
                (
                    s.er.as_slice(),
                    s.aj.as_slice(),
                    s.gn.as_slice(),
                    s.gg.as_slice(),
                )
            })
            .collect(),
        _ => {
            return Err(Error::Mismatched {
                what: "MF=32 Breit-Wigner or Reich-Moore parameters and the MF=2 range",
            })
        }
    };
    let mut best: Option<((f64, bool, f64), Location)> = None;
    for (section, (energies, spins, gn, gg)) in sections.iter().enumerate() {
        for (index, (&e, &j)) in energies.iter().zip(spins.iter()).enumerate() {
            if !close(j.abs(), aj.abs()) {
                continue;
            }
            let location = Location::Orbital { section, index };
            let widths = (
                gn.get(index).copied().unwrap_or(0.0),
                gg.get(index).copied().unwrap_or(0.0),
            );
            let f = fit((e - er).abs(), claimed.contains(&location), widths, mf32);
            if best.is_none_or(|(b, _)| better(f, b)) {
                best = Some((f, location));
            }
        }
    }
    Ok(best
        .filter(|((d, _, _), _)| *d <= ENERGY_TOLERANCE * er.abs().max(1e-30))
        .map(|((d, _, _), l)| (l, d <= 1e-6 * er.abs().max(1e-30))))
}

/// Each Breit-Wigner or Reich-Moore section's APL (zero where it has none:
/// Breit-Wigner sections never do).
fn section_radii(parameters: &ResonanceParameters) -> Vec<f64> {
    match parameters {
        ResonanceParameters::ReichMoore(rm) => rm.sections.iter().map(|s| s.apl).collect(),
        ResonanceParameters::BreitWigner(bw) => vec![0.0; bw.sections.len()],
        _ => Vec::new(),
    }
}

/// The radius parameter ISR=1 adds, and each section's step under it (see
/// the module documentation).
fn radius(builder: &mut Builder, dap: &Option<ScatteringRadiusUncertainty>, apl: &[f64]) {
    let values: Vec<f64> = match dap {
        None => return,
        Some(ScatteringRadiusUncertainty::Cont { dap }) => vec![*dap],
        Some(ScatteringRadiusUncertainty::List { values }) => values.clone(),
    };
    let Some(&global) = values.first() else {
        return;
    };
    let nls = apl.len();
    let mls = values.len();
    let own = |i: usize| -> f64 {
        if mls > 1 && i + 1 < mls {
            values[i + 1]
        } else {
            global
        }
    };
    let steps: Vec<f64> = (0..nls)
        .map(|i| if apl[i] == 0.0 { global } else { own(i) })
        .collect();
    if steps.iter().all(|s| *s == 0.0) {
        return;
    }
    let p = builder.push(Location::Range, Quantity::ScatteringRadius, 0.0);
    builder.set(p, p, 1.0);
    builder.radius_steps = steps;
}

/// The width quantities of a Breit-Wigner or Reich-Moore resonance, in the
/// order MF=32 lists them after ER.
fn widths(lrf: i64) -> [Quantity; 4] {
    let last = if lrf == 3 {
        Quantity::SecondFissionWidth
    } else {
        Quantity::CompetitiveWidth
    };
    [
        Quantity::NeutronWidth,
        Quantity::CaptureWidth,
        Quantity::FissionWidth,
        last,
    ]
}

fn read_range(range: &Range, parameters: &ResonanceParameters, b: &mut Builder) -> Result<()> {
    match &range.covariance {
        Covariance::Compatible(c) => {
            // Per resonance: ER, AJ, GT, GN, GG, GF, then DE2, DN2, DNDG,
            // DG2, DNDF, DGDF, DF2. ER is uncorrelated with the widths, and
            // no resonance with another.
            for s in &c.sections {
                for res in &s.resonances {
                    let found = orbital(parameters, res[0], res[1], (res[3], res[4]), &b.claimed)?;
                    let Some(location) = b.matched(found, res[0]) else {
                        continue;
                    };
                    let e = b.push(location, Quantity::Energy, res[0]);
                    let n = b.push(location, Quantity::NeutronWidth, res[3]);
                    let g = b.push(location, Quantity::CaptureWidth, res[4]);
                    let f = b.push(location, Quantity::FissionWidth, res[5]);
                    b.set(e, e, res[6]);
                    b.set(n, n, res[7]);
                    b.set(n, g, res[8]);
                    b.set(g, g, res[9]);
                    b.set(n, f, res[10]);
                    b.set(g, f, res[11]);
                    b.set(f, f, res[12]);
                }
            }
            Ok(())
        }
        Covariance::General(c) => {
            radius(b, &c.dap, &section_radii(parameters));
            let names = widths(range.lrf);
            for block in &c.blocks {
                let mpar = block.mpar as usize;
                let mut rows = Vec::with_capacity(mpar * block.resonances.len());
                for res in &block.resonances {
                    // ER, AJ, then GT, GN, GG, GF for Breit-Wigner and GN, GG,
                    // GFA, GFB for Reich-Moore.
                    let gn_gg = if range.lrf == 3 {
                        (res[2], res[3])
                    } else {
                        (res[3], res[4])
                    };
                    let found = orbital(parameters, res[0], res[1], gn_gg, &b.claimed)?;
                    let location = b.matched(found, res[0]);
                    let values = if range.lrf == 3 {
                        [res[2], res[3], res[4], res[5]]
                    } else {
                        [res[3], res[4], res[5], res[2] - res[3] - res[4] - res[5]]
                    };
                    rows.push(b.row(location, Quantity::Energy, res[0]));
                    for k in 0..mpar.saturating_sub(1) {
                        rows.push(b.row(location, names[k], values[k]));
                    }
                }
                b.packed(&rows, &block.covariance)?;
            }
            Ok(())
        }
        Covariance::Compact(c) => {
            radius(b, &c.dap, &section_radii(parameters));
            let nrsa = c.resonances.len();
            let nnn = c.correlation.nnn as usize;
            if nrsa == 0 || !nnn.is_multiple_of(nrsa) {
                return Err(Error::Mismatched {
                    what: "an MF=32 compact correlation order and its resonance count",
                });
            }
            let mpar = nnn / nrsa;
            if !(1..=5).contains(&mpar) {
                return Err(Error::Mismatched {
                    what: "an MF=32 compact correlation's parameters per resonance",
                });
            }
            let names = widths(range.lrf);
            let mut rows = Vec::with_capacity(nnn);
            let mut u = Vec::with_capacity(nnn);
            for res in &c.resonances {
                // Parameters ER, AJ, GN, GG, GFA, GFB (Reich-Moore) or ER, AJ,
                // GT, GN, GG, GF (Breit-Wigner), uncertainties alongside.
                let (p, d) = (&res.parameters, &res.uncertainties);
                let gn_gg = if range.lrf == 3 {
                    (p[2], p[3])
                } else {
                    (p[3], p[4])
                };
                let found = orbital(parameters, p[0], p[1], gn_gg, &b.claimed)?;
                let location = b.matched(found, p[0]);
                let order: [(f64, f64); 4] = if range.lrf == 3 {
                    [(p[2], d[2]), (p[3], d[3]), (p[4], d[4]), (p[5], d[5])]
                } else {
                    [
                        (p[3], d[3]),
                        (p[4], d[4]),
                        (p[5], d[5]),
                        (p[2] - p[3] - p[4] - p[5], 0.0),
                    ]
                };
                rows.push(b.row(location, Quantity::Energy, p[0]));
                u.push(d[0]);
                for k in 0..mpar - 1 {
                    rows.push(b.row(location, names[k], order[k].0));
                    u.push(order[k].1);
                }
            }
            b.compact(&rows, &u, &c.correlation)
        }
        Covariance::GeneralRMatrix(c) => {
            let ResonanceParameters::RMatrixLimited(rml) = parameters else {
                return Err(Error::Mismatched {
                    what: "MF=32 R-matrix limited parameters and the MF=2 range",
                });
            };
            channel_radii(b, &c.dap, rml);
            for block in &c.blocks {
                let mut rows = Vec::new();
                for sg in &block.spin_groups {
                    for k in 0..sg.nrb as usize {
                        let values = sg.resonance(k);
                        let found = spin_group_resonance(rml, sg.nch, values[0], &b.claimed);
                        let location = b.matched(found, values[0]);
                        rows.push(b.row(location, Quantity::Energy, values[0]));
                        for c in 0..sg.nch as usize {
                            rows.push(b.row(location, Quantity::ChannelWidth(c), values[1 + c]));
                        }
                    }
                }
                b.packed(&rows, &block.covariance)?;
            }
            Ok(())
        }
        Covariance::CompactRMatrix(c) => {
            let ResonanceParameters::RMatrixLimited(rml) = parameters else {
                return Err(Error::Mismatched {
                    what: "MF=32 R-matrix limited parameters and the MF=2 range",
                });
            };
            channel_radii(b, &c.dap, rml);
            let mut rows = Vec::new();
            let mut u = Vec::new();
            // Each MF=32 spin group is the next MF=2 group of its spin,
            // channel count and resonance count, so a group MF=32 leaves out
            // is skipped.
            let mut next = 0;
            for sg in &c.spin_groups {
                let group = (next..rml.spin_groups.len())
                    .find(|&g| {
                        let m = &rml.spin_groups[g];
                        m.nch == sg.nch && m.aj == sg.aj && m.er.len() == sg.nrsa as usize
                    })
                    .ok_or(Error::Mismatched {
                        what: "an MF=32 spin group and the MF=2 spin groups",
                    })?;
                next = group + 1;
                let mf2_group = &rml.spin_groups[group];
                for index in 0..sg.nrsa as usize {
                    let (p, d) = (sg.parameters(index), sg.uncertainties(index));
                    let e = mf2_group.er[index];
                    let location = if (p[0] - e).abs() <= ENERGY_TOLERANCE * e.abs().max(1e-30) {
                        if !close(p[0], e) {
                            b.approximate += 1;
                        }
                        Some(Location::SpinGroup { group, index })
                    } else {
                        b.unmatched.push(p[0]);
                        None
                    };
                    rows.push(b.row(location, Quantity::Energy, p[0]));
                    u.push(d[0]);
                    for ch in 0..sg.nch as usize {
                        rows.push(b.row(location, Quantity::ChannelWidth(ch), p[1 + ch]));
                        u.push(d[1 + ch]);
                    }
                }
            }
            b.compact(&rows, &u, &c.correlation)
        }
        Covariance::Unresolved(_) => Ok(()),
    }
}

/// An unresolved range's parameter covariance: per `(l, J)`, in MF=32's
/// listing order, unit-mean multipliers on its parameter tables (D, GN0, GG,
/// GF, GX, the first MPAR of them; with LFW=0 and MPAR=4 the four are D,
/// GN0, GG and GX), carrying MF=32's relative covariance. Sections are
/// matched to MF=2's by `l`, spins by `J`.
fn read_unresolved(
    range: &Range,
    parameters: &ResonanceParameters,
    lfw: i64,
    b: &mut Builder,
) -> Result<()> {
    let Covariance::Unresolved(u) = &range.covariance else {
        return Err(Error::Mismatched {
            what: "an MF=32 unresolved range without its covariance",
        });
    };
    let ResonanceParameters::Unresolved(mf2) = parameters else {
        return Err(Error::Mismatched {
            what: "MF=32 unresolved parameters and the MF=2 range",
        });
    };
    let mpar = u.mpar.max(0) as usize;
    let names: [Quantity; 5] = if lfw == 0 && mpar == 4 {
        [
            Quantity::LevelSpacing,
            Quantity::ReducedNeutronWidth,
            Quantity::CaptureWidth,
            Quantity::CompetitiveWidth,
            Quantity::FissionWidth,
        ]
    } else {
        [
            Quantity::LevelSpacing,
            Quantity::ReducedNeutronWidth,
            Quantity::CaptureWidth,
            Quantity::FissionWidth,
            Quantity::CompetitiveWidth,
        ]
    };
    let spin_of = |p: &crate::mf::mf2::UnresolvedParameters| match p {
        crate::mf::mf2::UnresolvedParameters::CaseB { aj, .. } => *aj,
        crate::mf::mf2::UnresolvedParameters::CaseC { aj, .. } => *aj,
    };
    let mut rows = Vec::new();
    for lv in &u.l_values {
        // An l MF=2 does not have (FENDL-3.2d La138 gives l=2, MF=2 only 0
        // and 1) moves no cross section: its spins are left out of the
        // matrix and listed as unmatched, as a missing resonance is.
        let Some(orbital) = mf2.ranges.iter().position(|r| r.l == lv.l) else {
            for par in &lv.parameters {
                b.unmatched.push(par[1]);
                for _ in 0..mpar {
                    rows.push(None);
                }
            }
            continue;
        };
        let section = &mf2.ranges[orbital];
        let spins: Vec<f64> = if section.parameters.is_empty() {
            section.aj.clone()
        } else {
            section.parameters.iter().map(spin_of).collect()
        };
        // By J where every J agrees; JEFF-4.0 writes many unresolved MF=32
        // sections whose J column does not (Ni66: 5, 1, 2 for MF=2's 0.5,
        // 1.5, 2.5), and there by position within the section, the order the
        // format lists them in, where the counts agree. Decided for the whole
        // section: a J that matches beside one that does not could otherwise
        // send two MF=32 spins to one MF=2 spin. Where neither holds, each J
        // that matches takes its MF=2 spin once and the rest are unmatched.
        let by_j: Vec<Option<usize>> = lv
            .parameters
            .iter()
            .map(|par| spins.iter().position(|&j| close(j, par[1])))
            .collect();
        let mut seen = std::collections::HashSet::new();
        let unique = by_j.iter().all(|m| m.is_some_and(|s| seen.insert(s)));
        let matched: Vec<Option<usize>> = if unique {
            by_j
        } else if lv.parameters.len() == spins.len() {
            b.approximate += lv
                .parameters
                .iter()
                .zip(&spins)
                .filter(|(par, &j)| !close(j, par[1]))
                .count();
            (0..spins.len()).map(Some).collect()
        } else {
            let mut taken = std::collections::HashSet::new();
            by_j.into_iter()
                .map(|m| m.filter(|&s| taken.insert(s)))
                .collect()
        };
        for (par, spin) in lv.parameters.iter().zip(matched) {
            let aj = par[1];
            let location = spin.map(|spin| Location::Unresolved { orbital, spin });
            if location.is_none() {
                b.unmatched.push(aj);
            }
            for name in names.iter().take(mpar) {
                rows.push(b.row(location, *name, 1.0));
            }
        }
    }
    b.packed(&rows, &u.relative_covariance)
}

/// The spin group of `rml` with `nch` channels holding the resonance nearest
/// `er`, within [`ENERGY_TOLERANCE`], preferring among equally near ones one
/// not yet claimed, its index there, and whether the energies agree to the
/// digit.
fn spin_group_resonance(
    rml: &crate::mf::mf2::RMatrixLimited,
    nch: i64,
    er: f64,
    claimed: &std::collections::HashSet<Location>,
) -> Option<(Location, bool)> {
    let mut best: Option<((f64, bool, f64), Location)> = None;
    for (group, sg) in rml.spin_groups.iter().enumerate() {
        if sg.nch != nch {
            continue;
        }
        for (index, &e) in sg.er.iter().enumerate() {
            let location = Location::SpinGroup { group, index };
            let f = ((e - er).abs(), claimed.contains(&location), 0.0);
            if best.is_none_or(|(b, _)| better(f, b)) {
                best = Some((f, location));
            }
        }
    }
    best.filter(|((d, _, _), _)| *d <= ENERGY_TOLERANCE * er.abs().max(1e-30))
        .map(|((d, _, _), l)| (l, d <= 1e-6 * er.abs().max(1e-30)))
}

/// The channel radius parameters ISR=1 adds to an R-matrix limited range:
/// one DAP per channel of each spin group, in order.
fn channel_radii(
    b: &mut Builder,
    dap: &Option<ScatteringRadiusUncertainty>,
    rml: &crate::mf::mf2::RMatrixLimited,
) {
    let Some(ScatteringRadiusUncertainty::List { values }) = dap else {
        return;
    };
    let mut k = 0;
    for (group, sg) in rml.spin_groups.iter().enumerate() {
        for channel in 0..sg.nch as usize {
            let Some(&d) = values.get(k) else {
                return;
            };
            k += 1;
            if d == 0.0 {
                continue;
            }
            let radius = sg.channels.apt.get(channel).copied().unwrap_or(0.0);
            let p = b.push(
                Location::Channel { group, channel },
                Quantity::ScatteringRadius,
                radius,
            );
            b.set(p, p, d * d);
        }
    }
}

/// `range`, the MF=2 range `cov` was matched to, with `cov`'s parameters at
/// `values` (one per parameter, in order) instead of their nominal values:
/// what a sampled parameter vector is rebuilt into cross sections from
/// ([`reconstruction_at`]).
///
/// Each value is read against its parameter's [`Parameter::value`], so the
/// nominal vector gives back `range` exactly, field for field, wherever MF=32
/// repeats MF=2 to fewer digits or matched a resonance only approximately:
///
/// - a resonance energy or width, or an R-matrix channel radius, moves its
///   MF=2 value by `values[i] - value` (eV, or 1e-12 cm). A Breit-Wigner
///   width moves GT with it, so the competitive width (GT less the others)
///   stays put, and the competitive width itself moves GT alone. A channel
///   radius moves the radii its penetrability and phase are taken at
///   together, as [`crate::resonance::RMatrixRange::radius_derivative`]
///   differentiates it.
/// - the radius parameter of a Breit-Wigner or Reich-Moore range (nominally
///   0, unit variance) moves each section's radius by `values[i]` times its
///   step in [`RangeCovariance::radius_steps`]: a Reich-Moore section with an
///   APL of its own through it, one without from AP, and a Breit-Wigner
///   range's AP by its one step, as the reconstructions' `radius_derivative`
///   take it.
/// - an unresolved `(l, J)` multiplier (nominally 1) scales its whole
///   parameter table (every energy's value) by `values[i] / value`. A width
///   the table does not have (the fission width of case A, the competitive
///   width of cases A and B) is zero in the average cross sections, and stays
///   zero.
///
/// Errors where `values` is not as long as `cov`'s parameters, a value is not
/// finite, or a parameter has no place in `range`: a location the range does
/// not have, or a quantity its formalism has no field for (a second fission
/// width in Breit-Wigner, a competitive width in Reich-Moore, a channel width
/// outside R-matrix limited). [`range_covariances`] emits neither of the last
/// two, so a refusal means the covariance is not `range`'s.
pub fn with_parameters(
    range: &ResonanceRange,
    cov: &RangeCovariance,
    values: &[f64],
) -> Result<ResonanceRange> {
    if values.len() != cov.len() {
        return Err(Error::Mismatched {
            what: "a parameter vector's length and its covariance's parameter count",
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Mismatched {
            what: "a parameter vector, whose values must be finite,",
        });
    }
    let misplaced = || Error::Mismatched {
        what: "a covariance parameter and the MF=2 range it is placed in",
    };
    let mut out = range.clone();
    for (p, &v) in cov.parameters.iter().zip(values) {
        if v == p.value {
            continue;
        }
        let shift = v - p.value;
        match (&mut out.parameters, p.location) {
            (ResonanceParameters::ReichMoore(rm), Location::Orbital { section, index }) => {
                let s = rm.sections.get_mut(section).ok_or_else(misplaced)?;
                let column = match p.quantity {
                    Quantity::Energy => &mut s.er,
                    Quantity::NeutronWidth => &mut s.gn,
                    Quantity::CaptureWidth => &mut s.gg,
                    Quantity::FissionWidth => &mut s.gfa,
                    Quantity::SecondFissionWidth => &mut s.gfb,
                    _ => return Err(misplaced()),
                };
                *column.get_mut(index).ok_or_else(misplaced)? += shift;
            }
            (ResonanceParameters::BreitWigner(bw), Location::Orbital { section, index }) => {
                let s = bw.sections.get_mut(section).ok_or_else(misplaced)?;
                let (column, total) = match p.quantity {
                    Quantity::Energy => (Some(&mut s.er), false),
                    Quantity::NeutronWidth => (Some(&mut s.gn), true),
                    Quantity::CaptureWidth => (Some(&mut s.gg), true),
                    Quantity::FissionWidth => (Some(&mut s.gf), true),
                    Quantity::CompetitiveWidth => (None, true),
                    _ => return Err(misplaced()),
                };
                if let Some(column) = column {
                    *column.get_mut(index).ok_or_else(misplaced)? += shift;
                }
                if total {
                    *s.gt.get_mut(index).ok_or_else(misplaced)? += shift;
                }
            }
            (ResonanceParameters::ReichMoore(rm), Location::Range)
                if p.quantity == Quantity::ScatteringRadius =>
            {
                let ap = rm.ap;
                for (s, step) in rm.sections.iter_mut().zip(&cov.radius_steps) {
                    let base = if s.apl != 0.0 { s.apl } else { ap };
                    s.apl = base + shift * step;
                }
            }
            (ResonanceParameters::BreitWigner(bw), Location::Range)
                if p.quantity == Quantity::ScatteringRadius =>
            {
                bw.ap += shift * cov.radius_steps.first().copied().unwrap_or(0.0);
            }
            (ResonanceParameters::RMatrixLimited(rml), Location::SpinGroup { group, index }) => {
                let sg = rml.spin_groups.get_mut(group).ok_or_else(misplaced)?;
                let column = match p.quantity {
                    Quantity::Energy => &mut sg.er,
                    Quantity::ChannelWidth(c) => sg.gam.get_mut(c).ok_or_else(misplaced)?,
                    _ => return Err(misplaced()),
                };
                *column.get_mut(index).ok_or_else(misplaced)? += shift;
            }
            (ResonanceParameters::RMatrixLimited(rml), Location::Channel { group, channel })
                if p.quantity == Quantity::ScatteringRadius =>
            {
                let sg = rml.spin_groups.get_mut(group).ok_or_else(misplaced)?;
                if !crate::resonance::shift_channel_radius(&mut sg.channels, channel, shift) {
                    return Err(misplaced());
                }
            }
            (ResonanceParameters::Unresolved(u), Location::Unresolved { orbital, spin }) => {
                if p.value == 0.0 {
                    return Err(misplaced());
                }
                let factor = v / p.value;
                let section = u.ranges.get_mut(orbital).ok_or_else(misplaced)?;
                scale_unresolved(section, spin, p.quantity, factor).ok_or_else(misplaced)?;
            }
            _ => return Err(misplaced()),
        }
    }
    Ok(out)
}

/// Scale the `quantity` column of spin `spin` of an unresolved section by
/// `factor`, at every energy: `None` where the section has no such spin or
/// the quantity is not an unresolved parameter. A width the case does not
/// tabulate is zero and is left so.
fn scale_unresolved(
    section: &mut crate::mf::mf2::UnresolvedRange,
    spin: usize,
    quantity: Quantity,
    factor: f64,
) -> Option<()> {
    use crate::mf::mf2::UnresolvedParameters;
    let scale = |x: &mut f64| *x *= factor;
    if section.parameters.is_empty() {
        // Case A: D, GN0 and GG per J, no fission or competition.
        let column = match quantity {
            Quantity::LevelSpacing => &mut section.d,
            Quantity::ReducedNeutronWidth => &mut section.gno,
            Quantity::CaptureWidth => &mut section.gg,
            Quantity::FissionWidth | Quantity::CompetitiveWidth => {
                return (spin < section.aj.len()).then_some(())
            }
            _ => return None,
        };
        scale(column.get_mut(spin)?);
        return Some(());
    }
    match section.parameters.get_mut(spin)? {
        UnresolvedParameters::CaseB { d, gn0, gg, gf, .. } => match quantity {
            Quantity::LevelSpacing => scale(d),
            Quantity::ReducedNeutronWidth => scale(gn0),
            Quantity::CaptureWidth => scale(gg),
            Quantity::FissionWidth => gf.iter_mut().for_each(scale),
            Quantity::CompetitiveWidth => {}
            _ => return None,
        },
        UnresolvedParameters::CaseC {
            d, gx, gn0, gg, gf, ..
        } => {
            let column = match quantity {
                Quantity::LevelSpacing => d,
                Quantity::ReducedNeutronWidth => gn0,
                Quantity::CaptureWidth => gg,
                Quantity::FissionWidth => gf,
                Quantity::CompetitiveWidth => gx,
                _ => return None,
            };
            column.iter_mut().for_each(scale);
        }
    }
    Some(())
}

/// The reconstruction of `range` with `cov`'s parameters at `values`: the
/// cross sections of a sampled parameter vector, exact in the parameters
/// rather than first order. [`with_parameters`] then
/// [`crate::resonance::reconstruction`], with their errors.
pub fn reconstruction_at(
    range: &ResonanceRange,
    cov: &RangeCovariance,
    values: &[f64],
) -> Result<Box<dyn crate::resonance::RangeReconstruction + Send + Sync>> {
    crate::resonance::reconstruction(&with_parameters(range, cov, values)?)
}

/// The relative covariance of a resolved range's group cross sections that
/// its resonance-parameter covariance implies.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupCovariance {
    /// Group edges, eV, ascending: `G + 1` of them.
    pub edges: Vec<f64>,
    /// The reactions, in block order: MT 2 and 102, 18 where the range has
    /// fission, and the MT of each other exit pair an R-matrix limited range
    /// has (600 for a proton, 800 for an alpha, 51 for an inelastic
    /// neutron).
    pub reactions: Vec<i32>,
    /// Per reaction, the 1/E-weighted group cross sections, barns.
    pub cross_sections: Vec<Vec<f64>>,
    /// Row-major `(R G) × (R G)`, reaction-major: element
    /// `(a G + g, b G + h)` is the relative covariance of reaction `a` in
    /// group `g` with reaction `b` in group `h`.
    pub relative: Vec<f64>,
}

impl GroupCovariance {
    pub fn groups(&self) -> usize {
        self.edges.len().saturating_sub(1)
    }

    /// The relative covariance of `(reaction a, group g)` with `(b, h)`.
    pub fn get(&self, a: usize, g: usize, b: usize, h: usize) -> f64 {
        let n = self.reactions.len() * self.groups();
        self.relative[(a * self.groups() + g) * n + b * self.groups() + h]
    }

    /// The same covariance relative to `totals` (per reaction, per group)
    /// instead of the resonance cross sections alone: what to use where the
    /// cross section that gets multiplied is the resonance part plus a
    /// background, as a transport replica's is. A group whose total is zero
    /// reads zero.
    pub fn relative_to(&self, totals: &[Vec<f64>]) -> GroupCovariance {
        let (r, g) = (self.reactions.len(), self.groups());
        let n = r * g;
        let scale: Vec<f64> = (0..n)
            .map(|k| {
                let (a, h) = (k / g, k % g);
                let total = totals.get(a).and_then(|t| t.get(h)).copied().unwrap_or(0.0);
                if total != 0.0 {
                    self.cross_sections[a][h] / total
                } else {
                    0.0
                }
            })
            .collect();
        let mut relative = self.relative.clone();
        for i in 0..n {
            for j in 0..n {
                relative[i * n + j] *= scale[i] * scale[j];
            }
        }
        GroupCovariance {
            edges: self.edges.clone(),
            reactions: self.reactions.clone(),
            cross_sections: (0..r)
                .map(|a| totals.get(a).cloned().unwrap_or_else(|| vec![0.0; g]))
                .collect(),
            relative,
        }
    }
}

/// Group edges for a resolved range: one group per resonance, the edges at
/// the geometric midpoints of neighbouring resonance energies, with a
/// thermal group below half the first resonance's energy. Groups are where a
/// transport replica's cross-section multiplier is constant, so a group per
/// resonance lets each resonance move on its own.
pub fn resonance_edges(range: &dyn crate::resonance::RangeReconstruction) -> Vec<f64> {
    let (el, eh) = range.bounds();
    let mut energies: Vec<f64> = range
        .resonances()
        .into_iter()
        .map(|(e, _)| e)
        .filter(|&e| e > el && e < eh)
        .collect();
    energies.sort_by(f64::total_cmp);
    energies.dedup();
    let mut edges = vec![el];
    if let Some(&first) = energies.first() {
        if 0.5 * first > el {
            edges.push(0.5 * first);
        }
    }
    for w in energies.windows(2) {
        edges.push((w[0] * w[1]).sqrt());
    }
    edges.push(eh);
    edges.dedup_by(|a, b| close(*a, *b));
    edges
}

/// Points a decade and points a resonance of [`integration_points`].
const PER_DECADE: usize = 400;
const PER_RESONANCE: usize = 256;

/// Errors unless `edges` are positive, ascending and within `range`, outside
/// which the resonance formula is not the cross section.
fn check_edges(range: &dyn crate::resonance::RangeReconstruction, edges: &[f64]) -> Result<()> {
    if edges.len() < 2 || edges.windows(2).any(|w| w[1] <= w[0]) || edges[0] <= 0.0 {
        return Err(Error::Mismatched {
            what: "group edges, which must be positive and ascending",
        });
    }
    let (el, eh) = range.bounds();
    if edges[0] < el || edges[edges.len() - 1] > eh {
        return Err(Error::Mismatched {
            what: "group edges and the range, which must hold them",
        });
    }
    Ok(())
}

/// The 0 K grid on which [`group_covariance`] takes its integrals, for
/// evaluating several reconstructions of one range on the same points:
/// typically the nominal one and one rebuilt from a sampled parameter vector
/// ([`reconstruction_at`]), whose difference a caller integrates against a
/// weight.
///
/// It holds every one of `edges` (group edges, or just the range's bounds)
/// and spans them: 400 logarithmic points a decade and, for every resonance
/// of every one of `reconstructions`, 256 points at `E_r + (G/2) tan(theta)`
/// (uniform in the Lorentzian's cumulative), then outwards at
/// `E_r +- (G/2) 1.1^k` to the edges, `G` the resonance's total width. The
/// outward points are what make a trapezoid of the difference of two
/// reconstructions accurate where it is a near-cancellation (the odd tail of
/// a moved resonance energy). A resonance two reconstructions share exactly
/// is traced once. An unresolved range has no resonances: its grid is the
/// logarithmic one. Pass the perturbed
/// reconstructions along with the nominal one: a sampled resonance energy
/// moves the peak, and a peak the grid traces only at its nominal place is
/// integrated well only while the shift is small against its width, so the
/// grid built from the union of the nominal and perturbed resonances is the
/// one on which each reconstruction's integral is as accurate as the nominal
/// one's.
///
/// Errors without a reconstruction, and unless `edges` are positive,
/// ascending and inside every reconstruction's bounds.
pub fn reconstruction_grid(
    reconstructions: &[&dyn crate::resonance::RangeReconstruction],
    edges: &[f64],
) -> Result<Vec<f64>> {
    if reconstructions.is_empty() {
        return Err(Error::Mismatched {
            what: "a reconstruction grid and the reconstructions it is for, of which there must be one",
        });
    }
    let mut resonances = Vec::new();
    for r in reconstructions {
        check_edges(*r, edges)?;
        resonances.extend(r.resonances());
    }
    resonances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    resonances.dedup();
    Ok(integration_points(
        &resonances,
        edges,
        PER_DECADE,
        PER_RESONANCE,
    ))
}

/// The points an integral over a resolved range is taken on: a logarithmic
/// grid at `per_decade` points a decade, each resonance traced at
/// `per_resonance` points spaced as `E_r + (G/2) tan(theta)` (uniform in the
/// Lorentzian's cumulative) and then outwards at `E_r +- (G/2) 1.1^k` to the
/// range's edges, and every group edge. At 400 a decade and 256 a resonance
/// Pb208's group variances are within 1% of their value at twice to four
/// times the density.
///
/// The outward points matter for the derivative with respect to a resonance's
/// energy: it is odd about the resonance and falls as a power of the
/// distance, so its integral over a group is a near-cancellation that
/// trapezoids on points symmetric about the resonance and geometric in the
/// distance get right, and the logarithmic grid alone, kilo-electronvolts
/// apart and lopsided about an 11 eV resonance at 571 keV, does not (it made
/// Pb208's capture variance there 1e4 times too large).
fn integration_points(
    resonances: &[(f64, f64)],
    edges: &[f64],
    per_decade: usize,
    per_resonance: usize,
) -> Vec<f64> {
    let (el, eh) = (edges[0], edges[edges.len() - 1]);
    let mut points = edges.to_vec();
    let decades = (eh / el).log10();
    let n = (decades * per_decade as f64).ceil().max(1.0) as usize;
    for i in 1..n {
        points.push(el * (eh / el).powf(i as f64 / n as f64));
    }
    for &(er, width) in resonances {
        if er <= el || er >= eh || width <= 0.0 {
            continue;
        }
        for j in 0..per_resonance {
            let theta = std::f64::consts::PI * ((j as f64 + 0.5) / per_resonance as f64 - 0.5);
            let e = er + 0.5 * width * theta.tan();
            if e > el && e < eh {
                points.push(e);
            }
        }
        let mut offset =
            0.5 * width * (std::f64::consts::FRAC_PI_2 * (1.0 - 1.0 / per_resonance as f64)).tan();
        while er - offset > el || er + offset < eh {
            for e in [er - offset, er + offset] {
                if e > el && e < eh {
                    points.push(e);
                }
            }
            offset *= 1.1;
        }
    }
    points.sort_by(f64::total_cmp);
    points.dedup();
    points
}

/// The relative covariance of `range`'s 1/E-weighted group cross sections on
/// `edges` that the parameter covariance `cov` implies, to first order:
///
/// ```text
/// C_(ag, bh) = sum_ij S_ag,i C_ij S_bh,j / (sigma_ag sigma_bh)
/// S_ag,i     = int_g (d sigma_a / d p_i) dE / E  /  int_g dE / E
/// ```
///
/// at infinite dilution and 0 K, as NJOY's ERRORR takes it. The integrals
/// are trapezoids in `ln E` on points that trace every resonance.
pub fn group_covariance(
    cov: &RangeCovariance,
    range: &dyn crate::resonance::RangeReconstruction,
    edges: &[f64],
) -> Result<GroupCovariance> {
    check_edges(range, edges)?;
    let groups = edges.len() - 1;
    let n_par = cov.len();
    let points = integration_points(&range.resonances(), edges, PER_DECADE, PER_RESONANCE);
    // Per group: the integral of each cross section and of each parameter's
    // gradient, and of the weight.
    const R: usize = crate::resonance::REACTIONS;
    let mut sigma = vec![[0.0; R]; groups];
    let mut sens = vec![[0.0; R]; groups * n_par];
    let mut width = vec![0.0; groups];
    let mut previous: Option<(
        f64,
        crate::resonance::CrossSections,
        Vec<crate::resonance::Gradient>,
    )> = None;
    let mut g = 0;
    for &e in &points {
        let x = range.cross_sections(e);
        let grad = range.parameter_gradients(e, cov)?;
        if let Some((e0, x0, grad0)) = &previous {
            while g + 1 < groups && *e0 >= edges[g + 1] {
                g += 1;
            }
            let h = 0.5 * (e / e0).ln();
            width[g] += 2.0 * h;
            let (a, b) = (x0.slots(), x.slots());
            for c in 0..R {
                sigma[g][c] += h * (a[c] + b[c]);
            }
            let row = &mut sens[g * n_par..(g + 1) * n_par];
            for (i, s) in row.iter_mut().enumerate() {
                for c in 0..R {
                    s[c] += h * (grad0[i][c] + grad[i][c]);
                }
            }
        }
        previous = Some((e, x, grad));
    }
    // Elastic and capture always; fission, and an R-matrix range's other
    // exit pairs, where the range has them.
    let present = |c: usize| sigma.iter().any(|s| s[c] != 0.0);
    let mut reactions: Vec<(usize, i32)> = vec![(0, 2), (1, 102)];
    if present(2) {
        reactions.push((2, 18));
    }
    for (k, mt) in range.other_reactions().into_iter().enumerate() {
        if let Some(mt) = mt.filter(|_| present(3 + k)) {
            reactions.push((3 + k, mt));
        }
    }
    let r = reactions.len();
    let rows = r * groups;
    // Relative sensitivities, row (a, g) by parameter.
    let mut s_rel = vec![0.0; rows * n_par];
    let mut cross_sections = vec![vec![0.0; groups]; r];
    for (ai, &(c, _)) in reactions.iter().enumerate() {
        for gi in 0..groups {
            let w = width[gi].max(f64::MIN_POSITIVE);
            let mean = sigma[gi][c] / w;
            cross_sections[ai][gi] = mean;
            if mean == 0.0 {
                continue;
            }
            for i in 0..n_par {
                s_rel[(ai * groups + gi) * n_par + i] = sens[gi * n_par + i][c] / w / mean;
            }
        }
    }
    // T = C S^T (n_par × rows), then S T.
    let mut t = vec![0.0; n_par * rows];
    for i in 0..n_par {
        let crow = &cov.covariance[i * n_par..(i + 1) * n_par];
        for k in 0..rows {
            let srow = &s_rel[k * n_par..(k + 1) * n_par];
            t[i * rows + k] = crow.iter().zip(srow).map(|(c, s)| c * s).sum();
        }
    }
    let mut relative = vec![0.0; rows * rows];
    for a in 0..rows {
        let srow = &s_rel[a * n_par..(a + 1) * n_par];
        for b in a..rows {
            let v: f64 = (0..n_par).map(|i| srow[i] * t[i * rows + b]).sum();
            relative[a * rows + b] = v;
            relative[b * rows + a] = v;
        }
    }
    Ok(GroupCovariance {
        edges: edges.to_vec(),
        reactions: reactions.iter().map(|&(_, mt)| mt).collect(),
        cross_sections,
        relative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::mf::mf32::Covariance;

    /// The MF=2 spin each MF=32 unresolved spin of one l=0 section lands on,
    /// for MF=2 spins 1/2, 3/2 and 5/2, and the builder's counts.
    fn unresolved_spins(mf32_j: &[f64]) -> (Vec<Option<usize>>, usize, Vec<f64>) {
        let mf2 = ResonanceParameters::Unresolved(Box::new(crate::mf::mf2::Unresolved {
            ranges: vec![crate::mf::mf2::UnresolvedRange {
                l: 0,
                aj: vec![0.5, 1.5, 2.5],
                ..Default::default()
            }],
            ..Default::default()
        }));
        let n = mf32_j.len();
        let range = Range {
            el: 1e3,
            eh: 1e4,
            lru: 2,
            lrf: 1,
            nro: 0,
            naps: 0,
            covariance: Covariance::Unresolved(Box::new(crate::mf::mf32::Unresolved {
                l_values: vec![crate::mf::mf32::UnresolvedL {
                    l: 0,
                    njs: n as i64,
                    parameters: mf32_j
                        .iter()
                        .map(|&j| [1.0, j, 1.0, 1.0, 0.0, 0.0])
                        .collect(),
                    ..Default::default()
                }],
                mpar: 1,
                relative_covariance: PackedCovariance {
                    order: n,
                    values: vec![0.01; n * (n + 1) / 2],
                },
                ..Default::default()
            })),
        };
        let mut b = Builder::default();
        read_unresolved(&range, &mf2, 0, &mut b).unwrap();
        let spins = b
            .parameters
            .iter()
            .map(|p| match p.location {
                Location::Unresolved { spin, .. } => Some(spin),
                _ => None,
            })
            .collect();
        (spins, b.approximate, b.unmatched)
    }

    #[test]
    fn unresolved_spins_never_share_an_mf2_spin() {
        // Every J matches: by J, in MF=32's order.
        assert_eq!(
            unresolved_spins(&[2.5, 0.5, 1.5]),
            (vec![Some(2), Some(0), Some(1)], 0, vec![])
        );
        // One J does not: by position for the whole section, where matching
        // each J alone would put 2 and 1.5 both on spin 1.
        assert_eq!(
            unresolved_spins(&[0.5, 2.0, 1.5]),
            (vec![Some(0), Some(1), Some(2)], 2, vec![])
        );
        // A J twice and the counts differ: the second is unmatched.
        assert_eq!(unresolved_spins(&[0.5, 0.5]), (vec![Some(0)], 0, vec![0.5]));
    }

    #[test]
    fn an_unresolved_l_mf2_does_not_have_is_left_out() {
        // MF=32 lists l=1 before l=0, MF=2 has only l=0 (FENDL-3.2d La138
        // gives l=2 over MF=2's 0 and 1): the matrix is l=0's own block.
        let mf2 = ResonanceParameters::Unresolved(Box::new(crate::mf::mf2::Unresolved {
            ranges: vec![crate::mf::mf2::UnresolvedRange {
                l: 0,
                aj: vec![0.5],
                ..Default::default()
            }],
            ..Default::default()
        }));
        let section = |l: i64, js: &[f64]| crate::mf::mf32::UnresolvedL {
            l,
            njs: js.len() as i64,
            parameters: js.iter().map(|&j| [1.0, j, 1.0, 1.0, 0.0, 0.0]).collect(),
            ..Default::default()
        };
        let covariance = PackedCovariance {
            order: 3,
            values: vec![0.01, 0.002, 0.003, 0.04, 0.005, 0.09],
        };
        let range = Range {
            el: 1e3,
            eh: 1e4,
            lru: 2,
            lrf: 1,
            nro: 0,
            naps: 0,
            covariance: Covariance::Unresolved(Box::new(crate::mf::mf32::Unresolved {
                l_values: vec![section(1, &[0.5, 1.5]), section(0, &[0.5])],
                mpar: 1,
                relative_covariance: covariance.clone(),
                ..Default::default()
            })),
        };
        let mut b = Builder::default();
        read_unresolved(&range, &mf2, 0, &mut b).unwrap();
        assert_eq!(b.parameters.len(), 1);
        assert_eq!(
            b.parameters[0].location,
            Location::Unresolved {
                orbital: 0,
                spin: 0
            }
        );
        assert_eq!(b.entries, [(0, 0, covariance.get(2, 2))]);
        assert_eq!(b.unmatched, [0.5, 1.5]);
    }

    const DY158: &[u8] = include_bytes!("../fixtures/n-066_Dy_158_mf2_mf32.endf.xz");
    const NA23: &[u8] = include_bytes!("../fixtures/n-011_Na_023_mf2_mf32.endf.xz");
    const PU244: &[u8] = include_bytes!("../fixtures/n-094_Pu_244_mf2_mf32.endf.xz");
    const TH232: &[u8] = include_bytes!("../fixtures/n-090_Th_232_mf2_mf32.endf.xz");
    const CL35: &[u8] = include_bytes!("../fixtures/n-017_Cl_035_mf2_mf32.endf.xz");

    /// MF=2 Reich-Moore sections each holding one 8 MeV J=1/2 resonance with
    /// the given neutron widths, as ENDF/B-VIII.1 Pb208 does in its L=0 and
    /// L=1 sections.
    fn same_energy_and_spin(gn: &[f64]) -> ResonanceParameters {
        ResonanceParameters::ReichMoore(crate::mf::mf2::ReichMoore {
            sections: gn
                .iter()
                .enumerate()
                .map(|(l, &gn)| crate::mf::mf2::ReichMooreSection {
                    l: l as i64,
                    er: vec![8.0e6],
                    aj: vec![0.5],
                    gn: vec![gn],
                    gg: vec![1.0],
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        })
    }

    #[test]
    fn resonances_of_one_spin_at_one_energy_are_told_apart_by_their_widths() {
        let mf2 = same_energy_and_spin(&[4.4577e5, 3.3064e7]);
        let mut b = Builder::default();
        // MF=32 lists the L=1 resonance first: its widths pick section 1,
        // where the energy and spin alone took whichever came first.
        for (gn, section) in [(3.3064e7, 1), (4.4577e5, 0)] {
            let found = orbital(&mf2, 8.0e6, 0.5, (gn, 1.0), &b.claimed).unwrap();
            let location = b.matched(found, 8.0e6);
            assert_eq!(location, Some(Location::Orbital { section, index: 0 }));
        }
        assert!(b.unmatched.is_empty() && b.approximate == 0);
    }

    #[test]
    fn an_mf2_resonance_is_given_to_one_mf32_resonance_at_most() {
        // Identical rows (JEFF-4.0 Zn66 and I127 have them): the second MF=32
        // resonance takes the second row, and a third finds none left.
        let mf2 = same_energy_and_spin(&[2.0, 2.0]);
        let mut b = Builder::default();
        let mut got = Vec::new();
        for _ in 0..3 {
            let found = orbital(&mf2, 8.0e6, 0.5, (2.0, 1.0), &b.claimed).unwrap();
            got.push(b.matched(found, 8.0e6));
        }
        assert_eq!(
            got,
            [
                Some(Location::Orbital {
                    section: 0,
                    index: 0
                }),
                Some(Location::Orbital {
                    section: 1,
                    index: 0
                }),
                None,
            ]
        );
        assert_eq!(b.unmatched, [8.0e6]);
    }

    fn read(fixture: &[u8]) -> (Material, Vec<RangeCovariance>) {
        let m = Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses");
        let v = range_covariances(m.mf2().unwrap(), m.mf32().unwrap()).expect("covariance");
        (m, v)
    }

    /// The trailing `k × k` block of a range's matrix, without the scattering
    /// radius parameters: `(trace, sum, C_01, C_12)`.
    fn checksum(r: &RangeCovariance, k: usize) -> (f64, f64, f64, f64) {
        let rows: Vec<usize> = (0..r.len())
            .filter(|&i| r.parameters[i].quantity != Quantity::ScatteringRadius)
            .collect();
        let rows = &rows[rows.len() - k..];
        let c = |a: usize, b: usize| r.get(rows[a], rows[b]);
        let trace = (0..k).map(|a| c(a, a)).sum();
        let sum = (0..k)
            .flat_map(|a| (0..k).map(move |b| (a, b)))
            .map(|(a, b)| c(a, b))
            .sum();
        (trace, sum, c(0, 1), c(1, 2))
    }

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
            "{a} against {b}"
        );
    }

    /// The layouts OpenMC 0.15.3 reads correctly give the same matrices: its
    /// checksums, taken from `ResonanceCovariances.from_endf` on the same
    /// ENDF/B-VIII.1 tapes. OpenMC keeps only the last LCOMP=1 block, so for
    /// Dy158 that block is compared.
    #[test]
    fn matches_openmc_where_openmc_reads_the_layout() {
        for (fixture, k, want) in [
            (
                DY158,
                12,
                (
                    4.328_531_116_379_6e-2,
                    4.311_801_048_426_772e-2,
                    -1.675_014e-8,
                    -4.711_39e-12,
                ),
            ),
            (
                NA23,
                69,
                (1.969_253_183_545_206e9, 1.969_253_183_545_206e9, 0.0, 0.0),
            ),
            (
                PU244,
                60,
                (7.312_618_670_523_5e-1, 7.312_618_670_523_5e-1, 0.0, 0.0),
            ),
        ] {
            let (_, v) = read(fixture);
            let got = checksum(&v[0], k);
            assert_close(got.0, want.0);
            assert_close(got.1, want.1);
            assert_close(got.2, want.2);
            assert_close(got.3, want.3);
        }
    }

    /// Every field of a compact correlation line lands in its own column, so
    /// the matrix has one non-zero correlation per non-zero field. OpenMC
    /// writes every field of a line to the line's first column, keeping only
    /// the last, which on Th232 loses most of them.
    #[test]
    fn every_compact_correlation_field_has_its_own_column() {
        let (m, v) = read(TH232);
        let Covariance::Compact(c) = &m.mf32().unwrap().isotopes[0].ranges[0].covariance else {
            panic!("Th232 is compact");
        };
        let fields = c.correlation.entries().count();
        let r = &v[0];
        let n = r.len();
        let off: usize = (0..n)
            .map(|i| (0..i).filter(|&j| r.get(i, j) != 0.0).count())
            .sum();
        assert_eq!(off, fields);
        for i in 0..n {
            for j in 0..i {
                let bound = (r.get(i, i) * r.get(j, j)).sqrt();
                assert!(r.get(i, j).abs() <= bound * (1.0 + 1e-12));
            }
        }
    }

    /// An R-matrix limited compact range: every parameter sits at an MF=2
    /// resonance whose energy and width it repeats, which needs the MF=2
    /// resonances read with their line padding.
    #[test]
    fn r_matrix_parameters_land_on_their_mf2_resonances() {
        let (m, v) = read(CL35);
        let ResonanceParameters::RMatrixLimited(rml) =
            &m.mf2().unwrap().isotopes[0].ranges[0].parameters
        else {
            panic!("Cl35 is R-matrix limited");
        };
        let r = &v[0];
        assert!(r.unmatched.is_empty() && r.approximate == 0);
        let mut checked = 0;
        for p in &r.parameters {
            if let Location::SpinGroup { group, index } = p.location {
                let sg = &rml.spin_groups[group];
                let want = match p.quantity {
                    Quantity::Energy => sg.er[index],
                    Quantity::ChannelWidth(c) => sg.gam[c][index],
                    _ => unreachable!(),
                };
                assert!(close(p.value, want), "{p:?} against {want}");
                checked += 1;
            }
        }
        assert!(checked > 100);
    }

    /// MF=2 R-matrix limited resonances are read with their line padding: a
    /// spin group of two or three channels has exactly NRS energies and NRS
    /// widths per channel, not the padding read as further resonances.
    #[test]
    fn mf2_r_matrix_resonances_skip_their_line_padding() {
        let (m, _) = read(CL35);
        let ResonanceParameters::RMatrixLimited(rml) =
            &m.mf2().unwrap().isotopes[0].ranges[0].parameters
        else {
            panic!("Cl35 is R-matrix limited");
        };
        for sg in &rml.spin_groups {
            assert!(sg.nch < 6);
            assert_eq!(sg.er.len(), sg.nrs as usize);
            assert!(sg.gam.iter().all(|row| row.len() == sg.nrs as usize));
            assert!(sg.er.windows(2).all(|w| w[0] <= w[1]), "energies ascend");
        }
    }

    /// Dy158's group covariance matches NJOY 2016 ERRORR's resonance-parameter
    /// contribution (ENDF/B-VIII.1, 1/E weight, the "contribution from
    /// resonance parameters (mf=32)" it prints) to its four printed digits and
    /// the 1% finite differences it takes them from, elastic and capture, over
    /// the resolved range.
    #[test]
    fn group_covariance_matches_errorr() {
        let m = Material::from_str(&crate::testdata::text(DY158)).expect("fixture parses");
        let cov = &range_covariances(m.mf2().unwrap(), m.mf32().unwrap()).unwrap()[0];
        let rm = crate::resonance::ReichMooreRange::new(&m.mf2().unwrap().isotopes[0].ranges[0])
            .unwrap();
        let edges = [1e-5, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 86.2];
        let g = group_covariance(cov, &rm, &edges).unwrap();
        assert_eq!(g.reactions, vec![2, 102]);
        let elastic = [
            3.635e-4, 3.106e-4, 2.243e-4, 2.366e-4, 3.692e-4, 1.065e-3, 2.343e-2, 3.569e-2,
        ];
        let capture = [
            4.477e-2, 4.326e-2, 3.303e-2, 1.469e-2, 1.634e-2, 2.292e-2, 4.058e-3, 1.930e-2,
        ];
        for h in 0..8 {
            for (a, want) in [(0, elastic[h]), (1, capture[h])] {
                let got = g.get(a, h, a, h);
                assert!(
                    (got / want - 1.0).abs() < 2e-3,
                    "reaction {a} group {h}: {got:e} against ERRORR's {want:e}"
                );
            }
        }
        // The cross block is the transpose of its mirror.
        for h in 0..8 {
            for k in 0..8 {
                assert_eq!(g.get(0, h, 1, k), g.get(1, k, 0, h));
            }
        }
    }

    /// One group per resonance: an edge between every pair of neighbours and
    /// a thermal group below the first.
    #[test]
    fn resonance_edges_put_one_resonance_in_each_group() {
        let m = Material::from_str(&crate::testdata::text(DY158)).expect("fixture parses");
        let rm = crate::resonance::ReichMooreRange::new(&m.mf2().unwrap().isotopes[0].ranges[0])
            .unwrap();
        let edges = resonance_edges(&rm);
        use crate::resonance::RangeReconstruction;
        let inside: Vec<f64> = rm
            .resonances()
            .into_iter()
            .map(|(e, _)| e)
            .filter(|&e| e > edges[0] && e < edges[edges.len() - 1])
            .collect();
        for w in edges.windows(2) {
            let n = inside.iter().filter(|&&e| e >= w[0] && e < w[1]).count();
            assert!(n <= 1, "group {w:?} holds {n} resonances");
        }
        assert!(edges.windows(2).all(|w| w[1] > w[0]));
    }

    /// The matrix is symmetric wherever it comes from.
    #[test]
    fn every_layout_gives_a_symmetric_matrix() {
        for fixture in [DY158, NA23, PU244, TH232, CL35] {
            let (_, v) = read(fixture);
            for r in &v {
                let n = r.len();
                for i in 0..n {
                    for j in 0..i {
                        assert_eq!(r.get(i, j), r.get(j, i));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod rebuild_tests {
    use super::*;
    use crate::material::Material;
    use crate::mf::mf2::{Unresolved, UnresolvedParameters, UnresolvedRange};
    use crate::resonance::{cross_sections_on, reconstruction, REACTIONS};

    /// Every MF=2 and MF=32 fixture: multi-level Breit-Wigner (Na23, Pu244),
    /// Reich-Moore (Dy158, Th232), R-matrix limited (Cl35, Cu63, Cu65, W183,
    /// W186, Rh103) and unresolved (Th232, Rh103).
    const FIXTURES: [&[u8]; 10] = [
        include_bytes!("../fixtures/n-011_Na_023_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-094_Pu_244_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-066_Dy_158_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-090_Th_232_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-017_Cl_035_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-029_Cu_063_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-029_Cu_065_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-074_W_183_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-074_W_186_mf2_mf32.endf.xz"),
        include_bytes!("../fixtures/n-045_Rh_103_mf2_mf32.endf.xz"),
    ];
    const NA23: &[u8] = FIXTURES[0];
    const DY158: &[u8] = FIXTURES[2];
    const W186: &[u8] = FIXTURES[8];
    const RH103: &[u8] = FIXTURES[9];
    const PU244: &[u8] = FIXTURES[1];
    const U235: &[u8] = include_bytes!("../fixtures/n-092_U_235_mf2.endf.xz");
    const V51: &[u8] = include_bytes!("../fixtures/n-023_V_051_mf2.endf.xz");

    fn material(fixture: &[u8]) -> Material {
        Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses")
    }

    /// Each of a fixture's covariances with the MF=2 range it was matched to.
    fn covariances(fixture: &[u8]) -> Vec<(ResonanceRange, RangeCovariance)> {
        let m = material(fixture);
        let mf2 = m.mf2().unwrap();
        range_covariances(mf2, m.mf32().unwrap())
            .unwrap()
            .into_iter()
            .map(|c| (mf2.isotopes[c.isotope].ranges[c.mf2_range].clone(), c))
            .collect()
    }

    fn nominal(cov: &RangeCovariance) -> Vec<f64> {
        cov.parameters.iter().map(|p| p.value).collect()
    }

    /// A covariance over `parameters` of `range` (unit matrix: only the
    /// parameters are read here).
    fn synthetic(
        range: &ResonanceRange,
        parameters: Vec<Parameter>,
        radius_steps: Vec<f64>,
    ) -> RangeCovariance {
        let n = parameters.len();
        let mut covariance = vec![0.0; n * n];
        for i in 0..n {
            covariance[i * n + i] = 1.0;
        }
        RangeCovariance {
            isotope: 0,
            range: 0,
            mf2_range: 0,
            el: range.el,
            eh: range.eh,
            lru: range.lru,
            lrf: range.lrf,
            parameters,
            covariance,
            approximate: 0,
            unmatched: Vec::new(),
            radius_steps,
        }
    }

    fn parameter(location: Location, quantity: Quantity, value: f64) -> Parameter {
        Parameter {
            location,
            quantity,
            value,
        }
    }

    /// The nominal vector gives back the MF=2 range field for field, and its
    /// reconstruction the nominal cross sections to the bit, on (a spread of)
    /// the grid an integral over the range is taken on.
    fn assert_nominal_is_exact(range: &ResonanceRange, cov: &RangeCovariance) {
        let values = nominal(cov);
        assert_eq!(&with_parameters(range, cov, &values).unwrap(), range);
        let a = reconstruction(range).unwrap();
        let b = reconstruction_at(range, cov, &values).unwrap();
        let (el, eh) = a.bounds();
        let grid = reconstruction_grid(&[a.as_ref(), b.as_ref()], &[el, eh]).unwrap();
        let points: Vec<f64> = grid
            .iter()
            .step_by((grid.len() / 300).max(1))
            .copied()
            .collect();
        for ((x, y), e) in cross_sections_on(a.as_ref(), &points)
            .iter()
            .zip(cross_sections_on(b.as_ref(), &points))
            .zip(&points)
        {
            for c in 0..REACTIONS {
                assert_eq!(
                    x.slots()[c].to_bits(),
                    y.slots()[c].to_bits(),
                    "reaction {c} at {e} eV"
                );
            }
        }
    }

    /// The step a central difference takes in parameter `i`, the energies it
    /// is taken at, and the width the derivative's size goes with.
    fn probe(cov: &RangeCovariance, i: usize, bounds: (f64, f64)) -> (f64, Vec<f64>, f64) {
        let p = cov.parameters[i];
        let (el, eh) = bounds;
        let spread = vec![
            el * (eh / el).powf(0.25),
            (el * eh).sqrt(),
            el * (eh / el).powf(0.8),
        ];
        match p.location {
            Location::Orbital { .. } | Location::SpinGroup { .. } => {
                let at = |q: &Parameter| q.location == p.location;
                let er = cov
                    .parameters
                    .iter()
                    .find(|q| at(q) && q.quantity == Quantity::Energy)
                    .map_or(p.value, |q| q.value);
                let width = cov
                    .parameters
                    .iter()
                    .filter(|q| at(q) && q.quantity != Quantity::Energy)
                    .map(|q| q.value.abs())
                    .sum::<f64>()
                    .max(1e-3);
                let h = match p.quantity {
                    Quantity::Energy => 1e-4 * width,
                    _ if cov.lrf == 3 => 1e-5 * p.value.abs(),
                    _ => 1e-3 * p.value.abs(),
                };
                let energies = [er, er + 0.7 * width, er * 1.3]
                    .into_iter()
                    .filter(|&e| e > el && e < eh)
                    .collect();
                (h, energies, width)
            }
            Location::Range => (1e-4, spread, eh - el),
            Location::Channel { .. } => (1e-5, spread, eh - el),
            Location::Unresolved { .. } => (1e-4, spread, eh - el),
        }
    }

    /// Moving parameter `i` by a small step moves the rebuilt cross sections
    /// by the range's analytic gradient times the step: a central difference
    /// of [`reconstruction_at`] against
    /// [`RangeReconstruction::parameter_gradients`]. Returns the number of
    /// comparisons.
    fn assert_first_order(range: &ResonanceRange, cov: &RangeCovariance, picks: &[usize]) -> usize {
        let base = reconstruction(range).unwrap();
        let mut checked = 0;
        for &i in picks {
            let p = cov.parameters[i];
            let (h, energies, width) = probe(cov, i, base.bounds());
            // A zero width has no derivative (its amplitude is not
            // differentiable there), and reads zero.
            if h == 0.0 {
                continue;
            }
            let moved = |delta: f64| {
                let mut values = nominal(cov);
                values[i] += delta;
                reconstruction_at(range, cov, &values).unwrap()
            };
            let (up, down) = (moved(h), moved(-h));
            for e in energies {
                let analytic = base.parameter_gradients(e, cov).unwrap()[i];
                let (u, d, x) = (
                    up.cross_sections(e).slots(),
                    down.cross_sections(e).slots(),
                    base.cross_sections(e).slots(),
                );
                for c in 0..REACTIONS {
                    let numeric = (u[c] - d[c]) / (2.0 * h);
                    // As the formalisms' own derivative tests: relative to
                    // the derivative, plus the difference's rounding noise,
                    // plus a floor for interference terms whose difference is
                    // still truncation.
                    let tolerance =
                        1e-4 * numeric.abs() + 1e-12 * x[c].abs() / h + 1e-7 * x[c].abs() / width;
                    assert!(
                        (analytic[c] - numeric).abs() <= tolerance,
                        "{:?} {:?} at {e} eV, reaction {c}: analytic {} against {numeric}",
                        p.location,
                        p.quantity,
                        analytic[c]
                    );
                    checked += 1;
                }
            }
        }
        checked
    }

    /// About `n` parameters spread across the covariance, and the first of
    /// every quantity, so each quantity a range has is tried.
    fn picks(cov: &RangeCovariance, n: usize) -> Vec<usize> {
        let mut out: Vec<usize> = (0..cov.len()).step_by((cov.len() / n).max(1)).collect();
        let mut seen = Vec::new();
        for (i, p) in cov.parameters.iter().enumerate() {
            let kind = std::mem::discriminant(&p.quantity);
            if !seen.contains(&kind) {
                seen.push(kind);
                out.push(i);
            }
        }
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn the_nominal_vector_rebuilds_every_fixture_bit_for_bit() {
        let mut formalisms = std::collections::BTreeSet::new();
        for fixture in FIXTURES {
            for (range, cov) in covariances(fixture) {
                assert_nominal_is_exact(&range, &cov);
                formalisms.insert((range.lru, range.lrf));
            }
        }
        assert_eq!(
            formalisms.into_iter().collect::<Vec<_>>(),
            [(1, 2), (1, 3), (1, 7), (2, 2)]
        );
    }

    #[test]
    fn a_small_step_moves_every_fixture_by_its_gradient() {
        for fixture in FIXTURES {
            for (range, cov) in covariances(fixture) {
                let checked = assert_first_order(&range, &cov, &picks(&cov, 6));
                assert!(checked >= 15, "{checked} comparisons");
            }
        }
    }

    /// A Reich-Moore range with fission: every quantity a resonance has (GFB
    /// included), and the radius parameter moving its one section's APL.
    #[test]
    fn reich_moore_places_every_quantity() {
        let range = material(U235).mf2().unwrap().isotopes[0].ranges[0].clone();
        let ResonanceParameters::ReichMoore(rm) = &range.parameters else {
            unreachable!()
        };
        let s = &rm.sections[0];
        let mut parameters = vec![parameter(Location::Range, Quantity::ScatteringRadius, 0.0)];
        for index in (0..s.er.len())
            .filter(|&i| s.gfb[i] != 0.0 && s.er[i] > range.el && s.er[i] < range.eh)
            .step_by(40)
        {
            let location = Location::Orbital { section: 0, index };
            for (q, v) in [
                (Quantity::Energy, s.er[index]),
                (Quantity::NeutronWidth, s.gn[index]),
                (Quantity::CaptureWidth, s.gg[index]),
                (Quantity::FissionWidth, s.gfa[index]),
                (Quantity::SecondFissionWidth, s.gfb[index]),
            ] {
                parameters.push(parameter(location, q, v));
            }
        }
        let cov = synthetic(&range, parameters, vec![0.02]);
        assert!(cov.len() > 20);
        assert_nominal_is_exact(&range, &cov);
        let all: Vec<usize> = (0..cov.len()).collect();
        assert!(assert_first_order(&range, &cov, &all) > 300);
        // Two standard deviations of the radius: AP plus twice the step,
        // written to the section's APL.
        let mut values = nominal(&cov);
        values[0] = 2.0;
        let moved = with_parameters(&range, &cov, &values).unwrap();
        let ResonanceParameters::ReichMoore(m) = &moved.parameters else {
            unreachable!()
        };
        let base = if s.apl != 0.0 { s.apl } else { rm.ap };
        assert_eq!(m.sections[0].apl, base + 2.0 * 0.02);
    }

    /// A multi-level Breit-Wigner range with a competitive width (Na23 given
    /// one): each width moves GT with it, the competitive width GT alone, and
    /// the radius parameter AP.
    #[test]
    fn breit_wigner_places_every_quantity() {
        let mut range = material(NA23).mf2().unwrap().isotopes[0].ranges[0].clone();
        let ResonanceParameters::BreitWigner(bw) = &mut range.parameters else {
            unreachable!()
        };
        let s = &mut bw.sections[0];
        s.lrx = 1;
        s.qx = -100.0;
        for i in 0..s.er.len() {
            s.gt[i] += 0.2 * s.gn[i];
        }
        let s = s.clone();
        let mut parameters = vec![parameter(Location::Range, Quantity::ScatteringRadius, 0.0)];
        for index in (0..s.er.len()).filter(|&i| s.er[i] > range.el && s.er[i] < range.eh) {
            let location = Location::Orbital { section: 0, index };
            for (q, v) in [
                (Quantity::Energy, s.er[index]),
                (Quantity::NeutronWidth, s.gn[index]),
                (Quantity::CaptureWidth, s.gg[index]),
                (Quantity::FissionWidth, s.gf[index]),
                (
                    Quantity::CompetitiveWidth,
                    s.gt[index] - s.gn[index] - s.gg[index] - s.gf[index],
                ),
            ] {
                parameters.push(parameter(location, q, v));
            }
        }
        let cov = synthetic(&range, parameters, vec![0.01; 3]);
        assert_nominal_is_exact(&range, &cov);
        let all: Vec<usize> = (0..cov.len()).collect();
        assert!(assert_first_order(&range, &cov, &all) > 100);
        // GN moves GT; GC moves GT alone.
        let at = |q: Quantity| {
            cov.parameters
                .iter()
                .position(|p| p.quantity == q && p.location != Location::Range)
                .unwrap()
        };
        let mut values = nominal(&cov);
        values[at(Quantity::NeutronWidth)] += 0.5;
        values[at(Quantity::CompetitiveWidth)] += 0.25;
        values[0] = -1.0;
        let moved = with_parameters(&range, &cov, &values).unwrap();
        let ResonanceParameters::BreitWigner(m) = &moved.parameters else {
            unreachable!()
        };
        let Location::Orbital { index, .. } = cov.parameters[at(Quantity::NeutronWidth)].location
        else {
            unreachable!()
        };
        assert_eq!(m.sections[0].gn[index], s.gn[index] + 0.5);
        assert_eq!(m.sections[0].gt[index], s.gt[index] + 0.5 + 0.25);
        assert_eq!(m.sections[0].gg[index], s.gg[index]);
        let ResonanceParameters::BreitWigner(b) = &range.parameters else {
            unreachable!()
        };
        assert_eq!(m.ap, b.ap - 0.01);
    }

    /// R-matrix limited channel radii: V51's group with distinct APE and APT
    /// moves both, a zero radius (the one standing for the other) stays zero,
    /// and the rebuilt cross sections follow the radius derivative.
    #[test]
    fn r_matrix_places_channel_radii() {
        for fixture in [V51, W186] {
            let range = material(fixture).mf2().unwrap().isotopes[0].ranges[0].clone();
            let ResonanceParameters::RMatrixLimited(rml) = &range.parameters else {
                unreachable!()
            };
            let mut parameters = Vec::new();
            for (group, sg) in rml.spin_groups.iter().enumerate().take(3) {
                for channel in 0..sg.nch as usize {
                    parameters.push(parameter(
                        Location::Channel { group, channel },
                        Quantity::ScatteringRadius,
                        sg.channels.apt[channel],
                    ));
                }
                for index in (0..sg.er.len()).step_by(5) {
                    let location = Location::SpinGroup { group, index };
                    parameters.push(parameter(location, Quantity::Energy, sg.er[index]));
                    for (c, row) in sg.gam.iter().enumerate() {
                        parameters.push(parameter(location, Quantity::ChannelWidth(c), row[index]));
                    }
                }
            }
            let cov = synthetic(&range, parameters, Vec::new());
            assert_nominal_is_exact(&range, &cov);
            let radii: Vec<usize> = (0..cov.len())
                .filter(|&i| cov.parameters[i].location != Location::Range)
                .collect();
            assert!(assert_first_order(&range, &cov, &radii) > 50);
            let mut values = nominal(&cov);
            for (v, p) in values.iter_mut().zip(&cov.parameters) {
                if p.quantity == Quantity::ScatteringRadius {
                    *v += 0.01;
                }
            }
            let moved = with_parameters(&range, &cov, &values).unwrap();
            let ResonanceParameters::RMatrixLimited(m) = &moved.parameters else {
                unreachable!()
            };
            for (group, sg) in rml.spin_groups.iter().enumerate().take(3) {
                let ch = &m.spin_groups[group].channels;
                for c in 0..sg.nch as usize {
                    let (ape, apt) = (sg.channels.ape[c], sg.channels.apt[c]);
                    let want = |x: f64, other: f64| {
                        if x != 0.0 || other == 0.0 {
                            x + 0.01
                        } else {
                            0.0
                        }
                    };
                    assert_eq!(ch.ape[c], want(ape, apt));
                    assert_eq!(ch.apt[c], want(apt, ape));
                }
            }
        }
    }

    /// Every unresolved quantity in each table case: A (D, GN0 and GG per J),
    /// B (energy-dependent GF) and C (all of them per energy), C with a
    /// competitive width (Rh103 given one) and a fission width (Pu244).
    #[test]
    fn unresolved_places_every_quantity_in_every_case() {
        let case_a = UnresolvedRange {
            awri: 100.0,
            l: 0,
            njs: 2,
            d: vec![20.0, 15.0],
            aj: vec![0.0, 1.0],
            amun: vec![1.0, 1.0],
            gno: vec![2e-3, 1e-3],
            gg: vec![0.05, 0.06],
            ..Default::default()
        };
        let case_b = UnresolvedRange {
            awri: 100.0,
            l: 0,
            njs: 1,
            parameters: vec![UnresolvedParameters::CaseB {
                muf: 2,
                d: 20.0,
                aj: 0.5,
                amun: 1.0,
                gn0: 2e-3,
                gg: 0.05,
                gf: vec![0.1, 0.12, 0.15],
            }],
            ..Default::default()
        };
        let synthetic_range = |section: UnresolvedRange, spi: f64, lrf: i64| ResonanceRange {
            el: 1e3,
            eh: 3e4,
            lru: 2,
            lrf,
            nro: 0,
            naps: 0,
            parameters: ResonanceParameters::Unresolved(Box::new(Unresolved {
                spi,
                ap: 0.6,
                nls: 1,
                es: vec![1e3, 1e4, 3e4],
                ranges: vec![section],
                ..Default::default()
            })),
        };
        let mut ranges = vec![
            synthetic_range(case_a, 0.5, 1),
            synthetic_range(case_b, 0.0, 1),
        ];
        for fixture in [RH103, PU244] {
            let m = material(fixture);
            let mut range = m.mf2().unwrap().isotopes[0]
                .ranges
                .iter()
                .find(|r| r.lru == 2)
                .unwrap()
                .clone();
            if let ResonanceParameters::Unresolved(u) = &mut range.parameters {
                for p in u.ranges.iter_mut().flat_map(|s| s.parameters.iter_mut()) {
                    if let UnresolvedParameters::CaseC { amux, gx, gg, .. } = p {
                        if gx.iter().all(|x| *x == 0.0) {
                            *gx = gg.iter().map(|g| 0.3 * g).collect();
                            *amux = amux.max(1.0);
                        }
                    }
                }
            }
            ranges.push(range);
        }
        let quantities = [
            Quantity::LevelSpacing,
            Quantity::ReducedNeutronWidth,
            Quantity::CaptureWidth,
            Quantity::FissionWidth,
            Quantity::CompetitiveWidth,
        ];
        for range in ranges {
            let ResonanceParameters::Unresolved(u) = &range.parameters else {
                unreachable!()
            };
            let mut parameters = Vec::new();
            for (orbital, s) in u.ranges.iter().enumerate() {
                let spins = s.parameters.len().max(s.aj.len());
                for spin in 0..spins {
                    for q in quantities {
                        parameters.push(parameter(Location::Unresolved { orbital, spin }, q, 1.0));
                    }
                }
            }
            let cov = synthetic(&range, parameters, Vec::new());
            assert_nominal_is_exact(&range, &cov);
            let all: Vec<usize> = (0..cov.len()).collect();
            assert!(assert_first_order(&range, &cov, &all) > 50);
            // Twice the level spacing of the first (l, J), at every energy.
            let mut values = nominal(&cov);
            values[0] = 2.0;
            let moved = with_parameters(&range, &cov, &values).unwrap();
            let ResonanceParameters::Unresolved(m) = &moved.parameters else {
                unreachable!()
            };
            let (was, now) = (&u.ranges[0], &m.ranges[0]);
            match (was.parameters.first(), now.parameters.first()) {
                (None, None) => assert_eq!(now.d[0], 2.0 * was.d[0]),
                (
                    Some(UnresolvedParameters::CaseB { d: a, .. }),
                    Some(UnresolvedParameters::CaseB { d: b, .. }),
                ) => assert_eq!(*b, 2.0 * a),
                (
                    Some(UnresolvedParameters::CaseC { d: a, .. }),
                    Some(UnresolvedParameters::CaseC { d: b, .. }),
                ) => {
                    assert!(a.iter().zip(b).all(|(a, b)| *b == 2.0 * a));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn a_vector_of_the_wrong_length_is_refused() {
        let (range, cov) = covariances(DY158).remove(0);
        let mut values = nominal(&cov);
        values.push(0.0);
        assert!(matches!(
            with_parameters(&range, &cov, &values),
            Err(Error::Mismatched { .. })
        ));
        values.truncate(cov.len() - 1);
        assert!(matches!(
            with_parameters(&range, &cov, &values),
            Err(Error::Mismatched { .. })
        ));
        assert!(reconstruction_at(&range, &cov, &[]).is_err());
        let mut values = nominal(&cov);
        values[3] = f64::NAN;
        assert!(with_parameters(&range, &cov, &values).is_err());
    }

    /// A quantity the formalism has no field for, or a location the range
    /// does not have, is refused rather than dropped. The nominal value is
    /// written nowhere, so only a moved one is refused.
    #[test]
    fn a_parameter_with_no_place_in_the_range_is_refused() {
        let reich_moore = material(DY158).mf2().unwrap().isotopes[0].ranges[0].clone();
        let breit_wigner = material(NA23).mf2().unwrap().isotopes[0].ranges[0].clone();
        let r_matrix = material(W186).mf2().unwrap().isotopes[0].ranges[0].clone();
        let orbital = Location::Orbital {
            section: 0,
            index: 0,
        };
        for (range, p) in [
            (
                &breit_wigner,
                parameter(orbital, Quantity::SecondFissionWidth, 0.0),
            ),
            (
                &reich_moore,
                parameter(orbital, Quantity::CompetitiveWidth, 0.0),
            ),
            (
                &reich_moore,
                parameter(orbital, Quantity::ChannelWidth(0), 0.0),
            ),
            (
                &reich_moore,
                parameter(orbital, Quantity::LevelSpacing, 1.0),
            ),
            (&r_matrix, parameter(orbital, Quantity::Energy, 1.0)),
            (
                &r_matrix,
                parameter(Location::Range, Quantity::ScatteringRadius, 0.0),
            ),
            (
                &r_matrix,
                parameter(
                    Location::SpinGroup {
                        group: 0,
                        index: 10_000,
                    },
                    Quantity::Energy,
                    1.0,
                ),
            ),
            (
                &reich_moore,
                parameter(
                    Location::Orbital {
                        section: 9,
                        index: 0,
                    },
                    Quantity::Energy,
                    1.0,
                ),
            ),
            (
                &reich_moore,
                parameter(
                    Location::Unresolved {
                        orbital: 0,
                        spin: 0,
                    },
                    Quantity::LevelSpacing,
                    1.0,
                ),
            ),
        ] {
            let cov = synthetic(range, vec![p], vec![0.01]);
            assert!(with_parameters(range, &cov, &[p.value]).is_ok());
            assert!(
                matches!(
                    with_parameters(range, &cov, &[p.value + 0.5]),
                    Err(Error::Mismatched { .. })
                ),
                "{p:?}"
            );
        }
        // Single-level Breit-Wigner is placed but not reconstructed.
        let mut single = breit_wigner.clone();
        single.lrf = 1;
        assert!(matches!(
            reconstruction(&single),
            Err(Error::Unsupported { .. })
        ));
    }

    /// The grid traces a perturbed resonance where it moved to, and holds
    /// every point of the nominal grid.
    #[test]
    fn the_grid_traces_nominal_and_perturbed_resonances() {
        let (range, cov) = covariances(DY158).remove(0);
        let i = cov
            .parameters
            .iter()
            .position(|p| p.quantity == Quantity::Energy && p.value > 1.0)
            .unwrap();
        let width: f64 = cov.parameters[i + 1].value + cov.parameters[i + 2].value;
        let shifted = cov.parameters[i].value + 7.3 * width;
        let mut values = nominal(&cov);
        values[i] = shifted;
        let a = reconstruction(&range).unwrap();
        let b = reconstruction_at(&range, &cov, &values).unwrap();
        let (el, eh) = a.bounds();
        let alone = reconstruction_grid(&[a.as_ref()], &[el, eh]).unwrap();
        let both = reconstruction_grid(&[a.as_ref(), b.as_ref()], &[el, eh]).unwrap();
        assert!(both.windows(2).all(|w| w[1] > w[0]));
        assert!(alone
            .iter()
            .all(|e| both.binary_search_by(|x| x.total_cmp(e)).is_ok()));
        // Half the points across a Lorentzian lie within half a width of it.
        let near = |grid: &[f64]| {
            grid.iter()
                .filter(|&&e| (e - shifted).abs() <= 0.5 * width)
                .count()
        };
        assert!(near(&both) >= PER_RESONANCE / 2, "{}", near(&both));
        assert!(near(&alone) < PER_RESONANCE / 8, "{}", near(&alone));
        // Nothing to build a grid for, or edges outside the range.
        assert!(reconstruction_grid(&[], &[el, eh]).is_err());
        assert!(reconstruction_grid(&[a.as_ref()], &[el, 2.0 * eh]).is_err());
    }
}
