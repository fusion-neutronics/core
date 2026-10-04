//! The covariance of a resolved range's resonance parameters, as one matrix
//! over named parameters.
//!
//! MF=32 writes a resolved range's covariance in one of five layouts
//! ([`crate::mf::mf32`]), each listing the parameters its own way: per
//! resonance in fixed 4 by 4 blocks (LCOMP=0), in short-range blocks over a
//! subset of resonances (LCOMP=1), or as uncertainties and an integer-coded
//! correlation matrix (LCOMP=2). [`resolved_covariances`] reads every one into
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
//! counted in [`ResolvedCovariance::approximate`]. A resonance MF=2 does not
//! have at all (JEFF-4.0 Sm151 at -0.08 eV) is left out of the matrix, which
//! keeps the rest of it exact (a marginal of a covariance is its sub-block),
//! and listed in [`ResolvedCovariance::unmatched`]. R-matrix spin groups are
//! matched in order by spin, parity and channel count, so an MF=2 group MF=32
//! leaves out (ENDF/B-VIII.1 W183's fifth) is skipped.
//!
//! # Scattering radius
//!
//! Where ISR=1 a Breit-Wigner or Reich-Moore range's radius uncertainty is one
//! parameter, uncorrelated with the resonance parameters (the format gives no
//! such correlation), of unit variance: one standard deviation of it moves
//! every section's radius at once, each by its own step
//! ([`ResolvedCovariance::radius_steps`]). The steps follow NJOY's ERRORR, the
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
use crate::mf::mf2::{Mf2, ResonanceParameters};
use crate::mf::mf32::{Covariance, Mf32, PackedCovariance, Range, ScatteringRadiusUncertainty};

/// Where a parameter sits in MF=2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Location {
    /// Resonance `index` of the MF=2 section `section` (the section of one
    /// orbital angular momentum), for Breit-Wigner and Reich-Moore.
    Orbital { section: usize, index: usize },
    /// Resonance `index` of spin group `group`, for R-matrix limited.
    SpinGroup { group: usize, index: usize },
    /// The range as a whole: its radius parameter, which moves every
    /// section's radius by its step in [`ResolvedCovariance::radius_steps`].
    Range,
    /// The radius of channel `channel` of spin group `group`, for R-matrix
    /// limited.
    Channel { group: usize, channel: usize },
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
    /// GX, the competitive width, for Breit-Wigner: GT less the others.
    CompetitiveWidth,
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
pub struct ResolvedCovariance {
    /// Index of the isotope in MF=2 and MF=32, and of the range within it.
    pub isotope: usize,
    pub range: usize,
    pub el: f64,
    pub eh: f64,
    /// The MF=2 representation: 1 or 2 Breit-Wigner, 3 Reich-Moore, 7
    /// R-matrix limited.
    pub lrf: i64,
    pub parameters: Vec<Parameter>,
    /// Row-major `n × n`, `n` the number of parameters.
    pub covariance: Vec<f64>,
    /// Resonances matched to an MF=2 resonance whose energy differs, within
    /// [`ENERGY_TOLERANCE`].
    pub approximate: usize,
    /// Energies of MF=32 resonances MF=2 does not list, left out of the
    /// matrix.
    pub unmatched: Vec<f64>,
    /// Per MF=2 section, how far one standard deviation of the range's radius
    /// parameter moves the section's radius (1e-12 cm). Empty where the
    /// range has no radius uncertainty.
    pub radius_steps: Vec<f64>,
}

impl ResolvedCovariance {
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
pub fn resolved_covariances(mf2: &Mf2, mf32: &Mf32) -> Result<Vec<ResolvedCovariance>> {
    let mut out = Vec::new();
    for (i, isotope) in mf32.isotopes.iter().enumerate() {
        let Some(parameters) = mf2.isotopes.get(i) else {
            return Err(Error::Mismatched {
                what: "MF=32 and MF=2 isotopes",
            });
        };
        for (r, range) in isotope.ranges.iter().enumerate() {
            if range.lru != 1 {
                continue;
            }
            let overlap = |el: f64, eh: f64| (eh.min(range.eh) - el.max(range.el)).max(0.0);
            let resonances = parameters
                .ranges
                .iter()
                .filter(|p| p.lru == 1 && overlap(p.el, p.eh) > 0.0)
                .max_by(|a, b| overlap(a.el, a.eh).total_cmp(&overlap(b.el, b.eh)))
                .ok_or(Error::Mismatched {
                    what: "an MF=32 resolved range and the MF=2 ranges",
                })?;
            let mut builder = Builder::default();
            read_range(range, &resonances.parameters, &mut builder)?;
            let covariance = builder.dense();
            out.push(ResolvedCovariance {
                isotope: i,
                range: r,
                el: range.el,
                eh: range.eh,
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
    /// approximate match and listing a missing one.
    fn matched(&mut self, found: Option<(Location, bool)>, er: f64) -> Option<Location> {
        match found {
            Some((location, exact)) => {
                if !exact {
                    self.approximate += 1;
                }
                Some(location)
            }
            None => {
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

/// The MF=2 section and index of the Breit-Wigner or Reich-Moore resonance
/// nearest `er` with spin `aj`, within [`ENERGY_TOLERANCE`], and whether the
/// energies agree to the digits ENDF writes. `None` where MF=2 has none.
fn orbital(parameters: &ResonanceParameters, er: f64, aj: f64) -> Result<Option<(Location, bool)>> {
    let sections: Vec<(&[f64], &[f64])> = match parameters {
        ResonanceParameters::BreitWigner(bw) => bw
            .sections
            .iter()
            .map(|s| (s.er.as_slice(), s.aj.as_slice()))
            .collect(),
        ResonanceParameters::ReichMoore(rm) => rm
            .sections
            .iter()
            .map(|s| (s.er.as_slice(), s.aj.as_slice()))
            .collect(),
        _ => {
            return Err(Error::Mismatched {
                what: "MF=32 Breit-Wigner or Reich-Moore parameters and the MF=2 range",
            })
        }
    };
    let mut best: Option<(f64, Location)> = None;
    for (section, (energies, spins)) in sections.iter().enumerate() {
        for (index, (&e, &j)) in energies.iter().zip(spins.iter()).enumerate() {
            if !close(j.abs(), aj.abs()) {
                continue;
            }
            let d = (e - er).abs();
            if best.is_none_or(|(b, _)| d < b) {
                best = Some((d, Location::Orbital { section, index }));
            }
        }
    }
    Ok(best
        .filter(|(d, _)| *d <= ENERGY_TOLERANCE * er.abs().max(1e-30))
        .map(|(d, l)| (l, d <= 1e-6 * er.abs().max(1e-30))))
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
                    let found = orbital(parameters, res[0], res[1])?;
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
                    let found = orbital(parameters, res[0], res[1])?;
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
                let found = orbital(parameters, p[0], p[1])?;
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
                        let found = spin_group_resonance(rml, sg.nch, values[0]);
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
            // parity and channel count, so a group MF=32 leaves out is
            // skipped.
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

/// The spin group of `rml` with `nch` channels holding the resonance nearest
/// `er`, within [`ENERGY_TOLERANCE`], its index there, and whether the
/// energies agree to the digit.
fn spin_group_resonance(
    rml: &crate::mf::mf2::RMatrixLimited,
    nch: i64,
    er: f64,
) -> Option<(Location, bool)> {
    let mut best: Option<(f64, Location)> = None;
    for (group, sg) in rml.spin_groups.iter().enumerate() {
        if sg.nch != nch {
            continue;
        }
        for (index, &e) in sg.er.iter().enumerate() {
            let d = (e - er).abs();
            if best.is_none_or(|(b, _)| d < b) {
                best = Some((d, Location::SpinGroup { group, index }));
            }
        }
    }
    best.filter(|(d, _)| *d <= ENERGY_TOLERANCE * er.abs().max(1e-30))
        .map(|(d, l)| (l, d <= 1e-6 * er.abs().max(1e-30)))
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

/// The relative covariance of a resolved range's group cross sections that
/// its resonance-parameter covariance implies.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupCovariance {
    /// Group edges, eV, ascending: `G + 1` of them.
    pub edges: Vec<f64>,
    /// The reactions, in block order: MT 2 and 102, and 18 where the range
    /// has fission.
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
}

/// Group edges for a resolved range: one group per resonance, the edges at
/// the geometric midpoints of neighbouring resonance energies, with a
/// thermal group below half the first resonance's energy. Groups are where a
/// transport replica's cross-section multiplier is constant, so a group per
/// resonance lets each resonance move on its own.
pub fn resonance_edges(range: &dyn crate::resonance::ResolvedRange) -> Vec<f64> {
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
    range: &dyn crate::resonance::ResolvedRange,
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
    for (er, width) in range.resonances() {
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
    cov: &ResolvedCovariance,
    range: &dyn crate::resonance::ResolvedRange,
    edges: &[f64],
) -> Result<GroupCovariance> {
    if edges.len() < 2 || edges.windows(2).any(|w| w[1] <= w[0]) || edges[0] <= 0.0 {
        return Err(Error::Mismatched {
            what: "group edges, which must be positive and ascending",
        });
    }
    let groups = edges.len() - 1;
    let n_par = cov.len();
    let points = integration_points(range, edges, 400, 256);
    // Per group: the integral of each cross section and of each parameter's
    // gradient, and of the weight.
    let mut sigma = vec![[0.0; 3]; groups];
    let mut sens = vec![[0.0; 3]; groups * n_par];
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
            let (a, b) = (
                [x0.elastic, x0.capture, x0.fission],
                [x.elastic, x.capture, x.fission],
            );
            for c in 0..3 {
                sigma[g][c] += h * (a[c] + b[c]);
            }
            let row = &mut sens[g * n_par..(g + 1) * n_par];
            for (i, s) in row.iter_mut().enumerate() {
                for c in 0..3 {
                    s[c] += h * (grad0[i][c] + grad[i][c]);
                }
            }
        }
        previous = Some((e, x, grad));
    }
    let fission = sigma.iter().any(|s| s[2] != 0.0);
    let reactions: Vec<(usize, i32)> = if fission {
        vec![(0, 2), (1, 102), (2, 18)]
    } else {
        vec![(0, 2), (1, 102)]
    };
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

    const DY158: &[u8] = include_bytes!("../fixtures/n-066_Dy_158_mf2_mf32.endf.xz");
    const NA23: &[u8] = include_bytes!("../fixtures/n-011_Na_023_mf2_mf32.endf.xz");
    const PU244: &[u8] = include_bytes!("../fixtures/n-094_Pu_244_mf2_mf32.endf.xz");
    const TH232: &[u8] = include_bytes!("../fixtures/n-090_Th_232_mf2_mf32.endf.xz");
    const CL35: &[u8] = include_bytes!("../fixtures/n-017_Cl_035_mf2_mf32.endf.xz");

    fn read(fixture: &[u8]) -> (Material, Vec<ResolvedCovariance>) {
        let m = Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses");
        let v = resolved_covariances(m.mf2().unwrap(), m.mf32().unwrap()).expect("covariance");
        (m, v)
    }

    /// The trailing `k × k` block of a range's matrix, without the scattering
    /// radius parameters: `(trace, sum, C_01, C_12)`.
    fn checksum(r: &ResolvedCovariance, k: usize) -> (f64, f64, f64, f64) {
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
        let cov = &resolved_covariances(m.mf2().unwrap(), m.mf32().unwrap()).unwrap()[0];
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
        use crate::resonance::ResolvedRange;
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
