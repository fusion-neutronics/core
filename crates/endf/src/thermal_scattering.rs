//! Thermal neutron scattering for one bound material, from a thermal
//! scattering law evaluation (NSUB=12).
//!
//! MF=7 gives the scattering law of a nuclide bound in a material (H in
//! water, C in graphite) in two parts: MT=4, the incoherent inelastic law
//! S(alpha, beta) at each temperature, and MT=2, the elastic part, which is
//! closed form (Bragg edges for a crystal, a Debye-Waller integral for an
//! incoherent scatterer, or both). MT=451, when present, names the atoms the
//! law describes. The formulas are those of the ENDF-6 manual, BNL-203218-2018
//! (section 7).
//!
//! This is what the evaluation says and nothing more. The inelastic law is
//! not sampleable as tabulated: turning it into outgoing energy and angle
//! distributions (the alpha-beta integration, the short-collision-time tail)
//! is a processing step done elsewhere.

use crate::data::K_BOLTZMANN;
use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};
use crate::material::Material;
use crate::mf::mf7::{Mf7Mt2, Mf7Mt4, ThermalElement};
use crate::records::{field, int_endf};

/// How a secondary (non-principal) atom of the material scatters, from its
/// analytic-model flag B(6i+1) in MT=4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryModel {
    /// The short-collision-time approximation, which has its own effective
    /// temperature.
    ShortCollisionTime,
    /// Free gas.
    FreeGas,
    /// Diffusive motion.
    Diffusive,
}

/// A secondary atom of the material, described by an analytic law.
#[derive(Debug, Clone, PartialEq)]
pub struct SecondaryAtom {
    pub model: SecondaryModel,
    /// Its free-atom scattering cross section, in barns, for all atoms of
    /// this kind in the molecule together.
    pub free_atom_xs: f64,
    /// Its mass in neutron masses.
    pub awr: f64,
    /// How many atoms of this kind the molecule has.
    pub count: f64,
}

/// The incoherent inelastic law S(alpha, beta, T) of the principal atom, as
/// tabulated.
#[derive(Debug, Clone, PartialEq)]
pub struct InelasticLaw {
    /// The alpha grid, shared by every beta and temperature.
    pub alpha: Vec<f64>,
    /// The beta grid.
    pub beta: Vec<f64>,
    /// Interpolation in alpha: the regions of the first beta's TAB1, which
    /// every beta shares.
    pub alpha_breakpoints: Vec<i32>,
    pub alpha_interpolation: Vec<i32>,
    /// Interpolation in beta, from the TAB2 record.
    pub beta_breakpoints: Vec<i32>,
    pub beta_interpolation: Vec<i32>,
    /// S at each temperature, `s[t][i_beta * alpha.len() + i_alpha]`. When
    /// [`ThermalScattering::lln`] is set these are ln S.
    pub s: Vec<Vec<f64>>,
    /// Interpolation between each temperature and the previous one (the
    /// first is 0, having no previous one).
    pub temperature_interpolation: Vec<i64>,
}

impl InelasticLaw {
    /// S (or ln S) at grid point `(i_alpha, i_beta)` of temperature `t`.
    pub fn s_at(&self, t: usize, i_alpha: usize, i_beta: usize) -> f64 {
        self.s[t][i_beta * self.alpha.len() + i_alpha]
    }
}

/// Coherent elastic scattering, off the Bragg edges of a crystal.
#[derive(Debug, Clone, PartialEq)]
pub struct CoherentElastic {
    /// The temperatures it is given at, in Kelvin.
    pub temperatures: Vec<f64>,
    /// The Bragg edges, in eV.
    pub bragg_edges: Vec<f64>,
    /// The cumulative structure factor at each edge, per temperature, in
    /// eV-barns: `factors[t][i]` is the sum over every edge up to and
    /// including edge `i`.
    pub factors: Vec<Vec<f64>>,
}

impl CoherentElastic {
    /// The cross section in barns at `energy` (eV) for temperature index `t`:
    /// the cumulative factor of the last edge at or below `energy`, over
    /// `energy`. Zero below the first edge.
    pub fn xs(&self, t: usize, energy: f64) -> f64 {
        let n = self.bragg_edges.partition_point(|&e| e <= energy);
        if n == 0 {
            0.0
        } else {
            self.factors[t][n - 1] / energy
        }
    }
}

/// Incoherent elastic scattering, from the Debye-Waller integral.
#[derive(Debug, Clone, PartialEq)]
pub struct IncoherentElastic {
    /// The characteristic bound cross section in barns, as the evaluation
    /// gives it (SB in the format).
    pub bound_xs: f64,
    /// The Debye-Waller integral W'(T) in 1/eV against temperature in Kelvin.
    pub debye_waller: Tabulated1D,
}

impl IncoherentElastic {
    /// The cross section in barns at `energy` (eV) and `temperature` (K):
    /// `(SB / 2) (1 - exp(-4 E W')) / (2 E W')`.
    pub fn xs(&self, temperature: f64, energy: f64) -> f64 {
        let w = self.debye_waller.eval(temperature);
        let x = 2.0 * energy * w;
        if x == 0.0 {
            return self.bound_xs;
        }
        self.bound_xs / 2.0 * -(-2.0 * x).exp_m1() / x
    }
}

/// The elastic part of a thermal scattering law (MF=7 MT=2), by its LTHR.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalElastic {
    Coherent(CoherentElastic),
    Incoherent(IncoherentElastic),
    /// Both, whose cross sections add.
    Mixed {
        coherent: CoherentElastic,
        incoherent: IncoherentElastic,
    },
}

impl ThermalElastic {
    /// The LTHR flag this representation is written with.
    pub fn lthr(&self) -> i64 {
        match self {
            ThermalElastic::Coherent(_) => 1,
            ThermalElastic::Incoherent(_) => 2,
            ThermalElastic::Mixed { .. } => 3,
        }
    }
}

/// One thermal scattering law evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalScattering {
    /// The material number.
    pub mat: i32,
    /// ZA as the evaluation writes it, which names the material, not a
    /// nuclide (134 is CH4).
    pub za: i64,
    /// The ZSYMAM text, such as `"H(H2O)"`.
    pub zsymam: Option<String>,
    /// The principal atom's mass in neutron masses (A0 in the format).
    pub awr: f64,
    /// The principal atom's free-atom scattering cross section in barns, for
    /// all principal atoms of the molecule together. Zero when the principal
    /// atom is described by an analytic law, in which case no S(alpha, beta)
    /// is tabulated.
    pub free_atom_xs: f64,
    /// The energy above which the law is not to be used, in eV.
    pub energy_max: f64,
    /// The value epsilon the evaluation gives for the principal atom, in eV.
    pub epsilon: f64,
    /// How many principal atoms the molecule has (M0).
    pub principal_count: f64,
    /// 1 when alpha and beta are in units of kT at 0.0253 eV, 0 when in kT
    /// at the actual temperature.
    pub lat: i64,
    /// True when S is symmetric in beta (LASYM = 0).
    pub symmetric: bool,
    /// 1 when S is stored as its natural logarithm.
    pub lln: i64,
    pub secondary: Vec<SecondaryAtom>,
    /// The temperatures of the inelastic law, in Kelvin. For a principal atom
    /// with an analytic law, the temperatures of its effective temperature
    /// table.
    pub temperatures: Vec<f64>,
    /// The tabulated inelastic law, absent for an analytic principal atom.
    pub inelastic: Option<InelasticLaw>,
    /// Effective temperatures against material temperature: the principal
    /// atom's first, then each short-collision-time secondary atom's.
    pub effective_temperature: Vec<Tabulated1D>,
    /// The elastic part, absent when the evaluation has no MT=2.
    pub elastic: Option<ThermalElastic>,
    /// The atoms MF=7 MT=451 names, when the evaluation has that section.
    pub atoms: Option<Vec<ThermalElement>>,
}

fn unsupported(what: &'static str) -> Error {
    Error::Unsupported { what }
}

impl ThermalScattering {
    /// Build from a thermal scattering law material.
    pub fn from_endf(material: &Material) -> Result<Self> {
        let mt4 = material.mf7_mt4().ok_or(unsupported(
            "a thermal scattering material with no MF=7 MT=4",
        ))?;
        let b = &mt4.b;
        let get = |i: usize| -> Result<f64> {
            b.get(i).copied().ok_or(unsupported(
                "an MF=7 MT=4 B list shorter than the format requires",
            ))
        };
        let secondary = (0..mt4.ns.max(0) as usize)
            .map(|i| {
                let base = 6 * (i + 1);
                let model = match get(base)? {
                    0.0 => SecondaryModel::ShortCollisionTime,
                    1.0 => SecondaryModel::FreeGas,
                    2.0 => SecondaryModel::Diffusive,
                    _ => {
                        return Err(unsupported(
                            "an MF=7 MT=4 secondary atom model other than 0, 1 or 2",
                        ))
                    }
                };
                Ok(SecondaryAtom {
                    model,
                    free_atom_xs: get(base + 1)?,
                    awr: get(base + 2)?,
                    count: get(base + 5)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let inelastic = inelastic_law(mt4)?;
        let temperatures = match &inelastic {
            Some(_) => {
                let first = &mt4.beta_data[0];
                std::iter::once(first.t)
                    .chain(first.others.iter().map(|o| o.t))
                    .collect()
            }
            None => mt4.teff.first().map(|t| t.x.clone()).unwrap_or_default(),
        };

        let elastic = material.mf7_mt2().map(elastic_part).transpose()?;
        let header = material.mf1_mt451();

        Ok(ThermalScattering {
            mat: material.mat,
            za: mt4.za,
            zsymam: header.and_then(|h| h.zsymam.as_ref().map(|s| s.trim().to_string())),
            awr: get(2)?,
            free_atom_xs: get(0)?,
            energy_max: get(3)?,
            epsilon: get(1)?,
            principal_count: get(5)?,
            lat: mt4.lat,
            symmetric: mt4.lasym == 0,
            lln: mt4.lln,
            secondary,
            temperatures,
            inelastic,
            effective_temperature: mt4.teff.clone(),
            elastic,
            atoms: material.mf7_mt451().map(|s| s.elements.clone()),
        })
    }

    /// kT in eV at each of [`Self::temperatures`].
    pub fn kts(&self) -> Vec<f64> {
        self.temperatures.iter().map(|t| K_BOLTZMANN * t).collect()
    }

    /// The ZA of every atom MF=7 MT=451 names, principal first, or `None`
    /// when the evaluation has no MT=451.
    pub fn atom_zas(&self) -> Option<Vec<i64>> {
        self.atoms.as_ref().map(|elements| {
            elements
                .iter()
                .flat_map(|e| e.zai.iter().map(|&z| z as i64))
                .collect()
        })
    }
}

fn tab2_regions(t: &Tabulated2D) -> (Vec<i32>, Vec<i32>) {
    (t.breakpoints.clone(), t.interpolation.clone())
}

fn inelastic_law(mt4: &Mf7Mt4) -> Result<Option<InelasticLaw>> {
    let Some(first) = mt4.beta_data.first() else {
        return Ok(None);
    };
    let alpha = first.s.x.clone();
    let n_alpha = alpha.len();
    let n_temps = 1 + first.others.len();
    let n_beta = mt4.beta_data.len();
    let mut s = vec![vec![0.0; n_alpha * n_beta]; n_temps];
    let mut beta = Vec::with_capacity(n_beta);
    for (i, law) in mt4.beta_data.iter().enumerate() {
        if law.s.x != alpha || law.others.len() + 1 != n_temps {
            return Err(unsupported(
                "an MF=7 MT=4 law whose betas do not share one alpha grid and one set of temperatures",
            ));
        }
        if law.s.breakpoints != first.s.breakpoints || law.s.interpolation != first.s.interpolation
        {
            return Err(unsupported(
                "an MF=7 MT=4 law whose betas interpolate in alpha differently",
            ));
        }
        beta.push(law.beta);
        s[0][i * n_alpha..(i + 1) * n_alpha].copy_from_slice(&law.s.y);
        for (t, other) in law.others.iter().enumerate() {
            if other.s.len() != n_alpha {
                return Err(unsupported(
                    "an MF=7 MT=4 temperature whose S list is not the length of the alpha grid",
                ));
            }
            s[t + 1][i * n_alpha..(i + 1) * n_alpha].copy_from_slice(&other.s);
        }
    }
    let temperature_interpolation = std::iter::once(0)
        .chain(first.others.iter().map(|o| o.li))
        .collect();
    let (beta_breakpoints, beta_interpolation) =
        mt4.beta_int.as_ref().map(tab2_regions).unwrap_or_default();
    Ok(Some(InelasticLaw {
        alpha,
        beta,
        alpha_breakpoints: first.s.breakpoints.clone(),
        alpha_interpolation: first.s.interpolation.clone(),
        beta_breakpoints,
        beta_interpolation,
        s,
        temperature_interpolation,
    }))
}

fn coherent(c: &crate::mf::mf7::CoherentElastic) -> Result<CoherentElastic> {
    let bragg_edges = c.s.x.clone();
    let mut factors = vec![c.s.y.clone()];
    let mut temperatures = vec![c.t];
    for o in &c.others {
        if o.s.len() != bragg_edges.len() {
            return Err(unsupported(
                "an MF=7 MT=2 coherent temperature whose factors are not one per Bragg edge",
            ));
        }
        temperatures.push(o.t);
        factors.push(o.s.clone());
    }
    Ok(CoherentElastic {
        temperatures,
        bragg_edges,
        factors,
    })
}

fn incoherent(i: &crate::mf::mf7::IncoherentElastic) -> IncoherentElastic {
    IncoherentElastic {
        bound_xs: i.sb,
        debye_waller: i.w.clone(),
    }
}

fn elastic_part(mt2: &Mf7Mt2) -> Result<ThermalElastic> {
    let missing = || unsupported("an MF=7 MT=2 section missing the part its LTHR names");
    match mt2.lthr {
        1 => Ok(ThermalElastic::Coherent(coherent(
            mt2.coherent.as_ref().ok_or_else(missing)?,
        )?)),
        2 => Ok(ThermalElastic::Incoherent(incoherent(
            mt2.incoherent.as_ref().ok_or_else(missing)?,
        ))),
        3 => Ok(ThermalElastic::Mixed {
            coherent: coherent(mt2.coherent.as_ref().ok_or_else(missing)?)?,
            incoherent: incoherent(mt2.incoherent.as_ref().ok_or_else(missing)?),
        }),
        _ => Err(unsupported("an MF=7 MT=2 LTHR other than 1, 2 or 3")),
    }
}

/// What [`first_temperature_only`] cut from an evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstTemperatureCut {
    /// The evaluation with only its first temperature, still valid ENDF-6:
    /// every record is copied as written except that the temperature counts
    /// (LT) of the MT=2 and MT=4 TAB1 records are zero and the LIST records
    /// they counted are gone.
    pub text: String,
    /// The temperature kept, in K.
    pub temperature: f64,
    /// The temperatures dropped, in K, in the order the file gives them.
    pub dropped_temperatures: Vec<f64>,
    /// How many LIST records were dropped from MT=2 and from MT=4.
    pub dropped_mt2_records: usize,
    pub dropped_mt4_records: usize,
    /// How many of the dropped MT=2 records did not give one factor per
    /// Bragg edge, and so were not valid ENDF-6.
    pub invalid_mt2_records: usize,
}

/// Whether `t` is a temperature NJOY cannot tell from `first`: THERMR's
/// `rdelas` takes the coherent elastic LIST of any temperature within
/// `temp/1000 + 5` K of the one asked for.
fn indistinct(first: f64, t: f64) -> bool {
    (t - first).abs() < first / 1000.0 + 5.0
}

/// The evaluation in `text` cut to its first temperature, when every later
/// temperature it gives in MF=7 MT=2 and MT=4 is within a few kelvin of the
/// first; `None` when it gives no later temperature, or any later one that is
/// farther away, so a genuine temperature grid is never touched.
///
/// JEFF-4.0 publishes its UO2, PuO2, ThO2 and Zircaloy-4 laws one file per
/// temperature T, and every one of those files also carries a copy at T+1 K
/// (a LIST record after each TAB1). These copies are not data to keep:
///
/// - In the Zircaloy-4 files at 350, 600 and 700 K the T+1 coherent elastic
///   LIST has fewer factors than the TAB1 has Bragg edges (1738 for 1740, for
///   example). ENDF-6 gives every later temperature the first one's edges, so
///   these records are invalid, and the factors in them belong to an edge set
///   at T+1 that the file does not give (two edges that are separate at
///   350 K are one step at 351 K). There is no faithful reading of them.
/// - NJOY's THERMR (`rdelas`) never compares the asked-for temperature with
///   the TAB1's: it takes the LIST of any temperature within `temp/1000 + 5`
///   K. Asked for T it uses the T+1 factors on the T edges (0.3% from the
///   TAB1 at the highest edges), and from a short LIST it reads past the
///   record's end. Handing NJOY the cut file is what makes its elastic part
///   at T the evaluation's TAB1.
///
/// Nothing is padded or interpolated: the first temperature's records are
/// kept as written, and the effective temperature and Debye-Waller tables,
/// which are functions of temperature rather than per-temperature records,
/// are kept whole.
pub fn first_temperature_only(text: &str) -> Result<Option<FirstTemperatureCut>> {
    let material = Material::from_str(text)?;
    let coherent = material.mf7_mt2().and_then(|m| m.coherent.as_ref());
    let mt4 = material.mf7_mt4();
    let first = match (mt4.and_then(|m| m.beta_data.first()), coherent) {
        (Some(law), _) => law.t,
        (None, Some(c)) => c.t,
        (None, None) => return Ok(None),
    };
    if coherent.is_some_and(|c| c.t != first) {
        return Ok(None);
    }
    let mut later: Vec<f64> = Vec::new();
    let mut invalid_mt2_records = 0;
    if let Some(c) = coherent {
        for o in &c.others {
            later.push(o.t);
            if o.s.len() != c.s.x.len() {
                invalid_mt2_records += 1;
            }
        }
    }
    for law in mt4.map(|m| m.beta_data.as_slice()).unwrap_or_default() {
        if law.t != first {
            return Ok(None);
        }
        later.extend(law.others.iter().map(|o| o.t));
    }
    if later.is_empty() || !later.iter().all(|&t| indistinct(first, t)) {
        return Ok(None);
    }
    let mut dropped_temperatures: Vec<f64> = Vec::new();
    for t in later {
        if !dropped_temperatures.contains(&t) {
            dropped_temperatures.push(t);
        }
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let (mut dropped_mt2_records, mut dropped_mt4_records) = (0, 0);
    let mut i = 0;
    while i < lines.len() {
        match mf_mt(lines[i]) {
            (7, 2) => {
                let mut walk = SectionWalk::new(&lines, i, 2, &mut out);
                let lthr = walk.int_at(0, 2)?;
                walk.keep(1)?;
                if lthr == 1 || lthr == 3 {
                    dropped_mt2_records += walk.tab1_first_temperature()?;
                }
                i = walk.rest()?;
            }
            (7, 4) => {
                let mut walk = SectionWalk::new(&lines, i, 4, &mut out);
                walk.keep(1)?;
                let npl = walk.int_at(0, 4)?;
                let b1 = crate::records::float_endf(field(walk.line(1)?, 0, 11));
                walk.keep(1 + lines_for(npl, 6))?;
                if b1 > 0.0 {
                    let nr = walk.int_at(0, 4)?;
                    let nb = walk.int_at(0, 5)?;
                    walk.keep(1 + lines_for(nr, 3))?;
                    for _ in 0..nb {
                        dropped_mt4_records += walk.tab1_first_temperature()?;
                    }
                }
                i = walk.rest()?;
            }
            _ => {
                out.push(lines[i].to_string());
                i += 1;
            }
        }
    }
    let mut cut = out.join("\n");
    cut.push('\n');

    // The cut must read back as one temperature, the first, with the same
    // records the evaluation gives at it.
    let check = Material::from_str(&cut)?;
    let cut_coherent = check.mf7_mt2().and_then(|m| m.coherent.as_ref());
    let same_coherent = match (coherent, cut_coherent) {
        (Some(a), Some(b)) => b.others.is_empty() && a.s == b.s,
        (None, None) => true,
        _ => false,
    };
    let same_mt4 = match (mt4, check.mf7_mt4()) {
        (Some(a), Some(b)) => {
            a.beta_data.len() == b.beta_data.len()
                && a.beta_data
                    .iter()
                    .zip(&b.beta_data)
                    .all(|(x, y)| y.others.is_empty() && x.s == y.s && x.beta == y.beta)
                && a.teff == b.teff
        }
        (None, None) => true,
        _ => false,
    };
    if !same_coherent || !same_mt4 {
        return Err(Error::Unsupported {
            what: "an MF=7 section that does not cut cleanly to its first temperature",
        });
    }
    Ok(Some(FirstTemperatureCut {
        text: cut,
        temperature: first,
        dropped_temperatures,
        dropped_mt2_records,
        dropped_mt4_records,
        invalid_mt2_records,
    }))
}

fn mf_mt(line: &str) -> (i64, i64) {
    (int_endf(field(line, 70, 72)), int_endf(field(line, 72, 75)))
}

/// Lines holding `n` values at `per_line` to a line.
fn lines_for(n: i64, per_line: i64) -> usize {
    (n.max(0) as usize).div_ceil(per_line as usize)
}

/// A walk through the lines of one MF=7 section, copying what it keeps.
struct SectionWalk<'a> {
    lines: &'a [&'a str],
    at: usize,
    mt: i64,
    out: &'a mut Vec<String>,
}

impl<'a> SectionWalk<'a> {
    fn new(lines: &'a [&'a str], at: usize, mt: i64, out: &'a mut Vec<String>) -> Self {
        SectionWalk { lines, at, mt, out }
    }

    fn line(&self, offset: usize) -> Result<&'a str> {
        self.lines
            .get(self.at + offset)
            .copied()
            .filter(|l| mf_mt(l) == (7, self.mt))
            .ok_or(Error::Unsupported {
                what: "an MF=7 section shorter than its records' counts",
            })
    }

    /// Integer field `n` (0 to 5) of the line `offset` lines on.
    fn int_at(&self, offset: usize, n: usize) -> Result<i64> {
        Ok(int_endf(field(self.line(offset)?, 11 * n, 11 * n + 11)))
    }

    fn keep(&mut self, n: usize) -> Result<()> {
        for k in 0..n {
            self.out.push(self.line(k)?.to_string());
        }
        self.at += n;
        Ok(())
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        for k in 0..n {
            self.line(k)?;
        }
        self.at += n;
        Ok(())
    }

    /// Copy a TAB1 with its temperature count LT set to zero, and drop the
    /// LT LIST records after it. Returns how many were dropped.
    fn tab1_first_temperature(&mut self) -> Result<usize> {
        let head = self.line(0)?;
        if !head.is_ascii() || head.len() < 33 {
            return Err(Error::Unsupported {
                what: "an MF=7 TAB1 record that is not a plain ENDF line",
            });
        }
        let lt = self.int_at(0, 2)?;
        let (nr, np) = (self.int_at(0, 4)?, self.int_at(0, 5)?);
        self.out
            .push(format!("{}{:>11}{}", &head[..22], 0, &head[33..]));
        self.at += 1;
        self.keep(lines_for(nr, 3) + lines_for(np, 3))?;
        for _ in 0..lt.max(0) {
            let npl = self.int_at(0, 4)?;
            self.skip(1 + lines_for(npl, 6))?;
        }
        Ok(lt.max(0) as usize)
    }

    /// Copy the rest of the section, and return the line after it.
    fn rest(self) -> Result<usize> {
        let mut at = self.at;
        while at < self.lines.len() && mf_mt(self.lines[at]) == (7, self.mt) {
            self.out.push(self.lines[at].to_string());
            at += 1;
        }
        Ok(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(fixture: &[u8]) -> ThermalScattering {
        load_text(&crate::testdata::text(fixture))
    }

    fn load_text(text: &str) -> ThermalScattering {
        ThermalScattering::from_endf(&Material::from_str(text).unwrap()).unwrap()
    }

    const SIO2: &[u8] = include_bytes!("../fixtures/tsl-SiO2-beta.endf.xz");
    const ZR_ZRH: &[u8] = include_bytes!("../fixtures/tsl-ZrinZrH.endf.xz");
    const CH4: &[u8] = include_bytes!("../fixtures/tsl-s-CH4.endf.xz");
    /// JEFF-4.0's Zr90 in Zircaloy-4 at 350 K, cut to its first three betas:
    /// a 351 K copy after every TAB1, the coherent one with 1738 factors for
    /// 1740 Bragg edges.
    const ZY4_350K: &[u8] = include_bytes!("../fixtures/tsl-Zr90inZy4-350K.jeff40.endf.xz");

    #[test]
    fn a_coherent_crystal_with_a_short_collision_time_secondary() {
        let t = load(SIO2);
        assert_eq!(t.temperatures.len(), 2);
        assert_eq!(t.secondary.len(), 1);
        assert_eq!(t.secondary[0].model, SecondaryModel::ShortCollisionTime);
        // The principal atom's effective temperature, then the SCT atom's.
        assert_eq!(t.effective_temperature.len(), 2);
        let law = t.inelastic.as_ref().unwrap();
        assert_eq!(law.s.len(), 2);
        assert_eq!(law.s[0].len(), law.alpha.len() * law.beta.len());
        // The second temperature carries its own LI, not the outer LT.
        assert_eq!(law.temperature_interpolation.len(), 2);
        assert_ne!(
            law.temperature_interpolation[1], 1,
            "LT, not LI, was stored"
        );
        let Some(ThermalElastic::Coherent(c)) = &t.elastic else {
            panic!("SiO2 is coherent")
        };
        assert_eq!(c.temperatures, t.temperatures);
        // Zero below the first Bragg edge, factor over E above it.
        let e0 = c.bragg_edges[0];
        assert_eq!(c.xs(0, 0.5 * e0), 0.0);
        let e = 0.5 * (c.bragg_edges[0] + c.bragg_edges[1]);
        assert_eq!(c.xs(0, e), c.factors[0][0] / e);
    }

    #[test]
    fn an_incoherent_scatterer_at_eight_temperatures() {
        let t = load(ZR_ZRH);
        assert_eq!(t.temperatures.len(), 8);
        assert!(t.symmetric);
        let Some(ThermalElastic::Incoherent(i)) = &t.elastic else {
            panic!("Zr in ZrH is incoherent")
        };
        // The cross section tends to SB as E W' goes to zero and falls as
        // 1/E at high energy.
        let lo = i.xs(296.0, 1e-9);
        assert!((lo - i.bound_xs).abs() < 1e-6 * i.bound_xs);
        let w = i.debye_waller.eval(296.0);
        let e = 10.0 / w;
        assert!((i.xs(296.0, e) - i.bound_xs / 2.0 / (2.0 * e * w)).abs() < 1e-9);
    }

    #[test]
    fn mixed_elastic_adds_both_parts() {
        let coherent = CoherentElastic {
            temperatures: vec![296.0],
            bragg_edges: vec![1e-3, 2e-3],
            factors: vec![vec![1e-3, 3e-3]],
        };
        let incoherent = IncoherentElastic {
            bound_xs: 4.0,
            debye_waller: Tabulated1D::new(vec![296.0, 400.0], vec![10.0, 12.0]),
        };
        let mixed = ThermalElastic::Mixed {
            coherent: coherent.clone(),
            incoherent: incoherent.clone(),
        };
        assert_eq!(mixed.lthr(), 3);
        assert_eq!(coherent.xs(0, 2.5e-3), 3e-3 / 2.5e-3);
        assert_eq!(
            coherent.xs(0, 1e-3),
            1.0,
            "an energy on an edge takes that edge"
        );
    }

    #[test]
    fn a_free_gas_secondary_has_no_effective_temperature() {
        let t = load(CH4);
        assert_eq!(t.secondary[0].model, SecondaryModel::FreeGas);
        assert_eq!(t.effective_temperature.len(), 1);
        assert_eq!(t.lat, 0);
        assert!(t.atoms.is_none() || t.atom_zas().is_some());
    }

    #[test]
    fn a_coherent_temperature_short_of_the_bragg_edges_is_refused() {
        let m = Material::from_str(&crate::testdata::text(ZY4_350K)).unwrap();
        let e = ThermalScattering::from_endf(&m).unwrap_err().to_string();
        assert!(e.contains("not one per Bragg edge"), "{e}");
    }

    #[test]
    fn a_one_kelvin_copy_is_cut_to_the_first_temperature() {
        let text = crate::testdata::text(ZY4_350K);
        let whole = Material::from_str(&text).unwrap();
        let cut = first_temperature_only(&text)
            .unwrap()
            .expect("351 K is a copy");
        assert_eq!(cut.temperature, 350.0);
        assert_eq!(cut.dropped_temperatures, vec![351.0]);
        assert_eq!(cut.dropped_mt2_records, 1);
        assert_eq!(cut.dropped_mt4_records, 3, "one per beta");
        assert_eq!(cut.invalid_mt2_records, 1);

        let t = load_text(&cut.text);
        assert_eq!(t.temperatures, vec![350.0]);
        let Some(ThermalElastic::Coherent(c)) = &t.elastic else {
            panic!("Zr in Zircaloy-4 is coherent")
        };
        // The 350 K TAB1 exactly as written, nothing filled in.
        let tab1 = &whole.mf7_mt2().unwrap().coherent.as_ref().unwrap().s;
        assert_eq!(c.bragg_edges, tab1.x);
        assert_eq!(c.factors, vec![tab1.y.clone()]);
        assert_eq!(c.bragg_edges.len(), 1740);
        let law = t.inelastic.as_ref().unwrap();
        assert_eq!(law.s.len(), 1);
        assert_eq!(law.beta.len(), 3);
        let first_beta = &whole.mf7_mt4().unwrap().beta_data[0].s;
        assert_eq!(law.s[0][..law.alpha.len()], first_beta.y[..]);
        // The effective temperature is a function of temperature, kept whole.
        assert_eq!(
            t.effective_temperature,
            whole.mf7_mt4().unwrap().teff.clone()
        );
        // Everything outside MF=7 MT=2 and MT=4 is untouched, and the cut is
        // a fixed point.
        let cut_material = Material::from_str(&cut.text).unwrap();
        assert_eq!(
            cut_material.section_text.get(&(1, 451)),
            whole.section_text.get(&(1, 451))
        );
        assert_eq!(first_temperature_only(&cut.text).unwrap(), None);
    }

    #[test]
    fn a_genuine_temperature_grid_is_not_cut() {
        for fixture in [SIO2, ZR_ZRH, CH4] {
            let text = crate::testdata::text(fixture);
            assert_eq!(first_temperature_only(&text).unwrap(), None);
        }
    }

    #[test]
    fn a_later_temperature_is_a_copy_only_within_njoys_window() {
        assert!(indistinct(350.0, 351.0));
        assert!(indistinct(1000.0, 1005.9));
        assert!(!indistinct(1000.0, 1006.0));
        assert!(!indistinct(296.0, 400.0));
    }
}
