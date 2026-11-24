//! Units of tally results.
//!
//! Transport scores per source particle. A score's unit per source particle
//! comes from the score (flux is a track length in cm, heating an energy in
//! eV, a reaction rate a count) times the units of any `energy_function`
//! filter. [`SourceStrength`] turns a result per source particle into one per
//! second or per pulse, and [`conversion`] works out the factor from a
//! result's own unit to a unit string such as `"W/cm3"`, dividing by each
//! bin's volume when the target is per volume.
//!
//! A small table of symbols, with SI prefixes, covers what tallies produce:
//! length (`m`), energy (`eV`, `J`), power (`W`), time (`s`, `min`, `h`, `d`,
//! `y`), dose (`Sv`, `Gy`), `pulse`, and counts (`particles`, `reactions`,
//! `n`, ...), which carry no dimension. No unit library is needed, so this
//! works in WASM and Pyodide.
use crate::filter::Filter;
use crate::score::Score;
use crate::tally::Tally;

/// Joules per electronvolt (exact, SI 2019).
const J_PER_EV: f64 = 1.602_176_634e-19;

/// Dimension exponents, in the order of [`DIM_NAMES`].
type Dims = [i8; 6];
const LENGTH: usize = 0;
const TIME: usize = 2;
const PULSE: usize = 5;
const DIM_NAMES: [&str; 6] = ["length", "energy", "time", "dose", "absorbed dose", "pulse"];

/// A unit as a factor times powers of the base units cm, eV, s, Sv, Gy and
/// pulse: `W` is `6.24e18 eV^1 s^-1`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Quantity {
    factor: f64,
    dims: Dims,
}

impl Quantity {
    const ONE: Self = Self {
        factor: 1.0,
        dims: [0; 6],
    };

    fn base(factor: f64, dim: usize) -> Self {
        let mut dims = [0; 6];
        dims[dim] = 1;
        Self { factor, dims }
    }

    fn mul(self, other: Self) -> Self {
        let mut dims = self.dims;
        for (d, o) in dims.iter_mut().zip(other.dims) {
            *d += o;
        }
        Self {
            factor: self.factor * other.factor,
            dims,
        }
    }

    fn powi(self, n: i8) -> Self {
        Self {
            factor: self.factor.powi(n as i32),
            dims: self.dims.map(|d| d * n),
        }
    }
}

/// How a source strength is counted: source particles per second (steady
/// state) or per pulse (ICF, one shot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrengthPer {
    Second,
    Pulse,
}

impl StrengthPer {
    /// `"second"` or `"pulse"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Second => "second",
            Self::Pulse => "pulse",
        }
    }

    /// Parse `"second"` / `"s"` or `"pulse"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "second" | "s" => Ok(Self::Second),
            "pulse" => Ok(Self::Pulse),
            other => Err(format!(
                "unknown per={other:?}: expected \"second\" (a steady source, particles per second) or \"pulse\" (particles per pulse)"
            )),
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Second => "s",
            Self::Pulse => "pulse",
        }
    }
}

/// Source particles per second or per pulse. Results are linear in it, so it
/// is applied after transport.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceStrength {
    pub value: f64,
    pub per: StrengthPer,
}

impl SourceStrength {
    /// A strength, refused unless finite and positive.
    pub fn new(value: f64, per: StrengthPer) -> Result<Self, String> {
        if value.is_finite() && value > 0.0 {
            Ok(Self { value, per })
        } else {
            Err(format!(
                "source strength must be finite and positive, got {value}"
            ))
        }
    }

    fn quantity(self) -> Quantity {
        Quantity::base(
            1.0,
            if self.per == StrengthPer::Second {
                TIME
            } else {
                PULSE
            },
        )
        .powi(-1)
    }
}

impl std::fmt::Display for SourceStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:e} per {}", self.value, self.per.name())
    }
}

/// The unit of a tally score per source particle, as text and as a
/// quantity. `None` for the quantity when an `energy_function` filter has no
/// units, or units this module does not know: such a result still reports a
/// unit string but cannot be converted.
struct ScoreUnit {
    text: String,
    quantity: Result<Quantity, String>,
}

fn score_unit(tally: &Tally, score: &Score) -> ScoreUnit {
    let base = score.base_units();
    let base_quantity = match base {
        "cm" => Quantity::base(1.0, LENGTH),
        "eV" => Quantity::base(1.0, 1),
        _ => Quantity::ONE,
    };
    let ef_units = tally.filters.iter().find_map(|f| match f {
        Filter::EnergyFunction(ef) => Some(ef.units.clone()),
        _ => None,
    });
    match ef_units {
        None => ScoreUnit {
            text: base.to_string(),
            quantity: Ok(base_quantity),
        },
        Some(None) => ScoreUnit {
            text: format!("(energy_function)·{base}"),
            quantity: Err(
                "the tally's energy_function has no units, so its results cannot be converted: give the energy_function its units".into(),
            ),
        },
        Some(Some(u)) => ScoreUnit {
            text: format!("{u}·{base}"),
            quantity: parse(&u)
                .map(|q| q.mul(base_quantity))
                .map_err(|e| format!("energy_function units {u:?}: {e}")),
        },
    }
}

/// Per score, the unit of a tally's results with `strength` applied (or per
/// source particle without one): `"eV/s"`, `"cm/pulse"`,
/// `"reactions/source-particle"`.
pub fn result_units(tally: &Tally, strength: Option<SourceStrength>) -> Vec<String> {
    let per = strength.map_or("source-particle", |s| s.per.symbol());
    tally
        .scores
        .iter()
        .map(|s| format!("{}/{per}", score_unit(tally, s).text))
        .collect()
}

/// What converting a score's results to a unit takes: a factor, and whether
/// each bin is also divided by its volume (in cm3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conversion {
    pub factor: f64,
    pub per_volume: bool,
}

/// The conversion of `score`'s results, with `strength` applied, to `target`.
pub fn conversion(
    tally: &Tally,
    score: &Score,
    strength: Option<SourceStrength>,
    target: &str,
) -> Result<Conversion, String> {
    let unit = score_unit(tally, score);
    let mut from = unit.quantity?;
    if let Some(s) = strength {
        from = from.mul(s.quantity());
    }
    let to = parse(target)?;
    let mut per_volume_dims = from.dims;
    per_volume_dims[LENGTH] -= 3;
    let per_volume = if to.dims == from.dims {
        false
    } else if to.dims == per_volume_dims {
        true
    } else {
        let have = format!(
            "{}/{}",
            unit.text,
            strength.map_or("source-particle", |s| s.per.symbol())
        );
        return Err(mismatch(&have, target, from.dims, to.dims, strength));
    };
    Ok(Conversion {
        factor: from.factor / to.factor,
        per_volume,
    })
}

fn mismatch(
    have: &str,
    target: &str,
    from: Dims,
    to: Dims,
    strength: Option<SourceStrength>,
) -> String {
    let head = format!("cannot convert {have} to {target:?}");
    let wants_time = to[TIME] != 0 && from[TIME] == 0;
    let wants_pulse = to[PULSE] != 0 && from[PULSE] == 0;
    match strength {
        None if wants_time || wants_pulse => format!(
            "{head}: the result is per source particle; call with_source_strength(...) first"
        ),
        Some(s) if s.per == StrengthPer::Pulse && wants_time => format!(
            "{head}: the source strength is per pulse, so the result is per pulse (for example J/pulse, not W)"
        ),
        Some(s) if s.per == StrengthPer::Second && wants_pulse => format!(
            "{head}: the source strength is per second; pass per=\"pulse\" to with_source_strength for a per-pulse source"
        ),
        Some(s) if from[TIME] + from[PULSE] != to[TIME] + to[PULSE] => format!(
            "{head}: the source strength is per {}, so the unit must be per {} too (for example W or eV/s for a steady source, J/pulse for a pulsed one)",
            s.per.name(),
            s.per.name()
        ),
        _ => {
            let describe = |d: Dims| {
                let parts: Vec<String> = d
                    .iter()
                    .zip(DIM_NAMES)
                    .filter(|(e, _)| **e != 0)
                    .map(|(e, n)| if *e == 1 { n.to_string() } else { format!("{n}^{e}") })
                    .collect();
                if parts.is_empty() {
                    "a count".to_string()
                } else {
                    parts.join(" ")
                }
            };
            format!(
                "{head}: the result is {}, the unit is {} (a per-volume unit, such as /cm3, divides by each bin's volume)",
                describe(from),
                describe(to)
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Symbols that are not prefixed.
fn plain_symbol(s: &str) -> Option<Quantity> {
    Some(match s {
        "min" => Quantity::base(60.0, TIME),
        "h" | "hr" => Quantity::base(3600.0, TIME),
        "d" | "day" => Quantity::base(86_400.0, TIME),
        "y" | "yr" | "year" => Quantity::base(365.25 * 86_400.0, TIME),
        "pulse" | "pulses" | "shot" | "shots" => Quantity::base(1.0, PULSE),
        "particle" | "particles" | "source" | "source-particle" | "source-particles" | "n"
        | "neutron" | "neutrons" | "photon" | "photons" | "reaction" | "reactions" | "atom"
        | "atoms" => Quantity::ONE,
        _ => return None,
    })
}

/// Symbols that take an SI prefix.
fn prefixable_symbol(s: &str) -> Option<Quantity> {
    Some(match s {
        "m" => Quantity::base(100.0, LENGTH),
        "s" => Quantity::base(1.0, TIME),
        "eV" => Quantity::base(1.0, 1),
        "J" => Quantity::base(1.0 / J_PER_EV, 1),
        "W" => Quantity::base(1.0 / J_PER_EV, 1).mul(Quantity::base(1.0, TIME).powi(-1)),
        "Sv" => Quantity::base(1.0, 3),
        "Gy" => Quantity::base(1.0, 4),
        _ => return None,
    })
}

fn prefix(c: char) -> Option<f64> {
    Some(match c {
        'p' => 1e-12,
        'n' => 1e-9,
        'u' | 'µ' | 'μ' => 1e-6,
        'm' => 1e-3,
        'c' => 1e-2,
        'k' => 1e3,
        'M' => 1e6,
        'G' => 1e9,
        'T' => 1e12,
        _ => return None,
    })
}

fn symbol(s: &str) -> Result<Quantity, String> {
    if let Some(q) = plain_symbol(s).or_else(|| prefixable_symbol(s)) {
        return Ok(q);
    }
    let mut chars = s.chars();
    if let (Some(p), rest) = (chars.next().and_then(prefix), chars.as_str())
        && let Some(q) = prefixable_symbol(rest)
    {
        return Ok(Quantity {
            factor: q.factor * p,
            ..q
        });
    }
    Err(format!(
        "unknown unit {s:?}; known units are m, s, min, h, d, y, eV, J, W, Sv, Gy (with SI prefixes such as c, m, k, M), pulse, and counts such as particles or reactions"
    ))
}

fn superscript_digit(c: char) -> Option<i8> {
    "⁰¹²³⁴⁵⁶⁷⁸⁹".chars().position(|d| d == c).map(|p| p as i8)
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_separators(&mut self) -> bool {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_whitespace() || matches!(c, '*' | '·' | '⋅'))
        {
            self.pos += 1;
        }
        self.pos > start
    }

    fn error(&self, what: &str) -> String {
        format!("unit {:?}: {what} at position {}", self.text, self.pos)
    }

    /// `factor (sep factor | '/' factor)*`
    fn product(&mut self) -> Result<Quantity, String> {
        self.skip_separators();
        let mut q = self.factor()?;
        loop {
            self.skip_separators();
            match self.peek() {
                None | Some(')') => return Ok(q),
                Some('/') => {
                    self.pos += 1;
                    self.skip_separators();
                    q = q.mul(self.factor()?.powi(-1));
                }
                Some(_) => q = q.mul(self.factor()?),
            }
        }
    }

    /// `'(' product ')' exponent? | number | symbol exponent?`
    fn factor(&mut self) -> Result<Quantity, String> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let q = self.product()?;
                if self.peek() != Some(')') {
                    return Err(self.error("missing ')'"));
                }
                self.pos += 1;
                Ok(q.powi(self.exponent()?))
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.pos;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E'))
                {
                    self.pos += 1;
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                let factor = s.parse::<f64>().map_err(|_| self.error("bad number"))?;
                Ok(Quantity {
                    factor,
                    dims: [0; 6],
                })
            }
            Some(c) if c.is_alphabetic() => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    let hyphen_word = c == '-'
                        && self
                            .chars
                            .get(self.pos + 1)
                            .is_some_and(|n| n.is_alphabetic());
                    if c.is_alphabetic() || hyphen_word {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                Ok(symbol(&s)?.powi(self.exponent()?))
            }
            _ => Err(self.error("expected a unit")),
        }
    }

    /// `('^'? '-'? digits) | superscripts`, default 1.
    fn exponent(&mut self) -> Result<i8, String> {
        if self.peek() == Some('^') {
            self.pos += 1;
        }
        let mut sign = 1;
        if matches!(self.peek(), Some('-' | '⁻')) {
            sign = -1;
            self.pos += 1;
        }
        let mut digits = String::new();
        let mut value: Option<i8> = None;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                digits.push(c);
            } else if let Some(d) = superscript_digit(c) {
                value = Some(value.unwrap_or(0) * 10 + d);
            } else {
                break;
            }
            self.pos += 1;
        }
        let n = match (digits.is_empty(), value) {
            (true, None) if sign == -1 => return Err(self.error("'-' without an exponent")),
            (true, None) => 1,
            (true, Some(v)) => v,
            (false, None) => digits
                .parse::<i8>()
                .map_err(|_| self.error("bad exponent"))?,
            (false, Some(_)) => return Err(self.error("mixed exponent digits")),
        };
        Ok(sign * n)
    }
}

fn parse(text: &str) -> Result<Quantity, String> {
    let mut p = Parser {
        chars: text.chars().collect(),
        pos: 0,
        text,
    };
    if p.chars.iter().all(|c| c.is_whitespace()) {
        return Err("empty unit".into());
    }
    let q = p.product()?;
    if p.pos != p.chars.len() {
        return Err(p.error("unexpected ')'"));
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, HeatingScore, ReactionRateScore};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    #[test]
    fn parses_common_units() {
        let w = parse("W").unwrap();
        assert!(close(w.factor, 1.0 / J_PER_EV));
        assert_eq!(w.dims, [0, 1, -1, 0, 0, 0]);
        assert_eq!(parse("W/cm3").unwrap().dims, [-3, 1, -1, 0, 0, 0]);
        assert_eq!(parse("W/m^3").unwrap().factor, w.factor / 1e6);
        let flux = parse("1/(cm2 s)").unwrap();
        assert_eq!(flux.dims, [-2, 0, -1, 0, 0, 0]);
        assert_eq!(parse("n/cm2/s").unwrap(), flux);
        assert_eq!(parse("cm^-2 s^-1").unwrap(), flux);
        assert_eq!(parse("cm⁻² s⁻¹").unwrap(), flux);
        assert!(close(parse("pSv·cm²").unwrap().factor, 1e-12));
        assert_eq!(parse("pSv·cm²").unwrap().dims, [2, 0, 0, 1, 0, 0]);
        assert!(close(parse("Sv/h").unwrap().factor, 1.0 / 3600.0));
        assert!(close(parse("MeV").unwrap().factor, 1e6));
        assert!(close(parse("ms").unwrap().factor, 1e-3));
        assert_eq!(parse("Gy").unwrap().dims, [0, 0, 0, 0, 1, 0]);
        assert_eq!(parse("J/pulse").unwrap().dims, [0, 1, 0, 0, 0, -1]);
        assert_eq!(parse("reactions/source-particle").unwrap(), Quantity::ONE);
    }

    #[test]
    fn rejects_bad_units() {
        assert!(parse("").is_err());
        assert!(parse("furlong").unwrap_err().contains("unknown unit"));
        assert!(parse("(cm").is_err());
        assert!(parse("cm)").is_err());
        assert!(parse("cm^-").is_err());
    }

    fn tally(score: Score) -> Tally {
        let mut t = Tally::new();
        t.scores = vec![score];
        t
    }

    #[test]
    fn heating_converts_with_and_without_strength() {
        let t = tally(Score::Heating(HeatingScore));
        let s = &t.scores[0];
        assert!(close(conversion(&t, s, None, "MeV").unwrap().factor, 1e-6));
        let per_s = SourceStrength::new(5e15, StrengthPer::Second).unwrap();
        let c = conversion(&t, s, Some(per_s), "W").unwrap();
        assert!(close(c.factor, J_PER_EV));
        assert!(!c.per_volume);
        assert!(conversion(&t, s, Some(per_s), "W/cm3").unwrap().per_volume);
        let err = conversion(&t, s, None, "W").unwrap_err();
        assert!(err.contains("with_source_strength"), "{err}");
        let per_pulse = SourceStrength::new(1e19, StrengthPer::Pulse).unwrap();
        assert!(close(
            conversion(&t, s, Some(per_pulse), "J/pulse")
                .unwrap()
                .factor,
            J_PER_EV
        ));
        let err = conversion(&t, s, Some(per_pulse), "W").unwrap_err();
        assert!(err.contains("per pulse"), "{err}");
        let err = conversion(&t, s, Some(per_s), "J").unwrap_err();
        assert!(err.contains("per second"), "{err}");
    }

    #[test]
    fn flux_is_a_track_length_until_divided_by_volume() {
        let t = tally(Score::Flux(FluxScore));
        let per_s = SourceStrength::new(1.0, StrengthPer::Second).unwrap();
        let c = conversion(&t, &t.scores[0], Some(per_s), "1/(cm2 s)").unwrap();
        assert!(c.per_volume && close(c.factor, 1.0));
        let c = conversion(&t, &t.scores[0], Some(per_s), "m/s").unwrap();
        assert!(!c.per_volume && close(c.factor, 0.01));
        let err = conversion(&t, &t.scores[0], Some(per_s), "eV/s").unwrap_err();
        assert!(err.contains("length"), "{err}");
    }

    #[test]
    fn dose_coefficients_give_dose_rate_per_volume() {
        use crate::filter::energy_function::EnergyFunctionFilter;
        let mut t = tally(Score::Flux(FluxScore));
        t.filters
            .push(Filter::EnergyFunction(EnergyFunctionFilter::with_units(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1.0; 4],
                crate::Interpolation::Cubic,
                "pSv·cm²",
            )));
        let per_s = SourceStrength::new(1.0, StrengthPer::Second).unwrap();
        let c = conversion(&t, &t.scores[0], Some(per_s), "Sv/h").unwrap();
        assert!(c.per_volume);
        assert!(close(c.factor, 1e-12 * 3600.0));
        assert_eq!(result_units(&t, Some(per_s)), vec!["pSv·cm²·cm/s"]);
    }

    #[test]
    fn reaction_rates_are_counts() {
        let t = tally(Score::ReactionRate(ReactionRateScore::total()));
        assert_eq!(result_units(&t, None), vec!["reactions/source-particle"]);
        let per_s = SourceStrength::new(2.0, StrengthPer::Second).unwrap();
        assert_eq!(result_units(&t, Some(per_s)), vec!["reactions/s"]);
        let c = conversion(&t, &t.scores[0], Some(per_s), "1/(cm3 s)").unwrap();
        assert!(c.per_volume);
    }

    #[test]
    fn strength_must_be_positive() {
        assert!(SourceStrength::new(0.0, StrengthPer::Second).is_err());
        assert!(SourceStrength::new(f64::NAN, StrengthPer::Pulse).is_err());
        assert!(StrengthPer::parse("hour").is_err());
    }
}
