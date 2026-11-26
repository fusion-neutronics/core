//! Unit strings for yamc and yani results.
//!
//! [`parse`] turns a unit string such as `"W/m3"`, `"kBq/g"`, `"1/(cm2 s)"` or
//! `"uSv/h"` into a [`Quantity`]: a factor times powers of the base units cm,
//! eV, s, Sv, pulse and g. A small table of symbols, with SI prefixes,
//! covers what transport and inventory results come in: length (`m`), mass
//! (`g`), energy (`eV`, `J`), power (`W`), activity (`Bq`, `Ci`), time (`s`,
//! `min`, `h`, `d`, `a` for a year), dose (`Sv`, `Gy`), `pulse`, and counts (`particles`,
//! `reactions`, `n`, ...), which carry no dimension. No unit library is
//! needed, so this works in WASM and Pyodide.
//!
//! [`convert_extensive`] is the conversion for a quantity a whole body has,
//! such as a material's activity: to the same unit, or to that unit per volume
//! or per mass. [`convert`] is for a quantity that is already intensive, such
//! as a dose rate.

/// Joules per electronvolt (exact, SI 2019).
pub const J_PER_EV: f64 = 1.602_176_634e-19;
/// Becquerels per curie (exact, by definition).
pub const BQ_PER_CI: f64 = 3.7e10;

/// How many base dimensions a [`Quantity`] has.
pub const N_DIMS: usize = 6;
/// Dimension exponents, in the order of [`DIM_NAMES`].
pub type Dims = [i8; N_DIMS];
pub const LENGTH: usize = 0;
pub const ENERGY: usize = 1;
pub const TIME: usize = 2;
/// Dose equivalent (Sv), kept apart from energy per mass because a sievert
/// is a weighted quantity that does not convert to a gray.
pub const DOSE: usize = 3;
pub const PULSE: usize = 4;
pub const MASS: usize = 5;
pub const DIM_NAMES: [&str; N_DIMS] = ["length", "energy", "time", "dose", "pulse", "mass"];

/// A unit as a factor times powers of the base units cm, eV, s, Sv, pulse and
/// g: `W` is `6.24e18 eV^1 s^-1`, and `Gy` is a J/kg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    pub factor: f64,
    pub dims: Dims,
}

impl Quantity {
    pub const ONE: Self = Self {
        factor: 1.0,
        dims: [0; N_DIMS],
    };

    pub fn base(factor: f64, dim: usize) -> Self {
        let mut dims = [0; N_DIMS];
        dims[dim] = 1;
        Self { factor, dims }
    }

    pub fn powi(self, n: i8) -> Self {
        Self {
            factor: self.factor.powi(n as i32),
            dims: self.dims.map(|d| d * n),
        }
    }
}

impl std::ops::Mul for Quantity {
    type Output = Self;

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
}

/// Dimensions in words, for error messages: `"energy time^-1 length^-3"`, or
/// `"a count"` for none.
pub fn describe(dims: Dims) -> String {
    let parts: Vec<String> = dims
        .iter()
        .zip(DIM_NAMES)
        .filter(|(e, _)| **e != 0)
        .map(|(e, n)| {
            if *e == 1 {
                n.to_string()
            } else {
                format!("{n}^{e}")
            }
        })
        .collect();
    if parts.is_empty() {
        "a count".to_string()
    } else {
        parts.join(" ")
    }
}

/// What a body-wide quantity is divided by to reach the unit asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalisation {
    /// The whole body: the same dimensions as the quantity.
    Total,
    /// Per cm³ of the body.
    PerVolume,
    /// Per g of the body.
    PerMass,
}

/// The factor from a quantity's own unit to the one asked for, after dividing
/// by what [`Normalisation`] says: the volume in cm³ or the mass in g.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtensiveConversion {
    pub factor: f64,
    pub normalisation: Normalisation,
}

/// Convert a quantity the whole body has (`from`, such as `"Bq"` or `"W"`) to
/// `to`: the same dimensions (`"TBq"`, `"MeV/s"`), or those per volume
/// (`"Bq/cm3"`, `"MW/m3"`) or per mass (`"kBq/g"`, `"W/kg"`).
pub fn convert_extensive(from: &str, to: &str) -> Result<ExtensiveConversion, String> {
    let have = parse(from)?;
    let want = parse(to)?;
    let mut per_volume = have.dims;
    per_volume[LENGTH] -= 3;
    let mut per_mass = have.dims;
    per_mass[MASS] -= 1;
    let normalisation = if want.dims == have.dims {
        Normalisation::Total
    } else if want.dims == per_volume {
        Normalisation::PerVolume
    } else if want.dims == per_mass {
        Normalisation::PerMass
    } else {
        return Err(format!(
            "cannot give {from} in {to:?}: expected {from} itself, per volume (e.g. {from}/cm3) \
             or per mass (e.g. {from}/kg), but {to:?} is {}",
            describe(want.dims)
        ));
    };
    Ok(ExtensiveConversion {
        factor: have.factor / want.factor,
        normalisation,
    })
}

/// The factor from `from` to `to`, which must have the same dimensions: for
/// a quantity that is already per unit of something, such as a dose rate.
pub fn convert(from: &str, to: &str) -> Result<f64, String> {
    let have = parse(from)?;
    let want = parse(to)?;
    if want.dims != have.dims {
        return Err(format!(
            "cannot give {from} in {to:?}: {from} is {}, {to:?} is {}",
            describe(have.dims),
            describe(want.dims)
        ));
    }
    Ok(have.factor / want.factor)
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
        // A Julian year, 365.25 days, written `a` (annum) and only `a`.
        "a" => Quantity::base(365.25 * 86_400.0, TIME),
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
        "W" => Quantity::base(1.0 / J_PER_EV, 1) * Quantity::base(1.0, TIME).powi(-1),
        "Sv" => Quantity::base(1.0, 3),
        // An absorbed dose is an energy per mass: J/kg.
        "Gy" => {
            Quantity::base(1.0 / (J_PER_EV * 1000.0), ENERGY) * Quantity::base(1.0, MASS).powi(-1)
        }
        "g" => Quantity::base(1.0, MASS),
        "Bq" => Quantity::base(1.0, TIME).powi(-1),
        "Ci" => Quantity::base(1.0 / BQ_PER_CI, TIME).powi(-1),
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

/// The hint for a year written another way than `a`: "yr", "year", ...
pub fn year_hint(s: &str) -> Option<&'static str> {
    matches!(s, "y" | "yr" | "yrs" | "year" | "years")
        .then_some("a year is written \"a\" (annum), e.g. \"mSv/a\" or (1, \"a\")")
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
    if let Some(hint) = year_hint(s) {
        return Err(format!("unknown unit {s:?}: {hint}"));
    }
    Err(format!(
        "unknown unit {s:?}; known units are m, g, s, min, h, d, a (year), eV, J, W, Bq, Ci, Sv, Gy (with SI prefixes such as c, m, k, M), pulse, and counts such as particles or reactions"
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
                    q = q * self.factor()?.powi(-1);
                }
                Some(_) => q = q * self.factor()?,
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
                    dims: [0; N_DIMS],
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

pub fn parse(text: &str) -> Result<Quantity, String> {
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
        assert_eq!(parse("Gy").unwrap(), parse("J/kg").unwrap());
        assert_eq!(parse("J/pulse").unwrap().dims, [0, 1, 0, 0, -1, 0]);
        assert_eq!(parse("reactions/source-particle").unwrap(), Quantity::ONE);
    }

    #[test]
    fn parses_mass_and_activity() {
        assert_eq!(parse("Bq").unwrap(), parse("1/s").unwrap());
        assert!(close(parse("TBq").unwrap().factor, 1e12));
        assert!(close(parse("Ci").unwrap().factor, BQ_PER_CI));
        assert!(close(parse("mCi").unwrap().factor, BQ_PER_CI * 1e-3));
        let kg = parse("kg").unwrap();
        assert!(close(kg.factor, 1e3));
        assert_eq!(kg.dims, [0, 0, 0, 0, 0, 1]);
        assert!(close(parse("mg").unwrap().factor, 1e-3));
        assert_eq!(parse("Bq/kg").unwrap().dims, [0, 0, -1, 0, 0, -1]);
        // Gy and y, cm and m, mCi and Ci: the whole symbol wins over a prefix.
        assert_eq!(parse("Gy").unwrap().dims[MASS], -1);
        assert!(close(parse("cm").unwrap().factor, 1.0));
    }

    #[test]
    fn a_year_is_a_alone() {
        assert!(close(parse("a").unwrap().factor, 365.25 * 86_400.0));
        for other in ["y", "yr", "year", "years"] {
            let err = parse(&format!("mSv/{other}")).unwrap_err();
            assert!(err.contains("a year is written \"a\""), "{other}: {err}");
        }
    }

    #[test]
    fn rejects_bad_units() {
        assert!(parse("").is_err());
        assert!(parse("furlong").unwrap_err().contains("unknown unit"));
        assert!(parse("(cm").is_err());
        assert!(parse("cm)").is_err());
        assert!(parse("cm^-").is_err());
    }

    #[test]
    fn extensive_conversions() {
        let c = convert_extensive("Bq", "TBq").unwrap();
        assert_eq!(c.normalisation, Normalisation::Total);
        assert!(close(c.factor, 1e-12));
        let c = convert_extensive("Bq", "Bq/m3").unwrap();
        assert_eq!(c.normalisation, Normalisation::PerVolume);
        assert!(close(c.factor, 1e6));
        let c = convert_extensive("Bq", "kBq/kg").unwrap();
        assert_eq!(c.normalisation, Normalisation::PerMass);
        assert!(close(c.factor, 1.0));
        let c = convert_extensive("W", "MeV/s").unwrap();
        assert!(close(c.factor, 1.0 / J_PER_EV / 1e6));
        let c = convert_extensive("Bq", "Ci/g").unwrap();
        assert!(close(c.factor, 1.0 / BQ_PER_CI));
        let err = convert_extensive("W", "eV").unwrap_err();
        assert!(err.contains("cannot give W in \"eV\""), "{err}");
        assert!(convert_extensive("Bq", "Bq/cm2").is_err());
    }

    #[test]
    fn intensive_conversions() {
        assert!(close(convert("Gy/h", "uGy/h").unwrap(), 1e6));
        assert!(close(convert("Sv/h", "mSv/h").unwrap(), 1e3));
        assert!(close(convert("Gy/h", "Gy/s").unwrap(), 1.0 / 3600.0));
        let err = convert("Gy/h", "Sv/h").unwrap_err();
        assert!(err.contains("dose"), "{err}");
    }
}
