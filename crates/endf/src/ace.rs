//! Reading ACE cross section tables.
//!
//! ACE is not ENDF: it is the processed, ready-to-sample form that NJOY writes
//! and transport codes read. The two meet here because several of the package's
//! higher-level types can be built from either.
//!
//! Only Type 1 (ASCII) tables are read. Type 2 (binary) tables are a different
//! on-disk layout and are reported as unsupported rather than misparsed.

use std::path::Path;

use crate::data::{gnds_name, ATOMIC_SYMBOL, EV_PER_MEV, K_BOLTZMANN};
use crate::error::{Error, Result};

/// Lines of header before the XSS array begins.
const ACE_HEADER_SIZE: usize = 12;

/// How a library encodes the metastable state in a ZAID.
///
/// The plain ZAID, 1000*Z + A, has nowhere to put it, so libraries disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetastableScheme {
    /// ZAID is 1000*Z + A + 100*m.
    #[default]
    Nndc,
    /// 400 is added for a metastable nuclide, except that 95242 is Am242m and
    /// 95642 is the ground state. Newer libraries use an SZA form above
    /// 1000000.
    Mcnp,
}

/// What a ZAID identifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// GNDS name, e.g. `"Am242_m1"`.
    pub name: String,
    /// Atomic symbol, e.g. `"Am"`.
    pub element: String,
    pub z: u32,
    pub mass_number: u32,
    pub metastable: u32,
}

/// Identify the nuclide a ZAID refers to.
pub fn get_metadata(zaid: i64, scheme: MetastableScheme) -> Result<Metadata> {
    let mut z = zaid / 1000;
    let mut mass_number = zaid % 1000;

    let metastable = match scheme {
        MetastableScheme::Mcnp => {
            if zaid > 1_000_000 {
                // Newer SZA form: the leading digits carry the state.
                z %= 1000;
                if zaid == 1_095_242 {
                    0
                } else {
                    zaid / 1_000_000
                }
            } else if zaid == 95242 {
                1
            } else if zaid == 95642 {
                0
            } else {
                i64::from(mass_number > 300)
            }
        }
        MetastableScheme::Nndc => i64::from(mass_number > 300),
    };

    // Undo the offset the state was encoded with, until the mass number is
    // physically possible.
    while mass_number > 3 * z {
        mass_number -= 100;
    }

    let element = ATOMIC_SYMBOL
        .get(z as usize)
        .copied()
        .ok_or_else(|| Error::UnknownElement {
            symbol: format!("Z={z}"),
        })?;

    Ok(Metadata {
        name: gnds_name(z as u32, mass_number as u32, metastable as u32),
        element: element.to_string(),
        z: z as u32,
        mass_number: mass_number as u32,
        metastable: metastable as u32,
    })
}

/// The kind of data an ACE table holds, from the letter its suffix ends with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    NeutronContinuous,
    NeutronDiscrete,
    ThermalScattering,
    Dosimetry,
    Photoatomic,
    Photonuclear,
    Proton,
    Deuteron,
    Triton,
    Helium3,
    Alpha,
}

impl TableType {
    /// The letter the format uses for this type.
    pub fn suffix(self) -> char {
        match self {
            TableType::NeutronContinuous => 'c',
            TableType::NeutronDiscrete => 'd',
            TableType::ThermalScattering => 't',
            TableType::Dosimetry => 'y',
            TableType::Photoatomic => 'p',
            TableType::Photonuclear => 'u',
            TableType::Proton => 'h',
            TableType::Deuteron => 'o',
            TableType::Triton => 'r',
            TableType::Helium3 => 's',
            TableType::Alpha => 'a',
        }
    }

    /// The type a suffix denotes, e.g. `"70c"` is a continuous neutron table.
    pub fn from_suffix(suffix: &str) -> Result<TableType> {
        const ALL: [TableType; 11] = [
            TableType::NeutronContinuous,
            TableType::NeutronDiscrete,
            TableType::ThermalScattering,
            TableType::Dosimetry,
            TableType::Photoatomic,
            TableType::Photonuclear,
            TableType::Proton,
            TableType::Deuteron,
            TableType::Triton,
            TableType::Helium3,
            TableType::Alpha,
        ];
        ALL.into_iter()
            .find(|t| suffix.ends_with(t.suffix()))
            .ok_or_else(|| Error::BadAceTable {
                what: format!("suffix {suffix:?} has no corresponding ACE table type"),
            })
    }
}

/// One ACE cross section table.
///
/// `nxs`, `jxs` and `xss` each carry an unused element at index 0, so that the
/// one-based indices the ACE specification uses can be written directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// Full identifier, e.g. `"92235.70c"`.
    pub name: String,
    pub atomic_weight_ratio: f64,
    /// Temperature in MeV, as the file stores it.
    pub kt: f64,
    /// The (IZ, AW) pairs of the header.
    pub pairs: Vec<(i64, f64)>,
    pub nxs: Vec<i64>,
    pub jxs: Vec<i64>,
    pub xss: Vec<f64>,
}

impl Table {
    /// The ZAID, the part of the name before the dot.
    pub fn zaid(&self) -> Result<i64> {
        self.name
            .split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::BadAceTable {
                what: format!("table name {:?} has no ZAID", self.name),
            })
    }

    /// What kind of data this table holds.
    pub fn data_type(&self) -> Result<TableType> {
        let suffix = self
            .name
            .split('.')
            .nth(1)
            .ok_or_else(|| Error::BadAceTable {
                what: format!("table name {:?} has no suffix", self.name),
            })?;
        TableType::from_suffix(suffix)
    }

    /// Temperature in kelvin.
    ///
    /// [`Table::kt`] is the raw value in MeV; the two differ by more than ten
    /// orders of magnitude.
    pub fn temperature(&self) -> f64 {
        self.kt * EV_PER_MEV / K_BOLTZMANN
    }
}

/// Parse a float as an ACE file writes it.
///
/// Mostly ordinary, but NJOY drops the `e` from values below 1e-100, writing
/// `1.234567-120`. Unlike an ENDF field this is not fixed-width, so the
/// eleven-character rule of [`crate::records::float_endf`] does not apply.
fn parse_float(s: &str) -> Result<f64> {
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
    }
    // Put the exponent marker back: the sign that follows a digit or a point,
    // with no marker already present.
    let bytes = s.as_bytes();
    for i in 1..bytes.len() {
        if (bytes[i] == b'+' || bytes[i] == b'-')
            && (bytes[i - 1].is_ascii_digit() || bytes[i - 1] == b'.')
        {
            let patched = format!("{}e{}", &s[..i], &s[i..]);
            if let Ok(v) = patched.parse::<f64>() {
                return Ok(v);
            }
        }
    }
    Err(Error::BadAceTable {
        what: format!("{s:?} is not a number"),
    })
}

fn parse_int(s: &str) -> Result<i64> {
    s.parse().map_err(|_| Error::BadAceTable {
        what: format!("{s:?} is not an integer"),
    })
}

/// Read every table in an ASCII ACE file.
pub fn get_tables(path: impl AsRef<Path>) -> Result<Vec<Table>> {
    let text = std::fs::read_to_string(path)?;
    tables_from_str(&text, None)
}

/// Read one named table, e.g. `"3006.01c"`.
pub fn get_table(path: impl AsRef<Path>, name: &str) -> Result<Table> {
    let text = std::fs::read_to_string(path)?;
    let wanted = [name.to_string()];
    tables_from_str(&text, Some(&wanted))?
        .into_iter()
        .next()
        .ok_or_else(|| Error::BadAceTable {
            what: format!("no table named {name:?} in the file"),
        })
}

/// Read tables from the text of an ASCII ACE file.
///
/// `wanted` restricts which tables are read; `None` reads all of them.
pub fn tables_from_str(text: &str, wanted: Option<&[String]>) -> Result<Vec<Table>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut tables = Vec::new();
    let mut at = 0usize;

    while at < lines.len() && !lines[at].trim().is_empty() {
        let header = lines[at];
        let first_word = header.split_whitespace().next().unwrap_or("");

        // A 2.0-style header opens with a version like "2.0.1", so its second
        // character is a dot. A 1.0-style one opens with the table name.
        let two_point_zero = first_word.as_bytes().get(1) == Some(&b'.');

        let (name, atomic_weight_ratio, kt, body) = if two_point_zero {
            let words: Vec<&str> = header.split_whitespace().collect();
            let name = words.get(1).copied().unwrap_or("").to_string();
            let second: Vec<&str> = lines
                .get(at + 1)
                .copied()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            let awr = parse_float(second.first().copied().unwrap_or("0"))?;
            let kt = parse_float(second.get(1).copied().unwrap_or("0"))?;
            let comment_lines: usize = parse_int(second.get(3).copied().unwrap_or("0"))? as usize;
            // The comment lines sit between the 2.0 header and the body that
            // is otherwise laid out exactly as the 1.0 form.
            (name, awr, kt, at + comment_lines)
        } else {
            let words: Vec<&str> = header.split_whitespace().collect();
            let name = words.first().copied().unwrap_or("").to_string();
            let awr = parse_float(words.get(1).copied().unwrap_or("0"))?;
            let kt = parse_float(words.get(2).copied().unwrap_or("0"))?;
            (name, awr, kt, at)
        };

        let line = |i: usize| lines.get(body + i).copied().unwrap_or("");

        // The (IZ, AW) pairs occupy four lines of four pairs each.
        let pair_words: Vec<&str> = (2..6).flat_map(|i| line(i).split_whitespace()).collect();
        let mut pairs = Vec::with_capacity(pair_words.len() / 2);
        for chunk in pair_words.chunks(2) {
            if chunk.len() == 2 {
                pairs.push((parse_int(chunk[0])?, parse_float(chunk[1])?));
            }
        }

        // NXS and JXS each get a leading zero so the format's one-based
        // indices can be used directly.
        let mut nxs = vec![0i64];
        for w in (6..8).flat_map(|i| line(i).split_whitespace()) {
            nxs.push(parse_int(w)?);
        }
        let mut jxs = vec![0i64];
        for w in (8..ACE_HEADER_SIZE).flat_map(|i| line(i).split_whitespace()) {
            jxs.push(parse_int(w)?);
        }

        let length = nxs.get(1).copied().unwrap_or(0).max(0) as usize;
        // Four values to a line.
        let n_lines = length.div_ceil(4);

        let skip_this = wanted.is_some_and(|names| !names.iter().any(|n| n == &name));
        if skip_this {
            at = body + ACE_HEADER_SIZE + n_lines;
            continue;
        }

        let mut xss = Vec::with_capacity(length + 1);
        xss.push(0.0);
        for i in 0..n_lines {
            for w in line(ACE_HEADER_SIZE + i).split_whitespace() {
                xss.push(parse_float(w)?);
            }
        }
        if xss.len() != length + 1 {
            return Err(Error::BadAceTable {
                what: format!(
                    "table {name:?} declares {length} values in XSS but {} were read",
                    xss.len() - 1
                ),
            });
        }

        tables.push(Table {
            name,
            atomic_weight_ratio,
            kt,
            pairs,
            nxs,
            jxs,
            xss,
        });

        at = body + ACE_HEADER_SIZE + n_lines;

        // Stop once every requested table has been found.
        if let Some(names) = wanted {
            if tables.len() == names.len() {
                break;
            }
        }
    }

    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zaids_are_read_the_way_each_library_encodes_them() {
        // Ordinary nuclide, both schemes agree.
        let m = get_metadata(3006, MetastableScheme::Nndc).unwrap();
        assert_eq!(
            (m.name.as_str(), m.z, m.mass_number, m.metastable),
            ("Li6", 3, 6, 0)
        );
        assert_eq!(m.element, "Li");

        // NNDC puts the state in the hundreds digit of A.
        let m = get_metadata(95342, MetastableScheme::Nndc).unwrap();
        assert_eq!(
            (m.name.as_str(), m.mass_number, m.metastable),
            ("Am242_m1", 242, 1)
        );

        // MCNP's two exceptions: 95242 is the metastable one, 95642 the ground
        // state, which is the reverse of what the digits suggest.
        let m = get_metadata(95242, MetastableScheme::Mcnp).unwrap();
        assert_eq!((m.name.as_str(), m.metastable), ("Am242_m1", 1));
        let m = get_metadata(95642, MetastableScheme::Mcnp).unwrap();
        assert_eq!((m.name.as_str(), m.metastable), ("Am242", 0));
        // And the newer SZA form, where the ground state is spelled out.
        let m = get_metadata(1_095_242, MetastableScheme::Mcnp).unwrap();
        assert_eq!((m.name.as_str(), m.metastable), ("Am242", 0));
    }

    #[test]
    fn suffixes_map_to_table_types() {
        assert_eq!(
            TableType::from_suffix("70c").unwrap(),
            TableType::NeutronContinuous
        );
        assert_eq!(
            TableType::from_suffix("12p").unwrap(),
            TableType::Photoatomic
        );
        assert_eq!(
            TableType::from_suffix("00t").unwrap(),
            TableType::ThermalScattering
        );
        assert!(TableType::from_suffix("70z").is_err());
    }

    #[test]
    fn floats_parse_including_the_form_njoy_writes_below_1e_100() {
        assert_eq!(parse_float("1.5").unwrap(), 1.5);
        assert_eq!(parse_float("-2.5E-3").unwrap(), -2.5e-3);
        // NJOY drops the 'e' when the exponent needs three digits.
        assert_eq!(parse_float("1.234567-120").unwrap(), 1.234567e-120);
        // The exact tokens that made TENDL-2025's Db262 and its neighbours
        // unreadable: a three-digit exponent overflows the field NJOY has to
        // write it in, so the `e` goes. See issue #20, and
        // `tests/synthetic-denormal.ace.xz`, which pins both readers on a
        // whole file.
        assert_eq!(
            parse_float("6.10562372605-318").unwrap(),
            6.10562372605e-318
        );
        assert_eq!(
            parse_float("-6.10562372605-318").unwrap(),
            -6.10562372605e-318
        );
        // Rounded to the nearest subnormal, as any correct parser does.
        assert_eq!(parse_float("9.99999999999-323").unwrap(), 1.0e-322);
        assert_eq!(parse_float("-9.8+100").unwrap(), -9.8e100);
        assert!(parse_float("banana").is_err());
    }
}
