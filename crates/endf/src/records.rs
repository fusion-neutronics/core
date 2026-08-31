//! The ENDF-6 record primitives: TEXT, CONT, HEAD, LIST, TAB1, TAB2 and INTG.
//!
//! ENDF-6 is a fixed-width line format. Every record is built from 11-character
//! fields, and the last 14 columns of each line carry the MAT/MF/MT control
//! fields that [`crate::material`] uses to split a file into sections.

use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};

/// Extract a fixed-width field, tolerating lines shorter than the format
/// requires.
///
/// Evaluators do trim trailing blanks, so a CONT record's last field is
/// routinely missing from the line entirely. Python slicing yields `""` there
/// and [`int_endf`] reads that as zero; this keeps that behaviour rather than
/// panicking on the range.
#[inline]
pub fn field(line: &str, start: usize, end: usize) -> &str {
    let len = line.len();
    let start = start.min(len);
    let end = end.min(len);
    if start >= end {
        return "";
    }
    // `get` returns None rather than panicking if a non-ASCII byte means the
    // range is not on a char boundary.
    line.get(start..end).unwrap_or("")
}

/// Convert an ENDF floating point field to `f64`.
///
/// ENDF-6 uses an "e-less" exponential notation, e.g. `-1.23481+10`, which the
/// usual float parsers reject. Whitespace anywhere in the field is ignored, an
/// all-blank field is zero, and `e`, `E`, `d` and `D` are all accepted as the
/// exponent marker. Only the first 11 characters are read, matching the width
/// of an ENDF field.
pub fn float_endf(s: &str) -> f64 {
    let bytes = s.as_bytes();
    let n = bytes.len().min(11);

    // 11 characters, plus at most one inserted `e`, plus room to spare.
    let mut buf = [0u8; 13];
    let mut j = 0usize;
    let mut found_significand = false;
    let mut found_exponent = false;

    for &c in &bytes[..n] {
        if c == b' ' {
            continue;
        }
        if found_significand {
            if !found_exponent {
                if c == b'+' || c == b'-' {
                    // A sign after the significand with no marker in between is
                    // the e-less exponent: supply the `e` ourselves.
                    buf[j] = b'e';
                    j += 1;
                    found_exponent = true;
                } else if matches!(c, b'e' | b'E' | b'd' | b'D') {
                    buf[j] = b'e';
                    j += 1;
                    found_exponent = true;
                    continue;
                }
            }
        } else if c == b'.' || c.is_ascii_digit() {
            found_significand = true;
        }
        buf[j] = c;
        j += 1;
    }

    let Ok(text) = std::str::from_utf8(&buf[..j]) else {
        return 0.0;
    };
    // C's atof takes the longest parseable prefix and yields zero when there is
    // none; Rust's parse is all-or-nothing, so walk the end back to match.
    if let Ok(v) = text.parse::<f64>() {
        return v;
    }
    for end in (1..text.len()).rev() {
        if let Ok(v) = text[..end].parse::<f64>() {
            return v;
        }
    }
    0.0
}

/// Convert an ENDF integer field to `i64`.
///
/// The format allows an integer to be written as an all-blank field, which
/// means zero.
pub fn int_endf(s: &str) -> i64 {
    let t = s.trim();
    if t.is_empty() {
        return 0;
    }
    t.parse::<i64>().unwrap_or(0)
}

/// The six fields of a CONT record.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cont {
    pub c1: f64,
    pub c2: f64,
    pub l1: i64,
    pub l2: i64,
    pub n1: i64,
    pub n2: i64,
}

/// A HEAD record: a CONT whose first two fields are ZA and AWR.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Head {
    pub za: i64,
    pub awr: f64,
    pub l1: i64,
    pub l2: i64,
    pub n1: i64,
    pub n2: i64,
}

/// A LIST record: a CONT header followed by `n1` floating point values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ListRecord {
    pub cont: Cont,
    pub values: Vec<f64>,
}

/// A TAB1 record: four header fields plus the tabulated function itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Tab1 {
    pub c1: f64,
    pub c2: f64,
    pub l1: i64,
    pub l2: i64,
    pub table: Tabulated1D,
}

/// A TAB2 record: a CONT header plus the interpolation rules for the
/// subrecords that follow it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tab2 {
    pub cont: Cont,
    pub table: Tabulated2D,
}

/// A square matrix, used for the correlation matrix an INTG record encodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    n: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// The `n` by `n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix {
            n,
            data: vec![0.0; n * n],
        };
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn order(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    /// Row-major values.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Reflect the lower triangle onto the upper, leaving the diagonal alone.
    fn symmetrize(&mut self) {
        for i in 0..self.n {
            for j in 0..i {
                let v = self.get(i, j) + self.get(j, i);
                self.set(i, j, v);
                self.set(j, i, v);
            }
        }
    }
}

/// A cursor over the lines of one ENDF section.
///
/// Records are read in sequence, mirroring how the format is defined: each
/// `*_record` call consumes as many lines as that record occupies.
pub struct Reader<'a> {
    lines: Vec<&'a str>,
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(text: &'a str) -> Self {
        Reader {
            lines: text.lines().collect(),
            pos: 0,
        }
    }

    /// Lines not yet consumed.
    pub fn remaining(&self) -> usize {
        self.lines.len().saturating_sub(self.pos)
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn next_line(&mut self, expected: &'static str) -> Result<&'a str> {
        let line = self
            .lines
            .get(self.pos)
            .copied()
            .ok_or(Error::UnexpectedEof { expected })?;
        self.pos += 1;
        Ok(line)
    }

    /// Skip `n` lines without interpreting them.
    ///
    /// Used for records the format defines but nothing reads, such as MF=4's
    /// obsolete energy transformation matrix.
    pub fn skip_lines(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            self.next_line("lines to skip")?;
        }
        Ok(())
    }

    /// A TEXT record: the 66 columns before the control fields.
    pub fn text_record(&mut self) -> Result<&'a str> {
        let line = self.next_line("a TEXT record")?;
        Ok(field(line, 0, 66))
    }

    pub fn cont_record(&mut self) -> Result<Cont> {
        let line = self.next_line("a CONT record")?;
        Ok(Cont {
            c1: float_endf(field(line, 0, 11)),
            c2: float_endf(field(line, 11, 22)),
            l1: int_endf(field(line, 22, 33)),
            l2: int_endf(field(line, 33, 44)),
            n1: int_endf(field(line, 44, 55)),
            n2: int_endf(field(line, 55, 66)),
        })
    }

    /// Read the next CONT record without consuming it.
    ///
    /// The format sometimes puts the flag that says how to read a record
    /// inside the record itself (MF=33's LB is the case here) so the reader
    /// has to look before it leaps. The Python reader does the same with
    /// `tell` and `seek`.
    pub fn peek_cont_record(&mut self) -> Result<Cont> {
        let mark = self.pos;
        let cont = self.cont_record();
        self.pos = mark;
        cont
    }

    pub fn head_record(&mut self) -> Result<Head> {
        let c = self.cont_record()?;
        Ok(Head {
            // ZA is written as a float but is conceptually an integer.
            za: c.c1 as i64,
            awr: c.c2,
            l1: c.l1,
            l2: c.l2,
            n1: c.n1,
            n2: c.n2,
        })
    }

    pub fn list_record(&mut self) -> Result<ListRecord> {
        let cont = self.cont_record()?;
        let npl = cont.n1.max(0) as usize;
        let mut values = Vec::with_capacity(npl);
        for _ in 0..npl.div_ceil(6) {
            let line = self.next_line("the values of a LIST record")?;
            let n = (npl - values.len()).min(6);
            for j in 0..n {
                values.push(float_endf(field(line, 11 * j, 11 * (j + 1))));
            }
        }
        Ok(ListRecord { cont, values })
    }

    /// Read the `(NBT, INT)` interpolation pairs shared by TAB1 and TAB2.
    fn interpolation_pairs(&mut self, n_regions: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut breakpoints = Vec::with_capacity(n_regions);
        let mut interpolation = Vec::with_capacity(n_regions);
        for _ in 0..n_regions.div_ceil(3) {
            let line = self.next_line("the interpolation regions of a TAB record")?;
            let n = (n_regions - breakpoints.len()).min(3);
            for j in 0..n {
                let o = 22 * j;
                breakpoints.push(int_endf(field(line, o, o + 11)) as i32);
                interpolation.push(int_endf(field(line, o + 11, o + 22)) as i32);
            }
        }
        Ok((breakpoints, interpolation))
    }

    pub fn tab1_record(&mut self) -> Result<Tab1> {
        let line = self.next_line("a TAB1 record")?;
        let c1 = float_endf(field(line, 0, 11));
        let c2 = float_endf(field(line, 11, 22));
        let l1 = int_endf(field(line, 22, 33));
        let l2 = int_endf(field(line, 33, 44));
        let n_regions = int_endf(field(line, 44, 55)).max(0) as usize;
        let n_pairs = int_endf(field(line, 55, 66)).max(0) as usize;

        let (breakpoints, interpolation) = self.interpolation_pairs(n_regions)?;

        let mut x = Vec::with_capacity(n_pairs);
        let mut y = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs.div_ceil(3) {
            let line = self.next_line("the (x, y) pairs of a TAB1 record")?;
            let n = (n_pairs - x.len()).min(3);
            for j in 0..n {
                let o = 22 * j;
                x.push(float_endf(field(line, o, o + 11)));
                y.push(float_endf(field(line, o + 11, o + 22)));
            }
        }

        Ok(Tab1 {
            c1,
            c2,
            l1,
            l2,
            table: Tabulated1D::with_regions(x, y, breakpoints, interpolation),
        })
    }

    pub fn tab2_record(&mut self) -> Result<Tab2> {
        let cont = self.cont_record()?;
        let n_regions = cont.n1.max(0) as usize;
        let (breakpoints, interpolation) = self.interpolation_pairs(n_regions)?;
        Ok(Tab2 {
            cont,
            table: Tabulated2D {
                breakpoints,
                interpolation,
            },
        })
    }

    /// An INTG record: a correlation matrix in the format's compact integer
    /// encoding.
    pub fn intg_record(&mut self) -> Result<Matrix> {
        let items = self.cont_record()?;
        let ndigit = items.l1;
        let npar = items.l2.max(0) as usize;
        let nlines = items.n1.max(0) as usize;

        let nrow: usize = match ndigit {
            2 => 18,
            3 => 12,
            4 => 11,
            5 => 9,
            6 => 8,
            _ => return Err(Error::BadNdigit { ndigit }),
        };
        let width = (ndigit + 1) as usize;
        let factor = 10f64.powi(ndigit as i32);

        let mut corr = Matrix::identity(npar);
        for _ in 0..nlines {
            let line = self.next_line("the rows of an INTG record")?;
            let ii = int_endf(field(line, 0, 5)) - 1;
            let jj = int_endf(field(line, 5, 10)) - 1;
            if ii < 0 || jj < 0 || ii as usize >= npar {
                continue;
            }
            for j in 0..nrow {
                if jj + j as i64 >= ii {
                    break;
                }
                let o = 11 + width * j;
                let element = int_endf(field(line, o, o + width));
                // NOTE: the column written to is `jj`, not `jj + j`. This
                // mirrors endf-python's reader exactly, including what looks
                // like an upstream indexing bug, so the two agree while the
                // port is validated against it. Revisit against ENDF-102 §33
                // before this crate is the only reader.
                let (i, k) = (ii as usize, jj as usize);
                if element > 0 {
                    corr.set(i, k, (element as f64 + 0.5) / factor);
                } else if element < 0 {
                    corr.set(i, k, (element as f64 - 0.5) / factor);
                }
            }
        }
        corr.symmetrize();
        Ok(corr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported one-for-one from tests/test_records.py so the two readers are
    // held to the same cases.

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} != {b}");
    }

    #[test]
    fn float_sign() {
        approx(float_endf("+3.2146"), 3.2146);
        approx(float_endf("-2.225002+6"), -2.225002e6);
    }

    #[test]
    fn float_no_leading_digit() {
        approx(float_endf(".12345"), 0.12345);
    }

    #[test]
    fn float_double_digit_exponent() {
        approx(float_endf("6.022+23"), 6.022e23);
        approx(float_endf("6.022-23"), 6.022e-23);
    }

    #[test]
    fn float_whitespace() {
        approx(float_endf(" +1.01+ 2"), 101.0);
        approx(float_endf(" -1.01- 2"), -0.0101);
        approx(float_endf("+ 2 . 3+ 1"), 23.0);
        approx(float_endf("-7 .8 -1"), -0.78);
    }

    // 3.14 here is test data for the exponent marker, not an approximation of pi.
    #[test]
    #[allow(clippy::approx_constant)]
    fn float_e_exponent() {
        approx(float_endf("3.14e0"), 3.14);
        approx(float_endf("3.14E0"), 3.14);
        approx(float_endf("3.14e-1"), 0.314);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn float_d_exponent() {
        approx(float_endf("3.14d0"), 3.14);
        approx(float_endf("3.14D0"), 3.14);
        approx(float_endf("3.14d-1"), 0.314);
    }

    #[test]
    fn float_only_leading_digit() {
        approx(float_endf("1+2"), 100.0);
        approx(float_endf("-1+2"), -100.0);
        approx(float_endf("1.+2"), 100.0);
        approx(float_endf("-1.+2"), -100.0);
    }

    #[test]
    fn float_empty() {
        assert_eq!(float_endf("        "), 0.0);
        assert_eq!(float_endf(""), 0.0);
    }

    #[test]
    fn float_buffer_size() {
        // Only the first 11 characters are part of the field.
        approx(float_endf("9.876540000000000"), 9.87654);
    }

    #[test]
    fn int_blank_is_zero() {
        assert_eq!(int_endf("           "), 0);
        assert_eq!(int_endf("        145"), 145);
        assert_eq!(int_endf("         -1"), -1);
    }

    #[test]
    fn field_tolerates_short_lines() {
        assert_eq!(field("abc", 0, 11), "abc");
        assert_eq!(field("abc", 5, 11), "");
    }

    #[test]
    fn reads_a_head_record() {
        let text = " 9.524400+4 2.419680+2          0          1          0          29552 1451\n";
        let mut r = Reader::new(text);
        let head = r.head_record().unwrap();
        assert_eq!(head.za, 95244);
        approx(head.awr, 241.968);
        assert_eq!(head.l2, 1);
        assert_eq!(head.n2, 2);
        assert!(r.is_empty());
    }

    #[test]
    fn reads_a_tab1_record() {
        let text = concat!(
            " 0.000000+0 0.000000+0          0          0          1          3\n",
            "          3          2\n",
            " 1.000000+0 2.000000+0 3.000000+0 4.000000+0 5.000000+0 6.000000+0\n",
        );
        let mut r = Reader::new(text);
        let tab = r.tab1_record().unwrap();
        assert_eq!(tab.table.x, vec![1.0, 3.0, 5.0]);
        assert_eq!(tab.table.y, vec![2.0, 4.0, 6.0]);
        assert_eq!(tab.table.breakpoints, vec![3]);
        assert_eq!(tab.table.interpolation, vec![2]);
    }

    #[test]
    fn running_off_the_end_is_an_error() {
        let mut r = Reader::new("");
        assert!(r.cont_record().is_err());
    }
}
