//! Splitting an ENDF-6 file into materials and their (MF, MT) sections.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::mf;
use crate::records::{field, Reader};

/// A parsed section, one variant per MF/MT family.
///
/// [`Section::Unparsed`] covers the files not yet ported to Rust. Their text is
/// still kept in [`Material::section_text`], so a consumer can fall back to the
/// Python reader for them while the port proceeds file by file.
#[derive(Debug, Clone, PartialEq)]
pub enum Section {
    /// MF=1 MT=451, descriptive data and the section directory.
    Mf1Mt451(Box<mf::mf1::Mf1Mt451>),
    /// MF=1 MT=452 or MT=456, total or prompt neutrons per fission.
    Mf1Mt452(mf::mf1::Mf1Mt452),
    /// MF=1 MT=455, delayed neutron data.
    Mf1Mt455(mf::mf1::Mf1Mt455),
    /// MF=1 MT=458, components of the fission energy release.
    Mf1Mt458(mf::mf1::Mf1Mt458),
    /// MF=1 MT=460, delayed photon data.
    Mf1Mt460(mf::mf1::Mf1Mt460),
    /// MF=2 MT=151, resonance parameters.
    Mf2(Box<mf::mf2::Mf2>),
    /// MF=3, reaction cross sections.
    Mf3(mf::mf3::Mf3),
    /// MF=4, angular distributions.
    Mf4(Box<mf::mf4::Mf4>),
    /// MF=5, energy distributions.
    Mf5(Box<mf::mf5::Mf5>),
    /// MF=6, energy-angle distributions of reaction products.
    Mf6(Box<mf::mf6::Mf6>),
    /// MF=7 MT=2, elastic thermal scattering.
    Mf7Mt2(Box<mf::mf7::Mf7Mt2>),
    /// MF=7 MT=4, incoherent inelastic thermal scattering.
    Mf7Mt4(Box<mf::mf7::Mf7Mt4>),
    /// MF=7 MT=451, thermal scattering general information.
    Mf7Mt451(Box<mf::mf7::Mf7Mt451>),
    /// MF=8, radioactive nuclide production.
    Mf8(Box<mf::mf8::Mf8>),
    /// MF=8 MT=454 or MT=459, fission product yields.
    Mf8Mt454(Box<mf::mf8::Mf8Mt454>),
    /// MF=8 MT=457, radioactive decay data.
    Mf8Mt457(Box<mf::mf8::Mf8Mt457>),
    /// MF=9 or MF=10, isomer multiplicities and production cross sections.
    Mf9Mf10(Box<mf::mf8::Mf9Mf10>),
    /// MF=12, photon production multiplicities.
    Mf12(Box<mf::photon::Mf12>),
    /// MF=13, photon production cross sections.
    Mf13(Box<mf::photon::Mf13>),
    /// MF=14, photon angular distributions.
    Mf14(Box<mf::photon::Mf14>),
    /// MF=15, continuous photon energy spectra.
    Mf15(Box<mf::photon::Mf15>),
    /// MF=23, photo-atomic and electro-atomic cross sections.
    Mf23(Box<mf::atomic::Mf23>),
    /// MF=26, electro-atomic secondary distributions.
    Mf26(Box<mf::atomic::Mf26>),
    /// MF=27, atomic form factors and scattering functions.
    Mf27(Box<mf::atomic::Mf27>),
    /// MF=28, atomic relaxation data.
    Mf28(Box<mf::atomic::Mf28>),
    /// MF=33, covariances of neutron cross sections.
    Mf33(Box<mf::covariance::Mf33>),
    /// MF=34, covariances of angular distributions.
    Mf34(Box<mf::covariance::Mf34>),
    /// MF=40, covariances of radionuclide production.
    Mf40(Box<mf::covariance::Mf40>),
    Unparsed {
        mf: i32,
        mt: i32,
    },
}

/// An ENDF material: one evaluation, made up of (MF, MT) sections.
#[derive(Debug, Clone)]
pub struct Material {
    /// ENDF material number.
    pub mat: i32,
    /// Raw text of each section, keyed by (MF, MT), control columns included.
    pub section_text: BTreeMap<(i32, i32), String>,
    /// Parsed form of each section.
    pub section_data: BTreeMap<(i32, i32), Section>,
}

/// Read a MAT/MF/MT control field.
///
/// A blank field reads as zero, which is what the format means by it. Anything
/// present but not an integer is a malformed line and an error, rather than
/// being silently taken as zero.
fn control_field(line: &str, start: usize, end: usize) -> Result<i32> {
    let s = field(line, start, end).trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse::<i32>().map_err(|_| Error::BadControlField {
        line: line.to_string(),
    })
}

/// The (MAT, MF, MT) control fields in the last 14 columns of a line.
fn control(line: &str) -> Result<(i32, i32, i32)> {
    Ok((
        control_field(line, 66, 70)?,
        control_field(line, 70, 72)?,
        control_field(line, 72, 75)?,
    ))
}

/// What [`Material::interpret`] built, one variant per sublibrary that has a
/// high-level interface.
#[derive(Debug, Clone, PartialEq)]
pub enum Interpretation {
    IncidentNeutron(Box<crate::IncidentNeutron>),
    IncidentPhoton(Box<crate::IncidentPhoton>),
}

impl Material {
    /// Parse the first material in `text`, skipping the leading TPID record.
    ///
    /// Not `std::str::FromStr`: that trait's `Err` would have to be this
    /// crate's error type anyway, and the inherent method keeps the name the
    /// Python reader uses.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Material> {
        let lines: Vec<&str> = text.lines().collect();
        let mut cursor = usize::from(!lines.is_empty());
        Material::parse(&lines, &mut cursor)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Material> {
        let text = std::fs::read_to_string(path)?;
        Material::from_str(&text)
    }

    /// Parse one material starting at `cursor`, leaving `cursor` on the line
    /// after the material's MEND record.
    fn parse(lines: &[&str], cursor: &mut usize) -> Result<Material> {
        // Evaluators sometimes write ill-formed TPID records, so the material
        // number is taken from the first line that actually belongs to a file
        // (MF != 0) rather than from the header.
        let mat = loop {
            let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                expected: "the start of a material",
            })?;
            let (m, mf, _) = control(line)?;
            if mf != 0 {
                break m;
            }
            *cursor += 1;
        };

        let mut section_text: BTreeMap<(i32, i32), String> = BTreeMap::new();
        loop {
            // Advance to the next section head, or to the end of the material.
            let (cur_mat, mf, mt) = loop {
                let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                    expected: "the next section",
                })?;
                let c = control(line)?;
                if c.2 > 0 || c.0 == 0 {
                    break c;
                }
                *cursor += 1;
            };

            // MAT=0 is the MEND record: this material is done.
            if cur_mat == 0 {
                *cursor += 1;
                break;
            }

            // Collect the section's lines up to its SEND record (MT=0).
            let mut body = String::new();
            loop {
                let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                    expected: "the end of a section",
                })?;
                *cursor += 1;
                if control(line)?.2 == 0 {
                    break;
                }
                body.push_str(line);
                body.push('\n');
            }
            section_text.insert((mf, mt), body);
        }

        let mut section_data = BTreeMap::new();
        for (&(mf, mt), text) in &section_text {
            section_data.insert((mf, mt), parse_section(mf, mt, text)?);
        }

        Ok(Material {
            mat,
            section_text,
            section_data,
        })
    }

    /// The (MF, MT) sections present, in ascending order.
    pub fn sections(&self) -> Vec<(i32, i32)> {
        self.section_text.keys().copied().collect()
    }

    pub fn contains(&self, mf: i32, mt: i32) -> bool {
        self.section_data.contains_key(&(mf, mt))
    }

    pub fn get(&self, mf: i32, mt: i32) -> Option<&Section> {
        self.section_data.get(&(mf, mt))
    }

    /// The MF=3 cross section for a reaction, if the material has one.
    pub fn mf3(&self, mt: i32) -> Option<&mf::mf3::Mf3> {
        match self.section_data.get(&(3, mt))? {
            Section::Mf3(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=4 angular distribution for a reaction.
    pub fn mf4(&self, mt: i32) -> Option<&mf::mf4::Mf4> {
        match self.section_data.get(&(4, mt))? {
            Section::Mf4(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=5 energy distributions for a reaction.
    pub fn mf5(&self, mt: i32) -> Option<&mf::mf5::Mf5> {
        match self.section_data.get(&(5, mt))? {
            Section::Mf5(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=6 product distributions for a reaction.
    pub fn mf6(&self, mt: i32) -> Option<&mf::mf6::Mf6> {
        match self.section_data.get(&(6, mt))? {
            Section::Mf6(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=8 radioactive production data for a reaction.
    pub fn mf8(&self, mt: i32) -> Option<&mf::mf8::Mf8> {
        match self.section_data.get(&(8, mt))? {
            Section::Mf8(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=8 MT=457 radioactive decay data.
    pub fn mf8_mt457(&self) -> Option<&mf::mf8::Mf8Mt457> {
        match self.section_data.get(&(8, 457))? {
            Section::Mf8Mt457(s) => Some(s),
            _ => None,
        }
    }

    /// Fission product yields: MT=454 independent, MT=459 cumulative.
    pub fn mf8_mt454(&self, mt: i32) -> Option<&mf::mf8::Mf8Mt454> {
        match self.section_data.get(&(8, mt))? {
            Section::Mf8Mt454(s) => Some(s),
            _ => None,
        }
    }

    /// MF=9 isomer multiplicities for a reaction.
    pub fn mf9(&self, mt: i32) -> Option<&mf::mf8::Mf9Mf10> {
        match self.section_data.get(&(9, mt))? {
            Section::Mf9Mf10(s) => Some(s),
            _ => None,
        }
    }

    /// MF=10 isomer production cross sections for a reaction.
    pub fn mf10(&self, mt: i32) -> Option<&mf::mf8::Mf9Mf10> {
        match self.section_data.get(&(10, mt))? {
            Section::Mf9Mf10(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=12 section for a reaction.
    pub fn mf12(&self, mt: i32) -> Option<&mf::photon::Mf12> {
        match self.section_data.get(&(12, mt))? {
            Section::Mf12(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=13 section for a reaction.
    pub fn mf13(&self, mt: i32) -> Option<&mf::photon::Mf13> {
        match self.section_data.get(&(13, mt))? {
            Section::Mf13(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=14 section for a reaction.
    pub fn mf14(&self, mt: i32) -> Option<&mf::photon::Mf14> {
        match self.section_data.get(&(14, mt))? {
            Section::Mf14(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=15 section for a reaction.
    pub fn mf15(&self, mt: i32) -> Option<&mf::photon::Mf15> {
        match self.section_data.get(&(15, mt))? {
            Section::Mf15(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=23 section for a reaction.
    pub fn mf23(&self, mt: i32) -> Option<&mf::atomic::Mf23> {
        match self.section_data.get(&(23, mt))? {
            Section::Mf23(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=26 section for a reaction.
    pub fn mf26(&self, mt: i32) -> Option<&mf::atomic::Mf26> {
        match self.section_data.get(&(26, mt))? {
            Section::Mf26(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=27 section for a reaction.
    pub fn mf27(&self, mt: i32) -> Option<&mf::atomic::Mf27> {
        match self.section_data.get(&(27, mt))? {
            Section::Mf27(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=28 section for a reaction.
    pub fn mf28(&self, mt: i32) -> Option<&mf::atomic::Mf28> {
        match self.section_data.get(&(28, mt))? {
            Section::Mf28(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=33 covariance section for a reaction.
    pub fn mf33(&self, mt: i32) -> Option<&mf::covariance::Mf33> {
        match self.section_data.get(&(33, mt))? {
            Section::Mf33(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=34 covariance section for a reaction.
    pub fn mf34(&self, mt: i32) -> Option<&mf::covariance::Mf34> {
        match self.section_data.get(&(34, mt))? {
            Section::Mf34(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=40 covariance section for a reaction.
    pub fn mf40(&self, mt: i32) -> Option<&mf::covariance::Mf40> {
        match self.section_data.get(&(40, mt))? {
            Section::Mf40(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=2 MT=151 resonance parameters.
    pub fn mf2(&self) -> Option<&mf::mf2::Mf2> {
        match self.section_data.get(&(2, 151))? {
            Section::Mf2(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=1 MT=451 descriptive data, which every material carries.
    pub fn mf1_mt451(&self) -> Option<&mf::mf1::Mf1Mt451> {
        match self.section_data.get(&(1, 451))? {
            Section::Mf1Mt451(s) => Some(s),
            _ => None,
        }
    }

    /// Build the high-level interface the material's sublibrary calls for.
    ///
    /// NSUB in MF=1 MT=451 says which: 3 is photoatomic, 10 incident neutron.
    /// A photoatomic evaluation is interpreted on its own, so any atomic
    /// relaxation data has to come from its own MF=28; pass a separate
    /// evaluation to [`IncidentPhoton::from_endf`] directly when there is one.
    pub fn interpret(&self) -> Result<Interpretation> {
        let nsub = self
            .mf1_mt451()
            .ok_or(Error::Unsupported {
                what: "a material with no MF=1 MT=451 to interpret",
            })?
            .nsub;
        match nsub {
            3 => Ok(Interpretation::IncidentPhoton(Box::new(
                crate::IncidentPhoton::from_endf(self, None)?,
            ))),
            10 => Ok(Interpretation::IncidentNeutron(Box::new(
                crate::IncidentNeutron::from_endf(self)?,
            ))),
            _ => Err(Error::Unsupported {
                what: "a sublibrary with no high-level interface (NSUB is not 3 or 10)",
            }),
        }
    }

    /// A neutron yield: MT=452 total, MT=456 prompt.
    pub fn mf1_mt452(&self, mt: i32) -> Option<&mf::mf1::Mf1Mt452> {
        match self.section_data.get(&(1, mt))? {
            Section::Mf1Mt452(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=1 MT=455 delayed neutron data.
    pub fn mf1_mt455(&self) -> Option<&mf::mf1::Mf1Mt455> {
        match self.section_data.get(&(1, 455))? {
            Section::Mf1Mt455(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=1 MT=458 fission energy release.
    pub fn mf1_mt458(&self) -> Option<&mf::mf1::Mf1Mt458> {
        match self.section_data.get(&(1, 458))? {
            Section::Mf1Mt458(s) => Some(s),
            _ => None,
        }
    }

    /// The MF=1 MT=460 delayed photon data.
    pub fn mf1_mt460(&self) -> Option<&mf::mf1::Mf1Mt460> {
        match self.section_data.get(&(1, 460))? {
            Section::Mf1Mt460(s) => Some(s),
            _ => None,
        }
    }

    /// The sublibrary number (NSUB) from MT=451, which says what kind of
    /// evaluation this is: 10 incident-neutron, 4 radioactive decay, and so on.
    pub fn nsub(&self) -> Option<i64> {
        Some(self.mf1_mt451()?.nsub)
    }
}

/// Dispatch a section to its file's parser.
///
/// Each MF ported to Rust adds an arm here; everything else stays
/// [`Section::Unparsed`] with its text preserved.
fn parse_section(mf: i32, mt: i32, text: &str) -> Result<Section> {
    let mut r = Reader::new(text);
    Ok(match (mf, mt) {
        (1, 451) => Section::Mf1Mt451(Box::new(mf::mf1::parse_mf1_mt451(&mut r)?)),
        (1, 452) | (1, 456) => Section::Mf1Mt452(mf::mf1::parse_mf1_mt452(&mut r)?),
        (1, 455) => Section::Mf1Mt455(mf::mf1::parse_mf1_mt455(&mut r)?),
        (1, 458) => Section::Mf1Mt458(mf::mf1::parse_mf1_mt458(&mut r)?),
        (1, 460) => Section::Mf1Mt460(mf::mf1::parse_mf1_mt460(&mut r)?),
        (2, 151) => Section::Mf2(Box::new(mf::mf2::parse_mf2(&mut r)?)),
        (3, _) => Section::Mf3(mf::mf3::parse_mf3(&mut r)?),
        (4, _) => Section::Mf4(Box::new(mf::mf4::parse_mf4(&mut r)?)),
        (5, _) => Section::Mf5(Box::new(mf::mf5::parse_mf5(&mut r)?)),
        (6, _) => Section::Mf6(Box::new(mf::mf6::parse_mf6(&mut r)?)),
        (7, 2) => Section::Mf7Mt2(Box::new(mf::mf7::parse_mf7_mt2(&mut r)?)),
        (7, 4) => Section::Mf7Mt4(Box::new(mf::mf7::parse_mf7_mt4(&mut r)?)),
        (7, 451) => Section::Mf7Mt451(Box::new(mf::mf7::parse_mf7_mt451(&mut r)?)),
        (8, 454) | (8, 459) => Section::Mf8Mt454(Box::new(mf::mf8::parse_mf8_mt454(&mut r)?)),
        (8, 457) => Section::Mf8Mt457(Box::new(mf::mf8::parse_mf8_mt457(&mut r)?)),
        (8, _) => Section::Mf8(Box::new(mf::mf8::parse_mf8(&mut r)?)),
        (9, _) | (10, _) => Section::Mf9Mf10(Box::new(mf::mf8::parse_mf9_mf10(&mut r, mf as i64)?)),
        (12, _) => Section::Mf12(Box::new(mf::photon::parse_mf12(&mut r)?)),
        (13, _) => Section::Mf13(Box::new(mf::photon::parse_mf13(&mut r)?)),
        (14, _) => Section::Mf14(Box::new(mf::photon::parse_mf14(&mut r)?)),
        (15, _) => Section::Mf15(Box::new(mf::photon::parse_mf15(&mut r)?)),
        (23, _) => Section::Mf23(Box::new(mf::atomic::parse_mf23(&mut r)?)),
        (26, _) => Section::Mf26(Box::new(mf::atomic::parse_mf26(&mut r)?)),
        (27, _) => Section::Mf27(Box::new(mf::atomic::parse_mf27(&mut r)?)),
        (28, _) => Section::Mf28(Box::new(mf::atomic::parse_mf28(&mut r)?)),
        (33, _) => Section::Mf33(Box::new(mf::covariance::parse_mf33(&mut r)?)),
        (34, _) => Section::Mf34(Box::new(mf::covariance::parse_mf34(&mut r, mt as i64)?)),
        (40, _) => Section::Mf40(Box::new(mf::covariance::parse_mf40(&mut r)?)),
        _ => Section::Unparsed { mf, mt },
    })
}

/// Read every material in an ENDF-6 file.
pub fn get_materials(path: impl AsRef<Path>) -> Result<Vec<Material>> {
    let text = std::fs::read_to_string(path)?;
    materials_from_str(&text)
}

/// Read every material in the text of an ENDF-6 file.
pub fn materials_from_str(text: &str) -> Result<Vec<Material>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    // Skip the TPID record that opens the file.
    let mut cursor = 1usize;
    let mut materials = Vec::new();
    while cursor < lines.len() {
        // MAT=-1 is the TEND record that closes the file.
        if control_field(lines[cursor], 66, 70)? == -1 {
            break;
        }
        let before = cursor;
        materials.push(Material::parse(&lines, &mut cursor)?);
        debug_assert!(cursor > before, "parsing a material must consume lines");
    }
    Ok(materials)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_the_material_number() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        assert_eq!(m.mat, 9552);
    }

    #[test]
    fn finds_the_expected_sections() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let sections = m.sections();
        // The evaluation opens with the descriptive MF=1/MT=451 section and
        // carries resonance parameters and cross sections.
        assert!(sections.contains(&(1, 451)));
        assert!(sections.contains(&(2, 151)));
        assert!(sections.contains(&(3, 1)));
        assert!(sections.contains(&(3, 2)));
        assert!(sections.contains(&(3, 18)));
        assert!(!sections.is_empty());
    }

    #[test]
    fn section_text_excludes_the_send_record() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let text = &m.section_text[&(3, 1)];
        for line in text.lines() {
            let (_, mf, mt) = control(line).unwrap();
            assert_eq!(mf, 3);
            assert_ne!(mt, 0, "a SEND record leaked into the section body");
        }
    }

    #[test]
    fn every_section_is_dispatched() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        assert_eq!(m.section_text.len(), m.section_data.len());
    }

    #[test]
    fn reads_the_whole_file_as_one_material() {
        let materials = materials_from_str(&crate::testdata::text(FIXTURE)).unwrap();
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0].mat, 9552);
    }

    #[test]
    fn interprets_by_sublibrary() {
        let neutron = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        match neutron.interpret().unwrap() {
            Interpretation::IncidentNeutron(n) => assert_eq!(n.name(), "Am244"),
            other => panic!("NSUB=10 gave {other:?}"),
        }

        const PHOTOATOMIC: &[u8] = include_bytes!("../fixtures/photoat-001_H_000.endf.xz");
        let photon = Material::from_str(&crate::testdata::text(PHOTOATOMIC)).unwrap();
        match photon.interpret().unwrap() {
            Interpretation::IncidentPhoton(p) => assert_eq!(p.atomic_number, 1),
            other => panic!("NSUB=3 gave {other:?}"),
        }
    }

    #[test]
    fn a_sublibrary_with_no_interface_says_so() {
        // Thermal scattering: NSUB=12, which has no high-level class in either
        // reader.
        const TSL: &[u8] = include_bytes!("../fixtures/tsl-s-CH4.endf.xz");
        let m = Material::from_str(&crate::testdata::text(TSL)).unwrap();
        assert!(m.interpret().is_err());
    }
}
