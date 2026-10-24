//! What each material in an ENDF-6 file is, read from the four records that
//! open its MF=1 MT=451 and nothing else.
//!
//! Those records say what an evaluation is and so what it is called: ZA and
//! LISO name the target, NSUB the sublibrary, NLIB and NVER the library it
//! came from. [`get_materials`](crate::get_materials) parses every section of
//! every material to reach them. [`get_headers`] reads the four records and,
//! on every other line, only the MAT, MF and MT columns, which are what show
//! where each material starts and ends.
//!
//! ```no_run
//! for header in endf::get_headers("IRDFF-II.endf")? {
//!     println!("MAT {} is {:?} (NSUB {})", header.mat, header.name(), header.nsub);
//! }
//! # Ok::<(), endf::Error>(())
//! ```

use std::io::BufRead;
use std::path::Path;

use crate::data::{gnds_name, ATOMIC_SYMBOL};
use crate::error::{Error, Result};
use crate::records::Reader;

/// The four records that open one material's MF=1 MT=451.
///
/// The fields have [`Mf1Mt451`](crate::mf::mf1::Mf1Mt451)'s names and hold
/// the values it holds. NWD and NXC are left out, because they only give the
/// length of the rest of the section, which is not read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    /// ENDF material number.
    pub mat: i32,
    /// 1000 Z + A of the target, with A = 0 for an element.
    pub za: i64,
    pub awr: f64,
    pub lrp: i64,
    pub lfi: i64,
    /// The library: 0 is ENDF/B, 2 JEFF, 6 JENDL and so on
    /// ([`library_name`](crate::njoy::library_name)).
    pub nlib: i64,
    pub nmod: i64,
    pub elis: f64,
    pub sta: f64,
    /// The target's excited level, which is not its isomeric state: In116m2
    /// is level 4.
    pub lis: i64,
    /// The target's isomeric state, 0 for the ground state.
    pub liso: i64,
    pub nfor: i64,
    pub awi: f64,
    pub emax: f64,
    pub lrel: i64,
    /// The sublibrary: 0 photonuclear, 3 photo-atomic, 4 radioactive decay,
    /// 6 atomic relaxation, 10 incident neutron, 11 neutron-induced fission
    /// yields, 12 thermal scattering, 113 electro-atomic.
    pub nsub: i64,
    pub nver: i64,
    pub temp: f64,
    pub ldrv: i64,
}

impl Header {
    /// The name the converters give this material's data: the GNDS name of
    /// the target, `"Am242_m1"`, or for the sublibraries that are per element
    /// (NSUB 3, 6 and 113) the element's symbol, `"Fe"`.
    ///
    /// `None` for thermal scattering data (NSUB 12), which is named for its
    /// compound and ZA does not give that, and for a Z with no element symbol.
    pub fn name(&self) -> Option<String> {
        let z = self.za / 1000;
        let symbol = *ATOMIC_SYMBOL.get(usize::try_from(z).ok()?)?;
        match self.nsub {
            12 => None,
            3 | 6 | 113 => Some(symbol.to_string()),
            _ => Some(gnds_name(
                z as u32,
                (self.za % 1000) as u32,
                self.liso as u32,
            )),
        }
    }
}

/// The header of every material in an ENDF-6 file, in file order.
///
/// The file is read a line at a time, so memory does not grow with its size.
/// Materials are found where [`get_materials`](crate::get_materials) finds
/// them, with two differences. `get_materials` always skips the first line
/// as the TPID record, so on a tape that has none and opens straight into
/// MF=1 MT=451 (JEFF-4.0's fission yield tape) it misreads the first
/// material; this reader sees that the first line is MF=1 MT=451 and reads
/// it. And it refuses what `get_materials` ignores after the TEND record, so
/// two files run together are an error rather than half read. It also reads
/// bytes that are not UTF-8 outside the four records it parses (IRDFF-II's
/// tape has a cp1252 dash in Zn68's description).
///
/// Fails with [`Error::NotEndf`] on a file with no material in it, or with a
/// line that has no MAT, MF and MT numbers before its first material: the
/// file is not ENDF-6 at all. Fails with [`Error::MalformedEndf`] on a file
/// that is ENDF-6 but cut short or broken: a line with no MAT, MF and MT
/// numbers once a material has started, a material with no MF=1 MT=451 or
/// with two, an MF=1 MT=451 shorter than its four opening records or with
/// one of them not UTF-8, a file that ends inside a material, and anything
/// but blank lines and further TEND records after the TEND record.
pub fn get_headers(path: impl AsRef<Path>) -> Result<Vec<Header>> {
    let file = std::fs::File::open(path)?;
    read(std::io::BufReader::with_capacity(1 << 16, file))
}

/// The header of every material in the text of an ENDF-6 file, as
/// [`get_headers`] reads a file.
pub fn headers_from_str(text: &str) -> Result<Vec<Header>> {
    read(text.as_bytes())
}

/// Where a line falls, the way [`get_materials`](crate::get_materials) walks
/// a file.
#[derive(PartialEq)]
enum Place {
    /// Before a material, where MEND and blank lines are skipped.
    Between,
    /// Inside a material, before the head of its next section.
    Seeking,
    /// Inside a section, before its SEND record.
    Section,
}

fn not_endf(what: String) -> Error {
    Error::NotEndf { what }
}

fn malformed(what: String) -> Error {
    Error::MalformedEndf { what }
}

/// MAT, MF and MT from columns 67 to 75, a blank field being zero as in
/// [`get_materials`](crate::get_materials). Read from the bytes, because the
/// columns are byte columns whatever the rest of the line holds.
fn control(line: &[u8]) -> Option<(i32, i32, i32)> {
    let field = |start: usize, end: usize| {
        let bytes = line.get(start..end.min(line.len())).unwrap_or_default();
        let text = std::str::from_utf8(bytes).ok()?.trim();
        if text.is_empty() {
            Some(0)
        } else {
            text.parse().ok()
        }
    };
    Some((field(66, 70)?, field(70, 72)?, field(72, 75)?))
}

fn read(mut input: impl BufRead) -> Result<Vec<Header>> {
    let mut headers = Vec::new();
    let mut line = Vec::new();
    let mut number = 0usize;
    let mut place = Place::Between;
    // The material being read: its MAT, its first line, and whether it has
    // had its MF=1 MT=451.
    let (mut mat, mut start, mut seen) = (0, 0, false);
    // The MF=1 MT=451 records read so far, and how many more there are to read.
    let mut opening = String::new();
    let mut pending = 0usize;
    // The line of the TEND record, once it has been read.
    let mut tend = None;

    loop {
        line.clear();
        if input.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        number += 1;
        // As `str::lines` ends a line.
        if line.last() == Some(&b'\n') {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }

        // Only blank lines and TEND again (the trimmed Am242 fixture writes
        // it twice) may follow the TEND record.
        if let Some(at) = tend {
            if line.iter().all(u8::is_ascii_whitespace)
                || control(&line).is_some_and(|(m, ..)| m == -1)
            {
                continue;
            }
            return Err(malformed(format!(
                "line {number} follows the TEND record at line {at}, which ends the tape"
            )));
        }

        let fields = control(&line);

        // The first line is the TPID record, unless the tape starts straight
        // into its first material.
        if number == 1 && !fields.is_some_and(|(_, mf, mt)| (mf, mt) == (1, 451)) {
            continue;
        }

        let (m, mf, mt) = fields.ok_or_else(|| {
            let shown: String = String::from_utf8_lossy(&line).chars().take(80).collect();
            let what = format!(
                "line {number} has no MAT, MF and MT numbers in columns 67 to 75: {shown:?}"
            );
            // Before any material this is not ENDF at all; inside one it is
            // an ENDF file that is broken.
            if headers.is_empty() && place == Place::Between {
                not_endf(what)
            } else {
                malformed(what)
            }
        })?;

        if place == Place::Between {
            // MAT=-1 is the TEND record that closes the tape.
            if m == -1 {
                tend = Some(number);
                continue;
            }
            if mf == 0 {
                continue;
            }
            (mat, start, seen) = (m, number, false);
            place = Place::Seeking;
        }

        if place == Place::Seeking {
            // MAT=0 is the MEND record that closes the material.
            if m == 0 {
                if !seen {
                    return Err(malformed(format!(
                        "MAT {mat} (lines {start} to {number}) has no MF=1 MT=451 section"
                    )));
                }
                place = Place::Between;
            } else if mt > 0 {
                place = Place::Section;
                if (mf, mt) == (1, 451) {
                    if seen {
                        return Err(malformed(format!(
                            "line {number} starts a second MF=1 MT=451 section in MAT {mat}, \
                             with no MEND record before it"
                        )));
                    }
                    seen = true;
                    opening.clear();
                    pending = 4;
                }
            }
        } else if mt == 0 {
            // A SEND record.
            if pending > 0 {
                return Err(malformed(format!(
                    "MF=1 MT=451 of MAT {mat} ends at line {number}, after {} of the four \
                     records that open it",
                    4 - pending
                )));
            }
            place = Place::Seeking;
            continue;
        }

        if place == Place::Section && pending > 0 {
            let text = std::str::from_utf8(&line).map_err(|_| {
                malformed(format!(
                    "line {number}, a record of MF=1 MT=451 in MAT {mat}, is not text"
                ))
            })?;
            opening.push_str(text);
            opening.push('\n');
            pending -= 1;
            if pending == 0 {
                headers.push(parse(mat, &opening)?);
            }
        }
    }

    if place != Place::Between {
        return Err(malformed(format!(
            "the file ends inside MAT {mat}, which has no MEND record"
        )));
    }
    if headers.is_empty() {
        return Err(not_endf("it holds no material".to_string()));
    }
    Ok(headers)
}

/// Read the four records that open MF=1 MT=451, as
/// [`parse_mf1_mt451`](crate::mf::mf1::parse_mf1_mt451) reads them.
fn parse(mat: i32, opening: &str) -> Result<Header> {
    let mut reader = Reader::new(opening);
    let head = reader.head_record()?;
    let c1 = reader.cont_record()?;
    let c2 = reader.cont_record()?;
    let c3 = reader.cont_record()?;
    Ok(Header {
        mat,
        za: head.za,
        awr: head.awr,
        lrp: head.l1,
        lfi: head.l2,
        nlib: head.n1,
        nmod: head.n2,
        elis: c1.c1,
        sta: c1.c2,
        lis: c1.l1,
        liso: c1.l2,
        nfor: c1.n2,
        awi: c2.c1,
        emax: c2.c2,
        lrel: c2.l1,
        nsub: c2.n1,
        nver: c2.n2,
        temp: c3.c1,
        ldrv: c3.l1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{materials_from_str, Material};
    use crate::testdata::text;

    const AM244: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");
    const IN116M1: &[u8] = include_bytes!("../fixtures/dec-049_In_116m1.endf.xz");
    const IN116M2: &[u8] = include_bytes!("../fixtures/dec-049_In_116m2.endf.xz");
    const HF177M1: &[u8] = include_bytes!("../fixtures/dec-072_Hf_177m1.endf.xz");
    const PHOTOATOMIC: &[u8] = include_bytes!("../fixtures/photoat-001_H_000.endf.xz");
    const RELAXATION: &[u8] = include_bytes!("../fixtures/atom-001_H_000.endf.xz");

    /// One record: its 66 columns, then MAT, MF and MT.
    fn record(body: &str, mat: i32, mf: i32, mt: i32) -> String {
        format!("{body:<66}{mat:>4}{mf:>2}{mt:>3}\n")
    }

    /// The materials of a file, which are everything between its TPID and
    /// TEND records.
    fn materials(file: &str) -> String {
        let lines: Vec<&str> = file.lines().collect();
        let tend = lines
            .iter()
            .rposition(|line| control(line.as_bytes()).is_some_and(|(mat, ..)| mat == -1))
            .expect("a TEND record");
        lines[1..tend]
            .iter()
            .map(|line| format!("{line}\n"))
            .collect()
    }

    /// Several files' materials in one tape, as a library ships a
    /// sublibrary: one TPID record, each material in turn, one TEND record.
    fn tape(files: &[&[u8]]) -> String {
        let mut out = record(" a tape", 1, 0, 0);
        for file in files {
            out.push_str(&materials(&text(file)));
        }
        out + &record("", -1, 0, 0)
    }

    /// What the full reader says the same materials are.
    fn full_reader(file: &str) -> Vec<Header> {
        materials_from_str(file)
            .unwrap()
            .iter()
            .map(|m| {
                let d = m.mf1_mt451().expect("MF=1 MT=451");
                Header {
                    mat: m.mat,
                    za: d.za,
                    awr: d.awr,
                    lrp: d.lrp,
                    lfi: d.lfi,
                    nlib: d.nlib,
                    nmod: d.nmod,
                    elis: d.elis,
                    sta: d.sta,
                    lis: d.lis,
                    liso: d.liso,
                    nfor: d.nfor,
                    awi: d.awi,
                    emax: d.emax,
                    lrel: d.lrel,
                    nsub: d.nsub,
                    nver: d.nver,
                    temp: d.temp,
                    ldrv: d.ldrv,
                }
            })
            .collect()
    }

    /// The name the high-level reader for the material's sublibrary gives
    /// it, where there is one.
    fn reader_name(m: &Material) -> Option<String> {
        Some(match m.nsub()? {
            3 | 113 => crate::IncidentPhoton::from_endf(m, None)
                .unwrap()
                .name()
                .to_string(),
            4 => crate::Decay::from_material(m).unwrap().nuclide.name,
            10 => crate::IncidentNeutron::from_endf(m).unwrap().name(),
            11 => {
                crate::FissionProductYields::from_material(m)
                    .unwrap()
                    .nuclide
                    .name
            }
            _ => return None,
        })
    }

    fn refusal(file: &str) -> String {
        match headers_from_str(file) {
            Err(e @ Error::NotEndf { .. }) => e.to_string(),
            other => panic!("expected NotEndf, got {other:?}"),
        }
    }

    fn broken(file: &str) -> String {
        match headers_from_str(file) {
            Err(e @ Error::MalformedEndf { .. }) => e.to_string(),
            other => panic!("expected MalformedEndf, got {other:?}"),
        }
    }

    #[test]
    fn every_fixture_reads_as_the_full_reader_reads_it() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let mut named = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let fixture = path.file_name().unwrap().to_string_lossy().into_owned();
            if !fixture.ends_with(".endf.xz") {
                continue;
            }
            let file = text(&std::fs::read(&path).unwrap());
            let headers = headers_from_str(&file).unwrap();
            assert_eq!(headers, full_reader(&file), "{fixture}");
            let material = Material::from_str(&file).unwrap();
            if let Some(name) = reader_name(&material) {
                assert_eq!(headers[0].name(), Some(name), "{fixture}");
                named.insert(headers[0].nsub);
            }
        }
        assert_eq!(named, [3, 4, 10, 11, 113].into(), "sublibraries named");
    }

    #[test]
    fn a_tape_gives_each_material_in_order() {
        let file = tape(&[AM244, IN116M1, PHOTOATOMIC]);
        let headers = headers_from_str(&file).unwrap();
        assert_eq!(headers, full_reader(&file));
        let seen: Vec<_> = headers.iter().map(|h| (h.mat, h.nsub, h.name())).collect();
        assert_eq!(
            seen,
            [
                (9552, 10, Some("Am244".to_string())),
                (1477, 4, Some("In116_m1".to_string())),
                (100, 3, Some("H".to_string())),
            ]
        );
    }

    #[test]
    fn a_tape_with_no_tpid_starts_with_its_first_material() {
        // JEFF-4.0's fission yield tape opens with its first material.
        let file = tape(&[AM244, IN116M1]);
        let without = file.split_once('\n').unwrap().1;
        assert_eq!(
            headers_from_str(without).unwrap(),
            headers_from_str(&file).unwrap()
        );
    }

    #[test]
    fn two_files_run_together_are_refused() {
        // `cat` leaves the first file's TEND record, and the second's TPID,
        // in the middle.
        let first = text(AM244);
        let tend = first.lines().count();
        assert_eq!(
            broken(&(first.clone() + &text(IN116M1))),
            format!(
                "malformed ENDF-6 file: line {} follows the TEND record at line {tend}, which \
                 ends the tape",
                tend + 1
            )
        );
        // Blank lines after it are fine.
        assert_eq!(
            headers_from_str(&(first.clone() + "\n   \n")).unwrap(),
            full_reader(&first)
        );
    }

    #[test]
    fn the_isomeric_state_is_liso_not_the_level() {
        for (fixture, lis, liso, name) in [
            (IN116M1, 1, 1, "In116_m1"),
            (IN116M2, 4, 2, "In116_m2"),
            (HF177M1, 48, 1, "Hf177_m1"),
        ] {
            let [header] = headers_from_str(&text(fixture)).unwrap()[..] else {
                panic!("one material");
            };
            assert_eq!(
                (header.lis, header.liso, header.name()),
                (lis, liso, Some(name.into()))
            );
        }
    }

    #[test]
    fn a_file_that_is_not_endf_is_refused() {
        // The ACE table of the same kind of data.
        let ace = refusal(&text(crate::testdata::LI6_ACE));
        assert!(
            ace.starts_with("not an ENDF-6 file: line 2 has no MAT, MF and MT numbers"),
            "{ace}"
        );

        // The page a host sends in place of an archive, one line per tag.
        let page = "<!DOCTYPE html>\n<html>\n<head><title>Download</title></head>\n\
                    <body>Please accept the terms of use.</body>\n</html>\n";
        assert_eq!(refusal(page), "not an ENDF-6 file: it holds no material");

        // The same page on one line, as it is usually served.
        let line = page.replace('\n', "");
        let one = refusal(&format!("{line}\n{line}\n"));
        assert!(
            one.starts_with("not an ENDF-6 file: line 2 has no MAT, MF and MT numbers"),
            "{one}"
        );

        assert_eq!(refusal(""), "not an ENDF-6 file: it holds no material");
    }

    #[test]
    fn a_file_that_is_cut_short_is_refused() {
        let file = text(AM244);
        let lines: Vec<&str> = file.lines().collect();
        let half = lines[..lines.len() / 2].join("\n");
        assert_eq!(
            broken(&half),
            "malformed ENDF-6 file: the file ends inside MAT 9552, which has no MEND record"
        );

        // A line that is not a record, once a material has started. (A short
        // one has blank MAT, MF and MT columns and reads as a MEND record.)
        let garbled = [lines[..10].join("\n"), "x".repeat(80)].join("\n");
        assert!(
            broken(&garbled)
                .starts_with("malformed ENDF-6 file: line 11 has no MAT, MF and MT numbers"),
            "{garbled}"
        );

        // MF=1 MT=451 closed after its HEAD and first CONT records.
        let short = [
            lines[0].to_string() + "\n",
            lines[1..3].join("\n") + "\n",
            record("", 9552, 1, 0),
            record("", 0, 0, 0),
        ]
        .concat();
        assert_eq!(
            broken(&short),
            "malformed ENDF-6 file: MF=1 MT=451 of MAT 9552 ends at line 4, after 2 of the four \
             records that open it"
        );
    }

    #[test]
    fn a_material_without_its_own_mend_is_refused() {
        // Two materials run together, with the MEND record between them gone.
        let file = tape(&[AM244, IN116M1]);
        let lines: Vec<&str> = file.lines().collect();
        let mend = lines
            .iter()
            .position(|line| control(line.as_bytes()).is_some_and(|(mat, ..)| mat == 0))
            .unwrap();
        let joined = [lines[..mend].join("\n"), lines[mend + 1..].join("\n")].join("\n");
        let message = broken(&joined);
        assert!(
            message.contains("starts a second MF=1 MT=451 section in MAT 9552"),
            "{message}"
        );

        // A material with no MF=1 MT=451 at all.
        let tpid = file.split_once('\n').unwrap().0;
        let bare = [
            format!("{tpid}\n"),
            record(
                " 9.524400+4 2.419680+2          0          0          0          0",
                9552,
                3,
                1,
            ),
            record("", 9552, 3, 0),
            record("", 0, 0, 0),
        ]
        .concat();
        assert_eq!(
            broken(&bare),
            "malformed ENDF-6 file: MAT 9552 (lines 2 to 4) has no MF=1 MT=451 section"
        );
    }

    #[test]
    fn every_field_comes_from_its_own_place() {
        // Every one of the 18 fields holds a different value, so reading one
        // from another's place fails.
        let opening = [
            record(
                " 2.604600+4 5.534450+1          1          2          3          4",
                2631,
                1,
                451,
            ),
            record(
                " 6.000000+0 7.000000+0          8          9          0         10",
                2631,
                1,
                451,
            ),
            record(
                " 1.100000+1 1.200000+1         13         14         15         16",
                2631,
                1,
                451,
            ),
            record(
                " 1.700000+1 0.000000+0         18          0          0          0",
                2631,
                1,
                451,
            ),
        ]
        .concat();
        assert_eq!(
            parse(2631, &opening).unwrap(),
            Header {
                mat: 2631,
                za: 26046,
                awr: 55.3445,
                lrp: 1,
                lfi: 2,
                nlib: 3,
                nmod: 4,
                elis: 6.0,
                sta: 7.0,
                lis: 8,
                liso: 9,
                nfor: 10,
                awi: 11.0,
                emax: 12.0,
                lrel: 13,
                nsub: 15,
                nver: 16,
                temp: 17.0,
                ldrv: 18,
            }
        );
    }

    #[test]
    fn some_materials_have_no_name() {
        let header = |za, nsub| Header {
            za,
            nsub,
            ..headers_from_str(&text(AM244)).unwrap()[0]
        };
        // Atomic relaxation, which has no reader that names it, is per element.
        let [relaxation] = headers_from_str(&text(RELAXATION)).unwrap()[..] else {
            panic!("one material");
        };
        assert_eq!(
            (relaxation.nsub, relaxation.name()),
            (6, Some("H".to_string()))
        );
        assert_eq!(header(26056, 6).name(), Some("Fe".to_string()));
        // Thermal scattering is named for its compound (ZA 134 is CH4).
        assert_eq!(header(134, 12).name(), None);
        assert_eq!(header(200_400, 10).name(), None);
        assert_eq!(header(-1000, 10).name(), None);
    }

    #[test]
    fn reads_a_file_from_disk() {
        let dir = std::env::temp_dir().join(format!("endf-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let file = tape(&[AM244, PHOTOATOMIC]);
        let endf = dir.join("tape.endf");
        std::fs::write(&endf, &file).unwrap();
        assert_eq!(get_headers(&endf).unwrap(), full_reader(&file));

        // IRDFF-II's tape has a cp1252 dash, byte 0x96, in Zn68's description,
        // which is not UTF-8. Here it goes into Am244's first TEXT record.
        let mut bytes = file.clone().into_bytes();
        let text_record = file.match_indices('\n').nth(4).unwrap().0 + 1;
        bytes[text_record + 32] = 0x96;
        std::fs::write(&endf, &bytes).unwrap();
        assert_eq!(get_headers(&endf).unwrap(), full_reader(&file));
        // A record that is parsed has to be text, though.
        let mut bytes = file.clone().into_bytes();
        bytes[file.match_indices('\n').nth(1).unwrap().0 + 6] = 0x96;
        std::fs::write(&endf, &bytes).unwrap();
        assert_eq!(
            get_headers(&endf).unwrap_err().to_string(),
            "malformed ENDF-6 file: line 3, a record of MF=1 MT=451 in MAT 9552, is not text"
        );

        // An archive, which is bytes rather than lines of text.
        let archive = dir.join("jendl5-n.tar.gz");
        let mut gzip = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\n".to_vec();
        gzip.extend((0..120u8).map(|b| b | 0x80));
        std::fs::write(&archive, &gzip).unwrap();
        let message = get_headers(&archive).unwrap_err().to_string();
        assert!(
            message.starts_with("not an ENDF-6 file: line 2 has no MAT, MF and MT numbers"),
            "{message}"
        );

        assert!(matches!(
            get_headers(dir.join("missing.endf")),
            Err(Error::Io(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
