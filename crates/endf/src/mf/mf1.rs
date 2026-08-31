//! MF=1, descriptive data, neutron yields and fission energy release.

use crate::error::Result;
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::{field, Reader};

/// A neutron yield (nu-bar), in whichever of the two forms the evaluation uses.
///
/// LNU=1 gives polynomial coefficients, LNU=2 a tabulation. Evaluations mix the
/// two freely (U235, U238 and Pu239 carry a tabulated prompt yield alongside
/// a polynomial delayed one) so the form is recorded per quantity, never assumed.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Nu {
    /// LNU=1: coefficients in ascending powers of incident energy.
    Polynomial(Vec<f64>),
    /// LNU=2: tabulated against incident energy.
    Tabulated(Tabulated1D),
    /// LNU was neither 1 nor 2.
    #[default]
    Absent,
}

/// MF=1 MT=451: descriptive data and the section directory.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf1Mt451 {
    pub za: i64,
    pub awr: f64,
    pub lrp: i64,
    pub lfi: i64,
    pub nlib: i64,
    pub nmod: i64,
    pub elis: f64,
    pub sta: f64,
    pub lis: i64,
    pub liso: i64,
    pub nfor: i64,
    pub awi: f64,
    pub emax: f64,
    pub lrel: i64,
    pub nsub: i64,
    pub nver: i64,
    pub temp: f64,
    pub ldrv: i64,
    pub nwd: i64,
    pub nxc: i64,
    /// Target name, e.g. `" 95-Am-244 "`. `None` when the evaluation writes
    /// fewer than five text records, which leaves the whole header absent.
    pub zsymam: Option<String>,
    pub alab: Option<String>,
    pub edate: Option<String>,
    pub auth: Option<String>,
    pub reference: Option<String>,
    pub ddate: Option<String>,
    pub rdate: Option<String>,
    pub endate: Option<String>,
    pub hsub: Vec<String>,
    pub description: Vec<String>,
    /// The directory: (MF, MT, number of records, modification number).
    pub section_list: Vec<(i64, i64, i64, i64)>,
}

/// MF=1 MT=452 or MT=456: total or prompt neutrons per fission.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf1Mt452 {
    pub za: i64,
    pub awr: f64,
    pub lnu: i64,
    pub nu: Nu,
}

/// Delayed-group constants at one incident energy (LDG=1).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DelayedGroupConstants {
    pub energy: f64,
    pub lambda: Vec<f64>,
    pub alpha: Vec<f64>,
}

/// MF=1 MT=455: delayed neutron data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf1Mt455 {
    pub za: i64,
    pub awr: f64,
    pub ldg: i64,
    pub lnu: i64,
    /// LDG=0: decay constants, energy independent.
    pub lambda: Vec<f64>,
    /// LDG=1: interpolation across incident energy.
    pub e_int: Option<Tabulated2D>,
    /// LDG=1: the constants at each incident energy.
    pub constants: Vec<DelayedGroupConstants>,
    pub nu: Nu,
}

/// One component of the fission energy release.
///
/// The two forms are what MT=458 actually stores, and they survive into the
/// converted data unchanged: a component is either polynomial coefficients or a
/// tabulation, never both, and which one it is varies per component within a
/// single evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum FissionEnergyRelease {
    /// (coefficient, uncertainty) in ascending powers of incident energy.
    Polynomial(Vec<(f64, f64)>),
    /// LFC=1: tabulated against incident energy.
    Tabulated { ldrv: i64, eifc: Tabulated1D },
}

/// The nine components of the fission energy release, in the order MT=458
/// stores them. `IFC` in the format indexes this list, one-based.
pub const FISSION_ENERGY_COMPONENTS: [&str; 9] =
    ["EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET"];

/// MF=1 MT=458: components of the fission energy release.
#[derive(Debug, Clone, PartialEq)]
pub struct Mf1Mt458 {
    /// Read from a CONT rather than a HEAD record, so this is the raw float the
    /// field holds and not the integer every other section reports. See
    /// <https://github.com/shimwell/endf-python/issues/14>.
    pub za: f64,
    pub awr: f64,
    pub lfc: i64,
    pub nply: i64,
    pub nfc: i64,
    /// One entry per name in [`FISSION_ENERGY_COMPONENTS`], same order.
    pub components: Vec<FissionEnergyRelease>,
}

impl Mf1Mt458 {
    /// A component by its MT=458 name, e.g. `"EGP"` for prompt fission photons.
    pub fn component(&self, name: &str) -> Option<&FissionEnergyRelease> {
        let i = FISSION_ENERGY_COMPONENTS.iter().position(|&n| n == name)?;
        self.components.get(i)
    }
}

/// MF=1 MT=460: delayed photon data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf1Mt460 {
    pub za: i64,
    pub awr: f64,
    pub lo: i64,
    pub ng: i64,
    /// LO=1: the energy of each discrete photon.
    pub energy: Vec<f64>,
    /// LO=1: time dependence of each photon, aligned with `energy`.
    pub time: Vec<Tabulated1D>,
    /// LO=2: decay constants for the precursors.
    pub lambda: Vec<f64>,
}

/// Read a nu-bar in whichever form `lnu` selects.
fn parse_nu(reader: &mut Reader, lnu: i64) -> Result<Nu> {
    Ok(match lnu {
        1 => Nu::Polynomial(reader.list_record()?.values),
        2 => Nu::Tabulated(reader.tab1_record()?.table),
        _ => Nu::Absent,
    })
}

/// Trim a fixed-width text field, keeping `None` distinct from empty.
fn text_field(line: &str, start: usize, end: usize) -> Option<String> {
    Some(field(line, start, end).to_string())
}

/// Parse MF=1 MT=451.
pub fn parse_mf1_mt451(reader: &mut Reader) -> Result<Mf1Mt451> {
    let head = reader.head_record()?;
    let c1 = reader.cont_record()?;
    let c2 = reader.cont_record()?;
    let c3 = reader.cont_record()?;

    let mut data = Mf1Mt451 {
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
        nwd: c3.n1,
        nxc: c3.n2,
        ..Default::default()
    };

    let nwd = data.nwd.max(0) as usize;
    let mut text = Vec::with_capacity(nwd);
    for _ in 0..nwd {
        text.push(reader.text_record()?.to_string());
    }

    // Fewer than five text records means the evaluation left the descriptive
    // header out entirely; the reader reports that rather than inventing it.
    if text.len() >= 5 {
        data.zsymam = text_field(&text[0], 0, 11);
        data.alab = text_field(&text[0], 11, 22);
        data.edate = text_field(&text[0], 22, 32);
        data.auth = text_field(&text[0], 32, 66);
        data.reference = text_field(&text[1], 1, 22);
        data.ddate = text_field(&text[1], 22, 32);
        data.rdate = text_field(&text[1], 33, 43);
        data.endate = text_field(&text[1], 55, 63);
        data.hsub = text[2..5].to_vec();
        data.description = text[5..].to_vec();
    }

    for _ in 0..data.nxc.max(0) {
        let c = reader.cont_record()?;
        data.section_list.push((c.l1, c.l2, c.n1, c.n2));
    }

    Ok(data)
}

/// Parse MF=1 MT=452 or MT=456.
pub fn parse_mf1_mt452(reader: &mut Reader) -> Result<Mf1Mt452> {
    let head = reader.head_record()?;
    let lnu = head.l2;
    Ok(Mf1Mt452 {
        za: head.za,
        awr: head.awr,
        lnu,
        nu: parse_nu(reader, lnu)?,
    })
}

/// Parse MF=1 MT=455.
pub fn parse_mf1_mt455(reader: &mut Reader) -> Result<Mf1Mt455> {
    let head = reader.head_record()?;
    let (ldg, lnu) = (head.l1, head.l2);
    let mut data = Mf1Mt455 {
        za: head.za,
        awr: head.awr,
        ldg,
        lnu,
        ..Default::default()
    };

    if ldg == 0 {
        data.lambda = reader.list_record()?.values;
    } else if ldg == 1 {
        let tab2 = reader.tab2_record()?;
        let ne = tab2.cont.n2.max(0);
        data.e_int = Some(tab2.table);
        for _ in 0..ne {
            let list = reader.list_record()?;
            data.constants.push(DelayedGroupConstants {
                energy: list.cont.c2,
                lambda: list.values.iter().step_by(2).copied().collect(),
                alpha: list.values.iter().skip(1).step_by(2).copied().collect(),
            });
        }
    }

    // With energy-independent group constants the abundances are not given
    // here at all; they have to come from MF=5 MT=455, which carries one
    // energy distribution per delayed group.
    data.nu = parse_nu(reader, lnu)?;
    Ok(data)
}

/// Parse MF=1 MT=458.
pub fn parse_mf1_mt458(reader: &mut Reader) -> Result<Mf1Mt458> {
    let head = reader.cont_record()?;
    let (lfc, nfc) = (head.l2, head.n2);

    let list = reader.list_record()?;
    let nply = list.cont.l2;
    let values = &list.values;

    // The LIST holds, for each polynomial order in turn, a value and an
    // uncertainty for each of the nine components: 18 numbers per order.
    let n = FISSION_ENERGY_COMPONENTS.len();
    let stride = 2 * n;
    let mut components = Vec::with_capacity(n);
    for i in 0..n {
        let pairs = values
            .iter()
            .skip(2 * i)
            .step_by(stride)
            .zip(values.iter().skip(2 * i + 1).step_by(stride))
            .map(|(&c, &d)| (c, d))
            .collect();
        components.push(FissionEnergyRelease::Polynomial(pairs));
    }

    let mut data = Mf1Mt458 {
        za: head.c1,
        awr: head.c2,
        lfc,
        nply,
        nfc,
        components,
    };

    // LFC=1 replaces some components with a tabulation. IFC says which.
    if lfc == 1 {
        for _ in 0..nfc.max(0) {
            let tab = reader.tab1_record()?;
            let (ldrv, ifc) = (tab.l1, tab.l2);
            let idx = (ifc - 1).max(0) as usize;
            if let Some(slot) = data.components.get_mut(idx) {
                *slot = FissionEnergyRelease::Tabulated {
                    ldrv,
                    eifc: tab.table,
                };
            }
        }
    }

    Ok(data)
}

/// Parse MF=1 MT=460.
pub fn parse_mf1_mt460(reader: &mut Reader) -> Result<Mf1Mt460> {
    let head = reader.head_record()?;
    let lo = head.l1;
    let mut data = Mf1Mt460 {
        za: head.za,
        awr: head.awr,
        lo,
        ..Default::default()
    };

    if lo == 1 {
        data.ng = head.n1;
        for _ in 0..data.ng.max(0) {
            let tab = reader.tab1_record()?;
            data.energy.push(tab.c1);
            data.time.push(tab.table);
        }
    } else if lo == 2 {
        data.lambda = reader.list_record()?.values;
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_the_descriptive_header() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let d = m.mf1_mt451().expect("MF=1 MT=451 is present");
        assert_eq!(d.za, 95244);
        // NSUB=10 is incident-neutron data.
        assert_eq!(d.nsub, 10);
        assert_eq!(d.zsymam.as_deref(), Some(" 95-Am-244 "));
        // The directory lists every section in the material.
        assert_eq!(d.section_list.len(), d.nxc as usize);
        assert!(!d.section_list.is_empty());
    }

    #[test]
    fn the_directory_agrees_with_the_sections_present() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let d = m.mf1_mt451().unwrap();
        for &(mf, mt, ..) in &d.section_list {
            assert!(
                m.contains(mf as i32, mt as i32),
                "the directory lists MF={mf} MT={mt} but it is not in the file"
            );
        }
    }
}
