//! MF=23, 26, 27 and 28: photo-atomic and electro-atomic data.
//!
//! Grouped into one module because the four files describe one thing between
//! them: the photon and electron interaction cross sections (MF=23), the
//! secondary distributions of the electro-atomic ones (MF=26), the form factors
//! and scattering functions that modify coherent and incoherent scattering
//! (MF=27), and how the ionised atom relaxes afterwards (MF=28).

use crate::error::Result;
use crate::function::Tabulated1D;
use crate::mf::mf6::{
    parse_continuum_energy_angle, parse_discrete_two_body, ContinuumEnergyAngle, DiscreteTwoBody,
};
use crate::records::Reader;

// -------------------------------------------------------------------------
// MF=23
// -------------------------------------------------------------------------

/// MF=23: a photo-atomic or electro-atomic cross section.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf23 {
    pub za: i64,
    pub awr: f64,
    /// Subshell binding energy, for the photoelectric subshell reactions.
    pub epe: f64,
    /// Fluorescence yield.
    pub efl: f64,
    pub sigma: Tabulated1D,
}

/// Parse an MF=23 section.
pub fn parse_mf23(reader: &mut Reader) -> Result<Mf23> {
    let head = reader.head_record()?;
    let tab = reader.tab1_record()?;
    Ok(Mf23 {
        za: head.za,
        awr: head.awr,
        epe: tab.c1,
        efl: tab.c2,
        sigma: tab.table,
    })
}

// -------------------------------------------------------------------------
// MF=26
// -------------------------------------------------------------------------

/// One product of an electro-atomic reaction.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicProduct {
    /// ZA of the product: 0 a photon, 11 an electron.
    pub zap: i64,
    pub awi: f64,
    pub law: i64,
    /// Multiplicity as a function of incident energy.
    pub yield_: Tabulated1D,
    pub distribution: ElectroAtomicDistribution,
}

/// The distribution of an electro-atomic product, by law.
#[derive(Debug, Clone, PartialEq)]
pub enum ElectroAtomicDistribution {
    /// A law the reader does not recognise, which the Python reader warns
    /// about and otherwise ignores.
    None,
    /// LAW=1, shared with MF=6.
    ContinuumEnergyAngle(Box<ContinuumEnergyAngle>),
    /// LAW=2, shared with MF=6.
    DiscreteTwoBody(Box<DiscreteTwoBody>),
    /// LAW=8: the energy transferred to the atom by excitation.
    EnergyTransfer(Tabulated1D),
}

/// MF=26: secondary distributions for electro-atomic data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf26 {
    pub za: i64,
    pub awr: f64,
    pub nk: i64,
    pub products: Vec<AtomicProduct>,
}

/// Parse an MF=26 section.
pub fn parse_mf26(reader: &mut Reader) -> Result<Mf26> {
    let head = reader.head_record()?;
    let nk = head.n1;
    let mut data = Mf26 {
        za: head.za,
        awr: head.awr,
        nk,
        products: Vec::new(),
    };

    for _ in 0..nk.max(0) {
        let tab = reader.tab1_record()?;
        let law = tab.l2;
        let distribution = match law {
            1 => ElectroAtomicDistribution::ContinuumEnergyAngle(Box::new(
                parse_continuum_energy_angle(reader)?,
            )),
            2 => ElectroAtomicDistribution::DiscreteTwoBody(Box::new(parse_discrete_two_body(
                reader,
            )?)),
            8 => ElectroAtomicDistribution::EnergyTransfer(reader.tab1_record()?.table),
            _ => ElectroAtomicDistribution::None,
        };
        data.products.push(AtomicProduct {
            zap: tab.c1 as i64,
            awi: tab.c2,
            law,
            yield_: tab.table,
            distribution,
        });
    }

    Ok(data)
}

// -------------------------------------------------------------------------
// MF=27
// -------------------------------------------------------------------------

/// MF=27: an atomic form factor or scattering function.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf27 {
    pub za: i64,
    pub awr: f64,
    /// Atomic number, written in a C field so it is a float here.
    pub z: f64,
    /// The form factor or scattering function against momentum transfer.
    pub h: Tabulated1D,
}

/// Parse an MF=27 section.
pub fn parse_mf27(reader: &mut Reader) -> Result<Mf27> {
    let head = reader.head_record()?;
    let tab = reader.tab1_record()?;
    Ok(Mf27 {
        za: head.za,
        awr: head.awr,
        z: tab.c2,
        h: tab.table,
    })
}

// -------------------------------------------------------------------------
// MF=28
// -------------------------------------------------------------------------

/// The relaxation data for one subshell.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Subshell {
    /// Subshell designator, written in a C field so it is a float here.
    pub subi: f64,
    /// Number of transitions.
    pub ntr: i64,
    /// Binding energy.
    pub ebi: f64,
    /// Number of electrons in the subshell when neutral.
    pub eln: f64,
    /// Secondary subshell of each transition.
    pub subj: Vec<f64>,
    /// Tertiary subshell, zero for a radiative transition.
    pub subk: Vec<f64>,
    /// Energy of each transition.
    pub etr: Vec<f64>,
    /// Fractional probability of each transition.
    pub ftr: Vec<f64>,
}

/// MF=28: atomic relaxation data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf28 {
    pub za: i64,
    pub awr: f64,
    pub nss: i64,
    pub shells: Vec<Subshell>,
}

fn column(values: &[f64], offset: usize, stride: usize) -> Vec<f64> {
    values
        .iter()
        .skip(offset)
        .step_by(stride)
        .copied()
        .collect()
}

/// Parse an MF=28 section.
pub fn parse_mf28(reader: &mut Reader) -> Result<Mf28> {
    let head = reader.head_record()?;
    let nss = head.n1;
    let mut data = Mf28 {
        za: head.za,
        awr: head.awr,
        nss,
        shells: Vec::new(),
    };

    for _ in 0..nss.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        // The first six values describe the subshell; the transitions follow
        // in groups of six.
        data.shells.push(Subshell {
            subi: list.cont.c1,
            ntr: list.cont.n2,
            ebi: v.first().copied().unwrap_or(0.0),
            eln: v.get(1).copied().unwrap_or(0.0),
            subj: column(v, 6, 6),
            subk: column(v, 7, 6),
            etr: column(v, 8, 6),
            ftr: column(v, 9, 6),
        });
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const PHOTOAT_H: &[u8] = include_bytes!("../../fixtures/photoat-001_H_000.endf.xz");
    const ATOM_H: &[u8] = include_bytes!("../../fixtures/atom-001_H_000.endf.xz");

    #[test]
    fn reads_photo_atomic_cross_sections() {
        let m = Material::from_str(&crate::testdata::text(PHOTOAT_H)).unwrap();
        // MT=501 is the total photon interaction cross section.
        let total = m.mf23(501).expect("MF=23 MT=501 is present");
        assert_eq!(total.za, 1000, "photo-atomic data is per element, so A=0");
        assert!(!total.sigma.x.is_empty());
        assert!(total.sigma.x.windows(2).all(|w| w[1] >= w[0]));
        assert!(total.sigma.y.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn reads_the_incoherent_scattering_function() {
        let m = Material::from_str(&crate::testdata::text(PHOTOAT_H)).unwrap();
        // MT=502 coherent form factor, MT=504 incoherent scattering function.
        let sf = m.mf27(504).expect("MF=27 MT=504 is present");
        assert_eq!(sf.z, 1.0, "hydrogen");
        // The incoherent scattering function rises from 0 towards Z.
        assert!(sf.h.y.first().copied().unwrap() >= 0.0);
        assert!(sf.h.y.last().copied().unwrap() <= 1.0 + 1e-9);
    }

    #[test]
    fn reads_atomic_relaxation() {
        let m = Material::from_str(&crate::testdata::text(ATOM_H)).unwrap();
        let relax = m.mf28(533).expect("MF=28 MT=533 is present");
        assert_eq!(relax.shells.len(), relax.nss as usize);
        // Hydrogen has only the K shell, and with one electron there is
        // nothing to relax from, so it lists no transitions.
        assert_eq!(relax.shells.len(), 1);
        let k = &relax.shells[0];
        assert_eq!(k.subi, 1.0, "the K shell");
        assert_eq!(k.eln, 1.0, "one electron");
        assert!(k.ebi > 0.0, "binding energy");
        assert_eq!(k.ntr, 0);
        assert!(k.subj.is_empty());
    }
}
