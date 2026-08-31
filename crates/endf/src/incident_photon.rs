//! Photon interaction data for one element.
//!
//! Built from the photoatomic sublibrary (MF=23 cross sections, MF=27 form
//! factors, MF=28 atomic relaxation) or from a processed ACE photoatomic
//! table.

use std::collections::BTreeMap;

use crate::ace::{self, Table};
use crate::data::{sum_rule, ATOMIC_SYMBOL, EV_PER_MEV};
use crate::error::{Error, Result};
use crate::function::Tabulated1D;
use crate::material::Material;

/// Atomic subshells, indexed by the designator the format uses. Index 0 is
/// unused, so a designator can be looked up directly.
pub const SUBSHELLS: [&str; 40] = [
    "", "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6",
    "N7", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4", "P5", "P6",
    "P7", "P8", "P9", "P10", "P11", "Q1", "Q2", "Q3",
];

/// The photon reactions the format names outright. The subshell
/// photoelectric reactions, MT=534 upward, are named for their subshell.
const PHOTON_REACTION_NAME_BASE: [(i32, &str); 11] = [
    (501, "total"),
    (502, "coherent"),
    (504, "incoherent"),
    (515, "pair_production_electron"),
    (516, "pair_production_total"),
    (517, "pair_production_nuclear"),
    (522, "photoelectric"),
    (525, "heating"),
    (526, "electro_atomic_scat"),
    (527, "electro_atomic_brem"),
    (528, "electro_atomic_excit"),
];

/// The name of a photon reaction, e.g. `"coherent"` for MT=502 or `"K"` for
/// MT=534, the K-shell photoelectric cross section.
pub fn photon_reaction_name(mt: i32) -> Option<&'static str> {
    if let Some(&(_, name)) = PHOTON_REACTION_NAME_BASE.iter().find(|&&(m, _)| m == mt) {
        return Some(name);
    }
    // MT = 533 + designator.
    let designator = mt - 533;
    if designator >= 1 {
        return SUBSHELLS.get(designator as usize).copied();
    }
    None
}

/// The MT of a named photon reaction.
pub fn photon_reaction_mt(name: &str) -> Option<i32> {
    if let Some(&(mt, _)) = PHOTON_REACTION_NAME_BASE.iter().find(|&&(_, n)| n == name) {
        return Some(mt);
    }
    SUBSHELLS
        .iter()
        .position(|&s| s == name && !s.is_empty())
        .map(|i| 533 + i as i32)
}

/// One photon interaction channel.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhotonReaction {
    pub mt: i32,
    /// Cross section against incident photon energy, in barns and eV.
    pub xs: Option<Tabulated1D>,
    /// The coherent or incoherent form factor.
    pub scattering_factor: Option<Tabulated1D>,
    /// Real part of the anomalous scattering factor, MT=502 only.
    pub anomalous_real: Option<Tabulated1D>,
    /// Imaginary part, likewise.
    pub anomalous_imag: Option<Tabulated1D>,
    /// Binding energy in eV, for the subshell photoelectric reactions.
    pub subshell_binding_energy: Option<f64>,
    /// Fluorescence yield, for MT=534 to 572.
    pub fluorescence_yield: Option<f64>,
}

impl PhotonReaction {
    pub fn new(mt: i32) -> PhotonReaction {
        PhotonReaction {
            mt,
            ..Default::default()
        }
    }

    pub fn name(&self) -> Option<&'static str> {
        photon_reaction_name(self.mt)
    }
}

/// One radiative or non-radiative transition filling a vacancy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transitions {
    /// The subshell the electron falls from.
    pub secondary_subshell: Vec<&'static str>,
    /// For a non-radiative transition, the subshell the Auger electron leaves.
    /// Empty for a radiative one.
    pub tertiary_subshell: Vec<&'static str>,
    /// Energy of the emitted particle in eV.
    pub energy: Vec<f64>,
    /// Fractional probability of each transition.
    pub probability: Vec<f64>,
}

/// How an atom relaxes after a photoelectric event leaves a vacancy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AtomicRelaxation {
    /// Binding energy in eV, by subshell.
    pub binding_energy: BTreeMap<&'static str, f64>,
    /// Electron occupancy of the neutral atom, by subshell.
    pub num_electrons: BTreeMap<&'static str, f64>,
    /// The transitions that fill a vacancy in each subshell.
    pub transitions: BTreeMap<&'static str, Transitions>,
}

impl AtomicRelaxation {
    /// The subshells present, in the format's own order rather than
    /// alphabetically.
    pub fn subshells(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = self.binding_energy.keys().copied().collect();
        out.sort_by_key(|s| {
            SUBSHELLS
                .iter()
                .position(|&x| x == *s)
                .unwrap_or(usize::MAX)
        });
        out
    }

    /// Read atomic relaxation data from an MF=28 MT=533 section.
    pub fn from_endf(material: &Material) -> Result<AtomicRelaxation> {
        let section = material.mf28(533).ok_or(Error::Unsupported {
            what: "atomic relaxation from a material with no MF=28 MT=533",
        })?;

        let name = |designator: f64| -> Result<&'static str> {
            SUBSHELLS
                .get(designator as usize)
                .copied()
                .ok_or(Error::Unsupported {
                    what: "an atomic subshell designator the format does not define",
                })
        };

        let mut out = AtomicRelaxation::default();
        for shell in &section.shells {
            let subi = name(shell.subi)?;
            out.binding_energy.insert(subi, shell.ebi);
            out.num_electrons.insert(subi, shell.eln);
            if shell.ntr > 0 {
                out.transitions.insert(
                    subi,
                    Transitions {
                        secondary_subshell: shell
                            .subj
                            .iter()
                            .map(|&s| name(s))
                            .collect::<Result<_>>()?,
                        tertiary_subshell: shell
                            .subk
                            .iter()
                            .map(|&s| name(s))
                            .collect::<Result<_>>()?,
                        energy: shell.etr.clone(),
                        probability: shell.ftr.clone(),
                    },
                );
            }
        }
        Ok(out)
    }

    /// Read atomic relaxation data from an ACE photoatomic table.
    pub fn from_ace(table: &Table) -> Result<AtomicRelaxation> {
        let at = |i: i64| -> f64 {
            usize::try_from(i)
                .ok()
                .and_then(|i| table.xss.get(i).copied())
                .unwrap_or(0.0)
        };
        let name = |designator: f64| -> Result<&'static str> {
            SUBSHELLS
                .get(designator as usize)
                .copied()
                .ok_or(Error::Unsupported {
                    what: "an atomic subshell designator the format does not define",
                })
        };

        let n = table.nxs[7].max(0);
        let shells: Vec<&'static str> = (0..n)
            .map(|i| name(at(table.jxs[11] + i)))
            .collect::<Result<_>>()?;

        let mut out = AtomicRelaxation::default();
        for (i, &shell) in shells.iter().enumerate() {
            out.num_electrons
                .insert(shell, at(table.jxs[12] + i as i64));
            out.binding_energy
                .insert(shell, at(table.jxs[13] + i as i64) * EV_PER_MEV);
        }

        let mut idx = table.jxs[18];
        for (i, &subi) in shells.iter().enumerate() {
            let n_transitions = at(table.jxs[15] + i as i64) as i64;
            if n_transitions <= 0 {
                continue;
            }
            let mut transitions = Transitions::default();
            for j in 0..n_transitions {
                transitions.secondary_subshell.push(name(at(idx))?);
                transitions.tertiary_subshell.push(name(at(idx + 1))?);
                transitions.energy.push(at(idx + 2) * EV_PER_MEV);
                // The table stores cumulative probabilities, so each one after
                // the first is the difference from the one before.
                transitions.probability.push(if j == 0 {
                    at(idx + 3)
                } else {
                    at(idx + 3) - at(idx - 1)
                });
                idx += 4;
            }
            out.transitions.insert(subi, transitions);
        }
        Ok(out)
    }
}

/// Photon interaction data for one element.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IncidentPhoton {
    pub atomic_number: i64,
    pub reactions: BTreeMap<i32, PhotonReaction>,
    pub atomic_relaxation: Option<AtomicRelaxation>,
    /// Compton profiles. An ACE photoatomic table carries them; an ENDF
    /// evaluation does not, so for those they come from the auxiliary
    /// tabulations via [`IncidentPhoton::add_photon_data`].
    pub compton_profiles: Option<ComptonProfiles>,
    /// Bremsstrahlung and density-effect data, which no evaluation carries.
    /// Filled by [`IncidentPhoton::add_photon_data`].
    pub bremsstrahlung: Option<Bremsstrahlung>,
}

/// Bremsstrahlung data attached to an element, on the shared grids.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bremsstrahlung {
    /// Mean excitation energy, in eV.
    pub i: f64,
    pub num_electrons: Vec<f64>,
    /// Ionization energy of each subshell, in eV.
    pub ionization_energy: Vec<f64>,
    /// Incident electron kinetic energies, in eV.
    pub electron_energy: Vec<f64>,
    /// Reduced photon energies.
    pub photon_energy: Vec<f64>,
    /// Scaled cross sections in barns: one row per electron energy, one
    /// column per reduced photon energy.
    pub dcs: Vec<Vec<f64>>,
}

/// The Compton profiles of an element, by shell.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ComptonProfiles {
    pub num_electrons: Vec<f64>,
    /// Binding energy in eV.
    pub binding_energy: Vec<f64>,
    /// The profile of each shell, against electron momentum.
    pub j: Vec<Tabulated1D>,
}

impl IncidentPhoton {
    /// Attach the auxiliary Compton profile and bremsstrahlung data.
    ///
    /// None of it is in the evaluation: it is looked up by atomic number in
    /// tabulations that ship alongside, which is what the Python package does
    /// at the end of `from_endf`. Kept as a separate call here because the
    /// data lives in files the caller has to find; see
    /// [`crate::PhotonData`].
    ///
    /// Compton profiles already read from an ACE table are left alone, since
    /// those came from the file being read and are specific to it.
    pub fn add_photon_data(&mut self, data: &crate::PhotonData) {
        if let Some(profile) = data.compton.get(&self.atomic_number) {
            if self.compton_profiles.is_none() {
                self.compton_profiles = Some(ComptonProfiles {
                    num_electrons: profile.num_electrons.clone(),
                    binding_energy: profile.binding_energy.clone(),
                    j: profile
                        .j
                        .iter()
                        .map(|row| Tabulated1D::new(data.pz.clone(), row.clone()))
                        .collect(),
                });
            }
        }
        if let Some(brem) = data.bremsstrahlung.get(&self.atomic_number) {
            self.bremsstrahlung = Some(Bremsstrahlung {
                i: brem.i,
                num_electrons: brem.num_electrons.clone(),
                ionization_energy: brem.ionization_energy.clone(),
                electron_energy: data.electron_energy.clone(),
                photon_energy: data.photon_energy.clone(),
                dcs: brem.dcs.clone(),
            });
        }
    }

    pub fn new(atomic_number: i64) -> IncidentPhoton {
        IncidentPhoton {
            atomic_number,
            ..Default::default()
        }
    }

    /// The element's symbol, e.g. `"Fe"`.
    pub fn name(&self) -> &'static str {
        ATOMIC_SYMBOL
            .get(self.atomic_number as usize)
            .copied()
            .unwrap_or("?")
    }

    pub fn contains(&self, mt: i32) -> bool {
        self.reactions.contains_key(&mt)
    }

    pub fn get(&self, mt: i32) -> Option<&PhotonReaction> {
        self.reactions.get(&mt)
    }

    /// One reaction, by name: `"coherent"`, `"K"`.
    pub fn get_by_name(&self, name: &str) -> Option<&PhotonReaction> {
        self.get(photon_reaction_mt(name)?)
    }

    /// Read photon data from a photoatomic evaluation.
    ///
    /// `relaxation` is a separate atomic relaxation evaluation, for libraries
    /// that ship one apart from the photoatomic file. When the photoatomic
    /// material carries its own MF=28, that is used and this overrides it.
    pub fn from_endf(
        photoatomic: &Material,
        relaxation: Option<&Material>,
    ) -> Result<IncidentPhoton> {
        let metadata = photoatomic.mf1_mt451().ok_or(Error::Unsupported {
            what: "a photoatomic evaluation with no MF=1 MT=451",
        })?;
        let mut data = IncidentPhoton::new(metadata.za / 1000);

        for &(mf, mt) in photoatomic.section_data.keys() {
            if mf != 23 {
                continue;
            }
            let Some(section) = photoatomic.mf23(mt) else {
                continue;
            };
            let mut rx = PhotonReaction::new(mt);
            rx.xs = Some(section.sigma.clone());
            if (534..=599).contains(&mt) {
                rx.subshell_binding_energy = Some(section.epe);
            }
            if (534..=572).contains(&mt) {
                rx.fluorescence_yield = Some(section.efl);
            }
            data.reactions.insert(mt, rx);
        }

        // MF=27 carries the form factors, keyed by their own MT numbers, which
        // belong to the MF=23 reactions rather than standing alone.
        for &(mf, mt) in photoatomic.section_data.keys() {
            if mf != 27 {
                continue;
            }
            let Some(section) = photoatomic.mf27(mt) else {
                continue;
            };
            let h = section.h.clone();
            match mt {
                502 | 504 => {
                    if let Some(rx) = data.reactions.get_mut(&mt) {
                        rx.scattering_factor = Some(h);
                    }
                }
                505 => {
                    if let Some(rx) = data.reactions.get_mut(&502) {
                        rx.anomalous_imag = Some(h);
                    }
                }
                506 => {
                    if let Some(rx) = data.reactions.get_mut(&502) {
                        rx.anomalous_real = Some(h);
                    }
                }
                _ => {}
            }
        }

        if photoatomic.mf28(533).is_some() {
            data.atomic_relaxation = Some(AtomicRelaxation::from_endf(photoatomic)?);
        }
        if let Some(relaxation) = relaxation {
            data.atomic_relaxation = Some(AtomicRelaxation::from_endf(relaxation)?);
        }

        Ok(data)
    }

    /// Read photon data from an ACE photoatomic table.
    pub fn from_ace(table: &Table) -> Result<IncidentPhoton> {
        if table.data_type()? != ace::TableType::Photoatomic {
            return Err(Error::BadAceTable {
                what: format!("{} is not a photoatomic table", table.name),
            });
        }
        let meta = ace::get_metadata(table.zaid()?, ace::MetastableScheme::Mcnp)?;
        let mut data = IncidentPhoton::new(meta.z as i64);

        let at = |i: i64| -> f64 {
            usize::try_from(i)
                .ok()
                .and_then(|i| table.xss.get(i).copied())
                .unwrap_or(0.0)
        };
        let slice = |i: i64, n: usize| -> Vec<f64> { (0..n as i64).map(|k| at(i + k)).collect() };

        // The energy grid is stored as logarithms.
        let n_energy = table.nxs[3].max(0) as usize;
        let energy: Vec<f64> = slice(table.jxs[1], n_energy)
            .into_iter()
            .map(|e| e.exp() * EV_PER_MEV)
            .collect();

        // The five main cross sections sit in columns beside the grid, also as
        // logarithms; a stored zero means zero rather than one.
        for (mt, column) in [(502, 2), (504, 1), (517, 4), (522, 3), (525, 5)] {
            let mut rx = PhotonReaction::new(mt);
            let raw = slice(table.jxs[1] + (column * n_energy) as i64, n_energy);
            let y: Vec<f64> = raw
                .iter()
                .map(|&v| if v == 0.0 { 0.0 } else { v.exp() })
                .collect();
            // Log-log, which is how the columns were stored.
            rx.xs = Some(Tabulated1D::with_regions(
                energy.clone(),
                y,
                vec![n_energy as i32],
                vec![5],
            ));
            data.reactions.insert(mt, rx);
        }

        // The Compton profiles the table carries.
        let n_shell = table.nxs[5].max(0) as usize;
        if n_shell != 0 {
            let mut profiles = Vec::with_capacity(n_shell);
            for k in 0..n_shell as i64 {
                let loca = at(table.jxs[9] + k) as i64;
                let jj = at(table.jxs[10] + loca - 1) as i32;
                let m = at(table.jxs[10] + loca) as usize;
                let idx = table.jxs[10] + loca + 1;
                profiles.push(Tabulated1D::with_regions(
                    slice(idx, m),
                    slice(idx + m as i64, m),
                    vec![m as i32],
                    vec![jj],
                ));
            }
            data.compton_profiles = Some(ComptonProfiles {
                num_electrons: slice(table.jxs[6], n_shell),
                binding_energy: slice(table.jxs[7], n_shell)
                    .into_iter()
                    .map(|e| e * EV_PER_MEV)
                    .collect(),
                j: profiles,
            });
        }

        // The subshell photoelectric cross sections, and the relaxation data
        // that names their binding energies.
        if table.nxs[7] > 0 {
            let relaxation = AtomicRelaxation::from_ace(table)?;
            let n_subshells = table.nxs[7];
            let designators: Vec<i64> = (0..n_subshells)
                .map(|i| at(table.jxs[11] + i) as i64)
                .collect();

            let mut idx = table.jxs[16];
            for d in designators {
                let mt = 533 + d as i32;
                let mut rx = PhotonReaction::new(mt);

                // Stored as logarithms above the threshold and zero below it.
                let raw = slice(idx, n_energy);
                let y: Vec<f64> = raw
                    .iter()
                    .map(|&v| if v == 0.0 { 0.0 } else { v.exp() })
                    .collect();
                let threshold = y.iter().position(|&v| v > 0.0).unwrap_or(0);
                rx.xs = Some(Tabulated1D::with_regions(
                    energy[threshold..].to_vec(),
                    y[threshold..].to_vec(),
                    vec![(n_energy - threshold) as i32],
                    vec![5],
                ));
                idx += n_energy as i64;

                rx.subshell_binding_energy = SUBSHELLS
                    .get(d as usize)
                    .and_then(|shell| relaxation.binding_energy.get(shell))
                    .copied();
                data.reactions.insert(mt, rx);
            }
            data.atomic_relaxation = Some(relaxation);
        }

        Ok(data)
    }

    /// Which reactions make up a redundant one, by the same sum rules the
    /// neutron data uses.
    pub fn reaction_components(&self, mt: i32) -> Vec<i32> {
        let mut mts = Vec::new();
        if let Some(rule) = sum_rule(mt) {
            for &component in rule {
                mts.extend(self.reaction_components(component));
            }
        }
        if !mts.is_empty() {
            return mts;
        }
        if self.contains(mt) {
            vec![mt]
        } else {
            Vec::new()
        }
    }
}

/// Cumulative distributions for a set of Compton profiles.
///
/// Each profile is integrated over the momentum grid by the trapezoidal rule.
/// The result is deliberately left un-normalised, so each row ends near 0.5
/// rather than 1: a profile is tabulated for positive momentum only and the
/// missing half is symmetric, which is what samplers expect.
///
/// `j` is `n_shells` rows of `pz.len()` values.
pub fn compton_profile_cdfs(j: &[Vec<f64>], pz: &[f64]) -> Vec<Vec<f64>> {
    j.iter()
        .map(|row| {
            let mut cdf = vec![0.0; row.len()];
            let mut running = 0.0;
            for i in 1..row.len().min(pz.len()) {
                running += 0.5 * (row[i - 1] + row[i]) * (pz[i] - pz[i - 1]);
                cdf[i] = running;
            }
            cdf
        })
        .collect()
}

/// Which atomic-relaxation subshells each Compton-profile shell corresponds to.
///
/// Returned in CSR form: Compton shell `c` maps to the subshells
/// `indices[offsets[c]..offsets[c + 1]]`, weighted by occupancy so the weights
/// of a group sum to one. An empty range means the shell has no clean
/// counterpart, which happens for the outer shells where the two orderings
/// diverge.
pub fn compton_subshell_map(
    compton_num_electrons: &[f64],
    subshell_num_electrons: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut offsets = vec![0usize];
    let mut indices: Vec<usize> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut cursor = 0usize;
    let mut stopped = false;

    for &c_occ in compton_num_electrons {
        if !stopped && c_occ > 0.0 {
            let mut acc = 0.0;
            let mut group = Vec::new();
            let mut j = cursor;
            // Take consecutive subshells until their occupancy reaches this
            // Compton shell's. The tolerance keeps the one that completes it.
            while j < subshell_num_electrons.len() && acc < c_occ - 1e-6 {
                acc += subshell_num_electrons[j];
                group.push(j);
                j += 1;
            }
            if !group.is_empty() && (acc - c_occ).abs() <= 1e-3 * 1.0f64.max(c_occ) {
                for &k in &group {
                    indices.push(k);
                    weights.push(subshell_num_electrons[k] / c_occ);
                }
                cursor = j;
            } else {
                // The first mismatch means the two orderings have diverged, so
                // this shell and every later one are dropped.
                stopped = true;
            }
        }
        offsets.push(indices.len());
    }
    (offsets, indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHOTOAT_H: &[u8] = include_bytes!("../fixtures/photoat-001_H_000.endf.xz");
    const ATOM_H: &[u8] = include_bytes!("../fixtures/atom-001_H_000.endf.xz");

    fn material(raw: &[u8]) -> Material {
        Material::from_str(&crate::testdata::text(raw)).unwrap()
    }

    #[test]
    fn names_the_photon_reactions_both_ways() {
        assert_eq!(photon_reaction_name(502), Some("coherent"));
        assert_eq!(photon_reaction_name(522), Some("photoelectric"));
        // The subshell photoelectric reactions are named for their subshell.
        assert_eq!(photon_reaction_name(534), Some("K"));
        assert_eq!(photon_reaction_name(535), Some("L1"));
        assert_eq!(photon_reaction_name(572), Some("Q3"));
        assert_eq!(photon_reaction_name(1), None);

        assert_eq!(photon_reaction_mt("coherent"), Some(502));
        assert_eq!(photon_reaction_mt("K"), Some(534));
        assert_eq!(photon_reaction_mt("Q3"), Some(572));
        assert_eq!(photon_reaction_mt("nonsense"), None);
        // The empty string is the unused zeroth entry, not a subshell.
        assert_eq!(photon_reaction_mt(""), None);

        for mt in 501..600 {
            if let Some(name) = photon_reaction_name(mt) {
                assert_eq!(photon_reaction_mt(name), Some(mt), "{name} came back wrong");
            }
        }
    }

    #[test]
    fn reads_a_photoatomic_evaluation() {
        let m = material(PHOTOAT_H);
        let d = IncidentPhoton::from_endf(&m, None).unwrap();
        assert_eq!(d.atomic_number, 1);
        assert_eq!(d.name(), "H");
        assert_eq!(
            d.reactions.keys().copied().collect::<Vec<_>>(),
            [501, 502, 504, 515, 516, 517, 522, 534]
        );

        // Coherent scattering carries a form factor and both anomalous terms,
        // which arrive from MF=27 under MTs of their own.
        let coherent = d.get(502).unwrap();
        assert!(coherent.scattering_factor.is_some());
        assert!(coherent.anomalous_real.is_some());
        assert!(coherent.anomalous_imag.is_some());
        // Incoherent has a form factor and no anomalous terms.
        let incoherent = d.get(504).unwrap();
        assert!(incoherent.scattering_factor.is_some());
        assert!(incoherent.anomalous_real.is_none());

        // The K-shell photoelectric reaction carries its binding energy and
        // fluorescence yield; the total does not.
        let k = d.get_by_name("K").unwrap();
        assert_eq!(k.mt, 534);
        assert_eq!(k.subshell_binding_energy, Some(13.6));
        assert!(k.fluorescence_yield.is_some());
        assert!(d.get(501).unwrap().subshell_binding_energy.is_none());

        // The total is a sum of the channels below it.
        let components = d.reaction_components(501);
        assert!(components.len() > 1);
        assert!(!components.contains(&501));
    }

    #[test]
    fn relaxation_data_can_come_from_a_separate_evaluation() {
        let photoatomic = material(PHOTOAT_H);
        let relaxation = material(ATOM_H);

        // Hydrogen's photoatomic file carries no MF=28 of its own.
        let without = IncidentPhoton::from_endf(&photoatomic, None).unwrap();
        assert!(without.atomic_relaxation.is_none());

        let with = IncidentPhoton::from_endf(&photoatomic, Some(&relaxation)).unwrap();
        let ar = with.atomic_relaxation.unwrap();
        // Hydrogen has one electron, in the K shell, bound by 13.6 eV.
        assert_eq!(ar.subshells(), ["K"]);
        assert_eq!(ar.binding_energy["K"], 13.6);
        assert_eq!(ar.num_electrons["K"], 1.0);
        // Nothing to relax into, so no transitions.
        assert!(ar.transitions.is_empty());
    }

    #[test]
    fn subshells_come_back_in_the_formats_order() {
        // Alphabetically L1 sorts before M1 but after K; the format's order is
        // by binding energy, which is what `subshells` has to give.
        let ar = AtomicRelaxation {
            binding_energy: BTreeMap::from([("M1", 1.0), ("K", 3.0), ("L2", 2.0)]),
            ..Default::default()
        };
        assert_eq!(ar.subshells(), ["K", "L2", "M1"]);
    }

    #[test]
    fn a_compton_cdf_integrates_the_profile() {
        // A flat profile of height 2 over a unit-spaced grid accumulates 2 per
        // step by the trapezoidal rule.
        let pz = [0.0, 1.0, 2.0, 3.0];
        let j = vec![vec![2.0, 2.0, 2.0, 2.0]];
        assert_eq!(
            compton_profile_cdfs(&j, &pz),
            vec![vec![0.0, 2.0, 4.0, 6.0]]
        );

        // A triangle: the first step averages 0 and 1.
        let j = vec![vec![0.0, 1.0, 0.0, 0.0]];
        assert_eq!(
            compton_profile_cdfs(&j, &pz),
            vec![vec![0.0, 0.5, 1.0, 1.0]]
        );
    }

    #[test]
    fn compton_shells_map_onto_the_subshells_that_fill_them() {
        // Two Compton shells of 2 and 8 electrons against K, L1, L2, L3.
        let (offsets, indices, weights) = compton_subshell_map(&[2.0, 8.0], &[2.0, 2.0, 2.0, 4.0]);
        assert_eq!(offsets, [0, 1, 4]);
        assert_eq!(indices, [0, 1, 2, 3]);
        assert_eq!(weights, [1.0, 0.25, 0.25, 0.5]);
        // The weights of each group sum to one.
        assert_eq!(weights[1] + weights[2] + weights[3], 1.0);
    }

    #[test]
    fn a_mismatched_occupancy_stops_the_mapping() {
        // The second Compton shell cannot be made from whole subshells, so it
        // and everything after it is dropped.
        let (offsets, indices, _) = compton_subshell_map(&[2.0, 7.0, 4.0], &[2.0, 2.0, 2.0, 4.0]);
        assert_eq!(offsets, [0, 1, 1, 1]);
        assert_eq!(indices, [0]);
    }
}
