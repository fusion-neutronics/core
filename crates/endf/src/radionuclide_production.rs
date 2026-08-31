//! Radionuclide production: which nuclides a reaction leaves behind, in which
//! state, and how much of each.
//!
//! MF=8 identifies each radioactive product and MF=9 or MF=10 gives the
//! energy-dependent yield or production cross section. The three are joined
//! here, keyed by reaction, because a consumer wants them together.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::Result;
use crate::function::Tabulated1D;
use crate::material::Material;

/// Production data for one final state of one reaction.
///
/// LFS is a level index of the *product* nuclide, not an isomeric-state
/// ordinal. Turning one into the other needs decay data; see
/// [`level_to_isomeric_state`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RadionuclideProduction {
    /// `1000*Z + A` of the product nuclide.
    pub zap: i64,
    /// Level number of the final state; 0 is the ground state.
    pub lfs: i64,
    /// Mass-difference Q value in eV.
    pub qm: f64,
    /// Reaction Q value for this state, in eV.
    pub qi: f64,
    /// Excitation energy of the final state in eV, from MF=8. `None` when the
    /// evaluation has no MF=8 subsection for this state.
    pub elfs: Option<f64>,
    /// MF=9 yield, as a multiplier on the reaction cross section.
    pub yields: Option<Tabulated1D>,
    /// MF=10 production cross section in barns.
    pub cross_section: Option<Tabulated1D>,
}

impl RadionuclideProduction {
    /// Excitation energy of the final state in eV.
    ///
    /// The MF=8 value when the evaluation gave one, and `QM - QI` otherwise,
    /// the same quantity read off the Q values.
    pub fn excitation_energy(&self) -> f64 {
        self.elfs.unwrap_or(self.qm - self.qi)
    }
}

/// Collect the radionuclide production data of a material, by MT.
///
/// Every reaction with an MF=9 or MF=10 section is included, its final states
/// in the order the evaluation writes them. MF=9 and MF=10 data for the same
/// `(ZAP, LFS)` pair are merged into one entry, and the MF=8 excitation energy
/// attached where there is one. The tabulated functions are returned exactly
/// as evaluated.
pub fn radionuclide_production(material: &Material) -> BTreeMap<i32, Vec<RadionuclideProduction>> {
    let mut by_mt: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
    for &(mf, mt) in material.section_data.keys() {
        if mf == 9 || mf == 10 {
            by_mt.entry(mt).or_default().insert(mf);
        }
    }

    let mut result = BTreeMap::new();
    for (mt, files) in by_mt {
        // MF=8 links each (ZAP, LFS) pair to an excitation energy.
        let mut elfs: BTreeMap<(i64, i64), f64> = BTreeMap::new();
        if let Some(mf8) = material.mf8(mt) {
            for sub in &mf8.subsections {
                elfs.insert((sub.zap as i64, sub.lfs), sub.elfs);
            }
        }

        // Insertion order is the evaluation's order, which is what a reader
        // expects back, so the states are kept in a vector and located by a
        // side map rather than being sorted.
        let mut ordered: Vec<RadionuclideProduction> = Vec::new();
        let mut index: BTreeMap<(i64, i64), usize> = BTreeMap::new();
        for mf in files {
            let section = match mf {
                9 => material.mf9(mt),
                _ => material.mf10(mt),
            };
            let Some(section) = section else { continue };
            for level in &section.levels {
                let key = (level.izap, level.lfs);
                let i = *index.entry(key).or_insert_with(|| {
                    ordered.push(RadionuclideProduction {
                        zap: key.0,
                        lfs: key.1,
                        qm: level.qm,
                        qi: level.qi,
                        elfs: elfs.get(&key).copied(),
                        ..Default::default()
                    });
                    ordered.len() - 1
                });
                if mf == 9 {
                    ordered[i].yields = Some(level.func.clone());
                } else {
                    ordered[i].cross_section = Some(level.func.clone());
                }
            }
        }
        result.insert(mt, ordered);
    }
    result
}

/// One isomeric state of a nuclide, as decay data describes it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Isomer {
    /// The nuclear level index the evaluation gives this state.
    pub lis: i64,
    /// Half-life in seconds, where the evaluation gave one.
    pub half_life: Option<f64>,
    /// Excitation energy in eV. `None` for a pure-beta isomer, which has no
    /// isomeric transition to measure it by.
    pub e_iso: Option<f64>,
}

/// Isomeric states by `(Z, A)`, then by isomeric-state ordinal (LISO).
pub type IsomerTable = BTreeMap<(i64, i64), BTreeMap<i64, Isomer>>;

/// What one decay file said, before the excitation energies are chained.
#[derive(Debug, Clone, Default)]
struct RawIsomer {
    lis: i64,
    half_life: Option<f64>,
    /// Q value of the isomeric-transition decay mode, when there is one.
    it_q: Option<f64>,
    /// The isomeric state that transition leaves behind.
    it_rfs: i64,
}

/// Build a table of isomeric states from decay data evaluations.
///
/// MF=8 identifies a radioactive product by a nuclear *level* index, not by an
/// isomeric-state ordinal, so relating production data to a named metastable
/// nuclide needs the excitation energies of the isomers, which is what decay
/// data provides.
///
/// Each isomer's absolute excitation energy is recovered from the Q value of
/// its isomeric-transition decay mode (RTYP = 3), chained through that mode's
/// final isomeric state down to lower isomers.
///
/// Only the metastable files are needed; ground states are implicit.
pub fn isomer_table<I, P>(decay_files: I) -> Result<IsomerTable>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut materials = Vec::new();
    for filename in decay_files {
        materials.push(Material::from_file(filename.as_ref())?);
    }
    Ok(isomer_table_from_materials(&materials))
}

/// The same, from materials already read.
///
/// This is where the work happens; [`isomer_table`] is the convenience that
/// opens the files first.
pub fn isomer_table_from_materials(materials: &[Material]) -> IsomerTable {
    let mut raw: BTreeMap<(i64, i64), BTreeMap<i64, RawIsomer>> = BTreeMap::new();
    for material in materials {
        let Some(section) = material.mf8_mt457() else {
            continue;
        };
        let (z, a) = (section.za / 1000, section.za % 1000);

        // The first isomeric transition is the one that fixes the energy; a
        // second would describe the same level.
        let it = section.modes.iter().find(|m| m.rtyp == 3.0);

        raw.entry((z, a)).or_default().insert(
            section.liso,
            RawIsomer {
                lis: section.lis,
                half_life: section.half_life.map(|(v, _)| v),
                it_q: it.map(|m| m.q.0),
                it_rfs: it.map_or(0, |m| m.rfs as i64),
            },
        );
    }

    let mut table = IsomerTable::new();
    for (za, isomers) in raw {
        // Energies resolved so far, which the chaining reads back.
        let mut resolved: BTreeMap<i64, f64> = BTreeMap::from([(0, 0.0)]);
        let mut out = BTreeMap::new();
        for (&liso, info) in &isomers {
            let energy = if liso == 0 {
                Some(0.0)
            } else {
                info.it_q
                    .map(|q| q + resolved.get(&info.it_rfs).copied().unwrap_or(0.0))
            };
            resolved.insert(liso, energy.unwrap_or(0.0));
            out.insert(
                liso,
                Isomer {
                    lis: info.lis,
                    half_life: info.half_life,
                    e_iso: energy,
                },
            );
        }
        table.insert(za, out);
    }
    table
}

/// Default tolerance in eV for matching a level energy to an isomer energy.
pub const ISOMER_ENERGY_TOLERANCE: f64 = 3000.0;

/// Map a production level to an isomeric-state ordinal.
///
/// The ground state maps to 0. Otherwise the level's excitation energy is
/// matched against the isomer energies in `table`; failing that, the level
/// index is compared against LIS; failing that, a nuclide with exactly one
/// isomer maps to it. A level that resolves to none of these is treated as
/// ground, on the basis that a short-lived level gamma-cascades down.
pub fn level_to_isomeric_state(
    z: i64,
    a: i64,
    lfs: i64,
    excitation_energy: Option<f64>,
    table: &IsomerTable,
    tol_ev: f64,
) -> i64 {
    let Some(isomers) = table.get(&(z, a)) else {
        return 0;
    };
    let metastable: Vec<(&i64, &Isomer)> = isomers.iter().filter(|(&liso, _)| liso > 0).collect();
    if metastable.is_empty() {
        return 0;
    }
    // A level below a keV is not a metastable state; nor is the ground state,
    // whatever energy it is given.
    match excitation_energy {
        _ if lfs == 0 => return 0,
        None => return 0,
        Some(e) if e < 1000.0 => return 0,
        Some(_) => {}
    }
    let excitation_energy = excitation_energy.unwrap_or(0.0);

    // 1. The energy match against the decay isomer energies.
    let mut best: Option<(i64, f64)> = None;
    for (&liso, isomer) in &metastable {
        if let Some(e_iso) = isomer.e_iso {
            let residual = (excitation_energy - e_iso).abs();
            match best {
                Some((_, r)) if r <= residual => {}
                _ => best = Some((liso, residual)),
            }
        }
    }
    if let Some((liso, residual)) = best {
        if residual <= tol_ev {
            return liso;
        }
    }

    // 2. The level index.
    for (&liso, isomer) in &metastable {
        if isomer.lis == lfs {
            return liso;
        }
    }

    // 3. A nuclide with one isomer can only mean that one.
    if metastable.len() == 1 {
        return *metastable[0].0;
    }

    // 4. Unresolved, so cascade to ground.
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN115: &[u8] = include_bytes!("../fixtures/n-049_In-115_trimmed.endf.xz");

    #[test]
    fn joins_mf8_mf9_and_mf10_for_each_reaction() {
        let m = Material::from_str(&crate::testdata::text(IN115)).unwrap();
        let production = radionuclide_production(&m);

        // The evaluation gives isomer production for three reactions, each to
        // a single excited state.
        assert_eq!(production.keys().copied().collect::<Vec<_>>(), [4, 16, 102]);
        assert!(production.values().all(|states| states.len() == 1));

        // Inelastic scattering and (n,2n) leave indium behind, and both are
        // given as MF=10 production cross sections.
        for (mt, zap) in [(4, 49115), (16, 49114)] {
            let state = &production[&mt][0];
            assert_eq!((state.zap, state.lfs), (zap, 1));
            assert!(state.cross_section.is_some());
            assert!(state.yields.is_none());
        }

        // Capture is given the other way, as an MF=9 yield on the MF=3 cross
        // section.
        let state = &production[&102][0];
        assert_eq!((state.zap, state.lfs), (49116, 1));
        assert!(state.yields.is_some());
        assert!(state.cross_section.is_none());

        // The excitation energy comes from MF=8's ELFS, not from QM - QI.
        assert_eq!(state.elfs, Some(127_269.7));
        assert_eq!(state.excitation_energy(), 127_269.7);
    }

    #[test]
    fn the_excitation_energy_falls_back_to_the_q_values() {
        // With no MF=8 subsection the energy is QM - QI, which is the same
        // quantity the evaluation would have written as ELFS.
        let state = RadionuclideProduction {
            qm: 6.0e6,
            qi: 5.8e6,
            elfs: None,
            ..Default::default()
        };
        assert_eq!(state.excitation_energy(), 2.0e5);

        let state = RadionuclideProduction {
            elfs: Some(1.0e5),
            ..state
        };
        assert_eq!(state.excitation_energy(), 1.0e5);
    }

    /// A two-isomer nuclide, as decay data would leave it: the second isomer
    /// transitions to the first, so its energy is the sum of the two Q values.
    fn two_isomers() -> IsomerTable {
        IsomerTable::from([(
            (95, 242),
            BTreeMap::from([
                (
                    1,
                    Isomer {
                        lis: 1,
                        half_life: Some(4.4e9),
                        e_iso: Some(48_600.0),
                    },
                ),
                (
                    2,
                    Isomer {
                        lis: 2,
                        half_life: Some(1.4e4),
                        e_iso: Some(2_200_000.0),
                    },
                ),
            ]),
        )])
    }

    #[test]
    fn an_energy_within_tolerance_picks_its_isomer() {
        let table = two_isomers();
        let at = |e: f64, lfs: i64| {
            level_to_isomeric_state(95, 242, lfs, Some(e), &table, ISOMER_ENERGY_TOLERANCE)
        };
        assert_eq!(at(48_600.0, 1), 1);
        assert_eq!(at(50_000.0, 1), 1);
        assert_eq!(at(2_200_500.0, 2), 2);
    }

    #[test]
    fn the_ground_state_and_low_levels_are_ground() {
        let table = two_isomers();
        let at = |e: Option<f64>, lfs: i64| {
            level_to_isomeric_state(95, 242, lfs, e, &table, ISOMER_ENERGY_TOLERANCE)
        };
        assert_eq!(at(Some(0.0), 0), 0);
        // A level index above zero but an energy too low to be metastable.
        assert_eq!(at(Some(500.0), 1), 0);
        // Nothing known about the energy.
        assert_eq!(at(None, 1), 0);
        // A nuclide the table has never heard of.
        assert_eq!(
            level_to_isomeric_state(1, 1, 1, Some(1.0e6), &table, ISOMER_ENERGY_TOLERANCE),
            0
        );
    }

    #[test]
    fn a_level_index_resolves_what_energy_cannot() {
        let table = two_isomers();
        // Far from either isomer energy, but the level index says which.
        assert_eq!(
            level_to_isomeric_state(95, 242, 2, Some(9.0e6), &table, ISOMER_ENERGY_TOLERANCE),
            2
        );
        // Neither energy nor index matches, and there are two isomers, so the
        // level is taken to cascade to ground.
        assert_eq!(
            level_to_isomeric_state(95, 242, 7, Some(9.0e6), &table, ISOMER_ENERGY_TOLERANCE),
            0
        );
    }

    #[test]
    fn a_nuclide_with_only_a_ground_state_is_ground() {
        let table = IsomerTable::from([(
            (26, 56),
            BTreeMap::from([(
                0,
                Isomer {
                    lis: 0,
                    half_life: None,
                    e_iso: Some(0.0),
                },
            )]),
        )]);
        assert_eq!(
            level_to_isomeric_state(26, 56, 3, Some(8.0e5), &table, ISOMER_ENERGY_TOLERANCE),
            0
        );
    }

    #[test]
    fn reads_the_isomers_of_in116_from_decay_data() {
        // The two metastable states of In116, which between them exercise both
        // branches: the first decays only by beta-, so it has no isomeric
        // transition to measure its energy by, and the second transitions down
        // to the first.
        const M1: &[u8] = include_bytes!("../fixtures/dec-049_In_116m1.endf.xz");
        const M2: &[u8] = include_bytes!("../fixtures/dec-049_In_116m2.endf.xz");
        let materials =
            [M1, M2].map(|raw| Material::from_str(&crate::testdata::text(raw)).unwrap());
        let table = isomer_table_from_materials(&materials);

        let in116 = &table[&(49, 116)];
        assert_eq!(in116.keys().copied().collect::<Vec<_>>(), [1, 2]);

        assert_eq!(in116[&1].lis, 1);
        assert_eq!(in116[&1].half_life, Some(3257.4));
        // Pure beta-, so the energy is unknown rather than zero.
        assert_eq!(in116[&1].e_iso, None);

        assert_eq!(in116[&2].lis, 4);
        assert_eq!(in116[&2].half_life, Some(2.18));
        // Its isomeric transition goes to state 1, whose energy is unknown and
        // so contributes nothing; the Q value stands alone.
        assert_eq!(in116[&2].e_iso, Some(162_393.0));
    }

    #[test]
    fn a_nuclide_with_one_isomer_resolves_to_it() {
        let table = IsomerTable::from([(
            (49, 116),
            BTreeMap::from([(
                1,
                Isomer {
                    lis: 1,
                    half_life: Some(3.3e3),
                    e_iso: None,
                },
            )]),
        )]);
        // No energy to match and no index to match, but only one candidate.
        assert_eq!(
            level_to_isomeric_state(49, 116, 4, Some(9.0e5), &table, ISOMER_ENERGY_TOLERANCE),
            1
        );
    }
}
