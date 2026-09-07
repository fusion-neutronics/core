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
    /// The excitation energy the file's own header states (MF=1 MT=451
    /// ELIS), when it states one. Zero there means unknown, not the ground
    /// state: the header of a metastable file is describing that isomer.
    elis: Option<f64>,
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
/// Each isomer's absolute excitation energy is the one its decay file states
/// in the MF=1 MT=451 header (ELIS), which 624 of ENDF/B-VIII.1's 738
/// metastable files give. Where the header says nothing the energy is
/// recovered from the Q value of the isomeric-transition decay mode
/// (RTYP = 3), chained through that mode's final isomeric state down to lower
/// isomers; a pure-beta isomer with no stated energy has none.
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
        let elis = material
            .mf1_mt451()
            .map(|header| header.elis)
            .filter(|&e| e > 0.0);

        raw.entry((z, a)).or_default().insert(
            section.liso,
            RawIsomer {
                lis: section.lis,
                half_life: section.half_life.map(|(v, _)| v),
                elis,
                it_q: it.map(|m| m.q.0),
                it_rfs: it.map_or(0, |m| m.rfs as i64),
            },
        );
    }

    raw.into_iter()
        .map(|(za, isomers)| (za, chain_isomer_energies(&isomers)))
        .collect()
}

/// The absolute excitation energy of each isomer of one nuclide.
///
/// The header's ELIS is taken as stated. Without it, an isomer's energy is the
/// Q of its transition plus the energy of the state that transition lands on,
/// so the states are walked from the lowest ordinal up and each one reads
/// back what was resolved before it. A state whose transition lands on a state
/// of unknown energy, or whose transition carries no Q at all, has an unknown
/// energy too: `None`, not the bare Q. Booking the bare Q used to put In116's
/// second isomer at 162 keV (its 290 keV transition lands on the pure-beta
/// first isomer), and a wrong energy is worse than a missing one, because a
/// production level that happens to sit near 162 keV then matches it.
fn chain_isomer_energies(isomers: &BTreeMap<i64, RawIsomer>) -> BTreeMap<i64, Isomer> {
    let mut resolved: BTreeMap<i64, Option<f64>> = BTreeMap::from([(0, Some(0.0))]);
    let mut out = BTreeMap::new();
    for (&liso, info) in isomers {
        let energy = if liso == 0 {
            Some(0.0)
        } else if info.elis.is_some() {
            info.elis
        } else {
            match (info.it_q, resolved.get(&info.it_rfs).copied().flatten()) {
                (Some(q), Some(base)) if q > 0.0 => Some(q + base),
                _ => None,
            }
        };
        resolved.insert(liso, energy);
        out.insert(
            liso,
            Isomer {
                lis: info.lis,
                half_life: info.half_life,
                e_iso: energy,
            },
        );
    }
    out
}

/// Default tolerance in eV for matching a level energy to an isomer energy.
pub const ISOMER_ENERGY_TOLERANCE: f64 = 3000.0;

/// Relative tolerance of the second, looser energy pass.
///
/// A production level and the decay data can place the same isomer some way
/// apart when they descend from different level schemes: TENDL-2017 puts the
/// 5.5 s isomer of Ir191 at 2201 keV where ENDF/B-VIII.1's transition energies
/// put it at 2046 keV. Within a tenth of the isomer's own energy, and with no
/// other isomer that close, it is the same state.
pub const ISOMER_ENERGY_RELATIVE_TOLERANCE: f64 = 0.10;

/// How a production level was matched to an isomeric state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LevelRoute {
    /// The ground state: level 0, no energy given, or below a keV.
    Ground,
    /// The decay data has no metastable state for this nuclide.
    NoIsomers,
    /// The energy matched an isomer within [`ISOMER_ENERGY_TOLERANCE`].
    Energy,
    /// No isomer within the absolute tolerance, but exactly one within
    /// [`ISOMER_ENERGY_RELATIVE_TOLERANCE`] of its own energy.
    NearEnergy,
    /// The level index equals an isomer's LIS.
    LevelIndex,
    /// The nuclide has one isomer, so the level can only mean that one.
    SingleIsomer,
    /// Nothing matched; the level is taken to cascade to ground.
    Unresolved,
}

impl LevelRoute {
    /// The route as a short label, for statistics keyed by name.
    pub fn label(self) -> &'static str {
        match self {
            LevelRoute::Ground => "ground",
            LevelRoute::NoIsomers => "no_isomers",
            LevelRoute::Energy => "energy",
            LevelRoute::NearEnergy => "near_energy",
            LevelRoute::LevelIndex => "level_index",
            LevelRoute::SingleIsomer => "single_isomer",
            LevelRoute::Unresolved => "unresolved",
        }
    }
}

/// A production level resolved to an isomeric state, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLevel {
    /// The isomeric-state ordinal; 0 is the ground state.
    pub liso: i64,
    pub route: LevelRoute,
    /// The isomer the level index pointed at, when the energy picked another.
    /// The two signals come from two level schemes and can disagree, as they
    /// do for Pm152, where the 150 keV level carries the first isomer's index
    /// and the second isomer's transition energy; the caller is told rather
    /// than left to trust whichever won.
    pub conflicting_liso: Option<i64>,
}

/// Map a production level to an isomeric-state ordinal.
///
/// The ground state maps to 0. Otherwise the level's excitation energy is
/// matched against the isomer energies in `table`, first within `tol_ev` and
/// then, for a single candidate, within a tenth of the isomer's energy;
/// failing that, the level index is compared against LIS; failing that, a
/// nuclide with exactly one isomer maps to it. A level that resolves to none
/// of these is treated as ground, on the basis that a short-lived level
/// gamma-cascades down. [`resolve_level`] says which of these happened.
pub fn level_to_isomeric_state(
    z: i64,
    a: i64,
    lfs: i64,
    excitation_energy: Option<f64>,
    table: &IsomerTable,
    tol_ev: f64,
) -> i64 {
    resolve_level(z, a, lfs, excitation_energy, table, tol_ev).liso
}

/// As [`level_to_isomeric_state`], with the route taken and any conflict.
pub fn resolve_level(
    z: i64,
    a: i64,
    lfs: i64,
    excitation_energy: Option<f64>,
    table: &IsomerTable,
    tol_ev: f64,
) -> ResolvedLevel {
    let resolved = |liso, route| ResolvedLevel {
        liso,
        route,
        conflicting_liso: None,
    };
    let Some(isomers) = table.get(&(z, a)) else {
        return resolved(0, LevelRoute::NoIsomers);
    };
    let metastable: Vec<(&i64, &Isomer)> = isomers.iter().filter(|(&liso, _)| liso > 0).collect();
    if metastable.is_empty() {
        return resolved(0, LevelRoute::NoIsomers);
    }
    // A level below a keV is not a metastable state; nor is the ground state,
    // whatever energy it is given.
    let excitation_energy = match excitation_energy {
        _ if lfs == 0 => return resolved(0, LevelRoute::Ground),
        None => return resolved(0, LevelRoute::Ground),
        Some(e) if e < 1000.0 => return resolved(0, LevelRoute::Ground),
        Some(e) => e,
    };

    // What the level index says, read once: it decides the third step and
    // qualifies the first two.
    let by_index = metastable
        .iter()
        .find(|(_, isomer)| isomer.lis == lfs)
        .map(|(&liso, _)| liso);
    let conflict = |liso: i64| by_index.filter(|&other| other != liso);

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
            return ResolvedLevel {
                liso,
                route: LevelRoute::Energy,
                conflicting_liso: conflict(liso),
            };
        }
    }

    // 1b. The looser pass: one isomer, and only one, within a tenth of its
    // own energy.
    let near: Vec<i64> = metastable
        .iter()
        .filter(|(_, isomer)| {
            isomer.e_iso.is_some_and(|e_iso| {
                (excitation_energy - e_iso).abs() <= ISOMER_ENERGY_RELATIVE_TOLERANCE * e_iso
            })
        })
        .map(|(&liso, _)| liso)
        .collect();
    if let [liso] = near[..] {
        return ResolvedLevel {
            liso,
            route: LevelRoute::NearEnergy,
            conflicting_liso: conflict(liso),
        };
    }

    // 2. The level index.
    if let Some(liso) = by_index {
        return resolved(liso, LevelRoute::LevelIndex);
    }

    // 3. A nuclide with one isomer can only mean that one.
    if let [(&liso, _)] = metastable[..] {
        return resolved(liso, LevelRoute::SingleIsomer);
    }

    // 4. Unresolved, so cascade to ground.
    resolved(0, LevelRoute::Unresolved)
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
        let level = resolve_level(95, 242, 2, Some(9.0e6), &table, ISOMER_ENERGY_TOLERANCE);
        assert_eq!((level.liso, level.route), (2, LevelRoute::LevelIndex));
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
        // Pure beta-, so no transition to measure the energy by; the header
        // states it.
        assert_eq!(in116[&1].e_iso, Some(127_267.0));

        assert_eq!(in116[&2].lis, 4);
        assert_eq!(in116[&2].half_life, Some(2.18));
        // The header's 289.66 keV. Chaining its 162 keV transition through
        // the first isomer would give 289.66 keV too, now that the first
        // isomer's energy is known, and used to give 162 keV.
        assert_eq!(in116[&2].e_iso, Some(289_660.0));
    }

    /// Without a header energy the transitions are chained; with one it wins,
    /// and it feeds the chaining of the states above it.
    #[test]
    fn a_stated_excitation_energy_is_taken_over_the_chained_one() {
        let raw = |lis, elis: Option<f64>, it_q: Option<f64>, it_rfs| RawIsomer {
            lis,
            half_life: Some(1.0),
            elis,
            it_q,
            it_rfs,
        };
        let isomers = BTreeMap::from([
            // A pure-beta isomer whose header states 127 keV.
            (1, raw(1, Some(127_267.0), None, 0)),
            // Its header is silent; the transition to state 1 is chained.
            (2, raw(4, None, Some(162_393.0), 1)),
            // The header disagrees with the chain, and the header wins.
            (3, raw(5, Some(500_000.0), Some(100_000.0), 2)),
        ]);
        let out = chain_isomer_energies(&isomers);
        assert_eq!(out[&1].e_iso, Some(127_267.0));
        assert_eq!(out[&2].e_iso, Some(289_660.0));
        assert_eq!(out[&3].e_iso, Some(500_000.0));
    }

    /// The chaining, on its own: a transition to a resolved state adds up, a
    /// transition to an unknown one or with no Q gives an unknown energy.
    #[test]
    fn isomer_energies_chain_only_through_known_states() {
        let raw = |lis, it_q: Option<f64>, it_rfs| RawIsomer {
            lis,
            half_life: Some(1.0),
            elis: None,
            it_q,
            it_rfs,
        };
        let isomers = BTreeMap::from([
            (1, raw(1, Some(48_600.0), 0)),
            (2, raw(2, Some(2_000_000.0), 1)),
            (3, raw(3, None, 0)),
            (4, raw(4, Some(100_000.0), 3)),
            (5, raw(5, Some(0.0), 0)),
        ]);
        let out = chain_isomer_energies(&isomers);
        assert_eq!(out[&1].e_iso, Some(48_600.0));
        assert_eq!(out[&2].e_iso, Some(2_048_600.0));
        // Pure beta: nothing to measure the energy by.
        assert_eq!(out[&3].e_iso, None);
        // Lands on a state of unknown energy.
        assert_eq!(out[&4].e_iso, None);
        // A transition the evaluation gave no Q for.
        assert_eq!(out[&5].e_iso, None);
    }

    /// Ir191 as TENDL-2017 and ENDF/B-VIII.1 describe it: the production level
    /// at 2201 keV is the 5.5 s isomer the decay data puts at 2046 keV.
    #[test]
    fn a_lone_isomer_within_a_tenth_of_its_energy_is_taken() {
        let table = IsomerTable::from([(
            (77, 191),
            BTreeMap::from([
                (
                    1,
                    Isomer {
                        lis: 1,
                        half_life: Some(4.94),
                        e_iso: Some(171_290.0),
                    },
                ),
                (
                    2,
                    Isomer {
                        lis: 2,
                        half_life: Some(5.5),
                        e_iso: Some(2_046_000.0),
                    },
                ),
            ]),
        )]);
        let level = resolve_level(
            77,
            191,
            30,
            Some(2_201_000.0),
            &table,
            ISOMER_ENERGY_TOLERANCE,
        );
        assert_eq!(level.liso, 2);
        assert_eq!(level.route, LevelRoute::NearEnergy);
        assert_eq!(level.conflicting_liso, None);
        // Twice the energy is not the same state.
        let level = resolve_level(
            77,
            191,
            30,
            Some(4_000_000.0),
            &table,
            ISOMER_ENERGY_TOLERANCE,
        );
        assert_eq!((level.liso, level.route), (0, LevelRoute::Unresolved));
    }

    /// Pm152 as the two libraries describe it: the level index says the first
    /// isomer, the transition energy says the second. The energy wins, as it
    /// always did, and the disagreement is reported.
    #[test]
    fn a_level_index_pointing_elsewhere_is_reported() {
        let table = IsomerTable::from([(
            (61, 152),
            BTreeMap::from([
                (
                    1,
                    Isomer {
                        lis: 4,
                        half_life: Some(451.2),
                        e_iso: None,
                    },
                ),
                (
                    2,
                    Isomer {
                        lis: 2,
                        half_life: Some(828.0),
                        e_iso: Some(150_000.0),
                    },
                ),
            ]),
        )]);
        let level = resolve_level(61, 152, 4, Some(150_000.0), &table, ISOMER_ENERGY_TOLERANCE);
        assert_eq!(level.liso, 2);
        assert_eq!(level.route, LevelRoute::Energy);
        assert_eq!(level.conflicting_liso, Some(1));
        // The 350 keV level matches nothing and has no index match either.
        let level = resolve_level(
            61,
            152,
            10,
            Some(350_000.0),
            &table,
            ISOMER_ENERGY_TOLERANCE,
        );
        assert_eq!((level.liso, level.route), (0, LevelRoute::Unresolved));
        // Agreement is not a conflict.
        let level = resolve_level(61, 152, 2, Some(150_000.0), &table, ISOMER_ENERGY_TOLERANCE);
        assert_eq!(level.conflicting_liso, None);
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
