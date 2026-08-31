//! Chain reaction type names and their ENDF MT numbers.
//!
//! The single source of truth for this mapping. It lives in `yani` rather than
//! in `yani-transmute` because the names are the chain file's own vocabulary,
//! produced by the chain reader in this crate, and because more than one
//! consumer needs to resolve them: the transmutation driver in `yani-transmute`
//! and D1S decay photon production in `yamc-physics`.
//!
//! Keeping a second copy is how `(n,2nd)` came to be mapped to MT 35
//! ((n,nd2a)) in the D1S path while every other copy in the workspace, and the
//! `endf` chain builder that wrote the names, said MT 11.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Shared mapping of chain reaction type names to MT numbers.
///
/// Used by both the operator (for standalone rate computation) and
/// transmutation tallies (for transport scoring).
///
/// The first name listed for an MT is its canonical one, since
/// [`mt_to_reaction_type`] returns the first match and its result is used as a
/// rate key that has to match the chain's own reaction kind. Later names for
/// the same MT are accepted aliases.
pub const REACTION_MT_MAP: &[(&str, i32)] = &[
    // Basic reactions
    ("(n,gamma)", 102),
    // The chain files spell MT 18 "fission", not "(n,fission)": all 88 fission
    // rows of the ENDF/B-VIII.1 reactions table use the bare name. Without it
    // here the kind resolves to no MT, so no fission rate is ever computed and
    // MT 18 is never tallied, which silently removes fission from the
    // transmutation network entirely.
    ("fission", 18),
    ("(n,fission)", 18),
    // Multiple neutron emission
    ("(n,2n)", 16),
    ("(n,3n)", 17),
    ("(n,4n)", 37),
    ("(n,5n)", 152),
    ("(n,6n)", 153),
    ("(n,7n)", 160),
    ("(n,8n)", 161),
    // Single charged particle emission
    ("(n,p)", 103),
    ("(n,d)", 104),
    ("(n,t)", 105),
    ("(n,3He)", 106),
    ("(n,a)", 107),
    // Two particle emission
    ("(n,2p)", 111),
    ("(n,2a)", 108),
    ("(n,3a)", 109),
    ("(n,na)", 22),
    ("(n,np)", 28),
    ("(n,nd)", 32),
    ("(n,nt)", 33),
    ("(n,pa)", 112),
    ("(n,da)", 117),
    ("(n,ta)", 155),
    ("(n,pd)", 115),
    ("(n,pt)", 116),
    ("(n,dt)", 182),
    ("(n,n3He)", 34),
    ("(n,2nd)", 11),
    // Three particle emission
    ("(n,n2a)", 29),
    ("(n,n3a)", 23),
    ("(n,2na)", 24),
    ("(n,3na)", 25),
    ("(n,2n2a)", 30),
    ("(n,nd2a)", 35),
    ("(n,nt2a)", 36),
    ("(n,t2a)", 113),
    ("(n,d2a)", 114),
    ("(n,npa)", 45),
    ("(n,nda)", 158),
    ("(n,nta)", 189),
    ("(n,2np)", 41),
    ("(n,3np)", 42),
    ("(n,n2p)", 44),
    ("(n,2nt)", 154),
    ("(n,3Hea)", 193),
    ("(n,npd)", 183),
    ("(n,npt)", 184),
    ("(n,ndt)", 185),
    ("(n,p3He)", 191),
    ("(n,d3He)", 192),
    ("(n,np3He)", 186),
    ("(n,nd3He)", 187),
    ("(n,nt3He)", 188),
    // Higher multiplicity emissions
    ("(n,4np)", 156),
    ("(n,5np)", 162),
    ("(n,6np)", 163),
    ("(n,7np)", 164),
    ("(n,3nd)", 157),
    ("(n,4nd)", 169),
    ("(n,5nd)", 170),
    ("(n,6nd)", 171),
    ("(n,3nt)", 172),
    ("(n,4nt)", 173),
    ("(n,5nt)", 174),
    ("(n,6nt)", 175),
    ("(n,4na)", 165),
    ("(n,5na)", 166),
    ("(n,6na)", 167),
    ("(n,7na)", 168),
    ("(n,2n3He)", 176),
    ("(n,3n3He)", 177),
    ("(n,4n3He)", 178),
    ("(n,3n2p)", 179),
    ("(n,3n2a)", 180),
    ("(n,3npa)", 181),
    ("(n,2npa)", 159),
    ("(n,2n2p)", 190),
    ("(n,4n2p)", 194),
    ("(n,4n2a)", 195),
    ("(n,4npa)", 196),
    ("(n,3p)", 197),
    ("(n,n3p)", 198),
    ("(n,3n2pa)", 199),
    ("(n,5n2p)", 200),
];

/// The two lookups over [`REACTION_MT_MAP`], indexed once per process.
///
/// Both used to be a linear `find` over 86 string literals. That is nothing on
/// its own, but `activation_mts` resolves every reaction kind of every one of a
/// chain's ~3820 nuclides on every `Material::transmute` call, which came to
/// roughly 5e6 string comparisons -- 40-60 ms per call, before any nuclear data
/// was read (issue #576, finding 8).
///
/// Built with `or_insert`, so the MT lookup keeps the FIRST name listed for an
/// MT, which is what `find` returned and what the table's own doc comment
/// promises: that name is used as a rate key and has to match the chain's
/// spelling, so "fission" must win over the "(n,fission)" alias below it.
struct Index {
    by_name: HashMap<&'static str, i32>,
    by_mt: HashMap<i32, &'static str>,
}

fn index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut by_name = HashMap::with_capacity(REACTION_MT_MAP.len());
        let mut by_mt = HashMap::with_capacity(REACTION_MT_MAP.len());
        for &(name, mt) in REACTION_MT_MAP {
            by_name.entry(name).or_insert(mt);
            by_mt.entry(mt).or_insert(name);
        }
        Index { by_name, by_mt }
    })
}

/// Get the MT number for a chain reaction type name.
pub fn reaction_type_to_mt(reaction_type: &str) -> Option<i32> {
    index().by_name.get(reaction_type).copied()
}

/// Get the chain reaction type name for an MT number.
pub fn mt_to_reaction_type(mt: i32) -> Option<&'static str> {
    index().by_mt.get(&mt).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(n,2nd)` is MT 11 and `(n,nd2a)` is MT 35. These two were swapped in the
    /// D1S copy of this table, so D1S decay photon production looked up the
    /// wrong channel's cross section for every `(n,2nd)` chain reaction.
    #[test]
    fn two_neutron_deuteron_is_mt_11_not_35() {
        assert_eq!(reaction_type_to_mt("(n,2nd)"), Some(11));
        assert_eq!(reaction_type_to_mt("(n,nd2a)"), Some(35));
        assert_eq!(mt_to_reaction_type(11), Some("(n,2nd)"));
        assert_eq!(mt_to_reaction_type(35), Some("(n,nd2a)"));
    }

    /// No MT may be claimed by two different canonical names, and no name may
    /// appear twice. Aliases are allowed (several names may share one MT, as
    /// `fission` and `(n,fission)` do), but a repeated *name* would make
    /// `reaction_type_to_mt` depend on table order.
    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = REACTION_MT_MAP.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "a reaction name appears twice");
    }

    /// The first name listed for an MT is the canonical one. MT 18 is the only
    /// MT with an alias, and the chain files spell it `fission`, so a rate key
    /// of `(n,fission)` would match no chain reaction and silently remove
    /// fission from the network. `find` gave first-wins for free; the index
    /// behind these lookups has to be built to preserve it.
    #[test]
    fn the_first_name_listed_for_an_mt_wins() {
        assert_eq!(mt_to_reaction_type(18), Some("fission"));
        assert_eq!(reaction_type_to_mt("fission"), Some(18));
        assert_eq!(reaction_type_to_mt("(n,fission)"), Some(18));
    }

    /// Every entry must round trip through the canonical name for its MT, so
    /// that a rate key produced by `mt_to_reaction_type` always resolves back
    /// to the same MT. This is what makes the table safe to use in both
    /// directions.
    #[test]
    fn every_entry_round_trips() {
        for (name, mt) in REACTION_MT_MAP {
            let canonical = mt_to_reaction_type(*mt).expect("MT resolves to a name");
            assert_eq!(
                reaction_type_to_mt(canonical),
                Some(*mt),
                "{name} -> MT {mt} -> {canonical} did not return to MT {mt}"
            );
        }
    }
}
