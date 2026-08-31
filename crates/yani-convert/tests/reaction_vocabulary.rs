//! The writer's reaction names must all resolve on the reader's side (#379).
//!
//! This crate writes `reactions/reactions.arrow` with the reaction name taken
//! verbatim from `endf::chain` (`ReactionPath::kind`), and `yani` resolves that
//! name to an MT through `REACTION_MT_MAP` to find the cross section to fold. A
//! name the map does not hold resolves to no MT, so the rate is never computed
//! and the reaction vanishes from the network. Nothing errors.
//!
//! That is #379, and it reached published data: the chain files spell MT 18
//! "fission", the consumer's map held only "(n,fission)", so no fission product
//! was ever produced by transmutation and every test on both sides passed.
//! Neither side could see the other's vocabulary.
//!
//! The check used to live in the Python converter's test suite, where it had to
//! parse `REACTION_MT_MAP` out of the Rust source with a regex (the resolver is
//! not exposed to Python) and compare it against the endf-python fork's
//! `REACTIONS` table, which meant it only ran where the fork was installed. Now
//! that both sides are Rust it calls the real resolver against the real table,
//! so a rename on either side is a test failure here rather than a silent
//! vocabulary drift.

use endf::REACTIONS;

/// Every reaction name this crate can write into `reactions/reactions.arrow`.
///
/// `endf::chain` emits `ReactionInfo::name` for a neutron-induced path, plus
/// the bare `"fission"` for a fissionable nuclide (see the `FISSION_MTS` arm of
/// `Chain::from_endf`). Those two sources are the whole vocabulary.
fn writer_vocabulary() -> Vec<&'static str> {
    // Asserted rather than assumed: if `REACTIONS` ever gains a "fission" entry
    // of its own, this list would quietly grow a duplicate and the reason for
    // the special case would be lost.
    assert!(
        !REACTIONS.iter().any(|r| r.name == "fission"),
        "REACTIONS now carries \"fission\" itself, so the special case below is stale"
    );
    REACTIONS
        .iter()
        .map(|r| r.name)
        .chain(std::iter::once("fission"))
        .collect()
}

/// The assertion whose absence was #379.
#[test]
fn every_name_the_writer_can_emit_resolves_on_the_reader_side() {
    let unresolvable: Vec<&str> = writer_vocabulary()
        .into_iter()
        .filter(|name| yani::reaction_type_to_mt(name).is_none())
        .collect();

    assert!(
        unresolvable.is_empty(),
        "the transmutation writer can emit reaction names that yani's \
         REACTION_MT_MAP does not resolve: {unresolvable:?}. These fail \
         silently: the name maps to no MT, so the rate is never computed and \
         the reaction vanishes from the network (#379). Add them to \
         REACTION_MT_MAP in crates/yani/src/reactions.rs."
    );
}

/// The other half of the failure mode: a shared name meaning different things.
///
/// Weaker than the closure check above, and it catches what that one cannot: a
/// name both sides hold, pointing at different reactions. The rate would then
/// be computed from the wrong channel's cross section, which is worse than a
/// reaction that never fires because the answer looks plausible.
#[test]
fn a_shared_name_means_the_same_mt_on_both_sides() {
    let mut disagreements = Vec::new();
    for info in REACTIONS.iter() {
        if let Some(mt) = yani::reaction_type_to_mt(info.name) {
            if !info.mts.contains(&mt) {
                disagreements.push((info.name, mt, info.mts));
            }
        }
    }

    assert!(
        disagreements.is_empty(),
        "a reaction name means a different MT on each side \
         (name, yani MT, endf MTs): {disagreements:?}"
    );
}

/// `"fission"` is the spelling that matters, and it must mean MT 18.
///
/// Pinned separately because it is the exact pair #379 got wrong, and because
/// the alias `"(n,fission)"` sits next to it in the map: whichever is listed
/// first is what `mt_to_reaction_type` hands back as a rate key, and a key of
/// `"(n,fission)"` matches no chain reaction.
#[test]
fn fission_resolves_to_mt_18_under_the_name_the_writer_uses() {
    assert_eq!(yani::reaction_type_to_mt("fission"), Some(18));
    assert_eq!(yani::mt_to_reaction_type(18), Some("fission"));
}
