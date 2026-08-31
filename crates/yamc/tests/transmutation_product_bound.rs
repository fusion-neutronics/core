//! The folded product bound must never undercut the rate the tally scores.
//!
//! `simulate_transmutation` decides which products to carry by rating every
//! chain nuclide against a flux spectrum and dropping the ones a bound says
//! cannot reach the solver's density floor (issue #404). The rating is a
//! per-bin cross-section maximum folded against the tallied flux, standing in
//! for the continuous-energy `sum(sigma(E_i) * TL_i)` the tally accumulates for
//! the nuclides it does carry.
//!
//! Everything rests on that fold being an upper bound. If it came out low the
//! bound would drop a nuclide the solve then populates, and nothing downstream
//! could tell: the nuclide would simply never react. So this pins the two
//! against each other over real cross sections, using the offline Li6 fixture
//! rather than a synthetic curve, because the fold has to hold against real
//! structure and not just a straight line.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yani::{BranchCurve, BranchQuantity, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::TransmutationTallies;

/// Track segments spanning thermal to 14 MeV, so the fold is exercised across
/// the whole grid rather than in one corner of it.
const SEGMENTS: &[(f64, f64)] = &[
    (2.53e-2, 5.0),
    (1.0, 3.0),
    (1.0e3, 2.0),
    (2.5e5, 4.0),
    (1.0e6, 1.5),
    (2.0e6, 2.5),
    (1.406e7, 6.0),
];
const N_PARTICLES: usize = 4;
const VOLUME: f64 = 2.0;
const SOURCE_RATE: f64 = 1.0e14;

fn li6_material() -> Material {
    let mut material = Material::new(
        HashMap::from([("Li6".to_string(), 1.0)]),
        "atom",
        "g/cc",
        Some(0.5),
    )
    .unwrap();
    material.set_temperature("294");
    material.set_material_id(1);
    material.transmutable = true;
    material.volume = Some(VOLUME);
    material
        .read_nuclear_data(
            &HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]),
            None,
        )
        .unwrap();
    material
}

fn rx(kind: &str, target: &str) -> ChainReaction {
    ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching: 1.0,
        q_value: None,
    }
}

fn stable(name: &str) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        decay_energy: 0.0,
        reactions: vec![],
        decays: vec![],
        fission_yields: None,
        sources: Vec::new(),
        half_life_uncertainty: None,
        decay_energy_uncertainty: None,
    }
}

/// Li6 with the channels the fixture actually carries, spanning a smooth
/// 1/v capture, a threshold reaction and a light-ejectile channel.
fn li6_chain() -> HashMap<String, ChainNuclide> {
    let mut chain = HashMap::new();
    chain.insert(
        "Li6".to_string(),
        ChainNuclide {
            name: "Li6".to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions: vec![
                rx("(n,gamma)", "Li7"),
                rx("(n,t)", "He4"),
                rx("(n,p)", "He6"),
                rx("(n,d)", "He5"),
                rx("(n,2n)", "Li5"),
            ],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        },
    );
    for name in ["Li7", "He4", "He6", "He5", "Li5"] {
        chain.insert(name.to_string(), stable(name));
    }
    chain
}

/// Per-nuclide, per-reaction-kind rates [1/s], as both extractors return them.
type Rates = HashMap<String, HashMap<String, f64>>;

fn scored_and_bounded(branch: &BranchTable) -> (Rates, Rates) {
    let material = li6_material();
    let chain = li6_chain();
    let cells: HashMap<u32, Vec<usize>> = HashMap::from([(1u32, vec![0usize])]);
    let materials: HashMap<u32, &Material> = HashMap::from([(1u32, &material)]);
    let tallies = TransmutationTallies::new(&cells, &materials, &chain, branch, &HashMap::new());

    for &(energy, track_length) in SEGMENTS {
        tallies.score(1, energy, track_length, &material);
    }
    tallies.accumulate_batch(N_PARTICLES);

    (
        tallies.get_reaction_rates(1, VOLUME, SOURCE_RATE),
        tallies.bounding_reaction_rates(1, &material, &chain, VOLUME, SOURCE_RATE),
    )
}

/// Every rate the tally scored must be covered by the folded bound, and the
/// bound must stay tight enough to be worth folding: a bound loose by orders of
/// magnitude per edge would compound over a product chain and keep the whole
/// closure, which is the outcome the pruning exists to avoid.
#[test]
fn folded_bound_covers_every_scored_rate() {
    let (scored, bounded) = scored_and_bounded(&BranchTable::new());
    assert!(
        !scored.is_empty(),
        "the fixture must score something, or this test proves nothing"
    );

    for (nuclide, kinds) in &scored {
        for (kind, &rate) in kinds {
            let bound = bounded
                .get(nuclide)
                .and_then(|k| k.get(kind))
                .copied()
                .unwrap_or(0.0);
            assert!(
                bound >= rate * (1.0 - 1e-12),
                "{nuclide} {kind}: folded bound {bound:e} undercuts the scored rate {rate:e}"
            );
            assert!(
                bound <= rate * 1.0e3,
                "{nuclide} {kind}: folded bound {bound:e} is {:.0}x the scored rate {rate:e}",
                bound / rate
            );
        }
    }
}

/// `(n,n')` has no transport MT, so its rate reaches the burnup matrix only
/// through the branching overlay's MF=10 partials. The bound has to fold those
/// too, or a metastable state reachable only by inelastic scatter would be
/// rated at zero and pruned out of a run that goes on to produce it.
#[test]
fn folded_bound_covers_overlay_only_kinds() {
    let mut branch = BranchTable::new();
    branch.entry("Li6".to_string()).or_default().insert(
        "(n,n')".to_string(),
        vec![BranchCurve {
            target: "Li6_m1".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e5, 2.0e7],
            values: vec![0.0, 2.0],
        }],
    );
    let (_, bounded) = scored_and_bounded(&Arc::new(branch));
    let rate = bounded
        .get("Li6")
        .and_then(|k| k.get("(n,n')"))
        .copied()
        .unwrap_or(0.0);
    assert!(
        rate > 0.0,
        "an MF=10 (n,n') partial must give the bound a rate; got {rate:e}"
    );
}
