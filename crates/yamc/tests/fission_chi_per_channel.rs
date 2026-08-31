//! Issue #425: an evaluation with partial fission channels carries a DIFFERENT
//! prompt spectrum on each one, so the flat chi has to be cached per (nuclide,
//! MT) rather than per nuclide.
//!
//! U240 is the only nuclide in ENDF/B-VIII.1 that exercises this. A scan of all
//! 553 cached endf-b8.1 nuclides (88 of them fissionable) found exactly one with
//! non-redundant partial fission MTs: U240 carries MT 19, 20, 21 and 38, each
//! with its own neutron product, while its MT 18 is redundant and carries no
//! neutron product at all. Every other fissionable nuclide has MT 18 alone and
//! `has_partial_fission == false`, which short-circuits the channel sampling
//! entirely, so nothing else in the library can see any of this.
//!
//! What separates U240's four channels is not the Maxwell temperature (all four
//! sit at about 1.49 MeV) but the RESTRICTION ENERGY, which caps the outgoing
//! energy at `E_in - u` and is what makes them second, third and fourth chance
//! fission:
//!
//! | MT | u | open above | E_out cap at 14.06 MeV |
//! |----|----|----|----|
//! | 19 | -30.0 MeV | always | unrestricted |
//! | 20 | +1.394 MeV | 1.39 MeV | 12.67 MeV |
//! | 21 | +9.958 MeV | 9.96 MeV | 4.10 MeV |
//! | 38 | +14.34 MeV | 14.34 MeV | closed |
//!
//! The cache this replaced held ONE slot per nuclide, so the run's whole fission
//! spectrum was whichever channel the first fission event happened to sample,
//! frozen for every fission after it. When that winner was a high-threshold
//! channel, every later fission at a lower energy drew from a chi that cannot
//! produce anything there: `sample_fission_chi_flat` returned `None` 64 times
//! and the caller fell back to emitting the neutron at its incident energy. The
//! result depended on which thread reached the first fission.

use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::energy::FissionChiFlatCache;
use yamc_nuclide::reaction_product::{
    AngleEnergyDistribution, EnergyDistribution, FissionChiFlat, ParticleType, ReactionProduct,
};
use yamc_nuclide::LoadScope;

const DATA: &str = "tests/U240.arrow";
const CHANNELS: [i32; 4] = [19, 20, 21, 38];
const E_14MEV: f64 = 14.06e6;

fn load() -> Option<Nuclide> {
    if !std::path::Path::new(DATA).exists() {
        eprintln!("skipping: {DATA} absent (run scripts/fetch_test_fixtures.py)");
        return None;
    }
    Some(read_nuclide_from_arrow(std::path::Path::new(DATA), &LoadScope::full()).expect("U240"))
}

fn first_neutron_product(rxn: &Reaction) -> Option<&ReactionProduct> {
    rxn.products
        .iter()
        .find(|p| p.is_particle_type(&ParticleType::Neutron))
}

/// The prompt chi lives on the first neutron product, matching what
/// `sample_fission_event` hands the cache.
fn prompt_dist(rxn: &Reaction) -> Option<&EnergyDistribution> {
    first_neutron_product(rxn).and_then(|p| {
        p.distribution.first().and_then(|d| match d {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => energy.as_ref(),
            _ => None,
        })
    })
}

/// Draw `n` outgoing energies at `e_in` on a fixed stream, so channels compare
/// like for like. `None` when the channel is closed at that energy, which is a
/// real answer rather than a failure: MT 38 is shut below 14.34 MeV.
fn mean_eout(chi: &FissionChiFlat, e_in: f64, n: usize) -> Option<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let (mut sum, mut got) = (0.0, 0usize);
    for _ in 0..n {
        if let Some(e) = yamc_physics::gpu::flat::sample_fission_chi_flat(chi, e_in, &mut state) {
            sum += e;
            got += 1;
        }
    }
    (got * 2 > n).then(|| sum / got as f64)
}

/// The data shape that makes the per-MT key necessary. If a library revision
/// changes this, the tests below stop meaning what they say.
#[test]
fn u240_is_the_partial_fission_case() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let grid = n.fast_xs.get(ti).expect("fast_xs");

    assert!(grid.has_partial_fission, "U240 carries partial fission");
    assert_eq!(
        grid.fission_mt_numbers,
        CHANNELS.to_vec(),
        "the non-redundant fission channels, MT 18 excluded as redundant"
    );

    let mt18 = n.reactions[ti].get(&18).expect("MT 18 present");
    assert!(mt18.redundant, "MT 18 is the redundant total");
    assert!(
        first_neutron_product(mt18).is_none(),
        "U240's MT 18 has no neutron product, so it cannot supply a chi"
    );
}

/// Each channel opens at its own threshold, which is the property a single
/// shared slot destroys.
#[test]
fn each_channel_opens_at_its_own_threshold() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let cache = FissionChiFlatCache::default();
    let chi = |mt: i32| {
        let rxn = n.reactions[ti].get(&mt).expect("channel");
        cache.get_or_build(mt, prompt_dist(rxn))
    };

    // At 1 MeV only first-chance fission is open.
    assert!(
        mean_eout(chi(19), 1.0e6, 2000).is_some(),
        "MT 19 is open at 1 MeV"
    );
    for mt in [20, 21, 38] {
        assert!(
            mean_eout(chi(mt), 1.0e6, 2000).is_none(),
            "MT {mt} must be shut at 1 MeV"
        );
    }

    // At 14.06 MeV the first three are open and fourth-chance is not.
    for mt in [19, 20, 21] {
        assert!(
            mean_eout(chi(mt), E_14MEV, 2000).is_some(),
            "MT {mt} is open at 14.06 MeV"
        );
    }
    assert!(
        mean_eout(chi(38), E_14MEV, 2000).is_none(),
        "MT 38 opens at 14.34 MeV, above 14.06"
    );
}

/// The open channels' spectra differ materially, because the restriction energy
/// truncates them at different places.
#[test]
fn open_channels_have_different_spectra() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let cache = FissionChiFlatCache::default();

    let mut means = Vec::new();
    for mt in [19, 20, 21] {
        let rxn = n.reactions[ti].get(&mt).expect("channel");
        let m = mean_eout(cache.get_or_build(mt, prompt_dist(rxn)), E_14MEV, 20_000)
            .expect("open at 14.06 MeV");
        println!("U240 MT{mt} mean E_out at 14.06 MeV = {m:.4e} eV");
        means.push(m);
    }

    let lo = means.iter().cloned().fold(f64::MAX, f64::min);
    let hi = means.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        (hi - lo) / lo > 0.05,
        "channels should differ materially, got {:?} (spread {:.2}%)",
        means,
        (hi - lo) / lo * 100.0
    );
}

/// The bug itself. Touch a high-threshold channel first, then ask for
/// first-chance fission at an energy only it can serve. The single-slot cache
/// handed back MT 21's chi, which produces nothing at 1 MeV, and the fission
/// neutron came out at its incident energy instead.
#[test]
fn a_high_threshold_channel_does_not_poison_a_low_energy_fission() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let cache = FissionChiFlatCache::default();
    let dist = |mt: i32| prompt_dist(n.reactions[ti].get(&mt).expect("channel"));

    // First fission of the "run" lands on third-chance fission at 14 MeV.
    assert!(mean_eout(cache.get_or_build(21, dist(21)), E_14MEV, 2000).is_some());

    // A later 1 MeV fission must still get MT 19's spectrum. With the shared
    // slot this returned `None` every time, and the caller emitted the neutron
    // at exactly its incident energy.
    let m = mean_eout(cache.get_or_build(19, dist(19)), 1.0e6, 20_000)
        .expect("MT 19 must still be sampleable at 1 MeV after MT 21 was cached");

    // MT 19 is effectively unrestricted (u = -30 MeV), so this is a plain
    // Maxwell whose mean is 1.5*theta, and theta interpolates to ~1.31 MeV at
    // 1 MeV incident. Fission neutrons leaving above the incident energy is
    // normal: the energy comes from the fragments, not the incident neutron.
    assert!(
        (1.8e6..2.1e6).contains(&m),
        "expected the 1.5*theta Maxwell mean near 1.96 MeV, got {m:.4e}"
    );
}

/// Cache population order must not change any channel's answer.
#[test]
fn channel_chi_is_order_independent() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let dist = |mt: i32| prompt_dist(n.reactions[ti].get(&mt).expect("channel"));

    let forward = FissionChiFlatCache::default();
    let a: Vec<_> = CHANNELS
        .iter()
        .map(|&mt| mean_eout(forward.get_or_build(mt, dist(mt)), E_14MEV, 20_000))
        .collect();

    let reverse = FissionChiFlatCache::default();
    let mut b = vec![None; CHANNELS.len()];
    for (i, &mt) in CHANNELS.iter().enumerate().rev() {
        b[i] = mean_eout(reverse.get_or_build(mt, dist(mt)), E_14MEV, 20_000);
    }

    assert_eq!(
        a, b,
        "a channel's chi must not depend on which channel was sampled first"
    );
}
