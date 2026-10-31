//! The per-reaction deposits: their mean outgoing energies against the
//! transport's own samplers (what a reaction's products carry away on
//! average is what the energy balance subtracts), and their sum against the
//! stored rows they split. Self-skip without the fixtures.

use std::collections::BTreeSet;

use rand::SeedableRng;
use yamc_nuclide::deposit::{deposits, product_mean_energy, unexplained_share, Response};
use yamc_nuclide::nuclide::load_nuclide;
use yamc_nuclide::reaction_product::ParticleType;
use yamc_nuclide::LoadScope;

/// The sampled mean lab energy of `n` draws, and its standard error.
fn sampled(
    product: &yamc_nuclide::reaction_product::ReactionProduct,
    e: f64,
    awr: f64,
    cm: bool,
    n: usize,
) -> (f64, f64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let (mut s, mut s2) = (0.0, 0.0);
    for _ in 0..n {
        let (e_out, mu) = product.sample(e, &mut rng);
        let lab = if cm {
            e_out + (e + 2.0 * mu * (awr + 1.0) * (e * e_out).sqrt()) / ((awr + 1.0) * (awr + 1.0))
        } else {
            e_out
        };
        s += lab;
        s2 += lab * lab;
    }
    let mean = s / n as f64;
    (
        mean,
        ((s2 / n as f64 - mean * mean).max(0.0) / n as f64).sqrt(),
    )
}

#[test]
fn mean_energies_match_the_samplers() {
    let mut checked = 0;
    for name in ["Fe56", "W184", "Li6"] {
        let Some(dir) = yamc_test_cache::nuclide(name) else {
            eprintln!("skipping {name}: fixture not fetched");
            continue;
        };
        let scope = LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
        let nuclide = load_nuclide(&dir, &scope).expect("load");
        let awr = nuclide.atomic_weight_ratio.unwrap();
        let reactions = &nuclide.reactions[0];
        for (mt, e) in [
            (51, 2.5e6),
            (91, 8e6),
            (91, 14.1e6),
            (16, 14.1e6),
            (32, 14.1e6),
            (102, 1e3),
            (91, 5e6),
        ] {
            let Some(r) = reactions.get(&mt) else {
                continue;
            };
            for product in r.products.iter() {
                if product.distribution.is_empty() {
                    continue;
                }
                let cm = r.scatter_in_cm && product.particle == ParticleType::Neutron;
                let analytic = product_mean_energy(product, e, awr, cm);
                let (mc, se) = sampled(product, e, awr, cm, 400_000);
                eprintln!(
                    "{name} MT {mt} {:?} at {e:.3e} eV: analytic {analytic:.6e}, sampled {mc:.6e} +- {se:.1e}",
                    product.particle
                );
                assert!(
                    (analytic - mc).abs() < 5.0 * se + 1e-4 * mc.abs(),
                    "{name} MT {mt} {:?} at {e} eV: {analytic} against {mc} +- {se}",
                    product.particle
                );
                checked += 1;
            }
        }
    }
    eprintln!("checked {checked} products");
}

/// The deposits of every reaction add up to the stored heating, local
/// heating and damage-energy rows, over lethargy, to within what each
/// nuclide's data leaves out (`partial_sums_against_the_stored_rows` in
/// `deposit_validation.rs` prints the energy-resolved table): Li6 to 0.01%,
/// Fe56 to 0.2% (its MT 5 takes what the others leave), Be9 to 5% (its
/// (n,2n) breakup recoils are in MF=6 spectra the data does not carry).
#[test]
fn deposits_add_up_to_the_stored_rows() {
    for (name, bounds) in [
        ("Li6", [1e-4, 1e-4, 1e-4]),
        ("Fe56", [1e-3, 3e-3, 1e-3]),
        ("Be9", [0.03, 0.05, 0.02]),
    ] {
        let Some(dir) = yamc_test_cache::nuclide(name) else {
            eprintln!("skipping {name}: fixture not fetched");
            continue;
        };
        let scope = LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
        let nuclide = load_nuclide(&dir, &scope).expect("load");
        let partials: BTreeSet<i32> = nuclide.reactions[0]
            .iter()
            .filter(|(_, r)| !r.redundant)
            .map(|(mt, _)| *mt)
            .collect();
        for (response, bound) in [
            Response::Heating,
            Response::DamageEnergy,
            Response::HeatingLocal,
        ]
        .into_iter()
        .zip(bounds)
        {
            let split = deposits(&nuclide, 0, response, &partials);
            let share = unexplained_share(&nuclide, 0, response, &split, (0.0, f64::INFINITY))
                .expect("a stored row");
            eprintln!(
                "{name} MT {}: {:.4}% unexplained",
                response.mt(),
                100.0 * share
            );
            assert!(
                share < bound,
                "{name} MT {}: {share} of the row unexplained, more than {bound}",
                response.mt()
            );
        }
    }
}
