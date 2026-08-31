//! Issue #371: the flattened correlated angle-energy sampler must reproduce the
//! reference `CorrelatedAngleEnergy::sample` it was derived from.
//!
//! The CPU transport samples continuum inelastic through the GPU-shared FLAT
//! path (`InelasticFlatCache` -> `sample_correlated_angle_energy`), not through
//! `CorrelatedAngleEnergy::sample`. The two drifted: the flat path took the
//! angular sub-table at the LOWER bracketing E_out point instead of the nearer
//! one, so μ was wrong wherever adjacent sub-tables disagree -- which is exactly
//! at a spectrum's falling edge, since the lab energy carries a `±2√(E_in·E_cm)
//! / (A+1)` μ term.
//!
//! On W184 that put a -6.9% hole in the 12.523-12.840 MeV flux of the V&V
//! sphere and doubled the population of the bin above it, the only |z| > 5 bin
//! in the whole 175-group spectrum. Pre-fix this test measures 0.954 and 1.996
//! against the reference.
//!
//! Data: endf-b8.1 Arrow tables in the fixture cache. Self-skips when absent.

use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::reaction_product::AngleEnergyDistribution;
use yamc_nuclide::LoadScope;
use yamc_physics::gpu::flat::cm_to_lab::cm_to_lab;
use yamc_physics::gpu::flat::inelastic_flat::InelasticFlatCache;

const NUCLIDE: &str = "W184";
const MT: i32 = 91;
const E_IN: f64 = 14.06e6;
const N: usize = 4_000_000;
/// VITAMIN-J-175 edges spanning the region the bug moved flux across.
const EDGES: [f64; 5] = [11.618e6, 12.214e6, 12.523e6, 12.840e6, 13.499e6];

fn data_path() -> String {
    yamc_test_cache::nuclide_path(NUCLIDE)
}

fn bin_of(e: f64) -> Option<usize> {
    (0..EDGES.len() - 1).find(|&k| e >= EDGES[k] && e < EDGES[k + 1])
}

#[test]
fn flat_correlated_sampler_matches_the_reference_implementation() {
    let path = data_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} absent");
        return;
    }
    // The directory existing is not the same as this test's data being there. A
    // cache entry is a directory of sections and a run that needed only some of
    // them leaves the rest unfetched, which is what W184 looks like on a machine
    // that has never needed its distributions. The loader already rejects a
    // half-populated directory with a precise message naming the missing
    // section, so defer to it and skip, rather than unwrapping: absent data is
    // not a sampler regression and the two should not fail alike.
    //
    // Deliberately catching every read error, not just that one. Enumerating the
    // sections here was tried and got it wrong (MT 91's law needs fast_xs.arrow
    // too, not just products and distributions), and a list that has to track
    // the loader's requirements will drift out of step with them again. Other
    // tests cover the loader itself; this one is about the sampler.
    let nuclide = match read_nuclide_from_arrow(std::path::Path::new(&path), &LoadScope::full()) {
        Ok(nuclide) => nuclide,
        Err(e) => {
            eprintln!("skipping: {path} does not load in full, refetch {NUCLIDE}. {e}");
            return;
        }
    };
    let ti = nuclide.get_temp_idx("294").expect("294 K");
    let reaction = nuclide.reactions[ti].get(&MT).expect("MT 91");
    let awr = nuclide.atomic_weight_ratio.expect("awr");

    // Reference: the standalone CorrelatedAngleEnergy sampler, which is a
    // direct port of OpenMC's `CorrelatedAngleEnergy::sample`.
    let correlated = reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))
        .and_then(|p| {
            p.distribution.iter().find_map(|d| match d {
                AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } => Some(correlated),
                _ => None,
            })
        })
        .expect("MT 91 is a correlated angle-energy law");

    let mut ref_hist = [0usize; EDGES.len() - 1];
    let mut rng = yamc::util::fast_rng::FastRng::new(0x5EED_0371);
    for _ in 0..N {
        let (e_cm, mu_cm) = correlated.sample(E_IN, &mut rng);
        if let Some((_, e_lab)) = cm_to_lab(E_IN, e_cm, mu_cm, awr) {
            if let Some(k) = bin_of(e_lab) {
                ref_hist[k] += 1;
            }
        }
    }

    // Production: the flattened path the transport actually calls.
    let cache = InelasticFlatCache::default();
    let flat = cache.get_or_build(&nuclide, reaction);
    assert!(flat.has_outgoing_energy_law(), "MT 91 carries an E_out law");
    let abs_q = flat.q_value.abs();
    let threshold = (awr + 1.0) / awr * abs_q;
    let mass_ratio = (awr / (awr + 1.0)).powi(2);
    let e_cm_closed = mass_ratio * (E_IN - threshold);

    let mut flat_hist = [0usize; EDGES.len() - 1];
    let mut pcg: u64 = 0x853c_49e6_748f_ea9b;
    for _ in 0..N {
        let xi3 = yamc_rng::next_xi(&mut pcg);
        let (_mu_lab, e_lab, ok) = flat.sample_kinematics(E_IN, awr, e_cm_closed, xi3, &mut pcg);
        if ok {
            if let Some(k) = bin_of(e_lab) {
                flat_hist[k] += 1;
            }
        }
    }

    for k in 0..EDGES.len() - 1 {
        let (a, b) = (flat_hist[k] as f64, ref_hist[k] as f64);
        assert!(b > 1000.0, "bin {k} needs statistics, got {b}");
        let ratio = a / b;
        // Independent streams, so allow Monte-Carlo scatter: 5 sigma on the
        // smaller count is well under the 4.6% / 100% the bug produced.
        let sigma = (1.0 / a + 1.0 / b).sqrt();
        assert!(
            (ratio - 1.0).abs() < (5.0 * sigma).max(0.02),
            "bin {:.3}-{:.3} MeV: flat/reference = {ratio:.4} (flat {a}, reference {b}); \
             the flat path must sample the same distribution as the reference",
            EDGES[k] / 1e6,
            EDGES[k + 1] / 1e6,
        );
    }
}
