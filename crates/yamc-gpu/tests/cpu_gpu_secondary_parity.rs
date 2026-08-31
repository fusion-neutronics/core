//! CPU-vs-GPU parity for secondary outgoing-energy samplers (issue #101).
//!
//! Pins the GPU `flat` samplers (`yamc_physics::gpu::flat::*`, used by both the
//! cubecl kernel and its CPU twin) against the PRODUCTION samplers
//! (`yamc_nuclide::reaction_product::EnergyDistribution`) statistically, so a
//! silently-divergent outgoing-energy spectrum is caught per-sub-step rather
//! than only at the integrated-flux level. The lack of exactly this comparison
//! is what let the multi-isotope epithermal residual (#88) hide: the existing
//! `gpu_*_matches_cpu` tests compare the kernel against the `flat` twin (which
//! is bit-identical to it), never against the production transport.
//!
//! These are STATISTICAL distribution comparisons (the two paths use different
//! RNGs and draw schedules), with a generous restriction energy so neither path
//! rejects, isolating the sampled spectrum itself.

use rand::{rngs::StdRng, SeedableRng};
use yamc_nuclide::reaction_product::{EnergyDistribution, Tabulated1D};
use yamc_physics::gpu::flat::evaporation::sample_evaporation;
use yamc_physics::gpu::flat::maxwell::sample_maxwell;
use yamc_physics::gpu::flat::watt::sample_watt_inelastic;
use yamc_rng::expand_seed;

/// A `Tabulated1D` that evaluates to a constant `val` across `[e_lo, e_hi]`
/// (lin-lin, single region), so the production `theta.evaluate(E)` and the
/// `flat` `linear_interp_on_grid` both return `val` and the interpolation
/// scheme does not confound the spectrum comparison.
fn const_tab(e_lo: f64, e_hi: f64, val: f64) -> Tabulated1D {
    Tabulated1D::Tabulated1D {
        x: vec![e_lo, e_hi],
        y: vec![val, val],
        breakpoints: vec![2],
        interpolation: vec![2], // lin-lin
    }
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Assert two large samples come from the same distribution: matching mean,
/// standard deviation, and coarse histogram. Tolerances are set well above the
/// Monte-Carlo noise floor at `n >= 4e5` but far below a real spectrum shift.
fn assert_same_distribution(prod: &[f64], flat: &[f64], hi: f64, label: &str) {
    let (mp, sp) = mean_std(prod);
    let (mf, sf) = mean_std(flat);
    let mean_rel = (mp - mf).abs() / mp.abs().max(1e-30);
    let std_rel = (sp - sf).abs() / sp.abs().max(1e-30);
    eprintln!(
        "{label}: prod mean={mp:.5e} std={sp:.5e} | flat mean={mf:.5e} std={sf:.5e} | mean_rel={mean_rel:.4} std_rel={std_rel:.4}"
    );
    assert!(
        mean_rel < 0.015,
        "{label}: mean differs by {mean_rel:.4} (prod {mp:.5e} vs flat {mf:.5e})"
    );
    assert!(
        std_rel < 0.03,
        "{label}: std differs by {std_rel:.4} (prod {sp:.5e} vs flat {sf:.5e})"
    );
    // Coarse histogram over [0, hi]: every bin holding >1% of the mass must
    // agree to <8% relative (about 5 sigma at n=4e5).
    let nb = 12usize;
    let (mut hp, mut hf) = (vec![0.0_f64; nb], vec![0.0_f64; nb]);
    for &x in prod {
        hp[(((x / hi) * nb as f64) as usize).min(nb - 1)] += 1.0;
    }
    for &x in flat {
        hf[(((x / hi) * nb as f64) as usize).min(nb - 1)] += 1.0;
    }
    let (np, nf) = (prod.len() as f64, flat.len() as f64);
    for b in 0..nb {
        let (pp, pf) = (hp[b] / np, hf[b] / nf);
        if pp > 0.01 {
            let rel = (pp - pf).abs() / pp;
            assert!(
                rel < 0.08,
                "{label}: histogram bin {b} differs by {rel:.3} (prod {pp:.4} vs flat {pf:.4})"
            );
        }
    }
}

const N: usize = 400_000;

#[test]
fn flat_maxwell_matches_production_spectrum() {
    let e_in = 14.0e6;
    let theta = 1.3e6;
    let u = 0.0; // generous: cap_e = e_in - u = 14 MeV >> theta, so ~no rejection
    let cap_e = e_in - u;

    let dist = EnergyDistribution::Maxwell {
        theta: const_tab(1.0, 2.0e7, theta),
        u,
    };
    let mut rng = StdRng::seed_from_u64(88_101);
    let prod: Vec<f64> = (0..N).map(|_| dist.sample(e_in, &mut rng)).collect();

    let grid = [1.0, 2.0e7];
    let theta_grid = [theta, theta];
    let mut state: u64 = expand_seed(0x5EED_0101);
    let mut flat = Vec::with_capacity(N);
    let mut exhausted = 0usize;
    while flat.len() < N {
        match sample_maxwell(e_in, &grid, &theta_grid, u, &mut state) {
            Some(e) => flat.push(e),
            None => exhausted += 1,
        }
    }
    assert!(
        exhausted < N / 100,
        "flat Maxwell exhausted its 32-iter cap {exhausted} times for a generous restriction energy"
    );
    assert_same_distribution(&prod, &flat, cap_e, "Maxwell");
}

#[test]
fn flat_watt_matches_production_spectrum() {
    let e_in = 14.0e6;
    let a = 0.988e6; // typical thermal-fission Watt parameters
    let b = 2.249e-6;
    let u = 0.0;
    let cap_e = e_in - u;

    let dist = EnergyDistribution::Watt {
        a: const_tab(1.0, 2.0e7, a),
        b: const_tab(1.0, 2.0e7, b),
        u,
    };
    let mut rng = StdRng::seed_from_u64(88_102);
    let prod: Vec<f64> = (0..N).map(|_| dist.sample(e_in, &mut rng)).collect();

    let grid = [1.0, 2.0e7];
    let a_grid = [a, a];
    let b_grid = [b, b];
    let mut state: u64 = expand_seed(0x5EED_0102);
    let mut flat = Vec::with_capacity(N);
    let mut exhausted = 0usize;
    while flat.len() < N {
        match sample_watt_inelastic(e_in, &grid, &a_grid, &b_grid, u, &mut state) {
            Some(e) => flat.push(e),
            None => exhausted += 1,
        }
    }
    assert!(
        exhausted < N / 100,
        "flat Watt exhausted its 32-iter cap {exhausted} times for a generous restriction energy"
    );
    assert_same_distribution(&prod, &flat, cap_e, "Watt");
}

#[test]
fn flat_evaporation_matches_production_spectrum() {
    let e_in = 14.0e6;
    let theta = 1.3e6;
    let u = 0.0;
    let cap_e = e_in - u;

    let dist = EnergyDistribution::Evaporation {
        theta: const_tab(1.0, 2.0e7, theta),
        u,
    };
    let mut rng = StdRng::seed_from_u64(88_103);
    let prod: Vec<f64> = (0..N).map(|_| dist.sample(e_in, &mut rng)).collect();

    let grid = [1.0, 2.0e7];
    let theta_grid = [theta, theta];
    let mut state: u64 = expand_seed(0x5EED_0103);
    let mut flat = Vec::with_capacity(N);
    let mut exhausted = 0usize;
    while flat.len() < N {
        match sample_evaporation(e_in, &grid, &theta_grid, u, &mut state) {
            Some(e) => flat.push(e),
            None => exhausted += 1,
        }
    }
    assert!(
        exhausted < N / 100,
        "flat Evaporation exhausted its 32-iter cap {exhausted} times for a generous restriction energy"
    );
    assert_same_distribution(&prod, &flat, cap_e, "Evaporation");
}
