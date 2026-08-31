//! CPU-vs-GPU parity for the elastic CM-cosine sampler (issues #111, #101, #40).
//!
//! After #111 step 1 the production CPU elastic scatter samples the CM cosine
//! through `yamc_physics::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm` -- the
//! SAME function the cubecl kernel and its CPU twin use -- fed by the flat
//! buffers from `AngleDistribution::to_elastic_flat()`. This pins that flat
//! sampler against the production `AngleDistribution::sample` over a realistic
//! tabulated angular distribution, so the CPU and GPU elastic angle sampling
//! cannot drift (the #88 divergence class). It is the per-event (level-1) form
//! of the #40 matched-stream diff: identical tabulated data, sampled by both
//! paths, compared statistically (the two use different RNGs / draw schedules).

use rand::{rngs::StdRng, SeedableRng};
use yamc_nuclide::reaction_product::{AngleDistribution, Tabulated, TabulatedInterp};
use yamc_physics::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm;
use yamc_rng::{expand_seed, next_xi};

const N: usize = 400_000;

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Assert two large `mu in [-1, 1]` samples come from the same distribution:
/// matching mean (absolute, since the mean can sit near 0), standard deviation,
/// and a 12-bin histogram. Tolerances are well above the Monte-Carlo noise floor
/// at `n >= 4e5` (SEM on the mean ~ 0.5/sqrt(n) ~ 8e-4) but far below a real
/// distribution shift.
fn assert_same_mu_distribution(prod: &[f64], flat: &[f64], label: &str) {
    let (mp, sp) = mean_std(prod);
    let (mf, sf) = mean_std(flat);
    let mean_abs = (mp - mf).abs();
    let std_rel = (sp - sf).abs() / sp.abs().max(1e-30);
    eprintln!("{label}: prod mean={mp:.5} std={sp:.5} | flat mean={mf:.5} std={sf:.5} | dmean={mean_abs:.4} dstd_rel={std_rel:.4}");
    assert!(
        mean_abs < 0.01,
        "{label}: mu mean differs by {mean_abs:.4} (prod {mp:.5} vs flat {mf:.5})"
    );
    assert!(
        std_rel < 0.03,
        "{label}: mu std differs by {std_rel:.4} (prod {sp:.5} vs flat {sf:.5})"
    );
    let nb = 12usize;
    let (mut hp, mut hf) = (vec![0.0_f64; nb], vec![0.0_f64; nb]);
    let bin = |x: f64| (((x + 1.0) / 2.0 * nb as f64) as usize).min(nb - 1);
    for &x in prod {
        hp[bin(x)] += 1.0;
    }
    for &x in flat {
        hf[bin(x)] += 1.0;
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

/// Build a normalized `Tabulated` mu-distribution from `(x, pdf)`: trapezoidal
/// CDF then `normalize()` to integral 1.0, exactly as the Arrow loader does, so
/// the production `Tabulated::sample` and `to_elastic_flat` (which renormalizes)
/// see the same data.
fn tab(x: Vec<f64>, p: Vec<f64>, interp: TabulatedInterp) -> Tabulated {
    let mut c = vec![0.0; x.len()];
    for i in 1..x.len() {
        c[i] = c[i - 1] + 0.5 * (p[i] + p[i - 1]) * (x[i] - x[i - 1]);
    }
    let mut t = Tabulated { x, p, c, interp };
    t.normalize();
    t
}

#[test]
fn flat_elastic_mu_cm_matches_production_angle_sample() {
    // Two incident energies with distinct anisotropy: isotropic at low E and
    // forward-peaked (rising toward mu = +1) at high E, the typical elastic
    // trend. Sampling at an intermediate energy exercises the stochastic
    // energy-bracket interpolation both paths perform.
    let iso = tab(vec![-1.0, 1.0], vec![0.5, 0.5], TabulatedInterp::LinLin);
    let fwd = tab(
        vec![-1.0, 0.0, 0.5, 1.0],
        vec![0.1, 0.3, 0.8, 1.6],
        TabulatedInterp::LinLin,
    );
    let angle = AngleDistribution {
        energy: vec![1.0e3, 1.0e7],
        mu: vec![iso, fwd],
    };
    let flat = angle.to_elastic_flat();

    // Sanity: full-resolution flatten (no subsampling) preserves the grid.
    assert_eq!(flat.energy_grid, vec![1.0e3, 1.0e7]);
    assert_eq!(flat.n_mu, vec![2, 4]);
    assert_eq!(flat.mu_offset, vec![0, 2]); // tight CSR: row1 starts after row0's 2 points

    // Compare at the low knot, an interpolated energy, and above the top knot.
    for (tag, e_in, seed) in [
        ("low_knot", 1.0e3_f64, 0x5EED_0E01u32),
        ("interp", 1.0e6, 0x5EED_0E02),
        ("above", 2.0e7, 0x5EED_0E03),
    ] {
        let mut rng = StdRng::seed_from_u64(0xA17E_0000 ^ seed as u64);
        let prod: Vec<f64> = (0..N).map(|_| angle.sample(e_in, &mut rng)).collect();
        let mut state: u64 = expand_seed(seed);
        let flat_mu: Vec<f64> = (0..N)
            .map(|_| {
                // xi3 fallback (unused while tabulated data is present); drawing
                // it keeps the schedule in step with the kernel/twin.
                let xi3 = next_xi(&mut state);
                sample_elastic_mu_cm(
                    e_in,
                    xi3,
                    &flat.energy_grid,
                    &flat.n_mu,
                    &flat.interp,
                    &flat.mu,
                    &flat.cdf,
                    &flat.pdf,
                    &flat.mu_offset,
                    &mut state,
                )
            })
            .collect();
        assert_same_mu_distribution(&prod, &flat_mu, &format!("elastic_mu[{tag}]"));
    }
}
