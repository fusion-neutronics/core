//! Issue #370: the free-gas elastic kernel against OpenMC's, via the equilibrium
//! of a collision chain.
//!
//! Repeatedly scattering one neutron off a free gas drives a Markov chain whose
//! fixed point is a property of the scattering kernel alone. Two kernels that sample
//! the same target-velocity distribution must share that fixed point, so this pins
//! yamc's equilibrium against OpenMC's.
//!
//! This exists because the V&V shows yamc 0.405% above OpenMC (z = +21) in the
//! 0.1 to 0.414 eV bin for H1, which is 3.9 to 16.3 kT: the shoulder between the
//! Maxwellian peak and the 1/E region, and exactly where an error in the
//! target-velocity sampling would surface. The rejection algorithm and the entry
//! condition were both verified against OpenMC's `sample_cxs_target_velocity` by
//! inspection, so this checks the kernel's OUTPUT rather than its code.
//!
//! The answer is that the kernel is FINE: yamc's chain settles at 1.7476 kT and
//! OpenMC's own sampler driven through the identical chain settles at 1.7502 kT, so
//! the H1 excess lies somewhere other than the target-velocity sampling.

use yamc_physics::gpu::flat::free_gas_elastic::sample_free_gas_elastic;

const K_B: f64 = 8.617333e-5;
const T: f64 = 294.0;
const KT: f64 = K_B * T;
/// H1: the case the V&V flags, and the one where target motion matters most.
const AWR_H1: f64 = 0.9991673;
/// Matches the transport default, in units of kT.
const FREE_GAS_THRESHOLD: f64 = 400.0;

/// Walk one neutron through many free-gas collisions and return the sampled
/// energies after discarding a burn-in.
fn equilibrium_energies(n_collisions: usize, burn_in: usize, seed: u32) -> Vec<f64> {
    equilibrium_energies_at(T, n_collisions, burn_in, seed)
}

/// The same chain at an arbitrary temperature, for the scale-invariance check.
fn equilibrium_energies_at(
    temperature_k: f64,
    n_collisions: usize,
    burn_in: usize,
    seed: u32,
) -> Vec<f64> {
    let kt = K_B * temperature_k;
    let mut state = yamc_rng::expand_seed(seed);
    let mut e = kt; // start near equilibrium so burn-in is short
    let (mut dx, mut dy, mut dz) = (0.0_f64, 0.0_f64, 1.0_f64);
    let mut out = Vec::with_capacity(n_collisions.saturating_sub(burn_in));
    for i in 0..n_collisions {
        // Isotropic in CM, which is what H1 elastic is at these energies.
        let mu_cm = 2.0 * yamc_rng::next_xi(&mut state) - 1.0;
        let (ndx, ndy, ndz, e_out, did_run) = sample_free_gas_elastic(
            e,
            dx,
            dy,
            dz,
            AWR_H1,
            temperature_k,
            FREE_GAS_THRESHOLD,
            mu_cm,
            &mut state,
        );
        assert!(
            did_run,
            "free-gas path must run for H1 at {e:.4e} eV (awr <= 1 has no threshold)"
        );
        assert!(
            e_out > 0.0 && e_out.is_finite(),
            "collision {i} produced a non-physical outgoing energy {e_out}"
        );
        e = e_out;
        dx = ndx;
        dy = ndy;
        dz = ndz;
        if i >= burn_in {
            out.push(e);
        }
    }
    out
}

/// Equilibrium the chain settles at, in kT, measured by driving OpenMC's
/// `sample_cxs_target_velocity` (physics.cpp) through this EXACT chain: same
/// isotropic-in-CM mu, same CM transform, 300k collisions.
///
/// Deliberately NOT 1.5 kT or 2 kT. A collision chain with no cross-section
/// weighting samples neither the Maxwellian density (1.5 kT) nor the flux (2 kT);
/// the fixed point of this particular chain is 1.750 kT, and asserting either
/// textbook value here would be wrong. The number's authority is that OpenMC's
/// kernel produces it, making this a cross-code parity bound rather than an
/// analytic identity.
const OPENMC_EQUILIBRIUM_KT: f64 = 1.7502;

/// The free-gas kernel must land on the same equilibrium as OpenMC's.
///
/// This is the test that exonerated the kernel for #370: yamc settles at 1.7476 kT
/// against OpenMC's 1.7502 kT, a 0.15% agreement, so the 0.4% H1 excess the V&V
/// reports in the 0.1 to 0.414 eV bin does NOT come from the target-velocity
/// sampling.
///
/// Measured sensitivity: making the sampled target 3% hotter moves the equilibrium
/// to 1.8005 kT and trips this. It does NOT catch a constant rescaling of the
/// rejection acceptance, which barely changes the accepted distribution's shape --
/// so this bounds the target-velocity DISTRIBUTION, not every possible edit to the
/// sampler.
#[test]
fn equilibrium_matches_openmcs_kernel() {
    let e = equilibrium_energies(400_000, 20_000, 0x01CE_5EED_u32);
    let mean_kt = (e.iter().sum::<f64>() / e.len() as f64) / KT;
    // Successive collisions are correlated, so the effective sample size is well
    // below the count; 1% sits above the chain-to-chain scatter and still an order
    // of magnitude below any bias worth caring about.
    assert!(
        (mean_kt - OPENMC_EQUILIBRIUM_KT).abs() < 0.01 * OPENMC_EQUILIBRIUM_KT,
        "free-gas equilibrium is {mean_kt:.4} kT against OpenMC's \
         {OPENMC_EQUILIBRIUM_KT:.4} kT. The two kernels sample the same target \
         velocity distribution, so they must share a fixed point (#370)"
    );
}

/// Shape, not just the mean: the equilibrium spectrum must be smooth and unimodal,
/// with no spike or hole of the kind a broken rejection branch would leave. A
/// mean-only check can hide probability moved from the peak into the tail.
#[test]
fn equilibrium_spectrum_is_smooth_and_unimodal() {
    let energies = equilibrium_energies(1_000_000, 50_000, 0xBA5E_BA11_u32);
    let edges: Vec<f64> = (0..=24).map(|i| 0.05 * KT * 1.35_f64.powi(i)).collect();
    let mut counts = vec![0usize; edges.len() - 1];
    for &e in &energies {
        if e >= edges[0] && e < edges[edges.len() - 1] {
            counts[edges.partition_point(|&x| x <= e) - 1] += 1;
        }
    }
    // Per-unit-energy density, so the geometric binning cannot fake a shape.
    let dens: Vec<f64> = counts
        .iter()
        .enumerate()
        .map(|(k, &c)| c as f64 / (edges[k + 1] - edges[k]))
        .collect();
    let peak = dens
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite densities"))
        .map(|(k, _)| k)
        .expect("non-empty");
    assert!(
        peak > 0 && peak + 1 < dens.len(),
        "the equilibrium peak landed on bin {peak} of {}, so the binning does not \
         bracket the distribution",
        dens.len()
    );
    for k in 1..dens.len() {
        if counts[k] < 500 || counts[k - 1] < 500 {
            continue;
        }
        let (prev, cur) = (dens[k - 1], dens[k]);
        let tol = 1.0 + 6.0 / (counts[k] as f64).sqrt();
        if k <= peak {
            assert!(
                cur * tol >= prev,
                "equilibrium spectrum dips at bin {k} ([{:.3e}, {:.3e}] eV) below the \
                 peak: density {cur:.4e} against {prev:.4e}. A hole here means a \
                 rejection branch is dropping part of the target distribution (#370)",
                edges[k],
                edges[k + 1]
            );
        } else {
            assert!(
                prev * tol >= cur,
                "equilibrium spectrum rises at bin {k} ([{:.3e}, {:.3e}] eV) above the \
                 peak: density {cur:.4e} against {prev:.4e} (#370)",
                edges[k],
                edges[k + 1]
            );
        }
    }
}

/// Above the threshold a target with `awr > 1` is treated as stationary, but H1
/// (`awr < 1`) must keep sampling target motion at every energy -- the condition
/// OpenMC writes as `E >= free_gas_threshold * kT && awr > 1.0`. If this regressed,
/// H's up-scattering would vanish above 400 kT.
#[test]
fn hydrogen_never_falls_back_to_a_stationary_target() {
    let mut state = yamc_rng::expand_seed(7);
    for &e in &[1.0e-3_f64, 1.0, 10.0, 1.0e3, 1.0e5, 1.0e6] {
        let (_, _, _, _, did_run) = sample_free_gas_elastic(
            e,
            0.0,
            0.0,
            1.0,
            AWR_H1,
            T,
            FREE_GAS_THRESHOLD,
            0.3,
            &mut state,
        );
        assert!(
            did_run,
            "H1 at {e:.3e} eV took the stationary-target path; awr = {AWR_H1} is below 1 \
             so the threshold must not apply (#370)"
        );
    }
    // A heavy target above the threshold must take the stationary path.
    let (_, _, _, _, heavy) = sample_free_gas_elastic(
        1.0e4,
        0.0,
        0.0,
        1.0,
        183.9,
        T,
        FREE_GAS_THRESHOLD,
        0.3,
        &mut state,
    );
    assert!(
        !heavy,
        "a heavy target well above {FREE_GAS_THRESHOLD} kT must be treated as stationary"
    );
}

/// The kernel depends on `E/kT` and nothing else, so its equilibrium expressed in
/// units of kT cannot depend on the temperature. For H1 the threshold never
/// applies (`awr < 1` always samples target motion), so the invariance is exact:
/// scale `E` and `kT` by the same factor and every sampled outgoing energy scales
/// with them.
///
/// Unlike `OPENMC_EQUILIBRIUM_KT`, which is a cross-code bound, this one IS an
/// identity, which makes it the right shape of test for issue #478: a material at
/// 900 K whose kernel is handed 294 K settles at 1.75 kT(294) rather than
/// 1.75 kT(900), a factor of 3.1 too cold, and no amount of internal consistency
/// would reveal it.
///
/// What this does NOT bound: which temperature `Material` hands to the kernel.
/// That is the plumbing #478 broke, and it is pinned by
/// `temperature_k_tracks_the_label` over in yamc-materials.
#[test]
fn equilibrium_in_kt_does_not_depend_on_temperature() {
    let reference = {
        let e = equilibrium_energies_at(294.0, 400_000, 20_000, 0x0478_0478_u32);
        (e.iter().sum::<f64>() / e.len() as f64) / (K_B * 294.0)
    };

    for &temperature_k in &[77.0_f64, 600.0, 900.0, 2500.0] {
        let e = equilibrium_energies_at(temperature_k, 400_000, 20_000, 0x0478_0478_u32);
        let mean_kt = (e.iter().sum::<f64>() / e.len() as f64) / (K_B * temperature_k);
        assert!(
            (mean_kt - reference).abs() < 0.01 * reference,
            "equilibrium is {mean_kt:.4} kT at {temperature_k} K against \
             {reference:.4} kT at 294 K. The kernel is a function of E/kT alone, \
             so these must agree (#478)"
        );
    }
}
