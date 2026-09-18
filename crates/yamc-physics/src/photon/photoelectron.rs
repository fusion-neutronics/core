//! Photoelectron emission direction.
//!
//! Split out of the CPU photoelectric branch so the GPU kernel's `#[cube]`
//! copy (`yamc_gpu::common::probes::sauter_direction`) is tested against the
//! exact sampler the CPU transport runs. Keep the two in lock-step, the same
//! arrangement `sample_ttb_photon_energy` uses for the TTB inverse CDF.

use rand::RngExt;
use yamc_element::photon::MASS_ELECTRON_EV;

/// Sample the photoelectron's emission direction from the non-relativistic
/// Sauter distribution (Sauter, Ann. Phys. 11, 454-488, 1931; sampling per
/// Kaltiaisenaho, Comput. Phys. Commun. 252, 107143, 2020, Eqns 3.19-3.20).
///
/// `electron_energy` is the photoelectron kinetic energy in eV.
///
/// The returned direction takes `mu` about the **lab +x axis**, not about the
/// incident photon's direction. That is what OpenMC does
/// (`sample_photon_reaction` in `src/physics.cpp`, which assigns
/// `u.x = mu` directly), and the photoelectron is only ever used as a
/// bremsstrahlung source, so this reproduces it rather than rotating into the
/// photon frame and diverging from the benchmark.
pub fn sample_photoelectron_direction<R: rand::Rng + ?Sized>(
    electron_energy: f64,
    rng: &mut R,
) -> [f64; 3] {
    let mu = loop {
        let r: f64 = rng.random::<f64>();
        if 4.0 * (1.0 - r) * r >= rng.random::<f64>() {
            let rel_vel = (electron_energy * (electron_energy + 2.0 * MASS_ELECTRON_EV)).sqrt()
                / (electron_energy + MASS_ELECTRON_EV);
            break (2.0 * r + rel_vel - 1.0) / (2.0 * rel_vel * r - rel_vel + 1.0);
        }
    };
    let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    [mu, sin_theta * phi.cos(), sin_theta * phi.sin()]
}
