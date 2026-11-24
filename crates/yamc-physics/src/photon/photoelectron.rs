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
/// `electron_energy` is the photoelectron kinetic energy in eV and `incident`
/// the photon's unit direction.
///
/// `mu` is the cosine of the angle to the incident photon's direction, which
/// is what the Sauter distribution describes. OpenMC instead takes it about
/// the lab +x axis (`sample_photon_reaction` assigns `u.x = mu`), and this did
/// too: since the photoelectron's bremsstrahlung is emitted along it, that
/// sent the bremsstrahlung preferentially towards +x whatever way the photon
/// was going, so rotating the geometry changed the photon field. This departs
/// from OpenMC on purpose.
pub fn sample_photoelectron_direction<R: rand::Rng + ?Sized>(
    electron_energy: f64,
    incident: [f64; 3],
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
    crate::neutron::interaction::rotate_direction_fast(
        incident[0],
        incident[1],
        incident[2],
        mu,
        phi,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// The emission is symmetric about the photon's direction: the mean
    /// direction is `<mu>` along it, with nothing transverse, whichever way
    /// the photon travels.
    #[test]
    fn the_photoelectron_is_emitted_about_the_photon_direction() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let n = 200_000;
        for incident in [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.267, -0.445, 0.855]] {
            let norm = incident.iter().map(|v: &f64| v * v).sum::<f64>().sqrt();
            let u = incident.map(|v| v / norm);
            let mut mean = [0.0; 3];
            let mut mu = 0.0;
            for _ in 0..n {
                let d = sample_photoelectron_direction(5.0e5, u, &mut rng);
                for k in 0..3 {
                    mean[k] += d[k] / n as f64;
                }
                mu += (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / n as f64;
            }
            // 500 keV is well forward.
            assert!(mu > 0.7, "{u:?}: <mu> {mu}");
            for k in 0..3 {
                assert!(
                    (mean[k] - mu * u[k]).abs() < 5e-3,
                    "{u:?}: mean direction {mean:?}, expected {mu} along it"
                );
            }
        }
    }
}
