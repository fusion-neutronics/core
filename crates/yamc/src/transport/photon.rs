//! Photon collision physics: coherent/Compton/photoelectric/pair-production
//! channels, TTB photon banking, and fluorescence/Auger collection.

use super::*;

/// Handle a photon collision: sample reaction type and process the interaction.
///
/// Samples from coherent, incoherent (Compton), photoelectric, and pair production channels.
/// Updates particle state (energy, direction, alive) and banks secondary
/// photons from atomic relaxation and pair production annihilation.
///
/// # Returns
/// `(mt, bank_second_photon_energy)` where `mt` is the sampled reaction MT number
/// and `bank_second_photon_energy` is the total kinetic energy [eV] of secondary
/// photons that were banked (for analog heating calculation).
pub(crate) fn handle_photon_collision<R: rand::Rng + ?Sized>(
    particle: &mut yamc_particle::particle::Particle,
    material: &Material,
    macro_xs: &MacroPhotonXS,
    particle_bank: &mut ParticleBank,
    photon_cutoff: f64,
    rng: &mut R,
) -> (i32, f64) {
    let alpha = particle.energy / MASS_ELECTRON_EV;

    // Sample which element the photon interacts with, then its micro XS.
    let (_i_element, element) = material.sample_element(macro_xs.total, particle.energy, rng);
    let micro_xs = element.calculate_xs(particle.energy);

    // Sample the reaction channel via cumulative comparison against the
    // total micro XS, then dispatch to the per-channel handler. Each
    // handler returns `(mt, banked_photon_energy)`. The RNG draw order
    // here AND inside the handlers is load-bearing for reproducibility,
    // so the handlers are invoked exactly where their branch used to run.
    let cutoff = rng.random::<f64>() * micro_xs.total;
    let mut prob = 0.0;

    prob += micro_xs.coherent;
    if prob > cutoff {
        return photon_coherent_scatter(particle, &element, alpha, rng);
    }

    prob += micro_xs.incoherent;
    if prob > cutoff {
        return photon_compton_scatter(
            particle,
            material,
            &element,
            alpha,
            photon_cutoff,
            particle_bank,
            rng,
        );
    }

    prob += micro_xs.photoelectric;
    if prob > cutoff {
        return photon_photoelectric(
            particle,
            material,
            &element,
            &micro_xs,
            photon_cutoff,
            particle_bank,
            rng,
        );
    }

    prob += micro_xs.pair_production;
    if prob > cutoff {
        return photon_pair_production(
            particle,
            material,
            &element,
            alpha,
            photon_cutoff,
            particle_bank,
            rng,
        );
    }

    // Should not reach here.
    (0, 0.0)
}

/// Bank one charged particle's thick-target bremsstrahlung (TTB) photons
/// into `particle_bank`, returning the total banked photon energy [eV].
///
/// No-op (returns `0.0`, draws no random numbers) when the material has no
/// TTB data or the electron/positron energy is at or below `photon_cutoff`
/// -- the same guard the inline call sites used, so callers can invoke it
/// unconditionally without changing behaviour or the RNG stream.
#[allow(clippy::too_many_arguments)]
fn bank_ttb_photons<R: rand::Rng + ?Sized>(
    material: &Material,
    energy: f64,
    is_positron: bool,
    position: [f64; 3],
    direction: [f64; 3],
    weight: f64,
    photon_cutoff: f64,
    particle_bank: &mut ParticleBank,
    rng: &mut R,
) -> f64 {
    if let Some(ref ttb_data) = material.ttb {
        if energy > photon_cutoff {
            return yamc_physics::photon::bremsstrahlung::thick_target_bremsstrahlung(
                energy,
                is_positron,
                ttb_data,
                position,
                direction,
                weight,
                photon_cutoff,
                particle_bank,
                rng,
            );
        }
    }
    0.0
}

/// Process an atomic-relaxation cascade's secondaries: bank fluorescent
/// photons at or above `photon_cutoff` (accumulating their energy) and
/// collect Auger electrons (energy + direction) for downstream TTB.
/// Returns `(banked_photon_energy, auger_electrons)`. Draws no random
/// numbers. `count_fluorescence_diag` gates the photoelectric
/// fluorescence diagnostics counters (Compton passes `false`).
pub(super) fn bank_fluorescence_collect_auger(
    secondaries: Vec<(f64, [f64; 3], bool)>,
    particle: &yamc_particle::particle::Particle,
    photon_cutoff: f64,
    particle_bank: &mut ParticleBank,
    count_fluorescence_diag: bool,
) -> (f64, Vec<(f64, [f64; 3])>) {
    let _ = count_fluorescence_diag; // read only under `debug_diagnostics`
    let mut banked = 0.0;
    let mut auger_electrons: Vec<(f64, [f64; 3])> = Vec::new();
    for (energy, direction, is_photon) in secondaries {
        if is_photon && energy >= photon_cutoff {
            // Only bank fluorescent photons above the energy cutoff
            let mut secondary =
                yamc_particle::particle::Particle::new(particle.position, direction, energy);
            secondary.particle_type = ParticleType::Photon;
            secondary.weight = particle.weight;
            secondary.alive = true;
            banked += energy;
            particle_bank.bank_secondary(secondary);
            #[cfg(feature = "debug_diagnostics")]
            if count_fluorescence_diag {
                photon_diag::FLUORESCENCE_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if (5000.0..10000.0).contains(&energy) {
                    photon_diag::FLUORESCENCE_5_10
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        } else if !is_photon && energy > 0.0 {
            // Auger electron: deposits locally (minus TTB below)
            auger_electrons.push((energy, direction));
        }
    }
    (banked, auger_electrons)
}

/// Coherent (Rayleigh) scattering: rotate the photon direction; no energy
/// change and no secondary photons. Returns `(502, 0.0)`.
fn photon_coherent_scatter<R: rand::Rng + ?Sized>(
    particle: &mut yamc_particle::particle::Particle,
    element: &PhotonInteraction,
    alpha: f64,
    rng: &mut R,
) -> (i32, f64) {
    let mu = element.rayleigh_scatter(alpha, rng);
    let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    particle.direction = rotate_direction_fast(
        particle.direction[0],
        particle.direction[1],
        particle.direction[2],
        mu,
        phi,
    );
    (502, 0.0) // COHERENT
}

/// Incoherent (Compton) scattering: scatter the photon, bank the
/// relaxation cascade, and convert the Compton + Auger electron energy to
/// TTB photons. Returns `(504, banked_photon_energy)`.
fn photon_compton_scatter<R: rand::Rng + ?Sized>(
    particle: &mut yamc_particle::particle::Particle,
    material: &Material,
    element: &PhotonInteraction,
    alpha: f64,
    photon_cutoff: f64,
    particle_bank: &mut ParticleBank,
    rng: &mut R,
) -> (i32, f64) {
    let mut bank_second_photon_energy = 0.0;
    let (alpha_out, mu, i_shell) = element.compton_scatter(alpha, true, rng);

    let e_out = alpha_out * MASS_ELECTRON_EV;

    // Compton electron kinetic energy = E_in - E_out - E_binding
    let binding_energy = if i_shell >= 0 {
        element.binding_energy[i_shell as usize]
    } else {
        0.0
    };
    let electron_energy = (particle.energy - e_out - binding_energy).max(0.0);

    // Sample azimuthal angle
    let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);

    // Compute Compton electron direction from kinematics BEFORE rotating the
    // photon. Relativistic energy-momentum conservation gives:
    //   mu_electron = (alpha - alpha_out * mu) / sqrt(alpha^2 + alpha_out^2
    //                 - 2*alpha*alpha_out*mu)
    let electron_direction = {
        let denom = (alpha * alpha + alpha_out * alpha_out - 2.0 * alpha * alpha_out * mu)
            .max(0.0)
            .sqrt();
        let mu_electron = if denom > 0.0 {
            ((alpha - alpha_out * mu) / denom).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        rotate_direction_fast(
            particle.direction[0],
            particle.direction[1],
            particle.direction[2],
            mu_electron,
            phi,
        )
    };

    // Rotate photon direction (phi+PI offset ensures photon/electron momentum conservation)
    particle.direction = rotate_direction_fast(
        particle.direction[0],
        particle.direction[1],
        particle.direction[2],
        mu,
        phi + std::f64::consts::PI,
    );
    particle.energy = e_out;

    // The Compton electron deposits its kinetic energy locally (minus the
    // TTB-radiated photons banked below), so it is NOT counted here.

    // Atomic relaxation for the Compton-ionized shell. The ionized Compton (n, l)
    // shell maps to one or two relaxation (n, l, j) subshells; sample which one
    // took the vacancy, weighted by occupancy (the j-split branching).
    if i_shell >= 0 {
        let targets = element
            .compton_relax_map
            .get(i_shell as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if !targets.is_empty() {
            let i_relax = if targets.len() == 1 {
                targets[0].shell_index
            } else {
                let r: f64 = rng.random();
                let mut acc = 0.0;
                let mut chosen = targets[targets.len() - 1].shell_index;
                for t in targets {
                    acc += t.weight;
                    if r < acc {
                        chosen = t.shell_index;
                        break;
                    }
                }
                chosen
            };
            let secondaries =
                element.atomic_relaxation(i_relax, particle.energy + binding_energy, rng);
            let (banked, auger_electrons) = bank_fluorescence_collect_auger(
                secondaries,
                particle,
                photon_cutoff,
                particle_bank,
                false,
            );
            bank_second_photon_energy += banked;
            // TTB for Auger electrons: each uses its own isotropic direction.
            for (auger_e, auger_dir) in &auger_electrons {
                bank_second_photon_energy += bank_ttb_photons(
                    material,
                    *auger_e,
                    false, // electron
                    particle.position,
                    *auger_dir,
                    particle.weight,
                    photon_cutoff,
                    particle_bank,
                    rng,
                );
            }
        }
    }

    // TTB for the Compton electron: photons inherit the electron's
    // kinematically-computed momentum direction, not the scattered photon's.
    bank_second_photon_energy += bank_ttb_photons(
        material,
        electron_energy,
        false, // electron, not positron
        particle.position,
        electron_direction,
        particle.weight,
        photon_cutoff,
        particle_bank,
        rng,
    );

    (504, bank_second_photon_energy) // INCOHERENT
}

/// Photoelectric absorption: sample the subshell + photoelectron, run the
/// relaxation cascade, convert photoelectron + Auger energy to TTB photons,
/// then kill the photon. Returns `(533 + subshell index, banked_photon_energy)`.
#[allow(clippy::too_many_arguments)]
fn photon_photoelectric<R: rand::Rng + ?Sized>(
    particle: &mut yamc_particle::particle::Particle,
    material: &Material,
    element: &PhotonInteraction,
    micro_xs: &ElementMicroXS,
    photon_cutoff: f64,
    particle_bank: &mut ParticleBank,
    rng: &mut R,
) -> (i32, f64) {
    let mut bank_second_photon_energy = 0.0;

    // Sample which subshell absorbs the photon
    let i_shell = element.sample_photoelectric_subshell(micro_xs, rng);
    #[cfg(feature = "debug_diagnostics")]
    {
        photon_diag::PE_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if i_shell == 0 {
            photon_diag::PE_KSHELL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // Photoelectron kinetic energy = photon_energy - binding_energy
    let binding_energy = element.shells[i_shell].binding_energy;
    let electron_energy = particle.energy - binding_energy;

    // Sample photoelectron direction from non-relativistic Sauter distribution
    // (Sauter, Ann. Phys. 11, 454-488, 1931; sampling per Kaltiaisenaho,
    // Comput. Phys. Commun. 252, 107143, 2020, Eqns 3.19-3.20).
    let electron_direction = {
        let mu_e = loop {
            let r: f64 = rng.random::<f64>();
            if 4.0 * (1.0 - r) * r >= rng.random::<f64>() {
                let rel_vel = (electron_energy * (electron_energy + 2.0 * MASS_ELECTRON_EV)).sqrt()
                    / (electron_energy + MASS_ELECTRON_EV);
                break (2.0 * r + rel_vel - 1.0) / (2.0 * rel_vel * r - rel_vel + 1.0);
            }
        };
        let phi_e: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        // Construct direction in global frame from polar and azimuthal angles
        let sin_theta = (1.0 - mu_e * mu_e).max(0.0).sqrt();
        [mu_e, sin_theta * phi_e.cos(), sin_theta * phi_e.sin()]
    };

    // The photoelectron deposits its kinetic energy locally (minus the
    // TTB-radiated photons banked below), so it is NOT counted here.

    // Atomic relaxation cascade -- fluorescent photons and Auger electrons.
    let secondaries = element.atomic_relaxation(i_shell, particle.energy, rng);
    let (banked, auger_electrons) =
        bank_fluorescence_collect_auger(secondaries, particle, photon_cutoff, particle_bank, true);
    bank_second_photon_energy += banked;

    // TTB: photoelectron first, then each Auger electron (original order).
    bank_second_photon_energy += bank_ttb_photons(
        material,
        electron_energy,
        false, // electron
        particle.position,
        electron_direction,
        particle.weight,
        photon_cutoff,
        particle_bank,
        rng,
    );
    for (auger_e, auger_dir) in &auger_electrons {
        bank_second_photon_energy += bank_ttb_photons(
            material,
            *auger_e,
            false, // electron
            particle.position,
            *auger_dir,
            particle.weight,
            photon_cutoff,
            particle_bank,
            rng,
        );
    }

    // Photon is absorbed
    particle.alive = false;
    particle.energy = 0.0;
    particle.weight = 0.0;

    // MT for photoelectric = 533 + shell designator index
    (
        533 + element.shells[i_shell].index_subshell,
        bank_second_photon_energy,
    )
}

/// Pair production: sample the e-/e+ pair, convert their kinetic energy to
/// TTB photons, bank the two 511 keV annihilation photons, then kill the
/// photon. Returns `(515, banked_photon_energy)`.
fn photon_pair_production<R: rand::Rng + ?Sized>(
    particle: &mut yamc_particle::particle::Particle,
    material: &Material,
    element: &PhotonInteraction,
    alpha: f64,
    photon_cutoff: f64,
    particle_bank: &mut ParticleBank,
    rng: &mut R,
) -> (i32, f64) {
    let mut bank_second_photon_energy = 0.0;

    // Sample electron and positron energies and angles
    let (e_electron, e_positron, mu_e, mu_p) = element.pair_production(alpha, rng);

    // Compute electron and positron directions by rotating the incident photon
    // direction using the sampled polar angle and a random azimuthal angle
    let electron_direction = rotate_direction_fast(
        particle.direction[0],
        particle.direction[1],
        particle.direction[2],
        mu_e,
        rng.random_range(0.0..std::f64::consts::TAU),
    );
    let positron_direction = rotate_direction_fast(
        particle.direction[0],
        particle.direction[1],
        particle.direction[2],
        mu_p,
        rng.random_range(0.0..std::f64::consts::TAU),
    );

    // The pair's kinetic energy deposits locally (minus the TTB-radiated
    // photons banked below); only the annihilation photons carry energy away.
    bank_second_photon_energy += bank_ttb_photons(
        material,
        e_electron,
        false, // electron
        particle.position,
        electron_direction,
        particle.weight,
        photon_cutoff,
        particle_bank,
        rng,
    );
    bank_second_photon_energy += bank_ttb_photons(
        material,
        e_positron,
        true, // positron
        particle.position,
        positron_direction,
        particle.weight,
        photon_cutoff,
        particle_bank,
        rng,
    );

    // Positron annihilation produces two 511 keV photons back-to-back
    let dir = isotropic_direction(rng);
    let neg_dir = [-dir[0], -dir[1], -dir[2]];

    let mut photon1 =
        yamc_particle::particle::Particle::new(particle.position, dir, MASS_ELECTRON_EV);
    photon1.particle_type = ParticleType::Photon;
    photon1.weight = particle.weight;
    photon1.alive = true;
    bank_second_photon_energy += MASS_ELECTRON_EV;
    particle_bank.bank_secondary(photon1);

    let mut photon2 =
        yamc_particle::particle::Particle::new(particle.position, neg_dir, MASS_ELECTRON_EV);
    photon2.particle_type = ParticleType::Photon;
    photon2.weight = particle.weight;
    photon2.alive = true;
    bank_second_photon_energy += MASS_ELECTRON_EV;
    particle_bank.bank_secondary(photon2);

    // Photon is absorbed
    particle.alive = false;
    particle.energy = 0.0;
    particle.weight = 0.0;

    (515, bank_second_photon_energy) // PAIR_PROD
}
