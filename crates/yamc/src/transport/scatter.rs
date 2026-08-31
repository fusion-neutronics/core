//! Neutron scattering kinematics: elastic, inelastic, and "other" scatter
//! channels, the CM elastic-mu sampler, and weight-cutoff roulette.

use super::*;

/// MT 2 elastic scattering, routed through the shared GPU/CPU samplers
/// `yamc_physics::gpu::flat::{elastic_mu_cm, free_gas_elastic}` driven by a
/// threaded 64-bit PCG state (issues #111, #274). The CPU and the GPU kernel +
/// CPU-twin now sample the *same* tabulated CM angular distribution and the
/// *same* free-gas thermal kinematics, so the two transport paths cannot drift
/// (the cause of the #88 epithermal divergence class).
///
/// Draw schedule (PCG `next_xi`, in the kernel/twin order): `xi3` (the angle
/// seed, drawn once at the reaction-type split and passed in -- issue #111)
/// seeds the isotropic fallback, then `elastic_mu_cm` (bracket + CDF inversion),
/// then `free_gas_elastic`, then ONE lab-azimuth draw
/// `phi = TAU * next_xi(pcg)`. For the cold-target high-energy regime
/// (`E >= 400*kT`, `AWR > 1`) `free_gas_elastic` reports `did_run = false` and
/// the closed-form two-body kinematics are applied, rotated by that azimuth
/// (equivalent to the previous full CM transform with a target at rest).
///
/// The azimuth draw is UNCONDITIONAL (issue #111 site D): the GPU kernel and
/// its CPU twin draw it in the common tail after the reaction branch, for warp
/// coherence, and skip only the ROTATION when the free-gas vector CM transform
/// already wrote the direction (that transform is not reducible to a
/// `(mu_lab, phi)` rotation about the incident direction). Skipping the draw as
/// well would consume one fewer value on the free-gas path and desynchronise
/// the rest of the history from the GPU stream -- worth ~100% of thermal
/// moderating histories in the issue-#40 matched-stream harness.
///
/// Always returns `true` (the free-gas sampler handles the near-zero CM-speed
/// case internally instead of terminating the history).
///
/// `free_gas_threshold` (the model option, default `400.0`) sets the free-gas
/// regime boundary at `free_gas_threshold * kT`. It is forwarded to the shared
/// `free_gas_elastic` sampler, which the GPU kernel + CPU twin pass the same
/// value, so the boundary cannot drift between backends (issue #102).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_elastic(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    material: &Material,
    constituent_reaction: &Reaction,
    free_gas_threshold: f64,
    particle_idx: usize,
    // Angle seed drawn once at the reaction-type split (issue #111), mirroring
    // the kernel/twin `xi3`. Replaces the draw this function used to make.
    xi3: f64,
    pcg: &mut u64,
) -> bool {
    use yamc_rng::next_xi;

    // Debug: count MT 2 (elastic)
    #[cfg(feature = "debug_runtime")]
    TOTAL_SCATTER_MT2.fetch_add(1, Ordering::Relaxed);

    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    let temperature_k = material.temperature_k();
    let e_in = particle.energy;
    let [dx, dy, dz] = particle.direction;

    // Locate the elastic neutron product's CM angular distribution (if any),
    // then fetch the per-nuclide flat table (built + cached on first use).
    let angle = constituent_reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))
        .and_then(|p| p.distribution.first())
        .and_then(|d| match d {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } => Some(angle),
            _ => None,
        });
    let flat = nuclide.elastic_flat_cache.get_or_build(angle);

    // (1) CM scattering cosine via the shared tabulated sampler. `xi3` (the
    // angle seed, passed in from the reaction-type split) seeds the isotropic
    // fallback, mirroring the kernel/twin `xi3` role.
    let mu_cm = yamc_physics::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm(
        e_in,
        xi3,
        &flat.energy_grid,
        &flat.n_mu,
        &flat.interp,
        &flat.mu,
        &flat.cdf,
        &flat.pdf,
        &flat.mu_offset,
        pcg,
    );

    // (2) Free-gas elastic kinematics (epithermal / thermal regime). Returns
    // `did_run = false` for the cold-target high-energy regime, handled below.
    let (new_dx, new_dy, new_dz, new_e, did_run) =
        yamc_physics::gpu::flat::free_gas_elastic::sample_free_gas_elastic(
            e_in,
            dx,
            dy,
            dz,
            awr,
            temperature_k,
            free_gas_threshold,
            mu_cm,
            pcg,
        );

    // (3) Lab azimuth, drawn UNCONDITIONALLY here (issue #111 site D). The GPU
    // kernel and its CPU twin make this draw in the common tail after the
    // reaction branch, i.e. immediately after the free-gas sampler on the
    // elastic arm, whether or not free-gas already wrote the direction, and
    // then skip only the ROTATION. Drawing it conditionally here consumed one
    // fewer value on the free-gas path and desynchronised the rest of the
    // history from the GPU stream.
    let phi = std::f64::consts::TAU * next_xi(pcg);

    let (e_out, out_dir) = if did_run {
        // Free-gas wrote the direction with its vector CM transform, which is
        // not reducible to a (mu_lab, phi) rotation about the incident
        // direction: the draw above is consumed, the rotation is skipped.
        (new_e, [new_dx, new_dy, new_dz])
    } else {
        // Cold-target asymptotic two-body kinematics (= the full CM transform
        // with the target at rest); the lab azimuth is uniform.
        let one_plus_a = awr + 1.0;
        let denom = one_plus_a * one_plus_a;
        let numer = awr * awr + 2.0 * awr * mu_cm + 1.0;
        let e_out = e_in * numer / denom;
        let mu_lab = (1.0 + awr * mu_cm) / numer.sqrt();
        (e_out, rotate_direction_fast(dx, dy, dz, mu_lab, phi))
    };

    if is_debug_elastic() {
        log_elastic_scatter(e_in, mu_cm, e_out, did_run);
    }

    if e_out > 1e-20 {
        particle.energy = e_out;
        particle.direction = out_dir;
        // Debug collision logging: elastic scatter result
        #[cfg(feature = "debug_collision")]
        log_scatter_result(particle_idx, e_in, e_out, 2);
    } else {
        // Extremely low energy - particle effectively stopped.
        particle.energy = 1e-11;
        #[cfg(feature = "debug_collision")]
        log_scatter_result(particle_idx, e_in, 1e-11, 2);
    }
    let _ = particle_idx; // used only under `debug_collision`
    true
}

/// Inelastic scatter (MT 50-91) and level-specific (n,2n) (MT 875-890):
/// sample the tabulated product distributions, update the primary, and
/// bank any extra outgoing neutrons. Returns `false` when the reaction
/// produces no particles (terminate the history, matching the original
/// inline `return`); `true` otherwise.
pub(super) fn scatter_inelastic(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    constituent_reaction: &Reaction,
    particle_idx: usize,
    particle_bank: &mut ParticleBank,
    rng: &mut FastRng,
) -> bool {
    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    if constituent_reaction.products.is_empty() {
        panic!(
            "Missing product distributions for sampled inelastic reaction MT {} at E={:.4e} eV",
            constituent_reaction.mt_number, particle.energy
        );
    }

    // Save incoming energy for debug logging
    #[cfg(feature = "debug_collision")]
    let e_in_inelastic = particle.energy;

    let outgoing_particles = yamc_physics::neutron::inelastic::sample_from_products_with_awr(
        particle,
        constituent_reaction,
        awr,
        rng,
    );

    // These reactions may produce multiple particles (yield > 1 for n,2n)
    if outgoing_particles.is_empty() {
        // No particles produced - this shouldn't happen for these MTs
        return false;
    }

    // Handle single or multiple outgoing particles
    if outgoing_particles.len() == 1 {
        let outgoing = &outgoing_particles[0];
        particle.energy = outgoing.energy;
        particle.direction = outgoing.direction;
        particle.position = outgoing.position;
        particle.weight = outgoing.weight;
        // Debug collision logging: inelastic scatter result
        #[cfg(feature = "debug_collision")]
        log_scatter_result(
            particle_idx,
            e_in_inelastic,
            outgoing.energy,
            constituent_reaction.mt_number,
        );
    } else {
        // Multi-neutron reaction - bank secondary particles
        for (i, outgoing_particle) in outgoing_particles.into_iter().enumerate() {
            if i == 0 {
                particle.energy = outgoing_particle.energy;
                particle.direction = outgoing_particle.direction;
                particle.position = outgoing_particle.position;
                particle.weight = outgoing_particle.weight;
            } else {
                particle_bank.bank_secondary(outgoing_particle);
            }
        }
    }
    let _ = particle_idx; // used only under `debug_collision`
    true
}

/// CONTINUUM / tabulated inelastic (MT 50-91, MT 875-890) on the shared PCG
/// stream (issue #111 sub-step 3): the same
/// `yamc_physics::gpu::flat::inelastic_dispatch::sample_inelastic_kinematics`
/// the GPU kernel and its CPU twin call, driven by the threaded 64-bit PCG
/// state instead of the legacy `FastRng`.
///
/// The reaction's per-law outgoing-energy + angular tables come from the
/// run-scoped [`InelasticFlatCache`], which flattens the reaction through the
/// SAME extractors yamc-gpu packs into its per-material buffers, so the CPU
/// and the GPU read byte-identical data (pinned by yamc-gpu's
/// `inelastic_flat_cache_parity` test).
///
/// Draw schedule, matching the GPU twin's inelastic arm
/// (`yamc-gpu::neutron::transport::shared`) exactly: `xi3` (the angle seed
/// drawn once at the reaction-type split) is passed in and seeds the
/// isotropic fallback, `sample_inelastic_kinematics` then makes the law's
/// draws, and finally ONE azimuth draw `phi = TAU * next_xi(pcg)` feeds
/// `rotate_direction_fast` (as `scatter_inelastic_level` does).
///
/// Returns `None` when the reaction carries no outgoing-energy law this flat
/// layer recognises (`InelasticFlat::has_outgoing_energy_law`); the caller
/// then keeps the legacy [`scatter_inelastic`] path so no CPU behaviour is
/// lost. Otherwise returns `Some(proceed)` with the same meaning as
/// `scatter_inelastic_level`: `false` only when the yield is zero (no neutron
/// produced), `true` otherwise, including the rare failed-kinematics case
/// where the particle is left unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_inelastic_shared(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    constituent_reaction: &Reaction,
    particle_idx: usize,
    particle_bank: &mut ParticleBank,
    // Angle seed drawn once at the reaction-type split, the twin's `xi3`.
    xi3: f64,
    pcg: &mut u64,
    flat_cache: &InelasticFlatCache,
) -> Option<bool> {
    use yamc_rng::next_xi;

    let flat = flat_cache.get_or_build(nuclide, constituent_reaction);
    if !flat.has_outgoing_energy_law() {
        return None;
    }
    // First neutron product carries the yield (the kinematics tables are
    // already flattened); no neutron product means nothing to sample here.
    let neutron_product = constituent_reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))?;

    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    let e_in = particle.energy;

    // Closed-form CM energy, computed exactly as the GPU twin does before its
    // `sample_inelastic_kinematics` call. Only the `LevelInelastic` fallback
    // branch consumes it (every law reaching here overrides it), but it is
    // passed with the same value and in the same expression order regardless.
    let abs_q = flat.q_value.abs();
    let threshold = (awr + 1.0) / awr * abs_q;
    let mass_ratio = (awr / (awr + 1.0)).powi(2);
    let e_cm_closed = mass_ratio * (e_in - threshold);

    let (mu_lab, e_out, ok) = flat.sample_kinematics(e_in, awr, e_cm_closed, xi3, pcg);
    if ok {
        let phi = std::f64::consts::TAU * next_xi(pcg);
        let [dx, dy, dz] = particle.direction;
        particle.direction = rotate_direction_fast(dx, dy, dz, mu_lab, phi);
        particle.energy = if e_out > 1e-20 { e_out } else { 1e-11 };
        #[cfg(feature = "debug_collision")]
        log_scatter_result(
            particle_idx,
            e_in,
            particle.energy,
            constituent_reaction.mt_number,
        );
    }

    // Yield / multi-neutron handling, identical to `scatter_inelastic_level`
    // (itself mirroring `sample_from_products_with_awr`).
    let yield_val = neutron_product
        .product_yield
        .as_ref()
        .map(|y| y.evaluate(e_in))
        .unwrap_or(1.0);
    if yield_val.abs() < 1e-10 {
        return Some(false);
    }
    let is_integral = (yield_val - yield_val.floor()).abs() < 1e-10;
    if is_integral {
        let n_total = yield_val.round() as usize;
        for _ in 1..n_total {
            particle_bank.bank_secondary(particle.clone());
        }
    } else {
        particle.weight *= yield_val;
    }
    let _ = particle_idx; // used only under `debug_collision`
    Some(true)
}

/// DISCRETE inelastic levels (MT 51-90, and LevelInelastic MT 875-890) with
/// closed-form-Q kinematics on the shared PCG stream (issue #111 sub-step 3).
///
/// Routed here (from the analog single-nuclide path) only when the constituent's
/// first neutron product carries a `LevelInelastic` energy distribution. The CM
/// outgoing energy is the closed-form Q value and the CM cosine is drawn from
/// the same `elastic_mu_cm` sampler the GPU reuses for inelastic, so the level
/// kinematics are bit-identical to the GPU twin. The lab azimuth uses
/// `TAU * next_xi(pcg)` (matching `scatter_elastic`). Yield / multi-neutron
/// handling preserves the existing convention (`sample_from_products_with_awr`):
/// integral yield banks copies, fractional yield scales the weight; the common
/// single-neutron levels are a no-op there.
///
/// Returns `false` only when the yield is zero (no neutron produced); `true`
/// otherwise, including the rare `e_lab <= 0` case where the particle is left
/// unchanged (matching the GPU fallback).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_inelastic_level(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    constituent_reaction: &Reaction,
    particle_idx: usize,
    particle_bank: &mut ParticleBank,
    xi3: f64,
    pcg: &mut u64,
) -> bool {
    use yamc_rng::next_xi;
    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    let e_in = particle.energy;

    // First neutron product carries the level's angular distribution + yield.
    let Some(neutron_product) = constituent_reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))
    else {
        return false;
    };
    let angle = neutron_product.distribution.first().and_then(|d| match d {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } => Some(angle),
        _ => None,
    });
    let flat = nuclide
        .inelastic_angle_flat_cache
        .get_or_build(constituent_reaction.mt_number, angle);

    // Closed-form-Q energy + shared CM-cosine sampler (bit-identical to the GPU);
    // `None` => non-positive lab energy, leave the particle unchanged.
    if let Some((e_lab, mu_lab)) = yamc_physics::neutron::inelastic::sample_level_inelastic(
        e_in,
        awr,
        constituent_reaction.q_value,
        constituent_reaction.scatter_in_cm,
        xi3,
        &flat.energy_grid,
        &flat.n_mu,
        &flat.interp,
        &flat.mu,
        &flat.cdf,
        &flat.pdf,
        &flat.mu_offset,
        pcg,
    ) {
        let phi = std::f64::consts::TAU * next_xi(pcg);
        let [dx, dy, dz] = particle.direction;
        particle.direction = rotate_direction_fast(dx, dy, dz, mu_lab, phi);
        particle.energy = if e_lab > 1e-20 { e_lab } else { 1e-11 };
        #[cfg(feature = "debug_collision")]
        log_scatter_result(
            particle_idx,
            e_in,
            particle.energy,
            constituent_reaction.mt_number,
        );
    }

    // Yield / multi-neutron handling, mirroring sample_from_products_with_awr.
    let yield_val = neutron_product
        .product_yield
        .as_ref()
        .map(|y| y.evaluate(e_in))
        .unwrap_or(1.0);
    if yield_val.abs() < 1e-10 {
        return false;
    }
    let is_integral = (yield_val - yield_val.floor()).abs() < 1e-10;
    if is_integral {
        let n_total = yield_val.round() as usize;
        for _ in 1..n_total {
            particle_bank.bank_secondary(particle.clone());
        }
    } else {
        particle.weight *= yield_val;
    }
    let _ = particle_idx; // used only under `debug_collision`
    true
}

/// (n,xn) / (n,n'x) channels -- MT 5 / 16 / 17 / 22 / 28 / 32 / 33 / 34 / ...,
/// everything outside the `50..=91 | 875..=890` inelastic series -- on the
/// shared PCG stream (issue #111). The GPU twin samples these MTs from the same
/// flat tables through the same `sample_inelastic_kinematics` dispatcher, so
/// after this routing a collision on one of them is bit-identical across the two
/// backends instead of diverging in [`scatter_other`]'s legacy `FastRng`
/// kinematics.
///
/// Draw schedule, mirroring the GPU twin's `k_out` loop
/// (`yamc-gpu::neutron::transport::shared`) draw-for-draw:
///   * `xi3` (the angle seed drawn once at the reaction-type split) is passed
///     in and seeds the walk's isotropic fallback;
///   * `sample_kinematics` makes the walk's law draws;
///   * each ANALOG (n,xn) secondary (`yield - 1` of them, see below) then draws
///     its own fresh isotropic-fallback uniform, its own law draws, and ONE
///     azimuth `TAU * next_xi(pcg)`, and is rotated around the INCIDENT
///     direction before being banked;
///   * finally ONE azimuth draw rotates the continuing walk, exactly as
///     [`scatter_inelastic_shared`] does. The walk's azimuth comes AFTER the
///     secondaries, as it does in the kernel.
///
/// MULTIPLICITY (issue #274 convention, unchanged here): an integral yield
/// banks `yield - 1` extra neutrons at the walk's weight; a fractional yield
/// multiplies the walk's weight instead. Each extra neutron is sampled
/// INDEPENDENTLY from the reaction's distributions, as the GPU kernel does.
/// (The legacy [`scatter_other`] path banks `yield - 1` CLONES of the one
/// sampled outcome, which has the same mean and the same multiplicity but
/// perfectly correlated secondaries; independent secondaries are what the GPU
/// samples and what the estimator variance should see.)
///
/// Returns `None` -- with no draw consumed and the particle untouched -- when
/// the flat layer does not recognise the reaction's outgoing-energy law
/// (`InelasticFlat::has_outgoing_energy_law`, which also covers a reaction with
/// no neutron product at all, i.e. the charged-particle-only channels). The
/// caller then keeps the legacy [`scatter_other`] path, so no CPU behaviour is
/// lost. `Some(())` means the collision was fully handled here; like
/// [`scatter_other`] this arm never terminates the history early (the caller
/// always proceeds to score the collision), it only marks the particle dead.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_other_shared(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    constituent_reaction: &Reaction,
    particle_idx: usize,
    particle_bank: &mut ParticleBank,
    // Angle seed drawn once at the reaction-type split, the twin's `xi3`.
    xi3: f64,
    pcg: &mut u64,
    flat_cache: &InelasticFlatCache,
) -> Option<()> {
    use yamc_rng::next_xi;

    let flat = flat_cache.get_or_build(nuclide, constituent_reaction);
    if !flat.has_outgoing_energy_law() {
        return None;
    }
    // First neutron product carries the yield (the kinematics tables are
    // already flattened); no neutron product means nothing to sample here.
    let neutron_product = constituent_reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))?;

    // Debug: count non-elastic scattering, as `scatter_other` does.
    #[cfg(feature = "debug_runtime")]
    TOTAL_SCATTER_MT_OTHER.fetch_add(1, Ordering::Relaxed);

    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    let e_in = particle.energy;
    // INCIDENT direction: the walk and every secondary are rotated around it
    // (the GPU rotates the walk only after its `k_out` loop, so the extras see
    // the pre-collision direction).
    let [dx, dy, dz] = particle.direction;

    // Yield / multi-neutron handling, keeping this arm's existing convention
    // (`sample_from_products_with_awr`): zero yield is absorption, an integral
    // yield emits that many neutrons, a fractional yield scales the weight.
    // This coincides with the kernel's rule for every yield >= 1.
    let yield_val = neutron_product
        .product_yield
        .as_ref()
        .map(|y| y.evaluate(e_in))
        .unwrap_or(1.0);
    if yield_val.abs() < 1e-10 {
        // No neutron produced: absorption, primary killed with its incident
        // energy / direction intact (the legacy path's empty-product outcome).
        particle.alive = false;
        return Some(());
    }
    let is_integral = (yield_val - yield_val.floor()).abs() < 1e-10;
    let n_out = if is_integral {
        yield_val.round() as usize
    } else {
        particle.weight *= yield_val;
        1
    };

    // Closed-form CM energy, computed exactly as the GPU twin does before its
    // `sample_inelastic_kinematics` call (only the `LevelInelastic` fallback
    // branch consumes it, but it is passed the same way regardless).
    let abs_q = flat.q_value.abs();
    let threshold = (awr + 1.0) / awr * abs_q;
    let mass_ratio = (awr / (awr + 1.0)).powi(2);
    let e_cm_closed = mass_ratio * (e_in - threshold);

    // Iteration 0 is the continuing walk; iterations 1.. are the analog (n,xn)
    // secondaries. Twin of the kernel's `k_out` loop.
    let mut walk_mu = 0.0;
    let mut walk_e = e_in;
    let mut walk_ok = false;
    for k_out in 0..n_out {
        let xi3_k = if k_out == 0 { xi3 } else { next_xi(pcg) };
        let (mu, e_out, ok) = flat.sample_kinematics(e_in, awr, e_cm_closed, xi3_k, pcg);
        if k_out == 0 {
            walk_mu = mu;
            walk_e = e_out;
            walk_ok = ok;
        } else if ok {
            // Extra analog secondary: one azimuth draw, rotated around the
            // INCIDENT direction, banked at the walk's weight and position.
            // A failed-kinematics extra is dropped, as the kernel drops it.
            let phi = std::f64::consts::TAU * next_xi(pcg);
            let mut secondary = particle.clone();
            secondary.energy = if e_out > 1e-20 { e_out } else { 1e-11 };
            secondary.direction = rotate_direction_fast(dx, dy, dz, mu, phi);
            particle_bank.bank_secondary(secondary);
        }
    }

    // The walk's own lab azimuth, drawn AFTER the secondaries and
    // unconditionally (the kernel draws it for every non-absorption channel),
    // then the rotation. Failed walk kinematics leave the energy untouched and
    // kill the walk, matching the kernel's `alive = 0` on that branch.
    let phi = std::f64::consts::TAU * next_xi(pcg);
    particle.direction = rotate_direction_fast(dx, dy, dz, walk_mu, phi);
    if walk_ok {
        particle.energy = if walk_e > 1e-20 { walk_e } else { 1e-11 };
        #[cfg(feature = "debug_collision")]
        log_scatter_result(
            particle_idx,
            e_in,
            particle.energy,
            constituent_reaction.mt_number,
        );
    } else {
        particle.alive = false;
    }
    let _ = particle_idx; // used only under `debug_collision`
    Some(())
}

/// General "other" scattering reactions -- (n,2n), (n,3n), (n,n'alpha),
/// etc. -- via the scatter module, with CM→LAB conversion. Updates the
/// primary and banks extra neutrons; an empty product set is treated as
/// absorption (the primary is killed). Never terminates the history early.
///
/// Legacy `FastRng` path: reached from the analog single-nuclide arm only when
/// [`scatter_other_shared`] does not recognise the reaction's outgoing-energy
/// law, and from the survival-biasing / multi-nuclide path (which selects its
/// constituent off `FastRng` and is not stream-matched to the GPU).
pub(super) fn scatter_other(
    particle: &mut yamc_particle::particle::Particle,
    nuclide: &Nuclide,
    nuclide_name: &str,
    constituent_reaction: &Reaction,
    particle_bank: &mut ParticleBank,
    rng: &mut FastRng,
) {
    // Debug: count non-elastic scattering
    #[cfg(feature = "debug_runtime")]
    TOTAL_SCATTER_MT_OTHER.fetch_add(1, Ordering::Relaxed);

    let awr = nuclide
        .atomic_weight_ratio
        .expect("No atomic weight ratio for nuclide");
    let outgoing_particles = yamc_physics::neutron::scatter::scatter_with_awr(
        particle,
        constituent_reaction,
        nuclide_name,
        awr,
        rng,
    );

    // Some reactions like MT 5 (n,misc) may produce no neutrons at certain energies
    // (e.g., when only charged particles are emitted). Treat this as absorption.
    if outgoing_particles.is_empty() {
        particle.alive = false;
    } else if outgoing_particles.len() == 1 {
        // Single neutron - update current particle
        let outgoing = &outgoing_particles[0];
        particle.energy = outgoing.energy;
        particle.direction = outgoing.direction;
        particle.position = outgoing.position;
        particle.weight = outgoing.weight;
    } else {
        // Multi-neutron reaction - bank secondary particles
        for (i, outgoing_particle) in outgoing_particles.into_iter().enumerate() {
            if i == 0 {
                particle.energy = outgoing_particle.energy;
                particle.direction = outgoing_particle.direction;
                particle.position = outgoing_particle.position;
                particle.weight = outgoing_particle.weight;
            } else {
                particle_bank.bank_secondary(outgoing_particle);
            }
        }
    }
}

/// Weight-cutoff Russian-roulette decision. Given the pre-roulette weight,
/// the survival weight, and the random draw `xi ∈ [0, 1)`, returns
/// `Some(weight_survive)` if the particle survives (continuing at that
/// weight) or `None` if it is killed. Pure -- the caller owns the RNG draw
/// and the tracker bookkeeping. Survival probability `weight / weight_survive`
/// leaves the expected weight unchanged.
pub(super) fn weight_cutoff_roulette(weight: f64, weight_survive: f64, xi: f64) -> Option<f64> {
    if xi < weight / weight_survive {
        Some(weight_survive)
    } else {
        None
    }
}
