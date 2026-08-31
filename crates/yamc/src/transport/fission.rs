//! Neutron-induced fission: sample the fission reaction and its neutrons.

use super::*;

/// Per-history ceiling on the LIVE particle bank, enforced at fission events.
/// Exceeding it is a hard error, not a cap (issue #348).
///
/// yamc is fixed-source only. In a subcritical model each source neutron seeds
/// a chain that dies out, so the bank drains back to empty; in a supercritical
/// one the chain multiplies faster than the bank drains, so the bank grows
/// without bound and the run simply never finishes. Nothing else notices:
/// `ParticleBank` is an unbounded `Vec`, and the only other per-history budget
/// (`WW_SPLIT_BUDGET_PER_HISTORY`) bounds weight-window splitting, not fission
/// progeny.
///
/// Truncating instead of erroring would be worse than refusing. A cap has to
/// throw away banked neutrons, which destroys their weight and biases the
/// tallies low -- the trap the DeGVR work already hit with a hard per-history
/// particle cap -- and a supercritical fixed-source model has no correct answer
/// to bias TOWARDS. So the chain is reported, not silently shortened.
///
/// # Why the live bank rather than the chain's total progeny
///
/// Total progeny per history does not separate the two regimes: it is
/// `k / (1 - k)` in the mean, with a tail that diverges as k approaches 1, so
/// any budget low enough to catch a runaway promptly also fires on a
/// legitimately subcritical model. Measured on bare U235 spheres at 19.1 g/cm3
/// (20k histories each), the worst history's total progeny climbs 130 -> 1,129
/// -> 19,547 -> 85,276 across r = 5, 7, 8, 8.5 cm.
///
/// The bank the chain is being explored on is far sharper, because the drain is
/// depth-first: the LIVE bank is the frontier of the chain's tree, not its
/// size. Over the same spheres the worst history's peak bank is 21 -> 56 ->
/// 226 -> 2,653. It stays in the hundreds at k ~ 0.99, where the total progeny
/// is already tens of thousands, and it is the quantity that actually grows
/// without bound once k > 1 (and the one that takes the memory with it).
///
/// 20,000 sits clear of both ends: ~90x the peak of a k ~ 0.99 model and ~8x
/// that of one held essentially at critical, while a genuinely diverging chain
/// crosses it within a single history. It is also above
/// `WW_SPLIT_BUDGET_PER_HISTORY`, so weight-window splitting cannot fill the
/// bank to this depth on its own and be mistaken for a runaway chain.
pub(crate) const FISSION_BANK_LIMIT_PER_HISTORY: usize = 20_000;

/// Charge `produced` fission neutrons to the current history and hold the live
/// bank to [`FISSION_BANK_LIMIT_PER_HISTORY`].
///
/// Called once per fission event, on both the analog and the survival-biased
/// arm, so the count covers the whole chain the source history seeded and the
/// check sees the bank at every point where fission grows it.
pub(super) fn charge_fission_progeny(
    particle_bank: &mut ParticleBank,
    produced: usize,
    material: &Material,
    nuclide_name: &str,
) {
    let total = particle_bank.record_fission_progeny(produced);
    if particle_bank.len() > FISSION_BANK_LIMIT_PER_HISTORY {
        let material_name = material
            .name
            .clone()
            .unwrap_or_else(|| "<unnamed>".to_string());
        panic!(
            "Fission chain did not terminate: one source history has \
             {FISSION_BANK_LIMIT_PER_HISTORY}+ fission neutrons still queued \
             ({total} produced so far) in material '{material_name}', most \
             recently in {nuclide_name}. This geometry appears to be \
             supercritical, and yamc does fixed-source transport only -- it has \
             no criticality/eigenvalue mode, so a chain that multiplies faster \
             than it dies out has no finite result to return and the run would \
             never finish. Reduce the fissile inventory, shrink the fissile \
             region, or take this geometry to a criticality code."
        );
    }
}

/// Sample a fission event's outgoing neutron batch: pick the fission
/// reaction (MT 18/19/20/21/38) from the partial cross-sections, evaluate
/// nu-bar at the incident energy, select the neutron products (with a
/// fallback scan for evaluations whose sampled reaction carries no neutron
/// data, e.g. MT 18 for U234/U236/U240), and sample the outgoing neutrons.
/// Extracted verbatim from the analog `ReactionType::Fission` arm of
/// [`handle_neutron_collision`] so survival biasing can reuse it.
///
/// `nu_scale` scales the expected neutron count: 1.0 for an analog fission
/// event; survival biasing passes `sigma_f / sigma_t` so that banking the
/// progeny at every collision reproduces the analog fission source in
/// expectation.
///
/// `mu_xi` is the uniform behind the CONTINUING progeny's isotropic emission
/// cosine. The analog caller passes the angle seed it drew at the reaction
/// split (the GPU kernel's `xi3`, which the kernel's fission branch reuses the
/// same way); a caller with no such seed to hand on draws its own.
///
/// Returns the sampled reaction's MT number (18 when unresolved, for tally
/// compatibility) and the sampled neutrons, each carrying the incident
/// particle's current weight.
/// Extract the prompt fission neutron product's outgoing-energy (chi)
/// distribution, used to build the per-nuclide flat chi cache (issue #111).
fn prompt_chi_dist(
    product: &yamc_nuclide::reaction_product::ReactionProduct,
) -> Option<&yamc_nuclide::reaction_product::EnergyDistribution> {
    product.distribution.first().and_then(|d| match d {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => energy.as_ref(),
        _ => None,
    })
}

/// Resolve which neutron products supply the prompt chi, and the MT they came
/// from. Returns `(None, empty)` when no fission channel carries a neutron
/// product at all.
///
/// Normally the products are the sampled channel's own. An evaluation whose
/// sampled reaction carries no neutron product falls back to the first other
/// fission channel that has some (ENDF hangs the redundant MT 18 of U234, U236
/// and U240 off photons alone), because fission must always produce neutrons.
///
/// The returned MT keys the per-channel chi cache, so it names the channel the
/// products actually came from rather than the sampled one (issue #425). Keying
/// a fallback channel's spectrum under the sampled channel's slot would be the
/// same cross-channel mix the per-MT key exists to prevent.
///
/// `channel_mts` and `channel_rxns` are the fast grid's parallel fission arrays.
fn resolve_chi_products<'a>(
    sampled: Option<&'a yamc_nuclide::reaction::Reaction>,
    channel_mts: &[i32],
    channel_rxns: &'a [Arc<yamc_nuclide::reaction::Reaction>],
) -> (
    Option<i32>,
    Vec<&'a yamc_nuclide::reaction_product::ReactionProduct>,
) {
    fn neutrons(
        rxn: &yamc_nuclide::reaction::Reaction,
    ) -> Vec<&yamc_nuclide::reaction_product::ReactionProduct> {
        rxn.products
            .iter()
            .filter(|p| p.is_particle_type(&yamc_nuclide::reaction_product::ParticleType::Neutron))
            .collect()
    }

    if let Some(rxn) = sampled {
        let products = neutrons(rxn);
        if !products.is_empty() {
            return (Some(rxn.mt_number), products);
        }
    }

    for (j, &mt) in channel_mts.iter().enumerate() {
        // Skip the reaction we already checked.
        if sampled.map(|r| r.mt_number) == Some(mt) {
            continue;
        }
        let Some(reaction) = channel_rxns.get(j) else {
            continue;
        };
        let products = neutrons(reaction);
        if !products.is_empty() {
            return (Some(mt), products);
        }
    }

    (None, Vec::new())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample_fission_event(
    nuclide: &yamc_nuclide::nuclide::Nuclide,
    temperature: &str,
    particle: &yamc_particle::particle::Particle,
    nu_scale: f64,
    mu_xi: f64,
    // Per-particle PCG state for the shared fission-chi path (issue #111).
    pcg: &mut u64,
    rng: &mut FastRng,
) -> (i32, yamc_physics::neutron::interaction::SecondaryNeutrons) {
    use yamc_rng::next_xi;

    // Fission - sample which fission reaction based on partial cross sections.
    // This samples among MT 18, 19, 20, 21, 38 on the shared PCG stream, the
    // last collision-path draw that was still coming off `rng` (issue #418).
    // The uniform is drawn only when the evaluation actually carries partial
    // channels: see `FastXSGrid::sample_fission_reaction` for why a draw taken
    // for a single-channel nuclide would desynchronise the CPU from the kernel.
    let sampled_fission_rxn =
        nuclide.sample_fission_reaction(particle.energy, temperature, || next_xi(pcg));

    // Use sampled reaction's MT for tally (default to 18 for compatibility)
    let tally_mt = sampled_fission_rxn.map(|r| r.mt_number).unwrap_or(18);

    // Get nu-bar (average neutrons per fission) at this energy
    let nu_bar = nuclide
        .fission_nu
        .as_ref()
        .map(|nu_data| nu_data.evaluate(particle.energy))
        .unwrap_or(2.5);

    let (channel_mts, channel_rxns) = nuclide
        .get_temp_idx(temperature)
        .and_then(|temp_idx| nuclide.fast_xs.get(temp_idx))
        .map(|grid| {
            (
                grid.fission_mt_numbers.as_slice(),
                grid.fission_mt_reactions.as_slice(),
            )
        })
        .unwrap_or((&[], &[]));

    // The MT whose products end up supplying the chi, which is the sampled one
    // unless the fallback has to go looking elsewhere. It keys the chi cache, so
    // it has to track the products rather than the sampled reaction (issue #425).
    let (resolved_chi_mt, neutron_products) =
        resolve_chi_products(sampled_fission_rxn, channel_mts, channel_rxns);
    let chi_mt = resolved_chi_mt.unwrap_or(tally_mt);

    let products_ref: Option<&[&yamc_nuclide::reaction_product::ReactionProduct]> =
        if neutron_products.is_empty() {
            None
        } else {
            Some(&neutron_products)
        };

    // Resolve the prompt fission chi, flattened for the shared GPU/CPU flat
    // samplers and cached per (nuclide, fission MT) (issues #111, #425). Keyed by
    // MT because an evaluation with partial fission channels carries a different
    // prompt spectrum on each: the spectrum is fixed per CHANNEL, not per nuclide.
    let chi_flat = nuclide.fission_chi_flat_cache.get_or_build(
        chi_mt,
        products_ref
            .and_then(|p| p.first().copied())
            .and_then(prompt_chi_dist),
    );

    // Delayed neutrons (issue #364). `nu_bar` is nu_TOTAL, so the batch already
    // has the right count; what the delayed groups add is that a `beta` fraction of
    // those neutrons are born from the (much softer) delayed spectrum instead of
    // the prompt one. `None` for a nuclide with no delayed data, which leaves both
    // the spectrum and the draw schedule exactly as they were.
    let delayed = nuclide.delayed_neutrons(temperature).and_then(|d| {
        // nu_d and nu_total come from different blocks of the evaluation, so clamp
        // rather than trust their ratio to stay in range.
        let beta = if nu_bar > 0.0 {
            (d.nu(particle.energy) / nu_bar).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (beta > 0.0).then(|| (beta, d.chi_flat()))
    });

    // Sample fission neutrons
    let fission_neutrons = yamc_physics::neutron::interaction::sample_fission_neutrons(
        particle,
        nu_bar * nu_scale,
        products_ref,
        chi_flat,
        delayed,
        mu_xi,
        pcg,
        rng,
    );

    (tally_mt, fission_neutrons)
}

#[cfg(test)]
mod chi_product_resolution_tests {
    use super::resolve_chi_products;
    use std::sync::Arc;
    use yamc_nuclide::reaction::Reaction;
    use yamc_nuclide::reaction_product::{ParticleType, ReactionProduct};

    fn product(particle: ParticleType) -> ReactionProduct {
        ReactionProduct {
            particle,
            emission_mode: "prompt".to_string(),
            decay_rate: 0.0,
            applicability: Vec::new(),
            distribution: Vec::new(),
            product_yield: None,
        }
    }

    fn channel(mt: i32, products: Vec<ReactionProduct>) -> Arc<Reaction> {
        Arc::new(Reaction {
            cross_section: vec![1.0].into(),
            threshold_idx: 0,
            energy: vec![1.0].into(),
            mt_number: mt,
            q_value: 0.0,
            products,
            scatter_in_cm: false,
            redundant: false,
        })
    }

    /// The ordinary case: the sampled channel carries its own neutrons, so it
    /// also supplies the chi and keys the cache.
    #[test]
    fn a_sampled_channel_supplies_its_own_chi() {
        let mts = [19, 20];
        let rxns = [
            channel(19, vec![product(ParticleType::Neutron)]),
            channel(20, vec![product(ParticleType::Neutron)]),
        ];
        let (mt, products) = resolve_chi_products(Some(&rxns[1]), &mts, &rxns);
        assert_eq!(mt, Some(20), "the sampled channel keys the cache");
        assert_eq!(products.len(), 1);
    }

    /// Issue #425. When the sampled channel carries no neutron product, the chi
    /// comes from a DIFFERENT channel, and the returned MT has to name that one.
    /// Reporting the sampled MT instead would file the fallback channel's
    /// spectrum under the sampled channel's cache slot, which is exactly the
    /// cross-channel mix the per-MT key exists to prevent.
    #[test]
    fn a_photon_only_channel_reports_the_fallback_mt_not_the_sampled_one() {
        let mts = [18, 19];
        let rxns = [
            channel(18, vec![product(ParticleType::Photon)]),
            channel(19, vec![product(ParticleType::Neutron)]),
        ];
        let (mt, products) = resolve_chi_products(Some(&rxns[0]), &mts, &rxns);
        assert_eq!(
            mt,
            Some(19),
            "the chi came from MT 19, so MT 19 must key the cache"
        );
        assert_eq!(products.len(), 1);
    }

    /// The sampled channel must not be re-examined by the fallback scan, or a
    /// photon-only channel could resolve to itself and hand back no products.
    #[test]
    fn the_sampled_channel_is_skipped_by_the_fallback_scan() {
        let mts = [19, 20];
        let rxns = [
            channel(19, vec![product(ParticleType::Photon)]),
            channel(20, vec![product(ParticleType::Neutron)]),
        ];
        let (mt, products) = resolve_chi_products(Some(&rxns[0]), &mts, &rxns);
        assert_eq!(mt, Some(20));
        assert_eq!(products.len(), 1);
    }

    /// An unresolved sample still finds a channel, which is what keeps the
    /// caller's "18 when unresolved" tally default off the chi key.
    #[test]
    fn an_unresolved_sample_still_finds_a_channel() {
        let mts = [19];
        let rxns = [channel(19, vec![product(ParticleType::Neutron)])];
        let (mt, products) = resolve_chi_products(None, &mts, &rxns);
        assert_eq!(mt, Some(19));
        assert_eq!(products.len(), 1);
    }

    /// No neutron product anywhere resolves to nothing, and the caller falls
    /// back to the tally MT with a `FissionChiFlat::None` slot.
    #[test]
    fn no_neutron_product_anywhere_resolves_to_nothing() {
        let mts = [18];
        let rxns = [channel(18, vec![product(ParticleType::Photon)])];
        let (mt, products) = resolve_chi_products(Some(&rxns[0]), &mts, &rxns);
        assert_eq!(mt, None);
        assert!(products.is_empty());
    }
}
