use crate::neutron::interaction::SecondaryNeutrons;
use yamc_nuclide::reaction::Reaction;
use yamc_particle::particle::Particle;

/// Handle general scattering reactions with explicit AWR for CM to LAB conversion
/// This includes reactions like (n,2n), (n,3n), (n,n'alpha), etc.
/// Returns vector of outgoing neutrons based on reaction products or analytical models
pub fn scatter_with_awr<R: rand::Rng>(
    particle: &Particle,
    reaction: &Reaction,
    _nuclide_name: &str,
    awr: f64,
    rng: &mut R,
) -> SecondaryNeutrons {
    // Sampled reactions are expected to have product data
    if reaction.products.is_empty() {
        panic!(
            "Missing product distributions for sampled scattering reaction MT {} at E={:.4e} eV",
            reaction.mt_number, particle.energy
        );
    }

    // Sample from product distributions when available (with AWR for CM to LAB)
    crate::neutron::inelastic::sample_from_products_with_awr(particle, reaction, awr, rng)
}
