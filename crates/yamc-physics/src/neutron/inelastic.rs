use crate::neutron::interaction::SecondaryNeutrons;
use rand::RngExt;
use smallvec::smallvec;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::ParticleType;
use yamc_particle::particle::Particle;

/// Handle inelastic scattering reactions
/// Returns vector of outgoing neutrons based on reaction products or analytical models
pub fn inelastic_scatter<R: rand::Rng>(
    particle: &Particle,
    reaction: &Reaction,
    awr: f64,
    rng: &mut R,
) -> SecondaryNeutrons {
    // Sampled reactions are expected to have product data
    if reaction.products.is_empty() {
        panic!(
            "Missing product distributions for sampled inelastic reaction MT {} at E={:.4e} eV",
            reaction.mt_number, particle.energy
        );
    }

    // Sample from product distributions when available
    sample_from_products_with_awr(particle, reaction, awr, rng)
}

/// Sample outgoing particles with explicit AWR for CM to LAB conversion
/// 1. Sample energy/angle from first neutron product only
/// 2. Apply CM-to-LAB transformation if needed
/// 3. If yield is integral, create (yield-1) secondaries with SAME energy/direction
/// 4. If yield is non-integral, use stochastic rounding
pub fn sample_from_products_with_awr<R: rand::Rng>(
    particle: &Particle,
    reaction: &Reaction,
    awr: f64,
    rng: &mut R,
) -> SecondaryNeutrons {
    let e_in = particle.energy;

    // Find the FIRST neutron product only
    let neutron_product = reaction
        .products
        .iter()
        .find(|product| product.is_particle_type(&ParticleType::Neutron));

    let Some(neutron_product) = neutron_product else {
        // No neutron products - particle is absorbed (e.g., (n,gamma), (n,p), etc.)
        return SecondaryNeutrons::new();
    };

    // Sample outgoing energy and scattering cosine from product distribution ONCE
    let (mut e_out, mut mu) = neutron_product.sample(e_in, rng);

    // If scattering is in center-of-mass frame, convert to LAB frame
    if reaction.scatter_in_cm {
        let e_cm = e_out;
        let a = awr;

        // Determine outgoing energy in lab frame
        // E = E_cm + (E_in + 2*mu*(A+1)*sqrt(E_in*E_cm)) / ((A+1)^2)
        e_out =
            e_cm + (e_in + 2.0 * mu * (a + 1.0) * (e_in * e_cm).sqrt()) / ((a + 1.0) * (a + 1.0));

        // Determine outgoing angle in lab frame
        // mu = mu * sqrt(E_cm/E) + 1/(A+1) * sqrt(E_in/E)
        mu = mu * (e_cm / e_out).sqrt() + 1.0 / (a + 1.0) * (e_in / e_out).sqrt();

        // Clamp mu to [-1, 1] due to floating point roundoff
        if mu.abs() > 1.0 {
            mu = 1.0_f64.copysign(mu);
        }
    }

    // Create the primary outgoing neutron
    let mut primary_particle = particle.clone();
    primary_particle.energy = e_out;
    rotate_direction(&mut primary_particle.direction, mu, rng);

    // Evaluate yield at incident energy
    let yield_val = neutron_product
        .product_yield
        .as_ref()
        .map(|y| y.evaluate(e_in))
        .unwrap_or(1.0);

    // Yield handling:
    // - If yield is zero: treat as absorption (no neutrons produced)
    // - If yield is integral (e.g., 2.0 for (n,2n)): create exactly (yield-1) secondary particles
    // - If yield is non-integral: modify particle weight by yield (no extra particles)
    if yield_val.abs() < 1e-10 {
        return SecondaryNeutrons::new();
    }

    let is_integral = (yield_val - yield_val.floor()).abs() < 1e-10;

    if is_integral {
        // Integral yield - create exactly yield particles, all with SAME energy/direction
        let n_total = yield_val.round() as usize;
        let mut outgoing_neutrons = SecondaryNeutrons::with_capacity(n_total);

        // First particle is the primary
        outgoing_neutrons.push(primary_particle.clone());

        // Create (yield-1) additional particles with SAME energy and direction
        for _ in 1..n_total {
            outgoing_neutrons.push(primary_particle.clone());
        }

        outgoing_neutrons
    } else {
        // Non-integral yield - modify particle weight
        // This is more accurate than stochastic rounding for variance reduction
        primary_particle.weight *= yield_val;
        smallvec![primary_particle]
    }
}
/// Rotate particle direction by scattering angle with cosine mu
/// This implements isotropic azimuthal angle sampling
/// Applies the standard Rodrigues rotation formula for particle direction change
pub fn rotate_direction<R: rand::Rng>(
    direction: &mut [f64; 3],
    mu: f64, // cosine of scattering angle
    rng: &mut R,
) {
    // Sample azimuthal angle uniformly
    let phi = rng.random_range(0.0..2.0 * std::f64::consts::PI);

    // Use the shared implementation
    let result = crate::neutron::interaction::rotate_direction_fast(
        direction[0],
        direction[1],
        direction[2],
        mu,
        phi,
    );

    *direction = result;

    // Normalize to ensure unit vector (numerical precision)
    let norm = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
    if norm > 1e-10 && (norm - 1.0).abs() > 1e-14 {
        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;
    }
}

/// Sample a DISCRETE inelastic level (MT 51-90) with closed-form-Q kinematics on
/// the shared per-particle PCG `state` (issue #111 sub-step 3).
///
/// Bit-identical to the GPU twin: the CM outgoing energy uses the same
/// closed-form Q formula (yamc-gpu shared.rs `e_cm = mass_ratio*(E_in - threshold)`,
/// `threshold = (A+1)/A*|Q|`, `mass_ratio = (A/(A+1))^2`); the CM scattering
/// cosine is drawn from the SAME `elastic_mu_cm` flat sampler the GPU reuses for
/// the inelastic angle (with `xi3` as the isotropic fallback); the lab transform
/// is the shared `cm_to_lab`. The level energy itself consumes no random number,
/// matching the GPU draw schedule (only `elastic_mu_cm`'s internal draws fire
/// here, after the caller's `xi3`).
///
/// Returns `(e_lab, mu_lab)`, or `None` when the lab energy is non-positive (the
/// caller then leaves the particle state unchanged, as the GPU does).
#[allow(clippy::too_many_arguments)]
pub fn sample_level_inelastic(
    e_in: f64,
    awr: f64,
    q_value: f64,
    scatter_in_cm: bool,
    xi3: f64,
    energy_grid: &[f64],
    n_mu_per_e: &[u32],
    interp_per_e: &[u32],
    mu_table: &[f64],
    cdf_table: &[f64],
    pdf_table: &[f64],
    mu_offset: &[u32],
    state: &mut u64,
) -> Option<(f64, f64)> {
    // Closed-form-Q CM outgoing energy (verbatim from yamc-gpu shared.rs).
    let abs_q = q_value.abs();
    let threshold = (awr + 1.0) / awr * abs_q;
    let mass_ratio = (awr / (awr + 1.0)).powi(2);
    let e_cm = mass_ratio * (e_in - threshold);

    // CM scattering cosine via the shared sampler (the GPU's inelastic angle
    // path uses this exact sampler); `xi3` seeds the isotropic fallback.
    let mu_cm = crate::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm(
        e_in,
        xi3,
        energy_grid,
        n_mu_per_e,
        interp_per_e,
        mu_table,
        cdf_table,
        pdf_table,
        mu_offset,
        state,
    );

    if scatter_in_cm {
        // cm_to_lab returns (mu_lab, e_lab); expose (e_lab, mu_lab). None when
        // e_lab <= 0 -> caller leaves the particle unchanged (GPU behaviour).
        crate::gpu::flat::cm_to_lab::cm_to_lab(e_in, e_cm, mu_cm, awr)
            .map(|(mu_lab, e_lab)| (e_lab, mu_lab))
    } else {
        // Rare ENDF lab-frame discrete level: e_cm/mu_cm are already lab.
        Some((e_cm, mu_cm))
    }
}

#[cfg(test)]
mod level_inelastic_tests {
    use super::sample_level_inelastic;

    /// Pin the closed-form-Q + cm_to_lab math against a hand computation for one
    /// (E_in, Q, AWR) tuple with an empty angular table (isotropic via xi3).
    #[test]
    fn level_inelastic_closed_form_pins() {
        let e_in: f64 = 5.0e6;
        let awr: f64 = 11.8969; // ~C12
        let q: f64 = -4.4389e6; // MT 51-like level Q (eV)
        let xi3: f64 = 0.25; // isotropic fallback -> mu_cm = 1 - 2*xi3 = 0.5
        let mut state: u64 = 12345;
        // Hand-computed closed form:
        let abs_q = q.abs();
        let threshold = (awr + 1.0) / awr * abs_q;
        let mass_ratio = (awr / (awr + 1.0)).powi(2);
        let e_cm = mass_ratio * (e_in - threshold);
        let mu_cm = 1.0 - 2.0 * xi3;
        let one_plus_a = awr + 1.0;
        let e_lab_exp = e_cm
            + (e_in + 2.0 * mu_cm * one_plus_a * (e_in * e_cm).sqrt()) / (one_plus_a * one_plus_a);
        let mu_lab_exp = (mu_cm * (e_cm / e_lab_exp).sqrt()
            + (1.0 / one_plus_a) * (e_in / e_lab_exp).sqrt())
        .clamp(-1.0, 1.0);
        // Empty angular table forces the xi3 isotropic fallback (no state draws).
        let (e_lab, mu_lab) = sample_level_inelastic(
            e_in,
            awr,
            q,
            true,
            xi3,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut state,
        )
        .expect("e_lab > 0");
        assert!(
            (e_lab - e_lab_exp).abs() <= 1e-9 * e_lab_exp,
            "e_lab {e_lab:e} vs {e_lab_exp:e}"
        );
        assert!(
            (mu_lab - mu_lab_exp).abs() <= 1e-12,
            "mu_lab {mu_lab} vs {mu_lab_exp}"
        );
    }
}
