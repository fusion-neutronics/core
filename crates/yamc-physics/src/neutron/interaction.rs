// Elastic scattering physics for Monte Carlo transport

use nalgebra::Vector3;
use rand::{Rng, RngExt};
use smallvec::SmallVec;
use yamc_particle::particle::Particle;

/// Inline capacity for outgoing-neutron vectors in fission / (n,xn) sites.
/// Covers all reachable (n,xn) multiplicities (2,3,4) and ~all fission ν samples
/// for fusion fixed-source blankets; rare tails spill to the heap.
pub const SECONDARY_NEUTRON_INLINE: usize = 4;

/// Heap-lift-avoiding vector of secondary neutrons for fission / (n,xn).
pub type SecondaryNeutrons = SmallVec<[Particle; SECONDARY_NEUTRON_INLINE]>;

/// Fast direction rotation using direct scalar math (no Vector3 allocation)
/// Rotates direction u by angle with cos(theta)=mu and azimuthal angle phi
/// Returns the rotated direction as [f64; 3]
#[inline(always)]
pub fn rotate_direction_fast(ux: f64, uy: f64, uz: f64, mu: f64, phi: f64) -> [f64; 3] {
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Use b = sqrt(1 - uz^2) > 1e-10 as threshold for numerical stability
    // This is equivalent to |uz| < sqrt(1 - 1e-20) ≈ 0.99999999999
    let b = (1.0 - uz * uz).max(0.0).sqrt();

    if b > 1e-10 {
        // General case - standard rotation formula
        [
            mu * ux + sin_theta * (ux * uz * cos_phi - uy * sin_phi) / b,
            mu * uy + sin_theta * (uy * uz * cos_phi + ux * sin_phi) / b,
            mu * uz - sin_theta * b * cos_phi,
        ]
    } else {
        // Special case: uz ≈ ±1 - expand about y component instead
        // Degenerate case: uz ~ +/-1, expand about y component instead
        let b_y = (1.0 - uy * uy).max(0.0).sqrt();
        if b_y > 1e-10 {
            [
                mu * ux + sin_theta * (-ux * uy * sin_phi + uz * cos_phi) / b_y,
                mu * uy + sin_theta * b_y * sin_phi,
                mu * uz - sin_theta * (uy * uz * sin_phi + ux * cos_phi) / b_y,
            ]
        } else {
            // Extremely degenerate case - use simple rotation
            let sign = if uz > 0.0 { 1.0 } else { -1.0 };
            [sin_theta * cos_phi, sin_theta * sin_phi, sign * mu]
        }
    }
}

/// Rotate a vector by a random angle mu_cm around a random axis
#[inline]
pub fn rotate_angle(u_cm: Vector3<f64>, mu_cm: f64, rng: &mut impl Rng) -> Vector3<f64> {
    let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
    let result = rotate_direction_fast(u_cm.x, u_cm.y, u_cm.z, mu_cm, phi);
    Vector3::new(result[0], result[1], result[2])
}

/// Rotate a direction vector by angle theta (cos(theta)=mu) around arbitrary axis
/// This rotates u_old to a new direction with cosine mu relative to original
#[inline]
pub fn rotate_direction_3d(u_old: &Vector3<f64>, mu: f64, phi: f64) -> Vector3<f64> {
    let result = rotate_direction_fast(u_old.x, u_old.y, u_old.z, mu, phi);
    Vector3::new(result[0], result[1], result[2])
}

/// Sample target velocity using the CXS (Constant Cross Section) approximation
/// Returns velocity as [f64; 3] in units consistent with neutron velocity (sqrt(eV))
///
/// The CXS method uses rejection sampling to properly weight the target velocity
/// distribution by the relative velocity (collision probability).
///
/// # Resonance Scattering Methods (Not Implemented)
///
/// More accurate methods exist for resonance scattering:
/// - **DBRC** (Doppler Broadening Rejection Correction): More accurate for
///   resonant scatterers by accounting for the energy-dependent cross section
///   in the target velocity sampling.
/// - **RVS** (Relative Velocity Sampling): An alternative approach that samples
///   the relative velocity distribution directly.
///
/// These methods are relevant for resonant scatterers (e.g., U-238 at low energies)
/// where the CXS approximation may be insufficient. For non-resonant materials
/// or energies away from resonances, CXS provides adequate accuracy.
#[inline]
pub fn sample_cxs_target_velocity(
    awr: f64,
    neutron_energy: f64,
    neutron_direction: &[f64; 3],
    temperature_k: f64,
    rng: &mut impl Rng,
) -> Vector3<f64> {
    let [vx, vy, vz] = sample_cxs_target_velocity_array(
        awr,
        neutron_energy,
        neutron_direction,
        temperature_k,
        rng,
    );
    Vector3::new(vx, vy, vz)
}

/// Sample target velocity - returns [f64; 3] array (faster, no Vector3 allocation)
#[inline]
pub fn sample_cxs_target_velocity_array(
    awr: f64,
    neutron_energy: f64,
    neutron_direction: &[f64; 3],
    temperature_k: f64,
    rng: &mut impl Rng,
) -> [f64; 3] {
    // Boltzmann constant in eV/K
    const K_B: f64 = 8.617333e-5;
    const PI: f64 = std::f64::consts::PI;
    const SQRT_PI: f64 = 1.7724538509055159; // sqrt(PI) as const

    let k_t = K_B * temperature_k;

    // Reduced neutron velocity: beta_vn = sqrt(awr * E / kT)
    let beta_vn = (awr * neutron_energy / k_t).sqrt();

    // Probability weighting factor
    let alpha = 1.0 / (1.0 + SQRT_PI * beta_vn * 0.5);

    let beta_vt_sq: f64;
    let mu: f64;

    loop {
        // Sample two random numbers
        let r1: f64 = rng.random::<f64>();
        let r2: f64 = rng.random::<f64>();

        let beta_vt_sq_candidate = if rng.random::<f64>() < alpha {
            // With probability alpha, sample from p(y) = y*e^(-y)
            // Using -log(r1 * r2) = -log(r1) - log(r2) but single log is faster
            -(r1 * r2).ln()
        } else {
            // With probability 1-alpha, sample from p(y) = y^2 * e^(-y^2)
            // Need separate logs here because c^2 only multiplies r2's log
            let c = (PI * 0.5 * rng.random::<f64>()).cos();
            -r1.ln() - r2.ln() * c * c
        };

        // Determine beta * vt
        let beta_vt = beta_vt_sq_candidate.sqrt();

        // Sample cosine of angle between neutron and target velocity
        let mu_candidate = 2.0 * rng.random::<f64>() - 1.0;

        // Determine rejection probability based on relative velocity
        // accept_prob = |v_rel| / |v_rel_max| = sqrt(vn^2 + vt^2 - 2*vn*vt*mu) / (vn + vt)
        let accept_prob = (beta_vn * beta_vn + beta_vt_sq_candidate
            - 2.0 * beta_vn * beta_vt * mu_candidate)
            .sqrt()
            / (beta_vn + beta_vt);

        // Perform rejection sampling
        if rng.random::<f64>() < accept_prob {
            beta_vt_sq = beta_vt_sq_candidate;
            mu = mu_candidate;
            break;
        }
    }

    // Determine speed of target nucleus
    let vt = (beta_vt_sq * k_t / awr).sqrt();

    // Rotate neutron direction by angle mu around random azimuthal angle, scale by vt
    let phi = 2.0 * PI * rng.random::<f64>();
    let [dx, dy, dz] = rotate_direction_fast(
        neutron_direction[0],
        neutron_direction[1],
        neutron_direction[2],
        mu,
        phi,
    );
    [vt * dx, vt * dy, vt * dz]
}

/// Perform elastic scattering for a neutron particle with thermal motion (isotropic in CM)
/// This is used when no angular distribution data is available
#[inline]
pub fn elastic_scatter(particle: &mut Particle, awr: f64, temperature_k: f64, rng: &mut impl Rng) {
    // Neutron velocity in LAB (velocity = sqrt(energy) in our units where neutron mass = 1)
    let vel = particle.energy.sqrt();
    let [dx, dy, dz] = particle.direction;
    let v_n = [dx * vel, dy * vel, dz * vel];

    // Sample target velocity using CXS approximation - use array version
    let v_t = sample_cxs_target_velocity_array(
        awr,
        particle.energy,
        &particle.direction,
        temperature_k,
        rng,
    );

    // Center-of-mass velocity
    let inv_awr_plus_1 = 1.0 / (awr + 1.0);
    let v_cm = [
        (v_n[0] + awr * v_t[0]) * inv_awr_plus_1,
        (v_n[1] + awr * v_t[1]) * inv_awr_plus_1,
        (v_n[2] + awr * v_t[2]) * inv_awr_plus_1,
    ];

    // Transform to CM frame
    let v_n_cm = [v_n[0] - v_cm[0], v_n[1] - v_cm[1], v_n[2] - v_cm[2]];
    let vel_cm_sq = v_n_cm[0] * v_n_cm[0] + v_n_cm[1] * v_n_cm[1] + v_n_cm[2] * v_n_cm[2];

    // Handle case where neutron and target have same velocity
    if vel_cm_sq < 1e-20 {
        // Just scatter isotropically in lab frame
        let mu = 2.0 * rng.random::<f64>() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
        let sin_theta = (1.0 - mu * mu).sqrt();
        particle.direction = [sin_theta * phi.cos(), sin_theta * phi.sin(), mu];
        return;
    }

    let vel_cm = vel_cm_sq.sqrt();

    // Sample scattering angle in CM (isotropic for elastic)
    let mu_cm = 2.0 * rng.random::<f64>() - 1.0;
    let phi_cm = 2.0 * std::f64::consts::PI * rng.random::<f64>();

    // Direction in CM (normalized)
    let inv_vel_cm = 1.0 / vel_cm;
    let u_cm = [
        v_n_cm[0] * inv_vel_cm,
        v_n_cm[1] * inv_vel_cm,
        v_n_cm[2] * inv_vel_cm,
    ];

    // Rotate and scale to get new CM velocity
    let rotated = rotate_direction_fast(u_cm[0], u_cm[1], u_cm[2], mu_cm, phi_cm);
    let v_n_cm_new = [
        rotated[0] * vel_cm,
        rotated[1] * vel_cm,
        rotated[2] * vel_cm,
    ];

    // Transform back to LAB frame
    let v_n_lab = [
        v_n_cm_new[0] + v_cm[0],
        v_n_cm_new[1] + v_cm[1],
        v_n_cm_new[2] + v_cm[2],
    ];

    // Update particle energy and direction
    let e_out = v_n_lab[0] * v_n_lab[0] + v_n_lab[1] * v_n_lab[1] + v_n_lab[2] * v_n_lab[2];
    particle.energy = e_out;
    let vel_lab = e_out.sqrt();

    if vel_lab > 1e-10 {
        let inv_vel_lab = 1.0 / vel_lab;
        particle.direction = [
            v_n_lab[0] * inv_vel_lab,
            v_n_lab[1] * inv_vel_lab,
            v_n_lab[2] * inv_vel_lab,
        ];
    }
}

/// DBRC (Doppler Broadening Rejection Correction) elastic scattering
/// Uses target-at-rest approximation with tabulated angular distribution
/// This is appropriate for the resonance region (typically ~1 eV to 210 keV)
#[inline]
pub fn dbrc_elastic_scatter(
    particle: &mut Particle,
    awr: f64,
    // temperature_k: f64,
    mu_cm: f64, // Scattering cosine from tabulated distribution
    rng: &mut impl Rng,
) {
    // For DBRC, use target-at-rest kinematics with the tabulated angular distribution
    // The thermal motion is already accounted for in the cross section sampling via relative energy

    // Target-at-rest: transform using simple kinematics
    // E_out = E_in * ((A^2 + 1 + 2*A*mu) / (A+1)^2) for elastic scattering

    let e_in = particle.energy;
    let a = awr;

    // Scattering cosine in lab frame (from CM using target-at-rest kinematics)
    // mu_lab = (1 + A*mu_cm) / sqrt(A^2 + 2*A*mu_cm + 1)
    let mu_lab = (1.0 + a * mu_cm) / (a * a + 2.0 * a * mu_cm + 1.0).sqrt();

    // Outgoing energy (target-at-rest kinematics)
    let e_out = e_in * (a * a + 1.0 + 2.0 * a * mu_cm) / ((a + 1.0) * (a + 1.0));

    // Update particle energy
    particle.energy = e_out;

    // Rotate direction using fast scalar math
    let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
    particle.direction = rotate_direction_fast(
        particle.direction[0],
        particle.direction[1],
        particle.direction[2],
        mu_lab,
        phi,
    );
}

// =====================
//   FISSION PHYSICS
// =====================

/// Maxwell (ENDF Law 7) and Watt (Law 11) fission-spectrum samplers.
///
/// Re-exported from [`yamc_nuclide::sampling`] -- the single source of truth
/// -- so `yamc_physics::neutron::interaction::sample_maxwell_spectrum` /
/// `sample_watt_spectrum_params` keep working for existing callers without a
/// byte-identical second copy.
pub use yamc_nuclide::sampling::{sample_maxwell_spectrum, sample_watt_spectrum_params};

/// Sample fission neutrons and return them as a vector of particles
///
/// # Arguments
/// * `particle` - The incident neutron particle
/// * `nu_bar` - Average number of neutrons per fission at this energy
/// * `fission_products` - Optional fission neutron products with energy distributions
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of fission neutron particles (may be empty in rare cases)
/// `chi_flat` is the prompt fission spectrum pre-flattened for the shared
/// GPU/CPU flat samplers (issue #111 fission sub-step). When it carries data,
/// the outgoing energy is drawn through `sample_fission_chi_flat` and the
/// isotropic emission angle through the per-particle PCG `pcg` stream, so the
/// CPU fission chi and the GPU kernel share one sampling implementation.
/// `FissionChiFlat::None` (an unmigrated/degenerate chi kind) falls back to the
/// legacy `prompt_product.sample` on `rng`, byte-identical to the prior path.
///
/// On the shared path the PCG draws come in the GPU kernel's fission order
/// (issue #111), which is what lets a fission collision stay bit-identical
/// between the two backends:
///   1. the CONTINUING progeny's chi -- before the multiplicity draw,
///   2. the stochastic-rounding uniform that turns `nu_bar` into `N`,
///   3. per banked progeny (batch entries 1..N): chi, isotropic mu, azimuth,
///   4. the continuing progeny's azimuth, LAST -- the kernel draws the lab
///      azimuth once per collision, after the per-reaction branch, and the
///      fission branch's banked progeny are sampled before it.
///
/// `mu_xi` is the isotropic cosine's uniform for the continuing progeny: the
/// angle seed the caller already drew at the reaction split (the kernel's
/// `xi3`, reused there rather than drawn again).
#[allow(clippy::too_many_arguments)]
pub fn sample_fission_neutrons<R: Rng>(
    particle: &yamc_particle::particle::Particle,
    nu_bar: f64,
    fission_products: Option<&[&yamc_nuclide::reaction_product::ReactionProduct]>,
    chi_flat: &yamc_nuclide::reaction_product::FissionChiFlat,
    delayed: Option<(f64, &yamc_nuclide::reaction_product::FissionChiFlat)>,
    mu_xi: f64,
    pcg: &mut u64,
    rng: &mut R,
) -> SecondaryNeutrons {
    use yamc_rng::next_xi;

    let e_in = particle.energy;
    let chi_usable = !matches!(
        chi_flat,
        yamc_nuclide::reaction_product::FissionChiFlat::None
    );

    // Continuing progeny's outgoing energy, drawn BEFORE the multiplicity so
    // the kernel's `sample_fission_chi` lands on the same stream position.
    let e_continuing = if chi_usable {
        fission_progeny_energy(chi_flat, delayed, e_in, pcg)
    } else {
        e_in
    };

    // Stochastic rounding to determine actual number of neutrons (on the PCG
    // stream so the fission multiplicity is shared with the GPU schedule).
    let n_neutrons = {
        let base = nu_bar.floor() as usize;
        let frac = nu_bar - nu_bar.floor();
        if next_xi(pcg) < frac {
            base + 1
        } else {
            base
        }
    };

    if n_neutrons == 0 {
        return SecondaryNeutrons::new();
    }

    // Check if we have product distributions to sample from
    // Product distributions should always exist for fission
    let products = match fission_products {
        Some(prods) if !prods.is_empty() => prods,
        _ => {
            // No product data available - this should not happen with proper nuclear data
            // Log a warning and skip fission neutron production
            // Valid product distributions are required for fission
            eprintln!(
                "Warning: No fission product distribution available at E={:.4e} eV. \
                 Check that nuclear data includes fission product spectra (maxwell/watt/continuous). \
                 Fission neutrons will not be produced for this event.",
                particle.energy
            );
            return SecondaryNeutrons::new();
        }
    };

    let mut neutrons = SecondaryNeutrons::with_capacity(n_neutrons);

    if chi_usable {
        // Banked progeny (batch entries 1..N) come first on the stream: each
        // gets an independent chi energy and an independent isotropic direction
        // about the INCIDENT direction, exactly as the kernel's bank loop does.
        for _ in 1..n_neutrons {
            let mut new_particle = particle.clone();
            new_particle.energy = fission_progeny_energy(chi_flat, delayed, e_in, pcg).max(1e-11);
            let mu = 1.0 - 2.0 * next_xi(pcg);
            let phi = std::f64::consts::TAU * next_xi(pcg);
            new_particle.direction = rotate_direction_fast(
                new_particle.direction[0],
                new_particle.direction[1],
                new_particle.direction[2],
                mu,
                phi,
            );
            neutrons.push(new_particle);
        }

        // Continuing progeny: energy from the chi drawn before the multiplicity,
        // isotropic cosine from the caller's reaction-split seed, and the
        // azimuth LAST (the kernel's shared post-collision lab azimuth). It goes
        // in at index 0 because the analog caller keeps entry 0 as the walk that
        // continues.
        let mut first = particle.clone();
        first.energy = e_continuing.max(1e-11);
        let mu = 1.0 - 2.0 * mu_xi;
        let phi = std::f64::consts::TAU * next_xi(pcg);
        first.direction = rotate_direction_fast(
            first.direction[0],
            first.direction[1],
            first.direction[2],
            mu,
            phi,
        );
        neutrons.insert(0, first);
    } else {
        // Legacy path for chi kinds not yet flattened (e.g. equiprobable
        // Tabulated): byte-identical to the pre-#111 fission sampling, all draws
        // on `rng`. No GPU counterpart to align with, so every progeny -- the
        // continuing one included -- takes its energy and angle from the
        // product's own distribution in emission order.
        for _ in 0..n_neutrons {
            let mut new_particle = particle.clone();
            let prompt_product = products[0];
            let (e_fission, mu) = prompt_product.sample(e_in, rng);
            new_particle.energy = e_fission.max(1e-11);
            crate::neutron::inelastic::rotate_direction(&mut new_particle.direction, mu, rng);
            neutrons.push(new_particle);
        }
    }

    neutrons
}

/// One fission progeny's outgoing energy: pick the emitting spectrum, then sample
/// it (issue #364).
///
/// `fission_nu` is nu_TOTAL, so the batch already has the right COUNT; what makes
/// the source spectrum right is that a `beta(E)` fraction of those neutrons are
/// born from the DELAYED spectrum, which is far softer (~0.5 MeV mean against
/// ~2.0 MeV prompt). Sampling every progeny from the prompt spectrum made yamc's
/// fission source too hard and its slowing-down flux low against OpenMC, by an
/// amount that tracked `beta` across the actinides (U238 -1.5%, U235 -0.65%,
/// Pu239 -0.21%; issue #364).
///
/// The choice costs ONE uniform, drawn only when the evaluation actually carries
/// delayed data (`beta > 0`). That condition is a property of the nuclide's data,
/// not of any sampled state, so both backends evaluate it identically and an
/// evaluation with no delayed groups keeps its previous draw schedule exactly.
fn fission_progeny_energy(
    prompt: &yamc_nuclide::reaction_product::FissionChiFlat,
    delayed: Option<(f64, &yamc_nuclide::reaction_product::FissionChiFlat)>,
    e_in: f64,
    pcg: &mut u64,
) -> f64 {
    use yamc_rng::next_xi;
    if let Some((beta, delayed_chi)) = delayed {
        if beta > 0.0 && next_xi(pcg) < beta {
            return fission_chi_or_incident(delayed_chi, e_in, pcg);
        }
    }
    fission_chi_or_incident(prompt, e_in, pcg)
}

/// One prompt-fission outgoing energy from the flat chi on the shared PCG
/// stream. Retries the (near-impossible) rejection-cap exhaustion and falls
/// back to the incident energy if every attempt fails.
fn fission_chi_or_incident(
    chi_flat: &yamc_nuclide::reaction_product::FissionChiFlat,
    e_in: f64,
    pcg: &mut u64,
) -> f64 {
    for _ in 0..64 {
        if let Some(e) = crate::gpu::flat::sample_fission_chi_flat(chi_flat, e_in, pcg) {
            return e;
        }
    }
    e_in
}

// =====================
//        TESTS
// =====================
