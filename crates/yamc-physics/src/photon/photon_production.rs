use crate::neutron::interaction::rotate_direction_fast;
/// Secondary photon production from neutron collisions.
///
/// After a neutron collision, photons may be emitted as secondary particles.
/// This module provides functions to sample the number and kinematics of
/// those photons.
use crate::util::bank::ParticleBank;
use rand::RngExt;
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_particle::particle::{Particle, ParticleType};

/// Sample secondary photons from a neutron collision and bank them.
///
/// # Arguments
/// * `particle` - The neutron that just collided (used for position, weight, cell index)
/// * `incoming_energy` - Neutron energy BEFORE the collision (must be pre-scatter value)
/// * `incoming_direction` - Neutron direction BEFORE the collision (must be pre-scatter value)
/// * `nuclide` - The nuclide that was hit
/// * `fast_grid` - Pre-computed fast XS grid for grid-consistent XS lookup
/// * `i_grid` - Energy grid index from lookup_grid_index
/// * `interp_factor` - Interpolation factor from lookup_grid_index
/// * `micro_xs_total` - Microscopic total XS at the collision energy
/// * `photon_prod_xs` - Microscopic photon production XS at the collision energy
/// * `bank` - Particle bank to push photon secondaries into
/// * `rng` - Random number generator
#[allow(clippy::too_many_arguments)]
pub fn sample_secondary_photons<R: rand::Rng>(
    particle: &Particle,
    incoming_energy: f64,
    incoming_direction: [f64; 3],
    _nuclide: &Nuclide,
    fast_grid: &FastXSGrid,
    i_grid: usize,
    interp_factor: f64,
    micro_xs_total: f64,
    photon_prod_xs: f64,
    bank: &mut ParticleBank,
    rng: &mut R,
) {
    if photon_prod_xs <= 0.0 || micro_xs_total <= 0.0 {
        return;
    }

    // Average number of photons per collision (using grid-interpolated photon_prod,
    // using grid-interpolated photon_prod)
    let y_t = photon_prod_xs / micro_xs_total;

    // Sample integer number of photons from the fractional yield
    let mut n_photons = y_t as usize; // floor
    if rng.random::<f64>() < (y_t - n_photons as f64) {
        n_photons += 1;
    }

    if n_photons == 0 {
        return;
    }

    let energy_in = incoming_energy;

    for _ in 0..n_photons {
        // Sample which reaction and product this photon comes from
        let (reaction, product_idx) =
            match sample_photon_product(fast_grid, i_grid, interp_factor, energy_in, rng) {
                Some(result) => result,
                None => continue,
            };

        // Get the photon product at the sampled index
        let mut photon_product_idx = 0;
        let mut product = None;
        for p in &reaction.products {
            if p.is_particle_type(&ParticleType::Photon) {
                if photon_product_idx == product_idx {
                    product = Some(p);
                    break;
                }
                photon_product_idx += 1;
            }
        }

        let product = match product {
            Some(p) => p,
            None => continue,
        };

        // Check if the product has a valid distribution for sampling.
        // Some photon products may have UncorrelatedAngleEnergy with energy=None
        // (from unrecognized distribution types during HDF5 loading). Skip those.
        let has_valid_distribution = if product.distribution.is_empty() {
            false
        } else {
            product.distribution.iter().all(|d| {
                use yamc_nuclide::reaction_product::AngleEnergyDistribution;
                !matches!(
                    d,
                    AngleEnergyDistribution::UncorrelatedAngleEnergy { energy: None, .. }
                )
            })
        };

        if !has_valid_distribution {
            continue;
        }

        // Sample outgoing energy and cosine of scattering angle
        let (e_out, mu) = product.sample(energy_in, rng);

        if e_out <= 0.0 {
            continue;
        }

        // Sample azimuthal angle uniformly
        let phi = rng.random::<f64>() * std::f64::consts::TAU;

        // Rotate direction relative to incident neutron direction
        // (must use pre-scatter direction, not post-scatter)
        let direction = rotate_direction_fast(
            incoming_direction[0],
            incoming_direction[1],
            incoming_direction[2],
            mu,
            phi,
        );

        // Create the photon particle
        let mut photon = Particle::new(particle.position, direction, e_out);
        photon.particle_type = ParticleType::Photon;
        photon.weight = particle.weight;
        photon.current_cell_index = particle.current_cell_index;

        bank.bank_secondary(photon);
    }
}

/// Sample which reaction and product to emit the photon from.
///
/// Uses grid-consistent XS values: per-reaction XS is interpolated on the same
/// unified energy grid as photon_prod, ensuring consistent relative weights
/// between reactions.
///
/// Uses a two-pass approach for internally consistent normalization:
/// first computes total_prob from grid-consistent XS * yield, then samples
/// using that same total as the cutoff normalization.
///
/// Returns `Some((&Reaction, photon_product_index))` or `None`.
fn sample_photon_product<'a, R: rand::Rng>(
    fast_grid: &'a FastXSGrid,
    i_grid: usize,
    interp_factor: f64,
    energy: f64,
    rng: &mut R,
) -> Option<(&'a yamc_nuclide::reaction::Reaction, usize)> {
    // Helper to get delayed photon scaling factor for fission reactions.
    // Scale fission photon production by delayed photon factor.
    let scaling_factor = |mt: i32| -> f64 {
        if yamc_nuclide::nuclide::is_fission_mt(mt) && !fast_grid.delayed_photon_scaling.is_empty()
        {
            if i_grid + 1 < fast_grid.delayed_photon_scaling.len() {
                fast_grid.delayed_photon_scaling[i_grid]
                    + interp_factor
                        * (fast_grid.delayed_photon_scaling[i_grid + 1]
                            - fast_grid.delayed_photon_scaling[i_grid])
            } else if !fast_grid.delayed_photon_scaling.is_empty() {
                fast_grid.delayed_photon_scaling
                    [i_grid.min(fast_grid.delayed_photon_scaling.len() - 1)]
            } else {
                1.0
            }
        } else {
            1.0
        }
    };

    // First pass: compute total photon production probability using grid-consistent XS.
    let mut total_prob = 0.0;
    for (j, &mt) in fast_grid.photon_rxn_mt_numbers.iter().enumerate() {
        let reaction = &fast_grid.photon_rxn_reactions[j];
        let rxn_xs = fast_grid.photon_rxn_xs_interp(i_grid, interp_factor, j);
        if rxn_xs <= 0.0 {
            continue;
        }
        let f = scaling_factor(mt);
        for product in &reaction.products {
            if product.is_particle_type(&ParticleType::Photon) {
                let y = product
                    .product_yield
                    .as_ref()
                    .map(|yld| yld.evaluate(energy))
                    .unwrap_or(1.0);
                total_prob += f * rxn_xs * y;
            }
        }
    }

    if total_prob <= 0.0 {
        return None;
    }

    // Second pass: sample using the consistently-computed total
    let cutoff = rng.random::<f64>() * total_prob;
    let mut prob = 0.0;
    let mut last_product: Option<(&yamc_nuclide::reaction::Reaction, usize)> = None;

    for (j, &mt) in fast_grid.photon_rxn_mt_numbers.iter().enumerate() {
        let reaction = &fast_grid.photon_rxn_reactions[j];
        let rxn_xs = fast_grid.photon_rxn_xs_interp(i_grid, interp_factor, j);
        if rxn_xs <= 0.0 {
            continue;
        }

        let f = scaling_factor(mt);
        let mut photon_product_idx = 0;
        for product in &reaction.products {
            if product.is_particle_type(&ParticleType::Photon) {
                let y = product
                    .product_yield
                    .as_ref()
                    .map(|yld| yld.evaluate(energy))
                    .unwrap_or(1.0);
                prob += f * rxn_xs * y;

                // Set last_product BEFORE cutoff check
                last_product = Some((reaction.as_ref(), photon_product_idx));
                if prob > cutoff {
                    return last_product;
                }
                photon_product_idx += 1;
            }
        }
    }

    // Fallback: return last photon product (handles floating-point edge cases)
    last_product
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Test arrow datasets live in `crates/yamc/tests/`. Resolve relative
    /// to this crate's manifest dir so `cargo test` works regardless of
    /// which crate's runner is invoked.
    fn td(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("yamc")
            .join("tests")
            .join(name)
    }

    /// Helper: load Fe56 nuclide from test data
    fn load_fe56() -> Nuclide {
        yamc_nuclide::nuclide_loader::load_nuclide(
            td("Fe56.arrow"),
            &yamc_nuclide::LoadScope::full(),
        )
        .expect("Failed to load Fe56.arrow")
    }

    #[test]
    fn test_photon_prod_in_fast_xs_grid_fe56() {
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }
        let nuclide = load_fe56();

        // Fe56 should have photon-producing reactions (n,gamma at minimum)
        assert!(
            !nuclide.fast_xs.is_empty(),
            "Fe56 should have fast_xs grids"
        );

        let fast_grid = &nuclide.fast_xs[0];
        assert!(
            !fast_grid.photon_prod.is_empty(),
            "Fe56 photon_prod should be populated"
        );

        // At thermal energies, n,gamma dominates so photon_prod should be large
        let (i_grid, f) = fast_grid.lookup_grid_index(0.0253); // 25.3 meV thermal
        let pp = fast_grid.lookup_photon_prod(i_grid, f);
        assert!(
            pp > 0.0,
            "Fe56 photon_prod should be > 0 at thermal energy, got {pp}"
        );

        // At 14 MeV, photon production should also exist
        let (i_grid, f) = fast_grid.lookup_grid_index(14.0e6);
        let pp_14mev = fast_grid.lookup_photon_prod(i_grid, f);
        assert!(
            pp_14mev > 0.0,
            "Fe56 photon_prod should be > 0 at 14 MeV, got {pp_14mev}"
        );
    }

    #[test]
    fn test_sample_secondary_photons_fe56_14mev() {
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }
        let nuclide = load_fe56();

        // Create a 14 MeV neutron
        let mut neutron = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 14.0e6);
        neutron.particle_type = ParticleType::Neutron;
        neutron.weight = 1.0;

        let mut rng = StdRng::seed_from_u64(42);
        let mut bank = ParticleBank::new();

        // Get the photon_prod and total XS
        let fast_grid = &nuclide.fast_xs[0];
        let (i_grid, f) = fast_grid.lookup_grid_index(14.0e6);
        let photon_prod_xs = fast_grid.lookup_photon_prod(i_grid, f);
        let (total_xs, _, _, _) = fast_grid.lookup(14.0e6);

        assert!(photon_prod_xs > 0.0, "Need photon_prod > 0 for this test");
        assert!(total_xs > 0.0, "Need total_xs > 0 for this test");

        // Run many trials
        let n_trials = 1000;
        let mut total_photons = 0;
        for _ in 0..n_trials {
            bank.clear();
            sample_secondary_photons(
                &neutron,
                neutron.energy,
                neutron.direction,
                &nuclide,
                fast_grid,
                i_grid,
                f,
                total_xs,
                photon_prod_xs,
                &mut bank,
                &mut rng,
            );
            total_photons += bank.len();
        }

        // Should produce some photons
        assert!(
            total_photons > 0,
            "Should have produced some secondary photons from Fe56 + 14 MeV neutron"
        );

        // Average should be approximately photon_prod / total
        let avg = total_photons as f64 / n_trials as f64;
        let expected = photon_prod_xs / total_xs;
        assert!(
            (avg - expected).abs() < 0.5,
            "Average photons per collision ({avg:.3}) should be ~{expected:.3}"
        );
    }

    #[test]
    fn test_secondary_photons_have_correct_properties() {
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }
        let nuclide = load_fe56();

        let neutron = Particle::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 14.0e6);
        let mut rng = StdRng::seed_from_u64(123);
        let mut bank = ParticleBank::new();

        let fast_grid = &nuclide.fast_xs[0];
        let (i_grid, f) = fast_grid.lookup_grid_index(14.0e6);
        let photon_prod_xs = fast_grid.lookup_photon_prod(i_grid, f);
        let (total_xs, _, _, _) = fast_grid.lookup(14.0e6);

        // Force high yield to ensure we get photons
        let boosted_prod = photon_prod_xs.max(total_xs * 5.0);

        for _ in 0..100 {
            bank.clear();
            sample_secondary_photons(
                &neutron,
                neutron.energy,
                neutron.direction,
                &nuclide,
                fast_grid,
                i_grid,
                f,
                total_xs,
                boosted_prod,
                &mut bank,
                &mut rng,
            );
            if !bank.is_empty() {
                break;
            }
        }

        assert!(!bank.is_empty(), "Should have produced at least one photon");

        let photon = bank.pop_particle().unwrap();
        assert_eq!(
            photon.particle_type,
            ParticleType::Photon,
            "Secondary should be a photon"
        );
        assert!(photon.energy > 0.0, "Photon energy should be positive");
        assert_eq!(
            photon.position, neutron.position,
            "Photon should be at neutron's position"
        );
        assert_eq!(
            photon.weight, neutron.weight,
            "Photon should inherit neutron weight"
        );

        // Direction should be unit vector
        let d = photon.direction;
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-10,
            "Photon direction should be unit vector, got magnitude {mag}"
        );
    }

    #[test]
    fn test_photon_rxn_xs_populated() {
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }
        let nuclide = load_fe56();
        let fast_grid = &nuclide.fast_xs[0];

        assert!(
            !fast_grid.photon_rxn_xs.is_empty(),
            "Fe56 should have photon-producing reactions in photon_rxn_xs"
        );

        // Check that photon_rxn_xs entries have the right structure
        let n_mts = fast_grid.photon_rxn_mt_numbers.len();
        assert_eq!(
            fast_grid.photon_rxn_xs.len(),
            fast_grid.energy.len() * n_mts,
            "flat photon_rxn_xs should be [n_energies * n_mts]"
        );
        for (j, &mt) in fast_grid.photon_rxn_mt_numbers.iter().enumerate() {
            let reaction = &fast_grid.photon_rxn_reactions[j];
            assert!(mt > 0, "MT should be positive");
            // Reaction should have photon products
            let has_photon = reaction
                .products
                .iter()
                .any(|p| p.is_particle_type(&ParticleType::Photon));
            assert!(
                has_photon,
                "Reaction MT {mt} in photon_rxn_xs should have photon products"
            );
            // Photon products should have distributions
            for product in &reaction.products {
                if product.is_particle_type(&ParticleType::Photon) {
                    assert!(
                        !product.distribution.is_empty(),
                        "Photon product in MT {mt} should have distributions"
                    );
                }
            }
        }
    }
}
