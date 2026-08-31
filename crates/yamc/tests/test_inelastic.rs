use rand::rngs::StdRng;
use rand::SeedableRng;
use yamc_nuclide::Reaction;
use yamc_particle::particle::Particle;
use yamc_physics::neutron::inelastic::*;

#[test]
#[should_panic(expected = "Missing product distributions for sampled inelastic reaction MT 16")]
fn test_inelastic_scatter_panics_without_products() {
    let mut rng = StdRng::seed_from_u64(42);

    // Create a test particle with sufficient energy
    let particle = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10e6); // 10 MeV

    // Create a test reaction without products - should panic
    let reaction = Reaction {
        cross_section: vec![1.0].into(),
        threshold_idx: 0,
        energy: vec![1e5].into(),
        mt_number: 16,   // (n,2n) reaction
        q_value: -6.0e6, // Endothermic, 6 MeV threshold
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };

    // Test the inelastic scatter function - should panic because no products
    let awr_be9 = 8.934;
    let _ = inelastic_scatter(&particle, &reaction, awr_be9, &mut rng);
}

#[test]
#[should_panic(expected = "Missing product distributions for sampled inelastic reaction MT 51")]
fn test_single_neutron_reaction_panics_without_products() {
    let mut rng = StdRng::seed_from_u64(42);

    // Create a test particle with sufficient energy
    let particle = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5e6); // 5 MeV

    // Create an inelastic scattering reaction without products - should panic
    let reaction = Reaction {
        cross_section: vec![1.0].into(),
        threshold_idx: 0,
        energy: vec![1e5].into(),
        mt_number: 51,   // Discrete inelastic level
        q_value: -1.0e6, // 1 MeV excitation energy
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };

    // Test the inelastic scatter function - should panic because no products
    let awr_li6 = 5.963;
    let _ = inelastic_scatter(&particle, &reaction, awr_li6, &mut rng);
}
