use rand::rng;
use yamc_nuclide::reaction_product::*;

fn main() {
    println!("=== Testing Reaction Product Sampling ===");

    // Create a simple tabulated distribution for testing
    let mu_dist = Tabulated {
        x: vec![-1.0, 0.0, 1.0],
        p: vec![0.2, 0.5, 1.0], // CDF values
        c: vec![],
        interp: TabulatedInterp::default(),
    };

    // Create an angular distribution
    let angle_dist = AngleDistribution {
        energy: vec![1e6, 1e7, 2e7],
        mu: vec![mu_dist.clone(), mu_dist.clone(), mu_dist.clone()],
    };

    // Create an uncorrelated angle-energy distribution
    let angle_energy_dist = AngleEnergyDistribution::UncorrelatedAngleEnergy {
        angle: angle_dist,
        energy: Some(EnergyDistribution::LevelInelastic {
            threshold: 1.0e6,  // 1 MeV threshold
            mass_ratio: 0.965, // (A/(A+1))^2 for A~56
        }),
    };

    // Create a reaction product (neutron)
    let product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![],
        distribution: vec![angle_energy_dist],
        product_yield: None,
    };

    let mut rng = rng();

    println!("\nSampling 10 outgoing particles at 14 MeV:");
    for i in 0..10 {
        let incoming_energy = 14e6; // 14 MeV
        let (e_out, mu) = product.sample(incoming_energy, &mut rng);
        println!(
            "  Sample {}: E_out = {:.3e} eV, mu = {:.3}",
            i + 1,
            e_out,
            mu
        );
    }

    println!("\nTesting CDF conversion:");
    let pdf = Tabulated {
        x: vec![-1.0, -0.5, 0.0, 0.5, 1.0],
        p: vec![0.1, 0.2, 0.4, 0.2, 0.1], // PDF values
        c: vec![],
        interp: TabulatedInterp::default(),
    };
    let cdf = pdf.to_cdf();
    println!("Original PDF: {:?}", pdf.p);
    println!("Converted CDF: {:?}", cdf.p);

    println!("\nSampling from CDF:");
    for i in 0..5 {
        let sample = cdf.sample(&mut rng);
        println!("  Sample {}: mu = {:.3}", i + 1, sample);
    }

    println!("\n=== Sampling Test Complete ===");
}
