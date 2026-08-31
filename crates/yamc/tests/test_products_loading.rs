use yamc_nuclide::nuclide::load_nuclide;

#[test]
fn test_pb208_products_are_loaded() {
    // Load Li6 nuclide data as the products fixture.
    let nuclide = load_nuclide("tests/Li6.arrow", &yamc_nuclide::LoadScope::full()).unwrap();

    // Check that we have reactions
    assert!(!nuclide.reactions.is_empty());

    // Print available temperatures
    println!("Loaded temperatures: {:?}", nuclide.loaded_temperatures);
    println!(
        "Available temperatures: {:?}",
        nuclide.available_temperatures
    );

    // Check specific temperature - use loaded_temperatures which is the canonical source
    let temp = if nuclide.loaded_temperatures.contains(&"294K".to_string()) {
        "294K"
    } else if nuclide.loaded_temperatures.contains(&"294".to_string()) {
        "294"
    } else {
        // Use the first available temperature
        nuclide.loaded_temperatures.first().unwrap()
    };
    println!("Using temperature: {}", temp);

    let reactions_at_temp = nuclide
        .reactions_for_temp(temp)
        .expect("Temperature should exist");

    // Check specific reaction (MT=2 should have products in Pb208)
    if let Some(reaction) = reactions_at_temp.get(&2) {
        println!("MT=2 reaction has {} products", reaction.products.len());
        // Should have products for elastic scattering
        assert!(!reaction.products.is_empty());

        // Print product details
        for (i, product) in reaction.products.iter().enumerate() {
            println!("Product {}: {:?}", i, product.particle);
            if !product.distribution.is_empty() {
                match &product.distribution[0] {
                    yamc_nuclide::reaction_product::AngleEnergyDistribution::UncorrelatedAngleEnergy {
                        angle,
                        energy,
                    } => {
                        println!(
                            "  Angle distribution has {} energy points",
                            angle.energy.len()
                        );
                        match energy {
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::LevelInelastic {
                                ..
                            }) => {
                                println!("  Energy distribution: LevelInelastic");
                            }
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::Tabulated {
                                energy,
                                ..
                            }) => {
                                println!(
                                    "  Energy distribution: Tabulated with {} energy points",
                                    energy.len()
                                );
                            }
                            Some(
                                yamc_nuclide::reaction_product::EnergyDistribution::ContinuousTabular {
                                    energy,
                                    ..
                                },
                            ) => {
                                println!("  Energy distribution: ContinuousTabular with {} energy points", energy.len());
                            }
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::Maxwell {
                                ..
                            }) => {
                                println!("  Energy distribution: Maxwell fission spectrum");
                            }
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::Watt { .. }) => {
                                println!("  Energy distribution: Watt fission spectrum");
                            }
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::Evaporation {
                                ..
                            }) => {
                                println!("  Energy distribution: Evaporation spectrum");
                            }
                            Some(yamc_nuclide::reaction_product::EnergyDistribution::DiscretePhoton {
                                primary_flag,
                                energy: photon_energy,
                                ..
                            }) => {
                                println!("  Energy distribution: DiscretePhoton (primary_flag={}, energy={:.4e})", primary_flag, photon_energy);
                            }
                            None => {
                                println!("  Energy distribution: None");
                            }
                        }
                    }
                    yamc_nuclide::reaction_product::AngleEnergyDistribution::KalbachMann { kalbach } => {
                        println!(
                            "  KalbachMann distribution has {} energy points",
                            kalbach.energy.len()
                        );
                    }
                    yamc_nuclide::reaction_product::AngleEnergyDistribution::CorrelatedAngleEnergy {
                        correlated,
                    } => {
                        println!(
                            "  CorrelatedAngleEnergy distribution has {} incoming energy points",
                            correlated.energy.len()
                        );
                        println!(
                            "  With {} energy_out distributions",
                            correlated.distributions.len()
                        );
                    }
                    yamc_nuclide::reaction_product::AngleEnergyDistribution::Evaporation { .. } => {
                        println!("  Evaporation distribution");
                    }
                    yamc_nuclide::reaction_product::AngleEnergyDistribution::NBodyPhaseSpace {
                        n_bodies,
                        ..
                    } => {
                        println!("  NBodyPhaseSpace distribution with {} bodies", n_bodies);
                    }
                }
            }
        }
    } else {
        panic!("MT=2 reaction not found");
    }

    // Check another reaction that should have products
    println!(
        "Available reactions: {:?}",
        reactions_at_temp.keys().collect::<Vec<_>>()
    );

    // Count total products across all reactions
    let mut total_products = 0;
    let mut reactions_with_products = 0;

    for (mt, reaction) in reactions_at_temp {
        if !reaction.products.is_empty() {
            println!("MT={} has {} products", mt, reaction.products.len());
            reactions_with_products += 1;
            total_products += reaction.products.len();
        }
    }

    println!("Total reactions with products: {}", reactions_with_products);
    println!("Total products across all reactions: {}", total_products);
    assert!(
        total_products > 0,
        "Expected at least some products to be loaded"
    );
}
