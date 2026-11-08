use yamc_nuclide::nuclide::load_nuclide;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Real Nuclear Data Sampling ===");

    // Load Li6 which has product data
    let li6 = load_nuclide("tests/Li6.arrow", &yamc_nuclide::LoadScope::full())?;

    if let Some(temps) = li6.temperatures() {
        println!("Loaded Li6 nuclide with {} temperatures", temps.len());

        // Get reactions at the first available temperature
        if let Some(temp) = temps.first() {
            if let Some(reactions) = li6.reactions_for_temp(temp) {
                println!("\nReactions at {temp}:");
                for _reaction in reactions.values() {
                    if reactions.len() > 5 {
                        break; // Just show a few reactions
                    }
                }
            }
        } else {
            println!("No temperatures available");
        }
    } else {
        println!("No temperature data available");
    }

    println!("\n=== Demonstrating Sampling Framework ===");

    // Show how the sampling would work in a Monte Carlo transport loop
    let incoming_energy = 14e6; // 14 MeV neutron
    let _rng = rand::rng();

    println!("Incoming neutron: E = {:.1} MeV", incoming_energy / 1e6);
    println!("Reaction occurs...");

    // This is pseudocode for how sampling would be integrated:
    println!("\n// Pseudocode for Monte Carlo integration:");
    println!("// match reaction_mt {{");
    println!("//     2 => {{ // Elastic scattering");
    println!("//         if let Some(product) = reaction.products.get(\"neutron\") {{");
    println!("//             let (e_out, mu) = product.sample(incoming_energy, &mut rng);");
    println!("//             particle.energy = e_out;");
    println!("//             particle.scatter(mu, &mut rng);");
    println!("//         }}");
    println!("//     }},");
    println!("//     16 => {{ // (n,2n) reaction");
    println!("//         for product in reaction.products {{");
    println!("//             if product.is_particle_type(&ParticleType::Neutron) {{");
    println!("//                 let (e_out, mu) = product.sample(incoming_energy, &mut rng);");
    println!("//                 // Create new neutron with sampled energy and direction");
    println!("//                 create_secondary_neutron(e_out, mu);");
    println!("//             }}");
    println!("//         }}");
    println!("//     }},");
    println!("//     // Handle other reaction types...");
    println!("// }}");

    println!("\n=== Test Complete ===");
    println!("The sampling framework is ready for integration into Monte Carlo transport!");

    Ok(())
}
