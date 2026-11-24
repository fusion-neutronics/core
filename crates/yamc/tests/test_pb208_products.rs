#[cfg(test)]
mod test_pb208_products {
    use std::path::Path;
    use yamc_nuclide::nuclide::load_nuclide;

    #[test]
    fn test_pb208_with_products() {
        // Use Li6 data as the products fixture.
        let path = Path::new("tests/Li6.arrow");
        let nuclide =
            load_nuclide(path, &yamc_nuclide::LoadScope::full()).expect("Failed to load Li6.arrow");

        // Check that we have reactions
        assert!(!nuclide.reactions.is_empty());

        // Check that some reactions have products
        let mut found_products = false;
        for temp_reactions in nuclide.reactions.iter() {
            for reaction in temp_reactions.values() {
                if !reaction.products.is_empty() {
                    found_products = true;
                    println!(
                        "Reaction MT {} has {} products",
                        reaction.mt_number,
                        reaction.products.len()
                    );

                    for (i, product) in reaction.products.iter().enumerate() {
                        println!(
                            "  Product {}: particle={:?}, emission_mode={}, distributions={}",
                            i,
                            product.particle,
                            product.emission_mode,
                            product.distribution.len()
                        );

                        for (j, dist) in product.distribution.iter().enumerate() {
                            println!("    Distribution {}: {:?}", j, dist);
                        }
                    }
                    break;
                }
            }
            if found_products {
                break;
            }
        }

        assert!(found_products, "No products found in any reactions");
    }
}
