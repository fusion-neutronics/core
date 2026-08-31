use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::Surface;
use yamc::geo::{HalfspaceType, Region};
use yamc_materials::Material;
use yamc_nuclide::Config;

fn main() {
    println!("=== Materials for MC Demo ===");

    // Set the path to the cross section data using local test files
    let cross_sections = HashMap::from([
        ("Li6".to_string(), "tests/Li6.arrow".to_string()),
        ("Li7".to_string(), "tests/Li7.arrow".to_string()),
        ("Fe56".to_string(), "tests/Fe56.arrow".to_string()),
    ]);
    Config::global().set_cross_sections(cross_sections);

    println!("Creating first material with Li6...");
    let mut mat = Material::new(
        HashMap::from([("Li6".into(), 0.05)]),
        "atom",
        "g/cm3",
        Some(19.1),
    )
    .unwrap();
    mat.set_temperature("294");
    mat.volume(Some(100.0)).unwrap();
    // mat.calculate_microscopic_xs_neutron(Some(&vec!["2".to_string()]));
    mat.calculate_macroscopic_xs(&vec![3], false);
    // mat.calculate_microscopic_xs_neutron(None);
    // mat.calculate_macroscopic_xs(None);

    // Create a second material and add element Li (lithium) with natural abundances
    println!("Creating lithium material...");
    let mut li_nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
    li_nuclides.insert("Fe56".into(), 1.0);
    let mut lithium_mat = Material::new(li_nuclides, "atom", "g/cm3", Some(0.534)).unwrap(); // Lithium density
    lithium_mat.set_temperature("294");
    lithium_mat.volume(Some(50.0)).unwrap();
    // lithium_mat.calculate_microscopic_xs_neutron(Some(&vec!["2".to_string()]));
    // lithium_mat.calculate_macroscopic_xs(Some(&vec!["2".to_string()]));
    // lithium_mat.calculate_microscopic_xs_neutron(None);
    // Print available MT numbers for each nuclide at temperature "294"

    lithium_mat.calculate_macroscopic_xs(&vec![1], true);

    println!("Testing nuclide sampling...");
    let mut rng = StdRng::seed_from_u64(123456);
    let energy = 1.0e3; // 1 MeV
                        // Sample nuclide
    let sampled_nuclide_name = lithium_mat.sample_interacting_nuclide(energy, &mut rng);
    println!("Sampled nuclide: {sampled_nuclide_name}");
    if let Some(nuclide) = lithium_mat.nuclide_data.get(&sampled_nuclide_name) {
        println!("fissionable: {}", nuclide.fissionable);
    } else {
        println!("Nuclide struct not found for {sampled_nuclide_name}");
    }

    // Print the per-nuclide macroscopic total cross sections for lithium_mat

    // ------------------------------------------------------------
    // Performance timing: mean free path sampling across energies
    // ------------------------------------------------------------
    println!("Running performance benchmark...");
    use std::time::Instant;
    let n_energies = 100; // number of distinct energies
    let n_samples_per_energy = 1000; // number of samples per energy
                                     // Build energies logarithmically spaced between 1e2 eV and 1e6 eV
    let e_min = 1.0e2_f64;
    let e_max = 1.0e6_f64;
    let log_min = e_min.ln();
    let log_max = e_max.ln();
    let mut energies: Vec<f64> = Vec::with_capacity(n_energies);
    for i in 0..n_energies {
        let f = i as f64 / (n_energies as f64 - 1.0);
        energies.push((log_min + f * (log_max - log_min)).exp());
    }
    // Ensure total xs grid is built once (MT=1)
    lithium_mat.calculate_macroscopic_xs(&vec![1], false);
    let start = Instant::now();
    let mut total_queries: u64 = 0;
    let mut accum_mfp = 0.0_f64; // accumulate to prevent optimization
    for &e in &energies {
        for _ in 0..n_samples_per_energy {
            if let Some(mfp) = lithium_mat.mean_free_path_neutron(e) {
                accum_mfp += mfp;
            }
            total_queries += 1;
        }
    }
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let per_call_ns = (secs * 1.0e9) / (total_queries as f64);
    println!(
        "Mean free path sampling: energies={n_energies} samples/energy={n_samples_per_energy} total_calls={total_queries} total_time={secs:.6}s avg_per_call={per_call_ns:.1} ns (accum_mfp={accum_mfp:.3})"
    );

    println!("Testing geometry features...");
    println!("Creating cube surfaces...");
    // Cube from x=0..1, y=0..1, z=0..1
    let sx0 = Arc::new(Surface::x_plane(0.0, Some(1), None));
    let sx1 = Arc::new(Surface::x_plane(1.0, Some(2), None));
    let sy0 = Arc::new(Surface::y_plane(0.0, Some(3), None));
    let sy1 = Arc::new(Surface::y_plane(1.0, Some(4), None));
    let sz0 = Arc::new(Surface::z_plane(0.0, Some(5), None));
    let sz1 = Arc::new(Surface::z_plane(1.0, Some(6), None));

    // Cube region: intersection of all halfspaces
    let cube = Region::new_from_halfspace(HalfspaceType::Above(sx0.clone()))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
            sx1.clone(),
        )))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
            sy0.clone(),
        )))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
            sy1.clone(),
        )))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
            sz0.clone(),
        )))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
            sz1.clone(),
        )));
    println!("Cube region created.");
    let cube_bb = cube.bounding_box();
    println!("Cube bounding box: {cube_bb:?}");

    println!("Creating sphere surface...");
    let sphere = Arc::new(Surface::new_sphere(0.5, 0.5, 0.5, 0.4, Some(7), None));
    let sphere_region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
    println!("Sphere region created.");
    let sphere_bb = sphere_region.bounding_box();
    println!("Sphere bounding box: {sphere_bb:?}");

    println!("Making intersection and union regions...");
    let intersection = cube.intersection(&sphere_region);
    let union = cube.union(&sphere_region);
    println!(
        "Intersection bounding box: {:?}",
        intersection.bounding_box()
    );
    println!("Union bounding box: {:?}", union.bounding_box());

    println!("Testing point containment...");
    let pt_inside = (0.5, 0.5, 0.5);
    let pt_outside = (1.5, 1.5, 1.5);
    println!(
        "Point {:?} in intersection: {}",
        pt_inside,
        intersection.contains(pt_inside)
    );
    println!(
        "Point {:?} in union: {}",
        pt_inside,
        union.contains(pt_inside)
    );
    println!(
        "Point {:?} in intersection: {}",
        pt_outside,
        intersection.contains(pt_outside)
    );
    println!(
        "Point {:?} in union: {}",
        pt_outside,
        union.contains(pt_outside)
    );

    println!("=== Demo completed successfully! ===");
}
