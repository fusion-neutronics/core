mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use yamc::geometry::cell::Cell;
    use yamc::geometry::Geometry;
    use yamc::model::*;

    use yamc::geo::{BoundaryType, Surface, SurfaceKind};
    use yamc::geo::{HalfspaceType, Region};
    use yamc_materials::Material;
    use yamc_source::source::{ParticleSource, Source};

    #[test]
    fn test_model_construction() {
        // Sphere surface with vacuum boundary
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        // Material with Li6 nuclide
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cc",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();
        // Prepare material cross sections before creating the Arc (include MT 301 for heating)
        material.calculate_macroscopic_xs(&vec![1, 301], true);
        let material_arc = Arc::new(material.clone());
        let cell = Cell::new(Some(1), region, Some("sphere_cell".to_string()), Some(0));
        let geometry = Geometry::new(vec![cell], vec![material_arc]).unwrap();
        let source = ParticleSource::Neutron(Source {
            space: yamc_source::source::SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: yamc_source::distribution::angular::AngularDistribution::new_monodirectional(
                0.0, 0.0, 1.0,
            ),
            energy: yamc_source::source::SourceEnergyDistribution::Discrete(
                yamc_source::distribution::energy::Discrete::new(vec![1e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });
        let particles = 100;
        let batches = 10;

        // Create a tally for absorption reactions with a custom name
        let mut absorption_tally = yamc_tallies::tally::Tally::new();
        absorption_tally.scores = vec![yamc_tallies::tally::Score::ReactionRate(
            yamc_tallies::tally::ReactionRateScore::from_mt(yamc_tallies::tally::Mt::new(101)),
        )]; // MT 101 = absorption
        absorption_tally.name = Some("Absorption Tally".to_string());
        absorption_tally.units = "events".to_string();
        absorption_tally.initialize_batches(batches);
        let absorption_tally_arc = Arc::new(absorption_tally);

        let mut model = Model::new(
            geometry,
            vec![source.clone()],
            vec![Arc::clone(&absorption_tally_arc)],
        );
        let settings = TransportSettings {
            total_particles: Some(particles * batches),
            ..Default::default()
        };
        assert_eq!(settings.total_particles, Some(particles * batches));
        assert_eq!(
            model.sources[0].source().energy,
            yamc_source::source::SourceEnergyDistribution::Discrete(
                yamc_source::distribution::energy::Discrete::new(vec![1e6], vec![1.0]).unwrap()
            )
        );
        // Check geometry and material
        assert_eq!(model.geometry.cells().len(), 1);
        let cell_material = model
            .geometry
            .material_for(&model.geometry.cells()[0])
            .expect("cell should have material");
        assert!(cell_material.nuclides.contains_key("Li6"));
        // Run the model and ensure it executes without panicking
        model.simulate_transport(&settings).unwrap();

        // Verify the absorption tally was updated in place
        assert_eq!(
            absorption_tally_arc.name,
            Some("Absorption Tally".to_string())
        );
        assert_eq!(absorption_tally_arc.units, "events");
        assert_eq!(absorption_tally_arc.n_batches.load(Ordering::Relaxed), 10);

        println!("Test tally results:");
        println!("Absorption Tally: {}", absorption_tally_arc);

        println!("✓ Absorption tally verified successfully - tally updated in place!");
    }

    #[test]
    fn test_model_threads_parameter() {
        // Semantics-only test: the `threads` parameter must be accepted, run
        // to completion, populate the timing metrics, and give thread-count
        // independent physics (per-history RNG streams make the tallies
        // invariant to how histories are scheduled). Wall-clock speedup is
        // deliberately NOT asserted: the harness runs sibling tests on all
        // cores concurrently, so on a loaded machine a 32-thread run of this
        // ~10 ms workload can come out SLOWER than the 1-thread run (observed
        // 288k vs 960k particles/s), which made every speedup threshold
        // flaky. Performance numbers are printed for eyeballing only.

        // Create a sphere with vacuum boundary
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 5.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

        // Material with multiple nuclides to make computation heavier
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "atom",
            "g/cc",
            Some(0.5),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();
        material.calculate_macroscopic_xs(&vec![1, 301], true);
        let material_arc = Arc::new(material);

        let cell = Cell::new(Some(1), region, Some("sphere_cell".to_string()), Some(0));
        let geometry = Geometry::new(vec![cell], vec![material_arc]).unwrap();

        // Use a larger number of particles to make timing difference apparent
        let source = ParticleSource::Neutron(Source {
            space: yamc_source::source::SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: yamc_source::distribution::angular::AngularDistribution::new_isotropic(),
            energy: yamc_source::source::SourceEnergyDistribution::Discrete(
                yamc_source::distribution::energy::Discrete::new(vec![1e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });
        let particles = 10000;
        let batches = 5;

        // One total-reaction-rate tally per run (MT 1 -- present in the
        // macroscopic XS prepared above): same seed + per-history streams
        // must give the same accumulated result whatever the thread count.
        let make_tally = || {
            let mut tally = yamc_tallies::tally::Tally::new();
            tally.scores = vec![yamc_tallies::tally::Score::ReactionRate(
                yamc_tallies::tally::ReactionRateScore::from_mt(yamc_tallies::tally::Mt::new(1)),
            )];
            tally.initialize_batches(batches);
            Arc::new(tally)
        };

        // Same seed + particle count for all three runs; only the thread
        // count varies. Per-history RNG streams make the tally invariant.
        let base_settings = TransportSettings {
            total_particles: Some(particles * batches),
            seed: 42,
            ..Default::default()
        };

        // Test with 1 thread
        let tally_1 = make_tally();
        let mut model_1 = Model::new(
            geometry.clone(),
            vec![source.clone()],
            vec![Arc::clone(&tally_1)],
        );
        model_1
            .simulate_transport(&TransportSettings {
                threads: Some(1),
                ..base_settings.clone()
            })
            .unwrap();

        // Test with 2 threads
        let tally_2 = make_tally();
        let mut model_2 = Model::new(
            geometry.clone(),
            vec![source.clone()],
            vec![Arc::clone(&tally_2)],
        );
        model_2
            .simulate_transport(&TransportSettings {
                threads: Some(2),
                ..base_settings.clone()
            })
            .unwrap();

        // Test with default (all threads)
        let tally_default = make_tally();
        let mut model_default = Model::new(
            geometry.clone(),
            vec![source.clone()],
            vec![Arc::clone(&tally_default)],
        );
        model_default.simulate_transport(&base_settings).unwrap();

        // Extract performance metrics
        let pps_1_thread = model_1.last_particles_per_second.unwrap();
        let pps_2_threads = model_2.last_particles_per_second.unwrap();
        let pps_default = model_default.last_particles_per_second.unwrap();

        let time_1_thread = model_1.last_elapsed_secs.unwrap();
        let time_2_threads = model_2.last_elapsed_secs.unwrap();
        let time_default = model_default.last_elapsed_secs.unwrap();

        println!("\nPerformance results:");
        println!(
            "  1 thread:  {} particles/s ({:.3}s)",
            pps_1_thread, time_1_thread
        );
        println!(
            "  2 threads: {} particles/s ({:.3}s)",
            pps_2_threads, time_2_threads
        );
        println!(
            "  Default:   {} particles/s ({:.3}s)",
            pps_default, time_default
        );
        println!(
            "  Speedup (2 vs 1): {:.2}x",
            pps_2_threads as f64 / pps_1_thread as f64
        );

        // The physics must not depend on the thread count: per-history RNG
        // streams mean the same seed gives the same histories, so the tally
        // may differ only by floating-point accumulation order.
        let mean_1 = tally_1.get_mean();
        let mean_2 = tally_2.get_mean();
        let mean_default = tally_default.get_mean();
        assert!(
            !mean_1.is_empty() && mean_1[0] > 0.0,
            "empty 1-thread tally"
        );
        for (label, other) in [("2 threads", &mean_2), ("default", &mean_default)] {
            for (a, b) in mean_1.iter().zip(other.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * a.abs().max(1e-300),
                    "total reaction-rate tally differs between 1 thread and {label}: {a} vs {b}"
                );
            }
        }

        // Verify elapsed_time and particles_per_second are populated
        assert!(model_1.last_elapsed_secs.is_some());
        assert!(model_2.last_elapsed_secs.is_some());
        assert!(model_default.last_elapsed_secs.is_some());
        assert!(model_1.last_particles_per_second.is_some());
        assert!(model_2.last_particles_per_second.is_some());
        assert!(model_default.last_particles_per_second.is_some());

        println!("✓ Threads parameter test passed - thread-count invariant results!");
    }
}
