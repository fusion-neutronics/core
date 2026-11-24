mod tests {
    use yamc_tallies::mesh::*;

    #[test]
    fn test_mesh_creation() {
        let mesh =
            RegularRectangularMesh::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        assert_eq!(mesh.num_voxels(), 8);
        assert_eq!(mesh.width(), [10.0, 10.0, 10.0]);
        assert_eq!(mesh.get_voxel_volume(0), 1000.0);
    }

    #[test]
    #[should_panic]
    fn test_mesh_invalid_dimension() {
        RegularRectangularMesh::new(
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
            [0, 2, 2], // Zero dimension should panic
        );
    }

    #[test]
    #[should_panic]
    fn test_mesh_invalid_bounds() {
        RegularRectangularMesh::new(
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, 10.0], // x_max < x_min should panic
            [2, 2, 2],
        );
    }

    #[test]
    fn test_get_bin_corners() {
        // Create a 2x2x2 mesh from [-10, -10, -10] to [10, 10, 10]
        let mesh =
            RegularRectangularMesh::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        // Test all 8 corner voxels (using positions inside each voxel)
        // Z-major ordering: bin = (iz * ny + iy) * nx + ix

        // Lower-left-bottom corner (ix=0, iy=0, iz=0)
        assert_eq!(mesh.get_bin([-9.0, -9.0, -9.0]), Some(0));

        // Lower-right-bottom corner (ix=1, iy=0, iz=0)
        assert_eq!(mesh.get_bin([9.0, -9.0, -9.0]), Some(1));

        // Upper-left-bottom corner (ix=0, iy=1, iz=0)
        assert_eq!(mesh.get_bin([-9.0, 9.0, -9.0]), Some(2));

        // Upper-right-bottom corner (ix=1, iy=1, iz=0)
        assert_eq!(mesh.get_bin([9.0, 9.0, -9.0]), Some(3));

        // Lower-left-top corner (ix=0, iy=0, iz=1)
        assert_eq!(mesh.get_bin([-9.0, -9.0, 9.0]), Some(4));

        // Lower-right-top corner (ix=1, iy=0, iz=1)
        assert_eq!(mesh.get_bin([9.0, -9.0, 9.0]), Some(5));

        // Upper-left-top corner (ix=0, iy=1, iz=1)
        assert_eq!(mesh.get_bin([-9.0, 9.0, 9.0]), Some(6));

        // Upper-right-top corner (ix=1, iy=1, iz=1)
        assert_eq!(mesh.get_bin([9.0, 9.0, 9.0]), Some(7));
    }

    #[test]
    fn test_get_bin_out_of_bounds() {
        let mesh =
            RegularRectangularMesh::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        // Test positions outside mesh
        assert_eq!(mesh.get_bin([-11.0, 0.0, 0.0]), None);
        assert_eq!(mesh.get_bin([11.0, 0.0, 0.0]), None);
        assert_eq!(mesh.get_bin([0.0, -11.0, 0.0]), None);
        assert_eq!(mesh.get_bin([0.0, 11.0, 0.0]), None);
        assert_eq!(mesh.get_bin([0.0, 0.0, -11.0]), None);
        assert_eq!(mesh.get_bin([0.0, 0.0, 11.0]), None);
    }

    #[test]
    fn test_get_bin_boundary() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 10, 10]);

        // Test exact lower boundary (should be in bin 0)
        assert_eq!(mesh.get_bin([0.0, 0.0, 0.0]), Some(0));

        // Test exact upper boundary (should be None or last bin)
        // Upper boundary is exclusive
        assert_eq!(mesh.get_bin([10.0, 10.0, 10.0]), None);

        // Just inside upper boundary should be last bin (ix=9, iy=9, iz=9)
        assert_eq!(mesh.get_bin([9.99, 9.99, 9.99]), Some(999));
    }

    #[test]
    fn test_fine_mesh() {
        // Test with a finer mesh
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 10, 10]);

        assert_eq!(mesh.num_voxels(), 1000);
        assert_eq!(mesh.width(), [1.0, 1.0, 1.0]);
        assert_eq!(mesh.get_voxel_volume(0), 1.0);

        // Test specific position
        assert_eq!(mesh.get_bin([0.5, 0.5, 0.5]), Some(0)); // First bin
        assert_eq!(mesh.get_bin([5.5, 5.5, 5.5]), Some(555)); // Middle bin: (5*10+5)*10+5 = 555
    }

    #[test]
    fn test_non_cubic_mesh() {
        // Test with different dimensions in each direction
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [2, 3, 4]);

        assert_eq!(mesh.num_voxels(), 24); // 2*3*4
        assert_eq!(mesh.width(), [5.0, 20.0 / 3.0, 7.5]);

        // Test bin calculation: (iz * ny + iy) * nx + ix
        // Position [6.0, 1.0, 8.0] should be in voxel (ix=1, iy=0, iz=1)
        // bin = (1 * 3 + 0) * 2 + 1 = 7
        assert_eq!(mesh.get_bin([6.0, 1.0, 8.0]), Some(7));
    }

    #[test]
    fn test_bins_crossed_single_voxel() {
        // Test a track that stays in a single voxel
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        let r0 = [1.0, 1.0, 1.0];
        let r1 = [2.0, 2.0, 2.0];
        let direction = [1.0 / 3.0_f64.sqrt(); 3]; // Unit diagonal

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Should only cross one bin
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].bin, 0); // First voxel
        assert!((crossings[0].length_fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bins_crossed_straight_line_x() {
        // Test a track along the x-axis crossing multiple voxels
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 1, 1]);

        let r0 = [0.5, 0.5, 0.5];
        let r1 = [9.5, 0.5, 0.5];
        let direction = [1.0, 0.0, 0.0];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Should cross all 10 voxels in x direction
        assert_eq!(crossings.len(), 10);

        // Check bins are sequential
        for (i, crossing) in crossings.iter().enumerate() {
            assert_eq!(crossing.bin, i);
        }

        // Check fractions sum to 1.0
        let total_fraction: f64 = crossings.iter().map(|c| c.length_fraction).sum();
        assert!((total_fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bins_crossed_diagonal() {
        // Test a diagonal track across a 2x2x2 mesh
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2, 2, 2]);

        let r0 = [0.1, 0.1, 0.1];
        let r1 = [1.9, 1.9, 1.9];
        let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
        let direction = [sqrt3_inv, sqrt3_inv, sqrt3_inv];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Diagonal should cross multiple voxels
        assert!(crossings.len() > 1);
        assert!(crossings.len() <= 8); // At most all voxels

        // Check fractions sum to approximately 1.0
        let total_fraction: f64 = crossings.iter().map(|c| c.length_fraction).sum();
        assert!((total_fraction - 1.0).abs() < 1e-6);

        // All bins should be valid
        for crossing in &crossings {
            assert!(crossing.bin < 8);
            assert!(crossing.length_fraction > 0.0);
            assert!(crossing.length_fraction <= 1.0);
        }
    }

    #[test]
    fn test_bins_crossed_outside_mesh_entering() {
        // Test a track that starts outside the mesh but enters it
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        let r0 = [-1.0, -1.0, -1.0]; // Outside mesh
        let r1 = [5.0, 5.0, 5.0];
        let direction = [1.0 / 3.0_f64.sqrt(); 3];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Track enters mesh at (0,0,0) and ends at (5,5,5), crossing multiple bins
        assert!(
            !crossings.is_empty(),
            "Track entering mesh should produce crossings"
        );
        assert!(crossings.len() <= 8);

        // All bins should be valid
        for crossing in &crossings {
            assert!(crossing.bin < 8);
            assert!(crossing.length_fraction > 0.0);
        }

        // Fractions should not sum to 1.0 (only the portion inside the mesh)
        let total_fraction: f64 = crossings.iter().map(|c| c.length_fraction).sum();
        assert!(
            total_fraction < 1.0,
            "Fraction should be < 1 since track starts outside"
        );
    }

    #[test]
    fn test_bins_crossed_outside_mesh_miss() {
        // Test a track that starts outside the mesh and misses it entirely
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);

        let r0 = [-1.0, -1.0, -1.0]; // Outside mesh
        let r1 = [-5.0, -5.0, -5.0]; // Moving away from mesh
        let direction = [-1.0 / 3.0_f64.sqrt(); 3];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Should return empty since track doesn't enter mesh
        assert_eq!(crossings.len(), 0);
    }

    #[test]
    fn test_bins_crossed_fractions() {
        // Test that length fractions are calculated correctly
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 1, 1]);

        // Track from 0.5 to 2.5 (crosses 2 voxels of width 1.0 each)
        let r0 = [0.5, 0.5, 0.5];
        let r1 = [2.5, 0.5, 0.5];
        let direction = [1.0, 0.0, 0.0];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        assert_eq!(crossings.len(), 3); // Voxels 0, 1, 2

        // First voxel: 0.5 to 1.0 = 0.5 units = 0.25 of total
        assert!((crossings[0].length_fraction - 0.25).abs() < 1e-10);

        // Second voxel: 1.0 to 2.0 = 1.0 units = 0.5 of total
        assert!((crossings[1].length_fraction - 0.5).abs() < 1e-10);

        // Third voxel: 2.0 to 2.5 = 0.5 units = 0.25 of total
        assert!((crossings[2].length_fraction - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_bins_crossed_parallel_to_grid() {
        // Test a track parallel to grid lines in one dimension
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [5, 5, 5]);

        // Track along x-axis at y=2.0, z=2.0
        // Voxel width is 2.0, so boundaries at x = 0, 2, 4, 6, 8, 10
        // From x=1.0 (voxel 0) to x=9.0 (voxel 4) crosses 5 voxels
        let r0 = [1.0, 2.0, 2.0];
        let r1 = [9.0, 2.0, 2.0];
        let direction = [1.0, 0.0, 0.0];

        let crossings = mesh.bins_crossed(r0, r1, direction);

        // Should cross 5 voxels in x direction (from ix=0 to ix=4)
        assert_eq!(crossings.len(), 5);

        // Check that all crossings are in the same y,z voxel
        for crossing in &crossings {
            // Extract y and z indices from bin
            let bin = crossing.bin;
            let nx = 5;
            let ny = 5;
            let ix = bin % nx;
            let iy = (bin / nx) % ny;
            let iz = bin / (nx * ny);

            assert_eq!(iy, 1); // All in same y voxel
            assert_eq!(iz, 1); // All in same z voxel
            assert!(ix < 5); // x voxel varies (0 to 4)
        }
    }
}
