mod tests {
    use yamc_tallies::filter::mesh::*;
    use yamc_tallies::mesh::RegularRectangularMesh;

    #[test]
    fn test_mesh_filter_creation() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);
        let filter = MeshFilter::new(mesh);

        assert_eq!(filter.num_bins(), 8);
    }

    #[test]
    fn test_mesh_filter_get_bin() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);
        let filter = MeshFilter::new(mesh);

        assert_eq!(filter.get_bin([1.0, 1.0, 1.0]), Some(0));
        assert_eq!(filter.get_bin([11.0, 11.0, 11.0]), None);
    }

    #[test]
    fn test_mesh_filter_matches() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 2, 2]);
        let filter = MeshFilter::new(mesh);

        assert!(filter.matches([5.0, 5.0, 5.0]));
        assert!(!filter.matches([11.0, 11.0, 11.0]));
    }

    #[test]
    fn test_mesh_filter_bins_crossed() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 1, 1]);
        let filter = MeshFilter::new(mesh);

        let crossings = filter.get_bins_crossed([0.5, 0.5, 0.5], [9.5, 0.5, 0.5], [1.0, 0.0, 0.0]);

        assert_eq!(crossings.len(), 10);

        // Check fractions sum to 1.0
        let total: f64 = crossings.iter().map(|c| c.length_fraction).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
