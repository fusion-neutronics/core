mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_source::distribution::spatial::*;

    #[test]
    fn test_point_construction() {
        let point = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(point.position(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_point_default() {
        let point = Point::default();
        assert_eq!(point.position(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_point_sampling() {
        let mut rng = StdRng::seed_from_u64(1);
        let point = Point::new([5.0, -3.0, 7.5]);

        // Point should always return the same position
        for _ in 0..100 {
            assert_eq!(point.sample(&mut rng), [5.0, -3.0, 7.5]);
        }
    }

    #[test]
    fn test_point_equality() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 3.1]);

        assert_eq!(point1, point2);
        assert_ne!(point1, point3);
    }

    #[test]
    fn test_point_clone() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = point1.clone();

        assert_eq!(point1, point2);
        assert_eq!(point1.position(), point2.position());
    }
}
