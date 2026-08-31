mod tests {
    use yamc_particle::particle::*;

    #[test]
    fn test_particle_construction() {
        let p = Particle::new([0.0, 1.0, 2.0], [1.0, 0.0, 0.0], 1e6);
        assert_eq!(p.position, [0.0, 1.0, 2.0]);
        assert_eq!(p.direction, [1.0, 0.0, 0.0]);
        assert_eq!(p.energy, 1e6);
        assert!(p.alive);
    }

    #[test]
    fn test_particle_move_by() {
        let mut p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e6);

        // Move 2 units in x direction
        p.move_by(2.0);
        assert_eq!(p.position, [2.0, 0.0, 0.0]);

        // Move another 1.5 units
        p.move_by(1.5);
        assert_eq!(p.position, [3.5, 0.0, 0.0]);

        // Test with diagonal direction (normalized)
        let sqrt_2_inv = 1.0 / 2.0_f64.sqrt();
        let mut p2 = Particle::new([0.0, 0.0, 0.0], [sqrt_2_inv, sqrt_2_inv, 0.0], 1e6);
        p2.move_by(2.0_f64.sqrt()); // Move sqrt(2) units to get 1 unit in both x and y

        // Allow for floating point precision
        assert!((p2.position[0] - 1.0).abs() < 1e-10);
        assert!((p2.position[1] - 1.0).abs() < 1e-10);
        assert_eq!(p2.position[2], 0.0);
    }
}
