mod tests {
    use yamc_particle::particle::{Particle, ParticleType};
    use yamc_physics::util::bank::*;

    /// Stand-in for `history_seed(base_seed, global_index)`.
    const SOURCE_SEED: u32 = 0x5EED_0001;

    #[test]
    fn test_particle_bank_basic() {
        let mut bank = ParticleBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);

        // Create a test particle
        let particle = Particle {
            particle_type: ParticleType::Neutron,
            energy: 1.0e6,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            weight: 1.0,
            alive: true,
            id: 1,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };

        bank.add_source_particle(particle.clone(), 0xABCD_1234);
        assert_eq!(bank.len(), 1);
        assert!(!bank.is_empty());

        let retrieved = bank.pop_particle().unwrap();
        assert_eq!(retrieved.energy, 1.0e6);
        assert_eq!(
            bank.walk_seed(),
            0xABCD_1234,
            "source particle keeps its history seed"
        );
        assert!(bank.is_empty());
    }

    #[test]
    fn test_particle_bank_secondary() {
        let mut bank = ParticleBank::new();

        let primary = Particle {
            particle_type: ParticleType::Neutron,
            energy: 14.0e6,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            weight: 1.0,
            alive: true,
            id: 1,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };

        let secondary = Particle {
            particle_type: ParticleType::Neutron,
            energy: 7.0e6,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
            weight: 1.0,
            alive: true,
            id: 2,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };

        bank.add_source_particle(primary, SOURCE_SEED);
        bank.bank_secondary(secondary);

        assert_eq!(bank.len(), 2);

        // Secondary comes out first (LIFO / stack order).
        // In Monte Carlo transport, processing order doesn't affect physics.
        let p1 = bank.pop_particle().unwrap();
        let s1 = bank.walk_seed();
        assert_eq!(p1.energy, 7.0e6);

        // Then primary
        let p2 = bank.pop_particle().unwrap();
        let s2 = bank.walk_seed();
        assert_eq!(p2.energy, 14.0e6);

        // Issue #111: the secondary transports on its OWN identity-derived
        // stream, keyed on the banking walk's seed and its ordinal there, not
        // on a continuation of the parent's state.
        assert_eq!(s2, SOURCE_SEED, "source particle keeps its history seed");
        assert_eq!(
            s1,
            yamc_rng::secondary_seed(SOURCE_SEED, 0),
            "secondary 0 must get secondary_seed(parent, 0)"
        );

        assert!(bank.is_empty());
    }

    /// Issue #111: a walk numbers ITS OWN secondaries from 0, so the seed a
    /// secondary gets is a function of where it sits in the history's emission
    /// tree and not of when the bank happened to hand it out. Popping resets
    /// the ordinal; two walks that each bank one secondary must therefore get
    /// `secondary_seed(their own seed, 0)`, not ordinals 0 and 1 off a single
    /// per-history counter (which is what would make the drain order matter,
    /// see the module docs on `secondary_seed`).
    #[test]
    fn test_secondary_seeds_are_keyed_on_the_banking_walk() {
        use yamc_rng::secondary_seed;

        let p = || Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e6);
        let mut bank = ParticleBank::new();
        bank.add_source_particle(p(), SOURCE_SEED);

        let _source = bank.pop_particle().unwrap();
        assert_eq!(bank.walk_seed(), SOURCE_SEED);
        // The source walk banks two secondaries: ordinals 0 and 1 under it.
        bank.bank_secondary(p());
        bank.bank_secondary(p());
        let a = secondary_seed(SOURCE_SEED, 0);
        let b = secondary_seed(SOURCE_SEED, 1);
        assert_ne!(a, b, "siblings must not share a stream");

        // LIFO: the second one comes out first.
        let _p1 = bank.pop_particle().unwrap();
        assert_eq!(bank.walk_seed(), b);
        // It banks one of its own, which is ordinal 0 UNDER IT -- the counter
        // restarted at the pop.
        bank.bank_secondary(p());
        let _p2 = bank.pop_particle().unwrap();
        assert_eq!(
            bank.walk_seed(),
            secondary_seed(b, 0),
            "a grandchild must be keyed on its parent, not on a flat per-history \
             ordinal (which the drain order would decide)"
        );

        let _p3 = bank.pop_particle().unwrap();
        assert_eq!(
            bank.walk_seed(),
            a,
            "the first sibling still gets its own key"
        );
        assert!(bank.is_empty());
    }

    #[test]
    fn test_particle_bank_with_capacity() {
        let bank = ParticleBank::with_capacity(100);
        assert!(bank.is_empty());
        // Cannot check internal queue capacity directly as it is private.
        // This test only checks that the bank is created and is empty.
    }

    #[test]
    fn test_particle_bank_clear() {
        let mut bank = ParticleBank::new();

        let particle = Particle {
            particle_type: ParticleType::Neutron,
            energy: 1.0e6,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            weight: 1.0,
            alive: true,
            id: 1,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };

        bank.add_source_particle(particle.clone(), 1);
        bank.add_source_particle(particle.clone(), 2);
        bank.add_source_particle(particle, 3);

        assert_eq!(bank.len(), 3);

        bank.clear();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
    }
}
