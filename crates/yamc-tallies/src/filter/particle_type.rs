use yamc_particle::ParticleType;

/// Particle type filter for tallies - restricts scoring to a specific particle type
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParticleTypeFilter {
    pub particle_type: ParticleType,
}

impl ParticleTypeFilter {
    pub fn new(particle_type: ParticleType) -> Self {
        Self { particle_type }
    }

    pub fn matches(&self, pt: ParticleType) -> bool {
        self.particle_type == pt
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_neutron_filter() {
        let filter = ParticleTypeFilter::new(ParticleType::Neutron);
        assert_eq!(filter.particle_type, ParticleType::Neutron);
    }

    #[test]
    fn test_new_photon_filter() {
        let filter = ParticleTypeFilter::new(ParticleType::Photon);
        assert_eq!(filter.particle_type, ParticleType::Photon);
    }

    #[test]
    fn test_matches_neutron() {
        let filter = ParticleTypeFilter::new(ParticleType::Neutron);
        assert!(filter.matches(ParticleType::Neutron));
        assert!(!filter.matches(ParticleType::Photon));
    }

    #[test]
    fn test_matches_photon() {
        let filter = ParticleTypeFilter::new(ParticleType::Photon);
        assert!(filter.matches(ParticleType::Photon));
        assert!(!filter.matches(ParticleType::Neutron));
    }

    #[test]
    fn test_num_bins_via_filter_enum() {
        use crate::filter::Filter;
        let filter = Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Neutron));
        assert_eq!(filter.num_bins(), 1);

        let filter = Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon));
        assert_eq!(filter.num_bins(), 1);
    }

    #[test]
    fn test_clone_and_eq() {
        let filter = ParticleTypeFilter::new(ParticleType::Neutron);
        let cloned = filter.clone();
        assert_eq!(filter, cloned);

        let other = ParticleTypeFilter::new(ParticleType::Photon);
        assert_ne!(filter, other);
    }

    #[test]
    fn test_debug_format() {
        let filter = ParticleTypeFilter::new(ParticleType::Neutron);
        let debug_str = format!("{:?}", filter);
        assert!(debug_str.contains("ParticleTypeFilter"));
        assert!(debug_str.contains("Neutron"));
    }
}
