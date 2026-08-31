use yamc_particle::particle::ParticleType;

/// Information about a particle that was lost during transport.
///
/// A particle is "lost" when it crosses a surface but the new position
/// is not contained in any cell. This indicates a gap in the geometry
/// (CSG regions don't fully cover the space).
#[derive(Debug, Clone)]
pub struct LostParticle {
    /// Type of the lost particle (Neutron or Photon)
    pub particle_type: ParticleType,
    /// Position where the particle was lost [x, y, z] in cm
    pub position: [f64; 3],
    /// Direction the particle was traveling [u, v, w]
    pub direction: [f64; 3],
    /// Energy of the particle in eV
    pub energy: f64,
    /// Index of the last cell the particle was in (before crossing)
    pub last_cell_index: Option<usize>,
    /// Cell ID of the last cell (user-facing ID)
    pub last_cell_id: Option<u32>,
    /// Name of the last cell
    pub last_cell_name: Option<String>,
    /// Surface ID of the surface that was crossed
    pub surface_id: Option<usize>,
}

impl LostParticle {
    /// Print a diagnostic message for this lost particle to stderr.
    pub fn print_diagnostic(&self) {
        let ptype = match self.particle_type {
            ParticleType::Neutron => "Neutron",
            ParticleType::Photon => "Photon",
        };

        let cell_info = match (&self.last_cell_name, self.last_cell_id) {
            (Some(name), Some(id)) => format!("\"{}\" (ID: {})", name, id),
            (None, Some(id)) => format!("ID: {}", id),
            (Some(name), None) => format!("\"{}\"", name),
            (None, None) => "unknown".to_string(),
        };

        let surface_info = match self.surface_id {
            Some(id) => format!("ID: {}", id),
            None => "unknown".to_string(),
        };

        eprintln!("  =======  LOST PARTICLE  =======");
        eprintln!("  Particle type:   {}", ptype);
        eprintln!("  Energy:          {:.6e} eV", self.energy);
        eprintln!(
            "  Position:        [{:.12e}, {:.12e}, {:.12e}] cm",
            self.position[0], self.position[1], self.position[2]
        );
        eprintln!(
            "  Direction:       [{:.12e}, {:.12e}, {:.12e}]",
            self.direction[0], self.direction[1], self.direction[2]
        );
        eprintln!("  Last cell:       {}", cell_info);
        eprintln!("  Surface crossed: {}", surface_info);
        eprintln!();
        eprintln!("  To replay this particle:");
        eprintln!("    source = yamc.NeutronSource(");
        eprintln!(
            "        position=yamc.Point({:.12e}, {:.12e}, {:.12e}),",
            self.position[0], self.position[1], self.position[2]
        );
        eprintln!(
            "        energy=yamc.Discrete([{:.6e}], [1.0]),",
            self.energy
        );
        eprintln!(
            "        direction=yamc.Monodirectional([{:.12e}, {:.12e}, {:.12e}]),",
            self.direction[0], self.direction[1], self.direction[2]
        );
        eprintln!("    )");
        eprintln!("  ===============================");
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lost_particle_diagnostic_with_all_fields() {
        let lp = LostParticle {
            particle_type: ParticleType::Neutron,
            position: [1.0, 2.0, 3.0],
            direction: [0.0, 0.0, 1.0],
            energy: 1e6,
            last_cell_index: Some(0),
            last_cell_id: Some(5),
            last_cell_name: Some("fuel".to_string()),
            surface_id: Some(10),
        };
        // Should not panic
        lp.print_diagnostic();
    }

    #[test]
    fn test_lost_particle_diagnostic_minimal() {
        let lp = LostParticle {
            particle_type: ParticleType::Photon,
            position: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
            energy: 500.0,
            last_cell_index: None,
            last_cell_id: None,
            last_cell_name: None,
            surface_id: None,
        };
        // Should not panic even with all None fields
        lp.print_diagnostic();
    }

    #[test]
    fn test_lost_particle_cell_info_formatting() {
        // name + id
        let lp = LostParticle {
            particle_type: ParticleType::Neutron,
            position: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            energy: 1.0,
            last_cell_index: None,
            last_cell_id: Some(3),
            last_cell_name: Some("moderator".to_string()),
            surface_id: None,
        };
        lp.print_diagnostic();

        // id only
        let lp2 = LostParticle {
            last_cell_name: None,
            last_cell_id: Some(7),
            ..lp.clone()
        };
        lp2.print_diagnostic();

        // name only
        let lp3 = LostParticle {
            last_cell_name: Some("void".to_string()),
            last_cell_id: None,
            ..lp
        };
        lp3.print_diagnostic();
    }
}
