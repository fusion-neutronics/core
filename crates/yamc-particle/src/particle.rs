use yamc_nuclide::nuclide_registry::NuclideId;

// `ParticleType` (the `{Neutron, Photon}` enum) is owned by
// yamc-nuclide because reaction-product code needs to encode emitted
// species without depending on this crate's runtime `Particle` state.
// Re-exported here so call sites that walk through
// `yamc::particle::ParticleType` keep resolving via the yamc re-export.
pub use yamc_nuclide::particle_type::ParticleType;

/// Sentinel for "no cached cell" on `Particle.current_cell_index` /
/// `Particle.previous_cell_index`. Chosen over `Option<usize>` to keep the
/// hot path branch-free on a plain integer compare and to shrink the
/// `Particle` struct (16 → 4 bytes per field) for better cache density and
/// GPU-friendliness. `u32` gives 4.29B cells, which overflows if exceeded.
pub const NO_CELL: u32 = u32::MAX;

/// Sentinel for "no last surface crossed" on `Particle.last_surface_id`.
/// Same rationale as `NO_CELL`.
pub const NO_SURFACE: u32 = u32::MAX;

/// Convert `Option<usize>` → `u32` with sentinel. Used at the boundary where
/// geometry / boundary-hit APIs still speak `Option<usize>`.
#[inline]
pub fn cell_index_to_u32(value: Option<usize>) -> u32 {
    match value {
        Some(idx) => idx as u32,
        None => NO_CELL,
    }
}

#[inline]
pub fn surface_id_to_u32(value: Option<usize>) -> u32 {
    match value {
        Some(idx) => idx as u32,
        None => NO_SURFACE,
    }
}

/// Convert back to `Option<usize>` for code that still uses the old API
/// (diagnostics, lost-particle records).
#[inline]
pub fn cell_index_to_option(value: u32) -> Option<usize> {
    if value == NO_CELL {
        None
    } else {
        Some(value as usize)
    }
}

#[inline]
pub fn surface_id_to_option(value: u32) -> Option<usize> {
    if value == NO_SURFACE {
        None
    } else {
        Some(value as usize)
    }
}

/// Sentinel for "no cached URR random number" on `Particle.urr_random`.
/// NaN is the natural sentinel for an absent f64: it's distinct from any
/// valid random number in [0, 1) and collapses `Option<f64>` (16 bytes) to
/// a plain `f64` (8 bytes), shrinking the `Particle` struct and giving a
/// GPU-friendly layout. CAUTION: `NaN != NaN`, so never test with `==` --
/// always use `.is_nan()` or `urr_to_option()`.
pub const NO_URR: f64 = f64::NAN;

/// Convert the `f64`-NaN-sentinel form used on `Particle.urr_random` to an
/// `Option<f64>` for the existing `Option<f64>` APIs in material/nuclide.
#[inline]
pub fn urr_to_option(value: f64) -> Option<f64> {
    if value.is_nan() {
        None
    } else {
        Some(value)
    }
}

/// Convert an `Option<f64>` back to the NaN-sentinel `f64` form.
#[inline]
pub fn urr_from_option(value: Option<f64>) -> f64 {
    value.unwrap_or(NO_URR)
}

/// A collision event for history tracking
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    pub energy_before: f64,
    pub energy_after: f64,
    pub reaction_mt: i32,
    pub nuclide: String,
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub particle_type: ParticleType,
    pub position: [f64; 3],
    pub last_position: [f64; 3], // Previous position for track-length tallying
    pub direction: [f64; 3],
    pub energy: f64,
    pub weight: f64, // Statistical weight of the particle
    pub alive: bool,
    /// Particle ID for tracking. `u32` (not `usize`) to shrink the struct by 4 bytes
    /// on 64-bit targets and give a GPU-friendly 32-bit layout; 4.29B IDs is plenty.
    /// This is distinct from the separate `u64` event-tracker ID space in `track.rs`.
    pub id: u32,
    /// Cached cell index; `NO_CELL` when unset (after initialization or surface crossing).
    pub current_cell_index: u32,
    /// Previous cell index, used for neighbor-list acceleration; `NO_CELL` when unset.
    pub previous_cell_index: u32,
    /// URR (Unresolved Resonance Range) random number for current collision.
    /// Preserves correlation between distance sampling and reaction sampling.
    /// This is cached per-energy: only resample when energy changes.
    /// Stored as `f64` with `NO_URR` (NaN) sentinel instead of `Option<f64>`
    /// to shrink the `Particle` struct by 8 bytes and keep the layout GPU-friendly.
    /// Test for "no value" with `.is_nan()` or `urr_to_option()`, never with `==`.
    pub urr_random: f64,
    /// Energy at which urr_random was last sampled.
    /// Used to detect when energy changes and URR random should be resampled.
    pub urr_energy: f64,
    /// Surface ID of the last surface crossed; `NO_SURFACE` when unset
    /// (used for lost-particle diagnostics).
    pub last_surface_id: u32,
    /// Parent nuclide for D1S decay photons.
    /// Set only on photons produced by the D1S method, used by ParentNuclideFilter.
    /// Stored as an interned `NuclideId` (2 bytes with niche-packed Option) so the
    /// hot path carries no heap pointer; resolve names via `NuclideRegistry`.
    pub parent_nuclide: Option<NuclideId>,
    /// Collision history for debugging (only populated when debug_history feature is enabled)
    #[cfg(feature = "debug_history")]
    pub history: Vec<CollisionEvent>,
}

impl Particle {
    pub fn new(position: [f64; 3], direction: [f64; 3], energy: f64) -> Self {
        Self {
            particle_type: ParticleType::Neutron,
            position,
            last_position: position, // Initialize to same as position
            direction,
            energy,
            weight: 1.0, // Default weight
            alive: true,
            id: 0,                        // Default ID
            current_cell_index: NO_CELL,  // Will be set on first transport step
            previous_cell_index: NO_CELL, // Set on surface crossings for neighbor acceleration
            urr_random: NO_URR,           // Set per-collision when in URR range
            urr_energy: -1.0,             // Invalid energy to force initial sampling
            last_surface_id: NO_SURFACE,  // Set on surface crossings for lost particle diagnostics
            parent_nuclide: None,         // Set only for D1S decay photons
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        }
    }

    /// Record a collision event in the particle's history
    #[cfg(feature = "debug_history")]
    pub fn record_collision(
        &mut self,
        energy_before: f64,
        energy_after: f64,
        reaction_mt: i32,
        nuclide: &str,
    ) {
        self.history.push(CollisionEvent {
            energy_before,
            energy_after,
            reaction_mt,
            nuclide: nuclide.to_string(),
        });
    }

    /// Print the particle history (for debugging)
    #[cfg(feature = "debug_history")]
    pub fn print_history(&self) {
        eprintln!(
            "=== Particle {} history ({} collisions) ===",
            self.id,
            self.history.len()
        );
        for (i, event) in self.history.iter().enumerate() {
            eprintln!(
                "  {:3}: E={:.4e} -> {:.4e} eV, MT={}, nuclide={}",
                i + 1,
                event.energy_before,
                event.energy_after,
                event.reaction_mt,
                event.nuclide
            );
        }
        eprintln!(
            "  Final: E={:.4e} eV, weight={}, alive={}",
            self.energy, self.weight, self.alive
        );
    }

    /// Move the particle along its current direction by the specified distance
    pub fn move_by(&mut self, distance: f64) {
        // Save current position as last position
        self.last_position = self.position;

        // Update position
        for i in 0..3 {
            self.position[i] += self.direction[i] * distance;
        }
    }
}
