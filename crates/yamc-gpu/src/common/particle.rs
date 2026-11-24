//! GPU-side particle representation.
//!
//! `GpuParticle` is the Pod (bytemuck), `repr(C)`, fixed-layout mirror of
//! `yamc::particle::Particle`. It exists *alongside* the CPU type -- the
//! CPU `Particle` is unchanged. This is deliberate: the CPU hot loop has
//! 30+ uses of `particle.alive: bool` and the existing `Option<…>` /
//! enum fields fit the inner-loop's branch patterns cleanly. Forcing the
//! CPU struct into Pod-compatible primitives just to share one type with
//! the GPU would touch every transport code path for no CPU benefit.
//!
//! Instead we define a separate Pod struct here, `From<&Particle>`-convert
//! at the GPU upload boundary, and keep the CPU representation untouched.
//! Cost: one struct-to-struct memcpy per particle at upload, which is
//! free compared to kernel time and amortized across an entire batch.
//!
//! # Sentinels
//!
//! Following the `NO_CELL` / `NO_URR` pattern already established in
//! `yamc::particle`:
//! - `alive` is a `u32` flag (0 = dead, 1 = alive). u32 not u8 for clean
//!   8-byte alignment alongside the other u32 fields and zero padding.
//! - `particle_type` is a `u16`: 0 = Neutron, 1 = Photon. Same as the
//!   CPU `ParticleType` enum's natural discriminants.
//! - `parent_nuclide` is a `u16` with `NO_PARENT_NUCLIDE = 0` meaning
//!   "no D1S parent". Mirrors the `NonZeroU16` niche-packing of the CPU
//!   `Option<NuclideId>` -- `NuclideId(NonZeroU16)` is 1..=65535, so 0 is
//!   the only safe sentinel.
//!
//! # Layout
//!
//! Fields are ordered largest-first (f64 arrays → f64 → u32 → u16) so
//! `repr(C)` emits zero internal padding. Total size is 128 bytes,
//! 8-byte aligned. A compile-time assertion at the bottom of this file
//! pins both -- adding a field that breaks the layout fails to compile.

use bytemuck::{Pod, Zeroable};
use yamc_nuclide::nuclide_registry::NuclideId;
use yamc_nuclide::particle_type::ParticleType;
use yamc_particle::Particle;

/// Sentinel for "no parent nuclide" on `GpuParticle.parent_nuclide`. The
/// CPU side stores `Option<NuclideId>` where `NuclideId(NonZeroU16)`, so
/// 0 cannot be a valid id and is the natural "absent" marker.
pub const NO_PARENT_NUCLIDE: u16 = 0;

/// `particle_type` discriminants. Match `ParticleType` declaration order.
pub const PARTICLE_TYPE_NEUTRON: u16 = 0;
pub const PARTICLE_TYPE_PHOTON: u16 = 1;

/// GPU-uploadable particle state. Pod, `repr(C)`, fixed 128-byte layout.
/// See module docs for the reasoning behind the field ordering and
/// sentinels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f64; 3],
    pub last_position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    pub weight: f64,
    pub urr_random: f64,
    pub urr_energy: f64,
    pub id: u32,
    pub current_cell_index: u32,
    pub previous_cell_index: u32,
    pub last_surface_id: u32,
    pub alive: u32,
    pub particle_type: u16,
    pub parent_nuclide: u16,
}

impl From<&Particle> for GpuParticle {
    fn from(p: &Particle) -> Self {
        let particle_type = match p.particle_type {
            ParticleType::Neutron => PARTICLE_TYPE_NEUTRON,
            ParticleType::Photon => PARTICLE_TYPE_PHOTON,
        };
        let parent_nuclide = p
            .parent_nuclide
            .map(NuclideId::get)
            .unwrap_or(NO_PARENT_NUCLIDE);
        GpuParticle {
            position: p.position,
            last_position: p.last_position,
            direction: p.direction,
            energy: p.energy,
            weight: p.weight,
            urr_random: p.urr_random,
            urr_energy: p.urr_energy,
            id: p.id,
            current_cell_index: p.current_cell_index,
            previous_cell_index: p.previous_cell_index,
            last_surface_id: p.last_surface_id,
            alive: u32::from(p.alive),
            particle_type,
            parent_nuclide,
        }
    }
}

// Compile-time layout pins. Any future field reorder, type widening, or
// added field that breaks size/alignment fails to build.
const _: () = assert!(std::mem::size_of::<GpuParticle>() == 128);
const _: () = assert!(std::mem::align_of::<GpuParticle>() == 8);
// The Pod derive itself enforces no padding, but pin offsets of the most
// hot-path fields explicitly so the kernel SSBO layout stays predictable.
const _: () = assert!(std::mem::offset_of!(GpuParticle, position) == 0);
const _: () = assert!(std::mem::offset_of!(GpuParticle, energy) == 72);
const _: () = assert!(std::mem::offset_of!(GpuParticle, alive) == 120);

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_particle::particle::{NO_CELL, NO_SURFACE, NO_URR};

    /// Spot-check that every field round-trips a representative value
    /// from the CPU `Particle` to its GPU counterpart, including the
    /// sentinel-mapped fields (alive, particle_type, parent_nuclide).
    #[test]
    fn particle_to_gpu_particle_field_round_trip() {
        // Build a Particle with non-default values everywhere so a
        // copy-paste bug in the conversion would change at least one field.
        let mut p = Particle::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 1e6);
        p.last_position = [0.5, 1.5, 2.5];
        p.weight = 0.75;
        p.alive = false;
        p.id = 12345;
        p.current_cell_index = 7;
        p.previous_cell_index = 3;
        p.urr_random = 0.42;
        p.urr_energy = 1.5e6;
        p.last_surface_id = 11;
        p.particle_type = ParticleType::Photon;
        // `parent_nuclide` left as None -- exercise the NO_PARENT sentinel.

        let g: GpuParticle = (&p).into();

        assert_eq!(g.position, [1.0, 2.0, 3.0]);
        assert_eq!(g.last_position, [0.5, 1.5, 2.5]);
        assert_eq!(g.direction, [0.0, 0.0, 1.0]);
        assert_eq!(g.energy, 1e6);
        assert_eq!(g.weight, 0.75);
        assert_eq!(g.alive, 0);
        assert_eq!(g.id, 12345);
        assert_eq!(g.current_cell_index, 7);
        assert_eq!(g.previous_cell_index, 3);
        assert_eq!(g.urr_random, 0.42);
        assert_eq!(g.urr_energy, 1.5e6);
        assert_eq!(g.last_surface_id, 11);
        assert_eq!(g.particle_type, PARTICLE_TYPE_PHOTON);
        assert_eq!(g.parent_nuclide, NO_PARENT_NUCLIDE);
    }

    #[test]
    fn alive_neutron_default_maps_correctly() {
        // A freshly-constructed neutron is alive -- make sure the
        // bool→u32 conversion keeps it that way.
        let p = Particle::new([0.0; 3], [1.0, 0.0, 0.0], 1.0);
        let g: GpuParticle = (&p).into();
        assert_eq!(g.alive, 1);
        assert_eq!(g.particle_type, PARTICLE_TYPE_NEUTRON);
    }

    /// Sentinel pass-through: `NO_CELL`, `NO_SURFACE`, `NO_URR` on the
    /// CPU side are plain primitives, not `Option<…>`, so they should
    /// flow through unchanged.
    #[test]
    fn unset_sentinels_flow_through() {
        let p = Particle::new([0.0; 3], [1.0, 0.0, 0.0], 1.0);
        let g: GpuParticle = (&p).into();
        assert_eq!(g.current_cell_index, NO_CELL);
        assert_eq!(g.previous_cell_index, NO_CELL);
        assert_eq!(g.last_surface_id, NO_SURFACE);
        assert!(g.urr_random.is_nan(), "NO_URR (NaN) should pass through");
        let _ = NO_URR; // referenced for documentation: the value above is exactly this sentinel
    }

    /// Pod is supposed to mean cast_slice works. Verify by uploading an
    /// arbitrary GpuParticle, casting back, and checking equality of the
    /// raw bytes -- `==` on the struct fails because `NO_URR` is `NaN`
    /// and `NaN != NaN` per IEEE-754, so we compare bytes directly.
    #[test]
    fn pod_byte_roundtrip() {
        let p = Particle::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 14.06e6);
        let g: GpuParticle = (&p).into();
        let bytes: &[u8] = bytemuck::bytes_of(&g);
        assert_eq!(bytes.len(), 128);
        let back: GpuParticle = *bytemuck::from_bytes::<GpuParticle>(bytes);
        let back_bytes: &[u8] = bytemuck::bytes_of(&back);
        assert_eq!(bytes, back_bytes);
    }
}
