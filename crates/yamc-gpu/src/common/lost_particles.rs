//! Device-side lost-particle diagnostics (issue #289).
//!
//! A particle is *lost* when its position is inside no cell: the geometry
//! does not cover the space it reached. On the CPU this is a hard, loud
//! failure (`handle_lost_particle` in `yamc::transport`, which records a
//! `LostParticle` and aborts the run once `max_lost_particles` is
//! exceeded). The GPU kernels used to just end the history, so a model
//! with a geometry gap returned quietly wrong tallies on `compute='gpu'`
//! while the same model refused to run on `compute='cpu'`.
//!
//! This module gives both kernels the counting + record-keeping half of
//! the CPU behaviour, using the same atomic slot-reservation protocol as
//! [`crate::common::particle_bank`]:
//!
//! - `lost_count[0]` is a single-element `Atomic<u64>` holding every loss
//!   the launch saw (it may exceed the record capacity),
//! - `lost_f64` holds the first `capacity = lost_f64.len() / LOST_F64_STRIDE`
//!   records, stride-packed for diagnostics.
//!
//! The host reads both back per launch, turns the records into
//! `yamc::util::lost_particle::LostParticle` values and enforces
//! `max_lost_particles`. Only the counter is load-bearing for the abort
//! decision, so a launch that loses more particles than the record
//! capacity still fails correctly, just with fewer printable records
//! (this mirrors the CPU, which also caps what it prints).
//!
//! The last cell index is carried as an `f64` in the record rather than in
//! a parallel `&mut [u32]`: cell indices are small integers and therefore
//! exact in `f64`, and the kernels are close enough to the per-stage
//! storage-buffer descriptor budget (see
//! `crate::neutron::transport::KERNEL_STORAGE_BUFFER_COUNT`) that one
//! binding is worth saving.

use cubecl::prelude::*;

/// f64 fields per lost-particle record:
/// `[px, py, pz, dx, dy, dz, energy, last_cell]`.
///
/// `last_cell` is the cell index the particle was in before the step that
/// lost it, or [`LOST_NO_CELL`] when it had none (a source particle born
/// outside every cell).
pub const LOST_F64_STRIDE: usize = 8;

/// Sentinel written into the `last_cell` slot when the lost particle had
/// no previous cell. Matches `yamc_particle::particle::NO_CELL` widened to
/// `f64`, and is far outside any real cell index.
pub const LOST_NO_CELL: f64 = 4_294_967_295.0;

/// Records kept per launch. The abort decision only needs the counter, so
/// this bounds diagnostics, not correctness. 64 is well past the default
/// `max_lost_particles = 10`, so a run that aborts has every record that
/// led to the abort.
pub const LOST_RECORD_CAPACITY: usize = 64;

/// Record one lost particle: bump the launch counter and, if there is room,
/// write the diagnostic record.
///
/// `fetch_add` returns the pre-add value, used directly as the write slot
/// (the same reservation protocol as the particle bank). Callers must set
/// the history's `alive` flag to 0 themselves, exactly where they did
/// before, so the control flow of the transport loop is unchanged.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn record_lost(
    lost_count: &mut [Atomic<u64>],
    lost_f64: &mut [f64],
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    energy: f64,
    last_cell: f64,
) {
    let capacity = (lost_f64.len() / 8) as u64;
    let slot = lost_count[0].fetch_add(1u64);
    if slot < capacity {
        let f = (slot * 8u64) as usize;
        lost_f64[f] = px;
        lost_f64[f + 1] = py;
        lost_f64[f + 2] = pz;
        lost_f64[f + 3] = dx;
        lost_f64[f + 4] = dy;
        lost_f64[f + 5] = dz;
        lost_f64[f + 6] = energy;
        lost_f64[f + 7] = last_cell;
    }
}

/// Host-side view of one launch's lost-particle diagnostics.
#[derive(Debug, Clone, Default)]
pub struct LostParticleResult {
    /// Every loss the launch saw (may exceed `records.len()`).
    pub count: u64,
    /// Decoded records, oldest-reserved first, at most
    /// [`LOST_RECORD_CAPACITY`] of them.
    pub records: Vec<LostParticleRecord>,
}

/// One decoded lost-particle record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LostParticleRecord {
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    /// Cell index the particle came from, or `None` when it had none.
    pub last_cell_index: Option<usize>,
}

impl LostParticleResult {
    /// Decode a read-back `lost_f64` buffer, keeping only the slots the
    /// kernel actually wrote (`min(count, capacity)`).
    pub fn from_device(count: u64, lost_f64: &[f64]) -> Self {
        let capacity = lost_f64.len() / LOST_F64_STRIDE;
        let written = (count as usize).min(capacity);
        let records = (0..written)
            .map(|i| {
                let f = i * LOST_F64_STRIDE;
                let cell = lost_f64[f + 7];
                LostParticleRecord {
                    position: [lost_f64[f], lost_f64[f + 1], lost_f64[f + 2]],
                    direction: [lost_f64[f + 3], lost_f64[f + 4], lost_f64[f + 5]],
                    energy: lost_f64[f + 6],
                    last_cell_index: if cell == LOST_NO_CELL {
                        None
                    } else {
                        Some(cell as usize)
                    },
                }
            })
            .collect();
        Self { count, records }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_only_written_slots() {
        let mut buf = vec![0.0f64; 3 * LOST_F64_STRIDE];
        buf[0] = 1.0;
        buf[1] = 2.0;
        buf[2] = 3.0;
        buf[3] = 0.0;
        buf[4] = 0.0;
        buf[5] = 1.0;
        buf[6] = 1.4e7;
        buf[7] = 5.0;
        // Two losses reserved, so the third (all-zero) slot must be ignored.
        let out = LostParticleResult::from_device(2, &buf);
        assert_eq!(out.count, 2);
        assert_eq!(out.records.len(), 2);
        assert_eq!(out.records[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(out.records[0].energy, 1.4e7);
        assert_eq!(out.records[0].last_cell_index, Some(5));
    }

    #[test]
    fn count_beyond_capacity_keeps_capacity_records() {
        let buf = vec![0.0f64; 2 * LOST_F64_STRIDE];
        let out = LostParticleResult::from_device(999, &buf);
        assert_eq!(out.count, 999);
        assert_eq!(out.records.len(), 2);
    }

    #[test]
    fn no_cell_sentinel_decodes_to_none() {
        let mut buf = vec![0.0f64; LOST_F64_STRIDE];
        buf[7] = LOST_NO_CELL;
        let out = LostParticleResult::from_device(1, &buf);
        assert_eq!(out.records[0].last_cell_index, None);
    }
}
