//! Device-side particle bank: a fixed-capacity, append-only queue of
//! secondary / split particles for the GPU transport passes.
//!
//! This mirrors the *role* of yamc-physics's CPU `ParticleBank`, but the
//! container is necessarily backend-specific: the CPU bank is a LIFO
//! `Vec` stack drained depth-first, whereas the GPU bank is an unordered,
//! fixed-capacity structure-of-arrays that producer threads append to
//! concurrently via an atomic fetch-add slot reservation. The two share
//! *semantics* (a queue of weighted secondaries carrying a generation /
//! depth field) rather than code; the per-secondary *sampling* math is
//! what will be shared as `#[cube]` + CPU twins in later phases (the TTB
//! sampler pattern).
//!
//! # Why a general particle (not photon-only)
//!
//! The record carries a `ptype` tag (neutron / photon) and a `weight`, so
//! the same bank serves coupled neutron->photon secondaries today and
//! variance-reduction *splitting* (which banks weighted neutrons and
//! photons) and secondary-neutron multiplication later -- the append +
//! overflow + drain machinery is particle-agnostic by design.
//!
//! # Layout
//!
//! Records are packed into two arrays (rather than one buffer per field)
//! to keep the kernel's descriptor-binding budget small:
//! - `bank_f64`, stride [`BANK_F64_STRIDE`]: `[energy, px, py, pz, dx, dy,
//!   dz, weight]`
//! - `bank_u32`, stride [`BANK_U32_STRIDE`]: `[ptype, cell, seed, gen]`
//!
//! plus two single-element `Atomic<u64>` counters:
//! - `bank_count` -- total slot reservations (may exceed capacity),
//! - `bank_overflow` -- reservations dropped because the bank was full.
//!
//! # Append protocol (the one novel GPU primitive this module proves)
//!
//! A producer thread reserves a slot with `slot =
//! bank_count[0].fetch_add(1)` -- the *pre-add* return value, used
//! directly as the write index. It writes its record iff `slot <
//! capacity`, otherwise adds 1 to `bank_overflow` and writes nothing. No
//! existing kernel uses an atomic add's return value, so the
//! `gpu_particle_bank_append` test pins this behaviour (distinct slot per
//! thread, no clobber, exact overflow count) on real hardware.
//!
//! The host MUST clamp any drain launch to `min(bank_count, capacity)`
//! and treat `bank_overflow > 0` as a hard error -- never a silent drop
//! (this is strictly safer than the CPU bank's silent cap).

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// f64 fields per banked particle: `[energy, px, py, pz, dx, dy, dz, weight]`.
pub const BANK_F64_STRIDE: usize = 8;
/// u32 fields per banked particle: `[ptype, cell, seed, gen]`.
pub const BANK_U32_STRIDE: usize = 4;

/// `ptype` tag for a banked neutron.
pub const PTYPE_NEUTRON: u32 = 0;
/// `ptype` tag for a banked photon.
pub const PTYPE_PHOTON: u32 = 1;

/// `gen` tag on a banked (n,xn) secondary that overflowed the neutron kernel's
/// thread-private pending stack (issue #111 phase 2).
///
/// The `gen` field is informational (the host bounds the drain by counting
/// passes, not by reading it), so it doubles as a provenance tag: a record
/// carrying this value is a secondary of a history transported in THIS pass,
/// not a fission progeny, which is what lets the diagnostics separate the two
/// without a second counter. Distinct from the `0` on banked photons and the
/// `1` on fission progeny.
///
/// # `gen` is only a provenance tag on a banked NEUTRON
///
/// Read it together with `ptype`, always. On a banked PHOTON the slot means
/// something else entirely: the D1S decay-photon emission site stores the
/// parent-nuclide id there (`kernel.rs`, `bank_u32[u + 3] = parent_id`), so a
/// decay photon from parent id 2 carries the same bit pattern as a spilled
/// neutron. Counting spills without qualifying by `ptype` reported one phantom
/// spill per such photon and made the coupled dispatch refuse every D1S run.
pub const BANK_GEN_NXN_SPILL: u32 = 2;

/// Count the banked (n,xn) spill records in `bank_u32`, whose first `count`
/// stride-`BANK_U32_STRIDE` rows are live.
///
/// Lives beside the tags it interprets because the interpretation is not
/// obvious: `gen` is a provenance tag ONLY on a banked neutron, and a banked
/// D1S decay photon reuses the slot for its parent-nuclide id. A photon from
/// parent id 2 is therefore bit-identical to a spilled neutron in that field,
/// so both `ptype` and `gen` have to match.
pub fn count_nxn_spills(bank_u32: &[u32], count: u64) -> u64 {
    bank_u32
        .as_chunks::<BANK_U32_STRIDE>()
        .0
        .iter()
        .take(count as usize)
        .filter(|r| r[0] == PTYPE_NEUTRON && r[3] == BANK_GEN_NXN_SPILL)
        .count() as u64
}

/// Test/validation kernel: each thread appends one pre-built record from
/// the stride-packed input buffers into the bank, exercising the atomic
/// fetch-add slot reservation and the overflow path. The append block is
/// inlined here (the same block production code will inline at the
/// neutron-kernel spawn sites); it is the load-bearing primitive under
/// test.
#[cube(launch_unchecked)]
fn bank_append_kernel(
    in_f64: &[f64],
    in_u32: &[u32],
    bank_f64: &mut [f64],
    bank_u32: &mut [u32],
    bank_count: &mut [Atomic<u64>],
    bank_overflow: &mut [Atomic<u64>],
) {
    let n_in = in_u32.len() / 4;
    if ABSOLUTE_POS >= n_in {
        terminate!();
    }
    let capacity = (bank_f64.len() / 8) as u64;

    // Reserve a slot; `fetch_add` returns the pre-add value.
    let slot = bank_count[0].fetch_add(1u64);
    if slot < capacity {
        let f = (slot * 8u64) as usize;
        let fi = ABSOLUTE_POS * 8;
        bank_f64[f] = in_f64[fi];
        bank_f64[f + 1] = in_f64[fi + 1];
        bank_f64[f + 2] = in_f64[fi + 2];
        bank_f64[f + 3] = in_f64[fi + 3];
        bank_f64[f + 4] = in_f64[fi + 4];
        bank_f64[f + 5] = in_f64[fi + 5];
        bank_f64[f + 6] = in_f64[fi + 6];
        bank_f64[f + 7] = in_f64[fi + 7];
        let u = (slot * 4u64) as usize;
        let ui = ABSOLUTE_POS * 4;
        bank_u32[u] = in_u32[ui];
        bank_u32[u + 1] = in_u32[ui + 1];
        bank_u32[u + 2] = in_u32[ui + 2];
        bank_u32[u + 3] = in_u32[ui + 3];
    } else {
        bank_overflow[0].fetch_add(1u64);
    }
}

/// Result of [`run_bank_append`]: the post-launch counters and the bank
/// contents (slots `0..min(count, capacity)` are written; the rest are
/// zero).
#[derive(Debug, Clone)]
pub struct BankAppendResult {
    /// Total slot reservations (== number of input records).
    pub count: u64,
    /// Reservations dropped because the bank was full.
    pub overflow: u64,
    /// Bank f64 fields, stride [`BANK_F64_STRIDE`], length `capacity * 8`.
    pub bank_f64: Vec<f64>,
    /// Bank u32 fields, stride [`BANK_U32_STRIDE`], length `capacity * 4`.
    pub bank_u32: Vec<u32>,
}

/// Append every input record (stride-packed: `in_f64` stride
/// [`BANK_F64_STRIDE`], `in_u32` stride [`BANK_U32_STRIDE`]) into a fresh
/// bank of `capacity` slots, concurrently on the GPU. For test/validation.
pub fn run_bank_append(
    ctx: &GpuContext,
    in_f64: &[f64],
    in_u32: &[u32],
    capacity: usize,
) -> BankAppendResult {
    assert!(in_f64.len().is_multiple_of(BANK_F64_STRIDE));
    assert!(in_u32.len().is_multiple_of(BANK_U32_STRIDE));
    let n = in_u32.len() / BANK_U32_STRIDE;
    assert_eq!(
        in_f64.len() / BANK_F64_STRIDE,
        n,
        "f64/u32 record counts differ"
    );

    let client = ctx.client();
    let in_f64_h = client.create_from_slice(bytemuck::cast_slice(in_f64));
    let in_u32_h = client.create_from_slice(bytemuck::cast_slice(in_u32));
    // Zero-initialise the bank so unwritten slots read back as 0.
    let bank_f64_z = vec![0.0_f64; capacity * BANK_F64_STRIDE];
    let bank_u32_z = vec![0u32; capacity * BANK_U32_STRIDE];
    let bank_f64_h = client.create_from_slice(bytemuck::cast_slice(&bank_f64_z));
    let bank_u32_h = client.create_from_slice(bytemuck::cast_slice(&bank_u32_z));
    let count_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));
    let overflow_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        bank_append_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(in_f64_h, in_f64.len()),
            BufferArg::from_raw_parts(in_u32_h, in_u32.len()),
            BufferArg::from_raw_parts(bank_f64_h.clone(), capacity * BANK_F64_STRIDE),
            BufferArg::from_raw_parts(bank_u32_h.clone(), capacity * BANK_U32_STRIDE),
            BufferArg::from_raw_parts(count_h.clone(), 1),
            BufferArg::from_raw_parts(overflow_h.clone(), 1),
        );
    }

    let count = bytemuck::cast_slice::<u8, u64>(&client.read_one(count_h).unwrap())[0];
    let overflow = bytemuck::cast_slice::<u8, u64>(&client.read_one(overflow_h).unwrap())[0];
    let bank_f64 = bytemuck::cast_slice::<u8, f64>(&client.read_one(bank_f64_h).unwrap()).to_vec();
    let bank_u32 = bytemuck::cast_slice::<u8, u32>(&client.read_one(bank_u32_h).unwrap()).to_vec();
    BankAppendResult {
        count,
        overflow,
        bank_f64,
        bank_u32,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};
    use std::collections::HashSet;

    /// Build `n` input records, each tagged with its index as a sentinel
    /// in the `seed` slot (u32 field 2) and `energy = index * 1000` so we
    /// can detect clobbered / duplicated / lost writes after the parallel
    /// append.
    fn sentinel_records(n: usize) -> (Vec<f64>, Vec<u32>) {
        let mut f = Vec::with_capacity(n * BANK_F64_STRIDE);
        let mut u = Vec::with_capacity(n * BANK_U32_STRIDE);
        for i in 0..n {
            let fi = i as f64;
            // energy, px, py, pz, dx, dy, dz, weight
            f.extend_from_slice(&[fi * 1000.0, fi, -fi, fi + 0.5, 1.0, 0.0, 0.0, 1.0]);
            // ptype, cell, seed(=sentinel index), gen
            u.extend_from_slice(&[PTYPE_PHOTON, (i % 7) as u32, i as u32, 0]);
        }
        (f, u)
    }

    /// Atomic fetch-add slot reservation: distinct slot per thread (no
    /// clobber / duplication / loss), and exact overflow accounting when
    /// the bank is over-filled. This pins the one novel GPU primitive --
    /// using an atomic add's return value as a write index -- on real
    /// hardware.
    #[test]
    fn gpu_particle_bank_append() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // --- Under capacity: every record lands, no overflow. ---
        let (inf, inu) = sentinel_records(100);
        let cap = 256;
        let r = run_bank_append(&ctx, &inf, &inu, cap);
        assert_eq!(r.count, 100, "all 100 reservations counted");
        assert_eq!(r.overflow, 0, "no overflow under capacity");
        // Slots 0..100 hold a permutation of sentinels {0..100}; the rest
        // stay zero (and energy round-trips through the f64 packing).
        let written: Vec<u32> = (0..100)
            .map(|s| r.bank_u32[s * BANK_U32_STRIDE + 2])
            .collect();
        let set: HashSet<u32> = written.iter().copied().collect();
        assert_eq!(set.len(), 100, "100 distinct sentinels (no clobber/dup)");
        assert_eq!(
            set,
            (0..100u32).collect::<HashSet<_>>(),
            "exactly sentinels 0..100"
        );
        for s in 0..100 {
            let sentinel = r.bank_u32[s * BANK_U32_STRIDE + 2];
            let energy = r.bank_f64[s * BANK_F64_STRIDE];
            assert_eq!(
                energy,
                sentinel as f64 * 1000.0,
                "f64 record packing must match its u32 sentinel at slot {s}"
            );
        }

        // --- Over capacity: exact overflow, written slots still distinct. ---
        let n = 300usize;
        let cap = 128usize;
        let (inf, inu) = sentinel_records(n);
        let r = run_bank_append(&ctx, &inf, &inu, cap);
        assert_eq!(r.count, n as u64, "all {n} reservations counted");
        assert_eq!(
            r.overflow,
            (n - cap) as u64,
            "overflow == reservations beyond capacity"
        );
        let written: Vec<u32> = (0..cap)
            .map(|s| r.bank_u32[s * BANK_U32_STRIDE + 2])
            .collect();
        let set: HashSet<u32> = written.iter().copied().collect();
        assert_eq!(
            set.len(),
            cap,
            "{cap} distinct sentinels written (no clobber/dup)"
        );
        assert!(
            set.iter().all(|&s| (s as usize) < n),
            "every written sentinel is a real input index"
        );
        println!(
            "particle bank append: under-cap 100/256 ok, over-cap {n}/{cap} ok (overflow {})",
            r.overflow
        );
    }
}

#[cfg(test)]
mod spill_count_tests {
    use super::*;

    /// One row of the u32 side of the bank.
    fn row(ptype: u32, gen: u32) -> [u32; BANK_U32_STRIDE] {
        // [ptype, cell, seed, gen]
        [ptype, 7, 0xDEAD_BEEF, gen]
    }

    fn bank(rows: &[[u32; BANK_U32_STRIDE]]) -> Vec<u32> {
        rows.iter().flatten().copied().collect()
    }

    /// A banked D1S decay photon stores its PARENT-NUCLIDE ID in the `gen`
    /// slot, so parent id 2 collides with `BANK_GEN_NXN_SPILL`. Counting on
    /// `gen` alone reported one phantom spill per such photon, which made the
    /// coupled dispatch refuse every D1S run outright.
    #[test]
    fn decay_photon_with_parent_id_two_is_not_a_spill() {
        let b = bank(&[
            row(PTYPE_PHOTON, BANK_GEN_NXN_SPILL),
            row(PTYPE_PHOTON, BANK_GEN_NXN_SPILL),
        ]);
        assert_eq!(count_nxn_spills(&b, 2), 0);
    }

    /// A genuine spill must still be counted: the fix must not mask the case
    /// the counter exists for.
    #[test]
    fn spilled_neutron_is_counted() {
        let b = bank(&[row(PTYPE_NEUTRON, BANK_GEN_NXN_SPILL)]);
        assert_eq!(count_nxn_spills(&b, 1), 1);
    }

    /// Mixed bank: only the neutron rows tagged as spills count, and photons
    /// with any parent id are ignored.
    #[test]
    fn counts_only_spilled_neutrons_in_a_mixed_bank() {
        let b = bank(&[
            row(PTYPE_PHOTON, 0),                   // ordinary secondary photon
            row(PTYPE_PHOTON, BANK_GEN_NXN_SPILL),  // D1S photon, parent id 2
            row(PTYPE_NEUTRON, 1),                  // fission progeny
            row(PTYPE_NEUTRON, BANK_GEN_NXN_SPILL), // the one real spill
            row(PTYPE_PHOTON, 5),                   // D1S photon, parent id 5
        ]);
        assert_eq!(count_nxn_spills(&b, 5), 1);
    }

    /// Rows past `count` are stale slots from an earlier launch and must not
    /// be read, even when they carry a spill pattern.
    #[test]
    fn rows_past_the_live_count_are_ignored() {
        let b = bank(&[
            row(PTYPE_NEUTRON, BANK_GEN_NXN_SPILL),
            row(PTYPE_NEUTRON, BANK_GEN_NXN_SPILL),
        ]);
        assert_eq!(count_nxn_spills(&b, 1), 1);
        assert_eq!(count_nxn_spills(&b, 0), 0);
    }
}
