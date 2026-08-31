//! PCG-32 random number generator (64-bit state, 32-bit output word,
//! PCG-XSH-RR 64/32; issue #274) -- bit-identical CPU and GPU
//! implementations.
//!
//! The shared collision path threads one 64-bit PCG state per history on
//! both backends. Per-history seeds stay 32-bit (built by the shared
//! `history_seed(base_seed, global_index)`) and are expanded to the 64-bit
//! state via splitmix64 (`expand_seed`) at history start, identically on
//! CPU and GPU, so the two backends consume the same stream. The original
//! GPU port used a 32-bit-state PCG; its marginal rare-tail
//! equidistribution inflated std_dev on rare tallies, which is why the
//! state is 64-bit now.
//!
//! `GpuRng::next_u32` here is the byte-for-byte reference for the
//! `pcg32_kernel` in [`crate::common::pcg32`]. A divergence between them
//! is caught by the `gpu_pcg32_matches_cpu_byte_for_byte` test.

use bytemuck::{Pod, Zeroable};

/// PCG constants and host-side helpers. Owned by the dependency-free
/// yamc-rng crate so the CPU companion (which calls into
/// `yamc_physics::gpu::flat`) and the cubecl kernel draw from the same
/// source of truth: the `#[cube]` twins in
/// [`crate::common::pcg32`] import these same `u64` constants directly.
/// Host code must use THIS `expand_seed` (wrapping arithmetic); the
/// `#[cube]` twin in `pcg32` uses plain ops and would overflow-panic in
/// debug builds if called on the host.
pub use yamc_rng::{
    expand_seed, fold_base_seed, history_seed, pcg_xsh_rr, secondary_seed, HISTORY_SEED_GOLDEN,
    PCG_INCR, PCG_MULT, SECONDARY_SEED_GOLDEN, SECONDARY_SEED_MIX_A, SECONDARY_SEED_MIX_B,
};

/// 64-bit PCG (PCG-XSH-RR 64/32) with the same algorithm as `pcg_next` in
/// [`crate::common::pcg32`] (issue #274). Stored as a `Pod` u64 so a
/// `Vec<GpuRng>` could be uploaded as a per-thread state buffer.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRng {
    pub state: u64,
}

impl GpuRng {
    /// Construct from a 32-bit per-history seed, expanded to the 64-bit state
    /// via splitmix64 (the same expansion the GPU kernel applies to `seeds[i]`),
    /// so this stays the byte-for-byte reference for the kernel's per-history
    /// stream.
    #[inline]
    pub fn new(seed: u32) -> Self {
        Self {
            state: expand_seed(seed),
        }
    }

    /// Construct directly from a raw 64-bit state (no seed expansion). For
    /// validation harnesses that thread a known state.
    #[inline]
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Advance the LCG and return a uniformly-distributed `u32`. Bit
    /// equivalent to one step of `pcg_next`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        pcg_xsh_rr(oldstate)
    }

    /// Uniformly-distributed `f32` in `[0, 1)`. Top 24 bits of `next_u32`
    /// fill the f32 mantissa exactly.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `next_f32` always lands in `[0, 1)`.
    #[test]
    fn cpu_f32_in_unit_interval() {
        let mut r = GpuRng::new(0xDEADBEEF);
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
        }
    }

    /// Mean of a large `next_f32` sample is close to 0.5 (uniform).
    #[test]
    fn cpu_f32_mean_near_half() {
        let mut r = GpuRng::new(7);
        let n = 100_000;
        let mean = (0..n).map(|_| r.next_f32() as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.005,
            "mean {mean} too far from 0.5 over {n} samples"
        );
    }

    /// Sequence determinism -- same seed gives same output across runs.
    /// The actual byte-for-byte cross-check against the GPU kernel lives
    /// in the kernel module's tests.
    #[test]
    fn cpu_seed_determinism() {
        let mut a = GpuRng::new(42);
        let mut b = GpuRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }
}
