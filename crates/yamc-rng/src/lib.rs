//! 64-bit PCG (PCG-XSH-RR 64/32) random number generator shared between the
//! CPU transport path and the GPU CPU-companion path (issue #111 / #274).
//!
//! The CPU companion in `yamc-gpu` must produce the same bit pattern as the
//! cubecl kernel. The kernel uses these constants as `u64` literals inside
//! `#[cube]` bodies; the CPU companion uses the same constants in regular Rust.
//! The `#[cube]` twin has to be written separately (a cubecl kernel body cannot
//! call ordinary Rust), but the constant values are owned here so a single
//! change propagates to both call sites. This crate is a leaf with no
//! dependencies precisely so every backend can reach the one stream.
//!
//! # Why 64-bit state (issue #274)
//!
//! The generator previously kept a 32-bit state (period `2^32`), whose limited
//! equidistribution over-disperses the rare tail that dominates an absorbing /
//! deep-penetration tally's variance -- inflating the reported std_dev (means
//! stayed unbiased) versus the CPU's 64-bit `FastRng` outer loop. Moving to a
//! 64-bit state (PCG-XSH-RR 64->32, period `2^64`, the same LCG constants as
//! `FastRng`) removes that gap while keeping the shared CPU/GPU stream
//! bit-identical (both seed via [`expand_seed`] and step this same generator).
//!
//! From O'Neill's PCG paper, the "XSH-RR 64/32" variant.

/// LCG multiplier (PCG / `FastRng` 64-bit constant).
pub const PCG_MULT: u64 = 6_364_136_223_846_793_005;
/// LCG increment (must be odd). PCG's default stream constant.
pub const PCG_INCR: u64 = 1_442_695_040_888_963_407;

/// Expand a 32-bit per-history seed into a well-mixed 64-bit PCG state via
/// splitmix64. The CPU driver and the GPU kernel MUST apply this identically to
/// the SAME per-history seed (always built by [`history_seed`]) so the two
/// backends consume the identical 64-bit stream and the #40 per-history
/// bit-identity holds.
#[inline]
pub fn expand_seed(seed: u32) -> u64 {
    let mut z = (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Odd multiplier that spreads consecutive history indices across the 32-bit
/// per-history seed space (Knuth's `2^32 / phi`). Odd, so multiplying by it is a
/// bijection on `u32`: distinct global indices always get distinct seeds.
pub const HISTORY_SEED_GOLDEN: u32 = 2_654_435_761;

/// Fold a run's 64-bit base seed (`TransportSettings::seed`) into the 32 bits a
/// per-history seed carries.
///
/// The per-history seed is a `u32` (that is what the GPU seed buffer holds), so
/// the wider base seed has to be reduced. Reducing by truncation would make
/// seeds that differ only above bit 32 collide, so instead the base seed goes
/// through the splitmix64 finaliser (full 64-bit avalanche) and the two halves
/// of the result are XOR-folded. Every bit of the base seed therefore influences
/// every bit of the fold, and two base seeds differing in a single bit give
/// unrelated folds. The reduction is 64 -> 32 bits, so collisions exist in
/// principle (about `2^-32` for a random pair); that is inherent to a 32-bit
/// per-history seed and is the same collision probability two arbitrary base
/// seeds already had.
#[inline]
pub fn fold_base_seed(base_seed: u64) -> u32 {
    let mut z = base_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z ^ (z >> 32)) as u32
}

/// THE single definition of per-history collision seeding (issue #315).
///
/// Every backend derives a history's collision-physics RNG state as
/// `expand_seed(history_seed(base_seed, global_index))`:
///
/// * the CPU transport loop, in `yamc::model`'s per-particle fold; and
/// * the GPU `seeds` buffer, built host-side in `yamc::gpu::translate`
///   (`sample_initial_particles_for_batch` / `_for_chunk`) and consumed by the
///   cubecl kernel as `expand_seed(seeds[i])`.
///
/// Both call this function, so the two backends cannot drift and the issue-#40
/// matched-stream per-history bit-identity holds by construction.
///
/// # Shape
///
/// ```text
/// history_seed(base, i) = fold_base_seed(base) ^ (i as u32 * HISTORY_SEED_GOLDEN)
/// ```
///
/// * It is a PURE function of `(base_seed, global_index)`. Nothing about how the
///   histories are grouped enters: not the chunk size, not the batch index, not
///   the MPI rank or the rank's slice of the index space, not the total history
///   count. Renumbering the partitioning therefore cannot change a result.
/// * For a fixed base seed it is injective in `global_index` (mod `2^32`): the
///   multiply is by an odd constant and the XOR is by a constant, so both steps
///   are bijections and no two histories of a run share a stream (up to `2^32`
///   histories, unchanged from before).
/// * Changing `base_seed` changes every history's collision stream. Before #315
///   the base seed was absent here, so re-running with a different `seed`
///   re-sampled only the source birth (the `FastRng` stream) and left every
///   collision realisation identical, which made a multi-seed spread a severe
///   under-estimate of the run-to-run error.
///
/// # Limit of a 32-bit per-history seed
///
/// The seed is a `u32` because that is what the GPU seed buffer carries, so two
/// runs of `N` histories with different base seeds still share about
/// `N^2 / 2^32` collision streams by the birthday argument (about 230 of a
/// million, 0.02%, and those histories are born differently anyway because the
/// `FastRng` source stream also moved). That floor belongs to the seed WIDTH,
/// not to this mixing function: any 32-bit per-history key has it. Before #315
/// the overlap between two runs was 100%.
#[inline]
pub fn history_seed(base_seed: u64, global_index: u64) -> u32 {
    fold_base_seed(base_seed) ^ (global_index as u32).wrapping_mul(HISTORY_SEED_GOLDEN)
}

/// Odd multiplier that spreads a walk's secondary ordinals across the 32-bit
/// seed space before the finaliser runs. Odd, so `ordinal -> ordinal * this` is
/// a bijection on `u32` and no two secondaries of one walk collide.
pub const SECONDARY_SEED_GOLDEN: u32 = 0x9E37_79B9;
/// First multiplier of the 32-bit (murmur3-style) finaliser in
/// [`secondary_seed`]. Owned here so the `#[cube]` twin in
/// `yamc_gpu::common::pcg32` imports the same value.
pub const SECONDARY_SEED_MIX_A: u32 = 0x85EB_CA6B;
/// Second multiplier of the finaliser in [`secondary_seed`].
pub const SECONDARY_SEED_MIX_B: u32 = 0xC2B2_AE35;

/// THE single definition of per-SECONDARY collision seeding (issue #111).
///
/// An (n,xn) reaction produces extra neutrons that both backends transport
/// INSIDE the parent history: the CPU banks them on a LIFO stack
/// (`yamc_physics::util::bank::ParticleBank`), the GPU queues them in an
/// in-thread FIFO. Before this function existed a secondary simply CONTINUED
/// the parent's PCG state at the moment it was popped, so what a secondary
/// sampled depended on WHEN it was scheduled: the two backends drained in
/// different orders and so produced different physics from the same collision.
///
/// Every secondary now gets a stream keyed on its IDENTITY instead. This is the
/// same architecture OpenMC uses (`init_particle_seeds` in `src/random_lcg.cpp`
/// derives a particle's streams from its id by skip-ahead), which is why OpenMC
/// can offer both a per-particle LIFO secondary bank and a shared flat bank
/// drained by offset and get identical results from either.
///
/// # Identity
///
/// A secondary is identified RECURSIVELY, by `(the seed of the walk that
/// created it, its ordinal among that walk's secondaries)`. The root of the
/// recursion is the source particle, whose seed is
/// [`history_seed(base_seed, global_index)`](history_seed).
///
/// The recursion is what makes the identity order-INDEPENDENT at every depth. A
/// FLAT `(history, n-th secondary created in this history)` counter is
/// order-independent only one level down: with two secondaries `A` and `B` of
/// the primary, a LIFO drain transports `B` first, so `B`'s own child takes
/// flat ordinal 2, while a FIFO drain gives flat ordinal 2 to a child of `A`.
/// Keying on the PARENT's seed plus the ordinal WITHIN that parent removes the
/// schedule from the key: a walk always numbers its own secondaries `0, 1, ...`
/// in the order it creates them, whenever it happens to run.
///
/// # Shape
///
/// ```text
/// secondary_seed(parent, k) = murmur3_fmix32(parent + (k + 1) * GOLDEN)
/// ```
///
/// * For a fixed parent it is injective in `k`: the multiply is by an odd
///   constant, the add is by a constant and the finaliser is a bijection on
///   `u32`, so a walk's secondaries never share a stream with each other.
/// * `k + 1` (rather than `k`) keeps secondary 0 off the parent's own key.
/// * A 32-bit finaliser (two `u32` multiplies), not the 64-bit splitmix64 of
///   [`expand_seed`]: this runs inside the GPU kernel, where 64-bit multiplies
///   are emulated. It is the same mixer `yamc::gpu::dispatch`'s
///   `split_progeny_seed` uses for the device bank's weight-split copies.
/// * Across DIFFERENT parents the streams collide at the `2^-32` birthday floor
///   that any 32-bit per-particle seed has (see [`history_seed`]); the GPU seed
///   buffer is `u32`, which is what fixes that width.
#[inline]
pub fn secondary_seed(parent_seed: u32, ordinal: u32) -> u32 {
    let mut z =
        parent_seed.wrapping_add(ordinal.wrapping_add(1).wrapping_mul(SECONDARY_SEED_GOLDEN));
    z = (z ^ (z >> 16)).wrapping_mul(SECONDARY_SEED_MIX_A);
    z = (z ^ (z >> 13)).wrapping_mul(SECONDARY_SEED_MIX_B);
    z ^ (z >> 16)
}

/// Advance the inline PCG state and return a uniform `(0, 1]` draw.
///
/// Bit-identical to the inline PCG step used inside the GPU kernel's `#[cube]`
/// body: read state, apply the XSH-RR 64->32 output permutation, advance,
/// scale. The `+1` / `/ 4_294_967_297.0` shape avoids returning exactly `0.0`
/// (matters for `ln(xi)` rejection loops like Maxwell/Watt).
#[inline]
pub fn next_xi(state: &mut u64) -> f64 {
    let r = pcg_xsh_rr(*state);
    *state = state.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    (r as f64 + 1.0) * (1.0 / 4_294_967_297.0)
}

/// PCG-XSH-RR 64->32 output permutation of a state word (no advance). Kept as a
/// standalone helper so the GPU `#[cube]` twin can mirror it exactly.
#[inline]
pub fn pcg_xsh_rr(s: u64) -> u32 {
    let xorshifted = (((s >> 18) ^ s) >> 27) as u32;
    let rot = (s >> 59) as u32;
    // rotate-right by `rot`; (32 - rot) & 31 handles rot == 0 (no-op rotate).
    (xorshifted >> rot) | (xorshifted << ((32u32.wrapping_sub(rot)) & 31))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_seed_is_well_mixed_and_distinct() {
        // Adjacent seeds must map to very different 64-bit states (avalanche),
        // so per-history streams don't start near each other.
        let a = expand_seed(0);
        let b = expand_seed(1);
        assert_ne!(a, b);
        assert!(
            (a ^ b).count_ones() > 16,
            "poor avalanche between adjacent seeds"
        );
    }

    /// Issue #315: the base seed must reach the collision stream. Two runs that
    /// differ only in `TransportSettings::seed` must give every history a
    /// different PCG state, otherwise a multi-seed spread measures only the
    /// source sampling.
    #[test]
    fn history_seed_depends_on_the_base_seed() {
        for i in 0..1000u64 {
            let a = history_seed(1, i);
            let b = history_seed(2, i);
            assert_ne!(a, b, "history {i} kept its stream when the base seed moved");
            let sa = expand_seed(a);
            let sb = expand_seed(b);
            assert!(
                (sa ^ sb).count_ones() > 16,
                "poor avalanche between base seeds for history {i}"
            );
        }
    }

    /// For a fixed run, distinct histories must keep distinct streams: the
    /// multiply is by an odd constant and the XOR by a constant, so the map is a
    /// bijection on `u32`.
    #[test]
    fn history_seed_is_injective_in_the_index() {
        use std::collections::HashSet;
        let seen: HashSet<u32> = (0..100_000u64)
            .map(|i| history_seed(0xDEAD_BEEF, i))
            .collect();
        assert_eq!(
            seen.len(),
            100_000,
            "per-history seeds collided within a run"
        );
    }

    /// The seed depends on the GLOBAL history index alone, never on how the
    /// histories were partitioned into chunks / batches / MPI ranks.
    #[test]
    fn history_seed_is_partition_independent() {
        let base = 4242u64;
        const N: usize = 4096;
        // One undivided sweep of the index space.
        let flat: Vec<u32> = (0..N as u64).map(|g| history_seed(base, g)).collect();
        // The same index space cut into chunks of any size (or split across MPI
        // ranks, which is the same arithmetic) must reproduce it entry for
        // entry. A chunk / batch / rank term in the seed would break this.
        for &chunk in &[1usize, 7, 64, 1000, N] {
            let mut split = Vec::with_capacity(N);
            let mut start = 0usize;
            while start < N {
                let n = chunk.min(N - start);
                for i in 0..n {
                    split.push(history_seed(base, (start + i) as u64));
                }
                start += n;
            }
            assert_eq!(split, flat, "chunking by {chunk} changed the seeds");
        }
        // The index enters only mod 2^32 (the seed is 32-bit), which the call
        // sites' u32 offset arithmetic already assumes.
        assert_eq!(history_seed(base, 5), history_seed(base, 5 + (1u64 << 32)));
    }

    /// The 64 -> 32 bit fold must use every bit of the base seed: seeds that
    /// differ only in their high half must not collide.
    #[test]
    fn fold_base_seed_uses_the_high_half() {
        assert_ne!(fold_base_seed(1), fold_base_seed(1 | (1 << 40)));
        assert_ne!(fold_base_seed(0), fold_base_seed(1 << 63));
        let a = fold_base_seed(7);
        let b = fold_base_seed(8);
        assert!((a ^ b).count_ones() > 8, "poor avalanche in the base fold");
    }

    /// A walk's secondaries must never share a stream with each other: for a
    /// fixed parent the map is a bijection on `u32`.
    #[test]
    fn secondary_seed_is_injective_in_the_ordinal() {
        use std::collections::HashSet;
        for parent in [0u32, 1, 0xDEAD_BEEF, u32::MAX] {
            let seen: HashSet<u32> = (0..100_000u32).map(|k| secondary_seed(parent, k)).collect();
            assert_eq!(
                seen.len(),
                100_000,
                "secondary seeds collided within one walk (parent {parent})"
            );
        }
    }

    /// A secondary must not inherit its parent's stream, and adjacent ordinals
    /// must land far apart (avalanche), so sibling secondaries of one collision
    /// are not near-duplicates of each other.
    #[test]
    fn secondary_seed_is_well_separated_from_its_parent_and_siblings() {
        for parent in [0u32, 7, 0x1234_5678, 0xFFFF_0000] {
            let a = secondary_seed(parent, 0);
            let b = secondary_seed(parent, 1);
            assert_ne!(a, parent, "secondary 0 reused the parent's stream");
            assert_ne!(b, parent, "secondary 1 reused the parent's stream");
            assert!(
                (expand_seed(a) ^ expand_seed(b)).count_ones() > 16,
                "poor avalanche between sibling secondaries of parent {parent}"
            );
        }
    }

    /// Two different walks that each number their secondaries from 0 must still
    /// get unrelated streams: the parent's seed is part of the key.
    #[test]
    fn secondary_seed_depends_on_the_parent() {
        for k in 0..1000u32 {
            let a = secondary_seed(history_seed(1, 10), k);
            let b = secondary_seed(history_seed(1, 11), k);
            assert_ne!(a, b, "ordinal {k} shared a stream across two parents");
        }
    }

    /// The whole point of issue #111's phase 1: the seed of a secondary depends
    /// only on the tree it sits in, never on the order the tree is walked. This
    /// replays the same emission tree under a LIFO drain (the CPU bank) and a
    /// FIFO drain (the GPU's in-thread queue) and demands the same seed for the
    /// same node. A FLAT per-history ordinal counter fails this at depth 2.
    #[test]
    fn secondary_seed_is_drain_order_independent() {
        // Emission tree: the root emits 2 secondaries, each of which emits 2.
        // `(path, seed)` where `path` names the node independently of the walk.
        fn expand(
            path: Vec<u32>,
            seed: u32,
            depth: u32,
            out: &mut Vec<(Vec<u32>, u32)>,
            queue: &mut std::collections::VecDeque<(Vec<u32>, u32, u32)>,
        ) {
            out.push((path.clone(), seed));
            if depth == 0 {
                return;
            }
            for k in 0..2u32 {
                let mut child = path.clone();
                child.push(k);
                queue.push_back((child, secondary_seed(seed, k), depth - 1));
            }
        }

        let root = history_seed(4242, 17);
        let mut lifo: Vec<(Vec<u32>, u32)> = Vec::new();
        let mut fifo: Vec<(Vec<u32>, u32)> = Vec::new();
        for use_lifo in [true, false] {
            let mut queue = std::collections::VecDeque::new();
            let out = if use_lifo { &mut lifo } else { &mut fifo };
            expand(Vec::new(), root, 2, out, &mut queue);
            while let Some((path, seed, depth)) = if use_lifo {
                queue.pop_back()
            } else {
                queue.pop_front()
            } {
                expand(path, seed, depth, out, &mut queue);
            }
        }
        assert_eq!(lifo.len(), fifo.len());
        lifo.sort();
        fifo.sort();
        assert_eq!(
            lifo, fifo,
            "a node's seed changed when the drain order changed"
        );
        // And every node of the tree really did get its own stream.
        let distinct: std::collections::HashSet<u32> = lifo.iter().map(|(_, s)| *s).collect();
        assert_eq!(distinct.len(), lifo.len(), "two tree nodes shared a stream");
    }

    #[test]
    fn next_xi_in_open_unit_interval() {
        let mut s = expand_seed(12345);
        for _ in 0..10_000 {
            let xi = next_xi(&mut s);
            assert!(xi > 0.0 && xi <= 1.0, "xi out of (0,1]: {xi}");
        }
    }

    #[test]
    fn next_xi_mean_near_half() {
        let mut s = expand_seed(7);
        let n = 2_000_000;
        let sum: f64 = (0..n).map(|_| next_xi(&mut s)).sum();
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 1e-3, "mean {mean} not near 0.5");
    }
}
