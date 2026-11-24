// Fast random number generator using a PCG-LCG implementation
//
// This is significantly faster than rand_pcg::Pcg64 because:
// 1. No struct construction overhead - just a u64 seed
// 2. Fully inlineable - compiler can optimize completely
// 3. Minimal state - 8 bytes vs Pcg64's larger internal state

use core::convert::Infallible;
use rand::{rand_core::TryRng, SeedableRng};

/// LCG multiplier
const PRN_MULT: u64 = 6364136223846793005;
/// LCG additive constant
const PRN_ADD: u64 = 1442695040888963407;

/// Fast RNG using a PCG-LCG algorithm.
///
/// This is a PCG (Permuted Congruential Generator) variant that uses
/// an LCG as the base generator with output permutation for quality.
///
/// Reference: Melissa E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
/// Statistically Good Algorithms for Random Number Generation"
#[derive(Clone, Copy, Debug)]
pub struct FastRng {
    seed: u64,
}

impl FastRng {
    /// Create a new FastRng with the given seed
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate a random f64 in [0, 1)
    #[inline(always)]
    pub fn random(&mut self) -> f64 {
        // Advance the LCG
        self.seed = PRN_MULT.wrapping_mul(self.seed).wrapping_add(PRN_ADD);

        // PCG output permutation (RXS-M-XS variant)
        let word =
            ((self.seed >> ((self.seed >> 59) + 5)) ^ self.seed).wrapping_mul(12605985483714917081);
        let result = (word >> 43) ^ word;

        // Convert to f64 in [0, 1) - equivalent to ldexp(result, -64)
        (result as f64) * 5.421010862427522e-20
    }

    /// Reseed the RNG (for reuse across particles)
    #[inline]
    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }
}

impl SeedableRng for FastRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            seed: u64::from_le_bytes(seed),
        }
    }
}

impl TryRng for FastRng {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        // Advance the LCG
        self.seed = PRN_MULT.wrapping_mul(self.seed).wrapping_add(PRN_ADD);

        // PCG output permutation
        let word =
            ((self.seed >> ((self.seed >> 59) + 5)) ^ self.seed).wrapping_mul(12605985483714917081);
        Ok((word >> 43) ^ word)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
        // Fill bytes using next_u64
        let mut left = dest;
        while left.len() >= 8 {
            let bytes = self.try_next_u64()?.to_le_bytes();
            left[..8].copy_from_slice(&bytes);
            left = &mut left[8..];
        }
        if !left.is_empty() {
            let bytes = self.try_next_u64()?.to_le_bytes();
            left.copy_from_slice(&bytes[..left.len()]);
        }
        Ok(())
    }
}
