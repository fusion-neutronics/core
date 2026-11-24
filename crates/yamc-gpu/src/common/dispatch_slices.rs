//! One logical launch, dispatched to the GPU in slices short enough for the
//! driver's watchdog.
//!
//! A launch's histories are fixed by the host (its seed band and its sampled
//! source particles), and each history is one thread whose work does not
//! depend on any other thread. So the kernel can run the launch in several
//! dispatches over consecutive ranges of histories, `[start, end)`, and give
//! exactly the result one dispatch would: every per-history output is indexed
//! by the history, and every tally is a fixed-point atomic sum, which does not
//! depend on the order the histories finish in.
//!
//! That makes the slice size free to follow wall time. A driver resets a
//! context whose job runs past its watchdog (10 s for amdgpu, 2 s for the
//! default Windows TDR), so each dispatch is sized to take about
//! [`TARGET_DISPATCH_SECS`]: the first from what the previous launches of the
//! same kernel achieved, the rest from the rate the last slice measured. A
//! launch of short histories goes in one dispatch; a launch of long ones goes
//! in several, and the results are the same either way.
//!
//! `YAMC_GPU_DISPATCH_SLICE` fixes the slice size, for exercising the slicing
//! in tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Wall time each dispatch aims for, half the strictest watchdog (Windows'
/// 2 s TDR).
///
/// A dispatch lasts as long as its slowest history, so its time is not
/// proportional to its slice: on a model with a long tail of slow histories a
/// slice four times larger takes barely longer, until the GPU is full. Below
/// that knee the GPU spends most of each dispatch waiting on a few threads.
/// On a 101-cell graphite model the knee sits near 0.6 s per dispatch, and a
/// target of 0.5 s held the slice under it at half the throughput.
pub const TARGET_DISPATCH_SECS: f64 = 1.0;

/// The first slice a kernel has no measurement for.
const FIRST_SLICE: usize = 4096;

/// The smallest slice: a few waves, so a pathologically slow launch still
/// advances.
const MIN_SLICE: usize = 256;

/// The most a slice grows or shrinks from one dispatch to the next.
const MAX_STEP: f64 = 4.0;

/// The slice size the neutron transport kernel last settled on.
pub static NEUTRON_SLICE: AtomicUsize = AtomicUsize::new(FIRST_SLICE);
/// The slice size the photon transport kernel last settled on.
pub static PHOTON_SLICE: AtomicUsize = AtomicUsize::new(FIRST_SLICE);

/// Walks one launch of `n` histories in slices.
pub struct DispatchSlices<'a> {
    learned: &'a AtomicUsize,
    fixed: Option<usize>,
    n: usize,
    start: usize,
    slice: usize,
}

impl<'a> DispatchSlices<'a> {
    /// A launch of `n` histories for the kernel whose learned slice is
    /// `learned`.
    pub fn new(learned: &'a AtomicUsize, n: usize) -> Self {
        let fixed = std::env::var("YAMC_GPU_DISPATCH_SLICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|s| s.max(1));
        let slice = fixed.unwrap_or_else(|| learned.load(Ordering::Relaxed).max(MIN_SLICE));
        DispatchSlices {
            learned,
            fixed,
            n,
            start: 0,
            slice,
        }
    }

    /// The next range of histories to dispatch, `[start, end)`, or `None`
    /// when the launch is done.
    pub fn next_range(&mut self) -> Option<(usize, usize)> {
        if self.start >= self.n {
            return None;
        }
        let end = (self.start + self.slice).min(self.n);
        let range = (self.start, end);
        self.start = end;
        Some(range)
    }

    /// Record that a dispatch of `len` histories took `elapsed`, and size the
    /// next one from it.
    pub fn record(&mut self, len: usize, elapsed: Duration) {
        if self.fixed.is_some() || len == 0 {
            return;
        }
        let secs = elapsed.as_secs_f64().max(1e-6);
        let ideal = len as f64 * TARGET_DISPATCH_SECS / secs;
        // A short last slice says little about the rate at full size, so it
        // may only shrink the slice, never grow it.
        let current = self.slice as f64;
        let mut next = ideal.clamp(current / MAX_STEP, current * MAX_STEP);
        if len < self.slice {
            next = next.min(current);
        }
        self.slice = (next as usize).max(MIN_SLICE);
        self.learned.store(self.slice, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_cover_the_launch_in_order_and_follow_the_measured_rate() {
        let learned = AtomicUsize::new(1000);
        let mut s = DispatchSlices::new(&learned, 10_000);
        let mut covered = 0;
        // A fast first slice grows the next, by at most MAX_STEP.
        let (a, b) = s.next_range().unwrap();
        assert_eq!((a, b), (0, 1000));
        s.record(b - a, Duration::from_millis(10));
        let (c, d) = s.next_range().unwrap();
        assert_eq!((c, d), (1000, 5000));
        covered += d;
        // A slow one shrinks it towards the target.
        s.record(d - c, Duration::from_secs(4));
        assert_eq!(learned.load(Ordering::Relaxed), 1000);
        while let Some((x, y)) = s.next_range() {
            assert_eq!(x, covered);
            covered = y;
            s.record(y - x, Duration::from_secs_f64(TARGET_DISPATCH_SECS));
        }
        assert_eq!(covered, 10_000);
    }

    #[test]
    fn a_slice_never_drops_below_the_minimum() {
        let learned = AtomicUsize::new(300);
        let mut s = DispatchSlices::new(&learned, 100_000);
        let (a, b) = s.next_range().unwrap();
        s.record(b - a, Duration::from_secs(100));
        assert_eq!(learned.load(Ordering::Relaxed), MIN_SLICE);
    }
}
