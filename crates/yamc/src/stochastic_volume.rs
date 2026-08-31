// TODO: For simple geometries (single sphere, spherical shell, box, finite cylinder),
// volumes could be computed analytically by pattern-matching the region expression tree.
// Fall back to stochastic for complex cases.

use crate::geo::BoundingBox;
use crate::util::fast_rng::FastRng;
use rayon::prelude::*;

/// Result of a stochastic volume calculation for a single region/cell.
#[derive(Debug, Clone)]
pub struct VolumeResult {
    pub volume: f64,
    pub std_dev: f64,
    pub num_hits: u64,
}

/// Estimate volumes by sampling random points within a bounding box.
///
/// # Arguments
/// * `n_samples` - Total number of random points to sample
/// * `bbox` - Bounding box to sample within
/// * `seed` - RNG seed for reproducibility
/// * `n_bins` - Number of bins (cells/regions) to track
/// * `classify` - Closure that maps a point `(f64, f64, f64)` to `Some(bin_index)` if inside,
///   or `None` if outside all tracked regions. `bin_index` must be in `0..n_bins`.
///
/// # Returns
/// A `Vec<VolumeResult>` of length `n_bins`, one per bin.
pub fn calculate_stochastic_volumes(
    n_samples: u64,
    bbox: &BoundingBox,
    seed: u64,
    n_bins: usize,
    classify: impl Fn((f64, f64, f64)) -> Option<usize> + Sync,
) -> Vec<VolumeResult> {
    let bbox_volume = bbox.volume();
    let ll = bbox.lower_left;
    let w = bbox.width();

    // Use rayon to split work into parallel chunks
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = n_samples / n_threads as u64;
    let remainder = n_samples % n_threads as u64;

    let chunk_results: Vec<Vec<u64>> = (0..n_threads)
        .into_par_iter()
        .map(|thread_idx| {
            let thread_idx = thread_idx as u64;
            // Each thread gets a unique seed derived from the base seed
            let thread_seed = seed.wrapping_add(thread_idx.wrapping_mul(6364136223846793005));
            let mut rng = FastRng::new(thread_seed);

            let my_samples = chunk_size + if thread_idx < remainder { 1 } else { 0 };
            let mut local_hits = vec![0u64; n_bins];

            for _ in 0..my_samples {
                let x = ll[0] + w[0] * rng.random();
                let y = ll[1] + w[1] * rng.random();
                let z = ll[2] + w[2] * rng.random();

                if let Some(bin) = classify((x, y, z)) {
                    if bin < n_bins {
                        local_hits[bin] += 1;
                    }
                }
            }

            local_hits
        })
        .collect();

    // Reduce: sum hits across all threads
    let mut total_hits = vec![0u64; n_bins];
    for local in &chunk_results {
        for (i, &count) in local.iter().enumerate() {
            total_hits[i] += count;
        }
    }

    // Compute volume and standard deviation for each bin
    let n = n_samples as f64;
    total_hits
        .iter()
        .map(|&hits| {
            let p = hits as f64 / n;
            let volume = p * bbox_volume;
            // Standard deviation of the volume estimate: sqrt(p*(1-p)/n) * bbox_volume
            let std_dev = if n > 1.0 {
                (p * (1.0 - p) / n).sqrt() * bbox_volume
            } else {
                0.0
            };
            VolumeResult {
                volume,
                std_dev,
                num_hits: hits,
            }
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::BoundingBox;

    /// Single bin that accepts every point -- estimated volume should approximate
    /// the full bounding-box volume.
    #[test]
    fn single_bin_all_hits() {
        let bbox = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let n_samples = 100_000;

        let results = calculate_stochastic_volumes(n_samples, &bbox, 42, 1, |_| Some(0));

        assert_eq!(results.len(), 1);
        // Every sample lands in bin 0, so volume == bbox volume exactly.
        let expected_volume = bbox.volume(); // 8.0
        assert!(
            (results[0].volume - expected_volume).abs() < 1e-12,
            "Expected volume {}, got {}",
            expected_volume,
            results[0].volume
        );
        assert_eq!(results[0].num_hits, n_samples);
    }

    /// Two bins splitting the box along x = 0.  Each half should get roughly
    /// half the total volume.
    #[test]
    fn two_bins_half_and_half() {
        let bbox = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let n_samples = 200_000;

        let results = calculate_stochastic_volumes(n_samples, &bbox, 123, 2, |(x, _y, _z)| {
            if x < 0.0 {
                Some(0)
            } else {
                Some(1)
            }
        });

        assert_eq!(results.len(), 2);
        let half_volume = bbox.volume() / 2.0; // 4.0
        let tolerance = 0.05 * half_volume; // 5 % relative tolerance

        assert!(
            (results[0].volume - half_volume).abs() < tolerance,
            "Bin 0: expected ~{}, got {}",
            half_volume,
            results[0].volume
        );
        assert!(
            (results[1].volume - half_volume).abs() < tolerance,
            "Bin 1: expected ~{}, got {}",
            half_volume,
            results[1].volume
        );
        // Total hits should equal n_samples (every point lands somewhere)
        assert_eq!(results[0].num_hits + results[1].num_hits, n_samples);
    }

    /// find_bin always returns None -- all volumes must be exactly 0.
    #[test]
    fn no_hits_all_none() {
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let n_samples = 10_000;

        let results = calculate_stochastic_volumes(n_samples, &bbox, 7, 3, |_| None);

        assert_eq!(results.len(), 3);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.volume, 0.0,
                "Bin {} should have volume 0, got {}",
                i, r.volume
            );
            assert_eq!(
                r.num_hits, 0,
                "Bin {} should have 0 hits, got {}",
                i, r.num_hits
            );
            assert_eq!(
                r.std_dev, 0.0,
                "Bin {} should have std_dev 0, got {}",
                i, r.std_dev
            );
        }
    }

    /// Zero samples: the function must not panic.  With 0 samples the hit
    /// fraction is 0/0 = NaN, so the resulting volumes are NaN.  We verify
    /// that the function completes without panic, returns the correct number
    /// of bins, and records zero hits.
    #[test]
    fn zero_samples() {
        let bbox = BoundingBox::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);

        let results = calculate_stochastic_volumes(0, &bbox, 0, 4, |_| Some(0));

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            // 0 / 0.0 produces NaN in IEEE 754 -- the volume is undefined.
            assert!(
                r.volume.is_nan(),
                "Bin {} volume should be NaN with 0 samples, got {}",
                i,
                r.volume
            );
            assert_eq!(
                r.num_hits, 0,
                "Bin {} should have 0 hits with 0 samples, got {}",
                i, r.num_hits
            );
        }
    }
}
