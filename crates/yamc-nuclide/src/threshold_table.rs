//! A per-channel cross-section table that stores no zeros below thresholds.
//!
//! The transport lookup reads each scattering channel at an energy index, and
//! the sampling walk reads every channel of one energy row. A dense
//! `[n_energy, n_channels]` layout serves both, but most of it is the zeros
//! below each channel's threshold: on the published heavy nuclides, level
//! inelastic channels open in the MeV range on grids whose points sit mostly in
//! the resolved resonances, so over 90 percent of the dense table is zero.
//!
//! Here the channels are ranked by the index of their first non-zero entry.
//! The channels open at row `i` are then always a prefix of that ranking, so
//! row `i` stores just that prefix, still contiguous, and an entry is found in
//! O(1) as `data[offsets[i] + rank[j]]`. An entry outside the prefix is below
//! its channel's threshold, and reads as the `+0.0` the dense table held there.
//! Every stored entry is the dense table's own value, bit for bit, so swapping
//! layouts cannot change a result.

use crate::buffer::F64Buffer;

/// See the module doc.
#[derive(Debug, Clone, Default)]
pub struct ThresholdTable {
    /// Every row's open channels, back to back, each row in rank order.
    data: F64Buffer,
    /// `offsets[i]..offsets[i + 1]` is row `i` within `data`. Empty when the
    /// table has no channels.
    offsets: Vec<u32>,
    /// `rank[j]` is channel `j`'s position within any row it is open in.
    rank: Vec<u32>,
}

impl ThresholdTable {
    /// Build from one column per channel, each `n_rows` long.
    ///
    /// A channel opens at its first entry that is not `+0.0`, compared by bits
    /// so that even a `-0.0` is stored rather than read back as `+0.0`. Entries
    /// after that are kept whatever their value, so a channel that closes again
    /// costs its trailing zeros and nothing is lost.
    ///
    /// Errors when the stored entries would not fit the 32-bit offsets, which
    /// no published library comes near (the largest grid is 163,746 points).
    pub fn from_columns(columns: &[Vec<f64>], n_rows: usize) -> Result<Self, String> {
        if columns.is_empty() || n_rows == 0 {
            return Ok(Self::default());
        }
        let first: Vec<usize> = columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .take(n_rows)
                    .position(|v| v.to_bits() != 0)
                    .unwrap_or(n_rows)
            })
            .collect();
        // Stable, so channels opening at the same row keep their storage order.
        let mut ranked: Vec<usize> = (0..columns.len()).collect();
        ranked.sort_by_key(|&j| first[j]);
        let mut rank = vec![0u32; columns.len()];
        for (r, &j) in ranked.iter().enumerate() {
            rank[j] = r as u32;
        }

        let mut data: Vec<f64> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(n_rows + 1);
        let mut open = 0usize;
        for i in 0..n_rows {
            offsets.push(u32::try_from(data.len()).map_err(|_| too_large(n_rows, columns.len()))?);
            while open < ranked.len() && first[ranked[open]] <= i {
                open += 1;
            }
            data.extend(
                ranked[..open]
                    .iter()
                    .map(|&j| columns[j].get(i).copied().unwrap_or(0.0)),
            );
        }
        offsets.push(u32::try_from(data.len()).map_err(|_| too_large(n_rows, columns.len()))?);
        data.shrink_to_fit();
        Ok(Self {
            data: data.into(),
            offsets,
            rank,
        })
    }

    /// Energy points the table spans.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Channels the table holds.
    #[inline]
    pub fn n_channels(&self) -> usize {
        self.rank.len()
    }

    /// True when the table holds no channel or spans no energy point.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows() == 0 || self.rank.is_empty()
    }

    /// Channel `j` at row `i`. Callers keep both in range, as they did for the
    /// dense table.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let start = self.offsets[i] as usize;
        let open = self.offsets[i + 1] as usize - start;
        let r = self.rank[j] as usize;
        if r < open {
            self.data[start + r]
        } else {
            0.0
        }
    }

    /// Channel `j` at every row, as the dense table's column.
    pub fn column(&self, j: usize) -> F64Buffer {
        (0..self.n_rows()).map(|i| self.get(i, j)).collect()
    }

    /// Entries actually stored, for measuring the saving.
    pub fn stored_len(&self) -> usize {
        self.data.len()
    }
}

fn too_large(n_rows: usize, n_channels: usize) -> String {
    format!(
        "a {n_rows}-point grid with {n_channels} scattering channels stores more entries \
         than a 32-bit offset can index"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(columns: &[Vec<f64>], n_rows: usize) -> Vec<f64> {
        crate::nuclide::flatten_row_major(columns, n_rows)
    }

    /// Every entry reads back as the dense table's, bit for bit, including
    /// channels that open late, close again, never open, or share a threshold.
    #[test]
    fn every_entry_matches_the_dense_table() {
        let n = 6;
        let columns = vec![
            vec![5.0, 5.0, 4.0, 3.0, 2.0, 1.0],      // open from the start
            vec![0.0, 0.0, 0.0, 7.0, 8.0, 9.0],      // opens at 3
            vec![0.0, 0.0, 1.5, 0.0, 0.0, 0.0],      // opens at 2, closes again
            vec![0.0; 6],                            // never opens
            vec![0.0, 0.0, 0.0, 6.0, 0.0, 2.5],      // shares row 3's threshold
            vec![0.0, -0.0, 0.0, 0.0, 0.0, 1.0e-30], // a -0.0 must survive
        ];
        let table = ThresholdTable::from_columns(&columns, n).unwrap();
        let reference = dense(&columns, n);
        assert_eq!(table.n_rows(), n);
        assert_eq!(table.n_channels(), columns.len());
        for i in 0..n {
            for j in 0..columns.len() {
                assert_eq!(
                    table.get(i, j).to_bits(),
                    reference[i * columns.len() + j].to_bits(),
                    "row {i}, channel {j}"
                );
            }
        }
        for (j, column) in columns.iter().enumerate() {
            assert_eq!(table.column(j).as_slice(), column.as_slice());
        }
        assert!(table.stored_len() < reference.len());
    }

    #[test]
    fn no_channels_is_an_empty_table() {
        let table = ThresholdTable::from_columns(&[], 10).unwrap();
        assert!(table.is_empty());
        assert_eq!(table.n_rows(), 0);
    }
}
