/// Dynamic neighbor lists for CSG cell-finding acceleration.
///
/// Each cell stores a list of neighboring cell indices, learned on-the-fly
/// during transport. When a particle crosses a surface and the next cell is
/// found, that cell is added to the previous cell's neighbor list. Future
/// particles check the neighbor list first (fast path, ~4 candidates) before
/// falling back to the full linear search (slow path, N cells).
///
/// Based on the approach described by Harper et al. (2020).
use crate::geometry::cell::Cell;

pub struct NeighborLists {
    lists: Vec<Vec<usize>>,
}

impl NeighborLists {
    pub fn new(num_cells: usize) -> Self {
        Self {
            lists: vec![Vec::new(); num_cells],
        }
    }

    /// Find cell containing point, checking neighbors of `prev_cell` first.
    /// If found via full search, records the new neighbor for future lookups.
    pub fn find_cell(
        &mut self,
        cells: &[Cell],
        point: (f64, f64, f64),
        prev_cell: usize,
    ) -> Option<usize> {
        // Fast path: check neighbor list (~4 candidates typically)
        for &neighbor in &self.lists[prev_cell] {
            if cells[neighbor].contains(point) {
                return Some(neighbor);
            }
        }
        // Slow path: full linear search
        let found = cells.iter().position(|cell| cell.contains(point));
        // Learn: add newly discovered neighbor
        if let Some(idx) = found {
            if idx != prev_cell && !self.lists[prev_cell].contains(&idx) {
                self.lists[prev_cell].push(idx);
            }
        }
        found
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::Surface;
    use crate::geo::{HalfspaceType, Region};
    use std::sync::Arc;

    /// Helper: create a 1D grid of 3 cells along x-axis: [0,1], [1,2], [2,3]
    fn make_three_cells() -> Vec<Cell> {
        let planes: Vec<Arc<Surface>> = (0..=3)
            .map(|i| Arc::new(Surface::x_plane(i as f64, None, None)))
            .collect();

        (0..3)
            .map(|i| {
                let above = Region::new_from_halfspace(HalfspaceType::Above(planes[i].clone()));
                let below = Region::new_from_halfspace(HalfspaceType::Below(planes[i + 1].clone()));
                let region = above.intersection(&below);
                Cell::new(None, region, None, None)
            })
            .collect()
    }

    #[test]
    fn first_lookup_learns_neighbor() {
        let cells = make_three_cells();
        let mut nl = NeighborLists::new(cells.len());

        // From cell 0, find the cell containing point in cell 1
        let result = nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);
        assert_eq!(result, Some(1));

        // Cell 1 should now be in cell 0's neighbor list
        assert!(nl.lists[0].contains(&1));
    }

    #[test]
    fn second_lookup_hits_fast_path() {
        let cells = make_three_cells();
        let mut nl = NeighborLists::new(cells.len());

        // First call: learns cell 1 as neighbor of cell 0
        nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);

        // Second call from same prev_cell: should hit fast path (neighbor list)
        let result = nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);
        assert_eq!(result, Some(1));

        // Still only one neighbor recorded
        assert_eq!(nl.lists[0].len(), 1);
    }

    #[test]
    fn no_duplicate_neighbors() {
        let cells = make_three_cells();
        let mut nl = NeighborLists::new(cells.len());

        // Look up cell 1 from cell 0 three times
        nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);
        nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);
        nl.find_cell(&cells, (1.5, 0.0, 0.0), 0);

        // Only one entry for cell 1
        assert_eq!(nl.lists[0].len(), 1);
        assert_eq!(nl.lists[0][0], 1);
    }

    #[test]
    fn learns_multiple_neighbors() {
        let cells = make_three_cells();
        let mut nl = NeighborLists::new(cells.len());

        // From cell 1, discover cell 0 and cell 2
        nl.find_cell(&cells, (0.5, 0.0, 0.0), 1);
        nl.find_cell(&cells, (2.5, 0.0, 0.0), 1);

        assert_eq!(nl.lists[1].len(), 2);
        assert!(nl.lists[1].contains(&0));
        assert!(nl.lists[1].contains(&2));
    }

    #[test]
    fn same_cell_not_added_as_neighbor() {
        let cells = make_three_cells();
        let mut nl = NeighborLists::new(cells.len());

        // Point is in cell 0, prev_cell is also 0
        let result = nl.find_cell(&cells, (0.5, 0.0, 0.0), 0);
        assert_eq!(result, Some(0));

        // Cell 0 should NOT be in its own neighbor list
        assert!(nl.lists[0].is_empty());
    }
}
