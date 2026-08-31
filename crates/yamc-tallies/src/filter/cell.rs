/// Cell filter for tallies -- filters/bins events by which cell they occur in.
///
/// A `CellFilter` holds one or more cell IDs. When it carries a single ID it acts
/// as a scalar gate (tally scores only when the event is in that cell). When it
/// carries multiple IDs the tally gains a cell-bin dimension, producing one result
/// bin per ID (analogous to a `CellFilter([c1, c2, c3])`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CellFilter {
    /// The cell IDs this filter bins over. Always at least one element.
    pub cell_ids: Vec<u32>,
}

impl CellFilter {
    /// Create a `CellFilter` from a single cell ID.
    pub fn from_id(cell_id: u32) -> Self {
        Self {
            cell_ids: vec![cell_id],
        }
    }

    /// Create a `CellFilter` from a list of cell IDs (one bin per ID).
    ///
    /// # Panics
    /// Panics if `cell_ids` is empty.
    pub fn from_ids(cell_ids: Vec<u32>) -> Self {
        assert!(
            !cell_ids.is_empty(),
            "CellFilter requires at least one cell ID",
        );
        Self { cell_ids }
    }

    /// Number of cell bins this filter produces.
    pub fn num_bins(&self) -> usize {
        self.cell_ids.len()
    }

    /// Return the bin index for a given cell ID, or `None` if the ID is not in the filter.
    pub fn get_bin(&self, cell_id: u32) -> Option<usize> {
        self.cell_ids.iter().position(|&id| id == cell_id)
    }

    /// True if `cell_id` is one of the IDs this filter bins over.
    pub fn matches(&self, cell_id: u32) -> bool {
        self.cell_ids.contains(&cell_id)
    }
}
