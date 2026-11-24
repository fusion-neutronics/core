//! Parts of mesh elements: the piece of an element inside one cell.
//!
//! A tally carrying [`MeshParts`] scores each mesh crossing into the bin of
//! the (element, cell) pair it lies in, rather than the element's: one bin per
//! part that exists, so an element straddling void and two materials gives
//! three separately scored parts, and an element in one cell costs one bin.

use rustc_hash::FxHashMap;

/// The parts of a mesh's elements, each numbered as a tally bin.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshParts {
    index: FxHashMap<(u32, u32), u32>,
}

impl MeshParts {
    /// Parts numbered in the order given, as (element bin, cell id).
    ///
    /// # Panics
    /// If a pair repeats.
    pub fn new(parts: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut index = FxHashMap::default();
        for (i, part) in parts.into_iter().enumerate() {
            assert!(
                index.insert(part, i as u32).is_none(),
                "mesh part {part:?} repeats"
            );
        }
        Self { index }
    }

    /// Number of parts: the tally's mesh bins.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The bin of the part of element `element` in cell `cell_id`, or `None`
    /// where that cell is not part of that element (or the cell has no id).
    #[inline]
    pub fn bin(&self, element: usize, cell_id: Option<u32>) -> Option<usize> {
        self.index
            .get(&(element as u32, cell_id?))
            .map(|&b| b as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_number_in_order_and_miss_elsewhere() {
        let p = MeshParts::new([(0, 7), (0, 9), (3, 7)]);
        assert_eq!(p.len(), 3);
        assert_eq!(
            [p.bin(0, Some(7)), p.bin(0, Some(9)), p.bin(3, Some(7))],
            [Some(0), Some(1), Some(2)]
        );
        assert_eq!((p.bin(3, Some(9)), p.bin(0, None)), (None, None));
    }
}
