use serde::{Deserialize, Serialize};

use crate::bounding_box::BoundingBox;
use crate::cell::GeoCell;
use crate::plot::{PlotGrid, PlotParams, PlotSample};

/// Lightweight CSG geometry for visualization -- serializable, no nuclear data deps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsgGeometry {
    pub cells: Vec<GeoCell>,
}

impl CsgGeometry {
    /// Find the first cell containing the given point, or None if not found
    pub fn find_cell(&self, point: (f64, f64, f64)) -> Option<&GeoCell> {
        self.cells.iter().find(|cell| cell.contains(point))
    }

    /// Find the index of the first cell containing the given point
    pub fn find_cell_index(&self, point: (f64, f64, f64)) -> Option<usize> {
        self.cells.iter().position(|cell| cell.contains(point))
    }

    /// Compute the bounding box of the entire geometry (enclosing all cells' regions)
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox_opt: Option<BoundingBox> = None;
        for cell in &self.cells {
            let cell_bbox = cell.region.bounding_box();
            match &mut bbox_opt {
                None => bbox_opt = Some(cell_bbox),
                Some(b) => b.expand_to_include(&cell_bbox),
            }
        }
        bbox_opt.unwrap_or_else(|| BoundingBox::new([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]))
    }

    /// Sample a 2D plot grid of cell/material IDs
    pub fn sample_grid(&self, params: &PlotParams) -> Result<PlotGrid, String> {
        crate::plot::sample_plot_grid(params, |point| {
            if let Some(cell) = self.find_cell(point) {
                PlotSample {
                    cell_id: cell.cell_id,
                    material_id: cell.material_id,
                    hover_text: String::new(),
                }
            } else {
                PlotSample {
                    cell_id: -1,
                    material_id: -1,
                    hover_text: String::new(),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CsgGeometry;
    use crate::cell::GeoCell;
    use crate::region::{HalfspaceType, Region};
    use crate::surface::Surface;
    use std::sync::Arc;

    /// Inside of a sphere of radius `r` centred at `(cx, 0, 0)`.
    fn sphere_cell(cell_id: i32, cx: f64, r: f64) -> GeoCell {
        let sphere = Surface::new_sphere(cx, 0.0, 0.0, r, Some(cell_id as usize), None);
        GeoCell {
            cell_id,
            material_id: cell_id + 100,
            name: None,
            material_name: None,
            region: Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere))),
        }
    }

    #[test]
    fn find_cell_returns_first_containing_cell() {
        // Two disjoint spheres: cell 1 around x=0, cell 2 around x=20.
        let geom = CsgGeometry {
            cells: vec![sphere_cell(1, 0.0, 3.0), sphere_cell(2, 20.0, 3.0)],
        };
        assert_eq!(geom.find_cell((0.0, 0.0, 0.0)).map(|c| c.cell_id), Some(1));
        assert_eq!(geom.find_cell((20.0, 0.0, 0.0)).map(|c| c.cell_id), Some(2));
        assert!(geom.find_cell((10.0, 0.0, 0.0)).is_none()); // in the gap
    }

    #[test]
    fn find_cell_index_matches_position() {
        let geom = CsgGeometry {
            cells: vec![sphere_cell(1, 0.0, 3.0), sphere_cell(2, 20.0, 3.0)],
        };
        assert_eq!(geom.find_cell_index((0.0, 0.0, 0.0)), Some(0));
        assert_eq!(geom.find_cell_index((20.0, 0.0, 0.0)), Some(1));
        assert_eq!(geom.find_cell_index((10.0, 0.0, 0.0)), None);
    }

    #[test]
    fn bounding_box_encloses_all_cells() {
        let geom = CsgGeometry {
            cells: vec![sphere_cell(1, 0.0, 3.0), sphere_cell(2, 20.0, 3.0)],
        };
        let bbox = geom.bounding_box();
        // Union of [-3,3] and [17,23] on x; [-3,3] on y and z.
        assert_eq!(bbox.lower_left, [-3.0, -3.0, -3.0]);
        assert_eq!(bbox.upper_right, [23.0, 3.0, 3.0]);
    }

    #[test]
    fn bounding_box_of_empty_geometry_is_inverted_infinity() {
        let geom = CsgGeometry { cells: vec![] };
        let bbox = geom.bounding_box();
        assert_eq!(bbox.lower_left, [f64::INFINITY; 3]);
        assert_eq!(bbox.upper_right, [f64::NEG_INFINITY; 3]);
    }
}
