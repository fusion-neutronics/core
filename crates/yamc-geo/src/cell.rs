use serde::{Deserialize, Serialize};

use crate::region::Region;

/// Lightweight cell for geometry visualization -- no nuclear data dependencies.
///
/// This is the serializable counterpart of yamc's transport-ready `Cell`
/// (which carries `Arc<Material>`). A `GeoCell` only stores IDs and names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoCell {
    /// Cell ID (-1 if unset)
    pub cell_id: i32,
    /// Material ID (-1 for void)
    pub material_id: i32,
    /// Optional cell name
    pub name: Option<String>,
    /// Optional material name
    pub material_name: Option<String>,
    /// Region defining the cell boundary
    pub region: Region,
}

impl GeoCell {
    /// Check if a point is inside this cell's region
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        self.region.contains(point)
    }
}

#[cfg(test)]
mod tests {
    use super::GeoCell;
    use crate::region::{HalfspaceType, Region};
    use crate::surface::Surface;
    use std::sync::Arc;

    /// A cell that is the inside of a sphere of radius 3 centred at the origin.
    fn sphere_cell() -> GeoCell {
        let sphere = Surface::new_sphere(0.0, 0.0, 0.0, 3.0, Some(1), None);
        GeoCell {
            cell_id: 1,
            material_id: 2,
            name: Some("ball".to_string()),
            material_name: Some("water".to_string()),
            region: Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere))),
        }
    }

    #[test]
    fn contains_is_true_inside_and_false_outside() {
        let cell = sphere_cell();
        assert!(cell.contains((0.0, 0.0, 0.0))); // centre
        assert!(cell.contains((2.9, 0.0, 0.0))); // just inside the surface
        assert!(!cell.contains((5.0, 0.0, 0.0))); // outside
        assert!(!cell.contains((0.0, 0.0, 10.0)));
    }
}
