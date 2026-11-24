//! Parse DAGMC-style naming conventions from physical group names and
//! build material/boundary-condition maps.

use std::collections::HashMap;

use crate::types::{BoundaryCondition, MaterialProps, SurfaceId, VolumeId};

/// Parsed metadata from physical groups.
#[derive(Debug, Clone, Default)]
pub struct PhysicalGroupData {
    /// Material properties per volume.
    pub volume_materials: HashMap<VolumeId, MaterialProps>,
    /// Boundary conditions per surface.
    pub surface_bcs: HashMap<SurfaceId, BoundaryCondition>,
}

/// Parse a physical group name using DAGMC-style conventions.
///
/// Supported prefixes:
/// - `mat:<name>` -- material assignment
/// - `boundary:<type>` -- boundary condition (vacuum, reflective, transmission)
/// - `temp:<value>` -- temperature
/// - Plain names -- treated as material name
pub fn parse_group_name(name: &str) -> GroupInfo {
    let mut material: Option<String> = None;
    let mut boundary: Option<BoundaryCondition> = None;
    let mut temperature: Option<f64> = None;

    // Handle compound names with '/' separator: "mat:fuel/temp:900"
    for part in name.split('/') {
        let part = part.trim();
        if let Some(mat_name) = part.strip_prefix("mat:") {
            material = Some(mat_name.to_string());
        } else if let Some(bc_str) = part.strip_prefix("boundary:") {
            boundary = Some(match bc_str.to_lowercase().as_str() {
                "vacuum" => BoundaryCondition::Vacuum,
                "reflective" | "reflecting" => BoundaryCondition::Reflective,
                "transmission" | "transmitting" => BoundaryCondition::Transmission,
                _ => BoundaryCondition::Transmission,
            });
        } else if let Some(temp_str) = part.strip_prefix("temp:") {
            temperature = temp_str.parse().ok();
        } else if material.is_none() && boundary.is_none() {
            // Plain name treated as material
            material = Some(part.to_string());
        }
    }

    GroupInfo {
        material,
        boundary,
        temperature,
    }
}

/// Parsed information from a single physical group name.
#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub material: Option<String>,
    pub boundary: Option<BoundaryCondition>,
    pub temperature: Option<f64>,
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_material() {
        let info = parse_group_name("mat:fuel");
        assert_eq!(info.material.as_deref(), Some("fuel"));
        assert_eq!(info.boundary, None);
    }

    #[test]
    fn test_parse_boundary() {
        let info = parse_group_name("boundary:vacuum");
        assert_eq!(info.boundary, Some(BoundaryCondition::Vacuum));
    }

    #[test]
    fn test_parse_reflective() {
        let info = parse_group_name("boundary:reflective");
        assert_eq!(info.boundary, Some(BoundaryCondition::Reflective));
    }

    #[test]
    fn test_parse_compound() {
        let info = parse_group_name("mat:fuel/temp:900.0");
        assert_eq!(info.material.as_deref(), Some("fuel"));
        assert_eq!(info.temperature, Some(900.0));
    }

    #[test]
    fn test_parse_plain_name() {
        let info = parse_group_name("water");
        assert_eq!(info.material.as_deref(), Some("water"));
    }
}
