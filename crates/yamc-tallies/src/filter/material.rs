use yamc_materials::Material;

/// Material filter for tallies -- filters/bins events by which material they occur in.
///
/// A `MaterialFilter` holds one or more material IDs. When it carries a single ID
/// it acts as a scalar gate. When it carries multiple IDs the tally gains a
/// material-bin dimension, producing one result bin per ID (analogous to a
/// `MaterialFilter([m1, m2, m3])`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MaterialFilter {
    /// The material IDs this filter bins over. Always at least one element.
    pub material_ids: Vec<u32>,
}

impl MaterialFilter {
    /// Create a `MaterialFilter` from a single material.
    ///
    /// # Panics
    /// Panics if the material has no material_id assigned.
    pub fn new(material: &Material) -> Self {
        let material_id = material.get_material_id().expect(
            "Cannot create MaterialFilter for material with no ID - assign a material_id first",
        );
        Self {
            material_ids: vec![material_id],
        }
    }

    /// Create a `MaterialFilter` from a list of materials (one bin per material).
    ///
    /// # Panics
    /// Panics if any material has no material_id assigned, or if the list is empty.
    pub fn from_materials(materials: &[&Material]) -> Self {
        assert!(
            !materials.is_empty(),
            "MaterialFilter requires at least one material",
        );
        let material_ids = materials
            .iter()
            .map(|m| {
                m.get_material_id().expect(
                    "Cannot create MaterialFilter for material with no ID - assign a material_id first",
                )
            })
            .collect();
        Self { material_ids }
    }

    /// Number of material bins this filter produces.
    pub fn num_bins(&self) -> usize {
        self.material_ids.len()
    }

    /// Return the bin index for a given material ID, or `None` if not in the filter.
    pub fn get_bin(&self, material_id: Option<u32>) -> Option<usize> {
        let id = material_id?;
        self.material_ids.iter().position(|&m| m == id)
    }

    /// True if `material_id` is one of the IDs this filter bins over.
    pub fn matches(&self, material_id: Option<u32>) -> bool {
        match material_id {
            Some(id) => self.material_ids.contains(&id),
            None => false,
        }
    }

    /// True if `material` is one of the materials this filter bins over.
    pub fn matches_material(&self, material: &Material) -> bool {
        material
            .get_material_id()
            .is_some_and(|id| self.material_ids.contains(&id))
    }
}
