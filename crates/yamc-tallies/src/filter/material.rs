use std::collections::HashMap;
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
    /// # Errors
    /// If the material has no id.
    pub fn new(material: &Material) -> Result<Self, String> {
        Self::from_materials(&[material])
    }

    /// Create a `MaterialFilter` from a list of materials, one bin per material
    /// in the order given.
    ///
    /// # Errors
    /// If the list is empty, a material has no id, or two materials share an
    /// id: the bins are told apart by id, so the second would never score.
    pub fn from_materials(materials: &[&Material]) -> Result<Self, String> {
        if materials.is_empty() {
            return Err("a material filter needs at least one material".into());
        }
        let mut positions: HashMap<u32, usize> = HashMap::with_capacity(materials.len());
        let mut material_ids = Vec::with_capacity(materials.len());
        for (position, material) in materials.iter().enumerate() {
            let id = material.get_material_id().ok_or_else(|| {
                let label = match material.get_name() {
                    Some(name) => format!("material {name:?}"),
                    None => format!("the material at position {position}"),
                };
                format!("cannot filter on {label}: it has no id; set one with `set_material_id`")
            })?;
            if let Some(first) = positions.insert(id, position) {
                return Err(format!(
                    "materials at positions {first} and {position} both have id {id}; a \
                     material filter has one bin per id, so give each material its own"
                ));
            }
            material_ids.push(id);
        }
        Ok(Self { material_ids })
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
