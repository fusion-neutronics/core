use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::geometry::PyCell;
use crate::material::PyMaterial;

pub fn parse_colors_ids_cells_materials(
    colors: Option<Bound<'_, PyDict>>,
) -> PyResult<Option<HashMap<i32, String>>> {
    let Some(colors_dict) = colors else {
        return Ok(None);
    };

    let mut map = HashMap::new();
    for (key, value) in colors_dict.iter() {
        let color: String = value.extract().map_err(|_| {
            PyTypeError::new_err(
                "colors values must be strings (color names, hex codes, or RGB strings)",
            )
        })?;

        let id: i32 = if let Ok(id) = key.extract::<i32>() {
            id
        } else if let Ok(cell) = key.extract::<PyCell>() {
            cell.inner
                .cell_id
                .map(|id| id as i32)
                .ok_or_else(|| PyValueError::new_err("Cell has no ID"))?
        } else if let Ok(material) = key.extract::<PyMaterial>() {
            material
                .internal
                .get_material_id()
                .map(|id| id as i32)
                .ok_or_else(|| PyValueError::new_err("Material has no ID"))?
        } else {
            return Err(PyTypeError::new_err(
                "colors keys must be integers, Cell objects, or Material objects",
            ));
        };

        map.insert(id, color);
    }
    Ok(Some(map))
}
