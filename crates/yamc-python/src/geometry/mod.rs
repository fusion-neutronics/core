//! Geometry wrappers: CSG geometry/cells/regions/surfaces, meshes, and CAD.

use crate::material::PyMaterial;
use pyo3::prelude::*;

mod bounding_box;
mod cell;
mod csg;
mod mesh;
mod region;
mod surface;

pub use bounding_box::*;
pub use cell::*;
pub use csg::*;
pub use mesh::*;
pub use region::*;
pub use surface::*;

#[cfg(feature = "mesh")]
mod mesh_geometry;
#[cfg(feature = "mesh")]
pub use mesh_geometry::*;

#[cfg(feature = "cad")]
mod cad;
#[cfg(feature = "cad")]
mod cad_scene;
#[cfg(feature = "cad")]
mod face_volumes;
#[cfg(feature = "cad")]
pub use cad::*;
#[cfg(feature = "cad")]
pub use cad_scene::*;
#[cfg(feature = "cad")]
pub use face_volumes::*;

/// Refuses two different `Material` objects with one id among the materials
/// a geometry is given. Ids are set when a Material is built, but an explicit
/// id can repeat another material's, and copies keep their original's. The
/// same object met again is one material.
#[derive(Default)]
pub(crate) struct DistinctMaterialIds<'py> {
    first_with_id: std::collections::HashMap<u32, Bound<'py, PyMaterial>>,
}

impl<'py> DistinctMaterialIds<'py> {
    pub(crate) fn check(&mut self, material: &Bound<'py, PyMaterial>) -> PyResult<()> {
        let id = material.try_borrow()?.material_id()?;
        let Some(first) = self.first_with_id.get(&id) else {
            self.first_with_id.insert(id, material.clone());
            return Ok(());
        };
        if first.as_ptr() == material.as_ptr() {
            return Ok(());
        }
        let name = |m: &Bound<'py, PyMaterial>| -> PyResult<Option<String>> {
            Ok(m.try_borrow()?.internal.get_name().map(String::from))
        };
        let who = match (name(first)?, name(material)?) {
            (Some(a), Some(b)) => format!("materials {a:?} and {b:?}"),
            (Some(a), None) | (None, Some(a)) => format!("material {a:?} and an unnamed material"),
            (None, None) => "two unnamed materials".to_string(),
        };
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Duplicate Material id {id} found: {who} are different Material objects with that \
             id. A Material's id is set when it is built (the id= given, or else a fresh one) \
             and kept by its copies, such as the materials a transmutation result returns, so \
             an explicit id must not repeat another material's. Give one of them an id no other \
             material has."
        )))
    }
}
