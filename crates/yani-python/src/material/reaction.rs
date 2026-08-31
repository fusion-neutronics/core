use crate::material::PyReactionProduct;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_nuclide::reaction::Reaction;

#[gen_stub_pyclass]
#[pyclass(name = "Reaction", from_py_object)]
/// Lightweight reaction summary exposed to Python.
///
/// Represents a single nuclear reaction channel with its products and cross section data.
#[derive(Clone, Debug)]
pub struct PyReaction {
    /// List of outgoing particles with their yield and secondary distributions.
    #[pyo3(get)]
    pub products: Vec<PyReactionProduct>,

    /// Cross section values in barns at each energy point.
    #[pyo3(get)]
    pub cross_section: Vec<f64>,

    /// Index into parent energy grid where this reaction becomes active (threshold).
    #[pyo3(get)]
    pub threshold_idx: usize,

    /// ENDF MT number identifying the reaction type (e.g., 2 for elastic, 16 for (n,2n)).
    #[pyo3(get)]
    pub mt_number: i32,

    /// Q-value of the reaction in eV (energy released if positive, absorbed if negative).
    #[pyo3(get)]
    pub q_value: f64,

    /// Reaction-specific energy grid in eV (subset of parent nuclide energy grid).
    #[pyo3(get)]
    pub energy_grid: Vec<f64>,

    /// Whether scattering is in center-of-mass frame (requires CM to LAB conversion).
    #[pyo3(get)]
    pub scatter_in_cm: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReaction {
    #[new]
    pub fn new(
        products: Vec<PyReactionProduct>,
        cross_section: Vec<f64>,
        threshold_idx: usize,
        mt_number: i32,
        q_value: f64,
        energy_grid: Vec<f64>,
        scatter_in_cm: bool,
    ) -> Self {
        PyReaction {
            products,
            cross_section,
            threshold_idx,
            mt_number,
            q_value,
            energy_grid,
            scatter_in_cm,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Reaction(mt={}, q_value={:.4e} eV, products={}, energy_points={})",
            self.mt_number,
            self.q_value,
            self.products.len(),
            self.energy_grid.len()
        )
    }
}

impl PyReaction {
    pub fn from_reaction(reaction: &Reaction, py: pyo3::Python) -> pyo3::PyResult<Self> {
        let products = reaction
            .products
            .iter()
            .map(|prod| PyReactionProduct::from_reaction_product(prod.clone(), py))
            .collect::<pyo3::PyResult<Vec<_>>>()?;
        Ok(PyReaction {
            products,
            // Owned copies: these are `#[pyo3(get)]` attributes, so Python
            // gets a list either way.
            cross_section: reaction.cross_section.to_vec(),
            threshold_idx: reaction.threshold_idx,
            mt_number: reaction.mt_number,
            q_value: reaction.q_value,
            energy_grid: reaction.energy.to_vec(),
            scatter_in_cm: reaction.scatter_in_cm,
        })
    }
}
