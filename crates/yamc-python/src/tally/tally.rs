use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;
use yamc_particle::particle::ParticleType;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{Score, Tally};
use yamc_tallies::{
    CellFilter, EnergyFilter, EnergyFunctionFilter, MaterialFilter, MeshFilter,
    ParentNuclideFilter, ParticleTypeFilter,
};

use crate::geometry::PyCell;
use crate::geometry::PyGeometry;
#[cfg(feature = "mesh")]
use crate::geometry::PyMeshGeometry;
use crate::geometry::{PyRegularCylindricalMesh, PyRegularRectangularMesh};
use crate::material::PyMaterial;

use super::mesh_slice::{parse_scores_arg, PyMeshSliceData};

/// A tally used to score physical quantities during particle transport simulation.
///
/// Tallies accumulate statistics over batches for quantities like flux, heating,
/// or reaction rates (via MT numbers). Results include mean values, standard
/// deviations, and relative errors.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Tally", unsendable, from_py_object)]
#[derive(Clone)]
pub struct PyTally {
    pub inner: Arc<Tally>,
}

/// Resolve a `Cell` or sequence of `Cell`s into the cell ids a `CellFilter`
/// bins over.
///
/// Cell ids are assigned when a cell is added to a `Geometry`, so a tally built
/// before its geometry sees `cell_id == None`. That is a user ordering mistake,
/// not a bug, so it must surface as a normal Python exception rather than a
/// panic crossing the FFI boundary (issue #305).
fn cell_filter_ids(cells: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let py_cells: Vec<pyo3::PyRef<'_, PyCell>> =
        if let Ok(list) = cells.extract::<Vec<pyo3::PyRef<'_, PyCell>>>() {
            list
        } else if let Ok(single) = cells.extract::<pyo3::PyRef<'_, PyCell>>() {
            vec![single]
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "cells must be a Cell or a list of Cell instances",
            ));
        };
    if py_cells.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cells must contain at least one Cell",
        ));
    }
    py_cells
        .iter()
        .map(|c| {
            c.inner.cell_id.ok_or_else(|| {
                let label = match c.inner.name.as_deref() {
                    Some(name) => format!(" (name={name:?})"),
                    None => String::new(),
                };
                pyo3::exceptions::PyValueError::new_err(format!(
                    "cannot filter on a Cell with no id{label}: ids are assigned when the \
                     cell is added to a Geometry, so build the Geometry before the Tally, \
                     or give the cell an explicit id with Cell(id=...)"
                ))
            })
        })
        .collect()
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTally {
    /// Create a new Tally.
    ///
    /// Args:
    ///     scores (list, optional): Scores to tally (e.g., ['flux'], [101, 'heating'])
    ///     name (str, optional): Name for the tally
    ///     id (int, optional): Numeric ID for the tally
    ///     nuclides (list, optional): Nuclides for per-nuclide *macroscopic*
    ///         breakdown at their real in-material density (e.g. ['Li6', 'Li7',
    ///         'total']). Mutually exclusive with ``response``.
    ///     response (str | list[str] | Material, optional): Score a *virtual*
    ///         response across the whole geometry (including void), decoupled
    ///         from the cell material:
    ///           * ``str`` / ``list[str]`` -- microscopic cross sections (barns)
    ///             for those nuclide(s) at unit density; one bin per nuclide.
    ///           * ``Material`` -- the material's *macroscopic* response,
    ///             weighting each nuclide's microscopic XS by its real atom
    ///             density; one combined bin (e.g. a SiO2 dose map).
    ///         Mutually exclusive with ``nuclides``.
    ///     cells (Cell | list[Cell], optional): Cell(s) to bin over -- one result bin
    ///         per cell. A single cell (not in a list) is also accepted.
    ///     materials (Material | list[Material], optional): Material(s) to bin over --
    ///         one result bin per material. A single material (not in a list) is also
    ///         accepted.
    ///     mesh (RegularRectangularMesh, optional): Mesh for spatial binning
    ///     unstructured_mesh (tuple, optional): (MeshGeometry, volume) for tet mesh scoring
    ///     energy_bins (list[float], optional): Energy bin boundaries in eV
    ///     energy_group_structure (str, optional): Named group structure (e.g. "VITAMIN-J-175")
    ///     energy_function (tuple, optional): (energy, y, units) for energy-dependent weighting
    ///     dose_coefficients (tuple, optional): (particle, geometry[, data_source]) for dose
    ///     particle (str, optional): "neutron" or "photon"
    ///     parent_nuclides (list[str], optional): Nuclides for D1S parent binning
    ///
    /// Notes:
    ///     ``cells`` and ``materials`` are mutually exclusive -- a single tally
    ///     binning over both cells and materials is not supported (use separate
    ///     tallies).
    ///
    ///     When multiple cells or materials are provided, the ``mean``/``std_dev``
    ///     arrays grow to hold one bin per cell/material, in the order given.
    ///     Flat bin layout (slowest → fastest varying) is:
    ///     ``score → cell → material → nuclide → parent_nuclide → energy → mesh``.
    #[new]
    #[pyo3(signature = (
        scores=None, name=None, id=None, nuclides=None, response=None,
        cells=None, materials=None,
        mesh=None, unstructured_mesh=None,
        energy_bins=None, energy_group_structure=None, energy_function=None,
        dose_coefficients=None, particle=None, parent_nuclides=None,
        estimator=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        #[gen_stub(override_type(
            type_repr = "typing.Sequence[builtins.str | builtins.int] | None"
        ))]
        scores: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        id: Option<u32>,
        nuclides: Option<Vec<String>>,
        #[gen_stub(override_type(type_repr = "str | typing.Sequence[str] | Material | None"))]
        response: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "Cell | typing.Sequence[Cell] | None"))] cells: Option<
            &Bound<'_, PyAny>,
        >,
        #[gen_stub(override_type(type_repr = "Material | typing.Sequence[Material] | None"))]
        materials: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(
            type_repr = "RegularRectangularMesh | RegularCylindricalMesh | None"
        ))]
        mesh: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "tuple[MeshGeometry, builtins.float] | None"))]
        unstructured_mesh: Option<&Bound<'_, PyAny>>,
        energy_bins: Option<Vec<f64>>,
        energy_group_structure: Option<String>,
        #[gen_stub(override_type(
            type_repr = "tuple[typing.Sequence[builtins.float], typing.Sequence[builtins.float], builtins.str] | None"
        ))]
        energy_function: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(
            type_repr = "tuple[builtins.str, builtins.str] | tuple[builtins.str, builtins.str, builtins.str] | None"
        ))]
        dose_coefficients: Option<&Bound<'_, PyAny>>,
        particle: Option<String>,
        parent_nuclides: Option<Vec<String>>,
        estimator: Option<String>,
    ) -> PyResult<Self> {
        let estimator_value = match estimator.as_deref() {
            None => yamc_tallies::Estimator::default(),
            Some(s) => s
                .parse::<yamc_tallies::Estimator>()
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        };
        // `response` selects the virtual-overlay regime (issue #341): the score
        // is evaluated across the whole geometry (including void), decoupled from
        // the cell material. Setting `multiply_density = false` only turns off the
        // usual "multiply by the *cell's* atom density" step -- it does NOT mean
        // the result is microscopic:
        //   * str / list[str] -> microscopic XS (barns) at unit density.
        //   * Material         -> a *macroscopic* response; the density weighting
        //                         comes from the overlay material's own atom
        //                         densities (see the Material branch below), not
        //                         from the cell the particle is in.
        // Mutually exclusive with the `nuclides` breakdown, which bins the real
        // in-material macroscopic response and shares the same nuclide axis.
        if response.is_some() && nuclides.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "response and nuclides are mutually exclusive",
            ));
        }
        // `multiply_density` means "scale microscopic XS by the *cell's* atom
        // density" (the normal macroscopic tally). A `response` clears it; for a
        // Material response the macroscopic weighting is instead supplied by
        // `overlay_material` below.
        let mut multiply_density = true;
        let mut overlay_material: Option<std::collections::BTreeMap<String, f64>> = None;
        let mut overlay_nuclide_bins: Option<Vec<yamc_tallies::NuclideBin>> = None;
        if let Some(resp) = response {
            use yamc_tallies::NuclideBin;
            multiply_density = false;
            // Extraction order matters: try str, then list[str], then Material.
            // `PyMaterial` is a pyclass and won't coerce to String/Vec<String>.
            if let Ok(name) = resp.extract::<String>() {
                overlay_nuclide_bins = Some(vec![NuclideBin::Specific(name)]);
            } else if let Ok(names) = resp.extract::<Vec<String>>() {
                if names.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "response list must not be empty",
                    ));
                }
                overlay_nuclide_bins = Some(names.into_iter().map(NuclideBin::Specific).collect());
            } else if let Ok(mat) = resp.extract::<pyo3::PyRef<'_, PyMaterial>>() {
                // Material response: weight each nuclide's microscopic XS by its
                // real atom density (atoms/barn-cm) and collapse to one combined
                // macroscopic bin (`nuclides = [Total]`, backed by `overlay_material`).
                let densities = mat
                    .internal
                    .get_atoms_per_barn_cm()
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                if densities.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "response material has no nuclides",
                    ));
                }
                overlay_material = Some(densities.into_iter().collect());
                overlay_nuclide_bins = Some(vec![NuclideBin::Total]);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "response must be a str, list[str], or Material",
                ));
            }
        }
        // Validate mutual exclusivity
        if cells.is_some() && materials.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cells and materials are mutually exclusive",
            ));
        }
        if mesh.is_some() && unstructured_mesh.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mesh and unstructured_mesh are mutually exclusive",
            ));
        }
        if energy_bins.is_some() && energy_group_structure.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "energy_bins and energy_group_structure are mutually exclusive",
            ));
        }
        if energy_function.is_some() && dose_coefficients.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "energy_function and dose_coefficients are mutually exclusive",
            ));
        }

        // Build filters from keyword arguments
        let mut filters = Vec::new();

        if let Some(cs) = cells {
            filters.push(Filter::Cell(CellFilter::from_ids(cell_filter_ids(cs)?)));
        }

        if let Some(ms) = materials {
            let py_mats: Vec<pyo3::PyRef<'_, PyMaterial>> =
                if let Ok(list) = ms.extract::<Vec<pyo3::PyRef<'_, PyMaterial>>>() {
                    list
                } else if let Ok(single) = ms.extract::<pyo3::PyRef<'_, PyMaterial>>() {
                    vec![single]
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "materials must be a Material or a list of Material instances",
                    ));
                };
            if py_mats.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "materials must contain at least one Material",
                ));
            }
            let mat_refs: Vec<&yamc_materials::material::Material> =
                py_mats.iter().map(|m| &m.internal).collect();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                MaterialFilter::from_materials(&mat_refs)
            })) {
                Ok(filter) => filters.push(Filter::Material(filter)),
                Err(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Cannot filter on material with no ID - assign an id first",
                    ))
                }
            }
        }

        if let Some(m) = mesh {
            if let Ok(rect) = m.extract::<PyRegularRectangularMesh>() {
                filters.push(Filter::Mesh(MeshFilter::new(rect.internal.clone())));
            } else if let Ok(cyl) = m.extract::<PyRegularCylindricalMesh>() {
                filters.push(Filter::Mesh(MeshFilter::new_cylindrical(
                    cyl.internal.clone(),
                )));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "mesh must be a RegularRectangularMesh or RegularCylindricalMesh instance",
                ));
            }
        }

        if let Some(um) = unstructured_mesh {
            #[cfg(feature = "mesh")]
            {
                use crate::geometry::PyMeshGeometry;
                let tuple: Vec<Bound<'_, PyAny>> = um.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "unstructured_mesh must be a tuple (MeshGeometry, volume)",
                    )
                })?;
                if tuple.len() != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "unstructured_mesh must be a 2-tuple (MeshGeometry, volume)",
                    ));
                }
                let mesh_geom: pyo3::PyRef<'_, PyMeshGeometry> =
                    tuple[0].extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "unstructured_mesh[0] must be a MeshGeometry instance",
                        )
                    })?;
                let mesh_ref = &mesh_geom.inner.mesh;
                let volume_id = if let Ok(id) = tuple[1].extract::<u32>() {
                    id
                } else if let Ok(name) = tuple[1].extract::<String>() {
                    resolve_volume_name(mesh_ref, &name)?
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "unstructured_mesh[1] must be an int (volume ID) or str (material name)",
                    ));
                };
                if volume_id >= mesh_ref.topology.num_volumes {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "volume_id {} out of range (mesh has {} volumes)",
                        volume_id, mesh_ref.topology.num_volumes
                    )));
                }
                let mesh_arc = std::sync::Arc::new(mesh_ref.clone());
                let filter = yamc_tallies::UnstructuredMeshFilter::new(mesh_arc, volume_id);
                // A surface-only mesh has no tetrahedra, so the tally would have
                // zero bins and silently score nothing. Refuse it at
                // construction with the actual remedy (issue #290).
                if filter.num_bins() == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unstructured_mesh volume {volume_id} has no tetrahedra, so a \
                         per-tet tally would have zero bins. Tet tallies need a \
                         tet-meshed volume (e.g. `CadToYamc.mesh(tet_volumes=[...])`); \
                         a surface-only mesh can be tallied with cells= or mesh= instead"
                    )));
                }
                filters.push(Filter::UnstructuredMesh(filter));
            }
            #[cfg(not(feature = "mesh"))]
            {
                let _ = um;
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unstructured_mesh requires the 'mesh' feature",
                ));
            }
        }

        if let Some(bins) = energy_bins {
            if bins.len() < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "energy_bins requires at least 2 boundaries",
                ));
            }
            for i in 1..bins.len() {
                if bins[i] <= bins[i - 1] {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "energy_bins must be in strictly ascending order",
                    ));
                }
            }
            filters.push(Filter::Energy(EnergyFilter::new(bins)));
        }

        if let Some(ref group_name) = energy_group_structure {
            let ef = EnergyFilter::from_group_structure(group_name)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            filters.push(Filter::Energy(ef));
        }

        if let Some(ef) = energy_function {
            let items: Vec<Bound<'_, PyAny>> = ef.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "energy_function must be a tuple (energy, y[, units])",
                )
            })?;
            if items.len() < 2 || items.len() > 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "energy_function must be a 2- or 3-tuple (energy, y[, units])",
                ));
            }
            let energy: Vec<f64> = items[0].extract()?;
            let y: Vec<f64> = items[1].extract()?;
            let units: Option<String> = if items.len() == 3 {
                items[2].extract().ok()
            } else {
                None
            };
            if energy.len() != y.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "energy and y arrays must have the same length",
                ));
            }
            if energy.len() < 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "energy_function requires at least 4 data points for cubic interpolation",
                ));
            }
            for i in 1..energy.len() {
                if energy[i] <= energy[i - 1] {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "energy grid must be monotonically increasing",
                    ));
                }
            }
            let mut filter = EnergyFunctionFilter::new(energy, y);
            filter.units = units;
            filters.push(Filter::EnergyFunction(filter));
        }

        if let Some(dc) = dose_coefficients {
            let items: Vec<Bound<'_, PyAny>> = dc.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "dose_coefficients must be a tuple (particle, geometry[, data_source])",
                )
            })?;
            if items.len() < 2 || items.len() > 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "dose_coefficients must be a 2- or 3-tuple (particle, geometry[, data_source])",
                ));
            }
            let dc_particle: String = items[0].extract()?;
            let dc_geometry: String = items[1].extract()?;
            let dc_source: String = if items.len() == 3 {
                items[2].extract()?
            } else {
                "icrp116".to_string()
            };

            // Call the Rust dose_coefficients function directly
            use yamc_nuclide::data::effective_dose::{
                dose_coefficients as rust_dose_coefficients, DoseDataSource, DoseGeometry,
                DoseParticle,
            };
            let dc_particle_kind = match dc_particle.as_str() {
                "neutron" => DoseParticle::Neutron,
                "photon" => DoseParticle::Photon,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Particle '{}' not supported for dose_coefficients. Must be 'neutron' or 'photon'.",
                        dc_particle
                    )))
                }
            };
            let geom = match dc_geometry.to_uppercase().as_str() {
                "AP" => DoseGeometry::AP,
                "PA" => DoseGeometry::PA,
                "LLAT" => DoseGeometry::LLAT,
                "RLAT" => DoseGeometry::RLAT,
                "ROT" => DoseGeometry::ROT,
                "ISO" => DoseGeometry::ISO,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid geometry '{}'. Must be one of: AP, PA, LLAT, RLAT, ROT, ISO",
                        dc_geometry
                    )))
                }
            };
            let source = match dc_source.to_lowercase().as_str() {
                "icrp74" => DoseDataSource::ICRP74,
                "icrp116" => DoseDataSource::ICRP116,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid data_source '{}'. Must be 'icrp74' or 'icrp116'",
                        dc_source
                    )))
                }
            };
            let (energy, coeffs) = rust_dose_coefficients(dc_particle_kind, geom, source);
            let mut filter = EnergyFunctionFilter::new(energy, coeffs);
            filter.units = Some("pSv·cm²".to_string());
            filters.push(Filter::EnergyFunction(filter));
        }

        if let Some(ref pt) = particle {
            let pt_enum = match pt.as_str() {
                "neutron" => ParticleType::Neutron,
                "photon" => ParticleType::Photon,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "particle must be 'neutron' or 'photon'",
                    ))
                }
            };
            filters.push(Filter::ParticleType(ParticleTypeFilter::new(pt_enum)));
        }

        if let Some(ref nucs) = parent_nuclides {
            // An empty list is allowed and distinct from `None`: it is a
            // decay-photon dose tally whose material has no gamma-emitting
            // activation products (e.g. pure Li6/Li7), so it correctly scores
            // zero. The parent dimension collapses to a single all-zero bin,
            // matching OpenMC's D1S, which returns a zero spectrum rather than
            // erroring. `None` still means "no parent filter" (tally all photons).
            filters.push(Filter::ParentNuclide(ParentNuclideFilter::new(
                nucs.clone(),
            )));
        }

        let mut tally = Tally::new();
        tally.name = name;
        tally.tally_id = id;
        tally.multiply_density = multiply_density;
        tally.overlay_material = overlay_material;
        tally.estimator = estimator_value;

        if let Some(scores) = scores {
            tally.set_scores_mixed(parse_scores_arg(scores)?);
        }
        if let Some(bins) = overlay_nuclide_bins {
            // Nuclide axis comes from `response` (virtual-overlay regime).
            tally.nuclides = bins;
        } else if let Some(nuclides) = nuclides {
            use yamc_tallies::NuclideBin;
            tally.nuclides = nuclides
                .into_iter()
                .map(|s| {
                    if s.eq_ignore_ascii_case("total") {
                        NuclideBin::Total
                    } else {
                        NuclideBin::Specific(s)
                    }
                })
                .collect();
        }

        // A per-nuclide axis on a score with no cross section is a construction
        // error, not a run-time one (issue #305): the user can still fix the call
        // here. Checked unconditionally, unlike the full `validate()` below,
        // which only runs when the tally has filters.
        tally
            .validate_nuclide_axis()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        if !filters.is_empty() {
            tally.filters = filters;
            tally
                .validate()
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }

        Ok(PyTally {
            inner: Arc::new(tally),
        })
    }

    /// List of scores to tally.
    ///
    /// Scores can be specified as:
    /// - Integers representing ENDF MT numbers (e.g., 1 for total cross section)
    /// - String "flux" for neutron flux
    /// - String "heating" for heating/energy deposition
    ///
    /// Returns:
    ///     List of scores (integers or strings)
    #[getter]
    pub fn scores(&self) -> Vec<Py<PyAny>> {
        Python::attach(|py| {
            self.inner
                .scores
                .iter()
                .map(|score| match score {
                    // Unnamed MT → return as Python int
                    Score::ReactionRate(r) if r.display_name.is_none() => {
                        r.mt.as_i32().into_pyobject(py).unwrap().into_any().unbind()
                    }
                    // Everything else → return as Python string
                    _ => score.name().into_pyobject(py).unwrap().into_any().unbind(),
                })
                .collect()
        })
    }

    /// List of nuclides for per-nuclide tally breakdown.
    ///
    /// When set, each score is broken down by the individual contribution
    /// of each listed nuclide. The special string "total" gives the
    /// material-total (macroscopic cross section summed over all nuclides).
    ///
    /// Compatible with the ``tally.nuclides`` property.
    ///
    /// Returns:
    ///     list[str]: Nuclide names (e.g., ['Li6', 'Li7', 'total']).
    ///         Empty list means material-total only (the default).
    ///
    /// Examples:
    ///     >>> tally.nuclides = ['Li6', 'Li7', 'Be9']
    ///     >>> tally.nuclides = ['U235', 'total']
    #[getter]
    pub fn nuclides(&self) -> Vec<String> {
        self.inner.nuclides.iter().map(|n| n.to_string()).collect()
    }

    /// Tally estimator: "track-length" (default) or "collision".
    ///
    /// "track-length" scores at every cell crossing, weighted by the
    /// segment length in each crossed bin. "collision" scores at every
    /// collision site as `weight / Σ_t`. Both converge to the same
    /// physical flux for a given history budget, with different
    /// variance characteristics -- collision is lower variance in
    /// optically-thick regions and touches fewer bins per history,
    /// which helps cache pressure on large mesh tallies.
    ///
    /// Returns:
    ///     str: "track-length" or "collision".
    #[getter]
    pub fn estimator(&self) -> &'static str {
        self.inner.estimator.as_str()
    }

    /// The virtual response target (issue #341), or ``None`` for a normal
    /// macroscopic tally.
    ///
    /// Mirrors the ``response=`` constructor argument, so one of:
    ///
    /// - ``dict[str, float]`` (nuclide -> atoms/barn-cm) for a material
    ///   response (the tally scores that material's macroscopic response).
    /// - ``list[str]`` of nuclide names for a unit-density nuclide overlay.
    /// - ``None`` for a normal (density-multiplied) macroscopic tally.
    #[getter]
    pub fn response(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if let Some(mat) = &self.inner.overlay_material {
            let dict = pyo3::types::PyDict::new(py);
            for (nuc, dens) in mat {
                dict.set_item(nuc, *dens).unwrap();
            }
            Some(dict.into_any().unbind())
        } else if !self.inner.multiply_density {
            let names: Vec<String> = self
                .inner
                .nuclides
                .iter()
                .filter_map(|n| match n {
                    yamc_tallies::NuclideBin::Specific(name) => Some(name.clone()),
                    yamc_tallies::NuclideBin::Total => None,
                })
                .collect();
            Some(names.into_pyobject(py).unwrap().into_any().unbind())
        } else {
            None
        }
    }

    /// Optional name for the tally.
    ///
    /// Returns:
    ///     String name or None
    #[getter]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Set the tally's name label.
    #[setter(name)]
    pub fn set_name(&mut self, name: Option<String>) -> PyResult<()> {
        let tally = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot modify tally: multiple references exist",
            )
        })?;
        tally.name = name;
        Ok(())
    }

    /// Optional numeric identifier for the tally.
    ///
    /// Returns:
    ///     Integer ID or None
    #[getter]
    pub fn id(&self) -> Option<u32> {
        self.inner.tally_id
    }

    /// Set the tally's numeric identifier.
    #[setter(id)]
    pub fn set_id(&mut self, id: u32) -> PyResult<()> {
        let tally = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot modify tally: multiple references exist",
            )
        })?;
        tally.tally_id = Some(id);
        Ok(())
    }

    /// Automatically derived physical units for each score.
    ///
    /// Units are derived from the score type and filters:
    /// - Flux: "cm / source-particle" (or "cm / cm³ / source-particle" with mesh)
    /// - Heating: "eV / source-particle"
    /// - Reactions: "reactions / source-particle"
    /// - With an ``energy_function=`` argument: uses its user-supplied units
    ///
    /// Returns:
    ///     list[str]: Units string for each score
    #[getter]
    pub fn units(&self) -> Vec<String> {
        self.inner.derive_units()
    }

    // -----------------------------------------------------------------------
    // Filter keyword properties (read-write)
    // -----------------------------------------------------------------------

    /// The RegularRectangularMesh used for spatial binning, or None (also None
    /// when the tally uses a cylindrical mesh -- see :attr:`cylindrical_mesh`).
    #[getter]
    pub fn mesh(&self) -> Option<PyRegularRectangularMesh> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::Mesh(mf) = f {
                mf.rectangular_mesh().map(|m| PyRegularRectangularMesh {
                    internal: m.clone(),
                })
            } else {
                None
            }
        })
    }

    /// The RegularCylindricalMesh used for spatial binning, or None.
    #[getter]
    pub fn cylindrical_mesh(&self) -> Option<PyRegularCylindricalMesh> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::Mesh(mf) = f {
                mf.cylindrical_mesh().map(|m| PyRegularCylindricalMesh {
                    internal: m.clone(),
                })
            } else {
                None
            }
        })
    }

    /// The particle type filter: "neutron", "photon", or None.
    #[getter]
    pub fn particle(&self) -> Option<String> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::ParticleType(pf) = f {
                Some(match pf.particle_type {
                    ParticleType::Neutron => "neutron".to_string(),
                    ParticleType::Photon => "photon".to_string(),
                })
            } else {
                None
            }
        })
    }

    /// Parent nuclide names for D1S binning, or None.
    #[getter]
    pub fn parent_nuclides(&self) -> Option<Vec<String>> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::ParentNuclide(pnf) = f {
                Some(pnf.nuclides.clone())
            } else {
                None
            }
        })
    }

    /// Energy bin boundaries in eV, or None.
    #[getter]
    pub fn energy_bins(&self) -> Option<Vec<f64>> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::Energy(ef) = f {
                Some(ef.bins.clone())
            } else {
                None
            }
        })
    }

    /// Energy function filter data as (energy, y, units) tuple, or None.
    #[getter]
    pub fn energy_function(&self) -> Option<(Vec<f64>, Vec<f64>, Option<String>)> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::EnergyFunction(ef) = f {
                Some((ef.energy().to_vec(), ef.y().to_vec(), ef.units.clone()))
            } else {
                None
            }
        })
    }

    /// Cell IDs of the cell filter, or None if no cell filter is set.
    #[getter]
    pub fn cells(&self) -> Option<Vec<u32>> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::Cell(cf) = f {
                Some(cf.cell_ids.clone())
            } else {
                None
            }
        })
    }

    /// Set the cell filter. Accepts a single Cell or a list of Cells.
    #[setter]
    pub fn set_cells(&mut self, cells: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let tally = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot modify tally: multiple references exist",
            )
        })?;
        tally.filters.retain(|f| !matches!(f, Filter::Cell(_)));
        if let Some(cs) = cells {
            let cell_ids = cell_filter_ids(cs)?;
            tally
                .filters
                .push(Filter::Cell(CellFilter::from_ids(cell_ids)));
        }
        Ok(())
    }

    /// Material IDs of the material filter, or None if no material filter is set.
    #[getter]
    pub fn materials(&self) -> Option<Vec<u32>> {
        self.inner.filters.iter().find_map(|f| {
            if let Filter::Material(mf) = f {
                Some(mf.material_ids.clone())
            } else {
                None
            }
        })
    }

    /// Set the material filter. Accepts a single Material or a list of Materials.
    #[setter]
    pub fn set_materials(&mut self, materials: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let tally = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot modify tally: multiple references exist",
            )
        })?;
        tally.filters.retain(|f| !matches!(f, Filter::Material(_)));
        if let Some(ms) = materials {
            let py_mats: Vec<pyo3::PyRef<'_, PyMaterial>> =
                if let Ok(list) = ms.extract::<Vec<pyo3::PyRef<'_, PyMaterial>>>() {
                    list
                } else if let Ok(single) = ms.extract::<pyo3::PyRef<'_, PyMaterial>>() {
                    vec![single]
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "materials must be a Material or a list of Material instances",
                    ));
                };
            if py_mats.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "materials must contain at least one Material",
                ));
            }
            let mat_refs: Vec<&yamc_materials::material::Material> =
                py_mats.iter().map(|m| &m.internal).collect();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                MaterialFilter::from_materials(&mat_refs)
            })) {
                Ok(filter) => tally.filters.push(Filter::Material(filter)),
                Err(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Cannot filter on material with no ID - assign an id first",
                    ))
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers for vtkhdf.py
    // -----------------------------------------------------------------------

    /// Number of energy bins (1 if no energy filter).
    #[getter]
    pub fn n_energy_bins(&self) -> usize {
        self.inner
            .filters
            .iter()
            .find_map(|f| {
                if let Filter::Energy(ef) = f {
                    Some(ef.num_bins())
                } else {
                    None
                }
            })
            .unwrap_or(1)
    }

    /// Number of parent nuclide bins (1 if no parent nuclide filter).
    #[getter]
    pub fn n_parent_bins(&self) -> usize {
        self.inner
            .filters
            .iter()
            .find_map(|f| {
                if let Filter::ParentNuclide(pnf) = f {
                    Some(pnf.num_bins())
                } else {
                    None
                }
            })
            .unwrap_or(1)
    }

    /// Number of mesh bins (0 if no mesh filter).
    #[getter]
    pub fn n_mesh_bins(&self) -> usize {
        self.inner
            .filters
            .iter()
            .find_map(|f| match f {
                Filter::Mesh(mf) => Some(mf.num_bins()),
                #[cfg(feature = "mesh")]
                Filter::UnstructuredMesh(umf) => Some(umf.num_bins()),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Whether this tally uses an unstructured mesh.
    #[getter]
    pub fn has_unstructured_mesh(&self) -> bool {
        self.inner.filters.iter().any(|f| {
            #[cfg(feature = "mesh")]
            if matches!(f, Filter::UnstructuredMesh(_)) {
                return true;
            }
            let _ = f;
            false
        })
    }

    /// Unstructured mesh vertices (for vtkhdf export). None if no unstructured mesh.
    #[getter]
    pub fn unstructured_mesh_vertices(&self) -> Option<Vec<[f64; 3]>> {
        #[cfg(feature = "mesh")]
        for f in &self.inner.filters {
            if let Filter::UnstructuredMesh(umf) = f {
                return Some(umf.mesh().topology.vertices.clone());
            }
        }
        None
    }

    /// Unstructured mesh connectivity (for vtkhdf export). None if no unstructured mesh.
    #[getter]
    pub fn unstructured_mesh_connectivity(&self) -> Option<Vec<[u32; 4]>> {
        #[cfg(feature = "mesh")]
        for f in &self.inner.filters {
            if let Filter::UnstructuredMesh(umf) = f {
                return Some(umf.mesh().topology.tetrahedra.clone());
            }
        }
        None
    }

    /// Volumes of every mesh cell in cm³, in one call, for any mesh filter
    /// (rectangular, cylindrical, or unstructured). ``None`` if the tally has
    /// no mesh filter.
    pub fn mesh_element_volumes(&self) -> Option<Vec<f64>> {
        for f in &self.inner.filters {
            if let Filter::Mesh(mf) = f {
                return Some(
                    (0..mf.num_bins())
                        .map(|b| mf.get_element_volume(b))
                        .collect(),
                );
            }
            #[cfg(feature = "mesh")]
            if let Filter::UnstructuredMesh(umf) = f {
                return Some(
                    (0..umf.num_bins())
                        .map(|b| umf.get_element_volume(b))
                        .collect(),
                );
            }
        }
        None
    }

    /// Volume of mesh cell ``bin`` in cm³, for any mesh filter (rectangular,
    /// cylindrical, or unstructured). ``None`` if the tally has no mesh.
    ///
    /// For a rectangular mesh every cell shares the same volume; for a
    /// cylindrical mesh outer radial rings are larger; for an unstructured
    /// mesh each tetrahedron has its own volume.
    #[pyo3(signature = (bin))]
    pub fn mesh_element_volume(&self, bin: usize) -> Option<f64> {
        for f in &self.inner.filters {
            if let Filter::Mesh(mf) = f {
                return Some(mf.get_element_volume(bin));
            }
            #[cfg(feature = "mesh")]
            if let Filter::UnstructuredMesh(umf) = f {
                return Some(umf.get_element_volume(bin));
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Accumulated results (populated by `simulate_transport`)
    // -----------------------------------------------------------------------

    /// Per-bin mean value, flat over
    /// ``[score][cell][material][nuclide][parent][energy][mesh]``
    /// (mesh fastest; dimensions without a filter have a single bin).
    ///
    /// Returns:
    ///     list[float]: Mean per bin. All zeros before a simulation has run.
    #[getter]
    pub fn mean(&self) -> Vec<f64> {
        self.inner.get_mean()
    }

    /// Per-bin standard error of the mean, flat in the same bin order
    /// as :attr:`mean`.
    ///
    /// Returns:
    ///     list[float]: Standard error of the mean per bin. All zeros
    ///         before a simulation has run.
    #[getter]
    pub fn standard_deviation(&self) -> Vec<f64> {
        self.inner.get_std_dev()
    }

    /// Per-bin relative error (``standard_deviation / mean``, 0 where the
    /// mean is not positive), flat in the same bin order as :attr:`mean`.
    ///
    /// Returns:
    ///     list[float]: Relative error per bin.
    #[getter]
    pub fn relative_error(&self) -> Vec<f64> {
        self.inner.get_rel_error()
    }

    /// Extract a 2D slice of mesh tally data for plotting.
    ///
    /// Returns a 2D list of tally values for the slice plane. The shape
    /// is ``[n_vertical][n_horizontal]`` matching the mesh dimensions for
    /// the chosen basis.
    ///
    /// Args:
    ///     basis: Slice plane - "xy", "xz", or "yz"
    ///     slice_coord: Float position along the fixed axis (default: mesh center)
    ///     score_index: Which score to extract, 0-indexed (default: 0)
    ///     energy_index: Which energy bin (default: sum over all)
    ///     value: Which statistic - "mean", "standard_deviation", or "relative_error" (default: "mean")
    ///
    /// Returns:
    ///     A :class:`MeshSliceData` with ``values``, ``h_edges``, ``v_edges``,
    ///     ``extent``, ``h_label``, ``v_label``.
    ///     Also behaves like a 2D list for backward compatibility
    ///     (``np.array(tally.extract_mesh_slice(...))`` still works).
    ///
    /// Raises:
    ///     ValueError: If the tally has no mesh or parameters are invalid
    #[pyo3(signature = (basis="xy", slice_coord=None, score_index=0, energy_index=None, value="mean"))]
    pub fn extract_mesh_slice(
        &self,
        basis: &str,
        slice_coord: Option<f64>,
        score_index: usize,
        energy_index: Option<usize>,
        value: &str,
    ) -> PyResult<PyMeshSliceData> {
        use crate::id_map_helper::basis_axes;

        let values = self
            .inner
            .extract_mesh_slice(basis, slice_coord, score_index, energy_index, value)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let mesh_filter = self.inner.get_mesh_filter().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "tally has no mesh (pass mesh= to the Tally constructor)",
            )
        })?;
        let mesh = mesh_filter.rectangular_mesh().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "mesh slice plotting requires a rectangular mesh",
            )
        })?;
        let ll = mesh.lower_left();
        let ur = mesh.upper_right();
        let dim = mesh.shape();

        let (h_idx, v_idx, _, h_label, v_label, _) =
            basis_axes(basis).map_err(pyo3::exceptions::PyValueError::new_err)?;

        let h_edges: Vec<f64> = (0..=dim[h_idx])
            .map(|i| ll[h_idx] + (ur[h_idx] - ll[h_idx]) * (i as f64) / (dim[h_idx] as f64))
            .collect();
        let v_edges: Vec<f64> = (0..=dim[v_idx])
            .map(|i| ll[v_idx] + (ur[v_idx] - ll[v_idx]) * (i as f64) / (dim[v_idx] as f64))
            .collect();

        let extent = (
            h_edges[0],
            *h_edges.last().unwrap(),
            v_edges[0],
            *v_edges.last().unwrap(),
        );

        Ok(PyMeshSliceData {
            values,
            h_edges,
            v_edges,
            extent,
            h_label: h_label.to_string(),
            v_label: v_label.to_string(),
        })
    }

    /// Create an interactive WASM-based mesh tally viewer.
    ///
    /// Returns an :class:`InteractiveTallyPlot` that renders inline in Jupyter
    /// and can be saved as a self-contained HTML file.
    ///
    /// Args:
    ///     geometry: Optional Geometry or MeshGeometry for outline overlay.
    ///     basis: Initial slice plane ("xy", "xz", or "yz").
    ///     slice_coord: Initial slice coordinate (default: mesh center).
    ///     slices: Which slices to embed. ``None`` = just the initial slice,
    ///         a list of ints = specific bin indices, ``"all"`` = every bin.
    ///     score_index: Which score to display.
    ///     energy_index: Energy bin (``None`` = sum over energy).
    ///     value: "mean", "standard_deviation", or "relative_error".
    ///     colorscale: Colorscale name (Viridis, Cividis, Hot, etc.).
    ///     log_scale: Use logarithmic color scale.
    ///     outline: "material", "cell", or ``None``.
    ///     resolution: Total pixel budget for geometry outline sampling.
    ///     axis_units: "mm", "cm", "m", or "km".
    ///     contour_kwargs: Dict with "colors" and "linewidths" for outlines.
    ///     title: Plot title (auto-generated if None).
    ///     colorbar_title: Colorbar label (auto-generated if None).
    ///     scaling_factor: Multiply all tally values by this factor (default 1.0).
    ///     font_size: Base font size in pixels for labels and ticks (default 12).
    ///
    /// Returns:
    ///     InteractiveTallyPlot
    #[gen_stub(skip)] // FIXME(stub): pyo3-stub-gen mishandles Option<&str> default; hand-add in stub patch-merge
    #[pyo3(signature = (geometry=None, basis="xy", slice_coord=None, slices=None, score_index=0, energy_index=None, value=None, colorscale="Viridis", log_scale=true, outline="material", resolution=400000, axis_units="cm", contour_kwargs=None, title=None, colorbar_title=None, scaling_factor=1.0, font_size=18, show_colorbar=true))]
    pub fn plot(
        &self,
        geometry: Option<&Bound<'_, PyAny>>,
        basis: &str,
        slice_coord: Option<f64>,
        slices: Option<&Bound<'_, PyAny>>,
        score_index: usize,
        energy_index: Option<usize>,
        value: Option<&Bound<'_, PyAny>>,
        colorscale: &str,
        log_scale: bool,
        outline: Option<&str>,
        resolution: Option<usize>,
        axis_units: &str,
        contour_kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
        title: Option<&str>,
        colorbar_title: Option<&str>,
        scaling_factor: f64,
        font_size: usize,
        show_colorbar: bool,
    ) -> PyResult<crate::ui::PyInteractiveTallyPlot> {
        use crate::ui::parse_contour_kwargs;
        use crate::ui::{
            build_interactive_tally_html, EmbeddedSlice, InteractiveTallyParams, MeshMeta,
            PyInteractiveTallyPlot,
        };

        // Validate basis
        if !["xy", "xz", "yz"].contains(&basis) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "basis must be 'xy', 'xz', or 'yz'",
            ));
        }

        // Get mesh info
        let mesh_filter = self.inner.get_mesh_filter().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "tally has no mesh (pass mesh= to the Tally constructor)",
            )
        })?;
        let mesh = mesh_filter.rectangular_mesh().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "mesh slice plotting requires a rectangular mesh",
            )
        })?;
        let ll = mesh.lower_left();
        let ur = mesh.upper_right();
        let dim = mesh.shape();
        let width = mesh.width();

        let (_, _, fixed_axis, _, _, _) = crate::id_map_helper::basis_axes(basis).unwrap();
        let fixed_dim = dim[fixed_axis];

        // Determine which slice bin indices to embed
        let slice_indices: Vec<usize> = if let Some(s) = slices {
            if let Ok(s_str) = s.extract::<String>() {
                if s_str == "all" {
                    (0..fixed_dim).collect()
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "slices must be None, 'all', or a list of ints",
                    ));
                }
            } else if let Ok(s_list) = s.extract::<Vec<usize>>() {
                s_list
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "slices must be None, 'all', or a list of ints",
                ));
            }
        } else {
            // Default: single center slice
            let coord = slice_coord.unwrap_or((ll[fixed_axis] + ur[fixed_axis]) / 2.0);
            let idx = ((coord - ll[fixed_axis]) / width[fixed_axis]).floor() as usize;
            vec![idx.min(fixed_dim - 1)]
        };

        // Parse value: str, list of str, or None (default: "mean")
        let values: Vec<String> = if let Some(val) = value {
            if let Ok(s) = val.extract::<String>() {
                vec![s]
            } else if let Ok(list) = val.extract::<Vec<String>>() {
                if list.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "value list must not be empty",
                    ));
                }
                list
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "value must be a string or list of strings ('mean', 'standard_deviation', 'relative_error')",
                ));
            }
        } else {
            vec!["mean".to_string()]
        };
        let display_value = &values[0];

        // Extract slices for the display value (heatmap)
        let raw_slices = self
            .inner
            .extract_mesh_slices(
                basis,
                &slice_indices,
                score_index,
                energy_index,
                display_value,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Extract slices for extra values (tooltip only)
        let mut extra_value_slices: Vec<(String, Vec<(usize, Vec<f64>)>)> = Vec::new();
        for v in &values[1..] {
            let extra = self
                .inner
                .extract_mesh_slices(basis, &slice_indices, score_index, energy_index, v)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            extra_value_slices.push((v.clone(), extra));
        }

        // Size warning
        let extra_bytes: usize = extra_value_slices
            .iter()
            .flat_map(|(_, slices)| slices.iter().map(|(_, d)| d.len() * 8))
            .sum();
        let total_bytes: usize =
            raw_slices.iter().map(|(_, d)| d.len() * 8).sum::<usize>() + extra_bytes;
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        // base64 is ~4/3 of raw + HTML overhead
        let est_html_mb = total_mb * 1.4 + 0.5;
        if est_html_mb > 16.0 {
            pyo3::Python::attach(|py| {
                let warnings = py.import("warnings").ok();
                if let Some(w) = warnings {
                    let _ = w.call_method1(
                        "warn",
                        (format!(
                            "Embedding {} slices will produce ~{:.0} MB HTML. Consider reducing slices.",
                            slice_indices.len(),
                            est_html_mb
                        ),),
                    );
                }
            });
        }

        let embedded_slices: Vec<EmbeddedSlice> = raw_slices
            .into_iter()
            .map(|(bin_idx, data)| EmbeddedSlice {
                basis: basis.to_string(),
                bin_index: bin_idx,
                data,
            })
            .collect();

        // Build extra value slices for tooltip
        let extra_values: Vec<crate::ui::ExtraValueSlices> = extra_value_slices
            .into_iter()
            .map(|(vname, raw)| crate::ui::ExtraValueSlices {
                value_name: vname,
                slices: raw
                    .into_iter()
                    .map(|(bin_idx, data)| EmbeddedSlice {
                        basis: basis.to_string(),
                        bin_index: bin_idx,
                        data,
                    })
                    .collect(),
            })
            .collect();

        let initial_slice_index = slice_indices[0].min(fixed_dim - 1);

        // Parse contour kwargs
        let contour = parse_contour_kwargs(contour_kwargs)?;

        // Build geometry JSON if provided. The optional surface_table
        // carries names for the surface-hover tooltip; only populated for
        // CSG geometries (mesh has no CSG surfaces to walk).
        let (geometry_json, geometry_kind, cell_names, material_names, surface_table) = {
            let csg_geom: Option<pyo3::PyRef<'_, PyGeometry>> =
                geometry.and_then(|g| g.extract().ok());
            #[cfg(feature = "mesh")]
            let mesh_geom: Option<pyo3::PyRef<'_, PyMeshGeometry>> = if csg_geom.is_none() {
                geometry.and_then(|g| g.extract().ok())
            } else {
                None
            };
            #[cfg(not(feature = "mesh"))]
            let mesh_geom: Option<()> = None;

            if geometry.is_some() && csg_geom.is_none() && mesh_geom.is_none() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "geometry must be a Geometry or MeshGeometry instance",
                ));
            }

            if let Some(ref g) = csg_geom {
                use yamc::geometry::conversion::geometry_to_csg;
                let csg = geometry_to_csg(&g.inner);
                let json = serde_json::to_string(&csg)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                // Collect names
                let mut cn = std::collections::HashMap::new();
                let mut mn = std::collections::HashMap::new();
                for cell in &g.inner.cells {
                    let cid = cell.cell_id.map(|id| id as i32).unwrap_or(-1);
                    if let Some(name) = &cell.name {
                        cn.entry(cid).or_insert_with(|| name.clone());
                    }
                    if let Some(mat) = cell
                        .material_idx
                        .and_then(|i| g.inner.materials.get(i as usize))
                    {
                        let mid = mat.get_material_id().map(|id| id as i32).unwrap_or(-1);
                        if let Some(name) = mat.get_name() {
                            mn.entry(mid).or_insert_with(|| name.to_string());
                        }
                    }
                }
                let surface_table = yamc_plot::build_surface_table(&csg);
                (Some(json), Some("csg"), cn, mn, Some(surface_table))
            } else if let Some(ref _m) = mesh_geom {
                #[cfg(feature = "mesh")]
                {
                    use yamc::geometry::conversion::mesh_geometry_to_geo_mesh;
                    let geo_mesh = mesh_geometry_to_geo_mesh(&_m.inner);
                    let json = serde_json::to_string(&geo_mesh)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    let mut cn = std::collections::HashMap::new();
                    let mut mn = std::collections::HashMap::new();
                    for cell in &_m.inner.cells {
                        let cid = cell.cell_id.map(|id| id as i32).unwrap_or(-1);
                        if let Some(name) = &cell.name {
                            cn.entry(cid).or_insert_with(|| name.clone());
                        }
                        if let Some(mat) = cell
                            .material_idx
                            .and_then(|i| _m.inner.materials.get(i as usize))
                        {
                            let mid = mat.get_material_id().map(|id| id as i32).unwrap_or(-1);
                            if let Some(name) = mat.get_name() {
                                mn.entry(mid).or_insert_with(|| name.to_string());
                            }
                        }
                    }
                    // Mesh has no CSG surfaces -- no surface_table.
                    (Some(json), Some("mesh"), cn, mn, None)
                }
                #[cfg(not(feature = "mesh"))]
                {
                    (
                        None,
                        None,
                        std::collections::HashMap::new(),
                        std::collections::HashMap::new(),
                        None,
                    )
                }
            } else {
                (
                    None,
                    None,
                    std::collections::HashMap::new(),
                    std::collections::HashMap::new(),
                    None,
                )
            }
        };

        // Build title
        let fixed_label = match basis {
            "xy" => "z",
            "xz" => "y",
            "yz" => "x",
            _ => "?",
        };
        let slice_pos = slice_coord.unwrap_or((ll[fixed_axis] + ur[fixed_axis]) / 2.0);
        let auto_title = format!(
            "Tally {} ({} plane at {} = {:.4} cm)",
            display_value,
            basis.to_uppercase(),
            fixed_label,
            slice_pos
        );
        let title = title.unwrap_or(&auto_title).to_string();

        let units = self.inner.derive_units();
        let score_unit = units
            .get(score_index)
            .cloned()
            .unwrap_or_else(|| "per source-particle".to_string());
        let auto_cb_title = format!(
            "{} [{}]",
            self.inner
                .name
                .clone()
                .unwrap_or_else(|| "Tally".to_string()),
            score_unit
        );
        let colorbar_title = colorbar_title.unwrap_or(&auto_cb_title).to_string();

        let mesh_meta = MeshMeta {
            lower_left: ll,
            upper_right: ur,
            shape: dim,
            width,
        };

        // Precompute outline grid for the initial slice (instant first render)
        let presampled_outline = if outline.is_some() {
            use crate::id_map_helper::{cell_to_ids, generate_presampled_grid, IdMapParams};
            let csg_geom: Option<pyo3::PyRef<'_, PyGeometry>> =
                geometry.and_then(|g| g.extract().ok());
            if let Some(ref g) = csg_geom {
                let op = resolution.unwrap_or(400000);
                let (h_idx, v_idx, _, _, _, _) = crate::id_map_helper::basis_axes(basis).unwrap();
                let mesh_w_h = ur[h_idx] - ll[h_idx];
                let mesh_w_v = ur[v_idx] - ll[v_idx];
                let aspect = mesh_w_h / mesh_w_v;
                let pv = (((op as f64) / aspect).sqrt().round() as usize).max(1);
                let ph = (((op as f64) / (pv as f64)).round() as usize).max(1);
                let center_h = (ll[h_idx] + ur[h_idx]) / 2.0;
                let center_v = (ll[v_idx] + ur[v_idx]) / 2.0;
                let fixed_coord =
                    ll[fixed_axis] + width[fixed_axis] * (initial_slice_index as f64 + 0.5);
                let origin = match basis {
                    "xy" => (center_h, center_v, fixed_coord),
                    "xz" => (center_h, fixed_coord, center_v),
                    _ => (fixed_coord, center_h, center_v),
                };
                let params = IdMapParams {
                    origin,
                    width: (mesh_w_h, mesh_w_v),
                    pixels: (ph, pv),
                    basis: basis.to_string(),
                };
                let geom = &g.inner;
                Some(generate_presampled_grid(&params, |point| {
                    geom.find_cell(point)
                        .map(|c| cell_to_ids(c, &geom.materials))
                        .unwrap_or((-1, -1))
                }))
            } else {
                None // Mesh geometry -- WASM handles it
            }
        } else {
            None
        };

        let tally_params = InteractiveTallyParams {
            initial_basis: basis.to_string(),
            initial_slice_index,
            colorscale: colorscale.to_string(),
            log_scale,
            outline: outline.map(|s| s.to_string()),
            outline_color: contour.colors.clone(),
            outline_thickness: contour.linewidths,
            axis_units: axis_units.to_string(),
            outline_pixels: resolution.unwrap_or(400000),
            title,
            colorbar_title,
            scaling_factor,
            font_size,
            show_colorbar,
            display_value: display_value.clone(),
        };

        let html = build_interactive_tally_html(
            geometry_json.as_deref(),
            geometry_kind,
            &mesh_meta,
            &embedded_slices,
            &extra_values,
            &tally_params,
            &cell_names,
            &material_names,
            presampled_outline.as_ref(),
            surface_table.as_deref(),
        );

        Ok(PyInteractiveTallyPlot::new(html))
    }

    fn __repr__(&self) -> String {
        // Format scores as Python list: integers for unnamed MT, strings for named scores
        let scores_str = self
            .inner
            .scores
            .iter()
            .map(|score| match score {
                Score::ReactionRate(r) if r.display_name.is_none() => r.mt.as_i32().to_string(),
                _ => format!("\"{}\"", score.name()),
            })
            .collect::<Vec<_>>()
            .join(", ");

        let units = self.inner.derive_units();
        let units_str = units
            .iter()
            .map(|u| format!("\"{}\"", u))
            .collect::<Vec<_>>()
            .join(", ");

        let name_str = match &self.inner.name {
            Some(n) => format!("\"{}\"", n),
            None => "None".to_string(),
        };
        let id_str = match self.inner.tally_id {
            Some(id) => id.to_string(),
            None => "None".to_string(),
        };

        format!(
            "Tally(scores=[{}], name={}, id={}, units=[{}])",
            scores_str, name_str, id_str, units_str
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

impl From<Tally> for PyTally {
    fn from(tally: Tally) -> Self {
        PyTally {
            inner: Arc::new(tally),
        }
    }
}

pub fn register_tally_classes(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTally>()?;
    Ok(())
}

/// Look up a volume ID by its material name (from a ``mat:<name>`` physical
/// group), for the unstructured-mesh `unstructured_mesh=(mesh, "mat:<name>")`
/// selector handled in `PyTally::new`.
#[cfg(feature = "mesh")]
fn resolve_volume_name(mesh: &yamt::MeshGeometry, name: &str) -> PyResult<u32> {
    for vol_id in 0..mesh.topology.num_volumes {
        if mesh.material_name(vol_id) == Some(name) {
            return Ok(vol_id);
        }
    }
    // Build list of known names for the error message
    let known: Vec<String> = (0..mesh.topology.num_volumes)
        .filter_map(|v| mesh.material_name(v).map(|n| format!("\"{n}\"")))
        .collect();
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "No volume with material name \"{name}\". Known names: [{}]",
        known.join(", ")
    )))
}
