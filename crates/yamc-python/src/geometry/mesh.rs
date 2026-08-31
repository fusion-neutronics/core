use crate::geometry::PyBoundingBox;
use crate::geometry::PyCell;
use crate::geometry::PyGeometry;
use crate::geometry::PyRegion;
use crate::material::PyMaterial;
use crate::simulation::PyModel;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_tallies::{CylindricalMesh, RegularRectangularMesh};

#[cfg(feature = "mesh")]
use crate::geometry::PyMeshGeometry;

/// An axis-aligned regular (structured) mesh used for spatial tally binning.
///
/// A RegularRectangularMesh divides a rectangular box (``lower_left`` to ``upper_right``)
/// into a grid of voxels of given ``shape``. Pass it to a :class:`Tally` via
/// ``mesh=`` to score quantities per voxel.
///
/// Construct from explicit corners and shape, or from a geometry's bounding
/// box via :meth:`RegularRectangularMesh.from_domain` (which accepts either a target
/// total voxel count as an int, or per-axis counts as a 3-tuple).
///
/// Examples:
///     >>> import yamc
///     >>> mesh = yamc.RegularRectangularMesh(
///     ...     lower_left=[-10, -10, -10],
///     ...     upper_right=[10, 10, 10],
///     ...     shape=[20, 20, 20],
///     ... )
///     >>> # shape may also be a single int (total voxels), split over the box:
///     >>> mesh = yamc.RegularRectangularMesh(
///     ...     lower_left=[-10, -10, -10], upper_right=[10, 10, 10], shape=8000,
///     ... )
///     >>> # Or fit to the geometry:
///     >>> mesh = yamc.RegularRectangularMesh.from_domain(geometry, shape=1_000_000)
///
/// Args:
///     lower_left: Lower-left corner of the mesh box, ``[x, y, z]`` in cm.
///     upper_right: Upper-right corner of the mesh box, ``[x, y, z]`` in cm.
///     shape: Per-axis voxel counts ``[nx, ny, nz]``, or a single int for the
///         total voxel count (split into roughly cubic bins using the box size).
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "RegularRectangularMesh", from_py_object)]
#[derive(Clone)]
pub struct PyRegularRectangularMesh {
    pub internal: RegularRectangularMesh,
}

/// Compute per-axis bin counts from a total cell count, distributing
/// bins proportionally to produce roughly cubic voxels.
fn shape_from_int(n: usize, width: [f64; 3]) -> [usize; 3] {
    let volume = width[0] * width[1] * width[2];
    let ideal = (volume / n as f64).cbrt();
    let nx = (width[0] / ideal).round().max(1.0) as usize;
    let ny = (width[1] / ideal).round().max(1.0) as usize;
    let nz = (width[2] / ideal).round().max(1.0) as usize;
    [nx, ny, nz]
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegularRectangularMesh {
    #[new]
    fn new(
        lower_left: [f64; 3],
        upper_right: [f64; 3],
        shape: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        // `shape` may be per-axis counts ``[nx, ny, nz]`` or a single int (total
        // voxel count, split into roughly cubic bins). The corners give the box
        // widths, so a single int can be distributed just like `from_domain`.
        let width = [
            upper_right[0] - lower_left[0],
            upper_right[1] - lower_left[1],
            upper_right[2] - lower_left[2],
        ];
        let resolved = parse_shape(Some(shape), width)?;
        Ok(PyRegularRectangularMesh {
            internal: RegularRectangularMesh::try_new(lower_left, upper_right, resolved)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }

    /// Create a mesh from a geometry domain object.
    ///
    /// The mesh extents are derived automatically from the domain's bounding
    /// box.  ``shape`` can be a single integer (total cell count --
    /// bins are distributed to produce roughly cubic voxels) or a list of
    /// three integers ``[nx, ny, nz]``.
    ///
    /// Args:
    ///     domain: A Region, Cell, Geometry, Model, MeshGeometry, or BoundingBox.
    ///     shape: Total number of mesh cells (int) or per-axis bin counts
    ///         ``[nx, ny, nz]``.  Defaults to ``1000``.
    ///     material: Optional Material -- only valid with MeshGeometry domains.
    ///         When given, the mesh covers only volumes matching that material.
    ///
    /// Returns:
    ///     RegularRectangularMesh: A new mesh covering the domain.
    #[classmethod]
    #[pyo3(signature = (domain, shape=None, material=None))]
    fn from_domain(
        _cls: &Bound<'_, PyType>,
        domain: &Bound<'_, pyo3::PyAny>,
        shape: Option<&Bound<'_, pyo3::PyAny>>,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<Self> {
        // --- Extract bounding box from domain ---------------------------
        let (lower_left, upper_right) = extract_bbox(domain, material.as_deref())?;

        // --- Validate bbox is finite and positive -----------------------
        for i in 0..3 {
            if !lower_left[i].is_finite() || !upper_right[i].is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Domain has infinite bounding box; specify a bounded domain",
                ));
            }
            if upper_right[i] <= lower_left[i] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Bounding box has zero or negative extent on axis {i}"
                )));
            }
        }

        // --- Parse shape ------------------------------------------------
        let width = [
            upper_right[0] - lower_left[0],
            upper_right[1] - lower_left[1],
            upper_right[2] - lower_left[2],
        ];
        let resolved_shape = parse_shape(shape, width)?;

        Ok(PyRegularRectangularMesh {
            internal: RegularRectangularMesh::try_new(lower_left, upper_right, resolved_shape)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }

    /// Lower-left corner of the mesh box, ``[x, y, z]`` in cm.
    #[getter]
    fn lower_left(&self) -> [f64; 3] {
        self.internal.lower_left()
    }

    /// Upper-right corner of the mesh box, ``[x, y, z]`` in cm.
    #[getter]
    fn upper_right(&self) -> [f64; 3] {
        self.internal.upper_right()
    }

    /// Number of voxels along each axis, ``[nx, ny, nz]``.
    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.internal.shape()
    }

    /// Voxel width along each axis, ``[dx, dy, dz]`` in cm.
    #[getter]
    fn width(&self) -> [f64; 3] {
        self.internal.width()
    }

    /// Total number of voxels (``nx * ny * nz``).
    ///
    /// This is also the length of a mesh tally's flat result array, and the
    /// number of weight-window bounds expected per energy group.
    #[getter]
    fn num_bins(&self) -> usize {
        self.internal.num_voxels()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegularRectangularMesh(lower_left={:?}, upper_right={:?}, shape={:?})",
            self.internal.lower_left(),
            self.internal.upper_right(),
            self.internal.shape()
        )
    }
}

/// Try to extract a `(lower_left, upper_right)` pair from a Python domain object.
fn extract_bbox(
    domain: &Bound<'_, pyo3::PyAny>,
    material: Option<&PyMaterial>,
) -> PyResult<([f64; 3], [f64; 3])> {
    // Reject `material` early for non-MeshGeometry types.
    // (MeshGeometry branch handles it below.)

    // 1. BoundingBox
    if let Ok(bb) = domain.cast::<PyBoundingBox>() {
        if material.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "material argument is only valid with MeshGeometry domains",
            ));
        }
        let bb = bb.borrow();
        return Ok((bb.lower_left(), bb.upper_right()));
    }

    // 2. Region
    if let Ok(r) = domain.extract::<PyRegion>() {
        if material.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "material argument is only valid with MeshGeometry domains",
            ));
        }
        let bbox = r.region.bounding_box();
        return Ok((bbox.lower_left, bbox.upper_right));
    }

    // 3. Cell
    if let Ok(c) = domain.extract::<PyCell>() {
        if material.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "material argument is only valid with MeshGeometry domains",
            ));
        }
        let bbox = c.inner.region.bounding_box();
        return Ok((bbox.lower_left, bbox.upper_right));
    }

    // 4. Geometry
    if let Ok(g) = domain.extract::<PyGeometry>() {
        if material.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "material argument is only valid with MeshGeometry domains",
            ));
        }
        let bbox = g.inner.bounding_box();
        return Ok((bbox.lower_left, bbox.upper_right));
    }

    // 5. Model
    if let Ok(m) = domain.extract::<PyModel>() {
        if material.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "material argument is only valid with MeshGeometry domains",
            ));
        }
        let bbox = m.inner.geometry.bounding_box();
        return Ok((bbox.lower_left, bbox.upper_right));
    }

    // 6. MeshGeometry (feature-gated)
    #[cfg(feature = "mesh")]
    if let Ok(mg) = domain.extract::<PyMeshGeometry>() {
        if let Some(mat) = material {
            let mat_id = mat.internal.get_material_id().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "Material must have an ID set to filter MeshGeometry volumes",
                )
            })?;
            let bbox = mg.inner.bounding_box_for_material(mat_id).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No MeshGeometry volumes match material ID {mat_id}"
                ))
            })?;
            return Ok((bbox.lower_left, bbox.upper_right));
        }
        let bbox = mg.inner.bounding_box();
        return Ok((bbox.lower_left, bbox.upper_right));
    }

    // If we have material but didn't match MeshGeometry above
    if material.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "material argument is only valid with MeshGeometry domains",
        ));
    }

    let type_name = domain
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "domain must be a Region, Cell, Geometry, Model, MeshGeometry, or BoundingBox, \
         got {type_name}",
    )))
}

/// Parse the `shape` argument into `[usize; 3]`.
///
/// - `None` → default 1000 total cells (proportional)
/// - Single int → proportional distribution for roughly cubic voxels
/// - List/tuple of 3 ints → used directly
fn parse_shape(shape: Option<&Bound<'_, pyo3::PyAny>>, width: [f64; 3]) -> PyResult<[usize; 3]> {
    let shape = match shape {
        None => {
            // Default: 1000 total cells
            return Ok(shape_from_int(1000, width));
        }
        Some(d) => d,
    };

    // Try single int first
    if let Ok(n) = shape.extract::<usize>() {
        if n < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shape must be >= 1",
            ));
        }
        return Ok(shape_from_int(n, width));
    }

    // Try list/tuple of 3
    if let Ok(seq) = shape.extract::<[usize; 3]>() {
        for (i, &v) in seq.iter().enumerate() {
            if v < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "shape[{i}] must be >= 1"
                )));
            }
        }
        return Ok(seq);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "shape must be an int or a list/tuple of 3 ints",
    ))
}

/// A cylindrical ``(r, φ, z)`` mesh with uniform spacing in each direction.
///
/// The mesh divides an annular cylinder into ``shape = (nr, nphi, nz)`` cells.
/// Cell volumes are **not** equal -- outer radial rings are larger
/// (``V = ½·(r_o² − r_i²)·Δφ·Δz``) -- so normalise per cell when comparing
/// flux across radii (see :meth:`element_volume`).
///
/// ``phi_bounds`` may be any sub-interval of ``[0, 2π]`` (default the full
/// circle), e.g. ``(0, math.pi)`` for a half or ``(math.pi, 2*math.pi)`` for a
/// sector. Coordinates are measured relative to ``origin`` and the cylinder
/// axis is parallel to +z.
///
/// Examples:
///     >>> import math, yamc
///     >>> mesh = yamc.RegularCylindricalMesh(
///     ...     r_bounds=(0.0, 10.0),
///     ...     z_bounds=(-5.0, 5.0),
///     ...     shape=(10, 16, 20),
///     ...     phi_bounds=(0.0, 2 * math.pi),
///     ...     origin=(0.0, 0.0, 0.0),
///     ... )
///
/// Args:
///     r_bounds: Radial extent ``(r_min, r_max)`` in cm. ``r_min`` must be >= 0.
///     z_bounds: Axial extent ``(z_min, z_max)`` in cm relative to ``origin``.
///     shape: Cell counts ``(nr, nphi, nz)``.
///     phi_bounds: Azimuthal extent ``(phi_min, phi_max)`` in radians. Defaults
///         to the full circle, ``(0.0, 2 * math.pi)``.
///     origin: Cylinder origin ``[x0, y0, z0]`` in cm. Defaults to
///         ``[0.0, 0.0, 0.0]``.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "RegularCylindricalMesh", from_py_object)]
#[derive(Clone)]
pub struct PyRegularCylindricalMesh {
    pub internal: CylindricalMesh,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegularCylindricalMesh {
    #[new]
    #[pyo3(signature = (
        r_bounds,
        z_bounds,
        shape,
        phi_bounds=(0.0, std::f64::consts::TAU),
        origin=[0.0, 0.0, 0.0],
    ), text_signature = "(r_bounds, z_bounds, shape, phi_bounds=(0.0, 6.283185307179586), origin=[0.0, 0.0, 0.0])")]
    fn new(
        r_bounds: (f64, f64),
        z_bounds: (f64, f64),
        shape: [usize; 3],
        phi_bounds: (f64, f64),
        origin: [f64; 3],
    ) -> PyResult<Self> {
        let err = pyo3::exceptions::PyValueError::new_err;
        if r_bounds.0 < 0.0 {
            return Err(err("r_bounds lower edge must be >= 0"));
        }
        if r_bounds.1 <= r_bounds.0 {
            return Err(err("r_bounds upper edge must exceed the lower edge"));
        }
        if z_bounds.1 <= z_bounds.0 {
            return Err(err("z_bounds upper edge must exceed the lower edge"));
        }
        if phi_bounds.1 <= phi_bounds.0 {
            return Err(err("phi_bounds upper edge must exceed the lower edge"));
        }
        let tau = std::f64::consts::TAU;
        if phi_bounds.0 < -1e-12 || phi_bounds.1 > tau + 1e-12 {
            return Err(err("phi_bounds must lie within [0, 2*pi]"));
        }
        if shape.iter().any(|&n| n < 1) {
            return Err(err("shape entries must be >= 1"));
        }
        Ok(PyRegularCylindricalMesh {
            internal: CylindricalMesh::try_uniform(origin, r_bounds, phi_bounds, z_bounds, shape)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }

    /// Radial extent ``(r_min, r_max)`` in cm.
    #[getter]
    fn r_bounds(&self) -> (f64, f64) {
        let g = self.internal.r_grid();
        (g[0], g[g.len() - 1])
    }

    /// Azimuthal extent ``(phi_min, phi_max)`` in radians.
    #[getter]
    fn phi_bounds(&self) -> (f64, f64) {
        let g = self.internal.phi_grid();
        (g[0], g[g.len() - 1])
    }

    /// Axial extent ``(z_min, z_max)`` in cm (relative to ``origin``).
    #[getter]
    fn z_bounds(&self) -> (f64, f64) {
        let g = self.internal.z_grid();
        (g[0], g[g.len() - 1])
    }

    /// Cell counts ``(nr, nphi, nz)``.
    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.internal.shape()
    }

    /// Cylinder origin ``[x0, y0, z0]`` in cm.
    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.internal.origin()
    }

    /// Total number of cells (``nr * nphi * nz``).
    #[getter]
    fn num_bins(&self) -> usize {
        self.internal.num_bins()
    }

    /// Volume of cell ``bin`` in cm³ (non-uniform: outer rings are larger).
    fn element_volume(&self, bin: usize) -> f64 {
        self.internal.get_voxel_volume(bin)
    }

    fn __repr__(&self) -> String {
        let (rmin, rmax) = self.r_bounds();
        let (pmin, pmax) = self.phi_bounds();
        let (zmin, zmax) = self.z_bounds();
        format!(
            "RegularCylindricalMesh(r_bounds=({rmin}, {rmax}), phi_bounds=({pmin}, {pmax}), \
             z_bounds=({zmin}, {zmax}), shape={:?}, origin={:?})",
            self.internal.shape(),
            self.internal.origin()
        )
    }
}
