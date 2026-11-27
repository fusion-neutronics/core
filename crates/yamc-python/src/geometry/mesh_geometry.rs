use crate::geometry::PyBoundingBox;
use crate::material::PyMaterial;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;
use yamc::geometry::mesh::MeshGeometry;

/// Mesh-based geometry loaded from a mesh file.
///
/// The mesh format is the YAMC Arrow IPC mesh (``.arrow``), as produced by
/// ``cad_to_yamc``. Other meshers are supported by writing that format.
///
/// Each volume in the mesh is named by a ``mat:<name>`` physical group. The
/// *materials* dictionary maps the bare ``<name>`` (as listed by
/// :attr:`volume_names`, e.g. ``{"first_wall": steel}``) to a
/// :class:`Material`. A key that matches no volume raises ``ValueError``
/// with a close match when one exists; volumes with no key are void.
///
/// Volumes given the same :class:`Material` object share it, as CSG cells
/// sharing a material do: its cross sections are held once, so memory
/// follows the number of materials rather than the number of volumes. In a
/// transmutation it is one material, depleted with the flux averaged over
/// all the volumes of the mesh it fills (its volume is set to their
/// total). Give volumes that should deplete separately a :class:`Material`
/// each.
///
/// Outer boundary surfaces should be tagged with a physical group named
/// ``boundary:vacuum`` (reflecting boundaries are not yet supported for
/// mesh geometry).
///
/// The implicit complement (unmeshed space between volumes) is void by
/// default -- particles free-stream through it with no collisions.  To
/// model air or coolant interactions in the gaps, pass a :class:`Material`
/// as *implicit_complement_material*.
///
/// A vacuum boundary box can be added by setting *graveyard_offset*.  This
/// adds an axis-aligned bounding box expanded by the given offset (cm)
/// around the geometry, with vacuum boundary conditions on all 6 faces.
/// Particles reaching this boundary are killed.
///
/// When this geometry is the whole model and the mesh has no vacuum
/// boundary (no *graveyard_offset* and no ``boundary:vacuum`` group, as is
/// usual for a mesh made from CAD), the :class:`Model` closes it before
/// transport:
///
/// - with a void implicit complement it adds a vacuum box 10 cm outside the
///   mesh's bounding box and prints its extent.  In a void the margin
///   changes no result; set *graveyard_offset* to choose it.
/// - with an *implicit_complement_material* it raises ``ValueError``: the
///   box position sets how much of that material surrounds the model, so
///   *graveyard_offset* must be given explicitly.
///
/// Used as the fill of a CSG cell, the mesh gets no box: the CSG geometry
/// bounds it.  Without any of these, a particle with no surface ahead of it
/// is a lost particle.
///
/// A mesh made without imprinting (``from_cadquery(..., imprint=False)``)
/// gives each solid its own surface, so touching solids meet along two
/// independently faceted surfaces that may overlap or leave thin gaps. Such a
/// mesh is tracked discontinuously: a particle leaving a solid is placed in
/// whichever touching solid contains it. Solids closer than the contact
/// tolerance count as touching, so an overlap or a gap thinner than it is
/// crossed straight into the next solid (an overlap stays with the solid the
/// particle was already in); a wider gap, or one the file records as being in
/// the CAD, is the implicit complement. The file records all this;
/// *contact_tolerance* forces it on for a mesh from another writer, or
/// overrides the tolerance the file records.
///
/// Loading warns when a mesh tracked conformally has solids that touch
/// without sharing a surface where they meet: a particle leaving one of them
/// is not placed in the solid it enters.
///
/// Made by :meth:`from_cadquery` or :meth:`from_tets`, or loaded with
/// :meth:`from_arrow`, whose arguments these are.
///
/// Args:
///     source: Path to an Arrow IPC mesh file (``.arrow``), or the file's
///         bytes, as :meth:`to_arrow` returns them with no path.
///     materials: Mapping of volume name (no ``mat:`` prefix) to :class:`Material`.
///     implicit_complement_material: Optional material for the unmeshed
///         space between volumes.  Default ``None`` means void (free-streaming).
///     graveyard_offset: Optional offset (cm) of a vacuum boundary box
///         around the mesh.  ``None`` lets the :class:`Model` add one with a
///         10 cm offset around a void implicit complement, as above.
///     contact_tolerance: Optional distance (cm) within which two solids'
///         surfaces count as touching. Setting it tracks the mesh
///         discontinuously. It must be at least the deepest overlap between
///         touching solids, and gaps narrower than it are filled by the solid
///         on the far side, except between solids the file records as apart
///         in the CAD (``from_cadquery`` checks this when meshing without
///         imprinting).
///
/// Raises:
///     ValueError: If the file is not an Arrow IPC mesh, or if it contains a
///         negatively oriented tetrahedron. Transport reads outward tet face
///         normals off a fixed vertex ordering, so an inverted tet is rejected
///         rather than silently re-wound; regenerate the mesh with
///         a writer that emits positively oriented tets.
///
/// Examples:
///     >>> fuel = yamc.Material(composition={"Li6": 1.0}, density=0.534, name="fuel")
///     >>> fuel.read_nuclear_data({"Li6": "tests/Li6.arrow"})
///     >>>
///     >>> mesh_geom = yamc.MeshGeometry.from_arrow("model.arrow", {"fuel": fuel},
///     ...                               graveyard_offset=5.0)
///     >>> model = yamc.Model(geometry=mesh_geom, source=source)
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "MeshGeometry",
    unsendable,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMeshGeometry {
    pub inner: MeshGeometry,
    /// Stored Arrow mesh data for rebuild when graveyard_offset changes.
    /// Shared, so cloning the wrapper does not copy it.
    arrow_data: Arc<yamt::ArrowMeshData>,
    /// Material map stored for rebuild.
    mat_map: HashMap<String, Arc<yamc_materials::material::Material>>,
    /// Implicit complement material stored for rebuild.
    ic_mat: Option<Arc<yamc_materials::material::Material>>,
    /// The Material objects given (the dict's values, then the implicit
    /// complement), so a Geometry this fills can refuse two different
    /// objects with one id across its cells and fills.
    pub(crate) py_materials: Arc<Vec<Py<PyMaterial>>>,
    /// Current graveyard offset.
    graveyard_offset: Option<f64>,
    /// Contact tolerance passed by the caller, or chosen on load for solids
    /// touching without a shared surface; re-applied on rebuild.
    contact_tolerance: Option<f64>,
    /// The arguments it was loaded with, for a geometry made from this one
    /// (``add_tets``).
    load_args: Arc<LoadArgs>,
    /// What the mesher reported making this mesh (``mesh_report``).
    mesh_report: Option<Arc<Py<PyAny>>>,
    /// The density factor of each tet (``tet_density_factors``).
    pub(crate) tet_density_factors: Option<Arc<Vec<f64>>>,
}

/// The arguments `load` was given.
struct LoadArgs {
    materials: Py<PyDict>,
    implicit_complement_material: Option<Py<PyMaterial>>,
    contact_tolerance: Option<f64>,
}

impl PyMeshGeometry {
    /// The geometry's tets as a tally mesh, sharing their memory: what
    /// ``Tally(mesh=geometry)`` scores on.
    pub(crate) fn tally_mesh(&self) -> PyResult<crate::geometry::PyUnstructuredMesh> {
        if self.inner.mesh.topology.tetrahedra.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "this geometry has no tets: tet-mesh volumes with add_tets, or \
                 from_cadquery(..., tet_volumes=[...])",
            ));
        }
        Ok(crate::geometry::PyUnstructuredMesh::of_geometry(
            &self.inner.mesh,
        ))
    }

    /// Load a mesh from a path or bytes, as `from_arrow` does.
    pub fn load<'py>(
        filename: &Bound<'py, PyAny>,
        materials: &Bound<'py, PyDict>,
        implicit_complement_material: Option<&Bound<'py, PyMaterial>>,
        graveyard_offset: Option<f64>,
        contact_tolerance: Option<f64>,
    ) -> PyResult<Self> {
        let contact_tolerance_arg = contact_tolerance;
        if let Some(tol) = contact_tolerance
            && !(tol.is_finite() && tol >= 0.0)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "contact_tolerance must be a non-negative number of cm, got {tol}"
            )));
        }
        // Two different Material objects with one id are refused. One object
        // under several names (or also as the implicit complement) is one
        // material, so materials are told apart by Python object identity.
        let mut material_ids = crate::geometry::DistinctMaterialIds::default();
        let mut py_materials: Vec<Py<PyMaterial>> = Vec::with_capacity(materials.len() + 1);
        for (_key, value) in materials.iter() {
            let py_mat: Bound<'py, PyMaterial> = value.cast_into()?;
            material_ids.check(&py_mat)?;
            py_materials.push(py_mat.unbind());
        }
        if let Some(ic) = implicit_complement_material {
            material_ids.check(ic)?;
            py_materials.push(ic.clone().unbind());
        }

        // One `Arc` per Material object, however many names it is under, so
        // the volumes given one object share one material slot and one set of
        // cross-section tables (`MeshGeometry::new` tells materials apart by
        // `Arc`).
        let mut shared: HashMap<usize, Arc<yamc_materials::material::Material>> = HashMap::new();
        let mut share = |py_mat: &Bound<'py, PyMaterial>| -> PyResult<_> {
            let object = py_mat.as_ptr() as usize;
            if let Some(material) = shared.get(&object) {
                return Ok(Arc::clone(material));
            }
            let material = Arc::new(py_mat.try_borrow()?.internal.clone());
            shared.insert(object, Arc::clone(&material));
            Ok(material)
        };
        let mut mat_map: HashMap<String, Arc<yamc_materials::material::Material>> = HashMap::new();
        for (key, value) in materials.iter() {
            let name: String = key.extract()?;
            let py_mat: Bound<'py, PyMaterial> = value.cast_into()?;
            mat_map.insert(name, share(&py_mat)?);
        }

        // Convert optional complement material
        let ic_mat = implicit_complement_material.map(&mut share).transpose()?;

        // Load the mesh, from its bytes or a file. The Arrow IPC mesh is the
        // only supported format.
        let loaded = if let Ok(bytes) = filename.cast::<pyo3::types::PyBytes>() {
            yamt::read_arrow_mesh_from(std::io::Cursor::new(bytes.as_bytes()))
        } else {
            let path: std::path::PathBuf = filename.extract()?;
            if path.extension().and_then(|e| e.to_str()) != Some("arrow") {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported mesh file '{}': MeshGeometry reads YAMC Arrow IPC \
                     meshes (.arrow), as written by cad_to_yamc",
                    path.display()
                )));
            }
            yamt::read_arrow_mesh(&path)
        };
        // the name messages give the mesh
        let filename = match filename.cast::<pyo3::types::PyBytes>() {
            Ok(_) => "<mesh bytes>".to_string(),
            Err(_) => filename
                .extract::<std::path::PathBuf>()?
                .display()
                .to_string(),
        };
        let mut data = loaded.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load Arrow mesh: {e}"))
        })?;
        let arrow_data = Arc::new(data.clone()); // Store original for rebuild
        if let Some(offset) = graveyard_offset {
            yamt::add_vacuum_boundary(&mut data, offset);
        }
        let topo = yamt::build_topology(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid mesh '{filename}': {e}"))
        })?;
        let mut mesh = yamt::MeshGeometry::from_topology(topo);
        let mut contact_tolerance = contact_tolerance;
        if contact_tolerance.is_some() {
            mesh.set_contact_tolerance(contact_tolerance);
        }
        // Solids that touch or overlap without sharing a surface, in a mesh
        // tracked conformally. Their facetings can cross where the CAD gap
        // is thinner than the facets' sag, and conformal tracking hands a
        // particle leaving one of them into such a crossing to the implicit
        // complement, where it skips the other solid or is lost. A file that
        // says it was imprinted (from_cadquery refuses overlaps deeper than its
        // facetings before writing one) is then tracked discontinuously,
        // which places the particle in the solid it is in; shared surfaces
        // still hand over by topology. A file that does not say how it was
        // made could hold a real overlap, which tracking would then hide, so
        // it is only warned about.
        if mesh.touches_without_shared_surface(4096) {
            let message = if mesh.topology.imprinted == Some(true) {
                let tol = mesh
                    .topology
                    .contact_tolerance
                    .unwrap_or_else(|| mesh.default_contact_tolerance());
                mesh.set_contact_tolerance(Some(tol));
                contact_tolerance = Some(tol);
                format!(
                    "'{filename}': the file says its solids were imprinted, but some of them \
                     touch without sharing a surface, so the imprint did not join them. \
                     Tracked conformally, a particle leaving one of these solids is not \
                     placed in the solid it enters, so the file is tracked discontinuously, \
                     with a contact tolerance of {tol:.3e} cm: shared surfaces still hand \
                     over by topology. Pass contact_tolerance=<distance in cm> to choose it."
                )
            } else {
                format!(
                    "'{filename}': some solids touch without sharing a surface, as in a mesh \
                     made without imprinting, but the file does not say it was made that way. \
                     Tracked as an imprinted mesh, a particle leaving one of these solids is \
                     not placed in the solid it enters, so tallies come out wrong or particles \
                     are lost. Pass contact_tolerance=<distance in cm> to track it \
                     discontinuously."
                )
            };
            materials
                .py()
                .import("warnings")?
                .call_method1("warn", (message,))?;
        }

        let inner = MeshGeometry::new(mesh, &mat_map, ic_mat.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("'{filename}': {e}")))?;

        Ok(PyMeshGeometry {
            inner,
            arrow_data,
            mat_map,
            ic_mat,
            py_materials: Arc::new(py_materials),
            graveyard_offset,
            contact_tolerance,
            load_args: Arc::new(LoadArgs {
                materials: materials.clone().unbind(),
                implicit_complement_material: implicit_complement_material
                    .map(|m| m.clone().unbind()),
                contact_tolerance: contact_tolerance_arg,
            }),
            mesh_report: None,
            tet_density_factors: None,
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshGeometry {
    /// Mesh a CadQuery assembly and return it as a geometry, ready for
    /// transport. No file is written; save it with :meth:`to_arrow` if wanted.
    ///
    /// Needs the ``cad`` extra (``pip install "yamc[cad]"``).
    ///
    /// Args:
    ///     assembly (cadquery.Assembly): The CAD, with named parts.
    ///     materials (dict[str, Material]): Material of each volume, keyed by
    ///         part name (or by the tag *material_tags* gives it). A part left
    ///         out is void.
    ///     material_tags (list[str] | str): How each solid is tagged:
    ///         ``"assembly_names"`` (default) uses each part's name,
    ///         ``"assembly_materials"`` its ``cq.Material`` name, or a list
    ///         gives one tag per solid.
    ///     tet_volumes (list[str] | None): Volumes to fill with tetrahedra,
    ///         by tag; the rest are surface meshes only.
    ///     target_edge_length (float | None): Target tet edge length (cm),
    ///         needed with *tet_volumes*.
    ///     implicit_complement_material (Material | None): As for
    ///         :meth:`from_arrow`.
    ///     graveyard_offset (float | None): As for :meth:`from_arrow`.
    ///     boundary_tags (dict[str, list[int] | str] | None): Boundary
    ///         conditions by name, such as ``"vacuum"``, each to the 1-based
    ///         surface ids it applies to, or to ``"exterior"`` for the model's
    ///         outer shell (:meth:`exterior_surface_ids`). With a vacuum
    ///         boundary on the exterior no graveyard is needed.
    ///     tolerance (float | dict): Chordal deflection of the surface mesh
    ///         (cm), for every solid or a dict by material tag or 1-based
    ///         solid id; smaller makes more triangles on curved surfaces.
    ///         A surface two solids share takes the finer of their values,
    ///         and curved surfaces are moved off the CAD so each solid's mesh
    ///         encloses its CAD volume. Default 0.5.
    ///     angular_tolerance (float | dict): Angular deflection (radians), as
    ///         *tolerance*. Default 0.2.
    ///     fidelity_check (bool): Refuse (``MeshFidelityError``) a mesh whose
    ///         face areas or volumes are far from the CAD's, or whose surfaces
    ///         are open or cross themselves. Default True.
    ///     imprint (bool): Imprint the assembly first, so touching solids
    ///         share a surface (a conformal mesh). ``False`` meshes each solid
    ///         on its own, faster on a large assembly, and the mesh is tracked
    ///         discontinuously. Default True.
    ///     contact_tolerance (float | None): With ``imprint=False``, the
    ///         distance (cm) within which solids count as touching. Defaults
    ///         to twice the largest deflection.
    ///     overlaps (str): Solids that overlap deeper than their facetings
    ///         explain: ``"error"`` (default) raises
    ///         ``OverlappingSolidsError``, ``"smaller_wins"`` (with
    ///         ``imprint=False``) gives each overlap to the smaller solid,
    ///         ``"ignore"`` skips the check.
    ///     imprint_threads (int | "all"): Threads for imprinting. Default all.
    ///     mesh_threads (int | "all"): Threads for meshing surfaces and tets.
    ///         Default all; the mesh is the same on any number.
    ///
    /// Examples:
    ///     >>> geometry = yamc.MeshGeometry.from_cadquery(
    ///     ...     assembly,
    ///     ...     materials={"breeder": breeder, "structure": steel},
    ///     ...     tet_volumes=["breeder", "structure"],
    ///     ...     target_edge_length=2.5,
    ///     ... )
    #[classmethod]
    #[pyo3(signature = (assembly, materials, *, material_tags=None, tet_volumes=None, target_edge_length=None, implicit_complement_material=None, graveyard_offset=None, boundary_tags=None, **mesh_options))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_cadquery<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        #[gen_stub(override_type(type_repr = "typing.Any"))] assembly: &Bound<'py, PyAny>,
        materials: &Bound<'py, PyDict>,
        #[gen_stub(override_type(type_repr = "list[str] | str | None"))] material_tags: Option<
            &Bound<'py, PyAny>,
        >,
        tet_volumes: Option<Vec<String>>,
        target_edge_length: Option<f64>,
        implicit_complement_material: Option<&Bound<'py, PyMaterial>>,
        graveyard_offset: Option<f64>,
        #[gen_stub(override_type(type_repr = "dict[str, list[int] | str] | None"))]
        boundary_tags: Option<&Bound<'py, PyDict>>,
        mesh_options: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<Self>> {
        let py = cls.py();
        let options = PyDict::new(py);
        if let Some(tags) = material_tags {
            options.set_item("material_tags", tags)?;
        }
        options.set_item("tet_volumes", tet_volumes)?;
        options.set_item("target_edge_length", target_edge_length)?;
        options.set_item("implicit_complement_material", implicit_complement_material)?;
        options.set_item("graveyard_offset", graveyard_offset)?;
        options.set_item("boundary_tags", boundary_tags)?;
        if let Some(extra) = mesh_options {
            options.update(extra.as_mapping())?;
        }
        py.import("yamc.cad.mesher")?
            .getattr("_mesh_geometry_from_cadquery")?
            .call((assembly, materials), Some(&options))?
            .cast_into::<Self>()
            .map(Bound::unbind)
            .map_err(Into::into)
    }

    /// A geometry from another code's tet mesh, such as FEniCSx's or
    /// gmsh's: every volume is tet-filled.
    ///
    /// The volumes are the distinct names in *tet_materials*, in order of
    /// first appearance. Their boundary surfaces are built from the tet faces
    /// where the name changes, and from the mesh's outer faces, so they are
    /// conformal with the tets by construction: a field on the tets (a
    /// temperature, a density) needs no mapping between the two codes.
    /// Negatively wound tets are re-wound.
    ///
    /// Args:
    ///     vertices (Sequence[Sequence[float]]): ``[x, y, z]`` per vertex (cm).
    ///     tets (Sequence[Sequence[int]]): Four vertex indices per tet. The
    ///         order is kept, so ``tetrahedra[i]`` is tet ``i`` given.
    ///     tet_materials (Sequence[str]): The volume name of each tet.
    ///     materials (dict[str, Material]): Material of each volume, by name.
    ///         A name left out is void.
    ///     implicit_complement_material: As for :meth:`from_arrow`.
    ///     graveyard_offset: As for :meth:`from_arrow`.
    ///
    /// Raises:
    ///     ValueError: For a tet with no volume, a vertex index out of
    ///         range, or a face shared by more than two tets.
    ///
    /// Examples:
    ///     >>> geometry = yamc.MeshGeometry.from_tets(
    ///     ...     vertices, tets, ["breeder"] * n_breeder + ["steel"] * n_steel,
    ///     ...     {"breeder": breeder, "steel": steel},
    ///     ... )
    #[classmethod]
    #[pyo3(signature = (vertices, tets, tet_materials, materials, implicit_complement_material=None, graveyard_offset=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_tets<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        vertices: Vec<[f64; 3]>,
        tets: Vec<[u32; 4]>,
        tet_materials: Vec<String>,
        materials: &Bound<'py, PyDict>,
        implicit_complement_material: Option<&Bound<'py, PyMaterial>>,
        graveyard_offset: Option<f64>,
    ) -> PyResult<Self> {
        let py = cls.py();
        let bytes = tet_mesh_bytes(py, vertices, tets, &tet_materials)?;
        Self::load(
            pyo3::types::PyBytes::new(py, &bytes).as_any(),
            materials,
            implicit_complement_material,
            graveyard_offset,
            None,
        )
    }

    /// Load a mesh that ``to_arrow`` saved.
    ///
    /// Args:
    ///     source (str | os.PathLike | bytes): A file path, or the bytes
    ///         ``to_arrow()`` returned.
    ///     materials: Mapping of volume name to :class:`Material`.
    ///     implicit_complement_material: See the class docstring.
    ///     graveyard_offset: See the class docstring.
    ///     contact_tolerance: See the class docstring.
    #[classmethod]
    #[pyo3(signature = (source, materials, implicit_complement_material=None, graveyard_offset=None, contact_tolerance=None))]
    pub fn from_arrow<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        #[gen_stub(override_type(type_repr = "str | os.PathLike[str] | bytes", imports = ("os")))]
        source: &Bound<'py, PyAny>,
        materials: &Bound<'py, PyDict>,
        implicit_complement_material: Option<&Bound<'py, PyMaterial>>,
        graveyard_offset: Option<f64>,
        contact_tolerance: Option<f64>,
    ) -> PyResult<Self> {
        Self::load(
            source,
            materials,
            implicit_complement_material,
            graveyard_offset,
            contact_tolerance,
        )
    }

    /// Save the mesh to an Arrow IPC file at ``path``, or return the file's
    /// bytes when ``path`` is None. Materials are not saved: give them again
    /// to ``from_arrow``. The graveyard is not saved either, as it is added
    /// when the mesh is loaded.
    ///
    /// Args:
    ///     path (str | os.PathLike | None): Where to write.
    ///
    /// Returns:
    ///     bytes | None: The file's bytes when ``path`` is None.
    #[pyo3(signature = (path=None))]
    pub fn to_arrow(
        &self,
        py: Python<'_>,
        path: Option<std::path::PathBuf>,
    ) -> PyResult<Option<Py<pyo3::types::PyBytes>>> {
        let mut bytes = Vec::new();
        yamt::write_arrow_mesh_to(yamt::MeshData::from(&*self.arrow_data), &mut bytes).map_err(
            |e| pyo3::exceptions::PyIOError::new_err(format!("Failed to write Arrow mesh: {e}")),
        )?;
        yani_python::file_io::write_or_return(py, path, &bytes)
    }

    /// True when the mesh is tracked discontinuously: a particle leaving a
    /// solid through a surface it does not share is placed in whichever
    /// touching solid contains it, rather than read off the surface. So is a
    /// mesh whose solids were meshed one by one without imprinting, and an
    /// imprinted one in which some solids touch without sharing a surface;
    /// shared surfaces still hand over by topology.
    #[getter]
    pub fn discontinuous(&self) -> bool {
        self.inner.mesh.is_discontinuous()
    }

    /// Distance (cm) within which two solids' surfaces count as touching, for
    /// a discontinuously tracked mesh; ``None`` for one tracked conformally.
    #[getter]
    pub fn contact_tolerance(&self) -> Option<f64> {
        self.inner.mesh.contact_tolerance()
    }

    /// Number of volumes (cells) in the mesh (excluding implicit complement).
    #[getter]
    pub fn num_volumes(&self) -> u32 {
        self.inner.mesh.topology.num_volumes
    }

    /// Material names for each volume, in volume-ID order.
    ///
    /// These are the names from ``mat:<name>`` physical groups.
    /// Pass any of these strings as the second element of a tally's
    /// ``mesh=(geometry, name)`` argument to bin on that volume by name
    /// (an integer volume ID is also accepted).
    #[getter]
    pub fn volume_names(&self) -> Vec<Option<String>> {
        (0..self.inner.mesh.topology.num_volumes)
            .map(|v| self.inner.mesh.material_name(v).map(|s| s.to_owned()))
            .collect()
    }

    /// All vertex coordinates as a list of ``[x, y, z]`` triples.
    #[getter]
    pub fn vertices(&self) -> Vec<[f64; 3]> {
        self.inner.mesh.topology.vertices.clone()
    }

    /// All triangle connectivity as a list of ``[v0, v1, v2]`` triples
    /// (vertex indices).
    #[getter]
    pub fn triangles(&self) -> Vec<[u32; 3]> {
        self.inner.mesh.topology.triangles.clone()
    }

    /// All tet connectivity as a list of ``[v0, v1, v2, v3]`` vertex indices,
    /// in the order ``Model.set_tet_density_factors`` takes its factors.
    #[getter]
    pub fn tetrahedra(&self) -> Vec<[u32; 4]> {
        self.inner.mesh.topology.tetrahedra.clone()
    }

    /// The volume ID of each tet, parallel to ``tetrahedra`` (``None`` for a
    /// tet in no volume). Index ``volume_names`` with it to find each tet's
    /// material name, e.g. to give density factors to one material's tets
    /// only.
    #[getter]
    pub fn tet_volume_ids(&self) -> Vec<Option<u32>> {
        self.inner.mesh.topology.tet_volume_ids.clone()
    }

    /// The volume name of each tet, parallel to ``tetrahedra``: the material
    /// tag given to ``from_cadquery``, or ``None`` for a tet in no named volume.
    /// ``numpy.array(geometry.tet_volume_names) == "breeder"`` picks out one
    /// material's tets.
    #[getter]
    pub fn tet_volume_names(&self) -> Vec<Option<String>> {
        let names = self.volume_names();
        self.inner
            .mesh
            .topology
            .tet_volume_ids
            .iter()
            .map(|v| v.and_then(|v| names.get(v as usize).cloned().flatten()))
            .collect()
    }

    /// The volume of each tet in cm³, parallel to ``tetrahedra``. A tet-mesh
    /// tally scores the integral over each tet, so dividing by these gives the
    /// per-cm³ density.
    #[getter]
    pub fn tet_volumes(&self) -> Vec<f64> {
        let topology = &self.inner.mesh.topology;
        let v = &topology.vertices;
        topology
            .tetrahedra
            .iter()
            .map(|t| {
                let [a, b, c, d] = t.map(|i| v[i as usize]);
                let e = |p: [f64; 3]| [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let (b, c, d) = (e(b), e(c), e(d));
                (b[0] * (c[1] * d[2] - c[2] * d[1]) - b[1] * (c[0] * d[2] - c[2] * d[0])
                    + b[2] * (c[0] * d[1] - c[1] * d[0]))
                    .abs()
                    / 6.0
            })
            .collect()
    }

    /// The density of each tet relative to its material's, parallel to
    /// ``tetrahedra``, or ``None`` (every factor 1).
    ///
    /// A field from a thermal or CFD solve on the same tets, for instance.
    /// Inside a tet every macroscopic cross section is ``f * Σ_material``, so
    /// it needs no extra nuclear data, and heating and reaction-rate scores
    /// follow the scaled density. A :class:`Model` built from this geometry
    /// starts from these factors, the way it starts from copies of its
    /// materials; change them between runs with
    /// ``Model.set_tet_density_factors``, as ``Model.set_material_density``
    /// does for a material. Needs ``tracking_mode="hybrid"`` (or
    /// ``"woodcock"``). Not saved by :meth:`to_arrow`.
    ///
    /// Set it to one non-negative factor per tet (a list, a numpy array or
    /// any iterable of numbers), or to ``None`` to clear it.
    #[getter]
    pub fn tet_density_factors(&self) -> Option<Vec<f64>> {
        self.tet_density_factors.as_ref().map(|f| f.to_vec())
    }

    #[setter]
    pub fn set_tet_density_factors(
        &mut self,
        #[gen_stub(override_type(type_repr = "typing.Iterable[builtins.float] | None"))]
        factors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let Some(factors) = factors.filter(|f| !f.is_none()) else {
            self.tet_density_factors = None;
            return Ok(());
        };
        let factors: Vec<f64> = factors
            .try_iter()?
            .map(|x| x?.extract::<f64>())
            .collect::<PyResult<_>>()?;
        let n = self.inner.mesh.topology.tetrahedra.len();
        if factors.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "one density factor per tet: got {} for {n} tets",
                factors.len()
            )));
        }
        if let Some(f) = factors.iter().find(|f| !(f.is_finite() && **f >= 0.0)) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "density factors must be finite and non-negative, got {f}"
            )));
        }
        self.tet_density_factors = Some(Arc::new(factors));
        Ok(())
    }

    /// What the mesher reported when it made this mesh, as a dict, or
    /// ``None`` for a mesh loaded from a file. ``"conform"`` holds what
    /// making the tets conformal repaired (``edges_split``, ``cones``,
    /// ``surface_triangles_refined``, ``conflicts``, ``skipped_solids``,
    /// ``reverted``), ``"overlaps"`` the solid pairs whose overlap was
    /// resolved, and ``"face_volumes"`` what keeping the faceted volumes
    /// equal to the CAD's did.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any] | None"))]
    pub fn mesh_report(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.mesh_report.as_ref().map(|r| r.clone_ref(py))
    }

    /// Internal: the same geometry, with ``report`` as its ``mesh_report``.
    #[gen_stub(skip)]
    fn _with_report(&self, report: Py<PyAny>) -> Self {
        Self {
            mesh_report: Some(Arc::new(report)),
            ..self.clone()
        }
    }

    /// Internal: the vertices and triangles of the mesh as loaded, without
    /// the graveyard, and the number of tets.
    #[gen_stub(skip)]
    #[allow(clippy::type_complexity)]
    fn _file_mesh(&self) -> (Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<[u32; 4]>, Vec<u32>) {
        let d = &self.arrow_data;
        (
            d.vertices.clone(),
            d.triangles.clone(),
            d.tetrahedra.clone(),
            d.tet_volume_ids.clone(),
        )
    }

    /// Internal: a new geometry with these vertices, triangles and tets, and
    /// everything else of this one. Each triangle takes the surface id and
    /// physical group of ``triangle_parent``, a triangle of this mesh; each
    /// tet is in the 0-based volume ``tet_volumes`` gives.
    #[gen_stub(skip)]
    fn _with_tets<'py>(
        &self,
        py: Python<'py>,
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        triangle_parent: Vec<usize>,
        tetrahedra: Vec<[u32; 4]>,
        tet_volumes: Vec<u32>,
    ) -> PyResult<Self> {
        let old = &self.arrow_data;
        if triangle_parent.len() != triangles.len() || tet_volumes.len() != tetrahedra.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "one parent per triangle and one volume per tet",
            ));
        }
        if let Some(&p) = triangle_parent.iter().find(|&&p| p >= old.triangles.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "triangle parent {p} is not a triangle of this mesh"
            )));
        }
        let mut data = yamt::MeshData::from(&**old);
        data.vertices = vertices.as_flattened().to_vec();
        data.triangles = triangles.as_flattened().to_vec();
        data.triangle_surface_ids = triangle_parent
            .iter()
            .map(|&p| old.triangle_surface_ids[p])
            .collect();
        data.triangle_physical_groups = triangle_parent
            .iter()
            .map(|&p| old.triangle_physical_groups[p])
            .collect();
        data.tetrahedra = tetrahedra.as_flattened().to_vec();
        // the file numbers volumes from 1, and a CAD volume's group is its id
        data.tet_volume_ids = tet_volumes.iter().map(|v| v + 1).collect();
        data.tet_physical_groups = tet_volumes.iter().map(|&v| v as i32 + 1).collect();
        data.tet_adjacency = yamt::mesh::cad_build::tet_adjacency_flat(&data.tetrahedra);
        let mut bytes = Vec::new();
        yamt::write_arrow_mesh_to(data, &mut bytes).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write Arrow mesh: {e}"))
        })?;
        let args = &self.load_args;
        let ic = args
            .implicit_complement_material
            .as_ref()
            .map(|m| m.bind(py).clone());
        let mut new = Self::load(
            pyo3::types::PyBytes::new(py, &bytes).as_any(),
            args.materials.bind(py),
            ic.as_ref(),
            self.graveyard_offset,
            args.contact_tolerance,
        )?;
        new.mesh_report = self.mesh_report.clone();
        Ok(new)
    }

    /// Surface ids (1-based, as the mesh file numbers them) on the model's
    /// outer shell: surfaces with a volume on one side and the implicit
    /// complement on the other. The surfaces between two volumes, and the
    /// graveyard box added on load, are left out. These are the surfaces a
    /// standalone mesh wants a vacuum boundary on, as
    /// ``boundary_tags={"vacuum": geometry.exterior_surface_ids()}`` to
    /// ``from_cadquery`` does with ``"exterior"``.
    ///
    /// Raises:
    ///     ValueError: For a mesh tracked discontinuously (made without
    ///         imprinting). Every surface of such a mesh belongs to one
    ///         volume, including where volumes touch, so tagging them vacuum
    ///         would kill particles inside the model.
    pub fn exterior_surface_ids(&self) -> PyResult<Vec<u32>> {
        if self.discontinuous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "exterior_surface_ids() is not defined for a mesh tracked \
                 discontinuously (made without imprinting): every surface there belongs \
                 to a single volume, including the surfaces where volumes touch, so \
                 tagging them vacuum would kill particles inside the model. Bound the \
                 model with graveyard_offset instead.",
            ));
        }
        // surfaces from the file, not the graveyard box added on load
        let n_file = self
            .arrow_data
            .triangle_surface_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let topology = &self.inner.mesh.topology;
        // the implicit complement is a volume of its own once loaded, or
        // absent on a side the file left open
        let outside = |v: Option<u32>| v.is_none_or(|v| v == topology.implicit_complement);
        Ok(topology
            .surface_volumes
            .iter()
            .enumerate()
            .filter(|&(i, &(a, b))| (i as u32) < n_file && outside(a) != outside(b))
            .map(|(i, _)| i as u32 + 1)
            .collect())
    }

    /// Number of triangles in the entire mesh.
    #[getter]
    pub fn num_triangles(&self) -> usize {
        self.inner.mesh.topology.triangles.len()
    }

    /// Analytical volume of each mesh volume (cm³), computed via the
    /// divergence theorem over the bounding surface triangles.
    #[getter]
    pub fn volume_measures(&self) -> Vec<f64> {
        self.inner.mesh.topology.volume_measures.clone()
    }

    /// Compute the volume of every mesh volume.
    ///
    /// Mesh geometries already know their volumes analytically (via the
    /// divergence theorem applied to their bounding triangles when the
    /// mesh is loaded), so this method is exact: ``std_dev`` is always 0
    /// and no stochastic sampling is performed. The signature matches
    /// :meth:`Geometry.calculate_volume` so the same call works on both
    /// CSG and mesh geometries.
    ///
    /// Args:
    ///     samples: Ignored (kept for signature compatibility).
    ///     bounding_box: Ignored (kept for signature compatibility).
    ///     seed: Ignored (kept for signature compatibility).
    ///
    /// Returns:
    ///     dict[int, VolumeResult]: Mapping volume_id -> VolumeResult.
    #[pyo3(signature = (samples=0, bounding_box=None, seed=None))]
    pub fn calculate_volume(
        &self,
        samples: u64,
        bounding_box: Option<&PyBoundingBox>,
        seed: Option<u64>,
    ) -> HashMap<u32, crate::geometry::PyVolumeResult> {
        let _ = (samples, bounding_box, seed);
        let measures = &self.inner.mesh.topology.volume_measures;
        let mut out = HashMap::with_capacity(measures.len());
        for (i, &v) in measures.iter().enumerate() {
            out.insert(
                i as u32,
                crate::geometry::PyVolumeResult {
                    volume: v,
                    std_dev: 0.0,
                    num_hits: 0,
                },
            );
        }
        out
    }

    /// Triangle indices belonging to a specific volume.
    ///
    /// Returns the global triangle indices for every surface of the given
    /// volume: those bounding it, and any internal face (a surface with the
    /// volume on both sides, which bounds nothing).  These can be used to
    /// index into :attr:`triangles`.
    ///
    /// Args:
    ///     volume_id (int): Volume index (0-based).
    ///
    /// Returns:
    ///     list[int]: Triangle indices for that volume.
    fn volume_triangle_indices(&self, volume_id: u32) -> PyResult<Vec<u32>> {
        let topo = &self.inner.mesh.topology;
        if volume_id >= topo.num_volumes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "volume_id {volume_id} out of range (num_volumes={})",
                topo.num_volumes
            )));
        }
        let mut tri_ids = Vec::new();
        for &(surf_id, _sense) in &topo.volume_surfaces[volume_id as usize] {
            let range = &topo.surface_tri_ranges[surf_id as usize];
            for idx in range.clone() {
                tri_ids.push(topo.surface_tri_indices[idx as usize]);
            }
        }
        tri_ids.sort_unstable();
        tri_ids.dedup();
        Ok(tri_ids)
    }

    /// Bounding box of the entire mesh geometry.
    ///
    /// Returns:
    ///     BoundingBox: Axis-aligned bounding box enclosing all mesh volumes.
    pub fn bounding_box(&self) -> PyBoundingBox {
        let bbox = self.inner.bounding_box();
        PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
    }

    /// Bounding box of volumes matching a specific material.
    ///
    /// Args:
    ///     material: A :class:`Material`, matched to volumes by its ``id``.
    ///
    /// Returns:
    ///     BoundingBox: Axis-aligned bounding box enclosing only the volumes
    ///         that use the given material.
    ///
    /// Raises:
    ///     ValueError: If no volumes match.
    pub fn bounding_box_for_material(&self, material: &PyMaterial) -> PyResult<PyBoundingBox> {
        let mat_id = material.material_id()?;
        let bbox = self
            .inner
            .bounding_box_for_material(mat_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No mesh volumes match material id {mat_id}"
                ))
            })?;
        Ok(PyBoundingBox::new(bbox.lower_left, bbox.upper_right))
    }

    /// The vacuum boundary (graveyard) offset in cm.
    ///
    /// Setting this property triggers a full rebuild (re-reads stored mesh
    /// data, adds the vacuum boundary box, and rebuilds the BVH).  Set to
    /// ``None`` to remove the boundary.
    #[getter]
    pub fn get_graveyard_offset(&self) -> Option<f64> {
        self.graveyard_offset
    }

    #[setter]
    pub fn set_graveyard_offset(&mut self, offset: Option<f64>) -> PyResult<()> {
        let mut data = (*self.arrow_data).clone();
        if let Some(off) = offset {
            yamt::add_vacuum_boundary(&mut data, off);
        }
        let topo = yamt::build_topology(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid mesh: {e}")))?;
        let mut mesh = yamt::MeshGeometry::from_topology(topo);
        if self.contact_tolerance.is_some() {
            mesh.set_contact_tolerance(self.contact_tolerance);
        }

        self.inner = MeshGeometry::new(mesh, &self.mat_map, self.ic_mat.clone())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.graveyard_offset = offset;
        Ok(())
    }

    /// Generate an interactive 2D geometry viewer as a self-contained HTML page.
    ///
    /// Produces an interactive viewer where you can switch slice planes (xy/xz/yz),
    /// pan with mouse drag, zoom with scroll wheel, and adjust all parameters
    /// in real time.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///     color_by: What to color by - "cell" or "material". Defaults to "cell".
    ///     outline: Add outline around regions - "material", "cell", or None. Defaults to "cell".
    ///     axis_units: Units for axis labels - "mm", "cm", "m", or "km". Defaults to "cm".
    ///     colors: Dictionary mapping integer IDs to color strings. Defaults to None.
    ///     font_size: Font scale for PNG axis labels (1=tiny, 2=normal, 3=large).
    ///         Defaults to 2.
    ///     contour_kwargs: Outline appearance options. Supported keys:
    ///         ``"colors"`` (hex string, default ``"#000000"``),
    ///         ``"linewidths"`` (int, default ``1``).
    ///
    /// Returns:
    ///     InteractivePlot: Result object that renders in Jupyter and supports
    ///         ``.to_html("file.html")`` and ``.to_png("file.png")``.
    #[gen_stub(skip)] // FIXME(stub): pyo3-stub-gen mishandles Option<&str> default; hand-add in stub patch-merge
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy", color_by="cell", outline="cell", axis_units="cm", colors=None, font_size=2, contour_kwargs=None))]
    pub fn plot(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        color_by: &str,
        outline: Option<&str>,
        axis_units: &str,
        #[allow(unused_variables)] colors: Option<pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
        font_size: usize,
        contour_kwargs: Option<pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
        py: pyo3::Python<'_>,
    ) -> PyResult<crate::ui::PyInteractivePlot> {
        yamc_plot::axis_unit_scale(axis_units).map_err(pyo3::exceptions::PyValueError::new_err)?;
        use crate::ui::{
            build_interactive_html, parse_contour_kwargs, InteractiveViewParams, PyInteractivePlot,
            SourceParams,
        };
        use yamc::geometry::conversion::mesh_geometry_to_geo_mesh;

        let contour = parse_contour_kwargs(contour_kwargs.as_ref())?;
        let source = SourceParams::default();
        let bbox = self.inner.bounding_box();
        let params = crate::id_map_helper::parse_sample_slice_params(
            origin, width, resolution, basis, &bbox, py,
        )?;

        let geo_mesh = mesh_geometry_to_geo_mesh(&self.inner);
        let geometry_json = serde_json::to_string(&geo_mesh)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut cell_names = std::collections::HashMap::new();
        let mut material_names = std::collections::HashMap::new();
        for vol in &geo_mesh.volumes {
            cell_names.insert(vol.cell_id, format!("Volume {}", vol.cell_id));
            if vol.material_id != -1
                && let Some(ref name) = vol.material_name
            {
                material_names.insert(vol.material_id, name.clone());
            }
        }

        let view_params = InteractiveViewParams {
            origin: params.origin,
            width: params.width,
            total_pixels: params.pixels.0 * params.pixels.1,
            basis: params.basis.clone(),
            color_by: color_by.to_string(),
            outline: outline.map(|s| s.to_string()),
            axis_units: axis_units.to_string(),
            bbox_widths: self.inner.bounding_box().width(),
            outline_color: contour.colors.clone(),
            outline_thickness: contour.linewidths,
            n_samples: None,
            plane_tolerance: 1.0,
            source_color: source.color.clone(),
            source_size: source.size,
        };

        // Pre-sample the initial view with native Rust (BVH)
        let presampled = {
            use crate::id_map_helper::{cell_to_ids, generate_presampled_grid};
            let mesh = &self.inner;
            generate_presampled_grid(&params, |point| {
                mesh.find_cell_index(point)
                    .map(|idx| cell_to_ids(&mesh.cells[idx], &mesh.materials))
                    .unwrap_or((-1, -1))
            })
        };

        let html = build_interactive_html(
            &geometry_json,
            "mesh",
            "",
            &view_params,
            &cell_names,
            &material_names,
            None,
            None,
            Some(&presampled),
            // A pure mesh geometry has no CSG fills; the WASM mesh sampler
            // resolves its volumes directly.
            false,
            // Mesh geometries don't carry CSG surfaces to hover over.
            None,
        );

        PyInteractivePlot::new(
            html,
            &presampled,
            color_by,
            outline,
            None,
            params.origin,
            params.width,
            basis,
            axis_units,
            font_size,
            &contour.colors,
            contour.linewidths,
            None,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshGeometry(volumes={}, triangles={})",
            self.inner.mesh.topology.num_volumes,
            self.inner.mesh.topology.triangles.len(),
        )
    }
}

/// The Arrow mesh file of a bare tet mesh whose tet `i` is in the volume
/// named `tet_materials[i]` (volumes in order of first appearance), its
/// surfaces built by `yamt::mesh::from_tets`.
pub(crate) fn tet_mesh_bytes(
    py: Python<'_>,
    vertices: Vec<[f64; 3]>,
    tets: Vec<[u32; 4]>,
    tet_materials: &[String],
) -> PyResult<Vec<u8>> {
    let mut names: Vec<String> = Vec::new();
    let mut index: HashMap<&str, u32> = HashMap::new();
    let mut tet_volume = Vec::with_capacity(tet_materials.len());
    for name in tet_materials {
        let v = *index.entry(name.as_str()).or_insert_with(|| {
            names.push(name.clone());
            names.len() as u32 - 1
        });
        tet_volume.push(v);
    }
    let n_volumes = names.len() as u32;
    let tables = py
        .detach(|| yamt::mesh::from_tets::from_tets(&vertices, &tets, &tet_volume, n_volumes))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let triangle_physical_groups = tables
        .triangle_surface_ids
        .iter()
        .map(|&s| tables.surface_volumes[s as usize - 1][0].map_or(-1, |v| v as i32 + 1))
        .collect();
    let tet_flat: Vec<u32> = tables.tetrahedra.as_flattened().to_vec();
    let measures: Vec<String> = tables
        .volume_measures
        .iter()
        .map(|m| format!("{m:e}"))
        .collect();
    let pairs: Vec<String> = tables
        .surface_volumes
        .iter()
        .map(|p| {
            let f = |v: Option<u32>| v.map_or("null".to_string(), |v| v.to_string());
            format!("[{}, {}]", f(p[0]), f(p[1]))
        })
        .collect();
    let data = yamt::MeshData {
        vertices: vertices.as_flattened().to_vec(),
        triangles: tables.triangles.as_flattened().to_vec(),
        triangle_surface_ids: tables.triangle_surface_ids.clone(),
        triangle_physical_groups,
        tet_adjacency: yamt::mesh::cad_build::tet_adjacency_flat(&tet_flat),
        tetrahedra: tet_flat,
        tet_volume_ids: tet_volume.iter().map(|v| v + 1).collect(),
        tet_physical_groups: tet_volume.iter().map(|&v| v as i32 + 1).collect(),
        physical_groups_json: physical_groups_json(&names, &[]),
        surface_volumes_json: format!("[{}]", pairs.join(", ")),
        volume_measures_json: format!("[{}]", measures.join(", ")),
        imprinted: Some(true),
        contact_tolerance: None,
        overlaps: vec![],
        separated: vec![],
    };
    let mut bytes = Vec::new();
    yamt::write_arrow_mesh_to(data, &mut bytes).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to write Arrow mesh: {e}"))
    })?;
    Ok(bytes)
}

/// Build the `yamc.physical_groups` JSON for a CAD export.
///
/// One `dim=3` group per material, numbered `1..` in `material_tags` order (so
/// group id `i + 1` is solid `i + 1`, which the reader maps to volume `i`),
/// followed by one `dim=2` group per boundary tag carrying the surface ids it
/// applies to. Boundary group names are prefixed `boundary:` so the reader's
/// group-name parser turns them into boundary conditions.
pub(crate) fn physical_groups_json(
    material_tags: &[String],
    boundary_tags: &[(String, Vec<u32>)],
) -> String {
    let mut groups = serde_json::Map::new();
    for (i, tag) in material_tags.iter().enumerate() {
        groups.insert(
            (i + 1).to_string(),
            serde_json::json!({"name": format!("mat:{tag}"), "dim": 3}),
        );
    }
    for (id, (name, surface_ids)) in (material_tags.len() + 1..).zip(boundary_tags) {
        groups.insert(
            id.to_string(),
            serde_json::json!({
                "name": format!("boundary:{name}"),
                "dim": 2,
                "surface_ids": surface_ids,
            }),
        );
    }
    serde_json::Value::Object(groups).to_string()
}
