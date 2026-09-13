use crate::distribution::{PyNeutronSource, PyPhotonSource};
use crate::geometry::PyGeometry;
use crate::tally::PyTally;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use yamc::model::Model;
use yamc::track::HistorySelection;
use yamc_source::source::ParticleSource;
use yani_python::PyTransmutationResults;
use yani_transmute::TransmutationResults;

/// Parse the `capture_tracks=` argument into a [`HistorySelection`].
///
/// Returns `Ok(None)` when tracking is disabled (the argument is omitted or
/// `None`). Otherwise maps the typed history-selection spec onto the core enum:
/// - `int N`         -> the first `N` histories (global indices `0..N`)
/// - `range(a, b, s)`-> a non-negative ascending index range, with stride `s`
/// - `Sequence[int]` -> exactly those global history indices
/// - `'all'`         -> every history
fn parse_capture_tracks(
    py: Python<'_>,
    obj: Option<pyo3::Py<pyo3::types::PyAny>>,
) -> PyResult<Option<HistorySelection>> {
    use pyo3::exceptions::{PyTypeError, PyValueError};

    let Some(obj) = obj else {
        return Ok(None);
    };
    let obj = obj.bind(py);
    if obj.is_none() {
        return Ok(None);
    }
    // `bool` is a subclass of `int`, so reject it explicitly -- otherwise
    // `capture_tracks=True` would silently mean "the first 1 history".
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err(
            "capture_tracks no longer accepts True/False; pass 'all' to capture every \
             history, an int/range/list of indices for a subset, or None to disable",
        ));
    }
    // 'all'
    if let Ok(s) = obj.extract::<String>() {
        if s == "all" {
            return Ok(Some(HistorySelection::All));
        }
        return Err(PyValueError::new_err(format!(
            "capture_tracks string must be 'all', got {s:?}"
        )));
    }
    // range(start, stop[, step]) -- checked before the generic int/sequence
    // arms (a range extracts as both an int-less object and an iterable).
    let range_type = py.import("builtins")?.getattr("range")?;
    if obj.is_instance(&range_type)? {
        let start: i64 = obj.getattr("start")?.extract()?;
        let stop: i64 = obj.getattr("stop")?.extract()?;
        let step: i64 = obj.getattr("step")?.extract()?;
        if step < 1 {
            return Err(PyValueError::new_err(
                "capture_tracks range must ascend with a positive step (e.g. range(20, 30) \
                 or range(0, 1000, 10)); pass an explicit list for any other ordering",
            ));
        }
        if start < 0 {
            return Err(PyValueError::new_err(
                "capture_tracks range bounds must be non-negative history indices",
            ));
        }
        return Ok(Some(HistorySelection::Range {
            start: start as u64,
            stop: stop.max(0) as u64, // stop <= start yields an empty selection
            step: step as u64,
        }));
    }
    // int N -> first N histories
    if let Ok(n) = obj.extract::<i64>() {
        if n < 0 {
            return Err(PyValueError::new_err(
                "capture_tracks count must be a non-negative integer",
            ));
        }
        return Ok(Some(HistorySelection::first(n as u64)));
    }
    // Sequence[int] -> exactly those indices
    if let Ok(indices) = obj.extract::<Vec<i64>>() {
        if indices.iter().any(|&i| i < 0) {
            return Err(PyValueError::new_err(
                "capture_tracks indices must be non-negative",
            ));
        }
        return Ok(Some(HistorySelection::Set(
            indices.into_iter().map(|i| i as u64).collect(),
        )));
    }
    Err(PyTypeError::new_err(
        "capture_tracks must be None, an int, a range, a list of ints, or 'all'",
    ))
}

#[cfg(feature = "mesh")]
use crate::geometry::PyMeshGeometry;

/// Extract a ParticleSource from a single PyAny item.
fn extract_one_source(item: &Bound<'_, PyAny>) -> PyResult<ParticleSource> {
    if let Ok(ns) = item.extract::<PyNeutronSource>() {
        return Ok(ns.inner);
    }
    if let Ok(ps) = item.extract::<PyPhotonSource>() {
        return Ok(ps.inner);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "source must be a NeutronSource, PhotonSource, or iterable of them",
    ))
}

/// Extract sources from a PyAny: accepts a single source or an iterable of them.
fn extract_sources(source: &Bound<'_, PyAny>) -> PyResult<Vec<ParticleSource>> {
    // Try single source first
    if let Ok(src) = extract_one_source(source) {
        if (src.strength() - 1.0).abs() > f64::EPSILON {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "strength is only meaningful for multi-source models. \
                 Pass [source] (a list) instead of a bare source if you want to weight it, \
                 or remove the explicit strength=.",
            ));
        }
        return Ok(vec![src]);
    }
    // Try iterable of sources
    if let Ok(iter) = source.try_iter() {
        let mut sources = Vec::new();
        for item in iter {
            let item = item?;
            sources.push(extract_one_source(&item)?);
        }
        if sources.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "source list must not be empty",
            ));
        }
        return Ok(sources);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "source must be a NeutronSource, PhotonSource, or iterable of them",
    ))
}

/// A complete Monte Carlo simulation model.
///
/// The Model combines geometry, source, tallies, and simulation parameters
/// into a single object that can be executed to perform particle transport.
///
/// The Model describes *what* to simulate -- the physical system and the
/// observations. Per-run execution parameters (``total_particles``,
/// ``seed``, ``threads``, ``max_runtime``, ``compute``) are arguments of
/// ``simulate_transport`` / ``simulate_transmutation``, not of the Model.
///
/// Args:
///     geometry: The geometry defining the problem's spatial domain.
///     tallies: Optional list of tallies to score during simulation.
///     source: Source distribution(s) for particle sampling.
///     transport_secondary_photons: Transport secondary photons produced by
///         neutron interactions, i.e. coupled neutron->photon (default: False).
///         A PhotonSource transports photons regardless of this flag.
///     use_decay_photons: Enable D1S decay photon mode (default: False).
///     photon_cutoff_energy: Photon energy cutoff in eV (default: 1000.0).
///     electron_treatment: Electron energy treatment, "ttb" (thick-target
///         bremsstrahlung) or "local" (local deposition). Default: "ttb".
///     free_gas_threshold: Free-gas threshold multiplier (default: 400.0).
///     max_lost_particles: Max lost particles before abort (default: 10).
///         Mesh-geometry transport verifies every surface crossing
///         spatially (issue #254): a crossing whose flight segment passes
///         through a surface foreign to the current volume (overlapping or
///         self-intersecting mesh volumes, or a corrupted tracking state)
///         records the particle as lost, exactly like a geometry gap.
///     gpu_max_steps_per_particle: Hard cap on transport steps per particle on
///         the GPU path, which needs a bounded loop. CPU transport runs every
///         history to completion (ending it on absorption, leakage or a lost
///         particle) and ignores the value, warning if you set it and then run
///         on the CPU, since the request has no effect there. Default: 100000,
///         high enough that the GPU matches the CPU's run-to-completion
///         behaviour for strong scatterers such as a pure-H2 sphere (a 1000
///         cap cost ~10% of the integral flux there); the loop exits early once
///         a particle leaks or is absorbed, so it is free for fast-escaping
///         problems. A cap that binds is an error: if any history in a GPU
///         launch is still transporting at the cap, ``simulate_transport``
///         raises ``ValueError`` instead of returning the under-counted
///         tallies. Raise the cap or run on the CPU.
///     verbose: Progress output as a list of independent flags (reported in
///         source particles). Any of ``"progress"`` (``Progress: N/total
///         particles (P%)`` lines), ``"eta"`` (progress with elapsed + ETA),
///         ``"tally"`` (each tally's running mean ± std),
///         ``"summary"`` (end-of-run completion + per-tally statistics), and
///         ``"nuclear_data"`` (one line per nuclear-data file as it loads, plus
///         the transmutation chain file). An empty list ``[]`` is fully silent.
///         ``None`` (default) uses ``["eta", "tally", "summary"]``.
///     tracking_mode: Particle-transport algorithm. One of:
///
///         - ``"surface"`` (default): standard distance-to-nearest-boundary
///           ray tracing. Always correct; the right choice for typical
///           geometries that mix dense and sparse regions and contain voids.
///         - ``"hybrid"``: Woodcock delta tracking with an automatic
///           per-cell fallback to surface tracking in voids / large
///           low-density regions. This is the recommended Woodcock mode for
///           real models, since most fusion geometries have a vacuum vessel
///           or air: it keeps delta tracking's speed in dense regions and
///           is never pathological in voids.
///         - ``"woodcock"``: *pure* delta tracking with no fallback. Only
///           use this on geometrically dense, **void-free** models (e.g.
///           finely diced or CAD-tessellated geometry); in a model with
///           large voids it churns fictitious collisions and is much slower
///           than ``"surface"``. Prefer ``"hybrid"`` unless you know your
///           model has no voids.
///
///         All three support neutrons and photons (photon sources, coupled
///         neutron->photon production, and D1S decay photons) and give the
///         same answer within statistics. CPU only: the GPU kernels always
///         surface-track, so ``compute='gpu'`` with ``"hybrid"`` or
///         ``"woodcock"`` prints a one-line notice to stderr (at every
///         ``verbose`` setting) and proceeds with surface tracking, whose
///         flux is an unbiased estimate of the same quantity.
///     variance_reduction: List of variance-reduction technique objects
///         applied during transport; an empty list or ``None`` (default)
///         is fully analog. Currently accepts ``yamc.SurvivalBiasing``
///         (at most one entry) and ``yamc.WeightWindowBounds``; source
///         biasing joins this list in a future release. Techniques compose
///         and the list order carries no meaning.
///
/// Examples:
///     >>> import yamc
///     >>> geometry = yamc.Geometry(cells)
///     >>> model = yamc.Model(
///     ...     geometry=geometry,
///     ...     source=source,
///     ... )
///     >>> model.simulate_transport(total_particles=100_000)
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Model", unsendable, from_py_object)]
#[derive(Clone)]
pub struct PyModel {
    pub inner: Model,
    /// Whether `gpu_max_steps_per_particle` was set explicitly (constructor argument
    /// or setter) rather than left at its default. Only the GPU path honours the
    /// cap, so `simulate_transport(compute='cpu')` warns that the request has no
    /// effect -- but only when it was actually asked for, never for the default.
    gpu_max_steps_explicit: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModel {
    /// Create a new Model.
    #[new]
    #[pyo3(signature = (geometry, tallies=None, source=None, transport_secondary_photons=false, use_decay_photons=false, photon_cutoff_energy=1000.0, electron_treatment=None, free_gas_threshold=400.0, max_lost_particles=10, gpu_max_steps_per_particle=None, verbose=None, tracking_mode="surface", variance_reduction=None, gpu_fission_bank=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        #[gen_stub(override_type(type_repr = "Geometry | MeshGeometry"))] geometry: &Bound<
            '_,
            pyo3::PyAny,
        >,
        tallies: Option<Vec<PyTally>>,
        #[gen_stub(override_type(
            type_repr = "NeutronSource | PhotonSource | typing.Sequence[NeutronSource | PhotonSource] | None"
        ))]
        source: Option<&Bound<'_, pyo3::PyAny>>,
        transport_secondary_photons: bool,
        use_decay_photons: bool,
        photon_cutoff_energy: f64,
        #[gen_stub(override_type(type_repr = "typing.Literal['ttb', 'local'] | None"))]
        electron_treatment: Option<String>,
        free_gas_threshold: f64,
        max_lost_particles: usize,
        gpu_max_steps_per_particle: Option<u32>,
        verbose: Option<Vec<String>>,
        tracking_mode: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Sequence[SurvivalBiasing | WeightWindowBounds] | None"
        ))]
        variance_reduction: Option<Vec<Bound<'_, pyo3::PyAny>>>,
        gpu_fission_bank: bool,
    ) -> PyResult<Self> {
        // Validate: use_decay_photons requires transport_secondary_photons
        if use_decay_photons && !transport_secondary_photons {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "use_decay_photons=True requires transport_secondary_photons=True",
            ));
        }

        // Parse + validate the variance-reduction list (entry types,
        // multiplicity, weight parameters) so a misconfigured model fails
        // at construction rather than at simulate_transport.
        let variance_reduction = crate::variance_reduction::parse_variance_reduction(
            variance_reduction.unwrap_or_default(),
        )?;

        // Resolve the verbose flag list early so the user gets a sensible
        // error before any expensive geometry conversion. `None` uses the
        // default set; an empty list is fully silent.
        let verbose_level = match verbose {
            None => yamc::model::Verbose::default(),
            Some(flags) => yamc::model::Verbose::from_flags(flags)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        };

        // Parse tracking_mode the same way -- accepts "surface",
        // "woodcock", or "hybrid". See [`yamc::model::TrackingMode`] for
        // the accepted values and when to use each.
        let tracking_mode_parsed = tracking_mode
            .parse::<yamc::model::TrackingMode>()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let sources = match source {
            Some(src) => extract_sources(src)?,
            None => vec![ParticleSource::Neutron(yamc_source::source::Source::new())],
        };

        let tallies: Vec<_> = if let Some(py_tallies) = tallies {
            py_tallies
                .into_iter()
                .map(|py_tally| py_tally.inner.clone())
                .collect()
        } else {
            Vec::new()
        };

        // Validate that tally names and ids are unique up front, so users
        // get an error at Model construction rather than at simulate_transport.
        let mut seen_names: HashMap<&str, usize> = HashMap::new();
        let mut seen_ids: HashMap<u32, usize> = HashMap::new();
        for (i, t) in tallies.iter().enumerate() {
            if let Some(name) = t.name.as_deref() {
                if let Some(&prev) = seen_names.get(name) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "duplicate tally name {name:?} at indices {prev} and {i}"
                    )));
                }
                seen_names.insert(name, i);
            }
            if let Some(id) = t.tally_id {
                if let Some(&prev) = seen_ids.get(&id) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "duplicate tally id {id} at indices {prev} and {i}"
                    )));
                }
                seen_ids.insert(id, i);
            }
        }

        // Electron treatment is the `electron_treatment=` string ("ttb" or
        // "local"). Default is TTB.
        let electron_treatment = match electron_treatment {
            Some(s) => s
                .parse::<yamc::model::ElectronTreatment>()
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            None => yamc::model::ElectronTreatment::Ttb,
        };

        let make_model = |geometry: yamc::geometry::backend::GeometryKind| Model {
            geometry,
            sources,
            free_gas_threshold,
            transport_secondary_photons,
            // GPU fission-chain bank (#78); default-on, no effect without
            // fissile material or on the CPU path.
            gpu_fission_bank,
            photon_cutoff_energy,
            electron_treatment,
            max_lost_particles,
            // `None` means "not given": fall back to the same default the Rust
            // core uses, and record below that the user did not ask for a cap.
            gpu_max_steps_per_particle: gpu_max_steps_per_particle.unwrap_or(100_000),
            use_decay_photons,
            tracking_mode: tracking_mode_parsed,
            variance_reduction,
            tallies,
            verbose: verbose_level,
            last_elapsed_secs: None,
            last_particles_per_second: None,
            last_data_load_secs: None,
            last_transport_secs: None,
            lost_particles: Vec::new(),
            convergence_targets: Vec::new(),
        };

        // CSG geometry path
        if let Ok(csg) = geometry.extract::<PyGeometry>() {
            return Ok(PyModel {
                inner: make_model(yamc::geometry::backend::GeometryKind::Csg(
                    csg.inner.clone(),
                )),
                gpu_max_steps_explicit: gpu_max_steps_per_particle.is_some(),
            });
        }

        // Mesh geometry path
        #[cfg(feature = "mesh")]
        if let Ok(mesh) = geometry.extract::<PyMeshGeometry>() {
            return Ok(PyModel {
                inner: make_model(yamc::geometry::backend::GeometryKind::Mesh(Box::new(
                    mesh.inner.clone(),
                ))),
                gpu_max_steps_explicit: gpu_max_steps_per_particle.is_some(),
            });
        }

        #[cfg(feature = "mesh")]
        let msg = "geometry must be Geometry or MeshGeometry";
        #[cfg(not(feature = "mesh"))]
        let msg = "geometry must be Geometry";

        Err(pyo3::exceptions::PyTypeError::new_err(msg))
    }

    /// Nuclide names referenced by this model's materials.
    ///
    /// Returns:
    ///     list[str]: Sorted, de-duplicated nuclide names.
    pub fn required_nuclides(&self) -> Vec<String> {
        self.inner.required_nuclides()
    }

    /// Element symbols this model needs photon data for.
    ///
    /// Photon data is per element (``Fe``, not ``Fe56``). Empty when no photons
    /// will be in flight, i.e. neither a photon source nor secondary-photon
    /// production.
    ///
    /// Returns:
    ///     list[str]: Sorted, de-duplicated element symbols.
    pub fn required_elements(&self) -> Vec<String> {
        self.inner.required_elements()
    }

    /// True when photons will be in flight during the run, either because a
    /// source emits them or because secondary-photon production is enabled.
    ///
    /// Returns:
    ///     bool
    pub fn has_photons(&self) -> bool {
        self.inner.has_photons()
    }

    /// Unstable activation products reachable from this model's materials.
    ///
    /// For every nuclide in the model's materials, looks up its chain reactions
    /// and collects the unstable targets (finite half-life with decay photon
    /// sources). Pass the result to ``Tally(parent_nuclides=...)`` for a D1S
    /// decay-photon tally.
    ///
    /// The transmutation network is assembled from the configured per-subsection
    /// sources (``yamc.transmutation_decay_data`` and friends).
    ///
    /// Returns:
    ///     list[str]: Sorted, de-duplicated radionuclide names.
    pub fn radionuclides(&self) -> pyo3::PyResult<Vec<String>> {
        yani_python::distribution::get_radionuclides_from_chain(self.inner.required_nuclides())
    }

    /// Geometry defining the problem's spatial domain.
    ///
    /// Returns:
    ///     Geometry object (only for CSG geometry)
    #[getter]
    pub fn geometry(&self) -> pyo3::PyResult<PyGeometry> {
        match &self.inner.geometry {
            yamc::geometry::backend::GeometryKind::Csg(g) => Ok(PyGeometry { inner: g.clone() }),
            #[cfg(feature = "mesh")]
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot access CSG geometry on a mesh-based model",
            )),
        }
    }

    /// Source distribution(s) for particle sampling.
    ///
    /// Returns:
    ///     list: List of NeutronSource / PhotonSource objects
    #[getter]
    pub fn source(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .sources
            .iter()
            .map(|s| match s {
                ParticleSource::Neutron(_) => {
                    Ok(Py::new(py, PyNeutronSource { inner: s.clone() })?.into_any())
                }
                ParticleSource::Photon(_) => {
                    Ok(Py::new(py, PyPhotonSource { inner: s.clone() })?.into_any())
                }
            })
            .collect()
    }

    /// Precision-based stopping criteria. When non-empty (single-process
    /// runs), the transport loop ends at the first checkpoint where every
    /// convergence target is satisfied.
    #[getter]
    pub fn convergence_targets(&self) -> Vec<crate::tally::PyConvergenceTarget> {
        self.inner
            .convergence_targets
            .iter()
            .cloned()
            .map(|inner| crate::tally::PyConvergenceTarget { inner })
            .collect()
    }

    #[setter]
    pub fn set_convergence_targets(
        &mut self,
        convergence_targets: Vec<crate::tally::PyConvergenceTarget>,
    ) {
        self.inner.convergence_targets = convergence_targets.into_iter().map(|t| t.inner).collect();
    }

    /// Whether photon transport is enabled.
    #[getter]
    pub fn transport_secondary_photons(&self) -> bool {
        self.inner.transport_secondary_photons
    }

    /// Whether D1S decay photon mode is enabled.
    #[getter]
    pub fn use_decay_photons(&self) -> bool {
        self.inner.use_decay_photons
    }

    /// Whether the GPU banks fission progeny on the device.
    ///
    /// Default `True`, and inert without fissile material or on the CPU. Turn
    /// it off to run a fissile model with a mesh tally on the GPU, which the
    /// dispatch otherwise refuses: the combination of per-source accumulation
    /// across fission generations and direct mesh scoring is not wired up.
    /// The cost is that fission progeny are not banked on the device, so the
    /// run takes the non-fissile loop, which is why the mesh path works.
    #[getter]
    pub fn gpu_fission_bank(&self) -> bool {
        self.inner.gpu_fission_bank
    }

    #[setter]
    pub fn set_gpu_fission_bank(&mut self, value: bool) {
        self.inner.gpu_fission_bank = value;
    }

    /// Photon energy cutoff in eV.
    #[getter]
    pub fn photon_cutoff_energy(&self) -> f64 {
        self.inner.photon_cutoff_energy
    }

    /// Electron treatment mode: ``"ttb"`` (thick-target bremsstrahlung) or
    /// ``"local"`` (local energy deposition).
    #[getter]
    pub fn electron_treatment(&self) -> String {
        self.inner.electron_treatment.as_str().to_string()
    }

    /// Free-gas threshold multiplier.
    #[getter]
    pub fn free_gas_threshold(&self) -> f64 {
        self.inner.free_gas_threshold
    }

    /// Maximum lost particles before abort.
    #[getter]
    pub fn max_lost_particles(&self) -> usize {
        self.inner.max_lost_particles
    }

    /// Set the lost-particle abort threshold for the next run.
    #[setter(max_lost_particles)]
    pub fn set_max_lost_particles(&mut self, value: usize) {
        self.inner.max_lost_particles = value;
    }

    /// Hard cap on transport steps per particle on the GPU path.
    /// Ignored by the CPU path.
    #[getter]
    pub fn gpu_max_steps_per_particle(&self) -> u32 {
        self.inner.gpu_max_steps_per_particle
    }

    /// Set the per-particle transport-step cap for the next run.
    #[setter(gpu_max_steps_per_particle)]
    pub fn set_gpu_max_steps_per_particle(&mut self, value: u32) {
        self.inner.gpu_max_steps_per_particle = value;
        self.gpu_max_steps_explicit = true;
    }

    /// Progress output as a list of flags (reported in source particles).
    /// Any combination of ``"progress"``, ``"eta"``, ``"tally"``,
    /// ``"summary"``, and ``"nuclear_data"``; an empty list ``[]`` is fully
    /// silent (not even the end-of-run summary). Default:
    /// ``["eta", "tally", "summary"]``.
    ///
    /// - ``"progress"``     -- periodic ``Progress: N/total particles (P%)`` lines
    /// - ``"eta"``          -- progress lines with elapsed wall-clock and ETA
    /// - ``"tally"`` -- each tally's running mean ± std (rel. err)
    /// - ``"summary"``      -- end-of-run completion line and per-tally statistics
    /// - ``"nuclear_data"`` -- one line per nuclear-data file as it loads
    ///   (nuclide + path), plus the transmutation chain file load
    #[getter]
    pub fn verbose(&self) -> Vec<String> {
        self.inner.verbose.flags()
    }

    #[setter(verbose)]
    pub fn set_verbose(&mut self, value: Vec<String>) -> PyResult<()> {
        self.inner.verbose = yamc::model::Verbose::from_flags(value)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Particle-transport tracking algorithm: ``"surface"`` (default),
    /// ``"hybrid"`` (recommended Woodcock mode -- delta tracking with a
    /// surface fallback in voids/low-density cells), or ``"woodcock"``
    /// (pure delta tracking, void-free models only). All support neutrons
    /// and photons. See the `Model` constructor for guidance on each.
    #[getter]
    pub fn tracking_mode(&self) -> String {
        self.inner.tracking_mode.to_string()
    }

    #[setter(tracking_mode)]
    pub fn set_tracking_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.tracking_mode = value
            .parse::<yamc::model::TrackingMode>()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Variance-reduction techniques applied during transport (list of
    /// ``SurvivalBiasing``; empty = fully analog). Returns copies:
    /// configure technique objects and assign a whole list, rather than
    /// mutating entries in place.
    #[getter]
    pub fn variance_reduction(&self, py: Python<'_>) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        self.inner
            .variance_reduction
            .iter()
            .map(|vr| -> PyResult<pyo3::Py<pyo3::PyAny>> {
                match vr {
                    yamc::variance_reduction::VarianceReduction::SurvivalBiasing(sb) => {
                        Ok(crate::variance_reduction::PySurvivalBiasing { inner: *sb }
                            .into_pyobject(py)?
                            .into_any()
                            .unbind())
                    }
                    yamc::variance_reduction::VarianceReduction::WeightWindowBounds(ww) => {
                        Ok(crate::variance_reduction::PyWeightWindowBounds {
                            inner: ww.as_ref().clone(),
                        }
                        .into_pyobject(py)?
                        .into_any()
                        .unbind())
                    }
                }
            })
            .collect()
    }

    #[setter(variance_reduction)]
    pub fn set_variance_reduction(&mut self, value: Vec<Bound<'_, pyo3::PyAny>>) -> PyResult<()> {
        self.inner.variance_reduction = crate::variance_reduction::parse_variance_reduction(value)?;
        Ok(())
    }

    /// Generate weight windows with the DeGVR method (density-extrapolation).
    ///
    /// Runs two reduced-density fixed-source passes internally and returns the
    /// resulting ``WeightWindowBounds`` for a clean production run (add it to a
    /// Model's ``variance_reduction`` list). The current model is not mutated.
    ///
    /// Each generation pass runs until its per-pass stop condition trips: a
    /// particle budget (``total_particles``), a wall-time budget
    /// (``max_runtime``), or the first of the two. At least one must be set.
    /// A time budget lets a weight-window generation cost be compared
    /// apples-to-apples against an equally time-bounded analog run.
    ///
    /// Args:
    ///     generator: A ``WeightWindowGeneratorDeGVR`` configuring the mesh,
    ///         energy groups, particle, and DeGVR parameters.
    ///     total_particles: Histories per generation pass, or ``None``
    ///         (default) for no particle cap. Must be positive when given.
    ///     seed: Base RNG seed (default 1).
    ///     threads: CPU worker threads (default: all cores).
    ///     max_runtime: Optional wall-time budget PER PASS. A float (seconds)
    ///         or a ``(value, unit)`` tuple with unit ``'s'``, ``'min'``,
    ///         ``'h'``, ``'d'``, or ``'a'`` (year), e.g. ``(5, 'min')``.
    ///     At least one of ``total_particles`` / ``max_runtime`` must be set
    ///     (a call with neither raises ``ValueError``).
    ///
    /// Returns:
    ///     WeightWindowBounds | list[WeightWindowBounds]: the generated windows,
    ///     matching the generator's ``particle`` form. A **string** ``particle``
    ///     returns a single ``WeightWindowBounds``; a **list** ``particle``
    ///     returns a ``list`` in that order (even a one-element list), e.g. a
    ///     coupled ``["neutron", "photon"]`` generation returns ``[neutron_ww,
    ///     photon_ww]``. Add a single window to a Model as
    ///     ``variance_reduction=[ww]``; pass a returned list straight through as
    ///     ``variance_reduction=windows``.
    #[pyo3(signature = (generator, total_particles=None, seed=1, threads=None, max_runtime=None, compute="cpu"))]
    pub fn generate_weight_windows<'py>(
        &self,
        py: Python<'py>,
        generator: crate::variance_reduction::PyWeightWindowGeneratorDeGVR,
        total_particles: Option<usize>,
        seed: u64,
        threads: Option<usize>,
        max_runtime: Option<pyo3::Py<pyo3::types::PyAny>>,
        compute: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Accepted so the request can be REFUSED in words rather than being
        // unreachable: without the argument there was no way to ask for the GPU
        // here at all, which reads as "not thought about" rather than "not
        // supported" (issue #339).
        if compute.trim() != "cpu" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "generate_weight_windows only supports compute='cpu': the GPU kernels do not \
                 honour weight windows yet, and generation is ordinary transport under a \
                 window, so it cannot run there before they do. Track it on the GPU feature \
                 gaps epic.",
            ));
        }
        // `total_particles=0` is an error (0 is not "unlimited"; use None).
        if total_particles == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "total_particles must be positive (use None for no particle cap)",
            ));
        }
        // Parse the optional per-pass wall-time budget (float seconds or a
        // (value, unit) tuple), while the GIL is held, before py.detach.
        let max_runtime_secs = match max_runtime {
            Some(obj) => Some(crate::distribution::parse_duration(obj.bind(py))?),
            None => None,
        };
        // Each pass needs at least one stop condition, else it never ends.
        if total_particles.is_none() && max_runtime_secs.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "generate_weight_windows needs a per-pass stop condition: set total_particles \
                 and/or max_runtime.",
            ));
        }
        let settings = yamc::model::TransportSettings {
            total_particles,
            seed,
            threads,
            max_runtime: max_runtime_secs,
        };
        let result = py.detach(|| {
            self.inner
                .generate_weight_windows(&generator.inner, &settings)
        });
        let mut windows = result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        // Return shape mirrors the `particle` argument form: a single-string
        // request yields one WeightWindowBounds; a list request yields a list in
        // request order (even of one). `particle_is_list` records that intent so
        // `["photon"]` round-trips as a one-element list, not a bare bounds.
        if !generator.particle_is_list {
            let single = crate::variance_reduction::PyWeightWindowBounds {
                inner: windows.pop().unwrap(),
            };
            Ok(single.into_pyobject(py)?.into_any())
        } else {
            let objs: Vec<Bound<'py, PyAny>> = windows
                .into_iter()
                .map(|inner| {
                    crate::variance_reduction::PyWeightWindowBounds { inner }
                        .into_pyobject(py)
                        .map(|b| b.into_any())
                })
                .collect::<PyResult<_>>()?;
            Ok(objs.into_pyobject(py)?.into_any())
        }
    }

    /// Estimate the auto DeGVR density-reduction factor N and the optical depth
    /// tau for ``generator``, via a particle-free ray-trace (source to deepest
    /// voxel). Returns ``(N, tau)`` without running generation, so you can inspect
    /// what the auto (``density_reduction=None``) path would choose.
    pub fn estimate_density_reduction(
        &self,
        generator: crate::variance_reduction::PyWeightWindowGeneratorDeGVR,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .estimate_density_reduction(&generator.inner)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Model(cells={}, tallies={})",
            self.inner.geometry.num_cells(),
            self.inner.tallies.len(),
        )
    }

    /// Serialize the model to a JSON string.
    ///
    /// This is what `save` writes to disk and what `Model.load` reads.
    /// The JSON describes the *specification* -- geometry, materials,
    /// source, tallies, run parameters. Nuclide cross-section data is
    /// **not** included; a loaded model still needs nuclide data to
    /// run (just like a model built from scratch).
    ///
    /// Args:
    ///     pretty: When True, emit multi-line indented JSON (useful for
    ///         inspecting models or for ``git diff``-friendly checked-in
    ///         model files). Defaults to compact single-line JSON.
    #[pyo3(signature = (pretty=false))]
    pub fn to_json(&self, pretty: bool) -> PyResult<String> {
        let result = if pretty {
            serde_json::to_string_pretty(&self.inner)
        } else {
            serde_json::to_string(&self.inner)
        };
        result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {e}")))
    }

    /// Save the model to a JSON file.
    ///
    /// Args:
    ///     path: Filesystem path to write to. Format is JSON; conventional
    ///         extension is ``.json``. Pretty-printed for readability.
    ///
    /// Examples:
    ///     >>> model.save("blanket_v3.json")
    pub fn save(&self, path: &str) -> PyResult<()> {
        let json = self.to_json(true)?;
        std::fs::write(path, json)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {path}: {e}")))?;
        Ok(())
    }

    /// Load a model from a JSON file (or string).
    ///
    /// Args:
    ///     source: Filesystem path (if it exists) or a JSON string.
    ///
    /// Returns:
    ///     A new Model. Cross-section data is **not** loaded -- call
    ///     ``read_nuclear_data`` on the materials before
    ///     ``simulate_transport``.
    ///
    /// Examples:
    ///     >>> model = yamc.Model.load("blanket_v3.json")
    #[staticmethod]
    pub fn load(source: &str) -> PyResult<Self> {
        // If `source` looks like a path that exists, read the file.
        // Otherwise treat as the JSON string directly.
        let json: String = if std::path::Path::new(source).is_file() {
            std::fs::read_to_string(source)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {source}: {e}")))?
        } else {
            source.to_string()
        };
        let inner: yamc::model::Model = serde_json::from_str(&json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("parse JSON: {e}")))?;
        let gpu_max_steps_explicit = inner.gpu_max_steps_per_particle != 100_000;
        Ok(PyModel {
            inner,
            gpu_max_steps_explicit,
        })
    }

    /// List of tallies to score during simulation.
    ///
    /// Returns:
    ///     List of Tally objects
    #[getter]
    pub fn tallies(&self) -> Vec<PyTally> {
        self.inner
            .tallies
            .iter()
            .map(|t| PyTally { inner: t.clone() })
            .collect()
    }

    /// Run the Monte Carlo simulation.
    ///
    /// Args:
    ///     total_particles: Particle-history budget for this run, or ``None``
    ///         (default) for no particle cap. A property of the run, not the
    ///         model: the same model can be run at several sample sizes
    ///         (e.g. a convergence sweep) without rebuilding geometry,
    ///         sources, or tallies. Must be positive when given (``0`` is an
    ///         error; use ``None`` for "no cap"). With ``None`` the run
    ///         continues until another stop condition trips, so set
    ///         ``max_runtime`` and/or convergence targets, or the run never
    ///         ends. A run with no stop condition at all raises ``ValueError``.
    ///         On ``compute='gpu'`` an uncapped run needs ``max_runtime`` to
    ///         stop (GPU can't stop on convergence targets yet).
    ///     seed: Base RNG seed for this run (default: 1). Per-particle
    ///         streams derive from it, so the seed fully determines the
    ///         run. Give each run a distinct seed when accumulating
    ///         statistics across runs with ``combine_results``.
    ///     threads: Number of threads to use for parallel execution. Defaults to None,
    ///         which uses all available CPU threads. Ignored when ``compute='gpu'``.
    ///     capture_tracks: Which particle histories to record full event tracks
    ///         for (debugging/visualisation). Histories are addressed by their
    ///         global 0-based index, so the captured set is the same regardless
    ///         of thread count. Can be:
    ///         - ``None`` (default): no tracking, zero overhead
    ///         - ``int N``: the first ``N`` histories (i.e. ``range(0, N)``)
    ///         - ``range(20, 30)``: histories 20..29 (half-open); strides too,
    ///           e.g. ``range(0, 1000, 10)`` for every 10th of the first 1000
    ///         - ``Sequence[int]`` e.g. ``[20, 30, 45]``: exactly those indices
    ///         - ``'all'``: every history (warns; high memory on large runs)
    ///         Only supported when ``compute='cpu'``.
    ///     compute: Where to run the transport. ``'cpu'`` (default) uses the
    ///         existing CPU implementation; ``'gpu'`` runs the cubecl/Vulkan
    ///         kernel via yamc-gpu. The GPU path handles neutron transport
    ///         (elastic, inelastic, and fission) over sphere/plane/cylinder/
    ///         torus surfaces with transmissive and vacuum boundaries, and
    ///         writes tally results back -- flux, reaction-rate, heating,
    ///         damage-energy, and production scores, including energy bins.
    ///         It does not support photon transport, particle tracking
    ///         (``capture_tracks``), or a custom ``threads`` count. Raises
    ///         ``RuntimeError`` if no GPU with f64 compute is available, or
    ///         ``ValueError`` if the model uses a feature the kernel doesn't
    ///         support.
    ///     max_runtime: Optional wall-time budget. The run stops at the first
    ///         chunk checkpoint where elapsed wall time has reached the budget,
    ///         then finalizes tallies normally, so the returned ``TallyResult``s
    ///         carry valid statistics and figure of merit. Accepts a float
    ///         (seconds) or a ``(value, unit)`` tuple with unit ``'s'``,
    ///         ``'min'``, ``'h'``, ``'d'``, or ``'a'`` (year), e.g.
    ///         ``(5, 'min')``. Composes with the other stop conditions: the run
    ///         ends at the first of {``total_particles`` exhausted, all
    ///         ``convergence_targets`` met, ``max_runtime`` elapsed}. Supported
    ///         on ``compute='cpu'`` and ``compute='gpu'`` (the GPU can only stop
    ///         between kernel launches, so it may overshoot the budget by up to
    ///         one launch). Under MPI (``mpi_size > 1``) the stop is collective
    ///         (any rank over budget stops them all at the same checkpoint); the
    ///         convergence early-stop is still single-process for now. A
    ///         time-bounded run is non-deterministic in history count, but the
    ///         results are statistically valid for the histories completed.
    ///
    /// Returns:
    ///     ``SimulationResults`` containing finalized ``TallyResult``s for
    ///     every tally attached to the model. When ``capture_tracks`` is set,
    ///     the captured ``Tracks`` are available on ``results.tracks``.
    ///
    /// Raises:
    ///     ValueError: if two tallies share the same name or the same id, or
    ///         if the model uses a feature the GPU kernel doesn't support,
    ///         including convergence targets (the GPU launch loop cannot stop
    ///         on them yet, so they are refused rather than ignored), or if a
    ///         GPU launch truncated histories at ``gpu_max_steps_per_particle``
    ///         (the under-counted tallies are never returned).
    ///     RuntimeError: if ``compute='gpu'`` and no GPU with f64 compute is
    ///         available.
    ///
    /// ``compute`` selects the device: ``'cpu'``, ``'gpu'`` (auto-selects an
    /// adapter, preferring a discrete GPU), or an adapter name from
    /// ``yamc.parallel.list_gpu_adapters()`` to pin a specific GPU.
    ///
    /// Examples:
    ///     >>> results = model.simulate_transport(total_particles=1000)
    ///     >>> results[flux_tally].mean
    ///     >>> results = model.simulate_transport(total_particles=1000, capture_tracks=100)
    ///     >>> df = pd.DataFrame(results.tracks.to_dataframe_records())
    ///     >>> results = model.simulate_transport(total_particles=1_000_000, compute='gpu')
    ///     >>> # run on a specific GPU by name:
    ///     >>> adapters = yamc.parallel.list_gpu_adapters()
    ///     >>> results = model.simulate_transport(total_particles=1_000_000, compute=adapters[0])
    ///     >>> # no particle cap: run for about 5 minutes, take whatever converged:
    ///     >>> results = model.simulate_transport(max_runtime=(5, "min"))
    #[pyo3(signature = (total_particles=None, seed=1, threads=None, capture_tracks=None, compute="cpu", max_runtime=None))]
    pub fn simulate_transport(
        &mut self,
        total_particles: Option<usize>,
        seed: u64,
        threads: Option<usize>,
        capture_tracks: Option<pyo3::Py<pyo3::types::PyAny>>,
        compute: &str,
        max_runtime: Option<pyo3::Py<pyo3::types::PyAny>>,
        py: Python,
    ) -> PyResult<crate::simulation::PySimulationResults> {
        // `total_particles=0` is an error (0 is not "unlimited"; use `None`).
        if total_particles == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "total_particles must be positive (use None for no particle cap)",
            ));
        }
        // `compute` is "cpu", "gpu" (auto-select adapter), or an adapter name
        // from `yamc.parallel.list_gpu_adapters()` to pin a specific GPU. Anything
        // other than "cpu"/"gpu" is treated as an adapter name (an unknown one fails
        // at GPU dispatch). Surrounding whitespace is ignored.
        let compute = compute.trim();
        // Parse the optional wall-time budget (float seconds or (value, unit)
        // tuple); like every other run parameter it lives in the per-call
        // settings, never on the model.
        let max_runtime_secs = match max_runtime {
            Some(obj) => Some(crate::distribution::parse_duration(obj.bind(py))?),
            None => None,
        };
        // A run needs at least one stop condition, else it never ends. The
        // conditions are OR-combined (the run ends at the first satisfied):
        // `total_particles`, `max_runtime`, or convergence targets on the
        // model. (A `convergence=` call argument is coming in a follow-up and
        // will join this OR.)
        if total_particles.is_none()
            && max_runtime_secs.is_none()
            && self.inner.convergence_targets.is_empty()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "A simulation needs at least one stop condition: set total_particles, \
                 max_runtime, and/or convergence targets. The run ends at the first one \
                 satisfied.",
            ));
        }
        let settings = yamc::model::TransportSettings {
            total_particles,
            seed,
            threads,
            max_runtime: max_runtime_secs,
        };
        if compute == "cpu" {
            self.warn_if_max_steps_ignored(py, "simulate_transport(compute='cpu')")?;
            return self.simulate_transport_cpu(&settings, capture_tracks, py);
        }
        // The GPU launch loop stops on total_particles and/or max_runtime,
        // checked between launches. It cannot evaluate convergence targets
        // (fusion-neutronics/core#29), so a model carrying them is refused
        // outright: it used to be refused only when neither budget was set,
        // and with a cap present it ran silently to the cap while the
        // precision the user asked to stop at was ignored
        // (fusion-neutronics/core#23). The OR-guard above already ensures a
        // cap or budget exists once there are no targets, so the Rust
        // dispatch's own UncappedWithoutRuntime is unreachable from here. The
        // Rust dispatch repeats this check for its other callers.
        if !self.inner.convergence_targets.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compute='gpu' cannot stop on convergence targets yet: the launch loop only \
                 checks total_particles and max_runtime between launches, so the {} target(s) \
                 on this model would be ignored and the run would go to the cap. Clear \
                 Model.convergence_targets to run this on the GPU, or use compute='cpu'.",
                self.inner.convergence_targets.len()
            )));
        }
        let device: Option<String> = if compute == "gpu" {
            None
        } else {
            Some(compute.to_string())
        };
        if parse_capture_tracks(py, capture_tracks)?.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "capture_tracks is not supported with compute='gpu'",
            ));
        }
        if threads.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threads is not used by compute='gpu' (the kernel \
                 parallelises across all particles in one launch); \
                 remove the argument",
            ));
        }
        self.simulate_transport_gpu(py, device, &settings)
    }

    /// Lost particles from the last run.
    ///
    /// Returns a list of LostParticle objects, each containing diagnostic
    /// information about a particle that was lost due to a geometry gap.
    /// Empty if no particles were lost. This stays on the Model (rather than
    /// SimulationResults) so the diagnostics remain inspectable after a
    /// ``max_lost_particles`` abort, when ``simulate_transport`` raises and
    /// returns no results object.
    ///
    /// Returns:
    ///     List of LostParticle objects
    #[getter]
    pub fn lost_particles(&self) -> Vec<crate::lost_particle::PyLostParticle> {
        self.inner
            .lost_particles
            .iter()
            .cloned()
            .map(crate::lost_particle::PyLostParticle::from)
            .collect()
    }

    /// Generate 2D maps of cell IDs and material IDs for the model's geometry.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///
    /// Returns:
    ///     A :class:`GeometrySliceData` -- also supports tuple unpacking.
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy"))]
    pub fn sample_slice(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<crate::id_map_helper::PyGeometrySliceData> {
        use crate::id_map_helper::{cell_to_ids, compute_sample_slice};

        let geometry = &self.inner.geometry;
        compute_sample_slice(
            origin,
            width,
            resolution,
            basis,
            &geometry.bounding_box(),
            py,
            |point| {
                geometry
                    .find_cell(point)
                    .map(|c| cell_to_ids(c, geometry.materials()))
                    .unwrap_or((-1, -1))
            },
        )
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
    ///     n_samples: Number of source particles to plot in the PNG overlay and
    ///         the initial HTML render. None hides the source overlay. In the
    ///         interactive HTML the user can change the ``n_samples`` input box
    ///         to resample any number of points without re-running Python.
    ///         Defaults to None; the HTML input box defaults to 1000.
    ///     plane_tolerance: Tolerance for source particles on plot plane, in cm.
    ///         Defaults to 1.0.
    ///     contour_kwargs: Outline appearance options. Supported keys:
    ///         ``colors`` (hex string, default "#000000"),
    ///         ``linewidths`` (int, default 1).
    ///     source_kwargs: Source point appearance options. Supported keys:
    ///         ``color`` (hex string, default "#ff0000"),
    ///         ``size`` (int resolution, default 3).
    ///
    /// Returns:
    ///     InteractivePlot: Result object that renders in Jupyter and supports
    ///         ``.save("file.html")`` and ``.save("file.png")``.
    #[gen_stub(skip)] // FIXME(stub): pyo3-stub-gen mishandles Option<&str> default; hand-add in stub patch-merge
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy", color_by="cell", outline="cell", axis_units="cm", colors=None, font_size=2, n_samples=None, plane_tolerance=1.0, contour_kwargs=None, source_kwargs=None))]
    pub fn plot(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        color_by: &str,
        outline: Option<&str>,
        axis_units: &str,
        colors: Option<Bound<'_, PyDict>>,
        font_size: usize,
        n_samples: Option<usize>,
        plane_tolerance: f64,
        contour_kwargs: Option<Bound<'_, PyDict>>,
        source_kwargs: Option<Bound<'_, PyDict>>,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<crate::ui::PyInteractivePlot> {
        use crate::id_map_helper::parse_sample_slice_params;
        use crate::ui::parse_colors_ids_cells_materials;
        use crate::ui::{
            build_interactive_html, parse_contour_kwargs, parse_source_kwargs,
            InteractiveViewParams, PyInteractivePlot,
        };

        let color_map = parse_colors_ids_cells_materials(colors)?;
        let contour = parse_contour_kwargs(contour_kwargs.as_ref())?;
        let source = parse_source_kwargs(source_kwargs.as_ref())?;
        let bbox = self.inner.geometry.bounding_box();
        let params = parse_sample_slice_params(origin, width, resolution, basis, &bbox, py)?;

        // Convert geometry depending on backend
        let (geometry_json, geometry_kind, cell_names, material_names) = match &self.inner.geometry
        {
            yamc::geometry::backend::GeometryKind::Csg(g) => {
                let csg = yamc::geometry::conversion::geometry_to_csg(g);
                let json = serde_json::to_string(&csg)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let mut cn = std::collections::HashMap::new();
                let mut mn = std::collections::HashMap::new();
                for cell in &csg.cells {
                    if let Some(ref name) = cell.name {
                        cn.insert(cell.cell_id, name.clone());
                    }
                    if cell.material_id != -1 {
                        if let Some(ref name) = cell.material_name {
                            mn.insert(cell.material_id, name.clone());
                        }
                    }
                }
                (json, "csg", cn, mn)
            }
            #[cfg(feature = "mesh")]
            yamc::geometry::backend::GeometryKind::Mesh(m) => {
                let geo_mesh = yamc::geometry::conversion::mesh_geometry_to_geo_mesh(m);
                let json = serde_json::to_string(&geo_mesh)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let mut cn = std::collections::HashMap::new();
                let mut mn = std::collections::HashMap::new();
                for vol in &geo_mesh.volumes {
                    cn.insert(vol.cell_id, format!("Volume {}", vol.cell_id));
                    if vol.material_id != -1 {
                        if let Some(ref name) = vol.material_name {
                            mn.insert(vol.material_id, name.clone());
                        }
                    }
                }
                (json, "mesh", cn, mn)
            }
        };

        let view_params = InteractiveViewParams {
            origin: params.origin,
            width: params.width,
            total_pixels: params.pixels.0 * params.pixels.1,
            basis: params.basis.clone(),
            color_by: color_by.to_string(),
            outline: outline.map(|s| s.to_string()),
            axis_units: axis_units.to_string(),
            bbox_widths: bbox.width(),
            outline_color: contour.colors.clone(),
            outline_thickness: contour.linewidths,
            n_samples,
            plane_tolerance,
            source_color: source.color.clone(),
            source_size: source.size,
        };

        // Sample exactly `n_samples` source positions in Rust for the PNG
        // overlay. The HTML doesn't use these -- it receives the source
        // distribution as JSON and runs a JS sampler, so users can re-run at
        // any count without re-running Python.
        let source_positions: Vec<[f64; 3]> = if let Some(ns) = n_samples {
            let mut rng = rand::rng();
            let selector = yamc_source::source::SourceSelector::new(&self.inner.sources);
            (0..ns)
                .map(|_| {
                    let p = self.inner.sample_source_with(&selector, &mut rng);
                    [p.position[0], p.position[1], p.position[2]]
                })
                .collect()
        } else {
            Vec::new()
        };

        // Distribution JSON for HTML -- JS samples from this on load and on
        // input-box change. Shape: {"sources": [{strength, spatial, energy}, ...]}.
        // Particle type (neutron/photon) is elided since plotting only cares
        // about positions.
        let source_dist_json = {
            let sources: Vec<serde_json::Value> = self
                .inner
                .sources
                .iter()
                .map(|ps| {
                    let src = ps.source();
                    serde_json::json!({
                        "strength": src.strength,
                        "spatial": &src.space,
                        "energy": &src.energy,
                    })
                })
                .collect();
            serde_json::to_string(&serde_json::json!({ "sources": sources }))
                .unwrap_or_else(|_| "null".to_string())
        };

        // Pre-sample the initial view with native Rust
        let presampled = {
            use crate::id_map_helper::{cell_to_ids, generate_presampled_grid};
            let geometry = &self.inner.geometry;
            generate_presampled_grid(&params, |point| {
                geometry
                    .find_cell(point)
                    .map(|c| cell_to_ids(c, geometry.materials()))
                    .unwrap_or((-1, -1))
            })
        };

        // A mesh-filled CSG cell (issue #291) must ship its raster too: the
        // browser sampler reads `geometry_json`, where a fill is only an identity
        // fingerprint, so it would draw the bare CSG frame and hide the fill body
        // the particles actually see. `presampled` resolves fills (it goes
        // through `Geometry::find_cell`), so it is the authoritative view and
        // `viewer.js` refuses to re-sample when `HAS_MESH_FILLS` is set.
        let has_mesh_fills = match &self.inner.geometry {
            yamc::geometry::backend::GeometryKind::Csg(g) => !g.fills.is_empty(),
            #[cfg(feature = "mesh")]
            yamc::geometry::backend::GeometryKind::Mesh(_) => false,
        };
        // Otherwise embed the pre-sample only for mesh geometries -- mesh still
        // hides WASM compile + first-sample latency behind the pre-sample. For
        // plain CSG the browser sampler is fast enough to render directly and the
        // ~3-4 MB base64 grid was pure file-size bloat.
        let presampled_for_html = if geometry_kind == "mesh" || has_mesh_fills {
            Some(&presampled)
        } else {
            None
        };
        // Surface name + identity table for hover tooltips. CSG-only;
        // mesh geometries don't have a CsgGeometry to walk.
        let surface_table = match &self.inner.geometry {
            yamc::geometry::backend::GeometryKind::Csg(g) => {
                let csg = yamc::geometry::conversion::geometry_to_csg(g);
                Some(yamc_plot::build_surface_table(&csg))
            }
            #[cfg(feature = "mesh")]
            yamc::geometry::backend::GeometryKind::Mesh(_) => None,
        };
        let html = build_interactive_html(
            &geometry_json,
            geometry_kind,
            "",
            &view_params,
            &cell_names,
            &material_names,
            color_map.as_ref(),
            Some(&source_dist_json),
            presampled_for_html,
            has_mesh_fills,
            surface_table.as_deref(),
        );

        // Build source overlay for PNG if n_samples is set
        let source_overlay = n_samples.map(|ns| crate::ui::SourceOverlay {
            points: source_positions,
            n_samples: ns,
            plane_tolerance,
            color: source.color.clone(),
            size: source.size,
            basis: basis.to_string(),
            origin: params.origin,
            width: params.width,
        });

        Ok(PyInteractivePlot::new(
            html,
            &presampled,
            color_by,
            outline,
            color_map.as_ref(),
            params.origin,
            params.width,
            basis,
            axis_units,
            font_size,
            &contour.colors,
            contour.linewidths,
            source_overlay.as_ref(),
        ))
    }

    /// Run transport-transmutation calculation.
    ///
    /// Args:
    ///     method: Transmutation method -- ``"coupled"`` or ``"independent"``.
    ///
    ///         * ``"coupled"`` -- transport runs every timestep with updated
    ///           material compositions (standard approach).
    ///         * ``"independent"`` -- transport runs **once**; the resulting
    ///           per-source-particle reaction rates are scaled by each step's
    ///           ``source_rate``.  Much faster for low-burnup / fusion scenarios
    ///           where compositions barely change.
    ///
    ///         Both modes run one extra scouting transport first, before any
    ///         transmutation product is loaded. It supplies the flux spectrum
    ///         used to work out which products the irradiation can actually
    ///         populate, and only those are then scored during the steps, which
    ///         is what keeps the per-step cost off the reachable closure.
    ///     schedule: A :class:`PulseSchedule` of ``Pulse`` / ``Cooldown`` steps.
    ///         Each ``Pulse`` contributes its ``duration`` and a source ``rate``
    ///         in n/s; each ``Cooldown`` (or a ``Pulse`` with rate 0) is a
    ///         decay-only step with no transport. The Pulse source distribution
    ///         is not used here: transport always uses the source configured on
    ///         the Model, and a schedule with more than one distinct Pulse source
    ///         is rejected.
    ///     total_particles: Particle histories per transport step, or ``None``
    ///         (default) for no particle cap. In ``"coupled"`` mode every
    ///         timestep's transport run uses this count; ``"independent"`` mode
    ///         runs transport once. The scouting run uses it too, so that it
    ///         sees the same histories the steps will. One of
    ///         ``total_particles`` / ``max_runtime`` must be set (they are the
    ///         per-step stop conditions); a run with neither raises
    ///         ``ValueError``.
    ///     seed: Base RNG seed for the transport runs (default: 1).
    ///     threads: Optional number of threads for transport. Defaults to all available.
    ///     max_runtime: Optional per-step wall-time budget for each transport
    ///         solve (per timestep in ``"coupled"`` mode; the single solve in
    ///         ``"independent"`` mode). Each solve stops at the first chunk
    ///         checkpoint where its elapsed time reaches the budget, then
    ///         finalizes normally. Accepts a float (seconds) or a
    ///         ``(value, unit)`` tuple with unit ``'s'``, ``'min'``, ``'h'``,
    ///         ``'d'``, or ``'a'`` (year), e.g. ``(30, 's')``. Composes with
    ///         ``total_particles``: a solve ends at the first of the two.
    ///         Because the budget is applied afresh to each timestep, a
    ///         time-bounded transmutation is non-deterministic in history count.
    ///
    /// Returns:
    ///     TransmutationResults object containing material compositions at each timestep.
    ///
    /// Examples:
    ///     >>> schedule = yamc.PulseSchedule([
    ///     ...     yamc.Pulse(rate=1e14, duration=86400.0, source=source),
    ///     ...     yamc.Cooldown(duration=86400.0),
    ///     ... ])
    ///     >>> results = model.simulate_transmutation(
    ///     ...     method="coupled", schedule=schedule, total_particles=1000)
    ///     >>> co60 = results.get_nuclide_evolution(1, "Co60")
    #[pyo3(signature = (method, schedule, total_particles=None, seed=1, threads=None, max_runtime=None, compute="cpu"))]
    pub fn simulate_transmutation(
        &mut self,
        method: String,
        schedule: &Bound<'_, PyAny>,
        total_particles: Option<usize>,
        seed: u64,
        threads: Option<usize>,
        max_runtime: Option<pyo3::Py<pyo3::types::PyAny>>,
        compute: &str,
        py: Python<'_>,
    ) -> PyResult<PyTransmutationResults> {
        // As on `generate_weight_windows`: present so the answer is a sentence
        // rather than a missing argument (issue #339).
        if compute.trim() != "cpu" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "simulate_transmutation only supports compute='cpu': the depletion solve has \
                 no GPU backend, and the per-step transport would have to hand its rates back \
                 to a CPU solve every step even if it ran on the device.",
            ));
        }
        // `total_particles=0` is an error (0 is not "unlimited").
        if total_particles == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "total_particles must be positive",
            ));
        }
        // `simulate_transmutation` has no `compute=` argument, so its transport
        // solves always run on the CPU, which ignores the step cap.
        self.warn_if_max_steps_ignored(py, "simulate_transmutation")?;
        // Parse the optional per-step wall-time budget (float seconds or a
        // (value, unit) tuple), while the GIL is held, before py.detach.
        let max_runtime_secs = match max_runtime {
            Some(obj) => Some(crate::distribution::parse_duration(obj.bind(py))?),
            None => None,
        };
        // Each transport solve needs at least one stop condition. Both budgets
        // are per transport step (per timestep in "coupled" mode): a step ends
        // at the first of {total_particles exhausted, max_runtime elapsed}.
        if total_particles.is_none() && max_runtime_secs.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "simulate_transmutation needs a per-step stop condition: set total_particles \
                 and/or max_runtime.",
            ));
        }
        let settings = yamc::model::TransportSettings {
            total_particles,
            seed,
            threads,
            max_runtime: max_runtime_secs,
        };
        // Unpack the PulseSchedule into the (timesteps, source_rates) the core
        // transmutation driver expects. Done while the GIL is held, before
        // py.detach, since it touches Python objects.
        let sched = schedule
            .cast::<crate::distribution::PyPulseSchedule>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("schedule must be a PulseSchedule")
            })?;
        let sched = sched.borrow();
        if sched.distinct_source_count(py) > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "simulate_transmutation uses the model's configured source for all pulses; a \
                 schedule with multiple distinct Pulse sources is not yet supported (per-pulse \
                 source override is not implemented). Use one source, or run separate transport \
                 per source and combine with PulseSchedule.time_correct_tally.",
            ));
        }
        // Clone the core numeric timeline so it is owned (Send) and can cross
        // the py.detach boundary; the Python source objects stay behind.
        let core_schedule = sched.core_schedule().clone();
        drop(sched);

        // With verbose "nuclear_data", log the chain file and each nuclide file
        // as they load (resolve_chain loads the chain; transmute loads the
        // per-nuclide cross sections). Process-global toggle, reset on every
        // path so it never leaks past this call.
        let log_loads = self.inner.verbose.nuclear_data;
        if log_loads {
            yamc_nuclide::set_load_logging(true);
        }
        let outcome = crate::distribution::resolve_chain().and_then(|loaded| {
            let chain = loaded.chain;
            let branch = loaded.branch;
            let parts = loaded.parts;
            // Release GIL during long-running transport computation
            let result: Result<TransmutationResults, String> = py.detach(|| {
                self.inner
                    .transmute(&method, &core_schedule, chain, branch, parts, &settings)
                    .map_err(|e| e.to_string())
            });
            match result {
                Ok(results) => Ok(PyTransmutationResults { inner: results }),
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
            }
        });
        if log_loads {
            yamc_nuclide::set_load_logging(false);
        }
        outcome
    }
}

// Private run helpers. Outside the #[pymethods] block: they take
// `&TransportSettings`, which is not a Python-extractable argument type.
impl PyModel {
    fn simulate_transport_cpu(
        &mut self,
        settings: &yamc::model::TransportSettings,
        capture_tracks: Option<pyo3::Py<pyo3::types::PyAny>>,
        py: Python,
    ) -> PyResult<crate::simulation::PySimulationResults> {
        let selection = parse_capture_tracks(py, capture_tracks)?;
        let start = std::time::Instant::now();
        let py_tracks = if let Some(selection) = selection {
            if matches!(selection, HistorySelection::All) {
                py.import("warnings")?.call_method1(
                    "warn",
                    (
                        "capture_tracks='all' records every history, which can use a lot of \
                      memory on large runs; pass an int, range, or list of history indices \
                      to bound it.",
                    ),
                )?;
            }
            let storage = py
                .detach(|| self.inner.run_with_tracking(settings, selection))
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Some(crate::simulation::PyTracks::from(storage))
        } else {
            py.detach(|| self.inner.simulate_transport(settings))
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            None
        };
        let elapsed = start.elapsed().as_secs_f64();
        crate::simulation::build_results(&self.inner, settings, "cpu", elapsed, py_tracks)
    }

    /// Warn that `gpu_max_steps_per_particle` does nothing on a CPU run.
    ///
    /// Only the GPU kernel reads the cap: it needs a bound in its loop condition
    /// (driver watchdog, lockstep workgroups). CPU transport runs
    /// `while particle.alive` and ends histories on absorption, leakage or the
    /// lost-particle diagnostics, so the value is ignored there. That asymmetry
    /// is documented, but silently ignoring a request the user typed is not the
    /// same as telling them, so say it once per run.
    ///
    /// Fires only when the value was set explicitly (constructor argument or
    /// setter), never for the default, which every model carries.
    fn warn_if_max_steps_ignored(&self, py: Python<'_>, entry_point: &str) -> PyResult<()> {
        if !self.gpu_max_steps_explicit {
            return Ok(());
        }
        py.import("warnings")?.call_method1(
            "warn",
            (format!(
                "gpu_max_steps_per_particle={} has no effect on {entry_point}: only the GPU \
                 kernel applies the cap (it needs a bounded loop), while CPU transport runs \
                 every history to completion and ends it on absorption, leakage or a lost \
                 particle. Remove the argument, or pass compute='gpu' if you meant to cap \
                 GPU histories.",
                self.inner.gpu_max_steps_per_particle
            ),),
        )?;
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn simulate_transport_gpu(
        &mut self,
        py: Python,
        device: Option<String>,
        settings: &yamc::model::TransportSettings,
    ) -> PyResult<crate::simulation::PySimulationResults> {
        use yamc::gpu::{run_on_gpu_with_device, GpuDispatchError};

        // CPU path runs `init_photon_data` on every material as part
        // of `run_internal`'s prep; the GPU dispatch goes straight to
        // `run_on_gpu` and would otherwise see materials with empty
        // `cached_elements`. Run the same prep here so a fresh model
        // built for a `compute='gpu'` run works without first calling
        // `compute='cpu'`.
        self.inner
            .ensure_photon_data_for_gpu()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let start = std::time::Instant::now();
        let device = device.as_deref();
        let gpu_result = py.detach(|| run_on_gpu_with_device(&mut self.inner, device, settings));
        let elapsed = start.elapsed().as_secs_f64();

        match gpu_result {
            Ok(_run) => {
                // `run_on_gpu` writes the kernel's flux into the
                // tally accumulator (when one is configured) before
                // returning. From here it's the same shape as the
                // CPU path: read finalised tally results out of
                // `self.inner.tallies` and wrap them in
                // `PySimulationResults`.
                crate::simulation::build_results(&self.inner, settings, "gpu", elapsed, None)
            }
            Err(GpuDispatchError::GpuUnavailable(e)) => Err(
                pyo3::exceptions::PyRuntimeError::new_err(format!("GPU unavailable: {e}")),
            ),
            Err(other) => Err(pyo3::exceptions::PyValueError::new_err(other.to_string())),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn simulate_transport_gpu(
        &mut self,
        _py: Python,
        _device: Option<String>,
        _settings: &yamc::model::TransportSettings,
    ) -> PyResult<crate::simulation::PySimulationResults> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "compute='gpu' requires yamc to be built with the `gpu` Cargo \
             feature, which is in the default set, so this build opted out of \
             it: rebuild without `--no-default-features`, or add `gpu` back to \
             the feature list you passed",
        ))
    }
}
