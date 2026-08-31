//! Python bindings for variance-reduction technique configuration.
//!
//! `Model(variance_reduction=[...])` takes a list of technique objects
//! (`SurvivalBiasing` and `WeightWindowBounds` today; source biasing joins
//! later). Techniques compose; the list order carries no meaning.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::variance_reduction::{
    SurvivalBiasing, VarianceReduction, WeightWindowBounds, WeightWindowGeneratorDeGVR,
};
use yamc_particle::particle::ParticleType;

use crate::geometry::PyRegularRectangularMesh;

/// Survival biasing (implicit capture) with weight-cutoff Russian roulette.
///
/// Absorption never terminates the particle; instead each collision
/// multiplies its weight by the scattering probability and the particle
/// always scatters. A particle whose weight drops below ``weight_cutoff``
/// plays roulette: it survives with probability ``weight / weight_survive``
/// (continuing at ``weight_survive``) or is killed, conserving weight in
/// expectation.
///
/// Args:
///     weight_cutoff: Weight below which a particle plays Russian roulette
///         (default: 0.25).
///     weight_survive: Weight given to a particle that survives roulette
///         (default: 1.0). Must be >= ``weight_cutoff``.
///
/// Examples:
///     >>> model = yamc.Model(
///     ...     geometry=geometry,
///     ...     source=source,
///     ...     variance_reduction=[yamc.SurvivalBiasing()],
///     ... )
///     >>> model.simulate_transport(total_particles=100_000)
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "SurvivalBiasing", from_py_object)]
#[derive(Clone)]
pub struct PySurvivalBiasing {
    pub inner: SurvivalBiasing,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySurvivalBiasing {
    #[new]
    #[pyo3(signature = (weight_cutoff=0.25, weight_survive=1.0))]
    pub fn new(weight_cutoff: f64, weight_survive: f64) -> PyResult<Self> {
        let inner = SurvivalBiasing {
            weight_cutoff,
            weight_survive,
        };
        inner
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Russian-roulette trigger weight.
    #[getter]
    pub fn weight_cutoff(&self) -> f64 {
        self.inner.weight_cutoff
    }

    #[setter(weight_cutoff)]
    pub fn set_weight_cutoff(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight_cutoff must be > 0 (got {value})"
            )));
        }
        self.inner.weight_cutoff = value;
        Ok(())
    }

    /// Weight assigned to roulette survivors.
    #[getter]
    pub fn weight_survive(&self) -> f64 {
        self.inner.weight_survive
    }

    #[setter(weight_survive)]
    pub fn set_weight_survive(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight_survive must be > 0 (got {value})"
            )));
        }
        self.inner.weight_survive = value;
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SurvivalBiasing(weight_cutoff={}, weight_survive={})",
            self.inner.weight_cutoff, self.inner.weight_survive
        )
    }
}

/// A mesh-based weight-window map applied at collisions: a particle whose
/// weight exceeds the local upper bound is split into equal-weight copies
/// (capped by ``max_split``), one at or below the lower bound plays Russian
/// roulette (survivor weight ``survival_factor * lower``), and one below
/// ``weight_floor`` is killed. Bounds are given per (energy group, voxel),
/// flat, indexed ``group * num_voxels + voxel``; a negative lower bound is
/// the "no window" sentinel. Add to ``Model(variance_reduction=[...])``;
/// several entries (for example one per particle type) are allowed.
///
/// Args:
///     mesh: RegularRectangularMesh the windows are defined on.
///     lower_bounds: Lower bound per (group, voxel), flat.
///     upper_bounds: Upper bound per (group, voxel), flat. If omitted, it is
///         ``ratio * lower_bounds`` (sentinels preserved).
///     ratio: Upper/lower ratio used when ``upper_bounds`` is omitted
///         (default 5.0).
///     energy_bins: Energy group edges in eV, ascending. None = single group.
///     particle: ``"neutron"`` (default) or ``"photon"``.
///     survival_factor: Roulette survivor weight = factor * lower (default 3).
///     max_split: Maximum copies produced in one split event (default 10).
///     weight_floor: Absolute weight floor below which a particle is killed
///         (default 1e-38).
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "WeightWindowBounds", from_py_object)]
#[derive(Clone)]
pub struct PyWeightWindowBounds {
    pub inner: WeightWindowBounds,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWeightWindowBounds {
    #[new]
    #[pyo3(signature = (mesh, lower_bounds, upper_bounds=None, ratio=5.0, energy_bins=None, particle="neutron", survival_factor=3.0, max_split=10, weight_floor=1e-38))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: PyRegularRectangularMesh,
        lower_bounds: Vec<f64>,
        upper_bounds: Option<Vec<f64>>,
        ratio: f64,
        energy_bins: Option<Vec<f64>>,
        particle: &str,
        survival_factor: f64,
        max_split: u32,
        weight_floor: f64,
    ) -> PyResult<Self> {
        let particle = match particle {
            "neutron" => ParticleType::Neutron,
            "photon" => ParticleType::Photon,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "particle must be 'neutron' or 'photon' (got {other:?})"
                )))
            }
        };
        let upper_bounds = match upper_bounds {
            Some(u) => u,
            None => {
                if ratio <= 1.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "ratio must be > 1 when upper_bounds is omitted (got {ratio})"
                    )));
                }
                lower_bounds
                    .iter()
                    .map(|&lo| if lo < 0.0 { lo } else { ratio * lo })
                    .collect()
            }
        };
        let inner = WeightWindowBounds {
            mesh: mesh.internal.clone(),
            particle,
            energy_bins,
            lower_bounds,
            upper_bounds,
            survival_factor,
            max_split,
            weight_floor,
        };
        inner
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Lower window bounds, flat over (group, voxel).
    #[getter]
    pub fn lower_bounds(&self) -> Vec<f64> {
        self.inner.lower_bounds.clone()
    }

    /// Upper window bounds, flat over (group, voxel).
    #[getter]
    pub fn upper_bounds(&self) -> Vec<f64> {
        self.inner.upper_bounds.clone()
    }

    /// Particle type these windows apply to (``"neutron"`` or ``"photon"``).
    #[getter]
    pub fn particle(&self) -> String {
        match self.inner.particle {
            ParticleType::Neutron => "neutron".to_string(),
            ParticleType::Photon => "photon".to_string(),
        }
    }

    /// Roulette survivor factor.
    #[getter]
    pub fn survival_factor(&self) -> f64 {
        self.inner.survival_factor
    }

    /// Maximum copies produced in one split event.
    #[getter]
    pub fn max_split(&self) -> u32 {
        self.inner.max_split
    }

    /// Absolute weight floor below which a particle is killed.
    #[getter]
    pub fn weight_floor(&self) -> f64 {
        self.inner.weight_floor
    }

    pub fn __repr__(&self) -> String {
        format!(
            "WeightWindowBounds(particle={}, n_groups={}, num_voxels={}, survival_factor={}, max_split={}, weight_floor={})",
            self.particle(),
            self.inner.n_groups(),
            self.inner.num_voxels(),
            self.inner.survival_factor,
            self.inner.max_split,
            self.inner.weight_floor,
        )
    }
}

/// DeGVR (Density-extrapolation Global Variance Reduction) weight-window
/// generator. Pass to ``Model.generate_weight_windows(...)``, which runs two
/// reduced-density passes internally and returns a ``WeightWindowBounds`` for a
/// clean production run. Suited to bulk deep-penetration shielding; weak on
/// streaming / void / near-source regions.
///
/// Args:
///     mesh: RegularRectangularMesh the windows are generated on.
///     energy_bins: Energy group edges in eV, ascending. None = single group.
///     particle: ``"neutron"`` (default) or ``"photon"``, or a list such as
///         ``["neutron", "photon"]`` to generate a window per particle from a
///         single coupled generation. A single string returns one
///         ``WeightWindowBounds``; a list returns a list, in the given order.
///     density_reduction: Factor N; the fiducial pass uses ``density / N`` and
///         the asymptotic pass ``density / (N / 2)``. None (default) = auto:
///         N is computed from a particle-free optical-depth ray-trace of the
///         problem (source to deepest voxel; the max over requested particles).
///         Pass a float to override.
///     ratio: Upper/lower ratio of the generated windows (default 5).
///     survival_factor: Roulette survivor factor (default 3).
///     max_split: Maximum copies per split event (default 10).
///     weight_floor: Absolute weight floor (default 1e-38).
///     photon_energy: Representative photon energy in eV for the auto-N
///         ray-trace of a photon window. None (default) = auto: the source
///         energy for a photon source, else ~1 MeV for neutron-driven secondary
///         / decay photons. Only affects the auto ``density_reduction``, never
///         bias.
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "WeightWindowGeneratorDeGVR",
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightWindowGeneratorDeGVR {
    pub inner: WeightWindowGeneratorDeGVR,
    /// Whether `particle` was given as a list (vs a single string). Records the
    /// caller's intent so the `particle` getter and `generate_weight_windows`
    /// return the matching shape (a list request always yields a list, even of
    /// one, while a string request yields a scalar). Not part of the serialized
    /// Rust config; a Python-ergonomics flag only.
    pub particle_is_list: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWeightWindowGeneratorDeGVR {
    #[new]
    #[pyo3(signature = (mesh, energy_bins=None, particle=None, density_reduction=None, ratio=5.0, survival_factor=3.0, max_split=10, weight_floor=1e-38, photon_energy=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: PyRegularRectangularMesh,
        energy_bins: Option<Vec<f64>>,
        particle: Option<Bound<'_, PyAny>>,
        density_reduction: Option<f64>,
        ratio: f64,
        survival_factor: f64,
        max_split: u32,
        weight_floor: f64,
        photon_energy: Option<f64>,
    ) -> PyResult<Self> {
        let (particles, particle_is_list) = parse_particles(particle)?;
        let inner = WeightWindowGeneratorDeGVR {
            mesh: mesh.internal.clone(),
            energy_bins,
            particles,
            photon_energy,
            density_reduction,
            ratio,
            survival_factor,
            max_split,
            weight_floor,
        };
        inner
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner,
            particle_is_list,
        })
    }

    /// Density-reduction factor N (None = auto, computed at generation time).
    #[getter]
    pub fn density_reduction(&self) -> Option<f64> {
        self.inner.density_reduction
    }

    /// Representative photon energy (eV) for the auto-N ray-trace, or None (auto).
    #[getter]
    pub fn photon_energy(&self) -> Option<f64> {
        self.inner.photon_energy
    }

    /// Particle type(s), round-tripping the constructor form: a single
    /// ``"neutron"`` / ``"photon"`` string when a string was passed, else a
    /// ``list[str]`` in request order (a list request always reads back as a
    /// list, even of one).
    #[getter]
    pub fn particle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let names: Vec<String> = self
            .inner
            .particles
            .iter()
            .map(|p| particle_type_name(*p).to_string())
            .collect();
        if self.particle_is_list {
            Ok(names.into_pyobject(py)?.into_any())
        } else {
            Ok(names
                .into_iter()
                .next()
                .unwrap()
                .into_pyobject(py)?
                .into_any())
        }
    }

    pub fn __repr__(&self) -> String {
        let dr = match self.inner.density_reduction {
            Some(n) => n.to_string(),
            None => "auto".to_string(),
        };
        let particles: Vec<&str> = self
            .inner
            .particles
            .iter()
            .map(|p| particle_type_name(*p))
            .collect();
        let pe = match self.inner.photon_energy {
            Some(e) => e.to_string(),
            None => "auto".to_string(),
        };
        format!(
            "WeightWindowGeneratorDeGVR(particle={:?}, density_reduction={}, ratio={}, survival_factor={}, max_split={}, photon_energy={})",
            particles,
            dr,
            self.inner.ratio,
            self.inner.survival_factor,
            self.inner.max_split,
            pe,
        )
    }
}

/// Map a particle-type name to the enum, rejecting anything else.
fn particle_type_from_str(s: &str) -> PyResult<ParticleType> {
    match s {
        "neutron" => Ok(ParticleType::Neutron),
        "photon" => Ok(ParticleType::Photon),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "particle must be 'neutron' or 'photon' (got {other:?})"
        ))),
    }
}

/// The canonical string name of a particle type.
fn particle_type_name(p: ParticleType) -> &'static str {
    match p {
        ParticleType::Neutron => "neutron",
        ParticleType::Photon => "photon",
    }
}

/// Parse the DeGVR generator's `particle` argument into the resolved particle
/// list and whether it was given as a **list** (vs a single string): a single
/// string (`"photon"`) -> `(vec![Photon], false)`, a list of strings
/// (`["neutron", "photon"]`) -> `(vec![...], true)`, or `None` (defaults to
/// `(vec![Neutron], false)`). The is-list flag drives the return/getter shape so
/// a list request round-trips as a list even of one. A single string is tried
/// before the list form (a Python `str` is itself an iterable of 1-char strings,
/// so the order matters).
fn parse_particles(obj: Option<Bound<'_, PyAny>>) -> PyResult<(Vec<ParticleType>, bool)> {
    let Some(obj) = obj else {
        return Ok((vec![ParticleType::Neutron], false));
    };
    if let Ok(s) = obj.extract::<String>() {
        return Ok((vec![particle_type_from_str(&s)?], false));
    }
    if let Ok(list) = obj.extract::<Vec<String>>() {
        if list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "particle list must not be empty",
            ));
        }
        let particles = list
            .iter()
            .map(|s| particle_type_from_str(s))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok((particles, true));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "particle must be a string or a list of strings ('neutron' / 'photon')",
    ))
}

/// Parse a Python `variance_reduction` list into the Rust configuration,
/// enforcing per-type multiplicity (at most one `SurvivalBiasing`). The
/// per-entry `extract` keeps the door open for further technique types.
pub fn parse_variance_reduction(
    entries: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<VarianceReduction>> {
    let mut parsed = Vec::with_capacity(entries.len());
    let mut survival_count = 0usize;
    for entry in &entries {
        if let Ok(sb) = entry.extract::<PySurvivalBiasing>() {
            survival_count += 1;
            parsed.push(VarianceReduction::SurvivalBiasing(sb.inner));
        } else if let Ok(ww) = entry.extract::<PyWeightWindowBounds>() {
            parsed.push(VarianceReduction::WeightWindowBounds(Box::new(ww.inner)));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "variance_reduction entries must be SurvivalBiasing or WeightWindowBounds (got {})",
                entry.get_type().name()?
            )));
        }
    }
    if survival_count > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "variance_reduction may contain at most one SurvivalBiasing entry (got {survival_count})"
        )));
    }
    Ok(parsed)
}
