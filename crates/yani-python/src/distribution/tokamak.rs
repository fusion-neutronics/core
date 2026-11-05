//! Python bindings for the parametric tokamak plasma source and its shape.
//!
//! The profile arguments keep the `openmc-plasma-source` package's names,
//! units and defaults, so its plasma parameters carry over as they are. The
//! shape is the exception: its five arguments (`major_radius`,
//! `minor_radius`, `elongation`, `triangularity`, `shafranov_factor`) make up
//! a `TokamakShape`, passed to `tokamak_source` as `shape`, as a
//! `StellaratorShape` is to `stellarator_source`. The names and units are
//! unchanged, but a call is no longer argument for argument the same.

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yamc_source::distribution::energy::FusionReactants;
use yamc_source::source::ParticleSource;
use yamc_source::tokamak::{
    neutron_source_density, normalised_radius, ConfinementMode, FuelIon, PlasmaProfiles,
    TokamakPlasma, TokamakShape,
};

use crate::distribution::PyNeutronSource;

/// A float argument that may also be given as a sequence of floats, so the
/// profile helpers can be called on a single radius or on a whole array of
/// them for plotting.
pub(crate) enum FloatArg {
    Scalar(f64),
    Sequence(Vec<f64>),
}

impl FloatArg {
    pub(crate) fn extract(name: &str, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<f64>() {
            return Ok(Self::Scalar(value));
        }
        match ob.extract::<Vec<f64>>() {
            Ok(values) => Ok(Self::Sequence(values)),
            Err(_) => Err(PyErr::new::<PyTypeError, _>(format!(
                "{name} must be a float or a sequence of floats"
            ))),
        }
    }

    pub(crate) fn values(&self) -> &[f64] {
        match self {
            Self::Scalar(value) => std::slice::from_ref(value),
            Self::Sequence(values) => values,
        }
    }

    pub(crate) fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

/// Return a float for a scalar input and a list for a sequence input.
pub(crate) fn shape_like(py: Python<'_>, scalar: bool, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    if scalar {
        Ok(values[0].into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(values.into_pyobject(py)?.into_any().unbind())
    }
}

/// Apply `f` elementwise, keeping the caller's scalar-or-sequence shape.
fn map_arg(
    py: Python<'_>,
    arg: &FloatArg,
    f: impl Fn(f64) -> Result<f64, String>,
) -> PyResult<Py<PyAny>> {
    let values = arg
        .values()
        .iter()
        .map(|value| f(*value))
        .collect::<Result<Vec<f64>, String>>()
        .map_err(PyErr::new::<PyValueError, _>)?;
    shape_like(py, arg.is_scalar(), values)
}

/// Pair up two arguments elementwise, broadcasting a scalar against a
/// sequence. The result is a scalar only when both inputs are.
fn zip_args(
    first_name: &str,
    first: &FloatArg,
    second_name: &str,
    second: &FloatArg,
) -> PyResult<(Vec<(f64, f64)>, bool)> {
    let (left, right) = (first.values(), second.values());
    let length = left.len().max(right.len());
    if left.len() != length && !first.is_scalar() || right.len() != length && !second.is_scalar() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{first_name} and {second_name} must be the same length (got {} and {})",
            left.len(),
            right.len()
        )));
    }
    let pairs = (0..length)
        .map(|i| (left[i % left.len()], right[i % right.len()]))
        .collect();
    Ok((pairs, first.is_scalar() && second.is_scalar()))
}

fn parse_reaction(reaction: &str) -> PyResult<FusionReactants> {
    match reaction {
        "DD" => Ok(FusionReactants::DD),
        "DT" => Ok(FusionReactants::DT),
        other => Err(PyErr::new::<PyValueError, _>(format!(
            "reaction must be \"DD\" or \"DT\" (got {other:?}). T-T fusion is not supported: \
             yamc has no T(t,2n) neutron spectrum model"
        ))),
    }
}

pub(crate) fn parse_fuel(fuel: Option<HashMap<String, f64>>) -> PyResult<Vec<(FuelIon, f64)>> {
    let Some(fuel) = fuel else {
        return Ok(vec![(FuelIon::Deuterium, 0.5), (FuelIon::Tritium, 0.5)]);
    };
    let mut parsed = fuel
        .into_iter()
        .map(|(species, fraction)| {
            FuelIon::parse(&species)
                .map(|ion| (ion, fraction))
                .map_err(PyErr::new::<PyValueError, _>)
        })
        .collect::<PyResult<Vec<_>>>()?;
    // A dict has no meaningful order; sort so the source list is reproducible.
    parsed.sort_by_key(|(ion, _)| *ion);
    Ok(parsed)
}

/// A tokamak plasma shape: nested flux surfaces with elongation,
/// triangularity and a Shafranov shift, in the parameterisation of Fausser et
/// al., Fus. Eng. Des. 87 (2012) 787.
///
/// The flux surface at normalised radius ``rho`` (0 on the magnetic axis, 1
/// on the plasma boundary) has minor radius ``a = rho * minor_radius``. At
/// poloidal angle ``theta`` it lies at
/// ``R = major_radius + a * cos(theta + triangularity * sin(theta)) + shift``
/// and ``Z = elongation * a * sin(theta)``, where the Shafranov shift
/// ``shift = shafranov_factor * (1 - rho**2)`` moves the magnetic axis out by
/// ``shafranov_factor`` and falls to zero on the boundary.
///
/// These are the five shape arguments of the ``openmc-plasma-source``
/// package's ``tokamak_source``, with the same names and units, made into the
/// one ``shape`` argument of ``tokamak_source`` here, as ``StellaratorShape``
/// is for ``stellarator_source``.
///
/// Args:
///     major_radius: Plasma major radius (cm).
///     minor_radius: Plasma minor radius (cm), less than ``major_radius``.
///     elongation: Plasma elongation (dimensionless).
///     triangularity: Plasma triangularity (dimensionless), from -1 to 1.
///     shafranov_factor: Shafranov factor (cm), the outward radial shift of
///         the magnetic axis, less than half the minor radius either way.
///
/// Raises:
///     ValueError: If a parameter is NaN or infinite (the message names it),
///         a radius or the elongation is not positive, the minor radius is
///         not less than the major radius, the triangularity is outside -1
///         to 1, or the Shafranov factor is half the minor radius or more.
///
/// Examples:
///     >>> import yamc
///     >>> shape = yamc.sources.TokamakShape(
///     ...     major_radius=906.0,
///     ...     minor_radius=292.258,
///     ...     elongation=1.557,
///     ...     triangularity=0.270,
///     ...     shafranov_factor=0.44789,
///     ... )
///     >>> r, z = shape.evaluate(1.0, 0.0)  # the outboard midplane
#[gen_stub_pyclass]
#[pyclass(name = "TokamakShape", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTokamakShape {
    pub inner: TokamakShape,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTokamakShape {
    #[new]
    #[pyo3(signature = (*, major_radius, minor_radius, elongation, triangularity, shafranov_factor))]
    fn new(
        major_radius: f64,
        minor_radius: f64,
        elongation: f64,
        triangularity: f64,
        shafranov_factor: f64,
    ) -> PyResult<Self> {
        let inner = TokamakShape::new(
            major_radius,
            minor_radius,
            elongation,
            triangularity,
            shafranov_factor,
        )
        .map_err(PyErr::new::<PyValueError, _>)?;
        Ok(Self { inner })
    }

    /// Plasma major radius (cm).
    #[getter]
    fn major_radius(&self) -> f64 {
        self.inner.major_radius()
    }

    /// Plasma minor radius (cm).
    #[getter]
    fn minor_radius(&self) -> f64 {
        self.inner.minor_radius()
    }

    /// Plasma elongation (dimensionless).
    #[getter]
    fn elongation(&self) -> f64 {
        self.inner.elongation()
    }

    /// Plasma triangularity (dimensionless).
    #[getter]
    fn triangularity(&self) -> f64 {
        self.inner.triangularity()
    }

    /// Shafranov factor (cm), the outward radial shift of the magnetic axis.
    #[getter]
    fn shafranov_factor(&self) -> f64 {
        self.inner.shafranov_factor()
    }

    /// Plasma volume (cm^3), inside the boundary ``rho = 1``.
    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    /// Cylindrical ``(R, Z)`` in cm at flux coordinates ``(rho, theta)``.
    ///
    /// Either argument is a float or a sequence of floats; a scalar
    /// broadcasts against a sequence, so a whole flux surface or radial
    /// chord evaluates in one call.
    ///
    /// Args:
    ///     rho: Normalised radius, 0 on the magnetic axis and 1 on the
    ///         boundary.
    ///     theta: Poloidal angle (radians).
    ///
    /// Returns:
    ///     An ``(R, Z)`` tuple in cm, each a float or a list of floats.
    ///
    /// Raises:
    ///     ValueError: If ``rho`` is outside 0 to 1, or ``rho`` and ``theta``
    ///         are sequences of different lengths.
    fn evaluate(
        &self,
        py: Python<'_>,
        rho: &Bound<'_, PyAny>,
        theta: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (pairs, scalar) = zip_args(
            "rho",
            &FloatArg::extract("rho", rho)?,
            "theta",
            &FloatArg::extract("theta", theta)?,
        )?;
        let (mut radii, mut heights) = (
            Vec::with_capacity(pairs.len()),
            Vec::with_capacity(pairs.len()),
        );
        for (rho, theta) in pairs {
            if !(0.0..=1.0).contains(&rho) {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "rho must be between 0 and 1 (got {rho})"
                )));
            }
            let (r, z) = self.inner.point(rho, theta);
            radii.push(r);
            heights.push(z);
        }
        Ok((
            shape_like(py, scalar, radii)?,
            shape_like(py, scalar, heights)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TokamakShape(major_radius={}, minor_radius={}, elongation={}, triangularity={}, \
             shafranov_factor={})",
            self.inner.major_radius(),
            self.inner.minor_radius(),
            self.inner.elongation(),
            self.inner.triangularity(),
            self.inner.shafranov_factor()
        )
    }
}

/// Build the neutron sources describing a parametric tokamak plasma.
///
/// The plasma is discretised onto a cylindrical (R, phi, Z) mesh: every
/// non-empty voxel becomes a ring source sampling uniformly across the voxel
/// and across the toroidal sector, emitting isotropically with the Ballabio
/// spectrum at that voxel's emission-weighted ion temperature. One source is
/// returned per (voxel, reaction), so D-D and D-T neutrons keep their own
/// spectra. Strengths are normalised to sum to 1; the absolute neutron rate
/// is set by the simulation, not here.
///
/// The shape and profile models are from Fausser et al., 'Tokamak D-T neutron
/// source models for different plasma physics confinement modes', Fus. Eng.
/// Des. 87 (2012) 787, and the arguments are those of the
/// ``openmc-plasma-source`` package's ``tokamak_source``, with the same
/// names, units and defaults, so the two can be compared. The exception is
/// the shape: its five arguments (``major_radius``, ``minor_radius``,
/// ``elongation``, ``triangularity`` and ``shafranov_factor``) are given here
/// as one ``TokamakShape``, as ``stellarator_source`` takes a
/// ``StellaratorShape``.
///
/// Two physics differences to expect in such a comparison: T(t,2n)4He is not
/// included, since yamc has no T-T neutron spectrum model, where
/// ``openmc-plasma-source`` takes one from NeSST (for D-T fuel the T-T yield
/// is a small fraction of the D-T yield); and the birth peaks sit a few tens
/// of keV apart, because yamc's Ballabio spectrum takes the zero-temperature
/// neutron energy from the reaction Q value rather than from Ballabio's
/// tabulated 14.021 / 2.4495 MeV.
///
/// Args:
///     shape: The plasma shape.
///     mode: Confinement mode, one of 'L', 'H' or 'A'.
///     ion_density_centre: Ion density at the plasma centre (m^-3).
///     ion_density_peaking_factor: Ion density peaking factor (dimensionless).
///     ion_density_pedestal: Ion density at the pedestal (m^-3).
///     ion_density_separatrix: Ion density at the separatrix (m^-3).
///     ion_temperature_centre: Ion temperature at the plasma centre (eV).
///     ion_temperature_peaking_factor: Ion temperature peaking factor
///             (dimensionless, alpha_T in the reference).
///     ion_temperature_beta: Ion temperature beta exponent (dimensionless,
///             beta_T in the reference).
///     ion_temperature_pedestal: Ion temperature at the pedestal (eV).
///     ion_temperature_separatrix: Ion temperature at the separatrix (eV).
///     pedestal_radius: Minor radius at the pedestal (cm), less than the
///             shape's ``minor_radius``.
///     start_angle: Toroidal angle at which the plasma starts, in radians.
///             Defaults to 0.
///     rotation_angle: Toroidal extent of the plasma, in radians. Negative
///             extends the plasma the other way from ``start_angle``.
///             Defaults to a full torus.
///     mesh_resolution: Number of (R, Z) mesh bins as a two-value tuple.
///             Defaults to (100, 100). The toroidal direction is not a
///             parameter: the plasma is axisymmetric, so it carries no
///             structure. Each non-empty voxel costs one source, so a coarser
///             mesh is cheaper to sample.
///     grid_density: Points per dimension of the internal (a, alpha) grid
///             mapped onto the mesh. Defaults to 500.
///     fuel: Fuel species as keys ('D', 'T') and atom fractions as values,
///             which must sum to 1. Defaults to 50:50 D-T.
///
/// Returns:
///     list[NeutronSource]: One ring source per (mesh voxel, reaction).
///
/// Raises:
///     ValueError: If a parameter is NaN or infinite (the message names it),
///             the plasma parameters are inconsistent (for example a pedestal
///             radius beyond the shape's minor radius), the fuel is not D
///             and/or T, or the plasma is too cold to make neutrons.
///
/// Examples:
///     >>> import yamc
///     >>> shape = yamc.sources.TokamakShape(
///     ...     major_radius=906.0,
///     ...     minor_radius=292.258,
///     ...     elongation=1.557,
///     ...     triangularity=0.270,
///     ...     shafranov_factor=0.44789,
///     ... )
///     >>> sources = yamc.sources.tokamak_source(
///     ...     shape=shape,
///     ...     mode="H",
///     ...     ion_density_centre=1.09e20,
///     ...     ion_density_peaking_factor=1,
///     ...     ion_density_pedestal=1.09e20,
///     ...     ion_density_separatrix=3e19,
///     ...     ion_temperature_centre=45.9e3,
///     ...     ion_temperature_peaking_factor=8.06,
///     ...     ion_temperature_beta=6.0,
///     ...     ion_temperature_pedestal=6.09e3,
///     ...     ion_temperature_separatrix=0.1e3,
///     ...     pedestal_radius=0.8 * 292.258,
///     ... )
///     >>> model = yamc.Model(geometry, sources, tallies)  # doctest: +SKIP
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "tokamak_source",
    signature = (
        *,
        shape,
        mode,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        ion_temperature_centre,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        pedestal_radius,
        start_angle = 0.0,
        rotation_angle = std::f64::consts::TAU,
        mesh_resolution = (100, 100),
        grid_density = 500,
        fuel = None,
    )
)]
pub fn py_tokamak_source(
    shape: PyTokamakShape,
    mode: &str,
    ion_density_centre: f64,
    ion_density_peaking_factor: f64,
    ion_density_pedestal: f64,
    ion_density_separatrix: f64,
    ion_temperature_centre: f64,
    ion_temperature_peaking_factor: f64,
    ion_temperature_beta: f64,
    ion_temperature_pedestal: f64,
    ion_temperature_separatrix: f64,
    pedestal_radius: f64,
    start_angle: f64,
    rotation_angle: f64,
    mesh_resolution: (usize, usize),
    grid_density: usize,
    fuel: Option<HashMap<String, f64>>,
) -> PyResult<Vec<PyNeutronSource>> {
    let plasma = TokamakPlasma {
        shape: shape.inner,
        mode: ConfinementMode::parse(mode).map_err(PyErr::new::<PyValueError, _>)?,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        ion_temperature_centre,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        pedestal_radius,
        start_angle,
        rotation_angle,
        mesh_resolution,
        grid_density,
        fuel: parse_fuel(fuel)?,
    };
    let sources = plasma
        .sources()
        .map_err(PyErr::new::<PyValueError, _>)?
        .into_iter()
        .map(|source| PyNeutronSource {
            inner: ParticleSource::Neutron(source),
        })
        .collect();
    Ok(sources)
}

/// Ion density of a tokamak plasma at a given minor radius.
///
/// The density depends only on the minor radius, so this is the profile the
/// parametric source is built from. Pass a float or a sequence of floats for
/// ``r``; the return keeps that shape.
///
/// Args:
///     mode: Confinement mode, one of 'L', 'H' or 'A'.
///     ion_density_centre: Ion density at the plasma centre (m^-3).
///     ion_density_peaking_factor: Ion density peaking factor (dimensionless).
///     ion_density_pedestal: Ion density at the pedestal (m^-3).
///     minor_radius: Plasma minor radius (cm).
///     pedestal_radius: Minor radius at the pedestal (cm).
///     ion_density_separatrix: Ion density at the separatrix (m^-3).
///     r: Local minor radius (cm), from 0 to ``minor_radius``.
///
/// Returns:
///     Ion density in m^-3, as a float or a list of floats.
///
/// Raises:
///     ValueError: If ``mode`` is not 'L', 'H' or 'A', or ``r`` falls outside
///             the plasma.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "tokamak_ion_density",
    signature = (
        *,
        mode,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        minor_radius,
        pedestal_radius,
        ion_density_separatrix,
        r,
    )
)]
pub fn py_tokamak_ion_density(
    py: Python<'_>,
    mode: &str,
    ion_density_centre: f64,
    ion_density_peaking_factor: f64,
    ion_density_pedestal: f64,
    minor_radius: f64,
    pedestal_radius: f64,
    ion_density_separatrix: f64,
    r: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let profiles = PlasmaProfiles {
        mode: ConfinementMode::parse(mode).map_err(PyErr::new::<PyValueError, _>)?,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        pedestal: pedestal_radius / minor_radius,
        // Only the density profile is evaluated.
        ion_temperature_centre: 0.0,
        ion_temperature_peaking_factor: 0.0,
        ion_temperature_beta: 0.0,
        ion_temperature_pedestal: 0.0,
        ion_temperature_separatrix: 0.0,
    };
    let radii = FloatArg::extract("r", r)?;
    map_arg(py, &radii, |value| {
        Ok(profiles.ion_density(normalised_radius(value, minor_radius)?))
    })
}

/// Ion temperature of a tokamak plasma at a given minor radius.
///
/// The temperature depends only on the minor radius, so this is the profile
/// that sets the neutron spectrum across the plasma. Pass a float or a
/// sequence of floats for ``r``; the return keeps that shape.
///
/// Unlike the ``openmc-plasma-source`` helper of the same name, which takes
/// keV and returns eV, every temperature here is in eV.
///
/// Args:
///     r: Local minor radius (cm), from 0 to ``minor_radius``.
///     mode: Confinement mode, one of 'L', 'H' or 'A'.
///     pedestal_radius: Minor radius at the pedestal (cm).
///     ion_temperature_pedestal: Ion temperature at the pedestal (eV).
///     ion_temperature_centre: Ion temperature at the plasma centre (eV).
///     ion_temperature_beta: Ion temperature beta exponent (dimensionless).
///     ion_temperature_peaking_factor: Ion temperature peaking factor
///             (dimensionless).
///     ion_temperature_separatrix: Ion temperature at the separatrix (eV).
///     minor_radius: Plasma minor radius (cm).
///
/// Returns:
///     Ion temperature in eV, as a float or a list of floats.
///
/// Raises:
///     ValueError: If ``mode`` is not 'L', 'H' or 'A', or ``r`` falls outside
///             the plasma.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "tokamak_ion_temperature",
    signature = (
        *,
        r,
        mode,
        pedestal_radius,
        ion_temperature_pedestal,
        ion_temperature_centre,
        ion_temperature_beta,
        ion_temperature_peaking_factor,
        ion_temperature_separatrix,
        minor_radius,
    )
)]
pub fn py_tokamak_ion_temperature(
    py: Python<'_>,
    r: &Bound<'_, PyAny>,
    mode: &str,
    pedestal_radius: f64,
    ion_temperature_pedestal: f64,
    ion_temperature_centre: f64,
    ion_temperature_beta: f64,
    ion_temperature_peaking_factor: f64,
    ion_temperature_separatrix: f64,
    minor_radius: f64,
) -> PyResult<Py<PyAny>> {
    let profiles = PlasmaProfiles {
        mode: ConfinementMode::parse(mode).map_err(PyErr::new::<PyValueError, _>)?,
        ion_temperature_centre,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        pedestal: pedestal_radius / minor_radius,
        // Only the temperature profile is evaluated.
        ion_density_centre: 0.0,
        ion_density_peaking_factor: 0.0,
        ion_density_pedestal: 0.0,
        ion_density_separatrix: 0.0,
    };
    let radii = FloatArg::extract("r", r)?;
    map_arg(py, &radii, |value| {
        Ok(profiles.ion_temperature(normalised_radius(value, minor_radius)?))
    })
}

/// Neutron source density of a fusing plasma.
///
/// Uses the Bosch-Hale parameterisation of the Maxwellian-averaged
/// reactivity (Nucl. Fusion 32 (1992) 611), which is fitted for ion
/// temperatures from 0.2 keV to 100 keV. Either numeric argument may be a
/// float or a sequence of floats, and a scalar broadcasts against a sequence.
///
/// Args:
///     ion_density: Density of reacting pairs (m^-6): ``n_D * n_T`` for D-T,
///             ``0.5 * n_D**2`` for D-D.
///     ion_temperature: Ion temperature (eV).
///     reaction: The fusing pair, either 'DD' or 'DT'. 'DD' counts the
///             neutron-producing D(d,n)3He branch only.
///
/// Returns:
///     Neutron source density in neutrons/s/m^3, as a float or a list of
///     floats.
///
/// Raises:
///     ValueError: If ``reaction`` is not 'DD' or 'DT'.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(
    name = "tokamak_neutron_source_density",
    signature = (*, ion_density, ion_temperature, reaction = "DT")
)]
pub fn py_tokamak_neutron_source_density(
    py: Python<'_>,
    ion_density: &Bound<'_, PyAny>,
    ion_temperature: &Bound<'_, PyAny>,
    reaction: &str,
) -> PyResult<Py<PyAny>> {
    let reactants = parse_reaction(reaction)?;
    let (pairs, scalar) = zip_args(
        "ion_density",
        &FloatArg::extract("ion_density", ion_density)?,
        "ion_temperature",
        &FloatArg::extract("ion_temperature", ion_temperature)?,
    )?;
    let densities = pairs
        .into_iter()
        .map(|(density, temperature)| neutron_source_density(density, temperature, reactants))
        .collect();
    shape_like(py, scalar, densities)
}
