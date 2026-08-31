//! Python bindings for the parametric tokamak plasma source.
//!
//! The signatures deliberately mirror the `openmc-plasma-source` package
//! (same argument names, same units, same defaults) so a model can be run
//! through both codes and compared without rewriting the plasma parameters.

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use yamc_source::distribution::energy::FusionReactants;
use yamc_source::source::ParticleSource;
use yamc_source::tokamak::{
    convert_a_alpha_to_r_z, neutron_source_density, ConfinementMode, FuelIon, TokamakPlasma,
};

use crate::distribution::PyNeutronSource;

/// A float argument that may also be given as a sequence of floats, so the
/// profile helpers can be called on a single radius or on a whole array of
/// them for plotting.
enum FloatArg {
    Scalar(f64),
    Sequence(Vec<f64>),
}

impl FloatArg {
    fn extract(name: &str, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
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

    fn values(&self) -> &[f64] {
        match self {
            Self::Scalar(value) => std::slice::from_ref(value),
            Self::Sequence(values) => values,
        }
    }

    fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

/// Return a float for a scalar input and a list for a sequence input.
fn shape_like(py: Python<'_>, scalar: bool, values: Vec<f64>) -> PyResult<Py<PyAny>> {
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

fn parse_fuel(fuel: Option<HashMap<String, f64>>) -> PyResult<Vec<(FuelIon, f64)>> {
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
/// The profile models are from Fausser et al., 'Tokamak D-T neutron source
/// models for different plasma physics confinement modes', Fus. Eng. Des. 87
/// (2012) 787, and the arguments match the ``openmc-plasma-source`` package's
/// ``tokamak_source`` so the two can be compared directly. Two physics
/// differences to expect in such a comparison: T(t,2n)4He is not included,
/// since yamc has no T-T neutron spectrum model, where
/// ``openmc-plasma-source`` takes one from NeSST (for D-T fuel the T-T yield
/// is a small fraction of the D-T yield); and the birth peaks sit a few tens
/// of keV apart, because yamc's Ballabio spectrum takes the zero-temperature
/// neutron energy from the reaction Q value rather than from Ballabio's
/// tabulated 14.021 / 2.4495 MeV.
///
/// Args:
///     major_radius: Plasma major radius (cm).
///     minor_radius: Plasma minor radius (cm).
///     elongation: Plasma elongation (dimensionless).
///     triangularity: Plasma triangularity (dimensionless).
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
///     pedestal_radius: Minor radius at the pedestal (cm).
///     shafranov_factor: Shafranov factor (cm), the outward radial shift of
///             the magnetic surfaces.
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
///     ValueError: If the plasma parameters are inconsistent (for example a
///             minor radius larger than the major radius), the fuel is not
///             D and/or T, or the plasma is too cold to make neutrons.
///
/// Examples:
///     >>> import yamc
///     >>> sources = yamc.sources.tokamak_source(
///     ...     major_radius=906.0,
///     ...     minor_radius=292.258,
///     ...     elongation=1.557,
///     ...     triangularity=0.270,
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
///     ...     shafranov_factor=0.44789,
///     ... )
///     >>> model = yamc.Model(geometry, sources, tallies)  # doctest: +SKIP
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "tokamak_source",
    signature = (
        *,
        major_radius,
        minor_radius,
        elongation,
        triangularity,
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
        shafranov_factor,
        start_angle = 0.0,
        rotation_angle = std::f64::consts::TAU,
        mesh_resolution = (100, 100),
        grid_density = 500,
        fuel = None,
    )
)]
pub fn py_tokamak_source(
    major_radius: f64,
    minor_radius: f64,
    elongation: f64,
    triangularity: f64,
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
    shafranov_factor: f64,
    start_angle: f64,
    rotation_angle: f64,
    mesh_resolution: (usize, usize),
    grid_density: usize,
    fuel: Option<HashMap<String, f64>>,
) -> PyResult<Vec<PyNeutronSource>> {
    let plasma = TokamakPlasma {
        major_radius,
        minor_radius,
        elongation,
        triangularity,
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
        shafranov_factor,
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
    let plasma = TokamakPlasma {
        mode: ConfinementMode::parse(mode).map_err(PyErr::new::<PyValueError, _>)?,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        minor_radius,
        pedestal_radius,
        ..TokamakPlasma::default()
    };
    let radii = FloatArg::extract("r", r)?;
    map_arg(py, &radii, |value| plasma.ion_density(value))
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
    let plasma = TokamakPlasma {
        mode: ConfinementMode::parse(mode).map_err(PyErr::new::<PyValueError, _>)?,
        pedestal_radius,
        ion_temperature_pedestal,
        ion_temperature_centre,
        ion_temperature_beta,
        ion_temperature_peaking_factor,
        ion_temperature_separatrix,
        minor_radius,
        ..TokamakPlasma::default()
    };
    let radii = FloatArg::extract("r", r)?;
    map_arg(py, &radii, |value| plasma.ion_temperature(value))
}

/// Convert plasma (a, alpha) coordinates to (R, Z) coordinates.
///
/// ``a`` is the minor radius of a flux surface and ``alpha`` the poloidal
/// angle; the mapping applies triangularity, elongation and the Shafranov
/// shift. Either argument may be a float or a sequence of floats, and a
/// scalar broadcasts against a sequence.
///
/// Args:
///     a: Minor radius of the flux surface (cm).
///     alpha: Poloidal angle (radians).
///     shafranov_factor: Shafranov factor (cm).
///     minor_radius: Plasma minor radius (cm).
///     major_radius: Plasma major radius (cm).
///     triangularity: Plasma triangularity (dimensionless).
///     elongation: Plasma elongation (dimensionless).
///
/// Returns:
///     An ``(R, Z)`` tuple in cm, each a float or a list of floats.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "tokamak_convert_a_alpha_to_r_z",
    signature = (
        *,
        a,
        alpha,
        shafranov_factor,
        minor_radius,
        major_radius,
        triangularity,
        elongation,
    )
)]
pub fn py_tokamak_convert_a_alpha_to_r_z(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    alpha: &Bound<'_, PyAny>,
    shafranov_factor: f64,
    minor_radius: f64,
    major_radius: f64,
    triangularity: f64,
    elongation: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (pairs, scalar) = zip_args(
        "a",
        &FloatArg::extract("a", a)?,
        "alpha",
        &FloatArg::extract("alpha", alpha)?,
    )?;
    let mut radii = Vec::with_capacity(pairs.len());
    let mut heights = Vec::with_capacity(pairs.len());
    for (a, alpha) in pairs {
        if a < 0.0 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "a must not be negative (got {a})"
            )));
        }
        let (r, z) = convert_a_alpha_to_r_z(
            a,
            alpha,
            shafranov_factor,
            minor_radius,
            major_radius,
            triangularity,
            elongation,
        );
        radii.push(r);
        heights.push(z);
    }
    Ok((
        shape_like(py, scalar, radii)?,
        shape_like(py, scalar, heights)?,
    ))
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
