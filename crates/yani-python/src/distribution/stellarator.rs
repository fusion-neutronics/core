//! Python bindings for the stellarator plasma shape and source.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yamc_source::source::ParticleSource;
use yamc_source::stellarator::{FluxShell, StellaratorPlasma, StellaratorShape};
use yamc_source::tokamak::{ConfinementMode, PlasmaProfiles};

use super::tokamak::{parse_fuel, shape_like, FloatArg};
use crate::distribution::PyNeutronSource;

fn value_error(message: String) -> PyErr {
    PyErr::new::<PyValueError, _>(message)
}

/// A stellarator plasma shape as a Fourier-Zernike series, in DESC's
/// conventions.
///
/// ``R(rho, theta, zeta) = sum R_lmn * Z_l^m(rho, theta) * F_n(zeta)`` and
/// likewise for ``Z``, where ``Z_l^m`` is the unnormalised Zernike radial
/// polynomial ``R_l^|m|(rho)`` (equal to 1 at ``rho = 1``) times
/// ``cos(m theta)`` for ``m >= 0`` or ``sin(|m| theta)`` for ``m < 0``, and
/// ``F_n`` is ``cos(n N zeta)`` for ``n >= 0`` or ``sin(|n| N zeta)`` for
/// ``n < 0``, with ``N`` the number of field periods. ``rho`` is the square
/// root of the normalised toroidal flux and ``zeta`` the cylindrical toroidal
/// angle.
///
/// A DESC equilibrium's ``R_lmn`` / ``Z_lmn`` with its ``R_basis.modes`` /
/// ``Z_basis.modes`` are these coefficients directly, times 100 for cm.
///
/// Args:
///     num_field_periods: Number of field periods ``N``.
///     r_modes: ``(l, m, n)`` mode numbers of the ``R`` series.
///     r_lmn: Coefficients of the ``R`` series (cm).
///     z_modes: ``(l, m, n)`` mode numbers of the ``Z`` series.
///     z_lmn: Coefficients of the ``Z`` series (cm).
///
/// Raises:
///     ValueError: If a mode is not a Zernike mode (``l >= |m|``,
///         ``l - |m|`` even), a mode is repeated, the lengths differ, a
///         coefficient is NaN or infinite, or the surfaces reach ``R <= 0``,
///         self-intersect or are not nested.
///
/// Examples:
///     >>> import yamc
///     >>> # A circular torus: R = 600 + 150 rho cos(theta), Z = 150 rho sin(theta)
///     >>> shape = yamc.sources.StellaratorShape(
///     ...     num_field_periods=1,
///     ...     r_modes=[(0, 0, 0), (1, 1, 0)],
///     ...     r_lmn=[600.0, 150.0],
///     ...     z_modes=[(1, -1, 0)],
///     ...     z_lmn=[150.0],
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "StellaratorShape", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStellaratorShape {
    pub inner: Arc<StellaratorShape>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStellaratorShape {
    #[new]
    #[pyo3(signature = (*, num_field_periods, r_modes, r_lmn, z_modes, z_lmn))]
    fn new(
        num_field_periods: u32,
        r_modes: Vec<[i32; 3]>,
        r_lmn: Vec<f64>,
        z_modes: Vec<[i32; 3]>,
        z_lmn: Vec<f64>,
    ) -> PyResult<Self> {
        let shape = StellaratorShape::new(num_field_periods, r_modes, r_lmn, z_modes, z_lmn)
            .map_err(value_error)?;
        Ok(Self {
            inner: Arc::new(shape),
        })
    }

    /// Number of field periods.
    #[getter]
    fn num_field_periods(&self) -> u32 {
        self.inner.num_field_periods()
    }

    /// ``(l, m, n)`` mode numbers of the ``R`` series.
    #[getter]
    fn r_modes(&self) -> Vec<(i32, i32, i32)> {
        self.inner
            .r_modes()
            .iter()
            .map(|&[l, m, n]| (l, m, n))
            .collect()
    }

    /// Coefficients of the ``R`` series (cm), in the order of ``r_modes``.
    #[getter]
    fn r_lmn(&self) -> Vec<f64> {
        self.inner.r_lmn().to_vec()
    }

    /// ``(l, m, n)`` mode numbers of the ``Z`` series.
    #[getter]
    fn z_modes(&self) -> Vec<(i32, i32, i32)> {
        self.inner
            .z_modes()
            .iter()
            .map(|&[l, m, n]| (l, m, n))
            .collect()
    }

    /// Coefficients of the ``Z`` series (cm), in the order of ``z_modes``.
    #[getter]
    fn z_lmn(&self) -> Vec<f64> {
        self.inner.z_lmn().to_vec()
    }

    /// Plasma volume (cm^3).
    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    /// Cylindrical ``(R, Z)`` in cm at flux coordinates ``(rho, theta, zeta)``.
    ///
    /// Each argument is a float or a sequence of floats; scalars broadcast
    /// against sequences, so a whole surface or cross-section evaluates in
    /// one call.
    ///
    /// Args:
    ///     rho: Normalised radius, 0 on the axis and 1 on the boundary.
    ///     theta: Poloidal angle (radians).
    ///     zeta: Toroidal angle (radians).
    ///
    /// Returns:
    ///     An ``(R, Z)`` tuple in cm, each a float or a list of floats.
    fn evaluate(
        &self,
        py: Python<'_>,
        rho: &Bound<'_, PyAny>,
        theta: &Bound<'_, PyAny>,
        zeta: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let args = [
            ("rho", FloatArg::extract("rho", rho)?),
            ("theta", FloatArg::extract("theta", theta)?),
            ("zeta", FloatArg::extract("zeta", zeta)?),
        ];
        let length = args
            .iter()
            .map(|(_, a)| a.values().len())
            .max()
            .unwrap_or(1);
        for (name, arg) in &args {
            if !arg.is_scalar() && arg.values().len() != length {
                return Err(value_error(format!(
                    "rho, theta and zeta must be floats or sequences of the same length \
                     ({name} has {})",
                    arg.values().len()
                )));
            }
        }
        let at = |i: usize, k: usize| {
            let values = args[k].1.values();
            values[i % values.len()]
        };
        let (mut radii, mut heights) = (Vec::with_capacity(length), Vec::with_capacity(length));
        for i in 0..length {
            let rho = at(i, 0);
            if !(0.0..=1.0).contains(&rho) {
                return Err(value_error(format!(
                    "rho must be between 0 and 1 (got {rho})"
                )));
            }
            let p = self.inner.point(rho, at(i, 1), at(i, 2));
            radii.push(p.r);
            heights.push(p.z);
        }
        let scalar = args.iter().all(|(_, a)| a.is_scalar());
        Ok((
            shape_like(py, scalar, radii)?,
            shape_like(py, scalar, heights)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "StellaratorShape(num_field_periods={}, r_modes={} modes, z_modes={} modes)",
            self.inner.num_field_periods(),
            self.inner.r_modes().len(),
            self.inner.z_modes().len()
        )
    }
}

/// The part of a stellarator plasma between two flux surfaces, the spatial
/// distribution of a ``stellarator_source`` source.
#[gen_stub_pyclass]
#[pyclass(name = "FluxShell", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyFluxShell {
    pub inner: FluxShell,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFluxShell {
    /// Inner and outer ``rho`` of the shell.
    #[getter]
    fn rho(&self) -> (f64, f64) {
        (self.inner.rho[0], self.inner.rho[1])
    }

    /// The plasma shape the shell belongs to.
    #[getter]
    fn shape(&self) -> PyStellaratorShape {
        PyStellaratorShape {
            inner: Arc::clone(&self.inner.shape),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FluxShell(rho=({}, {}))",
            self.inner.rho[0], self.inner.rho[1]
        )
    }
}

/// Build the neutron sources describing a stellarator plasma.
///
/// The plasma is split into radial shells in ``rho``. Each shell becomes one
/// source per reaction, sampling positions with density
/// ``S(rho) * R * |tau|`` (the emission density times the volume element of
/// the shape) and emitting isotropically with the Ballabio spectrum at the
/// shell's emission-weighted ion temperature, so D-D and D-T neutrons keep
/// their own spectra. Strengths are normalised to sum to 1; the absolute
/// neutron rate is set by the simulation, not here.
///
/// The density and temperature profiles are those of ``tokamak_source``
/// (Fausser et al., Fus. Eng. Des. 87 (2012) 787), with the normalised minor
/// radius they take there (the ``rho`` of a ``TokamakShape``) replaced by
/// this shape's ``rho``, so the pedestal is placed by ``pedestal_rho``. Plot
/// them with ``tokamak_ion_density`` / ``tokamak_ion_temperature`` using
/// ``minor_radius=1``, ``pedestal_radius=pedestal_rho`` and ``r=rho``.
///
/// Args:
///     shape: The plasma shape.
///     mode: Confinement mode, one of 'L', 'H' or 'A'.
///     ion_density_centre: Ion density on the magnetic axis (m^-3).
///     ion_density_peaking_factor: Ion density peaking factor (dimensionless).
///     ion_density_pedestal: Ion density at the pedestal (m^-3).
///     ion_density_separatrix: Ion density at the boundary (m^-3).
///     ion_temperature_centre: Ion temperature on the magnetic axis (eV).
///     ion_temperature_peaking_factor: Ion temperature peaking factor
///             (dimensionless, alpha_T in the reference).
///     ion_temperature_beta: Ion temperature beta exponent (dimensionless,
///             beta_T in the reference).
///     ion_temperature_pedestal: Ion temperature at the pedestal (eV).
///     ion_temperature_separatrix: Ion temperature at the boundary (eV).
///     pedestal_rho: ``rho`` of the pedestal, between 0 and 1.
///     fuel: Fuel species as keys ('D', 'T') and atom fractions as values,
///             which must sum to 1. Defaults to 50:50 D-T.
///
/// Returns:
///     list[NeutronSource]: One source per (radial shell, reaction).
///
/// Raises:
///     ValueError: If a parameter is NaN or infinite, out of range (for
///             example ``pedestal_rho`` outside 0 to 1), the fuel is not D
///             and/or T, or the plasma is too cold to make neutrons.
///
/// Examples:
///     >>> import yamc
///     >>> sources = yamc.sources.stellarator_source(
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
///     ...     pedestal_rho=0.8,
///     ... )  # doctest: +SKIP
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    name = "stellarator_source",
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
        pedestal_rho,
        fuel = None,
    )
)]
pub fn py_stellarator_source(
    shape: PyStellaratorShape,
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
    pedestal_rho: f64,
    fuel: Option<HashMap<String, f64>>,
) -> PyResult<Vec<PyNeutronSource>> {
    let plasma = StellaratorPlasma {
        shape: shape.inner,
        profiles: PlasmaProfiles {
            mode: ConfinementMode::parse(mode).map_err(value_error)?,
            ion_density_centre,
            ion_density_peaking_factor,
            ion_density_pedestal,
            ion_density_separatrix,
            ion_temperature_centre,
            ion_temperature_peaking_factor,
            ion_temperature_beta,
            ion_temperature_pedestal,
            ion_temperature_separatrix,
            pedestal: pedestal_rho,
        },
        fuel: parse_fuel(fuel)?,
    };
    Ok(plasma
        .sources()
        .map_err(value_error)?
        .into_iter()
        .map(|source| PyNeutronSource {
            inner: ParticleSource::Neutron(source),
        })
        .collect())
}
