//! `TransmutationResults`: the per-timestep inventories a transmutation
//! produces, whether it was driven by transport or by a supplied spectrum.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use yani_transmute::{Estimate, LineEstimate, TransmutationResults};

use crate::material::PyMaterial;

/// A quantity's value, and the spread the nuclear-data ensemble puts on it.
///
/// ``nominal`` is the unperturbed run, and is present whether or not
/// uncertainty was asked for. ``mean`` and ``std_dev`` are ``None`` below two
/// replicas: a spread over fewer than two samples is unmeasured, not zero, and
/// reporting it as zero would read as a quantity known exactly.
#[gen_stub_pyclass]
#[pyclass(name = "Estimate", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyEstimate {
    inner: Estimate,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEstimate {
    /// The quantity from the unperturbed inventory.
    #[getter]
    fn nominal(&self) -> f64 {
        self.inner.nominal
    }

    /// The ensemble mean, or None below two replicas.
    ///
    /// Worth comparing against ``nominal``: activity and decay heat are linear
    /// in the atom densities, so the two agree to within the sampling error,
    /// and a gap between them says the perturbation is biased rather than
    /// merely wide.
    #[getter]
    fn mean(&self) -> Option<f64> {
        self.inner.mean
    }

    /// The ensemble's sample standard deviation, or None below two replicas.
    #[getter]
    fn std_dev(&self) -> Option<f64> {
        self.inner.std_dev
    }

    /// ``std_dev`` as a fraction of ``nominal``, or None if either is absent.
    #[getter]
    fn relative_std_dev(&self) -> Option<f64> {
        self.inner.relative_std_dev()
    }

    /// How many replicas the ensemble held.
    #[getter]
    fn replicas(&self) -> usize {
        self.inner.replicas
    }

    fn __repr__(&self) -> String {
        match self.inner.std_dev {
            Some(sigma) => format!(
                "Estimate(nominal={:.4e}, std_dev={:.4e}, replicas={})",
                self.inner.nominal, sigma, self.inner.replicas
            ),
            None => format!(
                "Estimate(nominal={:.4e}, std_dev=None, replicas={})",
                self.inner.nominal, self.inner.replicas
            ),
        }
    }
}

impl From<Estimate> for PyEstimate {
    fn from(inner: Estimate) -> Self {
        Self { inner }
    }
}

/// One decay-photon line, with the ensemble's spread on its emission rate.
///
/// The set of lines is not the same in every replica: a nuclide that falls
/// below the density floor in one draw takes its lines out of that draw. The
/// spectrum is reported over the union, a line missing from a replica counts as
/// a zero in it, and ``emitting`` says how many replicas emitted it at all --
/// which is the difference between a line that is dim and a line that is
/// sometimes not there.
#[gen_stub_pyclass]
#[pyclass(name = "LineEstimate", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLineEstimate {
    inner: LineEstimate,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLineEstimate {
    /// Line energy [eV].
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// Emission rate from the unperturbed inventory [photons/s].
    #[getter]
    fn nominal(&self) -> f64 {
        self.inner.estimate.nominal
    }

    /// The ensemble mean [photons/s], or None below two replicas.
    #[getter]
    fn mean(&self) -> Option<f64> {
        self.inner.estimate.mean
    }

    /// The ensemble's sample standard deviation, or None below two replicas.
    #[getter]
    fn std_dev(&self) -> Option<f64> {
        self.inner.estimate.std_dev
    }

    /// ``std_dev`` as a fraction of ``nominal``, or None if either is absent.
    #[getter]
    fn relative_std_dev(&self) -> Option<f64> {
        self.inner.estimate.relative_std_dev()
    }

    /// How many replicas the ensemble held.
    #[getter]
    fn replicas(&self) -> usize {
        self.inner.estimate.replicas
    }

    /// Replicas that emitted this line at a positive rate.
    ///
    /// Below ``replicas`` when the emitter dropped out of some draws. A line
    /// with ``emitting`` far below ``replicas`` has a mean that is mostly
    /// zeros, and its spread says more about whether the line is there than
    /// about how bright it is.
    #[getter]
    fn emitting(&self) -> usize {
        self.inner.emitting
    }

    fn __repr__(&self) -> String {
        format!(
            "LineEstimate(energy={:.4e}, nominal={:.4e}, emitting={}/{})",
            self.inner.energy,
            self.inner.estimate.nominal,
            self.inner.emitting,
            self.inner.estimate.replicas
        )
    }
}

impl From<LineEstimate> for PyLineEstimate {
    fn from(inner: LineEstimate) -> Self {
        Self { inner }
    }
}

/// Which derived quantity an accessor is after.
#[derive(Clone, Copy)]
enum Derived {
    Activity,
    DecayHeat,
    ContactDose {
        quantity: yani_decay::DoseQuantity,
        build_up: f64,
    },
}

/// Results from a transmutation calculation.
///
/// Contains material compositions at each timestep for all transmuted materials.
/// Index 0 is the initial composition, index i is after timestep[i-1].
#[gen_stub_pyclass]
#[pyclass(name = "TransmutationResults", from_py_object)]
#[derive(Clone)]
pub struct PyTransmutationResults {
    pub inner: TransmutationResults,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTransmutationResults {
    /// Cumulative times [s] including t=0.
    #[getter]
    fn times(&self) -> Vec<f64> {
        self.inner.times.clone()
    }

    /// Timesteps used [s].
    #[getter]
    fn timesteps(&self) -> Vec<f64> {
        self.inner.timesteps.clone()
    }

    /// Source rates used [n/cm^2/s].
    #[getter]
    fn source_rates(&self) -> Vec<f64> {
        self.inner.source_rates.clone()
    }

    /// Number of transmutation steps.
    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    /// Get the evolution of a specific nuclide over all timesteps.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name (e.g., "Co60").
    ///
    /// Returns:
    ///     List of atom densities [atoms/barn-cm] at each time point
    ///     (index 0 = initial, index i = after step i).
    ///     Returns None if material_id not found.
    fn get_nuclide_evolution(&self, material_id: u32, nuclide: &str) -> Option<Vec<f64>> {
        self.inner.get_nuclide_evolution(material_id, nuclide)
    }

    /// Get nuclide density at a specific timestep.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name (e.g., "U235").
    ///     step: Timestep index (0 = initial).
    ///
    /// Returns:
    ///     Atom density [atoms/barn-cm] or None if not found.
    fn get_nuclide_density(&self, material_id: u32, nuclide: &str, step: usize) -> Option<f64> {
        self.inner.get_nuclide_density(material_id, nuclide, step)
    }

    /// Get the nuclear-data standard deviation on a nuclide density.
    ///
    /// The spread over an ensemble of solves with the activation cross sections
    /// resampled from their MF=33 covariance. In the same units as
    /// ``get_nuclide_density``, so the pair reads as ``mean ± sigma``.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name (e.g., "Mn56").
    ///     step: Timestep index (0 = initial).
    ///
    /// Returns:
    ///     Standard deviation [atoms/barn-cm], or None if the transmutation was
    ///     run without ``data_uncertainty``. Step 0 is the initial composition
    ///     and always reports 0.0: it is an input, and perturbing cross sections
    ///     does not move it.
    ///
    ///     A nuclide whose evaluation carries no covariance also reports 0.0.
    ///     That is not a claim of certainty -- check
    ///     ``data_uncertainty_info["no_covariance_data"]``, which lists exactly
    ///     those nuclides.
    fn get_nuclide_uncertainty(&self, material_id: u32, nuclide: &str, step: usize) -> Option<f64> {
        self.inner
            .get_nuclide_uncertainty(material_id, nuclide, step)
    }

    /// Get the standard deviation of a nuclide's density at every timestep.
    ///
    /// Parallel to ``get_nuclide_evolution``, leading zero included, so the two
    /// can be zipped without an index correction.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name.
    ///
    /// Returns:
    ///     List of standard deviations [atoms/barn-cm], or None if the
    ///     transmutation was run without ``data_uncertainty``.
    fn get_nuclide_uncertainty_evolution(
        &self,
        material_id: u32,
        nuclide: &str,
    ) -> Option<Vec<f64>> {
        self.inner
            .get_nuclide_uncertainty_evolution(material_id, nuclide)
    }

    /// Every replica's inventory at one timestep.
    ///
    /// For a quantity that has to be evaluated per sample rather than from the
    /// mean. Activity, decay heat and the decay-photon spectrum are all
    /// functions of a whole inventory, so evaluating one of them once per entry
    /// here and taking the spread keeps the correlations between nuclides.
    /// Taking the mean inventory and evaluating once throws them away, and
    /// summing per-nuclide sigmas in quadrature assumes an independence that the
    /// resampling exists precisely to avoid assuming.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial, which has no ensemble).
    ///
    /// Returns:
    ///     List of ``{nuclide: density}`` dicts, one per replica, or None if the
    ///     transmutation was run without ``data_uncertainty``.
    fn get_uncertainty_inventories(
        &self,
        material_id: u32,
        step: usize,
    ) -> Option<Vec<HashMap<String, f64>>> {
        self.inner
            .uncertainty_inventories(material_id, step)
            .map(|v| v.into_iter().cloned().collect())
    }

    /// Activity at one timestep [Bq], with the nuclear-data spread on it.
    ///
    /// Evaluated once per replica and summed within each, so the correlations
    /// between nuclides survive. Doing it any other way gives a plausible
    /// number that is wrong in a specific direction: evaluating from the mean
    /// inventory gives no spread at all, and adding the per-nuclide sigmas in
    /// quadrature double-counts a variance that partly cancels, since every
    /// Mn56 atom in an irradiated iron foil came out of an Fe56 atom.
    ///
    /// The volume is taken from the stored step material, so the nominal value
    /// and every replica are scaled by the same one.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition, whose spread is zero
    ///         because it is an input rather than a result).
    ///     by_nuclide (bool): Return a ``dict[str, Estimate]`` instead of one
    ///         ``Estimate`` for the total. These do not add up to the total in
    ///         quadrature, and are not meant to.
    ///
    /// Returns:
    ///     Estimate | dict[str, Estimate] | None: None if the transmutation
    ///     was run without ``data_uncertainty``.
    ///
    /// Raises:
    ///     ValueError: if the material has no ``volume`` in cm^3, which a
    ///         quantity in becquerel needs.
    #[pyo3(signature = (material_id, step, *, by_nuclide=false))]
    fn get_activity_uncertainty(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
        by_nuclide: bool,
    ) -> PyResult<Py<PyAny>> {
        self.derived(py, material_id, step, by_nuclide, Derived::Activity)
    }

    /// Decay heat at one timestep [W], with the nuclear-data spread on it.
    ///
    /// See ``get_activity_uncertainty``: same ensemble, same rule, and the
    /// quantity the FNS decay-heat benchmarks are written against. A calculated
    /// band beside the measured one is what turns a C/E into a statement about
    /// whether the disagreement is larger than the data uncertainty allows.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition).
    ///     by_nuclide (bool): Return a ``dict[str, Estimate]`` of W by nuclide
    ///         instead of one ``Estimate`` for the total.
    ///
    /// Returns:
    ///     Estimate | dict[str, Estimate] | None: None if the transmutation
    ///     was run without ``data_uncertainty``.
    ///
    /// Raises:
    ///     ValueError: if the material has no ``volume`` in cm^3.
    #[pyo3(signature = (material_id, step, *, by_nuclide=false))]
    fn get_decay_heat_uncertainty(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
        by_nuclide: bool,
    ) -> PyResult<Py<PyAny>> {
        self.derived(py, material_id, step, by_nuclide, Derived::DecayHeat)
    }

    /// Contact dose rate at one timestep, with the nuclear-data spread on it.
    ///
    /// The one derived quantity here that is **not** linear in the atom
    /// densities. The material attenuates its own decay photons, so the
    /// emitters sit in the numerator and the whole inventory in the
    /// denominator: a replica that makes more of an emitter also absorbs more
    /// of it. Scaling the nominal dose by the density spread would report the
    /// activity's uncertainty, which is wrong by the whole value; evaluating
    /// per replica gets the cancellation for free.
    ///
    /// Needs no ``volume``, unlike the other three, because the estimate takes
    /// the material for a half-space.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition).
    ///     dose_quantity (str): ``'absorbed-air'`` (Gy/h, the default) or
    ///         ``'effective'`` (Sv/h), as ``Material.contact_dose`` takes them.
    ///     build_up (float): Build-up factor, a plain multiplier on the answer.
    ///     by_nuclide (bool): Return a ``dict[str, Estimate]`` instead of one
    ///         ``Estimate`` for the total.
    ///
    /// Returns:
    ///     Estimate | dict[str, Estimate] | None: None if the transmutation
    ///     was run without ``data_uncertainty``.
    #[pyo3(signature = (material_id, step, *, dose_quantity="absorbed-air", build_up=2.0, by_nuclide=false))]
    fn get_contact_dose_uncertainty(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
        dose_quantity: &str,
        build_up: f64,
        by_nuclide: bool,
    ) -> PyResult<Py<PyAny>> {
        let quantity = match dose_quantity {
            "absorbed-air" => yani_decay::DoseQuantity::AbsorbedAir,
            "effective" => yani_decay::DoseQuantity::Effective,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dose_quantity must be 'absorbed-air' or 'effective', got '{other}'"
                )))
            }
        };
        self.derived(
            py,
            material_id,
            step,
            by_nuclide,
            Derived::ContactDose { quantity, build_up },
        )
    }

    /// The decay-photon spectrum at one timestep, line by line, with the
    /// nuclear-data spread on each emission rate.
    ///
    /// Ascending in energy and coincident lines summed, exactly as
    /// ``Material.decay_photon_spectrum`` returns them, and over the union of
    /// the lines the nominal run and every replica emit. A line a replica does
    /// not emit counts as a zero in it -- the same rule the densities follow,
    /// and the only one under which two lines' spreads are taken over the same
    /// sample -- and ``LineEstimate.emitting`` reports how many replicas
    /// emitted it, which is what the zero-fill would otherwise hide.
    ///
    ///     >>> lines = results.get_decay_photon_spectrum_uncertainty(mid, step)
    ///     >>> [(l.energy, l.nominal, l.std_dev) for l in lines[:2]]
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition).
    ///
    /// Returns:
    ///     list[LineEstimate] | None: None if the transmutation was run without
    ///     ``data_uncertainty``.
    ///
    /// Raises:
    ///     ValueError: if the material has no ``volume`` in cm^3.
    fn get_decay_photon_spectrum_uncertainty(
        &self,
        material_id: u32,
        step: usize,
    ) -> PyResult<Option<Vec<PyLineEstimate>>> {
        let chain = crate::distribution::resolve_chain()?.chain;
        Ok(self
            .inner
            .photon_spectrum_uncertainty(material_id, step, &chain)
            .map_err(PyValueError::new_err)?
            .map(|lines| lines.into_iter().map(PyLineEstimate::from).collect()))
    }

    /// What the nuclear-data uncertainty covered, and what it did not.
    ///
    /// ``None`` when the transmutation was run without ``data_uncertainty``.
    /// Otherwise a dict whose job is to make gaps visible rather than let them
    /// read as confidence:
    ///
    /// - ``perturbed`` / ``no_covariance_data``: which nuclides had usable
    ///   MF=33 covariance and which had none.
    /// - ``rate_fraction_covered_total``: the share of the production this run
    ///   drove that a covariance actually spans, weighted by rate and by parent
    ///   density. Read this before any sigma here. It is a different and much
    ///   sharper question than how many nuclides carry MF=33: an evaluation can
    ///   state covariance for every isotope in the material and none for the
    ///   channel making the product of interest, and the count then reads as
    ///   full coverage while the ensemble perturbs almost nothing.
    /// - ``rate_fraction_covered``: per nuclide and channel, the share of the
    ///   reaction rate the covariance grid actually spans. Below one means part
    ///   of the rate carries no stated uncertainty and the sigma is diluted.
    /// - ``skipped_nc``, ``skipped_cross_material``, ``unsupported_layouts``:
    ///   covariance blocks that were present but not consumed.
    /// - ``matrices_clipped`` / ``worst_relative_clip``: evaluations whose
    ///   covariance was not positive semi-definite and had to be repaired.
    /// - ``rates_floored`` / ``rates_sampled``: samples that went negative and
    ///   were truncated at zero, which biases the mean upward when common.
    /// - ``not_perturbed``: the sources this does not propagate at all.
    /// - ``samples`` / ``converged``: how many replicas ran, and whether the
    ///   sigmas settled or the cap was hit.
    #[getter]
    fn data_uncertainty_info<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.inner.uncertainty_info {
            None => Ok(None),
            Some(info) => Ok(Some(crate::data_uncertainty::info_to_dict(py, info)?)),
        }
    }

    /// What the self-shielding did, or ``None`` if the run was not shielded.
    ///
    /// Always present. ``chord_cm`` of ``None`` means the run was dilute and
    /// nothing was corrected; otherwise it says how: the ``method`` and
    /// ``chord_cm`` used, which nuclides were ``shielded``, which were
    /// ``not_shielded`` and why, and ``strongest_factor``, the smallest factor
    /// any group average was multiplied by. A run reporting ``1.0`` there
    /// shielded nothing in practice, which is a different statement from not
    /// having tried.
    ///
    /// ``would_shield`` is the other direction, and is filled only on a dilute
    /// run: nuclides whose own resonances are structured enough to have
    /// suppressed a reaction, each mapped to how strongly.
    ///
    /// It is an indicator, not a correction and not a bound. There is no
    /// geometry in it: the weight is ``1 / (1 + N * sigma_x)`` on that one
    /// reaction and that nuclide's own density, which fixes the background at
    /// 1/cm, while the correction proper uses ``1 / chord_cm`` against the
    /// material's total. So the number scales with how strongly a nuclide's own
    /// resonances could bite without predicting what a given lump would see,
    /// and it can sit either side of the real factor: a lump thinner than a
    /// centimetre of chord shields less, and a material whose other nuclides
    /// dominate the total at the resonance dips the flux further than this one
    /// reaction can express.
    ///
    /// On the FNS tungsten foil, ``would_shield`` reads 0.634 for W186 while
    /// the slowing-down correction on the same foil and spectrum saturates at
    /// 0.730 from a millimetre of chord upward. Read it as "this answer may be
    /// high, and this is a resonance absorber", which is the warning a dilute
    /// run should carry rather than silence. Read the size of the effect off a
    /// shielded run, by asking for one.
    #[getter]
    fn self_shielding_info<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(info) = &self.inner.shielding_info else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("method", info.method.clone())?;
        d.set_item("chord_cm", info.chord_cm)?;
        let mut shielded = info.shielded.clone();
        shielded.sort();
        shielded.dedup();
        d.set_item("shielded", shielded)?;
        let not = PyDict::new(py);
        for (name, why) in info.not_shielded.iter() {
            not.set_item(name, why)?;
        }
        d.set_item("not_shielded", not)?;
        d.set_item("strongest_factor", info.strongest_factor)?;
        let would = PyDict::new(py);
        for (name, factor) in info.would_shield.iter() {
            would.set_item(name, factor)?;
        }
        d.set_item("would_shield", would)?;
        Ok(Some(d))
    }

    /// Get material composition at a specific timestep as a dict.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial).
    ///
    /// Returns:
    ///     Dict of nuclide -> density [atoms/barn-cm], or None if not found.
    fn get_material_nuclides(&self, material_id: u32, step: usize) -> Option<HashMap<String, f64>> {
        let mat = self.inner.get_material(material_id, step)?;
        Some(mat.nuclides.clone())
    }

    /// Get material composition at a specific timestep.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial).
    ///
    /// Returns:
    ///     Material object with composition at that timestep, or None if not found.
    fn get_material(&self, material_id: u32, step: usize) -> Option<PyMaterial> {
        let mat = self.inner.get_material(material_id, step)?;
        Some(PyMaterial {
            internal: mat.clone(),
        })
    }

    /// Composition at the end of each schedule step, without the initial one.
    ///
    /// Indexed as ``timesteps``, so entry ``i`` is the state at the end of step
    /// ``i`` and pairs with ``get_reaction_rates(material_id, i)``. This is the
    /// list ``Material.transmute`` used to return on its own.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///
    /// Returns:
    ///     List[Material]: One material per step; empty if the material is
    ///     unknown or was never stepped.
    fn step_materials(&self, material_id: u32) -> Vec<PyMaterial> {
        self.inner
            .step_materials(material_id)
            .iter()
            .map(|m| PyMaterial {
                internal: m.clone(),
            })
            .collect()
    }

    /// Final composition for a material, after the last step.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///
    /// Returns:
    ///     Material at the end of the schedule, or None if the material is
    ///     unknown.
    fn get_final_material(&self, material_id: u32) -> Option<PyMaterial> {
        let mat = self.inner.get_final_material(material_id)?;
        Some(PyMaterial {
            internal: mat.clone(),
        })
    }

    /// Per-edge reaction rates the solve drove one step with.
    ///
    /// The rate of each individual production edge, which the solve computes to
    /// build the burnup matrix and used to throw away. Without it a consumer
    /// can enumerate the routes into a product but not weight them, so a route
    /// carrying 99.9% and one carrying 0.1% look alike.
    ///
    /// Both methods report the same quantity in the same shape.
    /// ``method="coupled"`` takes the rates from that step's transport tallies
    /// and ``method="independent"`` from the single transport's multigroup fold
    /// scaled by the step's source rate; either way the edge rate is that rate
    /// times the branching of the chain the step was solved with, isomeric
    /// overlay included.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Schedule step index, the same index as ``timesteps`` and
    ///         ``source_rates``. This is one less than the ``step`` the
    ///         composition getters take, where 0 is the initial composition.
    ///
    /// Returns:
    ///     dict[str, dict[str, list[tuple[str | None, float]]]] | None: parent
    ///     nuclide -> reaction kind -> [(target, rate [1/s])]. A ``target`` of
    ///     ``None`` is a channel naming no single product, which in practice
    ///     means fission, whose products come from the yields rather than from
    ///     an edge. Empty for a decay-only step, and ``None`` if the material
    ///     or the step is unknown.
    ///
    /// Examples:
    ///     >>> rates = results.get_reaction_rates(material_id=1, step=0)
    ///     >>> rates["Fe56"]["(n,gamma)"]
    ///     [('Fe57', 1.7e-09)]
    ///     >>> # share of Mn56 production arriving down each route
    ///     >>> into_mn56 = [
    ///     ...     (parent, kind, rate)
    ///     ...     for parent, kinds in rates.items()
    ///     ...     for kind, edges in kinds.items()
    ///     ...     for target, rate in edges
    ///     ...     if target == "Mn56"
    ///     ... ]
    fn get_reaction_rates(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
    ) -> Option<Py<PyAny>> {
        let edges = self.inner.get_reaction_rates(material_id, step)?;
        let out = PyDict::new(py);
        for (parent, kinds) in edges {
            let per_kind = PyDict::new(py);
            for (kind, targets) in kinds {
                let list = PyList::empty(py);
                for (target, rate) in targets {
                    list.append((target.as_deref(), rate)).unwrap();
                }
                per_kind.set_item(kind.as_str(), list).unwrap();
            }
            out.set_item(parent.as_str(), per_kind).unwrap();
        }
        Some(out.into_any().unbind())
    }

    /// One channel's reaction rate over one step, resolved onto the groups of
    /// the spectrum that drove it.
    ///
    /// ``get_reaction_rates`` answers with one number per edge, already
    /// collapsed against the whole spectrum. That number cannot say which part
    /// of the spectrum made it, and where a cross section spans decades the two
    /// readings are different physics: an effective ``W186(n,gamma)`` of 57 mb
    /// against a spectrum 89% of which sits in 12-16 MeV and 0.7% of which is
    /// below 100 keV is either a fast-capture rate or a resonance-region rate,
    /// and only the breakdown says which. A disagreement can then be pinned on
    /// resonance processing rather than guessed at, and a covariance grid that
    /// stops short of the spectrum can be checked against where the rate
    /// actually is.
    ///
    /// The per-group entries sum to the collapsed rate for the same channel, to
    /// floating-point rounding, because both come from the same walk of the
    /// same cross sections with the same self-shielding weighting.
    ///
    /// Nothing is stored for this. The breakdown is re-derived from the
    /// spectrum when asked for, which is one reaction over the group structure;
    /// keeping it for every channel would be tens of megabytes on a 709-group
    /// structure, and a run that never asks should not pay that.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: The parent the reaction happens on, e.g. ``"W186"``.
    ///     kind: The reaction, spelled as the chain spells it, e.g.
    ///         ``"(n,gamma)"``.
    ///     step: Schedule step index, as ``get_reaction_rates`` takes it.
    ///
    /// Returns:
    ///     dict | None: ``boundaries``, the group boundaries [eV] ascending and
    ///     one longer than the rates, and ``rates``, each group's contribution
    ///     to the rate [1/s]. None when the material, the step, the nuclide or
    ///     the channel is unknown; for a step that drove no flux, whether by
    ///     being a cooldown or by carrying a zero rate; for ``(n,n')``, whose
    ///     rate comes from the branching overlay's partials rather than from a
    ///     group average; and for a transport-coupled solve, which scores its
    ///     rates at the collision energy and keeps no group structure to
    ///     resolve them onto.
    ///
    /// Examples:
    ///     >>> spectrum = results.get_reaction_rate_spectrum(
    ///     ...     material_id=1, nuclide="W186", kind="(n,gamma)", step=0
    ///     ... )
    ///     >>> sum(spectrum["rates"])  # the collapsed rate
    ///     1.9e-09
    ///     >>> # where in energy that rate came from
    ///     >>> below_100_keV = sum(
    ///     ...     r
    ///     ...     for lo, r in zip(spectrum["boundaries"], spectrum["rates"])
    ///     ...     if lo < 1.0e5
    ///     ... )
    fn get_reaction_rate_spectrum(
        &self,
        py: Python<'_>,
        material_id: u32,
        nuclide: &str,
        kind: &str,
        step: usize,
    ) -> Option<Py<PyAny>> {
        let spectrum = self
            .inner
            .get_reaction_rate_spectrum(material_id, nuclide, kind, step)?;
        let out = PyDict::new(py);
        out.set_item("boundaries", spectrum.boundaries).unwrap();
        out.set_item("rates", spectrum.rates).unwrap();
        Some(out.into_any().unbind())
    }

    /// Flux-weighted isomeric branching at one step.
    ///
    /// Which state a reaction leaves its product in is energy dependent, so the
    /// single number describing a spectrum is the branching collapsed against
    /// it, and that number exists only inside a solve. The chain file carries
    /// the unweighted ratios, and where the branching overlay supplies the
    /// split it carries a placeholder instead: on TENDL-2025 the dominant
    /// tungsten channel reads ``W186 (n,2n) -> W185 1.000000`` and
    /// ``W186 (n,2n) -> W185_m1 0.000000`` in the file, and the overlay
    /// replaces both at solve time with roughly the 54/46 that spectrum gives.
    /// On a foil whose decay heat comes from an isomer, that difference is the
    /// whole answer.
    ///
    /// Only channels landing in more than one final state appear. A channel
    /// with one product has no branching to report, and listing it at 1.0
    /// buries the ones that do; ``get_reaction_rates`` has the unnormalised
    /// edges if the rest is wanted.
    ///
    /// This is the number that says whether a disagreement belongs to a cross
    /// section or to a branching ratio, which are different data and different
    /// fixes.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Schedule step index, the same index ``get_reaction_rates``
    ///         takes, which is one less than the composition getters' step.
    ///
    /// Returns:
    ///     dict[str, dict[str, list[tuple[str, float]]]] | None: parent ->
    ///     reaction kind -> [(target, fraction)], fractions summing to one and
    ///     ordered with the largest first. Empty for a decay-only step, and
    ///     None if the material or the step is unknown.
    ///
    /// Examples:
    ///     >>> results.get_isomeric_branching(material_id=1, step=0)["W186"]
    ///     {'(n,2n)': [('W185_m1', 0.535), ('W185', 0.465)]}
    fn get_isomeric_branching(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
    ) -> Option<Py<PyAny>> {
        let split = self.inner.get_isomeric_branching(material_id, step)?;
        let out = PyDict::new(py);
        for (parent, kinds) in &split {
            let per_kind = PyDict::new(py);
            for (kind, targets) in kinds {
                let list = PyList::empty(py);
                for (target, fraction) in targets {
                    list.append((target.as_str(), fraction)).unwrap();
                }
                per_kind.set_item(kind.as_str(), list).unwrap();
            }
            out.set_item(parent.as_str(), per_kind).unwrap();
        }
        Some(out.into_any().unbind())
    }

    /// Every way a product was made over one step, weighted by how much of it
    /// arrived down each.
    ///
    /// Enumerating routes is easy and weighting them is not. Asked what makes
    /// W187, a chain answers ``Os190(n,a)`` and ``Ir192(n,npa)`` as readily as
    /// ``W186(n,gamma)``, and nothing in a tungsten foil is osmium. So the walk
    /// starts from the nuclides the material actually began with, and each
    /// route is weighted by what its own reactions drove rather than by
    /// anything read off the chain.
    ///
    /// A route is ``reaction_depth`` neutron reactions followed by up to
    /// ``decay_depth`` decays. Its weight is the atoms it starts from, times
    /// each reaction step's per-atom production over the step, times the
    /// branching of every decay it passes through: a route through a 1% branch
    /// delivers 1% of what the reaction made. Reaction steps carry the step
    /// duration, so a two-reaction route is in the same units as a one-reaction
    /// route and comes out smaller by roughly a factor of the fluence, which is
    /// the honest answer for an irradiation short enough that products barely
    /// burn.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     product: The nuclide whose production is being explained.
    ///     step: Schedule step index, as ``get_reaction_rates`` takes it.
    ///     reaction_depth (int): Neutron reactions a route may use. 1 is what
    ///         the published pathway tables carry.
    ///     decay_depth (int): Decays a route may follow after them.
    ///
    /// Returns:
    ///     list[dict] | None: one entry per route, largest share first, each
    ///     with ``route`` (the string the published tables print, e.g.
    ///     ``"W186(n,2n)W185_m1(IT)W185"``), ``steps`` (the same thing as
    ///     ``(parent, kind, target)`` triples), ``share`` (of this product's
    ///     production, summing to one) and ``production`` (atoms per barn-cm,
    ///     before normalising, so that 100% of almost nothing is
    ///     distinguishable from 100% of the inventory). Empty when nothing in
    ///     this material makes the product, which is an answer. None if the
    ///     material, the step, or the chain is unknown.
    ///
    /// Examples:
    ///     >>> for r in results.get_production_routes(1, "W185", 0):
    ///     ...     print(f"{r['route']:<32} {r['share']:.1%}")
    ///     W186(n,2n)W185_m1(IT)W185        53.0%
    ///     W186(n,2n)W185                   46.8%
    #[pyo3(signature = (material_id, product, step, reaction_depth=1, decay_depth=3))]
    fn get_production_routes(
        &self,
        py: Python<'_>,
        material_id: u32,
        product: &str,
        step: usize,
        reaction_depth: usize,
        decay_depth: usize,
    ) -> Option<Py<PyAny>> {
        let routes = self.inner.get_production_routes(
            material_id,
            product,
            step,
            reaction_depth,
            decay_depth,
        )?;
        let out = PyList::empty(py);
        for r in &routes {
            let d = PyDict::new(py);
            d.set_item("route", r.text()).unwrap();
            let steps = PyList::empty(py);
            for (parent, kind, target) in &r.steps {
                steps
                    .append((parent.as_str(), kind.as_str(), target.as_str()))
                    .unwrap();
            }
            d.set_item("steps", steps).unwrap();
            d.set_item("share", r.share).unwrap();
            d.set_item("production", r.production).unwrap();
            out.append(d).unwrap();
        }
        Some(out.into_any().unbind())
    }

    /// List of material IDs that were transmuted.
    #[getter]
    fn material_ids(&self) -> Vec<u32> {
        self.inner.materials.keys().copied().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TransmutationResults(steps={}, materials={})",
            self.inner.num_steps(),
            self.inner.materials.len()
        )
    }
}

impl PyTransmutationResults {
    /// Shared body of the two derived-quantity accessors.
    ///
    /// Not a `#[pymethods]` entry: it exists only so the two differ in nothing
    /// but which `yani-transmute` call they make.
    fn derived(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
        by_nuclide: bool,
        which: Derived,
    ) -> PyResult<Py<PyAny>> {
        let chain = crate::distribution::resolve_chain()?.chain;
        if by_nuclide {
            let breakdown = match which {
                Derived::Activity => {
                    self.inner
                        .activity_uncertainty_by_nuclide(material_id, step, &chain)
                }
                Derived::DecayHeat => {
                    self.inner
                        .decay_heat_uncertainty_by_nuclide(material_id, step, &chain)
                }
                Derived::ContactDose { quantity, build_up } => {
                    self.inner.contact_dose_uncertainty_by_nuclide(
                        material_id,
                        step,
                        &chain,
                        quantity,
                        build_up,
                    )
                }
            }
            .map_err(PyValueError::new_err)?;
            let Some(breakdown) = breakdown else {
                return Ok(py.None());
            };
            let out = PyDict::new(py);
            for (nuclide, estimate) in breakdown {
                out.set_item(nuclide, Py::new(py, PyEstimate::from(estimate))?)?;
            }
            Ok(out.into_any().unbind())
        } else {
            let total =
                match which {
                    Derived::Activity => self.inner.activity_uncertainty(material_id, step, &chain),
                    Derived::DecayHeat => {
                        self.inner.decay_heat_uncertainty(material_id, step, &chain)
                    }
                    Derived::ContactDose { quantity, build_up } => self
                        .inner
                        .contact_dose_uncertainty(material_id, step, &chain, quantity, build_up),
                }
                .map_err(PyValueError::new_err)?;
            match total {
                None => Ok(py.None()),
                Some(estimate) => Ok(Py::new(py, PyEstimate::from(estimate))?.into_any()),
            }
        }
    }
}
