//! `TransmutationResults`: the per-timestep inventories a transmutation
//! produces, whether it was driven by transport or by a supplied spectrum.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::{BTreeMap, HashMap};
use yani_transmute::{Estimate, LineEstimate, TransmutationResults};

use crate::material::PyMaterial;

/// A quantity's value, and the spread the nuclear-data ensemble puts on it.
///
/// ``nominal`` is the unperturbed run, and is present whether or not
/// uncertainty was asked for. ``mean`` and ``std_dev`` are ``None`` below two
/// replicas: a spread over fewer than two samples is unmeasured, not zero, and
/// reporting it as zero would read as a quantity known exactly.
///
/// A quantity a decay photon intensity enters (contact dose, the photon
/// spectrum, and the decay heat through its gamma part) has a range rather
/// than one spread when the ``"decay_photon_lines"`` source is on, because
/// the decay data do not state how a nuclide's photon intensities are
/// correlated. ``std_dev`` is the lower end, every unstated correlation taken
/// as zero, and ``std_dev_correlated`` the upper end, every one taken as one;
/// ``std_dev_range`` gives both. For any other quantity, or with the source
/// off, the two are equal.
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
    ///
    /// The lower end of ``std_dev_range``: where the ``"decay_photon_lines"``
    /// source is drawn, the correlations the decay data leave unstated
    /// between a nuclide's photon intensities are taken as zero, which is the
    /// evaluation read literally.
    #[getter]
    fn std_dev(&self) -> Option<f64> {
        self.inner.std_dev
    }

    /// The ensemble's sample standard deviation with those correlations taken
    /// as one, or None below two replicas: within each nuclide the lines of a
    /// spectrum, the spectrum's normalisation and its lines, and its gamma and
    /// x-ray spectra all move together. The upper end of ``std_dev_range``.
    ///
    /// Evaluated on the same inventories as ``std_dev``, so the two differ by
    /// the line data alone. Equal to ``std_dev`` for activity, or when the
    /// ``"decay_photon_lines"`` source is not drawn.
    #[getter]
    fn std_dev_correlated(&self) -> Option<f64> {
        self.inner.std_dev_correlated
    }

    /// ``(std_dev, std_dev_correlated)``, the range every non-negative
    /// correlation between a nuclide's photon intensities gives, or None below
    /// two replicas.
    ///
    /// Negative correlations are not considered: what the intensities leave
    /// unstated is a shared normalisation, which moves every line it scales
    /// the same way and cannot anticorrelate them.
    #[getter]
    fn std_dev_range(&self) -> Option<(f64, f64)> {
        Some((self.inner.std_dev?, self.inner.std_dev_correlated?))
    }

    /// ``std_dev`` as a fraction of ``nominal``, or None if either is absent.
    #[getter]
    fn relative_std_dev(&self) -> Option<f64> {
        self.inner.relative_std_dev()
    }

    /// ``std_dev_correlated`` as a fraction of ``nominal``, or None if either
    /// is absent.
    #[getter]
    fn relative_std_dev_correlated(&self) -> Option<f64> {
        self.inner.relative_std_dev_correlated()
    }

    /// The standard error of ``std_dev``: how far another ensemble of the
    /// same size could put it, or None below four replicas.
    ///
    /// It allows for a heavy tail, ``Var(s^2) = s^4 (2/(n-1) + kappa/n)`` with
    /// ``kappa`` the sample excess kurtosis, so a lognormal-tailed quantity
    /// reads as less settled than a Gaussian one at the same replica count.
    #[getter]
    fn std_dev_standard_error(&self) -> Option<f64> {
        self.inner.std_dev_standard_error
    }

    /// How many replicas the ensemble held.
    #[getter]
    fn replicas(&self) -> usize {
        self.inner.replicas
    }

    fn __repr__(&self) -> String {
        let correlated = match (self.inner.std_dev, self.inner.std_dev_correlated) {
            (Some(low), Some(high)) if high != low => format!(", std_dev_correlated={high:.4e}"),
            _ => String::new(),
        };
        match (self.inner.std_dev, self.inner.std_dev_standard_error) {
            (Some(sigma), Some(se)) => format!(
                "Estimate(nominal={:.4e}, std_dev={:.4e} +/- {:.2e}{correlated}, replicas={})",
                self.inner.nominal, sigma, se, self.inner.replicas
            ),
            (Some(sigma), None) => format!(
                "Estimate(nominal={:.4e}, std_dev={:.4e}{correlated}, replicas={})",
                self.inner.nominal, sigma, self.inner.replicas
            ),
            (None, _) => format!(
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

/// One decay-photon line, with the ensemble's spread on its emission rate and
/// on the energy it is emitted at.
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
    /// Nominal line energy [eV], the one the evaluation states. Lines are
    /// matched across replicas on it.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// The mean energy the emitting replicas drew for this line [eV], or None
    /// below two of them. Equal to ``energy`` unless the
    /// ``"decay_photon_lines"`` source drew line energies.
    #[getter]
    fn energy_mean(&self) -> Option<f64> {
        self.inner.energy_estimate.mean
    }

    /// The sample standard deviation of the drawn energy over the emitting
    /// replicas [eV], or None below two of them. Zero unless the
    /// ``"decay_photon_lines"`` source drew line energies.
    #[getter]
    fn energy_std_dev(&self) -> Option<f64> {
        self.inner.energy_estimate.std_dev
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

    /// The ensemble's sample standard deviation [photons/s], or None below
    /// two replicas.
    ///
    /// The lower end of ``std_dev_range``: where the ``"decay_photon_lines"``
    /// source is drawn, the correlations the decay data leave unstated
    /// between a nuclide's photon intensities are taken as zero, which is the
    /// evaluation read literally.
    #[getter]
    fn std_dev(&self) -> Option<f64> {
        self.inner.estimate.std_dev
    }

    /// The ensemble's sample standard deviation with those correlations taken
    /// as one, or None below two replicas: within each nuclide the lines of a
    /// spectrum, the spectrum's normalisation and its lines, and its gamma and
    /// x-ray spectra all move together. The upper end of ``std_dev_range``.
    ///
    /// Evaluated on the same inventories as ``std_dev``, so the two differ by
    /// the line data alone. Equal to ``std_dev`` when the
    /// ``"decay_photon_lines"`` source is not drawn.
    #[getter]
    fn std_dev_correlated(&self) -> Option<f64> {
        self.inner.estimate.std_dev_correlated
    }

    /// ``(std_dev, std_dev_correlated)``, the range every non-negative
    /// correlation between a nuclide's photon intensities gives, or None below
    /// two replicas.
    ///
    /// Negative correlations are not considered: what the intensities leave
    /// unstated is a shared normalisation, which moves every line it scales
    /// the same way and cannot anticorrelate them.
    #[getter]
    fn std_dev_range(&self) -> Option<(f64, f64)> {
        Some((
            self.inner.estimate.std_dev?,
            self.inner.estimate.std_dev_correlated?,
        ))
    }

    /// ``std_dev`` as a fraction of ``nominal``, or None if either is absent.
    #[getter]
    fn relative_std_dev(&self) -> Option<f64> {
        self.inner.estimate.relative_std_dev()
    }

    /// ``std_dev_correlated`` as a fraction of ``nominal``, or None if either
    /// is absent.
    #[getter]
    fn relative_std_dev_correlated(&self) -> Option<f64> {
        self.inner.estimate.relative_std_dev_correlated()
    }

    /// The standard error of ``std_dev``, or None below four replicas (see
    /// ``Estimate.std_dev_standard_error``).
    #[getter]
    fn std_dev_standard_error(&self) -> Option<f64> {
        self.inner.estimate.std_dev_standard_error
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

    /// The rate each step drove a material at, one per timestep, zero for a
    /// cooldown.
    ///
    /// A spectrum solve (``Material.transmute``, ``transmute``) gives each
    /// material's own flux magnitude in n/cm^2/s, which differs between
    /// materials given their own schedules. ``Model.simulate_transmutation``
    /// gives the source strength in n/s, the same for every material.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///
    /// Returns:
    ///     List of rates, or None if the material is not in the results.
    fn get_source_rates(&self, material_id: u32) -> Option<Vec<f64>> {
        self.inner.get_source_rates(material_id).map(|r| r.to_vec())
    }

    /// Where a nuclide's uncertainty at one step comes from.
    ///
    /// Present when the run was asked for it with
    /// ``DataUncertainty(attribution=True)``, and ``None`` otherwise. A dict:
    ///
    /// - ``variance``: the total, resampled, the square of
    ///   ``get_nuclide_uncertainty``;
    /// - ``by_source``: each source alone, resampled the same way, so this
    ///   says how much is statistical and how much is each kind of nuclear
    ///   data. The sources are independent and these sum to the total;
    /// - ``unattributed``: what that sum leaves, interaction and sampling
    ///   noise, small when the attribution holds;
    /// - ``contributors``: first order, a list of ``(source, nuclide,
    ///   reaction, variance)``, largest reach first. Within the cross sections
    ///   a nuclide's whole evaluation has ``reaction`` of ``None`` and each
    ///   channel alone names it; a half-life has ``None``. A decay branching
    ///   contributor is a two-mode parent's one degree of freedom, with
    ///   ``reaction`` of ``None``. It says which evaluation to look at; the
    ///   total is the resampled one.
    /// - ``linearity``: how well that first order explains the replicas, per
    ///   source and as ``"all"`` for every source together. Each replica's
    ///   first-order prediction is the nominal plus every sensitivity times
    ///   that replica's own change in its input, so no extra solve is made.
    ///   Per entry: ``r2`` (squared correlation between replicas and
    ///   predictions), ``residual_share`` (the share of the variance first
    ///   order does not account for), ``by_contributor`` (``{source:
    ///   {nuclide: r2}}``, each contributor's term alone), ``ranking_agrees``
    ///   (whether the first-order top contributor is also the best
    ///   correlated), and ``flagged`` (``residual_share`` above 0.1, or the
    ///   ranking disagrees: read the contributors with care). First order has
    ///   terms only for ``cross_sections``, ``half_life`` and
    ///   ``decay_branching``. ``None`` for every other source
    ///   (``flux_spectrum``, ``statistical``, ``decay_energy``,
    ///   ``decay_photon_lines``, ``fission_yield``), and for ``"all"`` when
    ///   any applied source is one of those; absent for a nuclide with no
    ///   spread. The default source set applies several of them, so with the
    ///   defaults ``"all"`` is ``None``: restrict ``DataUncertainty(sources=...)``
    ///   to ``cross_sections``, ``half_life`` and ``decay_branching`` (or a
    ///   subset) for an ``"all"`` entry.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name.
    ///     step: As in ``get_nuclide_uncertainty``: 0 is the initial
    ///         composition, which carries none.
    fn get_uncertainty_breakdown<'py>(
        &self,
        py: Python<'py>,
        material_id: u32,
        nuclide: &str,
        step: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(b) = self.inner.uncertainty_breakdown(material_id, nuclide, step) else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("variance", b.variance)?;
        let by = PyDict::new(py);
        for (name, v) in &b.by_source {
            by.set_item(name, v)?;
        }
        d.set_item("by_source", by)?;
        d.set_item("unattributed", b.unattributed)?;
        d.set_item("contributors", b.contributors)?;
        let linearity = PyDict::new(py);
        for (source, l) in &b.linearity {
            match l {
                None => linearity.set_item(source, py.None())?,
                Some(l) => {
                    let e = PyDict::new(py);
                    e.set_item("r2", l.r2)?;
                    e.set_item("residual_share", l.residual_share)?;
                    let mut by: std::collections::BTreeMap<
                        &str,
                        std::collections::BTreeMap<&str, f64>,
                    > = std::collections::BTreeMap::new();
                    for ((s, n), r2) in &l.by_contributor {
                        by.entry(s.as_str()).or_default().insert(n.as_str(), *r2);
                    }
                    e.set_item("by_contributor", by)?;
                    e.set_item("ranking_agrees", l.ranking_agrees)?;
                    e.set_item("flagged", l.flagged)?;
                    linearity.set_item(source, e)?;
                }
            }
        }
        d.set_item("linearity", linearity)?;
        Ok(Some(d))
    }

    /// The statistical uncertainty of each transport-tallied reaction rate
    /// at one step.
    ///
    /// Present for ``Model.simulate_transmutation`` run with
    /// ``data_uncertainty`` including the ``"statistical"`` source, and
    /// ``None`` otherwise. Each entry is ``(nuclide, reaction, target, rate,
    /// std_dev)`` in 1/s per atom: a reaction total has ``target`` of ``None``,
    /// an isomeric partial names its final state. The rates are the tally's,
    /// scaled by the step's source rate, exactly as the step's solve used them
    /// before the branching fold.
    ///
    /// The rates are correlated, having been scored by the same histories,
    /// and the inventory sigmas are computed with those correlations. These
    /// standard deviations alone do not carry them.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Schedule step index, as in ``get_reaction_rates``.
    fn get_reaction_rate_uncertainty(
        &self,
        material_id: u32,
        step: usize,
    ) -> Option<Vec<(String, String, Option<String>, f64, f64)>> {
        let covariance = self.inner.rate_covariance.get(&material_id)?;
        let rate = *self.inner.get_source_rates(material_id)?.get(step)?;
        Some(
            (0..covariance.len())
                .map(|i| {
                    let label = &covariance.labels[i];
                    (
                        label.nuclide.clone(),
                        label.kind.clone(),
                        label.target.clone(),
                        covariance.rates[i] * rate,
                        covariance.std_dev(i) * rate,
                    )
                })
                .collect(),
        )
    }

    /// How much of the multigroup collapse work was shared, or ``None`` for a
    /// transport-coupled solve, which does none.
    ///
    /// A dict with ``performed``, the collapses actually run, and
    /// ``requested``, one per distinct spectrum per material. Materials with
    /// the same spectrum, composition, temperature and shielding collapse to
    /// the same rates and share one, so ``performed`` below ``requested`` is
    /// the saving ``transmute`` made over solving them one at a time.
    #[getter]
    fn collapse_reuse<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(reuse) = self.inner.collapse_reuse else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("performed", reuse.performed)?;
        d.set_item("requested", reuse.requested)?;
        Ok(Some(d))
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
    ///     ``get_data_uncertainty_info(material_id)["no_covariance_data"]``,
    ///     which lists exactly
    ///     those nuclides.
    fn get_nuclide_uncertainty(&self, material_id: u32, nuclide: &str, step: usize) -> Option<f64> {
        self.inner
            .get_nuclide_uncertainty(material_id, nuclide, step)
    }

    /// Get the standard error of ``get_nuclide_uncertainty``: how far another
    /// ensemble of the same size could put that sigma.
    ///
    /// It allows for a heavy tail, through the replicas' sample kurtosis, so a
    /// lognormal-tailed density reads as less settled than a Gaussian one at
    /// the same replica count.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     nuclide: Nuclide name.
    ///     step: As in ``get_nuclide_uncertainty``.
    ///
    /// Returns:
    ///     The standard error [atoms/barn-cm], or None if the transmutation was
    ///     run without ``data_uncertainty`` or with fewer than four replicas.
    ///     Step 0 reports 0.0.
    fn get_nuclide_uncertainty_standard_error(
        &self,
        material_id: u32,
        nuclide: &str,
        step: usize,
    ) -> Option<f64> {
        self.inner
            .get_nuclide_uncertainty_standard_error(material_id, nuclide, step)
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
    ///     step: Timestep index (0 = initial composition). The composition at
    ///         step 0 is an input, but each replica evaluates it with its own
    ///         sampled half-lives, so the activity at step 0 has a spread
    ///         whenever the ``"half_life"`` source is sampled.
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
    /// With the ``"decay_photon_lines"`` source on, each replica's gamma decay
    /// energy E_EM follows its drawn photon lines and continua rather than an
    /// independent ``"decay_energy"`` draw, so its gamma heat and its contact
    /// dose come from the same draw of one evaluation. E_EM moves by the drawn
    /// change in the photon energy per decay (each line's energy times its
    /// intensity, plus each continuum's energy integral), and the part of E_EM
    /// the tabulated spectra do not carry is held at nominal. The beta and
    /// alpha parts keep their ``"decay_energy"`` draws. The heat then has a
    /// range, ``Estimate.std_dev`` to ``Estimate.std_dev_correlated``, from the
    /// photon intensities' unstated correlations.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition). As in
    ///         ``get_activity_uncertainty``, step 0 has a spread whenever the
    ///         ``"half_life"``, ``"decay_energy"`` or ``"decay_photon_lines"``
    ///         source is sampled.
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
    /// The band is the spread of the replicas' inventories (each with its own
    /// half-lives when the ``"half_life"`` source is on), and of their decay
    /// photon line intensities and energies and continuum normalisations when
    /// the ``"decay_photon_lines"`` source is on. The photon attenuation
    /// (XCOM), air energy absorption (NIST SRD 126), ICRP-116 dose
    /// coefficients and the build-up factor are held at their nominal values
    /// and contribute nothing to it.
    ///
    /// With the ``"decay_photon_lines"`` source on, the band is a range:
    /// ``Estimate.std_dev`` takes each nuclide's photon intensities as
    /// independent where the decay data state no correlation, and
    /// ``Estimate.std_dev_correlated`` as fully correlated (the lines of a
    /// spectrum, its normalisation and lines, and its gamma and x-ray
    /// spectra). ENDF/B-VIII.1 folds each spectrum's normalisation sigma into
    /// every line's, so a multi-line emitter's range there is wide;
    /// ``get_data_uncertainty_info`` names those spectra under
    /// ``decay_photon_spectra_folded``.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition). As in
    ///         ``get_activity_uncertainty``, step 0 has a spread whenever the
    ///         ``"half_life"`` or ``"decay_photon_lines"`` source is sampled.
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
    /// emitted it, which is what the zero-fill would otherwise hide. Lines
    /// only, as there: a photon continuum is not a line and is not reported
    /// here.
    ///
    /// The band is the spread of the replicas' inventories (each with its own
    /// half-lives when the ``"half_life"`` source is on), and of each line's
    /// intensity per decay when the ``"decay_photon_lines"`` source is on.
    /// That source draws each line's energy too, so lines are matched across
    /// replicas on their nominal energy, and ``LineEstimate.energy_std_dev``
    /// gives the spread of the energy drawn. A line's rate spread is a range,
    /// ``LineEstimate.std_dev`` to ``LineEstimate.std_dev_correlated``, for the
    /// reason ``get_contact_dose_uncertainty`` gives.
    ///
    ///     >>> lines = results.get_decay_photon_spectrum_uncertainty(mid, step)
    ///     >>> [(l.energy, l.nominal, l.std_dev) for l in lines[:2]]
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition). As in
    ///         ``get_activity_uncertainty``, step 0 has a spread whenever the
    ///         ``"half_life"`` or ``"decay_photon_lines"`` source is sampled.
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

    /// Hydrogen and helium gas production in appm, at every time point.
    ///
    /// appm is gas atoms per million **initial** atoms of the material, so the
    /// denominator stays fixed as the material transmutes. The gas is what the
    /// inventory already holds: H1, H2, H3, He3 and He4 emitted by reactions
    /// and by decays, so tritium decaying to He3 during a cooldown shows up as
    /// He3 there.
    ///
    ///     >>> gas = results.get_gas_production(material_id=mid)
    ///     >>> gas["He4"][-1], gas["H"][-1]
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     produced (bool): Subtract the gas the material started with (water,
    ///         polymers, lithium compounds), the default, so index 0 is zero
    ///         and each value is what the schedule made by then. A nuclide
    ///         consumed faster than it is made reads negative, as H1 in water
    ///         can through H1(n,gamma)H2. ``False`` gives the gas present,
    ///         starting inventory included.
    ///
    /// Returns:
    ///     dict[str, list[float]] | None: appm keyed ``"H1"``, ``"H2"``,
    ///     ``"H3"``, ``"He3"``, ``"He4"`` and the totals ``"H"`` (H1 + H2 + H3)
    ///     and ``"He"`` (He3 + He4). Each list is parallel to ``times``, as
    ///     ``get_nuclide_evolution`` is: index 0 is the initial composition,
    ///     index i is after step i. None if the material is not in the results.
    ///
    /// Raises:
    ///     ValueError: if the chain the solve used has no entry for one of the
    ///         five gas nuclides. The solve follows an emitted particle only
    ///         when the chain has it, so that gas was dropped and a zero would
    ///         be wrong rather than measured. The message names the missing
    ///         nuclides.
    #[pyo3(signature = (material_id, *, produced=true))]
    fn get_gas_production(
        &self,
        material_id: u32,
        produced: bool,
    ) -> PyResult<Option<BTreeMap<String, Vec<f64>>>> {
        self.inner
            .gas_production(material_id, produced)
            .map_err(PyValueError::new_err)
    }

    /// Gas production in appm at one timestep, with the nuclear-data spread
    /// on it.
    ///
    /// See ``get_gas_production`` for the quantity. Evaluated on every
    /// replica's inventory against the one initial inventory, which is an
    /// input and the same in each, and the totals ``"H"`` and ``"He"`` are
    /// summed within a replica before the spread is taken, as
    /// ``get_activity_uncertainty`` does.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Timestep index (0 = initial composition, which has no spread).
    ///     produced (bool): As in ``get_gas_production``.
    ///
    /// Returns:
    ///     dict[str, Estimate] | None: keyed as ``get_gas_production``; None if
    ///     the transmutation was run without ``data_uncertainty``.
    ///
    /// Raises:
    ///     ValueError: as ``get_gas_production``, or if there is no such step.
    #[pyo3(signature = (material_id, step, *, produced=true))]
    fn get_gas_production_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        produced: bool,
    ) -> PyResult<Option<BTreeMap<String, PyEstimate>>> {
        Ok(self
            .inner
            .gas_production_uncertainty(material_id, step, produced)
            .map_err(PyValueError::new_err)?
            .map(|by_key| {
                by_key
                    .into_iter()
                    .map(|(key, estimate)| (key, PyEstimate::from(estimate)))
                    .collect()
            }))
    }

    /// What the nuclear-data uncertainty covered for one material, and what it
    /// did not.
    ///
    /// ``None`` when the transmutation was run without ``data_uncertainty``, or
    /// the material is not in the results.
    /// Otherwise a dict whose job is to make gaps visible rather than let them
    /// read as confidence:
    ///
    /// - ``perturbed`` / ``no_covariance_data``: which nuclides had usable
    ///   MF=33 covariance and which had none.
    /// - ``rate_fraction_covered_total``: the per-channel shares below,
    ///   averaged with each channel weighted by the production it drove (the
    ///   rate this run used times parent density): the share of the
    ///   production driven from energies where a covariance states a nonzero
    ///   variance. ``None`` on a decay-only schedule; on a transport run,
    ///   where the shares are of the dilute rate over the tally spectrum and
    ///   the covered share of the tallied production is not computed; and
    ///   under the ``1/E`` within-group weight (Rust API only), whose shares
    ///   are not exact where a covariance edge cuts a group. ``None`` too when
    ///   a channel is listed in ``partials_above_rate`` or
    ///   ``partials_below_rate``, whose rate is not the one its share is of:
    ///   the ``(n,n')`` of a nuclide with a metastable, whose rate is the
    ///   MF=10 production of the metastables while its covariance is MT 4's.
    ///   Read this before any sigma here. It is a different and much sharper
    ///   question than how many nuclides carry MF=33: an evaluation can state
    ///   covariance for every isotope in the material and none for the
    ///   channel making the product of interest, and the count then reads as
    ///   full coverage while the ensemble perturbs almost nothing.
    /// - ``rate_fraction_covered``: per nuclide and channel, the share of the
    ///   reaction rate from energies where the evaluation states a nonzero
    ///   variance for it, the rate being the dilute one on a dilute run and
    ///   the shielded one on a self-shielded run. Below one means part of the
    ///   rate carries no stated uncertainty and dilutes the sigma; on a
    ///   transport run it is a share of the dilute rate over the tally
    ///   spectrum, and the dilution applied differs from it. An
    ///   interval the covariance grid spans with a variance of zero counts as
    ///   uncovered: ENDF/B-VIII.1 W186 ``(n,gamma)`` states zero from 1e-5 eV
    ///   to 10 keV, where nearly all of its capture rate is. Every consumed
    ///   self-covariance block counts where it states a nonzero variance,
    ///   relative (LB=1 to 6), absolute (LB=0) and short-range (LB=8) alike.
    ///   Exact under the default flat within-group weight; under the ``1/E``
    ///   weight (Rust API only) the rate of a group a covariance edge cuts is
    ///   split by energy width, not lethargy, so the share is off there.
    /// - ``partials_above_rate``: per nuclide and channel, where the partial
    ///   rates the covariance was weighted with, zero variance intervals
    ///   included, add up to more than the rate it was divided by, their
    ///   ratio to it. Each entry is a channel whose sigma is overstated. Three
    ///   known causes: a tallied rate on a transport run, which the transport
    ///   can self-shield within its bins, against partials weighted flat
    ///   within each bin; a grafted ``(n,n')``, whose rate is the metastables'
    ///   MF=10 production while the partials are MT 4's; and the ``1/E``
    ///   within-group weight (Rust API only) with a covariance edge inside a
    ///   group. The share in ``rate_fraction_covered`` is of the fold's own
    ///   rate, not of the listed rate, and under the ``1/E`` weight it is off
    ///   as well wherever an edge cuts a group. On a channel derived through
    ///   an NC block, the partials of the reactions the block names are
    ///   checked the same way, and the check is also that they add up to the
    ///   one it derives over the block's range; a sum above it lands here.
    /// - ``partials_below_rate``: keyed the same way, where a covariance grid
    ///   spans the whole flux range and its partial rates add up to less than
    ///   the rate, their ratio to it: a channel whose sigma is understated.
    ///   The ``1/E`` weight gives one for a reaction falling with energy when
    ///   a covariance edge cuts a group. A grid that stops short of the flux
    ///   range cannot be checked from below, since rate from outside it
    ///   rightly leaves its partials short. A derived channel whose NC block
    ///   names reactions adding up to less than the one it derives lands here
    ///   too: ENDF/B-VIII.1 O16 ``(n,d)`` above 20 MeV, whose cross section
    ///   holds MT 660 to 669 while the block names 650 to 659.
    /// - ``derived_opposing_uncorrelated``: keyed the same way, for a channel
    ///   derived through an NC block whose terms name two reactions with
    ///   opposite signs, each with a variance of its own, and no covariance
    ///   between them: the ``[a, b]`` pairs. The absent block is read as zero,
    ///   since ENDF-102 33.3.2 a.1 lets a tape leave a zero covariance
    ///   unstated, and with opposing signs that reading sets the sigma.
    ///   FENDL-3.2d and TENDL-2017 H2 ``(n,2n)`` is ``σ_1 - σ_2 - σ_102`` and
    ///   folds to about 22% at 14 MeV and thousands of percent near
    ///   threshold, the tape's literal statement. Counted in ``has_gaps``.
    /// - ``lumped_covariance_not_assignable``: keyed ``"Nuclide MT852"``, a
    ///   lumped reaction (MT 851-870) with several components, to their kinds
    ///   (``"MT91"`` for one that is not a channel). ENDF-102 33.2.3 states
    ///   its covariance for the sum of the components and for none of them. A
    ///   lump with one component is that component, and its covariance is
    ///   folded as the component's. A lump an LTY=0 block names is folded
    ///   through that derivation, with its cross section the sum of its
    ///   components': ENDF/B-VIII.1 and TENDL-2017 U235 and U238 MT 4 is MT 51
    ///   plus MT 851. Any other lump is listed here and not folded, since
    ///   giving the sum's covariance to a component would be an assumption:
    ///   ENDF/B-VIII.1, FENDL-3.2d and JEFF-4.0 W180 to W186 give ``(n,2n)``
    ///   only as MT 852, the sum of MT 16 and 41, listed as ``"(n,2n)"`` and
    ///   ``"(n,2np)"`` when the chain carries both, and W186 MT 854 as
    ///   ``"(n,np)"`` and ``"MT91"``. Listed where the fold reaches a
    ///   component, or the reaction holding one as a level (MT 103 for MT 600
    ///   to 649, and so on). Counted in ``has_gaps``.
    /// - ``unsupported_layouts``: covariance blocks that were present but not
    ///   consumed, counted once per spectrum, so a run over several spectra
    ///   counts the same block once for each.
    /// - ``skipped_nc``: per nuclide, NC blocks (a covariance derived from
    ///   other reactions) that could not be derived. An LTY=0 block is
    ///   derived from the NI covariances of the reactions it names, cross
    ///   blocks included, over its own energy range: ENDF/B-VIII.1 O16
    ///   ``(n,p)`` is stated only that way. Left here are LTY 1 to 4, a block
    ///   in a cross-reaction subsection, one whose list of reactions is empty
    ///   or does not match its coefficients, one whose own energy range is
    ///   empty, one naming a reaction with no cross section, and one met
    ///   only circularly.
    /// - ``skipped_cross_material``: per nuclide, blocks on a reaction the
    ///   fold reaches (a channel, or one a channel is derived from) that
    ///   correlate it with another evaluation, not consumed. A block naming
    ///   the nuclide's own MAT is its own evaluation and is folded. The
    ///   partner is not checked, so a block is counted whether or not the
    ///   evaluation it names is in the run. ``skipped_other_file``: the same
    ///   for blocks whose partner is not a cross section.
    /// - ``mirrored_disagree``: keyed ``"Nuclide (n,a) (n,b)"``, where a pair
    ///   stored in both orientations has copies that are not each other's
    ///   transpose, the largest difference relative to the largest entry.
    ///   The copy in the lower MT's section is the one folded.
    /// - ``malformed_blocks``: covariance blocks not consumed because they
    ///   break ENDF-102's rules for their layout: arrays that disagree with
    ///   their declared sizes, an LB=0 to 2 block carrying a second energy
    ///   table, an LB=3 or 4 block without one or whose tables share no
    ///   energy range, or an LB=8 variance stated between two reactions.
    /// - ``covariance_source``: where each perturbed nuclide's covariance came
    ///   from, ``"<library>, MAT <n>"``, the library being the one its data
    ///   folder records (``"unknown"`` when none is recorded).
    /// - ``covariance_warnings``: perturbed nuclides whose library documents
    ///   a problem with this covariance, each with what it says and where:
    ///   every FENDL-3.2 covariance, which its own paper says should not be
    ///   used, and the ENDF/B-VIII.1 evaluations its release paper names (Fe,
    ///   Cr covariances reused from VIII.0; Cu with no fast-range covariance;
    ///   Ta, W, Pb resolved-resonance covariance flagged as too low; the Ta181
    ///   unresolved covariance overwritten). The evaluation is still sampled as
    ///   it is; any warning makes ``has_gaps`` true.
    /// - ``covariance_repaired``: nuclides the material can populate (bounded
    ///   at or above the solver's density floor over the schedule at nominal
    ///   rates; a replica's rates can sit above them) whose evaluated cell
    ///   covariance was not positive semi-definite past round-off, with a
    ///   channel a draw can move (a positive rate on a spectrum the schedule
    ///   irradiates with). Past round-off means the correlation matrix of the
    ///   cells has an eigenvalue below ``-m * 1e-12`` (``m`` the number of
    ///   cells with a positive stated variance), or a cell is stated with a
    ///   negative variance, or a zero one and a covariance to another cell.
    ///   The correlation matrix is replaced by the nearest correlation
    ///   matrix and rescaled by the evaluated sigmas, so every cell keeps its
    ///   evaluated sigma and only correlations move (a cell stated at zero or
    ///   negative variance is held at nominal); a channel folding several
    ///   cells can still be sampled at a sigma other than its evaluation's,
    ///   either way, and any makes ``has_gaps`` true.
    ///   ``covariance_repairs`` gives one dict per repaired populated nuclide
    ///   and spectrum, including repairs no draw can move, with ``lambda_min``
    ///   (the most negative eigenvalue of the cells' correlation matrix before
    ///   the repair), ``largest_correlation_change`` and
    ///   ``correlation_frobenius_change`` (the largest and the Frobenius
    ///   change of that correlation matrix), ``cells`` (in the coupled blocks
    ///   repaired), ``held_cells``, ``converged`` and, per channel keyed by
    ///   kind,
    ///   ``evaluated_variance`` (the folded diagonal as stated, which can be
    ///   negative), ``evaluated_sigma`` (``None`` when that variance is
    ///   negative) and ``sampled_sigma``. A repair of a nuclide outside the
    ///   populated bound has no dict; ``covariance_repaired_outside_bound``
    ///   names those with a channel a draw can move. The bound holds at
    ///   nominal rates only and a replica's rates can populate them, so any
    ///   also makes ``has_gaps`` true.
    /// - ``worst_sigma_change``: the largest ``|sampled / evaluated sigma -
    ///   1|`` over the repaired channels of populated nuclides with a
    ///   positive rate on a spectrum the schedule irradiates with,
    ///   ``float('inf')`` when a repair gave a spread to a channel whose stated
    ///   variance is zero or negative. ``rate_weighted_sigma_change`` is the
    ///   weighted mean of ``|sampled / evaluated sigma - 1|`` over every
    ///   sampled channel of a populated nuclide, each weighted by its unit-flux
    ///   rate times its spectrum's fluence in the schedule times its parent's
    ///   initial density, so it covers first-generation reactions only (a
    ///   produced nuclide carries no weight), shows the decomposition's
    ///   round-off on matrices that needed no repair, and is ``None`` when no
    ///   weighted channel has an evaluated sigma. Both can be
    ///   ``float('inf')``, which strict JSON does not accept.
    /// - ``sigma_at_least_one`` / ``sigma_at_least_ten``: sampled channels of
    ///   populated nuclides with a positive rate on a spectrum the schedule
    ///   irradiates with, keyed ``"Nuclide (n,x)"``, whose folded relative
    ///   sigma as evaluated, before any repair, is at least one or ten. At that
    ///   width the answer depends on the lognormal chosen to carry the
    ///   evaluation's two moments, not on the evaluation alone.
    ///   ``sigma_at_least_one_outside_bound`` is the same for nuclides outside
    ///   the populated bound, whose wide channels a replica's draw can take
    ///   past it; the ten-or-more subset reads off its values.
    /// - ``lognormal_not_carried``: nuclides whose evaluated relative
    ///   covariance is not a lognormal's, keyed by nuclide, each with
    ///   ``cells`` (cells whose sampled sigma or correlation differs from the
    ///   evaluated one), ``largest_sigma_change`` (the largest
    ///   ``|sampled / evaluated sigma - 1|``), ``largest_correlation_change``
    ///   and ``log_space_repair`` (``None``, or a dict with the keys of a
    ///   repair above, of the log-space correlation matrix).
    ///   Two fully correlated cells with different sigmas, or an
    ///   anticorrelation with ``1 + C <= 0``, are not, and the nearest
    ///   lognormal is sampled: where the log-space covariance is not PSD its
    ///   correlation matrix is replaced by the nearest correlation matrix,
    ///   which keeps every sigma. A property of the distribution rather than a
    ///   defect of the data, so not a gap. ``flux_lognormal_not_carried`` is
    ///   the same for a stated flux covariance, keyed by spectrum index,
    ///   whose ``log_space_repair`` is always ``None``: a flux covariance's
    ///   log-space negative eigenvalues are clipped.
    /// - ``rates_sampled``: cross-section rate draws made, each read off one
    ///   draw of the nuclide's cross sections. ``rates_floored`` counts those
    ///   that came out negative and were floored at zero, which only a channel
    ///   subtracting reactions or reading an additive (absolute or
    ///   short-range) covariance can.
    /// - ``half_lives_perturbed`` / ``no_half_life_uncertainty``: with the
    ///   ``"half_life"`` source, which reachable unstable nuclides had their
    ///   half-life sampled and which state no sigma to sample from.
    ///   ``half_life_uncertainty_not_carried`` names those whose stated sigma
    ///   no draw can carry (not finite, or not finite relative to the
    ///   half-life), held at nominal and counted as a gap.
    ///   ``half_lives_sampled`` counts the draws made. Each is a lognormal
    ///   matched to the evaluation's mean and sigma, so none can go
    ///   non-positive and none is floored.
    /// - ``decay_energies_perturbed`` / ``no_decay_energy_uncertainty``: the
    ///   same for the ``"decay_energy"`` source, drawn per nuclide as a
    ///   lognormal with the stated mean and sigma, per component where the
    ///   data splits it. ``decay_energy_uncertainty_not_carried`` names those
    ///   with a sigma stated on a zero energy, or not finite, which no draw
    ///   can carry; that energy is held at nominal and counted as a gap.
    /// - ``decay_photon_lines_perturbed`` /
    ///   ``no_decay_photon_line_uncertainty``: the same for the
    ///   ``"decay_photon_lines"`` source, over the reachable unstable
    ///   nuclides with decay photon data.
    ///   ``decay_photon_line_uncertainty_not_carried`` names those with a
    ///   sigma stated on a zero value, or not finite, which no draw can carry;
    ///   that value is held at nominal and counted as a gap.
    ///   ``decay_photon_spectra_folded`` maps each perturbed nuclide with a
    ///   spectrum written the ENDF/B way (a normalisation of 1 with no sigma,
    ///   its sigma folded into every line's dRI) to the radiation of each such
    ///   spectrum (``"gamma"``, ``"xray"``). How much of those dRI the lines
    ///   share is not stated, so they are where most of the range between a
    ///   photon output's ``std_dev`` and ``std_dev_correlated`` comes from.
    /// - ``fission_yields_perturbed`` / ``no_fission_yield_uncertainty``: the
    ///   same for the ``"fission_yield"`` source, over the reachable
    ///   fissioning parents. ``fission_yield_uncertainty_not_carried`` names
    ///   those with a DY on a zero yield, or not finite, which is held while
    ///   the parent's other yields are drawn.
    ///   ``fission_yields_mapping_mismatch`` names those whose tape yields,
    ///   named and summed by the converter's rule, do not give back the yields
    ///   the solver reads; they are held rather than drawn through a mapping
    ///   their yields were not built with. Each is counted as a gap.
    /// - ``decay_branchings_perturbed``: with the ``"decay_branching"``
    ///   source, the reachable two-mode parents whose split was sampled. The
    ///   multi-mode parents held at their evaluated ratios, each a gap:
    ///   ``no_decay_branching_uncertainty`` (no mode states a sigma),
    ///   ``decay_branchings_three_or_more_modes`` (a sigma, but no stated
    ///   covariance to share it between three or more modes),
    ///   ``decay_branchings_unequal_sigmas`` (two modes stating different
    ///   sigmas) and ``decay_branchings_too_wide`` (the smaller ratio under
    ///   five sigmas). ``decay_branchings_floored`` /
    ///   ``decay_branchings_sampled`` count draws clamped to the pair's total
    ///   and draws made.
    /// - ``statistical_rates``: with the ``"statistical"`` source on a
    ///   transport run, how many tallied rates were sampled from their
    ///   covariance; ``statistical_floored`` / ``statistical_sampled`` count
    ///   draws that came out negative and were floored.
    /// - ``resonance_parameters``: per nuclide with resonance parameters
    ///   (MF=2 and MF=32) and a rate in this run, how its resonance-range
    ///   uncertainty was sampled. ``method`` is ``"parameters sampled"``
    ///   (drawn per replica and the cross sections rebuilt from them) or
    ///   ``"first-order rows"`` (the MF=32 rows of ``covariance.arrow``),
    ///   ``reason`` why the parameters were not sampled (``None`` where they
    ///   were), and ``ranges`` one sampler report per sampled range:
    ///   ``isotope`` and ``range`` indices, the ``gaussian``, ``lognormal``
    ///   and ``held`` parameter counts, ``zero_mean_widths`` (widths stated
    ///   with a zero mean and a nonzero sigma, held at zero) and
    ///   ``negative_mean_widths`` (drawn as signed), each with ``index``,
    ///   ``location``, ``quantity``, ``value`` and ``sigma``,
    ///   ``zero_variance_with_covariance``, ``unattainable_pairs``, and
    ///   ``stated_repair`` / ``transformed_repair`` (the nearest-correlation
    ///   repair of the stated and of the log-space matrix, each with
    ///   ``lambda_min``, ``frobenius_change``, ``max_change``, ``parameters``,
    ///   ``iterations`` and ``converged``, ``None`` where none was needed).
    ///   How far the draws' parameter correlations are from the evaluated
    ///   ones, in the parameters themselves after both repairs and the
    ///   lognormal transform, reads off ``largest_correlation_change`` (the
    ///   largest change of one correlation) and
    ///   ``correlation_frobenius_change``, per range and per nuclide over its
    ///   ranges; every drawn parameter keeps its evaluated mean and sigma, so
    ///   that is the whole of the difference in the first two moments. Widths
    ///   stay lognormal, so a pair no lognormal carries
    ///   (``unattainable_pairs``) is where it is largest. Empty on a transport
    ///   run, which keeps the rows.
    /// - ``not_perturbed``: every input this run held at its nominal value,
    ///   such as any MF=32 resonance-parameter covariance neither sampled nor
    ///   in the library's ``covariance.arrow``, the photon and dose data, the
    ///   material composition, any source switched off, and, where they
    ///   applied, the self-shielding correction, the flux's response to a
    ///   perturbed cross section on a transport run, and the per-branch decay
    ///   emission of a parent whose branching was drawn.
    /// - ``samples`` / ``converged``: how many replicas ran, and whether the
    ///   sigmas settled or the cap was hit.
    ///
    /// Args:
    ///     material_id: Material ID number.
    fn get_data_uncertainty_info<'py>(
        &self,
        py: Python<'py>,
        material_id: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.uncertainty_info.get(&material_id) {
            None => Ok(None),
            Some(info) => Ok(Some(crate::data_uncertainty::info_to_dict(py, info)?)),
        }
    }

    /// What the self-shielding did for one material.
    ///
    /// Present for every material of a spectrum solve, shielded or not, and
    /// ``None`` for a transport-coupled solve or a material not in the
    /// results. ``chord_cm`` of ``None`` means the run was dilute and
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
    ///
    /// Args:
    ///     material_id: Material ID number.
    fn get_self_shielding_info<'py>(
        &self,
        py: Python<'py>,
        material_id: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(info) = self.inner.shielding_info.get(&material_id) else {
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

    /// Cumulative NRT displacements per atom (dpa) over the schedule.
    ///
    /// Present when the transmute call was given
    /// ``displacement_damage=True``, and ``None`` otherwise. One value per
    /// state, aligned with ``times``: entry 0 is the initial composition and
    /// is zero, and entry ``i`` is the total after schedule step ``i - 1``.
    /// Cooldowns add nothing.
    ///
    /// For an element ``X`` it is the damage energy deposited per atom of
    /// ``X``, from ``X``'s own nuclides, converted with ``X``'s displacement
    /// threshold energy as ``0.8 * E_damage / (2 * E_d)`` (the NRT model,
    /// ASTM E521). The material total, with ``element`` omitted, is the
    /// atom-fraction-weighted sum over elements: each element's recoils are
    /// treated as slowing down among atoms of their own kind, which reduces to
    /// the elemental value for a pure element and leaves out energy transfer
    /// between elements in a cascade. MT=444 is already integrated over the
    /// recoil spectrum, so the per-recoil threshold steps of the NRT model
    /// (no displacement below ``E_d``, one up to ``2 * E_d / 0.8``) are not
    /// applied, which is the standard practice for a damage-energy cross
    /// section. The ``E_d`` used and its source are in
    /// ``get_displacement_damage_info``.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     element: Element symbol, e.g. ``"W"``, for that element's dpa;
    ///         omit it for the material total.
    ///
    /// Returns:
    ///     List of cumulative dpa, one per state, or None if damage was not
    ///     asked for or the material is not in the results.
    ///
    /// Raises:
    ///     ValueError: If ``element`` has no dpa: it is not in the material
    ///         on an irradiated step, or it is a transmutation product with no
    ///         displacement threshold energy.
    #[pyo3(signature = (material_id, element = None))]
    fn get_dpa(&self, material_id: u32, element: Option<&str>) -> PyResult<Option<Vec<f64>>> {
        let Some(damage) = self.inner.get_displacement_damage(material_id) else {
            return Ok(None);
        };
        match element {
            None => Ok(Some(damage.dpa.clone())),
            Some(el) => damage
                .element_dpa
                .get(el)
                .cloned()
                .map(Some)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "no dpa for element {el:?}; elements with dpa: {}",
                        damage
                            .element_dpa
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }),
        }
    }

    /// Cumulative damage energy deposited per atom [eV] over the schedule.
    ///
    /// The quantity dpa is computed from, kept separate so the displacement
    /// model can be changed without the data: each nuclide's MT=444
    /// damage-energy cross section [eV barn] folded against the pulse
    /// spectrum by the same collapse as the reaction rates, times the flux
    /// magnitude and the step duration, at each step's composition. Indexed
    /// as ``get_dpa``. For an element it is per atom of that element; the
    /// material total weights the elements by atom fraction, so it is per atom
    /// of the material.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     element: Element symbol for that element's damage energy; omit it
    ///         for the material total.
    ///
    /// Returns:
    ///     List of cumulative damage energy [eV per atom], one per state, or
    ///     None if damage was not asked for or the material is not in the
    ///     results.
    ///
    /// Raises:
    ///     ValueError: If ``element`` is not in the material on an irradiated
    ///         step.
    #[pyo3(signature = (material_id, element = None))]
    fn get_damage_energy(
        &self,
        material_id: u32,
        element: Option<&str>,
    ) -> PyResult<Option<Vec<f64>>> {
        let Some(damage) = self.inner.get_displacement_damage(material_id) else {
            return Ok(None);
        };
        match element {
            None => Ok(Some(damage.damage_energy.clone())),
            Some(el) => damage
                .element_damage_energy
                .get(el)
                .cloned()
                .map(Some)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "no damage energy for element {el:?}; elements present: {}",
                        damage
                            .element_damage_energy
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }),
        }
    }

    /// What the displacement damage was computed with, and what it could not
    /// count.
    ///
    /// ``None`` unless the transmute call was given
    /// ``displacement_damage=True``. A dict:
    ///
    /// - ``model``: ``"NRT"``, and ``efficiency``: ``0.8``.
    /// - ``displacement_energies``: per element with dpa,
    ///   ``{"energy": E_d in eV, "source": ...}``, where ``source`` is
    ///   ``"ASTM E521"``, ``"OECD-NEA 2015"`` (Table 2.4 of NEA/NSC/DOC(2015)9,
    ///   for an element ASTM E521 does not cover) or ``"user"``.
    /// - ``without_damage_energy``: nuclides present on an irradiated step
    ///   whose data has no MT=444, each with the largest atom fraction it
    ///   reached there. Their damage energy is not counted, so a large entry
    ///   here means the totals are low by about that share. Only
    ///   transmutation products can appear: a nuclide of the starting
    ///   composition without MT=444 is refused.
    /// - ``without_displacement_energy``: transmutation-product elements with
    ///   no displacement threshold energy (hydrogen and helium, typically),
    ///   each with the largest atom fraction reached. Their damage energy is
    ///   counted in ``get_damage_energy``; they add nothing to ``get_dpa``.
    ///   Pass ``displacement_energies`` to include them.
    ///
    /// Args:
    ///     material_id: Material ID number.
    fn get_displacement_damage_info<'py>(
        &self,
        py: Python<'py>,
        material_id: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(damage) = self.inner.get_displacement_damage(material_id) else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("model", "NRT")?;
        d.set_item("efficiency", yamc_element::displacement::NRT_EFFICIENCY)?;
        let energies = PyDict::new(py);
        for (el, ed) in &damage.displacement_energies {
            let entry = PyDict::new(py);
            entry.set_item("energy", ed.energy_ev)?;
            entry.set_item("source", ed.source.label())?;
            energies.set_item(el, entry)?;
        }
        d.set_item("displacement_energies", energies)?;
        let without = PyDict::new(py);
        for (name, fraction) in &damage.without_damage_energy {
            without.set_item(name, fraction)?;
        }
        d.set_item("without_damage_energy", without)?;
        let without = PyDict::new(py);
        for (el, fraction) in &damage.without_displacement_energy {
            without.set_item(el, fraction)?;
        }
        d.set_item("without_displacement_energy", without)?;
        Ok(Some(d))
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
    ///         ``get_source_rates``. This is one less than the ``step`` the
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
    /// Channels come back ordered by production, the channel's rate times its
    /// parent's atom density at the start of the step. A rate on its own is per
    /// atom of the parent, so ordering on it promotes whatever sits on a trace
    /// isotope: on an FNS tungsten foil ``W180(n,2n)`` has the highest per-atom
    /// rate of any channel in the foil, and W180 is 0.12% of it, so by what it
    /// made the channel falls to fifth, two orders of magnitude below the
    /// ``W186(n,2n)`` carrying most of that foil's decay heat.
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Schedule step index, the same index ``get_reaction_rates``
    ///         takes, which is one less than the composition getters' step.
    ///
    /// Returns:
    ///     list[dict] | None: one entry per splitting channel, most produced
    ///     first, each with ``parent``, ``reaction``, ``production`` and
    ///     ``split``. ``split`` is [(target, fraction)] summing to one and
    ///     ordered with the largest share first. Empty for a decay-only step,
    ///     and None if the material or the step is unknown. A parent absent
    ///     from the step's starting composition has production 0.0 and sorts
    ///     last rather than being dropped.
    ///
    /// Examples:
    ///     >>> results.get_isomeric_branching(material_id=1, step=0)[0]
    ///     {'parent': 'W186', 'reaction': '(n,2n)', 'production': 9.35e-14,
    ///      'split': [('W185_m1', 0.535), ('W185', 0.465)]}
    fn get_isomeric_branching(
        &self,
        py: Python<'_>,
        material_id: u32,
        step: usize,
    ) -> Option<Py<PyAny>> {
        let channels = self.inner.get_isomeric_branching(material_id, step)?;
        let out = PyList::empty(py);
        for channel in &channels {
            let row = PyDict::new(py);
            row.set_item("parent", channel.parent.as_str()).unwrap();
            row.set_item("reaction", channel.reaction.as_str()).unwrap();
            row.set_item("production", channel.production).unwrap();
            let split = PyList::empty(py);
            for (target, fraction) in &channel.split {
                split.append((target.as_str(), fraction)).unwrap();
            }
            row.set_item("split", split).unwrap();
            out.append(row).unwrap();
        }
        Some(out.into_any().unbind())
    }

    /// What the isomeric-branching rule did over one step's spectrum.
    ///
    /// The branching evaluation gives the split and the cross-section library
    /// the total. How a list's values are read is decided by how the
    /// evaluation gives them, which the converter records: a complete MF=10
    /// list (its ground state listed) and every MF=9 list are shares of the
    /// transport total, applied at each energy; an MF=10 list of isomers only,
    /// and every ``(n,n')`` list, are absolute productions, the ground state
    /// taking the rest. This says, per channel, which of those applied and how
    /// much of the parent's removal rate rests on anything the evaluation
    /// does not give.
    ///
    /// MT=5, ``(n,anything)``, is the ``(n,X)`` reaction: its residuals, read
    /// from the reaction library's MF=6 MT=5, are shares of the MT=5 total
    /// at each energy (``file`` 6, ``representation`` ``"share"``), and its
    /// light particles H1 to He4 are their multiplicities times that total
    /// (``representation`` ``"multiplicity"``, each state's ``share`` the
    /// multiplicity folded over the spectrum, which can exceed one).
    ///
    /// A run refuses when the clipped or held production of its channels,
    /// each parent weighted by its density, is more than 0.1% of the
    /// material's neutron removal rate; each channel's ``clipped_share`` and
    /// ``extrapolated_share`` are of its own parent's removal. A multiplicity above what the target's nucleons allow is
    /// clipped like any other impossible value. ``unmodelled_mt5`` lists the
    /// parents whose MT=5 residuals the chain does not model, with the reason;
    /// on a reactions subsection that carries MT=5, a run refuses when those
    /// of the material's own nuclides carry more than 0.1% of the material's
    /// removal rate (one written before MT=5 was carried is reported only).
    ///
    /// Args:
    ///     material_id: Material ID number.
    ///     step: Schedule step index, as ``get_reaction_rates`` takes it.
    ///
    /// Returns:
    ///     dict | None: ``channels``, ``dropped`` and ``unmodelled_mt5``, or
    ///     None if the material or the step is unknown. Each channel has
    ///     ``parent``, ``reaction``, ``mt``, ``file`` (6, 9 or 10),
    ///     ``representation`` (``"share"``, ``"absolute"`` or
    ///     ``"multiplicity"``), ``complete``,
    ///     ``completeness_source``, ``denominator``, ``states`` (each with
    ///     ``target``, ``lfs``, ``level_route``, ``level_energy_difference``
    ///     and ``share``, its share of the reaction), ``removal_share`` (the
    ///     reaction's share of the parent's removal rate), ``clipped_share``
    ///     and ``extrapolated_share`` (of the same removal rate),
    ///     ``own_total_excess`` (``(energy_ev, ratio)`` where the listed values
    ///     most exceed the evaluation's own total, or None) and
    ///     ``normalisation``. Each dropped channel has ``parent``,
    ///     ``reaction``, ``target``, ``reason`` and ``removal_share`` (None
    ///     where it cannot be folded). ``unmodelled_mt5`` is
    ///     ``[(nuclide, share, reason)]``, MT=5's share of the removal rate
    ///     of each parent whose MT=5 residuals are not modelled, largest
    ///     first. Empty for a decay-only step.
    ///
    /// Examples:
    ///     >>> report = results.get_branching_report(material_id=1, step=0)
    ///     >>> report["channels"][0]["representation"]
    ///     'absolute'
    fn get_branching_report<'py>(
        &self,
        py: Python<'py>,
        material_id: u32,
        step: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(report) = self.inner.get_branching_report(material_id, step) else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        let channels = PyList::empty(py);
        for c in &report.channels {
            let d = PyDict::new(py);
            d.set_item("parent", c.parent.as_str())?;
            d.set_item("reaction", c.reaction.as_str())?;
            d.set_item("mt", c.mt)?;
            d.set_item("file", c.file)?;
            d.set_item("representation", c.representation.as_str())?;
            d.set_item("complete", c.complete)?;
            d.set_item("completeness_source", c.completeness_source.as_str())?;
            d.set_item("denominator", c.denominator.as_str())?;
            let states = PyList::empty(py);
            for s in &c.states {
                let sd = PyDict::new(py);
                sd.set_item("target", s.target.as_str())?;
                sd.set_item("lfs", s.lfs.clone())?;
                sd.set_item("level_route", s.level_route.clone())?;
                sd.set_item("level_energy_difference", s.level_energy_difference.clone())?;
                sd.set_item("share", s.share)?;
                states.append(sd)?;
            }
            d.set_item("states", states)?;
            d.set_item("removal_share", c.removal_share)?;
            d.set_item("clipped_share", c.clipped_share)?;
            d.set_item("extrapolated_share", c.extrapolated_share)?;
            d.set_item("own_total_excess", c.own_total_excess)?;
            d.set_item("normalisation", c.normalisation.clone())?;
            channels.append(d)?;
        }
        out.set_item("channels", channels)?;
        let dropped = PyList::empty(py);
        for c in &report.dropped {
            let d = PyDict::new(py);
            d.set_item("parent", c.parent.as_str())?;
            d.set_item("reaction", c.reaction.as_str())?;
            d.set_item("target", c.target.clone())?;
            d.set_item("reason", c.reason.as_str())?;
            d.set_item("removal_share", c.removal_share)?;
            dropped.append(d)?;
        }
        out.set_item("dropped", dropped)?;
        let mt5: Vec<(String, f64, String)> = report
            .unmodelled_mt5
            .iter()
            .map(|u| (u.nuclide.clone(), u.share, u.reason.clone()))
            .collect();
        out.set_item("unmodelled_mt5", mt5)?;
        Ok(Some(out))
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
