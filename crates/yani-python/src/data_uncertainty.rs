//! `DataUncertainty`: ask a transmutation for nuclear-data uncertainty.
//!
//! Optional everywhere it appears. Without it nothing is read, folded,
//! factorized or sampled, and the inventories are bit-identical to a build that
//! never had this feature.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use yani_transmute::nearest_correlation::CorrelationRepair;
use yani_transmute::resonance_rates::ResonanceMethod;
use yani_transmute::resonance_sampling::NotedParameter;
use yani_transmute::uncertainty::{DataUncertainty, Info, Source};

/// Request nuclear-data uncertainty on a transmutation.
///
/// Pass one to :meth:`Material.transmute` and the result carries a standard
/// deviation on every nuclide density alongside the mean.
///
/// What it can cover, by source (``DataUncertainty.available_sources()``):
///
/// - ``"cross_sections"``: the activation cross sections, sampled from the
///   ENDF MF=33 covariance folded against this material's own spectrum, and
///   where the library folder carries the evaluation's resonance parameters
///   (MF=2 and MF=32, ``resonance_parameters.arrow``), the parameters
///   themselves: each replica draws them from their covariance, rebuilds the
///   resonance range's cross sections from the draw and Doppler broadens the
///   change to the temperature in use, so the resonance range's uncertainty
///   is exact in the parameters rather than first order. A nuclide whose
///   parameters cannot be sampled keeps the first-order MF=32 rows of
///   ``covariance.arrow``, and the report says which and why;
/// - ``"flux_spectrum"``: the spectrum itself, from the per-bin
///   ``flux_std_dev`` given on a ``Pulse``;
/// - ``"half_life"``: every reachable nuclide's half-life, from the decay
///   data's own standard deviation. A replica's half-lives are used in its
///   solve AND in the activity, decay heat and dose evaluated from it, so a
///   saturated activity (``lambda N = R``) is correctly insensitive to its
///   own half-life rather than inheriting the density's spread;
/// - ``"decay_branching"``: the decay branching ratios of every reachable
///   parent with exactly two modes and one stated sigma between them (both
///   state the same one, or one states it and the other is its complement),
///   whose smaller ratio is at least five sigmas from zero. One draw per
///   parent moves one mode up and the other down by the same amount, so the
///   pair's total is kept. Other multi-mode parents stay at their evaluated
///   ratios and the report names them by why;
/// - ``"statistical"``: the Monte Carlo uncertainty of transport-tallied
///   reaction rates, from their per-history covariance, drawn independently
///   for each material since their tallies are separate estimates. It
///   applies to ``Model.simulate_transmutation``, as ``"flux_spectrum"``
///   applies only to ``Material.transmute``; each call ignores the other's,
///   and the report's ``sources`` lists what actually applied;
/// - ``"decay_energy"``: each nuclide's mean decay energy, from the sigma the
///   decay data gives each recoverable-heat component (beta, gamma, alpha),
///   or the total's where it gives no split. It moves decay heat only: a decay
///   energy never enters the solve, so the inventory and activity are
///   untouched;
/// - ``"decay_photon_lines"``: each decay photon spectrum's normalisation
///   (FD for lines, FC for a continuum), one draw per spectrum common to all
///   its lines, and each line's own intensity (dRI) and energy (dER), from
///   the decay data's MT=457 sigmas. It moves the decay photon spectrum, the
///   contact dose and, through the gamma decay energy E_EM, which follows
///   each replica's drawn lines in place of a ``"decay_energy"`` draw, the
///   decay heat; no photon enters the solve. The decay data do not state how
///   a nuclide's photon intensities are correlated beyond the normalisation
///   (between the lines' dRI, between a normalisation and its lines, between
///   gamma and x-ray spectra), so each output is evaluated at both ends:
///   ``Estimate.std_dev`` takes them independent and
///   ``Estimate.std_dev_correlated`` fully correlated. A D1S dose
///   (``PulseSchedule.time_correct_tally``) draws the normalisations alone;
/// - ``"fission_yield"``: each fissioning parent's independent yields (MT=454),
///   from the DY the evaluation states on each, drawn on the tape's own
///   products and summed onto the chain's the way the converter summed the
///   nominal yields. No evaluation states a correlation between yields, so
///   each is drawn independently and the yields of a draw are not
///   renormalised. An actinide that borrows another's yields shares its draw.
///   Needs a chain that carries the evaluated yields
///   (``fission_yields/evaluated_yields.arrow``); without one every parent is
///   listed under ``no_fission_yield_uncertainty``.
///
/// Each cross-section draw is a lognormal multiplier with the covariance's
/// own mean and variance, so a sampled rate is never negative and nothing is
/// floored. The correlated Gaussian deviates are kept (a Gaussian copula), so
/// the ordering between channels is preserved, but the Pearson correlations
/// come out weaker than evaluated as the sigmas grow. Half-lives and decay
/// energies are drawn the same way, one nuclide at a time: the decay data
/// states a mean and a sigma for each and no correlation, so the draws carry
/// exactly what the evaluation states and are never negative. So are the decay
/// photon normalisations, intensities and energies, with the one correlation
/// the data does state, a spectrum's normalisation common to its lines, and
/// the ones it does not state bounded rather than assumed.
///
/// Held at their nominal values, with uncertainties of their own that this
/// does not propagate:
///
/// - the decay branching ratios ``"decay_branching"`` does not sample (three
///   or more modes, unequal sigmas, too wide to sample untruncated, or no
///   sigma), and the per-decay photon lines and decay energy of a drawn
///   parent, which follow its nominal branching;
/// - the isomeric-branching overlay from MF=9/MF=10;
/// - covariance correlating two evaluations (MAT1 naming another material),
///   covariance with a quantity that is not a cross section (XMF1 not 0 or
///   3), covariance derived from other sections by an NC block that cannot
///   be derived (LTY 1-4, or an LTY=0 block counted in ``skipped_nc``), the
///   covariance of a lumped reaction (MT=851-870) with several components
///   that no derivation names, listed in ``lumped_covariance_not_assignable``,
///   and the resonance-parameter covariance (MF=32) of a range neither
///   sampled nor in ``covariance.arrow``: a library converted before the
///   converter wrote either, or a resonance range whose formalism ``endf``
///   does not reconstruct. What is sampled is each reaction's explicit MF=33
///   blocks, the resonance parameters where the folder carries them and
///   otherwise the resonance-range blocks the converter derives from MF=32
///   and writes beside them, the blocks of a lumped reaction
///   whose one component it is, and for a reaction an LTY=0 NC block states
///   as a sum of others (ENDF/B-VIII.1 O16 (n,p) as MT 600 to 603, U235 MT 4
///   as MT 51 plus the lumped MT 851), the covariance derived from the named
///   reactions' own blocks and the cross blocks between them;
/// - the self-shielding correction, when ``self_shielding_chord`` or
///   ``self_shielding_shape`` is given: the shielded flux is built once from
///   the nominal cross sections and reused by every replica, sampled
///   resonance parameters included;
/// - on a transport run, the flux's response to a perturbed cross section:
///   there is one transport, not one per replica. The tallied values
///   themselves are still drawn by the ``"statistical"`` source;
/// - the decay photon spectrum covariance and continuum shape (MF=8 MT=457
///   LCOV, which no library states), photon attenuation (XCOM), air energy
///   absorption (NIST SRD 126), the ICRP-116 fluence-to-dose coefficients and
///   the contact-dose build-up factor;
/// - the material's composition, density, natural isotopic abundances and the
///   AME2020 atomic masses used to convert mass fractions;
/// - any source switched off with ``sources``, or with nothing to act on (a
///   spectrum given without ``flux_std_dev``). When only some of a material's
///   spectra have one, the entry is ``"flux spectrum (spectra without a sigma
///   only)"`` and ``spectra_without_flux_sigma`` gives the count. A rate
///   under a spectrum with a sigma that the flux draw cannot move is named
///   in ``flux_rates_without_terms``.
///
/// ``TransmutationResults.get_data_uncertainty_info`` lists every one of these
/// that applied to a material under ``not_perturbed``, along with any nuclide
/// whose evaluation carries no covariance, any unstable nuclide whose
/// half-life or decay energy has no stated sigma, and any whose stated
/// half-life or decay-energy sigma no draw can carry.
///
/// Args:
///     seed (int): Base seed. A given nuclide's perturbation in a given replica
///         is a pure function of ``(seed, replica, nuclide)``, so the same seed
///         reproduces the same answer regardless of replica count, iteration
///         order, or what else is in the material.
///     samples (int, optional): Fixed replica count. Leave as ``None`` (the
///         default) to let the solver add replicas until the reported standard
///         deviations stop moving. A number here bounds cost or reproduces a
///         specific run; it is not an accuracy dial.
///     sources (list[str], optional): Which inputs to perturb. ``None`` (the
///         default) means every source this build implements. Restricting it is
///         how a run isolates one contribution, so that adding a source and
///         watching the inventory sigma grow is a measurement rather than a
///         guess.
///
///         Naming a source this build cannot perturb **raises**, rather than
///         being ignored. A source that has not landed yet must not look like
///         one that contributed nothing. ``DataUncertainty.available_sources()``
///         lists what there is.
///     attribution (bool): Also say where the uncertainty comes from, read
///         with ``TransmutationResults.get_uncertainty_breakdown``. Off by
///         default because it costs further solves: one ensemble per source,
///         each source alone, and one deterministic solve per contributor. It
///         changes none of the numbers the run otherwise reports.
///
/// Examples:
///     >>> results = iron.transmute(
///     ...     schedule=sched,
///     ...     data_uncertainty=yamc.DataUncertainty(seed=42),
///     ... )
///     >>> results.get_nuclide_density(iron.id or 0, "Mn56", 1)
///     1.234e-08
///     >>> results.get_nuclide_uncertainty(iron.id or 0, "Mn56", 1)
///     8.7e-10
#[gen_stub_pyclass]
// `from_py_object` because this is taken as an ARGUMENT to
// `Material.transmute`, so it has to convert back out of Python.
#[pyclass(name = "DataUncertainty", from_py_object)]
#[derive(Clone)]
pub struct PyDataUncertainty {
    pub inner: DataUncertainty,
    /// Whether `sources` was given, rather than left to mean every source
    /// the build implements. A caller that perturbs only some sources (a
    /// transport run perturbs cross sections) refuses an explicit request for
    /// the others but takes the default as everything it can do.
    pub sources_given: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDataUncertainty {
    #[new]
    #[pyo3(signature = (seed = 1, samples = None, sources = None, attribution = false))]
    fn new(
        seed: u64,
        samples: Option<usize>,
        sources: Option<Vec<String>>,
        attribution: bool,
    ) -> PyResult<Self> {
        if samples == Some(0) {
            return Err(PyValueError::new_err(
                "samples must be at least 1; pass samples=None to let the solver \
                 choose when the standard deviations have settled",
            ));
        }
        let sources_given = sources.is_some();
        let sources = match sources {
            None => Source::IMPLEMENTED.to_vec(),
            Some(names) => {
                if names.is_empty() {
                    return Err(PyValueError::new_err(
                        "sources must name at least one source; pass sources=None \
                         for every source this build implements",
                    ));
                }
                names
                    .iter()
                    .map(|n| Source::parse(n).map_err(PyValueError::new_err))
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        Ok(Self {
            inner: DataUncertainty {
                seed,
                samples,
                sources,
                attribution,
            },
            sources_given,
        })
    }

    /// The uncertainty sources this build can perturb.
    ///
    /// Returns:
    ///     list[str]: Names accepted by the ``sources`` argument.
    #[staticmethod]
    fn available_sources() -> Vec<String> {
        Source::IMPLEMENTED
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn samples(&self) -> Option<usize> {
        self.inner.samples
    }

    /// Whether the run also says where the uncertainty comes from.
    #[getter]
    fn attribution(&self) -> bool {
        self.inner.attribution
    }

    #[getter]
    fn sources(&self) -> Vec<String> {
        self.inner
            .sources
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        let samples = match self.inner.samples {
            Some(n) => n.to_string(),
            None => "None".to_string(),
        };
        format!(
            "DataUncertainty(seed={}, samples={samples}, sources={:?})",
            self.inner.seed,
            self.sources(),
        )
    }
}

/// Render an [`Info`] as a plain dict.
///
/// A dict rather than a class because it is a report, not an interface: a
/// caller prints it, logs it, or checks one key. Every entry exists so that a
/// gap is visible, since a nuclide with no published covariance and a nuclide
/// whose covariance is genuinely zero would otherwise both read as a confident
/// zero.
pub fn info_to_dict<'py>(py: Python<'py>, info: &Info) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("samples", info.samples)?;
    d.set_item("converged", info.converged)?;
    d.set_item(
        "perturbed",
        info.perturbed.iter().cloned().collect::<Vec<_>>(),
    )?;
    d.set_item(
        "no_covariance_data",
        info.no_covariance_data.iter().cloned().collect::<Vec<_>>(),
    )?;
    let cross = PyDict::new(py);
    for (nuclide, n) in &info.skipped_cross_material {
        cross.set_item(nuclide, n)?;
    }
    d.set_item("skipped_cross_material", cross)?;
    let other_file = PyDict::new(py);
    for (nuclide, n) in &info.skipped_other_file {
        other_file.set_item(nuclide, n)?;
    }
    d.set_item("skipped_other_file", other_file)?;
    // Keyed "Nuclide (n,a) (n,b)", the two kinds in MT order, like the
    // per-channel maps below.
    let mirrored = PyDict::new(py);
    for ((nuclide, a, b), mismatch) in &info.mirrored_disagree {
        mirrored.set_item(format!("{nuclide} {a} {b}"), mismatch)?;
    }
    d.set_item("mirrored_disagree", mirrored)?;
    let nc = PyDict::new(py);
    for (nuclide, n) in &info.skipped_nc {
        nc.set_item(nuclide, n)?;
    }
    d.set_item("skipped_nc", nc)?;

    let layouts = PyDict::new(py);
    for (lb, n) in &info.unsupported_layouts {
        layouts.set_item(lb, n)?;
    }
    d.set_item("unsupported_layouts", layouts)?;
    d.set_item("malformed_blocks", info.malformed_blocks)?;

    // Keyed "Nuclide (n,gamma)" rather than by a tuple, so the dict survives
    // JSON and a glance.
    let covered = PyDict::new(py);
    for ((nuclide, kind), fraction) in &info.rate_fraction_covered {
        covered.set_item(format!("{nuclide} {kind}"), fraction)?;
    }
    d.set_item("rate_fraction_covered", covered)?;

    // The one number that says whether the sigmas above are a spread over the
    // answer or over a corner of it. `None` for a decay-only schedule, which
    // drove no production and so has no share to report, and on a transport
    // run, where the share of the tallied production is not computed.
    d.set_item(
        "rate_fraction_covered_total",
        info.rate_fraction_covered_total,
    )?;

    // Keyed like `rate_fraction_covered`. Each entry is a channel whose sigma
    // is overstated, so it is a warning and not a detail: `has_gaps` counts it.
    let above = PyDict::new(py);
    for ((nuclide, kind), ratio) in &info.partials_above_rate {
        above.set_item(format!("{nuclide} {kind}"), ratio)?;
    }
    d.set_item("partials_above_rate", above)?;
    // The same inconsistency the other way, a sigma understated.
    let below = PyDict::new(py);
    for ((nuclide, kind), ratio) in &info.partials_below_rate {
        below.set_item(format!("{nuclide} {kind}"), ratio)?;
    }
    d.set_item("partials_below_rate", below)?;
    // A derived channel whose sigma rests on reading an absent covariance
    // between two opposing terms as zero, each pair as `[a, b]`.
    let opposing = PyDict::new(py);
    for ((nuclide, kind), pairs) in &info.derived_opposing_uncorrelated {
        let pairs: Vec<[&str; 2]> = pairs
            .iter()
            .map(|(a, b)| [a.as_str(), b.as_str()])
            .collect();
        opposing.set_item(format!("{nuclide} {kind}"), pairs)?;
    }
    d.set_item("derived_opposing_uncorrelated", opposing)?;
    // Keyed "Nuclide MT852", each a lumped reaction with several components
    // whose covariance was not folded, to the components by kind.
    let lumped = PyDict::new(py);
    for ((nuclide, mtl), components) in &info.lumped_covariance_not_assignable {
        lumped.set_item(
            format!("{nuclide} MT{mtl}"),
            components.iter().cloned().collect::<Vec<_>>(),
        )?;
    }
    d.set_item("lumped_covariance_not_assignable", lumped)?;

    d.set_item(
        "covariance_repaired",
        info.covariance_repaired.iter().cloned().collect::<Vec<_>>(),
    )?;
    d.set_item("covariance_source", &info.covariance_source)?;
    d.set_item("covariance_warnings", &info.covariance_warnings)?;
    // What one repair of a nuclide's cells did, into `entry`.
    let field_repair = |entry: &Bound<'py, PyDict>,
                        r: &yani_transmute::covariance_sample::FieldRepair|
     -> PyResult<()> {
        entry.set_item("lambda_min", r.lambda_min)?;
        entry.set_item("largest_correlation_change", r.largest_correlation_change)?;
        entry.set_item(
            "correlation_frobenius_change",
            r.correlation_frobenius_change,
        )?;
        entry.set_item("cells", r.cells)?;
        entry.set_item("held_cells", r.held_cells)?;
        entry.set_item("converged", r.converged)?;
        Ok(())
    };
    let repairs = PyList::empty(py);
    for r in &info.covariance_repairs {
        let entry = PyDict::new(py);
        entry.set_item("nuclide", &r.nuclide)?;
        entry.set_item("spectrum", r.spectrum)?;
        field_repair(&entry, &r.field)?;
        // Keyed by kind, each the evaluated variance and the evaluated and
        // sampled relative sigmas, so the change reads off a single entry.
        // A negative stated variance has no sigma and reads as None.
        let channels = PyDict::new(py);
        for c in &r.channels {
            let pair = PyDict::new(py);
            pair.set_item("evaluated_variance", c.evaluated_variance)?;
            pair.set_item("evaluated_sigma", c.evaluated_sigma())?;
            pair.set_item("sampled_sigma", c.sampled)?;
            channels.set_item(&c.kind, pair)?;
        }
        entry.set_item("channels", channels)?;
        repairs.append(entry)?;
    }
    d.set_item("covariance_repairs", repairs)?;
    d.set_item(
        "covariance_repaired_outside_bound",
        info.covariance_repaired_outside_bound
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item("worst_sigma_change", info.worst_sigma_change)?;
    d.set_item(
        "rate_weighted_sigma_change",
        info.rate_weighted_sigma_change,
    )?;
    for (key, channels) in [
        ("sigma_at_least_one", &info.sigma_at_least_one),
        ("sigma_at_least_ten", &info.sigma_at_least_ten),
        (
            "sigma_at_least_one_outside_bound",
            &info.sigma_at_least_one_outside_bound,
        ),
    ] {
        let wide = PyDict::new(py);
        for ((nuclide, kind), sigma) in channels {
            wide.set_item(format!("{nuclide} {kind}"), sigma)?;
        }
        d.set_item(key, wide)?;
    }
    d.set_item("rates_sampled", info.rates_sampled)?;
    d.set_item("rates_floored", info.rates_floored)?;
    d.set_item("spectra_with_flux_sigma", info.spectra_with_flux_sigma)?;
    d.set_item(
        "spectra_without_flux_sigma",
        info.spectra_without_flux_sigma,
    )?;
    d.set_item(
        "flux_rates_without_terms",
        info.flux_rates_without_terms
            .iter()
            .map(|(nuclide, kind)| format!("{nuclide} {kind}"))
            .collect::<Vec<_>>(),
    )?;
    let limit_dict =
        |l: &yani_transmute::covariance_sample::LognormalLimit| -> PyResult<Bound<'py, PyDict>> {
            let e = PyDict::new(py);
            e.set_item("cells", l.cells)?;
            e.set_item("largest_sigma_change", l.largest_sigma_change)?;
            e.set_item("largest_correlation_change", l.largest_correlation_change)?;
            match &l.log_space_repair {
                Some(r) => {
                    let repair = PyDict::new(py);
                    field_repair(&repair, r)?;
                    e.set_item("log_space_repair", repair)?;
                }
                None => e.set_item("log_space_repair", py.None())?,
            }
            Ok(e)
        };
    let flux_limits = PyDict::new(py);
    for (spectrum, l) in &info.flux_lognormal_not_carried {
        flux_limits.set_item(*spectrum, limit_dict(l)?)?;
    }
    d.set_item("flux_lognormal_not_carried", flux_limits)?;
    let limits = PyDict::new(py);
    for (nuclide, l) in &info.lognormal_not_carried {
        limits.set_item(nuclide, limit_dict(l)?)?;
    }
    d.set_item("lognormal_not_carried", limits)?;
    d.set_item(
        "resonance_parameters",
        resonance_parameters_dict(py, &info.resonance_parameters)?,
    )?;
    d.set_item("flux_bins_sampled", info.flux_bins_sampled)?;
    d.set_item(
        "half_lives_perturbed",
        info.half_lives_perturbed
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "no_half_life_uncertainty",
        info.no_half_life_uncertainty
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "half_life_uncertainty_not_carried",
        info.half_life_uncertainty_not_carried
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item("half_lives_sampled", info.half_lives_sampled)?;
    for (key, set) in [
        (
            "decay_branchings_perturbed",
            &info.decay_branchings_perturbed,
        ),
        (
            "no_decay_branching_uncertainty",
            &info.no_decay_branching_uncertainty,
        ),
        (
            "decay_branchings_three_or_more_modes",
            &info.decay_branchings_three_or_more_modes,
        ),
        (
            "decay_branchings_unequal_sigmas",
            &info.decay_branchings_unequal_sigmas,
        ),
        ("decay_branchings_too_wide", &info.decay_branchings_too_wide),
    ] {
        d.set_item(key, set.iter().cloned().collect::<Vec<_>>())?;
    }
    d.set_item("decay_branchings_floored", info.decay_branchings_floored)?;
    d.set_item("decay_branchings_sampled", info.decay_branchings_sampled)?;
    d.set_item(
        "decay_energies_perturbed",
        info.decay_energies_perturbed
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "no_decay_energy_uncertainty",
        info.no_decay_energy_uncertainty
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "decay_energy_uncertainty_not_carried",
        info.decay_energy_uncertainty_not_carried
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    for (key, set) in [
        (
            "decay_photon_lines_perturbed",
            &info.decay_photon_lines_perturbed,
        ),
        (
            "no_decay_photon_line_uncertainty",
            &info.no_decay_photon_line_uncertainty,
        ),
        (
            "decay_photon_line_uncertainty_not_carried",
            &info.decay_photon_line_uncertainty_not_carried,
        ),
    ] {
        d.set_item(key, set.iter().cloned().collect::<Vec<_>>())?;
    }
    d.set_item(
        "decay_photon_spectra_folded",
        info.decay_photon_spectra_folded.clone(),
    )?;
    for (key, set) in [
        ("fission_yields_perturbed", &info.fission_yields_perturbed),
        (
            "no_fission_yield_uncertainty",
            &info.no_fission_yield_uncertainty,
        ),
        (
            "fission_yield_uncertainty_not_carried",
            &info.fission_yield_uncertainty_not_carried,
        ),
        (
            "fission_yields_mapping_mismatch",
            &info.fission_yields_mapping_mismatch,
        ),
    ] {
        d.set_item(key, set.iter().cloned().collect::<Vec<_>>())?;
    }
    d.set_item("statistical_rates", info.statistical_rates)?;
    d.set_item("statistical_floored", info.statistical_floored)?;
    d.set_item("statistical_sampled", info.statistical_sampled)?;
    d.set_item("not_perturbed", info.not_perturbed.clone())?;
    d.set_item("sources", info.sources.clone())?;
    d.set_item("has_gaps", info.has_gaps())?;
    Ok(d)
}

/// `Info::resonance_parameters` as a dict keyed by nuclide: each entry's
/// `method`, the `reason` it fell back to the first-order rows (`None` where
/// the parameters were sampled), and per sampled range its sampler report.
fn resonance_parameters_dict<'py>(
    py: Python<'py>,
    methods: &std::collections::BTreeMap<String, ResonanceMethod>,
) -> PyResult<Bound<'py, PyDict>> {
    let noted = |p: &NotedParameter| -> PyResult<Bound<'py, PyDict>> {
        let e = PyDict::new(py);
        e.set_item("index", p.index)?;
        e.set_item("location", format!("{:?}", p.location))?;
        e.set_item("quantity", format!("{:?}", p.quantity))?;
        e.set_item("value", p.value)?;
        e.set_item("sigma", p.sigma)?;
        Ok(e)
    };
    let repair = |r: &Option<CorrelationRepair>| -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(r) = r else { return Ok(None) };
        let e = PyDict::new(py);
        e.set_item("lambda_min", r.lambda_min)?;
        e.set_item("frobenius_change", r.frobenius_change)?;
        e.set_item("max_change", r.max_change)?;
        e.set_item("parameters", r.parameters)?;
        e.set_item("iterations", r.iterations)?;
        e.set_item("converged", r.converged)?;
        Ok(Some(e))
    };
    let out = PyDict::new(py);
    for (nuclide, method) in methods {
        let entry = PyDict::new(py);
        entry.set_item("method", method.name())?;
        let (largest, frobenius) = method.correlation_change();
        entry.set_item("largest_correlation_change", largest)?;
        entry.set_item("correlation_frobenius_change", frobenius)?;
        let ranges = PyList::empty(py);
        match method {
            ResonanceMethod::Sampled { ranges: reports } => {
                entry.set_item("reason", py.None())?;
                for r in reports {
                    let range = PyDict::new(py);
                    range.set_item("isotope", r.isotope)?;
                    range.set_item("range", r.range)?;
                    range.set_item("gaussian", r.gaussian)?;
                    range.set_item("lognormal", r.lognormal)?;
                    range.set_item("held", r.held)?;
                    let zero = PyList::empty(py);
                    for p in &r.zero_mean_widths {
                        zero.append(noted(p)?)?;
                    }
                    range.set_item("zero_mean_widths", zero)?;
                    let negative = PyList::empty(py);
                    for p in &r.negative_mean_widths {
                        negative.append(noted(p)?)?;
                    }
                    range.set_item("negative_mean_widths", negative)?;
                    range.set_item(
                        "zero_variance_with_covariance",
                        r.zero_variance_with_covariance,
                    )?;
                    range.set_item("stated_repair", repair(&r.stated_repair)?)?;
                    range.set_item("unattainable_pairs", r.unattainable_pairs)?;
                    range.set_item("transformed_repair", repair(&r.transformed_repair)?)?;
                    range.set_item("largest_correlation_change", r.correlation_change)?;
                    range.set_item(
                        "correlation_frobenius_change",
                        r.correlation_frobenius_change,
                    )?;
                    ranges.append(range)?;
                }
            }
            ResonanceMethod::FirstOrder { reason } => {
                entry.set_item("reason", reason)?;
            }
        }
        entry.set_item("ranges", ranges)?;
        out.set_item(nuclide, entry)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yani_transmute::covariance_sample::{ChannelSigma, FieldRepair, Repair};

    /// A repaired, wide-sigma report reaches Python with the nested layout the
    /// docstring promises: channels keyed by kind, a negative stated variance
    /// as None rather than a zero sigma, and "Nuclide kind" string keys.
    #[test]
    fn a_repair_and_a_wide_channel_reach_the_dict() {
        let info = Info {
            covariance_repaired: ["W182".to_string()].into(),
            covariance_repairs: vec![Repair {
                nuclide: "W182".to_string(),
                spectrum: 1,
                field: FieldRepair {
                    lambda_min: -1.0e-4,
                    largest_correlation_change: 2.0e-4,
                    correlation_frobenius_change: 3.0e-4,
                    cells: 4,
                    held_cells: 1,
                    converged: true,
                },
                channels: vec![
                    ChannelSigma {
                        kind: "(n,2n)".to_string(),
                        evaluated_variance: 0.0036,
                        sampled: 0.084,
                    },
                    ChannelSigma {
                        kind: "(n,a)".to_string(),
                        evaluated_variance: -0.001,
                        sampled: 0.02,
                    },
                ],
            }],
            covariance_repaired_outside_bound: ["Xe135".to_string()].into(),
            worst_sigma_change: f64::INFINITY,
            rate_weighted_sigma_change: Some(0.25),
            sigma_at_least_one: [(("W186".to_string(), "(n,p)".to_string()), 12.0)].into(),
            sigma_at_least_ten: [(("W186".to_string(), "(n,p)".to_string()), 12.0)].into(),
            sigma_at_least_one_outside_bound: [(("W186".to_string(), "(n,p)".to_string()), 12.0)]
                .into(),
            ..Default::default()
        };
        Python::initialize();
        Python::attach(|py| {
            let d = info_to_dict(py, &info).unwrap();
            let get = |key: &str| d.get_item(key).unwrap().unwrap();
            assert_eq!(
                get("covariance_repaired").extract::<Vec<String>>().unwrap(),
                ["W182"]
            );
            assert_eq!(
                get("covariance_repaired_outside_bound")
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["Xe135"]
            );
            assert!(get("worst_sigma_change")
                .extract::<f64>()
                .unwrap()
                .is_infinite());
            assert_eq!(
                get("rate_weighted_sigma_change").extract::<f64>().unwrap(),
                0.25
            );
            assert!(get("has_gaps").extract::<bool>().unwrap());

            let repairs = get("covariance_repairs");
            let repair = repairs.get_item(0).unwrap();
            assert_eq!(
                repair
                    .get_item("nuclide")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "W182"
            );
            assert_eq!(
                repair
                    .get_item("spectrum")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                repair
                    .get_item("lambda_min")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                -1.0e-4
            );
            assert_eq!(
                repair
                    .get_item("largest_correlation_change")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                2.0e-4
            );
            assert_eq!(
                repair
                    .get_item("correlation_frobenius_change")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                3.0e-4
            );
            assert_eq!(
                repair
                    .get_item("held_cells")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            let channels = repair.get_item("channels").unwrap();
            let n2n = channels.get_item("(n,2n)").unwrap();
            assert_eq!(
                n2n.get_item("evaluated_variance")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                0.0036
            );
            assert!(
                (n2n.get_item("evaluated_sigma")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap()
                    - 0.06)
                    .abs()
                    < 1e-15
            );
            assert_eq!(
                n2n.get_item("sampled_sigma")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                0.084
            );
            let na = channels.get_item("(n,a)").unwrap();
            assert_eq!(
                na.get_item("evaluated_variance")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                -0.001
            );
            assert!(na.get_item("evaluated_sigma").unwrap().is_none());

            for key in [
                "sigma_at_least_one",
                "sigma_at_least_ten",
                "sigma_at_least_one_outside_bound",
            ] {
                let wide = get(key);
                assert_eq!(
                    wide.get_item("W186 (n,p)")
                        .unwrap()
                        .extract::<f64>()
                        .unwrap(),
                    12.0
                );
            }
        });
    }
    /// Each nuclide's resonance method reaches the dict: a sampled one with
    /// its ranges' reports and no reason, a fallback with its reason and no
    /// ranges.
    #[test]
    fn resonance_methods_reach_the_dict() {
        use yani_transmute::resonance_sampling::SamplerReport;
        let report = SamplerReport {
            isotope: 0,
            range: 0,
            gaussian: 3,
            lognormal: 5,
            held: 1,
            zero_mean_widths: Vec::new(),
            negative_mean_widths: Vec::new(),
            zero_variance_with_covariance: 0,
            stated_repair: Some(CorrelationRepair {
                lambda_min: -1e-3,
                frobenius_change: 2e-3,
                max_change: 1e-3,
                parameters: 9,
                iterations: 4,
                converged: true,
            }),
            unattainable_pairs: 0,
            transformed_repair: None,
            correlation_change: 1e-3,
            correlation_frobenius_change: 2e-3,
        };
        let info = Info {
            resonance_parameters: [
                (
                    "W186".to_string(),
                    ResonanceMethod::Sampled {
                        ranges: vec![report],
                    },
                ),
                (
                    "La138".to_string(),
                    ResonanceMethod::FirstOrder {
                        reason: "MF=32 does not match MF=2".to_string(),
                    },
                ),
            ]
            .into(),
            ..Default::default()
        };
        Python::initialize();
        Python::attach(|py| {
            let d = info_to_dict(py, &info).unwrap();
            let methods = d.get_item("resonance_parameters").unwrap().unwrap();
            let w = methods.get_item("W186").unwrap();
            assert_eq!(
                w.get_item("method").unwrap().extract::<String>().unwrap(),
                "parameters sampled"
            );
            assert!(w.get_item("reason").unwrap().is_none());
            let range = w.get_item("ranges").unwrap().get_item(0).unwrap();
            assert_eq!(
                range
                    .get_item("lognormal")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                5
            );
            let repair = range.get_item("stated_repair").unwrap();
            assert_eq!(
                repair
                    .get_item("iterations")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                4
            );
            assert!(range.get_item("transformed_repair").unwrap().is_none());
            assert_eq!(
                w.get_item("largest_correlation_change")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                1e-3
            );
            assert_eq!(
                w.get_item("correlation_frobenius_change")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                2e-3
            );
            let la = methods.get_item("La138").unwrap();
            assert_eq!(
                la.get_item("method").unwrap().extract::<String>().unwrap(),
                "first-order rows"
            );
            assert_eq!(
                la.get_item("reason").unwrap().extract::<String>().unwrap(),
                "MF=32 does not match MF=2"
            );
            assert_eq!(la.get_item("ranges").unwrap().len().unwrap(), 0);
        });
    }
}
