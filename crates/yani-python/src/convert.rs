//! `yani.convert_transmutation`: ENDF evaluations to the format yani reads.
//!
//! The point of the whole exercise. A user who wants transmutation data types
//! `pip install yani` and calls this: no `endf` package from a git branch, no
//! separate converter distribution, no NJOY. The parser is an implementation
//! detail this module never names in its signature.

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Convert ENDF decay, fission product yield and neutron evaluations into a
/// transmutation Arrow directory.
///
/// Parameters
/// ----------
/// decay_files, fpy_files, neutron_files : list[str]
///     ENDF inputs. All three are needed to build a network: the decay data
///     says what decays into what, the neutron data what transmutes into what,
///     and the fission yields what fission produces.
/// output_path : str
///     Directory to write. Created if absent, merged into if it already holds
///     other subsections of the same library.
/// library : str
///     Library name recorded in the provenance, e.g. ``"endf-b8.1"``.
/// data_version : str
///     Identifier of the published release. This is what a consumer compares a
///     cached copy against, so a rebuild that changes values must change this.
///     It is not the converter version: two rebuilds from one converter are
///     different data.
/// created_utc : str, optional
///     Timestamp recorded in the provenance. Supply one for a reproducible
///     directory; left unset, the current time is used.
/// branch_ratios : str, optional
///     Path to a JSON branching table in ``openmc_data`` shape,
///     ``{reaction: {parent: {target: fraction}}}``. Splits a reaction between
///     a ground state and its metastable partners. Without it every
///     ``(n,gamma)`` gives its whole rate to the ground state and the
///     metastable product is absent from the network entirely.
/// subsections : list[str], optional
///     Which of ``decay``, ``reactions`` and ``fission_yields`` to write.
///     Defaults to all three. A caller assembling a chain from more than one
///     library writes each subsection from the library that has it, and the
///     manifest merges rather than being overwritten.
/// reactions : list[str], optional
///     Reaction names to follow. Defaults to every reaction the chain builder
///     knows, not the six-name short set, so nothing is silently left out.
/// decay_fill_files : list[str], optional
///     Decay evaluations from a second library, read only to replace the
///     placeholder average decay energies in ``decay_files``. Some libraries
///     write a stand-in for nuclides nobody has evaluated: a third of each
///     beta or electron-capture branch's Q to the light particles and a third
///     to the photons, which for an electron-capture emitter can be several
///     times the recoverable energy (Sn111 in ENDF/B-VIII.1 is 1.63 MeV per
///     decay against 0.69 MeV from its decay scheme). A placeholder is
///     replaced only where the second library has an evaluated decay scheme
///     for the same nuclide with a half-life within 25% of the first's.
///     Half-lives and decay modes are never touched. The decay subsection's
///     ``provenance.json`` lists every placeholder and every replacement,
///     with or without a fill.
/// decay_fill_library : str, optional
///     The library ``decay_fill_files`` came from, e.g. ``"jendl-5.0"``.
///     Required with ``decay_fill_files``.
///
/// Returns
/// -------
/// int
///     The number of nuclides in the network written.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    decay_files,
    fpy_files,
    neutron_files,
    output_path,
    library = "",
    decay_library = "",
    data_version = "",
    created_utc = None,
    reactions = None,
    branch_ratios = None,
    subsections = None,
    decay_fill_files = Vec::new(),
    decay_fill_library = "",
))]
#[allow(clippy::too_many_arguments)]
pub fn convert_transmutation(
    decay_files: Vec<String>,
    fpy_files: Vec<String>,
    neutron_files: Vec<String>,
    output_path: &str,
    library: &str,
    decay_library: &str,
    data_version: &str,
    created_utc: Option<&str>,
    reactions: Option<Vec<String>>,
    branch_ratios: Option<&str>,
    subsections: Option<Vec<String>>,
    decay_fill_files: Vec<String>,
    decay_fill_library: &str,
) -> PyResult<usize> {
    // Which inputs are required depends on what is being written. A caller
    // asking only for the reaction topology, to graft onto decay data from
    // another library, legitimately has no fission yields to give.
    let wants = |name: &str| {
        subsections
            .as_ref()
            .map(|s| s.iter().any(|x| x == name))
            .unwrap_or(true)
    };
    if decay_files.is_empty() {
        return Err(PyValueError::new_err(
            "decay_files is required: the decay data is what says which \
             nuclides exist, so without it every reaction target is unknown",
        ));
    }
    if neutron_files.is_empty() && wants("reactions") {
        return Err(PyValueError::new_err(
            "neutron_files is required to write the reactions subsection",
        ));
    }
    if fpy_files.is_empty() && wants("fission_yields") {
        return Err(PyValueError::new_err(
            "fpy_files is required to write the fission_yields subsection; \
             pass subsections=[...] without it to skip that subsection",
        ));
    }

    let provenance = yani_convert::Provenance {
        library: library.to_string(),
        decay_library: decay_library.to_string(),
        data_version: data_version.to_string(),
        created_utc: created_utc.map(str::to_string).unwrap_or_default(),
    };

    yani_convert::convert_transmutation_files(
        &decay_files,
        &fpy_files,
        &neutron_files,
        &decay_fill_files,
        decay_fill_library,
        reactions.as_deref(),
        branch_ratios.map(PathBuf::from).as_deref(),
        subsections.as_deref(),
        &PathBuf::from(output_path),
        &provenance,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Extract isomeric branching from neutron evaluations and write the
/// ``branching/`` subsection.
///
/// Separate from :func:`convert_transmutation` because the branching curves and
/// the decay data routinely come from different libraries: TENDL branching
/// beside ENDF/B decay is the usual combination. Both calls merge into the same
/// ``manifest.json``.
///
/// Parameters
/// ----------
/// neutron_files : list[str]
///     Neutron evaluations of the parent nuclides. MF=8/9/10 radionuclide
///     production is what carries the branching.
/// decay_files : list[str]
///     Decay evaluations, read only to build the isomer table, so the
///     metastable ones suffice.
/// output_path : str
///     The transmutation directory to write ``branching/`` into.
/// library, data_version, created_utc
///     As for :func:`convert_transmutation`. The library here is the branching
///     source, which need not match the chain's.
/// tol_ev : float, optional
///     Energy tolerance when matching a nuclear level to an isomeric state.
/// linearize_tol : float, optional
///     Relative tolerance for resampling a non-lin-lin region onto the lin-lin
///     pairs the format stores.
///
/// Returns
/// -------
/// dict
///     Coverage: parents read, parents with data, curves linearized, duplicate
///     groups merged, the metastable targets found, ``level_routes`` (how many
///     excited production levels were matched to an isomer by energy, by
///     energy within a tenth, by level index, as the only isomer, or not at
///     all) and ``flagged_levels`` (one line per level that was unresolved,
///     matched only by the looser energy pass, or matched by energy while its
///     level index pointed at another isomer), and ``partial_sum_mismatches``
///     (one line per reaction whose MF=10 partial cross sections do not sum to
///     its MF=3 total, or whose MF=9 yields do not sum to one, within two
///     percent below 20 MeV).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    neutron_files,
    decay_files,
    output_path,
    library = "",
    decay_library = "",
    data_version = "",
    created_utc = None,
    tol_ev = 3000.0,
    linearize_tol = yani_convert::branching::DEFAULT_LINEARIZE_TOL,
))]
#[allow(clippy::too_many_arguments)]
pub fn convert_branching(
    py: Python<'_>,
    neutron_files: Vec<String>,
    decay_files: Vec<String>,
    output_path: &str,
    library: &str,
    decay_library: &str,
    data_version: &str,
    created_utc: Option<&str>,
    tol_ev: f64,
    linearize_tol: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    if neutron_files.is_empty() || decay_files.is_empty() {
        return Err(PyValueError::new_err(
            "branching needs neutron_files for the production data and \
             decay_files for the isomer table",
        ));
    }

    let provenance = yani_convert::Provenance {
        library: library.to_string(),
        decay_library: decay_library.to_string(),
        data_version: data_version.to_string(),
        created_utc: created_utc.map(str::to_string).unwrap_or_default(),
    };

    let stats = yani_convert::convert_branching_files(
        &neutron_files,
        &decay_files,
        &PathBuf::from(output_path),
        &provenance,
        tol_ev,
        linearize_tol,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let out = pyo3::types::PyDict::new(py);
    out.set_item("parents", stats.parents)?;
    out.set_item("parents_with_data", stats.parents_with_data)?;
    out.set_item("linearized_curves", stats.linearized_curves)?;
    out.set_item("merged_duplicate_groups", stats.merged_duplicate_groups)?;
    out.set_item("metastable_targets", stats.metastable_targets)?;
    out.set_item("level_routes", stats.level_routes)?;
    out.set_item("flagged_levels", stats.flagged_levels)?;
    out.set_item("partial_sum_mismatches", stats.partial_sum_mismatches)?;
    Ok(out.unbind())
}

/// Convert an evaluation into the cross-section data an activation
/// calculation reads.
///
/// Writes a ``{nuclide}.arrow/`` directory holding ``nuclide.arrow``,
/// ``reactions.arrow`` and ``version.json``, which is exactly what a
/// transport-free reaction-rate collapse reads. The secondary distributions and
/// lookup tables a transport run needs are deliberately not written: an
/// activation calculation never touches them, and building them from an ACE
/// table would cost heating data it does not carry.
///
/// Parameters
/// ----------
/// input_path : str
///     An ENDF evaluation, or an ACE table when ``source_format="ace"``.
/// output_dir : str
///     Directory to write ``{nuclide}.arrow/`` into.
/// source_format : {"endf", "ace"}
///     ``"endf"`` runs NJOY to reconstruct resonances and Doppler broaden,
///     which is the documented route and the only one that can reach an
///     arbitrary temperature. ``"ace"`` reads a table you already have and
///     needs no NJOY, at the cost of being stuck with the temperature it was
///     processed at.
/// njoy_exec : str, optional
///     The NJOY executable, for ``source_format="endf"``. Note that FENDL is
///     processed with the IAEA-NDS fork and upstream NJOY does not reproduce
///     the official release; both builds succeed, so the wrong choice is quiet.
/// temperatures : list[float], optional
///     Temperatures in Kelvin. Defaults to NJOY's own default. Ignored for
///     ``source_format="ace"``.
/// library, data_version, created_utc
///     Recorded in ``version.json``. ``data_version`` identifies the published
///     release and is what a consumer compares a cached copy against.
/// covariance : bool
///     Also write ``covariance.arrow``, the MF=33 cross-section covariance.
///     Off by default: the matrices are large and only an uncertainty
///     calculation reads them. Requires ``source_format="endf"`` -- MF=33 is
///     not carried through ACER, so asking for it from an ACE table raises.
///
/// Returns
/// -------
/// str
///     Path to the directory written.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_dir,
    source_format = "endf",
    njoy_exec = "njoy",
    temperatures = None,
    library = "",
    data_version = "",
    created_utc = None,
    covariance = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn convert_neutron_xs(
    input_path: &str,
    output_dir: &str,
    source_format: &str,
    njoy_exec: &str,
    temperatures: Option<Vec<f64>>,
    library: &str,
    data_version: &str,
    created_utc: Option<&str>,
    covariance: bool,
) -> PyResult<String> {
    let input = PathBuf::from(input_path);
    let source = match source_format {
        "ace" => yamc_convert::entry::Source::Ace { path: &input },
        "endf" => yamc_convert::entry::Source::Endf {
            path: &input,
            njoy_exec,
            temperatures: temperatures.unwrap_or_default(),
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown source_format {other:?}; expected \"endf\" or \"ace\""
            )))
        }
    };

    let provenance = yamc_convert::entry::Provenance {
        library: library.to_string(),
        data_version: data_version.to_string(),
        created_utc: created_utc.map(str::to_string).unwrap_or_default(),
    };

    yamc_convert::entry::convert_neutron_xs(
        &source,
        &PathBuf::from(output_dir),
        &provenance,
        covariance,
    )
    .map(|p| p.to_string_lossy().into_owned())
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert an ENDF evaluation into the full neutron transport data set.
///
/// Everything :func:`convert_neutron_xs` writes, plus the reaction products,
/// their secondary angle and energy distributions, the lookup accelerator the
/// transport hot path reads, and the two fissile-only sections.
///
/// ENDF only, deliberately. An ACE table holds one temperature, carries no
/// MT 901 heating and no fission energy release, so a transport conversion
/// from one would be missing the local KERMA and the fission photon scaling
/// with nothing on disk to say so.
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the ENDF evaluation.
/// output_dir : str
///     Directory to write ``{Nuclide}.arrow/`` into.
/// njoy_exec : str
///     The NJOY executable. FENDL needs the IAEA-NDS fork; upstream NJOY does
///     not reproduce the official release, and both builds succeed, so the
///     wrong choice is quiet.
/// temperatures : list[float], optional
///     Temperatures in Kelvin. Defaults to NJOY's own default.
/// library, data_version, created_utc
///     Recorded in ``version.json``.
/// covariance : bool
///     Also write ``covariance.arrow``, the MF=33 cross-section covariance.
///     Off by default: the matrices are large and only an uncertainty
///     calculation reads them.
///
/// Returns
/// -------
/// str
///     Path to the directory written.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_dir,
    njoy_exec = "njoy",
    temperatures = None,
    library = "",
    data_version = "",
    created_utc = None,
    covariance = false,
))]
pub fn convert_neutron_transport(
    input_path: &str,
    output_dir: &str,
    njoy_exec: &str,
    temperatures: Option<Vec<f64>>,
    library: &str,
    data_version: &str,
    created_utc: Option<&str>,
    covariance: bool,
) -> PyResult<String> {
    let input = PathBuf::from(input_path);
    let source = yamc_convert::entry::Source::Endf {
        path: &input,
        njoy_exec,
        temperatures: temperatures.unwrap_or_default(),
    };
    let provenance = yamc_convert::entry::Provenance {
        library: library.to_string(),
        data_version: data_version.to_string(),
        created_utc: created_utc.map(str::to_string).unwrap_or_default(),
    };

    yamc_convert::entry::convert_neutron_transport(
        &source,
        &PathBuf::from(output_dir),
        &provenance,
        covariance,
    )
    .map(|p| p.to_string_lossy().into_owned())
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert a photoatomic evaluation into the per-element photon sections.
///
/// Writes ``element.arrow`` always, and ``subshells.arrow``, ``compton.arrow``
/// and ``bremsstrahlung.arrow`` where there is data for them. No NJOY: a
/// photoatomic evaluation is already pointwise.
///
/// Parameters
/// ----------
/// photoatomic_path : str
///     The photoatomic evaluation. One file may hold several elements, and
///     each is written to its own directory.
/// output_dir : str
///     Directory to write ``{Element}.arrow/`` into.
/// relaxation_path : str, optional
///     The atomic relaxation evaluation for the same element, carrying binding
///     energies, occupancies and the transition cascade. Without it the
///     subshells are written with zero binding energy and no transitions,
///     which is a fluorescence-free atom rather than an error.
/// compton_profiles, density_effect, bremsstrahlung : str, optional
///     The auxiliary tabulations, which no evaluation carries. All three must
///     be given together or none of them; without them ``compton.arrow`` and
///     ``bremsstrahlung.arrow`` are absent.
/// library, data_version, created_utc
///     Recorded in ``version.json``.
///
/// Returns
/// -------
/// list[str]
///     One path per element written.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    photoatomic_path,
    output_dir,
    relaxation_path = None,
    compton_profiles = None,
    density_effect = None,
    bremsstrahlung = None,
    library = "",
    data_version = "",
    created_utc = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn convert_photon(
    photoatomic_path: &str,
    output_dir: &str,
    relaxation_path: Option<&str>,
    compton_profiles: Option<&str>,
    density_effect: Option<&str>,
    bremsstrahlung: Option<&str>,
    library: &str,
    data_version: &str,
    created_utc: Option<&str>,
) -> PyResult<Vec<String>> {
    let photoatomic = PathBuf::from(photoatomic_path);
    let relaxation = relaxation_path.map(PathBuf::from);

    let tabulation_paths = match (compton_profiles, density_effect, bremsstrahlung) {
        (Some(c), Some(d), Some(b)) => Some((PathBuf::from(c), PathBuf::from(d), PathBuf::from(b))),
        (None, None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "compton_profiles, density_effect and bremsstrahlung must be given \
                 together or not at all: the three are read as one set, and a \
                 partial set would silently drop a section",
            ))
        }
    };
    let tabulations =
        tabulation_paths
            .as_ref()
            .map(|(c, d, b)| yamc_convert::entry::PhotonTabulations {
                compton_profiles: c,
                density_effect: d,
                bremsstrahlung: b,
            });

    let provenance = yamc_convert::entry::Provenance {
        library: library.to_string(),
        data_version: data_version.to_string(),
        created_utc: created_utc.map(str::to_string).unwrap_or_default(),
    };

    yamc_convert::entry::convert_photon(
        &photoatomic,
        relaxation.as_deref(),
        tabulations.as_ref(),
        &PathBuf::from(output_dir),
        &provenance,
    )
    .map(|dirs| {
        dirs.into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// List the final states each reaction of an evaluation can leave its product
/// in.
///
/// A reaction that can leave its product in a metastable state says so in
/// MF=8, one subsection per final state. An evaluation that lists only the
/// ground state is not merely less accurate: the isomer is absent from any
/// network built from it, so no code can make it, and a measurement that sees
/// its decay heat cannot be reproduced by any means. TENDL-2017 omits the
/// 1706 keV state from ``Os190(n,n')``, which is why the FNS osmium foil comes
/// out at a third of the measured heat with that library, in yani and in
/// FISPACT-II alike. Nothing about the reaction looks wrong from outside: it
/// is present, its cross section is reasonable, and only the state list is
/// short. So comparing the state lists of several libraries is how such a gap
/// is found, and this is the read that makes the comparison possible.
///
/// A read, not a conversion. Nothing is written, no decay data is involved,
/// and the answer is what the files say rather than what a network built from
/// them would hold.
///
/// Parameters
/// ----------
/// neutron_files : list[str]
///     Neutron evaluations. Read one at a time rather than held, so a whole
///     sublibrary is a valid argument.
///
/// Returns
/// -------
/// list[dict]
///     One entry per (parent, reaction) carrying MF=9 or MF=10, in file order,
///     each with ``parent``, ``mt``, ``reaction`` (the transmutation reaction
///     name, or ``None`` for an MT no chain reaction covers) and ``states``.
///     Each state has ``excitation_energy_eV``, ``level_index``, ``product``
///     and ``source``.
///
///     ``product`` is the product's **ground-state** name even for an excited
///     state, because naming the isomer needs decay data to say which
///     isomeric ordinal a level is; pair it with ``excitation_energy_eV``.
///     ``level_index`` is the evaluation's own LFS and is not comparable
///     between libraries: Ir190's 377 keV isomer is level 3 in ENDF/B-VIII.1
///     and level 37 in JEFF-4.0. ``source`` is ``"cross_section"`` for MF=10
///     or ``"yield"`` for MF=9.
///
/// Examples
/// --------
///     >>> [c for c in yani.radionuclide_production(["n-Os190.tendl"])
///     ...  if c["mt"] == 4][0]["states"]
///     [{'excitation_energy_eV': 0.0, 'level_index': 0, 'product': 'Os190',
///       'source': 'cross_section'}]
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (neutron_files))]
pub fn radionuclide_production(
    py: Python<'_>,
    neutron_files: Vec<String>,
) -> PyResult<Py<pyo3::types::PyList>> {
    if neutron_files.is_empty() {
        return Err(PyValueError::new_err(
            "neutron_files is required: the production data is in the neutron \
             evaluations, so there is nothing to read without them",
        ));
    }
    let channels = yani_convert::production::production_from_files(&neutron_files)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let out = pyo3::types::PyList::empty(py);
    for channel in channels {
        let entry = pyo3::types::PyDict::new(py);
        entry.set_item("parent", channel.parent)?;
        entry.set_item("mt", channel.mt)?;
        entry.set_item("reaction", channel.reaction)?;
        let states = pyo3::types::PyList::empty(py);
        for state in channel.states {
            let row = pyo3::types::PyDict::new(py);
            row.set_item("excitation_energy_eV", state.excitation_energy)?;
            row.set_item("level_index", state.level_index)?;
            row.set_item("product", state.product)?;
            row.set_item("source", state.source)?;
            states.append(row)?;
        }
        entry.set_item("states", states)?;
        out.append(entry)?;
    }
    Ok(out.unbind())
}
