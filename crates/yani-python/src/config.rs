use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::collections::HashMap;
use yamc_nuclide::config::{BranchSources, SubsectionSource, CONFIG};

/// Refuse a branching-only library keyword (`irdff-ii`) where `role` names
/// what it was asked to be, before anything is stored or downloaded.
fn refuse_branching_only(source: &str, role: &str) -> PyResult<()> {
    if yamc_nuclide::url_cache::is_branching_only(source) {
        return Err(PyValueError::new_err(
            yamc_nuclide::url_cache::branching_only_error(source, role),
        ));
    }
    Ok(())
}

/// Get the current cross-section data mapping.
///
/// Returns ``None`` if nothing has been set, a string if a single global
/// library keyword or directory is in use, or a dict mapping nuclide names
/// to paths.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_cross_section_data(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let config = CONFIG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if config.cross_sections.is_empty() {
        return match config.default_cross_section.clone() {
            Some(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
            None => Ok(py.None()),
        };
    }

    let dict = PyDict::new(py);
    for (nuclide, path) in &config.cross_sections {
        dict.set_item(nuclide, path)?;
    }
    Ok(dict.into_any().unbind())
}

/// The directory downloaded nuclear data is cached in.
///
/// A folder in the platform's per-user cache directory (under ``~/.cache`` on
/// Linux, or ``$XDG_CACHE_HOME`` when that is set; ``~/Library/Caches`` on
/// macOS; ``%LOCALAPPDATA%`` on Windows), shared by both wheels. Inside it
/// each library keeps one folder per published release,
/// ``<keyword>/<release>/``, holding that release's ``manifest.json`` and the
/// files downloaded from it, each verified against the manifest. Read only:
/// to use data from somewhere else, point the data source at a local
/// directory instead.
///
/// Returns:
///     The cache directory as a string, whether or not it exists yet.
///
/// Raises:
///     RuntimeError: If the platform resolves no home directory.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn cache_dir() -> PyResult<String> {
    yamc_nuclide::url_cache::cache_root()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "no nuclear-data cache directory: the platform resolved no home directory",
            )
        })
}

/// Set the cross-section data mapping.
///
/// Accepts ``None`` to clear, a string for a global library keyword or
/// directory, or a dict mapping nuclide names to paths or keywords.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_cross_section_data(
    #[gen_stub(override_type(
        type_repr = "builtins.str | dict[builtins.str, builtins.str] | None"
    ))]
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut config = CONFIG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    match value {
        None => {
            config.cross_sections.clear();
            config.default_cross_section = None;
        }
        Some(v) if v.is_none() => {
            config.cross_sections.clear();
            config.default_cross_section = None;
        }
        Some(v) => {
            if let Ok(dict) = v.cast::<PyDict>() {
                if dict.is_empty() {
                    // An empty dict carries no entries, so treat it like `None`:
                    // clear both the per-nuclide mappings and the global default.
                    // (An empty `HashMap` passed to `set_cross_sections` is a
                    // no-op, which would otherwise leave a previously set global
                    // default in place -- so `cross_section_data = {}` would not
                    // actually clear the configuration.)
                    config.cross_sections.clear();
                    config.default_cross_section = None;
                } else {
                    let mut rust_map = HashMap::new();
                    for (k, val) in dict.iter() {
                        let key: String = k.extract()?;
                        let path: String = val.extract()?;
                        refuse_branching_only(&path, CROSS_SECTION_ROLE)?;
                        rust_map.insert(key, path);
                    }
                    config.set_cross_sections(rust_map);
                }
            } else if let Ok(string_val) = v.extract::<String>() {
                refuse_branching_only(&string_val, CROSS_SECTION_ROLE)?;
                config.set_cross_sections(string_val);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "cross_section_data must be None, a string, or a dict",
                ));
            }
        }
    }

    Ok(())
}

/// Look up the cross-section data path for a single nuclide, applying the global
/// default if no per-nuclide entry is set.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn lookup_cross_section_data(nuclide: &str) -> Option<String> {
    let config = CONFIG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    config.get_cross_section(nuclide)
}

/// Set the cross-section data path for a single nuclide, or set the global default
/// if ``path`` is omitted and ``nuclide_or_keyword`` is a library keyword /
/// directory.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (nuclide_or_keyword, path=None))]
pub fn set_cross_section_data_entry(nuclide_or_keyword: &str, path: Option<&str>) -> PyResult<()> {
    refuse_branching_only(path.unwrap_or(nuclide_or_keyword), CROSS_SECTION_ROLE)?;
    let mut config = CONFIG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    config.set_cross_section(nuclide_or_keyword, path);
    Ok(())
}

// Transmutation network subsections. Each part (decay, reactions,
// fission_yields, branch_ratios) is sourced independently by a library keyword
// or a path; unset parts fall back to a built-in default at load time
// (endf-b8.1 for decay/reactions/fission_yields; no branching overlay when
// branch_ratios is unset). branch_ratios also takes several sources in order.
// Getters return the raw setting (``None`` = default).

/// What a subsection setter accepts from Python: a keyword or path, `None`,
/// or `False`.
///
/// `reactions` and `fission_yields` have three states where the other two
/// sources have two, so `None` alone cannot spell them. `None` keeps the
/// meaning it has everywhere else in this module -- put it back to the
/// default -- and `False` is the one that turns the subsection off.
#[derive(FromPyObject)]
pub enum SubsectionArg {
    // Checked before the string arm: `False` is not a source name.
    Toggle(bool),
    Source(String),
}

impl pyo3_stub_gen::PyStubType for SubsectionArg {
    fn type_input() -> pyo3_stub_gen::TypeInfo {
        // `Literal[False]` rather than `bool`, because `True` is not accepted:
        // subsection_source below rejects it, naming the three spellings that
        // do work. A `bool` here advertises a fourth that raises.
        pyo3_stub_gen::TypeInfo::builtin("str")
            | pyo3_stub_gen::TypeInfo::with_module(
                "typing.Literal[False]",
                pyo3_stub_gen::ModuleRef::Named("typing".to_string()),
            )
    }

    fn type_output() -> pyo3_stub_gen::TypeInfo {
        Self::type_input()
    }
}

/// Read a subsection setter's argument, naming the three spellings when the
/// caller reaches for a fourth. `role` is what the source would be, for the
/// refusal of a branching-only library.
fn subsection_source(value: Option<SubsectionArg>, role: &str) -> PyResult<SubsectionSource> {
    match value {
        None => Ok(SubsectionSource::Default),
        Some(SubsectionArg::Toggle(false)) => Ok(SubsectionSource::Off),
        Some(SubsectionArg::Toggle(true)) => Err(PyValueError::new_err(
            "True is not a source: pass a library keyword or a path to choose \
             one, None to use the default library, or False to turn the \
             subsection off",
        )),
        Some(SubsectionArg::Source(source)) => {
            refuse_branching_only(&source, role)?;
            Ok(SubsectionSource::Set(source))
        }
    }
}

/// What a cross-section source is, as a refusal names it.
const CROSS_SECTION_ROLE: &str = "the neutron cross-section library";

/// Get the decay-data source for the transmutation network, or ``None``.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_transmutation_decay_data() -> Option<String> {
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_decay_data
        .clone()
}

/// Set the decay-data source. Pass ``None`` to reset to the default library.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_transmutation_decay_data(value: Option<&str>) -> PyResult<()> {
    if let Some(source) = value {
        refuse_branching_only(source, "the decay-data source")?;
    }
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_decay_data = value.map(|s| s.to_string());
    Ok(())
}

/// Get the reaction-topology source for the transmutation network.
///
/// Returns the source that will actually be used -- the default library when
/// the setting is untouched or was reset with ``None`` -- or ``None`` when it
/// has been turned off with ``False``.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_transmutation_reactions() -> Option<String> {
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_transmutation_reactions()
}

/// Set the reaction-topology source.
///
/// Pass a library keyword or a path to choose one, or ``None`` to go back to
/// the default library.
///
/// Pass ``False`` to turn the subsection off, for a decay-only calculation on
/// a material with no reaction rates. A rate that then needs it is refused
/// when the burnup matrix is built, naming the nuclide and the reaction,
/// rather than being solved as though the reaction produced nothing.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_transmutation_reactions(value: Option<SubsectionArg>) -> PyResult<()> {
    let source = subsection_source(value, "the reactions source")?;
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_reactions = source;
    Ok(())
}

/// Get the fission-yields source for the transmutation network.
///
/// Returns the source that will actually be used -- the default library when
/// the setting is untouched or was reset with ``None`` -- or ``None`` when it
/// has been turned off with ``False``.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_transmutation_fission_yields() -> Option<String> {
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_transmutation_fission_yields()
}

/// Set the fission-yields source.
///
/// Pass a library keyword or a path to choose one, or ``None`` to go back to
/// the default library.
///
/// Pass ``False`` to turn the subsection off, which is the honest setting for
/// a material nothing in which fissions and for a library that publishes no
/// yields of its own. A non-zero fission rate that then needs them is refused
/// when the burnup matrix is built, naming the nuclide, rather than losing its
/// fission products while still burning it.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_transmutation_fission_yields(value: Option<SubsectionArg>) -> PyResult<()> {
    let source = subsection_source(value, "the fission-yields source")?;
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_fission_yields = source;
    Ok(())
}

/// What the branching setting accepts from Python: one source, or several in
/// order.
#[derive(FromPyObject)]
pub enum BranchSourcesArg {
    // Checked before the list arm: a str is a sequence of str, and PyO3
    // refuses to read one as a list, so the order only makes the intent plain.
    One(String),
    Ordered(Vec<String>),
}

impl pyo3_stub_gen::PyStubType for BranchSourcesArg {
    fn type_input() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::builtin("str")
            | pyo3_stub_gen::TypeInfo::with_module(
                "typing.Sequence[builtins.str]",
                pyo3_stub_gen::ModuleRef::Named("typing".to_string()),
            )
    }

    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::builtin("str")
            | pyo3_stub_gen::TypeInfo::builtin("list[builtins.str]")
    }
}

/// Get the isomeric-branching sources of the transmutation network.
///
/// Returns:
///     What was assigned: one source as a string, a list of sources (in the
///     order a reaction takes its split from them) as a list, or ``None``
///     when nothing is set and no isomeric-branching overlay is applied.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "builtins.str | list[builtins.str] | None"))]
pub fn get_transmutation_branch_ratios(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let setting = CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_branch_ratios
        .clone();
    Ok(match setting {
        BranchSources::Unset => py.None(),
        BranchSources::One(source) => source.into_pyobject(py)?.into_any().unbind(),
        BranchSources::List(sources) => sources.into_pyobject(py)?.into_any().unbind(),
    })
}

/// Set the isomeric-branching sources of the transmutation network.
///
/// A source is a library keyword or a path to a ``branching`` subsection.
/// Give one, or several in order: each reaction of each parent (each
/// channel) takes its isomeric split, and the MF=40 covariance of that
/// split, from the first source that evaluates it, and the other sources'
/// data for that channel are not used. The branching report
/// (``TransmutationResults.get_branching_report``) names the source of every
/// channel, and says whether the run mixes libraries.
///
/// ``"irdff-ii"``, the IAEA dosimetry library IRDFF-II, supplies isomeric
/// branching only, so it is only ever a source here, and only where it is
/// named: it is never a default.
///
/// It reads back as it was assigned: a string as a string, a list (even one
/// of one) as a list.
///
/// Args:
///     value: A source, a list of sources in order, or ``None`` (or an empty
///         list) for no isomeric-branching overlay.
///
/// Raises:
///     ValueError: If a source is empty or is listed twice.
///
/// Examples:
///     >>> yani.transmutation_branch_ratios = ["irdff-ii", "endf-b8.1"]
///     >>> yani.transmutation_branch_ratios
///     ['irdff-ii', 'endf-b8.1']
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_transmutation_branch_ratios(value: Option<BranchSourcesArg>) -> PyResult<()> {
    let setting = match value {
        None => BranchSources::Unset,
        Some(BranchSourcesArg::One(source)) => BranchSources::One(source),
        Some(BranchSourcesArg::Ordered(sources)) => BranchSources::List(sources),
    };
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .set_transmutation_branch_ratios(setting)
        .map_err(PyValueError::new_err)
}

/// Install the function every nuclear-data download goes through.
///
/// Emscripten (Pyodide) builds only: they have no sockets, so the host
/// supplies the HTTP GETs. `fetcher(requests)` is called with a list of
/// `(url, range)` tuples and must answer a list of `(status, body)` tuples, one
/// per request and in the same order. `range` is `None` for a whole object,
/// otherwise the `Range` header value to send (`"bytes=0-99,200-299"`), and a
/// header naming several ranges is answered with the origin's
/// `multipart/byteranges` body as it came. A status of 0 means the request did
/// not complete, and is retried. The requests in one call are independent and
/// should be in flight together: a load of many nuclides batches them for
/// that. In a browser this blocks a Web Worker on `Atomics.wait` while a
/// helper worker fetches.
///
/// Here rather than in either wheel's module so both register the same hook:
/// it is `yamc._core._set_data_fetcher` and `yani._core._set_data_fetcher`,
/// and either one installs the fetcher for the whole process.
#[cfg(target_os = "emscripten")]
#[pyfunction]
pub fn _set_data_fetcher(fetcher: Py<PyAny>) {
    yamc_nuclide::url_cache::set_host_fetcher(Box::new(move |requests| {
        Python::attach(|py| -> PyResult<Vec<(u16, Vec<u8>)>> {
            let replies = fetcher.call1(py, (requests.to_vec(),))?;
            replies
                .bind(py)
                .try_iter()?
                .map(|reply| {
                    let (status, body): (u16, Bound<'_, pyo3::types::PyBytes>) =
                        reply?.extract()?;
                    Ok((status, body.as_bytes().to_vec()))
                })
                .collect()
        })
        .map_err(|e| format!("data fetcher failed: {e}"))
    }));
}
