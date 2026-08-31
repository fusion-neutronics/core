use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::collections::HashMap;
use yamc_nuclide::config::{SubsectionSource, CONFIG};

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
                        rust_map.insert(key, path);
                    }
                    config.set_cross_sections(rust_map);
                }
            } else if let Ok(string_val) = v.extract::<String>() {
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
// branch_ratios is unset). Getters return the raw setting (``None`` = default).

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
/// caller reaches for a fourth.
fn subsection_source(value: Option<SubsectionArg>) -> PyResult<SubsectionSource> {
    match value {
        None => Ok(SubsectionSource::Default),
        Some(SubsectionArg::Toggle(false)) => Ok(SubsectionSource::Off),
        Some(SubsectionArg::Toggle(true)) => Err(PyValueError::new_err(
            "True is not a source: pass a library keyword or a path to choose \
             one, None to use the default library, or False to turn the \
             subsection off",
        )),
        Some(SubsectionArg::Source(source)) => Ok(SubsectionSource::Set(source)),
    }
}

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
    let source = subsection_source(value)?;
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
    let source = subsection_source(value)?;
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_fission_yields = source;
    Ok(())
}

/// Get the isomeric branch-ratios source for the transmutation network, or ``None``.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_transmutation_branch_ratios() -> Option<String> {
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_branch_ratios
        .clone()
}

/// Set the isomeric branch-ratios source. Pass ``None`` to disable the overlay.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn set_transmutation_branch_ratios(value: Option<&str>) -> PyResult<()> {
    CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .transmutation_branch_ratios = value.map(|s| s.to_string());
    Ok(())
}
