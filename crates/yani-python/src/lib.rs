//! PyO3 bindings for the transmutation stack, shared by two wheels.
//!
//! Materials, nuclides, reactions, chains, elements, irradiation schedules and
//! the inventories a transmutation produces. Everything here is transport-free:
//! the crate graph stops at `yani-transmute` / `yamc-materials` and never
//! reaches geometry, tallies, meshing or the `yamc` crate (issue #381).
//!
//! Two consumers register the same classes:
//!
//! * the `yani` wheel, whose `_core` is [`_core`] below, built with the
//!   `extension-module` feature; and
//! * the `yamc` wheel, whose own `#[pymodule]` calls [`register_classes`] and
//!   then re-homes the classes' `__module__` onto `yamc._core`.
//!
//! The entry point is feature-gated so the `_core` symbol exists in exactly one
//! of the two cdylibs and the wheels can never collide.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::new_without_default,
    clippy::format_in_format_args,
    clippy::manual_range_contains
)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod config;
pub mod convert;
pub mod data_uncertainty;
pub mod distribution;
pub mod element;
pub mod group_structures;
pub mod html_repr;
pub mod material;
pub mod particle;
pub mod read;
mod shapes;
pub mod transmutation_results;

pub use transmutation_results::PyTransmutationResults;

/// Every class and function both wheels expose, added to `module`.
///
/// Also builds the runtime submodules (`sources`, `data`, `materials`,
/// `_decay_photons`) and registers them under `<package>._core.<name>` in
/// `sys.modules`, which is what makes `from yani._core.data import ...` work.
/// `package` is the wheel's import name (`"yamc"` or `"yani"`), and is the only
/// thing that differs between the two registrations.
pub fn register_classes(py: Python<'_>, m: &Bound<'_, PyModule>, package: &str) -> PyResult<()> {
    // Transport-only surface. These are reachable from the shared types
    // (`Nuclide.reactions` -> `Reaction` -> `ReactionProduct` ->
    // `AngleDistribution`), so they must remain *usable* in both wheels, but
    // there is nothing an inventory calculation does with a scattering cosine,
    // and `create_test_reaction_product` is a test helper that was never meant
    // to be public at all. Registering them top-level on the transmutation
    // wheel put them in its documented API. Issue #452.
    let transport = package == "yamc";
    if transport {
        m.add_class::<material::PyAngleDistribution>()?;
        m.add_class::<material::PyReactionProduct>()?;
        m.add_class::<material::PyTabulated>()?;
        m.add_class::<particle::PyParticle>()?;
        m.add_function(wrap_pyfunction!(material::sample_scatter_cosine, m)?)?;
        m.add_function(wrap_pyfunction!(material::create_test_reaction_product, m)?)?;
    }
    m.add_class::<material::PyDoseCoefficients>()?;
    m.add_class::<material::PyPhotonCoefficients>()?;
    m.add_class::<material::PyMaterial>()?;
    m.add_class::<material::PyEnriched>()?;
    m.add_function(wrap_pyfunction!(material::enriched, m)?)?;
    m.add_class::<material::PyNuclide>()?;
    m.add_class::<material::PyReaction>()?;
    m.add_class::<material::PyChain>()?;
    m.add_class::<element::PyElement>()?;
    m.add_class::<PyTransmutationResults>()?;
    m.add_class::<transmutation_results::PyEstimate>()?;
    m.add_class::<transmutation_results::PyLineEstimate>()?;
    m.add_class::<data_uncertainty::PyDataUncertainty>()?;

    m.add_class::<distribution::PyNeutronSource>()?;
    // PhotonSource takes the (x, p) pair `decay_photon_spectrum()` returns, but
    // the round-trip it exists for is a transport run, which this wheel cannot
    // do. Exposing it here would only advertise a dead end.
    if transport {
        m.add_class::<distribution::PyPhotonSource>()?;
    }
    // Irradiation/cooling timeline plus its typed dose result.
    m.add_class::<distribution::PyPulse>()?;
    m.add_class::<distribution::PyCooldown>()?;
    m.add_class::<distribution::PyPulseSchedule>()?;
    m.add_function(wrap_pyfunction!(distribution::cooldown_steps, m)?)?;
    m.add_class::<distribution::PyDoseResult>()?;

    // The built-in energy group structures as data, not just as a name that can
    // be passed somewhere: the edges behind `"CCFE-709"` (issue #492).
    m.add_function(wrap_pyfunction!(group_structures::group_structure, m)?)?;
    m.add_function(wrap_pyfunction!(
        group_structures::group_structure_names,
        m
    )?)?;

    // Nuclear-data configuration: which library, and which chain subsections.
    m.add_function(wrap_pyfunction!(convert::convert_transmutation, m)?)?;
    m.add_function(wrap_pyfunction!(convert::convert_branching, m)?)?;
    m.add_function(wrap_pyfunction!(convert::radionuclide_production, m)?)?;
    m.add_function(wrap_pyfunction!(convert::convert_neutron_xs, m)?)?;
    m.add_function(wrap_pyfunction!(convert::convert_neutron_transport, m)?)?;
    m.add_function(wrap_pyfunction!(convert::convert_photon, m)?)?;
    // Outside the `transport` gate with the converters: reading a written
    // directory back through the real loader is the gate on a conversion, and
    // both wheels convert.
    m.add_function(wrap_pyfunction!(read::read_nuclide_from_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(read::read_element_from_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(config::get_cross_section_data, m)?)?;
    m.add_function(wrap_pyfunction!(config::set_cross_section_data, m)?)?;
    m.add_function(wrap_pyfunction!(config::lookup_cross_section_data, m)?)?;
    m.add_function(wrap_pyfunction!(config::set_cross_section_data_entry, m)?)?;
    m.add_function(wrap_pyfunction!(config::get_transmutation_decay_data, m)?)?;
    m.add_function(wrap_pyfunction!(config::set_transmutation_decay_data, m)?)?;
    m.add_function(wrap_pyfunction!(config::get_transmutation_reactions, m)?)?;
    m.add_function(wrap_pyfunction!(config::set_transmutation_reactions, m)?)?;
    m.add_function(wrap_pyfunction!(
        config::get_transmutation_fission_yields,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        config::set_transmutation_fission_yields,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        config::get_transmutation_branch_ratios,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        config::set_transmutation_branch_ratios,
        m
    )?)?;

    // The source-building distributions live under `<package>.sources`;
    // NeutronSource/PhotonSource stay top level as the primary builders.
    let sources = PyModule::new(py, "sources")?;
    distribution::register_distribution_classes(py, &sources)?;
    add_submodule(py, m, &sources, package, "sources")?;

    // Lump shapes for self-shielding live under `<package>.shapes`, beside the
    // source distributions, for the same reason: they are small value types the
    // caller constructs and hands in.
    let shapes = PyModule::new(py, "shapes")?;
    shapes::register_shape_classes(py, &shapes)?;
    add_submodule(py, m, &shapes, package, "shapes")?;

    let data = PyModule::new(py, "data")?;
    data.add_function(wrap_pyfunction!(material::dose_coefficients, &data)?)?;
    data.add_function(wrap_pyfunction!(
        material::mass_attenuation_coefficient,
        &data
    )?)?;
    data.add_function(wrap_pyfunction!(
        material::mass_energy_absorption_coefficient,
        &data
    )?)?;
    data.add_function(wrap_pyfunction!(material::natural_abundance, &data)?)?;
    data.add_function(wrap_pyfunction!(material::split_nuclide, &data)?)?;
    data.add_function(wrap_pyfunction!(material::element_nuclides, &data)?)?;
    data.add_function(wrap_pyfunction!(material::element_names, &data)?)?;
    data.add_function(wrap_pyfunction!(material::reaction_names, &data)?)?;
    data.add_function(wrap_pyfunction!(material::clear_nuclide_cache, &data)?)?;
    data.add_function(wrap_pyfunction!(material::atomic_symbol, &data)?)?;
    data.add_function(wrap_pyfunction!(material::atomic_number, &data)?)?;
    add_submodule(py, m, &data, package, "data")?;

    // Curated material collections. `pnnl` is an object, not a submodule, so
    // it can be subscripted with the source document's verbatim names.
    let materials = PyModule::new(py, "materials")?;
    materials.add_class::<material::PyMaterialCollection>()?;
    materials.add("pnnl", Py::new(py, material::PyMaterialCollection::pnnl())?)?;
    materials.add_function(wrap_pyfunction!(material::collections, &materials)?)?;
    add_submodule(py, m, &materials, package, "materials")?;

    // Private: the chain walk behind Model.radionuclides().
    let decay_photons = PyModule::new(py, "_decay_photons")?;
    decay_photons.add_function(wrap_pyfunction!(
        distribution::get_radionuclides_from_chain,
        &decay_photons
    )?)?;
    add_submodule(py, m, &decay_photons, package, "_decay_photons")?;

    rehome(m, package)?;
    Ok(())
}

/// Attach `child` to `parent` and register it in `sys.modules` under
/// `<package>._core.<name>`, which is what lets Python import from it.
fn add_submodule(
    py: Python<'_>,
    parent: &Bound<'_, PyModule>,
    child: &Bound<'_, PyModule>,
    package: &str,
    name: &str,
) -> PyResult<()> {
    rehome(child, package)?;
    parent.add_submodule(child)?;
    let modules = py.import("sys")?.getattr("modules")?;
    modules.set_item(format!("{package}._core.{name}"), child)?;
    // Also register the public path, so `from yani.sources import Histogram`
    // resolves and not just the attribute access `yani.sources.Histogram`. Each
    // wheel's `__init__.py` used to repeat this for three of the four
    // submodules, which meant the list could drift from the one above; here a
    // submodule cannot be added without getting both names. `_decay_photons` is
    // private but registered the same way: hiding the dotted path bought
    // nothing, since the attribute was reachable regardless.
    modules.set_item(format!("{package}.{name}"), child)?;
    Ok(())
}

/// Point every class in `m` at the importing package.
///
/// The `#[pyclass]` attributes here deliberately name no module: there are two,
/// and a literal in the source could only ever be one of them (it would also
/// make stub generation for the other wheel refuse to run). pyo3 pyclasses are
/// heap types, so `__module__` is writable, and every class this function
/// registers is ours to set. Submodule classes get the same
/// `<package>._core`, matching where the generated stubs declare them.
fn rehome(m: &Bound<'_, PyModule>, package: &str) -> PyResult<()> {
    let core = format!("{package}._core");
    let names: Vec<String> = m
        .dict()
        .keys()
        .iter()
        .map(|k| k.extract::<String>())
        .collect::<PyResult<_>>()?;
    for name in names {
        let obj = m.getattr(name.as_str())?;
        if obj.is_instance_of::<pyo3::types::PyType>() {
            let _ = obj.setattr("__module__", core.as_str());
        }
    }
    Ok(())
}

/// The `yani` wheel's extension module.
///
/// Only compiled with the `extension-module` feature, which the `yani` wheel
/// turns on and the `yamc` wheel does not, so the two cdylibs never both
/// define `PyInit__core`.
#[cfg(feature = "extension-module")]
#[pyo3::pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_classes(py, m, "yani")
}

/// Build the [`pyo3_stub_gen::StubInfo`] the `stub_gen` binary uses to emit
/// `packages/yani-core/python/yani/_core.pyi`. The pyproject lives in the wheel's
/// package directory, not in this crate, so this cannot use
/// `define_stub_info_gatherer!`. Defining it in the lib (and calling it from
/// the bin) also forces the rlib, and thus the `inventory`-registered stub
/// metadata, to be linked.
pub fn gen_stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crate is at <workspace>/crates/yani-python");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(
        workspace_root.join("packages/yani-core/pyproject.toml"),
    )
}
