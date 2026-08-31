//! PyO3 bindings exposing yamc to Python as the `yamc._core` extension
//! module: materials, geometry, sources, tallies, transport, and transmutation.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::new_without_default,
    clippy::format_in_format_args,
    clippy::manual_range_contains
)]

use pyo3::prelude::*;
use pyo3::pymodule;
use pyo3::wrap_pyfunction;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

pub mod geometry;
pub mod id_map_helper;
pub mod lost_particle;
pub mod simulation;
pub mod tally;

// The transmutation-side bindings live in `yani-python` so the standalone
// `yani` wheel can ship them without any of the transport stack (issue #381).
// Re-exported under their original paths so the rest of this crate, and any
// `yamc_python::material::...` user, is unaffected by where they live.
pub use yani_python::{distribution, element, html_repr, material, particle};
pub mod tally_reduce;
pub mod ui;
pub mod variance_reduction;

/// Build the [`pyo3_stub_gen::StubInfo`] the `stub_gen` binary uses to emit
/// `packages/yamc-core/python/yamc/_core.pyi`. The pyproject lives in the
/// wheel's package directory, not in this crate, so this cannot use
/// `define_stub_info_gatherer!`. Defining this in the lib (and calling it from
/// the bin) also forces the rlib, and thus the `inventory`-registered stub
/// metadata, to be linked.
pub fn gen_stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crate is at <workspace>/crates/yamc-python");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(
        workspace_root.join("packages/yamc-core/pyproject.toml"),
    )
}

/// Return ``True`` if yamc was built with MPI support.
#[gen_stub_pyfunction]
#[pyfunction]
fn has_mpi() -> bool {
    yamc::mpi_context::has_mpi_support()
}

/// Return this process's MPI rank (``0`` when not running under MPI).
#[gen_stub_pyfunction]
#[pyfunction]
fn mpi_rank() -> i32 {
    yamc::mpi_context::mpi_rank()
}

/// Return the number of MPI ranks (``1`` when not running under MPI).
#[gen_stub_pyfunction]
#[pyfunction]
fn mpi_size() -> i32 {
    yamc::mpi_context::mpi_size()
}

/// Finalize MPI. Call once before process exit in MPI runs; a no-op otherwise.
#[gen_stub_pyfunction]
#[pyfunction]
fn mpi_finalize() {
    yamc::mpi_context::mpi_finalize()
}

/// Return ``True`` if the GPU compute path is usable -- yamc was built with the
/// ``gpu`` feature and an f64-capable Vulkan adapter is present.
#[gen_stub_pyfunction]
#[pyfunction]
fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        yamc_gpu::GpuContext::new().is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// List the names of the f64-capable Vulkan GPU adapters on this host.
///
/// Pass one of these names directly as ``compute=<name>`` to
/// :meth:`Model.simulate_transport` to run on that specific adapter; plain
/// ``compute='gpu'`` auto-selects (preferring a discrete GPU). Returns an
/// empty list when there's no usable GPU (or yamc was built without the
/// ``gpu`` feature). Pure enumeration -- it does not initialize the GPU.
#[gen_stub_pyfunction]
#[pyfunction]
fn list_gpu_adapters() -> Vec<String> {
    #[cfg(feature = "gpu")]
    {
        yamc_gpu::list_vulkan_f64_adapters()
            .into_iter()
            .map(|adapter| adapter.name)
            .collect()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Materials, nuclides, chains, elements, sources, schedules and
    // transmutation results, all shared with the standalone `yani` wheel.
    yani_python::register_classes(_py, m, "yamc")?;

    m.add_class::<geometry::PyGeometry>()?;
    m.add_class::<geometry::PyVolumeResult>()?;
    m.add_class::<geometry::PyBoundingBox>()?;
    m.add_class::<id_map_helper::PyGeometrySliceData>()?;
    m.add_class::<id_map_helper::PyVoxelData>()?;
    m.add_class::<tally::PyMeshSliceData>()?;
    m.add_class::<ui::PyInteractivePlot>()?;
    m.add_class::<ui::PyInteractiveTallyPlot>()?;
    m.add_class::<simulation::PyModel>()?;
    // The converters. They are implemented in `yani-python` because the
    // standalone transmutation wheel needs them too, and registered here as
    // well because THIS is the package that ships the auxiliary photon
    // tabulations: `yamc.convert_photon` can default to them, where
    // `yani.convert_photon` has to be given their paths. yani never needs
    // photon data, since transmutation is driven by the neutron flux.
    m.add_function(wrap_pyfunction!(
        yani_python::convert::convert_transmutation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        yani_python::convert::convert_branching,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        yani_python::convert::convert_neutron_xs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        yani_python::convert::convert_neutron_transport,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(yani_python::convert::convert_photon, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::combine_results, m)?)?;
    m.add_function(wrap_pyfunction!(tally_reduce::reduce_mesh_tally_block, m)?)?;
    m.add_class::<variance_reduction::PySurvivalBiasing>()?;
    m.add_class::<variance_reduction::PyWeightWindowBounds>()?;
    m.add_class::<variance_reduction::PyWeightWindowGeneratorDeGVR>()?;
    m.add_function(wrap_pyfunction!(geometry::Plane, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::Sphere, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::Cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::Cone, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::Torus, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::Quadric, m)?)?;
    // The surface factories above (Plane/Sphere/Cylinder/...) all return a
    // `Surface`, so the type must be importable for `-> Surface` annotations to
    // resolve and for `isinstance`/type hints to work.
    m.add_class::<geometry::PySurface>()?;
    m.add_class::<geometry::PyCell>()?;
    m.add_class::<geometry::PyRegion>()?;
    m.add_class::<geometry::PyHalfspace>()?;
    m.add_class::<geometry::PyRegularRectangularMesh>()?;
    m.add_class::<geometry::PyRegularCylindricalMesh>()?;
    // Decay-photon shutdown-dose-rate post-processing: an irradiation/cooling
    // timeline (Pulse/Cooldown steps) plus its typed dose result.
    tally::register_tally_classes(_py, m)?;

    m.add_class::<simulation::PyTallyResult>()?;
    m.add_class::<simulation::PySimulationResults>()?;
    m.add_class::<simulation::PyTallyIter>()?;
    m.add_class::<simulation::PyStatisticalChecks>()?;
    m.add_class::<simulation::PyConvergencePoint>()?;
    m.add_class::<simulation::PyScorePdf>()?;
    m.add_class::<tally::PyConvergenceTarget>()?;

    m.add_class::<simulation::PyTrackEvent>()?;
    m.add_class::<simulation::PyParticleTrack>()?;
    m.add_class::<simulation::PyTracks>()?;
    m.add_class::<lost_particle::PyLostParticle>()?;

    #[cfg(feature = "mesh")]
    {
        m.add_class::<geometry::PyMeshGeometry>()?;
    }

    #[cfg(feature = "cad")]
    {
        m.add_function(wrap_pyfunction!(geometry::mesh_face, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::mesh_faces, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::mesh_to_arrow, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::cad_mesh_to_arrow, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::cad_mesh_labels, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::weld_mesh, m)?)?;
        m.add_class::<geometry::PyFaceOutput>()?;
        m.add_function(wrap_pyfunction!(geometry::mesh_faces_scene_resolved, m)?)?;
        m.add_function(wrap_pyfunction!(geometry::mesh_volume_rs, m)?)?;
        m.add_class::<geometry::SceneFaceOutput>()?;
    }

    let parallel_submodule = PyModule::new(_py, "parallel")?;
    parallel_submodule.add_function(wrap_pyfunction!(has_mpi, &parallel_submodule)?)?;
    parallel_submodule.add_function(wrap_pyfunction!(gpu_available, &parallel_submodule)?)?;
    parallel_submodule.add_function(wrap_pyfunction!(list_gpu_adapters, &parallel_submodule)?)?;
    parallel_submodule.add_function(wrap_pyfunction!(mpi_rank, &parallel_submodule)?)?;
    parallel_submodule.add_function(wrap_pyfunction!(mpi_size, &parallel_submodule)?)?;
    parallel_submodule.add_function(wrap_pyfunction!(mpi_finalize, &parallel_submodule)?)?;
    m.add_submodule(&parallel_submodule)?;
    // Both names, matching what `yani_python::add_submodule` does for the
    // shared submodules: `yamc._core.parallel` for the stub layout, and
    // `yamc.parallel` so `import yamc.parallel` resolves. This one is built
    // here rather than there because it is transport-only.
    let modules = _py.import("sys")?.getattr("modules")?;
    modules.set_item("yamc._core.parallel", &parallel_submodule)?;
    modules.set_item("yamc.parallel", &parallel_submodule)?;

    #[cfg(feature = "mpi")]
    {
        let atexit = _py.import("atexit")?;
        let finalize_fn = parallel_submodule.getattr("mpi_finalize")?;
        atexit.call_method1("register", (finalize_fn,))?;
    }

    Ok(())
}
