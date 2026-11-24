//! GPU integration layer.
//!
//! Glue between yamc's `Model` (the user-facing API) and yamc-gpu's
//! flat-buffer kernel (`run_multi_cell_transport`). Lives on the yamc
//! side because it knows about both: it walks `Model`'s cells / surfaces
//! / materials / sources and produces the `Vec<f64>` / `Vec<u32>` inputs
//! the kernel takes.
//!
//! Gated behind the `gpu` cargo feature; pulls in yamc-gpu transitively.
//! On macOS the underlying yamc-gpu compiles to a stub whose
//! `GpuContext::new()` returns `NoF64Adapter`, so enabling the feature
//! is a no-op at runtime -- the future `compute='auto'` dispatch will
//! catch this and fall back to CPU.
//!
//! ## Supported subset
//!
//! The kernel covers a narrow slice of yamc's transport surface today.
//! `translate_for_gpu` rejects models that use features outside this
//! slice with a clear `GpuTranslateError`. The slice includes:
//!
//! - **Geometry**: CSG only (no mesh geometry, no mesh-filled cells).
//!   Cells carry a flat CSG region program the kernel evaluates during
//!   cell finding, so unions, complements and rings are all in scope;
//!   a cell whose bounding box cannot be derived is rejected.
//! - **Surfaces**: every CSG surface kind -- sphere, plane, cylinder
//!   (any axis), the X/Y/Z tori, the general quadric, and the
//!   arbitrary-axis double cone. Only mesh-boundary surfaces (which
//!   are not a `SurfaceKind`) remain outside the CSG path.
//! - **Materials**: any number, each with its own per-material union
//!   energy grid and sparse per-MT cross sections.
//! - **Sources**: neutron and photon. A photon source routes to the
//!   photon kernel and a mixed neutron+photon source to the mixed
//!   dispatch, so the neutron translation here only ever sees neutron
//!   sources.
//! - **Reactions**: elastic (including free-gas), discrete and
//!   continuum inelastic, fission (with a device fission bank), the
//!   unresolved resonance range, and secondary-photon production.
//! - **Boundaries**: transmission and vacuum, encoded per surface for
//!   the kernel (`BoundaryType` has no other variants).
//! - **Tallies**: the dispatch layer reads the model's tallies to
//!   derive bin params and rejects configurations the kernel can't
//!   score; cell, mesh (rectangular and cylindrical), energy and
//!   particle-type filters are supported.
//! - **Coupled neutron->photon transport and D1S decay photons** run
//!   via a two-pass dispatch (neutron kernel banks photons, photon
//!   kernel transports them).
//! - **No transmutation and no variance reduction on the GPU.**

pub mod dispatch;
pub mod error;
pub mod translate;
pub mod translate_photon;

pub use dispatch::{run_on_gpu, run_on_gpu_with_device, GpuDispatchError, GpuRunResult};
pub use error::GpuTranslateError;
pub use translate::{translate_for_gpu, GpuTransportInputs};
pub use translate_photon::{translate_photon_for_gpu, GpuPhotonTransportInputs};
