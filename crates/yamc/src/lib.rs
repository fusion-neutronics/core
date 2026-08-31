//! Core Monte Carlo particle transport: geometry and cells, the transport
//! kernel, materials/source wiring, and the `Model` simulation driver.
pub mod geo;
pub mod geometry;
pub mod model;
pub mod mpi_context;
pub mod options;
pub mod stochastic_volume;
pub mod track;
mod transport;
pub mod util;
pub mod variance_reduction;

// `Model::transmute`: the transport-coupled driver over `yani-transmute`.
// Nothing but an inherent impl, so the module itself stays private.
mod transmute;

pub use geo::*;
pub use geometry::backend::*;
pub use geometry::cell::*;
#[cfg(feature = "mesh")]
pub use geometry::mesh::MeshGeometry;
pub use geometry::*;

#[cfg(feature = "wasm")]
pub mod wasm {
    pub mod config_wasm;
    pub mod data_wasm;
    pub mod element_wasm;
    pub mod material_wasm;
    pub mod nuclide_wasm;
    pub mod reaction_wasm;
    pub mod simulation_wasm;
}
#[cfg(feature = "wasm")]
pub use wasm::config_wasm;
#[cfg(feature = "wasm")]
pub use wasm::data_wasm;
#[cfg(feature = "wasm")]
pub use wasm::element_wasm;
#[cfg(feature = "wasm")]
pub use wasm::material_wasm;
#[cfg(feature = "wasm")]
pub use wasm::material_wasm::WasmMaterial;
#[cfg(feature = "wasm")]
pub use wasm::nuclide_wasm;
#[cfg(feature = "wasm")]
pub use wasm::reaction_wasm;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use transport::debug as transport_debug;
pub use util::{interpolate_linear, interpolate_log_log};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

#[cfg(feature = "wasm")]
pub use wasm::config_wasm::*;
#[cfg(feature = "wasm")]
pub use wasm::data_wasm::*;
#[cfg(feature = "wasm")]
pub use wasm::element_wasm::*;
#[cfg(feature = "wasm")]
pub use wasm::nuclide_wasm::*;
#[cfg(feature = "wasm")]
pub use wasm::reaction_wasm::*;
