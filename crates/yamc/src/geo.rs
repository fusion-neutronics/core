//! Re-exports of the geometry primitives from the `yamc-geo` crate.
//!
//! `yamc` doesn't define these types itself -- it re-exports `yamc-geo`'s
//! surfaces, regions, bounding boxes, and plot helpers under `yamc::geo::*`
//! (and, via `lib.rs`, at the crate root) so transport code and the Python
//! bindings have one obvious place to find them.
pub use yamc_geo::{bounding_box::*, plot::*, region::*, surface::*};
