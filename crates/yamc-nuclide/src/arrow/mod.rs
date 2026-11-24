//! Apache Arrow IPC reader for neutron nuclear data.
//!
//! `arrow_helpers` holds the shared column-getter primitives; `nuclide_arrow`
//! is the full `.arrow/` directory loader built on them. Re-exported at the
//! crate root so the historic `yamc_nuclide::arrow_helpers` /
//! `yamc_nuclide::nuclide_arrow` paths stay stable.

pub mod arrow_helpers;
pub mod covariance_arrow;
pub mod nuclide_arrow;
