//! Building a whole nuclear data library locally, from the evaluated source
//! archives its publishers host, into the folder layout of a published
//! release.
//!
//! [`generate::generate`] is the whole build, in stages: [`recipe`] says
//! what each library keyword is built from and what its release holds,
//! [`sources`] gets those files onto local disk checked against their pins
//! (with one message for any a host only serves to a web browser),
//! [`extract`] unpacks them per sublibrary, and [`identify`] names every
//! material from its own header. Then [`convert`] turns each material into
//! its folder of the release, [`chain`] builds the transmutation chain,
//! [`gates`] check the result, [`index`] writes each particle's
//! `index.json`, and the manifest is written last.
//!
//! Native only: nothing in a browser builds a library.

pub mod chain;
pub mod convert;
pub mod extract;
pub mod gates;
pub mod generate;
pub mod identify;
pub mod index;
pub mod os_downloads;
pub mod recipe;
pub mod sources;
pub mod thermal;

use std::path::Path;

pub use convert::{Options, Outcome, Particle};
pub use generate::{generate, Report, Request};
pub use identify::Material;
pub use recipe::{Part, Recipe, Sublibrary};
pub use sources::Search;

/// Fetch, check, unpack and identify every source of `keyword` under
/// `<path>/.build/`, returning every material found.
///
/// `downloads` is a folder searched first for source files fetched by hand
/// (see [`sources`] for the full order).
pub fn prepare_sources(
    keyword: &str,
    path: &Path,
    downloads: Option<&Path>,
) -> Result<Vec<Material>, String> {
    let recipe = recipe::recipe(keyword)?;
    let fetched = sources::fetch(recipe, path, &Search::new(downloads))?;
    let folders = extract::extract(recipe, &fetched, path)?;
    identify::identify(&folders)
}
