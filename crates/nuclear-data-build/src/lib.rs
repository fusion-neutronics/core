//! Building a whole nuclear data library locally, from the evaluated source
//! archives its publishers host, into the folder layout of a published
//! release.
//!
//! This is the sources stage: [`recipe`] says what each library keyword is
//! built from, [`sources`] gets those files onto local disk checked against
//! their pins (with one message for any a host only serves to a web
//! browser), [`extract`] unpacks them per sublibrary, and [`identify`] names
//! every material from its own header. Conversion comes after this.
//!
//! Native only: nothing in a browser builds a library.

pub mod extract;
pub mod identify;
pub mod os_downloads;
pub mod recipe;
pub mod sources;

use std::path::Path;

pub use identify::Material;
pub use recipe::{Recipe, Sublibrary};
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
