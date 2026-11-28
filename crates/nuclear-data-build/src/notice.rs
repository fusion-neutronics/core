//! The release's `NOTICE.md`: who made the library, what to cite, the terms
//! its data is redistributed under, what the conversion changed, and the
//! source files it was built from.
//!
//! Licences such as CC BY 4.0 ask for the credit, the licence and a note of
//! any changes to travel with the data. Written into the release folder, the
//! notice is listed in the manifest and published with the release, so every
//! copy (a cache, a `download_library(path=...)` folder, a mirror) carries
//! it. What it says about the library comes from
//! [`yamc_nuclide::storage::credits`], the table the docs and a run's
//! `data_releases` read too.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use yamc_nuclide::storage::credits;
use yamc_nuclide::storage::release::NOTICE;

use crate::recipe::{Part, Recipe};

/// The notice for release `release` of `recipe`'s library: its `parts`,
/// converted by `converter` from `recipe.sources`.
pub fn render(
    recipe: &Recipe,
    parts: &[Part],
    release: &str,
    converter: &str,
) -> Result<String, String> {
    let credit = credits::credit(recipe.keyword).ok_or_else(|| {
        format!(
            "{} has no entry in yamc_nuclide::storage::credits, so its release would \
             carry no credit: add who made it and what to cite before building it",
            recipe.keyword
        )
    })?;
    let mut s = String::new();
    let line = |s: &mut String, text: &str| {
        s.push_str(text);
        s.push('\n');
    };
    line(
        &mut s,
        &format!(
            "# {} (`{}`), release {release}",
            credit.library, recipe.keyword
        ),
    );
    line(&mut s, "");
    line(
        &mut s,
        &format!(
            "This folder is {}, an evaluated nuclear data library, converted for yamc \
             and yani. It was made by {}. Its evaluators ask to be cited wherever their \
             data is used.",
            credit.library, credit.made_by
        ),
    );
    line(&mut s, "");
    line(&mut s, "## Cite");
    line(&mut s, "");
    for r in credit.references {
        if r.doi.is_empty() {
            line(&mut s, &format!("- {}", r.text));
        } else {
            line(
                &mut s,
                &format!("- {}, doi:{} (https://doi.org/{})", r.text, r.doi, r.doi),
            );
        }
    }
    line(&mut s, "");
    line(&mut s, &format!("The library's page: {}", credit.page));
    line(&mut s, "");
    line(&mut s, "## Terms");
    line(&mut s, "");
    line(&mut s, credit.terms);
    line(&mut s, "");
    line(&mut s, "## What was changed");
    line(&mut s, "");
    line(&mut s, &changes(parts, converter));
    line(&mut s, "");
    line(&mut s, "## Sources");
    line(&mut s, "");
    line(&mut s, "| file | bytes | sha256 |");
    line(&mut s, "| --- | ---: | --- |");
    for source in recipe.sources {
        let _ = writeln!(
            s,
            "| [{}]({}) | {} | `{}` |",
            source.name, source.url, source.pin.bytes, source.pin.sha256
        );
    }
    line(&mut s, "");
    line(&mut s, "## No warranty, no endorsement");
    line(&mut s, "");
    line(
        &mut s,
        &format!(
            "This conversion is provided as is, without warranty of any kind. It is \
             not endorsed by the makers of {}.",
            credit.library
        ),
    );
    Ok(s)
}

/// What the conversion did to the evaluations, for the parts built.
fn changes(parts: &[Part], converter: &str) -> String {
    let mut done = Vec::new();
    if parts.contains(&Part::Neutron) {
        done.push(
            "neutron data processed with NJOY at the temperatures the manifest records".to_string(),
        );
    }
    if parts.contains(&Part::Thermal) {
        done.push(
            "thermal scattering laws processed with NJOY against the library's own \
             neutron evaluations"
                .to_string(),
        );
    }
    if parts.contains(&Part::Photon) {
        done.push(
            "photo-atomic data converted, with the Compton profile, density effect and \
             bremsstrahlung tabulations yamc's photon transport adds beside it"
                .to_string(),
        );
    }
    if parts.contains(&Part::Photonuclear) {
        done.push("photonuclear evaluations converted as evaluated".to_string());
    }
    let chain: Vec<&str> = [
        (Part::Decay, "decay"),
        (Part::Reactions, "reaction"),
        (Part::FissionYields, "fission yield"),
        (Part::Branching, "isomeric branching"),
    ]
    .into_iter()
    .filter(|(p, _)| parts.contains(p))
    .map(|(_, name)| name)
    .collect();
    if !chain.is_empty() {
        let list = match chain.as_slice() {
            [one] => one.to_string(),
            [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
            [] => unreachable!(),
        };
        done.push(format!("the {list} data restructured into tables"));
    }
    format!(
        "Converted from ENDF-6 to the Arrow tables yamc and yani read, by {converter}: \
         {}. No evaluated value was intentionally changed. Evaluations the conversion \
         could not process are left out rather than replaced, and the manifest lists \
         every file the release holds.",
        done.join("; ")
    )
}

/// Write the notice into the release folder `root`.
pub fn write(
    root: &Path,
    recipe: &Recipe,
    parts: &[Part],
    release: &str,
    converter: &str,
) -> Result<(), String> {
    let path = root.join(NOTICE);
    fs::write(&path, render(recipe, parts, release, converter)?)
        .map_err(|e| format!("could not write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe;

    #[test]
    fn every_recipe_has_a_credit() {
        for keyword in recipe::keywords() {
            let recipe = recipe::recipe(keyword).unwrap();
            render(
                recipe,
                recipe.parts,
                "2026-11-27",
                "yani-python 0.0.0 (core abc)",
            )
            .unwrap();
        }
    }

    #[test]
    fn the_notice_cites_the_library_and_lists_its_sources() {
        let recipe = recipe::recipe("jendl-5.0").unwrap();
        let text = render(
            recipe,
            recipe.parts,
            "2026-11-27",
            "yani-python 0.0.0 (core abc)",
        )
        .unwrap();
        assert!(text.starts_with("# JENDL-5 (`jendl-5.0`), release 2026-11-27\n"));
        assert!(text.contains("doi:10.1080/00223131.2022.2141903"));
        assert!(text.contains("JAEA requires the reference"));
        assert!(text.contains("by yani-python 0.0.0 (core abc)"));
        for source in recipe.sources {
            assert!(text.contains(source.pin.sha256), "{}", source.name);
        }
        assert!(!text.contains('\u{2014}'), "no em dashes");
    }

    #[test]
    fn the_changes_name_only_what_was_built() {
        let chain_only = changes(&[Part::Branching], "b");
        assert!(!chain_only.contains("NJOY"), "{chain_only}");
        assert!(
            chain_only.contains("the isomeric branching data"),
            "{chain_only}"
        );
        let full = changes(recipe::recipe("endf-b8.1").unwrap().parts, "b");
        assert!(full.contains("NJOY") && full.contains("Compton"), "{full}");
        assert!(
            full.contains("the decay, reaction, fission yield and isomeric branching data"),
            "{full}"
        );
    }
}
