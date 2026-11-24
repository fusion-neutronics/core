//! Fetch, unpack and identify one library's sources, and count what was
//! found: `cargo run --release -p nuclear-data-build --example
//! prepare_sources -- <keyword> <path> [<downloads folder>]`.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [keyword, path, rest @ ..] = args.as_slice() else {
        eprintln!("usage: prepare_sources <keyword> <path> [<downloads folder>]");
        std::process::exit(2);
    };
    let downloads = rest.first().map(PathBuf::from);
    match nuclear_data_build::prepare_sources(keyword, path.as_ref(), downloads.as_deref()) {
        Ok(materials) => {
            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
            for m in &materials {
                *counts.entry(m.sublibrary.dir()).or_default() += 1;
            }
            for (sub, n) in counts {
                println!("{sub}: {n} materials");
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
