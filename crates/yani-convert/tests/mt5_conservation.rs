//! Every MT=5 list a whole library converts to is checked for conservation,
//! not a sample.
//!
//! Opt-in, since it reads whole local libraries: set `YANI_MT5_ENDF` to a
//! directory holding `endfb-viii.1-endf/`, `jeff-4.0-endf/`,
//! `tendl-2025-endf/` and `fendl-3.2d-endf/` as the build scripts unpack
//! them, and run
//! `RAYON_NUM_THREADS=8 cargo test -p yani-convert --test mt5_conservation -- --ignored --nocapture`.
//! The evaluations are streamed one at a time per thread, as a build does.

use std::path::{Path, PathBuf};

fn files(dir: &Path, patterns: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("readable directory") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if patterns
                .iter()
                .any(|(start, end)| name.starts_with(start) && name.ends_with(end))
            {
                out.push(path.to_string_lossy().to_string());
            }
        }
    }
    out.sort();
    out
}

#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_ENDF"]
fn every_mt5_list_of_every_library_is_checked() {
    let Some(root) = std::env::var_os("YANI_MT5_ENDF").map(PathBuf::from) else {
        eprintln!("YANI_MT5_ENDF unset; nothing to check");
        return;
    };
    let endf = root.join("endfb-viii.1-endf");
    let evaluations: [(&str, &str); 3] = [("", ".endf"), ("", ".dat"), ("", ".jeff")];
    let libraries: [(&str, PathBuf, PathBuf, PathBuf); 4] = [
        (
            "endf-b8.1",
            endf.join("decay-version.VIII.1"),
            endf.join("nfy-version.VIII.1"),
            endf.join("neutrons-version.VIII.1"),
        ),
        (
            "jeff-4.0",
            root.join("jeff-4.0-endf/decay"),
            root.join("jeff-4.0-endf/nfy"),
            root.join("jeff-4.0-endf/neutron"),
        ),
        (
            "tendl-2025",
            endf.join("decay-version.VIII.1"),
            endf.join("nfy-version.VIII.1"),
            root.join("tendl-2025-endf"),
        ),
        (
            "fendl-3.2d",
            endf.join("decay-version.VIII.1"),
            endf.join("nfy-version.VIII.1"),
            root.join("fendl-3.2d-endf/neutron"),
        ),
    ];
    for (library, decay, nfy, neutron) in libraries {
        let out = std::env::temp_dir().join(format!(
            "yani-mt5-conservation-{library}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out);
        let neutron_files = files(
            &neutron,
            &[
                ("", ".endf"),
                ("n-", ".tendl"),
                ("n_", ".jeff"),
                ("n_", ".dat"),
            ],
        );
        yani_convert::convert_transmutation_files(
            &files(&decay, &evaluations),
            &files(&nfy, &evaluations),
            &neutron_files,
            &[],
            "",
            None,
            None,
            Some(&["reactions".to_string()]),
            &out,
            &yani_convert::Provenance {
                library: library.to_string(),
                decay_library: String::new(),
                data_version: "test".to_string(),
                created_utc: String::new(),
            },
        )
        .expect("converts");
        let record: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("reactions/provenance.json")).unwrap(),
        )
        .unwrap();
        let mt5 = &record["mt5"];

        // The parents the file carries (n,X) for, from the file itself.
        let rows = std::fs::File::open(out.join("reactions/reactions.arrow")).unwrap();
        let reader = arrow_ipc::reader::FileReader::try_new(rows, None).unwrap();
        let mut parents = std::collections::BTreeSet::new();
        for batch in reader {
            let batch = batch.unwrap();
            let nuclides = batch
                .column_by_name("nuclide")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .clone();
            let kinds = batch
                .column_by_name("type")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .clone();
            for i in 0..batch.num_rows() {
                if kinds.value(i) == yani_convert::anything::ANYTHING {
                    parents.insert(nuclides.value(i).to_string());
                }
            }
        }
        let checked = mt5["conservation"].as_object().unwrap();
        let unchecked: Vec<&String> = parents
            .iter()
            .filter(|p| !checked.contains_key(*p) && !mt5_has(&mt5["no_product_data"], p))
            .collect();
        let count = |key: &str| mt5[key].as_array().unwrap().len();
        println!(
            "{library}: {} parents with (n,X), {} checked for conservation, {} not conserving \
             (MT=5-weighted imbalance above {}), {} with an impossible multiplicity, {} with \
             no residual given, {} whose residuals are not one per reaction, {} without MF=6",
            parents.len(),
            checked.len(),
            count("not_conserving"),
            mt5["conservation_tolerance"],
            count("impossible_multiplicity"),
            count("residual_not_given"),
            count("residuals_not_one_per_reaction"),
            count("no_product_data"),
        );
        for entry in mt5["not_conserving"].as_array().unwrap() {
            println!(
                "    {:8} charge {:+.3e} mass {:+.3e} (weighted), worst charge {:+.3e} at {:.3e} eV",
                entry["nuclide"].as_str().unwrap(),
                entry["charge_weighted"].as_f64().unwrap(),
                entry["mass_number_weighted"].as_f64().unwrap(),
                entry["charge"].as_f64().unwrap(),
                entry["charge_at_eV"].as_f64().unwrap()
            );
        }
        let _ = std::fs::remove_dir_all(&out);
        assert!(unchecked.is_empty(), "{library}: unchecked {unchecked:?}");
    }
}

fn mt5_has(list: &serde_json::Value, parent: &str) -> bool {
    list.as_array()
        .is_some_and(|l| l.iter().any(|v| v.as_str() == Some(parent)))
}
