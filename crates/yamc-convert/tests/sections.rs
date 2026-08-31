//! Convert a real ACE table and check the sections against the published data.
//!
//! The transmutation half of this work taught the lesson these tests are built
//! on: small fixtures cannot find the defects that matter. Every one of the
//! four bugs in that converter needed the real library to surface, and none was
//! visible to a hand-built chain. So these check against the published
//! `.arrow` output where it is available on this machine, and say so when it is
//! not, rather than passing quietly.

use std::path::{Path, PathBuf};

use endf::IncidentNeutron;

/// The Li6 ACE table committed with the parser.
const LI6_ACE: &[u8] = include_bytes!("../../endf/fixtures/Li6.ace.xz");

fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

fn li6() -> IncidentNeutron {
    let tables = endf::ace::tables_from_str(&text(LI6_ACE), None).expect("ACE parses");
    IncidentNeutron::from_ace(&tables[0], endf::ace::MetastableScheme::Mcnp).expect("Li6 reads")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yamc-convert-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

/// The published Li6 directory, if this machine has one.
fn published(nuclide: &str) -> Option<PathBuf> {
    let p = PathBuf::from(yamc_test_cache::nuclide_path(nuclide));
    p.is_dir().then_some(p)
}

fn read_batch(path: &Path) -> arrow_array::RecordBatch {
    use arrow_ipc::reader::FileReader;
    let reader = FileReader::try_new(std::fs::File::open(path).expect("open"), None).expect("ipc");
    let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>().expect("batches");
    assert!(!batches.is_empty(), "{} has no batches", path.display());
    batches.into_iter().next().unwrap()
}

/// nuclide.arrow carries the identity the loader keys everything else off.
#[test]
fn nuclide_section_matches_the_parsed_table() {
    use arrow_array::{cast::AsArray, types::Float64Type};

    let data = li6();
    let dir = scratch("nuclide");
    yamc_convert::nuclide::write_nuclide(&data, &dir).expect("writes");

    let batch = read_batch(&dir.join("nuclide.arrow"));
    assert_eq!(batch.num_rows(), 1, "nuclide.arrow is a single row");

    let name = batch.column_by_name("name").unwrap().as_string::<i32>();
    assert_eq!(name.value(0), "Li6");
    assert_eq!(
        batch
            .column_by_name("Z")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        3
    );
    assert_eq!(
        batch
            .column_by_name("A")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        6
    );

    // The two temperature lists are written from different sources and are not
    // required to be equal: the NJOY route adds a 0 K energy grid. For a pure
    // ACE table they must agree, and a mismatch here means the energy map and
    // the kT list came apart.
    let temps = batch
        .column_by_name("temperatures")
        .unwrap()
        .as_list::<i32>();
    let energy_temps = batch
        .column_by_name("energy_temperatures")
        .unwrap()
        .as_list::<i32>();
    let t: Vec<String> = temps
        .value(0)
        .as_string::<i32>()
        .iter()
        .flatten()
        .map(str::to_string)
        .collect();
    let et: Vec<String> = energy_temps
        .value(0)
        .as_string::<i32>()
        .iter()
        .flatten()
        .map(str::to_string)
        .collect();
    assert!(!t.is_empty(), "no temperatures were written");
    assert_eq!(t, et, "an ACE table has one energy grid per temperature");

    // Every energy grid must be non-empty and ascending. A grid that is not
    // sorted breaks every binary search downstream and nothing else checks it.
    let grids = batch
        .column_by_name("energy_values")
        .unwrap()
        .as_list::<i32>();
    let outer = grids.value(0);
    let outer = outer.as_list::<i32>();
    let n_grids = arrow_array::Array::len(outer);
    assert_eq!(n_grids, et.len(), "one grid per energy temperature");
    for i in 0..n_grids {
        let g = outer.value(i);
        let g: Vec<f64> = g.as_primitive::<Float64Type>().iter().flatten().collect();
        assert!(!g.is_empty(), "grid {i} is empty");
        assert!(
            g.windows(2).all(|w| w[1] >= w[0]),
            "grid {i} is not ascending"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Li6 has no unresolved resonance range, so no file should appear.
///
/// Writing an empty urr.arrow would be worse than writing none: the loader
/// treats an absent file as "no unresolved range" and a present one as data.
#[test]
fn no_urr_file_for_a_nuclide_without_one() {
    let data = li6();
    let dir = scratch("nourr");
    let wrote = yamc_convert::nuclide::write_urr(&data, &dir).expect("no error");
    assert!(!wrote, "Li6 has no unresolved range");
    assert!(
        !dir.join("urr.arrow").exists(),
        "an empty urr.arrow was written"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Li6 is not fissionable, and the flag is what stops the loader rescanning.
#[test]
fn fissionable_is_recorded() {
    assert!(!yamc_convert::nuclide::is_fissionable(&li6()));
}

/// The identity columns must equal the published file's, not merely be
/// self-consistent.
///
/// Skipped with a loud reason when the machine has no published Li6, rather
/// than passing: a test that silently checks nothing is how the earlier
/// fixture-cache traps worked.
#[test]
fn nuclide_section_matches_the_published_file() {
    use arrow_array::{cast::AsArray, types::Int32Type};

    let Some(ref_dir) = published("Li6") else {
        eprintln!(
            "SKIP: no published Li6 in ~/.cache/yamc; this test compares against \
             it and checked nothing"
        );
        return;
    };

    let data = li6();
    let dir = scratch("published");
    yamc_convert::nuclide::write_nuclide(&data, &dir).expect("writes");

    let mine = read_batch(&dir.join("nuclide.arrow"));
    let theirs = read_batch(&ref_dir.join("nuclide.arrow"));

    // The published file still holds the columns this build has retired, and
    // will until it is rebuilt, so parity is against the published set minus
    // those. Taken from nuclear_data_schema::retired rather than restated here:
    // a second copy of that list is how the schema and its compatibility half
    // drift apart, and this test is the thing that would stop noticing.
    let retired = nuclear_data_schema::retired("nuclide.arrow");
    let names = |b: &arrow_array::RecordBatch| -> Vec<String> {
        b.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };
    assert_eq!(
        names(&mine),
        names(&theirs)
            .into_iter()
            .filter(|n| !retired.contains(&n.as_str()))
            .collect::<Vec<_>>(),
        "column names differ from the published file, allowing for retired columns"
    );

    // `metastable` is retired, so it is no longer written and cannot be compared.
    for col in ["Z", "A"] {
        assert_eq!(
            mine.column_by_name(col)
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0),
            theirs
                .column_by_name(col)
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0),
            "{col} differs from the published file"
        );
    }
    assert_eq!(
        mine.column_by_name("name")
            .unwrap()
            .as_string::<i32>()
            .value(0),
        theirs
            .column_by_name("name")
            .unwrap()
            .as_string::<i32>()
            .value(0),
    );

    // The published Li6 came through NJOY, so it carries six temperatures and a
    // seventh 0 K energy grid, where this ACE fixture is a single temperature.
    // Comparing the lists directly would fail for a reason that is not a
    // defect, so compare what must hold either way: every temperature this
    // conversion produced must exist in the published file.
    let list = |b: &arrow_array::RecordBatch, c: &str| -> Vec<String> {
        b.column_by_name(c)
            .unwrap()
            .as_list::<i32>()
            .value(0)
            .as_string::<i32>()
            .iter()
            .flatten()
            .map(str::to_string)
            .collect()
    };
    let theirs_temps = list(&theirs, "temperatures");
    for t in list(&mine, "temperatures") {
        assert!(
            theirs_temps.contains(&t),
            "temperature {t} is not in the published file, whose set is {theirs_temps:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// reactions.arrow, checked against the published file rather than itself.
///
/// The published Li6 came through NJOY at six temperatures where this fixture
/// is a single-temperature ACE table, so the cross sections cannot be compared
/// value for value. What CAN be compared is the part that is pure conversion
/// logic and is silent when wrong: which MTs are present, what each is labelled
/// and which are marked redundant.
#[test]
fn reactions_section_matches_the_published_file() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_ipc::reader::FileReader;
    use std::collections::BTreeMap;

    let Some(ref_dir) = published("Li6") else {
        eprintln!("SKIP: no published Li6 in ~/.cache/yamc; this test checked nothing");
        return;
    };

    let data = li6();
    let dir = scratch("reactions");
    yamc_convert::reactions::write_reactions(&data, &dir).expect("writes");

    // Every batch, not just the first: this section is written one row per
    // batch on purpose and reading only batch zero would check one reaction.
    let labels = |path: &Path| -> BTreeMap<i32, (String, bool)> {
        let reader =
            FileReader::try_new(std::fs::File::open(path).expect("open"), None).expect("ipc");
        let mut out = BTreeMap::new();
        for batch in reader {
            let batch = batch.expect("batch");
            for i in 0..batch.num_rows() {
                let mt = batch
                    .column_by_name("mt")
                    .unwrap()
                    .as_primitive::<Int32Type>()
                    .value(i);
                let label = batch
                    .column_by_name("label")
                    .unwrap()
                    .as_string::<i32>()
                    .value(i)
                    .to_string();
                let redundant = batch
                    .column_by_name("redundant")
                    .unwrap()
                    .as_boolean()
                    .value(i);
                out.insert(mt, (label, redundant));
            }
        }
        out
    };

    let mine = labels(&dir.join("reactions.arrow"));
    let theirs = labels(&ref_dir.join("reactions.arrow"));
    assert!(mine.len() > 5, "only {} reactions were written", mine.len());

    // The canonical names for the synthesized sums (issue #438). The published
    // data used to spell these `(n,non-elastic)`, `(n,inelastic)` and
    // `(n,disappearance)`, which are not the format's names and did not resolve
    // through `REACTION_MT`, so a score read out of a published file was
    // rejected. The converter has written the canonical names since #438, and
    // the 2026-08-21 republish brought the published files into line (issue
    // #439), so this is no longer a deliberate difference: it is the spelling
    // both sides now agree on, pinned here so neither drifts back.
    let renamed: std::collections::HashMap<i32, &str> = [
        (3, "(n,nonelastic)"),
        (4, "(n,level)"),
        (101, "(n,disappear)"),
    ]
    .into_iter()
    .collect();

    // The five redundant MTs must all be present and marked redundant.
    for mt in [1, 3, 4, 27, 101] {
        let (mine_label, mine_redundant) = mine
            .get(&mt)
            .unwrap_or_else(|| panic!("MT {mt} is missing from the conversion"));
        if let Some(expected) = renamed.get(&mt) {
            assert_eq!(
                mine_label, expected,
                "MT {mt} must be labelled with the name the format uses"
            );
        }
        assert!(
            *mine_redundant,
            "MT {mt} is a sum and must be marked redundant"
        );
    }

    // Every other MT must agree with the published file. A disagreement there
    // is the #379 shape: a vocabulary split under one filename. The renamed
    // ones are exempt, and ONLY those: a fourth label drifting would fail here.
    for (mt, (label, _)) in &mine {
        if renamed.contains_key(mt) {
            continue;
        }
        if let Some((their_label, _)) = theirs.get(mt) {
            assert_eq!(
                label, their_label,
                "MT {mt} label differs from the published file"
            );
        }
    }

    // And the renamed ones must really be the labels the published file could
    // not resolve, rather than a rename that quietly grew.
    for (mt, expected) in &renamed {
        if let Some((their_label, _)) = theirs.get(mt) {
            if their_label == expected {
                continue;
            }
            assert!(
                yamc_nuclide::data::REACTION_MT
                    .get(their_label.as_str())
                    .is_none(),
                "MT {mt} is exempted from the published-label check, but the \
                 published label {their_label:?} resolves fine; the exemption \
                 is not needed and is hiding a real difference"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The synthesized total must reproduce the evaluation's own total.
///
/// The closed-form check, and the one that would have caught issue #23: MT 1 is
/// elastic plus everything else, and the ACE table carries its own MT 1 to
/// compare against. A double-counted channel shows up here as a total above the
/// evaluated one, with every individual number still looking reasonable.
#[test]
fn the_synthesized_total_reproduces_the_evaluated_total() {
    let data = li6();
    let temperature = data
        .temperatures()
        .into_iter()
        .next()
        .expect("a temperature");
    let grid = data.energy.get(&temperature).expect("an energy grid");

    let mut partials = std::collections::BTreeMap::new();
    for (&mt, rx) in &data.reactions {
        if yamc_convert::synthesis::SYNTHETIC_MTS.contains(&mt) {
            continue;
        }
        if let Some(xs) = rx.xs.get(&temperature) {
            partials.insert(
                mt,
                yamc_convert::synthesis::on_grid(&xs.y, xs.threshold_idx.unwrap_or(0), grid.len()),
            );
        }
    }
    let built = yamc_convert::synthesis::synthesize(&partials, grid.len());
    let mine = &built[&1];

    let evaluated = data
        .reactions
        .get(&1)
        .and_then(|rx| rx.xs.get(&temperature))
        .expect("the ACE table carries its own MT 1");
    let theirs = yamc_convert::synthesis::on_grid(
        &evaluated.y,
        evaluated.threshold_idx.unwrap_or(0),
        grid.len(),
    );

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (a, b)) in mine.iter().zip(theirs.iter()).enumerate() {
        if *b == 0.0 {
            continue;
        }
        let rel = ((a - b) / b).abs();
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
    }
    assert!(
        worst < 1e-6,
        "synthesized MT 1 differs from the evaluated total by {worst:.3e} at \
         E = {:.6e} eV; a channel is being counted twice or missed",
        grid[worst_at]
    );
    eprintln!("synthesized MT 1 agrees with the evaluated total to {worst:.3e}");
}

/// The whole conversion, from an ACE table to a directory yamc's own reader
/// accepts under the scope an activation calculation uses.
///
/// The point that no column-by-column check makes: the files must LOAD. A
/// conversion can satisfy every schema and still produce something the reader
/// refuses, and the reader is the only judge that matters.
#[test]
fn an_ace_conversion_loads_through_yamcs_own_reader() {
    let dir = scratch("entry");
    let ace = dir.join("Li6.ace");
    std::fs::write(&ace, text(LI6_ACE)).expect("write the ACE table");

    let out = yamc_convert::entry::convert_neutron_xs(
        &yamc_convert::entry::Source::Ace { path: &ace },
        &dir,
        &yamc_convert::entry::Provenance {
            library: "endf-b8.1".to_string(),
            data_version: "rust-check".to_string(),
            created_utc: "2026-08-10T00:00:00+00:00".to_string(),
        },
        false,
    )
    .expect("conversion succeeds");

    assert_eq!(out.file_name().unwrap(), "Li6.arrow");
    for f in ["nuclide.arrow", "reactions.arrow", "version.json"] {
        assert!(out.join(f).is_file(), "{f} was not written");
    }
    // No unresolved range for Li6, and an empty file would be worse than none.
    assert!(!out.join("urr.arrow").exists());

    // version.json carries the stamp that lets a stale cache be invalidated.
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("version.json")).expect("marker"))
            .expect("json");
    assert_eq!(marker["data_version"], "rust-check");
    assert_eq!(marker["format_version"], 1);

    // The conversion must load through the reader a real run uses, under the
    // scope an activation calculation asks for. XsOnly skips products and
    // distributions, which this conversion deliberately does not write, so a
    // Full load would fail for a reason that is not a defect.
    let mts: std::collections::HashSet<i32> = (1..1000).collect();
    let scope = yamc_nuclide::LoadScope::activation(mts);
    let loaded = yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(&out, &scope)
        .expect("yamc reads the converted directory");
    assert_eq!(loaded.name.as_deref(), Some("Li6"));
    assert_eq!(loaded.atomic_number, Some(3));
    assert!(!loaded.reactions.is_empty(), "no reactions were loaded");
    assert!(!loaded.fissionable, "Li6 is not fissionable");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The ENDF route must produce MT 901, which is not in the ACE file.
///
/// Needs NJOY and a real evaluation, so it announces a loud skip rather than
/// passing when either is absent. A heating tally without MT 901 counts only
/// what the neutron deposits and silently drops the photon contribution, which
/// is the whole reason an ENDF conversion is worth the NJOY cost.
#[test]
fn the_endf_route_builds_heating_local() {
    use arrow_ipc::reader::FileReader;

    let evaluation =
        yamc_test_cache::endf_evaluations().join("neutrons-version.VIII.1/n-003_Li_006.endf");
    if !evaluation.is_file() {
        eprintln!("SKIP: no ENDF/B-VIII.1 Li6 evaluation; this test checked nothing");
        return;
    }
    if std::process::Command::new("njoy")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP: no njoy on PATH; this test checked nothing");
        return;
    }

    let dir = scratch("heating");
    let out = yamc_convert::entry::convert_neutron_xs(
        &yamc_convert::entry::Source::Endf {
            path: &evaluation,
            njoy_exec: "njoy",
            temperatures: vec![294.0],
        },
        &dir,
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the ENDF route converts");

    let reader = FileReader::try_new(
        std::fs::File::open(out.join("reactions.arrow")).expect("open"),
        None,
    )
    .expect("ipc");
    let mut mts = Vec::new();
    for batch in reader {
        let batch = batch.expect("batch");
        for i in 0..batch.num_rows() {
            use arrow_array::cast::AsArray;
            mts.push(
                batch
                    .column_by_name("mt")
                    .unwrap()
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .value(i),
            );
        }
    }
    assert!(
        mts.contains(&901),
        "the ENDF route produced no MT 901; the HEATR tapes were not read. MTs: {mts:?}"
    );
    assert!(
        mts.contains(&301),
        "MT 301 is missing, so there was nothing to build MT 901 from"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

const LI6_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");

const U235_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-092_U_235_trimmed.endf.xz");

/// A fissile evaluation's heating correction is driven by MF=1/MT=458, and
/// MT 458 lives in the evaluation rather than in anything NJOY writes.
///
/// This is the shape of a bug that shipped. The release was read from the
/// HEATR tape, which is a PENDF and carries no MT 458, and the failure was
/// dropped with `.ok()`. A `None` release then meant both "this nuclide does
/// not fission" and "the correction could not be built", so no correction was
/// applied to any fissile nuclide at all. Measured on U235: MT 301 3.7% and
/// MT 901 6.7% below the published values, while MT 18 itself agreed to
/// 2.6e-4, so the fission cross section looked right and only the heating was
/// wrong.
///
/// Note what this test does NOT do, since it would be easy to read it as more
/// than it is. It does not run `add_heating_local`, which needs a pair of
/// HEATR tapes and so an NJOY run; there is no PENDF fixture here and a
/// synthetic stand-in would pair a KERMA with a nuclide it does not belong
/// to. What it pins is the contract the fix rests on: MT 458 is present in an
/// evaluation and absent from anything downstream of it, with real terms
/// rather than defaults. The end-to-end evidence is a U235 conversion, after
/// which MT 301 and MT 901 both agree with the published data to 2.6e-4.
///
/// Nor does it assert anything about nu-bar. Passing it makes no difference
/// to U235, whose evaluation gives the tabulated form; it matters for 25 of
/// the 79 fissile evaluations in ENDF/B-VIII.1, which take the prompt neutron
/// term from Sher-Beck and error without it. Those are the minor actinides,
/// and no fixture here is one, so that path is covered by the `.ok()` removal
/// rather than by a test.
#[test]
fn a_fissile_correction_reads_its_energy_release_from_the_evaluation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("u235.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &U235_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&path, raw).expect("fixture writes");

    let material = endf::Material::from_file(&path).expect("U235 parses");
    assert!(
        material.mf1_mt458().is_some(),
        "the fixture must be fissile for this test to mean anything"
    );

    let nu = material
        .mf1_mt452(452)
        .map(|section| &section.nu)
        .expect("a fissile evaluation carries MF=1 MT=452");
    let release = endf::FissionEnergyRelease::from_material(&material, Some(nu))
        .expect("the release builds from the evaluation");

    // The terms the correction multiplies by sigma_f, at thermal. Fragments
    // carry about 169 MeV of the roughly 200 MeV released, betas 6.5 MeV and
    // prompt photons 7.3 MeV. Asserted as values rather than "> 0" because a
    // defaulted zero is exactly the failure being guarded against, and zero
    // would leave the correction subtracting MT 318 and adding nothing back.
    assert!(
        (1.68e8..1.70e8).contains(&release.fragments.eval(0.0253)),
        "fission fragments carry about 169 MeV at thermal, got {:e} eV",
        release.fragments.eval(0.0253)
    );
    assert!(
        (6.0e6..7.0e6).contains(&release.betas.eval(0.0253)),
        "betas carry about 6.5 MeV, got {:e} eV",
        release.betas.eval(0.0253)
    );
    assert!(
        (7.0e6..7.6e6).contains(&release.prompt_photons.eval(0.0253)),
        "prompt photons carry about 7.3 MeV, got {:e} eV",
        release.prompt_photons.eval(0.0253)
    );

    // A non-fissile evaluation must take the other branch rather than error,
    // since most of the library is non-fissile and the correction does not
    // apply to it.
    let li6 = dir.path().join("li6.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &LI6_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&li6, raw).expect("fixture writes");
    let li6 = endf::Material::from_file(&li6).expect("Li6 parses");
    assert!(
        li6.mf1_mt458().is_none(),
        "Li6 does not fission, so the correction must not be attempted for it"
    );
}

/// A transport conversion must load through yamc's reader under a FULL scope.
///
/// The activation test above passes with no products and no distributions,
/// because an `XsOnly` scope never asks for them. This is the check that
/// cannot: a full load reads every section, resolves each product against its
/// distributions by `(mt, product_idx, dist_idx)`, and refuses any discriminant
/// it does not know. A ravel written in the wrong order, an offset off by one,
/// or a type string that is nearly right all fail here and nowhere else.
///
/// Needs NJOY and a real evaluation, so it announces a loud skip rather than
/// passing when either is absent.
#[test]
fn a_transport_conversion_loads_under_a_full_scope() {
    let evaluation =
        yamc_test_cache::endf_evaluations().join("neutrons-version.VIII.1/n-003_Li_006.endf");
    if !evaluation.is_file() {
        eprintln!("SKIP: no ENDF/B-VIII.1 Li6 evaluation; this test checked nothing");
        return;
    }
    if std::process::Command::new("njoy")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP: no njoy on PATH; this test checked nothing");
        return;
    }

    let dir = scratch("transport");
    let out = yamc_convert::entry::convert_neutron_transport(
        &yamc_convert::entry::Source::Endf {
            path: &evaluation,
            njoy_exec: "njoy",
            temperatures: vec![293.6],
        },
        &dir,
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the transport route converts");

    // Every section a transport load needs. urr, total_nu and fission_photon
    // are absent for Li6, which has no unresolved range and does not fission.
    for f in [
        "nuclide.arrow",
        "reactions.arrow",
        "products.arrow",
        "distributions.arrow",
        "fast_xs.arrow",
        "version.json",
    ] {
        assert!(out.join(f).is_file(), "{f} was not written");
    }
    assert!(!out.join("total_nu.arrow").exists(), "Li6 does not fission");
    assert!(
        !out.join("fission_photon.arrow").exists(),
        "Li6 has no fission energy release"
    );

    let loaded = yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
        &out,
        &yamc_nuclide::LoadScope::full(),
    )
    .expect("yamc reads the transport conversion under a full scope");

    assert_eq!(loaded.name.as_deref(), Some("Li6"));
    assert!(!loaded.fissionable, "Li6 is not fissionable");

    // The products and their distributions actually arrived, rather than the
    // sections merely being present and empty.
    // `reactions` is one map per loaded temperature; this conversion asked
    // for one.
    let at_temperature = loaded
        .reactions
        .first()
        .expect("one temperature was loaded");
    let elastic = at_temperature
        .get(&2)
        .expect("MT 2 is in a loaded transport nuclide");
    assert_eq!(
        elastic.products.len(),
        1,
        "elastic scattering emits one neutron"
    );
    assert_eq!(
        elastic.products[0].distribution.len(),
        1,
        "the elastic neutron has an angular distribution; an empty one would \
         make every scatter isotropic without failing to load"
    );

    // MT 102 emits three photons, each with its own discrete line. Checked
    // because it is the row where a mis-ordered ravel would still load.
    let capture = at_temperature.get(&102).expect("MT 102");
    assert_eq!(capture.products.len(), 3, "Li6 capture emits three photons");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Every label a conversion writes must resolve back to its own MT.
///
/// This is the check that was missing. The converter used to invent spellings
/// for the synthesized sums (`(n,non-elastic)` for MT 3, `(n,inelastic)` for
/// MT 4), so the format had two vocabularies and only one of them resolved:
/// `yamc.Tally(scores=["(n,inelastic)"])` was an "Unknown score" even though
/// that string came straight out of `reactions.arrow`. 77 of 80 published
/// nuclides carry the unresolvable spelling (issue #438).
///
/// Nothing reads the label back, so no test caught it. This one asserts the
/// round trip a person makes by hand: read a name out of a file, use it as a
/// score.
#[test]
fn every_label_written_resolves_back_to_its_mt() {
    use arrow_array::cast::AsArray;
    use arrow_ipc::reader::FileReader;

    let dir = scratch("labels");
    let ace = dir.join("Li6.ace");
    std::fs::write(&ace, text(LI6_ACE)).expect("write the ACE table");

    let out = yamc_convert::entry::convert_neutron_xs(
        &yamc_convert::entry::Source::Ace { path: &ace },
        &dir,
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("conversion succeeds");

    let reader = FileReader::try_new(
        std::fs::File::open(out.join("reactions.arrow")).expect("open"),
        None,
    )
    .expect("ipc");

    let mut checked = 0;
    for batch in reader {
        let batch = batch.expect("batch");
        for i in 0..batch.num_rows() {
            let mt = batch
                .column_by_name("mt")
                .expect("mt column")
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(i);
            let label = batch
                .column_by_name("label")
                .expect("label column")
                .as_string::<i32>()
                .value(i)
                .to_string();

            assert_eq!(
                yamc_nuclide::data::REACTION_MT.get(label.as_str()).copied(),
                Some(mt),
                "MT {mt} is written with the label {label:?}, which does not \
                 resolve back to it. A score read out of this file is rejected."
            );
            checked += 1;
        }
    }

    // Li6 has 19 MTs; a conversion that wrote nothing would pass vacuously.
    assert!(
        checked >= 19,
        "only {checked} labels checked; the conversion wrote almost nothing"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A nuclide with fission PARTIALS must still get its total nu-bar.
///
/// The bug this pins: the total nu-bar was looked up on MT 18 alone. The
/// parser attaches the ACE NU block to whichever reaction is marked TY = 19,
/// and for an evaluation that gives chance-by-chance partials rather than the
/// MT 18 total that is MT 19. So `total_nu.arrow` was silently not written for
/// U240 and every nuclide like it, and the reader falls back to treating the
/// fission product's own yield as the nu-bar, which is the PROMPT yield.
///
/// That is a several-percent undercount of fission neutrons with nothing on
/// disk or in a log to say so, which is issue #364 one nuclide at a time.
///
/// Needs NJOY and a real evaluation, so it announces a loud skip.
#[test]
fn a_nuclide_with_fission_partials_still_gets_its_total_nu() {
    let evaluation =
        yamc_test_cache::endf_evaluations().join("neutrons-version.VIII.1/n-092_U_240.endf");
    if !evaluation.is_file() {
        eprintln!("SKIP: no ENDF/B-VIII.1 U240 evaluation; this test checked nothing");
        return;
    }
    if std::process::Command::new("njoy")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP: no njoy on PATH; this test checked nothing");
        return;
    }

    let dir = scratch("totalnu");
    let out = yamc_convert::entry::convert_neutron_transport(
        &yamc_convert::entry::Source::Endf {
            path: &evaluation,
            njoy_exec: "njoy",
            temperatures: vec![293.6],
        },
        &dir,
        &yamc_convert::entry::Provenance::default(),
        false,
    )
    .expect("the transport route converts");

    assert!(
        out.join("total_nu.arrow").is_file(),
        "U240 gives fission partials rather than an MT 18 total, and its total \
         nu-bar was not written; the reader will fall back to the prompt yield"
    );

    // And it must be the real nu-bar rather than something defaulted. U240's
    // total nu runs from about 2.5 at thermal to 6.4 at 30 MeV, where its
    // PROMPT yield on MT 18 is the constant 8.18 the ACE table carries. The
    // two are not confusable, which is the point of asserting the value.
    let batch = {
        use arrow_ipc::reader::FileReader;
        let mut r = FileReader::try_new(
            std::fs::File::open(out.join("total_nu.arrow")).expect("open"),
            None,
        )
        .expect("ipc");
        r.next().expect("a row").expect("batch")
    };
    use arrow_array::cast::AsArray;
    let data: Vec<f64> = batch
        .column_by_name("yield_data")
        .expect("yield_data")
        .as_list::<i32>()
        .value(0)
        .as_primitive::<arrow_array::types::Float64Type>()
        .values()
        .to_vec();
    assert!(!data.is_empty(), "the total nu-bar is empty");
    let n = data.len() / 2;
    let (x, y) = (&data[..n], &data[n..]);
    assert!(
        (2.0..3.0).contains(&y[0]),
        "the total nu-bar at {:.3e} eV is {}, not the ~2.5 U240 releases at \
         thermal; this looks like the prompt yield or a default",
        x[0],
        y[0]
    );
    assert!(
        y[n - 1] > y[0],
        "the total nu-bar must rise with incident energy"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The photon route writes all four sections and they load through the reader.
///
/// A photoatomic evaluation needs no NJOY, so unlike the neutron tests this
/// one only needs the data to be on disk. It checks the two things a column
/// comparison does not: that the auxiliary tabulations shipped in the wheel
/// are found and parsed, and that `compton.arrow` is written, which depends on
/// the subshell rows being in most-bound-first order.
#[test]
fn the_photon_route_writes_every_section() {
    let root = yamc_test_cache::endf_evaluations();
    let photoatomic = root.join("photoat-version.VIII.1/photoat-026_Fe_000.endf");
    let relaxation = root.join("atomic_relax-version.VIII.1/atom-026_Fe_000.endf");
    if !photoatomic.is_file() || !relaxation.is_file() {
        eprintln!("SKIP: no ENDF/B-VIII.1 Fe photoatomic data; this test checked nothing");
        return;
    }

    // The tabulations that ship in the wheel. No evaluation carries them.
    let data =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/yamc-core/python/yamc/data");
    let compton = data.join("compton_profiles_biggs1975.txt");
    let density = data.join("density_effect_sternheimer1982.txt");
    let brem = data.join("bremsstrahlung_seltzer_berger1986.txt");
    assert!(
        compton.is_file() && density.is_file() && brem.is_file(),
        "the auxiliary photon tabulations are missing from packages/yamc-core/python/yamc/data"
    );

    let dir = scratch("photon");
    let written = yamc_convert::entry::convert_photon(
        &photoatomic,
        Some(&relaxation),
        Some(&yamc_convert::entry::PhotonTabulations {
            compton_profiles: &compton,
            density_effect: &density,
            bremsstrahlung: &brem,
        }),
        &dir,
        &yamc_convert::entry::Provenance::default(),
    )
    .expect("the photon route converts");

    assert_eq!(
        written.len(),
        1,
        "one element in the file, one directory out"
    );
    let out = &written[0];
    assert_eq!(out.file_name().unwrap(), "Fe.arrow");
    for f in [
        "element.arrow",
        "subshells.arrow",
        "compton.arrow",
        "bremsstrahlung.arrow",
        "version.json",
    ] {
        assert!(out.join(f).is_file(), "{f} was not written");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
