//! The transport reactions each cached evaluation covers.
//!
//! Every non-redundant reaction a nuclide's transport data holds must be
//! classified, none of the redundant sums may be, and every reaction read as
//! perturbed must be one a draw actually moves: one read as its own needs
//! cells of its own or of a reaction an NC block derives it from, over the
//! derivation's range, and one read through a sum needs the sum to have
//! cells. Short-range (`lb = 8`) blocks do not count, since transport holds
//! their noise at nominal. Skipped per evaluation where the cache does not
//! carry it, except for the committed ENDF/B-VIII.1 MF=33 tapes of Pb208 and
//! Be9, whose elastic and level reactions only NC blocks cover, converted
//! onto the fetched fixtures.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use yamc_materials::Material;
use yani_transmute::covariance_fold::{transport_fields, Read, TransportField};

const PB208: &[u8] = include_bytes!("../../endf/fixtures/n-082_Pb_208_endfb81_mf33.endf.xz");
const BE9: &[u8] = include_bytes!("../../endf/fixtures/n-004_Be_009_endfb81_mf33.endf.xz");

/// A copy of the fetched `nuclide` fixture with `covariance.arrow` written
/// from the committed MF=33 `tape`, or `None` when the fixture is not
/// fetched.
fn with_covariance(nuclide: &str, tape: &[u8], tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide(nuclide)?);
    let dir = tmp.join(format!("{nuclide}.arrow"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached fixture") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() && entry.file_name() != "covariance.arrow" {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &tape[..], &mut raw).expect("tape decompresses");
    let text = String::from_utf8(raw).expect("ENDF is text");
    let evaluation = endf::Material::from_str(&text).expect("tape parses");
    assert!(yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("writes"));
    Some(dir)
}

fn material(nuclide: &str, dir: &std::path::Path) -> Material {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.density = Some(8.0);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read nuclear data");
    m.ensure_covariance_loaded().expect("read covariance");
    m
}

/// Whether a draw moves reaction `mt` of `t` anywhere: its own cells, or a
/// derived term's cells over the term's range. Short-range blocks are not
/// cells.
fn perturbable(t: &TransportField, mt: i32) -> bool {
    let Some(f) = &t.field else {
        return false;
    };
    let cells = || f.relative_cells.iter().chain(&f.absolute_cells);
    let own = cells().any(|c| c.mt == mt);
    // Every derived term is kept only where it reaches cells.
    let derived = t.derived.get(&mt).is_some_and(|terms| {
        !terms.is_empty()
            && terms
                .iter()
                .all(|d| cells().any(|c| c.mt == d.mt && c.lo < d.range.1 && c.hi > d.range.0))
    });
    own || derived
}

#[test]
fn every_transport_partial_is_classified_and_reads_cells_that_exist() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let mut evaluations: Vec<(String, String, PathBuf)> = Vec::new();
    for (nuclide, tape) in [("Pb208", PB208), ("Be9", BE9)] {
        match with_covariance(nuclide, tape, tmp.path()) {
            Some(dir) => evaluations.push((
                "committed ENDF/B-VIII.1".to_string(),
                nuclide.to_string(),
                dir,
            )),
            None => eprintln!("skipping the committed {nuclide} tape: fixture not fetched"),
        }
    }
    for library in [
        "endf-b8.1",
        "jeff-4.0",
        "tendl-2017",
        "tendl-2025",
        "fendl-3.2d",
    ] {
        for nuclide in [
            "Fe56", "Cr52", "Co59", "W186", "Pb206", "Pb207", "Pb208", "Li6", "Be9", "C12", "O16",
            "F19", "Ti49",
        ] {
            let dir = yamc_test_cache::root().join(format!("{library}-{nuclide}.arrow"));
            if dir.join("covariance.arrow").is_file() && dir.join("reactions.arrow").is_file() {
                evaluations.push((library.to_string(), nuclide.to_string(), dir));
            }
        }
    }
    let mut checked = 0;
    for (library, nuclide, dir) in &evaluations {
        let nuclide = nuclide.as_str();
        let m = material(nuclide, dir);
        let (fields, without) = transport_fields(&m);
        let Some(t) = fields.get(nuclide) else {
            assert!(
                without.contains(nuclide),
                "{library} {nuclide} is neither covered nor named"
            );
            continue;
        };
        let data = &m.nuclide_data[nuclide];
        let held = data.reactions_for_temp("294").expect("294 K");
        let partials: BTreeSet<i32> = held
            .iter()
            .filter(|(_, r)| !r.redundant)
            .map(|(mt, _)| *mt)
            .collect();
        let classified: BTreeSet<i32> = t.reads.keys().copied().collect();
        assert_eq!(
            classified, partials,
            "{library} {nuclide}: every partial, and only partials"
        );

        let cells: BTreeSet<i32> = t
            .field
            .iter()
            .flat_map(|f| {
                f.relative_cells
                    .iter()
                    .chain(&f.absolute_cells)
                    .map(|c| c.mt)
            })
            .collect();
        let (mut own, mut parent, mut nominal) = (0, 0, 0);
        for (mt, read) in &t.reads {
            match read {
                Read::Own => {
                    own += 1;
                    assert!(
                        perturbable(t, *mt),
                        "{library} {nuclide} MT {mt} is read as its own, but no draw moves it"
                    );
                }
                Read::Parent(sum) => {
                    parent += 1;
                    assert!(
                        cells.contains(sum),
                        "{library} {nuclide} MT {mt} reads MT {sum}, which has no cells"
                    );
                }
                Read::Nominal => nominal += 1,
            }
        }
        for mt in t.derived.keys() {
            assert_eq!(
                t.reads.get(mt),
                Some(&Read::Own),
                "{library} {nuclide} MT {mt}: only a reaction read as its own is derived"
            );
        }
        eprintln!(
            "{library} {nuclide} nominal: {:?}",
            t.reads
                .iter()
                .filter(|(_, r)| **r == Read::Nominal)
                .map(|(m, _)| *m)
                .collect::<Vec<_>>()
        );
        eprintln!(
            "{library} {nuclide}: {} partials, {own} own, {parent} via a sum, {nominal} nominal, {} cells",
            t.reads.len(),
            t.field.as_ref().map_or(0, |f| f.relative_cells.len() + f.absolute_cells.len())
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipping: no cached evaluation with covariance");
    }
}
