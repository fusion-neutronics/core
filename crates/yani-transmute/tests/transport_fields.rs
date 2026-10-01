//! The transport reactions each cached evaluation covers.
//!
//! Every non-redundant reaction a nuclide's transport data holds must be
//! classified, none of the redundant sums may be, and every reaction read
//! through a sum must name one the field actually has cells for. Skipped per
//! evaluation where the cache does not carry it.

use std::collections::{BTreeSet, HashMap};

use yamc_materials::Material;
use yani_transmute::covariance_fold::{transport_fields, Read};

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

#[test]
fn every_transport_partial_is_classified_and_reads_cells_that_exist() {
    let mut checked = 0;
    for library in [
        "endf-b8.1",
        "jeff-4.0",
        "tendl-2017",
        "tendl-2025",
        "fendl-3.2d",
    ] {
        for nuclide in ["Fe56", "Cr52", "Co59", "W186", "Pb208", "Li6"] {
            let dir = yamc_test_cache::root().join(format!("{library}-{nuclide}.arrow"));
            if !(dir.join("covariance.arrow").is_file() && dir.join("reactions.arrow").is_file()) {
                continue;
            }
            let m = material(nuclide, &dir);
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
            let short: BTreeSet<i32> = t
                .field
                .iter()
                .flat_map(|f| f.short.iter().map(|s| s.mt))
                .collect();
            let (mut own, mut parent, mut nominal) = (0, 0, 0);
            for (mt, read) in &t.reads {
                match read {
                    Read::Own => own += 1,
                    Read::Parent(sum) => {
                        parent += 1;
                        assert!(
                            cells.contains(sum) || short.contains(sum),
                            "{library} {nuclide} MT {mt} reads MT {sum}, which has no cells"
                        );
                    }
                    Read::Nominal => nominal += 1,
                }
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
    }
    if checked == 0 {
        eprintln!("skipping: no cached evaluation with covariance");
    }
}
