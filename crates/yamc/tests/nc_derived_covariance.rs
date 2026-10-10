//! Transport reactions whose covariance an NC block derives from others.
//!
//! ENDF/B-VIII.1 gives Pb208 elastic above 1.5 MeV only as
//! `σ_1 - σ_4 - σ_16 - σ_102`, and Be9 elastic, `(n,p0)`, `(n,d0)`, `(n,t0)`
//! and `(n,α0)` only through their sums. None of them has cells of its own
//! there, so a run that reads only a reaction's own cells holds them at
//! nominal while calling them perturbed. These check that a rerun moves them
//! by the named reactions' changes, that the reaction rates move as the
//! activation fold says they do under the same perturbation, and that the
//! coverage report gives them the sigma the run applies.
//!
//! The covariance is converted from committed MF=33 tapes onto the fetched
//! fixtures, so these run wherever the fixtures do.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc::xs_perturbation::perturbed_nuclide;
use yamc_materials::Material;
use yamc_nuclide::reaction::Reaction;
use yani_transmute::covariance_fold::{cell_fields, transport_fields, FoldSpectrum, Read};

const PB208: &[u8] = include_bytes!("../../endf/fixtures/n-082_Pb_208_endfb81_mf33.endf.xz");
const BE9: &[u8] = include_bytes!("../../endf/fixtures/n-004_Be_009_endfb81_mf33.endf.xz");

/// A copy of the fetched `nuclide` fixture with `covariance.arrow` written
/// from the committed ENDF/B-VIII.1 MF=33 `tape`, or `None` when the fixture
/// is not fetched.
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
    assert!(
        yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance writes"),
        "the tape must carry MF=33 for these tests to mean anything"
    );
    Some(dir)
}

fn material(nuclide: &str, dir: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(8.0),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read nuclear data");
    m.ensure_covariance_loaded().expect("read covariance");
    m
}

/// A cross section on the full grid of `n` points, zero below threshold.
fn on_grid(r: &Reaction, n: usize) -> Vec<f64> {
    yamc_nuclide::synthesis::on_grid(r.cross_section.as_slice(), r.threshold_idx, n)
}

/// A relative change per reaction, the same on every one of its cells, so
/// two fields that cut a reaction's cells differently read the same
/// perturbation of its cross section.
const PB208_CHANGE: [(i32, f64); 4] = [(1, 0.03), (4, -0.05), (16, 0.07), (102, 0.11)];

fn per_cell(cells: &[yani_transmute::covariance_fold::Cell], change: &[(i32, f64)]) -> Vec<f64> {
    let by_mt: BTreeMap<i32, f64> = change.iter().copied().collect();
    cells
        .iter()
        .map(|c| by_mt.get(&c.mt).copied().unwrap_or(0.0))
        .collect()
}

#[test]
fn pb208_elastic_reads_its_derivation_from_the_total() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = with_covariance("Pb208", PB208, tmp.path()) else {
        eprintln!("skipping: Pb208 fixture not fetched");
        return;
    };
    let m = material("Pb208", &dir);
    let (fields, _) = transport_fields(&m);
    let t = &fields["Pb208"];
    assert_eq!(t.reads[&2], Read::Own);
    let terms: Vec<(f64, i32, (f64, f64))> = t.derived[&2]
        .iter()
        .map(|d| (d.coefficient, d.mt, d.range))
        .collect();
    let range = (1.5e6, 2.0e7);
    assert_eq!(
        terms,
        vec![
            (1.0, 1, range),
            (-1.0, 4, range),
            (-1.0, 16, range),
            (-1.0, 102, range)
        ]
    );
    let field = t.field.as_ref().expect("a field");
    assert!(
        !field.relative_cells.iter().any(|c| c.mt == 2),
        "Pb208 elastic has no cells of its own: only the derivation moves it"
    );
}

/// A rerun moves Pb208 elastic by exactly `Σ c_t δσ_t` inside the
/// derivation's range and leaves it alone outside, and the total still moves
/// by exactly what its partials moved.
#[test]
fn a_rerun_moves_pb208_elastic_by_the_named_reactions_changes() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = with_covariance("Pb208", PB208, tmp.path()) else {
        eprintln!("skipping: Pb208 fixture not fetched");
        return;
    };
    let m = material("Pb208", &dir);
    let (fields, _) = transport_fields(&m);
    let t = &fields["Pb208"];
    let field = t.field.as_ref().expect("a field");
    let relative = per_cell(&field.relative_cells, &PB208_CHANGE);
    let absolute = vec![0.0; field.absolute_cells.len()];
    let data = &m.nuclide_data["Pb208"];
    let (p, floored) = perturbed_nuclide(data, t, &relative, &absolute).expect("perturb");
    assert_eq!(floored, 0);

    let ti = data
        .loaded_temperatures
        .iter()
        .position(|x| x == "294")
        .expect("294 K");
    let energies = data.energy.as_ref().expect("grid")["294"].as_slice();
    let n = energies.len();
    let (before, after) = (&data.reactions[ti], &p.reactions[ti]);
    let xs = |map: &HashMap<i32, Arc<Reaction>>, mt: i32| on_grid(&map[&mt], n);
    let (b2, a2) = (xs(before, 2), xs(after, 2));
    let named: Vec<(f64, Vec<f64>, f64)> = [(1.0, 1), (-1.0, 4), (-1.0, 16), (-1.0, 102)]
        .iter()
        .map(|&(c, mt)| {
            let change = PB208_CHANGE.iter().find(|x| x.0 == mt).expect("listed").1;
            (c, xs(before, mt), change)
        })
        .collect();
    let mut inside = 0;
    for i in 0..n {
        let e = energies[i];
        let expected = if (1.5e6..2.0e7).contains(&e) {
            inside += 1;
            b2[i]
                + named
                    .iter()
                    .map(|(c, x, change)| c * change * x[i])
                    .sum::<f64>()
        } else {
            b2[i]
        };
        assert!(
            (a2[i] - expected).abs() <= 1e-12 * b2[i].abs().max(1.0),
            "MT 2 at {e} eV: {} against {expected}",
            a2[i]
        );
    }
    assert!(inside > 100, "the derivation covers the fast range");

    // The total moves by what its partials moved, which on the derived range
    // is what MT 1's own cells say, up to how far the stored MT 4 is from
    // the sum of its levels.
    let (b1, a1) = (xs(before, 1), xs(after, 1));
    let mut parts = vec![0.0; n];
    for (mt, r) in before.iter() {
        if r.redundant || yamc_nuclide::synthesis::SYNTHETIC_MTS.contains(mt) {
            continue;
        }
        let (b, a) = (on_grid(r, n), on_grid(&after[mt], n));
        for i in 0..n {
            parts[i] += a[i] - b[i];
        }
    }
    let mut worst: f64 = 0.0;
    for i in 0..n {
        assert!(
            ((a1[i] - b1[i]) - parts[i]).abs() <= 1e-9 * b1[i],
            "MT 1 moves by its partials at {} eV",
            energies[i]
        );
        if (2.0e6..2.0e7).contains(&energies[i]) {
            worst = worst.max(((a1[i] - b1[i]) / b1[i] - 0.03).abs());
        }
    }
    assert!(
        worst < 1e-3,
        "above 2 MeV Pb208's total moves by MT 1's +3%, off by {worst}"
    );
}

/// The `(n,p)` and `(n,t)` rates of Be9 move by the same relative amount
/// under a transport rerun as the activation fold's projection says, for the
/// same perturbation of the cross sections. Transport samples `(n,p0)`,
/// which ENDF/B-VIII.1 derives from MT 103, and `(n,t0)`, derived as
/// `σ_105 - σ_701`; activation reads MT 103 and MT 105 directly.
#[test]
fn be9_activation_and_transport_move_the_same_rates_by_the_same_amount() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = with_covariance("Be9", BE9, tmp.path()) else {
        eprintln!("skipping: Be9 fixture not fetched");
        return;
    };
    let m = material("Be9", &dir);
    let change = [
        (1, 0.02),
        (2, -0.01),
        (16, 0.04),
        (102, 0.08),
        (103, 0.13),
        (104, -0.06),
        (105, 0.17),
        (107, -0.09),
        (701, -0.23),
    ];

    // Transport: a rerun's redundant MT 103 and MT 105, rebuilt from the
    // partials, integrated over a flat flux.
    let (fields, _) = transport_fields(&m);
    let t = &fields["Be9"];
    for mt in [2, 600, 650, 700, 800] {
        assert_eq!(t.reads[&mt], Read::Own, "MT {mt}");
        assert!(t.derived.contains_key(&mt), "MT {mt} reads a derivation");
    }
    let field = t.field.as_ref().expect("a field");
    let relative = per_cell(&field.relative_cells, &change);
    let absolute = vec![0.0; field.absolute_cells.len()];
    let data = &m.nuclide_data["Be9"];
    let (p, _) = perturbed_nuclide(data, t, &relative, &absolute).expect("perturb");
    let ti = data
        .loaded_temperatures
        .iter()
        .position(|x| x == "294")
        .expect("294 K");
    let energies = data.energy.as_ref().expect("grid")["294"].as_slice();
    let n = energies.len();
    let integral = |v: &[f64]| -> f64 {
        (0..n - 1)
            .map(|i| 0.5 * (v[i] + v[i + 1]) * (energies[i + 1] - energies[i]))
            .sum()
    };
    let transport_change = |mt: i32| -> f64 {
        let (b, a) = (
            on_grid(&data.reactions[ti][&mt], n),
            on_grid(&p.reactions[ti][&mt], n),
        );
        let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        integral(&d) / integral(&b)
    };

    // Activation: the projection of the same change, over one flat group.
    let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = yani::parse_chain_arrow(&chain_path).expect("parse chain");
    let groups = [energies[0], energies[n - 1]];
    let flux = [1.0];
    let rates: yani::ReactionRates = HashMap::from([(
        "Be9".to_string(),
        HashMap::from([("(n,p)".to_string(), 1.0), ("(n,t)".to_string(), 1.0)]),
    )]);
    let spectrum = FoldSpectrum {
        chain: &chain,
        rates: &rates,
        multigroup_flux: &flux,
        group_boundaries: &groups,
    };
    let activation = cell_fields(&m, &chain, &[spectrum], None, &BTreeSet::new());
    let a = &activation["Be9"];
    let projection = a.projections[0].as_ref().expect("a projection");
    let dm = per_cell(&a.relative_cells, &change);
    let nr = a.relative_cells.len();
    let width = groups[1] - groups[0];
    for (kind, mt) in [("(n,p)", 103), ("(n,t)", 105)] {
        let i = projection
            .kinds
            .iter()
            .position(|k| k == kind)
            .expect("a Be9 channel");
        let moved: f64 = (0..nr)
            .map(|k| projection.relative[i * nr + k] * dm[k])
            .sum();
        // The projection's partial rates are in 1/s, cross sections in cm^2
        // times the flux density over the group; the rerun's integral is in
        // barn eV.
        let nominal = 1e-24 * integral(&on_grid(&data.reactions[ti][&mt], n)) / width;
        let activation_change = moved / nominal;
        let transport = transport_change(mt);
        eprintln!("Be9 {kind}: activation {activation_change:.6}, transport {transport:.6}");
        assert!(
            (activation_change - transport).abs() < 1e-6,
            "Be9 {kind}: activation {activation_change} against transport {transport}"
        );
    }
}

/// A one-cell model of `m`, for the coverage report.
fn model_of(m: Material) -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).expect("geometry");
    Model::new(geometry, vec![], vec![])
}

/// The coverage report gives a derived reaction the sigma a run applies to
/// it, not the zero its absent own cells would.
#[test]
fn coverage_reports_the_sigma_of_a_derived_reaction() {
    let tmp = tempfile::tempdir().expect("temp dir");
    for (nuclide, tape, mts) in [
        ("Pb208", PB208, vec![2]),
        ("Be9", BE9, vec![2, 600, 650, 700, 800]),
    ] {
        let Some(dir) = with_covariance(nuclide, tape, tmp.path()) else {
            eprintln!("skipping: {nuclide} fixture not fetched");
            continue;
        };
        let mut model = model_of(material(nuclide, &dir));
        let report = model.data_uncertainty_coverage().expect("coverage");
        let coverage = &report.nuclides[nuclide];
        let reported: BTreeSet<i32> = coverage.perturbed.keys().copied().collect();
        for mt in mts {
            assert!(reported.contains(&mt), "{nuclide} MT {mt} is perturbed");
            let sigma = coverage.perturbed[&mt].max_relative_sigma;
            eprintln!("{nuclide} MT {mt}: max relative sigma {sigma:.4}");
            assert!(
                sigma > 1e-3 && sigma.is_finite(),
                "{nuclide} MT {mt} reports sigma {sigma}"
            );
        }
    }
}
