//! Nuclear data with one replica's draw applied.
//!
//! On ENDF/B-VIII.1 Fe56, whose MF=33 covers elastic, capture and the total
//! inelastic: a zero draw must leave every row and the lookup bit-identical;
//! capture raised by 10% must move capture, the stored MT 1 and MT 101, and
//! the lookup's absorption column by exactly that, and leave elastic alone;
//! and the change must reach a transport run. Self-skips where the fixture
//! carries no covariance.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::xs_perturbation::{perturbed_material, perturbed_nuclide};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::covariance_fold::{transport_fields, Read, TransportField};
use yani_transmute::covariance_sample::Sampler;

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Fe56.arrow");
    dir.join("covariance.arrow").is_file().then_some(dir)
}

fn iron(dir: &std::path::Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Fe56");
    m.ensure_covariance_loaded().expect("read covariance");
    m
}

fn fe56_field(m: &Material) -> TransportField {
    let (fields, _) = transport_fields(m);
    fields.get("Fe56").cloned().expect("Fe56 has covariance")
}

/// `m_k - 1` of `rise` on every relative cell of `mt`, zero elsewhere.
fn raise(field: &TransportField, mt: i32, rise: f64) -> (Vec<f64>, Vec<f64>) {
    let f = field.field.as_ref().expect("a field");
    let relative = f
        .relative_cells
        .iter()
        .map(|c| if c.mt == mt { rise } else { 0.0 })
        .collect();
    (relative, vec![0.0; f.absolute_cells.len()])
}

#[test]
fn a_zero_draw_changes_nothing() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let field = fe56_field(&m);
    let (relative, absolute) = raise(&field, 102, 0.0);
    let nominal = &m.nuclide_data["Fe56"];
    let (perturbed, floored) =
        perturbed_nuclide(nominal, &field, &relative, &absolute).expect("perturb");
    assert_eq!(floored, 0);
    for (t, rows) in nominal.reactions.iter().enumerate() {
        for (mt, r) in rows {
            assert_eq!(
                r.cross_section.as_slice(),
                perturbed.reactions[t][mt].cross_section.as_slice(),
                "MT {mt} moved under a zero draw"
            );
        }
        assert_eq!(
            nominal.fast_xs[t].xs, perturbed.fast_xs[t].xs,
            "the lookup moved"
        );
    }
}

#[test]
fn raising_capture_moves_capture_and_its_sums_by_exactly_that() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let field = fe56_field(&m);
    assert_eq!(
        field.reads[&102],
        Read::Own,
        "Fe56 capture states its own covariance"
    );
    let (relative, absolute) = raise(&field, 102, 0.1);
    let nominal = &m.nuclide_data["Fe56"];
    let (perturbed, _) = perturbed_nuclide(nominal, &field, &relative, &absolute).expect("perturb");

    let t = nominal
        .loaded_temperatures
        .iter()
        .position(|l| l == "294")
        .expect("294 K");
    let grid = nominal.energy.as_ref().unwrap()["294"].as_slice();
    let cells = field.field.as_ref().unwrap();
    // A cell is `[lo, hi)`, the last one closed at the top, as the
    // perturbation reads it.
    let capture: Vec<_> = cells
        .relative_cells
        .iter()
        .filter(|c| c.mt == 102)
        .collect();
    let top = capture
        .iter()
        .map(|c| c.hi)
        .fold(f64::NEG_INFINITY, f64::max);
    let covered = |e: f64| capture.iter().any(|c| c.lo <= e && (e < c.hi || e == top));
    let row = |n: &yamc_nuclide::nuclide::Nuclide, mt: i32| -> Vec<f64> {
        let r = &n.reactions[t][&mt];
        let mut out = vec![0.0; grid.len()];
        out[r.threshold_idx..r.threshold_idx + r.cross_section.len()]
            .copy_from_slice(r.cross_section.as_slice());
        out
    };
    let (c0, c1) = (row(nominal, 102), row(&perturbed, 102));
    let (t0, t1) = (row(nominal, 1), row(&perturbed, 1));
    let (e0, e1) = (row(nominal, 2), row(&perturbed, 2));
    let mut inside = 0;
    for i in 0..grid.len() {
        let want = if covered(grid[i]) { 1.1 * c0[i] } else { c0[i] };
        assert!(
            (c1[i] - want).abs() <= 1e-12 * want.abs(),
            "capture at {} eV",
            grid[i]
        );
        inside += usize::from(covered(grid[i]));
        // The stored total moves by the capture change and nothing else.
        let d = c1[i] - c0[i];
        assert!(
            ((t1[i] - t0[i]) - d).abs() <= 1e-9 * t0[i].abs().max(1e-30),
            "MT 1 at {} eV moved by {} for a capture change of {d}",
            grid[i],
            t1[i] - t0[i]
        );
        assert_eq!(e0[i], e1[i], "elastic moved at {} eV", grid[i]);
        // The lookup's disappearance column carries the same change.
        let a0 = nominal.fast_xs[t].xs[i][1];
        let a1 = perturbed.fast_xs[t].xs[i][1];
        assert!(
            ((a1 - a0) - d).abs() <= 1e-9 * a0.abs().max(1e-30),
            "lookup absorption"
        );
    }
    assert!(inside > 100, "the capture cells cover the grid");
    if nominal.reactions[t].contains_key(&101) {
        let (s0, s1) = (row(nominal, 101), row(&perturbed, 101));
        for i in 0..grid.len() {
            let d = c1[i] - c0[i];
            assert!(
                ((s1[i] - s0[i]) - d).abs() <= 1e-9 * s0[i].abs().max(1e-30),
                "MT 101"
            );
        }
    }
}

/// A capture rate scored on a small iron sphere, nominal and with a material.
fn capture_rate(material: Material) -> f64 {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 0.2,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).expect("geometry");
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.scores = vec!["(n,gamma)".parse::<Score>().expect("score")];
    tally.estimator = Estimator::TrackLength;
    tally.initialize_batches(1);
    let tally = Arc::new(tally);
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![0.0253], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(20_000),
            seed: 7,
            ..Default::default()
        })
        .expect("transport");
    tally.get_mean().iter().sum()
}

/// On a sphere small enough that most neutrons leave uncollided, the same
/// histories cross it and the capture rate scales with the capture cross
/// section: raised by 10%, it reads 10% higher.
#[test]
fn a_perturbed_material_reaches_transport() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let field = fe56_field(&m);
    let (relative, absolute) = raise(&field, 102, 0.1);
    let nominal_rate = capture_rate(m.clone());

    let mut perturbed = m.clone();
    let (nuclide, _) =
        perturbed_nuclide(&m.nuclide_data["Fe56"], &field, &relative, &absolute).expect("perturb");
    perturbed
        .nuclide_data
        .insert("Fe56".to_string(), Arc::new(nuclide));
    perturbed.invalidate_xs_cache();
    let perturbed_rate = capture_rate(perturbed);

    let ratio = perturbed_rate / nominal_rate;
    assert!(
        (ratio - 1.1).abs() < 0.01,
        "capture rate moved by {ratio} for a 10% capture rise"
    );
}

/// A real draw through the sampler: the perturbed material differs from the
/// nominal, its stored total still equals the nominal total plus the change
/// in its partials, and the nominal material is untouched.
#[test]
fn a_sampled_draw_perturbs_a_copy_and_keeps_the_totals_consistent() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let (fields, _) = transport_fields(&m);
    let cells = fields
        .iter()
        .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
        .collect();
    let sampler = Sampler::new(&cells, &[]);
    let draw = sampler.draw(3, 0);
    let before = m.nuclide_data["Fe56"].reactions.clone();
    let (perturbed, _) = perturbed_material(&m, &fields, &draw).expect("perturb");

    assert!(
        perturbed.fast_xs.is_none(),
        "the tables are cleared for a rebuild"
    );
    let t = 0;
    let p = &perturbed.nuclide_data["Fe56"].reactions[t];
    let n = &m.nuclide_data["Fe56"].reactions[t];
    assert_ne!(
        p[&2].cross_section.as_slice(),
        n[&2].cross_section.as_slice(),
        "a draw moves elastic"
    );
    for (mt, r) in &before[t] {
        assert_eq!(
            r.cross_section.as_slice(),
            n[mt].cross_section.as_slice(),
            "nominal MT {mt} touched"
        );
    }
}
