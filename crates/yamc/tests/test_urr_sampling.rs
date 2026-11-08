//! URR (unresolved resonance region) sampling regression tests on the
//! CPU path. Co58 is the only URR-bearing nuclide fixture in the test
//! set: 10 keV sits inside its URR window and 14 MeV is far above it
//! (see gpu_urr.rs). These pin the contract of the shared URR
//! adjustment (`urr_adjusted_reaction_xs`, reached here through
//! `Nuclide::collision_xs`): inside the window the probability-table
//! adjustment must engage and follow the supplied correlated random;
//! outside the window the smooth grid columns must come back untouched.
//! The transport-level test guards analog-vs-survival agreement with
//! every collision routed through the URR-adjusted reaction path.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc::util::fast_rng::FastRng;
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

/// Energy (eV) inside Co58's URR window (same choice as gpu_urr.rs).
const E_IN_URR: f64 = 10_000.0;
/// Energy (eV) far above Co58's URR range.
const E_ABOVE_URR: f64 = 14.0e6;

fn co58_material(density: f64) -> Arc<Material> {
    let mut material = Material::new(
        HashMap::from([("Co58".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Co58".to_string(), "tests/Co58.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(material)
}

#[test]
fn collision_xs_engages_urr_inside_window() {
    let material = co58_material(8.9);
    let nuclide = material.nuclide_data.get("Co58").unwrap();
    assert!(nuclide.urr_present, "Co58 fixture must carry URR tables");

    let mut rng = FastRng::new(1);

    // Two different correlated randoms must select different probability
    // table bands: proves the adjustment is engaged AND driven by the
    // supplied random rather than an internal draw.
    let lo = nuclide
        .collision_xs(E_IN_URR, "294", Some(0.05), &mut rng)
        .unwrap();
    let hi = nuclide
        .collision_xs(E_IN_URR, "294", Some(0.95), &mut rng)
        .unwrap();
    assert_ne!(
        lo.total, hi.total,
        "URR adjustment not engaged at {E_IN_URR} eV (low/high bands identical)"
    );

    // Same random twice -> identical result (deterministic, no hidden
    // RNG consumption when the correlated random is supplied).
    let again = nuclide
        .collision_xs(E_IN_URR, "294", Some(0.05), &mut rng)
        .unwrap();
    assert_eq!(lo.total, again.total);
    assert_eq!(lo.scatter, again.scatter);
    assert_eq!(lo.fission, again.fission);
}

#[test]
fn collision_xs_returns_smooth_above_window() {
    let material = co58_material(8.9);
    let nuclide = material.nuclide_data.get("Co58").unwrap();
    let temp_idx = nuclide.get_temp_idx("294").unwrap();
    let fast_grid = &nuclide.fast_xs[temp_idx];

    let (smooth_total, _smooth_abs, smooth_scatter, smooth_fission) = fast_grid.lookup(E_ABOVE_URR);

    let mut rng = FastRng::new(1);
    let xs = nuclide
        .collision_xs(E_ABOVE_URR, "294", Some(0.5), &mut rng)
        .unwrap();

    // Outside the URR window the smooth grid columns must come back
    // bit-identical (no adjustment applied).
    assert_eq!(xs.total, smooth_total);
    assert_eq!(xs.scatter, smooth_scatter);
    assert_eq!(xs.fission, smooth_fission);
}

/// Co58 sphere (radius 5 cm) with a mono-energetic source inside the
/// URR window, so the URR-adjusted reaction path dominates collisions.
/// Cell tallies: track-length flux and MT-101 absorption rate.
fn co58_sphere_model(survival: bool, particles: usize, seed: u64) -> (Model, TransportSettings) {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![co58_material(8.9)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![E_IN_URR], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut flux = Tally::new();
    flux.filters.push(cell_filter.clone());
    flux.scores = vec![Score::Flux(FluxScore)];
    flux.name = Some("flux".to_string());

    let mut absorption = Tally::new();
    absorption.filters.push(cell_filter);
    absorption.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    absorption.name = Some("absorption".to_string());

    let mut model = Model::new(
        geometry,
        vec![source],
        vec![Arc::new(flux), Arc::new(absorption)],
    );
    if survival {
        model.variance_reduction = vec![VarianceReduction::SurvivalBiasing(
            SurvivalBiasing::default(),
        )];
    }
    let settings = TransportSettings {
        total_particles: Some(particles),
        seed,
        ..Default::default()
    };
    (model, settings)
}

#[test]
fn survival_matches_analog_through_urr_window() {
    // Analog samples the reaction channel through the URR-adjusted
    // partition; survival biasing consumes the same adjusted columns via
    // collision_xs. Both must estimate the same flux and MT-101
    // absorption rate; drift between the two URR consumers (the bug
    // class the shared helper exists to prevent) shows up here.
    let (mut analog, analog_settings) = co58_sphere_model(false, 20_000, 42);
    let (mut survival, survival_settings) = co58_sphere_model(true, 20_000, 42);
    analog.simulate_transport(&analog_settings).unwrap();
    survival.simulate_transport(&survival_settings).unwrap();

    for t in 0..2 {
        let name = analog.tallies[t].name.clone().unwrap();
        let a = analog.tallies[t].get_mean()[0];
        let s = survival.tallies[t].get_mean()[0];
        assert!(a > 0.0, "{name}: analog mean must be positive (got {a})");
        assert!(s > 0.0, "{name}: survival mean must be positive (got {s})");
        let rel = (a - s).abs() / a;
        assert!(
            rel < 0.05,
            "{name}: analog {a:.4e} vs survival {s:.4e} disagree (rel {rel:.3e})"
        );
    }
}
