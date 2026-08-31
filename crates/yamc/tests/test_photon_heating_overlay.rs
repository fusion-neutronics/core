//! Overlay (per-atom KERMA) vs standard (analog deposit) photon
//! heating consistency (issue #358).
//!
//! Standard tallies deposit photon heating analogically at collision
//! sites (electron energy local, minus banked TTB photons, #361).
//! Overlay tallies (`multiply_density = false`) instead score per-atom
//! track-length KERMA: the tabulated heating XS when present, else the
//! physics estimate (Klein-Nishina energy-transfer fraction for
//! Compton + photoelectric + pair production). Both estimate the same
//! physical heating, so overlay-per-atom x atom density must match the
//! analog volumetric result within a few percent.
//!
//! Guards two #358 bugs: the Compton energy-transfer closed form went
//! NEGATIVE below ~400 keV (overlay heating ~10x low across
//! 100-600 keV), and the analog eV deposit was also added into overlay
//! tallies (unit mixing with the eV*barn KERMA contributions).

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::tally::{HeatingScore, NuclideBin, Score, Tally};

fn build_fe_material() -> Material {
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nuclide_map, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    material
}

fn build_fe_photon_geometry() -> Geometry {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let cell = Cell::new(Some(1), region, Some("fe".to_string()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(build_fe_material())]).unwrap()
}

#[test]
fn overlay_photon_heating_matches_analog_deposit() {
    let geometry = build_fe_photon_geometry();

    let mut analog = Tally::new();
    analog.scores = vec![Score::Heating(HeatingScore)];
    // Collision estimator = the analog deposit reference (TrackLength
    // heating scores KERMA along tracks since #356).
    analog.estimator = yamc_tallies::Estimator::Collision;
    analog.name = Some("analog_heating".to_string());

    let mut overlay = Tally::new();
    overlay.scores = vec![Score::Heating(HeatingScore)];
    overlay.multiply_density = false;
    overlay.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
    overlay.estimator = yamc_tallies::Estimator::TrackLength;
    overlay.name = Some("overlay_heating".to_string());

    let source = ParticleSource::Photon(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let mut model = Model::new(
        geometry,
        vec![source],
        vec![Arc::new(analog), Arc::new(overlay)],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.transport_secondary_photons = true;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(40_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let analog_ev = model.tallies[0].total_mean();
    let overlay_ev_barn_cm = model.tallies[1].total_mean();
    let n_fe56 = build_fe_material()
        .get_atoms_per_barn_cm()
        .unwrap()
        .get("Fe56")
        .copied()
        .unwrap();
    let overlay_ev = overlay_ev_barn_cm * n_fe56;

    assert!(analog_ev > 0.0, "analog photon heating was zero");
    assert!(overlay_ev > 0.0, "overlay photon heating was zero");
    let ratio = overlay_ev / analog_ev;
    assert!(
        (0.95..1.06).contains(&ratio),
        "overlay KERMA x density = {overlay_ev:.4e} eV vs analog deposit \
         {analog_ev:.4e} eV (ratio {ratio:.3}); measured 1.009 at the fix \
         (issue #358) -- a collapse means the Compton transfer fraction \
         regressed, ~2 means the analog deposit leaked back into overlay \
         tallies",
    );
}

fn build_natural_fe_material() -> Material {
    let mut material = Material::new(
        HashMap::from([
            ("Fe54".into(), 0.05845),
            ("Fe56".into(), 0.91754),
            ("Fe57".into(), 0.02119),
            ("Fe58".into(), 0.00282),
        ]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([
        ("Fe54".to_string(), "tests/Fe54.arrow".to_string()),
        ("Fe56".to_string(), "tests/Fe56.arrow".to_string()),
        ("Fe57".to_string(), "tests/Fe57.arrow".to_string()),
        ("Fe58".to_string(), "tests/Fe58.arrow".to_string()),
    ]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nuclide_map, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    material
}

/// Issue #341: a material `response` must give the same macroscopic result as
/// the manual workaround -- per-nuclide microscopic overlays (unit density)
/// each weighted by that nuclide's atom density and summed. Running both in the
/// SAME model (one seed -> identical tracks) makes the comparison exact, and the
/// multi-isotope material (natural Fe, four isotopes sharing the Fe photon
/// element) guards against double-counting in the combined-bin sum.
#[test]
fn material_response_matches_nuclide_sum_workaround() {
    let geometry = build_fe_photon_geometry();

    let densities = build_natural_fe_material().get_atoms_per_barn_cm().unwrap();
    let isotopes = ["Fe54", "Fe56", "Fe57", "Fe58"];

    // Tally 0: material response -> one combined macroscopic bin.
    let mut material_tally = Tally::new();
    material_tally.scores = vec![Score::Heating(HeatingScore)];
    material_tally.multiply_density = false;
    material_tally.nuclides = vec![NuclideBin::Total];
    material_tally.overlay_material =
        Some(densities.iter().map(|(k, v)| (k.clone(), *v)).collect());
    material_tally.estimator = yamc_tallies::Estimator::TrackLength;
    material_tally.name = Some("material_response".to_string());
    assert_eq!(
        material_tally.num_nuclide_bins(),
        1,
        "a material response must produce a single combined bin"
    );

    // Tallies 1..=4: one unit-density nuclide overlay per isotope (the workaround).
    let mut tallies: Vec<Arc<Tally>> = vec![Arc::new(material_tally)];
    for iso in isotopes {
        let mut t = Tally::new();
        t.scores = vec![Score::Heating(HeatingScore)];
        t.multiply_density = false;
        t.nuclides = vec![NuclideBin::Specific(iso.to_string())];
        t.estimator = yamc_tallies::Estimator::TrackLength;
        t.name = Some(format!("overlay_{iso}"));
        tallies.push(Arc::new(t));
    }

    let source = ParticleSource::Photon(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = yamc::model::Verbose::silent();
    model.transport_secondary_photons = true;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(40_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let material_ev = model.tallies[0].total_mean();
    let workaround: f64 = isotopes
        .iter()
        .enumerate()
        .map(|(i, iso)| model.tallies[i + 1].total_mean() * densities.get(*iso).copied().unwrap())
        .sum();

    assert!(material_ev > 0.0, "material response heating was zero");
    assert!(workaround > 0.0, "nuclide-sum workaround was zero");
    let ratio = material_ev / workaround;
    assert!(
        (0.999..1.001).contains(&ratio),
        "material response {material_ev:.6e} eV must equal the density-weighted \
         nuclide-sum workaround {workaround:.6e} eV (identical tracks); ratio {ratio:.6} \
         -- a mismatch means the combined-bin sum dropped a nuclide, used the wrong \
         density, or double-counted the shared Fe photon element",
    );
}
