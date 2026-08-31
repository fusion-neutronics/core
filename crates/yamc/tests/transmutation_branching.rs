//! End-to-end coupled-path isomeric branching (issue #218): the per-final-state
//! partials are scored directly at the collision energy during transport and
//! re-partition the chain's ground/metastable split.
//!
//! Uses a synthetic chain (Li6 (n,gamma) -> Li7 / "Li7_m1") over the offline
//! Li6 fixture, with flat MF=10 partials of 3 b (ground) and 1 b (metastable).
//! Flat curves make the exact continuous-energy fraction spectrum-independent:
//! the metastable share must be exactly 1/4 whatever the flux looks like, so
//! the assertion is tight rather than statistical.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::source::{ParticleSource, Source};
use yani::{BranchCurve, BranchQuantity, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{Schedule, ScheduleStep};

fn li6_model() -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("Li6".to_string(), 1.0)]),
        "atom",
        "g/cc",
        Some(0.5),
    )
    .unwrap();
    material.set_temperature("294");
    material.set_material_id(1);
    material.transmutable = true;
    material.volume = Some(4.0 / 3.0 * std::f64::consts::PI * 125.0);
    let mut nuclide_paths = HashMap::new();
    nuclide_paths.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_paths, None).unwrap();
    material.calculate_macroscopic_xs(&vec![1], true);

    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: yamc_source::distribution::angular::AngularDistribution::new_isotropic(),
        energy: yamc_source::source::SourceEnergyDistribution::Discrete(
            yamc_source::distribution::energy::Discrete::new(vec![1.0e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    Model::new(geometry, vec![source], vec![])
}

fn stable(name: &str) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        decay_energy: 0.0,
        reactions: vec![],
        decays: vec![],
        fission_yields: None,
        sources: Vec::new(),
        half_life_uncertainty: None,
        decay_energy_uncertainty: None,
    }
}

/// Li6 captures to a two-state daughter with a 0.7/0.3 base split.
fn synthetic_chain() -> Arc<HashMap<String, ChainNuclide>> {
    let mut map = HashMap::new();
    map.insert(
        "Li6".to_string(),
        ChainNuclide {
            name: "Li6".to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions: vec![
                ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Li7".to_string()),
                    branching: 0.7,
                    q_value: None,
                },
                ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Li7_m1".to_string()),
                    branching: 0.3,
                    q_value: None,
                },
            ],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        },
    );
    map.insert("Li7".to_string(), stable("Li7"));
    map.insert("Li7_m1".to_string(), stable("Li7_m1"));
    Arc::new(map)
}

/// Flat MF=10 partials: 3 b to the ground state, 1 b to the metastable, over
/// the whole energy range, so the exact flux-weighted metastable fraction is
/// 1/4 for any transport spectrum.
fn synthetic_branch() -> Arc<BranchTable> {
    let mut branch = BranchTable::new();
    branch.entry("Li6".to_string()).or_default().insert(
        "(n,gamma)".to_string(),
        vec![
            BranchCurve {
                target: "Li7".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0e-5, 1.0e9],
                values: vec![3.0, 3.0],
            },
            BranchCurve {
                target: "Li7_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0e-5, 1.0e9],
                values: vec![1.0, 1.0],
            },
        ],
    );
    Arc::new(branch)
}

fn run(
    chain: Arc<HashMap<String, ChainNuclide>>,
    branch: Arc<BranchTable>,
) -> HashMap<String, f64> {
    let mut model = li6_model();
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e12,
        dt: 3600.0,
        is_pulse: true,
    }])
    .unwrap();
    let settings = TransportSettings {
        total_particles: Some(2000),
        ..Default::default()
    };
    let results = model
        .transmute(
            "coupled",
            &schedule,
            chain,
            branch,
            Default::default(),
            &settings,
        )
        .unwrap();
    let final_mat = results.get_final_material(1).expect("material 1 present");
    final_mat.nuclides.clone()
}

fn meta_fraction(nuclides: &HashMap<String, f64>) -> f64 {
    let li7 = nuclides.get("Li7").copied().unwrap_or(0.0);
    let li7m = nuclides.get("Li7_m1").copied().unwrap_or(0.0);
    assert!(li7 > 0.0 && li7m > 0.0, "capture products must appear");
    li7m / (li7 + li7m)
}

/// With the branching overlay on, the direct continuous-energy partials must
/// re-partition the 0.7/0.3 base split to exactly 0.75/0.25 (flat 3 b vs 1 b
/// partials); without the overlay the base split must survive untouched.
#[test]
fn coupled_branching_repartitions_metastable_split() {
    let f = meta_fraction(&run(synthetic_chain(), synthetic_branch()));
    assert!(
        (f - 0.25).abs() < 1e-9,
        "expected exact 1/4 metastable share from flat 3:1 partials, got {f}"
    );

    let base = meta_fraction(&run(synthetic_chain(), Arc::new(BranchTable::new())));
    assert!(
        (base - 0.3).abs() < 1e-9,
        "empty overlay must keep the 0.7/0.3 base split, got {base}"
    );
}

/// Non-flat curves: proportional 3:1 ramps (both rising from a shared
/// threshold) must also give exactly 1/4, since the energy shape cancels in
/// the fraction. This exercises the moment fold's slope terms end to end.
#[test]
fn coupled_branching_exact_for_ramp_partials() {
    let mut branch = BranchTable::new();
    branch.entry("Li6".to_string()).or_default().insert(
        "(n,gamma)".to_string(),
        vec![
            BranchCurve {
                target: "Li7".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0e2, 5.0e5, 2.0e6],
                values: vec![0.0, 3.0, 1.5],
            },
            BranchCurve {
                target: "Li7_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0e2, 5.0e5, 2.0e6],
                values: vec![0.0, 1.0, 0.5],
            },
        ],
    );
    let f = meta_fraction(&run(synthetic_chain(), Arc::new(branch)));
    assert!(
        (f - 0.25).abs() < 1e-9,
        "expected exact 1/4 metastable share from proportional 3:1 ramps, got {f}"
    );
}

/// MF=9 yields weight the parent's transport cross section at the collision
/// energy: flat 3:1 yields must give exactly the same 0.75/0.25 split (the
/// sigma_MT factor cancels in the fraction whatever the spectrum).
#[test]
fn coupled_branching_scores_mf9_yields() {
    let mut branch = BranchTable::new();
    branch.entry("Li6".to_string()).or_default().insert(
        "(n,gamma)".to_string(),
        vec![
            BranchCurve {
                target: "Li7".to_string(),
                quantity: BranchQuantity::Yield,
                energy: vec![1.0e-5, 1.0e9],
                values: vec![0.75, 0.75],
            },
            BranchCurve {
                target: "Li7_m1".to_string(),
                quantity: BranchQuantity::Yield,
                energy: vec![1.0e-5, 1.0e9],
                values: vec![0.25, 0.25],
            },
        ],
    );
    let f = meta_fraction(&run(synthetic_chain(), Arc::new(branch)));
    assert!(
        (f - 0.25).abs() < 1e-9,
        "expected exact 1/4 metastable share from flat 3:1 yields, got {f}"
    );
}

/// Chain parents that are NOT in the transport material (products that build
/// up during the step) must keep their branching overlay: here Li7 (produced
/// from Li6 capture during the pulse) carries a grafted (n,n') channel to
/// Li7_m1 whose rate exists only through the moment-folded injection. Without
/// that coverage Li7_m1 is exactly zero.
#[test]
fn coupled_branching_folds_parents_outside_material() {
    let mut map = HashMap::new();
    map.insert(
        "Li6".to_string(),
        ChainNuclide {
            name: "Li6".to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions: vec![ChainReaction {
                kind: "(n,gamma)".to_string(),
                target: Some("Li7".to_string()),
                branching: 1.0,
                q_value: None,
            }],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        },
    );
    // Li7 has a grafted (n,n') metastable channel; its production rate can
    // only come from the fold fallback (Li7 is not in the material, so it has
    // no partial channels, and MT 4 is never tallied).
    map.insert(
        "Li7".to_string(),
        ChainNuclide {
            name: "Li7".to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions: vec![ChainReaction {
                kind: "(n,n')".to_string(),
                target: Some("Li7_m1".to_string()),
                branching: 1.0,
                q_value: None,
            }],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        },
    );
    map.insert("Li7_m1".to_string(), stable("Li7_m1"));
    let chain = Arc::new(map);

    let mut branch = BranchTable::new();
    branch.entry("Li7".to_string()).or_default().insert(
        "(n,n')".to_string(),
        vec![BranchCurve {
            target: "Li7_m1".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e-5, 1.0e9],
            values: vec![0.5, 0.5], // flat 0.5 b
        }],
    );

    let nuclides = run(Arc::clone(&chain), Arc::new(branch));
    let li7m = nuclides.get("Li7_m1").copied().unwrap_or(0.0);
    assert!(
        li7m > 0.0,
        "fold fallback must inject the (n,n') rate for the built-up Li7, got {li7m}"
    );

    // Without the overlay there is no (n,n') rate at all.
    let nuclides = run(chain, Arc::new(BranchTable::new()));
    let li7m = nuclides.get("Li7_m1").copied().unwrap_or(0.0);
    assert_eq!(li7m, 0.0, "no overlay must mean no (n,n') production");
}
