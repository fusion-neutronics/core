//! End-to-end coupled-path isomeric branching: the per-final-state
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
use yani::{BranchCurve, BranchQuantity, BranchState, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{Schedule, ScheduleStep};

/// The converter's facts for a state whose list names its ground state
/// (`complete`) or gives its isomers alone.
fn facts(target: &str, complete: bool) -> Arc<[BranchState]> {
    Arc::from(vec![BranchState {
        mt: 102,
        lfs: if target.contains("_m") { 1 } else { 0 },
        lmf: Some(10),
        list_complete: complete,
        level_route: "energy".to_string(),
        level_energy: 0.0,
        level_energy_difference: Some(0.0),
        mf3_cross_section: None,
    }])
}

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
        decay_energy_components: Default::default(),
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
                    branching_uncertainty: None,
                    evaluated_branching: None,
                },
                ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Li7_m1".to_string()),
                    branching: 0.3,
                    q_value: None,
                    branching_uncertainty: None,
                    evaluated_branching: None,
                },
            ],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
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
    branch
        .curves_mut()
        .entry("Li6".to_string())
        .or_default()
        .insert(
            "(n,gamma)".to_string(),
            vec![
                BranchCurve {
                    target: "Li7".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e-5, 1.0e9],
                    values: vec![3.0, 3.0],
                    states: facts("Li7", true),
                    normalisation: None,
                },
                BranchCurve {
                    target: "Li7_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e-5, 1.0e9],
                    values: vec![1.0, 1.0],
                    states: facts("Li7_m1", true),
                    normalisation: None,
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
            None,
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
    branch
        .curves_mut()
        .entry("Li6".to_string())
        .or_default()
        .insert(
            "(n,gamma)".to_string(),
            vec![
                BranchCurve {
                    target: "Li7".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e2, 5.0e5, 2.0e6],
                    values: vec![0.0, 3.0, 1.5],
                    states: facts("Li7", true),
                    normalisation: None,
                },
                BranchCurve {
                    target: "Li7_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e2, 5.0e5, 2.0e6],
                    values: vec![0.0, 1.0, 0.5],
                    states: facts("Li7_m1", true),
                    normalisation: None,
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
    branch
        .curves_mut()
        .entry("Li6".to_string())
        .or_default()
        .insert(
            "(n,gamma)".to_string(),
            vec![
                BranchCurve {
                    target: "Li7".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: vec![1.0e-5, 1.0e9],
                    values: vec![0.75, 0.75],
                    states: facts("Li7", true),
                    normalisation: None,
                },
                BranchCurve {
                    target: "Li7_m1".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: vec![1.0e-5, 1.0e9],
                    values: vec![0.25, 0.25],
                    states: facts("Li7_m1", true),
                    normalisation: None,
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
                branching_uncertainty: None,
                evaluated_branching: None,
            }],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
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
                branching_uncertainty: None,
                evaluated_branching: None,
            }],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
        },
    );
    map.insert("Li7_m1".to_string(), stable("Li7_m1"));
    let chain = Arc::new(map);

    let mut branch = BranchTable::new();
    branch
        .curves_mut()
        .entry("Li7".to_string())
        .or_default()
        .insert(
            "(n,n')".to_string(),
            vec![BranchCurve {
                target: "Li7_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0e-5, 1.0e9],
                values: vec![0.5, 0.5], // flat 0.5 b
                states: facts("Li7_m1", true),
                normalisation: None,
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

/// Li6 capture shaped as ENDF/B-VIII.1 gives In115's: the base chain carries
/// the ground state at 1.0 and the isomer grafted at 0.0.
fn isomer_only_chain() -> Arc<HashMap<String, ChainNuclide>> {
    let mut map = (*synthetic_chain()).clone();
    let li6 = map.get_mut("Li6").expect("Li6");
    for rx in &mut li6.reactions {
        rx.branching = if rx.target.as_deref() == Some("Li7") {
            1.0
        } else {
            0.0
        };
    }
    Arc::new(map)
}

fn only_the_isomer(curve: BranchCurve) -> Arc<BranchTable> {
    let mut branch = BranchTable::new();
    branch
        .curves_mut()
        .entry("Li6".to_string())
        .or_default()
        .insert("(n,gamma)".to_string(), vec![curve]);
    Arc::new(branch)
}

/// An overlay listing only the isomer, the ground state being the remainder.
/// A flat MF=9 yield of 0.2 is the isomer's share exactly, the yield channel
/// and the total being scored against the same `sigma * TL`. Normalized over
/// the one listed state, and then confined to the zero mass its grafted edge
/// carried, it used to make no Li7_m1 at all.
#[test]
fn coupled_isomer_only_yield_is_a_share_of_the_tallied_total() {
    let branch = only_the_isomer(BranchCurve {
        target: "Li7_m1".to_string(),
        quantity: BranchQuantity::Yield,
        energy: vec![1.0e-5, 1.0e9],
        values: vec![0.2, 0.2],
        states: facts("Li7_m1", false),
        normalisation: None,
    });
    let f = meta_fraction(&run(isomer_only_chain(), branch));
    assert!(
        (f - 0.2).abs() < 1e-9,
        "expected the flat 0.2 yield as the isomer's share, got {f}"
    );
}

/// An MF=10 partial at 0.2 of Li6's own capture cross section, on that cross
/// section's grid up to `top` [eV].
fn fifth_of_the_capture(top: f64) -> BranchCurve {
    let mut li6 = Material::new(
        HashMap::from([("Li6".to_string(), 1.0)]),
        "atom",
        "g/cc",
        Some(0.5),
    )
    .unwrap();
    li6.set_temperature("294");
    li6.read_nuclear_data(
        &HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]),
        None,
    )
    .unwrap();
    let capture = &li6.nuclide_data["Li6"]
        .reactions_for_temp("294")
        .expect("Li6 at 294 K")[&102];
    let (energy, values): (Vec<f64>, Vec<f64>) = capture
        .energy
        .iter()
        .zip(capture.cross_section.iter())
        .filter(|(e, _)| **e <= top)
        .map(|(e, x)| (*e, 0.2 * x))
        .unzip();
    BranchCurve {
        target: "Li7_m1".to_string(),
        quantity: BranchQuantity::CrossSection,
        energy,
        values,
        states: facts("Li7_m1", false),
        normalisation: None,
    }
}

/// The MF=10 form: a partial at 0.2 of Li6's capture cross section, on that
/// cross section's grid, is 0.2 of the tallied total, and Li7 keeps the other
/// 0.8.
#[test]
fn coupled_isomer_only_partial_is_a_share_of_the_tallied_total() {
    let branch = only_the_isomer(fifth_of_the_capture(f64::INFINITY));
    let f = meta_fraction(&run(isomer_only_chain(), branch));
    assert!(
        (f - 0.2).abs() < 1e-9,
        "expected 0.2 of the capture total as the isomer's share, got {f}"
    );
}

/// The same partial stopping at 10 keV, under a 1 MeV source whose flux runs
/// on above it. Past its last point an isomer-only partial follows the tallied
/// total at the share it ends on, and that is the evaluation's fraction held,
/// not its data. Here most of the capture lies above 10 keV, so the run is
/// refused and says why, where it used to answer 0.2 on the held share alone.
#[test]
fn coupled_isomer_only_partial_past_its_last_point_is_refused() {
    let branch = only_the_isomer(fifth_of_the_capture(1.0e4));
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
    let err = model
        .transmute(
            "coupled",
            &schedule,
            isomer_only_chain(),
            branch,
            Default::default(),
            &settings,
            None,
        )
        .expect_err("refused")
        .to_string();
    assert!(err.contains("Li6 (n,gamma)"), "{err}");
    assert!(err.contains("tabulates no split"), "{err}");
}

/// A metastable the overlay grafts onto the chain enters it at branching 0.0,
/// the loader's placeholder for "the fold decides". The product bound that
/// picks which nuclides to score runs on the chain as loaded, before any fold,
/// so it must still see that edge: here Li6 capture feeds a grafted Li7_m2
/// whose own capture makes Li8. If the bound read the placeholder, Li7_m2
/// would be left unscored, its capture rate would be zero and no Li8 could
/// appear, although the overlay puts a quarter of every Li6 capture there.
#[test]
fn coupled_bound_keeps_grafted_metastable_reactions() {
    // Li7 cross sections stand in for the metastable's own: what matters is
    // that it has a capture channel to score, not what that channel's data is.
    // The mapping is process-global and the product preload has no per-call
    // path, so the metastable is named Li7_m2, which no other test in this
    // binary uses, rather than the Li7_m1 the tests running beside it share.
    yamc_nuclide::Config::global().set_cross_section("Li7_m2", Some("tests/Li7.arrow"));

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
                    branching: 1.0,
                    branching_uncertainty: None,
                    evaluated_branching: None,
                    q_value: None,
                },
                // Exactly what `parse_chain_parts_from_bytes` grafts.
                ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Li7_m2".to_string()),
                    branching: 0.0,
                    branching_uncertainty: None,
                    evaluated_branching: None,
                    q_value: None,
                },
            ],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
        },
    );
    map.insert("Li7".to_string(), stable("Li7"));
    let mut li7m = stable("Li7_m2");
    li7m.reactions.push(ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("Li8".to_string()),
        branching: 1.0,
        branching_uncertainty: None,
        evaluated_branching: None,
        q_value: None,
    });
    map.insert("Li7_m2".to_string(), li7m);
    map.insert("Li8".to_string(), stable("Li8"));

    // The same flat 3:1 partials as `synthetic_branch`, onto Li7_m2.
    let mut branch = (*synthetic_branch()).clone();
    for c in branch
        .curves_mut()
        .get_mut("Li6")
        .unwrap()
        .get_mut("(n,gamma)")
        .unwrap()
    {
        if c.target == "Li7_m1" {
            c.target = "Li7_m2".to_string();
        }
    }

    let mut model = li6_model();
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e16,
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
            Arc::new(map),
            Arc::new(branch),
            Default::default(),
            &settings,
            None,
        )
        .unwrap();
    let nuclides = &results.get_final_material(1).expect("material 1").nuclides;

    let li7 = nuclides.get("Li7").copied().unwrap_or(0.0);
    let li7m = nuclides.get("Li7_m2").copied().unwrap_or(0.0);
    assert!(
        (li7m / (li7 + li7m) - 0.25).abs() < 1e-9,
        "the fold must still route a quarter of Li6 capture to Li7_m2: Li7 {li7:e}, Li7_m2 {li7m:e}"
    );
    let li8 = nuclides.get("Li8").copied().unwrap_or(0.0);
    assert!(
        li8 > 0.0,
        "Li7_m2 capture must be scored, so Li8 must appear; got {li8:e} beside Li7_m2 {li7m:e}"
    );
}
