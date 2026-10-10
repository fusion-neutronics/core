//! A thick indium foil, where the isomeric split of capture depends on how the
//! 1.46 eV resonance shields itself, on both paths the branching rule serves.
//!
//! The coupled path scores each state's production at the collision energies,
//! so it sees the flux the foil really has. The multigroup path folds the same
//! list in the walk its capture total is collapsed with, so under the tallied
//! flux itself on a fine grid it must give the coupled answer. On a coarse
//! group across the resonance it takes the list under whatever within-group
//! weight the capture gets: the slowing-down shielding moves the split towards
//! the foil's, where the old fold took every share flat and unshielded, and so
//! gave the dilute split either way.
//!
//! The capture list is synthetic and complete, In116 and In116_m1 both listed,
//! with an isomer yield of 0.9 everywhere except across the resonance, where
//! it dips to 0.3, so the split depends on how the resonance's capture is
//! weighted.
//!
//! Needs the ENDF/B-VIII.1 In115, H1 and O16 fixtures at full scope, and skips
//! without them.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani::{BranchCurve, BranchQuantity, BranchState, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{
    transmute_material_shielded, MultigroupSpectrum, Schedule, ScheduleStep, Shielding,
    TransmuteStep,
};

/// The foil's radius [cm]: at 1.46 eV its capture mean free path is about
/// 10 um, so the resonance is black well inside it.
const RADIUS: f64 = 1.0;
const PARTICLES: usize = 100_000;
const SEED: u64 = 20260927;

fn data(nuclide: &str) -> Option<String> {
    let path = yamc_test_cache::nuclide(nuclide)?;
    // A cache entry fetched for a transmutation holds activation-scope
    // sections only, which transport cannot run on.
    std::path::Path::new(&path)
        .join("reactions.arrow")
        .exists()
        .then_some(path)
}

fn indium(path: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("In115".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.31),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    m.transmutable = true;
    m.volume = Some(4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3));
    m.read_nuclear_data(
        &HashMap::from([("In115".to_string(), path.to_string())]),
        None,
    )
    .unwrap();
    m
}

/// The foil at the centre of a water ball, a 2 MeV point source inside it, and
/// whatever tallies are asked for.
fn model(paths: &HashMap<&str, String>, tallies: Vec<Arc<Tally>>) -> Model {
    let foil = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let ball = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 8.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let inside = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&foil)));
    let shell = Region::new_from_halfspace(HalfspaceType::Below(ball))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(foil)));

    let mut water = Material::new(
        HashMap::from([("H1".to_string(), 2.0), ("O16".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    water.set_material_id(2);
    water.set_temperature("294");
    water
        .read_nuclear_data(
            &HashMap::from([
                ("H1".to_string(), paths["H1"].clone()),
                ("O16".to_string(), paths["O16"].clone()),
            ]),
            None,
        )
        .unwrap();

    let cells = vec![
        Cell::new(Some(1), inside, Some("foil".into()), Some(0)),
        Cell::new(Some(2), shell, Some("water".into()), Some(1)),
    ];
    let geometry = Geometry::new(
        cells,
        vec![Arc::new(indium(&paths["In115"])), Arc::new(water)],
    )
    .unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![2.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(PARTICLES),
        seed: SEED,
        ..Default::default()
    }
}

fn stable(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: vec![],
        fission_yields: None,
        sources: Vec::new(),
    }
}

/// In115 capture to In116 at 1.0 and In116_m1 grafted at 0.0, all stable.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let edge = |target: &str, branching: f64| ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some(target.to_string()),
        branching,
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
        q_value: Some(0.0),
    };
    Arc::new(HashMap::from([
        (
            "In115".to_string(),
            stable("In115", vec![edge("In116", 1.0), edge("In116_m1", 0.0)]),
        ),
        ("In116".to_string(), stable("In116", vec![])),
        ("In116_m1".to_string(), stable("In116_m1", vec![])),
    ]))
}

/// A complete MF=9 capture list whose isomer yield dips across the resonance.
fn branch() -> Arc<BranchTable> {
    let energy = vec![1.0e-5, 1.2, 1.4, 1.52, 1.7, 2.0e7];
    let isomer = vec![0.9, 0.9, 0.3, 0.3, 0.9, 0.9];
    let facts = |lfs: i32| -> Arc<[BranchState]> {
        Arc::from(vec![BranchState {
            mt: 102,
            lfs,
            lmf: Some(9),
            list_complete: true,
            level_route: if lfs == 0 { "ground" } else { "energy" }.to_string(),
            level_energy: 0.0,
            level_energy_difference: Some(0.0),
            mf3_cross_section: None,
        }])
    };
    let mut branch = BranchTable::new();
    branch
        .curves_mut()
        .entry("In115".to_string())
        .or_default()
        .insert(
            "(n,gamma)".to_string(),
            vec![
                BranchCurve {
                    target: "In116".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: energy.clone(),
                    values: isomer.iter().map(|y| 1.0 - y).collect(),
                    states: facts(0),
                    normalisation: None,
                },
                BranchCurve {
                    target: "In116_m1".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy,
                    values: isomer,
                    states: facts(1),
                    normalisation: None,
                },
            ],
        );
    Arc::new(branch)
}

/// In116_m1's share of In115 capture at step 0 of a result.
fn isomer_share(results: &yani_transmute::TransmutationResults, id: u32) -> f64 {
    let channels = results.get_isomeric_branching(id, 0).expect("step 0");
    let capture = channels
        .iter()
        .find(|c| c.parent == "In115" && c.reaction == "(n,gamma)")
        .expect("In115 capture splits");
    capture
        .split
        .iter()
        .find(|(t, _)| t == "In116_m1")
        .expect("In116_m1")
        .1
}

/// The foil's flux, tallied on `bins`, as a multigroup spectrum.
fn tallied_spectrum(paths: &HashMap<&str, String>, bins: &[f64]) -> MultigroupSpectrum {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(bins.to_vec())));
    t.scores = vec!["flux".parse::<Score>().expect("flux")];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    let t = Arc::new(t);
    let mut m = model(paths, vec![Arc::clone(&t)]);
    m.simulate_transport(&settings()).expect("transport");
    let flux = t.get_mean();
    let total: f64 = flux.iter().sum();
    MultigroupSpectrum {
        boundaries: bins.to_vec(),
        masses: flux.iter().map(|f| f / total).collect(),
        flux_error: None,
    }
}

/// In116_m1's share of capture from the multigroup path over `spectrum`.
fn group_share(
    paths: &HashMap<&str, String>,
    spectrum: MultigroupSpectrum,
    shielding: Option<&Shielding>,
) -> f64 {
    let mut material = indium(&paths["In115"]);
    let steps = [TransmuteStep {
        dt: 60.0,
        irradiation: Some((0, 1.0e10)),
    }];
    let results = transmute_material_shielded(
        &mut material,
        &[spectrum],
        &steps,
        chain(),
        &branch(),
        Default::default(),
        None,
        shielding,
    )
    .expect("transmute");
    isomer_share(&results, 1)
}

#[test]
fn a_thick_foil_splits_its_capture_alike_on_both_paths() {
    let mut paths = HashMap::new();
    for nuclide in ["In115", "H1", "O16"] {
        let Some(path) = data(nuclide) else {
            eprintln!("skipping -- {nuclide} fixture absent at full scope");
            return;
        };
        paths.insert(nuclide, path);
    }

    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e12,
        dt: 60.0,
        is_pulse: true,
    }])
    .unwrap();
    let coupled = model(&paths, vec![])
        .transmute(
            "coupled",
            &schedule,
            chain(),
            branch(),
            Default::default(),
            &settings(),
            None,
        )
        .expect("coupled transmute");
    let coupled = isomer_share(&coupled, 1);

    // The same histories tallied on a fine grid, 200 bins a decade and the
    // yield's own nodes, is the flux the foil had, shielding and all: the
    // group path over it is the same rule over the same flux.
    let mut fine: Vec<f64> = (0..=2460)
        .map(|k| 1.0e-5 * 10f64.powf(k as f64 / 200.0))
        .chain([1.2, 1.4, 1.52, 1.7])
        .collect();
    fine.sort_by(f64::total_cmp);
    fine.dedup();
    let tallied = tallied_spectrum(&paths, &fine);
    let over_fine = group_share(&paths, tallied.clone(), None);

    // The same flux with the bins across the resonance merged into one group,
    // folded unshielded and with the foil's chord (4R/3).
    let mut coarse = MultigroupSpectrum {
        boundaries: vec![tallied.boundaries[0]],
        masses: Vec::new(),
        flux_error: None,
    };
    for (g, &mass) in tallied.masses.iter().enumerate() {
        let (lo, hi) = (tallied.boundaries[g], tallied.boundaries[g + 1]);
        if lo >= 1.0 && hi <= 2.1 && lo > 1.0 {
            *coarse.masses.last_mut().expect("a group below") += mass;
            *coarse.boundaries.last_mut().expect("an edge") = hi;
        } else {
            coarse.masses.push(mass);
            coarse.boundaries.push(hi);
        }
    }
    let dilute = group_share(&paths, coarse.clone(), None);
    let chord = Shielding::new(4.0 * RADIUS / 3.0).expect("chord");
    let shielded = group_share(&paths, coarse, Some(&chord));

    eprintln!(
        "In116_m1 share: coupled {coupled:.5}, fine groups {over_fine:.5}, coarse dilute \
         {dilute:.5}, coarse shielded {shielded:.5}"
    );
    assert!(
        (over_fine - coupled).abs() < 1.0e-3,
        "the same flux must split alike: fine {over_fine} against coupled {coupled}"
    );
    // Unshielded, the one group weights the dip by its dilute capture, which
    // the foil does not have; shielded, the split moves towards the foil's.
    assert!(
        dilute < coupled - 0.02,
        "the dilute fold must overweight the dip: {dilute} against {coupled}"
    );
    assert!(
        shielded > dilute && (coupled - shielded).abs() < 0.67 * (coupled - dilute),
        "the shielded fold must close part of the gap: shielded {shielded}, dilute {dilute}, \
         coupled {coupled}"
    );
}
