//! The `flux_spectrum` source moves the rates and splits the isomeric
//! branching overlay makes, not only the collapsed transport channels.
//!
//! An `(n,n')` isomer rate (In115 to In115m, Rh103 to Rh103m, Nb93 to Nb93m)
//! is the folded MF=10 partial production itself, so it has no transport
//! channel's group averages to perturb. The flux draw used to pass it through
//! at nominal and report nothing: with a fully correlated 10% flux error, a
//! synthetic In115 gave In114 a relative sigma of 0.097 and In115_m1 one of
//! 0.0000. The same holds one level down for a split: an isomer-only (n,2n)
//! list's share is its production over the transport total, and a bin that
//! moves the isomer's production more than the total has to move the share.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yani::{BranchCurve, BranchQuantity, BranchState, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::flux_uncertainty::{FluxError, RelativeFluxCovariance};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn reaction(mt: i32, energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
    Reaction {
        cross_section: cross_section.into(),
        threshold_idx: 0,
        energy: energy.into(),
        mt_number: mt,
        q_value: 0.0,
        products: vec![].into(),
        scatter_in_cm: false,
        redundant: false,
    }
}

/// In115 with an inelastic total and a flat 1 b (n,2n) over both groups.
fn indium() -> Material {
    let temperature = "294".to_string();
    let reactions: HashMap<i32, Arc<Reaction>> = HashMap::from([
        (
            4,
            Arc::new(reaction(
                4,
                vec![1.0e5, 1.0e6, 1.0e7, 2.0e7],
                vec![0.5, 1.0, 1.0, 1.0],
            )),
        ),
        (
            16,
            Arc::new(reaction(16, vec![1.0e6, 2.0e7], vec![1.0, 1.0])),
        ),
    ]);
    let nuclide = Nuclide {
        name: Some("In115".to_string()),
        element: None,
        atomic_symbol: Some("In".to_string()),
        atomic_number: Some(49),
        neutron_number: Some(66),
        mass_number: Some(115),
        atomic_weight_ratio: Some(113.9),
        library: None,
        energy: None,
        reactions: vec![reactions],
        fissionable: false,
        available_temperatures: vec![temperature.clone()],
        loaded_temperatures: vec![temperature.clone()],
        data_path: None,
        data_source: None,
        fission_nu: None,
        fast_xs: vec![],
        urr_data: vec![],
        urr_present: false,
        fission_photon_release: None,
        covariance: None,
        angular_covariance: None,
        nubar_covariance: None,
        spectrum_covariance: None,
        resonance_parameters: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: Default::default(),
    };
    let mut m = Material::new(
        HashMap::from([("In115".to_string(), 3.8e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("indium");
    m.set_temperature(&temperature);
    m.nuclide_data
        .insert("In115".to_string(), Arc::new(nuclide));
    m.volume = Some(1.0);
    m
}

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
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

/// In115's (n,n') grafted to its isomer and its (n,2n) to the ground state at
/// 1.0 with the isomer grafted at 0.0. Every product is stable, so over a
/// short step each density is its production rate times the step.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let edge = |kind: &str, target: &str, branching: f64| ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching,
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
        q_value: Some(0.0),
    };
    let mut map = HashMap::new();
    map.insert(
        "In115".to_string(),
        nuclide(
            "In115",
            vec![
                edge("(n,n')", "In115_m1", 0.0),
                edge("(n,2n)", "In114", 1.0),
                edge("(n,2n)", "In114_m1", 0.0),
            ],
        ),
    );
    for name in ["In115_m1", "In114", "In114_m1"] {
        map.insert(name.to_string(), nuclide(name, vec![]));
    }
    Arc::new(map)
}

/// The converter's facts for an isomer its list gives alone.
fn isomer_only(mt: i32) -> Arc<[BranchState]> {
    Arc::from(vec![BranchState {
        mt,
        lfs: 1,
        lmf: Some(10),
        list_complete: false,
        level_route: "energy".to_string(),
        level_energy: 0.0,
        level_energy_difference: Some(0.0),
        mf3_cross_section: None,
    }])
}

/// The MF=10 partials: (n,n') to In115_m1 across both groups, and (n,2n) to
/// In114_m1 only in the upper group, rising from zero at its lower edge.
fn branch() -> BranchTable {
    let mut branch = BranchTable::new();
    let kinds = branch.curves_mut().entry("In115".to_string()).or_default();
    kinds.insert(
        "(n,n')".to_string(),
        vec![BranchCurve {
            target: "In115_m1".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e5, 1.0e6, 1.0e7, 2.0e7],
            values: vec![0.1, 0.3, 0.3, 0.3],
            states: isomer_only(4),
            normalisation: None,
        }],
    );
    kinds.insert(
        "(n,2n)".to_string(),
        vec![BranchCurve {
            target: "In114_m1".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e6, BOUNDARIES[1], 2.0e7],
            values: vec![0.0, 0.0, 0.5],
            states: isomer_only(16),
            normalisation: None,
        }],
    );
    branch
}

const BOUNDARIES: [f64; 3] = [1.0e6, 1.35e7, 2.0e7];
const FLUX: [f64; 2] = [1.0, 1.0];

fn spectrum(flux_error: Option<FluxError>) -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: BOUNDARIES.to_vec(),
        masses: FLUX.to_vec(),
        flux_error,
    }
}

fn steps() -> [TransmuteStep; 1] {
    [TransmuteStep {
        dt: 60.0,
        irradiation: Some((0, 1.0e14)),
    }]
}

fn flux_only(samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: 3,
        samples: Some(samples),
        sources: vec![Source::FluxSpectrum],
        attribution: false,
        ..Default::default()
    }
}

/// Each product's relative sigma, and the run's report, under `flux_error`.
fn relative_sigmas(
    flux_error: FluxError,
) -> (HashMap<String, f64>, yani_transmute::uncertainty::Info) {
    let mut material = indium();
    let results = transmute_material(
        &mut material,
        &[spectrum(Some(flux_error))],
        &steps(),
        chain(),
        &branch(),
        Default::default(),
        Some(&flux_only(400)),
    )
    .expect("transmute");
    let nominal = &results.get_material(0, 1).expect("step 1").nuclides;
    let sd = results.uncertainty[&0].std_dev_at(0);
    let rel = ["In115_m1", "In114", "In114_m1"]
        .iter()
        .map(|n| {
            (
                n.to_string(),
                sd.get(*n).copied().unwrap_or(0.0) / nominal[*n],
            )
        })
        .collect();
    (rel, results.uncertainty_info[&0].clone())
}

/// Every rate moves by the common factor of a fully correlated flux error, so
/// every product carries its 10%, the (n,n') isomer included.
#[test]
fn an_inelastic_isomer_moves_with_a_correlated_flux_error() {
    let s = 0.10;
    let cov = vec![vec![s * s, s * s], vec![s * s, s * s]];
    let error =
        FluxError::RelativeCovariance(RelativeFluxCovariance::from_absolute(&FLUX, &cov).unwrap());
    let (rel, info) = relative_sigmas(error);
    for n in ["In115_m1", "In114", "In114_m1"] {
        assert!(
            (rel[n] - 0.10).abs() < 0.01,
            "{n} should move with the whole flux: relative sigma {}",
            rel[n]
        );
    }
    assert!(
        info.flux_rates_without_terms.is_empty(),
        "every rate has terms: {:?}",
        info.flux_rates_without_terms
    );
}

/// Only the upper group is uncertain. In114_m1 is produced only there, so it
/// carries the whole 10%; the ground state, made in both groups, carries its
/// upper-group share. Holding the nominal split would give both the total's
/// share, 5%.
#[test]
fn an_isomer_split_moves_with_the_bins_its_state_is_produced_in() {
    let (rel, info) = relative_sigmas(FluxError::RelativeStdDev(vec![0.0, 0.10]));
    assert!(
        (rel["In114_m1"] - 0.10).abs() < 0.01,
        "In114_m1 is made only in the uncertain group: relative sigma {}",
        rel["In114_m1"]
    );
    // The ground state's (n,2n) is 1 b in the lower group and 1 - 0.25 b in
    // the upper, so 0.75 / 1.75 of it is in the uncertain group.
    let ground = 0.75 / 1.75 * 0.10;
    assert!(
        (rel["In114"] - ground).abs() < 0.006,
        "In114 should carry its upper-group share {ground}: relative sigma {}",
        rel["In114"]
    );
    assert!(info.flux_rates_without_terms.is_empty());
}

/// The nominal is the same with or without the flux source, and a flux error
/// of zero re-folds the branching to exactly the nominal in every replica.
#[test]
fn the_nominal_and_an_unperturbed_replica_are_unchanged() {
    let run = |flux_error: Option<FluxError>, request: Option<&DataUncertainty>| {
        let mut material = indium();
        transmute_material(
            &mut material,
            &[spectrum(flux_error)],
            &steps(),
            chain(),
            &branch(),
            Default::default(),
            request,
        )
        .expect("transmute")
    };
    let plain = run(None, None);
    let request = flux_only(8);
    let with_flux = run(
        Some(FluxError::RelativeStdDev(vec![0.05, 0.10])),
        Some(&request),
    );
    let exact = run(
        Some(FluxError::RelativeStdDev(vec![0.0, 0.0])),
        Some(&request),
    );

    let nominal = &plain.get_material(0, 1).expect("step 1").nuclides;
    assert!(nominal["In115_m1"] > 0.0 && nominal["In114_m1"] > 0.0);
    assert_eq!(
        &with_flux.get_material(0, 1).expect("step 1").nuclides,
        nominal
    );
    assert_eq!(&exact.get_material(0, 1).expect("step 1").nuclides, nominal);

    let ensemble = &exact.uncertainty[&0];
    for n in ["In115_m1", "In114", "In114_m1"] {
        for replica in ensemble.samples_at(0, n) {
            assert!(
                (replica - nominal[n]).abs() <= 1e-12 * nominal[n],
                "{n}: replica {replica} against nominal {}",
                nominal[n]
            );
        }
    }
}
