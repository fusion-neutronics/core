//! Regression test for issue #84: O16 neutron heating must fold in its
//! physically NEGATIVE KERMA (MT 301).
//!
//! In ENDF/B-VIII.1 the O16 macroscopic heating cross section (MT 301)
//! is negative across roughly 7.4-13 MeV: the dominant fast-neutron
//! charged-particle channels ((n,alpha) Q = -2.2 MeV, (n,p), (n,d),
//! (n,t)) are endothermic, so the KERMA they contribute is negative
//! there, turning positive again only near the 14 MeV source. OpenMC
//! and the yamc GPU kernel fold these negatives correctly. The yamc CPU
//! scoring used to guard the fold with `heating_xs > 0.0`, silently
//! dropping the negative window and over-counting neutron heating
//! (issue #84 quotes +18.4% on a single-collision spectral average).
//!
//! The fix relaxes the neutron heating / heating-local fold guards to
//! `!= 0.0` so negative KERMA accumulates. This test pins the corrected
//! behaviour two ways:
//!   1. The O16 macroscopic MT 301 KERMA really is negative across the
//!      fast charged-particle window (proves the data has the negatives
//!      the fix must keep).
//!   2. A 14.06 MeV O16-sphere CPU run lands at the folded value, below
//!      the inflated drop-negatives value the bug produced.
//!
//! MT 301 is the only neutron KERMA that goes negative in O16; MT 901
//! (heating-local) stays positive, so its result is unchanged by the
//! fix and acts as a control here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
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
use yamc_tallies::score::{FluxScore, HeatingLocalScore, HeatingScore, Score};
use yamc_tallies::tally::Tally;

const O16_PATH: &str = "tests/O16.arrow";

fn o16_material() -> Material {
    let mut material = Material::new(
        HashMap::from([("O16".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.141),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([("O16".to_string(), O16_PATH.to_string())]);
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    material
}

fn o16_sphere_model(seed: u64) -> (Model, Vec<Arc<Tally>>, TransportSettings) {
    // Small sphere: the flux stays dominated by high-energy neutrons in
    // the negative-MT-301 window, so dropping those negatives shows up
    // clearly in the heating tally.
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("o16_sphere".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(o16_material())]).unwrap();

    // 14.06 MeV monoenergetic point source (D-T fusion neutron).
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let n_particles = 20_000;
    let n_batches = 4;
    let make_tally = |score: Score, name: &str, energy_filter: Option<EnergyFilter>| {
        let mut t = Tally::new();
        t.filters
            .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
        if let Some(ef) = energy_filter {
            t.filters.push(Filter::Energy(ef));
        }
        t.scores = vec![score];
        t.name = Some(name.to_string());
        t.initialize_batches(n_batches);
        Arc::new(t)
    };
    // The fourth tally scores heating only in the 7.4-13 MeV window
    // where O16 MT 301 is negative; folding it must give a negative bin.
    let neg_window = EnergyFilter::new(vec![7.4e6, 13.0e6]);
    let tallies = vec![
        make_tally(Score::Flux(FluxScore), "flux", None),
        make_tally(Score::Heating(HeatingScore), "heating", None),
        make_tally(
            Score::HeatingLocal(HeatingLocalScore),
            "heating_local",
            None,
        ),
        make_tally(
            Score::Heating(HeatingScore),
            "heating_neg_window",
            Some(neg_window),
        ),
    ];
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tallies, settings)
}

fn tally_mean_sum(t: &Tally) -> f64 {
    t.get_mean().iter().sum()
}

/// The O16 macroscopic KERMA (MT 301) must be negative across the fast
/// charged-particle window (~7.4-13 MeV). This is the physics the fold
/// must not discard (issue #84).
#[test]
fn o16_macroscopic_kerma_is_negative_in_fast_window() {
    if !Path::new(O16_PATH).exists() {
        eprintln!("skipping: {O16_PATH} fixture not present");
        return;
    }
    let mut material = o16_material();
    // Build the macroscopic grid (incl. MT 301 / 901); read_nuclear_data
    // alone does not populate it.
    let _ = material.calculate_macroscopic_xs(&vec![1, 301, 901], true);

    let mut min_xs = f64::INFINITY;
    let mut min_e = 0.0;
    let mut e = 7.4e6;
    while e <= 13.0e6 {
        let xs = material.lookup_heating_xs(e);
        if xs < min_xs {
            min_xs = xs;
            min_e = e;
        }
        e += 1.0e4;
    }
    eprintln!("O16 min macroscopic MT 301 KERMA in 7.4-13 MeV: {min_xs:.4e} at {min_e:.3e} eV");
    assert!(
        min_xs < 0.0,
        "O16 MT 301 macroscopic KERMA should be negative in 7.4-13 MeV, got min {min_xs:.4e}"
    );
}

/// 14.06 MeV O16 sphere: CPU neutron heating (MT 301) must fold the
/// negative KERMA window. Dropping it (the bug) inflated the result;
/// the corrected fold lands lower. MT 901 never goes negative, so
/// heating-local is unchanged and serves as a control.
#[test]
fn o16_cpu_heating_folds_negative_kerma() {
    if !Path::new(O16_PATH).exists() {
        eprintln!("skipping: {O16_PATH} fixture not present");
        return;
    }
    let (mut model, tallies, settings) = o16_sphere_model(7);
    model.simulate_transport(&settings).unwrap();
    let flux = tally_mean_sum(&tallies[0]);
    let heating = tally_mean_sum(&tallies[1]);
    let heating_local = tally_mean_sum(&tallies[2]);
    let heating_neg_window = tally_mean_sum(&tallies[3]);
    eprintln!(
        "O16 CPU per source neutron: flux={flux:.4e}  heating={heating:.6e}  \
         heating_local={heating_local:.6e}  heating[7.4-13 MeV]={heating_neg_window:.6e}"
    );

    // Decisive check: in 7.4-13 MeV O16 MT 301 is negative everywhere,
    // so a heating tally restricted to that window must come out
    // NEGATIVE once the fold keeps the negatives. The buggy code
    // dropped every negative sample, leaving this bin at exactly 0.
    assert!(
        heating_neg_window < 0.0,
        "Heating in the 7.4-13 MeV window should be negative (folded KERMA), got \
         {heating_neg_window:.4e} -- negatives are being dropped (issue #84)"
    );

    // Sanity: the total heating still nets positive (the 14 MeV source
    // lobe dominates), and is lower than it would be with the negatives
    // discarded. MT 901 has no negative window, so heating-local is a
    // control that must stay positive and unaffected.
    assert!(
        heating > 0.0,
        "O16 total heating should net positive (14 MeV lobe dominates)"
    );
    assert!(
        heating_local > 0.0,
        "O16 heating-local (MT 901) should be positive"
    );
}

/// The yamc GPU kernel always folded the negative MT 301 KERMA (it
/// matches OpenMC, per issue #84). With the CPU fold fixed, CPU and GPU
/// O16 heating must now agree. Gated on `gpu`; skipped on the CPU CI.
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
#[test]
fn o16_cpu_matches_gpu_heating() {
    if !Path::new(O16_PATH).exists() {
        eprintln!("skipping: {O16_PATH} fixture not present");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no GPU with f64 compute available");
        return;
    }

    let (mut cpu_model, cpu_tallies, cpu_settings) = o16_sphere_model(7);
    cpu_model.simulate_transport(&cpu_settings).unwrap();
    let cpu_heating = tally_mean_sum(&cpu_tallies[1]);

    let (mut gpu_model, gpu_tallies, gpu_settings) = o16_sphere_model(7);
    yamc::gpu::run_on_gpu(&mut gpu_model, &gpu_settings).expect("GPU dispatch");
    let gpu_heating = tally_mean_sum(&gpu_tallies[1]);

    let ratio = cpu_heating / gpu_heating;
    eprintln!("O16 heating CPU={cpu_heating:.6e}  GPU={gpu_heating:.6e}  CPU/GPU={ratio:.4}");
    // Both fold the negatives now, so they agree to a few percent (seed
    // is shared but the RNG ladders differ between backends). Pre-fix
    // the CPU was high by the dropped-negative fraction.
    assert!(
        (0.90..=1.10).contains(&ratio),
        "O16 CPU/GPU heating ratio {ratio:.4} out of [0.90, 1.10] -- CPU fold may be dropping \
         negative MT 301 again (issue #84)"
    );
}
