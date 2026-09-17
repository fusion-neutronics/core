//! Th232's correlated prompt chi on the shared fission path
//! (fusion-neutronics/core#34 entry 2).
//!
//! Th232, Pa231 and Pa233 are the only fissionable nuclides in ENDF/B-VIII.1
//! whose prompt fission spectrum is a `CorrelatedAngleEnergy` (ENDF File 6
//! LAW 1). The GPU extractor always packed that table's outgoing-energy
//! marginal and emitted isotropically in the lab; the CPU's `prompt_chi_dist`
//! only recognised the uncorrelated encoding, so for these three the CPU fell
//! back to the legacy per-product sampler, off the shared PCG stream and
//! without the delayed-neutron split. Every history with a fission left the
//! GPU's stream at that fission: the twin localizer (`matched_stream_localize`)
//! measured 95.2% of Th232 histories identical within rounding, 4838 of 100k
//! first diverging at MT18, and a -1.28% collision-0 gap on the MT18 mean.
//!
//! The CPU now flattens the marginal too, so both backends draw the prompt chi
//! at the same stream position from the same table (the localizer reads 100%
//! within rounding, MT18 bit-equal). Two checks here: the cache entry Th232's
//! MT18 builds is a `Continuous` marginal rather than `None`, and a real GPU
//! run against a CPU run on the same seed keeps the lockstep signature.
//! Fission is 6% of a 14 MeV Th232 collision and most histories leave after
//! two, so the tally-level signature is diluted: the 7-bin reduced chi-square
//! read 0.23 before and 0.006 after on this box. The threshold sits between
//! them; the per-history claim is the localizer's.
//!
//! Run it (needs an f64 GPU and Th232 in the cache; Th232 is not a CI
//! fixture, so both tests self-skip there):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_th232_correlated_chi -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_nuclide::reaction_product::{AngleEnergyDistribution, FissionChiFlat, ParticleType};
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
const N_HISTORIES: usize = 100_000;
const RADIUS: f64 = 6.0;
const DENSITY: f64 = 11.7;

fn cache_path() -> Option<String> {
    let path = yamc_test_cache::nuclide("Th232");
    if path.is_none() {
        eprintln!("skipping: Th232 not in the cache");
    }
    path
}

fn spectrum_edges() -> Vec<f64> {
    vec![1.0e3, 1.0e5, 5.0e5, 1.0e6, 2.0e6, 4.0e6, 8.0e6, 1.5e7]
}

fn tally(cell_id: u32, score: &str, edges: Option<Vec<f64>>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    if let Some(e) = edges {
        t.filters.push(Filter::Energy(EnergyFilter::new(e)));
    }
    t.scores = vec![score.parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model(path: &str) -> (Model, [Arc<Tally>; 3], TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Th232".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(DENSITY),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("Th232".to_string(), path.to_string())]),
            None,
        )
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("th".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let flux = tally(cell_id, "flux", None);
    let fission = tally(cell_id, "fission", None);
    let spectrum = tally(cell_id, "flux", Some(spectrum_edges()));
    let mut model = Model::new(
        geometry,
        vec![source],
        vec![
            Arc::clone(&flux),
            Arc::clone(&fission),
            Arc::clone(&spectrum),
        ],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_max_steps_per_particle = 10_000;
    assert!(model.gpu_fission_bank);
    let settings = TransportSettings {
        total_particles: Some(N_HISTORIES),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, [flux, fission, spectrum], settings)
}

/// The fixture is what the entry says it is (Th232's MT18 first neutron product
/// is a `CorrelatedAngleEnergy`) and the CPU's cache builds a `Continuous`
/// marginal for it, so the shared flat sampler, not the legacy one, draws it.
#[test]
fn th232_prompt_chi_flattens_to_a_continuous_marginal() {
    let Some(path) = cache_path() else { return };
    let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(
        std::path::PathBuf::from(&path),
        &yamc_nuclide::LoadScope::full(),
    )
    .expect("load Th232");
    let ti = nuclide.get_temp_idx("294").expect("294 K on Th232");
    let mt18 = nuclide.reactions[ti].get(&18).expect("Th232 carries MT18");
    let product = mt18
        .products
        .iter()
        .find(|p| p.is_particle_type(&ParticleType::Neutron))
        .expect("MT18 has a neutron product");
    let dist = product.distribution.first();
    assert!(
        matches!(
            dist,
            Some(AngleEnergyDistribution::CorrelatedAngleEnergy { .. })
        ),
        "Th232's prompt chi is expected to be a CorrelatedAngleEnergy"
    );
    let flat = nuclide.fission_chi_flat_cache.get_or_build(18, dist);
    let FissionChiFlat::Continuous {
        energy_grid, n_x, ..
    } = flat
    else {
        panic!("Th232's MT18 slot is not a Continuous marginal: {flat:?}");
    };
    eprintln!(
        "Th232 MT18 marginal: {} incident energies [{:.3e}, {:.3e}] eV, {} to {} E_out points per row",
        energy_grid.len(),
        energy_grid[0],
        energy_grid[energy_grid.len() - 1],
        n_x.iter().min().unwrap(),
        n_x.iter().max().unwrap()
    );
    assert!(energy_grid.len() >= 2);
    assert!(
        n_x.iter().all(|&m| m >= 2),
        "every incident row carries a table"
    );
}

/// GPU against CPU on a Th232 sphere, same seed: means agree and the spectrum
/// comparison keeps the lockstep signature (reduced chi-square 0.006 measured;
/// 0.23 before the CPU flattened the correlated marginal, when every
/// fissioning history, about 5% of them, left the GPU's stream at its first
/// fission).
#[test]
fn th232_gpu_run_walks_the_cpu_histories() {
    let Some(path) = cache_path() else { return };
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no f64 GPU");
        return;
    }
    let (mut cpu, cpu_t, cpu_s) = build_model(&path);
    cpu.simulate_transport(&cpu_s).expect("CPU run");
    let (mut gpu, gpu_t, gpu_s) = build_model(&path);
    let result = yamc::gpu::run_on_gpu(&mut gpu, &gpu_s).expect("GPU run");
    assert!(result.n_bank_relaunched > 0, "the fissile loop did not run");
    for (name, c, g) in [
        ("flux", &cpu_t[0], &gpu_t[0]),
        ("fission", &cpu_t[1], &gpu_t[1]),
    ] {
        let (cm, cs) = (c.get_mean()[0], c.get_std_dev()[0]);
        let (gm, gs) = (g.get_mean()[0], g.get_std_dev()[0]);
        let z = (gm - cm) / (cs * cs + gs * gs).sqrt();
        eprintln!(
            "{name}: CPU {cm:.5e} +- {cs:.2e}  GPU {gm:.5e} +- {gs:.2e}  ratio {:.5}  z {z:+.2}  std ratio {:.4}",
            gm / cm,
            gs / cs
        );
        assert!(
            ((gm - cm) / cm).abs() < 0.01,
            "{name} mean off by more than 1%"
        );
        assert!(
            (gs / cs - 1.0).abs() < 0.15,
            "{name} std_dev ratio {}",
            gs / cs
        );
    }
    let (cm, cs, gm, gs) = (
        cpu_t[2].get_mean(),
        cpu_t[2].get_std_dev(),
        gpu_t[2].get_mean(),
        gpu_t[2].get_std_dev(),
    );
    let edges = spectrum_edges();
    let mut sum_z2 = 0.0;
    let mut n_bins = 0usize;
    for i in 0..cm.len() {
        let sigma = (cs[i] * cs[i] + gs[i] * gs[i]).sqrt();
        if cm[i] <= 0.0 || sigma <= 0.0 {
            continue;
        }
        let z = (gm[i] - cm[i]) / sigma;
        eprintln!(
            "[{:>8.2e}, {:>8.2e}) eV  GPU/CPU {:.4}  z {z:+.2}",
            edges[i],
            edges[i + 1],
            gm[i] / cm[i]
        );
        sum_z2 += z * z;
        n_bins += 1;
    }
    let chi2_dof = sum_z2 / n_bins as f64;
    eprintln!("spectrum reduced chi-square {chi2_dof:.3} over {n_bins} bins");
    // Lockstep: with the prompt chi drawn from the same marginal at the same
    // stream position on both backends, the histories keep walking together.
    assert!(
        chi2_dof < 0.1,
        "reduced chi-square {chi2_dof:.3}: the GPU and CPU histories diverged (the prompt chi \
         is drawn differently at a fission)"
    );
}
