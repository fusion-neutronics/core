//! Fission yield of the struck nuclide in a mixed fissile material
//! (fusion-neutronics/core#93).
//!
//! In a multi-nuclide material the kernel selects the struck nuclide and
//! splits the reaction on that nuclide's own partials, but the number of
//! fission progeny (`nu_bar`) and their prompt / delayed split (`beta`) came
//! from the material's fission-weighted averages, so a U238 fission at 14 MeV
//! in a U235 / U238 mixture was given U235's yield whenever U235 dominated
//! `sigma_f`. The expected production per collision was unchanged, so no flux
//! ever showed it; per nuclide and per history it was wrong, and it broke the
//! lockstep with the CPU (which takes both from the fissioning nuclide) at the
//! first fission of every history.
//!
//! Two checks. The extractor's per-nuclide yield equals each nuclide's own
//! single-nuclide fold exactly, and the two nuclides differ where they should.
//! And a GPU run against a CPU run on the same seed on the mixture: the
//! histories are keyed on the same PCG streams on both backends, so the two
//! runs largely walk the same histories and the per-bin z-scores sit near
//! zero. Measured on this stack: U235 nu_bar 4.469 against U238 4.511 at
//! 14.06 MeV (delayed fractions 0.0020 and 0.0058), so the material average
//! was only 1% off for either nuclide and the lockstep barely notices the
//! fix (reduced chi-square 0.12 before, 0.14 after); the guard is that the
//! mixture path keeps agreeing.
//!
//! Run it (needs an f64 GPU and U235 / U238 in the cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_mixed_fissile_yield -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
const N_HISTORIES: usize = 200_000;
const RADIUS: f64 = 6.0;
const DENSITY: f64 = 19.0;

fn cache_pair() -> Option<(String, String)> {
    let u235 = yamc_test_cache::nuclide("U235");
    let u238 = yamc_test_cache::nuclide("U238");
    match (u235, u238) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => {
            eprintln!("skipping: U235 / U238 not in the cache");
            None
        }
    }
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

/// 50 / 50 atom U235 / U238 sphere, 14.06 MeV point source, fission bank on.
fn build_model(paths: &(String, String)) -> (Model, [Arc<Tally>; 3], TransportSettings) {
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
        HashMap::from([("U235".to_string(), 0.5), ("U238".to_string(), 0.5)]),
        "atom",
        "g/cm3",
        Some(DENSITY),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([
                ("U235".to_string(), paths.0.clone()),
                ("U238".to_string(), paths.1.clone()),
            ]),
            None,
        )
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("u".into()), Some(0));
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

fn load(path: &str) -> yamc_nuclide::nuclide::Nuclide {
    yamc_nuclide::nuclide_loader::load_nuclide(
        std::path::PathBuf::from(path),
        &yamc_nuclide::LoadScope::full(),
    )
    .expect("load")
}

/// The per-nuclide yield the kernel reads equals each nuclide's own fold, and
/// the two nuclides differ at the source energy.
#[test]
fn per_nuclide_yield_matches_each_nuclides_own_fold() {
    let Some(paths) = cache_pair() else { return };
    let u235 = load(&paths.0);
    let u238 = load(&paths.1);
    let grid: Vec<f64> =
        yamc_gpu::neutron::xs::union_energy_grid(&[(&u235, 0.5), (&u238, 0.5)], "294")
            .expect("union grid");
    let pool =
        yamc_gpu::extract_per_nuclide_inelastic(&[(&u235, 0.5), (&u238, 0.5)], "294", &grid, &grid)
            .expect("pool");
    let n = grid.len();
    assert_eq!(pool.nu_bar.len(), 2 * n);
    assert_eq!(pool.beta_delayed.len(), 2 * n);
    for (row, nuc) in [(0usize, &u235), (1usize, &u238)] {
        let own = yamc_gpu::extract_xs_from_nuclide(nuc, "294").expect("single");
        // The single-nuclide extraction folds on the nuclide's own grid, every
        // point of which is in the union grid: walk the nuclide's grid and
        // binary-search the union for each point.
        let mut checked = 0usize;
        for (j, le) in own.log_energy_grid.iter().enumerate() {
            let e = le.exp();
            let i = grid.partition_point(|&g| g < e * (1.0 - 1e-12));
            if i >= n || (grid[i] - e).abs() > 1e-9 * e {
                continue;
            }
            let (nu_pool, nu_own) = (pool.nu_bar[row * n + i], own.nu_bar[j]);
            let (b_pool, b_own) = (pool.beta_delayed[row * n + i], own.beta_delayed[j]);
            if nu_own > 0.0 {
                checked += 1;
                assert!(
                    (nu_pool - nu_own).abs() <= 1e-12 * nu_own,
                    "row {row} at {e:e} eV: pool nu_bar {nu_pool} vs own {nu_own}"
                );
                assert!(
                    (b_pool - b_own).abs() <= 1e-12 * b_own.max(1e-300),
                    "row {row} at {e:e} eV: pool beta {b_pool} vs own {b_own}"
                );
            }
        }
        assert!(
            checked > 1000,
            "row {row}: only {checked} grid points compared"
        );
    }
    let at = |row: usize, e: f64| {
        let i = grid.iter().position(|&g| g >= e).unwrap();
        (pool.nu_bar[row * n + i], pool.beta_delayed[row * n + i])
    };
    let (nu5, b5) = at(0, SOURCE_E);
    let (nu8, b8) = at(1, SOURCE_E);
    eprintln!("at 14.06 MeV: U235 nu_bar {nu5:.4} beta {b5:.5}; U238 nu_bar {nu8:.4} beta {b8:.5}");
    // The two yields differ, if not by much at 14 MeV: nu_bar by about 1%
    // (4.47 against 4.51) and the delayed fraction by nearly a factor three.
    assert!(
        (nu5 - nu8).abs() > 0.01,
        "the two nu_bar should differ at 14 MeV"
    );
    assert!(
        b5 > 0.0 && b8 > 2.0 * b5,
        "U238's delayed fraction should be well above U235's"
    );
}

/// GPU against CPU on the mixture, same seed: means agree and the spectrum
/// comparison keeps the lockstep signature (reduced chi-square well below one).
/// This guards the mixture path rather than discriminating the yield fix: at
/// 14 MeV the two nu_bar differ by 1%, so the material average changed the
/// progeny count of only a few fissions in a hundred and the chi-square moved
/// from 0.12 to 0.14 between the two builds.
#[test]
fn mixture_gpu_run_walks_the_cpu_histories() {
    let Some(paths) = cache_pair() else { return };
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no f64 GPU");
        return;
    }
    let (mut cpu, cpu_t, cpu_s) = build_model(&paths);
    cpu.simulate_transport(&cpu_s).expect("CPU run");
    let (mut gpu, gpu_t, gpu_s) = build_model(&paths);
    let result = yamc::gpu::run_on_gpu(&mut gpu, &gpu_s).expect("GPU run");
    assert!(result.n_bank_relaunched > 0, "the fissile loop did not run");
    for (name, c, g) in [
        ("flux", &cpu_t[0], &gpu_t[0]),
        ("fission", &cpu_t[1], &gpu_t[1]),
    ] {
        let (cm, cs) = (c.get_mean()[0], c.get_std_dev()[0]);
        let (gm, gs) = (g.get_mean()[0], g.get_std_dev()[0]);
        eprintln!(
            "{name}: CPU {cm:.5e} +- {cs:.2e}  GPU {gm:.5e} +- {gs:.2e}  ratio {:.5}  std ratio {:.4}",
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
    // Lockstep: with the struck nuclide's own yield the two backends emit the
    // same progeny count at every fission and keep walking the same histories.
    assert!(
        chi2_dof < 0.3,
        "reduced chi-square {chi2_dof:.3}: the GPU and CPU histories diverged (yield or split \
         differs at a collision)"
    );
}
