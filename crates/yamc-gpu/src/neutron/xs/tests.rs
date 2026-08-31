//! Tests for the `neutron::xs` extraction pipeline.

use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::Reaction;

/// Build a synthetic nuclide in memory with two MTs (elastic and
/// capture) on a small log-spaced grid. Avoids depending on
/// HDF5/JSON test fixtures so this test runs anywhere.
fn make_synthetic_nuclide() -> Nuclide {
    make_synthetic_nuclide_with(12.0, 2.0, 1.0)
}

/// Parameterised version: pick atomic-weight ratio, elastic XS
/// (constant), and capture XS coefficient (capture follows
/// `coef / sqrt(E)`). Lets the multi-material test build
/// nuclides with different curves.
fn make_synthetic_nuclide_with(
    atomic_weight_ratio: f64,
    sigma_elastic: f64,
    capture_coef: f64,
) -> Nuclide {
    make_synthetic_nuclide_with_grid(atomic_weight_ratio, sigma_elastic, capture_coef, 50)
}

/// Like [`make_synthetic_nuclide_with`] but with a parametrised grid length, so
/// tests can build nuclides whose energy grids differ in resolution.
fn make_synthetic_nuclide_with_grid(
    atomic_weight_ratio: f64,
    sigma_elastic: f64,
    capture_coef: f64,
    n_grid: usize,
) -> Nuclide {
    let temp = "294".to_string();
    let last = (n_grid - 1).max(1) as f64;
    let energy_grid: Vec<f64> = (0..n_grid)
        .map(|i| {
            let frac = (i as f64) / last;
            (1e-3_f64.ln() + frac * (1e3_f64.ln() - 1e-3_f64.ln())).exp()
        })
        .collect();

    let xs_elastic: Vec<f64> = vec![sigma_elastic; energy_grid.len()];
    let xs_capture: Vec<f64> = energy_grid
        .iter()
        .map(|&e| capture_coef / e.sqrt())
        .collect();

    let elastic = Reaction {
        cross_section: xs_elastic.clone().into(),
        threshold_idx: 0,
        energy: energy_grid.clone().into(),
        mt_number: MT_ELASTIC,
        q_value: 0.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    let capture = Reaction {
        cross_section: xs_capture.clone().into(),
        threshold_idx: 0,
        energy: energy_grid.clone().into(),
        mt_number: MT_CAPTURE,
        q_value: 0.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };

    let mut reactions_for_temp: HashMap<i32, Arc<Reaction>> = HashMap::new();
    reactions_for_temp.insert(MT_ELASTIC, Arc::new(elastic));
    reactions_for_temp.insert(MT_CAPTURE, Arc::new(capture));

    let mut energy_map = HashMap::new();
    energy_map.insert(temp.clone(), energy_grid.clone().into());

    Nuclide {
        name: Some("Synthetic".to_string()),
        element: None,
        atomic_symbol: Some("Sy".to_string()),
        atomic_number: Some(1),
        neutron_number: Some(0),
        mass_number: Some(1),
        atomic_weight_ratio: Some(atomic_weight_ratio),
        library: None,
        energy: Some(energy_map),
        reactions: vec![reactions_for_temp],
        fissionable: false,
        available_temperatures: vec![temp.clone()],
        loaded_temperatures: vec![temp],
        data_path: None,
        fission_nu: None,
        fast_xs: vec![],
        urr_data: vec![],
        urr_present: false,
        fission_photon_release: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        covariance: None,
        load_scope: Default::default(),
    }
}

#[test]
fn extracts_elastic_and_capture_arrays() {
    let nuclide = make_synthetic_nuclide();
    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    assert_eq!(xs.log_energy_grid.len(), 50);
    assert_eq!(xs.xs_elastic.len(), 50);
    assert_eq!(xs.xs_absorption.len(), 50);
    assert_eq!(xs.target_mass, 12.0);

    // Elastic should be ~2.0 everywhere.
    for &v in &xs.xs_elastic {
        assert!((v - 2.0).abs() < 1e-12, "elastic should be 2.0, got {v}");
    }
    // Capture follows 1/sqrt(E). Spot-check: at E ≈ 1, σ_a ≈ 1.
    // At E ≈ 0.001, σ_a ≈ sqrt(1000) ≈ 31.6.
    let mid_idx = xs.log_energy_grid.len() / 2;
    let e_mid = xs.log_energy_grid[mid_idx].exp();
    let expected = 1.0 / e_mid.sqrt();
    let got = xs.xs_absorption[mid_idx];
    assert!(
        (got - expected).abs() < 1e-9,
        "capture at E={e_mid}: got {got}, expected {expected}"
    );

    // log_energy_grid must be sorted ascending (kernel's binary
    // search depends on it).
    for i in 1..xs.log_energy_grid.len() {
        assert!(
            xs.log_energy_grid[i] > xs.log_energy_grid[i - 1],
            "log_energy_grid not sorted at i={i}"
        );
    }
}

#[test]
fn missing_temperature_errors_cleanly() {
    let nuclide = make_synthetic_nuclide();
    let err = extract_xs_from_nuclide(&nuclide, "600").unwrap_err();
    assert!(matches!(err, NuclideXsError::TemperatureNotLoaded(_)));
}

/// Build a synthetic nuclide that has an inelastic-level reaction
/// with a tabulated forward-peaked angular distribution on its
/// neutron product. Used to verify that the GPU XS extractor
/// pulls the angle data into the per-MT flat buffers correctly.
fn make_synthetic_nuclide_with_inelastic_angle() -> Nuclide {
    use yamc_nuclide::particle_type::ParticleType;
    use yamc_nuclide::reaction_product::{
        AngleDistribution, AngleEnergyDistribution, EnergyDistribution, ReactionProduct, Tabulated,
        TabulatedInterp,
    };

    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    // Forward-peaked angular distribution at one incident energy:
    // `mu` runs over a linear ramp, with most CDF mass near 0.95.
    let mu_x = vec![-1.0_f64, 0.9, 0.95, 1.0];
    let mu_c = vec![0.0_f64, 0.05, 0.5, 1.0];
    let angle = AngleDistribution {
        energy: vec![1e6_f64],
        mu: vec![Tabulated {
            x: mu_x,
            p: vec![],
            c: mu_c,
            interp: TabulatedInterp::Histogram,
        }],
    };
    let energy_dist = EnergyDistribution::LevelInelastic {
        threshold: 1e3,
        mass_ratio: 0.85,
    };
    let neutron_product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![],
        distribution: vec![AngleEnergyDistribution::UncorrelatedAngleEnergy {
            angle,
            energy: Some(energy_dist),
        }],
        product_yield: None,
    };

    let mt_51 = Reaction {
        cross_section: vec![1.0; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 51,
        q_value: -1e3,
        products: vec![neutron_product],
        scatter_in_cm: true,
        redundant: false,
    };
    nuclide.reactions[0].insert(51, Arc::new(mt_51));
    nuclide
}

/// The angular-distribution extractor must pull the tabulated
/// CDF for the right MT slot, leave every other slot zero, and
/// pack the buffers tight (variable-length CSR, issue #104).
#[test]
fn extracts_per_mt_angle_buffers() {
    let nuclide = make_synthetic_nuclide_with_inelastic_angle();
    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    // Tight layout (issue #104): per-slot counts are `[MT_INELASTIC_COUNT]`;
    // the incident-energy rows (`energy_grid` / `n_mu` / `interp`) are
    // `[total ae-rows]`; the (mu, cdf) points are `[total mu-points]`. Here
    // only slot 0 (MT 51) has data: 1 incident energy -> 1 ae-row, 4 mu pts.
    assert_eq!(xs.angle_n_energies.len(), MT_INELASTIC_COUNT);
    assert_eq!(xs.angle_energy_grid.len(), 1);
    assert_eq!(xs.angle_n_mu.len(), 1);
    assert_eq!(xs.angle_interp.len(), 1);
    assert_eq!(xs.angle_mu.len(), 4);
    assert_eq!(xs.angle_cdf.len(), xs.angle_mu.len());
    assert_eq!(xs.scatter_in_cm.len(), MT_INELASTIC_COUNT);

    // Slot 0 (MT 51) carries the data we plugged in.
    assert_eq!(xs.angle_n_energies[0], 1);
    assert_eq!(xs.scatter_in_cm[0], 1);
    assert_eq!(xs.angle_energy_grid[0], 1e6);
    assert_eq!(xs.angle_n_mu[0], 4);
    assert!((xs.angle_mu[0] - -1.0).abs() < 1e-12);
    assert!((xs.angle_mu[3] - 1.0).abs() < 1e-12);
    // CDF was provided pre-normalised; first/last values must be 0/1.
    assert!(xs.angle_cdf[0].abs() < 1e-12);
    assert!((xs.angle_cdf[3] - 1.0).abs() < 1e-12);

    // Every other MT slot is zero (no data).
    for slot in 1..MT_INELASTIC_COUNT {
        assert_eq!(xs.angle_n_energies[slot], 0);
        assert_eq!(xs.scatter_in_cm[slot], 0);
    }
}

/// Build a synthetic nuclide whose MT 91 reaction has a
/// `ContinuousTabular` outgoing-energy distribution. The
/// extractor must encode the table into the per-MT eout buffers
/// with `EOUT_KIND_CONTINUOUS_TABULAR` for slot 40 (MT 91 −
/// MT_INELASTIC_FIRST = 40) and leave every other slot at the
/// `LevelInelastic` default.
fn make_synthetic_nuclide_with_continuum_eout() -> Nuclide {
    use yamc_nuclide::particle_type::ParticleType;
    use yamc_nuclide::reaction_product::{
        AngleDistribution, AngleEnergyDistribution, EnergyDistribution, ReactionProduct, Tabulated,
        TabulatedInterp, TabulatedProbability,
    };

    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    // Plain isotropic angle so the slot has tabulated data on
    // both axes (the kernel reads both buffers per slot).
    let angle = AngleDistribution {
        energy: vec![1e6_f64],
        mu: vec![Tabulated {
            x: vec![-1.0_f64, 1.0],
            p: vec![],
            c: vec![0.0_f64, 1.0],
            interp: TabulatedInterp::Histogram,
        }],
    };
    // ContinuousTabular at one incident energy with a 3-point
    // E_out CDF: 1, 5, 9 MeV with cumulative 0, 0.5, 1.0.
    let eout_dist = TabulatedProbability::Tabulated {
        x: vec![1e6_f64, 5e6, 9e6],
        p: vec![0.5, 0.0, 0.5],
        c: vec![0.0_f64, 0.5, 1.0],
        interp: TabulatedInterp::Histogram,
        n_discrete: 0,
    };
    let energy_dist = EnergyDistribution::ContinuousTabular {
        energy: vec![1e6_f64],
        energy_out: vec![eout_dist],
        histogram_interp: false,
    };
    let neutron_product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![],
        distribution: vec![AngleEnergyDistribution::UncorrelatedAngleEnergy {
            angle,
            energy: Some(energy_dist),
        }],
        product_yield: None,
    };

    let mt_91 = Reaction {
        cross_section: vec![1.0; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 91,
        q_value: -1e3,
        products: vec![neutron_product],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(91, Arc::new(mt_91));
    nuclide
}

/// Slice C: extracted GpuNuclideXs must carry the per-MT eout
/// buffers with the right shape and an
/// `EOUT_KIND_CONTINUOUS_TABULAR` discriminant on the slot whose
/// reaction had `ContinuousTabular` data.
#[test]
fn extracts_per_mt_eout_buffers() {
    let nuclide = make_synthetic_nuclide_with_continuum_eout();
    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    // Tight layout (issue #104): per-slot scalars are `[MT_INELASTIC_COUNT]`;
    // the incident-energy rows (`energy_grid` / `n_x`) are `[total ae-rows]`;
    // the (x, cdf) points are `[total x-points]`. Here only the MT 91 slot has
    // data: 1 incident energy -> 1 ae-row, 3 x-points.
    assert_eq!(xs.eout_kind.len(), MT_INELASTIC_COUNT);
    assert_eq!(xs.eout_n_energies.len(), MT_INELASTIC_COUNT);
    assert_eq!(xs.eout_energy_grid.len(), 1);
    assert_eq!(xs.eout_n_x.len(), 1);
    assert_eq!(xs.eout_x.len(), 3);
    assert_eq!(xs.eout_cdf.len(), xs.eout_x.len());

    // Slot 40 is MT 91. ContinuousTabular kind, n_energies = 1,
    // n_x = 3, mapped CDF endpoints. Every earlier slot is empty
    // (0 rows), so the MT 91 slot's ae-row is global index 0 and its
    // x-points start at global index 0.
    let slot = 91 - MT_INELASTIC_FIRST as usize;
    assert_eq!(xs.eout_kind[slot], EOUT_KIND_CONTINUOUS_TABULAR);
    assert_eq!(xs.eout_n_energies[slot], 1);
    let ae_off = 0usize;
    assert!((xs.eout_energy_grid[ae_off] - 1e6).abs() < 1e-9);
    assert_eq!(xs.eout_n_x[ae_off], 3);
    let x_off = 0usize;
    assert!((xs.eout_x[x_off] - 1e6).abs() < 1e-9);
    assert!((xs.eout_x[x_off + 2] - 9e6).abs() < 1e-9);
    assert!(xs.eout_cdf[x_off].abs() < 1e-12);
    assert!((xs.eout_cdf[x_off + 2] - 1.0).abs() < 1e-12);

    // Every other slot defaults to LevelInelastic + no data.
    for s in 0..MT_INELASTIC_COUNT {
        if s == slot {
            continue;
        }
        assert_eq!(xs.eout_kind[s], EOUT_KIND_LEVEL_INELASTIC);
        assert_eq!(xs.eout_n_energies[s], 0);
    }
}

/// End-to-end: build a synthetic nuclide → extract XS → run the
/// GPU multi-step kernel with the extracted arrays. Sanity-checks
/// that the extraction format is what the kernel expects, and
/// that the resulting transport produces sensible aggregate stats.
///
/// Gated to non-macOS because `crate::neutron::probes` is not built on
/// macOS (no f64 hardware path; cubecl deps excluded).
#[cfg(not(target_os = "macos"))]
#[test]
fn extract_and_run_multi_step() {
    use crate::neutron::probes::multi_step::run_multi_step;
    use crate::{GpuContext, GpuInitError};

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let nuclide = make_synthetic_nuclide();
    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    // Sanity: the extracted grid is what the kernel expects.
    assert_eq!(xs.log_energy_grid.len(), xs.xs_elastic.len());
    assert_eq!(xs.log_energy_grid.len(), xs.xs_absorption.len());

    // Run a small transport: 10k particles, 50 max steps. With the
    // synthetic XS (σ_e = 2 constant, σ_a ~ 1/sqrt(E)), particles
    // tend to thermalise (energy drops on each scatter) and absorb
    // somewhere in the lower-energy region where σ_a is largest.
    // Pick starting energy at the high end of the grid so there's
    // plenty of room to thermalise.
    let n = 10_000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies_in: Vec<f64> = vec![100.0; n];
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        directions.extend_from_slice(&[0.0, 0.0, 1.0]);
    }

    let log_e_min = xs.log_energy_grid[0];
    let log_e_max = *xs.log_energy_grid.last().unwrap();
    let n_bins = 8usize;

    let r = run_multi_step(
        &ctx,
        &seeds,
        &energies_in,
        &directions,
        &xs.log_energy_grid,
        &xs.xs_elastic,
        &xs.xs_absorption,
        xs.target_mass,
        log_e_min,
        log_e_max,
        n_bins,
        50,
    );

    // Most particles should be absorbed within 50 steps. With 1/v
    // capture, energy decays geometrically so absorption probability
    // stays bounded above zero -- virtually all should die.
    let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
    let total_tally: f64 = r.absorption_per_bin.iter().sum();
    let mean_steps: f64 = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    println!(
            "ENDF-style run: alive at cap = {alive_count} of {n}, mean steps = {mean_steps}, total tally = {total_tally}"
        );

    assert!(
        alive_count < n / 100,
        "{alive_count} particles still alive at cap suggests transport is wrong"
    );
    // Tally should be positive and reasonable. With energy-dependent
    // σ_a, we don't have a clean closed-form, but it should be
    // well below the trivial upper bound of `n * mean_steps * σ_a_max
    // * d_avg` which is ~1e6 here. A loose lower bound: each absorbed
    // particle scores at least the contribution of its final track,
    // so total > 0.
    assert!(total_tally > 0.0, "tally must be positive");
}

/// Multi-material aggregation produces XS that's the
/// density-weighted sum of per-nuclide XS. With two synthetic
/// nuclides (different XS values), the aggregated arrays should
/// equal `density_a * xs_a + density_b * xs_b` at every grid
/// point.
#[test]
fn aggregates_multi_nuclide_xs_correctly() {
    // Nuclide A: σ_e = 2.0, σ_a = 1.0/sqrt(E), A = 12
    // Nuclide B: σ_e = 5.0, σ_a = 0.3/sqrt(E), A = 56
    let a = make_synthetic_nuclide_with(12.0, 2.0, 1.0);
    let b = make_synthetic_nuclide_with(56.0, 5.0, 0.3);

    let dens_a = 0.4_f64;
    let dens_b = 0.6_f64;

    let mat = extract_material_xs(&[(&a, dens_a), (&b, dens_b)], "294").unwrap();

    // Elastic everywhere should be `0.4*2 + 0.6*5 = 0.8 + 3.0 = 3.8`.
    for &v in &mat.xs_elastic {
        assert!(
            (v - (dens_a * 2.0 + dens_b * 5.0)).abs() < 1e-12,
            "expected aggregated elastic {}, got {v}",
            dens_a * 2.0 + dens_b * 5.0
        );
    }
    // Absorption: density-weighted 1/sqrt(E) curve. Spot-check at
    // mid grid.
    let mid_idx = mat.log_energy_grid.len() / 2;
    let e_mid = mat.log_energy_grid[mid_idx].exp();
    let expected_a = dens_a * (1.0 / e_mid.sqrt()) + dens_b * (0.3 / e_mid.sqrt());
    let got = mat.xs_absorption[mid_idx];
    assert!(
        (got - expected_a).abs() < 1e-9,
        "expected aggregated absorption {expected_a}, got {got}"
    );

    // Effective target_mass = (0.4·12 + 0.6·56) / (0.4 + 0.6) = 38.4
    let expected_mass = (dens_a * 12.0 + dens_b * 56.0) / (dens_a + dens_b);
    assert!(
        (mat.target_mass - expected_mass).abs() < 1e-12,
        "expected effective mass {expected_mass}, got {}",
        mat.target_mass
    );
}

/// Issue #88 (extends the #74 order-independence regression): the dual energy
/// grid must be order-independent. The FINE grid is the exact UNION of the
/// material's per-nuclide grids -- mirroring the CPU's
/// `unified_energy_grid_neutron`, so every isotope's resonances survive in the
/// collision / selection cross sections -- and the COARSE grid is the finest
/// single per-nuclide grid (backing only the per-MT inelastic buffers). Both
/// must be byte-identical regardless of the order nuclides are passed in, which
/// is what makes the GPU result deterministic and resonance-faithful.
#[test]
fn master_grid_is_union_and_order_independent() {
    // Coarse 20-point nuclide + fine 200-point nuclide over the same
    // [1e-3, 1e3] range. The two log grids share only their endpoints, so the
    // union has 20 + 200 - 2 = 218 points and the finest single grid has 200.
    let coarse = make_synthetic_nuclide_with_grid(56.0, 3.0, 0.5, 20);
    let fine = make_synthetic_nuclide_with_grid(12.0, 2.0, 0.7, 200);

    let mat_cf = extract_material_xs(&[(&coarse, 0.5), (&fine, 0.5)], "294").unwrap();
    let mat_fc = extract_material_xs(&[(&fine, 0.5), (&coarse, 0.5)], "294").unwrap();

    // FINE grid = union of the per-nuclide grids (218 points), order-independent.
    assert_eq!(
        mat_cf.log_energy_grid.len(),
        218,
        "fine grid should be the union of the per-nuclide grids (218 points)"
    );
    // COARSE grid = finest single grid (200 points), order-independent.
    assert_eq!(
        mat_cf.coarse_log_energy_grid.len(),
        200,
        "coarse grid should be the finest nuclide's grid (200 points)"
    );
    assert_eq!(
        mat_fc.log_energy_grid.len(),
        mat_cf.log_energy_grid.len(),
        "fine grid length must not depend on nuclide order"
    );
    assert_eq!(
        mat_fc.coarse_log_energy_grid.len(),
        mat_cf.coarse_log_energy_grid.len(),
        "coarse grid length must not depend on nuclide order"
    );
    // Both grids (and hence every interpolation index) byte-identical regardless
    // of argument order.
    for (a, b) in mat_cf
        .log_energy_grid
        .iter()
        .zip(mat_fc.log_energy_grid.iter())
    {
        assert_eq!(a, b, "fine grid differs between nuclide orderings");
    }
    for (a, b) in mat_cf
        .coarse_log_energy_grid
        .iter()
        .zip(mat_fc.coarse_log_energy_grid.iter())
    {
        assert_eq!(a, b, "coarse grid differs between nuclide orderings");
    }
}

/// A single-nuclide material's master grid is exactly that nuclide's grid
/// (the finest-of-one), so the single-nuclide path stays unchanged.
#[test]
fn single_nuclide_master_grid_unchanged() {
    let n = make_synthetic_nuclide_with_grid(56.0, 3.0, 0.5, 137);
    let mat = extract_material_xs(&[(&n, 1.0)], "294").unwrap();
    assert_eq!(mat.log_energy_grid.len(), 137);
}

/// Empty-material call must error cleanly, not panic.
#[test]
fn empty_material_errors() {
    let err = extract_material_xs(&[], "294").unwrap_err();
    assert!(matches!(err, NuclideXsError::EmptyMaterial));
}

/// Zero-density material must error cleanly.
#[test]
fn zero_density_material_errors() {
    let n = make_synthetic_nuclide();
    let err = extract_material_xs(&[(&n, 0.0)], "294").unwrap_err();
    assert!(matches!(err, NuclideXsError::ZeroTotalDensity));
}

/// End-to-end: aggregate two synthetic nuclides into a material →
/// run the GPU multi-step kernel against the aggregate. Sanity
/// checks that the aggregation flows through cleanly.
///
/// Gated to non-macOS because `crate::neutron::probes` is not built on
/// macOS (no f64 hardware path; cubecl deps excluded).
#[cfg(not(target_os = "macos"))]
#[test]
fn multi_material_runs_on_gpu() {
    use crate::neutron::probes::multi_step::run_multi_step;
    use crate::{GpuContext, GpuInitError};

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let a = make_synthetic_nuclide_with(12.0, 2.0, 1.0);
    let b = make_synthetic_nuclide_with(56.0, 5.0, 0.3);
    let mat = extract_material_xs(&[(&a, 0.4), (&b, 0.6)], "294").unwrap();

    let n = 10_000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies_in: Vec<f64> = vec![100.0; n];
    let mut directions = Vec::with_capacity(3 * n);
    for _ in 0..n {
        directions.extend_from_slice(&[0.0, 0.0, 1.0]);
    }
    let log_e_min = mat.log_energy_grid[0];
    let log_e_max = *mat.log_energy_grid.last().unwrap();

    // 200 steps because this material is mostly elastic
    // (σ_e dominates σ_a at most energies); particles bounce
    // many times before absorption.
    let r = run_multi_step(
        &ctx,
        &seeds,
        &energies_in,
        &directions,
        &mat.log_energy_grid,
        &mat.xs_elastic,
        &mat.xs_absorption,
        mat.target_mass,
        log_e_min,
        log_e_max,
        8,
        200,
    );

    let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
    let total_tally: f64 = r.absorption_per_bin.iter().sum();
    let mean_steps = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;

    println!(
            "multi-material run: alive at cap = {alive_count} of {n}, mean steps = {mean_steps}, total tally = {total_tally}"
        );

    assert!(alive_count < n / 50, "too many alive at cap");
    assert!(total_tally > 0.0, "tally must be positive");
}

/// Build a synthetic nuclide whose MT 91 reaction has a
/// `CorrelatedAngleEnergy` distribution. The extractor must
/// flag the slot as `EOUT_KIND_CORRELATED`, populate the corr_*
/// buffers, and leave the `eout_x` / `eout_cdf` buffers empty
/// for that slot.
fn make_synthetic_nuclide_with_correlated() -> Nuclide {
    use yamc_nuclide::particle_type::ParticleType;
    use yamc_nuclide::reaction_product::{AngleEnergyDistribution, ReactionProduct};
    use yamc_nuclide::secondary_correlated::{
        CorrTable, CorrelatedAngleEnergy, Interpolation, Tabular as CorrTabular,
    };

    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    let angle_at_eout = CorrTabular {
        x: vec![-1.0_f64, 1.0],
        p: vec![],
        c: vec![0.0_f64, 1.0],
        interpolation: Interpolation::Histogram,
        n_discrete: 0,
    };
    let corr_table = CorrTable {
        interpolation: Interpolation::Histogram,
        n_discrete: 0,
        e_out: vec![1e6_f64, 5e6, 9e6],
        p: vec![0.5, 0.0, 0.5],
        c: vec![0.0_f64, 0.5, 1.0],
        angle: vec![angle_at_eout.clone(), angle_at_eout.clone(), angle_at_eout],
    };
    let correlated = CorrelatedAngleEnergy {
        energy: vec![1e6_f64],
        distributions: vec![corr_table],
    };
    let neutron_product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![],
        distribution: vec![AngleEnergyDistribution::CorrelatedAngleEnergy { correlated }],
        product_yield: None,
    };
    let mt_91 = Reaction {
        cross_section: vec![1.0; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 91,
        q_value: -1e3,
        products: vec![neutron_product],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(91, Arc::new(mt_91));
    nuclide
}

/// Slice D: extracted GpuNuclideXs flags MT 91 with
/// `EOUT_KIND_CORRELATED` and populates the corr_* buffers
/// with the right shape and CDF endpoints.
#[test]
fn extracts_per_mt_corr_buffers() {
    let nuclide = make_synthetic_nuclide_with_correlated();
    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    // Tight layout (issue #104): per-slot counts are `[MT_INELASTIC_COUNT]`;
    // the incident-energy ae-rows (`energy_grid` / `n_x` / `interp` /
    // `n_discrete`) are `[total ae-rows]`; the E_out x-points (`x` / `cdf` /
    // `p` / `n_mu` / `mu_interp`) are `[total x-points]`; the mu points (`mu`
    // / `mu_cdf` / `mu_pdf`) are `[total mu-points]`. Here only slot 40 (MT
    // 91) has data: 1 incident energy -> 1 ae-row, 3 E_out points, each with
    // a 2-point mu sub-table -> 3 x-points, 6 mu-points.
    assert_eq!(xs.corr_n_energies.len(), MT_INELASTIC_COUNT);
    assert_eq!(xs.corr_energy_grid.len(), 1);
    assert_eq!(xs.corr_n_x.len(), 1);
    assert_eq!(xs.corr_x.len(), 3);
    assert_eq!(xs.corr_cdf.len(), xs.corr_x.len());
    assert_eq!(xs.corr_n_mu.len(), 3);
    assert_eq!(xs.corr_mu.len(), 6);
    assert_eq!(xs.corr_mu_cdf.len(), xs.corr_mu.len());

    // Slot 40 is MT 91. Correlated kind, 1 incident energy, 3 E_out
    // points, populated mu sub-table.
    let slot = (91 - MT_INELASTIC_FIRST) as usize;
    assert_eq!(xs.eout_kind[slot], EOUT_KIND_CORRELATED);
    // The eout_* buffers stay empty for correlated slots.
    assert_eq!(xs.eout_n_energies[slot], 0);
    // The corr_* buffers carry the actual data. With tight CSR the per-nuclide
    // buffers concatenate only the populated slot, so the single ae-row /
    // x-points / mu-points start at index 0; the per-slot CSR base is built
    // later in translate.rs.
    assert_eq!(xs.corr_n_energies[slot], 1);
    assert!((xs.corr_energy_grid[0] - 1e6).abs() < 1e-9);
    assert_eq!(xs.corr_n_x[0], 3);
    assert!((xs.corr_x[0] - 1e6).abs() < 1e-9);
    assert!((xs.corr_x[2] - 9e6).abs() < 1e-9);
    assert!(xs.corr_cdf[0].abs() < 1e-12);
    assert!((xs.corr_cdf[2] - 1.0).abs() < 1e-12);
    // Each (E_in, E_out) bin has a 2-point angular sub-table.
    assert_eq!(xs.corr_n_mu[0], 2);
    assert_eq!(xs.corr_n_mu[1], 2);
    assert!((xs.corr_mu[0] - -1.0).abs() < 1e-12);
    assert!((xs.corr_mu[1] - 1.0).abs() < 1e-12);

    // Every other slot defaults to LevelInelastic, no corr data.
    for s in 0..MT_INELASTIC_COUNT {
        if s == slot {
            continue;
        }
        assert_eq!(xs.eout_kind[s], EOUT_KIND_LEVEL_INELASTIC);
        assert_eq!(xs.corr_n_energies[s], 0);
    }
}

/// Regression: the CM/lab `scatter_in_cm` flag is a property of the
/// reaction, not of which angle-distribution variant its neutron
/// product carries. A reaction whose product is `CorrelatedAngleEnergy`
/// (or KalbachMann / NBodyPhaseSpace) has no `UncorrelatedAngleEnergy`
/// angle, so the per-MT angular flatten takes an early-out path -- which
/// previously returned `scatter_in_cm = 0` (lab) even for a CM-framed
/// reaction. The kernel then skipped the CM->lab boost and biased the
/// outgoing-energy spectrum (observed end-to-end as Li6 MT 32 (n,n'd)
/// piling flux ~0.5 MeV too low). The flag must propagate regardless of
/// variant.
#[test]
fn correlated_reaction_propagates_cm_frame_flag() {
    let mut nuclide = make_synthetic_nuclide_with_correlated();
    // Rebuild MT 91's reaction with the CM frame flag set. Reuse the
    // existing correlated product; only the frame flag changes.
    let temp_idx = 0;
    let existing = nuclide.reactions[temp_idx].remove(&91).unwrap();
    let mt_91_cm = Reaction {
        scatter_in_cm: true,
        ..(*existing).clone()
    };
    nuclide.reactions[temp_idx].insert(91, Arc::new(mt_91_cm));

    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();
    let slot = (91 - MT_INELASTIC_FIRST) as usize;
    // The product is CorrelatedAngleEnergy (no Uncorrelated angle), so
    // this slot exercises the early-out branch of the per-MT angular flatten.
    assert_eq!(xs.eout_kind[slot], EOUT_KIND_CORRELATED);
    assert_eq!(
        xs.scatter_in_cm[slot], 1,
        "CM-framed correlated reaction must report scatter_in_cm = 1"
    );
}

/// Slice E: a nuclide with MT 16 (n,2n) data must populate slot
/// 41 of the per-MT XS buffer (and leave slots 0..=40 / 42 zero
/// for that MT). Verifies the `MT_SLOTS` re-indexing wires
/// correctly.
#[test]
fn extracts_mt_16_into_slot_41() {
    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();
    // Plug in a MT 16 reaction with a flat 0.5 b xs.
    let mt_16 = Reaction {
        cross_section: vec![0.5; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 16,
        q_value: -11_197_000.0, // typical Fe56 (n,2n) Q
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(16, Arc::new(mt_16));

    let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();
    let n_grid = xs.log_energy_grid.len();
    // Slot 41 should carry the MT 16 xs; slots 0..=40 + 42 should
    // be zero (no MT data plugged in for those slots).
    let slot_41 = 41;
    for i in 0..n_grid {
        let off = slot_41 * n_grid + i;
        assert!(
            (xs.xs_inelastic_per_mt[off] - 0.5).abs() < 1e-12,
            "slot 41 (MT 16) xs should be 0.5, got {}",
            xs.xs_inelastic_per_mt[off]
        );
    }
    for slot in 0..MT_INELASTIC_COUNT {
        if slot == slot_41 {
            continue;
        }
        for i in 0..n_grid {
            let off = slot * n_grid + i;
            assert!(
                xs.xs_inelastic_per_mt[off].abs() < 1e-12,
                "slot {slot} should have zero xs (only MT 16 was plugged in)"
            );
        }
    }
    assert!((xs.q_inelastic_per_mt[slot_41] - -11_197_000.0).abs() < 1.0);
}

/// Slice F: charged-particle-out + neutron MTs (MT 22 / 28 / 32 /
/// 33 / 34) must occupy their fixed slots (43..=47) in the per-MT
/// XS buffer with `MT_YIELDS == 1` (single-neutron-out). Verifies
/// the slice-F additions to `MT_SLOTS` and `MT_YIELDS` wire
/// through `extract_xs_from_nuclide` correctly so that channels
/// previously folded into `xs_absorption` are now sampled as
/// inelastic-branch slots with weight 1.0.
#[test]
fn extracts_slice_f_charged_particle_mts_into_slots_43_to_47() {
    // Each entry: (MT, expected slot, distinct flat xs in barns,
    // typical Fe56 Q-value).
    let cases: [(i32, usize, f64, f64); 5] = [
        (22, 43, 0.10, -7_614_000.0),  // (n,n'α)
        (28, 44, 0.07, -7_870_000.0),  // (n,n'p)
        (32, 45, 0.04, -10_310_000.0), // (n,n'd)
        (33, 46, 0.03, -11_130_000.0), // (n,n't)
        (34, 47, 0.02, -10_730_000.0), // (n,n'³He)
    ];

    for (mt, expected_slot, expected_xs, expected_q) in cases {
        assert_eq!(
            MT_SLOTS[expected_slot], mt,
            "MT_SLOTS[{expected_slot}] should be MT {mt}"
        );
        assert_eq!(
            MT_YIELDS[expected_slot], 1,
            "MT_YIELDS[{expected_slot}] should be 1 (single-neutron-out)"
        );

        let mut nuclide = make_synthetic_nuclide();
        let temp = "294".to_string();
        let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();
        let rxn = Reaction {
            cross_section: vec![expected_xs; energy_grid.len()].into(),
            threshold_idx: 0,
            energy: energy_grid.clone(),
            mt_number: mt,
            q_value: expected_q,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };
        nuclide.reactions[0].insert(mt, Arc::new(rxn));

        let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();
        let n_grid = xs.log_energy_grid.len();

        for i in 0..n_grid {
            let off = expected_slot * n_grid + i;
            assert!(
                (xs.xs_inelastic_per_mt[off] - expected_xs).abs() < 1e-12,
                "slot {expected_slot} (MT {mt}) xs should be {expected_xs}, got {}",
                xs.xs_inelastic_per_mt[off]
            );
        }
        for slot in 0..MT_INELASTIC_COUNT {
            if slot == expected_slot {
                continue;
            }
            for i in 0..n_grid {
                let off = slot * n_grid + i;
                assert!(
                    xs.xs_inelastic_per_mt[off].abs() < 1e-12,
                    "slot {slot} should have zero xs (only MT {mt} was plugged in)"
                );
            }
        }
        assert!((xs.q_inelastic_per_mt[expected_slot] - expected_q).abs() < 1.0);
    }
}

/// Slice F: pulling MT 22 / 28 / 32 / 33 / 34 into the inelastic
/// branch must remove their xs from the derived `xs_absorption`
/// slot. Plug a flat 0.5 b xs into one slice-F MT; verify that
/// `xs_absorption` is exactly the synthetic nuclide's
/// pre-existing absorption value (i.e. `σ_t − σ_e − σ_inelastic`
/// drops the new MT cleanly) rather than absorbing the 0.5 b.
#[test]
fn slice_f_mt_xs_subtracted_from_absorption_derivation() {
    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    // Baseline: extract before adding any slice-F MT to capture
    // the synthetic nuclide's pre-existing absorption.
    let xs_before = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    // Add MT 22 with flat 0.5 b xs.
    let mt_22 = Reaction {
        cross_section: vec![0.5; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 22,
        q_value: -7_614_000.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(22, Arc::new(mt_22));

    // After: σ_t now includes the 0.5 b (it's a non-redundant
    // reaction). xs_absorption derivation subtracts elastic +
    // every MT_SLOTS entry, so the 0.5 b lands in xs_inelastic
    // (slot 43) instead of xs_absorption.
    let xs_after = extract_xs_from_nuclide(&nuclide, "294").unwrap();
    let n_grid = xs_after.log_energy_grid.len();
    for i in 0..n_grid {
        assert!(
                (xs_after.xs_absorption[i] - xs_before.xs_absorption[i]).abs() < 1e-12,
                "absorption[{i}] changed by {} -- slice-F MT should be in xs_inelastic, not xs_absorption",
                xs_after.xs_absorption[i] - xs_before.xs_absorption[i]
            );
        // And the 0.5 b shows up in xs_inelastic at slot 43.
        assert!(
            (xs_after.xs_inelastic[i] - xs_before.xs_inelastic[i] - 0.5).abs() < 1e-12,
            "xs_inelastic[{i}] should grow by exactly 0.5 b after adding MT 22"
        );
    }
}

/// Issue #106: the breakup channels (MT 11 / 29 / 30 / 35 / 36 / 42) must
/// occupy slots 56..=61 with the multiplicity their reaction name implies, and
/// their xs must reach `xs_inelastic_per_mt` rather than the derived
/// absorption. These are the MTs that made the GPU over-absorb on TENDL data,
/// where MT 11 and 42 appear in the great majority of nuclides.
#[test]
fn extracts_breakup_mts_into_slots_56_to_61() {
    // (MT, expected slot, expected constant-yield reference, flat xs, Q).
    let cases: [(i32, usize, u32, f64, f64); 6] = [
        (11, 56, 2, 0.09, -20_000_000.0), // (n,2nd)
        (29, 57, 1, 0.08, -18_000_000.0), // (n,n'3α)
        (30, 58, 2, 0.07, -22_000_000.0), // (n,2n2α)
        (35, 59, 1, 0.06, -24_000_000.0), // (n,n'd2α)
        (36, 60, 1, 0.05, -26_000_000.0), // (n,n't2α)
        (42, 61, 3, 0.04, -28_000_000.0), // (n,3np)
    ];

    for (mt, expected_slot, expected_yield, expected_xs, expected_q) in cases {
        assert_eq!(
            MT_SLOTS[expected_slot], mt,
            "MT_SLOTS[{expected_slot}] should be MT {mt}"
        );
        assert_eq!(
            MT_YIELDS[expected_slot], expected_yield,
            "MT_YIELDS[{expected_slot}] should be {expected_yield} for MT {mt}"
        );

        let mut nuclide = make_synthetic_nuclide();
        let temp = "294".to_string();
        let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();
        let rxn = Reaction {
            cross_section: vec![expected_xs; energy_grid.len()].into(),
            threshold_idx: 0,
            energy: energy_grid.clone(),
            mt_number: mt,
            q_value: expected_q,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };
        nuclide.reactions[0].insert(mt, Arc::new(rxn));

        let xs = extract_xs_from_nuclide(&nuclide, "294").unwrap();
        let n_grid = xs.log_energy_grid.len();
        for i in 0..n_grid {
            let off = expected_slot * n_grid + i;
            assert!(
                (xs.xs_inelastic_per_mt[off] - expected_xs).abs() < 1e-12,
                "slot {expected_slot} (MT {mt}) xs should be {expected_xs}, got {}",
                xs.xs_inelastic_per_mt[off]
            );
        }
        assert!((xs.q_inelastic_per_mt[expected_slot] - expected_q).abs() < 1.0);
    }
}

/// Issue #106: the whole point of slotting the breakup channels is that their
/// cross section stops landing in the derived absorption, which is what killed
/// neutrons on the GPU that the CPU scattered. Plug MT 11 into the synthetic
/// nuclide and assert absorption is untouched while `xs_inelastic` grows by
/// exactly the added barn.
#[test]
fn breakup_mt_xs_subtracted_from_absorption_derivation() {
    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    let xs_before = extract_xs_from_nuclide(&nuclide, "294").unwrap();

    let mt_11 = Reaction {
        cross_section: vec![0.5; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 11,
        q_value: -20_000_000.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(11, Arc::new(mt_11));

    let xs_after = extract_xs_from_nuclide(&nuclide, "294").unwrap();
    let n_grid = xs_after.log_energy_grid.len();
    for i in 0..n_grid {
        assert!(
            (xs_after.xs_absorption[i] - xs_before.xs_absorption[i]).abs() < 1e-12,
            "absorption[{i}] changed by {} -- MT 11 belongs in xs_inelastic",
            xs_after.xs_absorption[i] - xs_before.xs_absorption[i]
        );
        assert!(
            (xs_after.xs_inelastic[i] - xs_before.xs_inelastic[i] - 0.5).abs() < 1e-12,
            "xs_inelastic[{i}] should grow by exactly 0.5 b after adding MT 11"
        );
    }
}

/// Phase 2b of the GPU particle-bank work: the per-nuclide macroscopic
/// total xs table that feeds the on-device nuclide selector.
///
/// Two checks pin it to the CPU semantics:
///  1. Each row equals `density * (sigma_elastic + capture_coef / sqrt(E))`,
///     i.e. `density * total_micro_xs` -- exactly what the CPU stores in
///     `Material::macroscopic_xs_neutron_total_by_nuclide` and walks in
///     `sample_interacting_nuclide`.
///  2. Summed over nuclides the rows reproduce the aggregate macroscopic
///     total assembled by `extract_material_xs` (elastic + absorption here;
///     the synthetic nuclides carry no inelastic), so the per-nuclide split
///     is consistent with the production aggregate buffers.
#[test]
fn per_nuclide_macro_total_xs_matches_cpu() {
    // Nuclide A: sigma_e = 2.0, sigma_a = 1.0/sqrt(E); B: 5.0, 0.3/sqrt(E).
    let a = make_synthetic_nuclide_with(12.0, 2.0, 1.0);
    let b = make_synthetic_nuclide_with(56.0, 5.0, 0.3);
    let dens_a = 0.4_f64;
    let dens_b = 0.6_f64;
    let pairs = [(&a, dens_a), (&b, dens_b)];

    // Same grid extract_material_xs derives (the first nuclide's grid), so
    // the per-nuclide rows align index-for-index with the aggregate buffers.
    let grid = a.energy.as_ref().unwrap().get("294").unwrap().clone();

    let per = extract_per_nuclide_macro_total_xs(&pairs, "294", &grid).unwrap();
    assert_eq!(per.n_nuclides, 2);
    assert_eq!(per.n_grid, grid.len());
    assert_eq!(per.macro_total_xs.len(), 2 * grid.len());

    // (1) Per-nuclide rows == analytic density * total_micro.
    for (i, &e) in grid.iter().enumerate() {
        let expect_a = dens_a * (2.0 + 1.0 / e.sqrt());
        let expect_b = dens_b * (5.0 + 0.3 / e.sqrt());
        let got_a = per.macro_total_xs[i];
        let got_b = per.macro_total_xs[grid.len() + i];
        assert!(
            (got_a - expect_a).abs() < 1e-12,
            "nuclide A @ E={e}: want {expect_a}, got {got_a}"
        );
        assert!(
            (got_b - expect_b).abs() < 1e-12,
            "nuclide B @ E={e}: want {expect_b}, got {got_b}"
        );
    }

    // (2) Rows sum to the aggregate macroscopic total (no inelastic here).
    let mat = extract_material_xs(&pairs, "294").unwrap();
    assert_eq!(mat.log_energy_grid.len(), grid.len());
    for i in 0..grid.len() {
        let row_sum = per.macro_total_xs[i] + per.macro_total_xs[grid.len() + i];
        let aggregate = mat.xs_elastic[i] + mat.xs_absorption[i];
        assert!(
            (row_sum - aggregate).abs() < 1e-12,
            "row sum {row_sum} != aggregate total {aggregate} at grid {i}"
        );
    }
}

/// #74 Stage 2b: `extract_per_nuclide_inelastic` for a SINGLE-nuclide material
/// must produce per-MT distribution buffers byte-identical to the material-
/// blended ones `extract_material_xs` produces (one nuclide => no blend), and
/// reaction partials whose elastic + absorption + inelastic + fission sums to
/// the material aggregate at every grid point. This is the data-layer guarantee
/// behind the kernel's single-nuclide bit-identity.
#[test]
fn per_nuclide_inelastic_single_nuclide_matches_material_blend() {
    let nuc = make_synthetic_nuclide_with_inelastic_angle();
    let density = 0.7_f64;
    let pairs = [(&nuc, density)];
    let grid = nuc.energy.as_ref().unwrap().get("294").unwrap().clone();

    let mat = extract_material_xs(&pairs, "294").unwrap();
    // Single nuclide: the union grid equals the finest grid, so fine == coarse.
    let pool = extract_per_nuclide_inelastic(&pairs, "294", &grid, &grid).unwrap();
    assert_eq!(pool.n_nuclides, 1);

    // Per-MT distribution buffers byte-identical to the material blend. The pool
    // stores the per-MT XS / yield SPARSE (issue #212): reconstruct the dense
    // `[MT_INELASTIC_COUNT × n_coarse]` layout (0 outside each slot's stored
    // range for XS, 1.0 for yield -- the dense defaults) and compare byte-for-byte
    // to the material's dense per-MT buffers.
    let n_coarse = grid.len();
    let mut xs_dense = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
    let mut yield_dense = vec![1.0_f64; MT_INELASTIC_COUNT * n_coarse];
    let mut off = 0usize;
    for slot in 0..MT_INELASTIC_COUNT {
        let i_start = pool.permt_i_start[slot] as usize;
        let n = pool.permt_n_stored[slot] as usize;
        for k in 0..n {
            xs_dense[slot * n_coarse + i_start + k] = pool.xs_inelastic_per_mt_sparse[off + k];
            yield_dense[slot * n_coarse + i_start + k] = pool.yield_per_mt_sparse[off + k];
        }
        off += n;
    }
    assert_eq!(xs_dense, mat.xs_inelastic_per_mt);
    assert_eq!(pool.q_inelastic_per_mt, mat.q_inelastic_per_mt);
    assert_eq!(yield_dense, mat.yield_per_mt);
    assert_eq!(pool.angle_n_energies, mat.angle_n_energies);
    assert_eq!(pool.angle_mu, mat.angle_mu);
    assert_eq!(pool.angle_cdf, mat.angle_cdf);
    assert_eq!(pool.scatter_in_cm, mat.scatter_in_cm);
    assert_eq!(pool.eout_kind, mat.eout_kind);

    // Reaction partials: per-nuclide elastic == material elastic; the four
    // partials sum to the material total at every grid point.
    for i in 0..grid.len() {
        assert!(
            (pool.sigma_elastic[i] - mat.xs_elastic[i]).abs() < 1e-12,
            "elastic partial mismatch at {i}"
        );
        assert!(
            (pool.sigma_inelastic[i] - mat.xs_inelastic[i]).abs() < 1e-12,
            "inelastic partial mismatch at {i}"
        );
        let sum = pool.sigma_elastic[i]
            + pool.sigma_absorption[i]
            + pool.sigma_inelastic[i]
            + pool.sigma_fission[i];
        let aggregate =
            mat.xs_elastic[i] + mat.xs_absorption[i] + mat.xs_inelastic[i] + mat.xs_fission[i];
        assert!(
            (sum - aggregate).abs() < 1e-12,
            "partial sum {sum} != material total {aggregate} at grid {i}"
        );
    }
}

/// Multi-distribution Evaporation: a neutron product carrying TWO
/// Evaporation laws gated by `applicability` over incident energy (the
/// ENDF File-6 shape of Na23 MT 91) must extract a per-incident-energy
/// restriction energy `u` that switches with the applicability, not a
/// single scalar from the FIRST law. Picking the first law there clamped
/// the GPU outgoing-energy band to the wrong value (E_out < E_in - u),
/// emptying the 8-11.6 MeV flux bins relative to the CPU.
#[test]
fn evap_multi_distribution_u_is_energy_dependent() {
    use yamc_nuclide::particle_type::ParticleType;
    use yamc_nuclide::reaction_product::{
        AngleDistribution, AngleEnergyDistribution, EnergyDistribution, ReactionProduct,
        Tabulated1D,
    };

    // Empty angular table -> isotropic fallback; the u-grid extraction
    // only inspects the energy distribution.
    let iso_angle = || AngleDistribution {
        energy: vec![],
        mu: vec![],
    };

    // Shared theta grid: 1 MeV .. 15 MeV. Two laws share these
    // breakpoints (as Na23 MT 91 does); only `u` differs.
    let theta = || Tabulated1D::Tabulated1D {
        x: vec![1.0e6, 5.0e6, 10.0e6, 15.0e6],
        y: vec![1.5e6, 1.5e6, 2.0e6, 2.3e6],
        breakpoints: vec![4],
        interpolation: vec![2],
    };
    // d0 (high u = 6.1 MeV): applicable BELOW 13 MeV.
    // d1 (low  u = 0.47 MeV): applicable AT/ABOVE 13 MeV.
    let app_low = Tabulated1D::Tabulated1D {
        x: vec![1.0e6, 13.0e6, 13.0e6, 20.0e6],
        y: vec![1.0, 1.0, 0.0, 0.0],
        breakpoints: vec![4],
        interpolation: vec![1],
    };
    let app_high = Tabulated1D::Tabulated1D {
        x: vec![1.0e6, 13.0e6, 13.0e6, 20.0e6],
        y: vec![0.0, 0.0, 1.0, 1.0],
        breakpoints: vec![4],
        interpolation: vec![1],
    };
    let neutron_product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![app_low, app_high],
        distribution: vec![
            AngleEnergyDistribution::UncorrelatedAngleEnergy {
                angle: iso_angle(),
                energy: Some(EnergyDistribution::Evaporation {
                    theta: theta(),
                    u: 6.1e6,
                }),
            },
            AngleEnergyDistribution::UncorrelatedAngleEnergy {
                angle: iso_angle(),
                energy: Some(EnergyDistribution::Evaporation {
                    theta: theta(),
                    u: 0.47e6,
                }),
            },
        ],
        product_yield: None,
    };
    let mt_91 = Reaction {
        cross_section: vec![1.0; 4].into(),
        threshold_idx: 0,
        energy: vec![1.0e6, 5.0e6, 10.0e6, 15.0e6].into(),
        mt_number: 91,
        q_value: -5.84e6,
        products: vec![neutron_product],
        scatter_in_cm: false,
        redundant: false,
    };
    let arc = Arc::new(mt_91);
    let (n_energies, _n_components, energy_grid, _theta, u) =
        distributions::build_per_mt_evap_buffers(
            |mt| if mt == 91 { Some(arc.as_ref()) } else { None },
        );

    // MT 91 is slot 40 (the 41st MT_SLOTS entry).
    let slot = MT_SLOTS.iter().position(|&m| m == 91).unwrap();
    // Tight CSR layout (issue #104): the slot's E_in rows start at the running
    // prefix sum of n_energies (mirrors `evap_ae_offset` built in translate.rs).
    let off: usize = n_energies[..slot].iter().map(|&n| n as usize).sum();
    let n = n_energies[slot] as usize;
    // Shared grid = theta breakpoints UNION applicability breakpoints
    // (deduped): [1, 5, 10, 13, 15, 20] MeV. The 13 MeV window edge must
    // be a grid point even when the theta tables skip it (the Ar38 MT91
    // shape), or the argmax collapse switches `u` at the wrong energy.
    assert_eq!(n, 6, "expected theta+applicability shared grid");
    assert!(
        energy_grid[off..off + n].contains(&13.0e6),
        "applicability window edge must be a grid point"
    );

    // At each grid point, u must be the applicability-dominant law's u:
    // 6.1 MeV below 13 MeV, 0.47 MeV at/above 13 MeV.
    for i in 0..n {
        let e_in = energy_grid[off + i];
        let expected_u = if e_in >= 13.0e6 { 0.47e6 } else { 6.1e6 };
        assert!(
            (u[off + i] - expected_u).abs() < 1.0,
            "grid point {i} (E_in={e_in:.3e}): u={:.3e}, expected {expected_u:.3e} \
             -- multi-distribution applicability selection regressed",
            u[off + i],
        );
    }
}

/// Build a synthetic URR-bearing nuclide with `n_e` energy points and `n_cdf`
/// probability-table bands, keyed on `Z*1000+A`. Reuses
/// [`make_synthetic_nuclide_with`] for the smooth reactions and layers a small
/// absolute-XS URR record on top.
fn make_synthetic_urr_nuclide(z: u32, a: u32, n_e: usize, n_cdf: usize) -> Nuclide {
    use yamc_nuclide::urr::{UrrData, UrrInterpolation, UrrXsSet};
    let mut nuc = make_synthetic_nuclide_with(2.0 * a as f64, 2.0, 1.0);
    nuc.atomic_number = Some(z);
    nuc.mass_number = Some(a);
    nuc.name = Some(format!("Z{z}A{a}"));
    let energy: Vec<f64> = (0..n_e).map(|i| 1.0e3 * (i as f64 + 1.0)).collect();
    let cdf_row: Vec<f64> = (1..=n_cdf).map(|j| j as f64 / n_cdf as f64).collect();
    let xs_row: Vec<UrrXsSet> = (0..n_cdf)
        .map(|j| UrrXsSet {
            total: 1.0 + j as f64,
            elastic: 0.5 + 0.1 * j as f64,
            fission: 0.0,
            n_gamma: 0.2 + 0.05 * j as f64,
            heating: 0.0,
        })
        .collect();
    let urr = UrrData {
        interp: UrrInterpolation::LinLin,
        inelastic_flag: -1,
        absorption_flag: 0,
        multiply_smooth: false,
        energy,
        cdf_values: vec![cdf_row; n_e],
        xs_values: vec![xs_row; n_e],
    };
    // `urr_data` is indexed by temperature index; the synthetic nuclide has a
    // single loaded temperature ("294") at index 0.
    nuc.urr_present = true;
    nuc.urr_data = vec![Some(urr)];
    nuc
}

/// `build_urr_buffers` must emit ONE slab row per nuclide of the material, in
/// input order, with URR nuclides marked `PRESENT` (carrying their ZA / table
/// dimensions and a tight concatenated table) and non-URR nuclides marked
/// absent with a zero-length table (issue #210). This is the per-nuclide
/// keying the GPU URR fix depends on.
#[test]
fn build_urr_buffers_is_per_slab_over_all_nuclides() {
    let n0 = make_synthetic_urr_nuclide(74, 182, 3, 2); // URR: 3 energies, 2 bands
    let n1 = make_synthetic_urr_nuclide(74, 183, 2, 3); // URR: 2 energies, 3 bands
    let n2 = make_synthetic_nuclide_with(56.0, 3.0, 0.5); // non-URR (urr_present=false)
    let nuclides = [(&n0, 0.3_f64), (&n1, 0.7_f64), (&n2, 1.0_f64)];

    let (meta, energy, cdf, xs, atom_density) = distributions::build_urr_buffers(&nuclides, "294");

    // One meta row per nuclide, in input order.
    assert_eq!(meta.len(), 3 * URR_META_COLS);
    let row = |s: usize, c: usize| meta[s * URR_META_COLS + c];

    // PRESENT flags: the two URR nuclides are present, the third absent.
    assert_eq!(row(0, URR_META_PRESENT), 1);
    assert_eq!(row(1, URR_META_PRESENT), 1);
    assert_eq!(row(2, URR_META_PRESENT), 0);

    // ZA (col 7) = Z*1000+A stream key.
    assert_eq!(row(0, URR_META_ZA), 74_182);
    assert_eq!(row(1, URR_META_ZA), 74_183);
    assert_eq!(row(2, URR_META_ZA), 0, "absent slab carries no ZA");

    // Per-slab table dimensions.
    assert_eq!(row(0, URR_META_N_ENERGIES), 3);
    assert_eq!(row(0, URR_META_N_CDF), 2);
    assert_eq!(row(1, URR_META_N_ENERGIES), 2);
    assert_eq!(row(1, URR_META_N_CDF), 3);
    assert_eq!(row(2, URR_META_N_ENERGIES), 0, "absent slab is zero-length");
    assert_eq!(row(2, URR_META_N_CDF), 0);

    // The concatenated tight tables carry only the two URR nuclides' rows:
    // energy = 3 + 2, cdf = 3*2 + 2*3, xs = cdf * URR_XS_COLS.
    assert_eq!(energy.len(), 5, "energy grid = n0.n_e + n1.n_e");
    assert_eq!(cdf.len(), 3 * 2 + 2 * 3);
    assert_eq!(xs.len(), cdf.len() * URR_XS_COLS);

    // The per-slab CSR bases translate.rs builds from these counts must be
    // monotonic non-decreasing: [0, 3] for energy, [0, 6] for cdf.
    let mut eg_base = 0u32;
    let mut cdf_base = 0u32;
    let mut eg_bases = Vec::new();
    let mut cdf_bases = Vec::new();
    for s in 0..3 {
        eg_bases.push(eg_base);
        cdf_bases.push(cdf_base);
        eg_base += row(s, URR_META_N_ENERGIES);
        cdf_base += row(s, URR_META_N_ENERGIES) * row(s, URR_META_N_CDF);
    }
    assert_eq!(eg_bases, vec![0, 3, 5]);
    assert_eq!(cdf_bases, vec![0, 6, 12]);

    // Atom density per slab; the non-URR slab is 0.
    assert_eq!(atom_density, vec![0.3, 0.7, 0.0]);
}

/// Issue #106: a nuclide carrying a neutron-emitting MT the kernel has no slot
/// for must be refused rather than silently over-absorbed. MT 160 ((n,7n)) is
/// in the CPU's scattering set and has no slot, so a fat one is a hard error.
#[test]
fn refuses_unslotted_scatter_mt_above_tolerance() {
    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    // The synthetic nuclide's elastic xs is O(1) b, so 0.5 b is percent-level.
    let big = Reaction {
        cross_section: vec![0.5; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 160,
        q_value: -30_000_000.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(160, Arc::new(big));

    let err = extract_xs_from_nuclide(&nuclide, "294")
        .expect_err("an unslotted neutron-emitting MT this large must be refused");
    match err {
        NuclideXsError::UnslottedScatterMts {
            ref mts,
            max_fraction,
            ..
        } => {
            assert_eq!(mts, &vec![160]);
            assert!(max_fraction > 1e-3, "fraction was {max_fraction}");
        }
        other => panic!("expected UnslottedScatterMts, got {other}"),
    }
    // The message has to name the MT and the size, or it is not actionable.
    let text = err.to_string();
    assert!(text.contains("160"), "message should name the MT: {text}");
    assert!(
        text.contains("CPU"),
        "message should name the way out: {text}"
    );
}

/// Issue #106: the guard must not refuse the negligible real case. ENDF/B-VIII.1
/// La139 carries the unslotted MT 152-200 series at 2.5e-6 of its total, and
/// only above 20 MeV; refusing that would cost a GPU run for nothing.
#[test]
fn tolerates_negligible_unslotted_scatter_mt() {
    let mut nuclide = make_synthetic_nuclide();
    let temp = "294".to_string();
    let energy_grid = nuclide.energy.as_ref().unwrap().get(&temp).unwrap().clone();

    let tiny = Reaction {
        cross_section: vec![1e-9; energy_grid.len()].into(),
        threshold_idx: 0,
        energy: energy_grid.clone(),
        mt_number: 160,
        q_value: -30_000_000.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    };
    nuclide.reactions[0].insert(160, Arc::new(tiny));

    extract_xs_from_nuclide(&nuclide, "294")
        .expect("a 1e-9 b unslotted channel is far below tolerance and must be accepted");
}
