//! Flat-buffer physics samplers.
//!
//! Bit-identical CPU mirrors of the cubecl `#[cube]` neutron kernel's
//! secondary-energy / secondary-angle samplers. They take pre-extracted
//! per-slot slices and an inline 64-bit PCG state (`&mut u64`) so the
//! caller (typically `yamc-gpu`'s `run_multi_cell_transport_cpu` and
//! `_rayon` companions) can validate the kernel output and provide a
//! pure-CPU fallback when no f64-capable GPU is available.
//!
//! The samplers are deliberately layout-independent: yamc-gpu owns the
//! GPU buffer layout (slot ordering, concatenation, CSR base offsets), the
//! caller does the flat-index → per-slot slice extraction, and these
//! functions only see one slot's data. That keeps physics here and
//! buffer-packing concerns in yamc-gpu. Two modules are deliberately wider
//! (issue #111, so the CPU production transport and the GPU host-side twin
//! share one path end to end):
//!   * [`inelastic_dispatch`] owns the per-(slab, MT-slot) extraction and
//!     `eout_kind` dispatch, plus the `MT_INELASTIC_COUNT` / `EOUT_KIND_*`
//!     constants it needs.
//!   * [`eout_extract`] owns the other half, turning ONE `Reaction` into the
//!     per-slot arrays these samplers read (plus the `MAX_EVAP_COMPONENTS` /
//!     `MAX_CORR_COMPONENTS` caps its multi-component paths enforce).
//!   * [`inelastic_flat`] bundles ONE reaction's extracted slot into the
//!     single-slot (`slab = 0`, `slot = 0`, zero-based CSR) form
//!     [`inelastic_dispatch`] takes, and caches those bundles per
//!     (nuclide, MT) for the CPU transport.
//!
//! Each sampler advances `state` exactly the same number of times as the
//! corresponding cubecl `#[cube]` branch -- replacing a call here with a
//! different RNG schedule breaks kernel/CPU bit-equivalence and the
//! `gpu_*` test suite will fail.

pub mod cm_to_lab;
pub mod correlated_angle_energy;
pub mod elastic_mu_cm;
pub mod eout_extract;
pub mod evaporation;
pub mod fission_eout_continuous;
pub mod free_gas_elastic;
pub(crate) mod grid;
pub mod inelastic_dispatch;
pub mod inelastic_flat;
pub(crate) mod interp;
pub mod kalbach_mann;
pub mod maxwell;
pub mod nbody_phase_space;
pub mod tabulated_continuous_eout;
pub mod tabulated_equiprobable;
pub mod watt;

use yamc_nuclide::reaction_product::FissionChiFlat;

/// Sample an outgoing fission / secondary chi energy from a pre-flattened
/// [`FissionChiFlat`], dispatching to the matching flat sampler driven by the
/// shared 64-bit PCG `state` (issue #111). Used by both the CPU transport
/// (`yamc::transport`'s fission path) and the parity tests, so the two share
/// one fission-chi sampling implementation. Returns `None` when the chi has no
/// usable data ([`FissionChiFlat::None`]) or the underlying sampler exhausts
/// its rejection cap; the caller then falls back (e.g. keeps the incident
/// energy).
pub fn sample_fission_chi_flat(chi: &FissionChiFlat, e_in: f64, state: &mut u64) -> Option<f64> {
    match chi {
        FissionChiFlat::None => None,
        FissionChiFlat::Watt {
            energy_grid,
            a,
            b,
            u,
        } => watt::sample_watt_inelastic(e_in, energy_grid, a, b, *u, state),
        FissionChiFlat::Maxwell {
            energy_grid,
            theta,
            u,
        } => maxwell::sample_maxwell(e_in, energy_grid, theta, *u, state),
        FissionChiFlat::Evaporation {
            energy_grid,
            theta,
            u,
        } => evaporation::sample_evaporation(e_in, energy_grid, theta, *u, state),
        FissionChiFlat::Continuous {
            energy_grid,
            n_x,
            interp,
            n_discrete,
            x,
            p,
            c,
            max_x,
            histogram_outer,
        } => {
            // Bridge: the fission-chi flat tables are still fixed-stride
            // `max_x` (the fission_eout family is migrated separately), so
            // row `i` starts at `i * max_x`. The sampler now takes a tight
            // CSR `x_offset` (issue #104); synthesise the strided offsets.
            let x_offset: Vec<u32> = (0..n_x.len()).map(|i| (i * *max_x) as u32).collect();
            tabulated_continuous_eout::sample_tabulated_continuous_eout(
                e_in,
                energy_grid,
                n_x,
                interp,
                n_discrete,
                *histogram_outer,
                x,
                p,
                c,
                &x_offset,
                state,
            )
        }
    }
}

#[cfg(test)]
mod fission_chi_flat_tests {
    //! Parity for the flat fission-chi path (issue #111 fission sub-step):
    //! `EnergyDistribution::to_fission_chi_flat` then `sample_fission_chi_flat`
    //! must reproduce the production `EnergyDistribution::sample` outgoing-energy
    //! distribution. Statistical comparison (different RNGs); validates the
    //! extraction before it is wired into the transport fission path.
    use super::sample_fission_chi_flat;
    use rand::{rngs::StdRng, SeedableRng};
    use yamc_nuclide::reaction_product::{
        EnergyDistribution, Tabulated1D, TabulatedInterp, TabulatedProbability,
    };

    const N: usize = 400_000;

    fn tab1d(val: f64) -> Tabulated1D {
        Tabulated1D::Tabulated1D {
            x: vec![1.0, 2.0e7],
            y: vec![val, val],
            breakpoints: vec![2],
            interpolation: vec![2], // lin-lin
        }
    }

    fn mean_std(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
        (m, s)
    }

    /// Drive production `dist.sample` and the flat chi path from the same
    /// distribution; assert matching mean (<1.5%) and std (<3%).
    fn check(dist: &EnergyDistribution, e_in: f64, seed: u64, label: &str) {
        let mut rng = StdRng::seed_from_u64(seed);
        let prod: Vec<f64> = (0..N).map(|_| dist.sample(e_in, &mut rng)).collect();

        let chi = dist.to_fission_chi_flat();
        let mut state: u64 = yamc_rng::expand_seed(seed as u32);
        let mut flat = Vec::with_capacity(N);
        let mut exhausted = 0usize;
        while flat.len() < N {
            match sample_fission_chi_flat(&chi, e_in, &mut state) {
                Some(e) => flat.push(e),
                None => {
                    exhausted += 1;
                    if exhausted > N {
                        panic!("{label}: flat chi exhausted too often");
                    }
                }
            }
        }
        let (mp, sp) = mean_std(&prod);
        let (mf, sf) = mean_std(&flat);
        let mr = (mp - mf).abs() / mp.abs().max(1e-30);
        let sr = (sp - sf).abs() / sp.abs().max(1e-30);
        eprintln!("{label}: prod ({mp:.4e},{sp:.4e}) flat ({mf:.4e},{sf:.4e}) mean_rel={mr:.4} std_rel={sr:.4}");
        assert!(mr < 0.015, "{label}: mean differs by {mr:.4}");
        assert!(sr < 0.03, "{label}: std differs by {sr:.4}");
    }

    #[test]
    fn flat_watt_chi_matches_production() {
        // U235-like Watt: a ~ 1 MeV, b ~ 2/MeV, u = 0. e_in = 2 MeV.
        let dist = EnergyDistribution::Watt {
            a: tab1d(1.0e6),
            b: tab1d(2.0e-6),
            u: 0.0,
        };
        check(&dist, 2.0e6, 0x5EED_F155, "Watt");
    }

    #[test]
    fn flat_maxwell_chi_matches_production() {
        // theta ~ 1.3 MeV, e_in = 14 MeV so cap_e >> theta (minimal rejection).
        let dist = EnergyDistribution::Maxwell {
            theta: tab1d(1.3e6),
            u: 0.0,
        };
        check(&dist, 14.0e6, 0x5EED_0A11, "Maxwell");
    }

    #[test]
    fn flat_continuous_chi_matches_production() {
        // Tabulated continuous chi with `histogram_interp = true` (suppresses
        // the stochastic bracket pick + stretch), two identical incident-energy
        // rows; sampling at the low knot reduces both paths to the within-
        // bracket CDF inversion (validated against the CPU in #114).
        let x = vec![1.0e5, 1.0e6, 4.0e6, 1.0e7];
        let c = vec![0.0, 0.4, 0.8, 1.0];
        let mut p = vec![0.0; 4];
        for k in 0..3 {
            p[k] = (c[k + 1] - c[k]) / (x[k + 1] - x[k]);
        }
        let row = TabulatedProbability::Tabulated {
            x: x.clone(),
            p: p.clone(),
            c: c.clone(),
            interp: TabulatedInterp::Histogram,
            n_discrete: 0,
        };
        let dist = EnergyDistribution::ContinuousTabular {
            energy: vec![1.0e3, 1.0e7],
            energy_out: vec![row.clone(), row],
            histogram_interp: true,
        };
        check(&dist, 1.0e3, 0x5EED_C047, "ContinuousTabular");
    }
}
