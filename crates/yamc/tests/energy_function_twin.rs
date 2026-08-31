//! Gate test for the packed energy-function descriptor (issue #271).
//!
//! `energy_function=` / `dose_coefficients=` interpolate a natural cubic spline
//! in LINEAR energy. The GPU cannot solve a spline, so the dispatch ships the
//! coefficients `EnergyFunctionFilter::new` already solved on the host and the
//! kernel only evaluates the polynomial. That is what makes the two backends
//! agree bit for bit rather than merely closely, and this test is what pins it:
//! `common::tallies::energy_function_weight` reading the PACKED descriptor must
//! return exactly what `EnergyFunctionFilter::get_weight` returns, including
//! the `None` that means "drop the whole scoring event".
//!
//! Following `cyl_mesh_dda_twin.rs`, this runs on the CPU only -- the twin is
//! pinned bit-equal to the cubecl kernel elsewhere -- so it gates the packing
//! in CI, with no GPU and no nuclear data. It needs the `gpu` feature only
//! because the twin lives in the optional crate.
//!
//! Run it:
//!   cargo test -p yamc --features gpu --test energy_function_twin
#![cfg(feature = "gpu")]

use yamc_gpu::common::tallies::{efunc_table_len, energy_function_weight};
use yamc_tallies::EnergyFunctionFilter;

/// Pack one filter the way `build_tallies_pack` does:
/// `[n_points, energy[n], coeffs[4*(n-1)]]`. Duplicated here on purpose --
/// a test that called the packer could not catch the packer drifting.
fn pack(filter: &EnergyFunctionFilter) -> Vec<f64> {
    let energies = filter.energy();
    let mut out = Vec::with_capacity(efunc_table_len(energies.len()));
    out.push(energies.len() as f64);
    out.extend_from_slice(energies);
    for c in filter.spline_coeffs() {
        out.extend_from_slice(c);
    }
    out
}

/// Every energy worth probing for a grid: below the floor, exactly on each
/// knot, each interval midpoint, and above the ceiling. Knots and boundaries
/// are where the CPU's `binary_search_by` tie-breaking and the twin's bracket
/// search could disagree.
fn probe_energies(energy: &[f64]) -> Vec<f64> {
    let mut out = vec![energy[0] * 0.5, energy[0] - 1e-9];
    for i in 0..energy.len() {
        out.push(energy[i]);
        if i + 1 < energy.len() {
            out.push(0.5 * (energy[i] + energy[i + 1]));
            out.push(energy[i] + 0.25 * (energy[i + 1] - energy[i]));
            // Just inside each knot from either side.
            out.push(energy[i] * (1.0 + 1e-12));
            out.push(energy[i + 1] * (1.0 - 1e-12));
        }
    }
    let last = *energy.last().unwrap();
    out.push(last * (1.0 + 1e-12));
    out.push(last * 2.0);
    out
}

fn assert_twin_matches(filter: &EnergyFunctionFilter, label: &str) {
    let params = pack(filter);
    assert_eq!(
        params.len(),
        efunc_table_len(filter.energy().len()),
        "{label}: packed length disagrees with efunc_table_len"
    );
    let mut n_in_range = 0usize;
    let mut n_out_of_range = 0usize;
    for e in probe_energies(filter.energy()) {
        let want = filter.get_weight(e);
        let got = energy_function_weight(&params, 0, e);
        match want {
            Some(_) => n_in_range += 1,
            None => n_out_of_range += 1,
        }
        // Exact equality, not a tolerance: same coefficients, same Horner
        // form, same bracket. Anything less would let a stray `exp(ln(E))`
        // round-trip or a log-space evaluation slip in unnoticed.
        assert_eq!(got, want, "{label}: weight at energy {e:e} differs");
    }
    assert!(
        n_in_range >= 8 && n_out_of_range >= 3,
        "{label}: probe set is not exercising both sides of the gate \
         ({n_in_range} in range, {n_out_of_range} out)"
    );
}

#[test]
fn packed_table_matches_the_filter_on_a_synthetic_curve() {
    // Non-monotonic y, so the spline genuinely curves and a wrong interval
    // index gives a visibly wrong answer rather than something plausible.
    let filter = EnergyFunctionFilter::new(
        vec![1.0, 10.0, 100.0, 1000.0, 1e4, 1e5],
        vec![0.5, 3.0, 1.25, 8.0, 2.0, 6.5],
    );
    assert_twin_matches(&filter, "synthetic");
}

#[test]
fn packed_table_matches_the_filter_on_a_uniform_grid() {
    // Uniform spacing exercises a different branch of the Thomas solve than
    // the decade-spaced grids above.
    let energy: Vec<f64> = (0..12).map(|i| 1.0 + i as f64).collect();
    let y: Vec<f64> = (0..12).map(|i| ((i as f64) * 0.7).sin() + 2.0).collect();
    assert_twin_matches(&EnergyFunctionFilter::new(energy, y), "uniform");
}

#[test]
fn packed_table_matches_the_filter_on_the_icrp116_dose_curve() {
    // The real payload behind `dose_coefficients=('neutron', 'AP')`: 68 points
    // spanning 1e-3 to 1e10 eV. A 13-decade grid is the stiffest case for the
    // bracket search, and it is what users will actually run.
    let (energy, coeffs) = yamc_nuclide::data::effective_dose::dose_coefficients(
        yamc_nuclide::data::effective_dose::DoseParticle::Neutron,
        yamc_nuclide::data::effective_dose::DoseGeometry::AP,
        yamc_nuclide::data::effective_dose::DoseDataSource::ICRP116,
    );
    assert!(energy.len() >= 60, "unexpected ICRP-116 table size");
    assert_twin_matches(&EnergyFunctionFilter::new(energy, coeffs), "icrp116-ap");
}

#[test]
fn constant_curve_evaluates_to_exactly_the_constant() {
    // A natural cubic spline through constant data is exactly constant (every
    // b, c, d solves to zero), which is what makes the "y == c scales the flux
    // tally by exactly c" invariance test in the transport suites exact rather
    // than approximate. Pin the property here so a failure there points at the
    // transport wiring rather than at the spline.
    let filter = EnergyFunctionFilter::new(vec![1.0, 10.0, 100.0, 1000.0], vec![3.5; 4]);
    let params = pack(&filter);
    for e in probe_energies(filter.energy()) {
        if let Some(w) = filter.get_weight(e) {
            assert_eq!(w, 3.5, "CPU filter not exactly constant at {e:e}");
            assert_eq!(
                energy_function_weight(&params, 0, e),
                Some(3.5),
                "packed table not exactly constant at {e:e}"
            );
        }
    }
}

#[test]
fn out_of_range_is_none_not_zero() {
    // The distinction matters: `None` means the CPU dropped the whole scoring
    // event, so the kernel must skip the atomic add rather than add zero. They
    // differ once several tallies or a mesh fan-out are involved.
    let filter =
        EnergyFunctionFilter::new(vec![10.0, 100.0, 1000.0, 1e4], vec![1.0, 2.0, 3.0, 4.0]);
    let params = pack(&filter);
    for e in [1.0, 9.999, 1.00001e4, 1e6] {
        assert_eq!(filter.get_weight(e), None, "CPU should gate {e:e}");
        assert_eq!(
            energy_function_weight(&params, 0, e),
            None,
            "packed table should gate {e:e}"
        );
    }
    // And immediately inside both boundaries it is Some, so the gate is not
    // simply always-on.
    assert!(energy_function_weight(&params, 0, 10.0).is_some());
    assert!(energy_function_weight(&params, 0, 1e4).is_some());
}

#[test]
fn tables_are_read_at_their_own_offset() {
    // Two tallies share one buffer, so the second must be read at its offset
    // and not from the front. An off-by-one in the CSR arithmetic would
    // silently give every tally the first tally's dose curve.
    let a = EnergyFunctionFilter::new(vec![1.0, 10.0, 100.0, 1000.0], vec![1.0, 2.0, 3.0, 4.0]);
    let b = EnergyFunctionFilter::new(vec![2.0, 20.0, 200.0, 2000.0], vec![9.0, 7.0, 5.0, 3.0]);
    let mut params = pack(&a);
    let off_b = params.len();
    params.extend(pack(&b));

    for e in probe_energies(a.energy()) {
        assert_eq!(energy_function_weight(&params, 0, e), a.get_weight(e));
    }
    for e in probe_energies(b.energy()) {
        assert_eq!(energy_function_weight(&params, off_b, e), b.get_weight(e));
    }
    // The two curves must actually differ somewhere both cover, otherwise the
    // offset check above would pass even if both reads hit table A.
    let shared = 500.0;
    assert_ne!(a.get_weight(shared), b.get_weight(shared));
}
