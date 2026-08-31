//! Regression for the correlated angle-energy mu-sub-table offset bug (found
//! while investigating #104).
//!
//! `parse_correlated` used to set the LAST mu sub-table of each incident-energy
//! corrtable to span to the end of the whole concatenated `corr_mu_data` array
//! instead of stopping at the next corrtable's offset. That made one angular
//! distribution per corrtable swallow tens of thousands of points (e.g. U238
//! MT5: a 181,298-point, non-monotonic table), corrupting the sampled emission
//! angle of correlated reactions on both the CPU and GPU paths.
//!
//! Uses cached ENDF/B-VIII.1 U238 (correlated MT5); self-skips when absent, so
//! it is inert without the data.
#![cfg(feature = "arrow")]

use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::reaction_product::AngleEnergyDistribution;

#[test]
fn correlated_mu_subtables_bounded_and_monotonic() {
    let p = std::path::PathBuf::from(yamc_test_cache::nuclide_path("U238"));
    if !p.exists() {
        eprintln!("skip: cached U238 not present");
        return;
    }
    // Presence is not enough. Since #389 a cache directory is routinely
    // populated at activation scope, holding cross sections and none of the
    // distributions this reads, and the directory exists either way.
    let Ok(nd) = read_nuclide_from_arrow(&p, &yamc_nuclide::LoadScope::full()) else {
        eprintln!("skip: cached U238 is narrower than full scope");
        return;
    };
    let mut checked = 0usize;
    let mut max_pts = 0usize;
    for temp_map in &nd.reactions {
        for rxn in temp_map.values() {
            for prod in &rxn.products {
                for ae in &prod.distribution {
                    if let AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } = ae {
                        for ct in &correlated.distributions {
                            for ang in &ct.angle {
                                checked += 1;
                                max_pts = max_pts.max(ang.x.len());
                                // A real angular distribution has O(2..300) mu
                                // points. Pre-fix the last-per-corrtable table
                                // held ~181k concatenated points.
                                assert!(
                                    ang.x.len() < 10_000,
                                    "correlated mu sub-table has {} points -- the offset bug is back",
                                    ang.x.len()
                                );
                                // A CDF column must be non-decreasing; a
                                // concatenation of many distributions is not.
                                assert!(
                                    ang.c.windows(2).all(|w| w[1] >= w[0] - 1e-9),
                                    "correlated mu CDF is not monotonic -- the offset bug is back"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no correlated angular sub-tables found in U238 (data layout changed?)"
    );
    eprintln!(
        "checked {checked} correlated mu sub-tables (max {max_pts} pts): all bounded + monotonic"
    );
}
