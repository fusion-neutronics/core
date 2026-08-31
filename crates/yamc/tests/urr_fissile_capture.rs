//! Issue #154: a fissile nuclide's URR-adjusted CAPTURE must survive the band.
//!
//! `Nuclide::urr_adjusted_reaction_xs` derived capture as
//! `(xs_absorption - xs_fission).max(0.0)`, but `FastXSGrid::lookup` already
//! returns absorption as a PARTIAL that excludes fission (its four partials sum
//! to the total, and the analog reaction split adds all four). The second
//! subtraction therefore clamped capture to ZERO for every nuclide whose in-band
//! fission exceeds its capture -- which is every fissile one. U235 at 10 keV has
//! absorption 1.06 b against fission 2.91 b.
//!
//! Consequences, all measured: the CPU lost ALL in-band capture, its in-band flux
//! read 17% high against the GPU, only 48.95% of in-band histories were
//! bit-identical to the GPU twin, and the integral flux on a fissile sphere sat
//! 0.2% apart end to end (issue #154, 6-7 sigma).
//!
//! W184 -- the only URR fixture in the matched-stream harnesses -- has no fission
//! at all, so subtracting it was harmless and the bug was invisible there.
//! `localize_u235_urr_band` in `matched_stream_localize` is the transport-level
//! guard; this is the direct one.

use std::collections::HashMap;

use yamc_materials::Material;

fn cache(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn load(nuclide: &str) -> Option<Material> {
    if !std::path::Path::new(&cache(nuclide)).exists() {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(18.95),
    )
    .ok()?;
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), cache(nuclide))]),
        None,
    )
    .ok()?;
    Some(m)
}

/// The grid's four partials sum to its total, i.e. `absorption` is capture only
/// and does NOT include fission. This is the invariant the URR code violated by
/// subtracting fission from it a second time.
#[test]
fn absorption_partial_excludes_fission() {
    let Some(m) = load("U235") else {
        eprintln!("skipping -- endf-b8.1-U235.arrow cache absent");
        return;
    };
    let nuc = m.nuclide_data.get("U235").expect("U235 loaded");
    let temp_idx = nuc.get_temp_idx("294").expect("294 K loaded");
    let grid = &nuc.fast_xs[temp_idx];
    for e in [2.5e3_f64, 5.0e3, 1.0e4, 2.0e4] {
        let (total, absorption, scattering, fission) = grid.lookup(e);
        let sum = scattering + absorption + fission;
        assert!(
            (sum - total).abs() <= 1e-9 * total,
            "E={e:.3e}: partials {sum:.6e} do not sum to total {total:.6e} \
             (scatter {scattering:.6e}, absorption {absorption:.6e}, fission {fission:.6e})"
        );
        // The precondition that made the old `(absorption - fission).max(0.0)`
        // destroy capture rather than merely perturb it.
        assert!(
            fission > absorption,
            "E={e:.3e}: expected U235's in-band fission {fission:.6e} to exceed its \
             capture {absorption:.6e}, which is what the clamp depended on"
        );
    }
}

/// In-band, the URR-adjusted capture must be positive. This is the assertion that
/// fails on the pre-fix code: it returned exactly 0.0 at every in-band energy.
#[test]
fn u235_urr_adjusted_capture_is_positive_in_band() {
    let Some(m) = load("U235") else {
        eprintln!("skipping -- endf-b8.1-U235.arrow cache absent");
        return;
    };
    let nuc = m.nuclide_data.get("U235").expect("U235 loaded");
    let mut rng = yamc::util::fast_rng::FastRng::new(12345);
    // U235's band is 2250 .. 24999 eV.
    for e in [2.3e3_f64, 5.0e3, 1.0e4, 2.0e4, 2.4e4] {
        // Sweep the band so the assertion cannot pass on one lucky draw.
        for k in 0..16 {
            let urr_random = (k as f64 + 0.5) / 16.0;
            let p = nuc
                .reaction_partials(e, "294", Some(urr_random), &mut rng)
                .expect("fast_xs grid present");
            assert!(
                p.sigma_a > 0.0,
                "E={e:.3e}, urr_random={urr_random:.4}: URR-adjusted capture is \
                 {:.6e}, so the band destroyed it (#154)",
                p.sigma_a
            );
            assert!(
                p.sigma_f > 0.0,
                "E={e:.3e}: expected nonzero in-band fission, got {:.6e}",
                p.sigma_f
            );
        }
    }
}

/// W184 has no fission, so its in-band capture was never at risk. Kept as the
/// contrast that explains why the existing URR fixture reads 100% bit-identical
/// against the twin and could not have caught this.
#[test]
fn w184_has_no_fission_to_subtract() {
    let Some(m) = load("W184") else {
        eprintln!("skipping -- endf-b8.1-W184.arrow cache absent");
        return;
    };
    let nuc = m.nuclide_data.get("W184").expect("W184 loaded");
    let temp_idx = nuc.get_temp_idx("294").expect("294 K loaded");
    let grid = &nuc.fast_xs[temp_idx];
    for e in [1.5e4_f64, 5.0e4, 9.0e4] {
        let (_total, absorption, _scattering, fission) = grid.lookup(e);
        assert_eq!(fission, 0.0, "E={e:.3e}: W184 should have no fission");
        assert!(absorption > 0.0, "E={e:.3e}: expected W184 capture");
    }
}
