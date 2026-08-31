//! Host-side verification of the photon "dominant element per reaction"
//! selection used by the single-element-per-material form-factor /
//! relaxation packs.
//!
//! Regression for the multi-element correctness gap (task #72): a trace
//! high-Z element in a low-Z bulk (W in polyethylene, or 50/50 WC) must
//! drive the coherent / incoherent / photoelectric / pair tables, because
//! those reactions scale steeply with Z and so the high-Z element
//! dominates the *macroscopic* cross section of that channel even when it
//! is a minority by atom count. The previous "most atoms" pick used H/C
//! and silently dropped the W K-fluorescence source.
//!
//! These tests load real W / C / H photon element data and assert the
//! selection directly -- no GPU adapter required, so they run everywhere.
//! The end-to-end CPU-vs-GPU score recovery (which needs a GPU) lives in
//! the `yamc` crate's `gpu_cpu_comparison_matrix` harness.

use std::sync::Arc;

use yamc_element::photon::{get_or_load_element, PhotonInteraction};
use yamc_gpu::photon::xs::atomic_relaxation_xs::extract_atomic_relaxation_for_gpu;
use yamc_gpu::photon::xs::incoherent_form_factor_xs::extract_incoherent_form_factor_for_gpu;
use yamc_gpu::photon::xs::pair_production_xs::extract_pair_production_for_gpu;
use yamc_gpu::photon::xs::photon_xs::{extract_photon_material_xs, MAX_RAYLEIGH_FF};
use yamc_gpu::photon::xs::{dominant_element_for_reaction, ReactionChannel};

const PHOTON_DIR: &str =
    "/home/jon/yamc-org/cross_section_data_fendl_3.2d_arrow/fendl-3.2d-arrow/photon";

/// Try to load an element's photon data; `None` (test self-skips) when
/// the local cross-section directory is absent.
fn try_load(symbol: &str) -> Option<Arc<PhotonInteraction>> {
    let path = format!("{PHOTON_DIR}/{symbol}.arrow");
    if !std::path::Path::new(&path).exists() {
        return None;
    }
    get_or_load_element(symbol, &path).ok()
}

/// `(name, element, atom_density)` triple list for a material from
/// `(symbol, atom_density)` pairs. Returns `None` if any element is
/// missing locally.
fn material(parts: &[(&str, f64)]) -> Option<Vec<(String, Arc<PhotonInteraction>, f64)>> {
    let mut out = Vec::new();
    for (sym, density) in parts {
        let el = try_load(sym)?;
        out.push((sym.to_string(), el, *density));
    }
    Some(out)
}

/// Polyethylene (CH2)n with a small W loading. Atom densities (arbitrary
/// scale -- only ratios matter for selection): per CH2, 2 H + 1 C, plus W
/// at ~10 at% of the total.
fn poly_w() -> Option<Vec<(String, Arc<PhotonInteraction>, f64)>> {
    // H : C : W = 60 : 30 : 10 (so W is 10 at%, the bulk is light).
    material(&[("H", 0.060), ("C", 0.030), ("W", 0.010)])
}

/// Tungsten carbide WC: 50 at% C, 50 at% W. Exercises the tie-break: the
/// old strict-`>` over an alphabetically ordered element list picked C
/// (alphabetically first) over W on an even atom split.
fn wc() -> Option<Vec<(String, Arc<PhotonInteraction>, f64)>> {
    material(&[("C", 0.050), ("W", 0.050)])
}

fn z_of(mat: &[(String, Arc<PhotonInteraction>, f64)], idx: usize) -> u32 {
    mat[idx].1.atomic_number
}

#[test]
fn poly_w_selects_w_for_all_z_steep_channels() {
    let Some(mat) = poly_w() else {
        eprintln!("skip: local photon data absent");
        return;
    };
    // Even at 10 at%, W dominates coherent / photoelectric / pair because
    // those cross sections scale steeply with Z.
    for ch in [
        ReactionChannel::Coherent,
        ReactionChannel::Photoelectric,
        ReactionChannel::Pair,
    ] {
        let idx = dominant_element_for_reaction(&mat, ch)
            .unwrap_or_else(|| panic!("no dominant element for {ch:?}"));
        assert_eq!(
            z_of(&mat, idx),
            74,
            "{ch:?} should pick W (Z=74) in poly+W, picked Z={}",
            z_of(&mat, idx)
        );
    }
    // Incoherent (Compton) scales ~linearly with Z, so per-atom it favours
    // W, but the bulk H+C atom count can still win. Whatever it picks, the
    // selection must be deterministic and not panic; document that the
    // incoherent channel is the weakly-Z one.
    let inc = dominant_element_for_reaction(&mat, ReactionChannel::Incoherent).unwrap();
    let _ = z_of(&mat, inc);
}

#[test]
fn wc_tie_break_prefers_tungsten_over_carbon() {
    let Some(mat) = wc() else {
        eprintln!("skip: local photon data absent");
        return;
    };
    // On the 50/50 split the steep-Z channels must pick W (Z=74), never
    // the alphabetically-first Carbon (Z=6).
    for ch in [
        ReactionChannel::Coherent,
        ReactionChannel::Photoelectric,
        ReactionChannel::Pair,
        ReactionChannel::Incoherent,
    ] {
        let idx = dominant_element_for_reaction(&mat, ch).unwrap();
        assert_eq!(
            z_of(&mat, idx),
            74,
            "{ch:?} should pick W (Z=74) in 50/50 WC, picked Z={}",
            z_of(&mat, idx)
        );
    }
}

#[test]
fn single_element_pick_is_the_only_element() {
    // Single-element materials must be unchanged: the only element wins
    // every channel trivially (this is what keeps Fe/W/Pb spheres exact).
    for sym in ["W", "C", "Fe"] {
        let Some(el) = try_load(sym) else {
            eprintln!("skip: {sym} photon data absent");
            continue;
        };
        let z = el.atomic_number;
        let mat = vec![(sym.to_string(), el, 0.05)];
        for ch in [
            ReactionChannel::Coherent,
            ReactionChannel::Incoherent,
            ReactionChannel::Photoelectric,
            ReactionChannel::Pair,
        ] {
            let idx = dominant_element_for_reaction(&mat, ch).unwrap();
            assert_eq!(idx, 0);
            assert_eq!(z_of(&mat, idx), z);
        }
    }
}

#[test]
fn extractors_pack_w_relaxation_and_form_factors_for_poly_w() {
    let Some(mat) = poly_w() else {
        eprintln!("skip: local photon data absent");
        return;
    };
    let materials = vec![mat.clone()];

    // #85 packs ONE Rayleigh slab PER ELEMENT (element-major within a material,
    // concatenated material-major), so the per-collision element selection can
    // read whichever element the photon struck. This material is the only one,
    // so its element slabs start at slab 0 in the order (H, C, W); element `e`
    // lives at slab `e`. Assert that W's OWN slab carries W's coherent
    // form-factor CDF (it must track W, not H), exercising the per-element
    // packing #85 introduced.
    let xs = extract_photon_material_xs(&materials);
    let log_grid = xs.log_energy_grid.clone();
    let w_idx = mat.iter().position(|(s, _, _)| s == "W").unwrap();
    let h_idx = mat.iter().position(|(s, _, _)| s == "H").unwrap();
    // Material 0's element-slab base is 0, so each element's slab index equals
    // its position within the material.
    let w_slab = w_idx;
    let n_pts = xs.rayleigh_n_points[w_slab] as usize;
    assert!(n_pts > 1, "W's Rayleigh slab should be populated");
    let w_off = w_slab * MAX_RAYLEIGH_FF;
    let mut err_w = 0.0_f64;
    let mut err_h = 0.0_f64;
    for j in 0..n_pts {
        let x2 = xs.rayleigh_x2[w_off + j];
        let packed = xs.rayleigh_cdf[w_off + j];
        err_w += (packed - coherent_cdf_at(&mat[w_idx].1, x2)).abs();
        err_h += (packed - coherent_cdf_at(&mat[h_idx].1, x2)).abs();
    }
    assert!(
        err_w < err_h,
        "W's packed coherent FF CDF slab should match W's, not H's (err_w={err_w:.3e}, err_h={err_h:.3e})"
    );

    // Atomic relaxation / photoelectric subshells: #85 packs one slab per
    // element, so W's OWN slab must carry W's subshells. W has ~22 subshells
    // (capped at the MAX_AR_SHELLS=16 GPU limit); H has 1, C has 4. Comparing
    // W's slab in the poly+W pack against a W-only pack (slab 0 = W) and an
    // H-only pack (slab 0 = H) is the direct signature that W's photoelectric
    // subshell + relaxation cascade (and hence W's K-fluorescence emission,
    // when the data carries relaxation transitions) is packed for selection.
    use yamc_gpu::photon::xs::atomic_relaxation_xs::MAX_AR_SHELLS;
    let n_grid = log_grid.len();
    let w_idx_el = mat[w_idx].1.clone();
    let w_only = vec![vec![("W".to_string(), w_idx_el, mat[w_idx].2)]];
    let h_only = vec![vec![("H".to_string(), mat[h_idx].1.clone(), mat[h_idx].2)]];
    let ar = extract_atomic_relaxation_for_gpu(&materials, &log_grid);
    let ar_w = extract_atomic_relaxation_for_gpu(&w_only, &log_grid);
    let ar_h = extract_atomic_relaxation_for_gpu(&h_only, &log_grid);
    assert_eq!(
        ar.n_shells[w_slab], ar_w.n_shells[0],
        "W's slab relaxation shell count should equal W-only's ({} vs W={}, H={})",
        ar.n_shells[w_slab], ar_w.n_shells[0], ar_h.n_shells[0]
    );
    assert!(
        ar.n_shells[w_slab] > ar_h.n_shells[0],
        "W's slab must pack more subshells than H-only ({} vs {})",
        ar.n_shells[w_slab],
        ar_h.n_shells[0]
    );
    // The K-shell photoelectric XS at a transport energy (10 keV) on W's slab
    // must be W's, not H's (which would be near-zero / below threshold).
    let i10 = log_grid
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.exp() - 1.0e4)
                .abs()
                .partial_cmp(&(b.exp() - 1.0e4).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    // K-shell is subshell index 0: PE-XS layout is
    // (m * n_grid + i_grid) * MAX_AR_SHELLS + s. W is slab `w_slab` in the
    // poly+W pack and slab 0 in the W-only pack; s = 0.
    let k0_w = (w_slab * n_grid + i10) * MAX_AR_SHELLS;
    // W-only / H-only packs have W / H at slab 0, so their K-shell slot is
    // simply `i10 * MAX_AR_SHELLS`.
    let k0_slab0 = i10 * MAX_AR_SHELLS;
    assert_eq!(
        ar.pe_subshell_xs_log[k0_w], ar_w.pe_subshell_xs_log[k0_slab0],
        "W-slab K-shell PE XS should equal W-only's, not H-only's ({})",
        ar_h.pe_subshell_xs_log[k0_slab0]
    );

    // Incoherent form factor + pair production must also be populated on W's
    // slab.
    let iff = extract_incoherent_form_factor_for_gpu(&materials);
    assert_eq!(iff.has_data[w_slab], 1);
    let pp = extract_pair_production_for_gpu(&materials);
    assert_eq!(
        pp.has_data[w_slab], 1,
        "pair-production constants should pack for W's slab"
    );
    // W's pair slab should carry W's Z (74) screening constants: a = Z/alpha^-1.
    let expected_a = 74.0 / 137.035_999_084;
    assert!(
        (pp.a[w_slab] - expected_a).abs() < 1e-9,
        "W-slab pair Born parameter should be W's (Z=74): got a={}",
        pp.a[w_slab]
    );
}

/// Integrated coherent form-factor CDF `F(x^2, Z)` of an element at a
/// given `x^2`.
fn coherent_cdf_at(el: &PhotonInteraction, x2: f64) -> f64 {
    el.coherent_int_form_factor.evaluate(x2)
}
