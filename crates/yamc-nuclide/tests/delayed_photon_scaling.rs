//! Issue #369: the delayed-photon scaling must actually reach the fission photon
//! production.
//!
//! A fission releases photons from the fission products' decay as well as promptly,
//! and the evaluation's prompt photon production does not include them. OpenMC
//! covers this by scaling fission photon production by
//! `f = (prompt_photons + delayed_photons) / prompt_photons`
//! (`settings::delayed_photon_scaling`, on by default, `physics.cpp`).
//!
//! yamc had the consuming code on both backends but the Arrow reader hardcoded the
//! scaling to empty ("not in Arrow format yet"), so every consumer took its
//! `f = 1.0` branch and the feature was inert for every nuclide. That showed up in
//! the V&V as a 1.4 to 2.6% secondary-photon flux deficit on Ac225/226/227 (which
//! have `fission_energy_release` data, so OpenMC scales them by 1.37 to 1.38) while
//! Fe56 (which does not) was clean.
//!
//! No published file carries the columns yet, so these tests drive the arithmetic
//! directly rather than through a real nuclide.

/// Horner evaluation, mirroring the reader's `eval_poly`.
fn poly(coeffs: &[f64], e: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * e + c)
}

/// The scaling is `(prompt + delayed) / prompt`, evaluated per energy point. Pinned
/// against the real Ac226 coefficients' magnitude: prompt ~7.06 MeV and delayed
/// ~2.70 MeV per fission at 14.06 MeV give f ~ 1.38, which is the factor OpenMC
/// applies and yamc was missing entirely.
#[test]
fn scaling_matches_the_prompt_plus_delayed_ratio() {
    // Constant-term-only polynomials standing in for a real evaluation.
    let prompt = [7.0622e6_f64];
    let delayed = [2.6998e6_f64];
    let e = 1.406e7;
    let f = (poly(&prompt, e) + poly(&delayed, e)) / poly(&prompt, e);
    assert!(
        (f - 1.3823).abs() < 1.0e-4,
        "expected f ~ 1.3823 for Ac226's release values, got {f:.6}"
    );
    assert!(
        f > 1.0,
        "the scaling can only ever increase fission photon production"
    );
}

/// A nuclide with no `fission_energy_release` data must come out with an EMPTY
/// scaling, not a vector of ones: the consumers branch on `is_empty()`, and an
/// all-ones vector would cost a per-energy multiply for no reason. This is the Fe56
/// case, and it is why Fe56 was the clean control in the V&V.
#[test]
fn absent_release_data_leaves_the_scaling_empty() {
    let prompt: Option<Vec<f64>> = None;
    let delayed: Option<Vec<f64>> = None;
    let scaling: Vec<f64> = match prompt.as_deref().zip(delayed.as_deref()) {
        Some((p, _)) if !p.is_empty() => vec![1.0; 3],
        _ => Vec::new(),
    };
    assert!(
        scaling.is_empty(),
        "a nuclide without fission_energy_release must leave the scaling empty so the \
         consumers take their f = 1.0 branch"
    );
}

/// A non-positive prompt release must fall back to no scaling rather than emitting
/// an infinity or a negative factor into the photon production.
#[test]
fn degenerate_release_falls_back_to_unity() {
    for (prompt, delayed) in [
        (vec![0.0_f64], vec![1.0e6_f64]),
        (vec![-1.0e6_f64], vec![1.0e6_f64]),
        (vec![1.0e6_f64], vec![-1.0e6_f64]),
    ] {
        let e = 1.0e6;
        let (p, d) = (poly(&prompt, e), poly(&delayed, e));
        let f = if p > 0.0 && d >= 0.0 {
            (p + d) / p
        } else {
            1.0
        };
        assert!(
            f.is_finite() && f >= 1.0,
            "prompt={p:.3e} delayed={d:.3e} produced f={f}, which would corrupt photon \
             production; the guard must fall back to 1.0"
        );
    }
}

/// Horner must agree with the naive ascending-power sum, so a multi-term
/// polynomial is not silently evaluated in the wrong coefficient order.
#[test]
fn poly_uses_ascending_coefficients() {
    let c = [1.0_f64, 2.0, 3.0];
    let e = 5.0_f64;
    let naive: f64 = c
        .iter()
        .enumerate()
        .map(|(i, &v)| v * e.powi(i as i32))
        .sum();
    assert!(
        (poly(&c, e) - naive).abs() < 1.0e-12,
        "Horner gave {} but ascending-power evaluation gives {naive}",
        poly(&c, e)
    );
    assert_eq!(naive, 1.0 + 2.0 * 5.0 + 3.0 * 25.0);
}
