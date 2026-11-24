//! Two-body CM→lab kinematic transform for inelastic scattering.
//!
//! Given an outgoing energy in the center-of-mass frame `e_cm`, a
//! CM-frame cosine `mu_cm`, the incident lab energy `e_in`, and the
//! target nucleus mass ratio `A`, return the laboratory-frame
//! `(mu_lab, e_lab)`. Bit-identical to the cubecl kernel's CM→lab
//! conversion at the end of `sample_inelastic_angle`.
//!
//! No RNG draws. Pure deterministic kinematic conversion.

/// Convert a CM-frame `(mu_cm, e_cm)` pair to lab-frame
/// `(mu_lab, e_lab)`.
///
/// # Arguments
/// * `e_in` -- incident lab-frame energy.
/// * `e_cm` -- outgoing CM-frame energy.
/// * `mu_cm` -- outgoing CM-frame cosine.
/// * `target_mass` -- atomic-mass ratio `A = m_t / m_n`.
///
/// # Returns
/// `Some((mu_lab, e_lab))` on success; `None` when the kinematic
/// solution gives a non-positive lab energy (callers fall back to
/// leaving the particle's pre-collision state unchanged).
///
/// # Formula
/// `e_lab = e_cm + (e_in + 2·μ·(A+1)·√(e_in·e_cm)) / (A+1)²`
/// `μ_lab = μ_cm · √(e_cm/e_lab) + √(e_in/e_lab) / (A+1)`
#[inline]
pub fn cm_to_lab(e_in: f64, e_cm: f64, mu_cm: f64, target_mass: f64) -> Option<(f64, f64)> {
    let one_plus_a = target_mass + 1.0;
    let denom = one_plus_a * one_plus_a;
    let e_lab = e_cm + (e_in + 2.0 * mu_cm * one_plus_a * (e_in * e_cm).sqrt()) / denom;
    if e_lab <= 0.0 {
        return None;
    }
    let mu_lab = (mu_cm * (e_cm / e_lab).sqrt() + (1.0 / one_plus_a) * (e_in / e_lab).sqrt())
        .clamp(-1.0, 1.0);
    Some((mu_lab, e_lab))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward-peaked CM (μ_cm ≈ 1) gives forward-peaked lab and
    /// `e_lab > e_cm` (the recoil contribution adds energy).
    #[test]
    fn forward_peak_cm_gives_forward_peak_lab_with_more_energy() {
        let (mu_lab, e_lab) = cm_to_lab(14.06e6, 1.0e6, 1.0, 56.0).unwrap();
        assert!(
            mu_lab > 0.9,
            "forward μ_cm = 1 should give μ_lab > 0.9, got {mu_lab}"
        );
        assert!(
            e_lab > 1.0e6,
            "e_lab {e_lab} should exceed e_cm = 1e6 (recoil contribution)"
        );
    }

    /// Back-scatter CM (μ_cm ≈ -1) on a heavy target gives μ_lab
    /// somewhere between μ_cm and the recoil-induced positive
    /// contribution. With A = 56 the recoil term is tiny, so
    /// μ_lab ≈ -1 still.
    #[test]
    fn back_scatter_heavy_target_stays_back() {
        let (mu_lab, _) = cm_to_lab(14.06e6, 1.0e6, -1.0, 56.0).unwrap();
        assert!(
            mu_lab < -0.9,
            "back-scatter on heavy target: μ_lab {mu_lab} should be ≈ -1"
        );
    }

    /// Hydrogen (A = 1): the CM and lab frames differ substantially.
    /// Forward CM with mu_cm = 0 (sideways) gives forward lab
    /// because of the recoil push.
    #[test]
    fn hydrogen_sideways_cm_becomes_forward_lab() {
        let (mu_lab, _) = cm_to_lab(1.0e6, 0.5e6, 0.0, 1.0).unwrap();
        assert!(
            mu_lab > 0.0,
            "with A=1 and μ_cm=0, recoil push gives μ_lab > 0, got {mu_lab}"
        );
    }

    /// Output μ_lab is always clamped to `[-1, 1]` even when the
    /// raw closed-form briefly numerical-overshoots at the boundary.
    #[test]
    fn mu_lab_is_clamped() {
        let (mu_lab, _) = cm_to_lab(14.06e6, 1.0e6, 1.0, 56.0).unwrap();
        assert!((-1.0..=1.0).contains(&mu_lab));
        let (mu_lab, _) = cm_to_lab(14.06e6, 1.0e6, -1.0, 56.0).unwrap();
        assert!((-1.0..=1.0).contains(&mu_lab));
    }
}
