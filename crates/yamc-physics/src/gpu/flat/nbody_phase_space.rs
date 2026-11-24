//! N-body phase-space (E_out, μ) sampler.
//!
//! Implements the ENDF N-body phase-space distribution used
//! for inelastic reactions that emit 3, 4, or 5 outgoing bodies (e.g.
//! `(n, 2n)` with the residual nucleus counted as a third body).
//! The energy distribution is `Maxwell(x) / (Maxwell(x) + Y_n(y))`
//! scaled to the kinematically-available CM-frame energy band
//! `E_max = (A_p − 1)/A_p · (m_t/(m_t+1) · E_in + Q)`.
//!
//! `μ` is isotropic in the CM frame (NBPS does not provide an angular
//! table); the caller is responsible for the CM→lab rotation.
//!
//! Bit-identical to the cubecl kernel's NBPS branch -- same RNG draw
//! schedule, same `e_max` formula, same `Maxwell(x)` shape, same per-
//! `n_bodies` `y` formula.

use yamc_rng::next_xi;

/// Sample an outgoing CM-frame energy and isotropic μ from the N-body
/// phase-space distribution.
///
/// # Arguments
/// * `e_in` -- incident neutron energy (laboratory frame).
/// * `target_mass` -- atomic-mass ratio of the target nucleus (in
///   neutron masses; `m_t / m_n`).
/// * `q_value` -- Q-value of the reaction in eV.
/// * `n_bodies` -- number of outgoing bodies (valid: 3, 4, or 5).
/// * `total_mass` -- total atomic-mass ratio of all outgoing bodies
///   (denoted `A_p` in ENDF; must be `> 1`).
/// * `state` -- inline PCG RNG state.
///
/// # Returns
/// `Some((e_cm, mu_cm))` on success; `None` when:
/// - `n_bodies` is outside `[3, 5]`,
/// - `total_mass ≤ 1`,
/// - the kinematic cap `e_max ≤ 0` (incident energy too low for the
///   reaction Q-value), or
/// - the sampled denominator `x + y` is non-positive.
///
/// # RNG schedule
/// - 3 draws for the Maxwell `x` sample (always)
/// - 3 / 3 / 6 draws for the `y` sample (n=3 / 4 / 5)
/// - 1 draw for the isotropic μ
pub fn sample_nbody_phase_space(
    e_in: f64,
    target_mass: f64,
    q_value: f64,
    n_bodies: u32,
    total_mass: f64,
    state: &mut u64,
) -> Option<(f64, f64)> {
    if !(3..=5).contains(&n_bodies) {
        return None;
    }
    if total_mass <= 1.0 {
        return None;
    }
    let e_max =
        (total_mass - 1.0) / total_mass * (target_mass / (target_mass + 1.0) * e_in + q_value);
    if e_max <= 0.0 {
        return None;
    }

    // Maxwell sample for x.
    let xr1 = next_xi(state);
    let xr2 = next_xi(state);
    let xr3 = next_xi(state);
    let cos_xr3 = (std::f64::consts::FRAC_PI_2 * xr3).cos();
    let x_m = -xr1.ln() - xr2.ln() * cos_xr3 * cos_xr3;

    let y_m = match n_bodies {
        3 => {
            let yr1 = next_xi(state);
            let yr2 = next_xi(state);
            let yr3 = next_xi(state);
            let cos_yr3 = (std::f64::consts::FRAC_PI_2 * yr3).cos();
            -yr1.ln() - yr2.ln() * cos_yr3 * cos_yr3
        }
        4 => {
            let yr1 = next_xi(state);
            let yr2 = next_xi(state);
            let yr3 = next_xi(state);
            -(yr1 * yr2 * yr3).ln()
        }
        _ => {
            // n_bodies == 5
            let yr1 = next_xi(state);
            let yr2 = next_xi(state);
            let yr3 = next_xi(state);
            let yr4 = next_xi(state);
            let yr5 = next_xi(state);
            let yr6 = next_xi(state);
            let cos_yr6 = (std::f64::consts::FRAC_PI_2 * yr6).cos();
            -(yr1 * yr2 * yr3 * yr4).ln() - yr5.ln() * cos_yr6 * cos_yr6
        }
    };

    let denom = x_m + y_m;
    let e_out = if denom > 0.0 {
        e_max * (x_m / denom)
    } else {
        return None;
    };

    let xi_mu = next_xi(state);
    let mu = 1.0 - 2.0 * xi_mu;
    Some((e_out.max(0.0), mu.clamp(-1.0, 1.0)))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `n_bodies` outside `[3, 5]` → `None`, no state advance.
    #[test]
    fn nbody_returns_none_for_invalid_body_count() {
        let mut state: u64 = 0xC0DE;
        let before = state;
        for n in [0u32, 1, 2, 6, 7, 100] {
            let r = sample_nbody_phase_space(14.06e6, 56.0, -5.0e6, n, 57.0, &mut state);
            assert!(r.is_none(), "n_bodies={n} should return None");
            assert_eq!(state, before, "invalid n_bodies must not advance RNG state");
        }
    }

    /// `total_mass ≤ 1` → `None`, no state advance.
    #[test]
    fn nbody_returns_none_for_degenerate_total_mass() {
        let mut state: u64 = 0xFADE;
        let before = state;
        for ap in [-1.0_f64, 0.0, 0.5, 1.0] {
            let r = sample_nbody_phase_space(14.06e6, 56.0, -5.0e6, 3, ap, &mut state);
            assert!(r.is_none(), "total_mass={ap} should return None");
            assert_eq!(
                state, before,
                "degenerate total_mass must not advance RNG state"
            );
        }
    }

    /// E_in below the reaction threshold (`m_t/(m_t+1)·E_in + Q ≤ 0`)
    /// → `None`, no state advance.
    #[test]
    fn nbody_returns_none_below_threshold() {
        let mut state: u64 = 0xBEAD;
        let before = state;
        // m_t = 56, Q = -100 MeV; need E_in > 100·(57/56) MeV ≈ 101.8 MeV.
        let r = sample_nbody_phase_space(1.0e6, 56.0, -100.0e6, 3, 57.0, &mut state);
        assert!(r.is_none());
        assert_eq!(state, before);
    }

    /// Happy path: each valid `n_bodies` (3, 4, 5) produces an
    /// (e_out, μ) pair with μ ∈ [-1, 1] and e_out positive.
    #[test]
    fn nbody_produces_bounded_output_for_each_branch() {
        for n in [3u32, 4, 5] {
            let mut state: u64 = 0xCAFE_F00D;
            let mut any_some = false;
            // Average over a few draws -- sometimes denom can be near
            // zero by chance; we just want to confirm valid samples
            // come out on the happy path.
            for _ in 0..32 {
                if let Some((e_out, mu)) =
                    sample_nbody_phase_space(14.06e6, 56.0, -5.0e6, n, 57.0, &mut state)
                {
                    any_some = true;
                    assert!(e_out >= 0.0, "n={n}: e_out {e_out} should be ≥ 0");
                    assert!((-1.0..=1.0).contains(&mu), "n={n}: μ {mu} out of [-1, 1]");
                }
            }
            assert!(
                any_some,
                "n_bodies={n} should produce at least one sample over 32 tries"
            );
        }
    }

    /// `μ` is isotropic -- the mean over many draws should sit near 0.
    #[test]
    fn nbody_mu_is_isotropic_on_average() {
        let mut state: u64 = 0xC1A551C5;
        let mut sum_mu = 0.0;
        let mut n_ok = 0;
        for _ in 0..4_000usize {
            if let Some((_e, mu)) =
                sample_nbody_phase_space(14.06e6, 56.0, -5.0e6, 4, 57.0, &mut state)
            {
                sum_mu += mu;
                n_ok += 1;
            }
        }
        assert!(n_ok > 100, "too few accepted samples: {n_ok}");
        let mean_mu = sum_mu / n_ok as f64;
        assert!(
            mean_mu.abs() < 0.05,
            "mean μ {mean_mu} should be ≈ 0 (isotropic), got |μ̄| ≥ 0.05"
        );
    }
}
