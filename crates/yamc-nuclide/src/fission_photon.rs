//! Fission energy release functions for the delayed-photon scaling (issue #369).
//!
//! A fission releases photons from the fission products' decay as well as
//! promptly, and an evaluation's prompt photon production does not include them.
//! OpenMC covers this by scaling fission photon production by
//! `f(E) = (prompt_photons(E) + delayed_photons(E)) / prompt_photons(E)`
//! (`settings::delayed_photon_scaling`, on by default, `physics.cpp`).
//!
//! Both terms come from ENDF MF=1/MT=458 and are `Function1D` objects, so they
//! are NOT always polynomials. Across ENDF/B-VIII.1, 79 of 557 nuclides carry
//! the data: 76 have both as polynomials, and three -- U235, U238 and Pu239, the
//! three that matter most -- store `prompt_photons` as a tabulated function.
//!
//! Storing the functions rather than pre-evaluated values keeps this independent
//! of any energy grid, so a regenerated grid can never silently mismatch a stale
//! vector of values.

use serde::{Deserialize, Serialize};

/// ENDF interpolation scheme code for linear-linear, the only scheme any
/// published `fission_energy_release` table uses.
pub const ENDF_LIN_LIN: i32 = 2;

/// One term of the fission energy release, in the representation the evaluation
/// stores it in.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ReleaseFunction {
    /// Coefficients in ASCENDING power order, evaluated with Horner.
    Polynomial(Vec<f64>),
    /// Single-region linear-linear table. Deliberately not a general ENDF
    /// `Tabulated1D`: every published table is one lin-lin region, and
    /// [`ReleaseFunction::from_tabulated`] refuses anything else rather than
    /// silently interpolating it the wrong way.
    LinLinTable { x: Vec<f64>, y: Vec<f64> },
}

impl ReleaseFunction {
    /// Build a tabulated term, rejecting any form this does not evaluate exactly.
    ///
    /// `interpolation` and `breakpoints` are the raw ENDF arrays. Anything other
    /// than a single linear-linear region is an error: silently treating a
    /// log-log region as linear would bias fission photon production by a few
    /// percent with nothing to show for it, which is precisely how #369 stayed
    /// invisible for so long.
    pub fn from_tabulated(
        x: Vec<f64>,
        y: Vec<f64>,
        interpolation: &[i32],
        breakpoints: &[i32],
        context: &str,
    ) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err(format!(
                "{context}: fission energy release table has {} x values and {} y \
                 values; they must match",
                x.len(),
                y.len()
            ));
        }
        if x.len() < 2 {
            return Err(format!(
                "{context}: fission energy release table needs at least 2 points, got {}",
                x.len()
            ));
        }
        if interpolation.len() > 1 || breakpoints.len() > 1 {
            return Err(format!(
                "{context}: fission energy release table is multi-region \
                 (interpolation {interpolation:?}, breakpoints {breakpoints:?}). Only a \
                 single linear-linear region is supported; evaluating this as one \
                 region would give the wrong photon production (#369)"
            ));
        }
        if let Some(&scheme) = interpolation.first() {
            if scheme != ENDF_LIN_LIN {
                return Err(format!(
                    "{context}: fission energy release table uses ENDF interpolation \
                     scheme {scheme}, but only {ENDF_LIN_LIN} (linear-linear) is \
                     supported. Treating it as linear-linear would silently bias \
                     fission photon production (#369)"
                ));
            }
        }
        Ok(Self::LinLinTable { x, y })
    }

    /// Evaluate at incident energy `e`, matching OpenMC's `Function1D` call.
    ///
    /// Off the ends the table is held flat, which is what OpenMC's `Tabulated1D`
    /// does outside its range.
    pub fn eval(&self, e: f64) -> f64 {
        match self {
            // Horner, so a high-order term cannot lose the low-order ones.
            Self::Polynomial(coeffs) => coeffs.iter().rev().fold(0.0, |acc, &c| acc * e + c),
            Self::LinLinTable { x, y } => {
                if e <= x[0] {
                    return y[0];
                }
                if e >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }
                // First index strictly greater than `e`, so `e` sits in
                // [x[i-1], x[i]).
                let i = x.partition_point(|&xi| xi <= e);
                let (x0, x1) = (x[i - 1], x[i]);
                let (y0, y1) = (y[i - 1], y[i]);
                let dx = x1 - x0;
                if dx <= 0.0 {
                    return y1;
                }
                y0 + (y1 - y0) * (e - x0) / dx
            }
        }
    }
}

/// The prompt and delayed photon release for one nuclide.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FissionPhotonRelease {
    pub prompt: ReleaseFunction,
    pub delayed: ReleaseFunction,
}

impl FissionPhotonRelease {
    /// `f(E) = (prompt + delayed) / prompt`, the factor fission photon production
    /// is scaled by.
    ///
    /// A non-positive prompt release, or a negative delayed one, would make the
    /// ratio meaningless, so those fall back to 1.0 (no scaling) rather than
    /// emitting an infinity or a factor below 1 into photon production. The
    /// scaling can only ever increase production.
    pub fn scaling(&self, e: f64) -> f64 {
        let p = self.prompt.eval(e);
        let d = self.delayed.eval(e);
        if p > 0.0 && d >= 0.0 {
            (p + d) / p
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_uses_ascending_coefficients() {
        let f = ReleaseFunction::Polynomial(vec![1.0, 2.0, 3.0]);
        // 1 + 2*5 + 3*25
        assert!((f.eval(5.0) - 86.0).abs() < 1e-12);
    }

    #[test]
    fn table_interpolates_linearly_and_holds_flat_outside() {
        let f = ReleaseFunction::LinLinTable {
            x: vec![1.0, 2.0, 4.0],
            y: vec![10.0, 20.0, 40.0],
        };
        assert!(
            (f.eval(1.5) - 15.0).abs() < 1e-12,
            "midpoint of the first region"
        );
        assert!(
            (f.eval(3.0) - 30.0).abs() < 1e-12,
            "midpoint of the second region"
        );
        assert!((f.eval(1.0) - 10.0).abs() < 1e-12, "on the first knot");
        assert!((f.eval(4.0) - 40.0).abs() < 1e-12, "on the last knot");
        assert!(
            (f.eval(0.1) - 10.0).abs() < 1e-12,
            "below the table is held flat"
        );
        assert!(
            (f.eval(99.0) - 40.0).abs() < 1e-12,
            "above the table is held flat"
        );
    }

    #[test]
    fn unsupported_interpolation_is_refused() {
        // 5 is log-log. Accepting it as linear is the silent-wrong-answer case.
        let err = ReleaseFunction::from_tabulated(
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            &[5],
            &[2],
            "U235 prompt_photons",
        )
        .expect_err("log-log must be refused, not silently treated as linear");
        assert!(
            err.contains("scheme 5"),
            "the message must name the scheme: {err}"
        );
        assert!(
            err.contains("U235"),
            "the message must name the nuclide: {err}"
        );
    }

    #[test]
    fn multi_region_is_refused() {
        let err = ReleaseFunction::from_tabulated(
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            &[2, 2],
            &[2, 3],
            "Pu239 prompt_photons",
        )
        .expect_err("a multi-region table must be refused");
        assert!(err.contains("multi-region"), "got: {err}");
    }

    #[test]
    fn single_lin_lin_region_is_accepted() {
        assert!(ReleaseFunction::from_tabulated(
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            &[ENDF_LIN_LIN],
            &[2],
            "U235 prompt_photons",
        )
        .is_ok());
    }

    /// Pinned against the real U235 numbers: prompt is tabulated, delayed is the
    /// polynomial `6.33e6 - 0.075 E`, and OpenMC evaluates f = 1.7734 at 2 MeV.
    #[test]
    fn u235_scaling_matches_openmc_at_2_mev() {
        let release = FissionPhotonRelease {
            prompt: ReleaseFunction::LinLinTable {
                x: vec![1.0e-5, 1.0e6, 2.0e6, 3.0e6],
                y: vec![7.281253e6, 7.969900e6, 7.990447e6, 8.461902e6],
            },
            delayed: ReleaseFunction::Polynomial(vec![6.33e6, -7.50e-2]),
        };
        let f = release.scaling(2.0e6);
        assert!(
            (f - 1.7734).abs() < 1.0e-4,
            "expected OpenMC's 1.7734 for U235 at 2 MeV, got {f}"
        );
    }

    #[test]
    fn degenerate_release_falls_back_to_no_scaling() {
        for (p, d) in [
            (vec![0.0], vec![1.0e6]),
            (vec![-1.0e6], vec![1.0e6]),
            (vec![1.0e6], vec![-1.0e6]),
        ] {
            let r = FissionPhotonRelease {
                prompt: ReleaseFunction::Polynomial(p),
                delayed: ReleaseFunction::Polynomial(d),
            };
            let f = r.scaling(1.0e6);
            assert!(
                f.is_finite() && f >= 1.0,
                "a degenerate release must fall back to 1.0, got {f}"
            );
        }
    }
}
