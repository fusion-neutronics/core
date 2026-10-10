//! Decay continua: a particle spectrum given as a density rather than as lines.
//!
//! ENDF-6 writes a decay spectrum as `S(E) = FC * RP(E) + FD * sum_r RI_r *
//! delta(E - ER_r)` (ENDF-102, section 8.4.2): discrete lines and, beside or
//! instead of them, a continuum `RP(E)` tabulated at points and read between
//! them by an interpolation law. A chain stores the continuum as an
//! emission-rate density per atom, per second per eV, so its values are not
//! rates. Its rate is its integral, and the law decides the integral: the same
//! points read as a histogram and as linear-linear give different totals.
//!
//! Summing the tabulated values as though they were lines would put a
//! continuum's emission low by a factor of about its grid spacing in eV (roughly 1e4 for Sm158 on
//! ENDF/B-VIII.1, 2e5 for Cf252 on JEFF-4.0). [`Continuum`] reads the density
//! the way the evaluation defines it: its value anywhere, its integral, and
//! the energy below which a given share of an interval's integral lies, all in
//! closed form.

use std::fmt;

/// The ENDF-6 interpolation law (INT) a continuum is read with between its
/// tabulated points.
///
/// All five are kept, so a chain round-trips what its file said. Every photon
/// continuum in ENDF/B-VIII.1, JEFF-4.0 and JENDL-5.0 is a histogram or
/// linear-linear; the log laws appear on neutron continua, which nothing here
/// integrates, and [`Continuum::new`] refuses them rather than reading them
/// as something else.
///
/// [`crate::chain::EvaluatedYields::interpolation`] keeps its law as the raw
/// ENDF code instead, so it carries any code the tape wrote. The law the
/// solver interpolates the yields with is a
/// [`crate::chain::YieldInterpolation`], which has only the two laws
/// evaluations use, and a chain stating another is refused when it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// INT=1: each point's value holds up to the next point.
    Histogram,
    /// INT=2.
    LinearLinear,
    /// INT=3: linear in the logarithm of the energy.
    LinearLog,
    /// INT=4: the logarithm of the value linear in the energy.
    LogLinear,
    /// INT=5.
    LogLog,
}

impl Interpolation {
    /// The law an ENDF interpolation code names, or `None` for a code outside
    /// 1 to 5.
    pub fn from_endf_code(code: i32) -> Option<Interpolation> {
        match code {
            1 => Some(Interpolation::Histogram),
            2 => Some(Interpolation::LinearLinear),
            3 => Some(Interpolation::LinearLog),
            4 => Some(Interpolation::LogLinear),
            5 => Some(Interpolation::LogLog),
            _ => None,
        }
    }

    /// The ENDF interpolation code.
    pub fn endf_code(self) -> i32 {
        match self {
            Interpolation::Histogram => 1,
            Interpolation::LinearLinear => 2,
            Interpolation::LinearLog => 3,
            Interpolation::LogLinear => 4,
            Interpolation::LogLog => 5,
        }
    }

    /// The name the endf crate and OpenMC use, e.g. `"linear-linear"`.
    pub fn name(self) -> &'static str {
        match self {
            Interpolation::Histogram => "histogram",
            Interpolation::LinearLinear => "linear-linear",
            Interpolation::LinearLog => "linear-log",
            Interpolation::LogLinear => "log-linear",
            Interpolation::LogLog => "log-log",
        }
    }
}

/// Why a tabulated continuum cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnreadableContinuum {
    /// The file states no interpolation law. A `decay/sources.arrow` written
    /// before the `interpolation` column states none for any continuum, and
    /// without one the continuum has no known integral.
    NoLaw,
    /// A law other than histogram or linear-linear.
    Unsupported(Interpolation),
    /// The energy and density lists differ in length, so some value has no
    /// partner.
    UnpairedLists { energies: usize, densities: usize },
    /// An energy lies below the one before it; the tabulation must ascend.
    Descending { index: usize },
    /// An energy is not finite, so no interval holding it is known.
    NonFiniteEnergy { index: usize },
    /// A density is negative or not finite. An emission rate cannot be
    /// negative, and the running integral a sampler searches must not fall.
    InvalidDensity { index: usize },
}

impl fmt::Display for UnreadableContinuum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnreadableContinuum::NoLaw => write!(
                f,
                "states no interpolation law, so its integral is not known. The chain's \
                 decay/sources.arrow was written before the interpolation column existed: \
                 regenerate it with a current converter, or use a transmutation data release \
                 whose decay/sources.arrow carries that column"
            ),
            UnreadableContinuum::Unsupported(law) => write!(
                f,
                "is tabulated {}, a law this build does not integrate",
                law.name()
            ),
            UnreadableContinuum::UnpairedLists {
                energies,
                densities,
            } => write!(
                f,
                "pairs {energies} energies with {densities} densities, so it has no single reading"
            ),
            UnreadableContinuum::Descending { index } => write!(
                f,
                "has energy {index} below the one before it; a continuum is tabulated ascending"
            ),
            UnreadableContinuum::NonFiniteEnergy { index } => {
                write!(f, "has energy {index} that is not a finite number")
            }
            UnreadableContinuum::InvalidDensity { index } => write!(
                f,
                "has density {index} negative or not a finite number; an emission rate is \
                 finite and not negative"
            ),
        }
    }
}

/// A tabulated density read exactly under its law: histogram or
/// linear-linear, the two every photon continuum in the evaluated libraries
/// uses.
///
/// `energies` ascend, as an ENDF TAB1 does, and may repeat one energy to
/// place a jump. `densities` pairs with them one to one.
#[derive(Clone, Copy, Debug)]
pub struct Continuum<'a> {
    energies: &'a [f64],
    densities: &'a [f64],
    linear: bool,
}

impl<'a> Continuum<'a> {
    /// A continuum tabulated at `energies`, with `densities` read between them
    /// by `interpolation`.
    pub fn new(
        energies: &'a [f64],
        densities: &'a [f64],
        interpolation: Option<Interpolation>,
    ) -> Result<Continuum<'a>, UnreadableContinuum> {
        let linear = match interpolation {
            None => return Err(UnreadableContinuum::NoLaw),
            Some(Interpolation::Histogram) => false,
            Some(Interpolation::LinearLinear) => true,
            Some(other) => return Err(UnreadableContinuum::Unsupported(other)),
        };
        // The chain readers refuse unpaired rows, but the distribution types
        // have public fields, so a hand-built continuum is checked here too:
        // trimming or reading an unsorted table would give a wrong integral.
        if energies.len() != densities.len() {
            return Err(UnreadableContinuum::UnpairedLists {
                energies: energies.len(),
                densities: densities.len(),
            });
        }
        // A NaN compares false, so the finite check runs before the order
        // check would let one through.
        if let Some(index) = energies.iter().position(|e| !e.is_finite()) {
            return Err(UnreadableContinuum::NonFiniteEnergy { index });
        }
        if let Some(i) = energies.windows(2).position(|w| w[1] < w[0]) {
            return Err(UnreadableContinuum::Descending { index: i + 1 });
        }
        if let Some(index) = densities.iter().position(|y| !(y.is_finite() && *y >= 0.0)) {
            return Err(UnreadableContinuum::InvalidDensity { index });
        }
        Ok(Continuum {
            energies,
            densities,
            linear,
        })
    }

    /// A continuum from lists [`Continuum::new`] has already accepted under
    /// `interpolation`, without scanning them again. For a sampler that keeps
    /// the validated lists and reads them once per draw; the checks still run
    /// in debug builds.
    pub fn already_checked(
        energies: &'a [f64],
        densities: &'a [f64],
        interpolation: Interpolation,
    ) -> Continuum<'a> {
        debug_assert!(
            Continuum::new(energies, densities, Some(interpolation)).is_ok(),
            "Continuum::already_checked on lists Continuum::new refuses"
        );
        Continuum {
            energies,
            densities,
            linear: interpolation == Interpolation::LinearLinear,
        }
    }

    /// The tabulated energies.
    pub fn energies(&self) -> &'a [f64] {
        self.energies
    }

    /// The interval `[energies[i], energies[i + 1]]` a density at `energy` is
    /// read from, or `None` outside the tabulated range. At a repeated energy
    /// it is the interval above the jump.
    fn interval(&self, energy: f64) -> Option<usize> {
        let n = self.energies.len();
        if n < 2 || !(energy >= self.energies[0] && energy <= self.energies[n - 1]) {
            return None;
        }
        let above = self.energies.partition_point(|&e| e <= energy);
        Some(above.clamp(1, n - 1) - 1)
    }

    /// The density at `energy` and its slope there, both zero outside the
    /// tabulated range. The slope is zero for a histogram.
    ///
    /// Between two tabulated points the density is exactly `value + slope *
    /// (E - energy)`, which is what lets a fold against a smooth weight take
    /// the continuum's law exactly and leave only the weight to a quadrature.
    pub fn value_and_slope(&self, energy: f64) -> (f64, f64) {
        let Some(i) = self.interval(energy) else {
            return (0.0, 0.0);
        };
        let (e0, e1) = (self.energies[i], self.energies[i + 1]);
        let (y0, y1) = (self.densities[i], self.densities[i + 1]);
        if !self.linear || e1 <= e0 {
            return (y0, 0.0);
        }
        let slope = (y1 - y0) / (e1 - e0);
        (y0 + slope * (energy - e0), slope)
    }

    /// The density at `energy`, zero outside the tabulated range.
    pub fn density(&self, energy: f64) -> f64 {
        self.value_and_slope(energy).0
    }

    /// The integral over each interval between consecutive points, in order.
    pub fn interval_integrals(&self) -> impl Iterator<Item = f64> + 'a {
        let linear = self.linear;
        self.energies
            .windows(2)
            .zip(self.densities.windows(2))
            .map(move |(e, y)| {
                let width = e[1] - e[0];
                if linear {
                    0.5 * (y[0] + y[1]) * width
                } else {
                    y[0] * width
                }
            })
    }

    /// The integral over the whole tabulated range.
    pub fn integral(&self) -> f64 {
        self.interval_integrals().sum()
    }

    /// The integral of energy times the density over the whole tabulated
    /// range, exact under the law: the energy a continuum of photons per eV
    /// carries, in eV per unit of `integral`.
    ///
    /// On a linear-linear interval the integrand is the product of two linear
    /// functions, whose integral is `width / 6 * (y0 (2 e0 + e1) + y1 (e0 + 2
    /// e1))`; on a histogram one it is `y0 (e1^2 - e0^2) / 2`.
    pub fn energy_integral(&self) -> f64 {
        self.energies
            .windows(2)
            .zip(self.densities.windows(2))
            .map(|(e, y)| {
                let width = e[1] - e[0];
                if self.linear {
                    width / 6.0 * (y[0] * (2.0 * e[0] + e[1]) + y[1] * (e[0] + 2.0 * e[1]))
                } else {
                    y[0] * 0.5 * (e[1] + e[0]) * width
                }
            })
            .sum()
    }

    /// The energy in interval `i` below which `part` of that interval's
    /// integral lies: the exact inverse of the running integral from
    /// `energies[i]`, for `part` between zero and the interval's integral.
    pub fn energy_within(&self, i: usize, part: f64) -> f64 {
        let (e0, e1) = (self.energies[i], self.energies[i + 1]);
        let y0 = self.densities[i];
        if part <= 0.0 {
            return e0;
        }
        let offset = if self.linear {
            // Solve y0 t + slope t^2 / 2 = part for t. This root form has no
            // cancellation, and it holds when y0 is zero or the slope is.
            let slope = (self.densities[i + 1] - y0) / (e1 - e0);
            2.0 * part / (y0 + (y0 * y0 + 2.0 * slope * part).max(0.0).sqrt())
        } else {
            part / y0
        };
        (e0 + offset).clamp(e0, e1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENERGIES: [f64; 4] = [1.0e3, 2.0e3, 4.0e3, 8.0e3];
    const DENSITIES: [f64; 4] = [3.0, 1.0, 2.0, 0.0];

    fn histogram() -> Continuum<'static> {
        Continuum::new(&ENERGIES, &DENSITIES, Some(Interpolation::Histogram)).unwrap()
    }

    fn linear() -> Continuum<'static> {
        Continuum::new(&ENERGIES, &DENSITIES, Some(Interpolation::LinearLinear)).unwrap()
    }

    #[test]
    fn codes_and_names_round_trip() {
        for (code, name) in [
            (1, "histogram"),
            (2, "linear-linear"),
            (3, "linear-log"),
            (4, "log-linear"),
            (5, "log-log"),
        ] {
            let law = Interpolation::from_endf_code(code).unwrap();
            assert_eq!(law.endf_code(), code);
            assert_eq!(law.name(), name);
        }
        assert_eq!(Interpolation::from_endf_code(0), None);
        assert_eq!(Interpolation::from_endf_code(6), None);
    }

    /// The same points integrate differently under the two laws, which is why
    /// the law has to be stored.
    #[test]
    fn the_integral_follows_the_law() {
        assert_eq!(histogram().integral(), 3.0 * 1e3 + 1.0 * 2e3 + 2.0 * 4e3);
        assert_eq!(
            linear().integral(),
            2.0 * 1e3 + 1.5 * 2e3 + 1.0 * 4e3,
            "trapezoids"
        );
    }

    /// The energy integral against a midpoint sum fine enough to agree to
    /// many digits, under each law.
    #[test]
    fn the_energy_integral_follows_the_law() {
        for continuum in [histogram(), linear()] {
            let n = 200_000;
            let (lo, hi) = (ENERGIES[0], ENERGIES[3]);
            let h = (hi - lo) / n as f64;
            let sum: f64 = (0..n)
                .map(|k| {
                    let e = lo + (k as f64 + 0.5) * h;
                    e * continuum.density(e) * h
                })
                .sum();
            let exact = continuum.energy_integral();
            assert!((exact / sum - 1.0).abs() < 1e-6, "{exact} against {sum}");
        }
        // One histogram bin of height 3 from 1 to 2 keV carries 3 * 1.5e6 eV.
        assert_eq!(
            Continuum::new(&[1.0e3, 2.0e3], &[3.0, 0.0], Some(Interpolation::Histogram))
                .unwrap()
                .energy_integral(),
            4.5e6
        );
    }

    #[test]
    fn the_density_between_points_follows_the_law() {
        assert_eq!(histogram().density(1.5e3), 3.0);
        assert_eq!(histogram().density(3.0e3), 1.0);
        assert_eq!(linear().density(1.5e3), 2.0);
        assert_eq!(linear().value_and_slope(3.0e3), (1.5, 0.5e-3));
        assert_eq!(histogram().value_and_slope(3.0e3), (1.0, 0.0));
        // Outside the tabulated range nothing is emitted.
        assert_eq!(linear().density(999.0), 0.0);
        assert_eq!(linear().density(8.001e3), 0.0);
    }

    /// `energy_within` inverts the running integral exactly under each law.
    #[test]
    fn energy_within_inverts_the_running_integral() {
        for continuum in [histogram(), linear()] {
            for (i, whole) in continuum.interval_integrals().enumerate() {
                for share in [0.0, 0.1, 0.5, 0.9, 1.0] {
                    let e = continuum.energy_within(i, share * whole);
                    let (e0, y0) = (ENERGIES[i], DENSITIES[i]);
                    let below = if continuum.linear {
                        0.5 * (y0 + continuum.density(e)) * (e - e0)
                    } else {
                        y0 * (e - e0)
                    };
                    assert!(
                        (below - share * whole).abs() <= 1e-12 * whole,
                        "interval {i}, share {share}: {below} != {}",
                        share * whole
                    );
                }
            }
        }
    }

    #[test]
    fn a_jump_reads_from_the_interval_above_it() {
        let energies = [1.0, 2.0, 2.0, 3.0];
        let densities = [1.0, 1.0, 5.0, 5.0];
        let c = Continuum::new(&energies, &densities, Some(Interpolation::LinearLinear)).unwrap();
        assert_eq!(c.density(1.5), 1.0);
        assert_eq!(c.density(2.0), 5.0);
        assert_eq!(c.integral(), 1.0 + 0.0 + 5.0);
    }

    #[test]
    fn a_continuum_without_a_readable_law_is_refused() {
        assert_eq!(
            Continuum::new(&ENERGIES, &DENSITIES, None).unwrap_err(),
            UnreadableContinuum::NoLaw
        );
        assert_eq!(
            Continuum::new(&ENERGIES, &DENSITIES, Some(Interpolation::LogLog)).unwrap_err(),
            UnreadableContinuum::Unsupported(Interpolation::LogLog)
        );
        let message = UnreadableContinuum::NoLaw.to_string();
        assert!(message.contains("interpolation column"), "{message}");
    }

    #[test]
    fn an_unpaired_or_descending_table_is_refused() {
        assert_eq!(
            Continuum::new(
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0],
                Some(Interpolation::Histogram)
            )
            .unwrap_err(),
            UnreadableContinuum::UnpairedLists {
                energies: 3,
                densities: 2
            }
        );
        assert_eq!(
            Continuum::new(
                &[1.0, 3.0, 2.0],
                &[1.0, 2.0, 3.0],
                Some(Interpolation::Histogram)
            )
            .unwrap_err(),
            UnreadableContinuum::Descending { index: 2 }
        );
    }

    #[test]
    fn a_non_finite_energy_or_a_negative_density_is_refused() {
        assert_eq!(
            Continuum::new(
                &[1.0, f64::NAN, 3.0],
                &[1.0, 2.0, 3.0],
                Some(Interpolation::LinearLinear)
            )
            .unwrap_err(),
            UnreadableContinuum::NonFiniteEnergy { index: 1 }
        );
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                Continuum::new(
                    &[1.0, 2.0, 3.0],
                    &[1.0, 2.0, bad],
                    Some(Interpolation::Histogram)
                )
                .unwrap_err(),
                UnreadableContinuum::InvalidDensity { index: 2 }
            );
        }
        // A zero density is a stated value, not a refusal.
        assert!(Continuum::new(
            &[1.0, 2.0, 3.0],
            &[0.0, 2.0, 0.0],
            Some(Interpolation::LinearLinear)
        )
        .is_ok());
    }
}
