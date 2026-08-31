//! Univariate probability distributions.
//!
//! The primitives the distribution readers are built from: a set of discrete
//! values with probabilities, a tabulated density, a uniform range, and a
//! weighted mixture of any of those.
//!
//! XML serialisation is deliberately absent. The Python module reads and writes
//! these as OpenMC XML elements; adding that here would mean an XML dependency
//! in a crate that has none, for a format the Arrow path never touches.

use crate::error::{Error, Result};

/// How a [`Tabular`] density is interpolated between its points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    Histogram,
    #[default]
    LinearLinear,
    LinearLog,
    LogLinear,
    LogLog,
}

impl Interpolation {
    /// The name the Python package uses, e.g. `"linear-linear"`.
    pub fn name(self) -> &'static str {
        match self {
            Interpolation::Histogram => "histogram",
            Interpolation::LinearLinear => "linear-linear",
            Interpolation::LinearLog => "linear-log",
            Interpolation::LogLinear => "log-linear",
            Interpolation::LogLog => "log-log",
        }
    }

    pub fn from_name(name: &str) -> Result<Interpolation> {
        Ok(match name {
            "histogram" => Interpolation::Histogram,
            "linear-linear" => Interpolation::LinearLinear,
            "linear-log" => Interpolation::LinearLog,
            "log-linear" => Interpolation::LogLinear,
            "log-log" => Interpolation::LogLog,
            _ => {
                return Err(Error::BadInterpolation {
                    name: name.to_string(),
                })
            }
        })
    }

    /// The ENDF interpolation code, as TAB1 records number them.
    pub fn from_endf_code(code: i32) -> Result<Interpolation> {
        Ok(match code {
            1 => Interpolation::Histogram,
            2 => Interpolation::LinearLinear,
            3 => Interpolation::LinearLog,
            4 => Interpolation::LogLinear,
            5 => Interpolation::LogLog,
            _ => {
                return Err(Error::BadInterpolation {
                    name: code.to_string(),
                })
            }
        })
    }
}

/// `(exp(x) - 1) / x`, without the cancellation that spoils it near zero.
fn exprel(x: f64) -> f64 {
    if x.abs() < 1e-16 {
        1.0
    } else {
        x.exp_m1() / x
    }
}

/// A distribution over a finite set of values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Discrete {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    /// The cumulative distribution as the source file gave it.
    ///
    /// ACE tables store a CDF next to the density; it is kept verbatim rather
    /// than recomputed so the values round-trip exactly. [`Self::cdf`] is the
    /// computed equivalent, and the two need not agree bit for bit. `None`
    /// when the source format supplied no CDF.
    pub c: Option<Vec<f64>>,
}

impl Discrete {
    pub fn new(x: Vec<f64>, p: Vec<f64>) -> Discrete {
        Discrete { x, p, c: None }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The cumulative distribution, opening with a zero, so it is one longer
    /// than `p`.
    pub fn cdf(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.p.len() + 1);
        out.push(0.0);
        let mut running = 0.0;
        for &v in &self.p {
            running += v;
            out.push(running);
        }
        out
    }

    pub fn integral(&self) -> f64 {
        self.p.iter().sum()
    }

    pub fn normalize(&mut self) {
        let total: f64 = self.p.iter().sum();
        for v in &mut self.p {
            *v /= total;
        }
    }

    /// Combine several discrete distributions, weighting each by a probability.
    ///
    /// A value appearing in more than one becomes a single entry whose
    /// probability is the weighted sum. The result is sorted by value.
    pub fn merge(dists: &[Discrete], probs: &[f64]) -> Result<Discrete> {
        if dists.len() != probs.len() {
            return Err(Error::Mismatched {
                what: "number of distributions and probabilities",
            });
        }
        // Values are floats, so they are collected by bit pattern rather than
        // through a hash map, and sorted afterwards.
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (dist, &weight) in dists.iter().zip(probs) {
            for (&x, &p) in dist.x.iter().zip(&dist.p) {
                match merged.iter_mut().find(|(v, _)| *v == x) {
                    Some((_, total)) => *total += p * weight,
                    None => merged.push((x, p * weight)),
                }
            }
        }
        merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Discrete {
            x: merged.iter().map(|&(v, _)| v).collect(),
            p: merged.iter().map(|&(_, p)| p).collect(),
            c: None,
        })
    }
}

/// A tabulated probability density.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tabular {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    pub interpolation: Interpolation,
    /// The cumulative distribution as the source file gave it; see
    /// [`Discrete::c`].
    pub c: Option<Vec<f64>>,
}

impl Tabular {
    pub fn new(x: Vec<f64>, p: Vec<f64>, interpolation: Interpolation) -> Tabular {
        Tabular {
            x,
            p,
            interpolation,
            c: None,
        }
    }

    /// The same, with the CDF the source file supplied alongside the density.
    pub fn with_cdf(
        x: Vec<f64>,
        p: Vec<f64>,
        interpolation: Interpolation,
        c: Vec<f64>,
    ) -> Tabular {
        Tabular {
            x,
            p,
            interpolation,
            c: Some(c),
        }
    }

    pub fn len(&self) -> usize {
        self.p.len()
    }

    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }

    /// The cumulative distribution at each tabulated point.
    ///
    /// Integrated analytically per interpolation law rather than numerically,
    /// so the result is exact for the density the table describes.
    pub fn cdf(&self) -> Vec<f64> {
        let n = self.x.len();
        let mut c = vec![0.0; n];
        if n < 2 {
            return c;
        }
        let (x, p) = (&self.x, &self.p);

        for i in 0..n - 1 {
            let dx = x[i + 1] - x[i];
            // The logs are subtracted rather than taken of a ratio. The two
            // are equal in exact arithmetic and not in floating point, and the
            // Python does it this way (`np.diff(np.log(x))`, not
            // `np.log(x[1:] / x[:-1])`) so matching it keeps the two readers
            // bit-identical rather than merely close.
            let dlog_x = || x[i + 1].ln() - x[i].ln();
            c[i + 1] = match self.interpolation {
                Interpolation::Histogram => p[i] * dx,
                Interpolation::LinearLinear => 0.5 * (p[i] + p[i + 1]) * dx,
                Interpolation::LinearLog => {
                    let dlog = dlog_x();
                    let m = (p[i + 1] - p[i]) / dlog;
                    p[i] * dx + m * (x[i + 1] * (dlog - 1.0) + x[i])
                }
                Interpolation::LogLinear => {
                    let m = (p[i + 1].ln() - p[i].ln()) / dx;
                    p[i] * dx * exprel(m * dx)
                }
                Interpolation::LogLog => {
                    let dlog = dlog_x();
                    let m = ((x[i + 1] * p[i + 1]).ln() - (x[i] * p[i]).ln()) / dlog;
                    x[i] * p[i] * dlog * exprel(m * dlog)
                }
            };
        }

        // Accumulate in place, as the Python does with cumsum.
        for i in 1..n {
            c[i] += c[i - 1];
        }
        c
    }

    pub fn integral(&self) -> f64 {
        self.cdf().last().copied().unwrap_or(0.0)
    }

    pub fn normalize(&mut self) {
        let max = self
            .cdf()
            .into_iter()
            .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        for v in &mut self.p {
            *v /= max;
        }
    }
}

/// A uniform distribution over `[a, b]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Default for Uniform {
    fn default() -> Self {
        Uniform { a: 0.0, b: 1.0 }
    }
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Uniform {
        Uniform { a, b }
    }

    /// The same distribution written as a two-point tabulation.
    pub fn to_tabular(&self) -> Tabular {
        let p = 1.0 / (self.b - self.a);
        Tabular::with_cdf(
            vec![self.a, self.b],
            vec![p, p],
            Interpolation::Histogram,
            vec![0.0, 1.0],
        )
    }

    pub fn cdf(&self) -> Vec<f64> {
        vec![0.0, 1.0]
    }

    pub fn integral(&self) -> f64 {
        1.0
    }
}

/// Any one of the univariate distributions.
#[derive(Debug, Clone, PartialEq)]
pub enum Univariate {
    Discrete(Discrete),
    Tabular(Tabular),
    Uniform(Uniform),
    Mixture(Mixture),
}

impl Univariate {
    pub fn integral(&self) -> f64 {
        match self {
            Univariate::Discrete(d) => d.integral(),
            Univariate::Tabular(t) => t.integral(),
            Univariate::Uniform(u) => u.integral(),
            Univariate::Mixture(m) => m.integral(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Univariate::Discrete(d) => d.len(),
            Univariate::Tabular(t) => t.len(),
            Univariate::Uniform(_) => 2,
            Univariate::Mixture(m) => m.distribution.iter().map(|d| d.len()).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A weighted mixture of other distributions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mixture {
    pub probability: Vec<f64>,
    pub distribution: Vec<Univariate>,
}

impl Mixture {
    pub fn new(probability: Vec<f64>, distribution: Vec<Univariate>) -> Mixture {
        Mixture {
            probability,
            distribution,
        }
    }

    /// A mixture has no single tabulated CDF; take each component's instead.
    pub fn integral(&self) -> f64 {
        self.probability
            .iter()
            .zip(&self.distribution)
            .map(|(&p, d)| p * d.integral())
            .sum()
    }

    pub fn normalize(&mut self) {
        let total: f64 = self.probability.iter().sum();
        for v in &mut self.probability {
            *v /= total;
        }
    }
}

/// Combine distributions, each weighted by a probability or an intensity.
///
/// Several [`Discrete`] distributions merge into one; the rest go into a
/// [`Mixture`]. A `Mixture` in the input is flattened first, so the result is
/// never nested.
///
/// [`Uniform`] is not accepted: the Python reader type-checks it out, and
/// nothing produces one here. It is folded into a [`Tabular`] by
/// [`Uniform::to_tabular`] if a caller needs it combined.
pub fn combine_distributions(dists: &[Univariate], probs: &[f64]) -> Result<Univariate> {
    if dists.len() != probs.len() {
        return Err(Error::Mismatched {
            what: "number of distributions and probabilities",
        });
    }

    // Flatten any mixture, scaling its components by its own weight.
    let mut flat: Vec<(f64, &Univariate)> = Vec::new();
    for (dist, &prob) in dists.iter().zip(probs) {
        match dist {
            Univariate::Mixture(m) => {
                for (&p, d) in m.probability.iter().zip(&m.distribution) {
                    flat.push((prob * p, d));
                }
            }
            _ => flat.push((prob, dist)),
        }
    }

    let mut discrete: Vec<Discrete> = Vec::new();
    let mut discrete_probs: Vec<f64> = Vec::new();
    let mut continuous: Vec<Univariate> = Vec::new();
    let mut continuous_probs: Vec<f64> = Vec::new();
    for (prob, dist) in flat {
        match dist {
            Univariate::Discrete(d) => {
                discrete.push(d.clone());
                discrete_probs.push(prob);
            }
            Univariate::Tabular(_) => {
                continuous.push(dist.clone());
                continuous_probs.push(prob);
            }
            other => {
                return Err(Error::Unsupported {
                    what: match other {
                        Univariate::Uniform(_) => "combining a uniform distribution",
                        _ => "combining a nested mixture",
                    },
                })
            }
        }
    }

    if !discrete.is_empty() {
        let merged = Univariate::Discrete(Discrete::merge(&discrete, &discrete_probs)?);
        if continuous.is_empty() {
            return Ok(merged);
        }
        // The merged discrete distribution already carries its weights, so it
        // enters the mixture with a probability of one.
        continuous_probs.push(1.0);
        continuous.push(merged);
        return Ok(Univariate::Mixture(Mixture::new(
            continuous_probs,
            continuous,
        )));
    }

    if continuous.len() == 1 {
        let Univariate::Tabular(t) = &continuous[0] else {
            unreachable!("only tabular distributions reach here");
        };
        let mut scaled = t.clone();
        for v in &mut scaled.p {
            *v *= continuous_probs[0];
        }
        return Ok(Univariate::Tabular(scaled));
    }

    Ok(Univariate::Mixture(Mixture::new(
        continuous_probs,
        continuous,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_accumulates_and_normalises() {
        let mut d = Discrete::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 1.0]);
        assert_eq!(d.cdf(), vec![0.0, 1.0, 3.0, 4.0]);
        assert_eq!(d.integral(), 4.0);
        d.normalize();
        assert_eq!(d.p, vec![0.25, 0.5, 0.25]);
        assert_eq!(d.integral(), 1.0);
    }

    #[test]
    fn discrete_merge_combines_shared_values() {
        let a = Discrete::new(vec![1.0, 2.0], vec![0.5, 0.5]);
        let b = Discrete::new(vec![2.0, 3.0], vec![1.0, 1.0]);
        // Weighted by 1.0 and 2.0: the shared value 2.0 gets 0.5 + 2.0.
        let m = Discrete::merge(&[a, b], &[1.0, 2.0]).unwrap();
        assert_eq!(m.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(m.p, vec![0.5, 2.5, 2.0]);
    }

    #[test]
    fn merge_rejects_mismatched_lengths() {
        let a = Discrete::new(vec![1.0], vec![1.0]);
        assert!(Discrete::merge(&[a], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn tabular_integrates_each_law_exactly() {
        // A flat density of 0.5 over [0, 2] integrates to 1 under histogram
        // and linear-linear alike.
        let x = vec![0.0, 1.0, 2.0];
        let p = vec![0.5, 0.5, 0.5];
        for law in [Interpolation::Histogram, Interpolation::LinearLinear] {
            let t = Tabular::new(x.clone(), p.clone(), law);
            assert!((t.integral() - 1.0).abs() < 1e-12, "{}", law.name());
        }

        // A straight line from 0 to 2 over [0, 2] has area 2 under
        // linear-linear, which the trapezoid rule gets exactly.
        let t = Tabular::new(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0],
            Interpolation::LinearLinear,
        );
        assert!((t.integral() - 2.0).abs() < 1e-12);
        assert_eq!(t.cdf(), vec![0.0, 0.5, 2.0]);
    }

    #[test]
    fn tabular_normalises_to_unit_area() {
        let mut t = Tabular::new(
            vec![0.0, 1.0, 2.0],
            vec![1.0, 1.0, 1.0],
            Interpolation::LinearLinear,
        );
        t.normalize();
        assert!((t.integral() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exprel_is_stable_at_zero() {
        // The naive (exp(x)-1)/x is 0/0 here; the limit is 1.
        assert_eq!(exprel(0.0), 1.0);
        assert!((exprel(1e-20) - 1.0).abs() < 1e-15);
        // And agrees with the naive form away from zero.
        assert!((exprel(2.0) - (2.0f64.exp() - 1.0) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn uniform_becomes_a_flat_table() {
        let u = Uniform::new(-1.0, 1.0);
        let t = u.to_tabular();
        assert_eq!(t.x, vec![-1.0, 1.0]);
        assert_eq!(t.p, vec![0.5, 0.5]);
        assert_eq!(u.integral(), 1.0);
    }

    #[test]
    fn mixture_weights_its_components() {
        let a = Univariate::Discrete(Discrete::new(vec![1.0], vec![1.0]));
        let b = Univariate::Discrete(Discrete::new(vec![2.0], vec![3.0]));
        let mut m = Mixture::new(vec![0.25, 0.75], vec![a, b]);
        assert_eq!(m.integral(), 0.25 * 1.0 + 0.75 * 3.0);
        m.normalize();
        assert_eq!(m.probability, vec![0.25, 0.75]);
    }

    #[test]
    fn interpolation_names_and_endf_codes_agree() {
        for (code, name) in [
            (1, "histogram"),
            (2, "linear-linear"),
            (3, "linear-log"),
            (4, "log-linear"),
            (5, "log-log"),
        ] {
            let by_code = Interpolation::from_endf_code(code).unwrap();
            assert_eq!(by_code.name(), name);
            assert_eq!(Interpolation::from_name(name).unwrap(), by_code);
        }
        assert!(Interpolation::from_endf_code(9).is_err());
        assert!(Interpolation::from_name("bilinear").is_err());
    }
}
