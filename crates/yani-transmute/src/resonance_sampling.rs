//! Sample a range's resonance parameters from their MF=32 covariance.
//!
//! [`range_covariances`](endf::resonance_covariance::range_covariances) gives
//! a range's parameters and their covariance as one dense matrix. A replica
//! draws a whole parameter vector from it and the cross sections are rebuilt
//! from that vector, so the draw has to be a distribution the
//! reconstruction can take: no negative widths, and the stated means and
//! covariance kept exactly, not to first order.
//!
//! # Marginals
//!
//! Each parameter is one of three kinds ([`Marginal`]):
//!
//! - **Gaussian** with the stated mean and variance: resonance energies,
//!   radius parameters, and everything signed. Reich-Moore GFA and GFB carry
//!   the sign of their amplitude, R-matrix limited channel widths are signed
//!   too (the reduced-width amplitude with IFG=1, the width with the
//!   amplitude's sign otherwise), and a width the evaluation states with a
//!   negative mean is one whose sign the evaluation means.
//! - **Lognormal** for a width that is positive by nature (neutron, capture
//!   and competitive widths, Breit-Wigner fission widths, and an unresolved
//!   range's average widths and level spacings, each a unit-mean multiplier),
//!   matched exactly to the stated mean and covariance. Every draw is
//!   positive.
//! - **Held** at its mean: a parameter with zero variance, and a width that
//!   is positive by nature with a zero mean and a nonzero sigma, which no
//!   positive distribution has. The second is not given a value of its own
//!   making; it is listed in [`SamplerReport::zero_mean_widths`].
//!
//! # The transform
//!
//! The draw is `x = m + L z` with `z` standard normal and `L L^T = Σ`, then
//! `y_i = exp(x_i)` for a lognormal parameter and `y_i = x_i` for a Gaussian
//! one. With `μ` and `C` the stated means and covariance, `Σ` and `m` are
//!
//! ```text
//! lognormal i, lognormal j   Σ_ij = ln(1 + C_ij / (μ_i μ_j))
//! lognormal i, Gaussian j    Σ_ij = C_ij / μ_i
//! Gaussian i, Gaussian j     Σ_ij = C_ij
//! lognormal i                m_i  = ln μ_i - Σ_ii / 2
//! Gaussian i                 m_i  = μ_i
//! ```
//!
//! The first is the moment matching of a multivariate lognormal: `E[y_i] =
//! exp(m_i + Σ_ii / 2) = μ_i` and `E[y_i y_j] = μ_i μ_j exp(Σ_ij)`. The second
//! is Stein's lemma: for `(x_i, x_j)` jointly Gaussian and any `g` with
//! `E|g'(x_i)|` finite, `Cov(g(x_i), x_j) = Cov(x_i, x_j) E[g'(x_i)]`. With
//! `g = exp`, `E[g'(x_i)] = E[y_i] = μ_i`, so `Cov(y_i, y_j) = μ_i Σ_ij`, and
//! `Σ_ij = C_ij / μ_i` reproduces `C_ij` exactly. Every first and second
//! moment of the draw is then the stated one, wherever `Σ` is a covariance.
//!
//! It need not be. Two lognormal widths whose `1 + C_ij / (μ_i μ_j)` is not
//! positive (a strong anticorrelation between widths of large relative
//! sigma) are a pair no lognormal carries; their `Σ_ij` is set to the most
//! negative value the two variances allow and counted in
//! [`SamplerReport::unattainable_pairs`]. And a `Σ` built from a PSD `C` can
//! come out indefinite, as two fully correlated widths of different relative
//! sigma do, or a width of large relative sigma `s` correlated with an
//! energy: their correlation in `Σ` is `ρ s / sqrt(ln(1 + s^2))`, past one
//! for `ρ` near one, and `s` reaches 56 among ENDF/B-VIII.1 Th232's widths.
//! Both are properties of the lognormal, not of the data, and the repair
//! below takes care of the second, reported apart from the repair of the
//! data ([`SamplerReport::transformed_repair`]).
//!
//! # Repair
//!
//! Evaluated matrices are often not positive semi-definite as written, from
//! rounding the correlations to the digits the tape carries (ENDF/B-VIII.1
//! W182 to W186 among them). Where the stated correlation matrix, or the
//! transformed one, has an eigenvalue below `-m * REPAIR_TOLERANCE` (`m` the
//! side of the block; the same threshold, and for the same reason, as
//! [`crate::covariance_sample::REPAIR_TOLERANCE`]), it is replaced by the
//! nearest correlation matrix in the Frobenius norm (N. J. Higham, IMA J.
//! Numer. Anal. 22 (2002) 329), found by Qi and Sun's Newton method (see
//! [`crate::nearest_correlation`], shared with the MF=33 sampler). That
//! keeps a unit diagonal, so every parameter keeps its evaluated sigma
//! exactly and only correlations move, where clipping the eigenvalues of the
//! covariance would widen the sigmas. What moved is in
//! [`CorrelationRepair`]. The stated matrix is repaired
//! first, so that the target of the transform is a covariance, then the
//! transformed one if it still needs it.
//!
//! A matrix that is PSD to within the threshold is not repaired. It is
//! factorized by Cholesky where that succeeds, and otherwise by its
//! eigendecomposition with the round-off negatives set to zero.
//!
//! # Seeds
//!
//! [`resonance_deviates`] keys the standard normals on `(seed, replica,
//! nuclide, range)` through `RESONANCE_PARAMETER_STREAM`, a stream of their
//! own, so they are reproducible and independent of the MF=33 cross-section
//! draws, which are keyed on the same nuclide without it.

use std::collections::BTreeMap;

use endf::resonance_covariance::{Location, Quantity, RangeCovariance};

use crate::covariance_sample::{eigen, name_ordinal, standard_normals};
use crate::nearest_correlation::{repaired, CorrelationRepair};

/// Keeps the resonance-parameter streams clear of every other per-nuclide
/// stream.
pub(crate) const RESONANCE_PARAMETER_STREAM: u32 = 0x2E50_7A2A;

/// How one parameter is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marginal {
    Gaussian,
    Lognormal,
    /// At its mean in every draw.
    Held,
}

/// A parameter the report names: its position in the range's parameter
/// list, what it is, and its stated mean and sigma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NotedParameter {
    pub index: usize,
    pub location: Location,
    pub quantity: Quantity,
    pub value: f64,
    pub sigma: f64,
}

/// What [`ResonanceSampler::new`] did with one range.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerReport {
    /// The range's isotope and range indices, as in [`RangeCovariance`].
    pub isotope: usize,
    pub range: usize,
    pub gaussian: usize,
    pub lognormal: usize,
    pub held: usize,
    /// Widths positive by nature, stated with a zero mean and a nonzero
    /// sigma: held at zero.
    pub zero_mean_widths: Vec<NotedParameter>,
    /// Widths positive by nature, stated with a negative mean: drawn
    /// Gaussian, as signed.
    pub negative_mean_widths: Vec<NotedParameter>,
    /// Parameters of zero variance with a nonzero covariance to another,
    /// which no covariance matrix has. Held at their mean; the covariances
    /// are dropped.
    pub zero_variance_with_covariance: usize,
    /// The repair of the stated correlation matrix, where it was not PSD.
    pub stated_repair: Option<CorrelationRepair>,
    /// Pairs of lognormal widths with `1 + C_ij / (μ_i μ_j) <= 0`.
    pub unattainable_pairs: usize,
    /// The repair of the transformed (log-space) correlation matrix, where
    /// it was not PSD.
    pub transformed_repair: Option<CorrelationRepair>,
}

/// One block of parameters the covariance couples, and its factor.
#[derive(Debug, Clone)]
struct Block {
    members: Vec<usize>,
    /// Row-major `m x m`, `L L^T = Σ` over `members`.
    factor: Vec<f64>,
    /// Whether `factor` is lower triangular (a Cholesky factor).
    triangular: bool,
}

/// A range's parameters, ready to draw.
#[derive(Debug, Clone)]
pub struct ResonanceSampler {
    marginals: Vec<Marginal>,
    /// The stated mean of every parameter.
    values: Vec<f64>,
    /// `m`, the Gaussian mean of `x`, per parameter (unused where held).
    location: Vec<f64>,
    blocks: Vec<Block>,
    dimension: usize,
}

/// Whether a quantity is a width that cannot be negative, in a range of
/// representation `lru`, `lrf`.
fn positive_by_nature(quantity: Quantity, lru: i64, lrf: i64) -> bool {
    match quantity {
        Quantity::NeutronWidth
        | Quantity::CaptureWidth
        | Quantity::CompetitiveWidth
        | Quantity::LevelSpacing
        | Quantity::ReducedNeutronWidth => true,
        // GF for Breit-Wigner and unresolved, GFA for Reich-Moore.
        Quantity::FissionWidth => lru == 2 || lrf != 3,
        // GFB, only ever Reich-Moore.
        Quantity::SecondFissionWidth => false,
        Quantity::Energy | Quantity::ScatteringRadius | Quantity::ChannelWidth(_) => false,
    }
}

impl ResonanceSampler {
    /// Classify every parameter, repair what is not PSD, and factorize.
    ///
    /// Errors where the matrix is not `n x n` for `n` parameters, or where a
    /// mean or covariance is not finite, or a variance is negative.
    pub fn new(range: &RangeCovariance) -> Result<(Self, SamplerReport), String> {
        let n = range.len();
        if range.covariance.len() != n * n {
            return Err(format!(
                "range {} of isotope {}: covariance has {} entries for {n} parameters",
                range.range,
                range.isotope,
                range.covariance.len()
            ));
        }
        if let Some(p) = range.parameters.iter().find(|p| !p.value.is_finite()) {
            return Err(format!(
                "range {} of isotope {}: {:?} {:?} has mean {}",
                range.range, range.isotope, p.location, p.quantity, p.value
            ));
        }
        if range.covariance.iter().any(|c| !c.is_finite()) {
            return Err(format!(
                "range {} of isotope {}: the covariance is not finite",
                range.range, range.isotope
            ));
        }
        let c = |i: usize, j: usize| range.covariance[i * n + j];
        let mut report = SamplerReport {
            isotope: range.isotope,
            range: range.range,
            gaussian: 0,
            lognormal: 0,
            held: 0,
            zero_mean_widths: Vec::new(),
            negative_mean_widths: Vec::new(),
            zero_variance_with_covariance: 0,
            stated_repair: None,
            unattainable_pairs: 0,
            transformed_repair: None,
        };
        let mut marginals = Vec::with_capacity(n);
        for (i, p) in range.parameters.iter().enumerate() {
            let variance = c(i, i);
            if variance < 0.0 {
                return Err(format!(
                    "range {} of isotope {}: {:?} {:?} has variance {variance}",
                    range.range, range.isotope, p.location, p.quantity
                ));
            }
            let note = NotedParameter {
                index: i,
                location: p.location,
                quantity: p.quantity,
                value: p.value,
                sigma: variance.sqrt(),
            };
            let marginal = if variance == 0.0 {
                if (0..n).any(|j| j != i && c(i, j) != 0.0) {
                    report.zero_variance_with_covariance += 1;
                }
                Marginal::Held
            } else if !positive_by_nature(p.quantity, range.lru, range.lrf) {
                Marginal::Gaussian
            } else if p.value > 0.0 {
                Marginal::Lognormal
            } else if p.value == 0.0 {
                report.zero_mean_widths.push(note);
                Marginal::Held
            } else {
                report.negative_mean_widths.push(note);
                Marginal::Gaussian
            };
            match marginal {
                Marginal::Gaussian => report.gaussian += 1,
                Marginal::Lognormal => report.lognormal += 1,
                Marginal::Held => report.held += 1,
            }
            marginals.push(marginal);
        }
        let values: Vec<f64> = range.parameters.iter().map(|p| p.value).collect();
        let mut location = values.clone();
        let mut blocks = Vec::new();
        for members in components(n, &marginals, &range.covariance) {
            let m = members.len();
            let sigma: Vec<f64> = members.iter().map(|&i| c(i, i).sqrt()).collect();
            // The stated correlation, repaired where it is not PSD.
            let mut stated = vec![0.0; m * m];
            for (a, &i) in members.iter().enumerate() {
                for (b, &j) in members.iter().enumerate() {
                    stated[a * m + b] = if a == b {
                        1.0
                    } else {
                        c(i, j) / (sigma[a] * sigma[b])
                    };
                }
            }
            if cholesky(&stated, m).is_none() {
                if let Some((repaired, record)) = repaired(&stated, m) {
                    report.stated_repair =
                        Some(CorrelationRepair::merge(report.stated_repair, record));
                    stated = repaired;
                }
            }
            // `Σ`, as its diagonal and correlation.
            let log_variance: Vec<f64> = members
                .iter()
                .zip(&sigma)
                .map(|(&i, s)| match marginals[i] {
                    Marginal::Lognormal => (s / values[i]).powi(2).ln_1p(),
                    _ => s * s,
                })
                .collect();
            let scale: Vec<f64> = log_variance.iter().map(|v| v.sqrt()).collect();
            let mut transformed = vec![0.0; m * m];
            for (a, &i) in members.iter().enumerate() {
                transformed[a * m + a] = 1.0;
                for (b, &j) in members.iter().enumerate().skip(a + 1) {
                    let cov = stated[a * m + b] * sigma[a] * sigma[b];
                    let entry = match (marginals[i], marginals[j]) {
                        (Marginal::Lognormal, Marginal::Lognormal) => {
                            let r = cov / (values[i] * values[j]);
                            if r > -1.0 {
                                r.ln_1p()
                            } else {
                                report.unattainable_pairs += 1;
                                -scale[a] * scale[b]
                            }
                        }
                        (Marginal::Lognormal, _) => cov / values[i],
                        (_, Marginal::Lognormal) => cov / values[j],
                        _ => cov,
                    };
                    let r = entry / (scale[a] * scale[b]);
                    transformed[a * m + b] = r;
                    transformed[b * m + a] = r;
                }
            }
            let mut factor = cholesky(&transformed, m);
            let triangular = factor.is_some();
            if factor.is_none() {
                if let Some((repaired, record)) = repaired(&transformed, m) {
                    report.transformed_repair =
                        Some(CorrelationRepair::merge(report.transformed_repair, record));
                    transformed = repaired;
                }
                factor = Some(eigen_factor(&transformed, m));
            }
            let mut factor = factor.expect("factorized above");
            for a in 0..m {
                for b in 0..m {
                    factor[a * m + b] *= scale[a];
                }
            }
            for (a, &i) in members.iter().enumerate() {
                if marginals[i] == Marginal::Lognormal {
                    location[i] = values[i].ln() - 0.5 * log_variance[a];
                }
            }
            blocks.push(Block {
                members,
                factor,
                triangular,
            });
        }
        let dimension = blocks.iter().map(|b| b.members.len()).sum();
        Ok((
            Self {
                marginals,
                values,
                location,
                blocks,
                dimension,
            },
            report,
        ))
    }

    /// The standard normals one draw takes: one per parameter that is not
    /// held.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// How each parameter is drawn, in the range's parameter order.
    pub fn marginals(&self) -> &[Marginal] {
        &self.marginals
    }

    /// The parameter vector for standard normals `z`, in the range's
    /// parameter order. `z` has [`Self::dimension`] entries.
    pub fn sample(&self, z: &[f64]) -> Vec<f64> {
        assert_eq!(
            z.len(),
            self.dimension,
            "a draw takes {} standard normals",
            self.dimension
        );
        let mut out = self.values.clone();
        let mut offset = 0;
        for block in &self.blocks {
            let m = block.members.len();
            let z = &z[offset..offset + m];
            for (a, &i) in block.members.iter().enumerate() {
                let width = if block.triangular { a + 1 } else { m };
                let row = &block.factor[a * m..a * m + width];
                let x = self.location[i] + row.iter().zip(z).map(|(l, z)| l * z).sum::<f64>();
                out[i] = match self.marginals[i] {
                    Marginal::Lognormal => x.exp(),
                    _ => x,
                };
            }
            offset += m;
        }
        out
    }

    /// One replica's parameter vector for this range: [`Self::sample`] of
    /// [`resonance_deviates`].
    pub fn draw(&self, seed: u64, replica: u64, nuclide: &str, range: usize) -> Vec<f64> {
        self.sample(&resonance_deviates(
            seed,
            replica,
            nuclide,
            range,
            self.dimension,
        ))
    }
}

/// `n` standard normals for one range of one nuclide in one replica, from a
/// stream keyed on `(seed, replica, nuclide, range)` alone. `range` is the
/// range's position in the list
/// [`range_covariances`](endf::resonance_covariance::range_covariances)
/// returns.
pub fn resonance_deviates(
    seed: u64,
    replica: u64,
    nuclide: &str,
    range: usize,
    n: usize,
) -> Vec<f64> {
    let replica_seed = yamc_rng::history_seed(seed, replica);
    let nuclide_seed = yamc_rng::secondary_seed(
        replica_seed,
        name_ordinal(nuclide) ^ RESONANCE_PARAMETER_STREAM,
    );
    let mut state = yamc_rng::expand_seed(yamc_rng::secondary_seed(nuclide_seed, range as u32));
    standard_normals(&mut state, n)
}

/// The parameters that are drawn, split into the blocks the covariance
/// couples, each in ascending order and the blocks in order of their first
/// member.
fn components(n: usize, marginals: &[Marginal], covariance: &[f64]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let drawn = |i: usize| marginals[i] != Marginal::Held;
    for i in (0..n).filter(|&i| drawn(i)) {
        for j in ((i + 1)..n).filter(|&j| drawn(j)) {
            if covariance[i * n + j] != 0.0 || covariance[j * n + i] != 0.0 {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in (0..n).filter(|&i| drawn(i)) {
        let r = root(&mut parent, i);
        blocks.entry(r).or_default().push(i);
    }
    blocks.into_values().collect()
}

/// The lower Cholesky factor of a symmetric `m x m` matrix, row-major, or
/// `None` where a pivot is not positive, which a positive definite matrix
/// never has.
fn cholesky(matrix: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * m + k] * l[j * m + k]).sum();
            let s = matrix[i * m + j] - dot;
            if i == j {
                if s.is_nan() || s <= 0.0 {
                    return None;
                }
                l[i * m + i] = s.sqrt();
            } else {
                l[i * m + j] = s / l[j * m + j];
            }
        }
    }
    Some(l)
}

/// `V sqrt(max(Λ, 0))` for a symmetric `m x m` matrix: a square root that
/// keeps a positive semi-definite matrix, with its round-off negatives at
/// zero.
fn eigen_factor(matrix: &[f64], m: usize) -> Vec<f64> {
    let (values, vectors) = eigen(matrix, m);
    let root: Vec<f64> = values.iter().map(|v| v.max(0.0).sqrt()).collect();
    let mut out = vectors;
    for a in 0..m {
        for (k, r) in root.iter().enumerate() {
            out[a * m + k] *= r;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance_sample::eigenvalues;
    use crate::nearest_correlation::tests::dykstra;
    use endf::resonance_covariance::{range_covariances, Parameter};

    /// A range of `parameters` `(quantity, mean, sigma)` with correlations
    /// `(i, j, ρ_ij)`.
    fn synthetic(
        lru: i64,
        lrf: i64,
        parameters: &[(Quantity, f64, f64)],
        correlations: &[(usize, usize, f64)],
    ) -> RangeCovariance {
        let n = parameters.len();
        let mut covariance = vec![0.0; n * n];
        for (i, p) in parameters.iter().enumerate() {
            covariance[i * n + i] = p.2 * p.2;
        }
        for &(i, j, rho) in correlations {
            let c = rho * parameters[i].2 * parameters[j].2;
            covariance[i * n + j] = c;
            covariance[j * n + i] = c;
        }
        RangeCovariance {
            isotope: 0,
            range: 0,
            mf2_range: 0,
            el: 1e-5,
            eh: 1e4,
            lru,
            lrf,
            parameters: parameters
                .iter()
                .enumerate()
                .map(|(index, &(quantity, value, _))| Parameter {
                    location: if quantity == Quantity::ScatteringRadius {
                        Location::Range
                    } else {
                        Location::Orbital { section: 0, index }
                    },
                    quantity,
                    value,
                })
                .collect(),
            covariance,
            approximate: 0,
            unmatched: Vec::new(),
            radius_steps: Vec::new(),
        }
    }

    impl ResonanceSampler {
        /// `Σ = L L^T` over every parameter (zero where held).
        fn sampled_log_covariance(&self) -> Vec<f64> {
            let n = self.values.len();
            let mut out = vec![0.0; n * n];
            for block in &self.blocks {
                let m = block.members.len();
                for (a, &i) in block.members.iter().enumerate() {
                    for (b, &j) in block.members.iter().enumerate() {
                        out[i * n + j] = (0..m)
                            .map(|k| block.factor[a * m + k] * block.factor[b * m + k])
                            .sum();
                    }
                }
            }
            out
        }
    }

    /// Mean and covariance of `draws` draws of `sampler`, on the stream,
    /// checked against the stated `means` and `covariance` (over the
    /// parameters `check`) to within `sigmas` standard errors of each
    /// estimate. The covariance is estimated about the stated mean, so each
    /// estimate is unbiased and its standard error is read off the draws.
    fn reproduces(
        sampler: &ResonanceSampler,
        means: &[f64],
        covariance: &[f64],
        check: &[usize],
        draws: u64,
        sigmas: f64,
    ) {
        let n = means.len();
        let k = check.len();
        let mut sum = vec![0.0; k];
        let mut sum_sq = vec![0.0; k];
        let mut product = vec![0.0; k * k];
        let mut product_sq = vec![0.0; k * k];
        for replica in 0..draws {
            let y = sampler.draw(11, replica, "Test", 0);
            let d: Vec<f64> = check.iter().map(|&i| y[i] - means[i]).collect();
            for a in 0..k {
                sum[a] += d[a];
                sum_sq[a] += d[a] * d[a];
                for b in a..k {
                    let q = d[a] * d[b];
                    product[a * k + b] += q;
                    product_sq[a * k + b] += q * q;
                }
            }
        }
        let count = draws as f64;
        for (a, &i) in check.iter().enumerate() {
            let mean = sum[a] / count;
            let error = (sum_sq[a] / count / count).sqrt();
            assert!(
                mean.abs() <= sigmas * error + 1e-14 * means[i].abs(),
                "parameter {i}: mean off by {mean} against a standard error {error}"
            );
            for (b, &j) in check.iter().enumerate().skip(a) {
                let c = product[a * k + b] / count;
                let variance = product_sq[a * k + b] / count - c * c;
                let error = (variance / count).sqrt();
                let stated = covariance[i * n + j];
                assert!(
                    (c - stated).abs() <= sigmas * error + 1e-14 * stated.abs(),
                    "C[{i},{j}] = {c} against {stated}, standard error {error}"
                );
            }
        }
    }

    /// The mixed case: a Gaussian energy, two lognormal widths at 50%
    /// relative sigma, a signed Reich-Moore fission width with a negative
    /// mean, and a radius parameter, all correlated.
    fn mixed() -> RangeCovariance {
        synthetic(
            1,
            3,
            &[
                (Quantity::Energy, 100.0, 0.5),
                (Quantity::NeutronWidth, 0.2, 0.1),
                (Quantity::CaptureWidth, 0.03, 0.015),
                (Quantity::FissionWidth, -0.01, 0.005),
                (Quantity::ScatteringRadius, 0.0, 1.0),
            ],
            &[
                (0, 1, 0.3),
                (0, 2, -0.2),
                (1, 2, 0.6),
                (1, 3, 0.4),
                (2, 3, 0.1),
                (0, 3, 0.1),
                (1, 4, 0.1),
            ],
        )
    }

    #[test]
    fn a_mixed_block_reproduces_its_mean_and_covariance() {
        let range = mixed();
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        assert_eq!(
            sampler.marginals(),
            [
                Marginal::Gaussian,
                Marginal::Lognormal,
                Marginal::Lognormal,
                Marginal::Gaussian,
                Marginal::Gaussian
            ]
        );
        assert_eq!((report.gaussian, report.lognormal, report.held), (3, 2, 0));
        assert_eq!(report.stated_repair, None);
        assert_eq!(report.transformed_repair, None);
        assert_eq!(report.unattainable_pairs, 0);
        // The Gaussian-space covariance is the transform of the stated one
        // exactly: lognormal pairs by ln(1 + C/(μμ)), mixed pairs by C/μ.
        let sampled = sampler.sampled_log_covariance();
        let sigma = |i: usize, j: usize| sampled[i * 5 + j];
        let c = |i: usize, j: usize| range.get(i, j);
        let mu = |i: usize| range.parameters[i].value;
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-13 * a.abs().max(b.abs());
        assert!(close(sigma(0, 1), c(0, 1) / mu(1)), "energy and width");
        assert!(
            close(sigma(1, 3), c(1, 3) / mu(1)),
            "width and signed width"
        );
        assert!(close(sigma(1, 4), c(1, 4) / mu(1)), "width and radius");
        assert!(close(sigma(1, 1), (c(1, 1) / (mu(1) * mu(1))).ln_1p()));
        assert!(close(sigma(1, 2), (c(1, 2) / (mu(1) * mu(2))).ln_1p()));
        assert!(close(sigma(0, 3), c(0, 3)));
        let means: Vec<f64> = range.parameters.iter().map(|p| p.value).collect();
        reproduces(
            &sampler,
            &means,
            &range.covariance,
            &[0, 1, 2, 3, 4],
            200_000,
            5.0,
        );
    }

    /// At 50% relative sigma a Gaussian width goes negative in about one
    /// draw in 44; the lognormal never does.
    #[test]
    fn widths_with_a_large_sigma_stay_positive() {
        let (sampler, _) = ResonanceSampler::new(&mixed()).unwrap();
        let mut gaussian_negative = 0;
        for replica in 0..200_000 {
            let z = resonance_deviates(3, replica, "W186", 1, sampler.dimension());
            let y = sampler.sample(&z);
            assert!(y[1] > 0.0 && y[2] > 0.0, "{y:?}");
            if 0.2 + 0.1 * z[1] < 0.0 {
                gaussian_negative += 1;
            }
        }
        assert!(gaussian_negative > 4000, "{gaussian_negative}");
    }

    /// Three energies whose rounded correlations leave the matrix just
    /// indefinite, at sigmas spanning three orders.
    fn indefinite() -> RangeCovariance {
        synthetic(
            1,
            3,
            &[
                (Quantity::Energy, 10.0, 0.1),
                (Quantity::Energy, 20.0, 2.0),
                (Quantity::Energy, 30.0, 30.0),
            ],
            &[(0, 1, 0.725), (0, 2, 0.725), (1, 2, 0.04)],
        )
    }

    #[test]
    fn an_indefinite_matrix_is_repaired_to_the_nearest_correlation_matrix() {
        let range = indefinite();
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        let repair = report.stated_repair.expect("repaired");
        // 1 + b/2 - sqrt(b^2/4 + 2a^2) for a = 0.725, b = 0.04.
        let lambda = 1.02 - (0.0004_f64 + 2.0 * 0.725 * 0.725).sqrt();
        assert!((repair.lambda_min - lambda).abs() < 1e-12, "{repair:?}");
        assert!(repair.converged && repair.parameters == 3, "{repair:?}");
        assert_eq!(report.transformed_repair, None);
        // Sigmas exact, the matrix PSD, and its correlation the one
        // Higham's own iteration converges to.
        let n = 3;
        let sigma = sampler.sampled_log_covariance();
        let s: Vec<f64> = (0..n).map(|i| range.get(i, i).sqrt()).collect();
        let mut stated = vec![0.0; n * n];
        let mut repaired = vec![0.0; n * n];
        for i in 0..n {
            assert!((sigma[i * n + i] / range.get(i, i) - 1.0).abs() < 1e-13);
            for j in 0..n {
                stated[i * n + j] = range.get(i, j) / (s[i] * s[j]);
                repaired[i * n + j] = sigma[i * n + j] / (s[i] * s[j]);
            }
        }
        let min = eigenvalues(&repaired, n)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        assert!(min >= -1e-14, "{min}");
        let reference = dykstra(&stated, n);
        for (a, b) in repaired.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-8, "{repaired:?} against {reference:?}");
        }
        let frobenius = stated
            .iter()
            .zip(&repaired)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let max = stated
            .iter()
            .zip(&repaired)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            (repair.frobenius_change - frobenius).abs() < 1e-12,
            "{repair:?}"
        );
        assert!((repair.max_change - max).abs() < 1e-12, "{repair:?}");
        assert!(
            repair.max_change > 1e-3 && repair.max_change < 1e-2,
            "{repair:?}"
        );
    }

    #[test]
    fn a_valid_gaussian_matrix_is_sampled_as_stated() {
        let range = synthetic(
            1,
            7,
            &[
                (Quantity::Energy, 10.0, 0.1),
                (Quantity::ChannelWidth(0), -0.5, 0.2),
                (Quantity::ChannelWidth(1), 0.0, 0.01),
            ],
            &[(0, 1, 0.5), (1, 2, -0.5)],
        );
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        assert_eq!(report.stated_repair, None);
        assert_eq!(report.transformed_repair, None);
        assert_eq!(sampler.marginals(), [Marginal::Gaussian; 3]);
        assert!(
            report.zero_mean_widths.is_empty(),
            "R-matrix widths are signed"
        );
        for (a, b) in sampler
            .sampled_log_covariance()
            .iter()
            .zip(&range.covariance)
        {
            assert!(
                (a - b).abs() <= 1e-15 * b.abs().max(1e-4),
                "{a} against {b}"
            );
        }
    }

    #[test]
    fn a_zero_mean_width_with_a_sigma_is_held_and_reported() {
        let range = synthetic(
            1,
            2,
            &[
                (Quantity::Energy, 10.0, 0.1),
                (Quantity::NeutronWidth, 0.0, 0.01),
                (Quantity::CaptureWidth, 0.05, 0.0),
                (Quantity::CaptureWidth, -0.05, 0.01),
            ],
            &[(0, 1, 0.5), (0, 3, 0.2)],
        );
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        assert_eq!(
            sampler.marginals(),
            [
                Marginal::Gaussian,
                Marginal::Held,
                Marginal::Held,
                Marginal::Gaussian
            ]
        );
        assert_eq!(report.held, 2);
        assert_eq!(report.zero_mean_widths.len(), 1);
        let held = report.zero_mean_widths[0];
        assert_eq!(
            (held.index, held.quantity, held.value, held.sigma),
            (1, Quantity::NeutronWidth, 0.0, 0.01)
        );
        assert_eq!(
            held.location,
            Location::Orbital {
                section: 0,
                index: 1
            }
        );
        assert_eq!(report.negative_mean_widths.len(), 1);
        assert_eq!(report.negative_mean_widths[0].index, 3);
        assert_eq!(sampler.dimension(), 2);
        for replica in 0..100 {
            let y = sampler.draw(5, replica, "Ta181", 0);
            assert_eq!((y[1], y[2]), (0.0, 0.05));
        }
    }

    #[test]
    fn a_zero_variance_with_a_covariance_is_held_and_counted() {
        let mut range = synthetic(
            1,
            3,
            &[
                (Quantity::Energy, 10.0, 0.1),
                (Quantity::NeutronWidth, 0.1, 0.0),
            ],
            &[],
        );
        range.covariance[1] = 1e-4;
        range.covariance[2] = 1e-4;
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        assert_eq!(report.zero_variance_with_covariance, 1);
        assert_eq!(sampler.marginals()[1], Marginal::Held);
    }

    #[test]
    fn a_malformed_range_is_an_error() {
        let mut range = mixed();
        range.covariance.pop();
        assert!(ResonanceSampler::new(&range).is_err());
        let mut range = mixed();
        range.covariance[0] = -1.0;
        assert!(ResonanceSampler::new(&range).is_err());
        let mut range = mixed();
        range.covariance[1] = f64::NAN;
        assert!(ResonanceSampler::new(&range).is_err());
    }

    #[test]
    fn an_anticorrelation_no_lognormal_carries_is_counted() {
        let range = synthetic(
            1,
            3,
            &[
                (Quantity::NeutronWidth, 0.1, 0.3),
                (Quantity::CaptureWidth, 0.1, 0.3),
            ],
            &[(0, 1, -0.5)],
        );
        // 1 + C/(μμ) = 1 - 0.5 * 9 < 0.
        let (sampler, report) = ResonanceSampler::new(&range).unwrap();
        assert_eq!(report.unattainable_pairs, 1);
        let y = sampler.draw(1, 2, "X", 0);
        assert!(y.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn the_draw_depends_on_seed_replica_nuclide_and_range_alone() {
        let (sampler, _) = ResonanceSampler::new(&mixed()).unwrap();
        let n = sampler.dimension();
        let a = resonance_deviates(7, 3, "W186", 0, n);
        assert_eq!(a, resonance_deviates(7, 3, "W186", 0, n));
        assert_eq!(sampler.draw(7, 3, "W186", 0), sampler.sample(&a));
        for other in [
            resonance_deviates(8, 3, "W186", 0, n),
            resonance_deviates(7, 4, "W186", 0, n),
            resonance_deviates(7, 3, "W184", 0, n),
            resonance_deviates(7, 3, "W186", 1, n),
        ] {
            assert!(a.iter().zip(&other).all(|(x, y)| x != y), "{a:?} {other:?}");
        }
        // Not the MF=33 stream of the same nuclide, which is keyed on the
        // bare name.
        let replica_seed = yamc_rng::history_seed(7, 3);
        let mut state =
            yamc_rng::expand_seed(yamc_rng::secondary_seed(replica_seed, name_ordinal("W186")));
        assert_ne!(standard_normals(&mut state, n), a);
    }

    fn fixture(compressed: &[u8]) -> Vec<RangeCovariance> {
        let mut raw = Vec::new();
        lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
        let text = String::from_utf8(raw).expect("ENDF is text");
        let material = endf::Material::from_str(&text).expect("evaluation parses");
        range_covariances(material.mf2().unwrap(), material.mf32().unwrap()).unwrap()
    }

    /// ENDF/B-VIII.1 Dy158's Reich-Moore range (LCOMP=1): four resonances,
    /// Gaussian energies and lognormal widths, PSD as written.
    #[test]
    fn a_real_range_reproduces_its_mean_and_covariance() {
        let ranges = fixture(include_bytes!(
            "../../endf/fixtures/n-066_Dy_158_mf2_mf32.endf.xz"
        ));
        let range = &ranges[0];
        let (sampler, report) = ResonanceSampler::new(range).unwrap();
        assert_eq!((report.gaussian, report.lognormal, report.held), (4, 8, 0));
        assert_eq!(report.stated_repair, None);
        assert_eq!(report.transformed_repair, None);
        let means: Vec<f64> = range.parameters.iter().map(|p| p.value).collect();
        let all: Vec<usize> = (0..range.len()).collect();
        reproduces(&sampler, &means, &range.covariance, &all, 200_000, 5.0);
    }

    /// ENDF/B-VIII.1 Rh103's unresolved range: unit-mean lognormal
    /// multipliers whose stated correlation is just indefinite. The repair
    /// keeps every mean and sigma, and moves no correlation by more than it
    /// reports.
    #[test]
    fn a_real_repaired_range_keeps_its_means_and_sigmas() {
        let ranges = fixture(include_bytes!(
            "../../endf/fixtures/n-045_Rh_103_mf2_mf32.endf.xz"
        ));
        let range = ranges.iter().find(|r| r.lru == 2).unwrap();
        let (sampler, report) = ResonanceSampler::new(range).unwrap();
        assert_eq!(report.lognormal, range.len());
        let repair = report.stated_repair.expect("repaired");
        assert!(repair.converged && repair.lambda_min < 0.0, "{repair:?}");
        let n = range.len();
        let means: Vec<f64> = range.parameters.iter().map(|p| p.value).collect();
        // Only the diagonal of the stated covariance is checked exactly.
        let mut diagonal = vec![0.0; n * n];
        for i in 0..n {
            diagonal[i * n + i] = range.get(i, i);
        }
        let drawn: Vec<usize> = (0..n).filter(|&i| range.get(i, i) > 0.0).collect();
        for &i in &drawn {
            reproduces(&sampler, &means, &diagonal, &[i], 50_000, 5.0);
        }
        let sigma = sampler.sampled_log_covariance();
        for &i in &drawn {
            for &j in &drawn {
                let stated = range.get(i, j) / (range.get(i, i) * range.get(j, j)).sqrt();
                let lognormal = sigma[i * n + j].exp_m1()
                    / (sigma[i * n + i].exp_m1() * sigma[j * n + j].exp_m1()).sqrt();
                assert!(
                    (lognormal - stated).abs() <= repair.max_change + 1e-12,
                    "{i} {j}: {lognormal} against {stated}"
                );
            }
        }
    }

    /// ENDF/B-VIII.1 W186 (compact R-matrix limited) is indefinite as
    /// written: its rounded correlations put the smallest eigenvalue of the
    /// 523 drawn parameters' correlation at -0.097.
    #[test]
    fn w186_is_repaired_keeping_every_sigma() {
        let ranges = fixture(include_bytes!(
            "../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz"
        ));
        let range = &ranges[0];
        let (sampler, report) = ResonanceSampler::new(range).unwrap();
        assert_eq!(report.gaussian, range.len());
        let repair = report.stated_repair.expect("repaired");
        assert!(repair.converged, "{repair:?}");
        assert_eq!(repair.parameters, 523);
        assert!((repair.lambda_min + 0.09724).abs() < 1e-4, "{repair:?}");
        let n = range.len();
        let sigma = sampler.sampled_log_covariance();
        for i in 0..n {
            let stated = range.get(i, i);
            assert!(
                (sigma[i * n + i] - stated).abs() <= 1e-12 * stated,
                "{i}: {} against {stated}",
                sigma[i * n + i]
            );
        }
    }
}
