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
//! Numer. Anal. 22 (2002) 329), found by Qi and Sun's Newton method. That
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

use crate::covariance_sample::{
    eigen, eigenvalues, name_ordinal, standard_normals, REPAIR_TOLERANCE,
};

/// Keeps the resonance-parameter streams clear of every other per-nuclide
/// stream.
pub(crate) const RESONANCE_PARAMETER_STREAM: u32 = 0x2E50_7A2A;

/// The nearest-correlation iteration stops when the PSD iterate's diagonal
/// is within this of one, as `||diag(X) - 1||_2 <= tolerance * sqrt(m)` on a
/// block of side `m`: a root-mean-square error per diagonal of 1e-10. The
/// result is PSD and has a unit diagonal exactly whatever the tolerance (the
/// iterate is scaled to one); the tolerance bounds how far it is from the
/// nearest such matrix, orders below the digits a tape writes correlations
/// to.
pub const NEAREST_CORRELATION_TOLERANCE: f64 = 1.0e-10;

/// The cap on Newton steps of the nearest-correlation iteration, each one
/// eigendecomposition. Convergence is quadratic: no MF=32 range of
/// ENDF/B-VIII.1, JEFF-4.0, JENDL-5.0, FENDL-3.2d or TENDL-2017 and 2025
/// takes more than ten. A block that hits
/// the cap is still repaired to a valid correlation matrix, only not the
/// nearest one, and [`CorrelationRepair::converged`] says so.
pub const NEAREST_CORRELATION_ITERATIONS: usize = 100;

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

/// What a nearest-correlation repair changed, over every block of a range
/// that needed one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationRepair {
    /// The most negative eigenvalue of the correlation matrix before repair.
    pub lambda_min: f64,
    /// `||R' - R||_F` over the whole correlation matrix, both triangles.
    pub frobenius_change: f64,
    /// The largest `|R'_ij - R_ij|`.
    pub max_change: f64,
    /// Parameters in the blocks that were repaired.
    pub parameters: usize,
    /// The most Newton steps any block took.
    pub iterations: usize,
    /// Whether every block met [`NEAREST_CORRELATION_TOLERANCE`] within
    /// [`NEAREST_CORRELATION_ITERATIONS`].
    pub converged: bool,
}

impl CorrelationRepair {
    fn merge(a: Option<Self>, b: Self) -> Self {
        match a {
            None => b,
            Some(a) => Self {
                lambda_min: a.lambda_min.min(b.lambda_min),
                frobenius_change: a.frobenius_change.hypot(b.frobenius_change),
                max_change: a.max_change.max(b.max_change),
                parameters: a.parameters + b.parameters,
                iterations: a.iterations.max(b.iterations),
                converged: a.converged && b.converged,
            },
        }
    }
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
    /// How far the correlations the draws actually have are from the
    /// evaluated ones: the largest `|ρ'_ij - ρ_ij|` over pairs of drawn
    /// parameters, `ρ` the stated correlation before any repair and `ρ'` the
    /// draw's own, in the parameters themselves rather than in log space.
    /// Every drawn parameter keeps its stated mean and sigma exactly (both
    /// repairs keep a unit diagonal), so this and
    /// [`SamplerReport::correlation_frobenius_change`] are the whole of the
    /// difference in the first two moments; held parameters are in `held`.
    /// A pair no lognormal carries ([`SamplerReport::unattainable_pairs`]) is
    /// where it is largest: ENDF/B-VIII.1 Th232.
    pub correlation_change: f64,
    /// `||ρ' - ρ||_F` over the drawn parameters, both triangles.
    pub correlation_frobenius_change: f64,
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
            correlation_change: 0.0,
            correlation_frobenius_change: 0.0,
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
            // The correlation the draws have, against the stated one: `Σ'` is
            // `transformed` scaled back, and the moments of `y` follow from it
            // as in the module documentation, read backwards.
            let mut frobenius = report.correlation_frobenius_change.powi(2);
            for (a, &i) in members.iter().enumerate() {
                for (b, &j) in members.iter().enumerate().skip(a + 1) {
                    let log_cov = transformed[a * m + b] * scale[a] * scale[b];
                    let cov = match (marginals[i], marginals[j]) {
                        (Marginal::Lognormal, Marginal::Lognormal) => {
                            values[i] * values[j] * log_cov.exp_m1()
                        }
                        (Marginal::Lognormal, _) => values[i] * log_cov,
                        (_, Marginal::Lognormal) => values[j] * log_cov,
                        _ => log_cov,
                    };
                    let change = (cov - c(i, j)).abs() / (sigma[a] * sigma[b]);
                    report.correlation_change = report.correlation_change.max(change);
                    frobenius += 2.0 * change * change;
                }
            }
            report.correlation_frobenius_change = frobenius.sqrt();
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

/// The nearest correlation matrix to `correlation` (`m x m`, unit
/// diagonal, one Cholesky has failed on) and what moved, or `None` where it
/// is PSD within `m * REPAIR_TOLERANCE` and needs no repair.
fn repaired(correlation: &[f64], m: usize) -> Option<(Vec<f64>, CorrelationRepair)> {
    let lambda_min = eigenvalues(correlation, m)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    if lambda_min >= -(m as f64) * REPAIR_TOLERANCE {
        return None;
    }
    let (near, iterations, converged) = nearest_correlation(correlation, m);
    let mut frobenius = 0.0_f64;
    let mut max_change = 0.0_f64;
    for (a, b) in near.iter().zip(correlation) {
        let d = (a - b).abs();
        frobenius += d * d;
        max_change = max_change.max(d);
    }
    Some((
        near,
        CorrelationRepair {
            lambda_min,
            frobenius_change: frobenius.sqrt(),
            max_change,
            parameters: m,
            iterations,
            converged,
        },
    ))
}

/// The nearest correlation matrix to `a` (`m x m`, symmetric, unit
/// diagonal) in the Frobenius norm, by the quadratically convergent Newton
/// method of H. Qi and D. Sun (SIAM J. Matrix Anal. Appl. 28 (2006) 360), as
/// N. J. Higham's alternating projections (IMA J. Numer. Anal. 22 (2002)
/// 329) converge only linearly, and on a block of a thousand resonance
/// parameters each step is a full eigendecomposition.
///
/// The nearest correlation matrix is `X = (A + diag(y))_+`, the PSD part of
/// `A` shifted on the diagonal, for the `y` that gives `X` a unit diagonal.
/// That `y` minimizes the convex dual `θ(y) = ||(A + diag(y))_+||_F^2 / 2 -
/// Σ y_i`, whose gradient is `diag(X) - 1`, and Newton's method with an
/// Armijo line search finds it (Qi and Sun, algorithm 5.1). Each Newton step
/// is solved by conjugate gradients on the generalized Hessian, whose product
/// with a vector costs `O(m^2 k)` for `k` the smaller of the counts of
/// positive and of non-positive eigenvalues, so the eigendecompositions,
/// one per step and usually under ten, are the whole cost.
///
/// Stops when `||diag(X) - 1||_2 <= NEAREST_CORRELATION_TOLERANCE *
/// sqrt(m)`, or after [`NEAREST_CORRELATION_ITERATIONS`] steps. Returns `X`
/// scaled to a unit diagonal, which is PSD and a correlation matrix whether
/// or not the iteration converged, the Newton steps taken and whether it
/// converged.
fn nearest_correlation(a: &[f64], m: usize) -> (Vec<f64>, usize, bool) {
    let mut y = vec![0.0; m];
    let mut spectrum = Spectrum::new(a, &y, m);
    let mut iterations = 0;
    let mut converged = false;
    loop {
        let gradient = spectrum.gradient();
        let size = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if size <= NEAREST_CORRELATION_TOLERANCE * (m as f64).sqrt() {
            converged = true;
            break;
        }
        if iterations >= NEAREST_CORRELATION_ITERATIONS {
            break;
        }
        iterations += 1;
        let rhs: Vec<f64> = gradient.iter().map(|g| -g).collect();
        let direction = spectrum.newton_direction(&rhs, size);
        let slope: f64 = gradient.iter().zip(&direction).map(|(g, d)| g * d).sum();
        // Armijo backtracking. Near the solution the decrease in `θ` falls
        // below the round-off of `θ` itself (a sum of `m` squared
        // eigenvalues), so a step that shrinks the gradient is taken too.
        let mut step = 1.0;
        loop {
            let trial: Vec<f64> = y
                .iter()
                .zip(&direction)
                .map(|(y, d)| y + step * d)
                .collect();
            let next = Spectrum::new(a, &trial, m);
            let shrinks = || {
                let g = next.gradient();
                g.iter().map(|g| g * g).sum::<f64>().sqrt() < size
            };
            if next.theta <= spectrum.theta + 1.0e-4 * step * slope || shrinks() || step < 1.0e-10 {
                y = trial;
                spectrum = next;
                break;
            }
            step *= 0.5;
        }
    }
    let x = spectrum.positive_part();
    // `x` is PSD, and so is `D^-1/2 x D^-1/2`; a diagonal clipped to zero
    // leaves its parameter uncorrelated with the rest.
    let diagonal: Vec<f64> = (0..m).map(|i| x[i * m + i]).collect();
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            out[i * m + j] = if i == j {
                1.0
            } else if diagonal[i] > 0.0 && diagonal[j] > 0.0 {
                x[i * m + j] / (diagonal[i] * diagonal[j]).sqrt()
            } else {
                0.0
            };
        }
    }
    (out, iterations, converged)
}

/// The eigendecomposition of `A + diag(y)` at one Newton iterate.
struct Spectrum {
    m: usize,
    values: Vec<f64>,
    /// Row-major `m x m`, eigenvector `k` in column `k`.
    vectors: Vec<f64>,
    /// `θ(y)`.
    theta: f64,
}

impl Spectrum {
    fn new(a: &[f64], y: &[f64], m: usize) -> Self {
        let mut z = a.to_vec();
        for i in 0..m {
            z[i * m + i] += y[i];
        }
        let (values, vectors) = eigen(&z, m);
        let theta = 0.5
            * values
                .iter()
                .filter(|v| **v > 0.0)
                .map(|v| v * v)
                .sum::<f64>()
            - y.iter().sum::<f64>();
        Self {
            m,
            values,
            vectors,
            theta,
        }
    }

    /// `diag((A + diag(y))_+) - 1`.
    fn gradient(&self) -> Vec<f64> {
        let m = self.m;
        (0..m)
            .map(|i| {
                let row = &self.vectors[i * m..(i + 1) * m];
                row.iter()
                    .zip(&self.values)
                    .filter(|(_, l)| **l > 0.0)
                    .map(|(p, l)| l * p * p)
                    .sum::<f64>()
                    - 1.0
            })
            .collect()
    }

    /// `(A + diag(y))_+`, row-major.
    fn positive_part(&self) -> Vec<f64> {
        let m = self.m;
        let mut out = vec![0.0; m * m];
        let columns: Vec<(f64, Vec<f64>)> = (0..m)
            .filter(|&k| self.values[k] > 0.0)
            .map(|k| {
                let v = (0..m).map(|i| self.vectors[i * m + k]).collect();
                (self.values[k], v)
            })
            .collect();
        for i in 0..m {
            let row = &mut out[i * m..(i + 1) * m];
            for (lambda, v) in &columns {
                let scale = lambda * v[i];
                for (o, vj) in row.iter_mut().zip(v) {
                    *o += scale * vj;
                }
            }
        }
        out
    }

    /// The Newton step: conjugate gradients on `(V + εI) d = rhs`, `V` the
    /// generalized Hessian, preconditioned by its diagonal, to a residual of
    /// `min(0.1, size) * size` (Qi and Sun's forcing term, which keeps the
    /// convergence quadratic).
    fn newton_direction(&self, rhs: &[f64], size: f64) -> Vec<f64> {
        let m = self.m;
        let hessian = Hessian::new(self);
        let tolerance = size.min(0.1) * size;
        let mut d = vec![0.0; m];
        let mut r = rhs.to_vec();
        let mut z: Vec<f64> = r
            .iter()
            .zip(&hessian.diagonal)
            .map(|(r, p)| r / p)
            .collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        for _ in 0..NEWTON_CG_ITERATIONS {
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= tolerance {
                break;
            }
            let vp = hessian.times(&p);
            let pvp: f64 = p.iter().zip(&vp).map(|(a, b)| a * b).sum();
            if pvp <= 0.0 {
                break;
            }
            let alpha = rz / pvp;
            for i in 0..m {
                d[i] += alpha * p[i];
                r[i] -= alpha * vp[i];
            }
            for i in 0..m {
                z[i] = r[i] / hessian.diagonal[i];
            }
            let next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = next / rz;
            rz = next;
            for i in 0..m {
                p[i] = z[i] + beta * p[i];
            }
        }
        d
    }
}

/// The generalized Hessian of `θ` at one iterate: `V h = diag(P (Ω ∘ (P^T
/// diag(h) P)) P^T)`, `Ω` 1 between positive eigenvalues, 0 between
/// non-positive ones, and `λ_i / (λ_i - λ_j)` for `λ_i` positive and `λ_j`
/// not (Qi and Sun, section 5), plus `ε` on the diagonal.
///
/// `Ω` and `1 - Ω` are each zero off the rows and columns of one set of
/// eigenvalues, so products are taken over the smaller set `S`: with `Ω'`
/// whichever of the two is supported there, only the columns of `M = P^T
/// diag(h) P` in `S` are needed, `O(m^2 |S|)`. `V h` is `diag(P (Ω' ∘ M)
/// P^T)` when `Ω' = Ω`, and `h` less it when `Ω' = 1 - Ω`, since `diag(P M
/// P^T) = h` for orthogonal `P`.
struct Hessian<'a> {
    spectrum: &'a Spectrum,
    set: Vec<usize>,
    complement: bool,
    /// `Ω'_ij` for `j` the `c`-th member of `S`, at `[c * m + i]`, doubled
    /// where `i` is outside `S`: its mirror in `M`'s rows in `S` is not
    /// stored.
    weights: Vec<f64>,
    /// The diagonal of `V + εI`, the preconditioner.
    diagonal: Vec<f64>,
}

impl<'a> Hessian<'a> {
    fn new(spectrum: &'a Spectrum) -> Self {
        let m = spectrum.m;
        let values = &spectrum.values;
        let positive = values.iter().filter(|v| **v > 0.0).count();
        let complement = positive > m - positive;
        let set: Vec<usize> = (0..m)
            .filter(|&k| (values[k] > 0.0) != complement)
            .collect();
        let mut in_set = vec![false; m];
        for &k in &set {
            in_set[k] = true;
        }
        let mut weights = vec![0.0; set.len() * m];
        for (c, &j) in set.iter().enumerate() {
            for i in 0..m {
                let (li, lj) = (values[i], values[j]);
                let w = match (li > 0.0, lj > 0.0) {
                    (true, true) => 1.0,
                    (false, false) => 0.0,
                    (true, false) => li / (li - lj),
                    (false, true) => lj / (lj - li),
                };
                let w = if complement { 1.0 - w } else { w };
                weights[c * m + i] = if in_set[i] { w } else { 2.0 * w };
            }
        }
        // `V_rr = Σ_ij Ω_ij P_ri^2 P_rj^2`, by the same split.
        let diagonal = indexed(m, |r| {
            let row = &spectrum.vectors[r * m..(r + 1) * m];
            let part: f64 = set
                .iter()
                .enumerate()
                .map(|(c, &j)| {
                    let w = &weights[c * m..(c + 1) * m];
                    row[j] * row[j] * row.iter().zip(w).map(|(p, w)| p * p * w).sum::<f64>()
                })
                .sum();
            let v = if complement { 1.0 - part } else { part };
            v.max(0.0) + NEWTON_REGULARIZATION
        });
        Self {
            spectrum,
            set,
            complement,
            weights,
            diagonal,
        }
    }

    /// `(V + εI) h`.
    fn times(&self, h: &[f64]) -> Vec<f64> {
        let m = self.spectrum.m;
        let vectors = &self.spectrum.vectors;
        let s = self.set.len();
        // `Ω' ∘ M` over the columns in `S`, stored by column.
        let mt: Vec<Vec<f64>> = indexed(s, |c| {
            let j = self.set[c];
            let mut column = vec![0.0; m];
            for r in 0..m {
                let row = &vectors[r * m..(r + 1) * m];
                let scale = h[r] * row[j];
                if scale == 0.0 {
                    continue;
                }
                for (out, p) in column.iter_mut().zip(row) {
                    *out += scale * p;
                }
            }
            for (out, w) in column.iter_mut().zip(&self.weights[c * m..(c + 1) * m]) {
                *out *= w;
            }
            column
        });
        indexed(m, |r| {
            let row = &vectors[r * m..(r + 1) * m];
            let part: f64 = self
                .set
                .iter()
                .zip(&mt)
                .map(|(&j, column)| {
                    let k: f64 = row.iter().zip(column).map(|(p, g)| p * g).sum();
                    k * row[j]
                })
                .sum();
            let vh = if self.complement { h[r] - part } else { part };
            vh + NEWTON_REGULARIZATION * h[r]
        })
    }
}

/// `f(0), ..., f(n - 1)`, in parallel off wasm32. Each entry is computed on
/// its own in a fixed order, so the result is the same on any thread count.
fn indexed<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).collect()
    }
}

/// Conjugate-gradient iterations allowed per Newton step.
const NEWTON_CG_ITERATIONS: usize = 200;

/// The `ε` that keeps the Newton system positive definite where the
/// generalized Hessian is singular (Qi and Sun's perturbation).
const NEWTON_REGULARIZATION: f64 = 1.0e-10;

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Higham's alternating projections with Dykstra's correction, plain
    /// and run long: the reference the Newton method has to agree with.
    fn dykstra(a: &[f64], m: usize) -> Vec<f64> {
        let mut y = a.to_vec();
        let mut correction = vec![0.0; m * m];
        for _ in 0..20_000 {
            let r: Vec<f64> = y.iter().zip(&correction).map(|(y, d)| y - d).collect();
            let (values, vectors) = eigen(&r, m);
            let mut x = vec![0.0; m * m];
            for (k, &l) in values.iter().enumerate() {
                for i in 0..m {
                    for j in 0..m {
                        x[i * m + j] += l.max(0.0) * vectors[i * m + k] * vectors[j * m + k];
                    }
                }
            }
            for i in 0..m * m {
                correction[i] = x[i] - r[i];
            }
            y = x;
            for i in 0..m {
                y[i * m + i] = 1.0;
            }
        }
        y
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
        assert!(report.correlation_change < 1e-12, "{report:?}");
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
        // Held at the most negative log-space covariance, ln(1 + 9) apart:
        // C' = 0.01 (exp(-ln 10) - 1) = -0.009, a correlation of -0.1 where
        // the evaluation states -0.5.
        assert!(
            (report.correlation_change - 0.4).abs() < 1e-12,
            "{report:?}"
        );
        assert!(
            (report.correlation_frobenius_change - 0.4 * 2f64.sqrt()).abs() < 1e-12,
            "{report:?}"
        );
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
