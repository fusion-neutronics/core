//! The nearest correlation matrix, for repairing an evaluated covariance that
//! is not positive semi-definite without moving any of its sigmas.
//!
//! Evaluated covariances are often not PSD as written: correlations rounded
//! to the digits a tape carries, or blocks evaluated apart and assembled
//! after. A matrix that is not PSD has no square root, so it cannot be
//! sampled as it is. Clipping its negative eigenvalues is the usual repair,
//! and it is not neutral: with `C = C+ + C-` split by the sign of the
//! eigenvalues, `-C-` is PSD, so every diagonal of `C+` is at least the
//! evaluated one, and clipping only ever adds variance.
//!
//! Here the repair is on the correlation matrix `R = D^-1/2 C D^-1/2`
//! instead: it is replaced by the nearest correlation matrix in the Frobenius
//! norm (N. J. Higham, IMA J. Numer. Anal. 22 (2002) 329), the PSD matrix of
//! unit diagonal closest to it, and rescaled by the evaluated sigmas,
//! `C' = D^1/2 R' D^1/2`. The diagonal is held at one, so every variance is
//! the evaluated one exactly and only correlations move, as little as any PSD
//! matrix allows. What moved is in [`CorrelationRepair`].
//!
//! The problem separates over the blocks the matrix couples: the nearest
//! correlation matrix of a block-diagonal matrix is block diagonal, each block
//! the nearest one to its own. So `repaired` splits the matrix first and
//! repairs only the blocks that need it, which leaves every other block
//! exactly as stated and makes each Newton step's eigendecomposition the size
//! of one block rather than of the whole matrix.

use crate::covariance_sample::{coupled_blocks, eigen, eigenvalues, REPAIR_TOLERANCE};

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
/// takes more than ten, and no ENDF/B-VIII.1 MF=33 cell covariance tested
/// (O16's 1102 coupled cells among them) more than seven. A block that hits
/// the cap is still repaired to a valid correlation matrix, only not the
/// nearest one, and [`CorrelationRepair::converged`] says so.
pub const NEAREST_CORRELATION_ITERATIONS: usize = 100;

/// What a nearest-correlation repair changed, over every block of a matrix
/// (a resonance range's parameters, a nuclide's MF=33 cells) that needed one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationRepair {
    /// The most negative eigenvalue of the correlation matrix before repair.
    pub lambda_min: f64,
    /// `||R' - R||_F` over the whole correlation matrix, both triangles.
    pub frobenius_change: f64,
    /// The largest `|R'_ij - R_ij|`.
    pub max_change: f64,
    /// Parameters (or cells) in the blocks that were repaired.
    pub parameters: usize,
    /// The most Newton steps any block took.
    pub iterations: usize,
    /// Whether every block met [`NEAREST_CORRELATION_TOLERANCE`] within
    /// [`NEAREST_CORRELATION_ITERATIONS`].
    pub converged: bool,
}

impl CorrelationRepair {
    pub(crate) fn merge(a: Option<Self>, b: Self) -> Self {
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

/// The nearest correlation matrix to `correlation` (`m x m`, unit
/// diagonal) and what moved, or `None` where it is PSD within the threshold
/// and needs no repair. See [`repaired_with_minimum`].
pub(crate) fn repaired(correlation: &[f64], m: usize) -> Option<(Vec<f64>, CorrelationRepair)> {
    repaired_with_minimum(correlation, m).1
}

/// [`repaired`], with the smallest eigenvalue of `correlation` whether or
/// not it needed a repair (one where `m < 2`).
///
/// The matrix needs a repair where its smallest eigenvalue is below
/// `-m * REPAIR_TOLERANCE` (the threshold of [`REPAIR_TOLERANCE`], and for
/// the same reason). It is then repaired block by block over the cells it
/// couples: a block of side `k` whose smallest eigenvalue is below
/// `-k * REPAIR_TOLERANCE` is replaced by its nearest correlation matrix, and
/// every other block is kept bit for bit. The record's `lambda_min` is the
/// smallest eigenvalue of the whole matrix.
pub(crate) fn repaired_with_minimum(
    correlation: &[f64],
    m: usize,
) -> (f64, Option<(Vec<f64>, CorrelationRepair)>) {
    let blocks: Vec<(Vec<usize>, Vec<f64>, f64)> = coupled_blocks(correlation, m)
        .into_iter()
        .filter(|members| members.len() > 1)
        .map(|members| {
            let k = members.len();
            let block: Vec<f64> = if k == m {
                correlation.to_vec()
            } else {
                members
                    .iter()
                    .flat_map(|&i| members.iter().map(move |&j| correlation[i * m + j]))
                    .collect()
            };
            let low = eigenvalues(&block, k)
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            (members, block, low)
        })
        .collect();
    let lambda_min = blocks.iter().map(|b| b.2).fold(1.0, f64::min);
    if lambda_min >= -(m as f64) * REPAIR_TOLERANCE {
        return (lambda_min, None);
    }
    let mut out = correlation.to_vec();
    let mut record: Option<CorrelationRepair> = None;
    for (members, block, low) in blocks {
        let k = members.len();
        if low >= -(k as f64) * REPAIR_TOLERANCE {
            continue;
        }
        let (near, iterations, converged) = nearest_correlation(&block, k);
        let mut frobenius = 0.0_f64;
        let mut max_change = 0.0_f64;
        for (a, b) in near.iter().zip(&block) {
            let d = (a - b).abs();
            frobenius += d * d;
            max_change = max_change.max(d);
        }
        for (a, &i) in members.iter().enumerate() {
            for (b, &j) in members.iter().enumerate() {
                out[i * m + j] = near[a * k + b];
            }
        }
        record = Some(CorrelationRepair::merge(
            record,
            CorrelationRepair {
                lambda_min: low,
                frobenius_change: frobenius.sqrt(),
                max_change,
                parameters: k,
                iterations,
                converged,
            },
        ));
    }
    let record = record.expect("the block holding the smallest eigenvalue is past its threshold");
    (
        lambda_min,
        Some((
            out,
            CorrelationRepair {
                lambda_min,
                ..record
            },
        )),
    )
}

/// The nearest correlation matrix to `a` (`m x m`, symmetric, unit
/// diagonal) in the Frobenius norm, by the quadratically convergent Newton
/// method of H. Qi and D. Sun (SIAM J. Matrix Anal. Appl. 28 (2006) 360), as
/// N. J. Higham's alternating projections (IMA J. Numer. Anal. 22 (2002)
/// 329) converge only linearly, and on a block of a thousand resonance
/// parameters or MF=33 cells each step is a full eigendecomposition.
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
pub(crate) mod tests {
    use super::*;

    /// Higham's alternating projections with Dykstra's correction, plain
    /// and run long: the reference the Newton method has to agree with.
    pub(crate) fn dykstra(a: &[f64], m: usize) -> Vec<f64> {
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

    /// A block-diagonal matrix is repaired block by block: the indefinite
    /// block goes to the nearest correlation matrix Higham's iteration gives
    /// it, the valid one is kept to the bit, and nothing couples them.
    #[test]
    fn only_the_blocks_that_need_it_are_repaired() {
        // Cells 0 and 2 just indefinite with 4; 1 and 3 a valid pair.
        let (a, b) = (0.725, 0.04);
        let m = 5;
        let mut r = vec![0.0; m * m];
        for i in 0..m {
            r[i * m + i] = 1.0;
        }
        let mut set = |i: usize, j: usize, v: f64| {
            r[i * m + j] = v;
            r[j * m + i] = v;
        };
        set(0, 2, a);
        set(0, 4, a);
        set(2, 4, b);
        set(1, 3, 0.3);
        let (near, record) = repaired(&r, m).expect("repaired");
        let lambda = 1.0 + b / 2.0 - (b * b / 4.0 + 2.0 * a * a).sqrt();
        assert!((record.lambda_min - lambda).abs() < 1e-12, "{record:?}");
        assert_eq!(record.parameters, 3);
        assert!(record.converged);
        assert_eq!(near[m + 3].to_bits(), 0.3_f64.to_bits());
        assert_eq!(near[3 * m + 1].to_bits(), 0.3_f64.to_bits());
        for i in [0, 2, 4] {
            assert_eq!(near[i * m + i], 1.0);
            for j in [1, 3] {
                assert_eq!(near[i * m + j], 0.0);
            }
        }
        let block = [1.0, a, a, a, 1.0, b, a, b, 1.0];
        let reference = dykstra(&block, 3);
        for (x, &i) in [0, 2, 4].iter().enumerate() {
            for (y, &j) in [0, 2, 4].iter().enumerate() {
                assert!((near[i * m + j] - reference[x * 3 + y]).abs() < 1e-8);
            }
        }
        let min = eigenvalues(&near, m)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        assert!(min >= -1e-14, "{min}");

        // A matrix PSD to within the threshold is not repaired at all.
        let mut valid = r.clone();
        valid[2] = 0.5;
        valid[2 * m] = 0.5;
        assert_eq!(repaired(&valid, m), None);
    }
}
