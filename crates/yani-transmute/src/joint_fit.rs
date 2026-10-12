//! The admissible joint covariance closest to a stated one, for drawing
//! evaluations a cross-material block links (see [`crate::cross_material`]).
//!
//! Each member `a` of a group is set up in whitened coordinates `u_a` of the
//! interval averages its links name: `Cov(u_a) = I`, and the averages'
//! correlations are `F_a F_aᵀ`, `F_a` a `q_a × r_a` factor with unit rows.
//! A joint draw is a covariance `W` of every member's `u` together whose
//! diagonal blocks are `I`, so that every member keeps its own covariance
//! exactly, and which is positive semidefinite. Its off-diagonal blocks
//! `K_ab` give the averages' correlations `F_a K_ab F_bᵀ`, which the links
//! state at some entries `(p, q)` as `t_pq`.
//!
//! The fit is the `W` that minimises `Σ (F_a K_ab F_bᵀ - t)_pq²` over the
//! stated entries, in correlation units: the norm judges every stated
//! entry alike, whatever its variance, and a weak direction of a member's
//! own correlation counts in it only as much as the averages it carries
//! do, rather than the whole mismatch along it.
//!
//! The constraint is carried by a factorisation (S. Burer and R. D. C.
//! Monteiro, Math. Program. 95 (2003) 329): `W = Z Zᵀ` with `Z` square and
//! each member's rows `Z_a` orthonormal, `Z_a Z_aᵀ = I`, so every `W` the
//! search visits is admissible and stopping early costs only optimality.
//! The rows are reached through the polar factor `Z_a = (M_a M_aᵀ)^-1/2 M_a`
//! of an unconstrained `M_a`, and `M` is searched by L-BFGS from `W = I`.
//! With `Z` square the factorised problem has no spurious local minimum
//! (N. Boumal, V. Voroninski and A. S. Bandeira, NeurIPS 2016): it is the
//! convex fit, solved without an eigendecomposition of `W`.
//!
//! Every matrix product runs on one thread and every parallel sum over
//! fixed chunks added in order, so the fit is the same on any thread count.

use faer::{Accum, MatMut, MatRef, Par};

use crate::covariance_sample::{clear_upper_state, eigen};
use crate::nearest_correlation::indexed;

/// One member: its averages' unit-row correlation factor `F`, row-major
/// `q × r`.
pub(crate) struct Member {
    pub q: usize,
    pub r: usize,
    pub factor: Vec<f64>,
}

/// The stated correlations between members `a < b`, as `(p, q, t)`: `a`'s
/// average `p` with `b`'s `q` stated as `t`.
pub(crate) struct Pair {
    pub a: usize,
    pub b: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

/// The fitted factor and how the search ended.
pub(crate) struct Fit {
    /// Row-major `R × R`, `R = Σ r_a`, the members' rows in order: `W = Z Zᵀ`.
    pub z: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// L-BFGS iterations allowed. Every iterate is admissible, so a fit that
/// reaches the cap is still a valid joint covariance, only not the closest.
const FIT_ITERATIONS: usize = 3000;

/// Pairs of steps L-BFGS remembers.
const MEMORY: usize = 6;

/// The search stops when the objective, half the mean squared correlation
/// error over the stated entries, falls by less than `PROGRESS` over the
/// last `WINDOW` iterations: a root-mean-square error `ε` then moves by
/// less than `PROGRESS / ε` per `WINDOW` iterations, 1.4e-4 at
/// FENDL-3.2d's group (`ε` 7e-4), 4e-7 at ENDF/B-VIII.1's (`ε` 0.27).
const PROGRESS: f64 = 1.0e-7;
const WINDOW: usize = 100;

/// Or when the largest gradient component falls below this.
const GRADIENT: f64 = 1.0e-12;

/// `dst = a b`, `a` row-major `m × k` (or `k × m` read transposed), `b`
/// row-major `k × n` (or `n × k` read transposed).
#[allow(clippy::too_many_arguments)]
fn product(
    dst: &mut [f64],
    a: &[f64],
    a_t: bool,
    b: &[f64],
    b_t: bool,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            dst[..m * n].fill(0.0);
        }
        return;
    }
    let lhs = if a_t {
        MatRef::from_row_major_slice(a, k, m).transpose()
    } else {
        MatRef::from_row_major_slice(a, m, k)
    };
    let rhs = if b_t {
        MatRef::from_row_major_slice(b, n, k).transpose()
    } else {
        MatRef::from_row_major_slice(b, k, n)
    };
    let out = MatMut::from_row_major_slice_mut(&mut dst[..m * n], m, n);
    let accum = if accumulate {
        Accum::Add
    } else {
        Accum::Replace
    };
    faer::linalg::matmul::matmul(out, accum, lhs, rhs, 1.0, Par::Seq);
}

struct Problem<'a> {
    members: &'a [Member],
    pairs: &'a [Pair],
    offsets: Vec<usize>,
    total: usize,
    /// `1 / (2 × stated entries)`: the objective is half the mean squared
    /// correlation error over the entries.
    scale: f64,
}

/// One member's polar factor of `M_a`: `S = M Mᵀ = V diag(s) Vᵀ`.
struct Polar {
    s: Vec<f64>,
    v: Vec<f64>,
    /// `S^-1/2`.
    root: Vec<f64>,
}

impl Problem<'_> {
    fn rows(&self, a: usize) -> std::ops::Range<usize> {
        self.offsets[a] * self.total..self.offsets[a + 1] * self.total
    }

    /// `Z_a = (M_a M_aᵀ)^-1/2 M_a` for every member.
    fn polar(&self, m: &[f64]) -> (Vec<f64>, Vec<Polar>) {
        let n = self.total;
        let parts = indexed(self.members.len(), |a| {
            let r = self.members[a].r;
            let ma = &m[self.rows(a)];
            let mut s = vec![0.0; r * r];
            product(&mut s, ma, false, ma, true, r, n, r, false);
            let (mut values, v) = eigen(&s, r);
            let top = values.iter().cloned().fold(0.0, f64::max);
            for x in &mut values {
                *x = x.max(1.0e-14 * top).max(f64::MIN_POSITIVE);
            }
            let mut scaled = v.clone();
            for i in 0..r {
                for j in 0..r {
                    scaled[i * r + j] /= values[j].sqrt();
                }
            }
            let mut root = vec![0.0; r * r];
            product(&mut root, &scaled, false, &v, true, r, r, r, false);
            let mut za = vec![0.0; r * n];
            product(&mut za, &root, false, ma, false, r, r, n, false);
            clear_upper_state();
            (za, Polar { s: values, v, root })
        });
        let mut z = Vec::with_capacity(n * n);
        let mut polar = Vec::with_capacity(parts.len());
        for (za, p) in parts {
            z.extend_from_slice(&za);
            polar.push(p);
        }
        (z, polar)
    }

    /// Per pair `(f, G)`, `f` its share of the objective and `G = ∂f/∂K`,
    /// `r_a × r_b`, at `z`.
    fn pairs(&self, z: &[f64]) -> Vec<(f64, Vec<f64>)> {
        let n = self.total;
        indexed(self.pairs.len(), |i| {
            let pair = &self.pairs[i];
            let (ma, mb) = (&self.members[pair.a], &self.members[pair.b]);
            let (ra, rb, qa, qb) = (ma.r, mb.r, ma.q, mb.q);
            let mut k = vec![0.0; ra * rb];
            product(
                &mut k,
                &z[self.rows(pair.a)],
                false,
                &z[self.rows(pair.b)],
                true,
                ra,
                n,
                rb,
                false,
            );
            // `F_a K`, `q_a × r_b`.
            let mut fk = vec![0.0; qa * rb];
            product(&mut fk, &ma.factor, false, &k, false, qa, ra, rb, false);
            let mut e = vec![0.0; qa * qb];
            let mut f = 0.0;
            for &(p, q, t) in &pair.entries {
                let x: f64 = fk[p * rb..(p + 1) * rb]
                    .iter()
                    .zip(&mb.factor[q * rb..(q + 1) * rb])
                    .map(|(a, b)| a * b)
                    .sum();
                let d = x - t;
                f += d * d;
                e[p * qb + q] += 2.0 * d * self.scale;
            }
            // `G = F_aᵀ E F_b`.
            let mut ef = vec![0.0; qa * rb];
            product(&mut ef, &e, false, &mb.factor, false, qa, qb, rb, false);
            let mut g = vec![0.0; ra * rb];
            product(&mut g, &ma.factor, true, &ef, false, ra, qa, rb, false);
            clear_upper_state();
            (f * self.scale, g)
        })
    }

    /// The objective and its gradient in `M`.
    fn evaluate(&self, m: &[f64]) -> (f64, Vec<f64>) {
        let n = self.total;
        let (z, polar) = self.polar(m);
        let per_pair = self.pairs(&z);
        let f: f64 = per_pair.iter().map(|(f, _)| f).sum();
        let grads = indexed(self.members.len(), |a| {
            let r = self.members[a].r;
            // `∂f/∂Z_a = Σ_b G_ab Z_b`, over the pairs in order.
            let mut gz = vec![0.0; r * n];
            for (pair, (_, g)) in self.pairs.iter().zip(&per_pair) {
                if pair.a == a {
                    let rb = self.members[pair.b].r;
                    product(
                        &mut gz,
                        g,
                        false,
                        &z[self.rows(pair.b)],
                        false,
                        r,
                        rb,
                        n,
                        true,
                    );
                } else if pair.b == a {
                    let ra = self.members[pair.a].r;
                    product(
                        &mut gz,
                        g,
                        true,
                        &z[self.rows(pair.a)],
                        false,
                        r,
                        ra,
                        n,
                        true,
                    );
                }
            }
            // Through the polar factor: `∇M = S^-1/2 G + 2 B M`, with
            // `B = V (Γ ∘ sym(Vᵀ G Mᵀ V)) Vᵀ` and
            // `Γ_ij = -1 / (√s_i √s_j (√s_i + √s_j))`.
            let Polar { s, v, root } = &polar[a];
            let ma = &m[self.rows(a)];
            let mut gm = vec![0.0; r * r];
            product(&mut gm, &gz, false, ma, true, r, n, r, false);
            let mut t = vec![0.0; r * r];
            product(&mut t, v, true, &gm, false, r, r, r, false);
            let mut a_ = vec![0.0; r * r];
            product(&mut a_, &t, false, v, false, r, r, r, false);
            let roots: Vec<f64> = s.iter().map(|x| x.sqrt()).collect();
            let mut h = vec![0.0; r * r];
            for i in 0..r {
                for j in 0..r {
                    let sym = 0.5 * (a_[i * r + j] + a_[j * r + i]);
                    h[i * r + j] = -sym / (roots[i] * roots[j] * (roots[i] + roots[j]));
                }
            }
            product(&mut t, v, false, &h, false, r, r, r, false);
            let mut b = vec![0.0; r * r];
            product(&mut b, &t, false, v, true, r, r, r, false);
            for x in &mut b {
                *x *= 2.0;
            }
            let mut out = vec![0.0; r * n];
            product(&mut out, root, false, &gz, false, r, r, n, false);
            product(&mut out, &b, false, ma, false, r, r, n, true);
            clear_upper_state();
            out
        });
        (f, grads.concat())
    }
}

/// Elements per chunk of the vector operations: a fixed size, so how a sum
/// is split, and so how it rounds, depends on the length alone.
const CHUNK: usize = 1 << 13;

/// `Σ a_i b_i`, chunk by chunk in parallel off wasm32, the chunks' sums
/// added in order.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    indexed(a.len().div_ceil(CHUNK), |k| {
        let range = k * CHUNK..((k + 1) * CHUNK).min(a.len());
        a[range.clone()]
            .iter()
            .zip(&b[range])
            .map(|(x, y)| x * y)
            .sum::<f64>()
    })
    .into_iter()
    .sum()
}

/// `y = s y + t x`, chunk by chunk in parallel off wasm32.
fn update(y: &mut [f64], s: f64, t: f64, x: &[f64]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        y.par_chunks_mut(CHUNK)
            .zip(x.par_chunks(CHUNK))
            .for_each(|(y, x)| {
                for (a, b) in y.iter_mut().zip(x) {
                    *a = s * *a + t * b;
                }
            });
    }
    #[cfg(target_arch = "wasm32")]
    for (a, b) in y.iter_mut().zip(x) {
        *a = s * *a + t * b;
    }
}

/// The admissible joint covariance of `members` closest to the stated
/// correlations `pairs` (see the module documentation).
///
/// L-BFGS with backtracking to the Armijo condition, from `W = I`. It stops
/// when the objective falls by less than [`PROGRESS`] over [`WINDOW`]
/// iterations, or after [`FIT_ITERATIONS`].
pub(crate) fn best_fit(members: &[Member], pairs: &[Pair]) -> Fit {
    let mut offsets = vec![0];
    for m in members {
        offsets.push(offsets.last().unwrap() + m.r);
    }
    let total = *offsets.last().unwrap();
    let entries: usize = pairs.iter().map(|p| p.entries.len()).sum();
    let problem = Problem {
        members,
        pairs,
        offsets,
        total,
        scale: 0.5 / entries.max(1) as f64,
    };
    let n = total * total;
    let mut x = vec![0.0; n];
    for i in 0..total {
        x[i * total + i] = 1.0;
    }
    let (mut f, mut g) = problem.evaluate(&x);
    // `(s, y, 1 / sᵀy)` of the last steps.
    let mut history: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> =
        std::collections::VecDeque::new();
    let mut objectives = vec![f];
    let (mut iterations, mut converged) = (0, false);
    let mut d = vec![0.0; n];
    let mut trial = vec![0.0; n];
    while iterations < FIT_ITERATIONS {
        if f == 0.0 || g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) <= GRADIENT {
            converged = true;
            break;
        }
        iterations += 1;
        // The two-loop recursion: `d = -H g`.
        update(&mut d, 0.0, -1.0, &g);
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * dot(s, &d);
            update(&mut d, 1.0, -alpha, y);
            alphas.push(alpha);
        }
        let gamma = match history.back() {
            Some((s, y, _)) => dot(s, y) / dot(y, y),
            None => 1.0 / dot(&g, &g).sqrt(),
        };
        update(&mut d, gamma, 0.0, &g);
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &d);
            update(&mut d, 1.0, alpha - beta, s);
        }
        let mut slope = dot(&g, &d);
        if slope >= 0.0 {
            history.clear();
            update(&mut d, 0.0, -1.0 / dot(&g, &g).sqrt(), &g);
            slope = dot(&g, &d);
        }
        // Backtracking to the Armijo condition.
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            trial.copy_from_slice(&x);
            update(&mut trial, 1.0, step, &d);
            let (ft, gt) = problem.evaluate(&trial);
            if ft <= f + 1.0e-4 * step * slope {
                accepted = Some((ft, gt));
                break;
            }
            step *= 0.5;
        }
        let Some((ft, gt)) = accepted else {
            converged = true;
            break;
        };
        // `s = trial - x` and `y = gt - g`, reusing the oldest pair's storage.
        let (mut s, mut y) = if history.len() == MEMORY {
            let (s, y, _) = history.pop_front().expect("full");
            (s, y)
        } else {
            (vec![0.0; n], vec![0.0; n])
        };
        s.copy_from_slice(&trial);
        update(&mut s, 1.0, -1.0, &x);
        y.copy_from_slice(&gt);
        update(&mut y, 1.0, -1.0, &g);
        let sy = dot(&s, &y);
        if sy > 1.0e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            history.push_back((s, y, 1.0 / sy));
        }
        std::mem::swap(&mut x, &mut trial);
        g = gt;
        f = ft;
        objectives.push(f);
        if iterations >= WINDOW && objectives[iterations - WINDOW] - f <= PROGRESS {
            converged = true;
            break;
        }
    }
    let (z, _) = problem.polar(&x);
    Fit {
        z,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(q: usize) -> Member {
        let mut factor = vec![0.0; q * q];
        for i in 0..q {
            factor[i * q + i] = 1.0;
        }
        Member { q, r: q, factor }
    }

    fn w(fit: &Fit, total: usize) -> Vec<f64> {
        let mut out = vec![0.0; total * total];
        product(
            &mut out, &fit.z, false, &fit.z, true, total, total, total, false,
        );
        out
    }

    /// A stated correlation within one is reproduced, one past it is drawn
    /// at one, and each member's own block stays the identity.
    #[test]
    fn two_independent_directions_are_fitted_apart() {
        let members = [unit(2), unit(2)];
        let pairs = [Pair {
            a: 0,
            b: 1,
            entries: vec![(0, 0, 1.0), (1, 1, 0.5), (0, 1, 0.0), (1, 0, 0.0)],
        }];
        let fit = best_fit(&members, &pairs);
        assert!(fit.converged);
        let w = w(&fit, 4);
        for (i, j, want) in [
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 3, 1.0),
            (0, 1, 0.0),
        ] {
            assert!((w[i * 4 + j] - want).abs() < 1e-12, "{i} {j} {w:?}");
        }
        assert!((w[2] - 1.0).abs() < 1e-5, "{w:?}");
        assert!((w[7] - 0.5).abs() < 1e-5, "{w:?}");
    }

    /// Three members each stated fully correlated with the next but the
    /// first anticorrelated with the third cannot all hold: the fit is the
    /// closest correlation matrix, `±1/2` (with the off-diagonal error
    /// spread evenly), and stays admissible.
    #[test]
    fn an_inconsistent_triangle_is_spread_evenly() {
        let members = [unit(1), unit(1), unit(1)];
        let pairs = [
            Pair {
                a: 0,
                b: 1,
                entries: vec![(0, 0, 1.0)],
            },
            Pair {
                a: 1,
                b: 2,
                entries: vec![(0, 0, 1.0)],
            },
            Pair {
                a: 0,
                b: 2,
                entries: vec![(0, 0, -1.0)],
            },
        ];
        let fit = best_fit(&members, &pairs);
        let w = w(&fit, 3);
        for i in 0..3 {
            assert!((w[i * 4] - 1.0).abs() < 1e-12);
        }
        assert!((w[1] - 0.5).abs() < 1e-4, "{w:?}");
        assert!((w[5] - 0.5).abs() < 1e-4, "{w:?}");
        assert!((w[2] + 0.5).abs() < 1e-4, "{w:?}");
    }

    /// With every entry stated on unit factors the fit is the nearest
    /// contraction in the Frobenius norm, which has a closed form: the
    /// stated block with its singular values clipped to one. Here
    /// `T = U diag(1.5, 0.3) Vᵀ` must come back as `U diag(1, 0.3) Vᵀ`.
    #[test]
    fn two_members_fit_the_clipped_singular_values() {
        let (c, s) = (0.6_f64, 0.8_f64);
        let u = [[c, -s], [s, c]];
        let v = [[1.0, 0.0], [0.0, 1.0]];
        let block = |d: [f64; 2]| {
            let mut t = [[0.0; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    t[i][j] = (0..2).map(|k| u[i][k] * d[k] * v[j][k]).sum();
                }
            }
            t
        };
        let stated = block([1.5, 0.3]);
        let want = block([1.0, 0.3]);
        let entries = (0..2)
            .flat_map(|p| (0..2).map(move |q| (p, q)))
            .map(|(p, q)| (p, q, stated[p][q]))
            .collect();
        let fit = best_fit(
            &[unit(2), unit(2)],
            &[Pair {
                a: 0,
                b: 1,
                entries,
            }],
        );
        let w = w(&fit, 4);
        for i in 0..2 {
            for j in 0..2 {
                assert!((w[i * 4 + 2 + j] - want[i][j]).abs() < 1e-4, "{w:?}");
            }
        }
    }

    /// The fit is the same on one thread as on several, to the bit.
    #[test]
    fn the_fit_does_not_depend_on_the_thread_count() {
        let members = [unit(3), unit(2), unit(3)];
        let pairs = [
            Pair {
                a: 0,
                b: 1,
                entries: vec![(0, 0, 0.9), (1, 1, -0.7), (2, 0, 0.4)],
            },
            Pair {
                a: 1,
                b: 2,
                entries: vec![(0, 0, 0.95), (1, 2, 0.6)],
            },
            Pair {
                a: 0,
                b: 2,
                entries: vec![(0, 0, -0.8), (2, 1, 0.3)],
            },
        ];
        let on = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| best_fit(&members, &pairs).z)
        };
        let one = on(1);
        assert_eq!(one, on(4));
        assert_eq!(one, on(7));
    }
}
