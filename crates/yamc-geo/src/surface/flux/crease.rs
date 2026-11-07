//! The crease-free distance of a boundary: how far outwards its offset
//! layers stay smooth, for the warning about creased layers.
//!
//! The layer at outward distance `d` is smooth while `d` is below the outer
//! reach: the distance to the nearest point outside that is a focal point
//! of the boundary or has two closest boundary points, where offset fronts
//! from two parts of the boundary meet. That is the smaller of
//!
//! - the tightest concave radius of curvature (the focal distance), and
//! - half the shortest outward normal chord, the distance along the outward
//!   normal from a boundary point to where it next meets the boundary.
//!   Past the middle of a chord its points are nearer its far end than its
//!   start, so the reach is at most half of it; and where fronts meet
//!   first, the nearest such point is the middle of a chord joining its two
//!   closest points, normal to the boundary at both ends.
//!
//! More generally, a boundary point `q` on the outer side of the tangent
//! plane at `p` bounds the reach by `|q - p|^2 / (2 n(p) . (q - p))`, the
//! radius of the ball touching the boundary at `p` with `q` on its sphere:
//! points of the normal from `p` beyond it are nearer `q` than `p`. For a
//! chord that is half its length.
//!
//! The engine's `reach_out` bounds the same distance from below for the ray
//! march, with a factor of 1.5 on sampled curvatures and box bounds on the
//! gaps, so it can be a third too small to warn by (far more beside a thin
//! part). Here the concave curvature is sampled on a grid over one field
//! period (one cut of an axisymmetric boundary) and the tightest samples
//! refined by a compass search. Chords are sampled on the same grid with
//! the certified ray march, and the shortest refined by Newton's method
//! towards the nearby chord normal to the boundary at both ends, where its
//! length is stationary: a few steps and no more rays, along any valley of
//! the length. Every value kept is the curvature at a boundary point or the
//! radius of a ball as above, so the distance is never too small; where the
//! grid resolves the shape it is exact to round-off (a circular torus gives
//! `R0 - a` to 1e-15). A concave dimple or a fold narrower than the grid
//! spacing (at least 48 samples round poloidally, twice as many for the
//! curvature), or a gap that only the normals of a band narrower than that
//! reach across (about `a / R0` rad round the hole of a torus), can be
//! missed, which makes the distance too large.

use super::patches::principal_curvatures;
use super::series::SurfacePoint;
use super::Engine;
use std::f64::consts::TAU;

/// Grid samples refined, at most: the smallest, each more than `APART`
/// cells from those refined before it and not equal to one of them (a
/// mirror image under a symmetry of the boundary).
const STARTS: usize = 6;
const APART: usize = 2;
/// The curvature's compass search stops at this step (rad).
const STEP_MIN: f64 = 1e-8;
/// Moves the compass search makes, at most.
const MOVES: usize = 1000;
/// Chords are followed up to this many times twice the tightest concave
/// radius of curvature: half a longer one cannot be the distance, and the
/// grid sample nearest a shorter one is longer by the grid's error (about
/// 2 % across the hole of a torus of aspect ratio 4).
const CHORD_REACH: f64 = 1.2;
/// Newton steps refining a chord, at most; it stops sooner once a step is
/// below `NEWTON_STEP_MIN` (rad).
const NEWTON_STEPS: usize = 50;
const NEWTON_STEP_MIN: f64 = 1e-12;

/// Samples `nt x nz` over `theta` in `[0, 2 pi)` and one field period of
/// `zeta`, `ht` and `hz` apart (`hz` 0 when nothing depends on `zeta`).
struct Grid {
    nt: usize,
    nz: usize,
    ht: f64,
    hz: f64,
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

impl Engine {
    /// The crease-free distance (cm), computed on first use.
    pub(super) fn crease_free_distance(&self) -> f64 {
        *self.crease_free.get_or_init(|| {
            let (radius, half_chord) = self.crease_free_parts();
            radius.min(half_chord)
        })
    }

    /// The tightest concave radius of curvature and half the shortest
    /// outward normal chord (infinite if none is shorter than
    /// `CHORD_REACH` times twice that radius), in cm.
    pub(super) fn crease_free_parts(&self) -> (f64, f64) {
        // The patch grid over a field period, `scale` times finer; a
        // boundary without toroidal modes is the same in every cut.
        let axisymmetric = self.series.n_max == 0;
        let grid = |scale: usize| {
            let nt = scale * self.patches.nt;
            let nz = scale * self.patches.nz / self.series.nfp as usize;
            if axisymmetric {
                Grid {
                    nt,
                    nz: 1,
                    ht: TAU / nt as f64,
                    hz: 0.0,
                }
            } else {
                Grid {
                    nt,
                    nz,
                    ht: TAU / nt as f64,
                    hz: TAU / (self.series.nfp * nz as f64),
                }
            }
        };
        // The concave curvature is minus the smaller principal curvature.
        let fine = grid(2);
        let curvature = |sp: &SurfacePoint| principal_curvatures(sp, self.orientation_3d).1;
        let at = |theta: f64, zeta: f64| curvature(&self.series.point(theta, zeta));
        let kappa = -self
            .starts(curvature, &fine)
            .into_iter()
            .map(|start| descend(&at, start, &fine))
            .fold(f64::INFINITY, f64::min);
        let radius = if kappa > 0.0 {
            1.0 / kappa
        } else {
            f64::INFINITY
        };
        let t_max = CHORD_REACH * 2.0 * radius;
        let chord = |sp: &SurfacePoint| {
            let normal = self.normal(sp);
            // A chord is at least twice the outer reach, which `reach_out`
            // bounds from below, so it needs a patch box on the normal
            // beyond `reach_out`.
            if !self
                .patches
                .segment_meets(sp.x, normal, self.reach_out, t_max)
            {
                return f64::INFINITY;
            }
            self.trace_ray(sp.x, normal, 0.0, Some(0.0), true, t_max)
                .crossing
                .unwrap_or(f64::INFINITY)
        };
        let half_chord = self
            .starts(chord, &grid(1))
            .into_iter()
            .map(|(t, theta, zeta)| self.refine_chord(t, theta, zeta))
            .fold(f64::INFINITY, f64::min);
        (radius, half_chord)
    }

    /// The smallest finite samples of `f` on the grid to refine, as
    /// `(value, theta, zeta)`.
    fn starts(&self, f: impl Fn(&SurfacePoint) -> f64, grid: &Grid) -> Vec<(f64, f64, f64)> {
        let mut samples = Vec::with_capacity(grid.nt * grid.nz);
        for j in 0..grid.nz {
            let zeta = j as f64 * grid.hz;
            let cut = self.series.collapse(zeta, 3);
            for i in 0..grid.nt {
                let v = f(&cut.section(i as f64 * grid.ht).cartesian(zeta));
                if v.is_finite() {
                    samples.push((v, i, j));
                }
            }
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cells = |a: usize, b: usize, n: usize| {
            let d = a.abs_diff(b);
            d.min(n - d)
        };
        let mut starts: Vec<(f64, usize, usize)> = Vec::with_capacity(STARTS);
        for s in samples {
            if starts.len() == STARTS {
                break;
            }
            if starts.iter().all(|p| {
                (cells(s.1, p.1, grid.nt) > APART || cells(s.2, p.2, grid.nz) > APART)
                    && (s.0 - p.0).abs() > 1e-12 * p.0.abs()
            }) {
                starts.push(s);
            }
        }
        starts
            .into_iter()
            .map(|(v, i, j)| (v, i as f64 * grid.ht, j as f64 * grid.hz))
            .collect()
    }

    /// Half the chord of length `t` along the outward normal at `(theta,
    /// zeta)`, refined: Newton's method moves both ends towards the nearby
    /// chord that is normal to the boundary at both ends, where
    /// `|q - p|^2 / 2` is stationary, and the smallest ball radius
    /// `|q - p|^2 / (2 n(p) . (q - p))` met on the way is kept (`t / 2` at
    /// the start, half the chord at the end).
    fn refine_chord(&self, t: f64, theta: f64, zeta: f64) -> f64 {
        let start = self.series.point(theta, zeta);
        let normal = self.normal(&start);
        let far = self.in_plane(std::array::from_fn(|k| start.x[k] + t * normal[k]));
        let mut u = [theta, zeta, far.theta, far.phi];
        let mut best = 0.5 * t;
        for _ in 0..NEWTON_STEPS {
            let p = self.series.point(u[0], u[1]);
            let q = self.series.point(u[2], u[3]);
            let d = sub(q.x, p.x);
            // Ends that slid together have left the chord, and as they meet
            // round-off decides the ball.
            if 4.0 * dot(d, d) < t * t {
                break;
            }
            let along = dot(self.normal(&p), d);
            if along > 0.0 {
                best = best.min(dot(d, d) / (2.0 * along));
            }
            let Some(step) = chord_step(&p, &q, d) else {
                break;
            };
            let (st, sz) = (
                step[0].abs().max(step[2].abs()),
                step[1].abs().max(step[3].abs()),
            );
            if st.max(sz) < NEWTON_STEP_MIN {
                break;
            }
            // At most a patch across per step, so that a critical point far
            // along a flat stretch is approached rather than jumped to.
            let scale = (self.patches.ht / st).min(self.patches.hz / sz).min(1.0);
            for (v, s) in u.iter_mut().zip(step) {
                *v += scale * s;
            }
        }
        best
    }
}

/// Newton step in `(theta_p, zeta_p, theta_q, zeta_q)` towards a
/// stationary point of `|q - p|^2 / 2` (`d = q - p`): the solution of
/// `H s = -g`, regularised as `(H^2 + mu^2) s = -H g` so that a direction
/// in which the length cannot change (turning an axisymmetric boundary
/// about its axis) is left alone rather than divided by zero.
fn chord_step(p: &SurfacePoint, q: &SurfacePoint, d: [f64; 3]) -> Option<[f64; 4]> {
    let g = [-dot(d, p.x_t), -dot(d, p.x_z), dot(d, q.x_t), dot(d, q.x_z)];
    let h = [
        [
            dot(p.x_t, p.x_t) - dot(d, p.x_tt),
            dot(p.x_t, p.x_z) - dot(d, p.x_tz),
            -dot(p.x_t, q.x_t),
            -dot(p.x_t, q.x_z),
        ],
        [
            dot(p.x_z, p.x_t) - dot(d, p.x_tz),
            dot(p.x_z, p.x_z) - dot(d, p.x_zz),
            -dot(p.x_z, q.x_t),
            -dot(p.x_z, q.x_z),
        ],
        [
            -dot(q.x_t, p.x_t),
            -dot(q.x_t, p.x_z),
            dot(q.x_t, q.x_t) + dot(d, q.x_tt),
            dot(q.x_t, q.x_z) + dot(d, q.x_tz),
        ],
        [
            -dot(q.x_z, p.x_t),
            -dot(q.x_z, p.x_z),
            dot(q.x_z, q.x_t) + dot(d, q.x_tz),
            dot(q.x_z, q.x_z) + dot(d, q.x_zz),
        ],
    ];
    // `mu`, a millionth of `H`, is far above the round-off in `H^2` and far
    // below the curvature along any stretch worth following.
    let scale = h.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mu2 = (1e-6 * scale).powi(2);
    let mut a = [[0.0; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = (0..4).map(|k| h[i][k] * h[k][j]).sum::<f64>();
        }
        a[i][i] += mu2;
        a[i][4] = -(0..4).map(|k| h[i][k] * g[k]).sum::<f64>();
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..4 {
        let pivot = (c..4).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))?;
        a.swap(c, pivot);
        let row = a[c];
        if row[c] == 0.0 || !row[c].is_finite() {
            return None;
        }
        for below in a.iter_mut().skip(c + 1) {
            let f = below[c] / row[c];
            for (x, p) in below.iter_mut().zip(row).skip(c) {
                *x -= f * p;
            }
        }
    }
    let mut s = [0.0; 4];
    for r in (0..4).rev() {
        let rest: f64 = (r + 1..4).map(|k| a[r][k] * s[k]).sum();
        s[r] = (a[r][4] - rest) / a[r][r];
    }
    s.iter().all(|v| v.is_finite()).then_some(s)
}

/// Compass search for a local minimum of `f` from `start` (its value and
/// `(theta, zeta)`), with steps of one grid cell halved whenever no move
/// along an axis lowers `f`.
fn descend(f: &impl Fn(f64, f64) -> f64, start: (f64, f64, f64), grid: &Grid) -> f64 {
    let (mut v, mut theta, mut zeta) = start;
    let (mut ht, mut hz) = (grid.ht, grid.hz);
    let mut moves = 0;
    while ht.max(hz) > STEP_MIN && moves < MOVES {
        let mut best = (v, theta, zeta);
        for (dt, dz) in [(ht, 0.0), (-ht, 0.0), (0.0, hz), (0.0, -hz)] {
            if dt == 0.0 && dz == 0.0 {
                continue;
            }
            let w = f(theta + dt, zeta + dz);
            if w < best.0 {
                best = (w, theta + dt, zeta + dz);
            }
        }
        if best.0 < v {
            (v, theta, zeta) = best;
            moves += 1;
        } else {
            ht *= 0.5;
            hz *= 0.5;
        }
    }
    v
}
