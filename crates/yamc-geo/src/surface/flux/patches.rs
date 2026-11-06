//! Conservative patch boxes over `(theta, zeta)`, a box hierarchy for
//! rigorous distance bounds, and per-patch curvature bounds.
//!
//! The parameter torus is cut into `nt x nz` patches. Each patch is
//! sampled on a 3 x 3 grid; the piecewise-bilinear interpolant of those
//! samples lies in their bounding box, and the surface lies within
//! `(h_t^2 / 8) sup|X_tt| + (h_z^2 / 8) sup|X_zz|` of that interpolant
//! (`h` the sample spacing, the tensor-product linear interpolation
//! bound), so the box grown by that sag plus a floating-point margin
//! contains the patch. The distance from a point to a patch's box is
//! therefore a rigorous lower bound on its distance to the patch.
//!
//! Each patch and each node of the hierarchy also carries a cone holding
//! its normals, so the pairs of patches facing each other across a thin
//! part or a hole can be found without visiting every pair.

use super::series::{Series, SurfacePoint};
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Factor on sampled curvatures, for the variation between samples.
const CURVATURE_SAFETY: f64 = 1.5;

#[derive(Debug, Clone, Copy)]
pub(super) struct Aabb {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

impl Aabb {
    /// Smallest distance between a point of this box and one of `other`.
    pub fn distance_to(&self, other: &Aabb) -> f64 {
        (0..3)
            .map(|i| {
                let gap = (other.lo[i] - self.hi[i])
                    .max(self.lo[i] - other.hi[i])
                    .max(0.0);
                gap * gap
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Range of `(x - c) . n` over the box.
    pub fn extent_along(&self, c: [f64; 3], n: [f64; 3]) -> (f64, f64) {
        let (mut lo, mut hi) = (0.0, 0.0);
        for i in 0..3 {
            let (a, b) = (n[i] * (self.lo[i] - c[i]), n[i] * (self.hi[i] - c[i]));
            lo += a.min(b);
            hi += a.max(b);
        }
        (lo, hi)
    }

    pub fn empty() -> Self {
        Self {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        }
    }

    pub fn grow_point(&mut self, p: [f64; 3]) {
        for ((lo, hi), v) in self.lo.iter_mut().zip(self.hi.iter_mut()).zip(p) {
            *lo = lo.min(v);
            *hi = hi.max(v);
        }
    }

    pub fn grow_box(&mut self, other: &Aabb) {
        for k in 0..3 {
            self.lo[k] = self.lo[k].min(other.lo[k]);
            self.hi[k] = self.hi[k].max(other.hi[k]);
        }
    }

    pub fn inflate(&mut self, by: f64) {
        for k in 0..3 {
            self.lo[k] -= by;
            self.hi[k] += by;
        }
    }

    /// Squared distance from `x` to the box (0 inside).
    #[inline]
    pub fn distance2(&self, x: [f64; 3]) -> f64 {
        let mut d2 = 0.0;
        for ((lo, hi), v) in self.lo.iter().zip(&self.hi).zip(x) {
            let d = (lo - v).max(v - hi).max(0.0);
            d2 += d * d;
        }
        d2
    }

    /// Smallest distance from the z axis to the box.
    pub fn min_cylindrical_radius(&self) -> f64 {
        let gap = |lo: f64, hi: f64| {
            if lo > 0.0 {
                lo
            } else if hi < 0.0 {
                -hi
            } else {
                0.0
            }
        };
        gap(self.lo[0], self.hi[0]).hypot(gap(self.lo[1], self.hi[1]))
    }

    /// Ray parameter interval `[t0, t1]` inside the box, if any.
    pub fn clip(&self, origin: [f64; 3], direction: [f64; 3]) -> Option<(f64, f64)> {
        let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
        for k in 0..3 {
            if direction[k] == 0.0 {
                if origin[k] < self.lo[k] || origin[k] > self.hi[k] {
                    return None;
                }
                continue;
            }
            let inv = 1.0 / direction[k];
            let (a, b) = (
                (self.lo[k] - origin[k]) * inv,
                (self.hi[k] - origin[k]) * inv,
            );
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        (t0 <= t1).then_some((t0, t1))
    }
}

/// Directions within `half` (rad) of the unit vector `axis`.
#[derive(Debug, Clone, Copy)]
struct Cone {
    axis: [f64; 3],
    half: f64,
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Cone {
    const ALL: Cone = Cone {
        axis: [0.0, 0.0, 1.0],
        half: PI,
    };

    /// The smallest cone of this construction holding both.
    fn union(self, other: Cone) -> Cone {
        if self.half >= PI || other.half >= PI {
            return Cone::ALL;
        }
        let angle = dot(self.axis, other.axis).clamp(-1.0, 1.0).acos();
        if angle + other.half <= self.half {
            return self;
        }
        if angle + self.half <= other.half {
            return other;
        }
        let half = 0.5 * (self.half + angle + other.half);
        if half >= PI {
            return Cone::ALL;
        }
        let sin = angle.sin();
        if sin < 1e-9 {
            // Nearly parallel axes (nearly opposite ones give half >= PI).
            return Cone {
                axis: self.axis,
                half: self.half.max(other.half) + angle + 1e-9,
            };
        }
        // Turn the axis towards the other one by `half - self.half`.
        let turn = half - self.half;
        let (w1, w2) = ((angle - turn).sin() / sin, turn.sin() / sin);
        let a = [
            w1 * self.axis[0] + w2 * other.axis[0],
            w1 * self.axis[1] + w2 * other.axis[1],
            w1 * self.axis[2] + w2 * other.axis[2],
        ];
        let len = dot(a, a).sqrt();
        Cone {
            axis: [a[0] / len, a[1] / len, a[2] / len],
            // Round-off allowance on the rotation.
            half: half + 1e-9,
        }
    }

    /// Could a direction in this cone be within `slack` (rad) of `dir`?
    fn near(&self, dir: [f64; 3], slack: f64) -> bool {
        let limit = self.half + slack;
        limit >= PI || dot(self.axis, dir) >= limit.cos()
    }
}

#[derive(Debug, Clone)]
struct Node {
    bx: Aabb,
    cone: Cone,
    /// Children, or `u32::MAX` for a leaf holding patch `patch`.
    left: u32,
    right: u32,
    patch: u32,
}

/// Outward unit normal at a surface point.
fn unit_normal(p: &SurfacePoint, orientation: f64) -> [f64; 3] {
    let n = [
        p.x_t[1] * p.x_z[2] - p.x_t[2] * p.x_z[1],
        p.x_t[2] * p.x_z[0] - p.x_t[0] * p.x_z[2],
        p.x_t[0] * p.x_z[1] - p.x_t[1] * p.x_z[0],
    ];
    let len = dot(n, n).sqrt();
    if len == 0.0 {
        return [0.0; 3];
    }
    let s = orientation / len;
    [n[0] * s, n[1] * s, n[2] * s]
}

/// Principal curvatures of the surface at a point, with the convention
/// that a convex surface (seen from outside) has positive curvature.
/// `orientation` makes `orientation * (X_t x X_z)` the outward normal.
pub(super) fn principal_curvatures(p: &SurfacePoint, orientation: f64) -> (f64, f64) {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let nrm = cross(p.x_t, p.x_z);
    let len = dot(nrm, nrm).sqrt();
    if len == 0.0 {
        return (f64::INFINITY, f64::INFINITY);
    }
    // Inward unit normal: curvature towards it is convex.
    let n_in = [
        -orientation * nrm[0] / len,
        -orientation * nrm[1] / len,
        -orientation * nrm[2] / len,
    ];
    let (e, f, g) = (dot(p.x_t, p.x_t), dot(p.x_t, p.x_z), dot(p.x_z, p.x_z));
    let (l, m, n) = (dot(p.x_tt, n_in), dot(p.x_tz, n_in), dot(p.x_zz, n_in));
    let a = e * g - f * f;
    let b = e * n + g * l - 2.0 * f * m;
    let c = l * n - m * m;
    let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
    ((b + disc) / (2.0 * a), (b - disc) / (2.0 * a))
}

/// Cone pairs wider than this (rad, summed) are split before comparing.
const WIDE: f64 = 0.6;
/// Splits of a patch for the gap search, at most (pieces 1/8 the size).
const MAX_DEPTH: u32 = 3;
/// Pieces built for the gap search, at most; past it pairs stay coarse.
const MAX_PIECES: usize = 250_000;

/// A piece of the surface over a parameter rectangle, bounded like a
/// patch: centre point, a radius covering it, a box and a normal cone.
#[derive(Clone, Copy)]
struct Piece {
    centre: [f64; 3],
    radius: f64,
    bx: Aabb,
    cone: Cone,
}

/// A piece: its patch, how many times the patch was halved, and the
/// piece's column and row among the `2^depth x 2^depth`.
type Key = (usize, u32, u32, u32);

/// The four pieces a piece splits into.
fn children(k: Key) -> [Key; 4] {
    let (p, d, a, b) = (k.0, k.1 + 1, 2 * k.2, 2 * k.3);
    [
        (p, d, a, b),
        (p, d, a + 1, b),
        (p, d, a, b + 1),
        (p, d, a + 1, b + 1),
    ]
}

/// Builds the pieces of patches on demand for the gap search.
struct Refiner<'a> {
    patches: &'a Patches,
    series: &'a Series,
    orientation: f64,
    margin: f64,
    pieces: HashMap<Key, Piece>,
}

impl Refiner<'_> {
    fn piece(&mut self, key: Key) -> Piece {
        let pt = self.patches;
        let (p, depth, a, b) = key;
        if depth == 0 {
            let (centre, radius) = pt.spheres[p];
            return Piece {
                centre,
                radius,
                bx: pt.boxes[p],
                cone: pt.cones[p],
            };
        }
        if let Some(piece) = self.pieces.get(&key) {
            return *piece;
        }
        let n = f64::from(1u32 << depth);
        let (ht, hz) = (pt.ht / n, pt.hz / n);
        let theta0 = (p % pt.nt) as f64 * pt.ht + f64::from(a) * ht;
        let zeta0 = (p / pt.nt) as f64 * pt.hz + f64::from(b) * hz;
        let mut points = [[0.0; 3]; 9];
        let mut normals = [[0.0; 3]; 9];
        for row in 0..3 {
            let zeta = zeta0 + 0.5 * hz * row as f64;
            let cut = self.series.collapse(zeta, 2);
            for col in 0..3 {
                let sp = cut.section(theta0 + 0.5 * ht * col as f64).cartesian(zeta);
                points[3 * row + col] = sp.x;
                normals[3 * row + col] = unit_normal(&sp, self.orientation);
            }
        }
        let (centre, normal) = (points[4], normals[4]);
        let mut bx = Aabb::empty();
        let (mut radius2, mut turn) = (0.0_f64, 0.0_f64);
        for (x, nx) in points.iter().zip(&normals) {
            bx.grow_point(*x);
            let d = [x[0] - centre[0], x[1] - centre[1], x[2] - centre[2]];
            radius2 = radius2.max(dot(d, d));
            turn = turn.max(dot(*nx, normal).clamp(-1.0, 1.0).acos());
        }
        let bounds = &self.series.bounds;
        let sag = (ht / 2.0).powi(2) / 8.0 * bounds.x_tt()
            + (hz / 2.0).powi(2) / 8.0 * bounds.x_zz()
            + self.margin;
        bx.inflate(sag);
        let radius = radius2.sqrt() + sag;
        let kappa = pt.convex[p].max(pt.concave[p]);
        let piece = Piece {
            centre,
            radius,
            bx,
            cone: Cone {
                axis: normal,
                half: (CURVATURE_SAFETY * turn).max(1.25 * kappa * radius) + 1e-6,
            },
        };
        self.pieces.insert(key, piece);
        piece
    }

    /// Fold the gap between pieces `pk` and `qk` into `best` (inner,
    /// outer), splitting them while their normal cones are too wide.
    fn pair(&mut self, pk: Key, qk: Key, best: &mut (f64, f64)) {
        let (pp, qq) = (self.piece(pk), self.piece(qk));
        let n = pp.cone.axis;
        if !qq.cone.near(n.map(|v| -v), pp.cone.half) {
            return;
        }
        let to_q = [
            qq.centre[0] - pp.centre[0],
            qq.centre[1] - pp.centre[1],
            qq.centre[2] - pp.centre[2],
        ];
        let from_centre = qq
            .bx
            .distance2(pp.centre)
            .sqrt()
            .max(dot(to_q, to_q).sqrt() - qq.radius);
        let gap = pp
            .bx
            .distance_to(&qq.bx)
            .max(from_centre - pp.radius)
            .max(0.0);
        // A bottleneck runs along the normal at both ends: `b - a` is
        // `+-2 rho n_a`, outwards across the outside, inwards across the
        // inside. With `n_a` within the cone of this piece's axis `n`, and
        // the cone narrower than a right angle, `(b - a) . n` then has that
        // sign, which the boxes' extents along `n` can rule out.
        let (p_lo, p_hi) = pp.bx.extent_along(pp.centre, n);
        let (q_lo, q_hi) = qq.bx.extent_along(pp.centre, n);
        let either = pp.cone.half >= FRAC_PI_2;
        let inner = (either || q_lo < p_hi) && gap < best.0;
        let outer = (either || q_hi > p_lo) && gap < best.1;
        if !inner && !outer {
            return;
        }
        if pp.cone.half + qq.cone.half > WIDE && self.pieces.len() < MAX_PIECES {
            // Split the piece with the wider cone.
            if qq.cone.half >= pp.cone.half && qk.1 < MAX_DEPTH {
                for child in children(qk) {
                    self.pair(pk, child, best);
                }
                return;
            }
            if pk.1 < MAX_DEPTH {
                for child in children(pk) {
                    self.pair(child, qk, best);
                }
                return;
            }
        }
        if inner {
            best.0 = gap;
        }
        if outer {
            best.1 = gap;
        }
    }
}

pub(super) struct Patches {
    pub nt: usize,
    pub nz: usize,
    pub ht: f64,
    pub hz: f64,
    boxes: Vec<Aabb>,
    /// Largest convex / concave curvature over each patch (times a
    /// safety factor).
    convex: Vec<f64>,
    concave: Vec<f64>,
    /// A sphere around each patch: its centre point and a radius covering
    /// the samples' convex hull plus the sag, so `|x - c| - r` bounds the
    /// distance to the patch from below (far tighter than the box for a
    /// tilted patch).
    spheres: Vec<([f64; 3], f64)>,
    /// Smallest patch extent (cm), for sizing neighbourhoods.
    pub min_size: f64,
    /// A cone holding each patch's normals: the centre normal, and the
    /// largest turn of the normal over the patch.
    cones: Vec<Cone>,
    nodes: Vec<Node>,
    pub extents: Aabb,
    /// Largest convex / concave curvature over the whole surface.
    pub max_convex: f64,
    pub max_concave: f64,
}

impl Patches {
    pub fn new(series: &Series, nt: usize, nz: usize, orientation: f64, margin: f64) -> Self {
        let (st, sz) = (2 * nt, 2 * nz);
        let (ht, hz) = (TAU / nt as f64, TAU / nz as f64);
        let mut samples = vec![[0.0; 3]; st * sz];
        let mut normals = vec![[0.0; 3]; st * sz];
        let mut curv = vec![(0.0, 0.0); st * sz];
        for j in 0..sz {
            let zeta = j as f64 * hz / 2.0;
            let collapsed = series.collapse(zeta, 3);
            for i in 0..st {
                let theta = i as f64 * ht / 2.0;
                let p = collapsed.section(theta).cartesian(zeta);
                samples[j * st + i] = p.x;
                normals[j * st + i] = unit_normal(&p, orientation);
                curv[j * st + i] = principal_curvatures(&p, orientation);
            }
        }
        let sag = (ht / 2.0).powi(2) / 8.0 * series.bounds.x_tt()
            + (hz / 2.0).powi(2) / 8.0 * series.bounds.x_zz()
            + margin;
        let mut boxes = Vec::with_capacity(nt * nz);
        let mut spheres = Vec::with_capacity(nt * nz);
        let mut cones = Vec::with_capacity(nt * nz);
        let mut convex = Vec::with_capacity(nt * nz);
        let mut concave = Vec::with_capacity(nt * nz);
        let mut extents = Aabb::empty();
        let mut min_size = f64::INFINITY;
        for j in 0..nz {
            for i in 0..nt {
                let mut bx = Aabb::empty();
                let (mut kv, mut kc) = (0.0_f64, 0.0_f64);
                let centre_index = ((2 * j + 1) % sz) * st + (2 * i + 1) % st;
                let (centre, normal) = (samples[centre_index], normals[centre_index]);
                let mut radius2 = 0.0_f64;
                let mut turn = 0.0_f64;
                for b in 0..3 {
                    for a in 0..3 {
                        let s = ((2 * j + b) % sz) * st + (2 * i + a) % st;
                        let x = samples[s];
                        bx.grow_point(x);
                        let d = [x[0] - centre[0], x[1] - centre[1], x[2] - centre[2]];
                        radius2 = radius2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                        turn = turn.max(dot(normals[s], normal).clamp(-1.0, 1.0).acos());
                        let (k1, k2) = curv[s];
                        kv = kv.max(k1).max(k2);
                        kc = kc.max(-k1).max(-k2);
                    }
                }
                bx.inflate(sag);
                extents.grow_box(&bx);
                boxes.push(bx);
                spheres.push((centre, radius2.sqrt() + sag));
                // The normal turns by at most the curvature times the arc
                // length from the centre; the sampled turn guards the
                // curvature sampling, and a floor covers flat patches.
                let reach = radius2.sqrt() + sag;
                cones.push(Cone {
                    axis: normal,
                    half: CURVATURE_SAFETY * turn.max(1.25 * kv.max(kc).max(0.0) * reach) + 1e-6,
                });
                min_size = min_size.min(radius2.sqrt());
                convex.push(kv.max(0.0) * CURVATURE_SAFETY);
                concave.push(kc.max(0.0) * CURVATURE_SAFETY);
            }
        }
        let max_convex = convex.iter().cloned().fold(0.0, f64::max);
        let max_concave = concave.iter().cloned().fold(0.0, f64::max);
        let mut patches = Self {
            nt,
            nz,
            ht,
            hz,
            boxes,
            convex,
            concave,
            spheres,
            min_size,
            cones,
            nodes: Vec::new(),
            extents,
            max_convex,
            max_concave,
        };
        patches.build(0, nt, 0, nz);
        patches
    }

    /// Build the subtree over patch columns `i0..i1`, rows `j0..j1`,
    /// returning its node index.
    fn build(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) -> u32 {
        let index = self.nodes.len() as u32;
        if i1 - i0 == 1 && j1 - j0 == 1 {
            let patch = j0 * self.nt + i0;
            self.nodes.push(Node {
                bx: self.boxes[patch],
                cone: self.cones[patch],
                left: u32::MAX,
                right: u32::MAX,
                patch: patch as u32,
            });
            return index;
        }
        self.nodes.push(Node {
            bx: Aabb::empty(),
            cone: Cone::ALL,
            left: 0,
            right: 0,
            patch: 0,
        });
        let (left, right) = if i1 - i0 >= j1 - j0 {
            let mid = (i0 + i1) / 2;
            (self.build(i0, mid, j0, j1), self.build(mid, i1, j0, j1))
        } else {
            let mid = (j0 + j1) / 2;
            (self.build(i0, i1, j0, mid), self.build(i0, i1, mid, j1))
        };
        let mut bx = self.nodes[left as usize].bx;
        bx.grow_box(&self.nodes[right as usize].bx);
        let cone = self.nodes[left as usize]
            .cone
            .union(self.nodes[right as usize].cone);
        self.nodes[index as usize] = Node {
            bx,
            cone,
            left,
            right,
            patch: 0,
        };
        index
    }

    /// Smallest distance from the z axis to any patch box.
    pub fn min_radius(&self) -> f64 {
        self.boxes
            .iter()
            .map(Aabb::min_cylindrical_radius)
            .fold(f64::INFINITY, f64::min)
    }

    pub fn patch_of(&self, theta: f64, zeta: f64) -> (usize, usize) {
        let i = (theta.rem_euclid(TAU) / self.ht) as usize % self.nt;
        let j = (zeta.rem_euclid(TAU) / self.hz) as usize % self.nz;
        (i, j)
    }

    /// Parameter rectangle of patch `(i, j)`.
    pub fn rect(&self, i: usize, j: usize) -> ([f64; 2], [f64; 2]) {
        (
            [i as f64 * self.ht, (i + 1) as f64 * self.ht],
            [j as f64 * self.hz, (j + 1) as f64 * self.hz],
        )
    }

    fn near(&self, patch: usize, centre: (usize, usize), radius: usize) -> bool {
        let (i, j) = (patch % self.nt, patch / self.nt);
        let di = (i as isize - centre.0 as isize).rem_euclid(self.nt as isize) as usize;
        let dj = (j as isize - centre.1 as isize).rem_euclid(self.nz as isize) as usize;
        di.min(self.nt - di) <= radius && dj.min(self.nz - dj) <= radius
    }

    /// Squared lower bound on the distance from `x` to patch `patch`: the
    /// larger of its box and sphere bounds.
    #[inline]
    fn leaf_bound2(&self, patch: usize, x: [f64; 3]) -> f64 {
        let (c, r) = self.spheres[patch];
        let d = ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2)).sqrt();
        let sphere = (d - r).max(0.0);
        self.boxes[patch].distance2(x).max(sphere * sphere)
    }

    /// Smallest squared lower bound on the distance from `x` to any patch,
    /// skipping those within `radius` patches of `exclude`; `cutoff2` if
    /// none is below it.
    pub fn min_distance2(
        &self,
        x: [f64; 3],
        exclude: Option<((usize, usize), usize)>,
        cutoff2: f64,
    ) -> f64 {
        let mut best = cutoff2;
        let mut stack = [0u32; 128];
        let mut top = 1;
        stack[0] = 0;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if node.bx.distance2(x) >= best {
                continue;
            }
            if node.left == u32::MAX {
                let patch = node.patch as usize;
                if exclude.is_some_and(|(c, r)| self.near(patch, c, r)) {
                    continue;
                }
                best = best.min(self.leaf_bound2(patch, x));
                continue;
            }
            // Visit the nearer child first (pushed last).
            let (l, r) = (node.left, node.right);
            let (dl, dr) = (
                self.nodes[l as usize].bx.distance2(x),
                self.nodes[r as usize].bx.distance2(x),
            );
            let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
            if top + 2 > stack.len() {
                // The tree is balanced (depth ~log2 of the patch count),
                // so the stack never fills; if it ever did, a full scan
                // keeps the bound exact.
                return self.scan_min(x, exclude, best);
            }
            stack[top] = second;
            stack[top + 1] = first;
            top += 2;
        }
        best
    }

    fn scan_min(&self, x: [f64; 3], exclude: Option<((usize, usize), usize)>, best: f64) -> f64 {
        self.boxes
            .iter()
            .enumerate()
            .filter(|(p, _)| !exclude.is_some_and(|(c, r)| self.near(*p, c, r)))
            .map(|(p, _)| self.leaf_bound2(p, x))
            .fold(best, f64::min)
    }

    /// Lower bounds on the narrowest gaps across the inside and across the
    /// outside of the surface, `(inner, outer)` (cm), from the patches in
    /// the first `rows` rows (a field period; the rest repeat it).
    ///
    /// The distance to the surface is smooth on a side up to its reach,
    /// which is either a focal distance (the curvature bounds cover that)
    /// or half the length of a bottleneck: a pair of surface points whose
    /// normals are exactly opposite and point at each other across a thin
    /// part (inside) or across a hole or a fold (outside). So only pairs of
    /// pieces whose normal cones hold opposite directions are compared,
    /// which skips every pair on one smooth sheet once the pieces are small
    /// enough for their cones to be narrow (patches whose cones are too
    /// wide to tell are split, a few times at most); a pair's distance less
    /// the piece's radius bounds the bottleneck length from below.
    pub fn gaps(&self, series: &Series, rows: usize, orientation: f64, margin: f64) -> (f64, f64) {
        let mut refiner = Refiner {
            patches: self,
            series,
            orientation,
            margin,
            pieces: HashMap::new(),
        };
        // A gap only matters below twice the focal distance on its side.
        let cap = |kappa: f64| {
            if kappa > 0.0 {
                2.0 / kappa
            } else {
                f64::INFINITY
            }
        };
        let mut best = (cap(self.max_convex), cap(self.max_concave));
        let mut stack = Vec::with_capacity(64);
        let mut queue = Vec::new();
        for p in 0..rows.min(self.nz) * self.nt {
            // Split the patch until its pieces' normal cones are narrow,
            // then look for pieces facing each one.
            queue.push((p, 0, 0, 0));
            while let Some(pk) = queue.pop() {
                let piece = refiner.piece(pk);
                if piece.cone.half > 0.5 * WIDE
                    && pk.1 < MAX_DEPTH
                    && refiner.pieces.len() < MAX_PIECES
                {
                    queue.extend(children(pk));
                    continue;
                }
                let (c, n) = (piece.centre, piece.cone.axis);
                let opposite = n.map(|v| -v);
                let either = piece.cone.half >= FRAC_PI_2;
                let (p_lo, p_hi) = piece.bx.extent_along(c, n);
                stack.clear();
                stack.push(0u32);
                while let Some(k) = stack.pop() {
                    let node = &self.nodes[k as usize];
                    if !node.cone.near(opposite, piece.cone.half) {
                        continue;
                    }
                    // Useful on a side only if near enough and with some
                    // point on that side of a point of this piece, along
                    // its normal (see `Refiner::pair`).
                    let d = piece.bx.distance_to(&node.bx);
                    let (lo, hi) = node.bx.extent_along(c, n);
                    let inner = (either || lo < p_hi) && d < best.0;
                    let outer = (either || hi > p_lo) && d < best.1;
                    if !inner && !outer {
                        continue;
                    }
                    if node.left == u32::MAX {
                        refiner.pair(pk, (node.patch as usize, 0, 0, 0), &mut best);
                    } else {
                        stack.push(node.left);
                        stack.push(node.right);
                    }
                }
            }
        }
        best
    }

    /// Patches whose box is within `cutoff2` (squared) of `x`, nearest
    /// first, as `(distance2, i, j)`.
    pub fn within(&self, x: [f64; 3], cutoff2: f64) -> Vec<(f64, usize, usize)> {
        let mut out = Vec::new();
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            let d2 = node.bx.distance2(x);
            if d2 >= cutoff2 {
                continue;
            }
            if node.left == u32::MAX {
                let patch = node.patch as usize;
                let d2 = self.leaf_bound2(patch, x);
                if d2 >= cutoff2 {
                    continue;
                }
                out.push((d2, patch % self.nt, patch / self.nt));
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// Whether the segment `origin + t direction`, `t` in `[t0, t1]`, meets
    /// a patch box (if not, it meets no part of the surface).
    pub fn segment_meets(&self, origin: [f64; 3], direction: [f64; 3], t0: f64, t1: f64) -> bool {
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if !node
                .bx
                .clip(origin, direction)
                .is_some_and(|(a, b)| a <= t1 && b >= t0)
            {
                continue;
            }
            if node.left == u32::MAX {
                return true;
            }
            stack.push(node.left);
            stack.push(node.right);
        }
        false
    }

    /// Largest curvature bending towards a point on `side` (+1 outside,
    /// -1 inside) over the patches within `radius` of `(i, j)`.
    pub fn curvature_near(&self, centre: (usize, usize), radius: usize, side: f64) -> f64 {
        let table = if side > 0.0 {
            &self.concave
        } else {
            &self.convex
        };
        let mut k = 0.0_f64;
        for dj in -(radius as isize)..=radius as isize {
            for di in -(radius as isize)..=radius as isize {
                let i = (centre.0 as isize + di).rem_euclid(self.nt as isize) as usize;
                let j = (centre.1 as isize + dj).rem_euclid(self.nz as isize) as usize;
                k = k.max(table[j * self.nt + i]);
            }
        }
        k
    }

    /// Whether every patch box contains the given surface point (test aid).
    #[cfg(test)]
    pub fn box_of(&self, i: usize, j: usize) -> Aabb {
        self.boxes[j * self.nt + i]
    }
}
