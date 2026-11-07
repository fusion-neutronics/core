//! Ray tests for implicit surfaces: the search against independent oracles
//! on the shapes and ray sets of the prototype study, and one test for
//! each way the prototype's search was found to fail.
//!
//! The oracles never use the interval machinery:
//! - yamc-geo's closed-form sphere and z-torus, with a fine scan judging
//!   any disagreement;
//! - exact roots, for rays grazing a surface;
//! - a scan of the point value (the sign the cell test reads) that
//!   rescans 1000 times finer wherever `|g|` is small next to its slope,
//!   where a close pair of crossings could hide, and bisects each crossing.
//!
//! A crossing is a point strictly across from the side the ray is on (an
//! exact zero is on neither side). A particle on the surface and leaving
//! it may skip the crossing at its feet (within 1e-6 cm).

use super::ray::trace;
use super::tests::{
    holes, tape_of, Rng, GYROID, SMOOTH_UNION, SPHERE, SUPERELLIPSOID, TWISTED, ZTORUS,
};
use super::*;
use crate::surface::search::{leaving, tolerance, Trace, MIN_ROOT};
use std::f64::consts::TAU;

/// Transport moves a crossing particle this far past the surface.
const NUDGE: f64 = 1e-8;
/// The longest stretch the search skips for a particle leaving the surface.
const SKIP: f64 = 1e-6;

fn side(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The value of `tape` at `p`.
fn value_at(tape: &Tape, p: [f64; 3]) -> f64 {
    tape.eval(p, &mut Vec::new())
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / n)
}

fn towards(from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    normalize([to[0] - from[0], to[1] - from[1], to[2] - from[2]])
}

/// A unit vector perpendicular to the unit `d`.
fn perpendicular(rng: &mut Rng, d: [f64; 3]) -> [f64; 3] {
    loop {
        let v = rng.direction();
        let dot = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        let w = [v[0] - dot * d[0], v[1] - dot * d[1], v[2] - dot * d[2]];
        if w.iter().map(|c| c * c).sum::<f64>() > 0.1 {
            return normalize(w);
        }
    }
}

/// The point value of a tape along one ray, for the oracles.
struct Probe<'a> {
    tape: &'a Tape,
    o: [f64; 3],
    u: [f64; 3],
    slots: Vec<f64>,
    duals: Vec<Dual>,
}

impl<'a> Probe<'a> {
    fn new(tape: &'a Tape, o: [f64; 3], u: [f64; 3]) -> Self {
        Self {
            tape,
            o,
            u,
            slots: Vec::new(),
            duals: Vec::new(),
        }
    }

    fn g(&mut self, t: f64) -> f64 {
        self.tape
            .eval(ray_point(self.o, self.u, t), &mut self.slots)
    }

    fn dual(&mut self, t: f64) -> Dual {
        self.tape
            .eval_dual(ray_point(self.o, self.u, t), self.u, &mut self.duals)
    }

    /// The first point in `(a, b]` on side `to`, scanning with step `h` (at
    /// least a float) and bisecting.
    fn first_on(&mut self, a: f64, b: f64, h: f64, to: f64) -> Option<f64> {
        let mut x = a;
        while x < b {
            let y = (x + h).max(x.next_up()).min(b);
            if side(self.g(y)) == to {
                let (mut lo, mut hi) = (x, y);
                loop {
                    let m = 0.5 * (lo + hi);
                    let width = (4.0 * f64::EPSILON * hi.abs()).max(1e-14);
                    if hi - lo <= width || !(lo < m && m < hi) {
                        return Some(hi);
                    }
                    if side(self.g(m)) == to {
                        hi = m;
                    } else {
                        lo = m;
                    }
                }
            }
            x = y;
        }
        None
    }

    /// The first point in `(a, b]` across from side `from`: a scan with
    /// step `h` (at least a float), rescanning 1000 times finer between
    /// samples where `|g|` is small next to the slope.
    fn next(&mut self, a: f64, b: f64, h: f64, from: f64) -> Option<f64> {
        let to = -from;
        let (mut x, mut dx) = (a, self.dual(a));
        while x < b {
            let y = (x + h).max(x.next_up()).min(b);
            let dy = self.dual(y);
            if side(dy.v) == to {
                return self.first_on(x, y, (y - x) / 64.0, to);
            }
            let slope = dx.d.abs().max(dy.d.abs());
            let far_from_zero = dx.v.abs().min(dy.v.abs()) > 4.0 * (y - x) * slope;
            if !far_from_zero && let Some(t) = self.first_on(x, y, (y - x) / 1000.0, to) {
                return Some(t);
            }
            (x, dx) = (y, dy);
        }
        None
    }

    /// The side the ray starts on at `from`, and where to look for
    /// crossings from: with no side at `from`, the ray is on the side
    /// `before` (if not 0), or else on that of the first point past `from`
    /// that has one, as the search takes it.
    fn start(&mut self, from: f64, end: f64, before: f64) -> (f64, f64) {
        let mut sigma = side(self.g(from));
        let mut a = from;
        if sigma == 0.0 {
            sigma = before;
        }
        let mut step = tolerance(from);
        while sigma == 0.0 && a < end {
            a = (a + step).min(end);
            step *= 2.0;
            sigma = side(self.g(a));
        }
        (sigma, a)
    }

    /// The side the ray starts on (see [`Probe::start`]) and its crossings
    /// after `from` and before `end`, up to `max` of them.
    fn crossings(
        &mut self,
        from: f64,
        end: f64,
        h: f64,
        before: f64,
        max: usize,
    ) -> (f64, Vec<f64>) {
        let (sigma, mut a) = self.start(from, end, before);
        let (mut found, mut s) = (Vec::new(), sigma);
        while found.len() < max
            && s != 0.0
            && let Some(t) = self.next(a, end, h, s)
        {
            found.push(t);
            (a, s) = (t, -s);
        }
        (sigma, found)
    }

    /// Whether the sign changes at `t` (found to the tolerance, at or just
    /// past the change).
    fn genuine(&mut self, t: f64) -> bool {
        let w = 2.0 * tolerance(t);
        let s = side(self.g(t - w));
        s != 0.0 && side(self.g(t + w)) == -s
    }
}

/// One query: a ray, its cap and the on-surface hint.
#[derive(Clone, Copy, Debug)]
struct Case {
    o: [f64; 3],
    u: [f64; 3],
    t_max: f64,
    hint: Option<f64>,
}

/// A test shape: the checked formula, the box ray origins are drawn from
/// (also the cell whose planes cap transport-like rays), its closed form if
/// yamc-geo has one, and the oracle's scanning step.
struct Shape {
    name: &'static str,
    formula: ImplicitFormula,
    region: ([f64; 3], [f64; 3]),
    closed: Option<Surface>,
    step: f64,
}

/// `[lo, hi]` scaled about its centre.
fn grown(lo: [f64; 3], hi: [f64; 3], factor: f64) -> ([f64; 3], [f64; 3]) {
    let c: [f64; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    (
        std::array::from_fn(|k| c[k] + factor * (lo[k] - c[k])),
        std::array::from_fn(|k| c[k] + factor * (hi[k] - c[k])),
    )
}

fn shape(name: &'static str) -> Shape {
    let make = |formula: &str, region, closed, step| Shape {
        name,
        formula: ImplicitFormula::new(formula).unwrap(),
        region,
        closed,
        step,
    };
    match name {
        "sphere" => make(
            SPHERE,
            grown([-3.7, -4.2, -3.9], [4.3, 3.8, 4.1], 1.25),
            Some(Surface::new_sphere(0.3, -0.2, 0.1, 4.0, None, None)),
            2e-3,
        ),
        "ztorus" => make(
            ZTORUS,
            grown([-4.8, -4.9, -1.6], [5.2, 5.1, 1.0], 1.25),
            Some(Surface::new_ztorus(
                0.2, 0.1, -0.3, 4.0, 1.3, 1.0, None, None,
            )),
            2e-3,
        ),
        "superellipsoid" => make(SUPERELLIPSOID, ([-2.5; 3], [2.5; 3]), None, 2e-3),
        "smooth union" => make(
            SMOOTH_UNION,
            grown([-3.2, -3.2, -2.1], [4.6, 3.2, 2.1], 1.2),
            None,
            1e-3,
        ),
        "twisted box" => make(
            TWISTED,
            grown([-2.7, -2.7, -3.0], [2.7, 2.7, 3.0], 1.25),
            None,
            1e-3,
        ),
        // No box: the cell's planes stop the search.
        "gyroid" => make(GYROID, ([-3.0; 3], [3.0; 3]), None, 1e-3),
        "holes block" => make(
            &holes(),
            ([-5.0, -5.0, -1.25], [5.0, 5.0, 1.25]),
            None,
            1e-3,
        ),
        _ => unreachable!(),
    }
}

/// Where the ray from `o` (inside the box) leaves the box.
fn exit(region: ([f64; 3], [f64; 3]), o: [f64; 3], u: [f64; 3]) -> f64 {
    (0..3)
        .filter(|&k| u[k] != 0.0)
        .map(|k| {
            let face = if u[k] > 0.0 { region.1[k] } else { region.0[k] };
            (face - o[k]) / u[k]
        })
        .fold(f64::INFINITY, f64::min)
}

fn uniform_in(rng: &mut Rng, region: ([f64; 3], [f64; 3])) -> [f64; 3] {
    std::array::from_fn(|k| rng.range(region.0[k], region.1[k]))
}

/// Origins uniform in the region (inside and outside the shape), isotropic
/// directions, capped where they leave the region.
fn transport(s: &Shape, n: usize, seed: u64) -> Vec<Case> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let o = uniform_in(&mut rng, s.region);
            let u = rng.direction();
            Case {
                o,
                u,
                t_max: exit(s.region, o, u),
                hint: None,
            }
        })
        .collect()
}

/// The oracle's first crossings of random rays in the region, as the rays
/// and the distances along them.
fn surface_points(s: &Shape, n: usize, seed: u64) -> Vec<([f64; 3], [f64; 3], f64)> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let o = uniform_in(&mut rng, s.region);
        let u = rng.direction();
        let end = exit(s.region, o, u);
        let mut probe = Probe::new(s.formula.tape(), o, u);
        if let Some(&t) = probe.crossings(MIN_ROOT, end, s.step, 0.0, 1).1.first() {
            out.push((o, u, t));
        }
    }
    out
}

/// Origins uniform in the region, aimed at points of the surface.
fn aimed(s: &Shape, n: usize, seed: u64) -> Vec<Case> {
    let mut rng = Rng(seed);
    surface_points(s, n, seed ^ 0xa1)
        .into_iter()
        .map(|(po, pu, t)| {
            let o = uniform_in(&mut rng, s.region);
            let u = towards(o, ray_point(po, pu, t));
            Case {
                o,
                u,
                t_max: exit(s.region, o, u),
                hint: None,
            }
        })
        .collect()
}

/// Particles that have just crossed the surface: half carry on along the
/// crossing ray from 1e-8 cm past it, half sit at the crossing with a new
/// direction, into the shape or out of it; with the hint or without it.
fn on_surface(s: &Shape, n: usize, seed: u64, hinted: bool) -> Vec<Case> {
    let mut rng = Rng(seed);
    surface_points(s, n, seed ^ 0x05)
        .into_iter()
        .enumerate()
        .map(|(i, (po, pu, t))| {
            let (o, u, offset) = if i % 2 == 0 {
                (ray_point(po, pu, t + NUDGE), pu, NUDGE)
            } else {
                (ray_point(po, pu, t), rng.direction(), 0.0)
            };
            Case {
                o,
                u,
                t_max: exit(s.region, o, u),
                hint: hinted.then_some(offset),
            }
        })
        .collect()
}

/// Counts over a set of queries.
#[derive(Default, Debug)]
struct Tally {
    rays: usize,
    roots: usize,
    /// A crossing beyond the first, or one that is not a sign change.
    wrong: usize,
    /// No crossing where there is one.
    missed: usize,
    /// The point 1e-8 cm past the crossing is not across it.
    sign: usize,
    exhausted: usize,
    /// A genuine sign change the oracle stepped over.
    oracle_missed: usize,
    /// The crossing at a leaving particle's feet, skipped.
    feet: usize,
    /// The closed form disagreed, and the scan sided with the search.
    closed_wrong: usize,
    evaluations: usize,
    most: usize,
    worst: f64,
    example: Option<String>,
}

impl Tally {
    fn failures(&self) -> usize {
        self.wrong + self.missed + self.sign + self.exhausted
    }

    fn mean(&self) -> f64 {
        self.evaluations as f64 / self.rays.max(1) as f64
    }

    fn fail(&mut self, what: String) {
        self.example.get_or_insert(what);
    }

    /// Judge one answer against the oracle's `crossings` (the first one or
    /// two) of a ray starting on side `sigma`, for a particle that `left`
    /// the surface and may skip the crossing at its feet.
    fn judge(
        &mut self,
        probe: &mut Probe,
        case: &Case,
        got: &Trace,
        (sigma, crossings): (f64, &[f64]),
        left: bool,
    ) {
        self.rays += 1;
        self.evaluations += got.evaluations;
        self.most = self.most.max(got.evaluations);
        if got.exhausted {
            self.exhausted += 1;
            self.fail(format!("exhausted: {case:?}"));
            return;
        }
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * y.abs().max(1.0);
        let first = crossings.first().copied();
        self.roots += usize::from(first.is_some());
        let feet = left && first.is_some_and(|y| y <= SKIP);
        // The side 1e-8 cm past the answer should be on.
        let mut to = -sigma;
        match (got.crossing, first) {
            (None, None) => return,
            (Some(x), Some(y)) if close(x, y) => self.worst = self.worst.max((x - y).abs()),
            (Some(x), Some(_)) if feet && crossings.get(1).is_some_and(|&z| close(x, z)) => {
                self.feet += 1;
                to = sigma;
            }
            (None, Some(_)) if feet && crossings.len() == 1 => {
                self.feet += 1;
                return;
            }
            (None, Some(y)) => {
                self.missed += 1;
                self.fail(format!("missed {y}: {case:?}"));
                return;
            }
            (Some(x), y) if y.is_none_or(|y| x < y) && probe.genuine(x) => {
                self.oracle_missed += 1;
            }
            (Some(x), y) => {
                self.wrong += 1;
                self.fail(format!("{x} where the first crossing is {y:?}: {case:?}"));
                return;
            }
        }
        let x = got.crossing.unwrap_or_default();
        if side(probe.g(x + NUDGE)) != to {
            self.sign += 1;
            self.fail(format!("not across 1e-8 cm past {x}: {case:?}"));
        }
    }
}

/// The search's answer to `case` on shape `s`.
fn query(s: &Shape, case: &Case) -> Trace {
    trace(
        s.formula.tape(),
        s.formula.bounding_box(),
        case.o,
        case.u,
        case.hint,
        false,
        case.t_max,
    )
}

/// Whether the search counts the particle as leaving the surface.
fn left(probe: &mut Probe, case: &Case) -> bool {
    let here = probe.dual(0.0);
    let distance = if case.hint.is_some() || here.v == 0.0 {
        0.0
    } else {
        (here.v / here.d).abs()
    };
    leaving(distance, here.v, here.d)
}

/// Run `cases` on `s` and judge them by the scan or, where `s` has a closed
/// form, by that, with a finer scan judging where the two disagree.
fn campaign(s: &Shape, cases: &[Case]) -> Tally {
    let mut tally = Tally::default();
    for case in cases {
        let got = query(s, case);
        let mut probe = Probe::new(s.formula.tape(), case.o, case.u);
        let left = left(&mut probe, case);
        let before = side(probe.g(0.0));
        let closed = s.closed.as_ref().map(|c| {
            c.distance_to_surface_on(case.o, case.u, case.hint)
                .filter(|&t| t <= case.t_max)
        });
        let truth = match (closed, got.crossing) {
            (Some(None), None) => (0.0, Vec::new()),
            (Some(Some(y)), Some(x)) if (x - y).abs() <= 1e-9 * y.max(1.0) => {
                (probe.start(MIN_ROOT, case.t_max, before).0, vec![y])
            }
            _ => {
                tally.closed_wrong += usize::from(closed.is_some());
                let step = if closed.is_some() { 1e-5 } else { s.step };
                let far = got
                    .crossing
                    .into_iter()
                    .chain(closed.flatten())
                    .fold(f64::NEG_INFINITY, f64::max);
                let far = if far.is_finite() { far } else { case.t_max };
                let stop = (far + 2.0 * s.step).min(case.t_max);
                probe.crossings(MIN_ROOT, stop, step, before, 2)
            }
        };
        tally.judge(&mut probe, case, &got, (truth.0, &truth.1), left);
    }
    tally
}

/// Assert a campaign found no failure and spent at most `mean` evaluations
/// a ray on average and `most` on any.
fn clean(what: &str, t: &Tally, mean: f64, most: usize) {
    assert_eq!(t.failures(), 0, "{what}: {t:?}");
    assert!(
        t.mean() <= mean,
        "{what}: {:.1} evaluations a ray: {t:?}",
        t.mean()
    );
    assert!(
        t.most <= most,
        "{what}: {} evaluations on one ray: {t:?}",
        t.most
    );
}

/// Every ray set on one shape, `n` transport-like rays and fewer of the
/// others, each clean, with at most `mean` evaluations a ray on average
/// and 250 on any.
fn run_shape(name: &'static str, n: usize, mean: f64) {
    let s = shape(name);
    for (set, cases) in [
        ("transport", transport(&s, n, 1)),
        ("aimed", aimed(&s, n / 2, 2)),
        ("hinted on the surface", on_surface(&s, n / 4, 3, true)),
        ("unhinted on the surface", on_surface(&s, n / 4, 4, false)),
    ] {
        let t = campaign(&s, &cases);
        println!(
            "{} {set}: {} rays, {} crossings, {:.1} evaluations a ray (most {}), worst error \
             {:.1e}, feet {}, oracle missed {}, closed form wrong {}",
            s.name,
            t.rays,
            t.roots,
            t.mean(),
            t.most,
            t.worst,
            t.feet,
            t.oracle_missed,
            t.closed_wrong
        );
        clean(&format!("{} {set}", s.name), &t, mean, 250);
    }
}

/// The oracle's scans step by at least a float, so one whose step is below
/// an ulp of where it is still ends.
#[test]
fn the_oracle_scans_always_move_on() {
    let tape = tape_of("x - 10");
    let mut probe = Probe::new(&tape, [0.0; 3], [1.0, 0.0, 0.0]);
    assert_eq!(probe.first_on(1.0, 1.0 + 1e-15, 1e-18, 1.0), None);
    assert_eq!(probe.next(1.0, 1.0 + 1e-15, 1e-18, -1.0), None);
}

#[test]
fn a_sphere_crosses_where_the_closed_form_does() {
    run_shape("sphere", 2_000, 25.0);
}

#[test]
fn a_torus_crosses_where_the_closed_form_does() {
    run_shape("ztorus", 2_000, 30.0);
}

#[test]
fn a_superellipsoid_crosses_where_the_scan_does() {
    run_shape("superellipsoid", 800, 25.0);
}

#[test]
fn a_smooth_union_crosses_where_the_scan_does() {
    run_shape("smooth union", 600, 35.0);
}

#[test]
fn a_twisted_box_crosses_where_the_scan_does() {
    run_shape("twisted box", 600, 35.0);
}

#[test]
fn a_gyroid_in_a_box_crosses_where_the_scan_does() {
    run_shape("gyroid", 600, 40.0);
}

#[test]
fn a_block_with_holes_crosses_where_the_scan_does() {
    run_shape("holes block", 300, 30.0);
}

/// Judge rays against their exact first crossings.
fn exact(tape: &Tape, bx: Option<([f64; 3], [f64; 3])>, rays: &[(Case, Option<f64>)]) -> Tally {
    let mut tally = Tally::default();
    for (case, want) in rays {
        let got = trace(tape, bx, case.o, case.u, None, false, case.t_max);
        let mut probe = Probe::new(tape, case.o, case.u);
        let sigma = probe.start(MIN_ROOT, case.t_max, 0.0).0;
        let want: Vec<f64> = want.iter().copied().collect();
        tally.judge(&mut probe, case, &got, (sigma, &want), false);
    }
    tally
}

#[test]
fn grazing_rays_find_the_exact_crossings() {
    let mut rng = Rng(7);
    let sphere = shape("sphere");
    let torus = ImplicitFormula::new("(sqrt(x^2 + y^2) - 4)^2 + z^2 - 1").unwrap();
    let flat = ImplicitFormula::new(SUPERELLIPSOID).unwrap();
    let n = 150;
    let check = |what: String, t: Tally| {
        println!(
            "{what}: {:.1} evaluations a ray (most {}), worst error {:.1e}",
            t.mean(),
            t.most,
            t.worst
        );
        clean(&what, &t, 120.0, 200);
    };
    for delta in [1e-3, 1e-6, 1e-9] {
        // The sphere (centre c, radius 4) passed at an impact parameter
        // 4 - delta (two crossings 2 sqrt(delta (8 - delta)) apart) or
        // 4 + delta (none).
        for hit in [true, false] {
            let (c, r, l) = ([0.3, -0.2, 0.1], 4.0, 8.0);
            let rays: Vec<_> = (0..n)
                .map(|_| {
                    let u = rng.direction();
                    let e = perpendicular(&mut rng, u);
                    let b = if hit { r - delta } else { r + delta };
                    let o: [f64; 3] = std::array::from_fn(|k| c[k] + b * e[k] - l * u[k]);
                    let case = Case {
                        o,
                        u,
                        t_max: 2.0 * l,
                        hint: None,
                    };
                    (case, hit.then(|| l - (delta * (2.0 * r - delta)).sqrt()))
                })
                .collect();
            let f = &sphere.formula;
            let t = exact(f.tape(), f.bounding_box(), &rays);
            check(format!("sphere grazed {delta:e} inside {hit}"), t);
        }
        // A circular torus (radii 4 and 1) crossed horizontally through its
        // axis at height 1 - delta (twice through the top of the tube on
        // each side) or 1 + delta (missed), with its box and with none.
        for hit in [true, false] {
            for boxed in [true, false] {
                let l = 7.5;
                let rays: Vec<_> = (0..n)
                    .map(|_| {
                        let phi = rng.range(0.0, TAU);
                        let u = [phi.cos(), phi.sin(), 0.0];
                        let z = if hit { 1.0 - delta } else { 1.0 + delta };
                        // The half-chord of the tube's top, w^2 = (1 - z)(1 + z).
                        let w2 = (1.0 - z) * (1.0 + z);
                        let case = Case {
                            o: [-l * u[0], -l * u[1], z],
                            u,
                            t_max: 2.0 * l,
                            hint: None,
                        };
                        (case, (w2 > 0.0).then(|| l - 4.0 - w2.sqrt()))
                    })
                    .collect();
                let bx = if boxed { torus.bounding_box() } else { None };
                let t = exact(torus.tape(), bx, &rays);
                check(format!("torus top {delta:e}, hit {hit}, boxed {boxed}"), t);
            }
        }
        // The same torus from inside the tube, along a line in the plane
        // z = 0 passing delta inside its inner wall (the first crossing is
        // a near double root, into the hole) or delta outside it (the line
        // stays in the tube up to the far outer wall).
        for hit in [true, false] {
            let rays: Vec<_> = (0..n)
                .map(|_| {
                    let phi = rng.range(0.0, TAU);
                    let (e1, e2) = ([phi.cos(), phi.sin(), 0.0], [-phi.sin(), phi.cos(), 0.0]);
                    let y0: f64 = if hit { 3.0 - delta } else { 3.0 + delta };
                    let s = ((4.0 - y0) * (4.0 + y0)).sqrt();
                    let o = [-s * e1[0] + y0 * e2[0], -s * e1[1] + y0 * e2[1], 0.0];
                    let want = if hit {
                        s - ((3.0 - y0) * (3.0 + y0)).sqrt()
                    } else {
                        s + ((5.0 - y0) * (5.0 + y0)).sqrt()
                    };
                    let case = Case {
                        o,
                        u: e1,
                        t_max: 20.0,
                        hint: None,
                    };
                    (case, Some(want))
                })
                .collect();
            let t = exact(torus.tape(), torus.bounding_box(), &rays);
            check(format!("torus inner wall {delta:e}, crossed {hit}"), t);
        }
        // x^4 + y^4 + z^4 = 16 crossed horizontally at height 2 - delta,
        // near its flat top (fourth-order contact).
        let l = 5.0;
        let rays: Vec<_> = (0..n)
            .map(|_| {
                let phi = rng.range(0.0, TAU);
                let u = [phi.cos(), phi.sin(), 0.0];
                let z: f64 = 2.0 - delta;
                let rz = (2.0 - z) * (2.0 + z) * (4.0 + z * z);
                let den = u[0].powi(4) + u[1].powi(4);
                let case = Case {
                    o: [-l * u[0], -l * u[1], z],
                    u,
                    t_max: 2.0 * l,
                    hint: None,
                };
                (case, Some(l - (rz / den).powf(0.25)))
            })
            .collect();
        let t = exact(flat.tape(), flat.bounding_box(), &rays);
        check(format!("superellipsoid flat top {delta:e}"), t);
    }
}

// The prototype's failures, each with the input that showed it; each test
// fails with that failure put back in the search. (Its formulas that were
// NaN where they are used are now refused when they are built: see
// `formulas_undefined_where_they_are_used_are_refused`.)

/// The prototype took the side where a ray enters the bounding box from
/// the point there, not from the box's contract (above outside it), so a
/// box touching the surface lost the entry crossing and returned the exit.
#[test]
fn a_box_touching_the_surface_keeps_the_entry_crossing() {
    let quartic = tape_of(SUPERELLIPSOID);
    let tight = Some(([-2.0; 3], [2.0; 3]));
    let got = trace(
        &quartic,
        tight,
        [3.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        None,
        false,
        10.0,
    );
    assert!(
        got.crossing.is_some_and(|t| (t - 1.0).abs() < 1e-12),
        "{:?}",
        got.crossing
    );
    let mut rng = Rng(77);
    for (formula, bx) in [
        (
            "max(abs(x), abs(y), abs(z)) - 1".to_string(),
            ([-1.0; 3], [1.0; 3]),
        ),
        (holes(), ([-4.0, -4.0, -1.0], [4.0, 4.0, 1.0])),
    ] {
        let tape = tape_of(&formula);
        let region = grown(bx.0, bx.1, 1.5);
        let mut tally = Tally::default();
        while tally.rays < 300 {
            let o = uniform_in(&mut rng, region);
            if (0..3).all(|k| bx.0[k] <= o[k] && o[k] <= bx.1[k]) {
                continue;
            }
            let u = towards(o, uniform_in(&mut rng, bx));
            let case = Case {
                o,
                u,
                t_max: exit(region, o, u),
                hint: None,
            };
            let got = trace(&tape, Some(bx), o, u, None, false, case.t_max);
            let mut probe = Probe::new(&tape, o, u);
            let stop = got
                .crossing
                .map_or(case.t_max, |x| (x + 2e-3).min(case.t_max));
            let (sigma, truth) = probe.crossings(MIN_ROOT, stop, 1e-3, 0.0, 2);
            tally.judge(&mut probe, &case, &got, (sigma, &truth), false);
        }
        clean(&formula, &tally, 60.0, 1_000);
    }
}

/// With `g'` overflowed and `g` finite, Newton's step is 0: the prototype
/// took it every time, crept half a tolerance a step, and returned the far
/// end of a bracket centimetres wide.
#[test]
fn an_overflowing_slope_still_closes_the_bracket() {
    let mut rng = Rng(99);
    for k in [300.0, 700.0, 1e3, 1e4] {
        let f = ImplicitFormula::new(&format!("exp({k} * (x - 1)) - 1")).unwrap();
        let (mut rays, mut most) = (0, 0);
        while rays < 2_000 {
            let o: [f64; 3] = std::array::from_fn(|_| rng.range(-1.0, 3.0));
            let u = rng.direction();
            let root = (1.0 - o[0]) / u[0];
            if !(root > 1e-6 && root < 4.9) {
                continue;
            }
            rays += 1;
            let got = trace(f.tape(), None, o, u, None, false, 5.0);
            assert!(
                got.crossing
                    .is_some_and(|t| (t - root).abs() <= 1e-9 * (1.0 + root)),
                "k {k}, from {o:?} along {u:?}: {:?}, the crossing is at {root}",
                got.crossing
            );
            most = most.max(got.evaluations);
        }
        assert!(most < 300, "k {k}: {most} evaluations on one ray");
    }
}

/// The two crossings of the sphere about the origin of radius `r`, by
/// the closest approach (no cancellation).
fn sphere_roots(r: f64, o: [f64; 3], u: [f64; 3]) -> Option<(f64, f64)> {
    let tc = -(o[0] * u[0] + o[1] * u[1] + o[2] * u[2]);
    let p = ray_point(o, u, tc);
    let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    (d <= r).then(|| {
        let w = ((r - d) * (r + d)).sqrt();
        (tc - w, tc + w)
    })
}

/// Grazing a thin feature, the plain enclosure over the box around a piece
/// clears it only once pieces are about as short as the feature is thin:
/// the prototype ran out of evaluations and returned the point reached as
/// a crossing, again and again along the chord. Walked as transport walks
/// (to each crossing, 1e-8 cm past it, on with the hint), a shell of
/// thickness `h` grazed within `3h` of a wall must give the exact
/// crossings, but for those in a pair closer than the nudge.
#[test]
fn grazing_a_thin_shell_neither_exhausts_nor_invents_crossings() {
    let r = 4.0;
    let mut rng = Rng(16);
    for h in [1e-4, 1e-5, 1e-6, 1e-7] {
        let f =
            ImplicitFormula::new(&format!("abs(sqrt(x^2 + y^2 + z^2) - 4) - {}", 0.5 * h)).unwrap();
        let (mut matched, mut evaluations, mut most) = (0, 0, 0);
        for i in 0..300 {
            let u = rng.direction();
            let e = perpendicular(&mut rng, u);
            let wall = if i % 2 == 0 { r - 0.5 * h } else { r + 0.5 * h };
            let b = wall + rng.range(-3.0 * h, 3.0 * h);
            let o: [f64; 3] = std::array::from_fn(|k| b * e[k] - 6.0 * u[k]);
            let mut want: Vec<f64> = [r - 0.5 * h, r + 0.5 * h]
                .iter()
                .filter_map(|&radius| sphere_roots(radius, o, u))
                .flat_map(|(a, b)| [a, b])
                .collect();
            want.sort_by(f64::total_cmp);
            // Walk the ray.
            let (mut at, mut p, mut hint) = (0.0, o, None);
            let mut got = Vec::new();
            while got.len() < 8 {
                let step = trace(f.tape(), f.bounding_box(), p, u, hint, false, 12.0 - at);
                assert!(!step.exhausted, "h {h}: exhausted from {p:?} along {u:?}");
                evaluations += step.evaluations;
                most = most.max(step.evaluations);
                let Some(t) = step.crossing else { break };
                got.push(at + t);
                at += t + NUDGE;
                p = ray_point(p, u, t + NUDGE);
                hint = Some(NUDGE);
            }
            // Each crossing found is an exact one, and each exact one is
            // found unless it is within the nudge of another.
            for &x in &got {
                assert!(
                    want.iter().any(|&w| (x - w).abs() < 1e-9 + 2.0 * NUDGE),
                    "h {h}: {x} is no crossing (exact {want:?}, found {got:?})"
                );
            }
            for (j, &w) in want.iter().enumerate() {
                if got.iter().any(|&x| (x - w).abs() < 1e-9 + 2.0 * NUDGE) {
                    matched += 1;
                    continue;
                }
                let paired = want
                    .iter()
                    .enumerate()
                    .any(|(k, &v)| k != j && (v - w).abs() < 3.0 * NUDGE);
                assert!(paired, "h {h}: missed {w} (exact {want:?}, found {got:?})");
            }
        }
        println!(
            "shell {h:e}: {matched} crossings matched, {:.1} evaluations a ray, most {most} a query",
            evaluations as f64 / 300.0
        );
        assert!(most < 1_000, "h {h}: {most} evaluations on one query");
    }
}

/// A particle leaving the surface skips at most 1e-6 cm past it, and only
/// where the formula is shown monotone over the stretch skipped, so a wall
/// thinner than that, crossed just before, is found all the same: through
/// a shell 1e-7 to 6e-7 cm thick, head on and obliquely, moved 1e-8 cm past
/// its outer wall as transport moves it, with the hint and without it.
#[test]
fn a_wall_thinner_than_the_skip_is_found_after_crossing_the_other() {
    for h in [1e-7, 3e-7, 6e-7] {
        let f =
            ImplicitFormula::new(&format!("abs(sqrt(x^2 + y^2 + z^2) - 4) - {}", 0.5 * h)).unwrap();
        for b in [0.0, 2.0, 3.0] {
            let (o, u) = ([10.0, b, 0.0], [-1.0, 0.0, 0.0]);
            let wall = |r: f64| sphere_roots(r, o, u).map(|(near, _)| near).unwrap();
            let (outer, inner) = (wall(4.0 + 0.5 * h), wall(4.0 - 0.5 * h));
            let first = trace(&f.tape, f.bounding_box(), o, u, None, false, 20.0);
            assert!(
                first.crossing.is_some_and(|t| (t - outer).abs() < 1e-9),
                "h {h}, b {b}: {:?}, the outer wall is at {outer}",
                first.crossing
            );
            let at = first.crossing.unwrap() + NUDGE;
            for hint in [Some(NUDGE), None] {
                let next = trace(
                    &f.tape,
                    f.bounding_box(),
                    ray_point(o, u, at),
                    u,
                    hint,
                    false,
                    20.0,
                );
                assert!(
                    next.crossing.is_some_and(|t| (at + t - inner).abs() < 1e-9),
                    "h {h}, b {b}, hint {hint:?}: {:?} from {at}, the inner wall is at {inner}",
                    next.crossing
                );
            }
        }
    }
}

/// Inigo Quilez's exact signed distances to a capped cylinder and a box add
/// `sqrt` of a sum of clamped squares, which is 0 on the surface itself,
/// where the slope of `sqrt` alone is unbounded. With no bound on the slope
/// the centred form could not be used near the surface, and rays grazing a
/// 30 cm cylinder within 1e-4 cm ran out of evaluations three times in four.
#[test]
fn grazing_an_exact_distance_neither_exhausts_nor_invents_crossings() {
    let mut rng = Rng(23);
    for (r, h) in [(1.0_f64, 2.0_f64), (30.0, 100.0)] {
        let f = ImplicitFormula::new(&format!(
            "min(max(sqrt(x^2 + y^2) - {r}, abs(z) - {h}), 0) \
             + sqrt(max(sqrt(x^2 + y^2) - {r}, 0)^2 + max(abs(z) - {h}, 0)^2)"
        ))
        .unwrap();
        for delta in [1e-4, 1e-6, 1e-8, 1e-10] {
            // Across the axis at an impact parameter r - delta (two
            // crossings 2 sqrt(delta (2r - delta)) apart) or r + delta (none).
            for hit in [true, false] {
                let rays: Vec<_> = (0..100)
                    .map(|_| {
                        let phi = rng.range(0.0, TAU);
                        let (u, e) = ([phi.cos(), phi.sin(), 0.0], [-phi.sin(), phi.cos(), 0.0]);
                        let b = if hit { r - delta } else { r + delta };
                        let l = 3.0 * r;
                        let case = Case {
                            o: [
                                b * e[0] - l * u[0],
                                b * e[1] - l * u[1],
                                rng.range(-0.7 * h, 0.7 * h),
                            ],
                            u,
                            t_max: 2.0 * l,
                            hint: None,
                        };
                        (case, hit.then(|| l - ((r - b) * (r + b)).sqrt()))
                    })
                    .collect();
                let t = exact(&f.tape, f.bounding_box(), &rays);
                println!(
                    "capped cylinder {r} grazed {delta:e}, hit {hit}: {:.1} evaluations a ray \
                     (most {})",
                    t.mean(),
                    t.most
                );
                clean(&format!("cylinder {r} {delta:e} {hit}"), &t, 100.0, 150);
            }
        }
    }
    // The box [-1, 1] x [-2, 2] x [-1.5, 1.5] entered through its face x = 1
    // at angles down to 1e-6 rad.
    let f = ImplicitFormula::new(
        "sqrt(max(abs(x) - 1, 0)^2 + max(abs(y) - 2, 0)^2 + max(abs(z) - 1.5, 0)^2) \
         + min(max(abs(x) - 1, abs(y) - 2, abs(z) - 1.5), 0)",
    )
    .unwrap();
    for alpha in [1e-1_f64, 1e-3, 1e-6] {
        let rays: Vec<_> = (0..100)
            .map(|_| {
                let side = if rng.below(2) == 0 { 1.0 } else { -1.0 };
                let u = [-alpha.sin(), side * alpha.cos(), 0.0];
                // 0.4 cm along y before the face, at y0.
                let s = 0.4 / alpha.cos();
                let y0 = rng.range(-1.5, 1.5);
                let case = Case {
                    o: [1.0 + s * alpha.sin(), y0 - side * 0.4, rng.range(-1.0, 1.0)],
                    u,
                    t_max: 3.0,
                    hint: None,
                };
                (case, Some(s))
            })
            .collect();
        let t = exact(&f.tape, f.bounding_box(), &rays);
        println!(
            "box entered at {alpha:e} rad: {:.1} evaluations a ray (most {})",
            t.mean(),
            t.most
        );
        clean(&format!("box {alpha:e}"), &t, 20.0, 40);
    }
}

/// The point value of a sphere is exactly 0 for about 4e-8 cm along a
/// tangent ray. The prototype took that for a crossing and stopped the
/// particle where it is in neither half-space.
#[test]
fn a_tangent_touch_is_no_crossing() {
    let f = ImplicitFormula::new("sqrt(x^2 + y^2 + z^2) - 4").unwrap();
    let (o, u) = ([-10.0, 0.0, 4.0], [1.0, 0.0, 0.0]);
    assert_eq!(f.evaluate(ray_point(o, u, 10.0)), 0.0);
    let got = trace(f.tape(), f.bounding_box(), o, u, None, false, 20.0);
    assert_eq!(got.crossing, None);
    assert!(
        !got.exhausted && got.evaluations < 400,
        "{}",
        got.evaluations
    );
    // Starting in that band: there is no side to start on, and the first
    // point past it that has one is above, so there is no crossing either.
    let got = trace(
        f.tape(),
        f.bounding_box(),
        [0.0, 0.0, 4.0],
        u,
        None,
        false,
        10.0,
    );
    assert_eq!(got.crossing, None);
    // Heading in from it, the crossing is the far side, not one at the
    // particle's feet: just past t = 8, where the value is exactly 0, at
    // the first point strictly above.
    let (o, u) = ([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]);
    assert_eq!(f.evaluate(ray_point(o, u, 8.0)), 0.0);
    let got = trace(f.tape(), f.bounding_box(), o, u, None, false, 10.0);
    assert!(
        got.crossing.is_some_and(|t| t > 8.0
            && t - 8.0 <= tolerance(8.0)
            && f.evaluate(ray_point(o, u, t)) > 0.0),
        "{:?}",
        got.crossing
    );
}

/// Where the value is exactly 0 along a stretch, a probe narrowing a
/// crossing can land in it, and the prototype returned such a probe as the
/// crossing: the particle moved 1e-8 cm past it was still in the stretch,
/// in neither half-space. Here `(x - 1) * 1e-320` underflows to 0 for
/// `|x - 1| < 2.5e-4`; the crossing returned is the first point past that
/// stretch strictly above, from either end of it, with the hint or not.
/// (Construction refuses the formula, as it is that small everywhere; the
/// search must still cope.)
#[test]
fn a_crossing_through_exact_zeros_lands_strictly_across() {
    let tape = tape_of("(x - 1) * 1e-320");
    let band = 2.47e-4;
    assert_eq!(value_at(&tape, [1.0 + 0.99 * band, 0.0, 0.0]), 0.0);
    assert!(value_at(&tape, [1.0 + 1.01 * band, 0.0, 0.0]) > 0.0);
    let mut rng = Rng(5);
    for _ in 0..200 {
        let o = [
            rng.range(-2.0, 0.9),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let u = towards(o, [rng.range(1.5, 3.0), rng.range(-1.0, 1.0), 0.0]);
        let hint = (rng.below(2) == 0).then_some(0.0);
        let got = trace(&tape, None, o, u, hint, false, 10.0);
        let t = got.crossing.expect("the ray crosses the plane");
        let mut probe = Probe::new(&tape, o, u);
        assert!(
            probe.g(t) > 0.0,
            "from {o:?} along {u:?}: {t} is not across"
        );
        // The first such point, to the tolerance.
        assert!(
            probe.g(t - 2.0 * tolerance(t)) <= 0.0,
            "from {o:?} along {u:?}: {t} is past the first point across"
        );
        assert!(got.evaluations < 300, "{}", got.evaluations);
    }
}

/// Starting on an exact zero or a NaN, the prototype took any later sign
/// for a crossing. `(x - 1) / (x - 1)` is 1 everywhere but at x = 1, where
/// it is NaN (construction would refuse it; the search must still cope).
#[test]
fn a_start_with_no_side_gives_no_crossing() {
    let tape = tape_of("(x - 1) / (x - 1)");
    let u = [1.0, 0.0, 0.0];
    // An origin whose first sample, at MIN_ROOT, lands exactly on x = 1.
    let mut ox: f64 = 1.0 - MIN_ROOT;
    while ray_point([ox, 0.0, 0.0], u, MIN_ROOT)[0] != 1.0 {
        ox = if ray_point([ox, 0.0, 0.0], u, MIN_ROOT)[0] < 1.0 {
            ox.next_up()
        } else {
            ox.next_down()
        };
    }
    for o in [[ox, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0; 3]] {
        let got = trace(&tape, None, o, u, None, false, 2.0);
        assert_eq!(got.crossing, None, "from {o:?}");
        assert!(!got.exhausted, "from {o:?}");
    }
}

/// The prototype took a particle as leaving the surface when `g * g' >=
/// 0`, which underflows to -0.0 for a formula scaled small enough: one
/// heading for the surface skipped it.
#[test]
fn a_particle_heading_for_a_faint_surface_finds_it() {
    let f = ImplicitFormula::new("(x - 5e-8) * 1e-160").unwrap();
    let here = f.evaluate([0.0; 3]);
    assert!(here < 0.0 && here * 1e-160 == 0.0);
    for hint in [None, Some(0.0)] {
        let got = trace(f.tape(), None, [0.0; 3], [1.0, 0.0, 0.0], hint, false, 1.0);
        assert!(
            got.crossing.is_some_and(|t| t > 5e-8
                && t - 5e-8 <= tolerance(5e-8)
                && f.evaluate([t, 0.0, 0.0]) > 0.0),
            "hint {hint:?}: {:?}",
            got.crossing
        );
    }
}

/// In the plane y = 0, `min(0, -abs(y))` is `min(+0, -0)`, which takes its
/// first operand, +0. The prototype decided such a tie for the second
/// operand when shortening a tape, and the shortened tape read -0, which
/// `1 / ...` turns into the other sign: here it would lose the surface,
/// the circle `x^2 + z^2 = 16`.
#[test]
fn shortened_tapes_cross_where_the_full_tape_does() {
    let tape = tape_of("x^2 + z^2 - 16 + 32 / (1 + exp(1 / min(0, -abs(y))))");
    assert!(tape.len() >= 16 && tape.choices() == 1);
    let (o, u) = ([-6.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
    let got = trace(&tape, None, o, u, None, false, 12.0);
    let want = 6.0 - 15.75_f64.sqrt();
    assert!(
        got.crossing.is_some_and(|t| (t - want).abs() < 1e-9),
        "{:?}, want {want}",
        got.crossing
    );
}

/// Far along a ray floats are more than `H_MIN` apart, so a piece's middle
/// can round onto an end: the prototype then split the same piece again
/// until the budget ran out. A ray passing just outside a sphere (the
/// computed value is at least 1e-20 along it, but no enclosure can show it
/// is positive near the touch) from up to 1e8 cm away.
#[test]
fn far_along_a_ray_pieces_stop_splitting() {
    let f = ImplicitFormula::new("sqrt((x - 1)^2 + y^2 + z^2) - 1 + 1e-20").unwrap();
    let loose = Some((
        [-1e-6, -1.0 - 1e-6, -1.0 - 1e-6],
        [2.0 + 1e-6, 1.0 + 1e-6, 1.0 + 1e-6],
    ));
    for bx in [f.bounding_box(), loose] {
        for far in [1e3, 1e5, 1e6, 1e7, 1e8] {
            let o = [0.0, -far, 0.0];
            let got = trace(f.tape(), bx, o, [0.0, 1.0, 0.0], None, false, 2.0 * far);
            assert_eq!(got.crossing, None, "{far:e}");
            assert!(
                !got.exhausted && got.evaluations < 1_000,
                "{far:e}: {}",
                got.evaluations
            );
        }
    }
}

/// Next to a crossing the computed values are noisy: here they read 0 at
/// the cap, a float past the crossing the search finds with no cap, and
/// strictly across a float or two before it. The search took the piece
/// ending at the cap, whose slope keeps its sign, to hold no crossing, as
/// its far end was not across.
#[test]
fn a_crossing_in_the_noise_just_before_the_cap_is_found() {
    let f = ImplicitFormula::new("abs(x)^2.5 + abs(y)^2.5 + abs(z)^2.5 - 1").unwrap();
    let o = [
        0.01929613419445131,
        -0.039502943238021615,
        1.2150783245522208,
    ];
    let u = [
        -0.6552824945997756,
        -0.48395887538336824,
        -0.5799902233734292,
    ];
    let free = trace(&f.tape, f.bounding_box(), o, u, None, false, f64::INFINITY);
    let x = free.crossing.unwrap();
    let cap = x.next_up();
    assert_eq!(f.evaluate(ray_point(o, u, cap)), 0.0);
    let got = trace(&f.tape, f.bounding_box(), o, u, None, false, cap);
    assert!(
        got.crossing
            .is_some_and(|t| t <= cap && f.evaluate(ray_point(o, u, t)) < 0.0),
        "{:?} with the cap at {cap}, {x} with none",
        got.crossing
    );
}

/// A cap just past the floor, where the prototype's start rule stepped past
/// an exact zero beyond the end of the stretch and built an inverted
/// interval: a panic in debug builds, and in release no crossing where
/// there is one.
#[test]
fn a_stretch_shorter_than_the_start_step_is_searched() {
    // Exactly 0 at the floor, above it from the next float on.
    let f = ImplicitFormula::new("x - 1e-10").unwrap();
    assert_eq!(f.evaluate([MIN_ROOT, 0.0, 0.0]), 0.0);
    for t_max in [MIN_ROOT, MIN_ROOT + 2e-13, MIN_ROOT + 1e-11, 2.0 * MIN_ROOT] {
        let got = trace(
            f.tape(),
            None,
            [0.0; 3],
            [1.0, 0.0, 0.0],
            None,
            false,
            t_max,
        );
        // None up to the floor itself; else the first point strictly
        // above, within the cap.
        if t_max == MIN_ROOT {
            assert_eq!(got.crossing, None);
        } else {
            assert!(
                got.crossing
                    .is_some_and(|t| t > MIN_ROOT && t <= t_max && f.evaluate([t, 0.0, 0.0]) > 0.0),
                "t_max {t_max}: {:?}",
                got.crossing
            );
        }
    }
}

/// With no bounding box and no cap, the prototype's search panicked. Now
/// it stops where the formula stops being checked, `WORLD` from the
/// origin.
#[test]
fn an_unbounded_search_ends() {
    let gyroid = ImplicitFormula::new(GYROID).unwrap();
    assert_eq!(gyroid.bounding_box(), None);
    let mut rng = Rng(9);
    for _ in 0..50 {
        let o = uniform_in(&mut rng, ([-3.0; 3], [3.0; 3]));
        let u = rng.direction();
        let t = gyroid
            .trace(o, u, None, false, f64::INFINITY)
            .crossing
            .expect("a gyroid is crossed within a period or two");
        let mut probe = Probe::new(gyroid.tape(), o, u);
        let (_, first) = probe.crossings(MIN_ROOT, t + 2e-3, 1e-3, 0.0, 1);
        assert!(
            first
                .first()
                .is_some_and(|&y| (t - y).abs() <= 1e-9 * y.max(1.0)),
            "from {o:?} along {u:?}: {t} vs {first:?}"
        );
    }
    // A plane and a wavy sheet the ray never meets: nothing, quickly.
    for (formula, o, u) in [
        ("x - 1", [0.0; 3], [0.0, 1.0, 0.0]),
        ("z - 0.5 * sin(x)", [0.0, 0.0, 5.0], [1.0, 0.0, 0.0]),
    ] {
        let f = ImplicitFormula::new(formula).unwrap();
        let got = trace(f.tape(), None, o, u, None, false, f64::INFINITY);
        assert_eq!(got.crossing, None, "{formula}");
        assert!(
            !got.exhausted && got.evaluations < 100,
            "{formula}: {}",
            got.evaluations
        );
    }
}

/// The kind as a surface: its name, value, box, crossings in the caller's
/// units, the refusal to offset it, and its JSON, which shares one formula
/// between the surfaces loaded from the same text.
#[test]
fn an_implicit_surface_is_an_ordinary_surface() {
    use crate::surface::{BoundaryType, SurfaceKind};
    let formula = Arc::new(ImplicitFormula::new("x^2 + y^2 + z^2 - 4").unwrap());
    let s = Surface::new_implicit(Arc::clone(&formula), Some(3), Some(BoundaryType::Vacuum));
    assert_eq!(s.kind.name(), "ImplicitSurface");
    assert!(s.is_free_form());
    let p = (0.5, -1.0, 0.25);
    assert_eq!(s.evaluate(p), formula.evaluate([0.5, -1.0, 0.25]));
    assert_eq!(s.evaluate_exact(p), s.evaluate(p));
    assert_eq!(s.evaluate((0.0, 0.0, 0.0)), -4.0);
    let (lo, hi) = s.bounding_box(true).unwrap();
    assert!((0..3).all(|k| lo[k] <= -2.0 && lo[k] > -2.1 && hi[k] >= 2.0 && hi[k] < 2.1));
    assert_eq!(s.bounding_box(false), None);
    // Crossings come back in the units of the direction.
    let t = s.distance_to_surface([0.0; 3], [0.0, 0.0, 2.0]).unwrap();
    assert!((t - 1.0).abs() < 1e-12, "{t}");
    assert_eq!(s.distance_to_surface([0.0; 3], [0.0; 3]), None);
    assert_eq!(
        s.distance_to_surface_capped([0.0; 3], [1.0, 0.0, 0.0], None, 1.5),
        None
    );
    let refused = s.offset(1.0, None, None).unwrap_err();
    assert!(
        refused.contains("ImplicitSurface cannot be offset"),
        "{refused}"
    );
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(
        json,
        r#"{"surface_id":3,"kind":{"Implicit":{"formula":"x^2 + y^2 + z^2 - 4"}},"boundary":"Vacuum"}"#
    );
    let a: Surface = serde_json::from_str(&json).unwrap();
    let b: Surface =
        serde_json::from_str(&json.replace("\"surface_id\":3", "\"surface_id\":4")).unwrap();
    match (&a.kind, &b.kind) {
        (SurfaceKind::Implicit { formula: fa }, SurfaceKind::Implicit { formula: fb }) => {
            assert!(Arc::ptr_eq(fa, fb));
            assert_eq!(fa.formula(), formula.formula());
        }
        _ => panic!("{json}"),
    }
    let bad = serde_json::from_str::<Surface>(&json.replace("x^2", "x^")).unwrap_err();
    assert!(bad.to_string().contains("column"), "{bad}");
}

/// A search that runs out of evaluations has found no crossing. Transport,
/// whose cell caps the search, gets the point it reached on a transmission
/// surface, but a query with no cap gets no crossing rather than a point
/// the surface does not cross: here along a twisted bar the ray never
/// meets, and along a plane in a formula that interval arithmetic cannot
/// see is flat.
#[test]
fn an_uncapped_search_that_runs_out_finds_no_crossing() {
    for (formula, o, u) in [
        (
            "((x * cos(0.6 * z) + y * sin(0.6 * z)) / 2.5)^4 \
             + ((y * cos(0.6 * z) - x * sin(0.6 * z)) / 2.5)^4 - 1",
            [6.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ),
        ("x^2 - x^2 + y - 1", [0.0; 3], [1.0, 0.0, 0.0]),
    ] {
        let f = Arc::new(ImplicitFormula::new(formula).unwrap());
        let reached = trace(&f.tape, None, o, u, None, false, f64::INFINITY);
        assert!(reached.exhausted && reached.crossing.is_some(), "{formula}");
        let s = Surface::new_implicit(f, Some(1), None);
        assert_eq!(s.distance_to_surface(o, u), None, "{formula}");
        assert_eq!(s.distance_to_surface_on(o, u, None), None, "{formula}");
        assert_eq!(
            s.distance_to_surface_capped(o, u, None, f64::INFINITY),
            reached.crossing,
            "{formula}"
        );
    }
}

/// Single-thread cost per query on each shape, for transport-like and
/// aimed rays, against the closed form where yamc-geo has one (the best of
/// five passes, for a shared machine); run it in release with `cargo test
/// --release -p yamc-geo --lib ray_timings -- --ignored --nocapture`.
#[test]
#[ignore]
fn ray_timings() {
    use std::hint::black_box;
    use std::time::Instant;
    /// Microseconds a case, the best of five passes.
    fn best(cases: &[Case], mut query: impl FnMut(&Case)) -> f64 {
        (0..5)
            .map(|_| {
                let start = Instant::now();
                cases.iter().for_each(&mut query);
                start.elapsed().as_secs_f64() * 1e6 / cases.len() as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
    let n = 20_000;
    println!(
        "| shape | instructions | rays | evaluations a ray | us a ray | closed form us a ray |"
    );
    println!("|---|---|---|---|---|---|");
    for name in [
        "sphere",
        "ztorus",
        "superellipsoid",
        "smooth union",
        "twisted box",
        "gyroid",
        "holes block",
    ] {
        let s = shape(name);
        for (set, cases) in [
            ("transport", transport(&s, n, 1)),
            ("aimed", aimed(&s, n / 4, 2)),
        ] {
            let evaluations: usize = cases.iter().map(|c| query(&s, c).evaluations).sum();
            let us = best(&cases, |c| {
                black_box(query(&s, black_box(c)));
            });
            let closed = s.closed.as_ref().map_or(String::new(), |closed| {
                let us = best(&cases, |c| {
                    black_box(closed.distance_to_surface_on(black_box(c.o), c.u, c.hint));
                });
                format!("{us:.3}")
            });
            println!(
                "| {name} {set} | {} | {} | {:.1} | {us:.2} | {closed} |",
                s.formula.tape().len(),
                cases.len(),
                evaluations as f64 / cases.len() as f64,
            );
        }
    }
}
