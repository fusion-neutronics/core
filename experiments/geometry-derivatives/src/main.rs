//! Prototype: forward-mode geometry derivative of a track-length reaction-rate
//! tally, one-group physics, on the real yamc-geo surface kernels.
//!
//! At each crossing of a differentiated interface (A -> B) at flight distance s,
//! with delta = ds/dtheta = -f_theta / (Omega . grad f):
//!   direct : w delta (R_A - R_B)                (track-length integrand jump)
//!   pass   : main particle's derivative weight += w delta (S_B - S_A)
//!   branch : forced collision in A, weight +w delta S_A; in B, weight -w delta S_B
use std::sync::Arc;
use yamc_geo::surface::{FluxBoundary, ImplicitFormula};
use yamc_geo::{BoundaryType, HalfspaceType, Region, Surface};

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03);
        r.u();
        r
    }
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn dir(&mut self) -> [f64; 3] {
        let mu = 2.0 * self.u() - 1.0;
        let phi = std::f64::consts::TAU * self.u();
        let s = (1.0 - mu * mu).sqrt();
        [s * phi.cos(), s * phi.sin(), mu]
    }
}

#[derive(Clone, Copy)]
struct Mat {
    st: f64, // total
    c: f64,  // scattering ratio
    sr: f64, // response cross section
}
const VOID: Mat = Mat {
    st: 0.0,
    c: 0.0,
    sr: 0.0,
};

struct Cell {
    region: Region,
    surfs: Vec<Arc<Surface>>,
    mat: Mat,
    tally: bool,
}
type Delta = Box<dyn Fn(usize, [f64; 3], [f64; 3]) -> f64 + Sync + Send>;
type Source = Box<dyn Fn(&mut Rng) -> [f64; 3] + Sync + Send>;
struct Model {
    cells: Vec<Cell>,
    source: Source,
    /// One per parameter: ds/dtheta at a crossing of surface `id` at x along dir.
    params: Vec<Delta>,
}
fn cell(region: Region, mat: Mat, tally: bool) -> Cell {
    let mut surfs: Vec<Arc<Surface>> = Vec::new();
    for (s, _) in region.surfaces_with_sense() {
        if !surfs.iter().any(|t| Arc::ptr_eq(t, &s)) {
            surfs.push(s);
        }
    }
    Cell {
        region,
        surfs,
        mat,
        tally,
    }
}
fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}
fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}
fn add(p: [f64; 3], d: [f64; 3], t: f64) -> [f64; 3] {
    [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

struct P {
    pos: [f64; 3],
    dir: [f64; 3],
    w: f64,
    d: Vec<f64>,
}
const EPS: f64 = 1e-7;
static DMAX: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
static NB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

impl Model {
    fn find(&self, p: [f64; 3]) -> Option<usize> {
        self.cells
            .iter()
            .position(|c| c.region.contains((p[0], p[1], p[2])))
    }
    /// One history: (score, d score / d theta_k, lost).
    fn history(&self, rng: &mut Rng, derivs: bool) -> (f64, Vec<f64>, bool) {
        let k = self.params.len();
        let mut score = 0.0;
        let mut ds = vec![0.0; k];
        let mut lost = false;
        let mut stack = vec![P {
            pos: (self.source)(rng),
            dir: rng.dir(),
            w: 1.0,
            d: vec![0.0; k],
        }];
        while let Some(mut p) = stack.pop() {
            let mut steps = 0;
            loop {
                steps += 1;
                if steps > 100_000 {
                    lost = true;
                    break;
                }
                let Some(ci) = self.find(p.pos) else { break };
                let c = &self.cells[ci];
                let dc = if c.mat.st > 0.0 {
                    -rng.u().ln() / c.mat.st
                } else {
                    f64::INFINITY
                };
                let mut best: Option<(f64, usize)> = None;
                for s in &c.surfs {
                    if let Some(t) = s.distance_to_surface(p.pos, p.dir) {
                        if t > 1e-10 && best.is_none_or(|b| t < b.0) {
                            best = Some((t, s.surface_id.unwrap()));
                        }
                    }
                }
                let track = |len: f64, p: &P, score: &mut f64, ds: &mut [f64]| {
                    if c.tally {
                        *score += p.w * c.mat.sr * len;
                        for (a, b) in ds.iter_mut().zip(&p.d) {
                            *a += b * c.mat.sr * len;
                        }
                    }
                };
                match best {
                    Some((t, sid)) if t < dc => {
                        track(t, &p, &mut score, &mut ds);
                        let x = add(p.pos, p.dir, t);
                        let next = add(p.pos, p.dir, t + EPS);
                        let ni = self.find(next);
                        if derivs && p.w != 0.0 && ni != Some(ci) {
                            let (a, ta) = (c.mat, c.tally);
                            let (b, tb) = match ni {
                                Some(i) => (self.cells[i].mat, self.cells[i].tally),
                                None => (VOID, false),
                            };
                            let mut ba = vec![0.0; k];
                            let mut bb = vec![0.0; k];
                            let (mut any_a, mut any_b) = (false, false);
                            for q in 0..k {
                                let delta = (self.params[q])(sid, x, p.dir);
                                let dmax = *DMAX.get_or_init(|| {
                                    std::env::var("DMAX")
                                        .ok()
                                        .and_then(|s| s.parse().ok())
                                        .unwrap_or(f64::INFINITY)
                                });
                                let delta = delta.clamp(-dmax, dmax);
                                if delta == 0.0 {
                                    continue;
                                }
                                let ra = if ta { a.sr } else { 0.0 };
                                let rb = if tb { b.sr } else { 0.0 };
                                ds[q] += p.w * delta * (ra - rb);
                                p.d[q] += p.w * delta * (b.st - a.st);
                                ba[q] = p.w * delta * a.st * a.c;
                                bb[q] = -p.w * delta * b.st * b.c;
                                any_a |= ba[q] != 0.0;
                                any_b |= bb[q] != 0.0;
                            }
                            // Forced collisions at x: scatter isotropically, then
                            // ordinary transport in the nominal geometry.
                            let nb = *NB.get_or_init(|| {
                                std::env::var("NB")
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(1)
                            });
                            for (any, d) in [(any_a, ba), (any_b, bb)] {
                                if any {
                                    for _ in 0..nb {
                                        let dir = rng.dir();
                                        let d = d.iter().map(|v| v / nb as f64).collect();
                                        stack.push(P {
                                            pos: add(x, dir, EPS),
                                            dir,
                                            w: 0.0,
                                            d,
                                        });
                                    }
                                }
                            }
                        }
                        p.pos = next;
                    }
                    _ => {
                        if dc.is_infinite() {
                            lost = true; // void with no surface ahead
                            break;
                        }
                        track(dc, &p, &mut score, &mut ds);
                        p.pos = add(p.pos, p.dir, dc);
                        if rng.u() < c.mat.c {
                            p.dir = rng.dir();
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        (score, ds, lost)
    }
}

struct Stat {
    mean: f64,
    err: f64,
}
/// Run n histories over all cores; mean and standard error of the score and of
/// each derivative.
fn run(model: &Model, n: u64, seed: u64, derivs: bool) -> (Stat, Vec<Stat>, u64, f64) {
    let k = model.params.len();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4) as u64;
    let t0 = std::time::Instant::now();
    let parts: Vec<(Vec<f64>, Vec<f64>, u64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut sum = vec![0.0; k + 1];
                    let mut sq = vec![0.0; k + 1];
                    let mut lost = 0;
                    let mut i = t;
                    while i < n {
                        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(i));
                        let (sc, d, l) = model.history(&mut rng, derivs);
                        lost += l as u64;
                        sum[0] += sc;
                        sq[0] += sc * sc;
                        for q in 0..k {
                            sum[q + 1] += d[q];
                            sq[q + 1] += d[q] * d[q];
                        }
                        i += threads;
                    }
                    (sum, sq, lost)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut sum = vec![0.0; k + 1];
    let mut sq = vec![0.0; k + 1];
    let mut lost = 0;
    for (a, b, l) in parts {
        lost += l;
        for q in 0..=k {
            sum[q] += a[q];
            sq[q] += b[q];
        }
    }
    let nf = n as f64;
    let stat = |q: usize| {
        let m = sum[q] / nf;
        Stat {
            mean: m,
            err: ((sq[q] / nf - m * m).max(0.0) / nf).sqrt(),
        }
    };
    (
        stat(0),
        (1..=k).map(stat).collect(),
        lost,
        t0.elapsed().as_secs_f64(),
    )
}

/// Compare the derivative estimator at theta0 with a central finite difference.
fn check(
    name: &str,
    build: &dyn Fn(&[f64]) -> Model,
    theta0: &[f64],
    h: &[f64],
    labels: &[&str],
    n: u64,
    n_fd: u64,
) {
    println!("\n=== {name} ===");
    let m = build(theta0);
    let (r0, _, _, t_plain) = run(&m, n, 1, false);
    let (r, d, lost, t_deriv) = run(&m, n, 1, true);
    println!(
        "response {:.6} +/- {:.6}  ({n} histories, {lost} lost, {:.1}s; without derivatives {:.6}, {:.1}s; cost x{:.2})",
        r.mean, r.err, t_deriv, r0.mean, t_plain, t_deriv / t_plain
    );
    for (q, label) in labels.iter().enumerate() {
        let mut tp = theta0.to_vec();
        let mut tm = theta0.to_vec();
        tp[q] += h[q];
        tm[q] -= h[q];
        let (rp, _, _, _) = run(&build(&tp), n_fd, 11 + q as u64, false);
        let (rm, _, _, _) = run(&build(&tm), n_fd, 111 + q as u64, false);
        let fd = (rp.mean - rm.mean) / (2.0 * h[q]);
        let fd_err = (rp.err * rp.err + rm.err * rm.err).sqrt() / (2.0 * h[q]);
        let z = (d[q].mean - fd) / (d[q].err * d[q].err + fd_err * fd_err).sqrt();
        println!(
            "d/d {label:<28} estimator {:+.4e} +/- {:.1e} ({:.1}%)   finite diff (h={}) {:+.4e} +/- {:.1e}   z={:+.2}",
            d[q].mean, d[q].err, 100.0 * d[q].err / d[q].mean.abs(), h[q], fd, fd_err, z
        );
    }
}

fn sphere(r: f64, id: usize, vacuum: bool) -> Arc<Surface> {
    Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        r,
        Some(id),
        vacuum.then_some(BoundaryType::Vacuum),
    ))
}

const WALL: Mat = Mat {
    st: 0.35,
    c: 0.9,
    sr: 0.0,
};
const BREEDER: Mat = Mat {
    st: 0.10,
    c: 0.8,
    sr: 0.02,
};

/// Concentric spheres: void | wall | breeder (tallied) | void. theta = [r1, r2, r3].
fn spheres(theta: &[f64]) -> Model {
    let (s1, s2, s3, w) = (
        sphere(theta[0], 1, false),
        sphere(theta[1], 2, false),
        sphere(theta[2], 3, false),
        sphere(200.0, 4, true),
    );
    let cells = vec![
        cell(below(&s1), VOID, false),
        cell(above(&s1).intersection(&below(&s2)), WALL, false),
        cell(above(&s2).intersection(&below(&s3)), BREEDER, true),
        cell(above(&s3).intersection(&below(&w)), VOID, false),
    ];
    // A sphere of radius r: f = |x| - r, f_theta = -1, grad f = x/|x|.
    let params: Vec<Delta> = (1..=3)
        .map(|id| -> Delta {
            Box::new(move |sid, x, dir| {
                if sid != id {
                    return 0.0;
                }
                let r = dot(x, x).sqrt();
                r / dot(dir, x)
            })
        })
        .collect();
    Model {
        cells,
        source: Box::new(|_| [0.0; 3]),
        params,
    }
}

/// A purely absorbing shell, where the answer is analytic.
fn absorber(theta: &[f64]) -> Model {
    let (s1, s2, w) = (
        sphere(10.0, 1, false),
        sphere(theta[0], 2, false),
        sphere(200.0, 4, true),
    );
    let m = Mat {
        st: 0.1,
        c: 0.0,
        sr: 0.1,
    };
    let cells = vec![
        cell(below(&s1), VOID, false),
        cell(above(&s1).intersection(&below(&s2)), m, true),
        cell(above(&s2).intersection(&below(&w)), VOID, false),
    ];
    let params: Vec<Delta> = vec![Box::new(|sid, x, dir| {
        if sid == 2 {
            dot(x, x).sqrt() / dot(dir, x)
        } else {
            0.0
        }
    })];
    Model {
        cells,
        source: Box::new(|_| [0.0; 3]),
        params,
    }
}

/// Layers around a flux boundary: plasma | gap | wall | breeder (tallied) | void.
/// `offsets` = [wall front, wall back, breeder back]; `coefs` are boundary
/// coefficients to differentiate: (is_z, index into r_mn / z_mn).
#[allow(clippy::too_many_arguments)]
fn layers(
    nfp: u32,
    r_modes: &[[i32; 2]],
    r_mn: &[f64],
    z_modes: &[[i32; 2]],
    z_mn: &[f64],
    offsets: [f64; 3],
    coefs: &[(bool, usize)],
    axis_r: f64,
    world: f64,
) -> Model {
    let b = Arc::new(
        FluxBoundary::new(
            nfp,
            r_modes.to_vec(),
            r_mn.to_vec(),
            z_modes.to_vec(),
            z_mn.to_vec(),
        )
        .unwrap(),
    );
    let plasma = Arc::new(Surface::new_flux(b.clone(), Some(1), None));
    let s: Vec<Arc<Surface>> = (0..3)
        .map(|i| Arc::new(plasma.offset(offsets[i], Some(2 + i), None).unwrap()))
        .collect();
    let w = sphere(world, 9, true);
    let cells = vec![
        cell(below(&plasma), VOID, false),
        cell(above(&plasma).intersection(&below(&s[0])), VOID, false),
        cell(above(&s[0]).intersection(&below(&s[1])), WALL, false),
        cell(above(&s[1]).intersection(&below(&s[2])), BREEDER, true),
        cell(above(&s[2]).intersection(&below(&w)), VOID, false),
    ];
    let mut params: Vec<Delta> = Vec::new();
    // Offset d of a level set of the signed distance D: f = D - d, f_theta = -1,
    // grad f = n (unit, at the closest boundary point), so ds/dd = 1 / (Omega . n).
    for id in 2..=4 {
        let b = b.clone();
        params.push(Box::new(move |sid, x, dir| {
            if sid != id {
                return 0.0;
            }
            let (_, t, z) = b.closest_point(x);
            1.0 / dot(dir, b.normal(t, z))
        }));
    }
    // A boundary coefficient c: every layer moves with the boundary, and the
    // normal velocity of a level set at x is n . d x_b / dc at x's closest
    // boundary point (the offset along n adds nothing to first order).
    for &(is_z, i) in coefs {
        let b = b.clone();
        let [m, n] = if is_z { z_modes[i] } else { r_modes[i] };
        let nfp = nfp as f64;
        params.push(Box::new(move |sid, x, dir| {
            if !(1..=4).contains(&sid) {
                return 0.0;
            }
            let (_, t, z) = b.closest_point(x);
            let tm = if m >= 0 {
                (m as f64 * t).cos()
            } else {
                (-m as f64 * t).sin()
            };
            let fn_ = if n >= 0 {
                (n as f64 * nfp * z).cos()
            } else {
                (-n as f64 * nfp * z).sin()
            };
            let dx = if is_z {
                [0.0, 0.0, tm * fn_]
            } else {
                [tm * fn_ * z.cos(), tm * fn_ * z.sin(), 0.0]
            };
            let nrm = b.normal(t, z);
            dot(nrm, dx) / dot(dir, nrm)
        }));
    }
    let source: Source = Box::new(move |rng| {
        let z = std::f64::consts::TAU * rng.u();
        [axis_r * z.cos(), axis_r * z.sin(), 0.0]
    });
    Model {
        cells,
        source,
        params,
    }
}

fn tokamak(theta: &[f64]) -> Model {
    // theta = [wall front, wall thickness, breeder thickness, major radius, elongation]
    let shape =
        yamc_source::tokamak::TokamakShape::new(theta[3], 292.258, theta[4], 0.270, 0.44789)
            .unwrap();
    let (rm, r, zm, z) = shape.surface_modes(1.0);
    let offsets = [
        theta[0],
        theta[0] + theta[1],
        theta[0] + theta[1] + theta[2],
    ];
    let mut m = layers(
        1,
        &rm,
        &r,
        &zm,
        &z,
        offsets,
        &[(false, 0), (true, 0)],
        906.0,
        1500.0,
    );
    chain_thickness(&mut m, 292.258);
    m
}

/// Turn the three offset parameters into (wall front, wall thickness, breeder
/// thickness), a later layer moving with every thickness before it; and the
/// Z_(-1,0) coefficient into elongation (Z_(-1,0) = elongation * minor radius).
fn chain_thickness(m: &mut Model, z_scale: f64) {
    let mut old: Vec<Arc<Delta>> = std::mem::take(&mut m.params)
        .into_iter()
        .map(Arc::new)
        .collect();
    let (o1, o2, o3) = (old[0].clone(), old[1].clone(), old[2].clone());
    let rest: Vec<Arc<Delta>> = old.drain(3..).collect();
    {
        let (a, b, c) = (o1.clone(), o2.clone(), o3.clone());
        m.params.push(Box::new(move |s, x, d| {
            a(s, x, d) + b(s, x, d) + c(s, x, d)
        }));
    }
    {
        let (b, c) = (o2.clone(), o3.clone());
        m.params
            .push(Box::new(move |s, x, d| b(s, x, d) + c(s, x, d)));
    }
    m.params.push(Box::new(move |s, x, d| o3(s, x, d)));
    for (i, p) in rest.into_iter().enumerate() {
        let scale = if i == 1 { z_scale } else { 1.0 };
        m.params.push(Box::new(move |s, x, d| scale * p(s, x, d)));
    }
}

fn stellarator(theta: &[f64]) -> Model {
    // Rotating ellipse, 5 field periods (the boundary of stellarator_csg_layers.py).
    // theta = [wall front, wall thickness, breeder thickness, R_(0,0), R_(1,1) helical]
    let (a, e) = (150.0, 50.0);
    let rm = [[0, 0], [1, 0], [1, 1], [-1, -1]];
    let r = [theta[3], a, theta[4], e];
    let zm = [[-1, 0], [-1, 1], [1, -1]];
    let z = [a, -e, e];
    let offsets = [
        theta[0],
        theta[0] + theta[1],
        theta[0] + theta[1] + theta[2],
    ];
    let mut m = layers(
        5,
        &rm,
        &r,
        &zm,
        &z,
        offsets,
        &[(false, 0), (false, 2)],
        600.0,
        1200.0,
    );
    chain_thickness(&mut m, 1.0);
    m
}

/// A gyroid-lattice breeder block (as examples/python/implicit_surfaces.py):
/// steel sheet |gyroid| < t inside a superellipsoid module of half-width a.
/// theta = [t, a]. The breeder channels are tallied.
fn gyroid(theta: &[f64]) -> Model {
    let lat = |t: f64| {
        format!(
            "abs(sin(pi / 2 * x) * cos(pi / 2 * y) + sin(pi / 2 * y) * cos(pi / 2 * z) + sin(pi / 2 * z) * cos(pi / 2 * x)) - {t}"
        )
    };
    let module = |a: f64| format!("((x - 21) / {a})^8 + (y / {a})^8 + (z / {a})^8 - 1");
    let f = |text: String| Arc::new(ImplicitFormula::new(&text).unwrap());
    let steel = Mat {
        st: 0.25,
        c: 0.85,
        sr: 0.0,
    };
    let lattice = Arc::new(Surface::new_implicit(f(lat(theta[0])), Some(1), None));
    let modsurf = Arc::new(Surface::new_implicit(f(module(theta[1])), Some(2), None));
    let w = sphere(100.0, 9, true);
    let inside = below(&modsurf);
    let cells = vec![
        cell(inside.intersection(&below(&lattice)), steel, false),
        cell(inside.intersection(&above(&lattice)), BREEDER, true),
        cell(above(&modsurf).intersection(&below(&w)), VOID, false),
    ];
    // Any formula f(x; theta): ds/dtheta = -f_theta / (Omega . grad f). Here both
    // derivatives are differenced; the real thing reads them off the tape's duals.
    let implicit = |id: usize,
                    nominal: Arc<ImplicitFormula>,
                    plus: Arc<ImplicitFormula>,
                    minus: Arc<ImplicitFormula>,
                    e: f64|
     -> Delta {
        Box::new(move |sid, x, dir| {
            if sid != id {
                return 0.0;
            }
            let f_theta = (plus.evaluate(x) - minus.evaluate(x)) / (2.0 * e);
            let h = 1e-5;
            let along =
                (nominal.evaluate(add(x, dir, h)) - nominal.evaluate(add(x, dir, -h))) / (2.0 * h);
            -f_theta / along
        })
    };
    let e = 1e-4;
    let params = vec![
        implicit(
            1,
            f(lat(theta[0])),
            f(lat(theta[0] + e)),
            f(lat(theta[0] - e)),
            e,
        ),
        implicit(
            2,
            f(module(theta[1])),
            f(module(theta[1] + e)),
            f(module(theta[1] - e)),
            e,
        ),
    ];
    Model {
        cells,
        source: Box::new(|_| [-20.0, 0.0, 0.0]),
        params,
    }
}

/// Shape parameters with a clear signal: plasma | gap | breeder (tallied) out to
/// a FIXED torus, so changing the boundary changes the breeder's thickness.
#[allow(clippy::too_many_arguments)]
fn shaped(
    nfp: u32,
    r_modes: &[[i32; 2]],
    r_mn: &[f64],
    z_modes: &[[i32; 2]],
    z_mn: &[f64],
    coefs: &[(bool, usize, f64)],
    torus: [f64; 3],
    axis_r: f64,
    world: f64,
) -> Model {
    let plain: Vec<(bool, usize)> = coefs.iter().map(|c| (c.0, c.1)).collect();
    let mut m = layers(
        nfp,
        r_modes,
        r_mn,
        z_modes,
        z_mn,
        [15.0, 16.0, 17.0],
        &plain,
        axis_r,
        world,
    );
    let front = m.cells[2]
        .surfs
        .iter()
        .find(|s| s.surface_id == Some(2))
        .unwrap()
        .clone();
    let plasma = m.cells[0].surfs[0].clone();
    let back = Arc::new(Surface::new_ztorus(
        0.0,
        0.0,
        0.0,
        torus[0],
        torus[1],
        torus[2],
        Some(8),
        None,
    ));
    let w = sphere(world, 9, true);
    m.cells = vec![
        cell(below(&plasma), VOID, false),
        cell(above(&plasma).intersection(&below(&front)), VOID, false),
        cell(above(&front).intersection(&below(&back)), BREEDER, true),
        cell(above(&back).intersection(&below(&w)), VOID, false),
    ];
    let shape: Vec<Delta> = m.params.drain(3..).collect();
    m.params = shape
        .into_iter()
        .zip(coefs)
        .map(|(p, c)| -> Delta {
            let scale = c.2;
            Box::new(move |s, x, d| scale * p(s, x, d))
        })
        .collect();
    m
}
fn tokamak_shape(theta: &[f64]) -> Model {
    // theta = [major radius, elongation, minor-radius-like R_(1,0) coefficient offset]
    let shape =
        yamc_source::tokamak::TokamakShape::new(theta[0], 292.258, theta[1], 0.270, 0.44789)
            .unwrap();
    let (rm, mut r, zm, z) = shape.surface_modes(1.0);
    r[1] += theta[2];
    shaped(
        1,
        &rm,
        &r,
        &zm,
        &z,
        &[(false, 0, 1.0), (true, 0, 292.258), (false, 1, 1.0)],
        [906.0, 490.0, 330.0],
        906.0,
        1500.0,
    )
}
fn stellarator_shape(theta: &[f64]) -> Model {
    // theta = [R_(0,0), R_(1,0), R_(1,1) helical, Z_(-1,1) helical]
    let e = 50.0;
    let rm = [[0, 0], [1, 0], [1, 1], [-1, -1]];
    let r = [theta[0], theta[1], theta[2], e];
    let zm = [[-1, 0], [-1, 1], [1, -1]];
    let z = [150.0, theta[3], e];
    shaped(
        5,
        &rm,
        &r,
        &zm,
        &z,
        &[
            (false, 0, 1.0),
            (false, 1, 1.0),
            (false, 2, 1.0),
            (true, 1, 1.0),
        ],
        [600.0, 235.0, 235.0],
        600.0,
        1200.0,
    )
}

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let on = |name: &str| which.is_empty() || which.iter().any(|w| w == name);
    let scale: f64 = std::env::var("SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let n = |x: f64| (x * scale) as u64;
    if on("absorber") {
        check("purely absorbing shell 10..30 cm (analytic: R = 1 - exp(-2) = 0.864665, dR/dr2 = 0.1 exp(-2) = 1.3534e-2)",
            &absorber, &[30.0], &[0.5], &["outer radius"], n(4e6), n(4e6));
    }
    if on("spheres") {
        check(
            "concentric spheres: void | wall 100..102 | breeder 102..152",
            &spheres,
            &[100.0, 102.0, 152.0],
            &[1.0, 0.25, 2.0],
            &[
                "wall inner radius",
                "wall/breeder radius",
                "breeder outer radius",
            ],
            n(4e6),
            n(2e7),
        );
    }
    if on("tokamak") {
        check(
            "tokamak (EU-DEMO-like TokamakShape), offset layers",
            &tokamak,
            &[15.0, 2.0, 50.0, 906.0, 1.557],
            &[2.0, 0.25, 2.0, 10.0, 0.03],
            &[
                "wall front offset",
                "wall thickness",
                "breeder thickness",
                "major radius",
                "elongation",
            ],
            n(1e6),
            n(1.5e6),
        );
    }
    if on("stellarator") {
        check(
            "stellarator (rotating ellipse, 5 field periods), offset layers",
            &stellarator,
            &[15.0, 2.0, 50.0, 600.0, 50.0],
            &[2.0, 0.25, 2.0, 5.0, 3.0],
            &[
                "wall front offset",
                "wall thickness",
                "breeder thickness",
                "R_(0,0)",
                "R_(1,1) helical",
            ],
            n(1e6),
            n(1.5e6),
        );
    }
    if on("tokamak-shape") {
        check(
            "tokamak shape parameters, breeder from the plasma-conforming front to a fixed torus",
            &tokamak_shape,
            &[906.0, 1.557, 0.0],
            &[8.0, 0.03, 8.0],
            &["major radius", "elongation", "R_(1,0) (minor radius)"],
            n(1e6),
            n(1.5e6),
        );
    }
    if on("stellarator-shape") {
        check("stellarator Fourier coefficients, breeder from the plasma-conforming front to a fixed torus", &stellarator_shape, &[600.0, 150.0, 50.0, -50.0],
            &[8.0, 8.0, 8.0, 8.0], &["R_(0,0)", "R_(1,0)", "R_(1,1) helical", "Z_(-1,1) helical"], n(1e6), n(1.5e6));
    }
    if on("gyroid") {
        check(
            "implicit: gyroid steel sheet in a superellipsoid module",
            &gyroid,
            &[0.3, 20.0],
            &[0.03, 0.5],
            &["sheet level t", "module half-width a"],
            n(1e6),
            n(1.5e6),
        );
    }
}
