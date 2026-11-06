//! The browser views evaluate CSG surfaces in JavaScript, in three copies of
//! `surfaceEval`: viewer.js (cell fills and hover), the CSG worker in
//! viewer_html.rs (cell fills) and tally_viewer.js (outlines and hover). For
//! every surface kind each copy must give the sign of Rust's
//! `Surface::evaluate` (else cells are drawn with the wrong fill), and the two
//! that hover a value hover can compare with half a pixel. For a kind it does
//! not know it must give NaN, which is never filled or hovered. The browser
//! app's surface editor (app.js) must give every kind fields and write an
//! unedited surface back as it was, in JSON that Rust loads.
//!
//! The JS runs under Node. Without `node` on PATH these tests fail rather than
//! skip: every CI runner that runs them ships Node, and a skip would pass
//! having checked nothing.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use yamc_geo::surface::{Surface, SurfaceKind};
use yamc_plot::{
    build_interactive_html, build_interactive_tally_html, EmbeddedSlice, InteractiveTallyParams,
    InteractiveViewParams, MeshMeta,
};

const APP_JS: &str =
    include_str!("../../../packages/yamc-core/python/yamc/_browser_template/app.js");

/// Every `SurfaceKind` as named in the JSON, from serde's own list (its error
/// for an unknown kind names them all). A new kind then fails these tests
/// until it has cases below, a JS formula and editor fields.
fn all_kinds() -> HashSet<String> {
    let error = serde_json::from_str::<SurfaceKind>("{\"NotAKind\": {}}")
        .unwrap_err()
        .to_string();
    let (_, list) = error.split_once("expected one of ").unwrap();
    list.split('`')
        .skip(1)
        .step_by(2)
        .map(String::from)
        .collect()
}

/// Centre, symmetry axis and size of a surface: where its test points are
/// drawn, out to three sizes from the centre and along the axis.
fn frame(kind: &SurfaceKind) -> ([f64; 3], Option<[f64; 3]>, f64) {
    match *kind {
        SurfaceKind::Plane { a, b, c, d } => ([a * d, b * d, c * d], Some([a, b, c]), 10.0),
        SurfaceKind::Sphere { x0, y0, z0, radius } => ([x0, y0, z0], None, radius),
        SurfaceKind::Cylinder {
            axis,
            origin,
            radius,
        } => (origin, Some(axis), radius),
        SurfaceKind::ZTorus {
            x0, y0, z0, a, c, ..
        } => ([x0, y0, z0], Some([0.0, 0.0, 1.0]), a + c),
        SurfaceKind::XTorus {
            x0, y0, z0, a, c, ..
        } => ([x0, y0, z0], Some([1.0, 0.0, 0.0]), a + c),
        SurfaceKind::YTorus {
            x0, y0, z0, a, c, ..
        } => ([x0, y0, z0], Some([0.0, 1.0, 0.0]), a + c),
        SurfaceKind::Quadric { .. } => ([0.0; 3], None, 4.0),
        SurfaceKind::Cone { apex, axis, .. } => (apex, Some(axis), 5.0),
    }
}

fn quadric(c: [f64; 10]) -> Surface {
    Surface::new_quadric(
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], None, None,
    )
}

/// Surfaces of every kind, each with points inside it that are tested too and
/// have random points drawn around them (a torus or an ellipsoid fills too
/// little of the box around it to be bracketed otherwise). Most are where the
/// gradient vanishes: a centre, a torus's core circle.
fn cases() -> Vec<(Surface, Vec<[f64; 3]>)> {
    let third = 1.0 / 3.0;
    vec![
        (Surface::new_plane(0.6, 0.8, 0.0, 2.0, None, None), vec![]),
        (Surface::x_plane(3.0, None, None), vec![]),
        (Surface::new_sphere(1.0, -2.0, 0.5, 3.0, None, None), vec![]),
        (Surface::z_cylinder(1.0, 2.0, 2.0, None, None), vec![]),
        (
            Surface::new_cylinder([1.0, 1.0, 1.0], [0.5, -1.0, 2.0], 1.5, None, None),
            vec![],
        ),
        // A tokamak-sized elliptical torus, and a small circular one whose
        // core circle passes through (4, 6, 3) exactly (rho = 5 = a).
        (
            Surface::new_ztorus(0.0, 0.0, 0.0, 600.0, 250.0, 150.0, None, None),
            vec![[600.0, 0.0, 0.0], [0.0, -600.0, 0.0]],
        ),
        (
            Surface::new_ztorus(1.0, 2.0, 3.0, 5.0, 1.0, 1.0, None, None),
            vec![[4.0, 6.0, 3.0], [-2.0, -2.0, 3.0]],
        ),
        (
            Surface::new_xtorus(1.0, -2.0, 3.0, 5.0, 1.5, 2.0, None, None),
            vec![[1.0, 1.0, 7.0], [1.0, -5.0, -1.0]],
        ),
        (
            Surface::new_ytorus(-1.0, 2.0, 0.5, 4.0, 2.0, 1.0, None, None),
            vec![[-1.0, 2.0, 4.5], [3.0, 2.0, 0.5]],
        ),
        // A spindle torus (c > a) about x.
        (
            Surface::new_xtorus(0.0, 0.0, 0.0, 1.0, 1.5, 2.0, None, None),
            vec![[0.0, 1.0, 0.0]],
        ),
        // Ellipsoid, hyperboloid of one sheet with cross terms, paraboloid,
        // double cone (apex at the origin), plane and offset sphere.
        (
            quadric([0.25, 1.0 / 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            vec![[0.0, 0.0, 0.0]],
        ),
        (
            quadric([1.0, 1.0, -1.0, 0.5, -0.3, 0.2, 1.0, -2.0, 0.5, -4.0]),
            vec![],
        ),
        (
            quadric([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
            vec![[0.0, 0.0, 1.0]],
        ),
        (
            quadric([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vec![[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
        ),
        (
            quadric([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, -1.0, -4.0]),
            vec![],
        ),
        (
            quadric([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -2.0, 4.0, -1.0, -3.75]),
            vec![[1.0, -2.0, 0.5]],
        ),
        (Surface::z_cone(0.5, -1.0, 2.0, 0.25, None, None), vec![]),
        (Surface::x_cone(0.0, 0.0, 0.0, 1.0, None, None), vec![]),
        (
            Surface::new_cone(
                [1.0, -1.0, 0.5],
                [third, -2.0 * third, 2.0 * third],
                1.5,
                None,
                None,
            ),
            vec![],
        ),
    ]
}

/// The kind's name in the JSON, e.g. "Cone".
fn kind_name(kind: &SurfaceKind) -> String {
    let value = serde_json::to_value(kind).unwrap();
    value.as_object().unwrap().keys().next().unwrap().clone()
}

/// SplitMix64, for the same points on every run.
struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

fn eval(s: &Surface, p: [f64; 3]) -> f64 {
    s.evaluate((p[0], p[1], p[2]))
}

/// The side `evaluate` puts a point on: 1 above, -1 below, 0 on the surface
/// (neither halfspace holds it).
fn side(f: f64) -> i64 {
    if f > 0.0 {
        1
    } else if f < 0.0 {
        -1
    } else {
        0
    }
}

fn add(p: [f64; 3], t: f64, d: [f64; 3]) -> [f64; 3] {
    [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Two adjacent points on either side of the surface between `p` and `q`
/// (which `evaluate` puts on opposite sides), found by bisection.
fn bracket(s: &Surface, mut p: [f64; 3], mut q: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let sp = side(eval(s, p));
    for _ in 0..200 {
        let m = [
            0.5 * (p[0] + q[0]),
            0.5 * (p[1] + q[1]),
            0.5 * (p[2] + q[2]),
        ];
        if m == p || m == q {
            break;
        }
        match side(eval(s, m)) {
            0 => return (m, m),
            v if v == sp => p = m,
            _ => q = m,
        }
    }
    (p, q)
}

/// Points around the surface of `case`: random ones out to three sizes from
/// its centre, its centre, axis and inside points, and points within 1e-12
/// to 1e-3 sizes of it, from bisections. Also returns a point from each
/// bisection, one on the surface.
fn points(case: &(Surface, Vec<[f64; 3]>), rng: &mut Rng) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let (surface, inside) = case;
    let (centre, axis, scale) = frame(&surface.kind);
    let mut box_point = |at: [f64; 3], half_width: f64| {
        let d = [rng.next(), rng.next(), rng.next()];
        add(at, half_width, d)
    };
    let mut cloud: Vec<[f64; 3]> = (0..100).map(|_| box_point(centre, 3.0 * scale)).collect();
    for &p in inside {
        cloud.extend((0..20).map(|_| box_point(p, 0.3 * scale)));
    }
    let mut out = cloud.clone();
    out.push(centre);
    out.extend(inside);
    if let Some(axis) = axis {
        out.extend((0..12).map(|_| add(centre, 3.0 * scale * rng.next(), axis)));
    }
    // Near the surface: the bracket itself, then off it along the segment.
    let on_side = |s: i64| -> Vec<[f64; 3]> {
        cloud
            .iter()
            .copied()
            .filter(|p| side(eval(surface, *p)) == s)
            .collect()
    };
    let (below, above) = (on_side(-1), on_side(1));
    assert!(
        !below.is_empty() && !above.is_empty(),
        "{:?}: no points on one side to bracket the surface with",
        surface.kind
    );
    let mut brackets = Vec::new();
    for i in 0..30 {
        let (p, q) = (below[i % below.len()], above[(7 * i) % above.len()]);
        let (lo, hi) = bracket(surface, p, q);
        let dir = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let dir = [dir[0] / norm(dir), dir[1] / norm(dir), dir[2] / norm(dir)];
        out.extend([lo, hi]);
        for delta in [1e-12, 1e-9, 1e-6, 1e-3] {
            out.push(add(lo, -delta * scale, dir));
            out.push(add(hi, delta * scale, dir));
        }
        brackets.push(lo);
    }
    (out, brackets)
}

fn view_params() -> InteractiveViewParams {
    InteractiveViewParams {
        origin: (0.0, 0.0, 0.0),
        width: (10.0, 10.0),
        total_pixels: 100,
        basis: "xy".into(),
        color_by: "cell".into(),
        outline: None,
        axis_units: "cm".into(),
        bbox_widths: [10.0, 10.0, 10.0],
        outline_color: "#000000".into(),
        outline_thickness: 1,
        n_samples: None,
        plane_tolerance: 1.0,
        source_color: "#ff0000".into(),
        source_size: 3,
    }
}

fn tally_params() -> InteractiveTallyParams {
    InteractiveTallyParams {
        initial_basis: "xy".into(),
        initial_slice_index: 0,
        colorscale: "Viridis".into(),
        log_scale: true,
        outline: Some("cell".into()),
        outline_color: "#000000".into(),
        outline_thickness: 1,
        axis_units: "cm".into(),
        outline_pixels: 40_000,
        title: "test".into(),
        colorbar_title: "flux".into(),
        scaling_factor: 1.0,
        font_size: 18,
        show_colorbar: true,
        display_value: "mean".into(),
    }
}

/// The JS that holds each `surfaceEval` copy, taken from the HTML the plots
/// ship, and whether it has the hover lookup `surfacesAtPoint`.
fn js_copies() -> Vec<(&'static str, String, bool)> {
    let html = build_interactive_html(
        "{\"cells\":[]}",
        "csg",
        "",
        &view_params(),
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
        None,
        false,
        None,
    );
    // The worker pool's source is a template literal ahead of viewer.js.
    let (worker, page) = html
        .split_once("const CSG_WORKER_SOURCE = `")
        .and_then(|(_, rest)| rest.split_once("`;"))
        .expect("no CSG_WORKER_SOURCE");
    let mesh = MeshMeta {
        lower_left: [-10.0, -10.0, -10.0],
        upper_right: [10.0, 10.0, 10.0],
        shape: [2, 2, 2],
        width: [10.0, 10.0, 10.0],
    };
    let slices = vec![EmbeddedSlice {
        basis: "xy".into(),
        bin_index: 0,
        data: vec![0.0; 4],
    }];
    let tally = build_interactive_tally_html(
        Some("{\"cells\":[]}"),
        Some("csg"),
        &mesh,
        &slices,
        &[],
        &tally_params(),
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
    );
    vec![
        ("viewer.js", page.to_string(), true),
        ("the CSG worker", worker.to_string(), false),
        ("tally_viewer.js", tally, true),
    ]
}

/// The top-level `function name(...) { ... }` of `js` (one starting a line),
/// up to its matching brace.
fn js_function<'a>(js: &'a str, name: &str) -> &'a str {
    let start = 1 + js
        .find(&format!("\nfunction {name}("))
        .unwrap_or_else(|| panic!("no function {name}"));
    let open = start + js[start..].find('{').unwrap();
    let mut depth = 0;
    js[open..]
        .char_indices()
        .find_map(|(i, ch)| {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            (depth == 0).then(|| &js[start..open + i + 1])
        })
        .unwrap_or_else(|| panic!("function {name} is not closed"))
}

/// `roots` and every top-level function of `js` they call, transitively, as
/// JS that defines them.
fn js_with(js: &str, roots: &[&str]) -> String {
    let names: Vec<&str> = js
        .match_indices("\nfunction ")
        .map(|(at, _)| {
            let rest = &js[at + "\nfunction ".len()..];
            rest[..rest.find('(').unwrap()].trim()
        })
        .collect();
    let mut wanted: Vec<&str> = roots.to_vec();
    let mut i = 0;
    while i < wanted.len() {
        let text = js_function(js, wanted[i]);
        for &name in &names {
            let called = text.match_indices(&format!("{name}(")).any(|(at, _)| {
                !text[..at]
                    .ends_with(|c: char| c.is_alphanumeric() || c == '_' || c == '$' || c == '.')
            });
            if called && !wanted.contains(&name) {
                wanted.push(name);
            }
        }
        i += 1;
    }
    let texts: Vec<&str> = wanted.iter().map(|name| js_function(js, name)).collect();
    texts.join("\n")
}

/// Runs `script` under Node and parses the JSON it prints.
fn node(script: &str) -> Value {
    let mut child = Command::new("node")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| {
            panic!(
                "could not run `node` ({e}). These tests run the browser views' JavaScript \
                 under Node.js: install it (https://nodejs.org) and put `node` on PATH."
            )
        });
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "node failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("node printed no JSON")
}

/// Prints the side `surfaceEval` puts each point of each case on.
const SIDES_JS: &str = "
console.log(JSON.stringify(CASES.map((c) => c.points.map(([x, y, z]) => {
  const f = surfaceEval(c.surface, x, y, z);
  return Number.isNaN(f) ? 'NaN' : Math.sign(f);
}))));
";

#[test]
fn every_surface_eval_copy_has_rusts_sign_for_every_kind() {
    let cases = cases();
    let covered: HashSet<String> = cases.iter().map(|(s, _)| kind_name(&s.kind)).collect();
    assert_eq!(covered, all_kinds());

    let mut rng = Rng(505);
    let points: Vec<Vec<[f64; 3]>> = cases.iter().map(|c| points(c, &mut rng).0).collect();
    let input: Vec<Value> = cases
        .iter()
        .zip(&points)
        .map(|((s, _), p)| json!({ "surface": s, "points": p }))
        .collect();
    for (copy, js, _) in js_copies() {
        let script = format!(
            "{}\nconst CASES = {};\n{SIDES_JS}",
            js_with(&js, &["surfaceEval"]),
            Value::Array(input.clone()),
        );
        let got = node(&script);
        let mut wrong = Vec::new();
        let mut n = 0;
        for (((surface, _), pts), sides) in cases.iter().zip(&points).zip(got.as_array().unwrap()) {
            for (p, js_side) in pts.iter().zip(sides.as_array().unwrap()) {
                let f = eval(surface, *p);
                assert!(
                    !f.is_nan(),
                    "{:?} at {p:?}: Rust evaluate is NaN",
                    surface.kind
                );
                n += 1;
                if js_side.as_i64() != Some(side(f)) {
                    wrong.push(format!(
                        "{:?} at {p:?}: Rust {f:e}, JS side {js_side}",
                        surface.kind
                    ));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{copy} disagrees with Surface::evaluate at {} of {n} points, e.g.\n{}",
            wrong.len(),
            wrong[..wrong.len().min(10)].join("\n")
        );
    }
}

/// Prints, for each case, how many of its `on` points hover does not find
/// the surface at and how many of its `off` points it does.
const HOVER_JS: &str = "
console.log(JSON.stringify(CASES.map((c) => {
  SURFACE_TABLE = [{ surface: c.surface, name: 's' }];
  const hovered = ([x, y, z]) => surfacesAtPoint(x, y, z, c.tol).length === 1;
  return [c.on.filter((p) => !hovered(p)).length, c.off.filter(hovered).length];
})));
";

/// The hover tooltip lists a surface where `|surfaceEval| <= tol`, half a
/// pixel in cm. For every kind that must mean within about `tol` of it:
/// points `tol / 2` off the surface are on it, points `4 tol` off are not.
#[test]
fn hover_finds_a_surface_within_half_a_pixel_of_it() {
    let mut rng = Rng(5050);
    let mut input = Vec::new();
    for case in cases() {
        let surface = &case.0;
        let (_, _, scale) = frame(&surface.kind);
        let tol = 1e-3 * scale;
        let h = 1e-6 * scale;
        let grad = |p: [f64; 3]| {
            let mut g = [0.0; 3];
            for (i, gi) in g.iter_mut().enumerate() {
                let (mut a, mut b) = (p, p);
                a[i] += h;
                b[i] -= h;
                *gi = (eval(surface, a) - eval(surface, b)) / (2.0 * h);
            }
            g
        };
        let (mut on, mut off) = (Vec::new(), Vec::new());
        for s in points(&case, &mut rng).1 {
            let g = grad(s);
            let gn = norm(g);
            let n = [g[0] / gn, g[1] / gn, g[2] / gn];
            // Only where the surface is smooth and f about linear across the
            // hover band, so that the distance there is known.
            let linear = [-4.0, 4.0].iter().all(|&k| {
                let p = add(s, k * tol, n);
                (eval(surface, p) - k * tol * gn).abs() <= 0.05 * 4.0 * tol * gn
                    && (norm(grad(p)) - gn).abs() <= 0.05 * gn
            });
            if gn > 0.0 && linear {
                on.extend([add(s, 0.5 * tol, n), add(s, -0.5 * tol, n)]);
                off.extend([add(s, 4.0 * tol, n), add(s, -4.0 * tol, n)]);
            }
        }
        assert!(
            on.len() >= 20,
            "{:?}: too few smooth surface points",
            surface.kind
        );
        input.push(json!({ "surface": surface, "tol": tol, "on": on, "off": off }));
    }
    for (copy, js, hover) in js_copies() {
        if !hover {
            continue;
        }
        let script = format!(
            "let SURFACE_TABLE = [];\n{}\nconst CASES = {};\n{HOVER_JS}",
            js_with(&js, &["surfacesAtPoint"]),
            Value::Array(input.clone()),
        );
        let got = node(&script);
        for (case, missed) in input.iter().zip(got.as_array().unwrap()) {
            assert_eq!(
                missed,
                &json!([0, 0]),
                "{copy}: [points tol/2 off not hovered, points 4 tol off hovered] for {}",
                case["surface"]["kind"]
            );
        }
    }
}

/// An unknown kind `U` beside the inside (`T`) and outside (`F`) of a ball,
/// at its centre, where hover would find any surface it can evaluate.
const UNKNOWN_JS: &str = "
const unknown = { surface_id: null, kind: { NotAKind: {} }, boundary: 'Transmission' };
const ball = {
  surface_id: null, kind: { Sphere: { x0: 0, y0: 0, z0: 0, radius: 1 } }, boundary: 'Transmission',
};
const U = { Halfspace: { Below: unknown } };
const T = { Halfspace: { Below: ball } };
const F = { Halfspace: { Above: ball } };
const at = (expr) => evalExpr(expr, 0, 0, 0);
SURFACE_TABLE = [{ surface: unknown, name: 'u' }];
const regions = {
  value: String(surfaceEval(unknown, 0, 0, 0)),
  below: at(U), above: at({ Halfspace: { Above: unknown } }), complement: at({ Complement: U }),
  and_false: at({ Intersection: [U, F] }), false_and: at({ Intersection: [F, U] }),
  and_true: at({ Intersection: [U, T] }),
  or_true: at({ Union: [U, T] }), true_or: at({ Union: [T, U] }), or_false: at({ Union: [U, F] }),
  true_and_true: at({ Intersection: [T, T] }), false_or_false: at({ Union: [F, F] }),
  not_true: at({ Complement: T }), not_false: at({ Complement: F }),
};
";

/// A kind `surfaceEval` does not know (one added in Rust and not in the JS)
/// is NaN: region evaluation calls it unknown (null, never filled) unless the
/// other side of an intersection or union decides, and hover skips it. Known
/// leaves combine as before.
#[test]
fn an_unknown_kind_is_nan_never_filled_and_never_hovered() {
    for (copy, js, hover) in js_copies() {
        let roots: &[&str] = if hover {
            &["evalExpr", "surfacesAtPoint"]
        } else {
            &["evalExpr"]
        };
        let hovered = if hover {
            "surfacesAtPoint(0, 0, 0, Infinity)"
        } else {
            "[]"
        };
        let script = format!(
            "let SURFACE_TABLE = [];\n{}\n{UNKNOWN_JS}\n\
             console.log(JSON.stringify({{ ...regions, hovered: {hovered} }}));",
            js_with(&js, roots),
        );
        assert_eq!(
            node(&script),
            json!({
                "value": "NaN",
                "below": null, "above": null, "complement": null,
                "and_false": false, "false_and": false, "and_true": null,
                "or_true": true, "true_or": true, "or_false": null,
                "true_and_true": true, "false_or_false": false,
                "not_true": false, "not_false": true,
                "hovered": [],
            }),
            "{copy}"
        );
    }
}

/// `a` and `b` are the same JSON, numbers compared as f64 (JS writes `1.0`
/// as `1`).
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same_json(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
        }
        _ => a == b,
    }
}

/// The editor's Apply: the card's fields read back as an `<input
/// type=number>` gives them, then `checkSurfaceParams`.
const APPLY_JS: &str = "
function applyCard(kindName, spec) {
  const read = (v) => parseFloat(String(v));
  const p = {};
  for (const field of spec) {
    p[field.name] = field.type === 'scalar' ? read(field.value) : field.value.map(read);
  }
  checkSurfaceParams(kindName, p);
  return { [kindName]: p };
}
const nameOf = (kind) => Object.keys(kind)[0];
";

/// Prints how many fields `surfaceSpec` gives each of `KINDS`, and what Apply
/// writes back for it unedited.
const ROUND_TRIP_JS: &str = "
console.log(JSON.stringify(KINDS.map((kind) => {
  const spec = surfaceSpec(nameOf(kind), kind[nameOf(kind)]);
  return [spec.length, applyCard(nameOf(kind), spec)];
})));
";

/// An unedited card's Apply writes back exactly the surface it was made from,
/// so it changes nothing, and Rust loads it.
#[test]
fn the_surface_editor_round_trips_every_kind_into_json_rust_loads() {
    // The cases, plus a Cylinder and a Cone whose axes are not unit vectors,
    // which Apply keeps as written (loading makes them unit vectors).
    let mut kinds: Vec<SurfaceKind> = cases().into_iter().map(|(s, _)| s.kind).collect();
    kinds.push(SurfaceKind::Cylinder {
        axis: [0.0, 0.0, 2.0],
        origin: [1.0, 2.0, 0.0],
        radius: 1.0,
    });
    kinds.push(SurfaceKind::Cone {
        apex: [1.0, 0.0, -1.0],
        axis: [1.0, 2.0, 2.0],
        tan2_theta: 0.5,
    });
    let script = format!(
        "{}\n{APPLY_JS}\nconst KINDS = {};\n{ROUND_TRIP_JS}",
        js_with(APP_JS, &["surfaceSpec", "checkSurfaceParams"]),
        serde_json::to_string(&kinds).unwrap(),
    );
    let got = node(&script);
    for (kind, out) in kinds.iter().zip(got.as_array().unwrap()) {
        let kind = serde_json::to_value(kind).unwrap();
        assert!(out[0].as_u64().unwrap() > 0, "no fields for {kind}");
        assert!(
            same_json(&out[1], &kind),
            "the editor wrote {} for {kind}",
            out[1]
        );
        serde_json::from_value::<SurfaceKind>(out[1].clone())
            .unwrap_or_else(|e| panic!("Rust cannot load {} ({e})", out[1]));
    }
}

/// Prints the message Apply refuses each of `REFUSED` with, and the kinds of
/// the cards the editor makes for `csgGeom`.
const REFUSED_JS: &str = "
console.log(JSON.stringify({
  refused: REFUSED.map((kind) => {
    try {
      applyCard(nameOf(kind), surfaceSpec(nameOf(kind), kind[nameOf(kind)]));
      return null;
    } catch (e) {
      return e.message;
    }
  }),
  cards: [...collectSurfaces().values()].map((group) => group.kindName),
}));
";

/// Apply refuses a degenerate surface rather than write it for Rust to load,
/// and an unknown kind gets no card, so Apply cannot write it back as `{}`.
#[test]
fn the_surface_editor_refuses_degenerate_surfaces_and_has_no_card_for_an_unknown_kind() {
    let (o, z) = ([0.0; 3], [0.0, 0.0, 1.0]);
    let refused = [
        (
            Surface::new_sphere(0.0, 0.0, 0.0, 0.0, None, None),
            "Sphere radius must be > 0",
        ),
        (
            Surface::new_cylinder(o, o, 1.0, None, None),
            "Cylinder axis cannot be zero vector",
        ),
        (
            Surface::new_cylinder(z, o, -1.0, None, None),
            "Cylinder radius must be > 0",
        ),
        (
            Surface::new_ztorus(0.0, 0.0, 0.0, 5.0, 1.0, 0.0, None, None),
            "ZTorus a, b and c must be > 0",
        ),
        (
            Surface::new_xtorus(0.0, 0.0, 0.0, 5.0, 0.0, 1.0, None, None),
            "XTorus a, b and c must be > 0",
        ),
        (
            Surface::new_ytorus(0.0, 0.0, 0.0, -5.0, 1.0, 1.0, None, None),
            "YTorus a, b and c must be > 0",
        ),
        (
            Surface::new_cone(o, o, 1.0, None, None),
            "Cone axis cannot be zero vector",
        ),
        (
            Surface::new_cone(o, z, 0.0, None, None),
            "Cone tan2_theta must be > 0",
        ),
        (
            quadric([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "Quadric needs a non-zero coefficient among a to j",
        ),
    ];
    let kinds: Vec<&SurfaceKind> = refused.iter().map(|(s, _)| &s.kind).collect();
    let unknown =
        json!({ "surface_id": null, "kind": { "NotAKind": {} }, "boundary": "Transmission" });
    let ball = serde_json::to_value(Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None)).unwrap();
    let below = |s: &Value| json!({ "Halfspace": { "Below": s } });
    let expr = json!({ "Intersection": [below(&unknown), below(&ball)] });
    let geometry = json!({ "cells": [{ "region": { "expr": expr } }] });
    let script = format!(
        "const csgGeom = {geometry};\n{}\n{APPLY_JS}\nconst REFUSED = {};\n{REFUSED_JS}",
        js_with(
            APP_JS,
            &["surfaceSpec", "checkSurfaceParams", "collectSurfaces"]
        ),
        serde_json::to_string(&kinds).unwrap(),
    );
    let got = node(&script);
    for ((surface, message), answer) in refused.iter().zip(got["refused"].as_array().unwrap()) {
        assert_eq!(
            answer.as_str(),
            Some(*message),
            "Apply of {:?}",
            surface.kind
        );
    }
    assert_eq!(got["cards"], json!(["Sphere"]));
}
