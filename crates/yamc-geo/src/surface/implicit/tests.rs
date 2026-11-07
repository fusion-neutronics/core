use super::compile::compile;
use super::parse::{number, parse};
use super::*;

/// Small deterministic generator (xorshift64*), so the tests need no
/// dependencies.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub(super) fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(super) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    pub(super) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    pub(super) fn direction(&mut self) -> [f64; 3] {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n > 0.1 && n < 1.0 {
                return v.map(|c| c / n);
            }
        }
    }
}

/// The tape of a formula, compiled but not checked.
pub(super) fn tape_of(text: &str) -> Tape {
    let ast = parse(text).unwrap_or_else(|e| panic!("{text}: {}", e.message));
    match compile(&ast) {
        Ok(c) => c.tape,
        Err(e) => panic!("{text}: {}", e.message),
    }
}

/// The tape of a formula if it compiles.
fn try_tape(text: &str) -> Option<Tape> {
    compile(&parse(text).ok()?).ok().map(|c| c.tape)
}

fn value(tape: &Tape, p: [f64; 3]) -> f64 {
    tape.eval(p, &mut Vec::new())
}

/// The canonical form of a formula (parsed and printed, not compiled).
fn canonical(text: &str) -> String {
    let ast = parse(text).unwrap_or_else(|e| panic!("{text}: {}", e.message));
    ast.print(ast.root())
}

/// A tape's instructions, constants by their bits.
fn bits(tape: &Tape) -> String {
    format!("{:?}", tape.ops())
}

pub(super) fn holes() -> String {
    let mut cylinders = Vec::new();
    for i in 0..7 {
        for j in 0..7 {
            let (cx, cy) = (i as f64 - 3.0, j as f64 - 3.0);
            cylinders.push(format!("sqrt((x - {cx})^2 + (y - {cy})^2) - 0.3"));
        }
    }
    format!(
        "max(abs(x) - 4, abs(y) - 4, abs(z) - 1, -min({}))",
        cylinders.join(", ")
    )
}

pub(super) const SPHERE: &str = "sqrt((x - 0.3)^2 + (y - -0.2)^2 + (z - 0.1)^2) - 4";
pub(super) const ZTORUS: &str =
    "(sqrt((x - 0.2)^2 + (y - 0.1)^2) - 4)^2 / 1 + (z - -0.3)^2 / 1.69 - 1";
pub(super) const SUPERELLIPSOID: &str = "x^4 + y^4 + z^4 - 16";
pub(super) const SMOOTH_UNION: &str = "smin(sqrt((sqrt(x^2 + y^2) - 2.5)^2 + z^2) - 0.6, \
                            sqrt((sqrt((x - 2.5)^2 + z^2) - 1.5)^2 + y^2) - 0.5, 0.4)";
pub(super) const TWISTED: &str = "((x * cos(0.6 * z) + y * sin(0.6 * z)) / 2.5)^4 \
                       + ((y * cos(0.6 * z) - x * sin(0.6 * z)) / 0.8)^4 + (z / 3)^4 - 1";
pub(super) const GYROID: &str = "sin(2 * pi / 1.5 * x) * cos(2 * pi / 1.5 * y) \
                      + sin(2 * pi / 1.5 * y) * cos(2 * pi / 1.5 * z) \
                      + sin(2 * pi / 1.5 * z) * cos(2 * pi / 1.5 * x) - 0.3";

#[test]
fn formulas_follow_the_usual_precedence() {
    fn fmin(a: f64, b: f64) -> f64 {
        if a <= b {
            a
        } else {
            b
        }
    }
    fn fmax(a: f64, b: f64) -> f64 {
        if a >= b {
            a
        } else {
            b
        }
    }
    fn bump(a: f64, b: f64, k: f64) -> f64 {
        let h = fmax(k - (a - b).abs(), 0.0);
        h * h / (4.0 * k)
    }
    type Exact = Box<dyn Fn([f64; 3]) -> f64>;
    let cases: Vec<(&str, Exact)> = vec![
        ("x - y - z", Box::new(|[x, y, z]| (x - y) - z)),
        ("x / y / z", Box::new(|[x, y, z]| (x / y) / z)),
        ("x - y * z", Box::new(|[x, y, z]| x - y * z)),
        ("x * (y + z)", Box::new(|[x, y, z]| x * (y + z))),
        ("-x^2", Box::new(|[x, _, _]| -(x * x))),
        ("x^-2", Box::new(|[x, _, _]| 1.0 / (x * x))),
        ("--x + +y", Box::new(|[x, y, _]| x + y)),
        ("x^3", Box::new(|[x, _, _]| (x * x) * x)),
        ("x^4", Box::new(|[x, _, _]| (x * x) * (x * x))),
        ("x^5", Box::new(|[x, _, _]| ((x * x) * (x * x)) * x)),
        (
            "x^6 * y^0 + z^1",
            Box::new(|[x, _, z]| ((x * x) * x) * ((x * x) * x) + z),
        ),
        ("x + 2^3^2", Box::new(|[x, _, _]| x + 512.0)),
        ("pi * x", Box::new(|[x, _, _]| std::f64::consts::PI * x)),
        (
            ".5 * x + 5. + 1e3 + 1E-3 + 1.5e+2",
            Box::new(|[x, _, _]| 0.5 * x + 5.0 + 1000.0 + 1e-3 + 150.0),
        ),
        ("x * x", Box::new(|[x, _, _]| x * x)),
        (
            "sqrt(x^2 + y^2) - abs(z)",
            Box::new(|[x, y, z]| (x * x + y * y).sqrt() - z.abs()),
        ),
        (
            "exp(x) + log(y^2) + sin(z) * cos(x)",
            Box::new(|[x, y, z]| libm::exp(x) + libm::log(y * y) + libm::sin(z) * libm::cos(x)),
        ),
        ("abs(x)^2.5", Box::new(|[x, _, _]| libm::pow(x.abs(), 2.5))),
        (
            "min(x, y, z) + max(x, y)",
            Box::new(|[x, y, z]| fmin(fmin(x, y), z) + fmax(x, y)),
        ),
        (
            "smin(x, y, 0.5) - smax(y, z, 2)",
            Box::new(|[x, y, z]| (fmin(x, y) - bump(x, y, 0.5)) - (fmax(y, z) + bump(y, z, 2.0))),
        ),
    ];
    let mut rng = Rng(1);
    for (text, exact) in cases {
        let tape = tape_of(text);
        for _ in 0..50 {
            let p = [
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
            ];
            let (got, want) = (value(&tape, p), exact(p));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{text} at {p:?}: {got} != {want}"
            );
        }
    }
}

#[test]
fn formulas_print_canonically() {
    for (text, want) in [
        ("x+y", "x + y"),
        ("((x))*y", "x * y"),
        ("x - (y - z)", "x - (y - z)"),
        ("(x - y) - z", "x - y - z"),
        ("x/(y*z)", "x / (y * z)"),
        ("(x/y)*z", "x / y * z"),
        ("-(x + y)", "-(x + y)"),
        ("(-x)^2", "(-x)^2"),
        ("-x^2", "-x^2"),
        ("- -x", "-(-x)"),
        ("x - -y", "x - -y"),
        ("x * -y", "x * -y"),
        ("+x", "x"),
        ("x^-2", "x^-2"),
        ("x^(y^z)", "x^y^z"),
        ("(x^y)^z", "(x^y)^z"),
        ("x^(1/3)", "x^(1 / 3)"),
        ("2.50 * x", "2.5 * x"),
        (
            "1e3 + 1e-7 + 0.1 + 1e16 + 123456789.5",
            "1000 + 1e-7 + 0.1 + 1e16 + 123456789.5",
        ),
        ("smin( x , y , 0.5 )", "smin(x, y, 0.5)"),
        ("min(x,y,z)", "min(x, y, z)"),
        ("sqrt(x)*abs(y)", "sqrt(x) * abs(y)"),
    ] {
        assert_eq!(canonical(text), want, "{text}");
        assert_eq!(canonical(want), want, "{want} is not a fixed point");
    }
}

fn number_signed(v: f64) -> String {
    if v < 0.0 {
        format!("-{}", number(-v))
    } else {
        number(v)
    }
}

/// A random formula over every operation and function.
fn random_formula(rng: &mut Rng, depth: u32) -> String {
    let leaves = [
        "x", "y", "z", "0", "0.5", "1", "2", "3.25", "1e-3", "1e3", "pi",
    ];
    if depth == 0 || rng.below(5) == 0 {
        return rng.pick(&leaves).to_string();
    }
    let a = random_formula(rng, depth - 1);
    let b = random_formula(rng, depth - 1);
    match rng.below(20) {
        0 => format!("{a} + {b}"),
        1 => format!("({a}) - ({b})"),
        2 => format!("({a}) * ({b})"),
        3 => format!("({a}) / ({b})"),
        4 => format!("-({a})"),
        5 => format!("sqrt({a})"),
        6 => format!("abs({a})"),
        7 => format!("exp({a})"),
        8 => format!("log({a})"),
        9 => format!("sin({a})"),
        10 => format!("cos({a})"),
        11 => format!("min({a}, {b})"),
        12 => format!("max({a}, {b}, x)"),
        13 => format!("smin({a}, {b}, 0.75)"),
        14 => format!("smax({a}, {b}, 2)"),
        15 => {
            let n = rng.range(-3.0, 6.99).floor();
            format!("({a})^{}", number_signed(n))
        }
        16 => {
            let p = *rng.pick(&[0.5, 1.5, 2.5, -0.5, 0.25]);
            format!("abs({a})^{}", number_signed(p))
        }
        17 => format!("({a}) - ({a})"),
        // A length, with a term clamped at zero: 0 where both terms are.
        18 => format!("sqrt(({a})^2 + max({b}, 0)^2 + (y - 0.5)^2)"),
        _ => format!("min(0, -abs({a})) * ({b})"),
    }
}

#[test]
fn canonical_strings_parse_back_to_the_same_tape() {
    let mut rng = Rng(2);
    let mut compiled = 0;
    for _ in 0..1500 {
        let depth = 1 + rng.below(5) as u32;
        let text = random_formula(&mut rng, depth);
        let printed = canonical(&text);
        assert_eq!(canonical(&printed), printed, "{text}");
        if let Some(tape) = try_tape(&text) {
            let again = tape_of(&printed);
            assert_eq!(bits(&tape), bits(&again), "{text} -> {printed}");
            compiled += 1;
        }
    }
    assert!(compiled > 500, "{compiled}");
}

#[test]
fn numbers_print_as_the_shortest_string_that_parses_back() {
    let mut rng = Rng(3);
    for _ in 0..20_000 {
        let v = f64::from_bits(rng.next() >> 1);
        if !v.is_finite() {
            continue;
        }
        let text = number(v);
        assert_eq!(
            text.parse::<f64>().unwrap().to_bits(),
            v.to_bits(),
            "{v:e}: {text}"
        );
        let tape = tape_of(&format!("x + {text}"));
        assert!(tape.ops().contains(&Op::Const(v)), "{text}");
    }
    assert_eq!(number(0.1), "0.1");
    assert_eq!(number(100.0), "100");
    assert_eq!(number(1e-7), "1e-7");
}

/// The message for `text`, which must be refused.
fn refusal(text: &str) -> String {
    match ImplicitFormula::new(text) {
        Ok(f) => panic!("{text} was accepted as {f:?}"),
        Err(e) => e,
    }
}

#[test]
fn bad_formulas_are_refused_where_they_go_wrong() {
    let deep = format!("{}x{}", "(".repeat(201), ")".repeat(201));
    for (text, says, column) in [
        ("", "the formula is empty", 1),
        ("   ", "the formula is empty", 4),
        ("x +", "found the end of the formula", 4),
        (
            "x + * y",
            "expected a number, x, y, z, pi, a function or '(', found '*'",
            5,
        ),
        (
            "(x + y",
            "expected ')' to close the '(' at column 1, found the end",
            7,
        ),
        ("x + y)", "this ')' has no matching '('", 6),
        (
            "2x",
            "expected an operator, found 'x' (write products with '*')",
            2,
        ),
        ("x(2)", "'x' is not a function: write products with '*'", 2),
        ("foo(x) + 1", "unknown name 'foo'", 1),
        ("X + 1", "unknown name 'X'", 1),
        ("sqrt(x, y)", "sqrt takes 1 argument (got 2)", 1),
        ("1 + min(x)", "min takes at least 2 arguments (got 1)", 5),
        ("smin(x, y)", "smin takes 3 arguments (got 2)", 1),
        ("sqrt x", "expected '(' after the function sqrt", 6),
        ("x # y", "unexpected character '#'", 3),
        ("x é y", "unexpected character 'é'", 3),
        ("x + 1e400", "the number is too large", 5),
        ("x + 1e", "expected the digits of the exponent", 6),
        ("x + .", "a number needs at least one digit", 5),
        ("x, y", "unexpected ','", 2),
        (deep.as_str(), "nested more than 200 levels deep", 201),
        ("x^y", "the exponent must be a constant", 3),
        (
            "x^0.5 - 1",
            "needs a base that cannot be negative, and x can be",
            2,
        ),
        (
            "(x - 1)^1.5",
            "needs a base that cannot be negative, and x - 1 can be",
            8,
        ),
        ("x^2000", "an integer exponent can be at most 1024", 3),
        (
            "smin(x, y, 0) - 1",
            "the blending distance k of smin must be a positive constant",
            12,
        ),
        ("smax(x, y, z)", "the blending distance k of smax", 12),
        (
            "x + 1 / 0",
            "1 / 0 is not a finite number (it evaluates to inf)",
            7,
        ),
        (
            "x + sqrt(-1)",
            "sqrt(-1) is not a finite number (it evaluates to NaN)",
            5,
        ),
        (
            "x - log(0)",
            "log(0) is not a finite number (it evaluates to -inf)",
            5,
        ),
        ("1 + 2 * pi", "the formula is a constant", 3),
    ] {
        let message = refusal(text);
        assert!(message.contains(says), "{text:?}: {message}");
        assert!(
            message.contains(&format!("(column {column})")),
            "{text:?}: {message}"
        );
    }
    // The caret sits under the column.
    let message = refusal("x + q");
    assert!(message.ends_with("\n  x + q\n      ^"), "{message}");
}

#[test]
fn the_parser_never_panics() {
    let pieces = [
        "x", "y", "z", "pi", "1", "0.5", "1e3", "e", ".", "+", "-", "*", "/", "^", "(", ")", ",",
        "sqrt", "min", "smin", "abs", " ", "é", "#", "1e", "--",
    ];
    let mut rng = Rng(4);
    for _ in 0..20_000 {
        let n = rng.below(12);
        let text: String = (0..n).map(|_| *rng.pick(&pieces)).collect();
        if let Ok(ast) = parse(&text) {
            let printed = ast.print(ast.root());
            assert_eq!(canonical(&printed), printed, "{text}");
            let _ = compile(&ast);
        }
    }
}

fn random_point(rng: &mut Rng) -> [f64; 3] {
    let scale = *rng.pick(&[1e-3, 0.1, 1.0, 5.0, 50.0]);
    [
        rng.range(-scale, scale),
        rng.range(-scale, scale),
        rng.range(-scale, scale),
    ]
}

/// A box around `p`: a single point, or of a random size per axis.
fn random_box(rng: &mut Rng, p: [f64; 3]) -> [Interval; 3] {
    std::array::from_fn(|k| {
        let w = *rng.pick(&[0.0, 0.0, 1e-12, 1e-6, 1e-3, 0.1, 2.0, 30.0]);
        Interval::new(p[k] - w * rng.uniform(), p[k] + w * rng.uniform())
    })
}

/// A point of the box: a corner, its centre or a random point.
fn point_in(rng: &mut Rng, b: &[Interval; 3]) -> [f64; 3] {
    match rng.below(3) {
        0 => std::array::from_fn(|k| if rng.below(2) == 0 { b[k].lo } else { b[k].hi }),
        1 => std::array::from_fn(|k| 0.5 * (b[k].lo + b[k].hi)),
        _ => std::array::from_fn(|k| rng.range(b[k].lo, b[k].hi).clamp(b[k].lo, b[k].hi)),
    }
}

/// Whether the computed sum under some `sqrt` among the point values `f` of
/// `tape` is positive but below 2^-1000, where its squares underflow and
/// the computed slope of a length is not enclosed (see `norm_slope`).
fn underflowing_root(tape: &Tape, f: &[f64]) -> bool {
    tape.ops
        .iter()
        .any(|op| matches!(op, Op::Sqrt(a) if f[*a as usize] > 0.0 && f[*a as usize] < 9.3e-302))
}

/// Every evaluator against the point evaluator, slot by slot, at `p` in
/// the box `b`: the intervals enclose the point values, the interval
/// derivatives the point derivatives (unless a sum of squares under a
/// `sqrt` underflows there), and the duals' values are the point values
/// bit for bit. Returns the slot checks made.
fn check_slots(tape: &Tape, p: [f64; 3], b: [Interval; 3], dir: [f64; 3]) -> usize {
    let (mut f, mut iv, mut du, mut idu, mut ch) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    tape.eval(p, &mut f);
    tape.eval_interval(b, &mut iv);
    tape.eval_dual(p, dir, &mut du);
    tape.eval_interval_dual(b, dir, &mut idu);
    let traced = tape.eval_traced(b, &mut Vec::new(), &mut ch);
    assert_eq!(traced, iv[iv.len() - 1]);
    let underflow = underflowing_root(tape, &f);
    let mut checks = 0;
    for i in 0..tape.len() {
        let (v, op) = (f[i], tape.ops()[i]);
        for (name, e) in [("interval", iv[i]), ("dual value", idu[i].v)] {
            assert!(
                !e.lo.is_nan() && !e.hi.is_nan() && e.lo <= e.hi,
                "{name} slot {i} {op:?}: {e:?}"
            );
            assert!(
                v.is_nan() || e.contains(v),
                "{name} slot {i} {op:?}: {v:e} not in {e:?} (box {b:?}, point {p:?})"
            );
        }
        assert!(
            du[i].v.to_bits() == v.to_bits() || (du[i].v.is_nan() && v.is_nan()),
            "dual value slot {i} {op:?}: {} != {v}",
            du[i].v
        );
        let (d, e) = (du[i].d, idu[i].d);
        assert!(
            !e.lo.is_nan() && !e.hi.is_nan(),
            "derivative slot {i}: {e:?}"
        );
        assert!(
            !d.is_finite() || underflow || e.contains(d),
            "derivative slot {i} {op:?}: {d:e} not in {e:?} (value {v:e}, box {b:?})"
        );
        checks += 1;
    }
    checks
}

#[test]
fn intervals_enclose_point_values_on_random_formulas() {
    let mut rng = Rng(5);
    let mut checks = 0;
    let mut formulas = 0;
    while formulas < 1_000 {
        let depth = 1 + rng.below(5) as u32;
        let Some(tape) = try_tape(&random_formula(&mut rng, depth)) else {
            continue;
        };
        formulas += 1;
        for _ in 0..10 {
            let p = random_point(&mut rng);
            let dir = rng.direction();
            checks += check_slots(&tape, p, p.map(Interval::point), dir);
            let b = random_box(&mut rng, p);
            for _ in 0..3 {
                checks += check_slots(&tape, point_in(&mut rng, &b), b, dir);
            }
        }
    }
    assert!(checks > 150_000, "{checks}");
}

#[test]
fn intervals_enclose_the_shapes_along_rays() {
    let mut rng = Rng(6);
    for text in [
        SPHERE,
        ZTORUS,
        SMOOTH_UNION,
        TWISTED,
        GYROID,
        holes().as_str(),
    ] {
        let tape = tape_of(text);
        for _ in 0..300 {
            let o = [
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
                rng.range(-2.0, 2.0),
            ];
            let u = rng.direction();
            let t0 = rng.range(0.0, 5.0);
            let w = *rng.pick(&[1e-9, 1e-4, 0.01, 0.5, 3.0]);
            let t = Interval::new(t0, t0 + w);
            let b = ray_box(o, u, t);
            let p = ray_point(o, u, rng.range(t.lo, t.hi).min(t.hi));
            check_slots(&tape, p, b, u);
        }
    }
}

#[test]
fn ray_boxes_hold_the_ray_points() {
    let mut rng = Rng(7);
    for _ in 0..20_000 {
        let scale = *rng.pick(&[1.0, 1e6, 1e300]);
        let o = [
            rng.range(-scale, scale),
            rng.range(-scale, scale),
            rng.range(-scale, scale),
        ];
        let tiny = *rng.pick(&[1.0, 1e-300]);
        let u = rng.direction().map(|c| c * tiny);
        let t0 = rng.range(0.0, 1e6);
        let t = Interval::new(t0, t0 + 10f64.powi(rng.below(15) as i32 - 12));
        let b = ray_box(o, u, t);
        for i in 0..=8 {
            let s = (t.lo + (t.hi - t.lo) * i as f64 / 8.0).min(t.hi);
            let p = ray_point(o, u, s);
            assert!((0..3).all(|k| b[k].contains(p[k])), "{b:?} misses {p:?}");
        }
    }
}

/// One operation, applied to adversarial operands: infinities, the
/// largest and smallest floats, signed zeros, the edges of `exp`, `sin`
/// and `cos` at their cut-off and next to their extrema. Every result must
/// be ordered, never NaN, and hold the point value at the ends, the zeros
/// and inner points.
#[test]
fn interval_operations_hold_at_special_values() {
    use std::f64::consts::{FRAC_PI_2, PI};
    let special = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -1e300,
        -1e8 - 0.5,
        -745.2,
        -709.8,
        -PI,
        -FRAC_PI_2,
        -1.0,
        -1e-8,
        -5e-324,
        -0.0,
        0.0,
        5e-324,
        1e-300,
        1e-8,
        1.0,
        FRAC_PI_2,
        PI,
        3.0 * FRAC_PI_2,
        709.78,
        709.8,
        1e7 * PI,
        1e8,
        1e300,
        f64::MAX,
        f64::INFINITY,
    ];
    let ops = [
        Op::Neg(0),
        Op::Square(0),
        Op::Sqrt(0),
        Op::Abs(0),
        Op::Exp(0),
        Op::Log(0),
        Op::Sin(0),
        Op::Cos(0),
        Op::Pow(0, 0.5),
        Op::Pow(0, 2.5),
        Op::Pow(0, -1.5),
        Op::Add(0, 1),
        Op::Sub(0, 1),
        Op::Mul(0, 1),
        Op::Div(0, 1),
        Op::Min(0, 1),
        Op::Max(0, 1),
    ];
    let mut rng = Rng(8);
    let pick = |rng: &mut Rng| -> Interval {
        let (a, b) = match rng.below(4) {
            0 => (*rng.pick(&special), *rng.pick(&special)),
            1 => {
                let c = *rng.pick(&special);
                let w = *rng.pick(&[0.0, 1e-300, 1e-12, 1e-6, 0.5, 3.0]);
                (c - w * rng.uniform(), c + w * rng.uniform())
            }
            2 => {
                let c = *rng.pick(&special);
                (c, c)
            }
            _ => {
                let m = 10f64.powf(rng.range(-10.0, 10.0));
                (rng.range(-m, m), rng.range(-m, m))
            }
        };
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        Interval::new(
            if lo.is_nan() { 0.0 } else { lo },
            if hi.is_nan() { 0.0 } else { hi },
        )
    };
    let samples = |e: Interval, rng: &mut Rng| -> Vec<f64> {
        let mut v = vec![
            e.lo,
            e.hi,
            e.lo.next_up().min(e.hi),
            e.hi.next_down().max(e.lo),
        ];
        if e.has_zero() {
            v.extend([0.0, -0.0]);
        }
        let (l, h) = (e.lo.max(-1e300), e.hi.min(1e300));
        if l <= h {
            for _ in 0..3 {
                v.push((l + rng.uniform() * (h - l)).clamp(e.lo, e.hi));
            }
        }
        v
    };
    let (mut checks, mut slots) = (0, Vec::new());
    for _ in 0..5_000 {
        let (a, b) = (pick(&mut rng), pick(&mut rng));
        let (sa, sb) = (samples(a, &mut rng), samples(b, &mut rng));
        for &op in &ops {
            let tape = Tape {
                ops: vec![Op::X, Op::Y, op],
            };
            let r = tape.eval_interval([a, b, Interval::point(0.0)], &mut Vec::new());
            assert!(
                !r.lo.is_nan() && !r.hi.is_nan() && r.lo <= r.hi,
                "{op:?} {a:?} {b:?}: {r:?}"
            );
            for &x in &sa {
                for &y in &sb {
                    let v = tape.eval([x, y, 0.0], &mut slots);
                    assert!(
                        v.is_nan() || r.contains(v),
                        "{op:?} of {a:?}, {b:?} = {r:?} misses f({x:e}, {y:e}) = {v:e}"
                    );
                    checks += 1;
                }
            }
        }
    }
    // sin and cos on narrow intervals at large arguments and next to
    // their extrema.
    for _ in 0..5_000 {
        let c = if rng.below(2) == 0 {
            FRAC_PI_2 * rng.range(-6e7, 6e7).round() + rng.range(-1e-6, 1e-6)
        } else {
            rng.range(-1e8, 1e8)
        };
        let w = 10f64.powf(rng.range(-15.0, 0.0));
        let e = Interval::new(c, c + w);
        let (s, k) = (e.sin(), e.cos());
        for i in 0..=16 {
            let x = (c + w * i as f64 / 16.0).min(e.hi);
            assert!(
                s.contains(libm::sin(x)) && k.contains(libm::cos(x)),
                "{e:?} at {x:e}"
            );
            checks += 1;
        }
    }
    assert!(checks > 1_000_000, "{checks}");
}

#[test]
fn derivatives_match_finite_differences() {
    // Smooth formulas: the derivative along a direction against a central
    // difference.
    let mut rng = Rng(9);
    for text in [
        SPHERE,
        ZTORUS,
        SUPERELLIPSOID,
        GYROID,
        TWISTED,
        "exp(-(x^2 + y^2)) * cos(3 * z) + log(1 + x^2) / (2 + sin(y)) - abs(z + 10)^1.5",
    ] {
        let tape = tape_of(text);
        for _ in 0..200 {
            let p = [
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
            ];
            let u = rng.direction();
            let d = tape.eval_dual(p, u, &mut Vec::new()).d;
            let h = 1e-6;
            let fd =
                (value(&tape, ray_point(p, u, h)) - value(&tape, ray_point(p, u, -h))) / (2.0 * h);
            assert!(
                (d - fd).abs() <= 1e-5 * (1.0 + d.abs()),
                "{text} at {p:?}: {d} vs {fd}"
            );
        }
    }
    // Any formula, kinks and all: the mean slope over a short stretch lies
    // in the interval derivative over a box around it, up to the rounding
    // of the two values (bounded by their own interval widths).
    let mut tested = 0;
    while tested < 3_000 {
        let depth = 1 + rng.below(4) as u32;
        let Some(tape) = try_tape(&random_formula(&mut rng, depth)) else {
            continue;
        };
        let p = random_point(&mut rng);
        let u = rng.direction();
        let h = *rng.pick(&[1e-3, 1e-5]);
        let (a, b) = (ray_point(p, u, -h), ray_point(p, u, h));
        let (fa, fb) = (value(&tape, a), value(&tape, b));
        let deriv = tape
            .eval_interval_dual(ray_box(p, u, Interval::new(-h, h)), u, &mut Vec::new())
            .d;
        if !(fa.is_finite() && fb.is_finite() && deriv.is_finite()) {
            continue;
        }
        tested += 1;
        let width = |q: [f64; 3]| {
            let e = tape.eval_interval(q.map(Interval::point), &mut Vec::new());
            e.hi - e.lo
        };
        let slack = (width(a) + width(b)) / (2.0 * h) + 1e-9 * (1.0 + deriv.magnitude());
        let slope = (fb - fa) / (2.0 * h);
        assert!(
            slope >= deriv.lo - slack && slope <= deriv.hi + slack,
            "slope {slope:e} outside {deriv:?} (slack {slack:e}) at {p:?}"
        );
    }
}

#[test]
fn shortened_tapes_are_exact_in_their_box() {
    let mut rng = Rng(10);
    let check = |tape: &Tape, b: [Interval; 3], rng: &mut Rng| -> bool {
        let mut choices = Vec::new();
        tape.eval_traced(b, &mut Vec::new(), &mut choices);
        let mut short = Tape::default();
        tape.simplify(&choices, &mut Vec::new(), &mut short);
        for _ in 0..20 {
            let p = point_in(rng, &b);
            let (full, cut) = (value(tape, p), value(&short, p));
            assert!(
                full.is_nan() || full.to_bits() == cut.to_bits(),
                "{full:e} != {cut:e} at {p:?} in {b:?}"
            );
        }
        short.len() < tape.len()
    };
    // The holes block, and random formulas full of min / max, abs and
    // signed zeros.
    let block = tape_of(&holes());
    let mut shorter = 0;
    for _ in 0..500 {
        let c = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-1.5, 1.5),
        ];
        let w = rng.range(0.01, 2.0);
        shorter += check(&block, c.map(|v| Interval::new(v - w, v + w)), &mut rng) as usize;
    }
    assert!(shorter > 300, "{shorter}");
    let mut formulas = 0;
    while formulas < 1_000 {
        let Some(tape) = try_tape(&random_formula(&mut rng, 4)) else {
            continue;
        };
        if tape.choices() == 0 {
            continue;
        }
        formulas += 1;
        let p = random_point(&mut rng);
        check(&tape, random_box(&mut rng, p), &mut rng);
    }
    // A tie between +0 and -0 is not decided for the second operand: at
    // y = 0, min(0, -abs(y)) is +0 (the first), and 1 / +0 = inf.
    let tie = tape_of("1 / min(0, -abs(y)) + x");
    let b = [
        Interval::new(-1.0, 1.0),
        Interval::new(-1.0, 0.0),
        Interval::point(0.0),
    ];
    let mut choices = Vec::new();
    tie.eval_traced(b, &mut Vec::new(), &mut choices);
    assert!(choices.iter().all(|&c| c == Choice::Both));
    let away = [b[0], Interval::new(-1.0, -0.5), b[2]];
    tie.eval_traced(away, &mut Vec::new(), &mut choices);
    assert!(choices.contains(&Choice::Right));
    assert!(check(&tie, away, &mut rng));
}

#[test]
fn transcendental_functions_give_the_same_bits_everywhere() {
    // libm's results, pinned: every platform computes these bits, whatever
    // its own maths library does (exp(1) is 1 ulp from the correctly
    // rounded e, which some libraries return instead).
    for (got, bits) in [
        (libm::sin(1.0), 0x3feaed548f090cee_u64),
        (libm::cos(1.0), 0x3fe14a280fb5068c),
        (libm::sin(1e22), 0xbfeb453ab76bf397),
        (libm::cos(1e22), 0x3fe0be2cef01c8f4),
        (libm::sin(0.5e8), 0x3fea6bb2b80d0cf6),
        (libm::exp(1.0), 0x4005bf0a8b14576a),
        (libm::exp(-700.5), 0x00c4ff475c68ca02),
        (libm::exp(709.7), 0x7fed75ae7a50ee14),
        (libm::log(10.0), 0x40026bb1bbb55516),
        (libm::log(1e-300), 0xc085963447f87fb5),
        (libm::pow(2.0, 0.5), 0x3ff6a09e667f3bcd),
        (libm::pow(10.0, 2.5), 0x4073c3a4edfa9758),
        (libm::pow(0.3, -1.7), 0x401ef88d0dbd9b43),
    ] {
        assert_eq!(got.to_bits(), bits, "{got:e}");
    }
    // And the formulas use them.
    let f = tape_of("exp(x) + sin(y) * cos(z) + log(10) + abs(x)^2.5");
    let p = [1.0, 1e22, 1.0];
    let want = libm::exp(1.0) + libm::sin(1e22) * libm::cos(1.0) + libm::log(10.0) + 1.0;
    assert_eq!(value(&f, p).to_bits(), want.to_bits());
}

/// `formula`'s box, which must exist, contain `want` and be within
/// `slack` of it on every face.
fn check_box(formula: &str, want: ([f64; 3], [f64; 3]), slack: f64) -> ImplicitFormula {
    let f = ImplicitFormula::new(formula).unwrap();
    let (lo, hi) = f
        .bounding_box()
        .unwrap_or_else(|| panic!("no box for {formula}"));
    for k in 0..3 {
        assert!(
            lo[k] <= want.0[k] && lo[k] >= want.0[k] - slack,
            "{formula}: lower {k} {} vs {}",
            lo[k],
            want.0[k]
        );
        assert!(
            hi[k] >= want.1[k] && hi[k] <= want.1[k] + slack,
            "{formula}: upper {k} {} vs {}",
            hi[k],
            want.1[k]
        );
    }
    f
}

#[test]
fn bounded_shapes_get_a_proved_box() {
    check_box(SPHERE, ([-3.7, -4.2, -3.9], [4.3, 3.8, 4.1]), 0.05);
    // Its top and bottom are circles, which take more cells to bound.
    check_box(ZTORUS, ([-4.8, -4.9, -1.6], [5.2, 5.1, 1.0]), 0.1);
    check_box(SUPERELLIPSOID, ([-2.0; 3], [2.0; 3]), 1e-4);
    check_box(SMOOTH_UNION, ([-3.1, -3.1, -2.0], [4.5, 3.1, 2.0]), 0.05);
    let block = check_box(&holes(), ([-4.0, -4.0, -1.0], [4.0, 4.0, 1.0]), 0.3);
    // A cube that is a box itself (a flat face takes many cells to bound),
    // and a small sphere far from the origin.
    check_box(
        "max(abs(x), abs(y), abs(z)) - 1",
        ([-1.0; 3], [1.0; 3]),
        0.1,
    );
    check_box(
        "sqrt((x - 900)^2 + y^2 + (z + 20)^2) - 0.01",
        ([899.99, -0.01, -20.01], [900.01, 0.01, -19.99]),
        1e-3,
    );
    // No point below the surface lies outside the box.
    let mut rng = Rng(11);
    for f in [ImplicitFormula::new(SMOOTH_UNION).unwrap(), block] {
        let (lo, hi) = f.bounding_box().unwrap();
        for _ in 0..20_000 {
            let p: [f64; 3] = std::array::from_fn(|k| {
                let pad = 0.2 * (hi[k] - lo[k]);
                rng.range(lo[k] - pad, hi[k] + pad)
            });
            if f.evaluate(p) <= 0.0 {
                assert!((0..3).all(|k| lo[k] <= p[k] && p[k] <= hi[k]), "{p:?}");
            }
        }
    }
}

#[test]
fn unbounded_shapes_get_no_box() {
    for formula in [
        TWISTED,
        GYROID,
        "x - 1",
        "x^2 + y^2 - 1",
        "-(x^2 + y^2 + z^2 - 1)",
        "x^2 + y^2 - z^2 - 1",
        "exp(x) - 2",
    ] {
        let f = ImplicitFormula::new(formula).unwrap();
        assert_eq!(f.bounding_box(), None, "{formula}");
    }
}

#[test]
fn formulas_undefined_where_they_are_used_are_refused() {
    for (formula, says) in [
        (
            "sqrt(1 - x^2 - y^2) - z",
            "sqrt(1 - x^2 - y^2) can take the square root of a negative number near",
        ),
        ("1 / (x - 1) - 2", "1 / (x - 1) can divide by zero near"),
        ("(x^2 - 1) / (x - 1) - 2.5", "can divide zero by zero near"),
        ("sin(x) / x - 0.5", "sin(x) / x can divide zero by zero"),
        ("abs(x)^-0.5 - 2", "can raise zero to a negative power"),
        (
            "log(x) - 1",
            "log(x) can take the logarithm of a negative number",
        ),
        (
            "sin(exp(x)) - 0.5",
            "can be undefined (NaN) after an overflow",
        ),
        (
            "sqrt(x^2 + y^2 + z^2) - 1 + 0 * sqrt(x - 5)",
            "sqrt(x - 5) can take the square root of a negative number",
        ),
    ] {
        let message = refusal(formula);
        assert!(message.contains(says), "{formula}: {message}");
    }
    // Where: the point of the pole nearest the middle of the region checked.
    let message = refusal("1 / (x - 1) - 2");
    let near = message.split("near (").nth(1).unwrap();
    let at: Vec<f64> = near
        .split(')')
        .next()
        .unwrap()
        .split(", ")
        .map(|v| v.parse().unwrap())
        .collect();
    assert!(
        (at[0] - 1.0).abs() < 1e-5 && at[1].abs() < 1e-5 && at[2].abs() < 1e-5,
        "{message}"
    );
}

#[test]
fn formulas_that_never_change_sign_are_refused() {
    for (formula, says) in [
        ("x^2 + y^2 + z^2 + 1", "is never negative"),
        ("(x^2 + y^2 + z^2 - 1)^2", "is never negative"),
        ("abs(x) + abs(y)", "is never negative"),
        ("-(x^2 + y^2 + z^2) - 1", "is never positive"),
        ("-exp(x)", "is never positive"),
    ] {
        let message = refusal(formula);
        assert!(message.contains(says), "{formula}: {message}");
    }
}

#[test]
fn defined_formulas_are_accepted() {
    for formula in [
        // log(0) at the origin is -inf: below, and no pole.
        "log(x^2 + y^2 + z^2) - 1",
        // An overflow to +inf is a sign like any other.
        "exp(x) - 2",
        "exp(x^2 + y^2 + z^2) - 3",
        "(x^2 + 1)^0.5 + abs(y)^1.5 - 3 + z",
        "smax(abs(x) - 1, abs(y) - 1, 0.25) + z",
        "x / (y^2 + 1)",
    ] {
        if let Err(e) = ImplicitFormula::new(formula) {
            panic!("{formula}: {e}");
        }
    }
}

#[test]
fn formulas_round_trip_through_json() {
    for formula in [
        SPHERE,
        SMOOTH_UNION,
        GYROID,
        "(x^2 + 1)^-2 - y * -z + pi^2.5 - 10",
    ] {
        let f = ImplicitFormula::new(formula).unwrap();
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(json, format!("{:?}", f.formula()));
        let back: ImplicitFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(back.formula(), f.formula());
        assert_eq!(bits(back.tape()), bits(f.tape()));
        assert_eq!(back.bounding_box(), f.bounding_box());
    }
    // A bad formula in a model is refused on load.
    let bad = serde_json::from_str::<ImplicitFormula>("\"x +\"").unwrap_err();
    assert!(
        bad.to_string().contains("found the end of the formula"),
        "{bad}"
    );
}

#[test]
fn a_sphere_evaluates_exactly_as_the_closed_form_does() {
    let f = ImplicitFormula::new(SPHERE).unwrap();
    let sphere = crate::surface::Surface::new_sphere(0.3, -0.2, 0.1, 4.0, None, None);
    let mut rng = Rng(12);
    for _ in 0..10_000 {
        let p = [
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
        ];
        let (implicit, closed) = (f.evaluate(p), sphere.evaluate((p[0], p[1], p[2])));
        assert_eq!(implicit.to_bits(), closed.to_bits(), "{p:?}");
    }
}

#[test]
fn an_overflow_that_turns_into_nan_is_refused() {
    // x^64 overflows at 6.5e4 cm, well within the world: two such shapes
    // blended give inf - inf there.
    let message = refusal("smin(x^64 + y^64 + z^64 - 1, (x - 3)^64 + y^64 + z^64 - 1, 0.5)");
    assert!(message.contains("after an overflow"), "{message}");
    // Each alone overflows only to +inf, which is a sign like any other.
    let f = ImplicitFormula::new("x^64 + y^64 + z^64 - 1").unwrap();
    assert_eq!(f.evaluate([1e5, 0.0, 0.0]), f64::INFINITY);
}

#[test]
fn segments_enclose_the_computed_values_along_rays() {
    let mut rng = Rng(13);
    let mut slots = SegmentSlots::default();
    let mut checks = 0;
    let mut tapes: Vec<Tape> = [SPHERE, ZTORUS, SMOOTH_UNION, TWISTED, GYROID]
        .iter()
        .map(|t| tape_of(t))
        .collect();
    tapes.push(tape_of(&holes()));
    while tapes.len() < 800 {
        let depth = 1 + rng.below(5) as u32;
        if let Some(tape) = try_tape(&random_formula(&mut rng, depth)) {
            tapes.push(tape);
        }
    }
    for tape in &tapes {
        for _ in 0..10 {
            let o = random_point(&mut rng);
            let u = rng.direction();
            let t0 = rng.range(0.0, 5.0);
            let w = *rng.pick(&[0.0, 1e-9, 1e-6, 1e-3, 0.1, 2.0]);
            let t = Interval::new(t0, t0 + w);
            let seg = tape.eval_segment(o, u, t, &mut slots);
            assert!(seg.v.lo <= seg.v.hi && !seg.v.lo.is_nan() && !seg.v.hi.is_nan());
            // The choices are those traced over the same box.
            let mut traced = Vec::new();
            tape.eval_traced(ray_box(o, u, t), &mut Vec::new(), &mut traced);
            assert_eq!(slots.choices(), &traced[..]);
            for i in 0..=8 {
                let s = match i {
                    0 => t.lo,
                    8 => t.hi,
                    _ => rng.range(t.lo, t.hi).min(t.hi),
                };
                let p = ray_point(o, u, s);
                let mut f = Vec::new();
                let v = tape.eval(p, &mut f);
                assert!(
                    v.is_nan() || seg.v.contains(v),
                    "{v:e} not in {:?} at t = {s} ({:?})",
                    seg.v,
                    tape.ops()
                );
                let d = tape.eval_dual(p, u, &mut Vec::new()).d;
                assert!(
                    !d.is_finite() || underflowing_root(tape, &f) || seg.d.contains(d),
                    "{d:e} not in {:?}",
                    seg.d
                );
                checks += 1;
            }
        }
    }
    assert!(checks > 50_000, "{checks}");
}

#[test]
fn the_centred_form_rules_out_a_grazing_pass() {
    // A diagonal ray passing 1e-6 cm outside a sphere of radius 4: over
    // 1e-4 cm either side of the closest approach the box around the ray
    // reaches into the sphere, the centred form does not.
    let tape = tape_of("sqrt(x^2 + y^2 + z^2) - 4");
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let u = [r, r, 0.0];
    let closest = [(4.0 + 1e-6) * r, -(4.0 + 1e-6) * r, 0.0];
    let o = ray_point(closest, u, -10.0);
    let t = Interval::new(10.0 - 1e-4, 10.0 + 1e-4);
    let plain = tape.eval_interval(ray_box(o, u, t), &mut Vec::new());
    let seg = tape.eval_segment(o, u, t, &mut SegmentSlots::default());
    assert!(plain.lo < 0.0, "{plain:?}");
    assert!(seg.v.lo > 0.0, "{:?}", seg.v);
    assert!(
        seg.v.hi - seg.v.lo < 1e-3 * (plain.hi - plain.lo),
        "{:?}",
        seg.v
    );
}

#[test]
fn a_length_has_a_bounded_slope_where_it_is_zero() {
    let slope = |text: &str, b: [Interval; 3], dir: [f64; 3]| {
        tape_of(text).eval_interval_dual(b, dir, &mut Vec::new()).d
    };
    // Over a box holding the z axis, sqrt(x^2 + y^2) slopes by at most 1;
    // the square root of a sum that is not of squares can slope by any
    // amount.
    let axis = [
        Interval::new(-1e-3, 1e-3),
        Interval::new(-2e-3, 1e-3),
        Interval::new(0.0, 1.0),
    ];
    let d = slope("sqrt(x^2 + y^2)", axis, [0.6, 0.8, 0.0]);
    assert!(d.lo >= -1.0 - 1e-12 && d.hi <= 1.0 + 1e-12, "{d:?}");
    assert_eq!(
        slope("sqrt(abs(x) + y^2)", axis, [0.6, 0.8, 0.0]),
        Interval::ENTIRE
    );
    // An exact distance to a capped cylinder, over a box across its side
    // wall: heading in, its slope is bounded and never positive, so a
    // stretch of ray there crosses the wall at most once.
    let wall = [
        Interval::new(0.99, 1.01),
        Interval::new(-0.01, 0.01),
        Interval::new(-0.5, 0.5),
    ];
    let d = slope(
        "min(max(sqrt(x^2 + y^2) - 1, abs(z) - 2), 0) \
         + sqrt(max(sqrt(x^2 + y^2) - 1, 0)^2 + max(abs(z) - 2, 0)^2)",
        wall,
        [-1.0, 0.0, 0.0],
    );
    assert!(d.is_finite() && d.lo > -2.1 && d.hi <= 0.0, "{d:?}");
}
