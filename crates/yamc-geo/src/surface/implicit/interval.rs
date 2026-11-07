//! Closed f64 intervals with outward rounding.
//!
//! Each arithmetic result is computed in round-to-nearest and moved one ulp
//! outward (`next_down` / `next_up`). Round-to-nearest is monotone, so the
//! result encloses both the exact result and the f64 result of the same
//! operation on any members of the operands; by induction an interval
//! evaluation of a tape encloses the point evaluation of the same tape at
//! every point of its box, which is the sign the cell test reads. The
//! functions from `libm` are not correctly rounded: their end values are
//! moved [`LIBM_ULPS`] outward instead. Where the sign of a result follows
//! from the signs of the operands (a sum or product of non-negative
//! values, say) the widening stops at zero, so the result is shown not to
//! change sign.
//!
//! No bound is ever NaN and no operation panics: one whose bounds would be
//! NaN gives [`Interval::ENTIRE`]. A point operation that gives NaN (the
//! square root of a negative number, say) is left out of the enclosure, as
//! a NaN has no sign; construction refuses a formula that can give one
//! where it is used.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Ulps by which a `libm` result is moved outward: `libm` keeps `sin`,
/// `cos`, `exp`, `log` and `pow` within 1 ulp, and enclosing the value
/// computed at an inner point as well needs twice that; 4 leaves room.
pub const LIBM_ULPS: u32 = 4;

/// Beyond this magnitude an argument of `sin` or `cos` is enclosed by the
/// full range.
const TRIG_LIMIT: f64 = 1e8;

/// Bound on the magnitude of `libm`'s `sin` and `cos` (1 within a few ulps).
const ONE_UP: f64 = 1.0 + 4.0 * f64::EPSILON;

/// A closed interval `[lo, hi]` of f64 values (the bounds may be infinite).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

fn dn(v: f64) -> f64 {
    v.next_down()
}

fn up(v: f64) -> f64 {
    v.next_up()
}

fn dn_libm(mut v: f64) -> f64 {
    for _ in 0..LIBM_ULPS {
        v = v.next_down();
    }
    v
}

fn up_libm(mut v: f64) -> f64 {
    for _ in 0..LIBM_ULPS {
        v = v.next_up();
    }
    v
}

/// Product of two bounds with `0 * inf = 0`: the zero is a value, the
/// infinity only a bound.
fn mul_bound(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

/// Whether `phase + 2 pi k` lies in `[lo, hi]` for some integer `k`,
/// erring towards yes: the slack, far above the rounding error of the test
/// and of `TAU` against 2 pi, only ever adds an extremum.
fn hits(lo: f64, hi: f64, phase: f64) -> bool {
    let slack = 1e-9 * (1.0 + lo.abs().max(hi.abs()));
    let k = ((lo - slack - phase) / TAU).ceil();
    phase + k * TAU <= hi + slack
}

impl Interval {
    /// Every value.
    pub const ENTIRE: Interval = Interval {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    /// `[lo, hi]`; the caller keeps `lo <= hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The single value `v` (the whole line for a NaN, which encloses
    /// nothing).
    pub fn point(v: f64) -> Self {
        if v.is_nan() {
            Self::ENTIRE
        } else {
            Self { lo: v, hi: v }
        }
    }

    /// `[lo, hi]` from computed bounds, the whole line if one is NaN.
    fn bounds(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() {
            Self::ENTIRE
        } else {
            Self { lo, hi }
        }
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "used by the tests"))]
    pub fn contains(self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn has_zero(self) -> bool {
        self.lo <= 0.0 && 0.0 <= self.hi
    }

    /// Both bounds finite.
    #[cfg_attr(not(test), expect(dead_code, reason = "used by the tests"))]
    pub fn is_finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    /// The smallest magnitude of a member (0 if it holds zero).
    pub fn mignitude(self) -> f64 {
        if self.has_zero() {
            0.0
        } else {
            self.lo.abs().min(self.hi.abs())
        }
    }

    /// The largest magnitude of a member.
    pub fn magnitude(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    /// The smallest interval holding both.
    pub fn hull(self, o: Self) -> Self {
        Self {
            lo: self.lo.min(o.lo),
            hi: self.hi.max(o.hi),
        }
    }

    pub fn square(self) -> Self {
        let (l2, h2) = (self.lo * self.lo, self.hi * self.hi);
        if self.lo >= 0.0 {
            Self::bounds(dn(l2).max(0.0), up(h2))
        } else if self.hi <= 0.0 {
            Self::bounds(dn(h2).max(0.0), up(l2))
        } else {
            Self::bounds(0.0, up(l2.max(h2)))
        }
    }

    /// The square root; negative members, whose roots are NaN, are left
    /// out.
    pub fn sqrt(self) -> Self {
        if self.hi < 0.0 {
            return Self::ENTIRE;
        }
        let lo = if self.lo > 0.0 {
            dn(self.lo.sqrt()).max(0.0)
        } else {
            0.0
        };
        Self::new(lo, up(self.hi.sqrt()))
    }

    pub fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            -self
        } else {
            Self::new(0.0, (-self.lo).max(self.hi))
        }
    }

    pub fn min(self, o: Self) -> Self {
        Self {
            lo: self.lo.min(o.lo),
            hi: self.hi.min(o.hi),
        }
    }

    pub fn max(self, o: Self) -> Self {
        Self {
            lo: self.lo.max(o.lo),
            hi: self.hi.max(o.hi),
        }
    }

    pub fn exp(self) -> Self {
        Self::new(
            dn_libm(libm::exp(self.lo)).max(0.0),
            up_libm(libm::exp(self.hi)),
        )
    }

    /// The natural logarithm; negative members, whose logarithms are NaN,
    /// are left out (zero gives `-inf`).
    pub fn log(self) -> Self {
        if self.hi.is_nan() || self.hi <= 0.0 {
            return Self::ENTIRE;
        }
        let lo = if self.lo > 0.0 {
            dn_libm(libm::log(self.lo))
        } else {
            f64::NEG_INFINITY
        };
        Self::new(lo, up_libm(libm::log(self.hi)))
    }

    /// `self^p`, taking a zero member as +0 (the powers differ only for an
    /// odd integer `p`); negative members, whose powers are NaN for the
    /// non-integer `p` the tapes use, are left out, but for `-inf`, whose
    /// power is `+inf` (`p > 0`) or `+0`.
    pub fn pow(self, p: f64) -> Self {
        if self.hi.is_nan() || self.hi < 0.0 {
            return Self::ENTIRE;
        }
        let lo = if self.lo > 0.0 { self.lo } else { 0.0 };
        let hi = if self.hi > 0.0 { self.hi } else { 0.0 };
        let (mut small, mut large) = if p > 0.0 {
            (libm::pow(lo, p), libm::pow(hi, p))
        } else {
            (libm::pow(hi, p), libm::pow(lo, p))
        };
        if self.lo == f64::NEG_INFINITY {
            if p > 0.0 {
                large = f64::INFINITY;
            } else {
                small = 0.0;
            }
        }
        Self::bounds(dn_libm(small).max(0.0), up_libm(large))
    }

    pub fn sin(self) -> Self {
        self.periodic(libm::sin, FRAC_PI_2, -FRAC_PI_2)
    }

    pub fn cos(self) -> Self {
        self.periodic(libm::cos, 0.0, PI)
    }

    /// `sin` or `cos` (`f`), with maxima at `top + 2 pi k` and minima at
    /// `bottom + 2 pi k`: monotone between them, so the extremes over the
    /// interval are at its ends unless it holds one of those.
    fn periodic(self, f: fn(f64) -> f64, top: f64, bottom: f64) -> Self {
        let unit = Self::new(-ONE_UP, ONE_UP);
        let width = self.hi - self.lo;
        if width.is_nan() || width >= TAU || self.magnitude() > TRIG_LIMIT {
            return unit;
        }
        let (a, b) = (f(self.lo), f(self.hi));
        let hi = if hits(self.lo, self.hi, top) {
            ONE_UP
        } else {
            up_libm(a.max(b)).min(ONE_UP)
        };
        let lo = if hits(self.lo, self.hi, bottom) {
            -ONE_UP
        } else {
            dn_libm(a.min(b)).max(-ONE_UP)
        };
        Self::bounds(lo, hi)
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl std::ops::Add for Interval {
    type Output = Self;

    fn add(self, o: Self) -> Self {
        let (lo, hi) = (self.lo + o.lo, self.hi + o.hi);
        if lo.is_nan() || hi.is_nan() {
            return Self::ENTIRE;
        }
        let mut r = Self::new(dn(lo), up(hi));
        if self.lo >= 0.0 && o.lo >= 0.0 {
            r.lo = r.lo.max(0.0);
        }
        if self.hi <= 0.0 && o.hi <= 0.0 {
            r.hi = r.hi.min(0.0);
        }
        r
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;

    fn sub(self, o: Self) -> Self {
        let (lo, hi) = (self.lo - o.hi, self.hi - o.lo);
        if lo.is_nan() || hi.is_nan() {
            return Self::ENTIRE;
        }
        let mut r = Self::new(dn(lo), up(hi));
        if self.lo >= 0.0 && o.hi <= 0.0 {
            r.lo = r.lo.max(0.0);
        }
        if self.hi <= 0.0 && o.lo >= 0.0 {
            r.hi = r.hi.min(0.0);
        }
        r
    }
}

impl std::ops::Mul for Interval {
    type Output = Self;

    fn mul(self, o: Self) -> Self {
        let p = [
            mul_bound(self.lo, o.lo),
            mul_bound(self.lo, o.hi),
            mul_bound(self.hi, o.lo),
            mul_bound(self.hi, o.hi),
        ];
        if p.iter().any(|v| v.is_nan()) {
            return Self::ENTIRE;
        }
        let lo = p[0].min(p[1]).min(p[2].min(p[3]));
        let hi = p[0].max(p[1]).max(p[2].max(p[3]));
        let mut r = Self::new(dn(lo), up(hi));
        let (a_pos, a_neg) = (self.lo >= 0.0, self.hi <= 0.0);
        let (b_pos, b_neg) = (o.lo >= 0.0, o.hi <= 0.0);
        if (a_pos && b_pos) || (a_neg && b_neg) {
            r.lo = r.lo.max(0.0);
        }
        if (a_pos && b_neg) || (a_neg && b_pos) {
            r.hi = r.hi.min(0.0);
        }
        r
    }
}

impl std::ops::Div for Interval {
    type Output = Self;

    /// Division; a divisor touching zero (either signed zero can occur in
    /// the point evaluation there) gives the whole line.
    fn div(self, o: Self) -> Self {
        if !(o.lo > 0.0 || o.hi < 0.0) {
            return Self::ENTIRE;
        }
        let q = [
            self.lo / o.lo,
            self.lo / o.hi,
            self.hi / o.lo,
            self.hi / o.hi,
        ];
        if q.iter().any(|v| v.is_nan()) {
            return Self::ENTIRE;
        }
        let lo = q[0].min(q[1]).min(q[2].min(q[3]));
        let hi = q[0].max(q[1]).max(q[2].max(q[3]));
        let mut r = Self::new(dn(lo), up(hi));
        let (a_pos, a_neg) = (self.lo >= 0.0, self.hi <= 0.0);
        let (b_pos, b_neg) = (o.lo > 0.0, o.hi < 0.0);
        if (a_pos && b_pos) || (a_neg && b_neg) {
            r.lo = r.lo.max(0.0);
        }
        if (a_pos && b_neg) || (a_neg && b_pos) {
            r.hi = r.hi.min(0.0);
        }
        r
    }
}
