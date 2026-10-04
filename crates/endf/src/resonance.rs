//! Resolved resonance cross sections from MF=2 parameters, at 0 K.
//!
//! The pointwise cross sections transport reads come from a processing code
//! (NJOY) that reconstructed and Doppler broadened these same parameters.
//! Reconstructing them here is for what processing does not give: how the
//! cross sections move when the parameters move, which is what turns the
//! resonance-parameter covariance of MF=32 into a cross-section covariance.
//! That is a property of the parameters, so it is taken at 0 K.
//!
//! # Reich-Moore (LRF=3)
//!
//! ENDF-102 Appendix D.1.4. For each orbital angular momentum `l`, channel
//! spin `s` and total spin `J`, the resonances of that spin group give
//!
//! ```text
//! R_cc'(E) = sum_r (1/2) sqrt(G_rc(E) G_rc'(E)) / (E_r - E - i G_rg / 2)
//! W = (I - i R)^-1
//! U_nn = exp(-2 i phi) (2 W_nn - 1)
//! ```
//!
//! over the neutron channel and up to two fission channels, the capture width
//! eliminated into the denominators. The neutron width scales with the
//! penetrability, `G_rn(E) = G_rn P_l(E) / P_l(|E_r|)`, the fission widths
//! not at all, and Reich-Moore has no level shift. Then
//!
//! ```text
//! elastic    = pi/k^2   sum g_J |1 - U_nn|^2
//! capture    = 4 pi/k^2 sum g_J [W Im(R) W^H]_nn
//! fission    = 4 pi/k^2 sum g_J sum_f |W_nf|^2
//! ```
//!
//! Capture is ENDF-102's `Re W_nn - |W_nn|^2 - sum_f |W_nf|^2`, rewritten:
//! `W M = I` with `M = I - i R` and `R` complex symmetric give
//! `(W + W^H)/2 - W W^H = W Im(R) W^H`. The ENDF form subtracts numbers within
//! a part in `k` of one, which `pi/k^2` then multiplies: at 1e-5 eV it puts
//! Pb208's 1/v capture 0.4% off. `Im(R)` is a sum of positive Lorentzians in
//! the capture widths, so the rewritten form loses nothing.
//!
//! summed over every `(s, J)` the target spin and `l` allow, with or without
//! resonances, so the potential scattering of every channel is in. Where a
//! `J` is reached from both channel spins the sign of AJ picks one: negative
//! for `s = I - 1/2`.
//!
//! The channel radius `a` (in the penetrability) and the scattering radius
//! (in the phase shift) follow NAPS: with NAPS=0 the channel radius is
//! `0.123 A^(1/3) + 0.08` and the scattering radius AP (or the section's APL
//! where non-zero); with NAPS=1 both are the scattering radius.

use crate::error::{Error, Result};
use crate::mf::mf2::{ReichMoore, ResonanceParameters, ResonanceRange};

/// `k = WAVE_NUMBER * A / (A + 1) * sqrt(E)`, `k` in 1/(1e-12 cm) for `E` in
/// eV: `sqrt(2 m_n eV) 1e-12 cm / hbar` with the CODATA 2018 constants NJOY
/// 2016 uses.
pub const WAVE_NUMBER: f64 = 2.196_807_690_264e-3;

/// The neutron wave number in the centre-of-mass frame, in 1/(1e-12 cm).
pub fn wave_number(awri: f64, energy: f64) -> f64 {
    WAVE_NUMBER * awri / (awri + 1.0) * energy.abs().sqrt()
}

/// The hard-sphere penetrability `P_l(rho)` and shift `S_l(rho)`, for
/// `l <= 4`.
pub fn penetration_shift(l: i64, rho: f64) -> (f64, f64) {
    let r2 = rho * rho;
    match l {
        0 => (rho, 0.0),
        1 => {
            let d = 1.0 + r2;
            (rho * r2 / d, -1.0 / d)
        }
        2 => {
            let d = 9.0 + 3.0 * r2 + r2 * r2;
            (rho * r2 * r2 / d, -(18.0 + 3.0 * r2) / d)
        }
        3 => {
            let d = 225.0 + 45.0 * r2 + 6.0 * r2 * r2 + r2 * r2 * r2;
            (
                rho * r2 * r2 * r2 / d,
                -(675.0 + 90.0 * r2 + 6.0 * r2 * r2) / d,
            )
        }
        _ => {
            let r4 = r2 * r2;
            let d = 11025.0 + 1575.0 * r2 + 135.0 * r4 + 10.0 * r4 * r2 + r4 * r4;
            (
                rho * r4 * r4 / d,
                -(44100.0 + 4725.0 * r2 + 270.0 * r4 + 10.0 * r4 * r2) / d,
            )
        }
    }
}

/// The hard-sphere phase shift `phi_l(rho)`, for `l <= 4`.
pub fn phase_shift(l: i64, rho: f64) -> f64 {
    let r2 = rho * rho;
    match l {
        0 => rho,
        1 => rho - rho.atan(),
        2 => rho - (3.0 * rho / (3.0 - r2)).atan(),
        3 => rho - (rho * (15.0 - r2) / (15.0 - 6.0 * r2)).atan(),
        _ => rho - (rho * (105.0 - 10.0 * r2) / (105.0 - 45.0 * r2 + r2 * r2)).atan(),
    }
}

/// A complex number, enough of one for a few-channel R-matrix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn inv(self) -> Self {
        let d = self.norm_sqr();
        Complex::new(self.re / d, -self.im / d)
    }

    pub fn scale(self, s: f64) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

/// The inverse of the `n × n` row-major complex matrix `a`, by Gauss-Jordan
/// elimination with partial pivoting. `n` is at most three here.
pub fn invert(a: &[Complex], n: usize) -> Vec<Complex> {
    let mut m = a.to_vec();
    let mut out = vec![Complex::ZERO; n * n];
    for i in 0..n {
        out[i * n + i] = Complex::ONE;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| {
                m[x * n + col]
                    .norm_sqr()
                    .total_cmp(&m[y * n + col].norm_sqr())
            })
            .unwrap_or(col);
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
                out.swap(col * n + k, pivot * n + k);
            }
        }
        let inv = m[col * n + col].inv();
        for k in 0..n {
            m[col * n + k] = m[col * n + k] * inv;
            out[col * n + k] = out[col * n + k] * inv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row * n + col];
            if f == Complex::ZERO {
                continue;
            }
            for k in 0..n {
                m[row * n + k] = m[row * n + k] - f * m[col * n + k];
                out[row * n + k] = out[row * n + k] - f * out[col * n + k];
            }
        }
    }
    out
}

/// Elastic, capture and fission cross sections in barns.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrossSections {
    pub elastic: f64,
    pub capture: f64,
    pub fission: f64,
}

/// One Reich-Moore resonance, as reconstruction reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RmResonance {
    /// Where it is in MF=2: section and index.
    section: usize,
    index: usize,
    energy: f64,
    gn: f64,
    gg: f64,
    gfa: f64,
    gfb: f64,
    /// `P_l(|E_r|)` at the channel radius.
    penetrability: f64,
}

/// One `(l, s, J)` channel: its statistical weight and resonances.
#[derive(Debug, Clone, PartialEq)]
struct RmChannel {
    g: f64,
    resonances: Vec<RmResonance>,
    fission: bool,
}

/// One orbital angular momentum's channels and radii.
#[derive(Debug, Clone, PartialEq)]
struct RmOrbital {
    l: i64,
    awri: f64,
    channel_radius: f64,
    scattering_radius: f64,
    channels: Vec<RmChannel>,
}

/// A Reich-Moore range, prepared for reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct ReichMooreRange {
    pub el: f64,
    pub eh: f64,
    orbitals: Vec<RmOrbital>,
    /// The parameters it was prepared from, for the radius derivatives.
    source: ReichMoore,
    naps: i64,
}

/// The derivatives of the elastic, capture and fission cross sections with
/// respect to one parameter: barns per unit of the parameter (per eV for an
/// energy or a width, per 1e-12 cm for a radius).
pub type Gradient = [f64; 3];

/// One resonance's parameter derivatives at one energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonanceGradient {
    /// The resonance's MF=2 section and index.
    pub section: usize,
    pub index: usize,
    /// With respect to ER, GN, GG, GFA and GFB, in that order. A fission
    /// width of zero has no derivative (its amplitude is not differentiable
    /// there) and reads zero.
    pub d: [Gradient; 5],
}

/// One channel at one energy: what its cross sections and their
/// derivatives are built from.
struct ChannelState {
    n: usize,
    /// `W = (I - i R)^-1`, row-major `n × n`, symmetric.
    w: Vec<Complex>,
    /// `Im R`, row-major.
    im_r: Vec<f64>,
    /// Per resonance, its amplitudes `(sqrt G_n(E), sqrt G_fa, sqrt G_fb)`
    /// (signed) and `D_r = (1/2) / (E_r - E - i G_g / 2)`.
    amplitudes: Vec<[f64; 3]>,
    d: Vec<Complex>,
}

impl ChannelState {
    fn new(ch: &RmChannel, energy: f64, p: f64) -> Self {
        let n = if ch.fission { 3 } else { 1 };
        let mut r = vec![Complex::ZERO; n * n];
        let mut amplitudes = Vec::with_capacity(ch.resonances.len());
        let mut d = Vec::with_capacity(ch.resonances.len());
        for res in &ch.resonances {
            let gn = res.gn * p / res.penetrability;
            let dr = Complex::new(res.energy - energy, -0.5 * res.gg)
                .inv()
                .scale(0.5);
            let a = [amplitude(gn), amplitude(res.gfa), amplitude(res.gfb)];
            for i in 0..n {
                for j in 0..n {
                    r[i * n + j] = r[i * n + j] + dr.scale(a[i] * a[j]);
                }
            }
            amplitudes.push(a);
            d.push(dr);
        }
        // I - i R
        let mut m: Vec<Complex> = r.iter().map(|x| Complex::new(x.im, -x.re)).collect();
        for i in 0..n {
            m[i * n + i] = m[i * n + i] + Complex::ONE;
        }
        let w = if n == 1 {
            vec![m[0].inv()]
        } else {
            invert(&m, n)
        };
        ChannelState {
            n,
            w,
            im_r: r.iter().map(|x| x.im).collect(),
            amplitudes,
            d,
        }
    }

    /// `|1 - U_nn|^2`, `[W Im(R) W^H]_nn` and `sum_f |W_nf|^2`, for a rotation
    /// `exp(-2 i phi)`.
    fn parts(&self, rotation: Complex) -> (f64, f64, f64) {
        let n = self.n;
        let w = &self.w[..n];
        let u = rotation * (w[0].scale(2.0) - Complex::ONE);
        let mut capture = 0.0;
        for i in 0..n {
            for j in 0..n {
                capture += self.im_r[i * n + j] * (w[i] * conj(w[j])).re;
            }
        }
        let fission = w[1..].iter().map(|x| x.norm_sqr()).sum();
        ((Complex::ONE - u).norm_sqr(), capture, fission)
    }
}

fn conj(x: Complex) -> Complex {
    Complex::new(x.re, -x.im)
}

impl ReichMooreRange {
    /// Prepare `range` (LRF=3) for reconstruction.
    pub fn new(range: &ResonanceRange) -> Result<Self> {
        let ResonanceParameters::ReichMoore(rm) = &range.parameters else {
            return Err(Error::Unsupported {
                what: "reconstruction of a resolved range other than Reich-Moore",
            });
        };
        if range.nro != 0 {
            return Err(Error::Unsupported {
                what: "an energy-dependent scattering radius (NRO/=0)",
            });
        }
        Ok(ReichMooreRange {
            el: range.el,
            eh: range.eh,
            orbitals: orbitals(rm, range.naps)?,
            source: rm.clone(),
            naps: range.naps,
        })
    }

    /// The cross sections at `energy` (eV).
    pub fn cross_sections(&self, energy: f64) -> CrossSections {
        let mut out = CrossSections::default();
        for o in &self.orbitals {
            let (mut elastic, mut capture, mut fission) = (0.0, 0.0, 0.0);
            let k = wave_number(o.awri, energy);
            let factor = std::f64::consts::PI / (k * k);
            let (p, _) = penetration_shift(o.l, k * o.channel_radius);
            let phi = phase_shift(o.l, k * o.scattering_radius);
            let rotation = Complex::new((2.0 * phi).cos(), -(2.0 * phi).sin());
            for ch in &o.channels {
                let (e, c, f) = ChannelState::new(ch, energy, p).parts(rotation);
                elastic += ch.g * e;
                capture += ch.g * 4.0 * c;
                fission += ch.g * 4.0 * f;
            }
            out.elastic += factor * elastic;
            out.capture += factor * capture;
            out.fission += factor * fission;
        }
        out
    }

    /// Every resonance's parameter derivatives at `energy` (eV), from the
    /// R-matrix algebra: `M = I - i R` gives `dW = i W dR W`, and with `W`
    /// symmetric a parameter of resonance `r`, which moves
    /// `R = sum_r D_r a_r a_r^T` through `D_r` and the amplitudes `a_r`,
    /// moves `W_nj` by
    ///
    /// ```text
    /// i [dD (W a)_n (W a)_j + D ((W da)_n (W a)_j + (W a)_n (W da)_j)]
    /// ```
    ///
    /// which costs a few channel-sized products per parameter. The neutron
    /// width enters through `a_n = sqrt(G_n P(E) / P(|E_r|))`, so ER moves it
    /// through `P(|E_r|)` as well as through `D_r`.
    pub fn derivatives(&self, energy: f64) -> Vec<ResonanceGradient> {
        let mut out = Vec::new();
        for o in &self.orbitals {
            let k = wave_number(o.awri, energy);
            let factor = std::f64::consts::PI / (k * k);
            let (p, _) = penetration_shift(o.l, k * o.channel_radius);
            let phi = phase_shift(o.l, k * o.scattering_radius);
            let rotation = Complex::new((2.0 * phi).cos(), -(2.0 * phi).sin());
            for ch in &o.channels {
                let st = ChannelState::new(ch, energy, p);
                let n = st.n;
                let w = |i: usize, j: usize| st.w[i * n + j];
                let u = rotation * (w(0, 0).scale(2.0) - Complex::ONE);
                let one_minus_u = Complex::ONE - u;
                // v_i = sum_j Im R_ij conj(W_nj); H_ij = Re(W_ni conj(W_nj)).
                let v: Vec<Complex> = (0..n)
                    .map(|i| {
                        (0..n).fold(Complex::ZERO, |acc, j| {
                            acc + conj(w(0, j)).scale(st.im_r[i * n + j])
                        })
                    })
                    .collect();
                let h = |i: usize, j: usize| (w(0, i) * conj(w(0, j))).re;
                for (r, res) in ch.resonances.iter().enumerate() {
                    let a = st.amplitudes[r];
                    let dr = st.d[r];
                    // W a, the column every derivative of this resonance uses.
                    let wa: Vec<Complex> = (0..n)
                        .map(|i| (0..n).fold(Complex::ZERO, |acc, j| acc + w(i, j).scale(a[j])))
                        .collect();
                    let den = Complex::new(res.energy - energy, -0.5 * res.gg);
                    let den2 = (den * den).inv();
                    // The neutron amplitude per unit G_n, and per unit ER
                    // through P(|E_r|).
                    let ratio = p / res.penetrability;
                    let dan_dgn = if res.gn != 0.0 {
                        ratio.sqrt() / (2.0 * res.gn.abs().sqrt())
                    } else {
                        0.0
                    };
                    let dan_der = if res.energy != 0.0 {
                        let kr = wave_number(o.awri, res.energy);
                        let rho = kr * o.channel_radius;
                        let dp_drho = penetrability_slope(o.l, rho);
                        let dp_de = dp_drho * rho / (2.0 * res.energy.abs()) * res.energy.signum();
                        -0.5 * a[0] * dp_de / res.penetrability
                    } else {
                        0.0
                    };
                    let half_inv_sqrt = |g: f64| {
                        if g != 0.0 {
                            1.0 / (2.0 * g.abs().sqrt())
                        } else {
                            0.0
                        }
                    };
                    // (dD, da) per parameter: ER, GN, GG, GFA, GFB.
                    let params: [(Complex, [f64; 3]); 5] = [
                        (den2.scale(-0.5), [dan_der, 0.0, 0.0]),
                        (Complex::ZERO, [dan_dgn, 0.0, 0.0]),
                        (Complex::new(0.0, 0.25) * den2, [0.0; 3]),
                        (Complex::ZERO, [0.0, half_inv_sqrt(res.gfa), 0.0]),
                        (Complex::ZERO, [0.0, 0.0, half_inv_sqrt(res.gfb)]),
                    ];
                    let mut d = [[0.0; 3]; 5];
                    for (q, (dd, da)) in params.iter().enumerate() {
                        if q >= 3 && n == 1 {
                            continue;
                        }
                        let wda: Vec<Complex> = (0..n)
                            .map(|i| {
                                (0..n).fold(Complex::ZERO, |acc, j| acc + w(i, j).scale(da[j]))
                            })
                            .collect();
                        let i_unit = Complex::new(0.0, 1.0);
                        let dw: Vec<Complex> = (0..n)
                            .map(|j| {
                                i_unit
                                    * (*dd * wa[0] * wa[j] + dr * (wda[0] * wa[j] + wa[0] * wda[j]))
                            })
                            .collect();
                        // Elastic: d|1 - U|^2 = -2 Re(conj(1 - U) dU), dU = 2 rot dW_nn.
                        let du = rotation * dw[0].scale(2.0);
                        let de = -2.0 * (conj(one_minus_u) * du).re;
                        // Capture: 2 Re sum_i dW_ni v_i + sum_ij W_ni Im(dR_ij) conj(W_nj).
                        let mut dc = 2.0 * (0..n).map(|i| (dw[i] * v[i]).re).sum::<f64>();
                        for i in 0..n {
                            for j in 0..n {
                                let im_dr =
                                    dd.im * a[i] * a[j] + dr.im * (da[i] * a[j] + a[i] * da[j]);
                                dc += im_dr * h(i, j);
                            }
                        }
                        // Fission: 2 Re sum_f conj(W_nf) dW_nf.
                        let df = 2.0 * (1..n).map(|f| (conj(w(0, f)) * dw[f]).re).sum::<f64>();
                        d[q] = [
                            factor * ch.g * de,
                            factor * ch.g * 4.0 * dc,
                            factor * ch.g * 4.0 * df,
                        ];
                    }
                    out.push(ResonanceGradient {
                        section: res.section,
                        index: res.index,
                        d,
                    });
                }
            }
        }
        out
    }

    /// The derivative at `energy` with respect to a scattering radius: the
    /// range's AP (`l` of `None`), which every section without an APL of
    /// its own reads, or the APL of the sections of orbital angular momentum
    /// `l`. With NAPS=1 the radius is the channel radius too. By central
    /// difference, the radius entering the phase shift, the penetrability
    /// and its ratio at every resonance together.
    pub fn radius_derivative(&self, energy: f64, l: Option<i64>) -> Result<Gradient> {
        let shifted = |delta: f64| -> Result<CrossSections> {
            let mut rm = self.source.clone();
            let h = match l {
                None => {
                    let h = delta * rm.ap.abs().max(1e-3);
                    rm.ap += h;
                    h
                }
                Some(l) => {
                    let mut h = 0.0;
                    for s in rm.sections.iter_mut().filter(|s| s.l == l) {
                        let base = if s.apl != 0.0 { s.apl } else { self.source.ap };
                        h = delta * base.abs().max(1e-3);
                        s.apl = base + h;
                    }
                    h
                }
            };
            let range = ReichMooreRange {
                el: self.el,
                eh: self.eh,
                orbitals: orbitals(&rm, self.naps)?,
                source: rm,
                naps: self.naps,
            };
            let mut x = range.cross_sections(energy);
            if h != 0.0 {
                x.elastic /= h;
                x.capture /= h;
                x.fission /= h;
            }
            Ok(x)
        };
        // No section of that l: nothing to move, where the division below
        // would otherwise be skipped and the cross sections returned.
        if let Some(l) = l {
            if !self.source.sections.iter().any(|s| s.l == l) {
                return Ok([0.0; 3]);
            }
        }
        const STEP: f64 = 1e-6;
        let (up, down) = (shifted(STEP)?, shifted(-STEP)?);
        // (sigma(+h) / h - sigma(-h) / (-h)) / 2 with h the same size each way.
        Ok([
            0.5 * (up.elastic + down.elastic),
            0.5 * (up.capture + down.capture),
            0.5 * (up.fission + down.fission),
        ])
    }
}

/// `dP_l / d rho` by central difference.
fn penetrability_slope(l: i64, rho: f64) -> f64 {
    let h = 1e-6 * rho.abs().max(1e-6);
    (penetration_shift(l, rho + h).0 - penetration_shift(l, rho - h).0) / (2.0 * h)
}

/// `sqrt(|x|)` with the sign of `x`: a width's amplitude.
fn amplitude(x: f64) -> f64 {
    x.signum() * x.abs().sqrt()
}

/// Every `(l, s, J)` channel of a Reich-Moore range, with its resonances.
fn orbitals(rm: &ReichMoore, naps: i64) -> Result<Vec<RmOrbital>> {
    let spin = rm.spi;
    let channel_spins: Vec<f64> = if spin == 0.0 {
        vec![0.5]
    } else {
        vec![spin - 0.5, spin + 0.5]
    };
    let mut out = Vec::with_capacity(rm.sections.len());
    for (section, s) in rm.sections.iter().enumerate() {
        if s.l > 4 {
            return Err(Error::Unsupported {
                what: "a Reich-Moore section with l > 4",
            });
        }
        let scattering_radius = if s.apl != 0.0 { s.apl } else { rm.ap };
        let channel_radius = match naps {
            0 => 0.123 * s.awri.cbrt() + 0.08,
            _ => scattering_radius,
        };
        let l = s.l as f64;
        // Every (s, J) the spins allow, with J weighted by (2J + 1) / (2 (2I + 1)).
        let mut channels: Vec<(f64, f64, RmChannel)> = Vec::new();
        for &cs in &channel_spins {
            let mut j = (l - cs).abs();
            while j <= l + cs + 1e-9 {
                channels.push((
                    cs,
                    j,
                    RmChannel {
                        g: (2.0 * j + 1.0) / (2.0 * (2.0 * spin + 1.0)),
                        resonances: Vec::new(),
                        fission: false,
                    },
                ));
                j += 1.0;
            }
        }
        for index in 0..s.er.len() {
            let j = s.aj[index].abs();
            let candidates: Vec<usize> = channels
                .iter()
                .enumerate()
                .filter(|(_, c)| (c.1 - j).abs() < 1e-6)
                .map(|(k, _)| k)
                .collect();
            let k = match candidates.as_slice() {
                [] => continue,
                [only] => *only,
                [low, high, ..] => {
                    if s.aj[index] < 0.0 {
                        *low
                    } else {
                        *high
                    }
                }
            };
            let energy = s.er[index];
            let (penetrability, _) =
                penetration_shift(s.l, wave_number(s.awri, energy) * channel_radius);
            let res = RmResonance {
                section,
                index,
                energy,
                gn: s.gn[index],
                gg: s.gg[index],
                gfa: s.gfa[index],
                gfb: s.gfb[index],
                penetrability,
            };
            let ch = &mut channels[k].2;
            ch.fission |= res.gfa != 0.0 || res.gfb != 0.0;
            ch.resonances.push(res);
        }
        out.push(RmOrbital {
            l: s.l,
            awri: s.awri,
            channel_radius,
            scattering_radius,
            channels: channels.into_iter().map(|c| c.2).collect(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    const DY158: &[u8] = include_bytes!("../fixtures/n-066_Dy_158_mf2_mf32.endf.xz");
    const TH232: &[u8] = include_bytes!("../fixtures/n-090_Th_232_mf2_mf32.endf.xz");
    const U235: &[u8] = include_bytes!("../fixtures/n-092_U_235_mf2.endf.xz");
    const PB208: &[u8] = include_bytes!("../fixtures/n-082_Pb_208_mf2.endf.xz");

    fn range(fixture: &[u8]) -> ReichMooreRange {
        let m = Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses");
        ReichMooreRange::new(&m.mf2().unwrap().isotopes[0].ranges[0]).expect("Reich-Moore")
    }

    // NJOY 2016 RECONR at 0 K (err 1e-5) on ENDF/B-VIII.1, less the MF=3
    // background: `(E, elastic, capture, fission)` at its own grid points,
    // written to the seven digits ENDF carries.
    const DY158_NJOY: &[(f64, f64, f64, f64)] = &[
        (1e-5, 6.522522e0, 2.218698e3, 0.000000e0),
        (8.75e-5, 6.522482e0, 7.500018e2, 0.000000e0),
        (7.988282e-4, 6.522113e0, 2.480553e2, 0.000000e0),
        (7.363282e-3, 6.518729e0, 8.120027e1, 0.000000e0),
        (7.050782e-2, 6.487268e0, 2.475542e1, 0.000000e0),
        (4.871094e-1, 6.318780e0, 6.706585e0, 0.000000e0),
        (8.085938e0, 5.515248e0, 2.906712e-1, 0.000000e0),
        (3.416771e1, 1.770951e0, 2.951286e0, 0.000000e0),
        (3.665166e1, 3.838921e-2, 1.995118e1, 0.000000e0),
        (3.742269e1, 9.579958e0, 1.037630e2, 0.000000e0),
        (3.789446e1, 5.109348e2, 2.409986e3, 0.000000e0),
        (3.815143e1, 3.880050e2, 1.316606e3, 0.000000e0),
        (3.892886e1, 2.391940e1, 4.047042e1, 0.000000e0),
        (4.301557e1, 1.515129e0, 9.145920e0, 0.000000e0),
        (4.431116e1, 5.253304e-1, 3.176088e1, 0.000000e0),
        (4.502149e1, 2.712181e1, 1.514356e2, 0.000000e0),
        (4.549854e1, 1.341677e3, 3.516981e3, 0.000000e0),
        (4.582858e1, 5.047300e2, 8.977368e2, 0.000000e0),
        (4.702782e1, 3.658503e1, 2.489158e1, 0.000000e0),
        (5.415614e1, 9.826010e0, 8.169185e-1, 0.000000e0),
        (7.925069e1, 3.192413e0, 1.326305e0, 0.000000e0),
        (8.362383e1, 3.171043e-2, 1.006309e1, 0.000000e0),
        (8.516703e1, 3.224629e1, 8.014492e1, 0.000000e0),
        (8.58367e1, 1.296311e3, 1.647464e3, 0.000000e0),
        (8.619777e1, 1.258930e3, 1.201908e3, 0.000000e0),
    ];
    const TH232_NJOY: &[(f64, f64, f64, f64)] = &[
        (1e-5, 1.302689e1, 3.734484e2, 0.000000e0),
        (1.288126e2, 9.441012e0, 3.109755e0, 0.000000e0),
        (2.632265e2, 2.295970e2, 3.538675e2, 0.000000e0),
        (4.011742e2, 3.689159e2, 9.382659e2, 0.000000e0),
        (5.700426e2, 9.813886e-1, 1.586962e1, 0.000000e0),
        (7.134932e2, 1.683865e2, 7.641163e1, 0.000000e0),
        (8.911608e2, 3.690209e1, 4.068602e0, 0.000000e0),
        (1.048651e3, 1.272971e1, 8.871328e-3, 0.000000e0),
        (1.210153e3, 1.054697e1, 5.953654e0, 0.000000e0),
        (1.373644e3, 9.760098e0, 1.609773e-1, 0.000000e0),
        (1.52484e3, 2.741897e0, 3.800675e0, 0.000000e0),
        (1.7074725e3, 1.100839e1, 6.940386e1, 0.000000e0),
        (1.875091e3, 1.257807e1, 1.826299e-2, 0.000000e0),
        (2.058165e3, 1.120136e1, 1.281281e0, 0.000000e0),
        (2.2186e3, 2.128555e1, 1.705278e0, 0.000000e0),
        (2.390451e3, 1.235052e1, 2.572392e-1, 0.000000e0),
        (2.565462e3, 7.523093e2, 5.065386e1, 0.000000e0),
        (2.746112e3, 1.720101e1, 1.046888e-1, 0.000000e0),
        (2.915365e3, 1.206926e1, 1.952789e0, 0.000000e0),
        (3.0803945e3, 1.368451e1, 1.723364e1, 0.000000e0),
        (3.243665e3, 1.108657e1, 4.815859e-2, 0.000000e0),
        (3.4395435e3, 1.262655e1, 8.565992e0, 0.000000e0),
        (3.6098675e3, 8.793223e0, 2.536470e0, 0.000000e0),
        (3.765513e3, 1.199396e1, 1.116069e0, 0.000000e0),
        (3.999991e3, 1.784173e0, 8.306196e0, 0.000000e0),
    ];
    const U235_NJOY: &[(f64, f64, f64, f64)] = &[
        (1e-5, 1.416891e1, 5.456465e3, 3.145416e4),
        (3.352642e1, 4.486374e1, 6.190747e2, 4.256915e2),
        (7.234617e1, 1.130133e1, 5.377284e1, 1.884434e2),
        (1.151946e2, 1.008678e1, 5.047165e0, 7.762199e0),
        (1.635931e2, 1.145416e1, 4.452145e1, 1.379687e2),
        (2.222725e2, 1.112586e1, 2.255684e0, 1.238752e1),
        (2.913089e2, 1.240493e1, 9.764711e-1, 5.871529e0),
        (3.615781e2, 1.307543e1, 8.473493e0, 4.949729e1),
        (4.445831e2, 1.198694e1, 1.262516e0, 6.549403e0),
        (5.315011e2, 9.003733e0, 3.702691e1, 1.407545e1),
        (6.285408e2, 1.208044e1, 2.544300e1, 6.648915e1),
        (7.237599e2, 1.253192e1, 2.220451e0, 1.675112e1),
        (8.247392e2, 1.142246e1, 9.354020e-1, 1.721328e0),
        (9.264195e2, 1.121230e1, 3.351066e0, 7.188375e0),
        (1.028459e3, 1.240776e1, 1.251730e0, 1.084396e0),
        (1.134419e3, 1.308997e1, 9.027337e-1, 5.069614e0),
        (1.250841e3, 1.212067e1, 2.071906e0, 4.803945e0),
        (1.371196e3, 9.030248e0, 3.469161e0, 3.378627e0),
        (1.493033e3, 1.106879e1, 4.405900e-1, 9.224983e-1),
        (1.606516e3, 5.521971e1, 2.070394e1, 5.275093e1),
        (1.73238e3, 1.384430e1, 1.067423e0, 3.020337e0),
        (1.865312e3, 1.016054e1, 8.319769e-1, 1.228107e0),
        (2.0025459e3, 8.877426e0, 4.045683e0, 7.937621e0),
        (2.118722e3, 1.088474e1, 1.298200e0, 4.749021e0),
        (2.249999e3, 1.021441e1, 5.775862e-1, 1.640655e0),
    ];

    /// The reconstruction agrees with NJOY to the seven digits NJOY writes,
    /// elastic, capture and fission, across each range.
    #[test]
    fn reich_moore_matches_njoy() {
        for (fixture, reference) in [(DY158, DY158_NJOY), (TH232, TH232_NJOY), (U235, U235_NJOY)] {
            let rm = range(fixture);
            for &(e, elastic, capture, fission) in reference {
                let x = rm.cross_sections(e);
                for (ours, njoy, what) in [
                    (x.elastic, elastic, "elastic"),
                    (x.capture, capture, "capture"),
                    (x.fission, fission, "fission"),
                ] {
                    // Seven digits of NJOY's total, less a background of up
                    // to a few barns, at an interference dip.
                    let tolerance = 6e-7 * njoy.abs() + 1e-6;
                    assert!(
                        (ours - njoy).abs() <= tolerance,
                        "{what} at {e} eV: {ours} against NJOY's {njoy}"
                    );
                }
            }
        }
    }

    /// Far below Pb208's first resonance (506 keV) capture is 1/v: below
    /// 0.01 eV the p-wave resonances add less than a part in 1e5. ENDF-102's
    /// `Re W - |W|^2` keeps only about four digits there (it was 1e-4 to
    /// 4e-3 off), subtracting numbers within 1e-12 of one; the form used here
    /// keeps them.
    #[test]
    fn thermal_capture_keeps_its_digits() {
        let rm = range(PB208);
        let at = |e: f64| rm.cross_sections(e).capture * e.sqrt();
        let reference = at(1e-3);
        for e in [1e-5, 3e-5, 1e-4, 4e-3, 1e-2] {
            assert!(
                (at(e) / reference - 1.0).abs() < 1e-5,
                "capture * sqrt(E) at {e} eV is {} against {reference}",
                at(e)
            );
        }
    }
}

#[cfg(test)]
mod derivative_tests {
    use super::*;
    use crate::material::Material;

    const DY158: &[u8] = include_bytes!("../fixtures/n-066_Dy_158_mf2_mf32.endf.xz");
    const TH232: &[u8] = include_bytes!("../fixtures/n-090_Th_232_mf2_mf32.endf.xz");
    const U235: &[u8] = include_bytes!("../fixtures/n-092_U_235_mf2.endf.xz");

    fn mf2_range(fixture: &[u8]) -> ResonanceRange {
        let m = Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses");
        m.mf2().unwrap().isotopes[0].ranges[0].clone()
    }

    /// `range` with parameter `q` (ER, GN, GG, GFA, GFB) of resonance
    /// `(section, index)` moved by `delta`.
    fn moved(
        range: &ResonanceRange,
        section: usize,
        index: usize,
        q: usize,
        delta: f64,
    ) -> ResonanceRange {
        let mut r = range.clone();
        let ResonanceParameters::ReichMoore(rm) = &mut r.parameters else {
            unreachable!()
        };
        let s = &mut rm.sections[section];
        let field = match q {
            0 => &mut s.er,
            1 => &mut s.gn,
            2 => &mut s.gg,
            3 => &mut s.gfa,
            _ => &mut s.gfb,
        };
        field[index] += delta;
        r
    }

    /// Every resonance parameter's analytic derivative matches a central
    /// difference of the reconstruction, at energies on, between and far
    /// from the resonances, for elastic, capture and fission.
    #[test]
    fn parameter_derivatives_match_central_differences() {
        let mut checked = 0;
        for fixture in [DY158, TH232, U235] {
            let range = mf2_range(fixture);
            let rm = ReichMooreRange::new(&range).unwrap();
            let ResonanceParameters::ReichMoore(params) = &range.parameters else {
                unreachable!()
            };
            // A spread of resonances, and energies on them and beside them.
            for (section, s) in params.sections.iter().enumerate() {
                let step = (s.er.len() / 4).max(1);
                for index in (0..s.er.len()).step_by(step) {
                    let er = s.er[index];
                    if er <= range.el || er >= range.eh {
                        continue;
                    }
                    let width = (s.gn[index].abs() + s.gg[index].abs()).max(1e-3);
                    for e in [er, er + 0.7 * width, er * 1.3, 0.0253] {
                        if e <= range.el || e >= range.eh {
                            continue;
                        }
                        let analytic = rm
                            .derivatives(e)
                            .into_iter()
                            .find(|g| g.section == section && g.index == index)
                            .expect("every resonance has a gradient");
                        let values = [er, s.gn[index], s.gg[index], s.gfa[index], s.gfb[index]];
                        for (q, &value_q) in values.iter().enumerate() {
                            if q >= 3 && value_q == 0.0 {
                                continue;
                            }
                            let h = if q == 0 {
                                1e-4 * width
                            } else {
                                1e-5 * value_q.abs().max(1e-6)
                            };
                            let up = ReichMooreRange::new(&moved(&range, section, index, q, h))
                                .unwrap()
                                .cross_sections(e);
                            let down = ReichMooreRange::new(&moved(&range, section, index, q, -h))
                                .unwrap()
                                .cross_sections(e);
                            let numeric = [
                                (up.elastic - down.elastic) / (2.0 * h),
                                (up.capture - down.capture) / (2.0 * h),
                                (up.fission - down.fission) / (2.0 * h),
                            ];
                            let x = rm.cross_sections(e);
                            for c in 0..3 {
                                let value = [x.elastic, x.capture, x.fission][c];
                                // Relative to the derivative, plus the rounding
                                // noise of the difference itself: a sum of
                                // hundreds of terms carries about 1e-14 of it.
                                let noise = 1e-14 * value.abs() / h;
                                let tolerance = 1e-4 * numeric[c].abs() + noise;
                                assert!(
                                    (analytic.d[q][c] - numeric[c]).abs() <= tolerance,
                                    "parameter {q} of resonance {section}/{index} at {e} eV, \
                                     reaction {c}: analytic {} against numeric {}",
                                    analytic.d[q][c],
                                    numeric[c]
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 500, "{checked}");
    }

    /// At thermal energy Pb208's elastic is s-wave, `4 pi a_eff^2` with
    /// `a_eff` its section radius APL less the distant resonances' share,
    /// which (NAPS=1, `P_0` proportional to the radius) does not depend on
    /// the radius. So the APL derivative is `8 pi a_eff`. Every Pb208 section
    /// has its own APL, so the range-wide AP moves nothing.
    #[test]
    fn radius_derivative_is_that_of_potential_scattering() {
        const PB208: &[u8] = include_bytes!("../fixtures/n-082_Pb_208_mf2.endf.xz");
        let rm = ReichMooreRange::new(&mf2_range(PB208)).unwrap();
        let d = rm.radius_derivative(0.0253, Some(0)).unwrap();
        let elastic = rm.cross_sections(0.0253).elastic;
        let want = 8.0 * std::f64::consts::PI * (elastic / (4.0 * std::f64::consts::PI)).sqrt();
        assert!((d[0] / want - 1.0).abs() < 1e-3, "{} against {want}", d[0]);
        let none = rm.radius_derivative(0.0253, None).unwrap();
        assert_eq!(none, [0.0, 0.0, 0.0]);
        // Pb208 has no l=7 section.
        assert_eq!(rm.radius_derivative(0.0253, Some(7)).unwrap(), [0.0; 3]);
    }
}
