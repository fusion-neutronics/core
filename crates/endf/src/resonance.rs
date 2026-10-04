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
//! `0.123 A^(1/3) + 0.08`, `A` the target mass in amu (`AWRI` times the
//! neutron's, as NJOY takes it), and the scattering radius AP (or the
//! section's APL where non-zero); with NAPS=1 both are the scattering radius.

use crate::error::{Error, Result};
use crate::mf::mf2::{ReichMoore, ResonanceParameters, ResonanceRange};

/// `k = WAVE_NUMBER * A / (A + 1) * sqrt(E)`, `k` in 1/(1e-12 cm) for `E` in
/// eV: `sqrt(2 m_n eV) 1e-12 cm / hbar` with the CODATA 2018 constants NJOY
/// 2016 uses.
pub const WAVE_NUMBER: f64 = 2.196_807_122_623e-3;

/// The neutron mass in atomic mass units, as NJOY 2016 has it: the channel
/// radius formula takes the target mass in amu, `AWRI` times this.
pub const NEUTRON_MASS_AMU: f64 = 1.008_664_915_95;

/// The channel radius ENDF-102 gives when NAPS=0, `0.123 A^(1/3) + 0.08`
/// (1e-12 cm), `A` the target mass in amu.
pub fn channel_radius_formula(awri: f64) -> f64 {
    0.123 * (awri * NEUTRON_MASS_AMU).cbrt() + 0.08
}

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

    /// The derivative at `energy` with respect to the range's radius
    /// parameter, which moves section `i`'s radius by `steps[i]` (1e-12 cm)
    /// per unit: a section without an APL of its own through AP, one with an
    /// APL through it. By central difference, the radius entering the phase
    /// shift, the penetrability and every resonance's penetrability ratio
    /// together.
    pub fn radius_derivative(&self, energy: f64, steps: &[f64]) -> Result<Gradient> {
        let shifted = |z: f64| -> Result<CrossSections> {
            let mut rm = self.source.clone();
            let ap = rm.ap;
            for (s, step) in rm.sections.iter_mut().zip(steps) {
                let base = if s.apl != 0.0 { s.apl } else { ap };
                s.apl = base + z * step;
            }
            let orbitals = orbitals(&rm, self.naps)?;
            Ok(ReichMooreRange {
                el: self.el,
                eh: self.eh,
                orbitals,
                source: rm,
                naps: self.naps,
            }
            .cross_sections(energy))
        };
        const Z: f64 = 1e-4;
        let (up, down) = (shifted(Z)?, shifted(-Z)?);
        Ok([
            (up.elastic - down.elastic) / (2.0 * Z),
            (up.capture - down.capture) / (2.0 * Z),
            (up.fission - down.fission) / (2.0 * Z),
        ])
    }
}

/// A resolved range that can give its cross sections and their derivatives
/// with respect to MF=32's parameters: what the group covariance of
/// [`crate::resonance_covariance::group_covariance`] needs from a formalism.
pub trait RangeReconstruction {
    /// The range's energy bounds, eV.
    fn bounds(&self) -> (f64, f64);

    /// The cross sections at `energy`.
    fn cross_sections(&self, energy: f64) -> CrossSections;

    /// Each resonance's energy and total width, eV, for placing the points
    /// an integral over the range needs.
    fn resonances(&self) -> Vec<(f64, f64)>;

    /// The derivatives at `energy` with respect to each of `cov`'s
    /// parameters, in order. A parameter this range does not have reads zero.
    fn parameter_gradients(
        &self,
        energy: f64,
        cov: &crate::resonance_covariance::RangeCovariance,
    ) -> Result<Vec<Gradient>>;
}

impl RangeReconstruction for ReichMooreRange {
    fn bounds(&self) -> (f64, f64) {
        (self.el, self.eh)
    }

    fn cross_sections(&self, energy: f64) -> CrossSections {
        ReichMooreRange::cross_sections(self, energy)
    }

    fn resonances(&self) -> Vec<(f64, f64)> {
        self.orbitals
            .iter()
            .flat_map(|o| o.channels.iter())
            .flat_map(|c| c.resonances.iter())
            .map(|r| {
                (
                    r.energy,
                    r.gn.abs() + r.gg.abs() + r.gfa.abs() + r.gfb.abs(),
                )
            })
            .collect()
    }

    fn parameter_gradients(
        &self,
        energy: f64,
        cov: &crate::resonance_covariance::RangeCovariance,
    ) -> Result<Vec<Gradient>> {
        let parameters = &cov.parameters;
        use crate::resonance_covariance::{Location, Quantity};
        let by_resonance: std::collections::HashMap<(usize, usize), [Gradient; 5]> = self
            .derivatives(energy)
            .into_iter()
            .map(|g| ((g.section, g.index), g.d))
            .collect();
        let mut radius: Option<Gradient> = None;
        let mut out = Vec::with_capacity(parameters.len());
        for p in parameters {
            let g = match (p.location, p.quantity) {
                (Location::Orbital { section, index }, q) => {
                    let row = match q {
                        Quantity::Energy => 0,
                        Quantity::NeutronWidth => 1,
                        Quantity::CaptureWidth => 2,
                        Quantity::FissionWidth => 3,
                        Quantity::SecondFissionWidth => 4,
                        _ => {
                            out.push([0.0; 3]);
                            continue;
                        }
                    };
                    by_resonance
                        .get(&(section, index))
                        .map_or([0.0; 3], |d| d[row])
                }
                (Location::Range, Quantity::ScatteringRadius) => match radius {
                    Some(g) => g,
                    None => {
                        let g = self.radius_derivative(energy, &cov.radius_steps)?;
                        radius = Some(g);
                        g
                    }
                },
                _ => [0.0; 3],
            };
            out.push(g);
        }
        Ok(out)
    }
}

/// One explicit channel of an R-matrix limited spin group: a neutron channel
/// (the photon channel is eliminated into the denominators).
#[derive(Debug, Clone, PartialEq)]
struct RmlChannel {
    /// Index among the spin group's channels in MF=2.
    index: usize,
    l: i64,
    /// Radius for the penetrability and shift (APT), and for the phase shift
    /// (APE).
    penetrability_radius: f64,
    phase_radius: f64,
    shift: bool,
    boundary: f64,
}

/// One R-matrix limited resonance, as reconstruction reads it.
#[derive(Debug, Clone, PartialEq)]
struct RmlResonance {
    index: usize,
    energy: f64,
    /// The eliminated photon channel's width.
    capture: f64,
    /// Per explicit channel: its width, and its penetrability at `|E_r|`.
    widths: Vec<f64>,
    penetrability: Vec<f64>,
}

/// One spin group: its statistical weight, mass ratio, channels and
/// resonances.
#[derive(Debug, Clone, PartialEq)]
struct RmlGroup {
    g: f64,
    mass_ratio: f64,
    /// The photon channel's index among the spin group's channels.
    photon: usize,
    channels: Vec<RmlChannel>,
    resonances: Vec<RmlResonance>,
}

/// An R-matrix limited range (LRF=7), prepared for reconstruction.
///
/// ENDF-102 Appendix D.1.6 in the Reich-Moore approximation (KRM=3) with the
/// widths themselves given (IFG=0): per spin group, over its explicit
/// (neutron) channels,
///
/// ```text
/// R~_cc'(E) = sum_r (1/2) sqrt(G_rc(E) G_rc'(E)) / (E_r - E - i G_rg / 2)
/// M = I - R~ (Delta + i I),  Delta_c = (S_c(E) - B_c) / P_c(E) with a shift
/// U = Omega (I + 2 i M^-1 R~) Omega
/// elastic = pi/k^2 g sum_cc' |delta_cc' - U_cc'|^2
/// capture = pi/k^2 g sum_c 4 [M^-1 Im(R~) M^-H]_cc
/// ```
///
/// with `G_rc(E) = G_rc P_c(E) / P_c(|E_r|)`, `Omega = exp(-i phi_c)`, the
/// photon channel's width in the denominators, the penetrability and shift
/// at the channel's true radius APT and the phase at its effective radius
/// APE (ENDF-102 2.2.1.6; V51 is where the two differ). Capture is
/// `1 - sum_c' |U_cc'|^2` rewritten (`M = R~ Z` with `Z = R~^-1 - Delta - iI`
/// complex symmetric), the form that keeps its digits far below a resonance,
/// as for Reich-Moore. Only the channels a spin group lists scatter: the
/// format gives no others.
///
/// Particle pairs other than the photon and the neutron (fission, charged
/// particles), background R-matrices, tabulated phase shifts and reduced
/// width amplitudes (IFG=1) are refused by name.
#[derive(Debug, Clone, PartialEq)]
pub struct RMatrixRange {
    pub el: f64,
    pub eh: f64,
    groups: Vec<RmlGroup>,
    source: crate::mf::mf2::RMatrixLimited,
}

/// One spin group at one energy.
struct RmlState {
    n: usize,
    /// `R~`, `M^-1` and `X = M^-1 R~`, row-major `n × n`.
    r: Vec<Complex>,
    inv: Vec<Complex>,
    x: Vec<Complex>,
    /// `Delta + i` per channel, `exp(-i phi)` per channel.
    k: Vec<Complex>,
    omega: Vec<Complex>,
    /// Per resonance: amplitudes per channel and `D_r`.
    amplitudes: Vec<Vec<f64>>,
    d: Vec<Complex>,
    /// `P_c(E)` per channel.
    p: Vec<f64>,
}

fn matmul(a: &[Complex], b: &[Complex], n: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex::ZERO;
            for k in 0..n {
                acc = acc + a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

impl RmlState {
    fn new(group: &RmlGroup, energy: f64) -> Self {
        let n = group.channels.len();
        let k_wave = WAVE_NUMBER * group.mass_ratio * energy.abs().sqrt();
        let mut p = vec![0.0; n];
        let mut k = vec![Complex::ZERO; n];
        let mut omega = vec![Complex::ONE; n];
        for (c, ch) in group.channels.iter().enumerate() {
            let (pc, sc) = penetration_shift(ch.l, k_wave * ch.penetrability_radius);
            p[c] = pc;
            let delta = if ch.shift && pc > 0.0 {
                (sc - ch.boundary) / pc
            } else {
                0.0
            };
            k[c] = Complex::new(delta, 1.0);
            let phi = phase_shift(ch.l, k_wave * ch.phase_radius);
            omega[c] = Complex::new(phi.cos(), -phi.sin());
        }
        let mut r = vec![Complex::ZERO; n * n];
        let mut amplitudes = Vec::with_capacity(group.resonances.len());
        let mut d = Vec::with_capacity(group.resonances.len());
        for res in &group.resonances {
            let a: Vec<f64> = (0..n)
                .map(|c| {
                    if res.penetrability[c] > 0.0 {
                        amplitude(res.widths[c] * p[c] / res.penetrability[c])
                    } else {
                        0.0
                    }
                })
                .collect();
            let dr = Complex::new(res.energy - energy, -0.5 * res.capture)
                .inv()
                .scale(0.5);
            for i in 0..n {
                for j in 0..n {
                    r[i * n + j] = r[i * n + j] + dr.scale(a[i] * a[j]);
                }
            }
            amplitudes.push(a);
            d.push(dr);
        }
        // M = I - R~ K
        let mut m = vec![Complex::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = -(r[i * n + j] * k[j]);
            }
            m[i * n + i] = m[i * n + i] + Complex::ONE;
        }
        let inv = if n == 1 {
            vec![m[0].inv()]
        } else {
            invert(&m, n)
        };
        let x = matmul(&inv, &r, n);
        RmlState {
            n,
            r,
            inv,
            x,
            k,
            omega,
            amplitudes,
            d,
            p,
        }
    }

    /// `U_cc'`.
    fn u(&self, c: usize, cp: usize) -> Complex {
        let n = self.n;
        let delta = if c == cp { Complex::ONE } else { Complex::ZERO };
        self.omega[c] * (delta + Complex::new(0.0, 2.0) * self.x[c * n + cp]) * self.omega[cp]
    }

    /// `sum_cc' |delta - U|^2` and `sum_c 4 [M^-1 Im(R~) M^-H]_cc`.
    fn parts(&self) -> (f64, f64) {
        let n = self.n;
        let mut elastic = 0.0;
        for c in 0..n {
            for cp in 0..n {
                let delta = if c == cp { Complex::ONE } else { Complex::ZERO };
                elastic += (delta - self.u(c, cp)).norm_sqr();
            }
        }
        let mut capture = 0.0;
        for c in 0..n {
            for i in 0..n {
                for j in 0..n {
                    capture +=
                        self.r[i * n + j].im * (self.inv[c * n + i] * conj(self.inv[c * n + j])).re;
                }
            }
        }
        (elastic, 4.0 * capture)
    }
}

impl RMatrixRange {
    /// Prepare `range` (LRF=7) for reconstruction.
    pub fn new(range: &ResonanceRange) -> Result<Self> {
        let ResonanceParameters::RMatrixLimited(rml) = &range.parameters else {
            return Err(Error::Unsupported {
                what: "reconstruction of a resolved range other than R-matrix limited",
            });
        };
        Ok(RMatrixRange {
            el: range.el,
            eh: range.eh,
            groups: rml_groups(rml)?,
            source: (**rml).clone(),
        })
    }

    /// The cross sections at `energy` (eV).
    pub fn cross_sections(&self, energy: f64) -> CrossSections {
        let mut out = CrossSections::default();
        for g in &self.groups {
            let st = RmlState::new(g, energy);
            let k = WAVE_NUMBER * g.mass_ratio * energy.abs().sqrt();
            let factor = std::f64::consts::PI / (k * k) * g.g;
            let (e, c) = st.parts();
            out.elastic += factor * e;
            out.capture += factor * c;
        }
        out
    }

    /// Every resonance's derivatives at `energy` with respect to ER and each
    /// of its spin group's channel widths (in the group's channel order, the
    /// photon channel's being the capture width), as `(group, index, d)`.
    /// `dM = -dR~ K` gives `dM^-1 = M^-1 dR~ K M^-1` and
    /// `dX = M^-1 dR~ (K X + I)`.
    pub fn derivatives(&self, energy: f64) -> Vec<(usize, usize, Vec<Gradient>)> {
        let mut out = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            let st = RmlState::new(g, energy);
            let n = st.n;
            let kw = WAVE_NUMBER * g.mass_ratio * energy.abs().sqrt();
            let factor = std::f64::consts::PI / (kw * kw) * g.g;
            // K X + I, shared by every parameter.
            let mut kx_i = vec![Complex::ZERO; n * n];
            for i in 0..n {
                for j in 0..n {
                    kx_i[i * n + j] = st.k[i] * st.x[i * n + j];
                }
                kx_i[i * n + i] = kx_i[i * n + i] + Complex::ONE;
            }
            // ER, then every MF=2 channel's width: the photon's and the
            // explicit channels'.
            let nch = 1 + g.channels.len() + 1;
            for (r, res) in g.resonances.iter().enumerate() {
                let a = &st.amplitudes[r];
                let dr = st.d[r];
                let den = Complex::new(res.energy - energy, -0.5 * res.capture);
                let den2 = (den * den).inv();
                // (dD, da) per parameter: ER, then each MF=2 channel's width.
                let mut params: Vec<(usize, Complex, Vec<f64>)> = Vec::with_capacity(nch);
                let mut da_er = vec![0.0; n];
                if res.energy != 0.0 {
                    let kr = WAVE_NUMBER * g.mass_ratio * res.energy.abs().sqrt();
                    for (c, ch) in g.channels.iter().enumerate() {
                        let rho = kr * ch.penetrability_radius;
                        let dp_de = penetrability_slope(ch.l, rho) * rho / (2.0 * res.energy.abs())
                            * res.energy.signum();
                        if res.penetrability[c] > 0.0 {
                            da_er[c] = -0.5 * a[c] * dp_de / res.penetrability[c];
                        }
                    }
                }
                params.push((0, den2.scale(-0.5), da_er));
                params.push((1 + g.photon, Complex::new(0.0, 0.25) * den2, vec![0.0; n]));
                for (c, ch) in g.channels.iter().enumerate() {
                    let mut da = vec![0.0; n];
                    if res.widths[c] != 0.0 && res.penetrability[c] > 0.0 {
                        da[c] = (st.p[c] / res.penetrability[c]).sqrt()
                            / (2.0 * res.widths[c].abs().sqrt());
                    }
                    params.push((1 + ch.index, Complex::ZERO, da));
                }
                let mut d = vec![[0.0; 3]; nch];
                for (slot, dd, da) in params {
                    // dR~ = dD a a^T + D (da a^T + a da^T)
                    let mut d_r = vec![Complex::ZERO; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            d_r[i * n + j] =
                                dd.scale(a[i] * a[j]) + dr.scale(da[i] * a[j] + a[i] * da[j]);
                        }
                    }
                    let left = matmul(&st.inv, &d_r, n);
                    let dx = matmul(&left, &kx_i, n);
                    // dM^-1 = M^-1 dR~ K M^-1
                    let mut lk = left.clone();
                    for i in 0..n {
                        for j in 0..n {
                            lk[i * n + j] = left[i * n + j] * st.k[j];
                        }
                    }
                    let d_inv = matmul(&lk, &st.inv, n);
                    let mut de = 0.0;
                    for c in 0..n {
                        for cp in 0..n {
                            let delta = if c == cp { Complex::ONE } else { Complex::ZERO };
                            let du = st.omega[c]
                                * Complex::new(0.0, 2.0)
                                * dx[c * n + cp]
                                * st.omega[cp];
                            de += -2.0 * (conj(delta - st.u(c, cp)) * du).re;
                        }
                    }
                    let mut dc = 0.0;
                    for c in 0..n {
                        for i in 0..n {
                            for j in 0..n {
                                let im_r = st.r[i * n + j].im;
                                let im_dr = d_r[i * n + j].im;
                                dc += 2.0 * im_r * (d_inv[c * n + i] * conj(st.inv[c * n + j])).re
                                    + im_dr * (st.inv[c * n + i] * conj(st.inv[c * n + j])).re;
                            }
                        }
                    }
                    d[slot] = [factor * de, factor * 4.0 * dc, 0.0];
                }
                out.push((gi, res.index, d));
            }
        }
        out
    }

    /// The derivative at `energy` with respect to the radius of channel
    /// `channel` of spin group `group` (its APE and APT together), by
    /// central difference.
    pub fn radius_derivative(&self, energy: f64, group: usize, channel: usize) -> Result<Gradient> {
        let shifted = |h: f64| -> Result<CrossSections> {
            let mut rml = self.source.clone();
            if let Some(sg) = rml.spin_groups.get_mut(group) {
                if let Some(x) = sg.channels.ape.get_mut(channel) {
                    *x += h;
                }
                if let Some(x) = sg.channels.apt.get_mut(channel) {
                    *x += h;
                }
            }
            let groups = rml_groups(&rml)?;
            Ok(RMatrixRange {
                el: self.el,
                eh: self.eh,
                groups,
                source: rml,
            }
            .cross_sections(energy))
        };
        const H: f64 = 1e-5;
        let (up, down) = (shifted(H)?, shifted(-H)?);
        Ok([
            (up.elastic - down.elastic) / (2.0 * H),
            (up.capture - down.capture) / (2.0 * H),
            0.0,
        ])
    }
}

/// The spin groups of an R-matrix limited range, with their explicit
/// channels and resonances.
fn rml_groups(rml: &crate::mf::mf2::RMatrixLimited) -> Result<Vec<RmlGroup>> {
    if rml.krm != 3 {
        return Err(Error::Unsupported {
            what: "R-matrix limited reconstruction other than Reich-Moore (KRM=3)",
        });
    }
    if rml.ifg != 0 {
        return Err(Error::Unsupported {
            what: "R-matrix limited reduced width amplitudes (IFG=1)",
        });
    }
    let pp = &rml.particle_pairs;
    let mut out = Vec::with_capacity(rml.spin_groups.len());
    for sg in &rml.spin_groups {
        if sg.kbk != 0 || sg.kps != 0 {
            return Err(Error::Unsupported {
                what: "an R-matrix limited background R-matrix or tabulated phase shift",
            });
        }
        let mut photon = None;
        let mut channels = Vec::new();
        let mut mass_ratio = 0.0;
        let mut spins = (0.5, 0.0);
        for c in 0..sg.nch as usize {
            let p = sg.channels.ppi[c] as usize - 1;
            let mt = pp.mt[p] as i64;
            if mt == 102 {
                photon = Some(c);
            } else if mt == 2 {
                mass_ratio = pp.mb[p] / (pp.ma[p] + pp.mb[p]);
                spins = (pp.ia[p].abs(), pp.ib[p].abs());
                let ape = sg.channels.ape[c];
                let apt = sg.channels.apt[c];
                channels.push(RmlChannel {
                    index: c,
                    l: sg.channels.l[c] as i64,
                    penetrability_radius: if apt != 0.0 { apt } else { ape },
                    phase_radius: if ape != 0.0 { ape } else { apt },
                    shift: pp.shf[p] == 1.0,
                    boundary: sg.channels.bnd[c],
                });
            } else {
                return Err(Error::Unsupported {
                    what: "an R-matrix limited particle pair other than the photon and the neutron",
                });
            }
        }
        let Some(photon) = photon else {
            return Err(Error::Unsupported {
                what: "an R-matrix limited spin group without a photon channel",
            });
        };
        if channels.iter().any(|c| c.l > 4) {
            return Err(Error::Unsupported {
                what: "an R-matrix limited channel with l > 4",
            });
        }
        let g = (2.0 * sg.aj.abs() + 1.0) / ((2.0 * spins.0 + 1.0) * (2.0 * spins.1 + 1.0));
        let resonances = (0..sg.er.len())
            .map(|index| {
                let energy = sg.er[index];
                let kr = WAVE_NUMBER * mass_ratio * energy.abs().sqrt();
                RmlResonance {
                    index,
                    energy,
                    capture: sg.gam[photon][index],
                    widths: channels.iter().map(|c| sg.gam[c.index][index]).collect(),
                    penetrability: channels
                        .iter()
                        .map(|c| penetration_shift(c.l, kr * c.penetrability_radius).0)
                        .collect(),
                }
            })
            .collect();
        out.push(RmlGroup {
            g,
            mass_ratio,
            photon,
            channels,
            resonances,
        });
    }
    Ok(out)
}

impl RangeReconstruction for RMatrixRange {
    fn bounds(&self) -> (f64, f64) {
        (self.el, self.eh)
    }

    fn cross_sections(&self, energy: f64) -> CrossSections {
        RMatrixRange::cross_sections(self, energy)
    }

    fn resonances(&self) -> Vec<(f64, f64)> {
        self.groups
            .iter()
            .flat_map(|g| g.resonances.iter())
            .map(|r| {
                (
                    r.energy,
                    r.capture.abs() + r.widths.iter().map(|w| w.abs()).sum::<f64>(),
                )
            })
            .collect()
    }

    fn parameter_gradients(
        &self,
        energy: f64,
        cov: &crate::resonance_covariance::RangeCovariance,
    ) -> Result<Vec<Gradient>> {
        use crate::resonance_covariance::{Location, Quantity};
        let by_resonance: std::collections::HashMap<(usize, usize), Vec<Gradient>> = self
            .derivatives(energy)
            .into_iter()
            .map(|(g, i, d)| ((g, i), d))
            .collect();
        let mut out = Vec::with_capacity(cov.parameters.len());
        for p in &cov.parameters {
            let g = match (p.location, p.quantity) {
                (Location::SpinGroup { group, index }, q) => {
                    let slot = match q {
                        Quantity::Energy => 0,
                        Quantity::ChannelWidth(c) => 1 + c,
                        _ => usize::MAX,
                    };
                    by_resonance
                        .get(&(group, index))
                        .and_then(|d| d.get(slot))
                        .copied()
                        .unwrap_or([0.0; 3])
                }
                (Location::Channel { group, channel }, Quantity::ScatteringRadius) => {
                    self.radius_derivative(energy, group, channel)?
                }
                _ => [0.0; 3],
            };
            out.push(g);
        }
        Ok(out)
    }
}

/// NJOY 2016's fluctuation-integral quadrature (`reconr`'s `gnrl`): ten
/// abscissae `QP` and weights `QW` for a chi-squared width distribution of one
/// to four degrees of freedom, `[dof - 1][point]`.
const URR_QW: [[f64; 10]; 4] = [
    [
        1.1120413e-1,
        2.3546798e-1,
        2.8440987e-1,
        2.2419127e-1,
        0.10967668,
        0.030493789,
        0.0042930874,
        2.5827047e-4,
        4.9031965e-6,
        1.4079206e-8,
    ],
    [
        0.033773418,
        0.079932171,
        0.12835937,
        0.17652616,
        0.21347043,
        0.21154965,
        0.13365186,
        0.022630659,
        1.6313638e-5,
        2.745383e-31,
    ],
    [
        3.3376214e-4,
        0.018506108,
        0.12309946,
        0.29918923,
        0.33431475,
        0.17766657,
        0.042695894,
        4.0760575e-3,
        1.1766115e-4,
        5.0989546e-7,
    ],
    [
        1.7623788e-3,
        0.021517749,
        0.080979849,
        0.18797998,
        0.30156335,
        0.29616091,
        0.10775649,
        2.5171914e-3,
        8.9630388e-10,
        0.0,
    ],
];
const URR_QP: [[f64; 10]; 4] = [
    [
        3.0013465e-3,
        7.8592886e-2,
        4.3282415e-1,
        1.3345267,
        3.0481846,
        5.8263198,
        9.9452656,
        1.5782128e1,
        23.996824,
        36.216208,
    ],
    [
        1.3219203e-2,
        7.2349624e-2,
        0.19089473,
        0.39528842,
        0.74083443,
        1.3498293,
        2.5297983,
        5.2384894,
        13.821772,
        75.647525,
    ],
    [
        1.0004488e-3,
        0.026197629,
        0.14427472,
        0.44484223,
        1.0160615,
        1.9421066,
        3.3150885,
        5.2607092,
        7.9989414,
        12.072069,
    ],
    [
        0.013219203,
        0.072349624,
        0.19089473,
        0.39528842,
        0.74083443,
        1.3498293,
        2.5297983,
        5.2384894,
        13.821772,
        75.647525,
    ],
];

/// One fluctuation integral (NJOY's `gnrl`): the average over the width
/// distributions of `G_n^2 / G` (`id` 1, elastic), `G_n / G` (2, capture) or
/// `G_n G_f / G` (3, fission), each in units of the average widths, for
/// average neutron, fission, capture and competitive widths `alpha`, `beta`,
/// `gamma` and `df` with `mu`, `nu` and `lambda` degrees of freedom.
#[allow(clippy::too_many_arguments)]
fn fluctuation(
    alpha: f64,
    beta: f64,
    gamma: f64,
    mu: usize,
    nu: usize,
    lambda: usize,
    df: f64,
    id: u8,
) -> f64 {
    if alpha <= 0.0 || gamma <= 0.0 || beta < 0.0 || (beta > 0.0 && df < 0.0) {
        return 0.0;
    }
    let dof = |m: usize| m.clamp(1, 4) - 1;
    let (qwm, qpm) = (URR_QW[dof(mu)], URR_QP[dof(mu)]);
    let (qwn, qpn) = (URR_QW[dof(nu)], URR_QP[dof(nu)]);
    let (qwl, qpl) = (URR_QW[dof(lambda)], URR_QP[dof(lambda)]);
    let num = |x: f64, y: f64| -> f64 {
        match id {
            1 => x * x,
            2 => x,
            _ => x * y,
        }
    };
    let mut s = 0.0;
    match (beta > 0.0, df > 0.0) {
        (false, false) => {
            if id == 3 {
                return 0.0;
            }
            for j in 0..10 {
                s += qwm[j] * num(qpm[j], 0.0) / (alpha * qpm[j] + gamma);
            }
        }
        (false, true) => {
            if id == 3 {
                return 0.0;
            }
            for j in 0..10 {
                for k in 0..10 {
                    s +=
                        qwm[j] * qwl[k] * num(qpm[j], 0.0) / (alpha * qpm[j] + gamma + df * qpl[k]);
                }
            }
        }
        (true, false) => {
            for j in 0..10 {
                for k in 0..10 {
                    s += qwm[j] * qwn[k] * num(qpm[j], qpn[k])
                        / (alpha * qpm[j] + beta * qpn[k] + gamma);
                }
            }
        }
        (true, true) => {
            for j in 0..10 {
                for k in 0..10 {
                    for l in 0..10 {
                        s += qwm[j] * qwn[k] * qwl[l] * num(qpm[j], qpn[k])
                            / (alpha * qpm[j] + beta * qpn[k] + gamma + df * qpl[l]);
                    }
                }
            }
        }
    }
    s
}

/// The average parameters of one `(l, J)`, tabulated in energy: a table of
/// one row is energy independent.
#[derive(Debug, Clone, PartialEq)]
struct UrrSpin {
    aj: f64,
    /// Degrees of freedom of the competitive, neutron and fission widths.
    mux: usize,
    mun: usize,
    muf: usize,
    energies: Vec<f64>,
    /// D, GX, GN0, GG, GF per energy.
    rows: Vec<[f64; 5]>,
}

impl UrrSpin {
    /// The parameters at `energy`, linearly interpolated and held constant
    /// beyond the table: how NJOY takes them at a node of the cross-section
    /// grid, and within a panel too wide to interpolate cross sections across
    /// (see [`UnresolvedAverages::cross_sections`]).
    fn at(&self, energy: f64) -> [f64; 5] {
        let n = self.energies.len();
        if n == 1 || energy <= self.energies[0] {
            return self.rows[0];
        }
        if energy >= self.energies[n - 1] {
            return self.rows[n - 1];
        }
        let i = self.energies.partition_point(|&e| e <= energy) - 1;
        let f = (energy - self.energies[i]) / (self.energies[i + 1] - self.energies[i]);
        std::array::from_fn(|q| self.rows[i][q] + f * (self.rows[i + 1][q] - self.rows[i][q]))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct UrrOrbital {
    l: i64,
    awri: f64,
    spins: Vec<UrrSpin>,
}

/// An unresolved range's infinitely dilute average cross sections at 0 K, as
/// NJOY's RECONR computes them (`csunr1` and `csunr2`, ENDF-102 D.2): the
/// single-level average over Porter-Thomas-like width distributions, with
/// NJOY's ten-point fluctuation quadrature, for every `(l, J)` of the range,
/// plus potential scattering. Only `l <= 2` is supported, as in NJOY.
///
/// Where the parameters depend on energy the cross sections are computed at
/// the evaluator's energies and interpolated between them, as ENDF-102 says
/// INT is for and as NJOY does, rather than computed from interpolated
/// parameters: the two differ by up to 3% in ENDF/B-VIII.1 Rh103 capture
/// between its parameter energies (see [`Self::cross_sections`]).
///
/// These averages are what the unresolved range's resonance-parameter
/// covariance (MF=32, LRU=2) moves; where the evaluation sets LSSF=1 the
/// average cross sections themselves are in MF=3 and the parameters only
/// shape self-shielding, so a caller relativizes by MF=3.
#[derive(Debug, Clone, PartialEq)]
pub struct UnresolvedAverages {
    pub el: f64,
    pub eh: f64,
    /// 1 where the averages are in MF=3 already.
    pub lssf: i64,
    spin: f64,
    channel_radius: f64,
    scattering_radius: f64,
    orbitals: Vec<UrrOrbital>,
    /// The energies the cross sections are computed at and interpolated
    /// between; empty where the parameters do not depend on energy.
    nodes: Vec<f64>,
    /// The ENDF interpolation law between `nodes`.
    law: i64,
}

/// A panel of the unresolved cross-section grid this many times wider than
/// its lower energy or more is too coarse to interpolate cross sections
/// across, and they are computed from interpolated parameters instead
/// (NJOY's `wide`).
const WIDE: f64 = 1.26;

/// `y` at `x` between `(x1, y1)` and `(x2, y2)` under ENDF interpolation law
/// `law`, as NJOY's `terp1` takes it. A logarithmic law across a zero or
/// negative value falls back to linear, where `terp1` would give a NaN.
fn interpolate(x1: f64, y1: f64, x2: f64, y2: f64, x: f64, law: i64) -> f64 {
    if x2 == x1 || law == 1 || y2 == y1 || x == x1 {
        return y1;
    }
    let linear = || y1 + (x - x1) * (y2 - y1) / (x2 - x1);
    match law {
        3 => y1 + (x / x1).ln() * (y2 - y1) / (x2 / x1).ln(),
        4 if y1 > 0.0 && y2 > 0.0 => y1 * ((x - x1) * (y2 / y1).ln() / (x2 - x1)).exp(),
        5 if y1 > 0.0 && y2 > 0.0 => y1 * ((x / x1).ln() * (y2 / y1).ln() / (x2 / x1).ln()).exp(),
        _ => linear(),
    }
}

/// What a multiplier on an unresolved parameter scales: column of
/// [`UrrSpin::rows`].
fn urr_column(q: crate::resonance_covariance::Quantity) -> Option<usize> {
    use crate::resonance_covariance::Quantity;
    match q {
        Quantity::LevelSpacing => Some(0),
        Quantity::CompetitiveWidth => Some(1),
        Quantity::ReducedNeutronWidth => Some(2),
        Quantity::CaptureWidth => Some(3),
        Quantity::FissionWidth => Some(4),
        _ => None,
    }
}

impl UnresolvedAverages {
    /// Prepare `range` (LRU=2) for reconstruction.
    pub fn new(range: &ResonanceRange) -> Result<Self> {
        use crate::mf::mf2::UnresolvedParameters;
        let ResonanceParameters::Unresolved(u) = &range.parameters else {
            return Err(Error::Unsupported {
                what: "average cross sections of a range other than an unresolved one",
            });
        };
        if range.nro != 0 {
            return Err(Error::Unsupported {
                what: "an energy-dependent scattering radius (NRO/=0)",
            });
        }
        let awri = u.ranges.first().map_or(1.0, |r| r.awri);
        let channel_radius = match range.naps {
            0 => channel_radius_formula(awri),
            _ => u.ap,
        };
        let mut orbitals = Vec::with_capacity(u.ranges.len());
        // NJOY interpolates on the first (l, J)'s energies, by the law of the
        // last (case C), or linearly on ES (case B).
        let mut grid: Vec<f64> = Vec::new();
        let mut law = 2;
        for r in &u.ranges {
            if r.l > 2 {
                return Err(Error::Unsupported {
                    what: "an unresolved section with l > 2",
                });
            }
            let mut spins = Vec::new();
            if r.parameters.is_empty() {
                // Case A: energy independent, no fission or competition.
                for j in 0..r.aj.len() {
                    spins.push(UrrSpin {
                        aj: r.aj[j],
                        mux: 1,
                        mun: r.amun[j].round() as usize,
                        muf: 1,
                        energies: vec![range.el],
                        rows: vec![[r.d[j], 0.0, r.gno[j], r.gg[j], 0.0]],
                    });
                }
            }
            for p in &r.parameters {
                match p {
                    UnresolvedParameters::CaseB {
                        muf,
                        d,
                        aj,
                        amun,
                        gn0,
                        gg,
                        gf,
                    } => {
                        let energies: Vec<f64> = u.es.clone();
                        if grid.is_empty() {
                            grid = energies.clone();
                        }
                        let rows = energies
                            .iter()
                            .enumerate()
                            .map(|(i, _)| [*d, 0.0, *gn0, *gg, gf.get(i).copied().unwrap_or(0.0)])
                            .collect();
                        spins.push(UrrSpin {
                            aj: *aj,
                            mux: 1,
                            mun: amun.round() as usize,
                            muf: *muf as usize,
                            energies,
                            rows,
                        });
                    }
                    UnresolvedParameters::CaseC {
                        aj,
                        interpolation,
                        amux,
                        amun,
                        amuf,
                        e,
                        d,
                        gx,
                        gn0,
                        gg,
                        gf,
                        ..
                    } => {
                        if grid.is_empty() {
                            grid = e.clone();
                        }
                        law = *interpolation;
                        spins.push(UrrSpin {
                            aj: *aj,
                            mux: amux.round() as usize,
                            mun: amun.round() as usize,
                            muf: amuf.round() as usize,
                            energies: e.clone(),
                            rows: (0..e.len())
                                .map(|i| [d[i], gx[i], gn0[i], gg[i], gf[i]])
                                .collect(),
                        });
                    }
                }
            }
            orbitals.push(UrrOrbital {
                l: r.l,
                awri: r.awri,
                spins,
            });
        }
        // NJOY's first panel starts at EL whatever the table's first energy.
        let mut nodes = Vec::new();
        if grid.len() >= 2 {
            nodes.push(range.el);
            nodes.extend(grid.into_iter().filter(|&e| e > range.el));
        }
        Ok(UnresolvedAverages {
            el: range.el,
            eh: range.eh,
            lssf: u.lssf,
            spin: u.spi,
            channel_radius,
            scattering_radius: u.ap,
            orbitals,
            nodes,
            law,
        })
    }

    /// The energies the evaluation tabulates the parameters on, within the
    /// range: where its cross sections are given.
    pub fn parameter_energies(&self) -> Vec<f64> {
        let mut e: Vec<f64> = self
            .orbitals
            .iter()
            .flat_map(|o| o.spins.iter())
            .flat_map(|s| s.energies.iter().copied())
            .filter(|&x| x >= self.el && x <= self.eh)
            .collect();
        e.sort_by(f64::total_cmp);
        e.dedup();
        e
    }

    /// The average cross sections at `energy` (eV): computed at the nodes of
    /// the evaluator's energy grid and interpolated between them by its INT,
    /// except in a panel [`WIDE`] or wider, and beyond the grid, where they
    /// are computed from interpolated parameters.
    pub fn cross_sections(&self, energy: f64) -> CrossSections {
        let n = &self.nodes;
        if n.len() < 2 || energy <= n[0] || energy >= n[n.len() - 1] {
            return self.with_parameters_at(energy);
        }
        let i = n.partition_point(|&e| e <= energy) - 1;
        let (e1, e2) = (n[i], n[i + 1]);
        if energy == e1 || e2 >= WIDE * e1 {
            return self.with_parameters_at(energy);
        }
        let (a, b) = (self.with_parameters_at(e1), self.with_parameters_at(e2));
        let at = |y1: f64, y2: f64| interpolate(e1, y1, e2, y2, energy, self.law);
        CrossSections {
            elastic: at(a.elastic, b.elastic),
            capture: at(a.capture, b.capture),
            fission: at(a.fission, b.fission),
        }
    }

    /// The average cross sections at `energy` from the parameters there.
    fn with_parameters_at(&self, energy: f64) -> CrossSections {
        let mut out = CrossSections::default();
        let pi = std::f64::consts::PI;
        for o in &self.orbitals {
            let ratio = o.awri / (o.awri + 1.0);
            let k = WAVE_NUMBER * ratio * energy.sqrt();
            let constant = 2.0 * pi * pi / (WAVE_NUMBER * ratio).powi(2);
            let rho = k * self.channel_radius;
            let rhoc = k * self.scattering_radius;
            let r2 = rho * rho;
            let (v, phase) = match o.l {
                0 => (1.0, rhoc),
                1 => (r2 / (1.0 + r2), rhoc - rhoc.atan()),
                _ => (
                    r2 * r2 / (9.0 + 3.0 * r2 + r2 * r2),
                    rhoc - (3.0 * rhoc / (3.0 - rhoc * rhoc)).atan(),
                ),
            };
            for sp in &o.spins {
                let [d, gx, gn0, gg, gf] = sp.at(energy);
                let gx = if gx < 1e-8 { 0.0 } else { gx };
                let gf = if gf < 1e-8 { 0.0 } else { gf };
                let gj = (2.0 * sp.aj + 1.0) / (4.0 * self.spin + 2.0);
                let gn = gn0 * v * sp.mun as f64 * energy.sqrt();
                let den = energy * d;
                if den <= 0.0 {
                    continue;
                }
                let temp = constant * gj * gn / den;
                let gs = fluctuation(gn, gf, gg, sp.mun, sp.muf, sp.mux, gx, 1) * temp * gn;
                let gc = fluctuation(gn, gf, gg, sp.mun, sp.muf, sp.mux, gx, 2) * temp * gg;
                let gff = fluctuation(gn, gf, gg, sp.mun, sp.muf, sp.mux, gx, 3) * temp * gf;
                let interference = constant * gj * 2.0 * gn * phase.sin().powi(2) / den;
                out.elastic += gs - interference;
                out.capture += gc;
                out.fission += gff;
            }
            out.elastic += 4.0 * pi * (2 * o.l + 1) as f64 * (phase.sin() / k).powi(2);
        }
        out
    }

    /// `self` with column `column` of `(orbital, spin)`'s table scaled by
    /// `factor` at every energy.
    fn scaled(&self, orbital: usize, spin: usize, column: usize, factor: f64) -> Self {
        let mut out = self.clone();
        if let Some(s) = out
            .orbitals
            .get_mut(orbital)
            .and_then(|o| o.spins.get_mut(spin))
        {
            for row in &mut s.rows {
                row[column] *= factor;
            }
        }
        out
    }
}

impl RangeReconstruction for UnresolvedAverages {
    fn bounds(&self) -> (f64, f64) {
        (self.el, self.eh)
    }

    fn cross_sections(&self, energy: f64) -> CrossSections {
        UnresolvedAverages::cross_sections(self, energy)
    }

    fn resonances(&self) -> Vec<(f64, f64)> {
        Vec::new()
    }

    /// The derivative with respect to a unit-mean multiplier on one `(l, J)`
    /// table, by central difference: the parameters are few.
    fn parameter_gradients(
        &self,
        energy: f64,
        cov: &crate::resonance_covariance::RangeCovariance,
    ) -> Result<Vec<Gradient>> {
        use crate::resonance_covariance::Location;
        const H: f64 = 1e-4;
        Ok(cov
            .parameters
            .iter()
            .map(|p| match (p.location, urr_column(p.quantity)) {
                (Location::Unresolved { orbital, spin }, Some(column)) => {
                    let up = self
                        .scaled(orbital, spin, column, 1.0 + H)
                        .cross_sections(energy);
                    let down = self
                        .scaled(orbital, spin, column, 1.0 - H)
                        .cross_sections(energy);
                    [
                        (up.elastic - down.elastic) / (2.0 * H),
                        (up.capture - down.capture) / (2.0 * H),
                        (up.fission - down.fission) / (2.0 * H),
                    ]
                }
                _ => [0.0; 3],
            })
            .collect())
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
            0 => channel_radius_formula(s.awri),
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
                    let tolerance = 2e-6 * njoy.abs() + 1e-6;
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
    /// the radius. So a unit step on the s-wave radius alone moves elastic by
    /// `8 pi a_eff`, and no step moves nothing.
    #[test]
    fn radius_derivative_is_that_of_potential_scattering() {
        const PB208: &[u8] = include_bytes!("../fixtures/n-082_Pb_208_mf2.endf.xz");
        let rm = ReichMooreRange::new(&mf2_range(PB208)).unwrap();
        let d = rm.radius_derivative(0.0253, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let elastic = rm.cross_sections(0.0253).elastic;
        let want = 8.0 * std::f64::consts::PI * (elastic / (4.0 * std::f64::consts::PI)).sqrt();
        assert!((d[0] / want - 1.0).abs() < 1e-3, "{} against {want}", d[0]);
        let none = rm.radius_derivative(0.0253, &[0.0; 4]).unwrap();
        assert_eq!(none, [0.0, 0.0, 0.0]);
    }
}

#[cfg(test)]
mod r_matrix_tests {
    use super::*;
    use crate::material::Material;
    use crate::resonance_covariance::{group_covariance, range_covariances};

    const W186: &[u8] = include_bytes!("../fixtures/n-074_W_186_mf2_mf32.endf.xz");
    const CU65: &[u8] = include_bytes!("../fixtures/n-029_Cu_065_mf2_mf32.endf.xz");
    const V51: &[u8] = include_bytes!("../fixtures/n-023_V_051_mf2.endf.xz");

    fn material(fixture: &[u8]) -> Material {
        Material::from_str(&crate::testdata::text(fixture)).expect("fixture parses")
    }

    fn range(fixture: &[u8]) -> RMatrixRange {
        RMatrixRange::new(&material(fixture).mf2().unwrap().isotopes[0].ranges[0])
            .expect("R-matrix limited")
    }

    // NJOY 2016 RECONR at 0 K (err 1e-5) on ENDF/B-VIII.1, less the MF=3
    // background: `(E, elastic, capture)` at its own grid points.
    const W186_NJOY: &[(f64, f64, f64)] = &[
        (1e-5, 8.271967e-2, 1.900542e3),
        (2.076627e2, 1.472565e-3, 9.121697e-1),
        (5.078018e2, 6.102830e-2, 1.460927e0),
        (7.154179e2, 5.908505e1, 3.660478e0),
        (1.07186e3, 1.747791e3, 1.767738e2),
        (1.464078e3, 3.848036e0, 3.471237e-2),
        (1.936519e3, 9.372047e1, 3.783920e0),
        (2.507229e3, 1.998538e-2, 1.436476e-1),
        (2.780709e3, 3.100004e2, 3.337881e1),
        (3.160018e3, 7.752326e2, 2.100326e1),
        (3.544189e3, 2.719665e2, 2.040216e1),
        (3.870351e3, 2.203334e2, 4.698132e0),
        (4.223023e3, 5.966325e2, 1.318790e1),
        (4.5869775e3, 3.481055e1, 1.349380e2),
        (5.133363e3, 2.493483e0, 6.784925e-1),
        (5.416642e3, 1.376228e1, 2.702355e-2),
        (5.891996e3, 1.315977e1, 1.865056e-1),
        (6.298326e3, 6.791252e1, 7.834726e-1),
        (6.746922e3, 1.347563e-3, 1.465816e-1),
        (7.176741e3, 9.082069e1, 6.729063e0),
        (7.631775e3, 9.036670e-3, 2.476986e-1),
        (8.04053e3, 1.415191e1, 3.834495e0),
        (8.669223e3, 2.491522e0, 2.296871e-1),
        (9.257214e3, 6.157844e-1, 3.512444e-2),
        (9.995325e3, 1.300591e1, 2.750419e-3),
    ];
    const CU65_NJOY: &[(f64, f64, f64)] = &[
        (1e-5, 1.380143e1, 1.079051e2),
        (2.324889e3, 3.541214e0, 9.192036e-1),
        (4.619948e3, 9.281443e0, 2.242511e-2),
        (7.668754e3, 2.781311e1, 1.758079e-1),
        (1.19238025e4, 3.464494e0, 4.748956e-1),
        (1.425242e4, 3.879635e1, 1.693935e-1),
        (1.897287e4, 1.209247e1, 5.187586e-3),
        (2.299154e4, 4.539647e0, 5.012440e-1),
        (2.592832e4, 4.202699e1, 7.556933e-1),
        (2.978651e4, 5.414469e0, 1.188704e-3),
        (3.493299e4, 4.824485e1, 8.898831e-2),
        (3.932424e4, 5.668947e0, 8.706035e-1),
        (4.437281e4, 8.459046e0, 5.865731e-3),
        (4.75433e4, 1.846617e1, 1.104660e-1),
        (5.29753e4, 4.132619e0, 9.821368e-3),
        (5.546307e4, 5.140781e0, 3.272325e-2),
        (6.018416e4, 3.570620e0, 1.175383e-3),
        (6.367904e4, 2.550025e0, 6.680812e-3),
        (6.949548e4, 2.699681e1, 1.262698e-1),
        (7.299451e4, 1.674086e1, 1.815522e-2),
        (7.819625e4, 4.916135e0, 9.871435e-3),
        (8.410924e4, 2.027356e1, 1.765158e-1),
        (8.853629e4, 5.294102e0, 1.359732e-3),
        (9.44143e4, 1.678731e1, 6.487461e-2),
        (9.99912e4, 7.665335e0, 1.498508e-3),
    ];
    const V51_NJOY: &[(f64, f64, f64)] = &[
        (1e-5, 4.895632e0, 2.471022e2),
        (2.424472e3, 9.614259e0, 3.880550e-2),
        (8.578279e3, 1.284070e1, 5.783970e-2),
        (1.574005e4, 2.905170e1, 1.512828e-1),
        (2.118094e4, 4.147802e0, 1.890418e-2),
        (2.960104e4, 3.129380e1, 1.730414e-1),
        (3.754263e4, 3.977332e0, 3.539704e-2),
        (4.245609e4, 2.622890e1, 7.570767e-1),
        (4.839581e4, 4.461941e0, 9.664563e-3),
        (5.295082e4, 1.463009e1, 2.691577e-2),
        (6.724623e4, 1.754831e1, 8.898576e-1),
        (7.269955e4, 9.666719e0, 6.065339e-1),
        (7.905059e4, 2.129457e1, 1.614942e-1),
        (8.715637e4, 1.479975e1, 1.926851e0),
        (9.51142e4, 4.091915e0, 2.335733e-2),
        (1.03711975e5, 1.868431e0, 1.724845e-1),
        (1.1375265e5, 7.448459e0, 1.009232e-1),
        (1.185446e5, 1.182729e1, 6.759970e-3),
        (1.297899e5, 7.108859e0, 2.845873e-3),
        (1.353416e5, 1.089117e1, 7.138237e-2),
        (1.515302e5, 8.113684e-1, 1.261546e-2),
        (1.627884e5, 1.209362e1, 3.558998e-3),
        (1.740548e5, 7.577261e0, 4.374221e-3),
        (1.82736e5, 1.150781e1, 8.270994e-3),
        (1.999991e5, 5.107940e0, 1.056029e-2),
    ];

    /// The reconstruction agrees with NJOY to the seven digits it writes:
    /// W186, Cu65 (shift factors, B = -1 on its p-wave channels) and V51 (whose
    /// true and effective radii differ).
    #[test]
    fn r_matrix_matches_njoy() {
        for (fixture, reference) in [(W186, W186_NJOY), (CU65, CU65_NJOY), (V51, V51_NJOY)] {
            let rm = range(fixture);
            for &(e, elastic, capture) in reference {
                let x = rm.cross_sections(e);
                for (ours, njoy, what) in [
                    (x.elastic, elastic, "elastic"),
                    (x.capture, capture, "capture"),
                ] {
                    assert!(
                        (ours - njoy).abs() <= 2e-6 * njoy.abs() + 1e-6,
                        "{what} at {e} eV: {ours} against NJOY's {njoy}"
                    );
                }
            }
        }
    }

    /// Every resonance's analytic derivatives (ER, the capture width and each
    /// neutron channel's width) match central differences of the
    /// reconstruction, with and without shift factors.
    #[test]
    fn r_matrix_derivatives_match_central_differences() {
        let mut checked = 0;
        for fixture in [W186, CU65] {
            let base = material(fixture).mf2().unwrap().isotopes[0].ranges[0].clone();
            let rm = RMatrixRange::new(&base).unwrap();
            let ResonanceParameters::RMatrixLimited(params) = &base.parameters else {
                unreachable!()
            };
            for (group, sg) in params.spin_groups.iter().enumerate() {
                let step = (sg.er.len() / 3).max(1);
                for index in (0..sg.er.len()).step_by(step) {
                    let er = sg.er[index];
                    if er <= base.el || er >= base.eh {
                        continue;
                    }
                    let width: f64 = sg
                        .gam
                        .iter()
                        .map(|row| row[index].abs())
                        .sum::<f64>()
                        .max(1e-3);
                    for e in [er, er + 0.7 * width, er * 1.3, 0.0253] {
                        if e <= base.el || e >= base.eh {
                            continue;
                        }
                        let analytic = rm
                            .derivatives(e)
                            .into_iter()
                            .find(|(g, i, _)| *g == group && *i == index)
                            .expect("every resonance has a gradient")
                            .2;
                        let x = rm.cross_sections(e);
                        for (slot, grad) in analytic.iter().enumerate().take(sg.nch as usize + 1) {
                            let value = if slot == 0 {
                                er
                            } else {
                                sg.gam[slot - 1][index]
                            };
                            if slot > 0 && value == 0.0 {
                                continue;
                            }
                            // A width step of 1e-3: smooth enough to leave no
                            // truncation, large enough that a thermal elastic
                            // near cancellation is not all rounding.
                            let h = if slot == 0 {
                                1e-4 * width
                            } else {
                                1e-3 * value.abs()
                            };
                            let moved = |delta: f64| {
                                let mut r = base.clone();
                                let ResonanceParameters::RMatrixLimited(p) = &mut r.parameters
                                else {
                                    unreachable!()
                                };
                                if slot == 0 {
                                    p.spin_groups[group].er[index] += delta;
                                } else {
                                    p.spin_groups[group].gam[slot - 1][index] += delta;
                                }
                                RMatrixRange::new(&r).unwrap().cross_sections(e)
                            };
                            let (up, down) = (moved(h), moved(-h));
                            let numeric = [
                                (up.elastic - down.elastic) / (2.0 * h),
                                (up.capture - down.capture) / (2.0 * h),
                            ];
                            for c in 0..2 {
                                let v = [x.elastic, x.capture][c];
                                let tolerance = 1e-4 * numeric[c].abs() + 1e-13 * v.abs() / h;
                                assert!(
                                    (grad[c] - numeric[c]).abs() <= tolerance,
                                    "slot {slot} of resonance {group}/{index} at {e} eV, reaction {c}: \
                                     analytic {} against numeric {}",
                                    grad[c],
                                    numeric[c]
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 200, "{checked}");
    }

    /// W186's group covariance against NJOY 2016 ERRORR (its SAMMY method)
    /// where ERRORR is self-consistent: elastic in the seven groups below
    /// 3 keV, and capture in the four whose ERRORR values are not erratic
    /// (ERRORR's thermal capture is 14 times its 0.1 to 1 eV group's, both 1/v,
    /// and its 1 to 3 keV capture is 7e-13; ours is 2.45e-4 in each 1/v group,
    /// the 1.5% W186 capture uncertainty ERRORR gives elsewhere).
    #[test]
    fn r_matrix_group_covariance_matches_errorr() {
        let m = material(W186);
        let cov = &range_covariances(m.mf2().unwrap(), m.mf32().unwrap()).unwrap()[0];
        let rm = RMatrixRange::new(&m.mf2().unwrap().isotopes[0].ranges[0]).unwrap();
        let edges = [1e-5, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 1e4];
        let g = group_covariance(cov, &rm, &edges).unwrap();
        let elastic = [
            7.939e-2, 5.306e-2, 4.156e-3, 2.080e-4, 1.916e-4, 7.433e-5, 3.950e-5,
        ];
        for (h, want) in elastic.iter().enumerate() {
            let got = g.get(0, h, 0, h);
            assert!(
                (got / want - 1.0).abs() < 0.015,
                "elastic group {h}: {got:e} against {want:e}"
            );
        }
        for (h, want) in [(1, 2.447e-4), (3, 2.424e-4), (4, 1.170e-4), (5, 7.032e-5)] {
            let got = g.get(1, h, 1, h);
            assert!(
                (got / want - 1.0).abs() < 0.005,
                "capture group {h}: {got:e} against {want:e}"
            );
        }
        for h in 0..3 {
            assert!(
                (g.get(1, h, 1, h) / 2.45e-4 - 1.0).abs() < 0.05,
                "1/v capture group {h}"
            );
        }
    }
}

#[cfg(test)]
mod unresolved_tests {
    use super::*;
    use crate::material::Material;
    use crate::mf::mf2::UnresolvedParameters;
    use crate::mf::mf32::Covariance;
    use crate::resonance_covariance::{group_covariance, range_covariances};

    const RH103: &[u8] = include_bytes!("../fixtures/n-045_Rh_103_mf2_mf32.endf.xz");

    fn material() -> Material {
        Material::from_str(&crate::testdata::text(RH103)).expect("fixture parses")
    }

    /// ENDF/B-VIII.1 Rh103's unresolved averages (LSSF=0, so NJOY 2016 RECONR
    /// puts them in MF=3), less the MF=3 background, at the evaluator's
    /// parameter energies, where RECONR evaluates them rather than
    /// interpolating: `(E, elastic, capture)`.
    const RH103_NJOY: [(f64, f64, f64); 8] = [
        (8.5e3, 6.597781, 1.405245),
        (1.05e4, 7.016208, 1.436239),
        (1.45e4, 7.094753, 1.333073),
        (1.95e4, 6.800235, 1.049889),
        (2.45e4, 6.842687, 0.9306573),
        (2.95e4, 7.624931, 0.9414835),
        (3.35e4, 7.541447, 0.8591245),
        (3.65e4, 6.858164, 0.7361913),
    ];

    fn unresolved() -> (Material, UnresolvedAverages) {
        let m = material();
        let range = m.mf2().unwrap().isotopes[0]
            .ranges
            .iter()
            .find(|r| r.lru == 2)
            .unwrap()
            .clone();
        let u = UnresolvedAverages::new(&range).unwrap();
        (m, u)
    }

    #[test]
    fn unresolved_averages_match_njoy() {
        let (_, u) = unresolved();
        for (e, elastic, capture) in RH103_NJOY {
            let x = u.cross_sections(e);
            assert!(
                (x.elastic / elastic - 1.0).abs() < 1e-6,
                "elastic at {e}: {} against {elastic}",
                x.elastic
            );
            assert!(
                (x.capture / capture - 1.0).abs() < 1e-6,
                "capture at {e}: {} against {capture}",
                x.capture
            );
        }
    }

    /// Between the parameter energies the cross sections are interpolated,
    /// by Rh103's INT, from their values at the panel's ends, not computed
    /// from interpolated parameters; the two differ by percents in capture.
    #[test]
    fn unresolved_cross_sections_are_interpolated_between_parameter_energies() {
        let (_, u) = unresolved();
        let n = &u.nodes;
        assert!(n.len() > 2 && n.windows(2).all(|w| w[1] < WIDE * w[0]));
        let mut largest = 0.0f64;
        for w in n.windows(2) {
            let mid = 0.5 * (w[0] + w[1]);
            let (a, b) = (u.cross_sections(w[0]), u.cross_sections(w[1]));
            let x = u.cross_sections(mid);
            let want = interpolate(w[0], a.capture, w[1], b.capture, mid, u.law);
            assert!((x.capture / want - 1.0).abs() < 1e-14, "{mid}");
            largest = largest.max((u.with_parameters_at(mid).capture / x.capture - 1.0).abs());
        }
        println!(
            "INT {} over {} nodes: largest capture difference {largest}",
            u.law,
            n.len()
        );
        assert!(largest > 0.01, "{largest}");
    }

    /// NJOY 2016 ERRORR takes an unresolved range's sensitivities from
    /// MF=32's own energy-independent average parameters, perturbed by 1%,
    /// and relativizes them by the MF=2 cross section; here they are taken
    /// from MF=2's tables, the parameters the evaluation's cross sections come
    /// from. Given ERRORR's parameters the group covariance reproduces its
    /// absolute variances: elastic to 0.3% and capture to 1.5%, its 1% finite
    /// differences against a derivative.
    #[test]
    fn unresolved_group_covariance_reproduces_errorr_on_its_own_parameters() {
        let (m, _) = unresolved();
        let covs = range_covariances(m.mf2().unwrap(), m.mf32().unwrap()).unwrap();
        let cov = covs
            .iter()
            .find(|c| c.lru == 2)
            .expect("an unresolved covariance");
        assert_eq!(cov.len(), 24);
        assert!(cov.unmatched.is_empty());
        let mut range = m.mf2().unwrap().isotopes[0].ranges[cov.mf2_range].clone();
        let mf32 = m.mf32().unwrap().isotopes[0]
            .ranges
            .iter()
            .find_map(|r| match &r.covariance {
                Covariance::Unresolved(u) => Some(u.clone()),
                _ => None,
            })
            .unwrap();
        if let ResonanceParameters::Unresolved(u) = &mut range.parameters {
            for (section, lv) in u.ranges.iter_mut().zip(&mf32.l_values) {
                for (p, par) in section.parameters.iter_mut().zip(&lv.parameters) {
                    if let UnresolvedParameters::CaseC {
                        e,
                        d,
                        gx,
                        gn0,
                        gg,
                        gf,
                        ..
                    } = p
                    {
                        let n = e.len();
                        *d = vec![par[0]; n];
                        *gn0 = vec![par[2]; n];
                        *gg = vec![par[3]; n];
                        *gf = vec![par[4]; n];
                        *gx = vec![par[5]; n];
                    }
                }
            }
        }
        let u = UnresolvedAverages::new(&range).unwrap();
        let edges = [8e3, 1e4, 1.5e4, 2e4, 3e4, 4.0146e4];
        let g = group_covariance(cov, &u, &edges).unwrap();
        // ERRORR's relative variances and group cross sections.
        let elastic = [
            (3.278e-4, 6.5102),
            (2.573e-4, 7.0800),
            (2.841e-4, 7.0722),
            (3.509e-4, 7.2484),
            (4.928e-4, 7.2306),
        ];
        let capture = [
            (1.961e-3, 1.3485),
            (1.506e-3, 1.3648),
            (1.645e-3, 1.1320),
            (1.582e-3, 9.8357e-1),
            (1.646e-3, 8.0582e-1),
        ];
        for h in 0..5 {
            for (a, (rel, xs), tolerance) in [(0, elastic[h], 3e-3), (1, capture[h], 1.5e-2)] {
                let ours = g.get(a, h, a, h) * g.cross_sections[a][h].powi(2);
                let njoy = rel * xs * xs;
                assert!(
                    (ours / njoy - 1.0).abs() < tolerance,
                    "reaction {a} group {h}: {ours:e} against {njoy:e}"
                );
            }
        }
    }
}
