//! The kernel's bound thermal scattering sampler: `#[cube]` twins of
//! `yamc_physics::gpu::flat::thermal_sab` reading the flat layout of
//! [`ThermalInputs`], with the draw schedule of the CPU's `sample_bound`. See
//! [`thermal_inputs`] for the layout and the schedule, and
//! [`sample_bound_cpu`] for the CPU twin.
//!
//! `exp`, `ln` and `expm1` go through the f64 polyfills, as everywhere in
//! the kernels.
//!
//! [`ThermalInputs`]: crate::neutron::thermal_inputs::ThermalInputs
//! [`thermal_inputs`]: crate::neutron::thermal_inputs
//! [`sample_bound_cpu`]: crate::neutron::thermal_inputs::sample_bound_cpu

use super::thermal_inputs::{
    SAB_ELASTIC_COHERENT, SAB_ELASTIC_INCOHERENT, SAB_ELASTIC_MIXED, SAB_NONE,
};
use cubecl::prelude::*;

/// The result of [`sample_bound()`]: `bound == 1` when the table scattered the
/// neutron, with its outgoing energy and lab cosine; `bound == 0` when the
/// slab is free, the energy is at or above energy_max, or the free share of
/// a partly bound nuclide was drawn (the caller then scatters it as a free
/// atom). `state` is the PCG state after the draws made.
#[derive(CubeType)]
pub struct BoundSample {
    pub bound: u32,
    pub e_out: f64,
    pub mu: f64,
    pub state: u64,
}

/// How many of the `n` values at `f[off..]` are `<= x` (they are ascending):
/// `partition_point(|&v| v <= x)`.
#[cube]
fn count_le(f: &[f64], off: u32, n: u32, x: f64) -> u32 {
    let mut lo = 0u32;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2u32;
        if f[(off + mid) as usize] <= x {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    lo
}

/// How many of the `n` values at `f[off..]` are `< x`.
#[cube]
fn count_lt(f: &[f64], off: u32, n: u32, x: f64) -> u32 {
    let mut lo = 0u32;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2u32;
        if f[(off + mid) as usize] < x {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Incoherent elastic cross section `b/2 (1 - e^{-2x}) / x`, `x = 2 E W'`.
#[cube]
fn incoherent_xs(bound_xs: f64, w: f64, e: f64) -> f64 {
    let x = 2.0f64 * e * w;
    let mut out = bound_xs;
    if x != 0.0f64 {
        out = bound_xs / 2.0f64 * -crate::common::polyfills::expm1_f64(-2.0f64 * x) / x;
    }
    out
}

/// Coherent elastic cross section: the cumulative factor of the edges below
/// `e`, over `e`.
#[cube]
fn coherent_xs(f: &[f64], el: u32, n_edges: u32, e: f64) -> f64 {
    let n = count_le(f, el + 2u32, n_edges, e);
    let mut out = e * 0.0f64;
    if n > 0u32 {
        out = f[(el + 2u32 + n_edges + n - 1u32) as usize] / e;
    }
    out
}

/// Table `t`'s (`sab_u32` word base) elastic cross section at `e`.
#[cube]
fn table_elastic_xs(u: &[u32], f: &[f64], t: u32, e: f64) -> f64 {
    let kind = u[(t + 4u32) as usize];
    let n_edges = u[(t + 5u32) as usize];
    let el = u[(t + 6u32) as usize];
    let mut out = e * 0.0f64;
    if kind == SAB_ELASTIC_COHERENT || kind == SAB_ELASTIC_MIXED {
        out += coherent_xs(f, el, n_edges, e);
    }
    if kind == SAB_ELASTIC_INCOHERENT || kind == SAB_ELASTIC_MIXED {
        out += incoherent_xs(f[el as usize], f[(el + 1u32) as usize], e);
    }
    out
}

/// Table `t`'s inelastic cross section at `e`: linear interpolation held at
/// the ends, as `yamc_nuclide::interpolate_linear`.
#[cube]
fn table_inelastic_xs(u: &[u32], f: &[f64], t: u32, e: f64) -> f64 {
    let n = u[t as usize];
    let off = u[(t + 1u32) as usize];
    let ys = off + n;
    let mut out = f[ys as usize];
    if n > 1u32 {
        if e >= f[(off + n - 1u32) as usize] {
            out = f[(ys + n - 1u32) as usize];
        } else if e > f[off as usize] {
            let mut low = 0u32;
            let mut high = n - 1u32;
            while high - low > 1u32 {
                let mid = (low + high) / 2u32;
                if f[(off + mid) as usize] <= e {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            let x1 = f[(off + low) as usize];
            let x2 = f[(off + low + 1u32) as usize];
            let y1 = f[(ys + low) as usize];
            let y2 = f[(ys + low + 1u32) as usize];
            out = y1 + (e - x1) * (y2 - y1) / (x2 - x1);
        }
    }
    out
}

/// Coherent elastic scattering off the Bragg edges below `e` (one draw).
// `clamp` is not a cube op; the bounds are compared by hand.
#[allow(clippy::manual_clamp)]
#[cube]
fn coherent_mu(f: &[f64], el: u32, n_edges: u32, e: f64, state: u64) -> BoundSample {
    let mut n = count_le(f, el + 2u32, n_edges, e);
    if n < 1u32 {
        n = 1u32;
    }
    let d = crate::common::pcg32::draw_uniform(state);
    let fac = el + 2u32 + n_edges;
    let target = d.xi * f[(fac + n - 1u32) as usize];
    let mut k = count_lt(f, fac, n, target);
    if k > n - 1u32 {
        k = n - 1u32;
    }
    BoundSample {
        bound: 1u32,
        e_out: e,
        mu: 1.0f64 - 2.0f64 * f[(el + 2u32 + k) as usize] / e,
        state: d.state,
    }
}

/// Incoherent elastic scattering with Debye-Waller integral `w` (one draw).
// `clamp` is not a cube op; the bounds are compared by hand.
#[allow(clippy::manual_clamp)]
#[cube]
fn incoherent_mu(w: f64, e: f64, state: u64) -> BoundSample {
    let d = crate::common::pcg32::draw_uniform(state);
    let c = 2.0f64 * e * w;
    let mut mu = 2.0f64 * d.xi - 1.0f64;
    if c >= 1e-10f64 {
        let low = crate::common::polyfills::exp_f64(-2.0f64 * c);
        let arg = low + d.xi * -crate::common::polyfills::expm1_f64(-2.0f64 * c);
        mu = -1.0f64;
        if arg > 0.0f64 {
            mu = 1.0f64 + crate::common::polyfills::ln_f64(arg) / c;
        }
        if mu < -1.0f64 {
            mu = -1.0f64;
        }
        if mu > 1.0f64 {
            mu = 1.0f64;
        }
    }
    BoundSample {
        bound: 1u32,
        e_out: e,
        mu,
        state: d.state,
    }
}

/// Incoherent inelastic scattering from table `t` (four draws), as
/// `thermal_sab::incoherent_inelastic`.
// Cube-friendly forms: hand clamps, the bounds check kept apart from the
// buffer read it guards, and `x != x` for NaN.
#[allow(clippy::manual_clamp, clippy::collapsible_if, clippy::eq_op)]
#[cube]
fn inelastic_scatter(
    u: &[u32],
    f: &[f64],
    mu_tab: &[f32],
    t: u32,
    row_base: u32,
    e_in: f64,
    state: u64,
) -> BoundSample {
    let n = u[t as usize];
    let off = u[(t + 1u32) as usize];
    let n_mu = u[(t + 2u32) as usize];
    let row_first = u[(t + 3u32) as usize];
    let mut st = state;

    // Grid point below e_in, clamped, and the fraction of the way to the next.
    let mut n2 = n;
    if n2 < 2u32 {
        n2 = 2u32;
    }
    let mut i = count_le(f, off, n, e_in);
    if i < 1u32 {
        i = 1u32;
    }
    if i > n2 - 1u32 {
        i = n2 - 1u32;
    }
    i -= 1u32;
    let mut f0 = e_in * 0.0f64;
    if n >= 2u32 {
        let ei = f[(off + i) as usize];
        let ei1 = f[(off + i + 1u32) as usize];
        f0 = (e_in - ei) / (ei1 - ei);
        if f0 < 0.0f64 {
            f0 = 0.0f64;
        }
        if f0 > 1.0f64 {
            f0 = 1.0f64;
        }
    }
    let mut l = i;
    if n >= 2u32 {
        let d = crate::common::pcg32::draw_uniform(st);
        st = d.state;
        if d.xi < f0 {
            l = i + 1u32;
        }
    }
    let r = row_base + 3u32 * (row_first + l);
    let m = u[r as usize];
    let o = u[(r + 1u32) as usize];
    let mo = u[(r + 2u32) as usize];
    let eo = o;
    let pdf = o + m;
    let cdf = o + 2u32 * m;

    let d1 = crate::common::pcg32::draw_uniform(st);
    st = d1.state;
    let r1 = d1.xi;
    // First cdf bin holding r1: partition_point over cdf[1..m] of `c <= r1`.
    let mut j = count_le(f, cdf + 1u32, m - 1u32, r1);
    if j > m - 2u32 {
        j = m - 2u32;
    }
    // A flat bin cannot hold the draw; take the first that can. The reads
    // stay out of the loop condition: cubecl-spirv's lowering of a `while`
    // whose `&&` condition indexes a buffer crashes the RADV compiler.
    let mut scanning = 1u32;
    while scanning == 1u32 {
        scanning = 0u32;
        if j + 2u32 < m {
            if f[(cdf + j + 1u32) as usize] <= f[(cdf + j) as usize] {
                j += 1u32;
                scanning = 1u32;
            }
        }
    }
    let e0 = f[(eo + j) as usize];
    let e1 = f[(eo + j + 1u32) as usize];
    let p0 = f[(pdf + j) as usize];
    let p1 = f[(pdf + j + 1u32) as usize];
    let c0 = f[(cdf + j) as usize];
    let c1 = f[(cdf + j + 1u32) as usize];
    let slope = (p1 - p0) / (e1 - e0);
    let mut slope_abs = slope;
    if slope < 0.0f64 {
        slope_abs = -slope;
    }
    #[allow(unused_assignments)]
    let mut e_out = e0;
    // `slope == 0 || !slope.is_finite()`: infinite above the largest finite
    // f64, NaN the only value unequal to itself.
    if slope == 0.0f64 || slope_abs > 1.797_693_134_862_315_7e308f64 || slope != slope {
        if p0 > 0.0f64 {
            e_out = e0 + (r1 - c0) / p0;
        } else {
            let mut q = (r1 - c0) / (c1 - c0);
            if q < 0.0f64 {
                q = 0.0f64;
            }
            if q > 1.0f64 {
                q = 1.0f64;
            }
            e_out = e0 + (e1 - e0) * q;
        }
    } else {
        let mut disc = p0 * p0 + 2.0f64 * slope * (r1 - c0);
        if disc < 0.0f64 {
            disc = 0.0f64;
        }
        e_out = e0 + (disc.sqrt() - p0) / slope;
    }
    let e_l = f[(off + l) as usize];
    if e_out < 0.5f64 * e_l {
        e_out *= 2.0f64 * e_in / e_l - 1.0f64;
    } else {
        e_out += e_in - e_l;
    }

    let mut fr = e_in * 0.0f64;
    if c1 > c0 {
        fr = (r1 - c0) / (c1 - c0);
        if fr < 0.0f64 {
            fr = 0.0f64;
        }
        if fr > 1.0f64 {
            fr = 1.0f64;
        }
    }
    let d2 = crate::common::pcg32::draw_uniform(st);
    st = d2.state;
    let mut k = (d2.xi * n_mu as f64) as u32;
    if k > n_mu - 1u32 {
        k = n_mu - 1u32;
    }
    let lo_row = mo + j * n_mu;
    let hi_row = mo + (j + 1u32) * n_mu;
    let mu_k_lo = mu_tab[(lo_row + k) as usize] as f64;
    let mu_k_hi = mu_tab[(hi_row + k) as usize] as f64;
    let mu = mu_k_lo + fr * (mu_k_hi - mu_k_lo);
    let mut left = -1.0f64 - (mu + 1.0f64);
    if k > 0u32 {
        let a = mu_tab[(lo_row + k - 1u32) as usize] as f64;
        let b = mu_tab[(hi_row + k - 1u32) as usize] as f64;
        left = a + fr * (b - a);
    }
    let mut right = 1.0f64 + (1.0f64 - mu);
    if k < n_mu - 1u32 {
        let a = mu_tab[(lo_row + k + 1u32) as usize] as f64;
        let b = mu_tab[(hi_row + k + 1u32) as usize] as f64;
        right = a + fr * (b - a);
    }
    let mut half = mu - left;
    if right - mu < half {
        half = right - mu;
    }
    let d3 = crate::common::pcg32::draw_uniform(st);
    st = d3.state;
    let mut mu_out = mu + half * (d3.xi - 0.5f64);
    if mu_out < -1.0f64 {
        mu_out = -1.0f64;
    }
    if mu_out > 1.0f64 {
        mu_out = 1.0f64;
    }
    if e_out < 0.0f64 {
        e_out = 0.0f64;
    }
    BoundSample {
        bound: 1u32,
        e_out,
        mu: mu_out,
        state: st,
    }
}

/// Scatter a neutron at `e_in` off slab `slab` by its thermal table, if one
/// binds it: the twin of `thermal_inputs::sample_bound_cpu`, same draws in the
/// same order. `sigma_el` is the slab's macroscopic elastic cross section at
/// `e_in`, its table applied.
// `slab * 0` seeds a runtime u32 (see below).
#[cube]
#[allow(clippy::too_many_arguments, clippy::erasing_op)]
pub fn sample_bound(
    sab_u32: &[u32],
    sab_f64: &[f64],
    sab_mu: &[f32],
    slab: u32,
    e_in: f64,
    sigma_el: f64,
    state: u64,
) -> BoundSample {
    // The result, field by field: cubecl cannot reassign a struct value.
    let mut bound = 0u32;
    let mut e_out = e_in;
    let mut mu = e_in * 0.0f64;
    let mut st = state;
    // A slab past the map's end (added after the tables were laid out) is
    // free. Seeded from a runtime value so it can take the runtime read below.
    let mut b = slab * 0u32 + SAB_NONE;
    if slab < sab_u32[0] {
        b = sab_u32[(4u32 + slab) as usize];
    }
    if b != SAB_NONE {
        let bind_base = sab_u32[1];
        let table_base = sab_u32[2];
        let row_base = sab_u32[3];
        let bf = b * 4u32;
        let fraction = sab_f64[bf as usize];
        let energy_max = sab_f64[(bf + 1u32) as usize];
        let w = sab_f64[(bf + 2u32) as usize];
        let density = sab_f64[(bf + 3u32) as usize];
        if e_in < energy_max {
            let lo = sab_u32[(bind_base + 2u32 * b) as usize];
            let hi = sab_u32[(bind_base + 2u32 * b + 1u32) as usize];
            let t_lo = table_base + 8u32 * lo;
            let mut is_bound = 1u32;
            if fraction < 1.0f64 {
                let mut blended = table_elastic_xs(sab_u32, sab_f64, t_lo, e_in)
                    + table_inelastic_xs(sab_u32, sab_f64, t_lo, e_in);
                if hi != SAB_NONE {
                    let t_hi = table_base + 8u32 * hi;
                    let both_hi = table_elastic_xs(sab_u32, sab_f64, t_hi, e_in)
                        + table_inelastic_xs(sab_u32, sab_f64, t_hi, e_in);
                    blended = (1.0f64 - w) * blended + w * both_hi;
                }
                let thermal = fraction * blended;
                let total = sigma_el / density;
                if total <= 0.0f64 {
                    is_bound = 0u32;
                } else {
                    let d = crate::common::pcg32::draw_uniform(st);
                    st = d.state;
                    if d.xi * total >= thermal {
                        is_bound = 0u32;
                    }
                }
            }
            if is_bound == 1u32 {
                let mut t = t_lo;
                if hi != SAB_NONE {
                    let d = crate::common::pcg32::draw_uniform(st);
                    st = d.state;
                    if d.xi < w {
                        t = table_base + 8u32 * hi;
                    }
                }
                let inelastic = table_inelastic_xs(sab_u32, sab_f64, t, e_in);
                let elastic = table_elastic_xs(sab_u32, sab_f64, t, e_in);
                let mut choose_elastic = 0u32;
                if elastic > 0.0f64 {
                    let d = crate::common::pcg32::draw_uniform(st);
                    st = d.state;
                    if d.xi * (elastic + inelastic) < elastic {
                        choose_elastic = 1u32;
                    }
                }
                if choose_elastic == 1u32 {
                    let kind = sab_u32[(t + 4u32) as usize];
                    let n_edges = sab_u32[(t + 5u32) as usize];
                    let el = sab_u32[(t + 6u32) as usize];
                    let w_dw = sab_f64[(el + 1u32) as usize];
                    let mut coherent = 0u32;
                    if kind == SAB_ELASTIC_COHERENT {
                        coherent = 1u32;
                    } else if kind == SAB_ELASTIC_MIXED {
                        let coh = coherent_xs(sab_f64, el, n_edges, e_in);
                        let inc = incoherent_xs(sab_f64[el as usize], w_dw, e_in);
                        if coh > 0.0f64 {
                            let d = crate::common::pcg32::draw_uniform(st);
                            st = d.state;
                            if d.xi * (coh + inc) < coh {
                                coherent = 1u32;
                            }
                        }
                    }
                    if coherent == 1u32 {
                        let r = coherent_mu(sab_f64, el, n_edges, e_in, st);
                        e_out = r.e_out;
                        mu = r.mu;
                        st = r.state;
                    } else {
                        let r = incoherent_mu(w_dw, e_in, st);
                        e_out = r.e_out;
                        mu = r.mu;
                        st = r.state;
                    }
                } else {
                    let r = inelastic_scatter(sab_u32, sab_f64, sab_mu, t, row_base, e_in, st);
                    e_out = r.e_out;
                    mu = r.mu;
                    st = r.state;
                }
                bound = 1u32;
            }
        }
    }
    BoundSample {
        bound,
        e_out,
        mu,
        state: st,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neutron::thermal_inputs::{sample_bound_cpu, SlabThermal, ThermalInputs};
    use crate::{GpuContext, GpuInitError};
    use yamc_nuclide::thermal_scattering::{
        Elastic, Inelastic, OutgoingDistribution, ThermalTemperature,
    };

    /// One sample per thread: energy, the slab's elastic, a seed.
    #[cube(launch)]
    fn sample_bound_kernel(
        sab_u32: &[u32],
        sab_f64: &[f64],
        sab_mu: &[f32],
        slab: &[u32],
        e_in: &[f64],
        sigma_el: &[f64],
        seeds: &[u32],
        out_f: &mut [f64],
        out_u: &mut [u32],
    ) {
        if ABSOLUTE_POS >= e_in.len() {
            terminate!();
        }
        let i = ABSOLUTE_POS;
        let r = sample_bound(
            sab_u32,
            sab_f64,
            sab_mu,
            slab[i],
            e_in[i],
            sigma_el[i],
            crate::common::pcg32::expand_seed(seeds[i]),
        );
        out_f[2 * i] = r.e_out;
        out_f[2 * i + 1] = r.mu;
        out_u[3 * i] = r.bound;
        out_u[3 * i + 1] = (r.state & 0xFFFF_FFFFu64) as u32;
        out_u[3 * i + 2] = (r.state >> 32u64) as u32;
    }

    /// A small table with every part the sampler reads: mixed or coherent
    /// elastic, and inelastic rows with a flat cdf bin and a falling pdf.
    fn temperature(kelvin: f64, scale: f64, elastic: Elastic) -> ThermalTemperature {
        let energy = vec![1e-5, 0.01, 0.1, 1.0];
        let n_mu = 4;
        let distributions = energy
            .iter()
            .map(|&e| {
                let energy_out: Vec<f64> = vec![0.0, 0.25 * e, 0.5 * e, e, 2.0 * e + 0.01];
                let pdf: Vec<f64> = vec![0.0, 2.0, 2.0, 1.0, 0.0];
                let mut cdf = vec![0.0; 5];
                for j in 1..5 {
                    cdf[j] = cdf[j - 1]
                        + 0.5 * (pdf[j] + pdf[j - 1]) * (energy_out[j] - energy_out[j - 1]);
                }
                let mu: Vec<f32> = (0..5)
                    .flat_map(|j| {
                        let s = 0.1 * j as f32;
                        [-0.8 + s, -0.2 + s, 0.3 + s * 0.5, 0.8 + s * 0.2]
                    })
                    .collect();
                OutgoingDistribution {
                    energy_out,
                    pdf,
                    cdf,
                    mu,
                }
            })
            .collect();
        ThermalTemperature {
            kelvin,
            kt: 8.617333262e-5 * kelvin,
            inelastic: Inelastic {
                energy,
                xs: vec![5.0 * scale, 4.0 * scale, 3.0 * scale, 2.0 * scale],
                distributions,
                n_mu,
            },
            elastic: Some(elastic),
        }
    }

    /// The kernel's bound sampler makes the CPU twin's draws and returns its
    /// outgoing energy and cosine, across both temperatures, mixed and
    /// coherent elastic, inelastic, a partly bound nuclide's free share,
    /// and energies above energy_max and on a free slab.
    #[test]
    fn gpu_sample_bound_matches_cpu_twin() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let lo = temperature(
            300.0,
            1.0,
            Elastic::Mixed {
                bragg_edges: vec![0.002, 0.004, 0.009],
                factors: vec![0.01, 0.01, 0.04],
                bound_xs: 4.0,
                debye_waller: 2.0,
            },
        );
        let hi = temperature(
            400.0,
            1.5,
            Elastic::Coherent {
                bragg_edges: vec![0.003, 0.006],
                factors: vec![0.02, 0.05],
            },
        );
        let inputs = ThermalInputs::from_slabs(&[
            None,
            Some(SlabThermal {
                fraction: 0.7,
                energy_max: 1.0,
                density: 0.08,
                lo: &lo,
                hi: Some((&hi, 0.35)),
            }),
            Some(SlabThermal {
                fraction: 1.0,
                energy_max: 1.0,
                density: 0.02,
                lo: &lo,
                hi: None,
            }),
        ]);
        assert!(inputs.any_bound());

        let energies = [1e-4, 0.001, 0.0025, 0.005, 0.02, 0.3, 0.99, 1.5];
        let mut slab = Vec::new();
        let mut e_in = Vec::new();
        let mut sigma_el = Vec::new();
        let mut seeds = Vec::new();
        for s in 0..3u32 {
            for &e in &energies {
                for k in 0..64u32 {
                    slab.push(s);
                    e_in.push(e);
                    // An elastic about twice the bound part, so a partly bound
                    // slab draws free about half the time.
                    sigma_el.push(0.08 * 10.0 / (1.0 + e));
                    seeds.push(1_000 * s + 97 * k + (e * 1e4) as u32);
                }
            }
        }
        let n = e_in.len();
        let client = ctx.client();
        let h = |v: &[u8]| client.create_from_slice(v);
        let out_f = client.empty(2 * n * 8);
        let out_u = client.empty(3 * n * 4);
        launch!(
            sample_bound_kernel,
            &client,
            CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(
                h(bytemuck::cast_slice(&inputs.sab_u32)),
                inputs.sab_u32.len(),
            ),
            BufferArg::from_raw_parts(
                h(bytemuck::cast_slice(&inputs.sab_f64)),
                inputs.sab_f64.len(),
            ),
            BufferArg::from_raw_parts(h(bytemuck::cast_slice(&inputs.sab_mu)), inputs.sab_mu.len(),),
            BufferArg::from_raw_parts(h(bytemuck::cast_slice(&slab)), n),
            BufferArg::from_raw_parts(h(bytemuck::cast_slice(&e_in)), n),
            BufferArg::from_raw_parts(h(bytemuck::cast_slice(&sigma_el)), n),
            BufferArg::from_raw_parts(h(bytemuck::cast_slice(&seeds)), n),
            BufferArg::from_raw_parts(out_f.clone(), 2 * n),
            BufferArg::from_raw_parts(out_u.clone(), 3 * n),
        );
        let gf: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_f).unwrap()).to_vec();
        let gu: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_u).unwrap()).to_vec();

        let (mut bound, mut free) = (0, 0);
        for i in 0..n {
            let mut pcg = yamc_rng::expand_seed(seeds[i]);
            let cpu = sample_bound_cpu(
                &inputs.sab_u32,
                &inputs.sab_f64,
                &inputs.sab_mu,
                slab[i] as usize,
                e_in[i],
                sigma_el[i],
                &mut pcg,
            );
            let state = gu[3 * i + 1] as u64 | ((gu[3 * i + 2] as u64) << 32);
            assert_eq!(
                state, pcg,
                "draw count differs at sample {i} (E = {})",
                e_in[i]
            );
            match cpu {
                None => {
                    assert_eq!(gu[3 * i], 0, "GPU scattered bound where the CPU did not");
                    free += 1;
                }
                Some((e, mu)) => {
                    assert_eq!(gu[3 * i], 1, "GPU scattered free where the CPU did not");
                    let (ge, gmu) = (gf[2 * i], gf[2 * i + 1]);
                    assert!(
                        (ge - e).abs() <= 1e-12 * e.abs().max(1e-12),
                        "E' {ge} against {e} at sample {i}"
                    );
                    assert!(
                        (gmu - mu).abs() <= 1e-10,
                        "mu {gmu} against {mu} at sample {i}"
                    );
                    bound += 1;
                }
            }
        }
        // Both outcomes were exercised.
        assert!(bound > n / 3 && free > n / 4, "bound {bound}, free {free}");
    }
}
