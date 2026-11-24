//! Bound thermal scattering on the GPU: a material's thermal-table data laid
//! out flat per nuclide slab, and the CPU twin of the kernel's sampler
//! (`super::thermal_sab::sample_bound`).
//!
//! The samplers are those of `yamc_physics::gpu::flat::thermal_sab` and the
//! draw schedule is the CPU's `sample_bound` (`crates/yamc/src/transport/
//! scatter.rs`), in this order, before the caller's azimuth:
//! 1. bound or free, with probability `f (sigma_el,b + sigma_inel,b)` over the
//!    nuclide's elastic cross section, only when the bound fraction f is
//!    below 1;
//! 2. the upper of the two table temperatures with probability equal to its
//!    blend weight, only when the material sits between two;
//! 3. elastic or inelastic in proportion to that table's cross sections, only
//!    when the table has an elastic part;
//! 4. coherent or incoherent in proportion, only for mixed elastic;
//! 5. the sampler's own draws.
//!
//! The one difference from the CPU is the denominator of 1: the nuclide's
//! elastic cross section with its table applied, which the kernel holds as
//! the struck nuclide's macroscopic elastic partial on its grid, divided by
//! the nuclide's atom density. The CPU evaluates `f (...) + (1 - f) sigma_el,free`
//! at the collision energy; the grid carries the table's own points, so the
//! two agree to interpolation.
//!
//! # Layout
//!
//! `sab_u32`:
//! - `[0]` slab count, `[1]` binding base, `[2]` table base, `[3]` row base;
//! - `[SAB_HEADER + slab]`: the slab's binding, [`SAB_NONE`] for a free one;
//! - binding `b` at `binding base + 2 b`: lower table, upper table
//!   ([`SAB_NONE`] when the material sits on one temperature);
//! - table `t` at `table base + SAB_TABLE_COLS t`: inelastic point count, its
//!   `sab_f64` offset (energies, then cross sections), cosines per outgoing
//!   energy, first row, elastic kind, Bragg edge count, elastic `sab_f64`
//!   offset (bound cross section, Debye-Waller integral, edges, factors);
//! - row `r` at `row base + SAB_ROW_COLS r`, one per inelastic incident energy:
//!   outgoing point count, its `sab_f64` offset (energies, pdf, cdf), its
//!   `sab_mu` offset.
//!
//! `sab_f64`: binding `b` at `SAB_BINDING_F64 b`: bound fraction, energy_max,
//! upper-table weight, the nuclide's atom density; then the data the offsets
//! above point at. `sab_mu`: the cosines, `n_mu` per outgoing energy, as f32,
//! as the tables store them.

use yamc_nuclide::thermal_scattering::{Elastic, ThermalTemperature};
use yamc_physics::gpu::flat::thermal_sab;

/// Words before the slab map in `sab_u32`.
pub const SAB_HEADER: usize = 4;
/// No binding / no upper table.
pub const SAB_NONE: u32 = u32::MAX;
/// `sab_u32` words per table.
pub const SAB_TABLE_COLS: usize = 8;
/// `sab_u32` words per inelastic incident-energy row.
pub const SAB_ROW_COLS: usize = 3;
/// `sab_f64` words per binding.
pub const SAB_BINDING_F64: usize = 4;

/// Elastic kinds in a table's `el_kind` word.
pub const SAB_ELASTIC_NONE: u32 = 0;
pub const SAB_ELASTIC_COHERENT: u32 = 1;
pub const SAB_ELASTIC_INCOHERENT: u32 = 2;
pub const SAB_ELASTIC_MIXED: u32 = 3;

/// One nuclide slab's thermal binding, as the host translator sees it.
pub struct SlabThermal<'a> {
    /// The fraction of the nuclide that is bound, in (0, 1].
    pub fraction: f64,
    /// Below this energy (eV) the table applies.
    pub energy_max: f64,
    /// The nuclide's atom density in the material (atoms per barn-cm).
    pub density: f64,
    /// The table temperature the material sits on, or the lower of two.
    pub lo: &'a ThermalTemperature,
    /// The upper of two table temperatures and its blend weight.
    pub hi: Option<(&'a ThermalTemperature, f64)>,
}

/// Every slab's thermal binding, flat, for the kernel and its CPU twin.
#[derive(Debug, Clone)]
pub struct ThermalInputs {
    pub sab_u32: Vec<u32>,
    pub sab_f64: Vec<f64>,
    pub sab_mu: Vec<f32>,
}

impl Default for ThermalInputs {
    /// No slab is bound.
    fn default() -> Self {
        Self::empty(0)
    }
}

impl ThermalInputs {
    /// `n_slab` slabs, none bound.
    pub fn empty(n_slab: usize) -> Self {
        Self::from_slabs(&(0..n_slab).map(|_| None).collect::<Vec<_>>())
    }

    /// Lay out `slabs` (one entry per global nuclide slab). A table temperature
    /// shared by several slabs (a table binding two nuclides, or two materials
    /// on the same temperature) is stored once.
    pub fn from_slabs(slabs: &[Option<SlabThermal<'_>>]) -> Self {
        let n_slab = slabs.len();
        // Each distinct table temperature, by identity, and its index.
        fn index_of<'t>(
            t: &'t ThermalTemperature,
            tables: &mut Vec<&'t ThermalTemperature>,
        ) -> u32 {
            match tables.iter().position(|&x| std::ptr::eq(x, t)) {
                Some(i) => i as u32,
                None => {
                    tables.push(t);
                    (tables.len() - 1) as u32
                }
            }
        }
        let mut tables: Vec<&ThermalTemperature> = Vec::new();
        // Bindings, in slab order.
        let mut slab_map = vec![SAB_NONE; n_slab];
        let mut bindings_u32: Vec<u32> = Vec::new();
        let mut sab_f64: Vec<f64> = Vec::new();
        for (s, slab) in slabs.iter().enumerate() {
            let Some(b) = slab else { continue };
            slab_map[s] = (bindings_u32.len() / 2) as u32;
            let lo = index_of(b.lo, &mut tables);
            let (hi, w) = match b.hi {
                Some((t, w)) => (index_of(t, &mut tables), w),
                None => (SAB_NONE, 0.0),
            };
            bindings_u32.extend_from_slice(&[lo, hi]);
            sab_f64.extend_from_slice(&[b.fraction, b.energy_max, w, b.density]);
        }

        let n_bind = bindings_u32.len() / 2;
        let bind_base = SAB_HEADER + n_slab;
        let table_base = bind_base + 2 * n_bind;
        let n_rows: usize = tables.iter().map(|t| t.inelastic.energy.len()).sum();
        let row_base = table_base + SAB_TABLE_COLS * tables.len();
        let mut sab_u32 = Vec::with_capacity(row_base + SAB_ROW_COLS * n_rows);
        sab_u32.extend_from_slice(&[
            n_slab as u32,
            bind_base as u32,
            table_base as u32,
            row_base as u32,
        ]);
        sab_u32.extend_from_slice(&slab_map);
        sab_u32.extend_from_slice(&bindings_u32);

        let mut rows: Vec<u32> = Vec::with_capacity(SAB_ROW_COLS * n_rows);
        let mut sab_mu: Vec<f32> = Vec::new();
        for t in &tables {
            let inel = &t.inelastic;
            let inel_off = sab_f64.len() as u32;
            sab_f64.extend_from_slice(&inel.energy);
            sab_f64.extend_from_slice(&inel.xs);
            let row_first = (rows.len() / SAB_ROW_COLS) as u32;
            for d in &inel.distributions {
                let out_off = sab_f64.len() as u32;
                sab_f64.extend_from_slice(&d.energy_out);
                sab_f64.extend_from_slice(&d.pdf);
                sab_f64.extend_from_slice(&d.cdf);
                rows.extend_from_slice(&[d.energy_out.len() as u32, out_off, sab_mu.len() as u32]);
                sab_mu.extend_from_slice(&d.mu);
            }
            let el_off = sab_f64.len() as u32;
            let (kind, edges, factors, bound_xs, dw): (u32, &[f64], &[f64], f64, f64) =
                match &t.elastic {
                    None => (SAB_ELASTIC_NONE, &[], &[], 0.0, 0.0),
                    Some(Elastic::Coherent {
                        bragg_edges,
                        factors,
                    }) => (SAB_ELASTIC_COHERENT, bragg_edges, factors, 0.0, 0.0),
                    Some(Elastic::Incoherent {
                        bound_xs,
                        debye_waller,
                    }) => (SAB_ELASTIC_INCOHERENT, &[], &[], *bound_xs, *debye_waller),
                    Some(Elastic::Mixed {
                        bragg_edges,
                        factors,
                        bound_xs,
                        debye_waller,
                    }) => (
                        SAB_ELASTIC_MIXED,
                        bragg_edges,
                        factors,
                        *bound_xs,
                        *debye_waller,
                    ),
                };
            sab_f64.extend_from_slice(&[bound_xs, dw]);
            sab_f64.extend_from_slice(edges);
            sab_f64.extend_from_slice(factors);
            sab_u32.extend_from_slice(&[
                inel.energy.len() as u32,
                inel_off,
                inel.n_mu as u32,
                row_first,
                kind,
                edges.len() as u32,
                el_off,
                0,
            ]);
        }
        sab_u32.extend_from_slice(&rows);
        // cubecl rejects empty bindings.
        if sab_f64.is_empty() {
            sab_f64.push(0.0);
        }
        if sab_mu.is_empty() {
            sab_mu.push(0.0);
        }
        ThermalInputs {
            sab_u32,
            sab_f64,
            sab_mu,
        }
    }

    /// Whether any slab is bound.
    pub fn any_bound(&self) -> bool {
        let n_slab = self.sab_u32[0] as usize;
        self.sab_u32[SAB_HEADER..SAB_HEADER + n_slab]
            .iter()
            .any(|&b| b != SAB_NONE)
    }
}

// ----------------------------- CPU twin -----------------------------

/// A table's elastic cross section at `e` (barns), from the flat layout.
fn elastic_xs_cpu(u: &[u32], f: &[f64], t: usize, e: f64) -> f64 {
    let kind = u[t + 4];
    let n_edges = u[t + 5] as usize;
    let el = u[t + 6] as usize;
    let (bound_xs, w) = (f[el], f[el + 1]);
    let edges = &f[el + 2..el + 2 + n_edges];
    let factors = &f[el + 2 + n_edges..el + 2 + 2 * n_edges];
    let coherent = || {
        let n = edges.partition_point(|&x| x <= e);
        if n == 0 {
            0.0
        } else {
            factors[n - 1] / e
        }
    };
    let incoherent = || {
        let x = 2.0 * e * w;
        if x == 0.0 {
            bound_xs
        } else {
            bound_xs / 2.0 * -(-2.0 * x).exp_m1() / x
        }
    };
    match kind {
        SAB_ELASTIC_COHERENT => coherent(),
        SAB_ELASTIC_INCOHERENT => incoherent(),
        SAB_ELASTIC_MIXED => coherent() + incoherent(),
        _ => 0.0,
    }
}

/// A table's inelastic cross section at `e` (barns), from the flat layout.
fn inelastic_xs_cpu(u: &[u32], f: &[f64], t: usize, e: f64) -> f64 {
    let n = u[t] as usize;
    let off = u[t + 1] as usize;
    yamc_nuclide::interpolate_linear(&f[off..off + n], &f[off + n..off + 2 * n], e)
}

/// CPU twin of the kernel's `thermal_sab::sample_bound`: `(E', mu_lab)` of a neutron at `e_in`
/// scattering elastically off slab `slab`, or `None` when the slab is free,
/// `e_in` is at or above energy_max, or the free share of a partly bound
/// nuclide was drawn. `sigma_el` is the slab's macroscopic elastic cross
/// section at `e_in` with its table applied.
pub fn sample_bound_cpu(
    sab_u32: &[u32],
    sab_f64: &[f64],
    sab_mu: &[f32],
    slab: usize,
    e_in: f64,
    sigma_el: f64,
    pcg: &mut u64,
) -> Option<(f64, f64)> {
    use yamc_rng::next_xi;
    let (u, f) = (sab_u32, sab_f64);
    if slab >= u[0] as usize {
        return None;
    }
    let b = u[SAB_HEADER + slab];
    if b == SAB_NONE {
        return None;
    }
    let (bind_base, table_base, row_base) = (u[1] as usize, u[2] as usize, u[3] as usize);
    let bf = b as usize * SAB_BINDING_F64;
    let (fraction, energy_max, w, density) = (f[bf], f[bf + 1], f[bf + 2], f[bf + 3]);
    if e_in >= energy_max {
        return None;
    }
    let lo = u[bind_base + 2 * b as usize] as usize;
    let hi = u[bind_base + 2 * b as usize + 1];
    let tab = |t: usize| table_base + SAB_TABLE_COLS * t;
    if fraction < 1.0 {
        let both =
            |t: usize| elastic_xs_cpu(u, f, tab(t), e_in) + inelastic_xs_cpu(u, f, tab(t), e_in);
        let blended = if hi == SAB_NONE {
            both(lo)
        } else {
            (1.0 - w) * both(lo) + w * both(hi as usize)
        };
        let thermal = fraction * blended;
        let total = sigma_el / density;
        if total <= 0.0 || next_xi(pcg) * total >= thermal {
            return None;
        }
    }
    let t = if hi != SAB_NONE && next_xi(pcg) < w {
        hi as usize
    } else {
        lo
    };
    let t = tab(t);
    let inelastic = inelastic_xs_cpu(u, f, t, e_in);
    let elastic = elastic_xs_cpu(u, f, t, e_in);
    if elastic > 0.0 && next_xi(pcg) * (elastic + inelastic) < elastic {
        let n_edges = u[t + 5] as usize;
        let el = u[t + 6] as usize;
        let w_dw = f[el + 1];
        let edges = &f[el + 2..el + 2 + n_edges];
        let factors = &f[el + 2 + n_edges..el + 2 + 2 * n_edges];
        return Some(match u[t + 4] {
            SAB_ELASTIC_COHERENT => thermal_sab::coherent_elastic(e_in, edges, factors, pcg),
            SAB_ELASTIC_INCOHERENT => thermal_sab::incoherent_elastic(e_in, w_dw, pcg),
            _ => {
                let n = edges.partition_point(|&x| x <= e_in);
                let coh = if n == 0 { 0.0 } else { factors[n - 1] / e_in };
                let x = 2.0 * e_in * w_dw;
                let bound_xs = f[el];
                let inc = if x == 0.0 {
                    bound_xs
                } else {
                    bound_xs / 2.0 * -(-2.0 * x).exp_m1() / x
                };
                if coh > 0.0 && next_xi(pcg) * (coh + inc) < coh {
                    thermal_sab::coherent_elastic(e_in, edges, factors, pcg)
                } else {
                    thermal_sab::incoherent_elastic(e_in, w_dw, pcg)
                }
            }
        });
    }
    let n = u[t] as usize;
    let off = u[t + 1] as usize;
    let n_mu = u[t + 2] as usize;
    let row_first = u[t + 3] as usize;
    Some(thermal_sab::incoherent_inelastic(
        e_in,
        &f[off..off + n],
        |l| {
            let r = row_base + SAB_ROW_COLS * (row_first + l);
            let (m, o, mo) = (u[r] as usize, u[r + 1] as usize, u[r + 2] as usize);
            thermal_sab::Outgoing {
                energy_out: &f[o..o + m],
                pdf: &f[o + m..o + 2 * m],
                cdf: &f[o + 2 * m..o + 3 * m],
                mu: &sab_mu[mo..mo + m * n_mu],
                n_mu,
            }
        },
        pcg,
    ))
}
