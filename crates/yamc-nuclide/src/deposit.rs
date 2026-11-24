//! What each reaction deposits: the per-reaction split of the heating (MT 301
//! and MT 901) and damage-energy (MT 444) rows.
//!
//! The published data carries only the totals, `K(E) = Σ_mt σ_mt(E) h_mt(E)`,
//! from NJOY's HEATR. A nuclear-data uncertainty run moves each partial cross
//! section, and the heating it scores should move with them, by
//! `Σ_mt h_mt (σ'_mt - σ_mt)`: the energy a reaction deposits is a property
//! of the reaction, not of how often it happens. This module rebuilds
//! `h_mt(E)`, in eV per reaction, from the data the loader already has, the
//! way HEATR builds the totals:
//!
//! - **Heating by energy balance**, `h = E + Q - ν Ē_n - Ē_γ`: what comes in,
//!   less what the outgoing neutrons and photons carry away. `Q` is the
//!   ground-state (mass-difference) Q value, so a level's excitation is
//!   deposited unless its photons are subtracted. `Ē_n` is the mean lab
//!   energy of the reaction's first neutron product (the one the transport
//!   emits) times its yield, and `Ē_γ` the yield-weighted mean energy of its
//!   photon products. MT 301 subtracts the photons, which a coupled run
//!   transports and scores itself; MT 901 keeps them, deposited where they
//!   are made. Photon production an evaluation lumps on a sum (MT 3, MT 4,
//!   MT 103 to 107) is subtracted from each partial of the sum, per reaction
//!   alike.
//! - **Capture** under MT 301 is the recoil by momentum conservation, as
//!   HEATR computes it: the neutron's momentum, `E / (A+1)`, plus each
//!   photon's kick, `Σ y <E_γ²> / (2 (A+1) m_n c²)`. Under MT 901 it is that
//!   recoil plus the photons' energy as the photon data gives it, which is
//!   what a coupled run deposits (`E + Q` where the data has no capture
//!   photons).
//! - **Damage energy** is the Lindhard partition (with Robinson's fit) of
//!   the recoils' energy, integrated over their spectrum: two-body
//!   kinematics with the evaluated angular distribution for elastic and the
//!   inelastic levels; the one-particle approximation (the residual takes the
//!   momentum the first emitted neutron leaves) for continuum neutron
//!   emission; each capture photon's kick for capture; isotropic two-body
//!   breakup for a charged-particle level, or a charged-particle reaction
//!   stated without levels (Li6 (n,t)), counting both the residual and the
//!   light particle, each as the ion it is; and HEATR's Coulomb-limited
//!   isotropic emission for a charged-particle continuum (MT 649, 849, ...).
//!   The displacement thresholds are HEATR's defaults by element, so a
//!   partial matches the total it splits.
//! - **MT 5**, which lumps several channels (above 20 MeV usually the whole
//!   non-elastic) under one Q value and whose charged-particle spectra the
//!   data does not carry, takes what the other partials leave of the stored
//!   row, where its cross section is not zero, up to `E + Q` per reaction
//!   ([`deposits`]).
//!
//! The rebuilt partials reproduce the stored rows closely where the data
//! the stored rows were built from is all in the published files (Fe56, Li6,
//! Cu63, Al27, O16 heating to a fraction of a percent at most energies).
//! Where HEATR read spectra the published files do not carry (recoil and
//! charged-particle distributions in MF=6, as for W184 and Be9 above a few
//! MeV), the split differs from HEATR's. The partials are kept as computed
//! there rather than scaled to the row: W184's stored damage energy above
//! 8 MeV is below what its (n,2n) alone deposits by momentum conservation
//! (the residual moves at least with the compound nucleus, `E A / (A+1)²`),
//! so scaling would move the well-reproduced elastic share wrongly too.
//!
//! Not split: fission. The fragment and beta energies HEATR's MT 301 counts
//! come from MF=1/MT=458, which the published data does not carry, and a
//! fission Q value includes neutrinos and delayed energy. Fission's share of
//! the stored row, and whatever else the partials do not reproduce, has no
//! partial, so it does not move with the cross sections;
//! [`unexplained_share`] says how much that is.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::nuclide::Nuclide;
use crate::reaction::Reaction;
use crate::reaction_product::{
    AngleDistribution, AngleEnergyDistribution, EnergyDistribution, ParticleType, ReactionProduct,
    Tabulated, TabulatedInterp, TabulatedProbability,
};
use crate::secondary_correlated::{self, CorrTable, CorrelatedAngleEnergy};
use crate::secondary_kalbach::{self, KMTable, KalbachMann};

/// Which stored row a set of deposits splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Response {
    /// MT 301: neutron KERMA with the photons' energy carried away.
    Heating,
    /// MT 901: neutron KERMA with the photons' energy deposited locally.
    HeatingLocal,
    /// MT 444: damage energy production.
    DamageEnergy,
}

impl Response {
    /// Every response, in MT order.
    pub const ALL: [Response; 3] = [
        Response::Heating,
        Response::DamageEnergy,
        Response::HeatingLocal,
    ];

    /// The MT of the stored row.
    pub fn mt(self) -> i32 {
        match self {
            Response::Heating => 301,
            Response::DamageEnergy => 444,
            Response::HeatingLocal => 901,
        }
    }

    /// The response a stored row's MT holds, if it is one.
    pub fn from_mt(mt: i32) -> Option<Self> {
        match mt {
            301 => Some(Response::Heating),
            444 => Some(Response::DamageEnergy),
            901 => Some(Response::HeatingLocal),
            _ => None,
        }
    }
}

/// The neutron rest energy, eV (NJOY's `amassn * amu * c²`).
const NEUTRON_MASS_EV: f64 = 1.00866491595 * 931.49410242e6;

/// Light-particle masses over the neutron's, as HEATR uses them (ENDF-102
/// Appendix H): proton, deuteron, triton, helion, alpha.
const PARTICLE_MASS: [f64; 5] = [
    1.007276466621 / 1.00866491595,
    2.013553212745 / 1.00866491595,
    3.01550071621 / 1.00866491595,
    3.014932247175 / 1.00866491595,
    4.001506179127 / 1.00866491595,
];

/// Light-particle charges, in the order of [`PARTICLE_MASS`].
const PARTICLE_CHARGE: [f64; 5] = [1.0, 1.0, 1.0, 2.0, 2.0];

/// Incident energies per decade the deposits are evaluated at, besides the
/// distributions' own incident grids, before interpolation onto the
/// nuclide's grid.
const POINTS_PER_DECADE: f64 = 64.0;

/// HEATR's default displacement threshold for element `z`, eV: the energy a
/// recoil must exceed to be counted at all.
pub fn heatr_displacement_threshold(z: u32) -> f64 {
    match z {
        4 | 6 => 31.0,
        12 | 14 => 25.0,
        13 => 27.0,
        20 | 22..=29 | 40 | 41 => 40.0,
        42 | 47 => 60.0,
        73 | 74 => 90.0,
        79 => 30.0,
        _ => 25.0,
    }
}

/// The Lindhard partition of a recoil's energy into atomic motion, with
/// Robinson's fit to the electronic stopping, for a recoil `(z_r, a_r)` in a
/// lattice `(z_l, a_l)`: HEATR's `df`. Zero below the displacement threshold.
#[derive(Debug, Clone, Copy)]
pub struct Lindhard {
    inverse_el: f64,
    k: f64,
    threshold: f64,
    zero: bool,
}

impl Lindhard {
    /// The partition for one recoil and lattice, `threshold` the displacement
    /// energy in eV. Masses are in neutron masses, as AWRs are.
    pub fn new(z_r: f64, a_r: f64, z_l: f64, a_l: f64, threshold: f64) -> Self {
        if z_r <= 0.0 || a_r <= 0.0 {
            return Lindhard {
                inverse_el: 0.0,
                k: 0.0,
                threshold,
                zero: true,
            };
        }
        let two_thirds = 2.0 / 3.0;
        let el =
            30.724 * z_r * z_l * (z_r.powf(two_thirds) + z_l.powf(two_thirds)).sqrt() * (a_r + a_l)
                / a_l;
        let denom =
            (z_r.powf(two_thirds) + z_l.powf(two_thirds)).powf(0.75) * a_r.powf(1.5) * a_l.sqrt();
        let k = 0.0793 * z_r.powf(two_thirds) * z_l.sqrt() * (a_r + a_l).powf(1.5) / denom;
        Lindhard {
            inverse_el: 1.0 / el,
            k,
            threshold,
            zero: false,
        }
    }

    /// The damage energy of a recoil of energy `e`, eV.
    pub fn damage(&self, e: f64) -> f64 {
        if self.zero || e < self.threshold || e <= 0.0 {
            return 0.0;
        }
        let x = e * self.inverse_el;
        e / (1.0 + self.k * (3.4008 * x.powf(1.0 / 6.0) + 0.40244 * x.powf(0.75) + x))
    }
}

/// Gauss-Legendre nodes and weights on `[-1, 1]`.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// The cached rules: 2 points per tabulated bin, 16 over a smooth angular
/// shape or an analytic spectrum.
fn rule(n: usize) -> &'static (Vec<f64>, Vec<f64>) {
    static TWO: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    static SIXTEEN: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    static SIXTY_FOUR: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    match n {
        2 => TWO.get_or_init(|| gauss_legendre(2)),
        16 => SIXTEEN.get_or_init(|| gauss_legendre(16)),
        _ => SIXTY_FOUR.get_or_init(|| gauss_legendre(64)),
    }
}

/// A tabulated density in one variable: `p` at `x`, histogram or linear
/// between points.
#[derive(Clone, Copy)]
struct Density<'a> {
    x: &'a [f64],
    p: &'a [f64],
    linlin: bool,
}

impl Density<'_> {
    /// `∫ f p` over the table, divided by `∫ p`, each bin by a two-point rule
    /// (exact for a linear `f` against a linear `p`).
    fn expect(&self, mut f: impl FnMut(f64) -> f64) -> f64 {
        let (gx, gw) = rule(2);
        let (mut num, mut area) = (0.0, 0.0);
        for k in 0..self.x.len().saturating_sub(1) {
            let (a, b) = (self.x[k], self.x[k + 1]);
            if b <= a {
                continue;
            }
            let (pa, pb) = (
                self.p[k],
                if self.linlin {
                    self.p[k + 1]
                } else {
                    self.p[k]
                },
            );
            let half = 0.5 * (b - a);
            for (t, w) in gx.iter().zip(gw) {
                let x = a + half * (t + 1.0);
                let p = pa + (pb - pa) * (x - a) / (b - a);
                num += half * w * p * f(x);
                area += half * w * p;
            }
        }
        if area > 0.0 {
            num / area
        } else {
            0.0
        }
    }
}

/// An outgoing particle's cosine distribution at one node.
#[derive(Clone, Copy)]
enum Angle<'a> {
    Isotropic,
    Table(Density<'a>),
    /// Kalbach-Mann: `p(μ) = a / (2 sinh a) (cosh aμ + r sinh aμ)`.
    Kalbach {
        r: f64,
        a: f64,
    },
}

impl Angle<'_> {
    fn of_tabulated(t: &Tabulated) -> Angle<'_> {
        if t.x.len() < 2 {
            return Angle::Isotropic;
        }
        Angle::Table(Density {
            x: &t.x,
            p: &t.p,
            linlin: t.interp == TabulatedInterp::LinLin,
        })
    }

    fn of_correlated(t: &secondary_correlated::Tabular) -> Angle<'_> {
        if t.x.len() < 2 {
            return Angle::Isotropic;
        }
        Angle::Table(Density {
            x: &t.x,
            p: &t.p,
            linlin: t.interpolation == secondary_correlated::Interpolation::LinLin,
        })
    }

    fn mean(&self) -> f64 {
        match self {
            Angle::Isotropic => 0.0,
            Angle::Table(d) => d.expect(|mu| mu),
            Angle::Kalbach { r, a } => {
                if a.abs() < 1e-3 {
                    r * a / 3.0
                } else {
                    r * (1.0 / a.tanh() - 1.0 / a)
                }
            }
        }
    }

    /// `∫ f(μ) p(μ) dμ`.
    fn expect(&self, f: impl Fn(f64) -> f64) -> f64 {
        match self {
            Angle::Isotropic => {
                let (gx, gw) = rule(16);
                gx.iter().zip(gw).map(|(x, w)| 0.5 * w * f(*x)).sum()
            }
            Angle::Table(d) => d.expect(f),
            Angle::Kalbach { r, a } => {
                let (gx, gw) = rule(16);
                if a.abs() < 1e-6 {
                    return gx.iter().zip(gw).map(|(x, w)| 0.5 * w * f(*x)).sum();
                }
                let norm = a / (2.0 * a.sinh());
                gx.iter()
                    .zip(gw)
                    .map(|(x, w)| w * norm * ((a * x).cosh() + r * (a * x).sinh()) * f(*x))
                    .sum()
            }
        }
    }
}

/// One quadrature node of an outgoing distribution at one incident energy:
/// a weight, an outgoing energy and the cosine distribution there.
struct Node<'a> {
    w: f64,
    e: f64,
    angle: Angle<'a>,
}

/// The incident-energy bracket `(i, r)` of `e` on `grid`, clamped to its
/// ends, as the samplers find it.
fn bracket(grid: &[f64], e: f64) -> (usize, f64) {
    let n = grid.len();
    if n < 2 || e <= grid[0] {
        return (0, 0.0);
    }
    if e >= grid[n - 1] {
        return (n - 2, 1.0);
    }
    let i = grid
        .partition_point(|&x| x <= e)
        .saturating_sub(1)
        .min(n - 2);
    let r = if grid[i + 1] > grid[i] {
        (e - grid[i]) / (grid[i + 1] - grid[i])
    } else {
        0.0
    };
    (i, r)
}

/// One incident energy's outgoing table, as the three tabulated laws hold
/// it: discrete lines first, then a continuous density.
struct OutTable<'a> {
    x: &'a [f64],
    p: &'a [f64],
    c: &'a [f64],
    linlin: bool,
    n_discrete: usize,
}

impl OutTable<'_> {
    fn first_continuous(&self) -> f64 {
        self.x
            .get(self.n_discrete)
            .or(self.x.first())
            .copied()
            .unwrap_or(0.0)
    }

    fn last(&self) -> f64 {
        self.x.last().copied().unwrap_or(0.0)
    }

    /// The table's nodes before the incident-energy scaling: `(weight, x,
    /// bin, whether discrete, the CDF at x)`, weights summing to one.
    fn nodes(&self) -> Vec<(f64, f64, usize, bool, f64)> {
        let n = self.x.len();
        let mut out = Vec::new();
        if n == 0 {
            return out;
        }
        if n == 1 {
            out.push((1.0, self.x[0], 0, self.n_discrete > 0, 1.0));
            return out;
        }
        let total = self.c.last().copied().filter(|t| *t > 0.0).unwrap_or(1.0);
        let mut prev = 0.0;
        for j in 0..self.n_discrete.min(n) {
            let c = self.c.get(j).copied().unwrap_or(prev);
            let w = (c - prev).max(0.0);
            if w > 0.0 {
                out.push((w / total, self.x[j], j, true, c));
            }
            prev = c;
        }
        let (gx, gw) = rule(2);
        for k in self.n_discrete..n - 1 {
            let (a, b) = (self.x[k], self.x[k + 1]);
            let mass =
                self.c.get(k + 1).copied().unwrap_or(0.0) - self.c.get(k).copied().unwrap_or(0.0);
            if b <= a || mass <= 0.0 {
                continue;
            }
            let (pa, pb) = (
                self.p[k],
                if self.linlin {
                    self.p[k + 1]
                } else {
                    self.p[k]
                },
            );
            let half = 0.5 * (b - a);
            let nodes: Vec<(f64, f64)> = gx
                .iter()
                .zip(gw)
                .map(|(t, w)| {
                    let x = a + half * (t + 1.0);
                    (
                        x,
                        (half * w * (pa + (pb - pa) * (x - a) / (b - a))).max(0.0),
                    )
                })
                .collect();
            let sum: f64 = nodes.iter().map(|n| n.1).sum();
            for (x, w) in nodes {
                let share = if sum > 0.0 { w / sum } else { 0.5 };
                let dx = x - a;
                let cdf = self.c[k] + pa * dx + 0.5 * (pb - pa) / (b - a) * dx * dx;
                out.push((mass * share / total, x, k, false, cdf));
            }
        }
        out
    }
}

/// Nodes of a tabulated law at `e`: tables `i` and `i + 1` weighted `1 - r`
/// and `r` (only `i` under histogram interpolation), each continuous node
/// scaled by unit-base interpolation of the tables' energy bounds, as the
/// samplers place them. `angle(l, k, x, cdf)` is the cosine distribution of
/// table `l` at its bin `k`.
fn tabulated_nodes<'a>(
    grid: &[f64],
    tables: &[OutTable<'a>],
    e: f64,
    histogram: bool,
    angle: impl Fn(usize, usize, f64, f64, bool) -> Angle<'a>,
) -> Vec<Node<'a>> {
    if tables.is_empty() || grid.is_empty() {
        return Vec::new();
    }
    if tables.len() == 1 || grid.len() == 1 {
        return tables[0]
            .nodes()
            .into_iter()
            .map(|(w, x, k, d, c)| Node {
                w,
                e: x,
                angle: angle(0, k, x, c, d),
            })
            .collect();
    }
    let (i, r) = bracket(grid, e);
    let i = i.min(tables.len() - 2);
    let (lo, hi) = (&tables[i], &tables[i + 1]);
    let e_1 = lo.first_continuous() + r * (hi.first_continuous() - lo.first_continuous());
    let e_k = lo.last() + r * (hi.last() - lo.last());
    let picks: Vec<(usize, f64)> = if histogram {
        vec![(i, 1.0)]
    } else {
        vec![(i, 1.0 - r), (i + 1, r)]
    };
    let mut out = Vec::new();
    for (l, wl) in picks {
        if wl <= 0.0 {
            continue;
        }
        let t = &tables[l];
        let (l1, lk) = (t.first_continuous(), t.last());
        for (w, x, k, discrete, cdf) in t.nodes() {
            let scaled = if histogram || discrete || lk <= l1 || e_k <= e_1 {
                x
            } else {
                e_1 + (x - l1) * (e_k - e_1) / (lk - l1)
            };
            out.push(Node {
                w: wl * w,
                e: scaled,
                angle: angle(l, k, x, cdf, discrete),
            });
        }
    }
    out
}

/// Nodes of an analytic spectrum `pdf` on `[0, top]`, isotropic.
fn analytic_nodes<'a>(top: f64, pdf: impl Fn(f64) -> f64) -> Vec<Node<'a>> {
    if top <= 0.0 {
        return Vec::new();
    }
    let (gx, gw) = rule(64);
    let half = 0.5 * top;
    let raw: Vec<(f64, f64)> = gx
        .iter()
        .zip(gw)
        .map(|(t, w)| {
            let x = half * (t + 1.0);
            (x, (half * w * pdf(x)).max(0.0))
        })
        .collect();
    let sum: f64 = raw.iter().map(|r| r.1).sum();
    if sum <= 0.0 {
        return Vec::new();
    }
    raw.into_iter()
        .map(|(x, w)| Node {
            w: w / sum,
            e: x,
            angle: Angle::Isotropic,
        })
        .collect()
}

/// Nodes of an energy law at incident energy `e`, each with `angle`.
fn energy_nodes(dist: &EnergyDistribution, e: f64) -> Vec<(f64, f64)> {
    match dist {
        EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        } => {
            if e > *threshold {
                vec![(1.0, mass_ratio * (e - threshold))]
            } else {
                Vec::new()
            }
        }
        EnergyDistribution::Tabulated { energy, energy_out } => {
            if energy.is_empty() || energy_out.is_empty() {
                return Vec::new();
            }
            let i = energy
                .partition_point(|&x| x <= e)
                .saturating_sub(1)
                .min(energy_out.len() - 1);
            let row = &energy_out[i];
            let w = 1.0 / row.len().max(1) as f64;
            row.iter().map(|x| (w, *x)).collect()
        }
        EnergyDistribution::ContinuousTabular {
            energy,
            energy_out,
            histogram_interp,
        } => {
            let tables: Vec<OutTable> = energy_out
                .iter()
                .map(|t| match t {
                    TabulatedProbability::Tabulated {
                        x,
                        p,
                        c,
                        interp,
                        n_discrete,
                    } => OutTable {
                        x,
                        p,
                        c,
                        linlin: *interp == TabulatedInterp::LinLin,
                        n_discrete: *n_discrete,
                    },
                })
                .collect();
            tabulated_nodes(energy, &tables, e, *histogram_interp, |_, _, _, _, _| {
                Angle::Isotropic
            })
            .into_iter()
            .map(|n| (n.w, n.e))
            .collect()
        }
        EnergyDistribution::Maxwell { theta, u } => {
            let t = theta.evaluate(e);
            nodes_xy(analytic_nodes(spectrum_top(e - u, t), |x| {
                x.sqrt() * (-x / t).exp()
            }))
        }
        EnergyDistribution::Watt { a, b, u } => {
            let (a, b) = (a.evaluate(e), b.evaluate(e));
            nodes_xy(analytic_nodes(spectrum_top(e - u, a), |x| {
                (-x / a).exp() * (b * x).sqrt().sinh()
            }))
        }
        EnergyDistribution::Evaporation { theta, u } => {
            let t = theta.evaluate(e);
            nodes_xy(analytic_nodes(spectrum_top(e - u, t), |x| {
                x * (-x / t).exp()
            }))
        }
        EnergyDistribution::DiscretePhoton {
            primary_flag,
            energy,
            atomic_weight_ratio,
        } => {
            let x = if *primary_flag == 2 {
                energy + atomic_weight_ratio / (atomic_weight_ratio + 1.0) * e
            } else {
                *energy
            };
            vec![(1.0, x)]
        }
    }
}

fn nodes_xy(nodes: Vec<Node>) -> Vec<(f64, f64)> {
    nodes.into_iter().map(|n| (n.w, n.e)).collect()
}

/// The upper end of an exponential spectrum's quadrature: its restriction
/// `limit`, or 40 temperatures, whichever is lower.
fn spectrum_top(limit: f64, temperature: f64) -> f64 {
    if temperature > 0.0 {
        limit.min(40.0 * temperature)
    } else {
        limit
    }
}

/// The cosine distribution of `angle` at `e` as a mixture of its two
/// bracketing tables.
fn angle_mixture(angle: &AngleDistribution, e: f64) -> Vec<(f64, Angle<'_>)> {
    if angle.energy.is_empty() || angle.mu.is_empty() {
        return vec![(1.0, Angle::Isotropic)];
    }
    if angle.mu.len() == 1 || angle.energy.len() == 1 {
        return vec![(1.0, Angle::of_tabulated(&angle.mu[0]))];
    }
    let (i, r) = bracket(&angle.energy, e);
    let i = i.min(angle.mu.len() - 2);
    vec![
        (1.0 - r, Angle::of_tabulated(&angle.mu[i])),
        (r, Angle::of_tabulated(&angle.mu[i + 1])),
    ]
}

fn kalbach_nodes(km: &KalbachMann, e: f64) -> Vec<Node<'_>> {
    let tables: Vec<OutTable> = km
        .distributions
        .iter()
        .map(|t: &KMTable| OutTable {
            x: &t.e_out,
            p: &t.p,
            c: &t.c,
            linlin: matches!(t.interpolation, secondary_kalbach::Interpolation::LinLin),
            n_discrete: t.n_discrete,
        })
        .collect();
    tabulated_nodes(&km.energy, &tables, e, false, |l, k, x, _, _| {
        let t = &km.distributions[l];
        let (r, a) = match t.interpolation {
            secondary_kalbach::Interpolation::LinLin if k + 1 < t.e_out.len() => {
                let span = t.e_out[k + 1] - t.e_out[k];
                let f = if span > 0.0 {
                    (x - t.e_out[k]) / span
                } else {
                    0.0
                };
                (
                    t.r[k] + f * (t.r[k + 1] - t.r[k]),
                    t.a[k] + f * (t.a[k + 1] - t.a[k]),
                )
            }
            _ => (t.r[k], t.a[k]),
        };
        Angle::Kalbach { r, a }
    })
}

fn correlated_nodes(corr: &CorrelatedAngleEnergy, e: f64) -> Vec<Node<'_>> {
    let tables: Vec<OutTable> = corr
        .distributions
        .iter()
        .map(|t: &CorrTable| OutTable {
            x: &t.e_out,
            p: &t.p,
            c: &t.c,
            linlin: t.interpolation == secondary_correlated::Interpolation::LinLin,
            n_discrete: t.n_discrete,
        })
        .collect();
    tabulated_nodes(&corr.energy, &tables, e, false, |l, k, _, cdf, discrete| {
        let t = &corr.distributions[l];
        if t.angle.is_empty() {
            return Angle::Isotropic;
        }
        // The sampler reads the bin's lower table in its lower half by
        // probability and the upper one above it.
        let upper = !discrete
            && t.interpolation == secondary_correlated::Interpolation::LinLin
            && k + 1 < t.c.len()
            && cdf - t.c[k] >= t.c[k + 1] - cdf;
        let j = if upper { k + 1 } else { k }.min(t.angle.len() - 1);
        Angle::of_correlated(&t.angle[j])
    })
}

/// The nodes of a product's distribution `d` at incident energy `e`, for a
/// target of mass `awr`.
fn distribution_nodes(d: &AngleEnergyDistribution, e: f64, awr: f64) -> Vec<Node<'_>> {
    match d {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, energy } => {
            let energies = match energy {
                Some(dist) => energy_nodes(dist, e),
                // Elastic: two-body kinematics with no Q, so the outgoing CM
                // energy is fixed.
                None => vec![(1.0, e * (awr / (awr + 1.0)).powi(2))],
            };
            let angles = angle_mixture(angle, e);
            let mut out = Vec::with_capacity(energies.len() * angles.len());
            for (we, x) in energies {
                for (wa, a) in &angles {
                    if *wa > 0.0 {
                        out.push(Node {
                            w: we * wa,
                            e: x,
                            angle: *a,
                        });
                    }
                }
            }
            out
        }
        AngleEnergyDistribution::KalbachMann { kalbach } => kalbach_nodes(kalbach, e),
        AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } => {
            correlated_nodes(correlated, e)
        }
        AngleEnergyDistribution::Evaporation { theta, u } => {
            let t = theta.as_ref().map_or(1.0, |t| t.evaluate(e));
            analytic_nodes(spectrum_top(e - u, t), |x| x * (-x / t).exp())
        }
        AngleEnergyDistribution::NBodyPhaseSpace {
            n_bodies,
            total_mass,
            awr,
            q_value,
        } => {
            let ap = *total_mass;
            let top = (ap - 1.0) / ap * (awr / (awr + 1.0) * e + q_value);
            let power = 1.5 * *n_bodies as f64 - 4.0;
            analytic_nodes(top, |x| x.sqrt() * (top - x).max(0.0).powf(power))
        }
    }
}

/// The nodes of every distribution of `product` at `e`, each weighted by its
/// applicability there.
fn product_nodes(product: &ReactionProduct, e: f64, awr: f64) -> Vec<Node<'_>> {
    let n = product.distribution.len();
    let mut out = Vec::new();
    for (j, d) in product.distribution.iter().enumerate() {
        let p = if n > 1 && product.applicability.len() == n {
            product.applicability[j].evaluate(e)
        } else {
            1.0
        };
        if p <= 0.0 {
            continue;
        }
        for mut node in distribution_nodes(d, e, awr) {
            node.w *= p;
            out.push(node);
        }
    }
    out
}

fn product_yield(product: &ReactionProduct, e: f64) -> f64 {
    product
        .product_yield
        .as_ref()
        .map_or(1.0, |y| y.evaluate(e))
}

/// The mean lab energy of an outgoing particle of unit mass over `nodes`,
/// from a target of mass `awr` hit at `e`, the nodes in the CM frame when
/// `cm`.
fn mean_lab_energy(nodes: &[Node], e: f64, awr: f64, cm: bool) -> f64 {
    let a1 = awr + 1.0;
    nodes
        .iter()
        .map(|n| {
            if cm {
                n.w * (n.e + e / (a1 * a1) + 2.0 * (e * n.e).sqrt() * n.angle.mean() / a1)
            } else {
                n.w * n.e
            }
        })
        .sum()
}

/// The mean lab energy of one particle of `product` (of unit mass) emitted
/// at incident energy `e` from a target of mass `awr`, its distribution in
/// the centre-of-mass frame when `cm`: what the transport's sampler gives on
/// average, before the product's yield.
pub fn product_mean_energy(product: &ReactionProduct, e: f64, awr: f64, cm: bool) -> f64 {
    mean_lab_energy(&product_nodes(product, e, awr), e, awr, cm)
}

/// `(Σ y Ē_γ, Σ y <E_γ²>)` over the photon products of `reaction` at `e`.
fn photon_moments(reaction: &Reaction, e: f64, awr: f64) -> (f64, f64) {
    let (mut first, mut second) = (0.0, 0.0);
    for product in reaction.products.iter() {
        if product.particle != ParticleType::Photon {
            continue;
        }
        let y = product_yield(product, e);
        if y == 0.0 {
            continue;
        }
        for n in product_nodes(product, e, awr) {
            first += y * n.w * n.e;
            second += y * n.w * n.e * n.e;
        }
    }
    (first, second)
}

/// How a reaction deposits its energy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Elastic,
    Capture,
    /// A reaction whose first product is a neutron.
    NeutronEmitting,
    /// A discrete level of a charged-particle reaction: the particle's index
    /// in [`PARTICLE_MASS`].
    ChargedLevel(usize),
    /// Continuum charged-particle emission, `(z, a)` of the particle HEATR
    /// assumes.
    ChargedContinuum(f64, f64),
}

/// How `mt` deposits, or `None` for a reaction this does not split.
/// `has_levels(mt)` says whether the nuclide holds any discrete level of the
/// charged-particle sum `mt` (MT 103 to 107).
fn kind_of(mt: i32, reaction: &Reaction, has_levels: impl Fn(i32) -> bool) -> Option<Kind> {
    let family = |first: i32| -> Option<usize> {
        [600, 650, 700, 750, 800].iter().position(|&f| f == first)
    };
    match mt {
        2 => Some(Kind::Elastic),
        102 => Some(Kind::Capture),
        5 | 18 | 19 | 20 | 21 | 38 => None,
        600..=848 if (mt - 600) % 50 < 49 => {
            Some(Kind::ChargedLevel(family(600 + 50 * ((mt - 600) / 50))?))
        }
        649 | 699 | 749 | 799 | 849 => {
            let i = family(mt - 49)?;
            Some(Kind::ChargedContinuum(PARTICLE_CHARGE[i], PARTICLE_MASS[i]))
        }
        _ if reaction
            .products
            .iter()
            .any(|p| p.particle == ParticleType::Neutron) =>
        {
            Some(Kind::NeutronEmitting)
        }
        // A charged-particle reaction stated without its levels: a two-body
        // breakup to the ground state (Li6 (n,t), He3 (n,p)).
        103..=107 if !has_levels(mt) => Some(Kind::ChargedLevel((mt - 103) as usize)),
        103..=107 => {
            let i = (mt - 103) as usize;
            Some(Kind::ChargedContinuum(
                PARTICLE_CHARGE[i],
                PARTICLE_MASS[i].round(),
            ))
        }
        _ if mt > 102 && mt < 200 => Some(Kind::ChargedContinuum(1.0, 1.0)),
        _ => None,
    }
}

/// The ground-state (mass-difference) Q value of `mt` among `held`: zero for
/// scattering, the family's ground-state level's for a charged-particle
/// level or continuum, the reaction's own otherwise (the stored Q values are
/// to the level a reaction leaves).
fn ground_state_q(mt: i32, held: &dyn Fn(i32) -> Option<f64>, own: f64) -> f64 {
    match mt {
        2 | 4 | 50..=91 => 0.0,
        600..=849 => {
            let ground = 600 + 50 * ((mt - 600) / 50);
            held(ground).unwrap_or(own)
        }
        _ => own,
    }
}

/// Everything one reaction's deposit reads.
struct ReactionDeposit<'a> {
    kind: Kind,
    reaction: &'a Reaction,
    awr: f64,
    z: f64,
    /// The ground-state Q value.
    q: f64,
    /// The stored Q value, to the level the reaction leaves.
    q_level: f64,
    lattice: Lindhard,
    /// The first neutron product, as the transport emits it.
    neutron: Option<&'a ReactionProduct>,
}

impl ReactionDeposit<'_> {
    /// Whether the neutron's distribution is in the centre-of-mass frame:
    /// elastic's always is.
    fn cm(&self) -> bool {
        self.reaction.scatter_in_cm || self.kind == Kind::Elastic
    }

    /// The deposit per reaction at `e`, eV.
    fn at(&self, response: Response, e: f64) -> f64 {
        match response {
            Response::Heating | Response::HeatingLocal => self.heating(response, e),
            Response::DamageEnergy => self.damage(e),
        }
    }

    fn heating(&self, response: Response, e: f64) -> f64 {
        let a = self.awr;
        let photons = || photon_moments(self.reaction, e, a);
        if self.kind == Kind::Capture {
            let (first, second) = photons();
            let recoil = e / (a + 1.0) + second / (2.0 * (a + 1.0) * NEUTRON_MASS_EV);
            if response == Response::Heating {
                return recoil;
            }
            if first > 0.0 {
                return recoil + first;
            }
        }
        let mut h = e + self.q;
        if let Some(n) = self.neutron {
            let nodes = product_nodes(n, e, a);
            h -= product_yield(n, e) * mean_lab_energy(&nodes, e, a, self.cm());
        }
        if response == Response::Heating {
            h -= photons().0;
        }
        h
    }

    fn damage(&self, e: f64) -> f64 {
        let a = self.awr;
        let a1 = a + 1.0;
        match self.kind {
            Kind::Elastic | Kind::NeutronEmitting => {
                let Some(n) = self.neutron else {
                    return 0.0;
                };
                let cm = self.cm();
                let nodes = product_nodes(n, e, a);
                let total: f64 = nodes.iter().map(|n| n.w).sum();
                if total <= 0.0 {
                    return 0.0;
                }
                // One-particle recoil: the residual, of mass A, takes the
                // momentum the neutron leaves behind.
                let d: f64 = nodes
                    .iter()
                    .map(|node| {
                        let x = node.e;
                        let recoil = |mu: f64| {
                            let er = if cm {
                                (e * (a / a1).powi(2) + x - 2.0 * (e * x).sqrt() * mu * a / a1) / a
                            } else {
                                (e + x - 2.0 * (e * x).sqrt() * mu) / a
                            };
                            self.lattice.damage(er)
                        };
                        node.w * node.angle.expect(recoil)
                    })
                    .sum();
                d / total
            }
            Kind::Capture => {
                // Each photon's kick, as HEATR takes it from the photon data:
                // the neutron's momentum is left out there.
                let kick = Lindhard::new(self.z, a1, self.z, a, self.lattice.threshold);
                let mut from_photons = 0.0;
                let mut any = false;
                for product in self.reaction.products.iter() {
                    if product.particle != ParticleType::Photon {
                        continue;
                    }
                    any = true;
                    let y = product_yield(product, e);
                    for n in product_nodes(product, e, a) {
                        from_photons +=
                            y * n.w * kick.damage(n.e * n.e / (2.0 * a1 * NEUTRON_MASS_EV));
                    }
                }
                if any {
                    return from_photons;
                }
                let available = a * e / a1 + self.q;
                let er = e / a1 + available * available / (2.0 * a1 * NEUTRON_MASS_EV);
                Lindhard::new(self.z, a1, self.z, a, self.lattice.threshold).damage(er)
            }
            Kind::ChargedLevel(i) => {
                let m = PARTICLE_MASS[i];
                let residual = a1 - m;
                let threshold = -self.q_level * a1 / a;
                if e <= threshold.max(0.0) || residual <= 0.0 {
                    return 0.0;
                }
                let g = (m * a * (1.0 - threshold / e) / residual).sqrt();
                let scale = residual * e / (a1 * a1);
                // The residual and the light particle both slow down in the
                // lattice, each partitioned as the ion it is.
                let heavy = Lindhard::new(
                    self.z - PARTICLE_CHARGE[i],
                    residual,
                    self.z,
                    a,
                    self.lattice.threshold,
                );
                let recoil = Angle::Isotropic
                    .expect(|mu| heavy.damage(scale * (1.0 - 2.0 * g * mu + g * g)));
                let light = Lindhard::new(PARTICLE_CHARGE[i], m, self.z, a, self.lattice.threshold);
                let cm_motion = m * e / (a1 * a1);
                let own = residual / (m + residual) * (a * e / a1 + self.q_level);
                recoil
                    + Angle::Isotropic.expect(|mu| {
                        light.damage(cm_motion + own + 2.0 * (cm_motion * own).max(0.0).sqrt() * mu)
                    })
            }
            Kind::ChargedContinuum(zx, ax) => {
                let coulomb = 1.029e6 * zx * self.z / (ax.cbrt() + a.cbrt());
                let mut available = self.q_level + a * e / a1;
                if available < 0.0 {
                    return 0.0;
                }
                available = available.min(coulomb);
                let et = (a1 - ax) * e / a1;
                let recoil = Lindhard::new(self.z - zx, a1 - ax, self.z, a, self.lattice.threshold);
                let (gx, gw) = rule(16);
                gx.iter()
                    .zip(gw)
                    .map(|(mu, w)| {
                        let er =
                            (et - 2.0 * (et * ax * available).sqrt() * mu + ax * available) / a1;
                        0.5 * w * recoil.damage(er)
                    })
                    .sum()
            }
        }
    }
}

/// The incident energies to evaluate a deposit at from `lo` to `hi`: a log
/// grid, both ends and the incident grid of `neutron`'s distributions, where
/// the tables it interpolates between change.
fn incident_points(lo: f64, hi: f64, neutron: Option<&ReactionProduct>) -> Vec<f64> {
    let lo = lo.max(1e-5);
    let mut out = vec![lo, hi.max(lo)];
    if hi > lo {
        let decades = (hi / lo).log10();
        let n = (decades * POINTS_PER_DECADE).ceil() as usize;
        for k in 1..n {
            out.push(lo * 10f64.powf(decades * k as f64 / n as f64));
        }
    }
    if let Some(neutron) = neutron {
        for d in &neutron.distribution {
            let grid: &[f64] = match d {
                AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } => &angle.energy,
                AngleEnergyDistribution::KalbachMann { kalbach } => &kalbach.energy,
                AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } => &correlated.energy,
                _ => &[],
            };
            out.extend(grid.iter().copied().filter(|&x| x > lo && x < hi));
        }
    }
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

/// The deposit of every reaction in `mts` that this splits (see the module
/// documentation), per reaction, eV, on the nuclide's grid at temperature
/// index `t`: zero below each reaction's threshold. Photon production lumped
/// on a sum is subtracted from each of the sum's partials in `mts` under
/// [`Response::Heating`].
///
/// MT 5, the reactions an evaluation lumps (above 20 MeV it is usually the
/// whole non-elastic), has no Q value of its own to balance and no
/// charged-particle spectra in the data, so it takes the part of the stored
/// row the other partials leave, wherever its cross section is not zero, as
/// the particle-production scores attribute their residual to it: it is the
/// only candidate there. Not for a fissile nuclide, whose residual is
/// fission's.
pub fn deposits(
    nuclide: &Nuclide,
    t: usize,
    response: Response,
    mts: &BTreeSet<i32>,
) -> BTreeMap<i32, Vec<f64>> {
    let reactions = match nuclide.reactions.get(t) {
        Some(r) => r,
        None => return BTreeMap::new(),
    };
    let lumped = reactions.get(&5).filter(|r| !r.redundant);
    let fissile = reactions
        .iter()
        .any(|(mt, r)| matches!(mt, 18 | 19 | 20 | 21 | 38) && !r.redundant);
    let (Some(r5), false, true) = (lumped, fissile, mts.contains(&5)) else {
        return split(nuclide, t, response, mts);
    };
    let Some(stored) = stored_row(nuclide, t, response) else {
        return split(nuclide, t, response, mts);
    };
    let every: BTreeSet<i32> = reactions
        .iter()
        .filter(|(_, r)| !r.redundant)
        .map(|(mt, _)| *mt)
        .collect();
    let mut all = split(nuclide, t, response, &every);
    let n = stored.len();
    let mut residual = stored;
    for (mt, h) in &all {
        let r = &reactions[mt];
        for (j, x) in r.cross_section.as_slice().iter().enumerate() {
            if let Some(v) = residual.get_mut(r.threshold_idx + j) {
                *v -= x * h[r.threshold_idx + j];
            }
        }
    }
    // No reaction deposits more than comes in, `E + Q`: a residual that would
    // need more is not MT 5's, and what is over stays unexplained.
    let grid = &nuclide.energy.as_ref().expect("grid")[&nuclide.loaded_temperatures[t]];
    let mut h5 = vec![0.0; n];
    for (j, x) in r5.cross_section.as_slice().iter().enumerate() {
        let i = r5.threshold_idx + j;
        if i < n && *x > 0.0 {
            let most = (grid[i] + r5.q_value.max(0.0)).max(0.0);
            h5[i] = (residual[i].max(0.0) / x).min(most);
        }
    }
    all.insert(5, h5);
    all.retain(|mt, _| mts.contains(mt));
    all
}

/// [`deposits`] of the reactions this splits, before MT 5's share.
fn split(
    nuclide: &Nuclide,
    t: usize,
    response: Response,
    mts: &BTreeSet<i32>,
) -> BTreeMap<i32, Vec<f64>> {
    let mut out = BTreeMap::new();
    let Some(label) = nuclide.loaded_temperatures.get(t) else {
        return out;
    };
    let Some(grid) = nuclide.energy.as_ref().and_then(|m| m.get(label)) else {
        return out;
    };
    let (Some(awr), Some(reactions)) = (nuclide.atomic_weight_ratio, nuclide.reactions.get(t))
    else {
        return out;
    };
    let grid = grid.as_slice();
    let n = grid.len();
    if n == 0 {
        return out;
    }
    let z = nuclide.atomic_number.unwrap_or(0) as f64;
    let threshold = heatr_displacement_threshold(z as u32);
    let lattice = Lindhard::new(z, awr, z, awr, threshold);
    let q_of = |mt: i32| reactions.get(&mt).map(|r| r.q_value);
    for &mt in mts {
        let Some(reaction) = reactions.get(&mt).filter(|r| !r.redundant) else {
            continue;
        };
        let has_levels = |sum: i32| {
            let first = 600 + 50 * (sum - 103);
            (first..first + 49).any(|l| reactions.contains_key(&l))
        };
        let Some(kind) = kind_of(mt, reaction, has_levels) else {
            continue;
        };
        let start = reaction.threshold_idx.min(n - 1);
        let deposit = ReactionDeposit {
            kind,
            reaction,
            awr,
            z,
            q: ground_state_q(mt, &q_of, reaction.q_value),
            q_level: reaction.q_value,
            lattice,
            neutron: reaction
                .products
                .iter()
                .find(|p| p.particle == ParticleType::Neutron),
        };
        let points = incident_points(grid[start], grid[n - 1], deposit.neutron);
        out.insert(
            mt,
            on_grid(grid, start, &points, |e| deposit.at(response, e)),
        );
    }
    if response == Response::Heating {
        subtract_lumped_photons(nuclide, t, grid, awr, &mut out);
    }
    out
}

/// `f` from grid index `start` up, evaluated at the incident energies `xs`
/// (ascending) and interpolated linearly onto the grid; zero below `start`.
fn on_grid(grid: &[f64], start: usize, xs: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    let n = grid.len();
    let mut out = vec![0.0; n];
    let ys: Vec<f64> = xs.iter().map(|&e| f(e)).collect();
    let mut j = 0;
    for i in start..n {
        let e = grid[i];
        while j + 2 < xs.len() && xs[j + 1] < e {
            j += 1;
        }
        out[i] = if xs.len() == 1 {
            ys[0]
        } else {
            let (x0, x1) = (xs[j], xs[j + 1]);
            if x1 > x0 {
                let f = ((e - x0) / (x1 - x0)).clamp(0.0, 1.0);
                ys[j] + f * (ys[j + 1] - ys[j])
            } else {
                ys[j]
            }
        };
    }
    out
}

/// Subtract the photons a sum's row carries (an evaluation that gives the
/// photon production of MT 3, MT 4 or MT 103 to 107 lumped rather than per
/// level) from each of the sum's partials in `out`: per reaction of the sum
/// alike, as the photon production itself is read.
fn subtract_lumped_photons(
    nuclide: &Nuclide,
    t: usize,
    grid: &[f64],
    awr: f64,
    out: &mut BTreeMap<i32, Vec<f64>>,
) {
    let reactions = &nuclide.reactions[t];
    for (&sum, reaction) in reactions {
        // A partial's own photons are in its own deposit already.
        if !reaction.redundant
            || !reaction
                .products
                .iter()
                .any(|p| p.particle == ParticleType::Photon)
        {
            continue;
        }
        let partials = partials_of(sum, reactions);
        if partials.iter().all(|p| !out.contains_key(p)) {
            continue;
        }
        let start = reaction.threshold_idx.min(grid.len() - 1);
        let points = incident_points(grid[start], grid[grid.len() - 1], None);
        let energy = on_grid(grid, start, &points, |e| photon_moments(reaction, e, awr).0);
        for p in partials {
            if let Some(h) = out.get_mut(&p) {
                for (v, g) in h.iter_mut().zip(&energy) {
                    *v -= g;
                }
            }
        }
    }
}

/// The non-redundant partials a sum `mt` adds up, among `reactions`.
fn partials_of(
    mt: i32,
    reactions: &std::collections::HashMap<i32, std::sync::Arc<Reaction>>,
) -> BTreeSet<i32> {
    let mut out = BTreeSet::new();
    if let Some(rule) = endf::data::sum_rule(mt) {
        for &c in rule {
            match reactions.get(&c) {
                Some(r) if !r.redundant => {
                    out.insert(c);
                }
                _ => out.extend(partials_of(c, reactions)),
            }
        }
    }
    out
}

/// The stored row of `response` on the nuclide's grid at temperature index
/// `t`, zero below its threshold, or `None` where the nuclide has none.
pub fn stored_row(nuclide: &Nuclide, t: usize, response: Response) -> Option<Vec<f64>> {
    let label = nuclide.loaded_temperatures.get(t)?;
    let n = nuclide.energy.as_ref()?.get(label)?.len();
    let row = nuclide.reactions.get(t)?.get(&response.mt())?;
    let mut out = vec![0.0; n];
    let xs = row.cross_section.as_slice();
    let start = row.threshold_idx.min(n);
    let len = xs.len().min(n - start);
    out[start..start + len].copy_from_slice(&xs[..len]);
    Some(out)
}

/// The energy bands [`unexplained_share`] is reported in, eV: thermal and
/// epithermal, the resolved resonances, the fast range below 1 MeV, the
/// fast range up to 20 MeV, and above.
pub const BANDS: [(f64, f64); 5] = [
    (0.0, 1.0),
    (1.0, 1e3),
    (1e3, 1e6),
    (1e6, 2e7),
    (2e7, f64::INFINITY),
];

/// How much of a stored row the partials leave out over `[lo, hi)`:
/// `∫ |K - Σ σ_mt h_mt| du / ∫ |K| du` over lethargy on the nuclide's grid,
/// `deposits` the partials' deposits ([`deposits`]) and `K` the stored row.
/// `None` where the nuclide has no such row or it is zero over the band.
pub fn unexplained_share(
    nuclide: &Nuclide,
    t: usize,
    response: Response,
    deposits: &BTreeMap<i32, Vec<f64>>,
    (lo, hi): (f64, f64),
) -> Option<f64> {
    let stored = stored_row(nuclide, t, response)?;
    let label = &nuclide.loaded_temperatures[t];
    let grid = nuclide.energy.as_ref()?.get(label)?.as_slice();
    let reactions = nuclide.reactions.get(t)?;
    let n = grid.len();
    let mut explained = vec![0.0; n];
    for (mt, h) in deposits {
        let Some(r) = reactions.get(mt) else {
            continue;
        };
        let xs = r.cross_section.as_slice();
        for (j, x) in xs.iter().enumerate() {
            let i = r.threshold_idx + j;
            if i < n {
                explained[i] += x * h[i];
            }
        }
    }
    let (mut residual, mut total) = (0.0, 0.0);
    for i in 0..n.saturating_sub(1) {
        if grid[i] <= 0.0 || grid[i + 1] <= grid[i] || grid[i] < lo || grid[i] >= hi {
            continue;
        }
        let du = (grid[i + 1] / grid[i]).ln();
        residual += 0.5
            * du
            * ((stored[i] - explained[i]).abs() + (stored[i + 1] - explained[i + 1]).abs());
        total += 0.5 * du * (stored[i].abs() + stored[i + 1].abs());
    }
    (total > 0.0).then(|| residual / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        let (x, w) = gauss_legendre(16);
        let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(6)).sum();
        assert!((integral - 2.0 / 7.0).abs() < 1e-13);
        let (x, w) = rule(2);
        let integral: f64 = x.iter().zip(w).map(|(x, w)| w * x * x).sum();
        assert!((integral - 2.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn lindhard_matches_heatr_for_iron() {
        // NJOY's df for Fe in Fe at 10 keV and 1 MeV, by hand from its
        // constants: E_L, the Lindhard k and Robinson's g(ε).
        let (z, a): (f64, f64) = (26.0, 55.454);
        let l = Lindhard::new(z, a, z, a, 40.0);
        let el = 30.724 * z * z * (2.0 * z.powf(2.0 / 3.0)).sqrt() * 2.0;
        let k = 0.0793 * z.powf(2.0 / 3.0) * z.sqrt() * (2.0 * a).powf(1.5)
            / ((2.0 * z.powf(2.0 / 3.0)).powf(0.75) * a.powf(1.5) * a.sqrt());
        for e in [1e4, 1e6] {
            let x = e / el;
            let expected =
                e / (1.0 + k * (3.4008 * x.powf(1.0 / 6.0) + 0.40244 * x.powf(0.75) + x));
            assert!((l.damage(e) - expected).abs() < 1e-9 * expected);
        }
        assert_eq!(l.damage(39.0), 0.0);
        // Most of a low-energy recoil goes into atomic motion, little of a
        // fast one.
        assert!(l.damage(1e3) / 1e3 > 0.8);
        assert!(l.damage(1e7) / 1e7 < 0.4);
    }

    #[test]
    fn kalbach_mean_cosine() {
        let a = Angle::Kalbach { r: 0.4, a: 2.0 };
        let direct = a.expect(|mu| mu);
        assert!((direct - a.mean()).abs() < 1e-12, "{direct} {}", a.mean());
    }

    #[test]
    fn elastic_recoil_is_two_body() {
        // Isotropic CM elastic on A: mean recoil 2A/(A+1)^2 E.
        let a: f64 = 55.0;
        let e = 1e6;
        let nodes = vec![Node {
            w: 1.0,
            e: e * (a / (a + 1.0)).powi(2),
            angle: Angle::Isotropic,
        }];
        let lab = mean_lab_energy(&nodes, e, a, true);
        let recoil = e - lab;
        let expected = 2.0 * a / ((a + 1.0) * (a + 1.0)) * e;
        assert!((recoil - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn tabulated_nodes_reproduce_a_flat_mean() {
        let x = [0.0, 1.0, 3.0];
        let p = [0.25, 0.25, 0.25];
        let c = [0.0, 0.25, 0.75];
        let t = OutTable {
            x: &x,
            p: &p,
            c: &c,
            linlin: false,
            n_discrete: 0,
        };
        let nodes = t.nodes();
        let w: f64 = nodes.iter().map(|n| n.0).sum();
        let mean: f64 = nodes.iter().map(|n| n.0 * n.1).sum::<f64>() / w;
        // A flat density on [0, 3] whose CDF ends at 0.75: normalised, mean 1.5.
        assert!((w - 1.0).abs() < 1e-12);
        assert!((mean - 1.5).abs() < 1e-12);
    }
}
