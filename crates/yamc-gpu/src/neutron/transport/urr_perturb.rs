//! Per-(material, nuclide) URR probability-table perturbation of the
//! macroscopic reaction partials, shared by the transport kernel's CPU twin.
//!
//! This is the plain-Rust form of the block in `kernel.rs` (search
//! `urr_any_in_range`), which the cubecl kernel keeps as its own `#[cube]`
//! transcription for the same reason every other sampler does: a `#[cube]`
//! function cannot be called from ordinary CPU code. The two are pinned equal
//! by the `cpu_gpu_equivalence_*` tests.
//!
//! Semantics, all mirroring CPU `Material::compute_urr_macro_xs`:
//!
//! * URR applies to EVERY in-range URR nuclide of the material, each drawing
//!   its own band from the shared per-collision base via
//!   `urr_nuclide_random(base, ZA)` (issue #204), because isotopes' resonance
//!   structures are statistically independent.
//! * The unperturbed partials are the floor; each nuclide contributes a
//!   macroscopic delta against its own `nuc_partial_xs` smooth baseline. The
//!   summed deltas are applied and clamped ONCE, since a single nuclide's
//!   self-shielding delta is legitimately negative.
//! * The band itself is held per ENERGY by the caller, not redrawn per step
//!   (issue #342).
//!
//! Tight CSR (issue #104): the URR buffers carry no `MAX_URR_*` padding, so
//! every bracket search is bounded by the slab's own counts.

use crate::neutron::xs::{URR_META_COLS, URR_XS_COLS};

/// Column indices into a `nuc_partial_xs` row.
const P_ELASTIC: usize = 0;
const P_ABSORPTION: usize = 1;
const P_INELASTIC: usize = 2;
const P_FISSION: usize = 3;

/// The flat URR tables for one translated model, as the kernel binds them.
pub(super) struct UrrTables<'a> {
    /// `[n_slab × URR_META_COLS]`; see the `URR_META_*` constants.
    pub meta: &'a [u32],
    /// Per-slab base into `energy_grid`.
    pub ae_offset: &'a [u32],
    /// Per-slab base into `cdf` (and, times `URR_XS_COLS`, into `xs`).
    pub cdf_offset: &'a [u32],
    pub energy_grid: &'a [f64],
    pub cdf: &'a [f64],
    pub xs: &'a [f64],
    /// Per-slab atom density, used only when the table holds absolute XS.
    pub atom_density: &'a [f64],
}

/// Macroscopic partials, in the kernel's `sigma_*` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Partials {
    pub elastic: f64,
    pub absorption: f64,
    pub inelastic: f64,
    pub fission: f64,
}

/// Outcome of a URR perturbation step.
pub(super) struct UrrPerturbation {
    /// The perturbed partials (unchanged when nothing fired).
    pub partials: Partials,
    /// Whether any URR nuclide was sampled, i.e. whether the tally score loops
    /// should substitute the perturbed macros for the smooth `xs_score_per_mt`.
    pub fired: bool,
    /// Full-material URR-modified macroscopic capture (n,gamma, EXCLUDING
    /// fission). Includes non-URR nuclides' smooth capture, so mixed materials
    /// still score capture correctly (issue #210).
    pub macro_capture: f64,
}

impl<'a> UrrTables<'a> {
    /// Does this material have at least one URR nuclide whose table covers
    /// `energy`? Mirrors CPU `Material::has_urr_in_range`, and gates the base
    /// draw so the RNG advances at most once per energy.
    pub(super) fn any_in_range(&self, nuc_off: usize, nuc_count: usize, energy: f64) -> bool {
        (0..nuc_count).any(|us| {
            let slab = nuc_off + us;
            let mo = slab * URR_META_COLS;
            if self.meta[mo] != 1 {
                return false;
            }
            let n_e = self.meta[mo + 1] as usize;
            if n_e < 2 {
                return false;
            }
            let eg_off = self.ae_offset[slab] as usize;
            let first = self.energy_grid[eg_off];
            let last = self.energy_grid[eg_off + n_e - 1];
            energy > first && energy < last
        })
    }

    /// Interpolate this slab's URR micro cross sections at `energy` in the band
    /// `r_urr` selects, returning `(elastic, fission, n_gamma)`.
    fn sample_slab(&self, slab: usize, energy: f64, r_urr: f64) -> (f64, f64, f64) {
        let mo = slab * URR_META_COLS;
        let n_e = self.meta[mo + 1] as usize;
        let n_cdf = self.meta[mo + 2] as usize;
        let log_log = self.meta[mo + 3] == 1;
        let eg_off = self.ae_offset[slab] as usize;

        // Energy bracket within this slab's tight grid.
        let mut i_e = 0usize;
        for k in 0..n_e - 1 {
            if energy >= self.energy_grid[eg_off + k] && energy < self.energy_grid[eg_off + k + 1] {
                i_e = k;
            }
        }
        let i_e_next = i_e + 1;
        let e_lo = self.energy_grid[eg_off + i_e];
        let e_hi = self.energy_grid[eg_off + i_e_next];
        let interp_f = if e_hi > e_lo {
            if log_log && e_lo > 0.0 {
                (energy / e_lo).ln() / (e_hi / e_lo).ln()
            } else {
                (energy - e_lo) / (e_hi - e_lo)
            }
        } else {
            0.0
        };

        // Band index at both energy points: the first CDF entry above `r_urr`
        // (upper bound), clamped, mirroring CPU `UrrData::find_cdf_index`.
        let cdf_base = self.cdf_offset[slab] as usize;
        let cdf_off_lo = cdf_base + i_e * n_cdf;
        let cdf_off_hi = cdf_base + i_e_next * n_cdf;
        let mut j_lo = 0usize;
        let mut j_hi = 0usize;
        for kc in 0..n_cdf {
            if self.cdf[cdf_off_lo + kc] <= r_urr {
                j_lo = kc + 1;
            }
            if self.cdf[cdf_off_hi + kc] <= r_urr {
                j_hi = kc + 1;
            }
        }
        let j_lo = j_lo.min(n_cdf - 1);
        let j_hi = j_hi.min(n_cdf - 1);

        let xs_off_lo = (cdf_off_lo + j_lo) * URR_XS_COLS;
        let xs_off_hi = (cdf_off_hi + j_hi) * URR_XS_COLS;
        // Log-log only when the record says so AND both endpoints are positive;
        // some tables carry negative values in extreme bands, which the CPU
        // clamps to zero after interpolating.
        let pick = |col: usize| -> f64 {
            let lo = self.xs[xs_off_lo + col];
            let hi = self.xs[xs_off_hi + col];
            let v = if log_log && lo > 0.0 && hi > 0.0 {
                ((1.0 - interp_f) * lo.ln() + interp_f * hi.ln()).exp()
            } else {
                (1.0 - interp_f) * lo + interp_f * hi
            };
            v.max(0.0)
        };
        // Column order is total / elastic / fission / n_gamma.
        (pick(1), pick(2), pick(3))
    }
}

/// One slab's URR-perturbed macroscopic partials, or `None` when this slab has
/// no URR table covering `energy` (the caller keeps its smooth values).
///
/// This is the single place the perturbation is defined. All three consumers
/// call it with the same `r_base`, so the material aggregate that governs the
/// flight, the per-nuclide weights that govern which nuclide is struck, and the
/// partials that govern the reaction split all ride the SAME sampled band
/// (issue #347). `urr_nuclide_random` is a pure function of `(r_base, ZA)`, so
/// recomputing it at each site is free of drift by construction.
pub(super) fn perturb_slab(
    tables: &UrrTables<'_>,
    slab: usize,
    energy: f64,
    r_base: f64,
    base: Partials,
) -> Option<Partials> {
    let mo = slab * URR_META_COLS;
    if tables.meta[mo] != 1 {
        return None;
    }
    let n_e = tables.meta[mo + 1] as usize;
    let inel_flag = tables.meta[mo + 4];
    let mult_smooth = tables.meta[mo + 6] == 1;
    let za = tables.meta[mo + 7];
    let eg_off = tables.ae_offset[slab] as usize;
    if n_e < 2
        || energy <= tables.energy_grid[eg_off]
        || energy >= tables.energy_grid[eg_off + n_e - 1]
    {
        return None;
    }

    // Independent per-nuclide band from the shared base (issue #204).
    let r_urr = crate::common::urr::urr_nuclide_random_cpu(r_base, za);
    let (urr_e, urr_f, urr_g) = tables.sample_slab(slab, energy, r_urr);

    // `multiply_smooth`: table entries are factors on the smooth macroscopic
    // baseline; otherwise absolute micro XS to be scaled by atom density. The
    // CPU multiplies the capture column by (smooth_absorption - smooth_fission)
    // == `base.absorption`.
    let (m_e, m_g, m_f) = if mult_smooth {
        (
            urr_e * base.elastic,
            urr_g * base.absorption,
            urr_f * base.fission,
        )
    } else {
        let n = tables.atom_density[slab];
        (n * urr_e, n * urr_g, n * urr_f)
    };
    Some(Partials {
        elastic: m_e,
        absorption: m_g,
        fission: m_f,
        // Smooth-inelastic exclusion (issue #105): with `inelastic_flag <= 0`
        // the CPU drops inelastic from the URR-window total.
        inelastic: if inel_flag == 0 { 0.0 } else { base.inelastic },
    })
}

/// This slab's smooth macroscopic partials, interpolated in the fine bracket.
pub(super) fn slab_baseline(
    nuc_partial_xs: &[f64],
    p_lo: usize,
    p_hi: usize,
    frac_f: f64,
) -> Partials {
    let at = |col: usize| -> f64 {
        let a = nuc_partial_xs[p_lo + col];
        let b = nuc_partial_xs[p_hi + col];
        a + (b - a) * frac_f
    };
    Partials {
        elastic: at(P_ELASTIC),
        absorption: at(P_ABSORPTION),
        inelastic: at(P_INELASTIC),
        fission: at(P_FISSION),
    }
}

impl Partials {
    pub(super) fn total(&self) -> f64 {
        self.elastic + self.absorption + self.inelastic + self.fission
    }
}

/// Apply every in-range URR nuclide's perturbation to a material's macroscopic
/// partials, given the band base `r_base` the walk holds at this energy.
///
/// `nuc_partial_xs` is the per-(material, nuclide) smooth baseline block on the
/// material's FINE grid; `row_of(slab_within_material)` gives a slab's row base
/// so the caller keeps ownership of the `fine_nuc_base` / `fine_n` arithmetic.
#[allow(clippy::too_many_arguments)]
pub(super) fn perturb(
    tables: &UrrTables<'_>,
    smooth: Partials,
    nuc_partial_xs: &[f64],
    row_of: impl Fn(usize) -> (usize, usize),
    nuc_off: usize,
    nuc_count: usize,
    energy: f64,
    r_base: f64,
    frac_f: f64,
) -> UrrPerturbation {
    let mut d_e = 0.0;
    let mut d_a = 0.0;
    let mut d_f = 0.0;
    let mut d_i = 0.0;
    let mut fired = false;

    for ku in 0..nuc_count {
        let slab = nuc_off + ku;
        let (p_lo, p_hi) = row_of(ku);
        let base = slab_baseline(nuc_partial_xs, p_lo, p_hi, frac_f);
        let Some(p) = perturb_slab(tables, slab, energy, r_base, base) else {
            continue;
        };
        d_e += p.elastic - base.elastic;
        d_a += p.absorption - base.absorption;
        d_f += p.fission - base.fission;
        d_i += p.inelastic - base.inelastic;
        fired = true;
    }

    if !fired {
        return UrrPerturbation {
            partials: smooth,
            fired: false,
            macro_capture: 0.0,
        };
    }

    let partials = Partials {
        elastic: (smooth.elastic + d_e).max(0.0),
        absorption: (smooth.absorption + d_a).max(0.0),
        inelastic: (smooth.inelastic + d_i).max(0.0),
        fission: (smooth.fission + d_f).max(0.0),
    };
    let macro_capture = (partials.absorption - partials.fission).max(0.0);
    UrrPerturbation {
        partials,
        fired: true,
        macro_capture,
    }
}
