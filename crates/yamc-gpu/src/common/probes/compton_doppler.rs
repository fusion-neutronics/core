//! Compton Doppler broadening on the GPU -- given the incident energy, the
//! sampled Klein-Nishina cosine `mu`, and the free-electron KN result, sample
//! a bound-electron shell weighted by occupancy and kinematically accessible
//! profile mass, draw a signed longitudinal momentum `pz` over the whole
//! allowed interval, and solve the bound-electron Compton kinematics for the
//! broadened `E_out` on the branch the sign of `pz` selects.
//!
//! The procedure is Kaltiaisenaho (2016, Sec. 3.4.8) as adopted by OpenMC in
//! openmc-dev/openmc#4036 and by the CPU `PhotonInteraction::compton_doppler`
//! (fusion-neutronics/core#22); the two are the same algorithm on the same
//! normalised tables, so the GPU is tested against the CPU distribution rather
//! than byte for byte. Extracted from the inline block in
//! `multi_cell_photon_transport` so the mega kernel calls it (single source of
//! truth). Differences from the CPU that are forced by `#[cube]`:
//!
//! - `exp` and `ln` for the log-linear profile tail go through the software
//!   polyfills, since the f64 GLSL ops are broken on RADV (cubecl#1316).
//! - The conditional shell PMF is not stored: the per-shell kinematics are
//!   recomputed while walking the cumulative sum, which costs a few flops per
//!   shell per attempt and no thread-private array.
//! - The retry budget on the `E'/E` rejection is 1024 attempts (the CPU allows
//!   100000); at the worst case, 10 MeV backscatter with `E'/E` near 0.025,
//!   the chance of exhausting it is below 1e-11. Past it the free-electron
//!   energy is returned, as on the CPU.
//!
//! Returns `e_out_kn` unchanged when the material has no Doppler data or no
//! shell is kinematically open.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::{exp_f64, ln_f64};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::photon::transport::{DOP_MAX_PZ, DOP_MAX_SHELLS, FINE_STRUCTURE, MASS_ELECTRON_EV};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// `f64::EPSILON`, spelled out because the associated constant is not
/// available inside `#[cube]` code.
const F64_EPSILON: f64 = 2.220_446_049_250_313e-16;

/// Result of one Doppler-broadened Compton energy sample.
#[derive(CubeType)]
pub struct DopplerSample {
    /// Advanced PCG-32 state.
    pub state: u64,
    /// Doppler-broadened outgoing photon energy (eV); equals `e_out_kn` when
    /// Doppler is skipped.
    pub e_out: f64,
    /// Compton-profile shell sampled from `dop_electron_pdf` (the shell the
    /// event ionizes). Set for every code path once a shell is drawn, matching
    /// the CPU `compton_doppler`, which returns `i_shell` regardless of the
    /// Doppler skip / no-root branches. `u32::MAX` (the none-shell sentinel)
    /// means no shell was sampled (no Doppler data), mirroring the CPU
    /// `i_shell = -1` case; the caller then skips atomic relaxation (its
    /// `< DOP_MAX_SHELLS` guard rejects the sentinel).
    pub shell: u32,
}

/// Kinematic bounds of one shell for one collision (Kaltiaisenaho Eqs. 3.73,
/// 3.117, 3.118): the upper momentum bound, the half-profile integral at its
/// magnitude, and the accessible profile mass. `mass <= 0` means the shell is
/// closed for this collision.
#[derive(CubeType)]
pub struct ShellKinematics {
    pub pz_max: f64,
    pub c_limit: f64,
    pub mass: f64,
}

/// `E'/E` for one signed momentum: `valid == 0` means no physical root on the
/// branch the sign of `pz` selects (the CPU twin returns `None`).
#[derive(CubeType)]
pub struct EnergyRatio {
    pub valid: u32,
    pub ratio: f64,
}

/// One momentum-and-energy draw for a chosen shell: `ok == 1` carries an
/// accepted `e_out`; `ok == 0` is a rejection (no physical root, above the
/// binding limit, or failed the `E'/E` factor) and the caller redraws.
#[derive(CubeType)]
pub struct MomentumSample {
    pub state: u64,
    pub ok: u32,
    pub e_out: f64,
}

/// `K_i(pz)`: the normalised half-profile integral of the shell whose tables
/// start at `table_off`, from `0` to `pz >= 0`, through the tabulated trapezoid
/// cdf inside the grid and the log-linear tail past it. Zero for `pz <= 0`,
/// capped at 1/2. Mirror of the CPU `PhotonInteraction::compton_profile_cdf`.
#[cube]
pub fn dop_profile_cdf_at(
    pz: f64,
    table_off: u32,
    n_pz: u32,
    tail_slope: f64,
    dop_pz_grid: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
) -> f64 {
    let mut c = 0.0_f64;
    if pz > 0.0_f64 {
        let last = n_pz - 1u32;
        let pz_last = dop_pz_grid[last as usize];
        if pz >= pz_last {
            let j_last = dop_profile_pdf[(table_off + last) as usize];
            c = dop_profile_cdf[(table_off + last) as usize]
                + j_last * (exp_f64(tail_slope * (pz - pz_last)) - 1.0_f64) / tail_slope;
        } else {
            let mut lo = 0u32;
            let mut hi = n_pz;
            let mut it = 0u32;
            while it < 16u32 && lo + 1u32 < hi {
                let mid = (lo + hi) / 2u32;
                if dop_pz_grid[mid as usize] <= pz {
                    lo = mid;
                } else {
                    hi = mid;
                }
                it += 1u32;
            }
            let mut i = lo;
            if i >= n_pz - 1u32 {
                i = n_pz - 2u32;
            }
            let pz_l = dop_pz_grid[i as usize];
            let pz_r = dop_pz_grid[(i + 1u32) as usize];
            let p_l = dop_profile_pdf[(table_off + i) as usize];
            let p_r = dop_profile_pdf[(table_off + i + 1u32) as usize];
            let c_l = dop_profile_cdf[(table_off + i) as usize];
            let slope = (p_r - p_l) / (pz_r - pz_l);
            let delta = pz - pz_l;
            c = c_l + p_l * delta + 0.5_f64 * slope * delta * delta;
        }
        if c > 0.5_f64 {
            c = 0.5_f64;
        }
    }
    c
}

/// Inverse of [`dop_profile_cdf_at`]: the `pz >= 0` at which the half-profile
/// integral reaches `c`. Piecewise-linear inversion inside the grid, in the
/// rationalised form that stays conditioned when the local slope is small
/// (Kaltiaisenaho Eq. 3.126), and the log-linear tail past it (Eq. 3.123).
#[cube]
#[allow(unused_assignments)]
pub fn dop_invert_profile_cdf(
    c: f64,
    table_off: u32,
    n_pz: u32,
    tail_slope: f64,
    dop_pz_grid: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
) -> f64 {
    let last = n_pz - 1u32;
    let c_last = dop_profile_cdf[(table_off + last) as usize];
    let mut pz = 0.0_f64;
    if c >= c_last {
        let j_last = dop_profile_pdf[(table_off + last) as usize];
        pz = dop_pz_grid[last as usize]
            + ln_f64(1.0_f64 + tail_slope * (c - c_last) / j_last) / tail_slope;
    } else {
        let mut lo = 0u32;
        let mut hi = n_pz;
        let mut it = 0u32;
        while it < 16u32 && lo + 1u32 < hi {
            let mid = (lo + hi) / 2u32;
            if dop_profile_cdf[(table_off + mid) as usize] <= c {
                lo = mid;
            } else {
                hi = mid;
            }
            it += 1u32;
        }
        let mut i = lo;
        if i >= n_pz - 1u32 {
            i = n_pz - 2u32;
        }
        let pz_l = dop_pz_grid[i as usize];
        let pz_r = dop_pz_grid[(i + 1u32) as usize];
        let p_l = dop_profile_pdf[(table_off + i) as usize];
        let p_r = dop_profile_pdf[(table_off + i + 1u32) as usize];
        let c_l = dop_profile_cdf[(table_off + i) as usize];
        let delta_c = c - c_l;
        if p_l == p_r {
            pz = pz_l + delta_c / p_l;
        } else {
            let slope = (p_r - p_l) / (pz_r - pz_l);
            let mut disc = p_l * p_l + 2.0_f64 * slope * delta_c;
            if disc < 0.0_f64 {
                disc = 0.0_f64;
            }
            pz = pz_l + 2.0_f64 * delta_c / (p_l + disc.sqrt());
        }
    }
    pz
}

/// `E'/E` for a bound electron with signed longitudinal momentum `pz` (atomic
/// units): the impulse-approximation kinematics quadratic solved for `E'`,
/// taking the lower positive root for `pz < 0` and the upper for `pz > 0`.
/// Mirror of the CPU `compton_energy_ratio`.
#[cube]
#[allow(unused_assignments)]
pub fn dop_energy_ratio(alpha: f64, mu: f64, pz: f64) -> EnergyRatio {
    let one_minus_mu = 1.0_f64 - mu;
    let f = 1.0_f64 + alpha * one_minus_mu;
    let mut ratio = 0.0_f64;
    let mut valid = 0u32;
    if pz == 0.0_f64 {
        ratio = 1.0_f64 / f;
        valid = 1u32;
    } else {
        let m = pz / FINE_STRUCTURE;
        let m2 = m * m;
        let a = m2 - f * f;
        let b = 2.0_f64 * (f - m2 * mu);
        let c = m2 - 1.0_f64;
        let four_ac = 4.0_f64 * a * c;
        let four_ac_abs = four_ac.abs();
        let b2 = b * b;
        let mut disc = b2 - four_ac;
        let tol = 16.0_f64 * F64_EPSILON * (b2 + four_ac_abs);
        let neg_tol = 0.0_f64 - tol;
        if disc >= neg_tol {
            if disc < 0.0_f64 {
                disc = 0.0_f64;
            }
            let mut root1 = 0.0_f64;
            let mut root2 = 0.0_f64;
            let mut have = 1u32;
            let a_abs = a.abs();
            let b_abs = b.abs();
            let c_abs = c.abs();
            let small_a = 1.0e-14_f64 * (b_abs + c_abs);
            if a_abs < small_a {
                if b == 0.0_f64 {
                    have = 0u32;
                } else {
                    root1 = (0.0_f64 - c) / b;
                    root2 = root1;
                }
            } else {
                let sq = disc.sqrt();
                let mut signed_sq = sq;
                if b < 0.0_f64 {
                    signed_sq = 0.0_f64 - sq;
                }
                let q = (0.0_f64 - 0.5_f64) * (b + signed_sq);
                root1 = q / a;
                if q == 0.0_f64 {
                    root2 = (0.0_f64 - b + sq) / (2.0_f64 * a);
                } else {
                    root2 = c / q;
                }
            }
            if have == 1u32 {
                let mut rmin = 0.0_f64;
                let mut rmax = 0.0_f64;
                let mut any = 0u32;
                if root1 > 0.0_f64 {
                    rmin = root1;
                    rmax = root1;
                    any = 1u32;
                }
                if root2 > 0.0_f64 {
                    if any == 0u32 {
                        rmin = root2;
                        rmax = root2;
                    } else {
                        if root2 < rmin {
                            rmin = root2;
                        }
                        if root2 > rmax {
                            rmax = root2;
                        }
                    }
                    any = 1u32;
                }
                if any == 1u32 {
                    let mut r = rmax;
                    if pz < 0.0_f64 {
                        r = rmin;
                    }
                    let free = 1.0_f64 / f;
                    let mut scale = 1.0_f64;
                    if free > 1.0_f64 {
                        scale = free;
                    }
                    let tol2 = 16.0_f64 * F64_EPSILON * scale;
                    let upper = free + tol2;
                    let lower = free - tol2;
                    let mut bad = 0u32;
                    if pz < 0.0_f64 && r > upper {
                        bad = 1u32;
                    }
                    if pz > 0.0_f64 && r < lower {
                        bad = 1u32;
                    }
                    if bad == 0u32 {
                        ratio = r;
                        valid = 1u32;
                    }
                }
            }
        }
    }
    EnergyRatio { valid, ratio }
}

/// Kinematic bounds of shell `shell` of the element slab whose per-shell
/// scalars start at `shell_off` (Kaltiaisenaho Eqs. 3.73 and 3.118). Mirror of
/// the CPU `compton_shell_kinematics`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dop_shell_kinematics(
    alpha: f64,
    mu: f64,
    e_in: f64,
    shell_off: u32,
    shell: u32,
    n_pz: u32,
    dop_pz_grid: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_tail_slope: &[f64],
    dop_negative_mass: &[f64],
) -> ShellKinematics {
    let mut pz_max = 0.0_f64;
    let mut c_limit = 0.0_f64;
    let mut mass = 0.0_f64;
    let e_b = dop_binding_energy[(shell_off + shell) as usize];
    if e_in > e_b {
        let e_minus_eb = e_in - e_b;
        let denom_sq = 2.0_f64 * e_in * e_minus_eb * (1.0_f64 - mu) + e_b * e_b;
        pz_max = (0.0_f64 - FINE_STRUCTURE) * (e_b - e_minus_eb * alpha * (1.0_f64 - mu))
            / denom_sq.sqrt();
        let lower_bound = 0.0_f64 - FINE_STRUCTURE;
        if pz_max > lower_bound {
            let table_off = (shell_off + shell) * DOP_MAX_PZ;
            let slope = dop_tail_slope[(shell_off + shell) as usize];
            c_limit = dop_profile_cdf_at(
                pz_max.abs(),
                table_off,
                n_pz,
                slope,
                dop_pz_grid,
                dop_profile_pdf,
                dop_profile_cdf,
            );
            let c_negative = dop_negative_mass[(shell_off + shell) as usize];
            if pz_max < 0.0_f64 {
                mass = c_negative - c_limit;
            } else {
                mass = c_negative + c_limit;
            }
        }
    }
    ShellKinematics {
        pz_max,
        c_limit,
        mass,
    }
}

/// Draw a signed `pz` for the chosen shell, conditional on its allowed
/// interval, and turn it into `E'` with the `E - E_b` limit and the `E'/E`
/// rejection (Kaltiaisenaho Eqs. 3.120 to 3.127). Two PCG draws. Mirror of
/// the CPU `sample_compton_momentum`.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn dop_sample_momentum(
    state: u64,
    alpha: f64,
    mu: f64,
    e_in: f64,
    shell_off: u32,
    shell: u32,
    n_pz: u32,
    kin: ShellKinematics,
    dop_pz_grid: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_tail_slope: &[f64],
    dop_negative_mass: &[f64],
) -> MomentumSample {
    let mut st = state;
    let table_off = (shell_off + shell) * DOP_MAX_PZ;
    let slope = dop_tail_slope[(shell_off + shell) as usize];
    let c_negative = dop_negative_mass[(shell_off + shell) as usize];

    let r1 = pcg_out(st);
    st = st * PCG_MULT + PCG_INCR;
    let xi_pz = (r1 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
    // The tabulated profile is symmetric, so the negative branch is the
    // reflected half-profile cdf.
    let mut pz = 0.0_f64;
    if kin.pz_max < 0.0_f64 {
        let c = kin.c_limit + xi_pz * kin.mass;
        pz = 0.0_f64
            - dop_invert_profile_cdf(
                c,
                table_off,
                n_pz,
                slope,
                dop_pz_grid,
                dop_profile_pdf,
                dop_profile_cdf,
            );
    } else {
        let c = xi_pz * kin.mass;
        if c < c_negative {
            pz = 0.0_f64
                - dop_invert_profile_cdf(
                    c_negative - c,
                    table_off,
                    n_pz,
                    slope,
                    dop_pz_grid,
                    dop_profile_pdf,
                    dop_profile_cdf,
                );
        } else {
            pz = dop_invert_profile_cdf(
                c - c_negative,
                table_off,
                n_pz,
                slope,
                dop_pz_grid,
                dop_profile_pdf,
                dop_profile_cdf,
            );
        }
    }

    let mut ok = 0u32;
    let mut e_out = 0.0_f64;
    let er = dop_energy_ratio(alpha, mu, pz);
    let mut ratio = er.ratio;
    if er.valid == 1u32 && ratio > 0.0_f64 {
        let max_ratio = 1.0_f64 - dop_binding_energy[(shell_off + shell) as usize] / e_in;
        let scale = if max_ratio > 1.0_f64 {
            max_ratio
        } else {
            1.0_f64
        };
        let tol = 16.0_f64 * F64_EPSILON * scale;
        let limit = max_ratio + tol;
        if ratio <= limit {
            if ratio > max_ratio {
                ratio = max_ratio;
            }
            // Eq. 3.127: the E'/E factor of the approximate RIA DDCS as a
            // rejection once E' is known.
            let r2 = pcg_out(st);
            st = st * PCG_MULT + PCG_INCR;
            let xi_acc = (r2 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
            if xi_acc <= ratio {
                ok = 1u32;
                e_out = ratio * e_in;
            }
        }
    }
    MomentumSample {
        state: st,
        ok,
        e_out,
    }
}

/// Sample the Doppler-broadened Compton `E_out`. `e_in` is the incident
/// photon energy (eV), `mu` the KN cosine, `e_out_kn` the free-electron KN
/// result (the fallback). The `dop_*` slices are the per-element Compton
/// profile tables (`mat_idx` selects the element slab). THE single source of
/// truth, called by both the test launcher and the inline Compton branch of
/// `multi_cell_photon_transport`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn compton_doppler_sample(
    state: u64,
    e_in: f64,
    mu: f64,
    e_out_kn: f64,
    mat_idx: u32,
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_tail_slope: &[f64],
    dop_negative_mass: &[f64],
) -> DopplerSample {
    let mut st = state;
    let mut e_out_final = e_out_kn;
    // `u32::MAX` = "no shell sampled" (no Doppler data): the kernel's universal
    // none-shell sentinel, which `< DOP_MAX_SHELLS` on the caller side rejects.
    let mut shell_out = 4_294_967_295u32;
    let n_sh = dop_n_shells[mat_idx as usize];
    let n_pz: u32 = dop_pz_grid.len() as u32;
    if dop_has_data[mat_idx as usize] == 1u32 && n_sh >= 1u32 && n_pz >= 2u32 {
        let alpha = e_in / MASS_ELECTRON_EV;
        let shell_off = mat_idx * DOP_MAX_SHELLS;
        let mut done = 0u32;

        // Phase 1 (Kaltiaisenaho Eq. 3.119): propose the shell by occupancy
        // and accept it with its accessible profile mass, twice.
        let mut attempt = 0u32;
        while attempt < 2u32 && done == 0u32 {
            let r_sh = pcg_out(st);
            st = st * PCG_MULT + PCG_INCR;
            let xi_sh = (r_sh as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
            let mut cum = 0.0_f64;
            let mut shell = n_sh - 1u32;
            let mut found = 0u32;
            let mut i = 0u32;
            while i < n_sh && found == 0u32 {
                cum += dop_electron_pdf[(shell_off + i) as usize];
                if xi_sh < cum {
                    shell = i;
                    found = 1u32;
                }
                i += 1u32;
            }
            shell_out = shell;
            let kin = dop_shell_kinematics(
                alpha,
                mu,
                e_in,
                shell_off,
                shell,
                n_pz,
                dop_pz_grid,
                dop_binding_energy,
                dop_profile_pdf,
                dop_profile_cdf,
                dop_tail_slope,
                dop_negative_mass,
            );
            if kin.mass > 0.0_f64 {
                let r_acc = pcg_out(st);
                st = st * PCG_MULT + PCG_INCR;
                let xi_acc = (r_acc as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                if xi_acc < kin.mass {
                    let ms = dop_sample_momentum(
                        st,
                        alpha,
                        mu,
                        e_in,
                        shell_off,
                        shell,
                        n_pz,
                        kin,
                        dop_pz_grid,
                        dop_binding_energy,
                        dop_profile_pdf,
                        dop_profile_cdf,
                        dop_tail_slope,
                        dop_negative_mass,
                    );
                    st = ms.state;
                    if ms.ok == 1u32 {
                        e_out_final = ms.e_out;
                        done = 1u32;
                    }
                }
            }
            attempt += 1u32;
        }

        // Phase 2 (Eq. 3.116): the conditional shell PMF, f_i times the
        // accessible mass, walked with the kinematics recomputed per shell so
        // no thread-private array is needed. Bounded so a run of E'/E
        // rejections cannot spin the thread.
        if done == 0u32 {
            let mut norm = 0.0_f64;
            let mut i = 0u32;
            while i < n_sh {
                let kin = dop_shell_kinematics(
                    alpha,
                    mu,
                    e_in,
                    shell_off,
                    i,
                    n_pz,
                    dop_pz_grid,
                    dop_binding_energy,
                    dop_profile_pdf,
                    dop_profile_cdf,
                    dop_tail_slope,
                    dop_negative_mass,
                );
                if kin.mass > 0.0_f64 {
                    norm += dop_electron_pdf[(shell_off + i) as usize] * kin.mass;
                }
                i += 1u32;
            }
            if norm > 0.0_f64 {
                let mut tries = 0u32;
                while tries < 1024u32 && done == 0u32 {
                    let r_sh = pcg_out(st);
                    st = st * PCG_MULT + PCG_INCR;
                    let rn = (r_sh as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64) * norm;
                    let mut cum = 0.0_f64;
                    let mut shell = n_sh - 1u32;
                    let mut found = 0u32;
                    let mut j = 0u32;
                    while j < n_sh && found == 0u32 {
                        let kin_j = dop_shell_kinematics(
                            alpha,
                            mu,
                            e_in,
                            shell_off,
                            j,
                            n_pz,
                            dop_pz_grid,
                            dop_binding_energy,
                            dop_profile_pdf,
                            dop_profile_cdf,
                            dop_tail_slope,
                            dop_negative_mass,
                        );
                        if kin_j.mass > 0.0_f64 {
                            cum += dop_electron_pdf[(shell_off + j) as usize] * kin_j.mass;
                        }
                        if rn < cum {
                            shell = j;
                            found = 1u32;
                        }
                        j += 1u32;
                    }
                    shell_out = shell;
                    let kin = dop_shell_kinematics(
                        alpha,
                        mu,
                        e_in,
                        shell_off,
                        shell,
                        n_pz,
                        dop_pz_grid,
                        dop_binding_energy,
                        dop_profile_pdf,
                        dop_profile_cdf,
                        dop_tail_slope,
                        dop_negative_mass,
                    );
                    if kin.mass > 0.0_f64 {
                        let ms = dop_sample_momentum(
                            st,
                            alpha,
                            mu,
                            e_in,
                            shell_off,
                            shell,
                            n_pz,
                            kin,
                            dop_pz_grid,
                            dop_binding_energy,
                            dop_profile_pdf,
                            dop_profile_cdf,
                            dop_tail_slope,
                            dop_negative_mass,
                        );
                        st = ms.state;
                        if ms.ok == 1u32 {
                            e_out_final = ms.e_out;
                            done = 1u32;
                        }
                    }
                    tries += 1u32;
                }
            }
        }
    }

    DopplerSample {
        state: st,
        e_out: e_out_final,
        shell: shell_out,
    }
}

/// Per-thread test launcher: single material (`mat_idx = 0`), fixed incident
/// energy + `mu`. Writes the Doppler-broadened `E_out` to `out_e[tid]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn compton_doppler_kernel(
    seeds: &[u32],
    e_in_buf: &[f64],
    mu_buf: &[f64],
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_tail_slope: &[f64],
    dop_negative_mass: &[f64],
    out_e: &mut [f64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let e_in = e_in_buf[0usize];
    let mu = mu_buf[0usize];
    // Free-electron KN result as the fallback (e_out_kn).
    let alpha = e_in / MASS_ELECTRON_EV;
    let e_out_kn = alpha / (1.0_f64 + alpha * (1.0_f64 - mu)) * MASS_ELECTRON_EV;
    let state = expand_seed(seeds[ABSOLUTE_POS]);
    let ds = compton_doppler_sample(
        state,
        e_in,
        mu,
        e_out_kn,
        0u32,
        dop_n_shells,
        dop_has_data,
        dop_pz_grid,
        dop_electron_pdf,
        dop_binding_energy,
        dop_profile_pdf,
        dop_profile_cdf,
        dop_tail_slope,
        dop_negative_mass,
    );
    out_e[ABSOLUTE_POS] = ds.e_out;
}

/// Run the Doppler sampler for a single material at fixed `(e_in, mu)`.
#[allow(clippy::too_many_arguments)]
pub fn run_compton_doppler(
    ctx: &GpuContext,
    seeds: &[u32],
    e_in: f64,
    mu: f64,
    pz_grid: &[f64],
    n_shells: u32,
    electron_pdf: &[f64],
    binding_energy: &[f64],
    profile_pdf: &[f64],
    profile_cdf: &[f64],
    tail_slope: &[f64],
    negative_mass: &[f64],
) -> Vec<f64> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let e_in_h = client.create_from_slice(bytemuck::cast_slice(&[e_in]));
    let mu_h = client.create_from_slice(bytemuck::cast_slice(&[mu]));
    let ns_h = client.create_from_slice(bytemuck::cast_slice(&[n_shells]));
    let has_h = client.create_from_slice(bytemuck::cast_slice(&[1u32]));
    let pz_h = client.create_from_slice(bytemuck::cast_slice(pz_grid));
    let epdf_h = client.create_from_slice(bytemuck::cast_slice(electron_pdf));
    let be_h = client.create_from_slice(bytemuck::cast_slice(binding_energy));
    let ppdf_h = client.create_from_slice(bytemuck::cast_slice(profile_pdf));
    let pcdf_h = client.create_from_slice(bytemuck::cast_slice(profile_cdf));
    let slope_h = client.create_from_slice(bytemuck::cast_slice(tail_slope));
    let negm_h = client.create_from_slice(bytemuck::cast_slice(negative_mass));
    let out_h = client.empty(n * std::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        compton_doppler_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(e_in_h, 1),
            BufferArg::from_raw_parts(mu_h, 1),
            BufferArg::from_raw_parts(ns_h, 1),
            BufferArg::from_raw_parts(has_h, 1),
            BufferArg::from_raw_parts(pz_h, pz_grid.len()),
            BufferArg::from_raw_parts(epdf_h, electron_pdf.len()),
            BufferArg::from_raw_parts(be_h, binding_energy.len()),
            BufferArg::from_raw_parts(ppdf_h, profile_pdf.len()),
            BufferArg::from_raw_parts(pcdf_h, profile_cdf.len()),
            BufferArg::from_raw_parts(slope_h, tail_slope.len()),
            BufferArg::from_raw_parts(negm_h, negative_mass.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
