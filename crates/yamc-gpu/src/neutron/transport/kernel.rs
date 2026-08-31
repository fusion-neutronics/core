//! The neutron transport `#[cube]` kernel body. cubecl transpiles
//! `multi_cell_transport_kernel` to SPIR-V via the `#[cube]` macro,
//! which also generates a `multi_cell_transport_kernel` module
//! containing the host-side `launch_unchecked` function -- re-exported
//! by `mod.rs` so the host driver can launch it.
//!
//! The kernel is bit-for-bit equivalent to the CPU mirrors in
//! `cpu.rs` and `cpu_rayon.rs` (same PCG-32 sequence, same physics
//! sampling -- those mirrors call into `yamc_physics::gpu::flat::*`, the
//! kernel reimplements the math here because cubecl can't call
//! regular Rust). Drift between this body and yamc-physics is
//! caught by the `gpu_*` test suite that compares kernel output
//! against the CPU mirrors.

use cubecl::prelude::*;

// Glob-import the parent so all layout constants (`COL_*`,
// `MT_SLOT_*`, `MAT_F64_COLS`, `BOUNDARY_*`, `FISSION_WEIGHT_CAP`,
// `MAX_*`) and the polyfilled f64 ops (`cos_f64`, `exp_f64`,
// `ln_f64`, `sin_f64`) reach the kernel without a 30-line
// explicit-import block.
use super::*;
use crate::common::geometry::region_eval::region_contains;
use crate::common::geometry::surface_distance::{
    cone_smallest_positive, quadric_smallest_positive, torus_smallest_positive,
};
use crate::common::particle_bank::{BANK_GEN_NXN_SPILL, PTYPE_NEUTRON, PTYPE_PHOTON};
use crate::common::polyfills::{atan2_f64, cos_f64, exp_f64, ln_f64, sin_f64};
use crate::common::sampling::fission_chi::sample_fission_progeny_energy;
use crate::neutron::transport::roulette::weight_cutoff_roulette;
use crate::photon::production_kinematics::sample_photon_kinematics;
use crate::photon::production_select::{sample_photon_count, sample_photon_product};

// `manual_clamp`: cubecl's `#[cube]` does not lower `f64::clamp` to a
// SPIR-V op the way native Rust does, so the in-kernel mu/μ_lab
// guards are written as paired `if` branches.
// `assign_op_pattern`: same -- `*=` on a `#[cube]` register doesn't
// always lower correctly, so the kernel uses explicit `x = x * y`.
// `identity_op`: the explicit `+ 0u32` reads in `mt_slot_u32_meta`
// keep all 7 column accesses uniform (`+ 0u32` for COL_EOUT_KIND
// through `+ 6u32` for COL_NBPS_N_BODIES), which is much easier to
// scan against the `COL_*` constants than dropping just the first.

/// Thread-private touched-list capacity for batch-free per-history
/// variance (issue #233). A history accumulates each distinct
/// `(tally, cell, energy)` bin it touches into this register array,
/// then flushes `sum` + `sum_sq` once at history end. `32` entries
/// (`u32` bin index + `f64` running total) cover essentially every
/// realistic GPU history (few cells + modest energy bins) with zero
/// global-memory traffic. A history that touches MORE than `PERHIST_K`
/// distinct bins spills the overflow into a per-history global buffer
/// (`spill_bin` / `spill_val`), so the variance is EXACT for any number
/// of distinct bins -- no cap, no fallback, no under-count. `pub` so the
/// dispatch layer can size the per-history spill buffer / launch chunk.
pub const PERHIST_K: u32 = 32;

/// Energy-function weight for the table packed at `off` in
/// `TalliesPack::efunc_params` (`energy_function=` / `dose_coefficients=`,
/// issue #271). `#[cube]` twin of `common::tallies::energy_function_weight`.
///
/// Callers MUST range-test before calling: the CPU drops the whole scoring
/// event when the energy falls outside the table, which is a different
/// outcome from a zero weight, and a cubecl fn cannot return the `Option`
/// that would carry that. The test is two comparisons against
/// `params[off + 1]` and `params[off + n]`, so it costs nothing at the call
/// site and keeps this fn a pure evaluator.
///
/// Two things it deliberately does not do. It does not solve the spline --
/// the natural-cubic coefficients arrive precomputed from
/// `EnergyFunctionFilter::new`, which is what lets both backends interpolate
/// identically instead of merely closely. And it works in LINEAR energy, not
/// the log energy every surrounding cross-section lookup uses.
#[cube]
pub(crate) fn energy_function_weight_kernel(params: &[f64], off: u32, energy: f64) -> f64 {
    let n = params[off as usize] as u32;
    let e0 = off + 1u32;
    let c0 = e0 + n;
    // First index with `e[idx] > energy`; the bracket is the one below it.
    // Phrased as a lower bound rather than an upper one so the midpoint stays a
    // plain `(lo + hi) / 2`, like the other binary searches in this kernel.
    // Callers range-test before calling, so `e[0] <= energy` and `lo >= 1`
    // here: the decrement cannot underflow.
    //
    // This mirrors the CPU `binary_search_by` tie-breaking: on an exact knot
    // hit both land on that knot (so `dx == 0`), and at the top edge both fall
    // back to the final interval evaluated at its right end.
    let mut lo = 0u32;
    let mut hi = n;
    let mut iter = 0u32;
    while iter < 32u32 && lo < hi {
        let mid: u32 = (lo + hi) / 2u32;
        if params[(e0 + mid) as usize] <= energy {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
        iter += 1u32;
    }
    let mut idx = lo - 1u32;
    if idx > n - 2u32 {
        idx = n - 2u32;
    }
    let dx = energy - params[(e0 + idx) as usize];
    let a = params[(c0 + 4u32 * idx) as usize];
    let b = params[(c0 + 4u32 * idx + 1u32) as usize];
    let c = params[(c0 + 4u32 * idx + 2u32) as usize];
    let d = params[(c0 + 4u32 * idx + 3u32) as usize];
    a + dx * (b + dx * (c + dx * d))
}

/// `#[cube]` twin of `crate::common::tallies::rect_mesh_crossings` + the mesh
/// branch of the CPU `accumulate_tallies`, specialised to the issue-#234
/// mesh_direct per-source variance path. Walks the step segment
/// `r0 -> r0 + d*dir` across a rectangular row-major mesh (packed descriptor
/// at `mesh_params[mo..]`, layout `[ll(3), ur(3), inv_width(3), width(3),
/// shape(3)]`) with an Amanatides-Woo DDA and atomic-adds each voxel's
/// `d * length_fraction * score * weight` (fixed-point at `scale`, symmetric
/// round) into `src_acc[src_base + base + voxel]`.
///
/// The arithmetic mirrors the CPU twin exactly (segment length from `r1 - r0`,
/// `d * lf * score * weight` in that association) so the summed mean is
/// bit-identical to a matched-stream CPU run. Only cubecl-spirv-validated ops
/// are used: `+ - * /`, comparisons, unary `-`, `sqrt`, and casts. Floors of
/// non-negative reals are `as u32` casts (equal to the CPU's `.floor() as
/// usize`); `abs`/`min`/`max`/`INFINITY` are expanded to `if`/`else` and a
/// large finite sentinel (parallel axes are never selected, so the sentinel
/// never reaches the output).
#[allow(
    clippy::manual_clamp,
    clippy::assign_op_pattern,
    clippy::too_many_arguments,
    // `total_length - total_length` intentionally forces a RUNTIME-typed 0.0
    // (see rt_zero); a literal would make dependent accumulators comptime and
    // fail cubecl's expand.
    clippy::eq_op
)]
#[cube]
pub(crate) fn mesh_rect_score_src_acc(
    mesh_params: &[f64],
    mo: u32,
    r0x: f64,
    r0y: f64,
    r0z: f64,
    dirx: f64,
    diry: f64,
    dirz: f64,
    d: f64,
    score: f64,
    weight: f64,
    base: u32,
    src_base: u32,
    scale: f64,
    src_acc: &mut [Atomic<u64>],
) {
    // Comptime thresholds (compared against runtime values; never reassigned).
    let tiny = 1e-8_f64;
    let fp = 1e-14_f64;
    let mob = mo as usize;
    let ll0 = mesh_params[mob];
    let ll1 = mesh_params[mob + 1];
    let ll2 = mesh_params[mob + 2];
    let ur0 = mesh_params[mob + 3];
    let ur1 = mesh_params[mob + 4];
    let ur2 = mesh_params[mob + 5];
    let iw0 = mesh_params[mob + 6];
    let iw1 = mesh_params[mob + 7];
    let iw2 = mesh_params[mob + 8];
    let w0 = mesh_params[mob + 9];
    let w1 = mesh_params[mob + 10];
    let w2 = mesh_params[mob + 11];
    let sh0 = mesh_params[mob + 12] as u32;
    let sh1 = mesh_params[mob + 13] as u32;
    let sh2 = mesh_params[mob + 14] as u32;
    let r1x = r0x + dirx * d;
    let r1y = r0y + diry * d;
    let r1z = r0z + dirz * d;
    let ex = r1x - r0x;
    let ey = r1y - r0y;
    let ez = r1z - r0z;
    let total_length = (ex * ex + ey * ey + ez * ez).sqrt();
    // Runtime-typed 0.0 and large sentinel. cubecl's `#[cube]` frontend keeps an
    // f64 `let mut` initialised from a bare literal COMPTIME; the instant such a
    // variable is read (running min/max, boundary distance) and then
    // conditionally assigned a runtime value it fails to unify
    // (`expected f64, found NativeExpand<f64>`). Seeding every read-modify-write
    // f64 accumulator from a runtime expression forces the runtime type.
    // `rt_zero` is exactly 0.0 (total_length is finite) and `big` exactly 1e300.
    let rt_zero = total_length - total_length;
    let big = rt_zero + 1e300_f64;

    let mut valid = true;
    if total_length < 2.0 * tiny {
        valid = false;
    }

    // Per-axis |dir|, signed reciprocal, and t_delta (voxel-crossing length).
    let mut absx = dirx;
    let mut absy = diry;
    let mut absz = dirz;
    if dirx < 0.0 {
        absx = -dirx;
    }
    if diry < 0.0 {
        absy = -diry;
    }
    if dirz < 0.0 {
        absz = -dirz;
    }
    let mut invx = rt_zero;
    let mut invy = rt_zero;
    let mut invz = rt_zero;
    if absx >= fp {
        invx = 1.0 / dirx;
    }
    if absy >= fp {
        invy = 1.0 / diry;
    }
    if absz >= fp {
        invz = 1.0 / dirz;
    }
    // t_delta = width * |1/dir| = width * (1/|dir|), bit-identical to the CPU
    // twin's `width * inv_direction.abs()` (|1/dir| and 1/|dir| share bits).
    let mut tdx = big;
    let mut tdy = big;
    let mut tdz = big;
    if absx >= fp {
        tdx = w0 * (1.0 / absx);
    }
    if absy >= fp {
        tdy = w1 * (1.0 / absy);
    }
    if absz >= fp {
        tdz = w2 * (1.0 / absz);
    }

    // Entry voxel: direct floor if r0 is inside the mesh, else a ray/AABB slab
    // test to find where the segment enters (nudged in by TINY_BIT).
    let mut ixv = 0u32;
    let mut iyv = 0u32;
    let mut izv = 0u32;
    let mut curx = r0x;
    let mut cury = r0y;
    let mut curz = r0z;
    let mut traveled = rt_zero;
    let mut in_mesh = true;
    if r0x < ll0 || r0x >= ur0 {
        in_mesh = false;
    }
    if r0y < ll1 || r0y >= ur1 {
        in_mesh = false;
    }
    if r0z < ll2 || r0z >= ur2 {
        in_mesh = false;
    }
    if in_mesh {
        ixv = ((r0x - ll0) * iw0) as u32;
        iyv = ((r0y - ll1) * iw1) as u32;
        izv = ((r0z - ll2) * iw2) as u32;
        if ixv >= sh0 {
            ixv = sh0 - 1;
        }
        if iyv >= sh1 {
            iyv = sh1 - 1;
        }
        if izv >= sh2 {
            izv = sh2 - 1;
        }
    }
    if valid && !in_mesh {
        let mut t_enter = -big;
        let mut t_exit = big;
        let mut miss = false;
        if absx < fp {
            if r0x < ll0 || r0x >= ur0 {
                miss = true;
            }
        } else {
            let invd = 1.0 / dirx;
            let t1 = (ll0 - r0x) * invd;
            let t2 = (ur0 - r0x) * invd;
            let mut tnear = t2;
            let mut tfar = t1;
            if invd > 0.0 {
                tnear = t1;
                tfar = t2;
            }
            if tnear > t_enter {
                t_enter = tnear;
            }
            if tfar < t_exit {
                t_exit = tfar;
            }
        }
        if absy < fp {
            if r0y < ll1 || r0y >= ur1 {
                miss = true;
            }
        } else {
            let invd = 1.0 / diry;
            let t1 = (ll1 - r0y) * invd;
            let t2 = (ur1 - r0y) * invd;
            let mut tnear = t2;
            let mut tfar = t1;
            if invd > 0.0 {
                tnear = t1;
                tfar = t2;
            }
            if tnear > t_enter {
                t_enter = tnear;
            }
            if tfar < t_exit {
                t_exit = tfar;
            }
        }
        if absz < fp {
            if r0z < ll2 || r0z >= ur2 {
                miss = true;
            }
        } else {
            let invd = 1.0 / dirz;
            let t1 = (ll2 - r0z) * invd;
            let t2 = (ur2 - r0z) * invd;
            let mut tnear = t2;
            let mut tfar = t1;
            if invd > 0.0 {
                tnear = t1;
                tfar = t2;
            }
            if tnear > t_enter {
                t_enter = tnear;
            }
            if tfar < t_exit {
                t_exit = tfar;
            }
        }
        if miss || t_enter >= t_exit || t_exit <= 0.0 || t_enter >= total_length {
            valid = false;
        } else {
            let mut t_start = rt_zero + 1e-8_f64;
            if t_enter > 0.0 {
                t_start = t_enter + tiny;
            }
            traveled = t_start;
            curx = r0x + dirx * t_start;
            cury = r0y + diry * t_start;
            curz = r0z + dirz * t_start;
            let mut im2 = true;
            if curx < ll0 || curx >= ur0 {
                im2 = false;
            }
            if cury < ll1 || cury >= ur1 {
                im2 = false;
            }
            if curz < ll2 || curz >= ur2 {
                im2 = false;
            }
            if im2 {
                ixv = ((curx - ll0) * iw0) as u32;
                iyv = ((cury - ll1) * iw1) as u32;
                izv = ((curz - ll2) * iw2) as u32;
                if ixv >= sh0 {
                    ixv = sh0 - 1;
                }
                if iyv >= sh1 {
                    iyv = sh1 - 1;
                }
                if izv >= sh2 {
                    izv = sh2 - 1;
                }
            } else {
                valid = false;
            }
        }
    }

    // Seed per-axis distance-to-next-boundary + next index (Amanatides-Woo).
    let mut dstx = big;
    let mut dsty = big;
    let mut dstz = big;
    let mut nix = ixv;
    let mut niy = iyv;
    let mut niz = izv;
    if valid {
        if absx >= fp {
            if dirx > 0.0 {
                nix = ixv + 1;
                let boundary = ll0 + (ixv + 1) as f64 * w0;
                dstx = traveled + (boundary - curx) * invx;
            } else {
                let boundary = ll0 + ixv as f64 * w0;
                dstx = traveled + (boundary - curx) * invx;
                if ixv > 0 {
                    nix = ixv - 1;
                } else {
                    nix = sh0;
                }
            }
        }
        if absy >= fp {
            if diry > 0.0 {
                niy = iyv + 1;
                let boundary = ll1 + (iyv + 1) as f64 * w1;
                dsty = traveled + (boundary - cury) * invy;
            } else {
                let boundary = ll1 + iyv as f64 * w1;
                dsty = traveled + (boundary - cury) * invy;
                if iyv > 0 {
                    niy = iyv - 1;
                } else {
                    niy = sh1;
                }
            }
        }
        if absz >= fp {
            if dirz > 0.0 {
                niz = izv + 1;
                let boundary = ll2 + (izv + 1) as f64 * w2;
                dstz = traveled + (boundary - curz) * invz;
            } else {
                let boundary = ll2 + izv as f64 * w2;
                dstz = traveled + (boundary - curz) * invz;
                if izv > 0 {
                    niz = izv - 1;
                } else {
                    niz = sh2;
                }
            }
        }
    }

    // DDA walk. Safety-bounded by the max crossings a straight segment can make.
    let cap = sh0 + sh1 + sh2 + 8u32;
    let mut iter = 0u32;
    while valid && iter < cap {
        let mut min_dist = dstx;
        let mut min_dim = 0u32;
        if dsty < min_dist {
            min_dist = dsty;
            min_dim = 1u32;
        }
        if dstz < min_dist {
            min_dist = dstz;
            min_dim = 2u32;
        }
        let mut length_in_voxel = min_dist - traveled;
        if min_dist >= total_length {
            length_in_voxel = total_length - traveled;
        }
        if length_in_voxel > tiny {
            let voxel = (izv * sh1 + iyv) * sh0 + ixv;
            let lf = length_in_voxel / total_length;
            let contrib = d * lf * score * weight;
            let sc = contrib * scale;
            let mut sb = -((-sc + 0.5) as i64);
            if sc >= 0.0 {
                sb = (sc + 0.5) as i64;
            }
            let idx = src_base + base + voxel;
            src_acc[idx as usize].fetch_add(u64::reinterpret(sb));
        }
        if min_dist >= total_length {
            valid = false;
        } else {
            traveled = min_dist;
            if min_dim == 0u32 {
                ixv = nix;
                if ixv >= sh0 {
                    valid = false;
                } else {
                    dstx = dstx + tdx;
                    if dirx > 0.0 {
                        nix = ixv + 1;
                    } else if ixv > 0 {
                        nix = ixv - 1;
                    } else {
                        nix = sh0;
                    }
                }
            } else if min_dim == 1u32 {
                iyv = niy;
                if iyv >= sh1 {
                    valid = false;
                } else {
                    dsty = dsty + tdy;
                    if diry > 0.0 {
                        niy = iyv + 1;
                    } else if iyv > 0 {
                        niy = iyv - 1;
                    } else {
                        niy = sh1;
                    }
                }
            } else {
                izv = niz;
                if izv >= sh2 {
                    valid = false;
                } else {
                    dstz = dstz + tdz;
                    if dirz > 0.0 {
                        niz = izv + 1;
                    } else if izv > 0 {
                        niz = izv - 1;
                    } else {
                        niz = sh2;
                    }
                }
            }
        }
        iter += 1u32;
    }
}

/// `#[cube]` twin of `crate::common::tallies::rect_mesh_bin_at` /
/// `RegularRectangularMesh::get_bin`: the row-major voxel index for a point in
/// a rectangular mesh (packed descriptor at `mesh_params[mo..]`), or
/// `u32::MAX` if the point is outside the mesh. Used by the collision estimator
/// (issue #234). Floors are `as u32` casts of non-negative reals.
#[allow(clippy::manual_clamp)]
#[cube]
pub(crate) fn rect_mesh_bin_at_kernel(
    mesh_params: &[f64],
    mo: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> u32 {
    let mob = mo as usize;
    let ll0 = mesh_params[mob];
    let ll1 = mesh_params[mob + 1];
    let ll2 = mesh_params[mob + 2];
    let ur0 = mesh_params[mob + 3];
    let ur1 = mesh_params[mob + 4];
    let ur2 = mesh_params[mob + 5];
    let iw0 = mesh_params[mob + 6];
    let iw1 = mesh_params[mob + 7];
    let iw2 = mesh_params[mob + 8];
    let sh0 = mesh_params[mob + 12] as u32;
    let sh1 = mesh_params[mob + 13] as u32;
    let sh2 = mesh_params[mob + 14] as u32;
    let mut out = 4_294_967_295u32;
    let mut inside = true;
    if px < ll0 || px >= ur0 {
        inside = false;
    }
    if py < ll1 || py >= ur1 {
        inside = false;
    }
    if pz < ll2 || pz >= ur2 {
        inside = false;
    }
    if inside {
        let mut vx = ((px - ll0) * iw0) as u32;
        let mut vy = ((py - ll1) * iw1) as u32;
        let mut vz = ((pz - ll2) * iw2) as u32;
        if vx >= sh0 {
            vx = sh0 - 1;
        }
        if vy >= sh1 {
            vy = sh1 - 1;
        }
        if vz >= sh2 {
            vz = sh2 - 1;
        }
        out = (vz * sh1 + vy) * sh0 + vx;
    }
    out
}

// =====================================================================
// Cylindrical mesh scoring (issue #279)
//
// `#[cube]` mirrors of the plain-Rust twins in `crate::common::tallies`
// (`cyl_mesh_crossings` / `cyl_mesh_bin_at`), themselves faithful ports of
// `yamc_tallies::mesh::CylindricalMesh`. The `cyl_mesh_dda_twin` gate test
// validates that DDA in plain Rust; these kernels reproduce it under the cubecl
// constraints (1e300 sentinel for INFINITY, explicit if/else for min/abs,
// signed-round for the fixed-point cast, `atan2_f64`/`sin_f64`/`cos_f64`
// polyfills for the transcendentals). Descriptor layout: header
// `[origin[3], nr, nphi, nz, full_phi]` (`MESH_CYL_HEADER` = 7 words) then grids
// `r_grid[nr+1]`, `r_grid_sq[nr+1]`, `phi_grid[nphi+1]`, `z_grid[nz+1]`.
// =====================================================================

/// Bracket `v` into a cell of the strictly-increasing grid stored at
/// `mesh_params[off..off + n + 1]`: the index `i` with `grid[i] <= v < grid[i+1]`,
/// or `n` (a non-cell sentinel) when `v` is outside `[grid[0], grid[n])`. Twin of
/// `cyl_bracket` / `yamc_tallies::mesh::bracket`.
#[cube]
fn cyl_bracket_kernel(mesh_params: &[f64], off: u32, n: u32, v: f64) -> u32 {
    let base = off as usize;
    let lo = mesh_params[base];
    let hi = mesh_params[base + n as usize];
    let mut idx = n; // outside sentinel
    if v >= lo && v < hi {
        // partition_point(|g| g <= v): count leading edges <= v, cell = count-1.
        let mut count = 0u32;
        let mut i = 0u32;
        while i <= n {
            if mesh_params[base + i as usize] <= v {
                count += 1u32;
            }
            i += 1u32;
        }
        idx = count - 1u32;
    }
    idx
}

/// First root `> l` of `a·t² + 2b·t + (c - R²) = 0`, or the `1e300` sentinel
/// when there is none. Twin of `CylMeshView::shell_crossing`.
// `l - l` forces a RUNTIME-typed 0.0 for the sentinel (see rt_zero in
// mesh_rect_score_src_acc); a literal would make cubecl expand it as comptime.
#[allow(clippy::eq_op)]
#[cube]
fn cyl_shell_crossing(a: f64, b: f64, c: f64, r_sq: f64, r: f64, l: f64) -> f64 {
    let rt_zero = l - l;
    let big = rt_zero + 1e300_f64;
    let fp = 1e-14_f64;
    let coincident = 1e-10_f64;
    let mut out = big;
    if r > 0.0 && a >= fp {
        let pn = b / a;
        let disc = pn * pn - (c - r_sq) / a;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let t1 = -pn - sq;
            let t2 = -pn + sq;
            let thresh = l + coincident;
            if t1 > thresh {
                out = t1;
            } else if t2 > thresh {
                out = t2;
            }
        }
    }
    out
}

/// Distance `> l` at which the track crosses the half-plane at angle `phi`, or
/// the `1e300` sentinel (parallel, behind, or on the antipodal half-plane). Twin
/// of `CylMeshView::phi_crossing`.
#[allow(clippy::eq_op)] // `l - l` forces a runtime-typed 0.0 (see rt_zero).
#[cube]
fn cyl_phi_crossing(px: f64, py: f64, dx: f64, dy: f64, phi: f64, l: f64) -> f64 {
    let rt_zero = l - l;
    let big = rt_zero + 1e300_f64;
    let fp = 1e-14_f64;
    let coincident = 1e-10_f64;
    let s = sin_f64(phi);
    let co = cos_f64(phi);
    let denom = dx * s - dy * co;
    let mut adenom = denom;
    if denom < 0.0 {
        adenom = -denom;
    }
    let mut out = big;
    if adenom >= fp {
        let t = -(px * s - py * co) / denom;
        if t > l + coincident {
            let x = px + t * dx;
            let y = py + t * dy;
            if co * x + s * y > 0.0 {
                out = t;
            }
        }
    }
    out
}

/// `1` if the local-frame point `(x, y, zc)` lies inside the mesh, else `0`.
/// Twin of `CylMeshView::indices_local(...).is_some()`.
// `phi = phi + two_pi` stays explicit because cubecl needs the non-`+=` form.
#[allow(clippy::assign_op_pattern)]
#[cube]
fn cyl_inside(
    mesh_params: &[f64],
    r_off: u32,
    phi_off: u32,
    z_off: u32,
    nr: u32,
    nphi: u32,
    nz: u32,
    full_phi: u32,
    x: f64,
    y: f64,
    zc: f64,
) -> u32 {
    let fp = 1e-14_f64;
    let two_pi = std::f64::consts::TAU;
    let rho = (x * x + y * y).sqrt();
    let ir = cyl_bracket_kernel(mesh_params, r_off, nr, rho);
    let iz = cyl_bracket_kernel(mesh_params, z_off, nz, zc);
    let mut ok = 1u32;
    if ir >= nr {
        ok = 0u32;
    }
    if iz >= nz {
        ok = 0u32;
    }
    if rho >= fp {
        let mut phi = atan2_f64(y, x);
        if phi < 0.0 {
            phi = phi + two_pi;
        }
        let last_phi = mesh_params[(phi_off + nphi) as usize];
        if full_phi == 1u32 && phi >= last_phi {
            phi = last_phi - fp;
        }
        let iphi = cyl_bracket_kernel(mesh_params, phi_off, nphi, phi);
        if iphi >= nphi {
            ok = 0u32;
        }
    }
    ok
}

/// Smallest distance in `(l, total]` at which the track first lies inside the
/// mesh, or the `1e300` sentinel when it never does. Handles initial entry and
/// re-entry through the central hole / a φ wedge. Twin of
/// `CylMeshView::first_entry`. The `consider` closure is inlined per candidate
/// (cubecl has no closures / `&mut f64` params); every candidate keeps the
/// smallest valid `t`, so the fixed candidate order matches the twin's result.
#[cube]
// `l - l` forces a runtime-typed 0.0 for the sentinel (see rt_zero).
#[allow(clippy::too_many_arguments, clippy::eq_op)]
fn cyl_first_entry(
    mesh_params: &[f64],
    r_off: u32,
    rsq_off: u32,
    phi_off: u32,
    z_off: u32,
    nr: u32,
    nphi: u32,
    nz: u32,
    full_phi: u32,
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    a: f64,
    b: f64,
    c: f64,
    l: f64,
    total: f64,
) -> f64 {
    let rt_zero = l - l;
    let big = rt_zero + 1e300_f64;
    let fp = 1e-14_f64;
    let tiny = 1e-8_f64;
    let coincident = 1e-10_f64;
    let mut best = big;

    // Both bounding radial shells (inner index 0, outer index nr).
    let mut si = 0u32;
    while si < 2u32 {
        let shell = if si == 0u32 { 0u32 } else { nr };
        let r = mesh_params[(r_off + shell) as usize];
        let r_sq = mesh_params[(rsq_off + shell) as usize];
        if r > 0.0 && a >= fp {
            let pn = b / a;
            let disc = pn * pn - (c - r_sq) / a;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let t1 = -pn - sq;
                if t1 > l + coincident && t1 < best && t1 <= total {
                    let ex = px + (t1 + tiny) * dx;
                    let ey = py + (t1 + tiny) * dy;
                    let ez = pz + (t1 + tiny) * dz;
                    if cyl_inside(
                        mesh_params,
                        r_off,
                        phi_off,
                        z_off,
                        nr,
                        nphi,
                        nz,
                        full_phi,
                        ex,
                        ey,
                        ez,
                    ) == 1u32
                    {
                        best = t1;
                    }
                }
                let t2 = -pn + sq;
                if t2 > l + coincident && t2 < best && t2 <= total {
                    let ex = px + (t2 + tiny) * dx;
                    let ey = py + (t2 + tiny) * dy;
                    let ez = pz + (t2 + tiny) * dz;
                    if cyl_inside(
                        mesh_params,
                        r_off,
                        phi_off,
                        z_off,
                        nr,
                        nphi,
                        nz,
                        full_phi,
                        ex,
                        ey,
                        ez,
                    ) == 1u32
                    {
                        best = t2;
                    }
                }
            }
        }
        si += 1u32;
    }

    // φ end walls (only meaningful for a partial sector).
    if full_phi == 0u32 {
        let mut pi = 0u32;
        while pi < 2u32 {
            let phi_edge = if pi == 0u32 {
                mesh_params[phi_off as usize]
            } else {
                mesh_params[(phi_off + nphi) as usize]
            };
            let t = cyl_phi_crossing(px, py, dx, dy, phi_edge, l);
            if t > l + coincident && t < best && t <= total {
                let ex = px + (t + tiny) * dx;
                let ey = py + (t + tiny) * dy;
                let ez = pz + (t + tiny) * dz;
                if cyl_inside(
                    mesh_params,
                    r_off,
                    phi_off,
                    z_off,
                    nr,
                    nphi,
                    nz,
                    full_phi,
                    ex,
                    ey,
                    ez,
                ) == 1u32
                {
                    best = t;
                }
            }
            pi += 1u32;
        }
    }

    // z end planes.
    let mut adz = dz;
    if dz < 0.0 {
        adz = -dz;
    }
    if adz >= fp {
        let mut zi = 0u32;
        while zi < 2u32 {
            let zp = if zi == 0u32 {
                mesh_params[z_off as usize]
            } else {
                mesh_params[(z_off + nz) as usize]
            };
            let t = (zp - pz) / dz;
            if t > l + coincident && t < best && t <= total {
                let ex = px + (t + tiny) * dx;
                let ey = py + (t + tiny) * dy;
                let ez = pz + (t + tiny) * dz;
                if cyl_inside(
                    mesh_params,
                    r_off,
                    phi_off,
                    z_off,
                    nr,
                    nphi,
                    nz,
                    full_phi,
                    ex,
                    ey,
                    ez,
                ) == 1u32
                {
                    best = t;
                }
            }
            zi += 1u32;
        }
    }
    best
}

/// Analytic `(r, φ, z)` voxel walk of the step `r0 -> r0 + dir·d`, atomic-adding
/// `d · length_fraction · score · weight · scale` (signed fixed-point) into
/// `src_acc[src_base + base + voxel]` per voxel crossed. `#[cube]` twin of
/// `crate::common::tallies::cyl_mesh_crossings`; same signature shape as
/// `mesh_rect_score_src_acc`. Degenerate (sub-`2·tiny`) steps are dropped
/// (statistically negligible), matching the rect kernel.
#[cube]
// `total_length - total_length` forces a runtime-typed 0.0 (see rt_zero);
// `phi = phi + two_pi` stays explicit because cubecl needs the non-`+=` form.
#[allow(clippy::too_many_arguments, clippy::eq_op, clippy::assign_op_pattern)]
pub(crate) fn cyl_mesh_score_src_acc(
    mesh_params: &[f64],
    mo: u32,
    r0x: f64,
    r0y: f64,
    r0z: f64,
    dirx: f64,
    diry: f64,
    dirz: f64,
    d: f64,
    score: f64,
    weight: f64,
    base: u32,
    src_base: u32,
    scale: f64,
    src_acc: &mut [Atomic<u64>],
) {
    let tiny = 1e-8_f64;
    let fp = 1e-14_f64;
    let coincident = 1e-10_f64;
    let two_pi = std::f64::consts::TAU;

    let mob = mo as usize;
    let ox = mesh_params[mob];
    let oy = mesh_params[mob + 1];
    let oz = mesh_params[mob + 2];
    let nr = mesh_params[mob + 3] as u32;
    let nphi = mesh_params[mob + 4] as u32;
    let nz = mesh_params[mob + 5] as u32;
    let full_phi = if mesh_params[mob + 6] > 0.5 {
        1u32
    } else {
        0u32
    };
    let r_off = mo + 7u32;
    let rsq_off = r_off + nr + 1u32;
    let phi_off = rsq_off + nr + 1u32;
    let z_off = phi_off + nphi + 1u32;

    let r1x = r0x + dirx * d;
    let r1y = r0y + diry * d;
    let r1z = r0z + dirz * d;
    let ex = r1x - r0x;
    let ey = r1y - r0y;
    let ez = r1z - r0z;
    let total_length = (ex * ex + ey * ey + ez * ez).sqrt();
    let rt_zero = total_length - total_length;
    let big = rt_zero + 1e300_f64;

    // Local frame start `p` and radial quadratic coefficients ρ²=a·t²+2b·t+c.
    let px0 = r0x - ox;
    let py0 = r0y - oy;
    let pz0 = r0z - oz;
    let a = dirx * dirx + diry * diry;
    let b = px0 * dirx + py0 * diry;
    let c = px0 * px0 + py0 * py0;

    let mut run = true;
    if total_length < 2.0 * tiny {
        run = false; // drop degenerate steps
    }
    let mut l = rt_zero;
    let cap = 4u32 * (nr + nphi + nz) + 32u32;
    let mut iter = 0u32;
    while run && iter < cap {
        iter += 1u32;
        if l >= total_length {
            run = false;
        } else {
            // Probe just past the cursor so we are inside the new cell.
            let mut lp = l + tiny;
            if lp > total_length {
                lp = total_length;
            }
            let prx = px0 + lp * dirx;
            let pry = py0 + lp * diry;
            let prz = pz0 + lp * dirz;
            // indices_local(prx, pry, prz), inlined.
            let rho = (prx * prx + pry * pry).sqrt();
            let ir = cyl_bracket_kernel(mesh_params, r_off, nr, rho);
            let iz = cyl_bracket_kernel(mesh_params, z_off, nz, prz);
            let mut iphi = 0u32;
            let mut inside = true;
            if ir >= nr {
                inside = false;
            }
            if iz >= nz {
                inside = false;
            }
            if rho >= fp {
                let mut phi = atan2_f64(pry, prx);
                if phi < 0.0 {
                    phi = phi + two_pi;
                }
                let last_phi = mesh_params[(phi_off + nphi) as usize];
                if full_phi == 1u32 && phi >= last_phi {
                    phi = last_phi - fp;
                }
                iphi = cyl_bracket_kernel(mesh_params, phi_off, nphi, phi);
                if iphi >= nphi {
                    inside = false;
                }
            }

            if inside {
                // Distance to leave this cell = nearest wall crossing > l.
                let ri = mesh_params[(r_off + ir) as usize];
                let ro = mesh_params[(r_off + ir + 1u32) as usize];
                let rsi = mesh_params[(rsq_off + ir) as usize];
                let rso = mesh_params[(rsq_off + ir + 1u32) as usize];
                let dr_in = cyl_shell_crossing(a, b, c, rsi, ri, l);
                let dr_out = cyl_shell_crossing(a, b, c, rso, ro, l);
                let mut dr = dr_in;
                if dr_out < dr {
                    dr = dr_out;
                }
                let plo = mesh_params[(phi_off + iphi) as usize];
                let phiup = mesh_params[(phi_off + iphi + 1u32) as usize];
                let dphi_lo = cyl_phi_crossing(px0, py0, dirx, diry, plo, l);
                let dphi_hi = cyl_phi_crossing(px0, py0, dirx, diry, phiup, l);
                let mut dphi = dphi_lo;
                if dphi_hi < dphi {
                    dphi = dphi_hi;
                }
                let mut dz = big;
                let mut adz = dirz;
                if dirz < 0.0 {
                    adz = -dirz;
                }
                if adz >= fp {
                    let mut plane = mesh_params[(z_off + iz) as usize];
                    if dirz > 0.0 {
                        plane = mesh_params[(z_off + iz + 1u32) as usize];
                    }
                    let tz = (plane - pz0) / dirz;
                    if tz > l + coincident {
                        dz = tz;
                    }
                }
                // Radial perigee t* = -b/a: forced event (radial turnaround /
                // through-axis φ flip).
                let mut d_peri = big;
                if a >= fp {
                    let t_star = -b / a;
                    if t_star > l + coincident {
                        d_peri = t_star;
                    }
                }
                let mut dmin = dr;
                if dphi < dmin {
                    dmin = dphi;
                }
                if dz < dmin {
                    dmin = dz;
                }
                if d_peri < dmin {
                    dmin = d_peri;
                }
                if total_length < dmin {
                    dmin = total_length;
                }
                let seg = dmin - l;
                if seg > tiny {
                    let voxel = (iz * nphi + iphi) * nr + ir;
                    let lf = seg / total_length;
                    let contrib = d * lf * score * weight;
                    let sc = contrib * scale;
                    let mut sb = -((-sc + 0.5) as i64);
                    if sc >= 0.0 {
                        sb = (sc + 0.5) as i64;
                    }
                    let idx = src_base + base + voxel;
                    src_acc[idx as usize].fetch_add(u64::reinterpret(sb));
                }
                if dmin >= total_length {
                    run = false;
                } else {
                    l = dmin;
                }
            } else {
                // Outside: jump to the next boundary that lands us inside.
                let te = cyl_first_entry(
                    mesh_params,
                    r_off,
                    rsq_off,
                    phi_off,
                    z_off,
                    nr,
                    nphi,
                    nz,
                    full_phi,
                    px0,
                    py0,
                    pz0,
                    dirx,
                    diry,
                    dirz,
                    a,
                    b,
                    c,
                    l,
                    total_length,
                );
                if te < total_length {
                    l = te;
                } else {
                    run = false;
                }
            }
        }
    }
}

/// `#[cube]` twin of `crate::common::tallies::cyl_mesh_bin_at` /
/// `CylindricalMesh::get_bin`: the flat `(iz*nphi+iphi)*nr+ir` voxel index for a
/// point, or `u32::MAX` if outside the mesh. Used by the collision estimator.
// `phi = phi + two_pi` stays explicit because cubecl needs the non-`+=` form.
#[allow(clippy::assign_op_pattern)]
#[cube]
pub(crate) fn cyl_mesh_bin_at_kernel(
    mesh_params: &[f64],
    mo: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> u32 {
    let fp = 1e-14_f64;
    let two_pi = std::f64::consts::TAU;
    let mob = mo as usize;
    let ox = mesh_params[mob];
    let oy = mesh_params[mob + 1];
    let oz = mesh_params[mob + 2];
    let nr = mesh_params[mob + 3] as u32;
    let nphi = mesh_params[mob + 4] as u32;
    let nz = mesh_params[mob + 5] as u32;
    let full_phi = if mesh_params[mob + 6] > 0.5 {
        1u32
    } else {
        0u32
    };
    let r_off = mo + 7u32;
    let phi_off = r_off + 2u32 * (nr + 1u32);
    let z_off = phi_off + nphi + 1u32;

    let x = px - ox;
    let y = py - oy;
    let zc = pz - oz;
    let rho = (x * x + y * y).sqrt();
    let ir = cyl_bracket_kernel(mesh_params, r_off, nr, rho);
    let iz = cyl_bracket_kernel(mesh_params, z_off, nz, zc);
    let mut out = 4_294_967_295u32;
    let mut inside = true;
    if ir >= nr {
        inside = false;
    }
    if iz >= nz {
        inside = false;
    }
    let mut iphi = 0u32;
    if rho >= fp {
        let mut phi = atan2_f64(y, x);
        if phi < 0.0 {
            phi = phi + two_pi;
        }
        let last_phi = mesh_params[(phi_off + nphi) as usize];
        if full_phi == 1u32 && phi >= last_phi {
            phi = last_phi - fp;
        }
        iphi = cyl_bracket_kernel(mesh_params, phi_off, nphi, phi);
        if iphi >= nphi {
            inside = false;
        }
    }
    if inside {
        out = (iz * nphi + iphi) * nr + ir;
    }
    out
}

#[allow(clippy::manual_clamp, clippy::assign_op_pattern, clippy::identity_op)]
#[cube(launch_unchecked)]
pub(crate) fn multi_cell_transport_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    positions_in: &[f64],
    directions_in: &[f64],
    cell_aabbs: &[f64],
    cell_to_material: &[u32],
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    bvh_prim_indices: &[u32],
    bvh_unbounded: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    surface_boundaries: &[u32],
    // Flat per-cell CSG region program (see `region_eval`). Used to
    // confirm point-in-region during cell finding so nested / overlapping
    // cells resolve to the same cell the CPU picks, not just the first
    // matching AABB.
    region_program: &[u32],
    log_energy_grid: &[f64],
    // COARSE grid backing the sparse per-MT inelastic buffers
    // (xs_inelastic_per_mt_sparse, yield_per_mt_sparse). Issue #212: every
    // material carries its OWN coarse grid (its finest single per-nuclide grid),
    // concatenated tight here; the coarse bracket is keyed per material via
    // `coarse_meta`, so no material's inelastic thresholds are smeared onto
    // another's grid. Equal to `log_energy_grid` for a single-nuclide
    // single-material problem.
    coarse_log_energy_grid: &[f64],
    // Packed `[n_mat × COARSE_META_COLS]` (stride 3u32) per-material coarse-grid
    // descriptor (issue #212). Columns: 0 = base into `coarse_log_energy_grid`,
    // 1 = this material's coarse length, 2 = base into `permt_meta` ROWS for the
    // material's first slab (= nuc_off * MT_INELASTIC_COUNT). Single-material
    // problems have `[0, coarse_len, 0]`, so indexing is byte-identical.
    coarse_meta: &[u32],
    // FINE grid backing the resonance-critical buffers: the per-material
    // aggregate macro XS below (`xs_elastic_per_material` etc.) and the
    // per-(material, nuclide) `nuc_macro_total` / `nuc_partial_xs` (issue #212).
    // Every material carries its OWN fine grid (its union grid), concatenated
    // tight here; the fine bracket and the aggregate-XS / nuc-buffer strides are
    // keyed per material via `fine_meta`, so no material's resonance structure is
    // smeared onto another's grid. Distinct from the GLOBAL `log_energy_grid`
    // above, which is retained for the grid-shared score / photon / decay
    // lookups. Equal to `log_energy_grid` for a single-material problem.
    fine_log_energy_grid: &[f64],
    // Packed `[n_mat × FINE_META_COLS]` (stride 3u32) per-material fine-grid
    // descriptor (issue #212). Columns: 0 = base into `fine_log_energy_grid`
    // (ALSO the per-material aggregate-XS row base), 1 = this material's fine
    // length, 2 = base into `nuc_macro_total` (element units) for the material's
    // first slab. Single-material problems have `[0, fine_len, 0]`, so indexing
    // is byte-identical to the old single shared fine grid.
    fine_meta: &[u32],
    // Aggregate macro XS below are flat TIGHT CSR `[sum_m fine_n_m]` on the FINE
    // grid: material `m`'s row starts at `fine_meta[m][COL_FINE_GRID_OFFSET]` and
    // is `fine_meta[m][COL_FINE_N]` wide. (Was `[n_materials × n_grid]`.)
    xs_elastic_per_material: &[f64],
    xs_absorption_per_material: &[f64],
    // Aggregated inelastic xs, flat `[n_materials × n_grid]` (sum
    // across MT slots in `xs_inelastic_per_mt`). Used to pick the
    // inelastic branch vs elastic / absorption.
    xs_inelastic_per_material: &[f64],
    // Total fission xs (sum across MT 18 / 19 / 20 / 21 / 38), flat
    // `[n_materials × n_grid]`. Pulls fission out of the absorption
    // branch into its own collision sample. Zero everywhere for
    // non-fissionable materials -- kernel never selects the branch.
    xs_fission_per_material: &[f64],
    // Average prompt neutrons per fission ν̄(E), flat `[n_materials ×
    // n_grid]`. The kernel multiplies the surviving neutron's weight
    // by this when sampling fission -- variance-reduction equivalent
    // of emitting `nu_bar` independent particles.
    nu_bar_per_material: &[f64],
    // Delayed-neutron fraction beta(E) = nu_d(E) / nu_t(E), same flat
    // `[n_materials x n_grid]` shape and fine-grid indexing as
    // `nu_bar_per_material` (issue #364). Per fission progeny the kernel draws one
    // uniform against this and takes the outgoing energy from the DELAYED chi row
    // instead of the prompt one when it lands below beta. Zero everywhere for a
    // material with no delayed data, which costs no draw and leaves the stream
    // unchanged.
    beta_delayed_per_material: &[f64],
    // Per-material fission outgoing-energy sampler dispatch.
    //   `fission_eout_kind == EOUT_KIND_CONTINUOUS_TABULAR (1)`:
    //     kernel samples E_out from the tabulated `(x, cdf, p)` buffers
    //     using the per-row interp-aware inversion (histogram or lin-lin
    //     via `fission_eout_interp`), the same scheme as the inelastic
    //     ContinuousTabular eout path. Stochastic E_in bracket pick
    //     between adjacent slices. (Also covers the
    //     `CorrelatedAngleEnergy` E_out marginal -- same buffer layout.)
    //   `fission_eout_kind == EOUT_KIND_MAXWELL (6)`:
    //     closed-form Maxwell χ `√E·exp(-E/θ)`. θ(E_in) is tabulated in
    //     `fission_eout_x` column 0 (one value per
    //     `fission_eout_energy_grid` point) and the scalar restriction
    //     energy `u` in `fission_eout_cdf[0]`. Samples via the shared
    //     `maxwell_rejection_draw` (same helper as the inelastic path).
    //   `fission_eout_kind == EOUT_KIND_EVAPORATION (4)`:
    //     closed-form Evaporation χ `E·exp(-E/θ)`; same θ / u packing as
    //     Maxwell, sampled via the shared `evaporation_rejection_draw`.
    //   `fission_eout_kind == EOUT_KIND_WATT (7)`:
    //     kernel uses the Watt-rejection branch with the per-material
    //     `(watt_a, watt_b)` from `mat_f64_meta`. Retained as a fallback
    //     for nuclides whose χ uses none of the above encodings.
    fission_eout_kind_per_material: &[u32],
    fission_eout_n_energies_per_material: &[u32],
    // Two chi ROWS per material (issue #364): row `2*mat` is the prompt spectrum
    // and row `2*mat + 1` the delayed groups' folded spectrum, so every
    // `*_per_material` buffer below is indexed by chi row, not by material. A
    // material with no delayed data carries an empty delayed row and `beta == 0`.
    //
    // Tight CSR (issue #104): `fission_eout_ae_offset[chi_row]` is the row's
    // first ae-row into `fission_eout_n_x` / `energy_grid`;
    // `fission_eout_x_offset[ae_row]` is the ae-row's first (x, cdf) point.
    fission_eout_ae_offset: &[u32],
    fission_eout_energy_grid_per_material: &[f64],
    fission_eout_n_x_per_material: &[u32],
    fission_eout_x_offset: &[u32],
    fission_eout_x_per_material: &[f64],
    fission_eout_cdf_per_material: &[f64],
    // PDF alongside the CDF (same shape as `fission_eout_x`), and the
    // per-ae-row interpolation code (0 histogram, 1 lin-lin) driving the
    // interp-aware within-bin inversion in `sample_fission_chi`.
    fission_eout_p_per_material: &[f64],
    fission_eout_interp_per_material: &[u32],
    // Packed `[n_mat × MAT_F64_COLS]` buffer of per-material f64s.
    // Layout: `mat_f64_meta[mat_idx * MAT_F64_COLS + col]`.
    //   COL_TARGET_MASS    : neutron-mass-units target mass (used in
    //                        elastic kinematics + free-gas threshold)
    //   COL_TEMPERATURE_K  : material temperature in Kelvin (drives
    //                        free-gas thermal scattering branch)
    //   COL_FISSION_A      : Watt-spectrum `a` (eV) for fission
    //                        χ-spectrum sampling -- non-fissionable
    //                        materials carry a default that's never
    //                        read (σ_f = 0)
    //   COL_FISSION_B      : Watt-spectrum `b` (1/eV)
    // Pre-pack each was its own per-material f64 binding; the pack
    // saves 3 storage-buffer descriptor bindings.
    mat_f64_meta: &[f64],
    // SPARSE per-MT inelastic xs values (issue #212). Concatenated tight in
    // (slab, MT slot) order: each (slab, MT slot) stores only its nonzero
    // (above-threshold) coarse-grid range, located by `permt_meta`. The dense
    // `[n_slab × MT_INELASTIC_COUNT × coarse_n]` buffer was 90-99.9% zeros and
    // overran the ~4 GB `maxStorageBufferRange`; the sparse lookup returns 0
    // outside the stored range, matching the dense zeros exactly.
    xs_inelastic_per_mt_sparse: &[f64],
    // Per-MT Q-value, flat `[n_slab × MT_INELASTIC_COUNT]`. Slot
    // `k` is MT `MT_INELASTIC_FIRST + k`'s Q-value, used in the
    // closed-form inelastic energy formula.
    q_inelastic_per_mt: &[f64],
    // SPARSE per-MT outgoing-neutron yield ν(E) values (issue #212). Parallel to
    // `xs_inelastic_per_mt_sparse`, located by the SAME `permt_meta` row. Stored
    // over each slot's nonzero-XS range; the kernel uses 1.0 outside that range
    // (the dense default, since the dense `yield_per_mt` was 1.0 wherever XS was
    // 0), so the yield interpolation stays bit-identical.
    yield_per_mt_sparse: &[f64],
    // Packed `[n_slab × MT_INELASTIC_COUNT × PERMT_META_COLS]` (stride 3u32)
    // per-(slab, MT slot) sparse descriptor (issue #212): col 0 = value_offset
    // (base into `xs_inelastic_per_mt_sparse` / `yield_per_mt_sparse`), col 1 =
    // i_start (first nonzero coarse-grid index, relative to the material's coarse
    // grid), col 2 = n_stored (contiguous stored-point count; 0 = absent slot).
    // Indexed `permt_meta[(slab * MT_INELASTIC_COUNT + slot) * 3 + col]`.
    permt_meta: &[u32],
    // Per-MT angular distribution (Tabular `(mu, cdf)` slices on a
    // tabulated incident-energy grid). Tight variable-length CSR layout
    // (issue #104): per-(slab,MT) `angle_n_energies` / `angle_ae_offset`
    // index into the back-to-back ae-row arrays, and per-ae-row
    // `angle_n_mu` / `angle_mu_offset` index the back-to-back mu-point
    // arrays. `angle_n_energies[slab * MT_INELASTIC_COUNT + slot] = 0`
    // means the kernel falls back to isotropic-in-CM sampling.
    //
    // #74 Stage 2b: every per-MT inelastic pool below (angle / eout / corr /
    // km / evap / maxwell / watt / nbps + xs/q/yield) is keyed per-(material,
    // nuclide) SLAB, not per material -- the leading `mat_slot` factor is the
    // struck nuclide's global `slab`, not `mat_idx`. Single-nuclide materials
    // have `slab == mat_idx`'s old row, so the layout (and result) is
    // byte-identical there. `n_slab = sum_mat n_nuclides`.
    //
    // Slot indexing:
    //   mat_slot = slab * 41 + slot              // [n_slab × 41]
    //   ae_off   = mat_slot * 32 + ae_idx        // [n_slab × 41 × 32]
    //   mu_off   = ae_off * 33                   // [n_slab × 41 × 32 × 33]
    angle_n_energies: &[u32],
    // CSR base: global ae-row where (slab, MT) slot's rows start in the tight
    // angle_energy_grid / n_mu / interp arrays (issue #104). n_ae for the slot
    // is angle_n_energies[mat_slot].
    angle_ae_offset: &[u32],
    angle_energy_grid: &[f64],
    angle_n_mu: &[u32],
    // CSR base: index into the tight angle_mu / cdf / pdf arrays where ae-row's
    // (mu, cdf, pdf) points start (issue #104). Row length is angle_n_mu[ae].
    angle_mu_offset: &[u32],
    angle_mu: &[f64],
    angle_cdf: &[f64],
    // Per-point PDF for per-MT inelastic angular table, same shape
    // as `angle_cdf`. Needed for the quadratic LinLin CDF inversion
    // that mirrors CPU's `TabulatedAngleDistribution::sample` in
    // `reaction_product.rs:128-139`.
    angle_pdf: &[f64],
    angle_interp: &[u32],
    // Packed `[n_mat × MT_INELASTIC_COUNT × MT_SLOT_U32_COLS]` buffer
    // holding every per-MT-slot u32 the kernel reads (eout
    // discriminator, eout/corr/km/evap n_energies, scatter-in-CM
    // flag, n-body count). Indexed as
    // `mt_slot_u32_meta[(mat_slot) * MT_SLOT_U32_COLS + COL_*]`.
    // Pre-pack each of these was its own per-MT-slot u32 binding;
    // the pack saves 6 storage-buffer descriptor bindings, head-room the
    // kernel needs to keep adding distribution samplers without
    // tripping the per-stage descriptor budget.
    //
    // Per-MT outgoing-energy distribution (slice C). `COL_EOUT_KIND`
    // dispatches between the closed-form Q-value energy
    // (`EOUT_KIND_LEVEL_INELASTIC = 0`) and CDF-inversion sampling
    // from a tabulated `(E_out, cdf)` grid
    // (`EOUT_KIND_CONTINUOUS_TABULAR = 1`). When the table is
    // present and selected, the kernel does stochastic
    // incident-energy bracket interpolation + CDF inversion and
    // produces an `E_out` in the same frame as the angular table --
    // `COL_SCATTER_IN_CM` then drives the optional CM→lab
    // conversion.
    //
    // Tight CSR layout for the eout buffers (issue #104):
    //   mat_slot    = mat_idx * 41 + slot          [n_mat × 41]
    //   eg_off_e    = eout_ae_offset[mat_slot]      (slot's first ae-row)
    //   x_off       = eout_x_offset[eg_off_e + bin] (row's first x-point)
    // No fixed per-axis stride.
    mt_slot_u32_meta: &[u32],
    // CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    // start in the tight eout_energy_grid / n_x / interp / n_discrete arrays
    // (issue #104). n_eout for the slot is the packed COL_EOUT_N_ENERGIES.
    eout_ae_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    // CSR base: index into the tight eout_x / cdf / p arrays where ae-row's
    // (x, cdf, p) points start (issue #104). Row length is eout_n_x[ae].
    eout_x_offset: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    // Per-MT `histogram_interp` flag (1 = histogram in E_in, 0 = lin-lin).
    // Length `n_mat × MT_INELASTIC_COUNT`. When set the kernel skips the
    // stochastic E_in bracket pick and the bracket-bound stretch,
    // mirroring CPU's `ContinuousTabular::sample`.
    eout_histogram_interp: &[u32],
    // Per-point PDF for the eout CDF. Same shape as `eout_x` and
    // `eout_cdf`. Needed by the quadratic LinLin CDF inversion that
    // mirrors CPU's `Tabular::sample`.
    eout_p: &[f64],
    // Per (MT slot, E_in slice) interpolation discriminant for the
    // inner Tabular (0 = Histogram, 1 = LinLin). Same encoding as
    // `angle_interp` / `corr_mu_interp`. One entry per ae-row
    // (tight CSR, issue #104).
    eout_interp: &[u32],
    // Per (MT slot, E_in slice) discrete-line prefix count. The first
    // `n_discrete` bins of each Tabular are discrete photon lines
    // (delta functions); kernel returns the bin endpoint and skips
    // bracket-bound stretch when the sampled bin is `< n_discrete`.
    // One entry per ae-row (tight CSR, issue #104).
    eout_n_discrete: &[u32],
    // Per-MT correlated angle-energy distribution (slice D). When
    // `eout_kind == EOUT_KIND_CORRELATED = 2` and
    // `mt_slot_u32_meta[..COL_CORR_N_ENERGIES] > 0` for a slot, the
    // kernel samples `E_out` from `corr_x` / `corr_cdf` (same shape
    // pattern as `eout_*` but with smaller caps) and `mu` from a
    // per-`(E_in, E_out)` angular sub-table (`corr_n_mu`, `corr_mu`,
    // `corr_mu_cdf`). The slice-B per-MT angular table is bypassed
    // for these slots since `mu` is joint with `E_out`.
    //
    // Tight CSR layout for the corr buffers (issue #104), three nesting
    // levels (per (slab,MT) slot -> incident-energy ae-rows -> E_out
    // x-points -> mu points):
    //   mat_slot   = mat_idx * 41 + slot           [n_mat × 41]
    //   eg_off_c   = corr_ae_offset[mat_slot]       (slot's first ae-row)
    //   x_off      = corr_x_offset[eg_off_c + bin]  (ae-row's first x-point)
    //   mu_off     = corr_mu_offset[x_off + j]      (x-point's first mu point)
    // No fixed per-axis stride.
    // CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    // start in the tight corr_energy_grid / n_x / interp / n_discrete arrays.
    corr_ae_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    // CSR base: global x-point where ae-row's (x, cdf, p) / n_mu / mu_interp
    // entries start. Row length is corr_n_x[ae-row].
    corr_x_offset: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    // Per-point PDF for `corr_x` / `corr_cdf`. Used by the quadratic
    // LinLin CDF inversion that mirrors CPU's
    // `Tabular::sample_with_discrete_info`. Shape same as `corr_cdf`.
    corr_p: &[f64],
    // Per (MT slot, E_in slice) interpolation discriminant for the
    // outgoing-energy Tabular (0 = Histogram, 1 = LinLin). Same
    // encoding as `angle_interp` / `eout_interp`. One entry per ae-row
    // (tight CSR, issue #104).
    corr_interp: &[u32],
    // Per (MT slot, E_in slice) discrete-line prefix count. Same
    // semantics as `eout_n_discrete`: bins `< n_discrete` are
    // discrete delta functions, kernel returns bin endpoint and
    // skips bracket-bound stretch.
    corr_n_discrete: &[u32],
    corr_n_mu: &[u32],
    // CSR base: global mu point where the x-point's (mu, cdf, pdf) sub-table
    // starts. Indexed by the global x-point index (= corr_x_offset[ae] + j).
    corr_mu_offset: &[u32],
    corr_mu: &[f64],
    corr_mu_cdf: &[f64],
    corr_mu_pdf: &[f64],
    corr_mu_interp: &[u32],
    // The per-MT scatter-frame flag (`1` = tabulated angular
    // distribution is CM, `0` = lab) lives in `mt_slot_u32_meta` at
    // `COL_SCATTER_IN_CM`. CM means the kernel applies CM→lab
    // two-body kinematics with the closed-form CM-frame outgoing
    // energy `E_cm = (A/(A+1))^2 * (E - (A+1)/A * |Q|)`; lab means
    // `mu` and the closed-form energy are used directly.
    //
    // Per-material elastic (MT 2) angular distribution. Same tight
    // variable-length CSR rules as the inelastic per-MT angular
    // buffers but for a single MT (no slot dimension): per-slab
    // `elastic_angle_n_energies` / `elastic_angle_ae_offset` index the
    // back-to-back ae-row arrays, per-ae-row `elastic_angle_n_mu` /
    // `elastic_angle_mu_offset` index the back-to-back mu-point arrays.
    // `n_ae == 0` means no tabulated data -- the kernel falls
    // back to isotropic-in-CM `mu_cm = 1 - 2·xi3`. Tabulated data is
    // the norm; isotropic only fires when the nuclide library
    // genuinely lacks an MT 2 angular table.
    elastic_angle_n_energies: &[u32],
    // CSR base: global ae-row where slab s's incident-energy rows start in
    // the tight elastic_angle_energy_grid / n_mu / interp arrays (issue
    // #104). n_ae for slab s is elastic_angle_n_energies[s].
    elastic_angle_ae_offset: &[u32],
    elastic_angle_energy_grid: &[f64],
    elastic_angle_n_mu: &[u32],
    // CSR base: index into the tight elastic_angle_mu / cdf / pdf arrays
    // where ae-row's (mu, cdf, pdf) points start (issue #104). Row length
    // is elastic_angle_n_mu[ae].
    elastic_angle_mu_offset: &[u32],
    elastic_angle_mu: &[f64],
    elastic_angle_cdf: &[f64],
    // Per-point PDF for the elastic angular table, same shape as
    // `elastic_angle_cdf`. Needed for the quadratic LinLin CDF
    // inversion that mirrors CPU's `TabulatedAngleDistribution::sample`
    // in `reaction_product.rs:128-139`.
    elastic_angle_pdf: &[f64],
    // Per (incident-energy index) interpolation discriminant for the
    // elastic angular table. `0` = histogram, `1` = LinLin (matches
    // `ANGLE_INTERP_HISTOGRAM` / `ANGLE_INTERP_LINLIN` in
    // `nuclide_xs.rs`). LinLin uses the quadratic inversion formula
    // and reads `elastic_angle_pdf`; Histogram uses the standard
    // `x_i + (c − c_i) / p_i` form (also reads PDF).
    elastic_angle_interp: &[u32],
    // Per-material temperature in Kelvin lives in `mat_f64_meta` at
    // `COL_TEMPERATURE_K`. Drives the kernel's free-gas thermal
    // scattering branch: when a neutron's energy drops below
    // `free_gas_threshold · K_B · T` (and the target is heavier than
    // the neutron), the kernel samples a target velocity from a CXS Maxwell-
    // Boltzmann distribution and runs full vector-based CM-frame
    // elastic kinematics. Above the threshold the closed-form
    // A-mass formula is used, identical to the pre-free-gas
    // behaviour, so RNG-stream bit-equivalence is preserved at MeV
    // energies.
    //
    // Per-MT Kalbach-Mann correlated angle-energy distribution.
    // When `eout_kind[mat_slot] == EOUT_KIND_KALBACH_MANN` the
    // kernel samples (E_out_cm, mu_cm) from these buffers
    // (replacing the level-inelastic closed-form energy and any
    // slice-B angular table). Tight CSR layout (issue #104):
    //   mat_slot   = mat_idx * MT_INELASTIC_COUNT + slot
    //   eg_off_k   = km_ae_offset[mat_slot]      (slot's first ae-row)
    //   x_off      = km_x_offset[eg_off_k + bin] (row's first x-point)
    // No fixed per-axis stride. CM-frame semantics follow
    // `COL_SCATTER_IN_CM`; the existing CM→lab block downstream applies
    // the conversion.
    // CSR base: global ae-row where the (slab, MT) slot's incident-energy
    // rows start in the tight km_energy_grid / n_x / interp / n_discrete
    // arrays. n_kae for the slot is the packed COL_KM_N_ENERGIES.
    km_ae_offset: &[u32],
    km_energy_grid: &[f64],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_n_x: &[u32],
    // CSR base: index into the tight km_x / p / c / r / a arrays where
    // ae-row's (x, p, c, r, a) points start. Row length is km_n_x[ae].
    km_x_offset: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    // Per-MT Evaporation distribution data. Tight CSR layout (issue #104):
    //   COL_EVAP_N_ENERGIES (in mt_slot_u32_meta) : θ(E_in) point count n_tae
    //   eg_off    = evap_ae_offset[mat_slot]      (slot's first E_in row)
    //   theta_off = evap_theta_offset[mat_slot] + comp * n_tae (component row)
    //   evap_energy_grid[eg_off + i]
    //   evap_theta[theta_off + i]
    //   evap_u[eg_off + i]                        : restriction energy (eV),
    //     tabulated per incident-energy point. Per-grid (not a per-slot
    //     scalar) so products that gate several Evaporation laws by
    //     applicability over incident energy -- e.g. Na23 MT 91, where a
    //     low-u law is active at 14 MeV and a high-u law lower down -- get
    //     the right outgoing-energy band at each incident energy, matching
    //     the CPU's per-collision applicability pick.
    // Sampler runs the standard Maxwell-style rejection algorithm
    // (`E_out = -ln((1 - v·xi1)·(1 - v·xi2)) · θ`, accept when
    // `result <= y = (E_in - u) / θ`). mu_sampled stays at the
    // slice-B / isotropic value already computed above (UAE
    // angular table when present, isotropic 1 - 2·xi3 otherwise).
    // CSR base: per-(slab,MT) global E_in-row start into evap_energy_grid /
    // evap_u (issue #104). Slot row count is COL_EVAP_N_ENERGIES.
    evap_ae_offset: &[u32],
    // CSR base: per-(slab,MT) global start into the component-major evap_theta
    // (issue #104). Component `c`'s row begins at
    // `evap_theta_offset[mat_slot] + c * n_tae`.
    evap_theta_offset: &[u32],
    evap_energy_grid: &[f64],
    evap_theta: &[f64],
    evap_u: &[f64],
    // Per-MT Maxwell distribution data. Tight CSR layout (issue #104):
    //   COL_MAXWELL_N_ENERGIES (in mt_slot_u32_meta) : θ(E_in) point count
    //   mg_off = maxwell_ae_offset[mat_slot]         (slot's first E_in row)
    //   maxwell_energy_grid[mg_off + i]
    //   maxwell_theta[mg_off + i]
    //   COL_MAXWELL_U (in mt_slot_f64_meta)          : restriction energy (eV)
    // Sampler runs the standard 3-uniform Maxwell rejection
    // (`E_out = -θ · (ln r1 + ln r2 · cos²(π/2 · r3))`, retry until
    // `E_out ≤ E_in - u`). mu stays at the slice-B / isotropic
    // value already computed above. Same buffer shape as the Evap
    // path but kept distinct so a future nuclide carrying both
    // Maxwell and Evaporation across different MTs needs no
    // special-casing.
    // CSR base: per-(slab,MT) global E_in-row start into maxwell_energy_grid /
    // maxwell_theta. Slot row count is COL_MAXWELL_N_ENERGIES.
    maxwell_ae_offset: &[u32],
    maxwell_energy_grid: &[f64],
    maxwell_theta: &[f64],
    // Per-MT inelastic Watt distribution data (ENDF File 5, Law 11).
    // Tight CSR layout (issue #104):
    //   COL_WATT_N_ENERGIES (in mt_slot_u32_meta) : a/b tabulation point count
    //   wg_off = watt_ae_offset[mat_slot]         (slot's first E_in row)
    //   watt_energy_grid[wg_off + i]
    //   watt_ab[(wg_off + i) * 2 + 0] : a(E_in) (eV)
    //   watt_ab[(wg_off + i) * 2 + 1] : b(E_in) (1/eV)
    //   COL_WATT_U (in mt_slot_f64_meta)              : restriction energy (eV)
    // `a` and `b` are interleaved into a single stride-2 buffer
    // (saves one descriptor binding).
    // Sampler: 4 RNG draws per rejection iteration (3 for the
    // Maxwell sample of `w` with parameter `a`, 1 for the uniform
    // correction). E_out = w + a²b/4 + (2 r4 - 1) · sqrt(a²b · w),
    // retry until E_out ≤ E_in - u. Bit-matches
    // `sample_watt_spectrum_params` in `yamc-nuclide::sampling`.
    // CSR base: per-(slab,MT) global E_in-row start into watt_energy_grid /
    // watt_ab. Slot row count is COL_WATT_N_ENERGIES.
    watt_ae_offset: &[u32],
    watt_energy_grid: &[f64],
    watt_ab: &[f64],
    // Packed `[n_mat × MT_INELASTIC_COUNT × MT_SLOT_F64_COLS]`
    // buffer of per-MT-slot f64 scalars. Layout:
    // `mt_slot_f64_meta[(mat_slot) * MT_SLOT_F64_COLS + col]`.
    // Replaces 4 separate per-slot bindings (`evap_u`, `maxwell_u`,
    // `watt_u`, `nbps_total_mass`) with one, freeing 3 descriptor
    // bindings -- without the consolidation the kernel hits the
    // per-stage descriptor budget on some drivers (silent aliasing
    // past the limit corrupts adjacent buffers).
    //
    // Per-MT NBodyPhaseSpace data lives in:
    //   COL_NBPS_N_BODIES   (in mt_slot_u32_meta) : 0 (not NBPS) or 3/4/5
    //   COL_NBPS_TOTAL_MASS (in mt_slot_f64_meta) : sum of product AWRs (Ap)
    // Sampler computes
    //   E_max = (Ap-1)/Ap · (A/(A+1)·E_in + Q),
    //   x = -ln(r1) - ln(r2)·cos(π/2·r3)²   (Maxwell sample)
    //   y = depends on n_bodies (3/4/5 cases)
    //   v = x/(x+y);  E_out = E_max·v
    // Mu is isotropic in CM (1 - 2·xi). H2 MT 16 is the canonical
    // example.
    mt_slot_f64_meta: &[f64],
    // Per-(material, nuclide) URR (unresolved resonance region) probability
    // table data, keyed on the global slab index (issue #210: URR is applied
    // to EVERY in-range URR nuclide of the material, each with an independent
    // probability-table band, so it is slab-keyed and aligns 1:1 with
    // `mat_nuclide_meta` / `nuc_partial_xs`). `urr_meta` carries the per-slab
    // flags (present, n_energies, n_cdf, interp, inelastic/absorption/
    // multiply-smooth bits, and the per-nuclide `ZA` stream key -- see
    // `URR_META_*` constants). Tight CSR (issue #104): `urr_ae_offset[slab]`
    // is the per-slab base into the concatenated `urr_energy_grid` and
    // `urr_cdf_offset[slab]` the base into `urr_cdf`; `n_energies` / `n_cdf`
    // come from `urr_meta` (NO MAX_URR_* caps). `urr_xs` reuses the
    // `urr_cdf_offset` base (the cdf cell index × `URR_XS_COLS`) and holds the
    // four columns (total / elastic / fission / n_gamma). The smooth baselines
    // the per-nuclide URR delta is taken against come from `nuc_partial_xs`
    // (already density-weighted macroscopic, per slab), so no per-material
    // smooth-micro buffer is needed. `urr_atom_density[slab]` scales a URR
    // nuclide's perturbed micro XS up to a macroscopic contribution.
    urr_meta: &[u32],
    urr_ae_offset: &[u32],
    urr_cdf_offset: &[u32],
    urr_energy_grid: &[f64],
    urr_cdf: &[f64],
    urr_xs: &[f64],
    urr_atom_density: &[f64],
    // Multi-tally pack (slice "tally-pack"). Each tally has its
    // own score kind, cell-bin map, energy-bin edges, and slot
    // range in `tally_out`. Per particle step the kernel walks
    // every tally and atomic-adds `track_length × score_xs ×
    // weight` (in fixed-point) into the indexed slot. See
    // `crate::common::tallies::TalliesPack` for the exact layout.
    tally_score_kinds: &[u32],
    tally_cell_to_bin: &[u32],
    tally_n_cells: &[u32],
    tally_edges_offsets: &[u32],
    tally_n_bins: &[u32],
    tally_log_edges: &[f64],
    tally_out_offsets: &[u32],
    // Per-tally auxiliary integer. For `SCORE_PER_MT` it's the slot
    // index into `xs_score_per_mt`; ignored for other score kinds.
    tally_score_data: &[u32],
    // Macroscopic XS for `SCORE_PER_MT` lookups, flat
    // `[n_materials × n_score_mts × n_grid]`. Padded to a single
    // all-zero slot when no tally requires per-MT scoring (cubecl
    // rejects zero-length buffers).
    xs_score_per_mt: &[f64],
    // Per-tally fixed-point scale for the atomic-add accumulator.
    // Length `n_tallies`. Default scale (2^30) preserves ~9
    // digits of precision; KERMA-shape tallies (heating /
    // heating-local / damage-energy) override to 1.0 because
    // their eV-scale per-step contributions otherwise overflow
    // the u64 sum across millions of source particles.
    tally_fixed_point_scales: &[f64],
    // Per-tally estimator flag. `1` = collision estimator (score
    // `weight × score_xs / Σ_t` once per real collision), `0` =
    // track-length (the per-step `d × score_xs × weight` path).
    // The per-step scoring block is gated on `== 0` and the
    // collision branch runs a parallel loop for `== 1`, so a tally
    // never gets both contributions. Length `n_tallies`.
    tally_is_collision: &[u32],
    // Per-tally ENDF MT for `SCORE_PER_MT` tallies (`0` otherwise),
    // length `n_tallies`. Inside the URR window the smooth
    // `xs_score_per_mt` value loses the per-collision resonance
    // self-shielding the transport step sampled, biasing capture
    // high. When `urr_fired` and this MT is 102 (n,gamma) the kernel
    // scores the URR-perturbed macroscopic capture
    // (`urr_macro_capture`) the step actually used; for MT 27
    // (absorption) it scores `sigma_a + sigma_f` (the already-
    // perturbed absorption macro). The CPU twin has no URR block, so
    // `urr_fired` is always false there and the substitution is a
    // no-op (matched-stream bit-equivalence preserved).
    tally_score_mt: &[u32],
    // ----------------------- mesh (voxel) tally binning (issue #234) --------
    // Per-tally spatial mesh dimension. A `MESH_NONE` tally has
    // `tally_n_mesh[t] == 1` and `voxel_bin == 0`, so the flat index
    // `out_off + (cell_bin * n_bins + e) * n_mesh + voxel` collapses to the
    // pre-mesh `out_off + cell_bin * n_bins + e` (byte-identical). A mesh tally
    // fans a track-length step across the voxels it crosses (Amanatides-Woo DDA
    // over the step segment) and bins the collision estimator at the collision
    // point. `tally_mesh_params` packs the per-tally geometry descriptor (see
    // `TalliesPack::mesh_params`); `tally_mesh_params_offsets[t]..[t+1]` is
    // tally `t`'s slice. Mesh tallies only ever appear in `mesh_direct` launches.
    tally_n_mesh: &[u32],
    tally_mesh_kind: &[u32],
    tally_mesh_params_offsets: &[u32],
    tally_mesh_params: &[f64],
    // ----------------------- energy-function weighting (issue #271) ---------
    // Per-tally `energy_function=` / `dose_coefficients=` table. Unlike every
    // other per-tally payload here this is NOT a bin dimension: it multiplies
    // the score by a tabulated curve evaluated at the particle's LINEAR energy
    // and drops the event outright when the energy falls outside the table.
    // `tally_efunc_offsets[t]..[t+1]` is tally `t`'s slice, laid out
    // `[n_points, energy[n], coeffs[4*(n-1)]]`; an EMPTY range means the tally
    // has no such filter, so no separate presence flag is needed.
    tally_efunc_offsets: &[u32],
    tally_efunc_params: &[f64],
    // ----------------------- survival biasing -------------------------------
    // Runtime gate + weight parameters, length 3:
    //   [0] enable flag: `1.0` survival biasing on, `0.0` off.
    //   [1] weight_cutoff  (weight-cutoff Russian-roulette trigger).
    //   [2] weight_survive (roulette survivor weight).
    // When `survival_params[0] == 0.0` the reaction-selection block takes
    // the analog absorption-kill branch with the SAME RNG schedule and the
    // post-collision roulette is skipped (no extra draw), so a non-VR run
    // stays byte-identical. When `1.0`:
    //   - implicit capture: capture is never a terminal event; the
    //     scatter/fission selection is renormalised over
    //     `(sigma_e + sigma_i + sigma_f)` (the capture mass dropped) and the
    //     weight is multiplied by `(sigma_e + sigma_i + sigma_f) / sigma_t`
    //     (the GPU twin of CPU `weight *= xs.scatter / xs.total`; the fission
    //     branch still applies `weight *= nu_bar` downstream, the GPU's
    //     pre-discount fission-bank equivalent). A pure absorber at this
    //     energy (`sigma_e + sigma_i + sigma_f == 0`) still terminates.
    //   - weight-cutoff roulette: after the collision is fully processed, a
    //     still-alive particle below `weight_cutoff` is rouletted (one draw,
    //     only below the cutoff) -- survive with probability
    //     `weight / weight_survive` at `weight_survive`, else killed.
    survival_params: &[f64],
    // Free-gas resonance/thermal cutoff multiplier (model option, default
    // 400.0), a single-element buffer. The free-gas branch treats the target
    // as a free Maxwell gas below `free_gas_threshold[0] * kt` and as a cold
    // target at/above it. Passed as a 1-element f64 buffer (like
    // `survival_params`) because cubecl comptime values must be `Hash` and
    // `f64` is not; the CPU twin reads the same value from `TransportInputs`
    // (issue #102).
    free_gas_threshold: &[f64],
    // ----------------------- coupled neutron->photon (S4b) -----------------
    // Runtime gate (1 element). When `coupled_enabled[0] == 0` NO photon RNG
    // is drawn and NO bank write happens -- the neutron transport is then
    // byte-identical to a neutron-only run. When `1`, each REAL collision
    // emits secondary photons into the device particle bank using the
    // PRE-scatter energy / direction / position / weight. Photon sampling
    // runs on a CHILD PCG state forked from the neutron `state`, so the
    // neutron stream is never perturbed (the coupled-on neutron tallies equal
    // the neutron-only ones).
    coupled_enabled: &[u32],
    // Per-material aggregate photon-production macro xs on the shared grid,
    // concatenated material-major: material `m`'s row of `n_grid` values
    // starts at `photon_prod_base_per_material[m]`. Drives the photon COUNT
    // yield `y_t = photon_prod[i_grid] / sigma_t`.
    photon_prod: &[f64],
    // Per-(material, photon reaction) macro production xs, concatenated
    // material-major and row-major within a material; material `m`'s first row
    // is at `photon_rxn_base_per_material[m]` (in element units, a multiple of
    // `n_grid`). Read by the product-selection walk via the GLOBAL
    // `photon_prod_rxn_idx`.
    photon_rxn_xs: &[f64],
    // GLOBAL reaction-row index per packed product (already offset by the
    // material's rxn base on the host), concatenated material-major. Length =
    // total products across materials.
    photon_prod_rxn_idx: &[u32],
    // Per-product yield curve on the shared grid, product-major and
    // concatenated material-major. Index `i * n_grid + g` is GLOBAL.
    photon_prod_yield_grid: &[f64],
    // Per-product outgoing-energy metadata (S1/S3), concatenated material-major
    // (one entry per global product, except the AE/CT-strided buffers).
    photon_prod_eout_kind: &[u32],
    photon_prod_line_energy: &[f64],
    photon_prod_primary_flag: &[i32],
    photon_prod_awr: &[f64],
    // GLOBAL continuous-tabular slot per product (already offset by the
    // material's ct base on the host).
    photon_prod_dist_slot: &[u32],
    // Per-product angular table (tight CSR, issue #104), concatenated.
    // `photon_pa_ae_offset[product]` is the global ae-row base into the
    // per-row arrays; `photon_pa_mu_offset[ae_row]` the per-row mu-point base.
    photon_pa_n_energies: &[u32],
    photon_pa_ae_offset: &[u32],
    photon_pa_mu_offset: &[u32],
    photon_pa_energy_grid: &[f64],
    photon_pa_n_mu: &[u32],
    photon_pa_mu: &[f64],
    photon_pa_cdf: &[f64],
    photon_pa_pdf: &[f64],
    photon_pa_interp: &[u32],
    // Per-continuous-slot outgoing-energy table (tight CSR, issue #104),
    // concatenated. `photon_ct_ae_offset[slot]` is the global ae-row base;
    // `photon_ct_x_offset[ae_row]` the per-row outgoing-point base.
    photon_ct_ae_offset: &[u32],
    photon_ct_x_offset: &[u32],
    photon_ct_energy_grid: &[f64],
    photon_ct_n_x: &[u32],
    photon_ct_x: &[f64],
    photon_ct_cdf: &[f64],
    photon_ct_p: &[f64],
    photon_ct_interp: &[u32],
    photon_ct_n_discrete: &[u32],
    photon_ct_n_eout: &[u32],
    photon_ct_hist: &[u32],
    // Per-material counts + base offsets into the concatenated buffers above.
    // `photon_n_product_per_material[m]` products live at base
    // `photon_prod_base_per_material[m]` (into the per-product arrays). The
    // rxn rows start at `photon_rxn_base_per_material[m]` (element units, a
    // multiple of `n_grid`). The aggregate `photon_prod` row starts at
    // `photon_pp_base_per_material[m]` (element units, a multiple of n_grid).
    photon_n_product_per_material: &[u32],
    photon_prod_base_per_material: &[u32],
    photon_pp_base_per_material: &[u32],
    // D1S (Direct-1-Step) decay-photon production. Mutually exclusive with the
    // prompt `coupled_enabled` gate above (the CPU treats them as
    // `if use_decay_photons { decay } else { prompt }`): when D1S is on,
    // `coupled_enabled[0] == 0` and `decay_enabled[0] == 1`. Gated identically
    // to the prompt block -- OFF draws zero decay RNG and writes zero decay bank
    // records, so a non-D1S run is byte-identical. See
    // `decay_photon_emission::DecayPhotonInputs` for the host packing.
    decay_enabled: &[u32],
    // Per-material aggregate decay photon-production macro xs on the shared
    // grid, concatenated material-major; material `m`'s row starts at
    // `decay_meta[m * DECAY_META_COLS + 0]`. Drives `y_t = photon_prod /
    // sigma_t`.
    decay_photon_prod: &[f64],
    // Per-channel weighted reaction xs rows (`n_grid` each), concatenated
    // material-major then channel-major. Global channel `c`'s row starts at
    // `c * n_grid`. The aggregate `decay_photon_prod` equals the per-grid sum
    // of these rows over a material's channels.
    decay_ch_xs: &[f64],
    // Per-channel parent-nuclide id (`NuclideId.get()` widened to u32). Stamped
    // on the emitted photon's bank `gen` slot for `parent_nuclides` binning.
    decay_ch_parent_id: &[u32],
    // GLOBAL per-channel discrete-line `[base, count]` into `decay_ch_energies`
    // / `decay_ch_intensity_cdf`, interleaved stride 2. Channel `c` reads
    // `decay_ch_e_meta[c*2]` / `decay_ch_e_meta[c*2 + 1]`.
    decay_ch_e_meta: &[u32],
    // Per-channel discrete decay-line energies [eV], concatenated.
    decay_ch_energies: &[f64],
    // Per-channel cumulative intensity CDF (normalized), concatenated, parallel
    // to `decay_ch_energies`.
    decay_ch_intensity_cdf: &[f64],
    // Packed per-material decay metadata, `[mat * DECAY_META_COLS + col]`:
    // col 0 = `pp_base` (aggregate-row offset), col 1 = `ch_base` (global index
    // of the material's first channel), col 2 = `ch_count`.
    decay_meta: &[u32],
    // ----------------------- per-collision nuclide selection (issue #74) ----
    // Per-(material, nuclide) macroscopic total xs on the shared grid, flat
    // `[n_slab x n_grid]` (nuclide-major within a material, concatenated
    // material-major). Slab row `s` starts at `s * n_grid`. At a collision in a
    // multi-nuclide material the kernel walks these rows cumulatively (linearly
    // interpolated at the collision bracket `[idx_lo, idx_hi]`, factor `frac`)
    // against one scaled uniform draw to pick the struck nuclide, then overrides
    // `target_mass` with that nuclide's AWR so elastic kinematics use the exact
    // per-nuclide mass. Gated on `count > 1`: single-nuclide materials draw NO
    // selection random and stay byte-identical to the pre-#74 stream.
    nuc_macro_total: &[f64],
    // Per-(material, nuclide) AWR, flat `[n_slab]`. Indexed by the global slab.
    nuc_awr: &[f64],
    // Stride-2 `[offset, count]` per material: `mat_nuclide_meta[mat*2]` is the
    // material's slab base, `mat_nuclide_meta[mat*2 + 1]` its nuclide count.
    mat_nuclide_meta: &[u32],
    // Per-(slab, energy) reaction partials, DENSITY-WEIGHTED, packed
    // `[n_slab x n_grid x NUC_PARTIAL_COLS]` (#74 Stage 2b). The entry for slab
    // `s`, grid point `i`, column `c` is `nuc_partial_xs[(s * n_grid + i) * 4 +
    // c]`; columns are elastic(0) / absorption(1) / inelastic(2) / fission(3).
    // After selecting nuclide `s` (count > 1), the kernel splits the reaction
    // type from THESE partials (mirroring CPU `Nuclide::sample_reaction_type`,
    // scattering = elastic + inelastic), instead of the material-aggregate
    // `sigma_*`. Single-nuclide materials never read this (the aggregate split
    // is exact and byte-identical).
    nuc_partial_xs: &[f64],
    // Runtime gate for the device fission bank (issue #78, 1 element). When
    // `fission_bank_enabled[0] == 0` the fission branch keeps the legacy
    // `weight *= nu_bar` + `FISSION_WEIGHT_CAP` terminator (byte-identical to a
    // pre-bank run). When `1` it stochastically rounds nu_bar to N, continues
    // ONE chi-sampled progeny in the current walk (weight unchanged), and
    // appends the other N-1 chi-sampled progeny to the device bank tagged as
    // neutrons -- the GPU twin of CPU `sample_fission_neutrons` + `bank_secondary`.
    fission_bank_enabled: &[u32],
    // Device particle bank (append-only SoA, see common::particle_bank). The
    // emission site inlines the atomic fetch-add slot-reservation + record
    // write; `bank_count` / `bank_overflow` are single-element u64 atomics.
    bank_f64: &mut [f64],
    bank_u32: &mut [u32],
    bank_count: &mut [Atomic<u64>],
    bank_overflow: &mut [Atomic<u64>],
    // Exit diagnostics. For a history that popped in-thread (n,xn)
    // secondaries (issue #274) these describe the LAST transported
    // secondary, not the primary walk: `out_n_steps` restarts per popped
    // particle and `out_alive` / `out_final_energy` are the final
    // particle's. Do not gate lost-particle or truncation detection on
    // them for multiplying histories.
    out_alive: &mut [u32],
    out_n_steps: &mut [u32],
    out_final_energy: &mut [f64],
    tally_out: &mut [Atomic<u64>],
    // Batch-free per-history variance spill (issue #233). Per-history
    // overflow list for a history touching MORE than `PERHIST_K` distinct
    // bins: each thread owns the slice `[ABSOLUTE_POS * spill_cap ..
    // + spill_cap)` (no cross-thread access, so no atomics). `spill_bin`
    // holds the flat `tally_out` bin index, `spill_val` its running
    // per-history total. Both are size-1 dummies (never indexed) when
    // `per_history_var` is false or `spill_cap == 0`.
    spill_bin: &mut [u32],
    spill_val: &mut [f64],
    // Per-source variance (issue #233 Stage 2, fissile). `source_idx[i]` is the
    // index (within the launch chunk of ORIGINAL source neutrons) of the source
    // neutron thread `i` descends from: identity `i` for a source launch, the
    // banked parent's index for a fission-generation launch. When
    // `per_source_var`, a history flushes its per-bin totals into
    // `src_acc[source_idx[i] * total_bins + bin]` (fixed-point sum, atomic:
    // progeny of one source can race the same slot) INSTEAD of the Stage 1
    // sum/sum_sq flush -- so a source's contributions accumulate across the
    // source + every generation launch before being squared at finalize.
    // `bank_source_idx[slot]` records the originating source index of each
    // banked fission progeny so the next generation inherits it (sized to the
    // bank capacity, like `bank_u32`). `source_idx` / `src_acc` are size-1
    // dummies when `per_source_var` is false.
    source_idx: &[u32],
    src_acc: &mut [Atomic<u64>],
    bank_source_idx: &mut [u32],
    // Lost-particle diagnostics (issue #289). `lost_count` is a single-element
    // u64 atomic counting every history that ended in no cell; `lost_f64` keeps
    // the first `lost_f64.len() / LOST_F64_STRIDE` records for the host to turn
    // into `LostParticle` diagnostics. The host enforces `max_lost_particles`
    // from the counter, so a gap in the geometry aborts the GPU run exactly as
    // it aborts the CPU run instead of returning a quietly truncated tally.
    lost_count: &mut [Atomic<u64>],
    lost_f64: &mut [f64],
    #[comptime] max_steps: u32,
    // When true, accumulate each history's per-bin total in a thread-private
    // touched-list (+ global spill) and flush `sum` + `sum_sq` at history end
    // instead of a per-step atomic add. `tally_out` is then sized
    // `2 * total_out_len`: the first half holds `sum` (linear fixed-point,
    // `tally_fixed_point_scales[t]`), the second half `sum_sq`
    // (`sum_sq_fixed_point_scale`). Off => byte-identical per-step path.
    #[comptime] per_history_var: bool,
    // Per-history spill capacity (`total_out_len - PERHIST_K`, clamped to
    // `>= 0`). Comptime so the per-thread base `ABSOLUTE_POS * spill_cap`
    // folds to a constant stride.
    #[comptime] spill_cap: u32,
    // Per-source variance mode (issue #233 Stage 2). When true the history flush
    // targets `src_acc` keyed by `source_idx` instead of the sum/sum_sq halves
    // of `tally_out`. Implies `per_history_var` (the touched-list still gathers
    // each thread's own steps).
    #[comptime] per_source_var: bool,
    // Per-source accumulator row stride (= `total_out_len`). Comptime so the
    // per-source base `source_idx[i] * total_bins` folds cleanly.
    #[comptime] total_bins: u32,
    // Mesh-tally variance path (issue #234). When true, every tally contribution
    // (each voxel crossing of a track-length mesh tally, or a single bin for a
    // non-mesh tally) is atomic-added, in the SUM fixed-point scale, DIRECTLY
    // into `src_acc[source_idx[i] * total_bins + flat_idx]` -- bypassing the
    // touched-list entirely (no O(distinct^2) dedup, no per-thread spill). The
    // host squares each source's grand total at finalize (per-source variance,
    // exactly as the fissile Stage-2 path). Set together with `per_source_var`
    // (the host uses the same src_acc/source_idx/finalize infrastructure);
    // `spill_cap` is 0 and the touched-list stays empty, so the history-end
    // flush is a no-op. Only mesh models use this; non-mesh launches keep the
    // Stage 1/2 touched-list path (byte-identical).
    #[comptime] mesh_direct: bool,
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    // Per-particle state in registers.
    let i3 = ABSOLUTE_POS * 3;
    let mut energy = energies_in[ABSOLUTE_POS];
    let mut px = positions_in[i3];
    let mut py = positions_in[i3 + 1];
    let mut pz = positions_in[i3 + 2];
    let mut dx = directions_in[i3];
    let mut dy = directions_in[i3 + 1];
    let mut dz = directions_in[i3 + 2];
    // Seed of the walk this thread is currently transporting: the source
    // particle's to begin with, then each popped (n,xn) secondary's own
    // identity-derived seed (issue #111). It is the parent half of the key
    // `secondary_seed` uses for anything this walk queues.
    let mut walk_seed = seeds[ABSOLUTE_POS];
    let mut state = crate::common::pcg32::expand_seed(walk_seed);
    // Secondaries this walk has queued so far == the ordinal the next one
    // takes. Reset on every pop, so each walk numbers its own secondaries
    // from 0 and the numbering does not depend on the drain order.
    let mut walk_secondaries = 0u32;

    // Held URR probability-table band for this walk (issue #342). The base
    // uniform is drawn ONCE per energy, not once per step: OpenMC advances its
    // URR seed only when the energy changes (`physics.cpp:164`) and the CPU
    // does the same (`Particle.urr_energy` / `urr_random`, PR #207), so the
    // same isotope keeps one resonance realisation across boundary crossings
    // and void excursions until a collision moves the neutron off that energy.
    // `urr_held` is the CPU's `NO_URR` sentinel in flag form. The two f64
    // registers are seeded from a runtime value (not a literal) because a
    // literal-initialised f64 local is comptime in cubecl and cannot take a
    // runtime assignment; both are overwritten before first use, since
    // `urr_held == 0` forces the initial draw.
    let mut urr_base = energies_in[ABSOLUTE_POS];
    let mut urr_energy = energies_in[ABSOLUTE_POS];
    let mut urr_held = 0u32;

    // alive: 1 = transporting, 0 = absorbed or escaped.
    let mut alive = 1u32;
    let mut n_steps = 0u32;
    // weight: slice-E multiplier for tally accumulation. Starts at
    // 1.0 and grows by `MT_YIELDS[slot]` whenever an inelastic
    // collision picks a multi-neutron-out MT (slot 41 = MT 16
    // doubles weight, slot 42 = MT 17 triples it). Track-length and
    // absorption tallies are scaled by `weight` so the expected
    // contribution matches `yield` independent neutrons (yamc's CPU
    // side clones the primary; this is the variance-reduction form).
    let mut weight = 1.0;

    // `angle_interp` is currently unused in the kernel -- slice B
    // approximates every (mu, cdf) bracket as piecewise-linear in
    // c → x, which matches Histogram interp exactly and is a fine
    // approximation of LinLin for ~33-point tables. Reading the
    // length keeps the binding live so a future kernel-side
    // upgrade can branch on it without re-plumbing the launch.
    let _angle_interp_len = angle_interp.len();

    let n_cells: u32 = (cell_aabbs.len() / 6) as u32;
    let n_surfaces: u32 = surface_types.len() as u32;
    let n_tallies: u32 = tally_score_kinds.len() as u32;
    let n_bvh_nodes: u32 = (bvh_aabbs.len() / 6) as u32;
    let n_bvh_unbounded: u32 = bvh_unbounded.len() as u32;

    // Batch-free per-history variance (issue #233): thread-private
    // touched-list. `th_bin[j]` = flat `tally_out` bin index, `th_val[j]` =
    // this history's running total for that bin. Declared UNCONDITIONALLY (a
    // conditional `Array::new` trips a cubecl-SPIRV/RADV codegen quirk), so
    // the per-step path also pays ~384 bytes of thread-private state; it is
    // only READ/written when `per_history_var`. Overflow past `PERHIST_K`
    // goes to the per-history global spill (`spill_bin`/`spill_val`).
    let mut th_bin = Array::<u32>::new(PERHIST_K as usize);
    let mut th_val = Array::<f64>::new(PERHIST_K as usize);
    let mut th_count = 0u32;
    let mut spill_count = 0u32;
    // In-thread (n,xn) pending-secondary STACK (issue #274). Analog
    // multiplicity: an integral multi-neutron yield keeps the walk's weight
    // unchanged and pushes the extra secondaries here; when the current
    // particle dies the thread pops the most recent one and keeps
    // transporting. Every secondary of the history stays in this thread, so
    // per-history variance, per-source-direct mesh scoring, and every dispatch
    // path see one complete history exactly like the CPU's in-history bank.
    // Same LIFO discipline as `yamc_physics::util::bank::ParticleBank` and as
    // OpenMC's per-particle secondary bank.
    //
    // `pend_n` is the number of secondaries OUTSTANDING, and a pop reclaims its
    // slot, so `PEND_SLOTS` bounds the emission tree's DFS depth rather than
    // its size. Every (n,xn) is endothermic and splits what is left of the
    // incident energy between its products, so the chain bottoms out against
    // the reaction threshold within a few levels and the stack does not fill in
    // practice (`nxn_spill_depth_is_sufficient` measures it). A history that
    // does exceed the depth spills the extra secondary to the device particle
    // bank, tagged `PTYPE_NEUTRON` / `BANK_GEN_NXN_SPILL`, for the host to
    // drain and transport in a later pass -- nothing is dropped and nothing
    // reverts to weight multiplication.
    //
    // `pend_seed` carries each entry's identity-derived collision seed (issue
    // #111): the thread re-seeds its PCG from it on pop rather than letting the
    // parent's state carry over, so where a secondary is transported (this
    // stack, or a later host pass) cannot change what it samples. 16 bytes of
    // thread-private state beside the 256 the eight f64 arrays already cost.
    let mut pend_seed = Array::<u32>::new(PEND_SLOTS);
    let mut pend_e = Array::<f64>::new(PEND_SLOTS);
    let mut pend_px = Array::<f64>::new(PEND_SLOTS);
    let mut pend_py = Array::<f64>::new(PEND_SLOTS);
    let mut pend_pz = Array::<f64>::new(PEND_SLOTS);
    let mut pend_dx = Array::<f64>::new(PEND_SLOTS);
    let mut pend_dy = Array::<f64>::new(PEND_SLOTS);
    let mut pend_dz = Array::<f64>::new(PEND_SLOTS);
    let mut pend_w = Array::<f64>::new(PEND_SLOTS);
    let mut pend_n = 0u32;
    // Per-thread base into the global spill buffers. `spill_cap` is comptime,
    // so this is a constant stride multiply. `usize` to match the kernel's
    // buffer-index type (ABSOLUTE_POS).
    let spill_base = ABSOLUTE_POS * spill_cap as usize;
    // Per-source variance (issue #233 Stage 2): the source neutron this thread
    // descends from, and its base row in `src_acc`. Only read/used when
    // `per_source_var` (else `source_idx` is a size-1 dummy). Stamped into every
    // fission progeny this thread banks so descendants inherit it.
    let mut my_source_idx = 0u32;
    if per_source_var || mesh_direct {
        my_source_idx = source_idx[ABSOLUTE_POS];
    }
    let src_base = my_source_idx as usize * total_bins as usize;

    // Cell this history occupied on the previous step, for the lost-particle
    // record (issue #289). `4_294_967_295` = "none yet", i.e. a source particle
    // born outside every cell. Kept as u32 (not f64) so the running
    // read-modify-write is free of the cubecl f64-literal-init quirk; widened
    // at the single record site.
    let mut last_cell = 4_294_967_295u32;

    let mut step = 0u32;
    while (alive == 1u32 && step < max_steps) || pend_n > 0u32 {
        // Current particle finished but (n,xn) secondaries are queued: pop
        // the most recent one and keep transporting inside the same history
        // (issue #274). The per-history accumulators carry over; the
        // particle registers reset, and so does the RNG -- the popped
        // secondary starts on the stream its identity picked out when it
        // was queued (issue #111), not on the tail of whatever the previous
        // walk left behind. Popping frees the slot for a later push.
        if alive == 0u32 || step >= max_steps {
            pend_n -= 1u32;
            let pi = pend_n as usize;
            energy = pend_e[pi];
            px = pend_px[pi];
            py = pend_py[pi];
            pz = pend_pz[pi];
            dx = pend_dx[pi];
            dy = pend_dy[pi];
            dz = pend_dz[pi];
            weight = pend_w[pi];
            walk_seed = pend_seed[pi];
            state = crate::common::pcg32::expand_seed(walk_seed);
            walk_secondaries = 0u32;
            alive = 1u32;
            step = 0u32;
            // A popped secondary is its own particle and holds no band yet,
            // matching `Particle::new`'s `NO_URR` (issue #342).
            urr_held = 0u32;
        }
        // 1. Find current cell via stackless BVH traversal (rope
        // encoding: each node carries the next-in-DFS index used
        // when its AABB rejects the point).
        //
        // We don't fast-path tiny geometries here even though the
        // BVH adds ~one wasted AABB test for n_cells ≤ 4 -- adding
        // a host-driven `if n_bvh_nodes <= 1` branch in the kernel
        // bloated the SPIR-V enough to drop large-geometry
        // occupancy by ~11%, much bigger than the ~5% small-N win.
        // The CPU helper still takes the small-N fast path because
        // there's no register-pressure cost on CPU.
        let mut cell = 4_294_967_295u32; // CELL_NOT_FOUND
        let mut bi = 0u32;
        while bi < n_bvh_nodes && cell == 4_294_967_295u32 {
            let ab: u32 = bi * 6u32;
            let min_x = bvh_aabbs[ab as usize];
            let min_y = bvh_aabbs[(ab + 1u32) as usize];
            let min_z = bvh_aabbs[(ab + 2u32) as usize];
            let max_x = bvh_aabbs[(ab + 3u32) as usize];
            let max_y = bvh_aabbs[(ab + 4u32) as usize];
            let max_z = bvh_aabbs[(ab + 5u32) as usize];
            let in_aabb = px >= min_x
                && px <= max_x
                && py >= min_y
                && py <= max_y
                && pz >= min_z
                && pz <= max_z;
            let mb: u32 = bi * 3u32;
            let r_or_f = bvh_meta[mb as usize];
            let n_prims = bvh_meta[(mb + 1u32) as usize];
            let escape = bvh_meta[(mb + 2u32) as usize];
            if in_aabb {
                if n_prims > 0u32 {
                    let mut p = 0u32;
                    while p < n_prims && cell == 4_294_967_295u32 {
                        let prim_idx = bvh_prim_indices[(r_or_f + p) as usize];
                        let cb: u32 = prim_idx * 6u32;
                        let cmin_x = cell_aabbs[cb as usize];
                        let cmin_y = cell_aabbs[(cb + 1u32) as usize];
                        let cmin_z = cell_aabbs[(cb + 2u32) as usize];
                        let cmax_x = cell_aabbs[(cb + 3u32) as usize];
                        let cmax_y = cell_aabbs[(cb + 4u32) as usize];
                        let cmax_z = cell_aabbs[(cb + 5u32) as usize];
                        // AABB is a cheap pre-filter; the actual CSG region
                        // test disambiguates nested / overlapping cells.
                        if px >= cmin_x
                            && px <= cmax_x
                            && py >= cmin_y
                            && py <= cmax_y
                            && pz >= cmin_z
                            && pz <= cmax_z
                            && region_contains(
                                region_program,
                                surface_types,
                                surface_params,
                                n_cells,
                                prim_idx,
                                px,
                                py,
                                pz,
                            )
                        {
                            cell = prim_idx;
                        }
                        p += 1u32;
                    }
                }
                bi += 1u32;
            } else {
                bi = escape;
            }
        }
        // Unbounded fallback (BVH-uncovered primitives).
        if cell == 4_294_967_295u32 {
            let mut u = 0u32;
            while u < n_bvh_unbounded && cell == 4_294_967_295u32 {
                let prim_idx = bvh_unbounded[u as usize];
                let cb: u32 = prim_idx * 6u32;
                let cmin_x = cell_aabbs[cb as usize];
                let cmin_y = cell_aabbs[(cb + 1u32) as usize];
                let cmin_z = cell_aabbs[(cb + 2u32) as usize];
                let cmax_x = cell_aabbs[(cb + 3u32) as usize];
                let cmax_y = cell_aabbs[(cb + 4u32) as usize];
                let cmax_z = cell_aabbs[(cb + 5u32) as usize];
                if px >= cmin_x
                    && px <= cmax_x
                    && py >= cmin_y
                    && py <= cmax_y
                    && pz >= cmin_z
                    && pz <= cmax_z
                    && region_contains(
                        region_program,
                        surface_types,
                        surface_params,
                        n_cells,
                        prim_idx,
                        px,
                        py,
                        pz,
                    )
                {
                    cell = prim_idx;
                }
                u += 1u32;
            }
        }

        if cell == 4_294_967_295u32 {
            // Particle is in no explicit cell, i.e. the geometry does not
            // cover the space it reached. The CPU calls this a LOST particle
            // (`handle_lost_particle`): it records the diagnostic and aborts
            // the run once `max_lost_particles` is exceeded. Record it here
            // for the same treatment host-side (issue #289) instead of
            // silently ending the history, which used to make a model with a
            // geometry gap "succeed" on GPU while refusing to run on CPU.
            //
            // A leak through a `boundary='vacuum'` surface never reaches this
            // branch: the crossing block below kills the history at the
            // surface, so the loop exits without re-locating the particle.
            crate::common::lost_particles::record_lost(
                lost_count,
                lost_f64,
                px,
                py,
                pz,
                dx,
                dy,
                dz,
                energy,
                last_cell as f64,
            );
            alive = 0u32;
        } else {
            last_cell = cell;
            // 2. Look up the material for this cell.
            let mat_idx = cell_to_material[cell as usize];
            // Packed `mat_f64_meta` columns (stride 5u32 = MAT_F64_COLS):
            //   0 = COL_TARGET_MASS
            //   1 = COL_TEMPERATURE_K
            //   2 = COL_FISSION_A
            //   3 = COL_FISSION_B
            //   4 = COL_URR_ATOM_DENSITY
            let mat_meta_off = mat_idx * 5u32;
            // `target_mass` is the density-weighted material average; for a
            // multi-nuclide material the per-collision nuclide-selection block
            // below overrides it with the struck nuclide's exact AWR (issue #74).
            let mut target_mass = mat_f64_meta[(mat_meta_off + 0u32) as usize];
            // GLOBAL grid length -- backs the grid-shared score / photon / decay
            // lookups (their buffers stay on `log_energy_grid`); the aggregate
            // macro XS + nuc buffers moved to the per-material FINE grid below.
            let n_grid: u32 = log_energy_grid.len() as u32;

            // 3. XS lookup at current energy.
            let log_e = ln_f64(energy);
            let mut lo = 0u32;
            let mut hi = n_grid;
            let mut iter = 0u32;
            while iter < 32u32 && lo < hi {
                let mid: u32 = (lo + hi) / 2u32;
                if log_energy_grid[mid as usize] < log_e {
                    lo = mid + 1u32;
                } else {
                    hi = mid;
                }
                iter += 1u32;
            }
            // Clamp to [1, n_grid − 1] so the [idx_lo, idx_hi]
            // interval always exists. Energies below the grid
            // minimum or above the maximum get clamped at the
            // endpoints -- same fallback as the CPU `partition_point`
            // mirror.
            let mut idx_hi = lo;
            if idx_hi < 1u32 {
                idx_hi = 1u32;
            }
            if idx_hi >= n_grid {
                idx_hi = n_grid - 1u32;
            }
            let idx_lo: u32 = idx_hi - 1u32;
            let x_lo = log_energy_grid[idx_lo as usize];
            let x_hi = log_energy_grid[idx_hi as usize];
            // Interpolation factor is computed in **linear** E to
            // mirror CPU's `Material::lookup_xs_by_mt` /
            // `FastXSGrid::lookup_total` -- those interpolate σ(E)
            // linearly on the unified energy grid in linear-E space.
            // A linear-in-log frac (`(log_e − x_lo) / (x_hi − x_lo)`)
            // systematically over-estimates concave-up curves like
            // 1/v capture, biasing MT 102 high by a few percent on
            // actinides at fast energies. Recover the bracket's
            // linear energies via `exp_f64` of the stored log values
            // (cheaper than passing a parallel linear grid buffer).
            let e_lo = exp_f64(x_lo);
            let e_hi = exp_f64(x_hi);
            let denom = e_hi - e_lo;
            let mut frac = 0.0_f64;
            if denom > 0.0 {
                frac = (energy - e_lo) / denom;
            }
            // Fine-grid bracket for the resonance-critical per-material aggregate
            // macro XS + the per-(material, nuclide) nuc buffers (issue #212).
            // This material's own fine grid is the slice
            // `fine_log_energy_grid[fine_off .. fine_off + fine_n]` (stride-3
            // `fine_meta` cols GRID_OFFSET / N). `idx_lo_f` / `idx_hi_f` stay
            // RELATIVE to `fine_off`; `fine_off` doubles as the aggregate-XS row
            // base (those buffers are concatenated in the same per-material order
            // + length as the grid). `fine_nuc_base` (col 2) is the material's
            // first-slab base into the nuc buffers. Single-material problems have
            // `fine_off == 0`, `fine_n == fine_log_energy_grid.len()`,
            // `fine_nuc_base == 0`, so this is bit-identical to the old shared
            // fine grid.
            let fine_meta_off = mat_idx * 3u32; // stride 3u32 = FINE_META_COLS
            let fine_off: u32 = fine_meta[(fine_meta_off + 0u32) as usize];
            let fine_n: u32 = fine_meta[(fine_meta_off + 1u32) as usize];
            let fine_nuc_base: u32 = fine_meta[(fine_meta_off + 2u32) as usize];
            let mut lo_f = 0u32;
            let mut hi_f = fine_n;
            let mut iter_f = 0u32;
            while iter_f < 32u32 && lo_f < hi_f {
                let mid_f: u32 = (lo_f + hi_f) / 2u32;
                if fine_log_energy_grid[(fine_off + mid_f) as usize] < log_e {
                    lo_f = mid_f + 1u32;
                } else {
                    hi_f = mid_f;
                }
                iter_f += 1u32;
            }
            let mut idx_hi_f = lo_f;
            if idx_hi_f < 1u32 {
                idx_hi_f = 1u32;
            }
            if idx_hi_f >= fine_n {
                idx_hi_f = fine_n - 1u32;
            }
            let idx_lo_f: u32 = idx_hi_f - 1u32;
            let e_lo_f = exp_f64(fine_log_energy_grid[(fine_off + idx_lo_f) as usize]);
            let e_hi_f = exp_f64(fine_log_energy_grid[(fine_off + idx_hi_f) as usize]);
            let denom_f = e_hi_f - e_lo_f;
            let mut frac_f = 0.0_f64;
            if denom_f > 0.0 {
                frac_f = (energy - e_lo_f) / denom_f;
            }
            // Coarse-grid bracket for the per-MT inelastic / yield buffers
            // (issue #88 / #212). This material's own coarse grid is the slice
            // `coarse_log_energy_grid[coarse_base .. coarse_base + coarse_n]`
            // (stride-3 `coarse_meta` cols 0 / 1). `idx_lo_c` / `idx_hi_c` stay
            // RELATIVE to `coarse_base` so they double as the per-MT-buffer
            // energy index; the base is added only when reading the grid itself.
            // Single-material problems have `coarse_base == 0` and `coarse_n ==
            // coarse_log_energy_grid.len()`, so this is bit-identical.
            let coarse_meta_off = mat_idx * 3u32; // stride 3u32 = COARSE_META_COLS
            let coarse_base: u32 = coarse_meta[(coarse_meta_off + 0u32) as usize];
            let coarse_n: u32 = coarse_meta[(coarse_meta_off + 1u32) as usize];
            let mut lo_c = 0u32;
            let mut hi_c = coarse_n;
            let mut iter_c = 0u32;
            while iter_c < 32u32 && lo_c < hi_c {
                let mid_c: u32 = (lo_c + hi_c) / 2u32;
                if coarse_log_energy_grid[(coarse_base + mid_c) as usize] < log_e {
                    lo_c = mid_c + 1u32;
                } else {
                    hi_c = mid_c;
                }
                iter_c += 1u32;
            }
            let mut idx_hi_c = lo_c;
            if idx_hi_c < 1u32 {
                idx_hi_c = 1u32;
            }
            if idx_hi_c >= coarse_n {
                idx_hi_c = coarse_n - 1u32;
            }
            let idx_lo_c: u32 = idx_hi_c - 1u32;
            let e_lo_c = exp_f64(coarse_log_energy_grid[(coarse_base + idx_lo_c) as usize]);
            let e_hi_c = exp_f64(coarse_log_energy_grid[(coarse_base + idx_hi_c) as usize]);
            let denom_c = e_hi_c - e_lo_c;
            let mut frac_c = 0.0_f64;
            if denom_c > 0.0 {
                frac_c = (energy - e_lo_c) / denom_c;
            }
            // Aggregate macro XS ride the per-material FINE grid (issue #212):
            // row base `fine_off`, energy index `idx_lo_f` / `idx_hi_f`, factor
            // `frac_f`. Single-material => `fine_off == mat_idx * n_grid`, so
            // byte-identical.
            let xs_e_lo = xs_elastic_per_material[(fine_off + idx_lo_f) as usize];
            let xs_e_hi = xs_elastic_per_material[(fine_off + idx_hi_f) as usize];
            let sigma_e = xs_e_lo + (xs_e_hi - xs_e_lo) * frac_f;
            let xs_a_lo = xs_absorption_per_material[(fine_off + idx_lo_f) as usize];
            let xs_a_hi = xs_absorption_per_material[(fine_off + idx_hi_f) as usize];
            let sigma_a = xs_a_lo + (xs_a_hi - xs_a_lo) * frac_f;
            let xs_i_lo = xs_inelastic_per_material[(fine_off + idx_lo_f) as usize];
            let xs_i_hi = xs_inelastic_per_material[(fine_off + idx_hi_f) as usize];
            let sigma_i = xs_i_lo + (xs_i_hi - xs_i_lo) * frac_f;
            let xs_f_lo = xs_fission_per_material[(fine_off + idx_lo_f) as usize];
            let xs_f_hi = xs_fission_per_material[(fine_off + idx_hi_f) as usize];
            let sigma_f = xs_f_lo + (xs_f_hi - xs_f_lo) * frac_f;
            let nu_lo = nu_bar_per_material[(fine_off + idx_lo_f) as usize];
            let nu_hi = nu_bar_per_material[(fine_off + idx_hi_f) as usize];
            let nu_bar = nu_lo + (nu_hi - nu_lo) * frac_f;
            // Delayed fraction on the same fine grid (issue #364).
            let beta_lo = beta_delayed_per_material[(fine_off + idx_lo_f) as usize];
            let beta_hi = beta_delayed_per_material[(fine_off + idx_hi_f) as usize];
            let beta_delayed = beta_lo + (beta_hi - beta_lo) * frac_f;

            // Per-(material, nuclide) URR probability-table sampling (issue
            // #210). URR is applied to EVERY in-range URR nuclide of the
            // material, each drawing an independent probability-table band
            // (issue #204), mirroring CPU `Material::compute_urr_macro_xs`.
            // The unperturbed `sigma_e` / `sigma_a` / `sigma_f` / `sigma_i`
            // stay the floor; each URR nuclide adds a macroscopic delta taken
            // against its own `nuc_partial_xs` smooth baseline. The summed
            // deltas are applied and clamped ONCE after the slab loop (an
            // individual nuclide's self-shielding delta is legitimately
            // negative), while the per-slab micro-XS >= 0 clamps stay inside
            // the loop.
            //
            // Tight CSR (issue #104): the URR energy / cdf / xs buffers are
            // packed without the old MAX_URR_* padding, so the bracket-find
            // loops MUST be variable-bound -- a fixed comptime bound would
            // read past a slab's tight region (launch_unchecked has no bounds
            // check). Same variable-bound pattern the migrated families use.
            let mut sigma_e_use = sigma_e;
            let mut sigma_a_use = sigma_a;
            let mut sigma_f_use = sigma_f;
            let mut sigma_i_use = sigma_i;
            // URR scoring-correlation state (hoisted so the tally score loops
            // below can substitute the URR-perturbed capture for the smooth
            // `xs_score_per_mt` value). `urr_fired` is set when any URR
            // nuclide is sampled this step; `urr_macro_capture` is the
            // full-material URR-modified macroscopic capture (n,gamma,
            // EXCLUDING fission) = `sigma_a_use - sigma_f_use` (the GPU twin
            // of CPU `UrrMacroXs.capture`, including non-URR nuclides' smooth
            // capture). Both stay at their defaults when no URR nuclide fires.
            let mut urr_fired = false;
            let mut urr_macro_capture = 0.0_f64;

            // This material's slab range [nuc_off, nuc_off + nuc_count) into
            // the global per-(material, nuclide) tables (the same table the
            // nuclide-selection walk uses).
            let urr_nuc_off = mat_nuclide_meta[(mat_idx * 2u32) as usize];
            let urr_nuc_count = mat_nuclide_meta[(mat_idx * 2u32 + 1u32) as usize];

            // First pass: does the material have >= 1 in-range URR nuclide?
            // (Mirrors CPU `has_urr_in_range`; the base uniform is drawn once
            // per collision only when at least one nuclide's URR range covers
            // the energy, keeping the RNG stream advance to one draw.)
            let mut urr_any_in_range = false;
            let mut us = 0u32;
            while us < urr_nuc_count {
                let slab = urr_nuc_off + us;
                let mo = slab * 8u32; // URR_META_COLS = 8
                if urr_meta[mo as usize] == 1u32 {
                    let n_e = urr_meta[(mo + 1u32) as usize]; // URR_META_N_ENERGIES
                    if n_e >= 2u32 {
                        let eg_off = urr_ae_offset[slab as usize];
                        let e_first = urr_energy_grid[eg_off as usize];
                        let e_last = urr_energy_grid[(eg_off + n_e - 1u32) as usize];
                        if energy > e_first && energy < e_last {
                            urr_any_in_range = true;
                        }
                    }
                }
                us += 1u32;
            }

            if urr_any_in_range {
                // One base uniform per ENERGY (issues #204, #342); each
                // nuclide's band is decorrelated from it by
                // `urr_nuclide_random(base, ZA)`. A band already held at this
                // exact energy is reused, so crossing into another material,
                // or streaming back through a void, keeps the isotope on the
                // one resonance realisation it presented before. Drawing per
                // step instead decorrelates the realisation across crossings
                // and over-predicts transmitted flux ~30% (issue #342).
                // Held bands are only ever touched here, so a void or
                // non-URR step leaves the band intact, matching PR #207.
                if urr_held == 0u32 || urr_energy != energy {
                    let d_urr = crate::common::pcg32::draw_uniform(state);
                    state = d_urr.state;
                    urr_base = d_urr.xi;
                    urr_energy = energy;
                    urr_held = 1u32;
                }
                let r_base = urr_base;

                // Accumulate macroscopic deltas across the material's URR
                // slabs; the SUMMED result is applied once after the loop.
                let mut sum_delta_e = 0.0_f64;
                let mut sum_delta_a = 0.0_f64;
                let mut sum_delta_f = 0.0_f64;
                let mut sum_delta_i = 0.0_f64;

                let mut ku = 0u32;
                while ku < urr_nuc_count {
                    let slab = urr_nuc_off + ku;
                    // Smooth macroscopic baselines for this slab from
                    // `nuc_partial_xs` (density-weighted, per-material FINE
                    // grid, issue #212): col 0 elastic, col 1 absorption
                    // (capture + other_abs, EXCLUDING fission), col 2
                    // inelastic, col 3 fission. Same FINE bracket the smooth
                    // macro lookup used.
                    let p_lo = (fine_nuc_base + ku * fine_n + idx_lo_f) * 4u32;
                    let p_hi = (fine_nuc_base + ku * fine_n + idx_hi_f) * 4u32;
                    let be_lo = nuc_partial_xs[p_lo as usize];
                    let be_hi = nuc_partial_xs[p_hi as usize];
                    let base_e = be_lo + (be_hi - be_lo) * frac_f;
                    let ba_lo = nuc_partial_xs[(p_lo + 1u32) as usize];
                    let ba_hi = nuc_partial_xs[(p_hi + 1u32) as usize];
                    let base_a = ba_lo + (ba_hi - ba_lo) * frac_f;
                    let bi_lo = nuc_partial_xs[(p_lo + 2u32) as usize];
                    let bi_hi = nuc_partial_xs[(p_hi + 2u32) as usize];
                    let base_i = bi_lo + (bi_hi - bi_lo) * frac_f;
                    let bf_lo = nuc_partial_xs[(p_lo + 3u32) as usize];
                    let bf_hi = nuc_partial_xs[(p_hi + 3u32) as usize];
                    let base_f = bf_lo + (bf_hi - bf_lo) * frac_f;

                    let up = crate::common::urr::urr_slab_partials(
                        urr_meta,
                        urr_ae_offset,
                        urr_cdf_offset,
                        urr_energy_grid,
                        urr_cdf,
                        urr_xs,
                        urr_atom_density,
                        slab,
                        energy,
                        r_base,
                        base_e,
                        base_a,
                        base_i,
                        base_f,
                    );
                    if up.fired == 1u32 {
                        sum_delta_e += up.elastic - base_e;
                        sum_delta_a += up.absorption - base_a;
                        sum_delta_f += up.fission - base_f;
                        sum_delta_i += up.inelastic - base_i;
                        urr_fired = true;
                    }
                    ku += 1u32;
                }

                // Apply the SUMMED deltas once, then clamp (a single nuclide's
                // delta can be legitimately negative from self-shielding, so
                // clamping per slab would be wrong).
                let mut new_e = sigma_e + sum_delta_e;
                let mut new_a = sigma_a + sum_delta_a;
                let mut new_f = sigma_f + sum_delta_f;
                let mut new_i = sigma_i + sum_delta_i;
                if new_e < 0.0 {
                    new_e = 0.0;
                }
                if new_a < 0.0 {
                    new_a = 0.0;
                }
                if new_f < 0.0 {
                    new_f = 0.0;
                }
                if new_i < 0.0 {
                    new_i = 0.0;
                }
                sigma_e_use = new_e;
                sigma_a_use = new_a;
                sigma_f_use = new_f;
                sigma_i_use = new_i;

                // Full-material URR-modified macroscopic capture (n,gamma) for
                // tally scoring = summed absorption minus fission. This
                // INCLUDES non-URR nuclides' smooth capture (they sit inside
                // `sigma_a`), so mixed materials where only some isotopes are
                // URR-in-range still score capture correctly (issue #210). Do
                // NOT build a URR-only capture accumulator here.
                let mut new_g = new_a - new_f;
                if new_g < 0.0 {
                    new_g = 0.0;
                }
                urr_macro_capture = new_g;
            }
            let sigma_e = sigma_e_use;
            let sigma_a = sigma_a_use;
            let sigma_f = sigma_f_use;
            let sigma_i = sigma_i_use;
            let sigma_t = sigma_e + sigma_a + sigma_i + sigma_f;

            // 3. Free-flight sample.
            let d_xi1 = crate::common::pcg32::draw_uniform(state);
            state = d_xi1.state;
            let xi1 = d_xi1.xi;
            let d_xs = -ln_f64(xi1) / sigma_t;

            // 4. Distance to nearest surface. Surface params layout is
            // stride-10 (wide enough for the general quadric's ten
            // coefficients; narrower kinds zero-pad the tail). All eight
            // SurfaceKind discriminants (stype 0..=7) are dispatched below.
            let mut d_boundary = 1e30_f64;
            let mut winner_surface = 0u32;
            let mut s = 0u32;
            while s < n_surfaces {
                let stype = surface_types[s as usize];
                let sb = (s * 10u32) as usize;
                let p0 = surface_params[sb];
                let p1 = surface_params[sb + 1];
                let p2 = surface_params[sb + 2];
                let p3 = surface_params[sb + 3];
                let p4 = surface_params[sb + 4];
                let p5 = surface_params[sb + 5];
                let p6 = surface_params[sb + 6];
                let p7 = surface_params[sb + 7];
                let p8 = surface_params[sb + 8];
                let p9 = surface_params[sb + 9];
                let mut hit = 1e30_f64;
                if stype == 0u32 {
                    // sphere
                    let qx = px - p0;
                    let qy = py - p1;
                    let qz = pz - p2;
                    let b = 2.0 * (qx * dx + qy * dy + qz * dz);
                    let cc = qx * qx + qy * qy + qz * qz - p3 * p3;
                    let disc = b * b - 4.0 * cc;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let t1 = (-b - sqrt_disc) * 0.5;
                        let t2 = (-b + sqrt_disc) * 0.5;
                        if t2 > 1e-12 {
                            hit = t2;
                        }
                        if t1 > 1e-12 {
                            hit = t1;
                        }
                    }
                }
                if stype == 1u32 {
                    // plane
                    let n_dot_d = p0 * dx + p1 * dy + p2 * dz;
                    if n_dot_d != 0.0 {
                        let n_dot_p = p0 * px + p1 * py + p2 * pz;
                        let t = (p3 - n_dot_p) / n_dot_d;
                        // Skip surfaces we're sitting on (within 1e-12)
                        // so a particle that just crossed the surface
                        // doesn't immediately re-cross it.
                        if t > 1e-12 {
                            hit = t;
                        }
                    }
                }
                if stype == 2u32 {
                    // cylinder: p0..p2 origin, p3 radius, p4..p6 axis
                    let qx = px - p0;
                    let qy = py - p1;
                    let qz = pz - p2;
                    let q_dot_a = qx * p4 + qy * p5 + qz * p6;
                    let mx = qx - q_dot_a * p4;
                    let my = qy - q_dot_a * p5;
                    let mz = qz - q_dot_a * p6;
                    let d_dot_a = dx * p4 + dy * p5 + dz * p6;
                    let dxp = dx - d_dot_a * p4;
                    let dyp = dy - d_dot_a * p5;
                    let dzp = dz - d_dot_a * p6;
                    let a_q = dxp * dxp + dyp * dyp + dzp * dzp;
                    let b_q = 2.0 * (dxp * mx + dyp * my + dzp * mz);
                    let c_q = mx * mx + my * my + mz * mz - p3 * p3;
                    if a_q > 0.0 {
                        let disc = b_q * b_q - 4.0 * a_q * c_q;
                        if disc >= 0.0 {
                            let sqrt_disc = disc.sqrt();
                            let inv_2a = 0.5 / a_q;
                            let t1 = (-b_q - sqrt_disc) * inv_2a;
                            let t2 = (-b_q + sqrt_disc) * inv_2a;
                            if t2 > 1e-12 {
                                hit = t2;
                            }
                            if t1 > 1e-12 {
                                hit = t1;
                            }
                        }
                    }
                }
                if stype == 3u32 {
                    // ZTorus: p0..p2 centre, p3 = a (major), p4 = b
                    // (axial minor), p5 = c (radial minor). Axial coord
                    // is z; transverse pair is (x, y).
                    hit =
                        torus_smallest_positive(px - p0, py - p1, pz - p2, dx, dy, dz, p3, p4, p5);
                }
                if stype == 4u32 {
                    // XTorus: axial coord is x; transverse pair is (y, z).
                    hit =
                        torus_smallest_positive(py - p1, pz - p2, px - p0, dy, dz, dx, p3, p4, p5);
                }
                if stype == 5u32 {
                    // YTorus: axial coord is y; transverse pair is (x, z).
                    hit =
                        torus_smallest_positive(px - p0, pz - p2, py - p1, dx, dz, dy, p3, p4, p5);
                }
                if stype == 6u32 {
                    // Quadric: p0..p9 = a, b, c, d, e, f, g, h, j, k.
                    hit = quadric_smallest_positive(
                        px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                    );
                }
                if stype == 7u32 {
                    // Cone: p0..p2 apex, p3..p5 unit axis, p6 = tan²θ.
                    hit =
                        cone_smallest_positive(px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6);
                }
                if hit < d_boundary {
                    d_boundary = hit;
                    winner_surface = s;
                }
                s += 1u32;
            }

            // 5. Take the min.
            let collide_first = d_xs < d_boundary;
            let mut d = d_boundary;
            if collide_first {
                d = d_xs;
            }

            // 6. Score per-tally per-cell-per-energy contributions.
            // For each tally `t`:
            //   - look up the cell-bin via `tally_cell_to_bin`
            //   - skip if the geom cell isn't in the tally's
            //     `CellFilter` (sentinel = u32::MAX)
            //   - binary-search the tally's `log_edges` slice for
            //     the energy bin (slice F: arbitrary monotonic edges)
            //   - score = 1.0 (Flux) | σ_t (Total) | σ_a (Absorption)
            //   - atomic-add `track_length × score × weight` (in
            //     fixed-point) into the indexed slot
            let mut t = 0u32;
            while t < n_tallies {
                let cell_bin = tally_cell_to_bin[(t * n_cells + cell) as usize];
                // Collision-estimator tallies (is_collision == 1) score at
                // the collision site below, NOT per step -- skip them here
                // so they never get both contributions.
                let in_tally = cell_bin != 4_294_967_295u32;
                let track_length_tally = tally_is_collision[t as usize] == 0u32;
                if in_tally && track_length_tally {
                    let n_bins = tally_n_bins[t as usize];
                    let edges_off = tally_edges_offsets[t as usize];
                    // Out-of-range guard: energies below the first edge or
                    // above the last edge are NOT scored (dropped), matching
                    // the CPU `EnergyFilter::get_bin`, which returns `None`
                    // outside range. Without this, sub-floor flux (e.g. He4
                    // down-scatter below a 1 keV tally floor) piles into bin 0
                    // and inflates the spectrum. The first edge is inclusive.
                    let lo_edge = tally_log_edges[edges_off as usize];
                    let hi_edge = tally_log_edges[(edges_off + n_bins) as usize];
                    let in_range = log_e >= lo_edge && log_e <= hi_edge;
                    // Binary-search log_e in this tally's edges slice.
                    let mut lo = 0u32;
                    let mut hi = n_bins;
                    let mut iter = 0u32;
                    while iter < 16u32 && lo + 1u32 < hi {
                        let mid: u32 = (lo + hi) / 2u32;
                        let mid_edge = tally_log_edges[(edges_off + mid) as usize];
                        // `<=` so a particle exactly AT an interior edge
                        // scores into the LOWER bin -- matches the CPU
                        // `EnergyFilter::get_bin` convention
                        // `bins[i] < E <= bins[i+1]` (a discrete source line
                        // on a group boundary otherwise lands one bin high).
                        if log_e <= mid_edge {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                        iter += 1u32;
                    }
                    let mut bin = lo;
                    if bin >= n_bins {
                        bin = n_bins - 1u32;
                    }

                    // Score factor.
                    let kind = tally_score_kinds[t as usize];
                    let mut score = 1.0; // SCORE_FLUX = 0 default
                    if kind == 1u32 {
                        score = sigma_t; // SCORE_TOTAL
                    } else if kind == 2u32 {
                        // SCORE_ABSORPTION = ENDF MT 27 (neutron
                        // disappearance) = σ_a + σ_f. The kernel's
                        // `sigma_a` excludes fission so the kill-vs-
                        // fission branch sampling above stays exact;
                        // re-adding σ_f here matches the CPU MC
                        // convention (`Mt::ABSORPTION` → MT 27) and
                        // a reference code, which both treat fission as part of
                        // absorption.
                        score = sigma_a + sigma_f;
                    } else if kind == 3u32 {
                        // SCORE_PER_MT -- look up the per-tally MT slot
                        // in xs_score_per_mt[mat × n_score_mts × n_grid]
                        // and linearly interpolate at the same idx_lo /
                        // idx_hi the existing kernel uses.
                        let slot = tally_score_data[t as usize];
                        // Recover `n_materials` from `mat_f64_meta` --
                        // it has one row per material with stride
                        // `MAT_F64_COLS = 4u32`.
                        let n_materials: u32 = (mat_f64_meta.len() / 5) as u32;
                        let n_score_mts: u32 =
                            xs_score_per_mt.len() as u32 / (n_materials * n_grid);
                        let mt_grid_off = mat_idx * n_score_mts * n_grid + slot * n_grid;
                        let xs_lo = xs_score_per_mt[(mt_grid_off + idx_lo) as usize];
                        let xs_hi = xs_score_per_mt[(mt_grid_off + idx_hi) as usize];
                        score = xs_lo + (xs_hi - xs_lo) * frac;
                        // URR self-shielding correlation (W184 capture bug).
                        // Inside the URR window the smooth `xs_score_per_mt`
                        // value is uncorrelated with the per-collision URR
                        // draw that perturbed the transport macro, biasing the
                        // score. Score each URR-perturbed reaction with the
                        // perturbed macro the step actually used: capture
                        // (MT 102) = urr_macro_capture, absorption (MT 27) =
                        // sigma_a + sigma_f, elastic (MT 2) = sigma_e, fission
                        // (MT 18) = sigma_f (all rebound to their *_use values
                        // above). No-op when no URR sample fired this step.
                        if urr_fired {
                            let mt_t = tally_score_mt[t as usize];
                            if mt_t == 102u32 {
                                score = urr_macro_capture;
                            } else if mt_t == 27u32 {
                                score = sigma_a + sigma_f;
                            } else if mt_t == 2u32 {
                                score = sigma_e;
                            } else if mt_t == 18u32 {
                                score = sigma_f;
                            }
                        }
                    }

                    // Energy-function weighting (issue #271). Applied to
                    // `score` rather than to `contrib` so the mesh DDA helpers
                    // below, which take `score` by value, pick it up for free.
                    // An out-of-range energy drops the whole event (CPU
                    // `get_weight() == None` -> `return`), which is why this
                    // ANDs into `in_range` instead of scoring zero.
                    let ef_lo = tally_efunc_offsets[t as usize];
                    let ef_hi = tally_efunc_offsets[(t + 1u32) as usize];
                    let mut ef_in_range = true;
                    if ef_hi > ef_lo {
                        let n_ef = tally_efunc_params[ef_lo as usize] as u32;
                        let e_first = tally_efunc_params[(ef_lo + 1u32) as usize];
                        let e_last = tally_efunc_params[(ef_lo + n_ef) as usize];
                        if energy < e_first || energy > e_last {
                            ef_in_range = false;
                        } else {
                            score = score
                                * energy_function_weight_kernel(tally_efunc_params, ef_lo, energy);
                        }
                    }

                    let contrib = d * score * weight;
                    let scale = tally_fixed_point_scales[t as usize];
                    // Symmetric round-to-nearest (twin: `round_fixed_point_bits`).
                    // Non-negative for transport tallies; the else-branch keeps the
                    // two signs symmetric so the matched-stream twin stays bit-identical.
                    let scaled_f = contrib * scale;
                    let scaled = if scaled_f >= 0.0 {
                        (scaled_f + 0.5) as i64
                    } else {
                        -((-scaled_f + 0.5) as i64)
                    };
                    let bits = u64::reinterpret(scaled);
                    let n_t_cells = tally_n_cells[t as usize];
                    let out_off = tally_out_offsets[t as usize];
                    // Non-mesh flat index (`n_mesh == 1`): `out_off + cell*n_bins + e`.
                    // The mesh_direct branch below recomputes the mesh-general
                    // index `out_off + (cell*n_bins + e)*n_mesh + voxel`.
                    let tally_idx = out_off + cell_bin * n_bins + bin;
                    let _ = n_t_cells;
                    if in_range && ef_in_range {
                        if mesh_direct {
                            // Issue #234: mesh models accumulate each contribution
                            // straight into the per-source accumulator (no touched
                            // list). A MESH_NONE tally in a mesh model writes its
                            // single bin; a mesh tally fans the step across voxels.
                            if tally_mesh_kind[t as usize] == 0u32 {
                                src_acc[src_base + tally_idx as usize]
                                    .fetch_add(u64::reinterpret(scaled));
                            } else {
                                let n_mesh = tally_n_mesh[t as usize];
                                let ce = cell_bin * n_bins + bin;
                                let base = out_off + ce * n_mesh;
                                let mo = tally_mesh_params_offsets[t as usize];
                                // Cylindrical (kind 3) vs rectangular (1/2)
                                // voxel walk (issue #279).
                                if tally_mesh_kind[t as usize] == 3u32 {
                                    cyl_mesh_score_src_acc(
                                        tally_mesh_params,
                                        mo,
                                        px,
                                        py,
                                        pz,
                                        dx,
                                        dy,
                                        dz,
                                        d,
                                        score,
                                        weight,
                                        base,
                                        src_base as u32,
                                        scale,
                                        src_acc,
                                    );
                                } else {
                                    mesh_rect_score_src_acc(
                                        tally_mesh_params,
                                        mo,
                                        px,
                                        py,
                                        pz,
                                        dx,
                                        dy,
                                        dz,
                                        d,
                                        score,
                                        weight,
                                        base,
                                        src_base as u32,
                                        scale,
                                        src_acc,
                                    );
                                }
                            }
                        } else if per_history_var {
                            // Batch-free per-history variance (issue #233):
                            // accumulate this history's per-bin PHYSICAL total
                            // locally (dedup on tally_idx). Squared + summed at
                            // history end. Overflow past PERHIST_K goes to the
                            // per-history global spill (EXACT, never dropped).
                            let mut found = false;
                            let mut j = 0u32;
                            while j < th_count {
                                if th_bin[j as usize] == tally_idx {
                                    th_val[j as usize] += contrib;
                                    found = true;
                                }
                                j += 1u32;
                            }
                            if !found {
                                if th_count < PERHIST_K {
                                    th_bin[th_count as usize] = tally_idx;
                                    th_val[th_count as usize] = contrib;
                                    th_count += 1u32;
                                } else {
                                    let mut sfound = false;
                                    let mut k = 0u32;
                                    while k < spill_count {
                                        if spill_bin[spill_base + k as usize] == tally_idx {
                                            spill_val[spill_base + k as usize] += contrib;
                                            sfound = true;
                                        }
                                        k += 1u32;
                                    }
                                    if !sfound {
                                        spill_bin[spill_base + spill_count as usize] = tally_idx;
                                        spill_val[spill_base + spill_count as usize] = contrib;
                                        spill_count += 1u32;
                                    }
                                }
                            }
                        } else {
                            tally_out[tally_idx as usize].fetch_add(bits);
                        }
                    }
                }
                t += 1u32;
            }

            // 7. Move particle by d.
            px += dx * d;
            py += dy * d;
            pz += dz * d;

            // 8. Collision or surface crossing.
            if collide_first {
                // 8a. Collision-estimator tallies. The collision-density
                // flux estimator scores `weight × score_xs / Σ_t` once per
                // real collision (vs the per-step `d × score_xs × weight`
                // track-length path above). Same energy-bin / score-factor
                // / atomic-add path -- only the effective distance changes
                // from `d` to `1 / Σ_t`, and only `is_collision == 1`
                // tallies fire here. Σ_t > 0 holds (the distance was sampled
                // against it), so the reciprocal is finite.
                let inv_sigma_t = 1.0 / sigma_t;
                let mut tc = 0u32;
                while tc < n_tallies {
                    let cell_bin_c = tally_cell_to_bin[(tc * n_cells + cell) as usize];
                    let in_tally_c = cell_bin_c != 4_294_967_295u32;
                    let collision_tally = tally_is_collision[tc as usize] == 1u32;
                    if in_tally_c && collision_tally {
                        let n_bins_c = tally_n_bins[tc as usize];
                        let edges_off_c = tally_edges_offsets[tc as usize];
                        // Out-of-range guard (see the per-step block above):
                        // drop energies outside the filter range instead of
                        // piling them into the first/last bin.
                        let lo_edge_c = tally_log_edges[edges_off_c as usize];
                        let hi_edge_c = tally_log_edges[(edges_off_c + n_bins_c) as usize];
                        let in_range_c = log_e >= lo_edge_c && log_e <= hi_edge_c;
                        let mut lo_c = 0u32;
                        let mut hi_c = n_bins_c;
                        let mut iter_c = 0u32;
                        while iter_c < 16u32 && lo_c + 1u32 < hi_c {
                            let mid_c: u32 = (lo_c + hi_c) / 2u32;
                            let mid_edge_c = tally_log_edges[(edges_off_c + mid_c) as usize];
                            if log_e <= mid_edge_c {
                                hi_c = mid_c;
                            } else {
                                lo_c = mid_c;
                            }
                            iter_c += 1u32;
                        }
                        let mut bin_c = lo_c;
                        if bin_c >= n_bins_c {
                            bin_c = n_bins_c - 1u32;
                        }

                        // Same score factor as the per-step block.
                        let kind_c = tally_score_kinds[tc as usize];
                        let mut score_c = 1.0; // SCORE_FLUX
                        if kind_c == 1u32 {
                            score_c = sigma_t; // SCORE_TOTAL
                        } else if kind_c == 2u32 {
                            score_c = sigma_a + sigma_f; // SCORE_ABSORPTION (MT 27)
                        } else if kind_c == 3u32 {
                            let slot_c = tally_score_data[tc as usize];
                            let n_materials_c: u32 = (mat_f64_meta.len() / 5) as u32;
                            let n_score_mts_c: u32 =
                                xs_score_per_mt.len() as u32 / (n_materials_c * n_grid);
                            let mt_grid_off_c = mat_idx * n_score_mts_c * n_grid + slot_c * n_grid;
                            let xs_lo_c = xs_score_per_mt[(mt_grid_off_c + idx_lo) as usize];
                            let xs_hi_c = xs_score_per_mt[(mt_grid_off_c + idx_hi) as usize];
                            score_c = xs_lo_c + (xs_hi_c - xs_lo_c) * frac;
                            // URR self-shielding correlation (see the per-step
                            // block above). Score URR-window capture / absorption
                            // with the perturbed macro the step used. No-op when
                            // no URR sample fired this step.
                            if urr_fired {
                                let mt_tc = tally_score_mt[tc as usize];
                                if mt_tc == 102u32 {
                                    score_c = urr_macro_capture;
                                } else if mt_tc == 27u32 {
                                    score_c = sigma_a + sigma_f;
                                } else if mt_tc == 2u32 {
                                    score_c = sigma_e;
                                } else if mt_tc == 18u32 {
                                    score_c = sigma_f;
                                }
                            }
                        }

                        // Energy-function weighting (issue #271), same rule as
                        // the per-step block: multiply the score, or drop the
                        // event when the incident energy is off the table.
                        let ef_lo_c = tally_efunc_offsets[tc as usize];
                        let ef_hi_c = tally_efunc_offsets[(tc + 1u32) as usize];
                        let mut ef_in_range_c = true;
                        if ef_hi_c > ef_lo_c {
                            let n_ef_c = tally_efunc_params[ef_lo_c as usize] as u32;
                            let e_first_c = tally_efunc_params[(ef_lo_c + 1u32) as usize];
                            let e_last_c = tally_efunc_params[(ef_lo_c + n_ef_c) as usize];
                            if energy < e_first_c || energy > e_last_c {
                                ef_in_range_c = false;
                            } else {
                                score_c = score_c
                                    * energy_function_weight_kernel(
                                        tally_efunc_params,
                                        ef_lo_c,
                                        energy,
                                    );
                            }
                        }

                        let contrib_c = score_c * weight * inv_sigma_t;
                        let scale_c = tally_fixed_point_scales[tc as usize];
                        // Symmetric round-to-nearest (twin: `round_fixed_point_bits`).
                        let scaled_cf = contrib_c * scale_c;
                        let scaled_c = if scaled_cf >= 0.0 {
                            (scaled_cf + 0.5) as i64
                        } else {
                            -((-scaled_cf + 0.5) as i64)
                        };
                        let bits_c = u64::reinterpret(scaled_c);
                        let out_off_c = tally_out_offsets[tc as usize];
                        let tally_idx_c = out_off_c + cell_bin_c * n_bins_c + bin_c;
                        if in_range_c && ef_in_range_c {
                            if mesh_direct {
                                // Issue #234: collision estimator bins at the
                                // collision point (position already advanced by
                                // `d`). A mesh tally scores the single voxel
                                // containing the point (dropped if outside the
                                // mesh, matching the CPU `get_bin` -> None).
                                if tally_mesh_kind[tc as usize] == 0u32 {
                                    src_acc[src_base + tally_idx_c as usize]
                                        .fetch_add(u64::reinterpret(scaled_c));
                                } else {
                                    let mo = tally_mesh_params_offsets[tc as usize];
                                    // Cylindrical (kind 3) vs rectangular (1/2)
                                    // point binning (issue #279).
                                    let voxel = if tally_mesh_kind[tc as usize] == 3u32 {
                                        cyl_mesh_bin_at_kernel(tally_mesh_params, mo, px, py, pz)
                                    } else {
                                        rect_mesh_bin_at_kernel(tally_mesh_params, mo, px, py, pz)
                                    };
                                    if voxel != 4_294_967_295u32 {
                                        let n_mesh = tally_n_mesh[tc as usize];
                                        let ce = cell_bin_c * n_bins_c + bin_c;
                                        let idx = out_off_c + ce * n_mesh + voxel;
                                        src_acc[src_base + idx as usize]
                                            .fetch_add(u64::reinterpret(scaled_c));
                                    }
                                }
                            } else if per_history_var {
                                // Batch-free per-history variance (issue #233):
                                // same touched-list + spill accumulation as the
                                // per-step (track-length) block above, for the
                                // collision-estimator contribution.
                                let mut found = false;
                                let mut j = 0u32;
                                while j < th_count {
                                    if th_bin[j as usize] == tally_idx_c {
                                        th_val[j as usize] += contrib_c;
                                        found = true;
                                    }
                                    j += 1u32;
                                }
                                if !found {
                                    if th_count < PERHIST_K {
                                        th_bin[th_count as usize] = tally_idx_c;
                                        th_val[th_count as usize] = contrib_c;
                                        th_count += 1u32;
                                    } else {
                                        let mut sfound = false;
                                        let mut k = 0u32;
                                        while k < spill_count {
                                            if spill_bin[spill_base + k as usize] == tally_idx_c {
                                                spill_val[spill_base + k as usize] += contrib_c;
                                                sfound = true;
                                            }
                                            k += 1u32;
                                        }
                                        if !sfound {
                                            spill_bin[spill_base + spill_count as usize] =
                                                tally_idx_c;
                                            spill_val[spill_base + spill_count as usize] =
                                                contrib_c;
                                            spill_count += 1u32;
                                        }
                                    }
                                }
                            } else {
                                tally_out[tally_idx_c as usize].fetch_add(bits_c);
                            }
                        }
                    }
                    tc += 1u32;
                }

                // 8b. Coupled neutron->photon production (S4b). Gated on the
                // runtime `coupled_enabled` flag: when OFF, ZERO photon RNG is
                // drawn and ZERO bank writes happen, so this whole block is
                // inert and the neutron path is byte-identical to a
                // neutron-only run. The photon emission uses the PRE-scatter
                // energy / direction / position / weight / cell (the MT sample
                // below has not run yet) and interpolates the photon-production
                // tables linearly in the collision bracket `[idx_lo, idx_hi]`
                // (factor `frac`), the same lookup the macroscopic `sigma_t`
                // and the CPU `lookup_photon_prod` / `photon_rxn_xs_interp` use.
                //
                // RNG isolation: a CHILD PCG state is forked from the neutron
                // `state` by one `pcg_next` step seeded with the neutron state
                // mixed with a constant. ALL photon draws thread the child
                // state; the neutron `state` is never touched here, so the
                // coupled-on neutron transport is identical to neutron-only.
                if coupled_enabled[0] == 1u32 {
                    // Fork a child PCG state. Mixing in a constant before the
                    // PCG step decorrelates the photon stream from the neutron
                    // stream that continues with `state` untouched.
                    let fork = crate::common::pcg32::pcg_next(state ^ 0x9E37_79B9_7F4A_7C15u64);
                    let mut photon_state = fork.state;

                    // Per-material photon-table bases / counts.
                    let pp_base = photon_pp_base_per_material[mat_idx as usize];
                    let prod_base = photon_prod_base_per_material[mat_idx as usize];
                    let n_product = photon_n_product_per_material[mat_idx as usize];

                    // Photon COUNT: y_t = photon_prod(E) / sigma_t, split into
                    // floor + Bernoulli (one child draw). `photon_prod` is
                    // linearly interpolated in the collision bracket
                    // `[idx_lo, idx_hi]` (factor `frac`) so the numerator
                    // tracks the exact collision energy, matching the CPU
                    // `lookup_photon_prod(i_grid, f)` against the interpolated
                    // `sigma_t` -- the same pattern the decay block (8c) uses.
                    let pp_lo = photon_prod[(pp_base + idx_lo) as usize];
                    let pp_hi = photon_prod[(pp_base + idx_hi) as usize];
                    let pp_g = pp_lo + (pp_hi - pp_lo) * frac;
                    let n_photons = sample_photon_count(pp_g, sigma_t, photon_state);
                    // `sample_photon_count` consumes one draw; advance the
                    // child state by re-running the same draw so the per-photon
                    // loop below starts from the post-count state.
                    let d_cnt = crate::common::pcg32::draw_uniform(photon_state);
                    photon_state = d_cnt.state;

                    // Per-photon loop, bounded by the comptime cap. `emitted`
                    // counts how many we have banked so the loop stops once the
                    // sampled count is reached (cubecl-friendly: full-range
                    // loop guarded by a predicate, no early break).
                    let cap = 16u32; // MAX_PHOTONS_PER_COLLISION
                    let mut emitted = 0u32;
                    let mut pk = 0u32;
                    while pk < cap {
                        if emitted < n_photons {
                            // Select the (reaction, product) pair (one child
                            // draw); the returned index is GLOBAL. The per-
                            // product weights are interpolated in the collision
                            // bracket `[idx_lo, idx_hi]` (factor `frac`) so the
                            // selection matches the CPU `sample_photon_product`,
                            // which uses `photon_rxn_xs_interp` /
                            // `product_yield.evaluate` at the exact energy.
                            let product_idx = sample_photon_product(
                                photon_rxn_xs,
                                photon_prod_rxn_idx,
                                photon_prod_yield_grid,
                                prod_base,
                                n_product,
                                n_grid,
                                idx_lo,
                                idx_hi,
                                frac,
                                photon_state,
                            );
                            let d_sel = crate::common::pcg32::draw_uniform(photon_state);
                            photon_state = d_sel.state;

                            // Sample (E_out, mu) for the selected product using
                            // the PRE-scatter neutron energy.
                            let kin = sample_photon_kinematics(
                                product_idx,
                                energy,
                                photon_prod_eout_kind,
                                photon_prod_line_energy,
                                photon_prod_primary_flag,
                                photon_prod_awr,
                                photon_prod_dist_slot,
                                photon_pa_n_energies,
                                photon_pa_ae_offset,
                                photon_pa_mu_offset,
                                photon_pa_energy_grid,
                                photon_pa_n_mu,
                                photon_pa_mu,
                                photon_pa_cdf,
                                photon_pa_pdf,
                                photon_pa_interp,
                                photon_ct_ae_offset,
                                photon_ct_x_offset,
                                photon_ct_energy_grid,
                                photon_ct_n_x,
                                photon_ct_x,
                                photon_ct_cdf,
                                photon_ct_p,
                                photon_ct_interp,
                                photon_ct_n_discrete,
                                photon_ct_n_eout,
                                photon_ct_hist,
                                photon_state,
                            );
                            photon_state = kin.state;

                            // Azimuth: the same Marsaglia uniform-azimuth
                            // sampler the neutron scatter uses (statistically
                            // matches the CPU's uniform phi; S7 is a
                            // statistical, not bit-exact, acceptance test).
                            let mphi =
                                crate::common::sampling::marsaglia_phi::marsaglia_cos_sin_phi(
                                    photon_state,
                                );
                            photon_state = mphi.state;

                            // Rotate (dx, dy, dz) by (mu, phi) -- the SAME
                            // rotation the kernel applies to scattered neutrons.
                            let mu_p = kin.mu;
                            let cos_phi_p = mphi.cos_phi;
                            let sin_phi_p = mphi.sin_phi;
                            let sin_th_sq = 1.0 - mu_p * mu_p;
                            let mut sin_th = 0.0;
                            if sin_th_sq > 0.0 {
                                sin_th = sin_th_sq.sqrt();
                            }
                            let one_minus_w_sq = 1.0 - dz * dz;
                            let mut new_dx = sin_th * cos_phi_p;
                            let mut new_dy = sin_th * sin_phi_p;
                            let mut new_dz = mu_p;
                            if dz < 0.0 {
                                new_dy = -new_dy;
                                new_dz = -mu_p;
                            }
                            if one_minus_w_sq > 1e-14 {
                                let sin_phi_w = one_minus_w_sq.sqrt();
                                new_dx = mu_p * dx
                                    + sin_th * (dx * dz * cos_phi_p - dy * sin_phi_p) / sin_phi_w;
                                new_dy = mu_p * dy
                                    + sin_th * (dy * dz * cos_phi_p + dx * sin_phi_p) / sin_phi_w;
                                new_dz = mu_p * dz - sin_th * sin_phi_w * cos_phi_p;
                            }

                            // Append the photon to the device bank (inlined
                            // atomic fetch-add slot reservation + record write,
                            // mirroring `bank_append_kernel`). The banked
                            // photon carries its own child-derived transport
                            // seed and generation 0; S5 drains and transports
                            // it.
                            // One unconditional child draw per emitted-photon
                            // attempt: the advance keeps successive photons on
                            // distinct sub-streams, and its output word is the
                            // banked u32 transport seed (the sub-pass expands
                            // it back to a 64-bit state, issue #274). Drawn
                            // BEFORE the e_out / capacity gates so the child
                            // stream schedule stays unconditional.
                            let d_seed = crate::common::pcg32::pcg_next(photon_state);
                            photon_state = d_seed.state;
                            // Only bank a physically real photon. A non-positive
                            // outgoing energy means the product had no sampleable
                            // outgoing distribution (an empty n_x == 0 CT row, or
                            // a NONE-classified law); the CPU
                            // `sample_secondary_photons` drops these (e_out <= 0),
                            // so skip the bank rather than emit a zero-energy
                            // photon (issue #175). All RNG draws stay
                            // unconditional, so the stream is unchanged.
                            if kin.e_out > 0.0 {
                                let capacity = (bank_f64.len() / 8) as u64;
                                let slot = bank_count[0].fetch_add(1u64);
                                if slot < capacity {
                                    let f = (slot * 8u64) as usize;
                                    bank_f64[f] = kin.e_out;
                                    bank_f64[f + 1] = px;
                                    bank_f64[f + 2] = py;
                                    bank_f64[f + 3] = pz;
                                    bank_f64[f + 4] = new_dx;
                                    bank_f64[f + 5] = new_dy;
                                    bank_f64[f + 6] = new_dz;
                                    bank_f64[f + 7] = weight;
                                    let u = (slot * 4u64) as usize;
                                    bank_u32[u] = PTYPE_PHOTON;
                                    bank_u32[u + 1] = cell;
                                    bank_u32[u + 2] = d_seed.rand;
                                    bank_u32[u + 3] = 0u32;
                                    // Per-source variance (issue #233 Stage 3): stamp the
                                    // originating source neutron so the coupled photon
                                    // sub-pass folds this secondary into the same sample.
                                    bank_source_idx[slot as usize] = my_source_idx;
                                } else {
                                    bank_overflow[0].fetch_add(1u64);
                                }
                            }

                            emitted += 1u32;
                        }
                        pk += 1u32;
                    }
                }

                // 8c. D1S decay-photon production. Mutually exclusive with the
                // prompt block (8b) above: only one of `coupled_enabled` /
                // `decay_enabled` is ever 1. Emits EXACTLY ONE photon per
                // collision with weight `w * y_t` (`y_t = decay_photon_prod /
                // sigma_t`), matching the CPU `sample_decay_photons` implicit-
                // capture estimator (no floor+Bernoulli count). The photon's
                // energy is a discrete decay line of a chain emitter and its
                // parent-nuclide id is stamped into the bank `gen` slot for
                // `parent_nuclides` tally binning. Isotropic direction.
                //
                // RNG isolation mirrors 8b: a CHILD PCG state forked from the
                // neutron `state` with a DISTINCT mixing constant, so the
                // neutron stream is never perturbed (decay-on neutron transport
                // equals neutron-only) and the decay stream is decorrelated.
                if decay_enabled[0] == 1u32 {
                    let dfork = crate::common::pcg32::pcg_next(state ^ 0xFF51_AFD7_ED55_8CCDu64);
                    let mut dstate = dfork.state;

                    // Per-material decay metadata: pp_base, ch_base, ch_count.
                    let dmeta_off = mat_idx * 3u32; // DECAY_META_COLS = 3
                    let d_pp_base = decay_meta[dmeta_off as usize];
                    let d_ch_base = decay_meta[(dmeta_off + 1u32) as usize];
                    let d_ch_count = decay_meta[(dmeta_off + 2u32) as usize];

                    // Aggregate decay photon-production xs at the collision
                    // energy (linear-E interpolation on the master grid, the
                    // same `idx_lo` / `frac` the macroscopic lookup used --
                    // mirrors the CPU `lookup_photon_prod(i_grid, interp)`).
                    let dpp_lo = decay_photon_prod[(d_pp_base + idx_lo) as usize];
                    let dpp_hi = decay_photon_prod[(d_pp_base + idx_hi) as usize];
                    let decay_pp = dpp_lo + (dpp_hi - dpp_lo) * frac;

                    if decay_pp > 0.0 && d_ch_count > 0u32 {
                        let y_t = decay_pp / sigma_t;
                        let photon_wgt = weight * y_t;

                        // Channel selection: cutoff = xi * decay_pp; walk the
                        // material's channels accumulating their interpolated
                        // weighted xs until the cumulative exceeds the cutoff.
                        // Matches CPU `sample_decay_photons`'s channel walk.
                        let dsel = crate::common::pcg32::draw_uniform(dstate);
                        dstate = dsel.state;
                        let cutoff = dsel.xi * decay_pp;

                        let mut acc = 0.0_f64;
                        // Default to the last channel (CPU's `selected_idx`
                        // initialisation) so a floating-point shortfall still
                        // picks a real channel.
                        let mut sel = d_ch_count - 1u32;
                        let mut found = false;
                        let mut ci = 0u32;
                        while ci < d_ch_count {
                            if !found {
                                let gch = d_ch_base + ci;
                                let row = gch * n_grid;
                                let cx_lo = decay_ch_xs[(row + idx_lo) as usize];
                                let cx_hi = decay_ch_xs[(row + idx_hi) as usize];
                                let cx = cx_lo + (cx_hi - cx_lo) * frac;
                                if cx > 0.0 {
                                    acc += cx;
                                    sel = ci;
                                    if acc > cutoff {
                                        found = true;
                                    }
                                }
                            }
                            ci += 1u32;
                        }

                        let gsel = d_ch_base + sel;

                        // Discrete energy: sample a line from the channel's
                        // cumulative intensity CDF (normalized to 1.0 at the
                        // last line). Find the first line with cdf >= xi2.
                        let e_base = decay_ch_e_meta[(gsel * 2u32) as usize];
                        let e_count = decay_ch_e_meta[(gsel * 2u32 + 1u32) as usize];
                        let denergy_draw = crate::common::pcg32::draw_uniform(dstate);
                        dstate = denergy_draw.state;
                        let xi2 = denergy_draw.xi;
                        // Default to the last line (covers the xi2 == 1.0 edge).
                        let mut e_idx = e_count - 1u32;
                        let mut e_found = false;
                        let mut li = 0u32;
                        while li < e_count {
                            if !e_found {
                                let cdf = decay_ch_intensity_cdf[(e_base + li) as usize];
                                if xi2 <= cdf {
                                    e_idx = li;
                                    e_found = true;
                                }
                            }
                            li += 1u32;
                        }
                        let e_out = decay_ch_energies[(e_base + e_idx) as usize];

                        if e_out > 0.0 {
                            // Isotropic direction: mu = 2u - 1, phi = 2pi u
                            // (matches CPU `sample_isotropic_direction`).
                            let dmu_draw = crate::common::pcg32::draw_uniform(dstate);
                            dstate = dmu_draw.state;
                            let mu_d = 2.0 * dmu_draw.xi - 1.0;
                            let dphi_draw = crate::common::pcg32::draw_uniform(dstate);
                            dstate = dphi_draw.state;
                            let phi_d = std::f64::consts::TAU * dphi_draw.xi;
                            let sin_th_sq_d = 1.0 - mu_d * mu_d;
                            let mut sin_th_d = 0.0;
                            if sin_th_sq_d > 0.0 {
                                sin_th_d = sin_th_sq_d.sqrt();
                            }
                            let ddx = sin_th_d * cos_f64(phi_d);
                            let ddy = sin_th_d * sin_f64(phi_d);
                            let ddz = mu_d;

                            // Bank the decay photon: parent-nuclide id in the
                            // `gen` slot (u32 idx 3). The banked transport
                            // seed is the output word of the final child
                            // state (no advance; the state is dead after).
                            let parent_id = decay_ch_parent_id[gsel as usize];
                            let dcapacity = (bank_f64.len() / 8) as u64;
                            let dslot = bank_count[0].fetch_add(1u64);
                            if dslot < dcapacity {
                                let f = (dslot * 8u64) as usize;
                                bank_f64[f] = e_out;
                                bank_f64[f + 1] = px;
                                bank_f64[f + 2] = py;
                                bank_f64[f + 3] = pz;
                                bank_f64[f + 4] = ddx;
                                bank_f64[f + 5] = ddy;
                                bank_f64[f + 6] = ddz;
                                bank_f64[f + 7] = photon_wgt;
                                let u = (dslot * 4u64) as usize;
                                bank_u32[u] = PTYPE_PHOTON;
                                bank_u32[u + 1] = cell;
                                // Banked u32 transport seed: the XSH-RR output
                                // word of the (dead-after-this) child state; the
                                // sub-pass expands it back to a 64-bit state
                                // (issue #274). No advance, so the child stream
                                // schedule is unchanged.
                                bank_u32[u + 2] = crate::common::pcg32::pcg_out(dstate);
                                bank_u32[u + 3] = parent_id;
                                // Per-source variance (issue #233 Stage 3): D1S decay
                                // photons descend from the source neutron (via the
                                // activated nuclide), so they inherit its source index.
                                bank_source_idx[dslot as usize] = my_source_idx;
                            } else {
                                bank_overflow[0].fetch_add(1u64);
                            }
                        }
                    }
                }

                // 8c. Per-collision nuclide selection (issue #74). In a
                // multi-nuclide material, pick which nuclide is struck
                // proportional to its macroscopic total xs at the collision
                // energy, then override `target_mass` with that nuclide's AWR so
                // the elastic kinematics below use the exact per-nuclide mass
                // (the material-average mass under-moderates -- an H2O sphere
                // loses ~half its thermal flux). The reaction-type four-way
                // split stays driven by the (already correct) aggregate
                // macroscopic partials; only the elastic AWR is per-nuclide in
                // Stage 1. Gated on `count > 1`: a single-nuclide material draws
                // NO random here and is byte-identical to the pre-#74 stream.
                let nuc_off = mat_nuclide_meta[(mat_idx * 2u32) as usize];
                let nuc_count = mat_nuclide_meta[(mat_idx * 2u32 + 1u32) as usize];
                // Global slab of the struck nuclide: `nuc_off` (the material's
                // first nuclide) unless the selection below picks another. Used
                // to index per-nuclide buffers (elastic AWR + elastic angle).
                // Single-nuclide materials keep `slab == nuc_off`, which equals
                // the old `mat_idx` row, so behaviour is byte-identical.
                let mut slab = nuc_off;
                // Reaction-split partials. Default to the material-aggregate
                // values (exact + byte-identical for single-nuclide materials);
                // a multi-nuclide collision overrides them below with the struck
                // nuclide's own partials so the four-way split mirrors CPU
                // `Nuclide::sample_reaction_type` on the selected nuclide.
                let mut sigma_e_rx = sigma_e;
                let mut sigma_a_rx = sigma_a;
                let mut sigma_i_rx = sigma_i;
                let mut sigma_f_rx = sigma_f;
                if nuc_count > 1u32 {
                    // Per-nuclide macro total at the collision energy, linearly
                    // interpolated in the FINE bracket `[idx_lo_f, idx_hi_f]`
                    // (issue #212). Sum is the selection denominator (equals the
                    // smooth Σ_t when no URR). `nuc_macro_total` rides the
                    // per-material FINE grid: slab `nuc_off + j`'s row base is
                    // `fine_nuc_base + j * fine_n`; single-material =>
                    // `fine_nuc_base == 0`, `fine_n == n_grid`, byte-identical.
                    // A nuclide's share of the collision density follows the
                    // cross section that actually governed the flight, so an
                    // in-range URR nuclide is weighted by its PERTURBED total,
                    // not the table average (issue #347). `urr_band_live` is 1
                    // only when this walk holds a band drawn at THIS energy.
                    // Belt and braces: a band held from an earlier energy
                    // survives non-URR steps by design (#342), but it cannot
                    // corrupt a weight anyway, since `urr_held` is only set when
                    // some slab was in range at that energy and
                    // `urr_slab_partials` re-checks the same range per slab, so
                    // a stale band reaches only slabs that report `fired == 0`.
                    // Kept because it makes the invariant local, and the two
                    // range tests living in different functions is exactly the
                    // kind of coupling that rots.
                    let urr_band_live = if urr_held == 1u32 && urr_energy == energy {
                        1u32
                    } else {
                        0u32
                    };
                    let mut sigma_t_nuc = 0.0_f64;
                    let mut j = 0u32;
                    while j < nuc_count {
                        let row = fine_nuc_base + j * fine_n;
                        let v_lo = nuc_macro_total[(row + idx_lo_f) as usize];
                        let v_hi = nuc_macro_total[(row + idx_hi_f) as usize];
                        let mut w = v_lo + (v_hi - v_lo) * frac_f;
                        if urr_band_live == 1u32 {
                            let q_lo = (fine_nuc_base + j * fine_n + idx_lo_f) * 4u32;
                            let q_hi = (fine_nuc_base + j * fine_n + idx_hi_f) * 4u32;
                            let we_lo = nuc_partial_xs[q_lo as usize];
                            let we_hi = nuc_partial_xs[q_hi as usize];
                            let wb_e = we_lo + (we_hi - we_lo) * frac_f;
                            let wa_lo = nuc_partial_xs[(q_lo + 1u32) as usize];
                            let wa_hi = nuc_partial_xs[(q_hi + 1u32) as usize];
                            let wb_a = wa_lo + (wa_hi - wa_lo) * frac_f;
                            let wi_lo = nuc_partial_xs[(q_lo + 2u32) as usize];
                            let wi_hi = nuc_partial_xs[(q_hi + 2u32) as usize];
                            let wb_i = wi_lo + (wi_hi - wi_lo) * frac_f;
                            let wf_lo = nuc_partial_xs[(q_lo + 3u32) as usize];
                            let wf_hi = nuc_partial_xs[(q_hi + 3u32) as usize];
                            let wb_f = wf_lo + (wf_hi - wf_lo) * frac_f;
                            let wp = crate::common::urr::urr_slab_partials(
                                urr_meta,
                                urr_ae_offset,
                                urr_cdf_offset,
                                urr_energy_grid,
                                urr_cdf,
                                urr_xs,
                                urr_atom_density,
                                nuc_off + j,
                                energy,
                                urr_base,
                                wb_e,
                                wb_a,
                                wb_i,
                                wb_f,
                            );
                            if wp.fired == 1u32 {
                                w = wp.elastic + wp.absorption + wp.inelastic + wp.fission;
                            }
                        }
                        sigma_t_nuc += w;
                        j += 1u32;
                    }
                    // One scaled uniform draw, then a cumulative walk: pick the
                    // first nuclide whose running sum exceeds `xi_n`, clamped to
                    // the last (mirrors CPU `sample_collision_data`'s strict-`<`
                    // walk and the `select_nuclide` twin).
                    let d_nuc = crate::common::pcg32::draw_uniform(state);
                    state = d_nuc.state;
                    let xi_n = d_nuc.xi * sigma_t_nuc;
                    let mut accum = 0.0_f64;
                    let mut chosen = 0u32;
                    let mut k = 0u32;
                    while k < nuc_count {
                        let row = fine_nuc_base + k * fine_n;
                        let v_lo = nuc_macro_total[(row + idx_lo_f) as usize];
                        let v_hi = nuc_macro_total[(row + idx_hi_f) as usize];
                        let mut w = v_lo + (v_hi - v_lo) * frac_f;
                        if urr_band_live == 1u32 {
                            let q_lo = (fine_nuc_base + k * fine_n + idx_lo_f) * 4u32;
                            let q_hi = (fine_nuc_base + k * fine_n + idx_hi_f) * 4u32;
                            let we_lo = nuc_partial_xs[q_lo as usize];
                            let we_hi = nuc_partial_xs[q_hi as usize];
                            let wb_e = we_lo + (we_hi - we_lo) * frac_f;
                            let wa_lo = nuc_partial_xs[(q_lo + 1u32) as usize];
                            let wa_hi = nuc_partial_xs[(q_hi + 1u32) as usize];
                            let wb_a = wa_lo + (wa_hi - wa_lo) * frac_f;
                            let wi_lo = nuc_partial_xs[(q_lo + 2u32) as usize];
                            let wi_hi = nuc_partial_xs[(q_hi + 2u32) as usize];
                            let wb_i = wi_lo + (wi_hi - wi_lo) * frac_f;
                            let wf_lo = nuc_partial_xs[(q_lo + 3u32) as usize];
                            let wf_hi = nuc_partial_xs[(q_hi + 3u32) as usize];
                            let wb_f = wf_lo + (wf_hi - wf_lo) * frac_f;
                            let wp = crate::common::urr::urr_slab_partials(
                                urr_meta,
                                urr_ae_offset,
                                urr_cdf_offset,
                                urr_energy_grid,
                                urr_cdf,
                                urr_xs,
                                urr_atom_density,
                                nuc_off + k,
                                energy,
                                urr_base,
                                wb_e,
                                wb_a,
                                wb_i,
                                wb_f,
                            );
                            if wp.fired == 1u32 {
                                w = wp.elastic + wp.absorption + wp.inelastic + wp.fission;
                            }
                        }
                        accum += w;
                        if xi_n >= accum {
                            chosen += 1u32;
                        }
                        k += 1u32;
                    }
                    if chosen >= nuc_count {
                        chosen = nuc_count - 1u32;
                    }
                    slab = nuc_off + chosen;
                    target_mass = nuc_awr[slab as usize];

                    // Reaction-type partials for the SELECTED nuclide (#74
                    // Stage 2b): density-weighted elastic / absorption /
                    // inelastic / fission interpolated at the collision bracket.
                    // Replaces the material-aggregate split so a chosen nuclide's
                    // inelastic branch is sampled in proportion to ITS own
                    // inelastic xs (the H2O / natFe fix). `nuc_partial_xs` rides
                    // the per-material FINE grid, packed tight-CSR
                    // `[sum_m nuc_count_m * fine_n_m x 4]` (cols e/a/i/f, issue
                    // #212): slab `slab`'s element base is `fine_nuc_base +
                    // (slab - nuc_off) * fine_n`, times 4. Single-material =>
                    // `fine_nuc_base == 0`, `nuc_off == 0`, `fine_n == n_grid`,
                    // byte-identical.
                    let p_lo = (fine_nuc_base + (slab - nuc_off) * fine_n + idx_lo_f) * 4u32;
                    let p_hi = (fine_nuc_base + (slab - nuc_off) * fine_n + idx_hi_f) * 4u32;
                    let e_lo_p = nuc_partial_xs[p_lo as usize];
                    let e_hi_p = nuc_partial_xs[p_hi as usize];
                    sigma_e_rx = e_lo_p + (e_hi_p - e_lo_p) * frac_f;
                    let a_lo_p = nuc_partial_xs[(p_lo + 1u32) as usize];
                    let a_hi_p = nuc_partial_xs[(p_hi + 1u32) as usize];
                    sigma_a_rx = a_lo_p + (a_hi_p - a_lo_p) * frac_f;
                    let i_lo_p = nuc_partial_xs[(p_lo + 2u32) as usize];
                    let i_hi_p = nuc_partial_xs[(p_hi + 2u32) as usize];
                    sigma_i_rx = i_lo_p + (i_hi_p - i_lo_p) * frac_f;
                    let f_lo_p = nuc_partial_xs[(p_lo + 3u32) as usize];
                    let f_hi_p = nuc_partial_xs[(p_hi + 3u32) as usize];
                    sigma_f_rx = f_lo_p + (f_hi_p - f_lo_p) * frac_f;
                    // Split the struck nuclide on the SAME band its selection
                    // and the flight used (issue #347). Smooth partials here
                    // would mis-weight capture against scatter: the URR capture
                    // fraction moves strongly with the band (W184 at 50 keV is
                    // 7.3% in a low band against 1.95% mid-band).
                    if urr_band_live == 1u32 {
                        let sp = crate::common::urr::urr_slab_partials(
                            urr_meta,
                            urr_ae_offset,
                            urr_cdf_offset,
                            urr_energy_grid,
                            urr_cdf,
                            urr_xs,
                            urr_atom_density,
                            slab,
                            energy,
                            urr_base,
                            sigma_e_rx,
                            sigma_a_rx,
                            sigma_i_rx,
                            sigma_f_rx,
                        );
                        if sp.fired == 1u32 {
                            sigma_e_rx = sp.elastic;
                            sigma_a_rx = sp.absorption;
                            sigma_i_rx = sp.inelastic;
                            sigma_f_rx = sp.fission;
                        }
                    }
                }

                // MT sample. Four-way: elastic / inelastic / fission /
                // absorption. Fission used to fold into σ_a; slice G
                // pulls it out so the kernel can sample the surviving
                // neutron's outgoing energy from a Watt χ-spectrum
                // and multiply weight by ν̄ (variance-reduction
                // equivalent of emitting `nu_bar` independent prompt
                // neutrons -- same trick as MT 16/17 with their fixed
                // yields of 2/3, just generalised to a fractional ν̄).
                let d_xi2 = crate::common::pcg32::draw_uniform(state);
                state = d_xi2.state;
                let xi2 = d_xi2.xi;
                // Survival biasing (implicit capture): when enabled and the
                // scatter+fission mass is positive, renormalise the
                // selection over `sigma_sf = sigma_e + sigma_i + sigma_f`
                // (drop the capture mass) and discount the weight by
                // `sigma_sf / sigma_t`. The `xi2` draw above is unchanged, so
                // an OFF run keeps the analog branch boundaries and RNG
                // schedule byte-for-byte. A pure absorber (`sigma_sf == 0`)
                // falls through to the analog kill below.
                // Reaction-type split runs against the SELECTED nuclide's
                // partials (#74 Stage 2b). For single-nuclide materials the
                // `*_rx` values equal the material-aggregate (URR-perturbed)
                // `sigma_*` exactly, and `sigma_t_rx == sigma_t`, so the split
                // is byte-identical to the pre-Stage-2b flow.
                let survival_on = survival_params[0] != 0.0;
                let sigma_t_rx = sigma_e_rx + sigma_a_rx + sigma_i_rx + sigma_f_rx;
                let sigma_sf = sigma_e_rx + sigma_i_rx + sigma_f_rx;
                let mut sel_denom = sigma_t_rx;
                if survival_on && sigma_sf > 0.0 {
                    sel_denom = sigma_sf;
                    weight = weight * (sigma_sf / sigma_t_rx);
                }
                let p_elastic = sigma_e_rx / sel_denom;
                let p_scatter = (sigma_e_rx + sigma_i_rx) / sel_denom;
                // With survival biasing on (and `sigma_sf > 0`) the capture
                // mass is gone, so `p_fission_or_scatter == 1.0` and the
                // absorption-kill branch is never taken. Off (or pure
                // absorber): the analog `sigma_sf / sigma_t_rx` threshold.
                let p_fission_or_scatter = sigma_sf / sel_denom;

                if xi2 >= p_fission_or_scatter {
                    // Absorbed.
                    alive = 0u32;
                } else {
                    // Sample scatter mu and update energy. The
                    // Marsaglia rejection + 3D rotation that follows
                    // is shared between elastic and inelastic -- only
                    // the energy and `mu_lab` differ, computed inside
                    // the per-branch block below.
                    let d_xi3 = crate::common::pcg32::draw_uniform(state);
                    state = d_xi3.state;
                    let xi3 = d_xi3.xi;
                    let mut mu_lab = 0.0_f64;
                    // `skip_lab_rotation` is set to 1 by the elastic
                    // branch when free-gas kinematics produce the
                    // new direction directly (vector-based CM
                    // transformation can't be reduced to a
                    // (mu_lab, phi) rotation around the incoming
                    // direction). Inelastic / fission / absorption
                    // continue to use the (mu_lab, phi) rotation.
                    let mut skip_lab_rotation = 0u32;
                    if xi2 < p_elastic {
                        // Elastic: CM cosine sampled from the tabulated
                        // angular distribution when present, else
                        // isotropic. Two extra RNG draws (`xi_eb_e` for
                        // the incident-energy bracket pick, `xi_mu_e`
                        // for the CDF inversion) advance the state ONLY
                        // in this branch -- the inelastic / fission /
                        // absorption branches keep their own draw
                        // schedule, so RNG-stream parity with their
                        // pre-fix flow is preserved.
                        //
                        // ENDF MT 2 elastic at MeV is heavily forward-
                        // peaked for heavy targets (Pb, Ac, U, …);
                        // the pre-fix isotropic fallback gave too many
                        // particles a back-scatter and inflated their
                        // residence time in the cell, biasing flux /
                        // (n,γ) tallies upward (~10% on the actinide
                        // verification spheres). Tabulated CDF
                        // inversion brings GPU within statistical
                        // noise of CPU.
                        // Index by the selected nuclide's global `slab`
                        // (#74 Stage 2a): the elastic angular table is now
                        // per-(material, nuclide), not material-blended. A
                        // single-nuclide material has `slab == nuc_off`, the
                        // same row as the pre-Stage-2a `mat_idx` layout.
                        let n_ae_e = elastic_angle_n_energies[slab as usize];
                        let mut mu_cm = 1.0 - 2.0 * xi3;
                        if n_ae_e > 0u32 {
                            // Tight CSR layout (issue #104): the slab's
                            // incident-energy rows start at this global base
                            // in the variable-length elastic arrays; no fixed
                            // per-axis stride.
                            let eg_off_e = elastic_angle_ae_offset[slab as usize];
                            let mut i_ae_e = 0u32;
                            let e_first_e = elastic_angle_energy_grid[eg_off_e as usize];
                            let e_last_e =
                                elastic_angle_energy_grid[(eg_off_e + n_ae_e - 1u32) as usize];
                            let mut r_ae_e = 0.0_f64;
                            if energy >= e_last_e {
                                if n_ae_e > 1u32 {
                                    i_ae_e = n_ae_e - 2u32;
                                }
                                r_ae_e = 1.0;
                            } else if energy > e_first_e {
                                let mut k = 0u32;
                                while k + 1u32 < n_ae_e {
                                    let e_k = elastic_angle_energy_grid[(eg_off_e + k) as usize];
                                    let e_k1 =
                                        elastic_angle_energy_grid[(eg_off_e + k + 1u32) as usize];
                                    if energy >= e_k && energy < e_k1 {
                                        i_ae_e = k;
                                        let de = e_k1 - e_k;
                                        if de > 0.0 {
                                            r_ae_e = (energy - e_k) / de;
                                        }
                                    }
                                    k += 1u32;
                                }
                            }

                            let pick_e =
                                crate::common::sampling::energy_bracket::pick_energy_bracket(
                                    r_ae_e, i_ae_e, n_ae_e, state,
                                );
                            state = pick_e.state;
                            let bin_e = pick_e.bin;

                            // Mirror CPU's `TabulatedAngleDistribution::sample`
                            // (reaction_product.rs:128-139): histogram /
                            // LinLin-quadratic CDF inversion. The LinLin path
                            // closes the actinide MT 102 (n,γ) over-count
                            // (see `feedback_gpu_must_mirror_cpu_exactly`).
                            let n_mu_e = elastic_angle_n_mu[(eg_off_e + bin_e) as usize];
                            let interp_e = elastic_angle_interp[(eg_off_e + bin_e) as usize];
                            let mu_off_e = elastic_angle_mu_offset[(eg_off_e + bin_e) as usize];
                            let ang_e = crate::common::sampling::angle_cdf_invert::invert_angle_cdf(
                                mu_cm,
                                mu_off_e,
                                n_mu_e,
                                interp_e,
                                elastic_angle_mu,
                                elastic_angle_cdf,
                                elastic_angle_pdf,
                                state,
                            );
                            state = ang_e.state;
                            mu_cm = ang_e.mu;
                        }
                        // Free-gas thermal scattering: when the
                        // neutron's kinetic energy is below
                        // `free_gas_threshold · K_B · T` (and the
                        // target is heavier than the neutron), the
                        // target nucleus's thermal motion can no
                        // longer be ignored.
                        // Sample a target velocity from the CXS
                        // (Constant Cross Section) Maxwell
                        // distribution and run full vector-based
                        // CM-frame elastic kinematics. Above the
                        // threshold (or when `temperature_k <= 0`,
                        // the sentinel test fixtures use to disable
                        // free-gas for bit-equivalence checks) the
                        // closed-form A-mass formula is used,
                        // identical to the pre-free-gas behaviour.
                        const K_B: f64 = 8.617333e-5; // eV/K
                        const SQRT_PI: f64 = 1.7724538509055159;
                        const TWO_PI: f64 = std::f64::consts::TAU;
                        let temp_k = mat_f64_meta[(mat_meta_off + 1u32) as usize];
                        let kt = K_B * temp_k;
                        let do_free_gas = if temp_k > 0.0 && target_mass > 1.0 {
                            energy < free_gas_threshold[0] * kt
                        } else {
                            temp_k > 0.0 && target_mass <= 1.0
                        };
                        if do_free_gas {
                            // CXS rejection-sample beta_vt² and mu_t
                            // (cosine of target velocity vs incoming
                            // neutron direction).
                            let beta_vn_sq = target_mass * energy / kt;
                            let beta_vn = beta_vn_sq.sqrt();
                            let alpha_w = 1.0 / (1.0 + SQRT_PI * beta_vn * 0.5);
                            let mut beta_vt_sq = 0.0;
                            let mut mu_t = 0.0;
                            let mut accepted_t = 0u32;
                            let mut iter_t = 0u32;
                            // Rejection sampling: typical acceptance
                            // probability is high (~0.5–0.9) so 32
                            // iterations is a generous cap that
                            // virtually never fires in practice.
                            // When it does fire we fall through with
                            // the last candidate; the per-history
                            // bias is bounded by exp(-32) ≈ 1e-14.
                            while iter_t < 32u32 && accepted_t == 0u32 {
                                let d_a = crate::common::pcg32::draw_uniform(state);
                                state = d_a.state;
                                let r1 = d_a.xi;

                                let d_b = crate::common::pcg32::draw_uniform(state);
                                state = d_b.state;
                                let r2 = d_b.xi;

                                let d_c = crate::common::pcg32::draw_uniform(state);
                                state = d_c.state;
                                let r3 = d_c.xi;

                                #[allow(unused_assignments)]
                                let mut cand = 0.0;
                                if r3 < alpha_w {
                                    // p(y) = y · exp(-y); inversion
                                    // via -ln(r1·r2). Mirrors CPU's
                                    // `sample_cxs_target_velocity`.
                                    cand = -ln_f64(r1 * r2);
                                } else {
                                    // p(y) = y² · exp(-y²); needs an
                                    // extra draw for the cos arg.
                                    let d_d = crate::common::pcg32::draw_uniform(state);
                                    state = d_d.state;
                                    let r4 = d_d.xi;
                                    let cos_arg = std::f64::consts::FRAC_PI_2 * r4;
                                    let cv = cos_f64(cos_arg);
                                    cand = -ln_f64(r1) - ln_f64(r2) * cv * cv;
                                }

                                let d_e = crate::common::pcg32::draw_uniform(state);
                                state = d_e.state;
                                let r5 = d_e.xi;
                                let mu_cand = 2.0 * r5 - 1.0;

                                let beta_vt = cand.sqrt();
                                let v_rel_sq =
                                    beta_vn_sq + cand - 2.0 * beta_vn * beta_vt * mu_cand;
                                let mut v_rel = 0.0_f64;
                                if v_rel_sq > 0.0 {
                                    v_rel = v_rel_sq.sqrt();
                                }
                                let acc = v_rel / (beta_vn + beta_vt);

                                let d_f = crate::common::pcg32::draw_uniform(state);
                                state = d_f.state;
                                let r6 = d_f.xi;
                                if r6 < acc {
                                    beta_vt_sq = cand;
                                    mu_t = mu_cand;
                                    accepted_t = 1u32;
                                }
                                iter_t += 1u32;
                            }

                            // Magnitude of target velocity (in
                            // sqrt(eV) units, neutron mass = 1).
                            let vt_mag = (beta_vt_sq * kt / target_mass).sqrt();

                            // Direction of v_t: rotate u_n by
                            // (mu_t, phi_t) -- same formula as the
                            // kernel's existing `rotate_direction`
                            // step but inlined so we can plumb
                            // both v_t and the CM rotation later.
                            let d_pt = crate::common::pcg32::draw_uniform(state);
                            state = d_pt.state;
                            let xi_phi_t = d_pt.xi;
                            let phi_t = TWO_PI * xi_phi_t;
                            let sin_th_t_sq = (1.0 - mu_t * mu_t).max(0.0);
                            let sin_th_t = sin_th_t_sq.sqrt();
                            let cos_phi_t = cos_f64(phi_t);
                            let sin_phi_t = sin_f64(phi_t);
                            let b_t = (1.0 - dz * dz).max(0.0).sqrt();
                            let mut t_dx = sin_th_t * cos_phi_t;
                            let mut t_dy = sin_th_t * sin_phi_t;
                            let mut t_dz = mu_t;
                            if dz < 0.0 {
                                t_dy = -t_dy;
                                t_dz = -mu_t;
                            }
                            if b_t > 1e-10 {
                                t_dx = mu_t * dx
                                    + sin_th_t * (dx * dz * cos_phi_t - dy * sin_phi_t) / b_t;
                                t_dy = mu_t * dy
                                    + sin_th_t * (dy * dz * cos_phi_t + dx * sin_phi_t) / b_t;
                                t_dz = mu_t * dz - sin_th_t * b_t * cos_phi_t;
                            }
                            let v_tx = vt_mag * t_dx;
                            let v_ty = vt_mag * t_dy;
                            let v_tz = vt_mag * t_dz;

                            // Lab-frame neutron velocity.
                            let vel_n = energy.sqrt();
                            let v_nx = dx * vel_n;
                            let v_ny = dy * vel_n;
                            let v_nz = dz * vel_n;

                            // CM velocity v_cm = (v_n + A·v_t)/(A+1).
                            let inv_apl1 = 1.0 / (target_mass + 1.0);
                            let v_cm_x = (v_nx + target_mass * v_tx) * inv_apl1;
                            let v_cm_y = (v_ny + target_mass * v_ty) * inv_apl1;
                            let v_cm_z = (v_nz + target_mass * v_tz) * inv_apl1;

                            // Neutron in CM frame.
                            let v_ncm_x = v_nx - v_cm_x;
                            let v_ncm_y = v_ny - v_cm_y;
                            let v_ncm_z = v_nz - v_cm_z;
                            let vel_cm_sq =
                                v_ncm_x * v_ncm_x + v_ncm_y * v_ncm_y + v_ncm_z * v_ncm_z;

                            // Sample CM-frame phi.
                            let d_pc = crate::common::pcg32::draw_uniform(state);
                            state = d_pc.state;
                            let xi_phi_cm = d_pc.xi;
                            let phi_cm = TWO_PI * xi_phi_cm;

                            if vel_cm_sq < 1e-20 {
                                // Degenerate: neutron at rest in CM.
                                // Sample isotropically in lab using
                                // (mu_cm, phi_cm).
                                let mu_iso = mu_cm;
                                let sin_th = (1.0 - mu_iso * mu_iso).max(0.0).sqrt();
                                let cphi = cos_f64(phi_cm);
                                let sphi = sin_f64(phi_cm);
                                dx = sin_th * cphi;
                                dy = sin_th * sphi;
                                dz = mu_iso;
                                // Energy unchanged (small or zero
                                // since vel_cm_sq ≈ 0 means
                                // |v_n_lab| ≈ |v_cm|, which is
                                // already in v_cm_*; use it).
                                let new_e = v_cm_x * v_cm_x + v_cm_y * v_cm_y + v_cm_z * v_cm_z;
                                if new_e > 0.0 {
                                    energy = new_e;
                                }
                            } else {
                                let vel_cm = vel_cm_sq.sqrt();
                                let inv_vel_cm = 1.0 / vel_cm;
                                let u_cmx = v_ncm_x * inv_vel_cm;
                                let u_cmy = v_ncm_y * inv_vel_cm;
                                let u_cmz = v_ncm_z * inv_vel_cm;

                                // Rotate u_cm by (mu_cm, phi_cm).
                                let sin_th_cm = (1.0 - mu_cm * mu_cm).max(0.0).sqrt();
                                let cphi_cm = cos_f64(phi_cm);
                                let sphi_cm = sin_f64(phi_cm);
                                let b_cm = (1.0 - u_cmz * u_cmz).max(0.0).sqrt();
                                let mut u_new_x = sin_th_cm * cphi_cm;
                                let mut u_new_y = sin_th_cm * sphi_cm;
                                let mut u_new_z = mu_cm;
                                if u_cmz < 0.0 {
                                    u_new_y = -u_new_y;
                                    u_new_z = -mu_cm;
                                }
                                if b_cm > 1e-10 {
                                    u_new_x = mu_cm * u_cmx
                                        + sin_th_cm * (u_cmx * u_cmz * cphi_cm - u_cmy * sphi_cm)
                                            / b_cm;
                                    u_new_y = mu_cm * u_cmy
                                        + sin_th_cm * (u_cmy * u_cmz * cphi_cm + u_cmx * sphi_cm)
                                            / b_cm;
                                    u_new_z = mu_cm * u_cmz - sin_th_cm * b_cm * cphi_cm;
                                }
                                let v_ncm_new_x = vel_cm * u_new_x;
                                let v_ncm_new_y = vel_cm * u_new_y;
                                let v_ncm_new_z = vel_cm * u_new_z;
                                let v_nlab_x = v_ncm_new_x + v_cm_x;
                                let v_nlab_y = v_ncm_new_y + v_cm_y;
                                let v_nlab_z = v_ncm_new_z + v_cm_z;
                                let new_e =
                                    v_nlab_x * v_nlab_x + v_nlab_y * v_nlab_y + v_nlab_z * v_nlab_z;
                                if new_e > 0.0 {
                                    let inv_vlab = 1.0 / new_e.sqrt();
                                    dx = v_nlab_x * inv_vlab;
                                    dy = v_nlab_y * inv_vlab;
                                    dz = v_nlab_z * inv_vlab;
                                    energy = new_e;
                                }
                            }
                            // Free-gas branch already updated
                            // direction; signal the lab-frame
                            // rotation block below to skip.
                            skip_lab_rotation = 1u32;
                        } else {
                            let one_plus_a = target_mass + 1.0;
                            let denom = one_plus_a * one_plus_a;
                            let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
                            energy = energy * numer / denom;
                            mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();
                        }
                    }
                    if xi2 >= p_elastic && xi2 < p_scatter {
                        // Inelastic: sample which MT proportional to
                        // per-MT xs at the current energy, then use
                        // that MT's Q in the level-inelastic
                        // closed-form energy formula. The angular
                        // sample comes from the per-MT tabulated
                        // distribution (slice B of the inelastic
                        // physics work) -- when the table is in CM
                        // frame the closed-form energy is interpreted
                        // as the CM-frame outgoing energy and a
                        // two-body kinematics step converts to lab.
                        let d_xi_mt = crate::common::pcg32::draw_uniform(state);
                        state = d_xi_mt.state;
                        let xi_mt = d_xi_mt.xi;

                        // Walk MT slots, accumulating xs until the
                        // running fraction crosses xi_mt × σ_i. The
                        // last visited slot whose cumulative was below
                        // the target is the selected MT. `mt_count`
                        // comes from `MT_INELASTIC_COUNT`
                        // (`crate::neutron::xs`): slots 0..=40 = MT
                        // 51..=91, slots 41–42 = MT 16/17 multi-
                        // neutron-out, slots 43–47 = MT 22/28/32/33/34
                        // slice-F charged-particle-out + neutron, slots
                        // 48..=55 = MT 5/23/24/25/37/41/44/45 coverage-
                        // closure neutron-emitting channels; the
                        // launcher asserts the buffer length agrees so
                        // any mismatch fails loudly at the dispatch
                        // boundary.
                        // Per-MT inelastic buffers are keyed per-(material,
                        // nuclide) SLAB (#74 Stage 2b), not per material: the
                        // chosen nuclide's own discrete-level / continuum
                        // inelastic distributions are sampled. Single-nuclide
                        // materials have `slab == mat_idx`'s old row, so the
                        // index (and result) is byte-identical.
                        let mt_count = MT_INELASTIC_COUNT as u32;
                        let mat_mt_off = slab * mt_count;
                        // SPARSE per-MT inelastic storage (issue #212). Each (slab,
                        // MT slot) has a `permt_meta` row [value_offset, i_start,
                        // n_stored]: the slot's xs at coarse index `k` is
                        // `xs_inelastic_per_mt_sparse[value_offset + (k - i_start)]`
                        // when `i_start <= k < i_start + n_stored`, else 0.0 (the
                        // dense zeros). The per-slab permt row base mirrors the old
                        // coarse per-MT CSR base: `coarse_meta` col 2 (the
                        // material's first-slab row base = nuc_off * mt_count) plus
                        // the slab's offset within the material -- reducing to the
                        // global `slab * mt_count`, the same ordering as
                        // `q_inelastic_per_mt`. Single-material => `coarse_mt_base
                        // == 0`, `nuc_off == 0`, so byte-identical.
                        let coarse_mt_base: u32 = coarse_meta[(coarse_meta_off + 2u32) as usize];
                        let permt_row_base = coarse_mt_base + (slab - nuc_off) * mt_count;
                        // Walk against the SELECTED nuclide's inelastic xs. Reading
                        // 0 below threshold at BOTH bracket endpoints (idx_lo_c /
                        // idx_hi_c) reproduces the dense buffer exactly: at a
                        // threshold bracket where idx_lo_c is below i_start, the
                        // dense buffer stored 0 there too, so `mt_sigma` matches.
                        let target_cumulative = xi_mt * sigma_i_rx;
                        let mut cumulative = 0.0_f64;
                        let mut selected_slot = 0u32;
                        let mut found = 0u32;
                        let mut slot = 0u32;
                        while slot < mt_count {
                            let meta_off = (permt_row_base + slot) * 3u32; // PERMT_META_COLS
                            let value_offset = permt_meta[(meta_off + 0u32) as usize];
                            let i_start = permt_meta[(meta_off + 1u32) as usize];
                            let n_stored = permt_meta[(meta_off + 2u32) as usize];
                            let i_end = i_start + n_stored;
                            let mut mt_xs_lo = 0.0_f64;
                            if idx_lo_c >= i_start && idx_lo_c < i_end {
                                mt_xs_lo = xs_inelastic_per_mt_sparse
                                    [(value_offset + (idx_lo_c - i_start)) as usize];
                            }
                            let mut mt_xs_hi = 0.0_f64;
                            if idx_hi_c >= i_start && idx_hi_c < i_end {
                                mt_xs_hi = xs_inelastic_per_mt_sparse
                                    [(value_offset + (idx_hi_c - i_start)) as usize];
                            }
                            let mt_sigma = mt_xs_lo + (mt_xs_hi - mt_xs_lo) * frac_c;
                            cumulative += mt_sigma;
                            if found == 0u32 && cumulative >= target_cumulative {
                                selected_slot = slot;
                                found = 1u32;
                            }
                            slot += 1u32;
                        }
                        if found == 0u32 {
                            // Numerical edge case (cumulative xs < σ_i
                            // due to interpolation drift): fall back
                            // to slot 40 (MT 91, continuum). Don't
                            // pick the multi-neutron-out slots (41–42)
                            // here -- those have weight multipliers
                            // that shouldn't apply to a numerical
                            // fallback.
                            selected_slot = 40u32;
                        }

                        // Multiply the surviving particle's weight by
                        // the MT slot's per-energy yield ν(E_in).
                        // Reads `yield_per_mt[mat × MT × n_grid +
                        // slot × n_grid + i]` interpolated at the
                        // current incident energy. For most MTs
                        // ν = 1.0; MT 16 carries 2.0, MT 17 carries
                        // 3.0; some libraries store an energy-
                        // dependent yield curve which the extractor
                        // pre-tabulated on the master grid.
                        // Selected slot's yield from the SAME `permt_meta` row as
                        // its xs. Outside the stored range the yield is 1.0 (the
                        // dense default, since the dense `yield_per_mt` held 1.0
                        // wherever xs was 0), so a threshold bracket with one
                        // endpoint below `i_start` interpolates bit-identically.
                        let sel_meta_off = (permt_row_base + selected_slot) * 3u32;
                        let sel_value_offset = permt_meta[(sel_meta_off + 0u32) as usize];
                        let sel_i_start = permt_meta[(sel_meta_off + 1u32) as usize];
                        let sel_i_end = sel_i_start + permt_meta[(sel_meta_off + 2u32) as usize];
                        let mut yield_lo = 1.0_f64;
                        if idx_lo_c >= sel_i_start && idx_lo_c < sel_i_end {
                            yield_lo = yield_per_mt_sparse
                                [(sel_value_offset + (idx_lo_c - sel_i_start)) as usize];
                        }
                        let mut yield_hi = 1.0_f64;
                        if idx_hi_c >= sel_i_start && idx_hi_c < sel_i_end {
                            yield_hi = yield_per_mt_sparse
                                [(sel_value_offset + (idx_hi_c - sel_i_start)) as usize];
                        }
                        let slot_yield = yield_lo + (yield_hi - yield_lo) * frac_c;
                        // Analog (n,xn) multiplicity (issue #274): an integral
                        // yield >= 2 keeps the walk's weight UNCHANGED; the
                        // kinematics loop below samples `yield - 1` extra
                        // independent secondaries from this same slot and
                        // queues them for in-thread transport. The CPU banks
                        // real secondaries the same way, so the per-history
                        // score distribution (and thus std_dev) matches;
                        // weight multiplication gave the correct mean but
                        // Var(2t) = 4 Var(t) instead of ~2 Var(t). Fractional
                        // yields keep the legacy weight multiplication (the
                        // CPU convention), as does a yield of exactly 1
                        // (multiply by 1.0, bit-identical to the old path).
                        let mut n_out = 1u32;
                        let yield_round = (slot_yield + 0.5) as u32;
                        let mut ydiff = slot_yield - (yield_round as f64);
                        if ydiff < 0.0 {
                            ydiff = -ydiff;
                        }
                        if yield_round >= 2u32 && ydiff < 1e-10 {
                            n_out = yield_round;
                        } else {
                            weight = weight * slot_yield;
                            // Hard kill if the compounded weight crosses
                            // the cap -- see `FISSION_WEIGHT_CAP` doc for
                            // why this matters (i64 atomic-add overflow
                            // in the fixed-point tally accumulator).
                            if weight > 1000.0_f64 {
                                alive = 0u32;
                            }
                        }

                        let q = q_inelastic_per_mt[(mat_mt_off + selected_slot) as usize];
                        let abs_q = if q < 0.0 { -q } else { q };
                        let threshold = (target_mass + 1.0) / target_mass * abs_q;
                        let mass_ratio = (target_mass / (target_mass + 1.0))
                            * (target_mass / (target_mass + 1.0));
                        let e_diff = energy - threshold;
                        let e_cm_closed = mass_ratio * e_diff;

                        // Sample the outgoing (mu, E) kinematics `n_out` times
                        // from this slot's distributions: iteration 0 is the
                        // continuing walk (bit-identical to the pre-#274 body
                        // when n_out == 1); iterations 1.. are the extra
                        // analog (n,xn) secondaries, each rotated around the
                        // INCIDENT direction and pushed to the in-thread
                        // stack (or, past its depth, to the device bank).
                        let e_in_inel = energy;
                        let mut k_out = 0u32;
                        while k_out < n_out {
                            let mut e_cm_k = e_cm_closed;
                            let mut xi3_k = xi3;
                            if k_out > 0u32 {
                                // Fresh isotropic-fallback uniform per extra
                                // secondary (the walk reuses the pre-drawn
                                // xi3, keeping its stream unchanged).
                                let d_x3 = crate::common::pcg32::draw_uniform(state);
                                state = d_x3.state;
                                xi3_k = d_x3.xi;
                            }
                            let mut out_e = e_in_inel;
                            let mut out_mu = 0.0_f64;
                            let mut out_alive = 1u32;

                            // Sample mu from the MT slot's tabulated
                            // angular distribution. `xi3` (already drawn
                            // above) is reserved as the fallback for
                            // isotropic when no data is present; new
                            // streams `xi_eb` / `xi_mu` advance the PCG
                            // state for the energy-bracket stochastic
                            // bin pick and the CDF inversion respectively.
                            // The tight CSR `angle_*_offset` bases index the
                            // back-to-back ae-row / mu-point arrays; the
                            // launcher asserts the per-row / per-point buffer
                            // lengths agree so any mismatch fails loudly at
                            // the dispatch boundary.
                            // Slab-keyed slot index (#74 Stage 2b): the per-MT
                            // angle / eout / corr / km / evap / maxwell / watt /
                            // nbps pools are now per-(material, nuclide). Single-
                            // nuclide => `slab * mt_count` equals the old
                            // `mat_idx * mt_count` row, byte-identical.
                            let mat_slot = slab * mt_count + selected_slot;
                            let n_ae = angle_n_energies[mat_slot as usize];
                            let mut mu_sampled = 1.0 - 2.0 * xi3_k;
                            if n_ae > 0u32 {
                                // Bracket the incident energy on the slot's tabulated
                                // grid. Tight CSR layout (issue #104): the slot's rows
                                // start at this global base; no fixed per-axis stride.
                                let eg_off = angle_ae_offset[mat_slot as usize];
                                let mut i_ae = 0u32;
                                let e_first = angle_energy_grid[eg_off as usize];
                                let e_last = angle_energy_grid[(eg_off + n_ae - 1u32) as usize];
                                let mut r_ae = 0.0_f64;
                                if e_in_inel >= e_last {
                                    if n_ae > 1u32 {
                                        i_ae = n_ae - 2u32;
                                    }
                                    r_ae = 1.0;
                                } else if e_in_inel > e_first {
                                    let mut k = 0u32;
                                    while k + 1u32 < n_ae {
                                        let e_k = angle_energy_grid[(eg_off + k) as usize];
                                        let e_k1 = angle_energy_grid[(eg_off + k + 1u32) as usize];
                                        if e_in_inel >= e_k && e_in_inel < e_k1 {
                                            i_ae = k;
                                            let de = e_k1 - e_k;
                                            if de > 0.0 {
                                                r_ae = (e_in_inel - e_k) / de;
                                            }
                                        }
                                        k += 1u32;
                                    }
                                }

                                // Stochastic interpolation: pick the lower
                                // or upper bracket weighted by `r_ae`.
                                let pick =
                                    crate::common::sampling::energy_bracket::pick_energy_bracket(
                                        r_ae, i_ae, n_ae, state,
                                    );
                                state = pick.state;
                                let bin = pick.bin;

                                // CDF inversion within the picked bracket. Mirrors
                                // CPU's `TabulatedAngleDistribution::sample`
                                // (reaction_product.rs:128-139), same shape as the
                                // elastic LinLin form (PR #116); applies to every
                                // MT in MT_SLOTS (discrete-level and continuum
                                // inelastic, multi-neutron-out, charged-particle).
                                let n_mu = angle_n_mu[(eg_off + bin) as usize];
                                let interp_kind = angle_interp[(eg_off + bin) as usize];
                                let mu_off = angle_mu_offset[(eg_off + bin) as usize];
                                let ang =
                                    crate::common::sampling::angle_cdf_invert::invert_angle_cdf(
                                        mu_sampled,
                                        mu_off,
                                        n_mu,
                                        interp_kind,
                                        angle_mu,
                                        angle_cdf,
                                        angle_pdf,
                                        state,
                                    );
                                state = ang.state;
                                mu_sampled = ang.mu;
                            }

                            // Slice C / D: when the slot has tabulated
                            // outgoing-energy data, override the closed-
                            // form `e_cm` (and `mu_sampled` for the
                            // correlated path) with sampled values.
                            // `eout_kind == 1` (CONTINUOUS_TABULAR) and
                            // `eout_kind == 2` (CORRELATED) follow the
                            // same bracket / CDF inversion pattern but
                            // read different buffers. Slice D adds the
                            // CPU-parity bracket-bound stretch
                            // interpolation that maps the sampled E_out
                            // into the incident-energy-interpolated
                            // bounds, and the correlated path also
                            // samples mu from a per-(E_in, E_out) angular
                            // sub-table.
                            // Packed `mt_slot_u32_meta` columns:
                            //   0 = COL_EOUT_KIND
                            //   1 = COL_EOUT_N_ENERGIES
                            //   2 = COL_CORR_N_ENERGIES
                            //   3 = COL_SCATTER_IN_CM
                            //   4 = COL_KM_N_ENERGIES
                            //   5 = COL_EVAP_N_ENERGIES
                            //   6 = COL_NBPS_N_BODIES
                            //   7 = COL_MAXWELL_N_ENERGIES
                            //   8 = COL_WATT_N_ENERGIES
                            //   9 = COL_EVAP_N_COMPONENTS
                            //  10 = COL_CORR_N_COMPONENTS
                            // Stride 11u32 matches MT_SLOT_U32_COLS; the
                            // launcher's pack uses the same stride, any
                            // mismatch silently corrupts dispatch (the
                            // KM regression test catches it).
                            let meta_off = mat_slot * 11u32;
                            let kind = mt_slot_u32_meta[(meta_off + 0u32) as usize];
                            let n_eout = mt_slot_u32_meta[(meta_off + 1u32) as usize];
                            if kind == 1u32 && n_eout > 0u32 {
                                // Tight CSR (issue #104): the slot's ae-rows start at
                                // this global base; the shared sampler reads each
                                // row's (x, cdf, p) via `eout_x_offset`. No stride.
                                let eg_off_e = eout_ae_offset[mat_slot as usize];
                                let hist_outer = eout_histogram_interp[mat_slot as usize];
                                let eout_res = crate::common::sampling::eout_continuous_tabular::sample_continuous_tabular_eout(
                                e_cm_k, e_in_inel, eg_off_e, n_eout, hist_outer, eout_x_offset,
                                eout_energy_grid, eout_n_x, eout_x, eout_cdf, eout_p,
                                eout_interp, eout_n_discrete, state,
                            );
                                e_cm_k = eout_res.e_cm;
                                state = eout_res.state;
                            } else if kind == 2u32 && n_eout == 0u32 {
                                // CORRELATED slot. The eout_n_energies
                                // here is 0 by construction (slice C
                                // marker only); the actual data lives
                                // in the corr_* buffers and is keyed by
                                // COL_CORR_N_ENERGIES.
                                let n_corr_total = mt_slot_u32_meta[(meta_off + 2u32) as usize];
                                if n_corr_total > 0u32 {
                                    // Multi-component mixture (issue #111): the
                                    // neutron product carried several equally-
                                    // weighted correlated laws (F19 MT16 n,2n, two
                                    // at 0.5/0.5). Pick one uniformly per collision
                                    // -- mirrors the CPU
                                    // `sample_distribution_index` for equal
                                    // applicability. One uniform only when >= 2
                                    // components, so single-component slots stay
                                    // bit-identical. Component `comp` occupies the
                                    // `n_corr` rows at `corr_ae_offset + comp *
                                    // n_corr`.
                                    let n_comp = mt_slot_u32_meta[(meta_off + 10u32) as usize];
                                    let mut comp = 0u32;
                                    let mut n_corr = n_corr_total;
                                    if n_comp >= 2u32 {
                                        n_corr = n_corr_total / n_comp;
                                        let d_c = crate::common::pcg32::draw_uniform(state);
                                        state = d_c.state;
                                        comp = (d_c.xi * n_comp as f64) as u32;
                                        if comp >= n_comp {
                                            comp = n_comp - 1u32;
                                        }
                                    }
                                    // Tight CSR (issue #104): the chosen component's
                                    // ae-rows start at this global base; the shared
                                    // sampler reads each ae-row's (x, cdf, p) via
                                    // `corr_x_offset`, and the mu sub-table via
                                    // `corr_mu_offset`. No MAX_CORR_* stride.
                                    let eg_off_c =
                                        corr_ae_offset[mat_slot as usize] + comp * n_corr;
                                    // Sample E_out: bracket E_in, pick a bracket,
                                    // invert the slice's outgoing-energy CDF, then
                                    // stretch into the interpolated bounds. Returns
                                    // the matching angular sub-table offset
                                    // (`mu_idx_off = x_off + j`) and a `valid` flag
                                    // (set iff the chosen slice had >= 2 points, so
                                    // both `e_cm` and `mu_sampled` are updated only
                                    // then -- mirroring the old `if n_x >= 2`).
                                    let corr_res =
                                    crate::common::sampling::eout_correlated::sample_correlated_eout(
                                        e_in_inel,
                                        eg_off_c,
                                        n_corr,
                                        corr_x_offset,
                                        corr_energy_grid,
                                        corr_n_x,
                                        corr_x,
                                        corr_cdf,
                                        corr_p,
                                        corr_interp,
                                        corr_n_discrete,
                                        state,
                                    );
                                    state = corr_res.state;
                                    if corr_res.valid == 1u32 {
                                        e_cm_k = corr_res.e_out;

                                        // Sample mu from the angular sub-table at
                                        // (bin_e, j). Replaces the slice-B per-MT
                                        // angular sample for correlated slots.
                                        let mu_idx_off = corr_res.mu_idx_off;
                                        let n_mu = corr_n_mu[mu_idx_off as usize];
                                        let interp_kind = corr_mu_interp[mu_idx_off as usize];
                                        let mu_off = corr_mu_offset[mu_idx_off as usize];
                                        let ang_c =
                                        crate::common::sampling::angle_cdf_invert::invert_angle_cdf(
                                            mu_sampled,
                                            mu_off,
                                            n_mu,
                                            interp_kind,
                                            corr_mu,
                                            corr_mu_cdf,
                                            corr_mu_pdf,
                                            state,
                                        );
                                        state = ang_c.state;
                                        mu_sampled = ang_c.mu;
                                    }
                                }
                            } else if kind == 3u32 {
                                // KALBACH-MANN slot. Sample (E_out_cm,
                                // mu_cm) from the per-(E_in, E_out) PDF +
                                // (r, a) parameters via the shared helper
                                // (mirrors CPU's `KalbachMann::sample`). RNG
                                // draw order, preserved by the helper:
                                //   incident-energy bracket pick,
                                //   E_out CDF inversion,
                                //   compound-vs-precompound pick,
                                //   mu sample.
                                let n_kae = mt_slot_u32_meta[(meta_off + 4u32) as usize];
                                if n_kae > 0u32 {
                                    // Tight CSR (issue #104): the slot's ae-rows
                                    // start at this global base; the shared sampler
                                    // reads each row's (x, p, c, r, a) via
                                    // `km_x_offset`. No stride.
                                    let eg_off_k = km_ae_offset[mat_slot as usize];
                                    let km =
                                        crate::common::sampling::kalbach_mann::sample_kalbach_mann(
                                            e_in_inel,
                                            eg_off_k,
                                            n_kae,
                                            km_x_offset,
                                            km_energy_grid,
                                            km_n_x,
                                            km_interp,
                                            km_n_discrete,
                                            km_x,
                                            km_p,
                                            km_c,
                                            km_r,
                                            km_a,
                                            state,
                                        );
                                    state = km.state;
                                    if km.e_valid == 1u32 {
                                        e_cm_k = km.e_out;
                                    }
                                    if km.mu_valid == 1u32 {
                                        mu_sampled = km.mu;
                                    }
                                }
                            } else if kind == 4u32 {
                                // EVAPORATION slot. Sample E_out from
                                //   p(E) ∝ E · exp(-E/θ),  0 < E < E_in - u
                                // via the standard rejection algorithm:
                                //   y = (E_in - u) / θ
                                //   v = 1 - exp(-y)
                                //   loop:
                                //     E = -ln((1 - v·xi1)(1 - v·xi2))
                                //     accept if E ≤ y
                                //   E_out = E · θ
                                // mu_sampled stays at the slice-B
                                // angular value (or isotropic fallback)
                                // already computed above.
                                let n_tae = mt_slot_u32_meta[(meta_off + 5u32) as usize];
                                if n_tae > 0u32 {
                                    // Linearly interpolate θ(E_in) on
                                    // the slot's tabulated grid.
                                    // Component selection: a neutron product may
                                    // carry several equally-weighted Evaporation
                                    // laws (the (n,xn) channels of a few endf-b8.1
                                    // nuclides). The CPU draws one component per
                                    // collision (`sample_distribution_index`); we
                                    // mirror that with ONE uniform when there are
                                    // >= 2 components, leaving single-component
                                    // slots bit-identical (no extra draw). θ is
                                    // stored component-major (tight, issue #104):
                                    // component `comp` occupies its own `n_tae`-point
                                    // row within this slot's `evap_theta_offset`
                                    // region. The shared incident-energy grid and
                                    // `u` rows start at `evap_ae_offset[mat_slot]`.
                                    let n_comp = mt_slot_u32_meta[(meta_off + 9u32) as usize];
                                    let mut comp = 0u32;
                                    if n_comp >= 2u32 {
                                        let d_c = crate::common::pcg32::draw_uniform(state);
                                        state = d_c.state;
                                        comp = (d_c.xi * n_comp as f64) as u32;
                                        if comp >= n_comp {
                                            comp = n_comp - 1u32;
                                        }
                                    }
                                    // Tight CSR (issue #104): the slot's E_in rows
                                    // start at `evap_ae_offset[mat_slot]`; the
                                    // component-major theta starts at
                                    // `evap_theta_offset[mat_slot]`, with component
                                    // `comp` occupying its own `n_tae`-point row.
                                    let eg_off_e = evap_ae_offset[mat_slot as usize];
                                    let theta_off =
                                        evap_theta_offset[mat_slot as usize] + comp * n_tae;
                                    let e_first = evap_energy_grid[eg_off_e as usize];
                                    let e_last =
                                        evap_energy_grid[(eg_off_e + n_tae - 1u32) as usize];
                                    let mut theta_val =
                                        evap_theta[(theta_off + n_tae - 1u32) as usize];
                                    // `u` is selected at the nearest-lower incident-
                                    // energy grid point (NOT interpolated): the
                                    // multi-law applicability switch is a step, so
                                    // interpolating `u` across it would blend two
                                    // distinct restriction energies. `u_idx` tracks
                                    // that bracket index alongside the theta interp.
                                    let mut u_idx = n_tae - 1u32;
                                    if e_in_inel <= e_first {
                                        theta_val = evap_theta[theta_off as usize];
                                        u_idx = 0u32;
                                    } else if e_in_inel < e_last {
                                        let mut k = 0u32;
                                        while k + 1u32 < n_tae {
                                            let e_k = evap_energy_grid[(eg_off_e + k) as usize];
                                            let e_k1 =
                                                evap_energy_grid[(eg_off_e + k + 1u32) as usize];
                                            if e_in_inel >= e_k && e_in_inel < e_k1 {
                                                let de = e_k1 - e_k;
                                                let mut f = 0.0_f64;
                                                if de > 0.0 {
                                                    f = (e_in_inel - e_k) / de;
                                                }
                                                let t_k = evap_theta[(theta_off + k) as usize];
                                                let t_k1 =
                                                    evap_theta[(theta_off + k + 1u32) as usize];
                                                theta_val = t_k + f * (t_k1 - t_k);
                                                u_idx = k;
                                            }
                                            k += 1u32;
                                        }
                                    }

                                    let u_ev = evap_u[(eg_off_e + u_idx) as usize];
                                    if theta_val > 0.0 && e_in_inel > u_ev {
                                        let y = (e_in_inel - u_ev) / theta_val;
                                        let v_e = 1.0 - exp_f64(-y);
                                        let mut accepted = 0u32;
                                        let mut sampled_e = 0.0;
                                        let mut iter = 0u32;
                                        // 32-iteration rejection cap --
                                        // typical acceptance is high; the
                                        // tail risk is bounded. The 2-uniform
                                        // draw + ln/ln candidate is the shared
                                        // `evaporation_rejection_draw` helper.
                                        while iter < 32u32 && accepted == 0u32 {
                                            let rj = crate::common::sampling::eout_rejection::evaporation_rejection_draw(
                                            v_e, y, theta_val, state,
                                        );
                                            state = rj.state;
                                            if rj.accepted == 1u32 {
                                                sampled_e = rj.e_out;
                                                accepted = 1u32;
                                            }
                                            iter += 1u32;
                                        }
                                        if accepted == 1u32 && sampled_e > 0.0 {
                                            e_cm_k = sampled_e;
                                        }
                                    }
                                }
                            } else if kind == 5u32 {
                                // NBODY PHASE SPACE slot. Sample E_out
                                // via the standard Maxwellian-product
                                // algorithm for n_bodies ∈ {3, 4, 5}.
                                // Mu is isotropic in CM (overrides any
                                // slice-B sample). E_max combines the
                                // CM-frame source energy with Q.
                                let n_bodies = mt_slot_u32_meta[(meta_off + 6u32) as usize];
                                let ap = mt_slot_f64_meta[(mat_slot * 4u32 + 3u32) as usize];
                                // `q` (the per-MT inelastic Q-value) is already in
                                // scope from the closed-form `e_cm` computation above.
                                let nb = crate::common::sampling::nbody_phase_space::sample_nbody_phase_space(
                                e_in_inel,
                                target_mass,
                                q,
                                n_bodies,
                                ap,
                                state,
                            );
                                state = nb.state;
                                if nb.e_valid == 1u32 {
                                    e_cm_k = nb.e_out;
                                }
                                if nb.mu_valid == 1u32 {
                                    // Mu is isotropic for NBPS: override
                                    // slice-B sample.
                                    mu_sampled = nb.mu;
                                }
                            } else if kind == 6u32 {
                                // MAXWELL slot. Sample E_out from
                                //   p(E) ∝ sqrt(E) · exp(-E/θ),  0 < E < E_in - u
                                // via the standard 3-uniform Maxwell
                                // rejection algorithm (ENDF File 5,
                                // Law 7), bit-matching `sample_maxwell_spectrum`
                                // in `yamc-nuclide::sampling`:
                                //   loop:
                                //     r1, r2, r3 ← RNG
                                //     c = cos(π/2 · r3)
                                //     E = -θ · (ln r1 + ln r2 · c²)
                                //     accept if E ≤ E_in - u
                                // mu_sampled stays at the slice-B
                                // angular value (or isotropic fallback)
                                // already computed above.
                                let n_mae = mt_slot_u32_meta[(meta_off + 7u32) as usize];
                                if n_mae > 0u32 {
                                    // Linearly interpolate θ(E_in) on
                                    // the slot's tabulated grid. Mirrors
                                    // the Evaporation interpolation
                                    // exactly -- same grid shape.
                                    // Tight CSR (issue #104): the slot's E_in rows
                                    // start at `maxwell_ae_offset[mat_slot]`.
                                    let mg_off_e = maxwell_ae_offset[mat_slot as usize];
                                    let me_first = maxwell_energy_grid[mg_off_e as usize];
                                    let me_last =
                                        maxwell_energy_grid[(mg_off_e + n_mae - 1u32) as usize];
                                    let mut mtheta_val =
                                        maxwell_theta[(mg_off_e + n_mae - 1u32) as usize];
                                    if e_in_inel <= me_first {
                                        mtheta_val = maxwell_theta[mg_off_e as usize];
                                    } else if e_in_inel < me_last {
                                        let mut k = 0u32;
                                        while k + 1u32 < n_mae {
                                            let e_k = maxwell_energy_grid[(mg_off_e + k) as usize];
                                            let e_k1 =
                                                maxwell_energy_grid[(mg_off_e + k + 1u32) as usize];
                                            if e_in_inel >= e_k && e_in_inel < e_k1 {
                                                let de = e_k1 - e_k;
                                                let mut f = 0.0_f64;
                                                if de > 0.0 {
                                                    f = (e_in_inel - e_k) / de;
                                                }
                                                let t_k = maxwell_theta[(mg_off_e + k) as usize];
                                                let t_k1 =
                                                    maxwell_theta[(mg_off_e + k + 1u32) as usize];
                                                mtheta_val = t_k + f * (t_k1 - t_k);
                                            }
                                            k += 1u32;
                                        }
                                    }

                                    let u_mx = mt_slot_f64_meta[(mat_slot * 4u32 + 1u32) as usize];
                                    if mtheta_val > 0.0 && e_in_inel > u_mx {
                                        let cap_e = e_in_inel - u_mx;
                                        let mut accepted = 0u32;
                                        let mut sampled_e = 0.0;
                                        let mut iter = 0u32;
                                        // 32-iteration rejection cap --
                                        // typical acceptance is high, the
                                        // tail risk is bounded. The 3-uniform
                                        // draw + cos/ln candidate is the shared
                                        // `maxwell_rejection_draw` helper.
                                        while iter < 32u32 && accepted == 0u32 {
                                            let rj = crate::common::sampling::eout_rejection::maxwell_rejection_draw(
                                            mtheta_val, cap_e, state,
                                        );
                                            state = rj.state;
                                            if rj.accepted == 1u32 {
                                                sampled_e = rj.e_out;
                                                accepted = 1u32;
                                            }
                                            iter += 1u32;
                                        }
                                        if accepted == 1u32 && sampled_e > 0.0 {
                                            e_cm_k = sampled_e;
                                        }
                                    }
                                }
                            } else if kind == 7u32 {
                                // WATT-INELASTIC slot. Sample E_out from
                                //   p(E) ∝ exp(-E/a) · sinh(sqrt(b·E)),
                                //   0 < E < E_in - u
                                // via the Watt-Maxwell relationship
                                // (ENDF File 5, Law 11):
                                //   loop:
                                //     r1, r2, r3, r4 ← RNG
                                //     w = -a · (ln r1 + ln r2 · cos²(π/2 · r3))
                                //     u_ξ = 2 r4 - 1                  (uniform [-1, 1])
                                //     E = w + a²b/4 + u_ξ · sqrt(a²b · w)
                                //     accept if E ≤ E_in - u
                                // Bit-matches `sample_watt_spectrum_params`
                                // in yamc-nuclide::sampling. mu_sampled
                                // stays at the slice-B angular value.
                                let n_wae = mt_slot_u32_meta[(meta_off + 8u32) as usize];
                                if n_wae > 0u32 {
                                    // Tight CSR (issue #104): the slot's E_in rows
                                    // start at `watt_ae_offset[mat_slot]`; `watt_ab`
                                    // interleaves a/b at stride 2 over that base.
                                    let wg_off = watt_ae_offset[mat_slot as usize];
                                    let we_first = watt_energy_grid[wg_off as usize];
                                    let we_last =
                                        watt_energy_grid[(wg_off + n_wae - 1u32) as usize];
                                    // `watt_ab` interleaves a and b at
                                    // stride 2: index `(wg_off + i) * 2`
                                    // is `a`, `+ 1` is `b`.
                                    let last_off = (wg_off + n_wae - 1u32) * 2u32;
                                    let mut watt_a_val = watt_ab[last_off as usize];
                                    let mut watt_b_val = watt_ab[(last_off + 1u32) as usize];
                                    if e_in_inel <= we_first {
                                        let off0 = wg_off * 2u32;
                                        watt_a_val = watt_ab[off0 as usize];
                                        watt_b_val = watt_ab[(off0 + 1u32) as usize];
                                    } else if e_in_inel < we_last {
                                        let mut k = 0u32;
                                        while k + 1u32 < n_wae {
                                            let e_k = watt_energy_grid[(wg_off + k) as usize];
                                            let e_k1 =
                                                watt_energy_grid[(wg_off + k + 1u32) as usize];
                                            if e_in_inel >= e_k && e_in_inel < e_k1 {
                                                let de = e_k1 - e_k;
                                                let mut f = 0.0_f64;
                                                if de > 0.0 {
                                                    f = (e_in_inel - e_k) / de;
                                                }
                                                let off_k = (wg_off + k) * 2u32;
                                                let off_k1 = (wg_off + k + 1u32) * 2u32;
                                                let a_k = watt_ab[off_k as usize];
                                                let a_k1 = watt_ab[off_k1 as usize];
                                                let b_k = watt_ab[(off_k + 1u32) as usize];
                                                let b_k1 = watt_ab[(off_k1 + 1u32) as usize];
                                                watt_a_val = a_k + f * (a_k1 - a_k);
                                                watt_b_val = b_k + f * (b_k1 - b_k);
                                            }
                                            k += 1u32;
                                        }
                                    }

                                    let u_w = mt_slot_f64_meta[(mat_slot * 4u32 + 2u32) as usize];
                                    if watt_a_val > 0.0 && e_in_inel > u_w {
                                        let cap_e = e_in_inel - u_w;
                                        let a2b = watt_a_val * watt_a_val * watt_b_val;
                                        let mut accepted = 0u32;
                                        let mut sampled_e = 0.0;
                                        let mut iter = 0u32;
                                        // The 4-uniform Maxwell-w + Watt-correction
                                        // candidate is the shared
                                        // `watt_rejection_draw` helper.
                                        while iter < 32u32 && accepted == 0u32 {
                                            let rj = crate::common::sampling::eout_rejection::watt_rejection_draw(
                                            watt_a_val, a2b, cap_e, state,
                                        );
                                            state = rj.state;
                                            if rj.accepted == 1u32 {
                                                sampled_e = rj.e_out;
                                                accepted = 1u32;
                                            }
                                            iter += 1u32;
                                        }
                                        if accepted == 1u32 && sampled_e > 0.0 {
                                            e_cm_k = sampled_e;
                                        }
                                    }
                                }
                            } else if kind == 8u32 && n_eout > 0u32 {
                                // TABULATED equiprobable slot. Flat
                                // bracket-find without early-out
                                // branching -- empirically the early-
                                // exit + while-loop pattern other
                                // branches use pushes cubecl-spirv past
                                // a silent threshold on RADV here (the
                                // `gpu_free_gas_thermal` regression test
                                // catches it via elastic descriptor
                                // aliasing). Flat full-range scan keeps
                                // it within budget.
                                // Tight CSR (issue #104): the slot's ae-rows start at
                                // this global base and run for `n_eout` rows; the
                                // scan is bounded by `n_eout` (not a fixed
                                // per-axis stride) so it never reads into a
                                // neighbouring slot's concatenated rows.
                                let eg_off_t = eout_ae_offset[mat_slot as usize];
                                let mut i_t = 0u32;
                                let mut k = 0u32;
                                while k + 1u32 < n_eout {
                                    let e_k = eout_energy_grid[(eg_off_t + k) as usize];
                                    let e_k1 = eout_energy_grid[(eg_off_t + k + 1u32) as usize];
                                    if e_in_inel >= e_k && e_in_inel < e_k1 {
                                        i_t = k;
                                    }
                                    k += 1u32;
                                }
                                // Above-grid clamp: the scan leaves `i_t = 0` when
                                // `energy` is at/above the last incident-energy
                                // point, which would sample the wrong (lowest-E)
                                // row. The flat twin `sample_tabulated_equiprobable`
                                // clamps to `n_e - 1` there, so mirror it (issue
                                // #107). `n_eout > 0` per the branch guard.
                                let e_last_t =
                                    eout_energy_grid[(eg_off_t + n_eout - 1u32) as usize];
                                if e_in_inel >= e_last_t {
                                    i_t = n_eout - 1u32;
                                }
                                let d_tx = crate::common::pcg32::draw_uniform(state);
                                state = d_tx.state;
                                let xi_tx = d_tx.xi;
                                let n_x_t = eout_n_x[(eg_off_t + i_t) as usize];
                                if n_x_t > 0u32 {
                                    let mut bin = (xi_tx * (n_x_t as f64)) as u32;
                                    if bin >= n_x_t {
                                        bin = n_x_t - 1u32;
                                    }
                                    let off_t = eout_x_offset[(eg_off_t + i_t) as usize] + bin;
                                    let e_sampled = eout_x[off_t as usize];
                                    if e_sampled > 0.0 {
                                        e_cm_k = e_sampled;
                                    }
                                }
                            }

                            // Energy update. Two paths depending on the
                            // table's frame:
                            //   scatter_in_cm == 1:
                            //     E_cm  comes from the eout sampler above
                            //          (or the closed-form fallback)
                            //     E_lab = E_cm + (E_in + 2·μ·(A+1)·sqrt(E_in·E_cm)) / (A+1)²
                            //     μ_lab = μ·sqrt(E_cm/E_lab) + 1/(A+1)·sqrt(E_in/E_lab)
                            //   scatter_in_cm == 0:
                            //     E_lab = E_cm                     (table already lab)
                            //     μ_lab = μ                        (table already lab)
                            // The threshold check terminates the particle
                            // for the same numerical-drift reason as the
                            // previous iteration of this kernel.
                            let in_cm = mt_slot_u32_meta[(meta_off + 3u32) as usize];
                            if e_cm_k <= 0.0 {
                                out_alive = 0u32;
                            }
                            if e_cm_k > 0.0 {
                                if in_cm == 1u32 {
                                    let one_plus_a = target_mass + 1.0;
                                    let denom = one_plus_a * one_plus_a;
                                    let e_lab = e_cm_k
                                        + (e_in_inel
                                            + 2.0
                                                * mu_sampled
                                                * one_plus_a
                                                * (e_in_inel * e_cm_k).sqrt())
                                            / denom;
                                    if e_lab > 0.0 {
                                        let inv_e_lab = 1.0 / e_lab;
                                        let mut mu_l = mu_sampled * (e_cm_k * inv_e_lab).sqrt()
                                            + (1.0 / one_plus_a) * (e_in_inel * inv_e_lab).sqrt();
                                        if mu_l < -1.0 {
                                            mu_l = -1.0;
                                        }
                                        if mu_l > 1.0 {
                                            mu_l = 1.0;
                                        }
                                        out_e = e_lab;
                                        out_mu = mu_l;
                                    } else {
                                        out_alive = 0u32;
                                    }
                                } else {
                                    out_e = e_cm_k;
                                    out_mu = mu_sampled;
                                }
                            }
                            if k_out == 0u32 {
                                // Commit the walk's outcome (identical to the
                                // pre-loop behaviour: kinematics failures kill
                                // the walk and leave energy/mu untouched).
                                if out_alive == 0u32 {
                                    alive = 0u32;
                                }
                                if out_alive == 1u32 {
                                    energy = out_e;
                                    mu_lab = out_mu;
                                }
                            } else if out_alive == 1u32 {
                                // Extra analog secondary: lab azimuth (one PCG
                                // draw, the same TAU * xi construction as the
                                // walk's shared rotation) around the INCIDENT
                                // direction, then queue for in-thread
                                // transport. A failed-kinematics extra
                                // (out_alive == 0) is dropped, mirroring the
                                // walk's kill on the same condition.
                                let mphi_x =
                                    crate::common::sampling::marsaglia_phi::azimuth_cos_sin_phi(
                                        state,
                                    );
                                let cos_phi_x = mphi_x.cos_phi;
                                let sin_phi_x = mphi_x.sin_phi;
                                state = mphi_x.state;
                                let sin_th_sq_x = 1.0 - out_mu * out_mu;
                                let mut sin_th_x = 0.0;
                                if sin_th_sq_x > 0.0 {
                                    sin_th_x = sin_th_sq_x.sqrt();
                                }
                                let one_minus_w_sq_x = 1.0 - dz * dz;
                                let mut xdx = sin_th_x * cos_phi_x;
                                let mut xdy = sin_th_x * sin_phi_x;
                                let mut xdz = out_mu;
                                if dz < 0.0 {
                                    xdy = -xdy;
                                    xdz = -out_mu;
                                }
                                if one_minus_w_sq_x > 1e-14 {
                                    let sin_phi_w_x = one_minus_w_sq_x.sqrt();
                                    xdx = out_mu * dx
                                        + sin_th_x * (dx * dz * cos_phi_x - dy * sin_phi_x)
                                            / sin_phi_w_x;
                                    xdy = out_mu * dy
                                        + sin_th_x * (dy * dz * cos_phi_x + dx * sin_phi_x)
                                            / sin_phi_w_x;
                                    xdz = out_mu * dz - sin_th_x * sin_phi_w_x * cos_phi_x;
                                }
                                // The secondary's own collision stream, fixed
                                // by its place in the emission tree (issue
                                // #111). Derived identically whether the
                                // secondary lands in the stack or the bank, so
                                // spilling cannot change what it samples.
                                let x_seed = crate::common::pcg32::secondary_seed(
                                    walk_seed,
                                    walk_secondaries,
                                );
                                if pend_n < PEND_SLOTS_U32 {
                                    let qi = pend_n as usize;
                                    pend_e[qi] = out_e;
                                    pend_px[qi] = px;
                                    pend_py[qi] = py;
                                    pend_pz[qi] = pz;
                                    pend_dx[qi] = xdx;
                                    pend_dy[qi] = xdy;
                                    pend_dz[qi] = xdz;
                                    pend_w[qi] = weight;
                                    pend_seed[qi] = x_seed;
                                    pend_n += 1u32;
                                } else {
                                    // Stack full: hand the secondary to the
                                    // device bank instead of folding it into
                                    // the walk weight, so the GPU transports
                                    // exactly the neutrons the CPU does. The
                                    // host drains `PTYPE_NEUTRON` records in a
                                    // later pass, carrying `my_source_idx` so
                                    // the contributions still land in the
                                    // ORIGINATING history's variance sample
                                    // (issue #233 Stage 2). Costs no PCG draw,
                                    // so the walk's stream is the same whether
                                    // or not it spilled.
                                    let xcap = (bank_f64.len() / 8) as u64;
                                    let xslot = bank_count[0].fetch_add(1u64);
                                    if xslot < xcap {
                                        let f = (xslot * 8u64) as usize;
                                        bank_f64[f] = out_e;
                                        bank_f64[f + 1] = px;
                                        bank_f64[f + 2] = py;
                                        bank_f64[f + 3] = pz;
                                        bank_f64[f + 4] = xdx;
                                        bank_f64[f + 5] = xdy;
                                        bank_f64[f + 6] = xdz;
                                        bank_f64[f + 7] = weight;
                                        let u = (xslot * 4u64) as usize;
                                        bank_u32[u] = PTYPE_NEUTRON;
                                        bank_u32[u + 1] = cell;
                                        bank_u32[u + 2] = x_seed;
                                        bank_u32[u + 3] = BANK_GEN_NXN_SPILL;
                                        bank_source_idx[xslot as usize] = my_source_idx;
                                    } else {
                                        bank_overflow[0].fetch_add(1u64);
                                    }
                                }
                                // Advance the ordinal whether the secondary
                                // went to the stack or the bank, so its key
                                // never depends on how full the stack was; the
                                // CPU bank, which has no depth limit, numbers
                                // the same secondaries the same way.
                                walk_secondaries += 1u32;
                            }
                            k_out += 1u32;
                        }
                    }
                    if xi2 >= p_scatter && xi2 < p_fission_or_scatter {
                        // Fission. The continuing walk gets ONE chi-sampled
                        // outgoing energy (via the shared `sample_fission_chi`
                        // helper, which preserves the original inline draw order
                        // bit-for-bit) and an isotropic-in-lab μ
                        // (`1 - 2·xi3` + the Marsaglia rotation that follows).
                        //
                        // Two transport modes, selected by `fission_bank_enabled`:
                        //   OFF (`0`): legacy variance-reduction terminator --
                        //     `weight *= nu_bar` then the `FISSION_WEIGHT_CAP`
                        //     kill. Byte-identical to the pre-bank kernel.
                        //   ON  (`1`): true fission-chain branching (issue #78).
                        //     Stochastically round nu_bar -> N, keep progeny 0 as
                        //     the current walk (weight UNCHANGED), and append the
                        //     other N-1 chi-sampled progeny to the device bank as
                        //     neutrons (gen+1). The host drains and transports
                        //     them in a second pass, folding into the same
                        //     tallies -- the GPU twin of CPU
                        //     `sample_fission_neutrons` + `bank_secondary`.
                        let watt_a_fis = mat_f64_meta[(mat_meta_off + 2u32) as usize];
                        let watt_b_fis = mat_f64_meta[(mat_meta_off + 3u32) as usize];
                        let e_incident_fis = energy;

                        // Continuing progeny (progeny 0): one chi draw.
                        let chi0 = sample_fission_progeny_energy(
                            e_incident_fis,
                            mat_idx,
                            beta_delayed,
                            watt_a_fis,
                            watt_b_fis,
                            fission_eout_kind_per_material,
                            fission_eout_n_energies_per_material,
                            fission_eout_ae_offset,
                            fission_eout_energy_grid_per_material,
                            fission_eout_n_x_per_material,
                            fission_eout_x_offset,
                            fission_eout_x_per_material,
                            fission_eout_cdf_per_material,
                            fission_eout_p_per_material,
                            fission_eout_interp_per_material,
                            state,
                        );
                        state = chi0.state;
                        energy = chi0.e_out;
                        // Isotropic-in-lab μ for the continuing walk (reuse the
                        // already-drawn `xi3`, exactly as before).
                        mu_lab = 1.0 - 2.0 * xi3;

                        if fission_bank_enabled[0] == 0u32 {
                            // Legacy weight-multiply terminator. RNG stream and
                            // weight evolution are byte-identical to the pre-bank
                            // kernel (the only extra work above was refactoring
                            // the chi draw into a helper -- same draws, same
                            // order).
                            weight = weight * nu_bar;
                            if weight > 1000.0_f64 {
                                alive = 0u32;
                            }
                        } else {
                            // Fission-bank branching. Stochastically round
                            // nu_bar -> N with one uniform (CPU
                            // `sample_fission_neutrons` rounding). Progeny 0 is
                            // the continuing walk just set above; bank progeny
                            // 1..N. Weight is left UNCHANGED (analog branching,
                            // not weight multiplication).
                            let d_nr = crate::common::pcg32::draw_uniform(state);
                            state = d_nr.state;
                            let n_floor = nu_bar as u32; // floor (nu_bar >= 0)
                            let frac_nr = nu_bar - (n_floor as f64);
                            let mut n_prog = n_floor;
                            if d_nr.xi < frac_nr {
                                n_prog = n_floor + 1u32;
                            }

                            // N == 0 (nu_bar rounded down to zero): the fission
                            // emits no neutrons, so the continuing walk dies --
                            // matching CPU `sample_fission_neutrons` returning an
                            // empty batch. (For fast actinides nu_bar >= ~2.4, so
                            // this is essentially never taken; it is here for
                            // correctness, not the hot path.)
                            if n_prog == 0u32 {
                                alive = 0u32;
                            }

                            // Bank progeny 1..N (comptime-bounded loop, guarded
                            // by `prog < n_prog`, mirroring the photon-emission
                            // loop). Each banked neutron gets an INDEPENDENT chi
                            // energy + isotropic direction, and carries the
                            // current weight.
                            let cap_prog = 8u32; // FISSION_BANK_PROGENY_CAP
                            let mut prog = 1u32;
                            while prog < cap_prog {
                                if prog < n_prog {
                                    // Independent chi energy.
                                    let chi = sample_fission_progeny_energy(
                                        e_incident_fis,
                                        mat_idx,
                                        beta_delayed,
                                        watt_a_fis,
                                        watt_b_fis,
                                        fission_eout_kind_per_material,
                                        fission_eout_n_energies_per_material,
                                        fission_eout_ae_offset,
                                        fission_eout_energy_grid_per_material,
                                        fission_eout_n_x_per_material,
                                        fission_eout_x_offset,
                                        fission_eout_x_per_material,
                                        fission_eout_cdf_per_material,
                                        fission_eout_p_per_material,
                                        fission_eout_interp_per_material,
                                        state,
                                    );
                                    state = chi.state;
                                    let e_bank = chi.e_out;

                                    // Isotropic-in-lab direction: uniform μ in
                                    // [-1, 1] + a `TAU * xi` azimuth (ONE draw,
                                    // like every other azimuth in this kernel
                                    // since #136 / #111), rotating the incident
                                    // direction (dx, dy, dz). Same construction
                                    // the continuing walk uses, so each banked
                                    // progeny is independently isotropic in lab.
                                    // Marsaglia rejection used to sample this
                                    // one, which spent a VARIABLE number of
                                    // draws (2 to 16) and so put the rest of the
                                    // walk on a different stream position from
                                    // the CPU's single-draw `TAU * next_xi`.
                                    let d_mu = crate::common::pcg32::draw_uniform(state);
                                    state = d_mu.state;
                                    let mu_b = 1.0 - 2.0 * d_mu.xi;
                                    let mphi_b =
                                        crate::common::sampling::marsaglia_phi::azimuth_cos_sin_phi(
                                            state,
                                        );
                                    state = mphi_b.state;
                                    let cos_phi_b = mphi_b.cos_phi;
                                    let sin_phi_b = mphi_b.sin_phi;
                                    let sin_th_sq_b = 1.0 - mu_b * mu_b;
                                    let mut sin_th_b = 0.0;
                                    if sin_th_sq_b > 0.0 {
                                        sin_th_b = sin_th_sq_b.sqrt();
                                    }
                                    let one_minus_w_sq_b = 1.0 - dz * dz;
                                    let mut bdx = sin_th_b * cos_phi_b;
                                    let mut bdy = sin_th_b * sin_phi_b;
                                    let mut bdz = mu_b;
                                    if dz < 0.0 {
                                        bdy = -bdy;
                                        bdz = -mu_b;
                                    }
                                    if one_minus_w_sq_b > 1e-14 {
                                        let sin_phi_w_b = one_minus_w_sq_b.sqrt();
                                        bdx = mu_b * dx
                                            + sin_th_b * (dx * dz * cos_phi_b - dy * sin_phi_b)
                                                / sin_phi_w_b;
                                        bdy = mu_b * dy
                                            + sin_th_b * (dy * dz * cos_phi_b + dx * sin_phi_b)
                                                / sin_phi_w_b;
                                        bdz = mu_b * dz - sin_th_b * sin_phi_w_b * cos_phi_b;
                                    }

                                    // The progeny's own collision stream, fixed
                                    // by its place in the emission tree (issue
                                    // #111 / #322), exactly as the (n,xn)
                                    // secondaries above derive theirs -- NOT a
                                    // drawn seed word, which cost the walk an
                                    // extra draw the CPU never makes. Derived
                                    // before the capacity check so the numbering
                                    // cannot depend on how full the bank was.
                                    let f_seed = crate::common::pcg32::secondary_seed(
                                        walk_seed,
                                        walk_secondaries,
                                    );

                                    // Append to the device bank (inlined atomic
                                    // fetch-add slot reservation + record write,
                                    // mirroring the photon-emission append).
                                    let fcap = (bank_f64.len() / 8) as u64;
                                    let fslot = bank_count[0].fetch_add(1u64);
                                    if fslot < fcap {
                                        let f = (fslot * 8u64) as usize;
                                        bank_f64[f] = e_bank;
                                        bank_f64[f + 1] = px;
                                        bank_f64[f + 2] = py;
                                        bank_f64[f + 3] = pz;
                                        bank_f64[f + 4] = bdx;
                                        bank_f64[f + 5] = bdy;
                                        bank_f64[f + 6] = bdz;
                                        bank_f64[f + 7] = weight;
                                        let u = (fslot * 4u64) as usize;
                                        bank_u32[u] = PTYPE_NEUTRON;
                                        bank_u32[u + 1] = cell;
                                        bank_u32[u + 2] = f_seed;
                                        // `gen` field is informational; the host
                                        // enforces the generation cap by counting
                                        // bank-drain passes (a banked neutron that
                                        // fissions again appends to the bank the
                                        // host re-drains next pass).
                                        bank_u32[u + 3] = 1u32;
                                        // Per-source variance (issue #233 Stage 2):
                                        // stamp this progeny with the SOURCE neutron
                                        // it descends from (inherited transitively
                                        // through generations), so its contributions
                                        // fold into that source's variance sample.
                                        // Written unconditionally (a size-1 dummy
                                        // when per_source_var is off, but the fission
                                        // bank is only exercised on the neutron
                                        // path); `fslot < fcap` guards the index.
                                        bank_source_idx[fslot as usize] = my_source_idx;
                                    } else {
                                        bank_overflow[0].fetch_add(1u64);
                                    }
                                    // Advance the ordinal whether the progeny
                                    // made it into the bank or overflowed, so
                                    // its key never depends on the capacity --
                                    // the same rule the (n,xn) secondaries
                                    // above follow, and the numbering the CPU
                                    // `ParticleBank` gives the same progeny.
                                    walk_secondaries += 1u32;
                                }
                                prog += 1u32;
                            }
                        }
                    }
                    // Shared angular sampling: Marsaglia rejection
                    // for (cos_phi, sin_phi) + 3D direction rotation.
                    // Same code path for elastic, inelastic, and
                    // fission; only `mu_lab` and `energy` differ
                    // (set by the per-branch block above).

                    // Lab azimuth: phi = TAU * xi (ONE PCG draw), matching the
                    // production CPU / the shared.rs twin so the issue-#40
                    // matched stream stays in lockstep past the first collision
                    // (#136 / #111). cos/sin via the polyfills; the twin inlines
                    // std libm cos/sin (agree within ulps, as ln/exp do).
                    let mphi = crate::common::sampling::marsaglia_phi::azimuth_cos_sin_phi(state);
                    let cos_phi = mphi.cos_phi;
                    let sin_phi = mphi.sin_phi;
                    state = mphi.state;

                    // 3D direction rotation by (mu_lab, phi). Skip
                    // when the free-gas elastic branch already wrote
                    // the new direction directly -- its vector-based
                    // CM transformation isn't reducible to a
                    // (mu_lab, phi) rotation about v_n. The Marsaglia
                    // draws above run unconditionally so RNG-stream
                    // parity is preserved across both paths.
                    if skip_lab_rotation == 0u32 {
                        let sin_theta_sq = 1.0 - mu_lab * mu_lab;
                        let mut sin_theta = 0.0;
                        if sin_theta_sq > 0.0 {
                            sin_theta = sin_theta_sq.sqrt();
                        }
                        let one_minus_w_sq = 1.0 - dz * dz;

                        let mut new_u = sin_theta * cos_phi;
                        let mut new_v = sin_theta * sin_phi;
                        let mut new_w = mu_lab;
                        if dz < 0.0 {
                            new_v = -new_v;
                            new_w = -mu_lab;
                        }
                        if one_minus_w_sq > 1e-14 {
                            let sin_phi_w = one_minus_w_sq.sqrt();
                            new_u = mu_lab * dx
                                + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / sin_phi_w;
                            new_v = mu_lab * dy
                                + sin_theta * (dy * dz * cos_phi + dx * sin_phi) / sin_phi_w;
                            new_w = mu_lab * dz - sin_theta * sin_phi_w * cos_phi;
                        }
                        dx = new_u;
                        dy = new_v;
                        dz = new_w;
                    }
                }

                // Weight-cutoff Russian roulette. After the collision is
                // fully processed (reaction selected, weight discounted, any
                // fission/photon banking done), a still-alive particle whose
                // weight dropped below `weight_cutoff` is rouletted: it
                // survives with probability `weight / weight_survive`
                // (continuing at `weight_survive`) or is killed. Exactly ONE
                // uniform is drawn, and only below the cutoff -- so an
                // above-cutoff history draws nothing extra and a survival-OFF
                // run (`survival_params[0] == 0`) never enters this block,
                // staying byte-identical to the analog kernel. Variance-
                // neutral: the expected weight is unchanged. CPU twin in
                // `shared.rs` mirrors this in the same loop position.
                if survival_on && alive == 1u32 && weight < survival_params[1] {
                    let d_rr = crate::common::pcg32::draw_uniform(state);
                    state = d_rr.state;
                    weight = weight_cutoff_roulette(weight, survival_params[2], d_rr.xi);
                    if weight == 0.0 {
                        alive = 0u32;
                    }
                }
            }
            // else: surface crossing -- kill the particle if the
            // surface it just crossed is a vacuum boundary, otherwise
            // let the next iteration re-find the cell. Vacuum is the
            // only non-transmissive boundary type yamc supports today
            // (matches `yamc_geo::BoundaryType::Vacuum` discriminant).
            if !collide_first {
                if surface_boundaries[winner_surface as usize] == BOUNDARY_VACUUM {
                    alive = 0u32;
                } else {
                    // The step landed the particle exactly ON the surface,
                    // where the point-in-region cell test (strict `<`/`>`
                    // on the surface value) is in NEITHER neighbouring cell.
                    // Nudge a hair past the surface so the next cell-find
                    // lands strictly inside the cell being entered -- the
                    // same `eps` the CPU `Cell::distance_to_surface` /
                    // `Region::is_exit_surface` use for the handoff.
                    px += dx * REGION_CROSS_EPS;
                    py += dy * REGION_CROSS_EPS;
                    pz += dz * REGION_CROSS_EPS;
                }
            }
        }

        n_steps = step + 1u32;
        step += 1u32;
    }

    // Batch-free per-history variance flush (issue #233). Emit this history's
    // per-bin `sum` (into the first half of `tally_out`) and `sum_sq` (into the
    // second half) for every touched-list AND spill entry: one atomic add per
    // touched bin, the ONLY tally atomic traffic on this path (vs one per step
    // on the per-step path). The SUM reuses the per-tally
    // `tally_fixed_point_scales[t]`; the SUM_SQ uses the derived
    // `min(S*S/2^40, 2^62)` (the plain-Rust `sum_sq_fixed_point_scale`, computed
    // identically host-side for unpacking). The owning tally `t` is recovered
    // from the flat bin index by scanning `tally_out_offsets` (n_tallies is
    // small; this runs once per history, not per step).
    if per_history_var {
        let sq_off = (tally_out.len() / 2) as u32;
        let sq_anchor = crate::common::tallies::SUMSQ_SCALE_ANCHOR;
        let sq_ceiling = crate::common::tallies::FIXED_POINT_ACC_CEILING;
        // KERMA-shape tallies (sum scale 1.0) use a dedicated sum_sq scale; the
        // generic `s*s/2^40` form underflows their eV-magnitude squares (twin:
        // `sum_sq_fixed_point_scale`).
        let kerma_scale = crate::common::tallies::KERMA_FIXED_POINT_SCALE;
        let kerma_sumsq = crate::common::tallies::KERMA_SUMSQ_SCALE;
        // Touched-list (registers).
        let mut j = 0u32;
        while j < th_count {
            let idx = th_bin[j as usize];
            let x = th_val[j as usize];
            let mut t = 0u32;
            let mut tt = 0u32;
            while tt < n_tallies {
                if idx >= tally_out_offsets[tt as usize] {
                    t = tt;
                }
                tt += 1u32;
            }
            let s = tally_fixed_point_scales[t as usize];
            // sum (symmetric round-to-nearest, twin: round_fixed_point_bits).
            let sx = x * s;
            let sbits = if sx >= 0.0 {
                (sx + 0.5) as i64
            } else {
                -((-sx + 0.5) as i64)
            };
            if per_source_var {
                // Stage 2 (fissile): accumulate into this history's SOURCE row so
                // fission descendants (later launches) fold into the same sample.
                // sum_sq is deferred to the host finalize (once per source).
                src_acc[src_base + idx as usize].fetch_add(u64::reinterpret(sbits));
            } else {
                // Stage 1 (non-fissile): flush sum + sum_sq directly.
                tally_out[idx as usize].fetch_add(u64::reinterpret(sbits));
                let s_sq = if s <= kerma_scale {
                    kerma_sumsq
                } else {
                    let s2 = s * s / sq_anchor;
                    if s2 < sq_ceiling {
                        s2
                    } else {
                        sq_ceiling
                    }
                };
                let qx = x * x * s_sq;
                let qbits = (qx + 0.5) as i64;
                tally_out[(sq_off + idx) as usize].fetch_add(u64::reinterpret(qbits));
            }
            j += 1u32;
        }
        // Spill (per-history global overflow, exact for any distinct-bin count).
        let mut k = 0u32;
        while k < spill_count {
            let idx = spill_bin[spill_base + k as usize];
            let x = spill_val[spill_base + k as usize];
            let mut t = 0u32;
            let mut tt = 0u32;
            while tt < n_tallies {
                if idx >= tally_out_offsets[tt as usize] {
                    t = tt;
                }
                tt += 1u32;
            }
            let s = tally_fixed_point_scales[t as usize];
            let sx = x * s;
            let sbits = if sx >= 0.0 {
                (sx + 0.5) as i64
            } else {
                -((-sx + 0.5) as i64)
            };
            if per_source_var {
                src_acc[src_base + idx as usize].fetch_add(u64::reinterpret(sbits));
            } else {
                tally_out[idx as usize].fetch_add(u64::reinterpret(sbits));
                let s_sq = if s <= kerma_scale {
                    kerma_sumsq
                } else {
                    let s2 = s * s / sq_anchor;
                    if s2 < sq_ceiling {
                        s2
                    } else {
                        sq_ceiling
                    }
                };
                let qx = x * x * s_sq;
                let qbits = (qx + 0.5) as i64;
                tally_out[(sq_off + idx) as usize].fetch_add(u64::reinterpret(qbits));
            }
            k += 1u32;
        }
    }

    out_alive[ABSOLUTE_POS] = alive;
    out_n_steps[ABSOLUTE_POS] = n_steps;
    out_final_energy[ABSOLUTE_POS] = energy;
}
