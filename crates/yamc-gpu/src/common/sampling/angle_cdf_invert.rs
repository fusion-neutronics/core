//! Shared tabulated angle-CDF inverter (ENDF `TabulatedAngleDistribution`).
//!
//! Given a picked incident-energy bracket's angular `(mu, cdf, pdf)` slice,
//! draw one PCG-32 uniform, binary-search (linear walk) the CDF row for the
//! bin containing the draw, then invert: histogram bins use
//! `x_j + (xi - c_j)/p_j`, LinLin bins use the quadratic CDF inversion
//! `x_j + (sqrt(p_j^2 + 2*m*(xi - c_j)) - p_j)/m` with
//! `m = (p_{j+1} - p_j)/(x_{j+1} - x_j)`. Faithful to `yamc_nuclide`'s
//! `TabulatedAngleDistribution::sample` (reaction_product.rs:128-139).
//!
//! Extracted verbatim from the neutron transport kernel's three inline
//! occurrences -- elastic angular table, inelastic (slice-B) per-MT angular
//! table, and the correlated angle-energy sub-table -- which were
//! bit-identical apart from a parameterizable default mu and an unused
//! `cdf[j+1]` dead load. One `#[cube]` helper now serves all three.
//!
//! The draw of the angular uniform happens INSIDE this helper, matching every
//! call site's RNG draw order (the uniform is drawn right after the
//! `n_mu >= 2` guard). When `n_mu < 2` no draw is made and the caller's
//! `mu_default` is returned with the state unchanged, mirroring the kernel's
//! `if n_mu >= 2` guard. Pure integer-PCG + f64 arithmetic apart from one
//! `sqrt` in the LinLin inversion, so the `#[cube]` kernel matches the
//! [`invert_angle_cdf_cpu`] twin to within a few ULPs
//! (`gpu_invert_angle_cdf_matches_cpu`).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::neutron::xs::constants::ANGLE_INTERP_LINLIN;
use cubecl::prelude::*;

/// Result of [`invert_angle_cdf`]: the sampled (and clamped to `[-1, 1]`)
/// scattering cosine and the advanced 64-bit PCG state.
#[derive(CubeType)]
pub struct AngleSample {
    pub mu: f64,
    pub state: u64,
}

/// Sample a scattering cosine from a tabulated angular CDF row. `mu_off` is
/// the slice's base offset into the flat `(mu, cdf, pdf)` buffers, `n_mu` the
/// number of points in the row, and `interp_kind` the row's interpolation
/// flag (`ANGLE_INTERP_LINLIN` vs histogram). When `n_mu < 2` the row carries
/// no usable distribution: `mu_default` is returned unchanged and the RNG is
/// not advanced. Otherwise one uniform is drawn, the CDF is inverted, the
/// result is clamped to `[-1, 1]`, and the advanced `state` is returned.
//
// `manual_clamp`: cubecl's `#[cube]` does not lower `f64::clamp` to a SPIR-V
// op, so the `[-1, 1]` guard is written as paired `if` branches (matching the
// kernel's convention); the CPU twin keeps the same shape for bit-parity.
#[allow(clippy::manual_clamp)]
#[cube]
pub fn invert_angle_cdf(
    mu_default: f64,
    mu_off: u32,
    n_mu: u32,
    interp_kind: u32,
    angle_mu: &[f64],
    angle_cdf: &[f64],
    angle_pdf: &[f64],
    state_in: u64,
) -> AngleSample {
    let mut state = state_in;
    let mut mu = mu_default;

    if n_mu >= 2u32 {
        let d_xi = crate::common::pcg32::draw_uniform(state);
        state = d_xi.state;
        let xi = d_xi.xi;

        // Walk the CDF for `j` such that `cdf[j] <= xi < cdf[j + 1]`.
        let mut j = 0u32;
        let mut j_found = 0u32;
        let mut k = 0u32;
        while k + 1u32 < n_mu {
            let c_k1 = angle_cdf[(mu_off + k + 1u32) as usize];
            if j_found == 0u32 && xi <= c_k1 {
                j = k;
                j_found = 1u32;
            }
            k += 1u32;
        }
        if j_found == 0u32 {
            j = n_mu - 2u32;
        }
        let c_j = angle_cdf[(mu_off + j) as usize];
        let x_j = angle_mu[(mu_off + j) as usize];
        let x_j1 = angle_mu[(mu_off + j + 1u32) as usize];
        let p_j = angle_pdf[(mu_off + j) as usize];
        let p_j1 = angle_pdf[(mu_off + j + 1u32) as usize];
        let dx = x_j1 - x_j;
        // Default: bracket lower endpoint `x_j`. The branches below overwrite
        // this only when a meaningful sub-bin position can be computed; the
        // degenerate `p_j == 0` and zero-slope LinLin cases fall through to
        // `x_j` (mirrors the kernel's explicit `else { mu = x_j }`).
        mu = x_j;
        if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
            let m = (p_j1 - p_j) / dx;
            let abs_m = if m < 0.0 { -m } else { m };
            if abs_m < 1e-30 {
                if p_j > 0.0 {
                    mu = x_j + (xi - c_j) / p_j;
                }
            } else {
                let mut disc = p_j * p_j + 2.0 * m * (xi - c_j);
                if disc < 0.0 {
                    disc = 0.0;
                }
                mu = x_j + (disc.sqrt() - p_j) / m;
            }
        } else if p_j > 0.0 {
            mu = x_j + (xi - c_j) / p_j;
        }
        if mu < -1.0 {
            mu = -1.0;
        }
        if mu > 1.0 {
            mu = 1.0;
        }
    }

    AngleSample { mu, state }
}

/// CPU twin of [`invert_angle_cdf`]. Same algorithm and same 64-bit-state PCG
/// draw (via `wrapping_*`), so it matches the `#[cube]` kernel to within a few
/// ULPs. Returns `(mu, state)`.
#[allow(clippy::too_many_arguments, clippy::manual_clamp)]
pub fn invert_angle_cdf_cpu(
    mu_default: f64,
    mu_off: u32,
    n_mu: u32,
    interp_kind: u32,
    angle_mu: &[f64],
    angle_cdf: &[f64],
    angle_pdf: &[f64],
    state_in: u64,
) -> (f64, u64) {
    let mut state = state_in;
    let mut mu = mu_default;

    if n_mu >= 2 {
        let s = state;
        let rand = pcg_out(s);
        state = s.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xi = (rand as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let mut j = 0u32;
        let mut j_found = 0u32;
        let mut k = 0u32;
        while k + 1 < n_mu {
            let c_k1 = angle_cdf[(mu_off + k + 1) as usize];
            if j_found == 0 && xi <= c_k1 {
                j = k;
                j_found = 1;
            }
            k += 1;
        }
        if j_found == 0 {
            j = n_mu - 2;
        }
        let c_j = angle_cdf[(mu_off + j) as usize];
        let x_j = angle_mu[(mu_off + j) as usize];
        let x_j1 = angle_mu[(mu_off + j + 1) as usize];
        let p_j = angle_pdf[(mu_off + j) as usize];
        let p_j1 = angle_pdf[(mu_off + j + 1) as usize];
        let dx = x_j1 - x_j;
        mu = x_j;
        if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
            let m = (p_j1 - p_j) / dx;
            let abs_m = if m < 0.0 { -m } else { m };
            if abs_m < 1e-30 {
                if p_j > 0.0 {
                    mu = x_j + (xi - c_j) / p_j;
                }
            } else {
                let mut disc = p_j * p_j + 2.0 * m * (xi - c_j);
                if disc < 0.0 {
                    disc = 0.0;
                }
                mu = x_j + (disc.sqrt() - p_j) / m;
            }
        } else if p_j > 0.0 {
            mu = x_j + (xi - c_j) / p_j;
        }
        if mu < -1.0 {
            mu = -1.0;
        }
        if mu > 1.0 {
            mu = 1.0;
        }
    }

    (mu, state)
}

/// Test/validation kernel: one sample per thread, each with its own slice
/// offset, point count, interpolation flag, default mu, and PCG seed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn angle_cdf_test_kernel(
    mu_default: &[f64],
    mu_off: &[u32],
    n_mu: &[u32],
    interp_kind: &[u32],
    angle_mu: &[f64],
    angle_cdf: &[f64],
    angle_pdf: &[f64],
    seeds: &[u32],
    out_mu: &mut [f64],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_mu.len() {
        terminate!();
    }
    let r = invert_angle_cdf(
        mu_default[ABSOLUTE_POS],
        mu_off[ABSOLUTE_POS],
        n_mu[ABSOLUTE_POS],
        interp_kind[ABSOLUTE_POS],
        angle_mu,
        angle_cdf,
        angle_pdf,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_mu[ABSOLUTE_POS] = r.mu;
    out_state[ABSOLUTE_POS] = r.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neutron::xs::constants::ANGLE_INTERP_HISTOGRAM;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    const MX: usize = 8; // points-per-slice stride for the test fixture

    /// Build a 3-slice angular fixture:
    ///   slice 0: 4-point LinLin distribution (rising mu spectrum)
    ///   slice 1: 3-point histogram distribution
    ///   slice 2: 2-point LinLin with zero slope (degenerate -> linear)
    #[allow(clippy::type_complexity)]
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let n_slices = 3usize;
        let mut mu = vec![0.0_f64; n_slices * MX];
        let mut cdf = vec![0.0_f64; n_slices * MX];
        let mut pdf = vec![0.0_f64; n_slices * MX];
        let mut n_mu = vec![0u32; n_slices];
        let mut interp = vec![0u32; n_slices];

        // Slice 0: 4-point LinLin over [-1, 1].
        n_mu[0] = 4;
        interp[0] = ANGLE_INTERP_LINLIN;
        let xs0 = [-1.0, -0.2, 0.5, 1.0];
        let cs0 = [0.0, 0.3, 0.75, 1.0];
        mu[0..4].copy_from_slice(&xs0);
        cdf[0..4].copy_from_slice(&cs0);
        for k in 0..3 {
            pdf[k] = (cs0[k + 1] - cs0[k]) / (xs0[k + 1] - xs0[k]);
        }
        pdf[3] = pdf[2];

        // Slice 1: 3-point histogram over [-1, 1].
        n_mu[1] = 3;
        interp[1] = ANGLE_INTERP_HISTOGRAM;
        let off = MX;
        let xs1 = [-1.0, 0.0, 1.0];
        let cs1 = [0.0, 0.6, 1.0];
        mu[off..off + 3].copy_from_slice(&xs1);
        cdf[off..off + 3].copy_from_slice(&cs1);
        pdf[off] = (cs1[1] - cs1[0]) / (xs1[1] - xs1[0]);
        pdf[off + 1] = (cs1[2] - cs1[1]) / (xs1[2] - xs1[1]);
        pdf[off + 2] = pdf[off + 1];

        // Slice 2: 2-point LinLin with equal pdf (zero slope -> linear path).
        n_mu[2] = 2;
        interp[2] = ANGLE_INTERP_LINLIN;
        let off2 = 2 * MX;
        mu[off2] = -1.0;
        mu[off2 + 1] = 1.0;
        cdf[off2] = 0.0;
        cdf[off2 + 1] = 1.0;
        pdf[off2] = 0.5;
        pdf[off2 + 1] = 0.5;

        (mu, cdf, pdf, n_mu, interp)
    }

    /// GPU angle-CDF inversion must match the CPU twin: the PCG state advances
    /// bit-for-bit (pure u64 integer math) and the sampled mu agrees to a few
    /// ULPs (one `sqrt` in the LinLin inversion). Sweeps all three slices
    /// (LinLin, histogram, degenerate) plus an `n_mu < 2` no-draw slot, over
    /// many seeds.
    #[test]
    fn gpu_invert_angle_cdf_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let (mu_buf, cdf, pdf, _n_mu_buf, _interp_buf) = fixture();

        // (mu_off, n_mu, interp) per case; case (.., 0, ..) exercises the
        // `n_mu < 2` no-draw path (the expanded seed must pass through
        // unchanged).
        let cases: &[(u32, u32, u32)] = &[
            (0, 4, ANGLE_INTERP_LINLIN),
            (MX as u32, 3, ANGLE_INTERP_HISTOGRAM),
            (2 * MX as u32, 2, ANGLE_INTERP_LINLIN),
            (0, 0, ANGLE_INTERP_LINLIN), // n_mu < 2 -> default mu, no draw
        ];

        let mut mu_default = Vec::new();
        let mut mu_off = Vec::new();
        let mut n_mu = Vec::new();
        let mut interp = Vec::new();
        let mut seeds = Vec::new();
        let n_seeds = 200u32;
        for &(off, ne, ik) in cases {
            for s in 0..n_seeds {
                mu_default.push(0.123_456); // sentinel default mu
                mu_off.push(off);
                n_mu.push(ne);
                interp.push(ik);
                seeds.push((off.wrapping_mul(7919) + s).wrapping_mul(2_654_435_761));
            }
        }
        let n = seeds.len();

        // CPU reference.
        let mut cpu_mu = vec![0.0_f64; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (m, st) = invert_angle_cdf_cpu(
                mu_default[i],
                mu_off[i],
                n_mu[i],
                interp[i],
                &mu_buf,
                &cdf,
                &pdf,
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_mu[i] = m;
            cpu_state[i] = st;
        }

        // GPU run.
        let client = ctx.client();
        let md_h = client.create_from_slice(bytemuck::cast_slice(&mu_default));
        let off_h = client.create_from_slice(bytemuck::cast_slice(&mu_off));
        let nmu_h = client.create_from_slice(bytemuck::cast_slice(&n_mu));
        let ik_h = client.create_from_slice(bytemuck::cast_slice(&interp));
        let mu_h = client.create_from_slice(bytemuck::cast_slice(&mu_buf));
        let cdf_h = client.create_from_slice(bytemuck::cast_slice(&cdf));
        let pdf_h = client.create_from_slice(bytemuck::cast_slice(&pdf));
        let seed_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let outmu_h = client.empty(n * core::mem::size_of::<f64>());
        let outs_h = client.empty(n * core::mem::size_of::<u64>());

        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            angle_cdf_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(md_h, mu_default.len()),
                BufferArg::from_raw_parts(off_h, mu_off.len()),
                BufferArg::from_raw_parts(nmu_h, n_mu.len()),
                BufferArg::from_raw_parts(ik_h, interp.len()),
                BufferArg::from_raw_parts(mu_h, mu_buf.len()),
                BufferArg::from_raw_parts(cdf_h, cdf.len()),
                BufferArg::from_raw_parts(pdf_h, pdf.len()),
                BufferArg::from_raw_parts(seed_h, seeds.len()),
                BufferArg::from_raw_parts(outmu_h.clone(), n),
                BufferArg::from_raw_parts(outs_h.clone(), n),
            );
        }
        let gpu_mu = bytemuck::cast_slice::<u8, f64>(&client.read_one(outmu_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(outs_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "sample {i}: PCG state diverged (gpu {} cpu {}, off {}, n_mu {}, seed {})",
                gpu_state[i], cpu_state[i], mu_off[i], n_mu[i], seeds[i]
            );
            let tol = 1e-9 * cpu_mu[i].abs().max(1.0);
            assert!(
                (gpu_mu[i] - cpu_mu[i]).abs() <= tol,
                "sample {i}: mu diverged (gpu {} cpu {}, tol {tol}, off {}, n_mu {}, seed {})",
                gpu_mu[i],
                cpu_mu[i],
                mu_off[i],
                n_mu[i],
                seeds[i]
            );
            // n_mu < 2 must leave the default mu and not advance the state.
            if n_mu[i] < 2 {
                assert_eq!(cpu_mu[i], mu_default[i], "sample {i}: no-draw mu changed");
                assert_eq!(
                    cpu_state[i],
                    crate::common::rng::expand_seed(seeds[i]),
                    "sample {i}: no-draw state advanced"
                );
            } else {
                assert!(
                    (-1.0..=1.0).contains(&cpu_mu[i]),
                    "sample {i}: mu out of [-1,1]: {}",
                    cpu_mu[i]
                );
            }
        }
        println!(
            "invert_angle_cdf: {} samples, GPU state bit-exact, mu within 1e-9 rel",
            n
        );
    }
}
