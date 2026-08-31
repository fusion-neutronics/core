//! On-device coupled neutron->photon production kinematics sampling (slice S3).
//!
//! After a neutron collision selects a secondary photon PRODUCT (slice S2,
//! [`crate::photon::production_select`]), this slice samples that product's
//! emitted photon **outgoing energy `E_out`** and **scattering cosine `mu`** --
//! the GPU analogue of the CPU `product.sample(energy_in, rng)` call inside
//! `yamc_physics::photon::photon_production::sample_secondary_photons`
//! (lines 60-132). S3 stops at `(E_out, mu)`: the uniform azimuth `phi` and the
//! `rotate_direction_fast` rotation are kernel-emission work (S4), not here.
//!
//! # Faithful to the CPU draw order
//!
//! The CPU `sample_uncorrelated`
//! (`yamc_nuclide::secondary::secondary_uncorrelated`) draws **angle FIRST,
//! then energy**:
//!
//! ```text
//! let mu    = angle.sample(incoming_energy, rng);   // AngleDistribution::sample
//! let e_out = energy_dist.sample(incoming_energy, rng);
//! ```
//!
//! [`sample_photon_kinematics`] preserves that order exactly so the RNG stream
//! advances identically to the CPU reference:
//!
//! 1. **Angle.** An empty angular table samples isotropic `mu = 2*xi - 1`
//!    (one draw, mirroring `AngleDistribution::sample`'s empty path). Otherwise
//!    the incident-energy bin `(i, r)` is found the same way
//!    `AngleDistribution::sample` finds it (clamp below -> `(0, 0)`, above ->
//!    `(n-2, 1)`, else lower-bound index + linear fraction), one stochastic
//!    bracket pick selects `bin in {i, i+1}` ([`pick_energy_bracket`], one
//!    draw), and the row's tabulated CDF is inverted ([`invert_angle_cdf`], one
//!    draw when `n_mu >= 2`). `mu` is clamped to `[-1, 1]`.
//! 2. **Energy.** Dispatched on the product's eout kind (packed by slice S1):
//!    - `DISCRETE`: closed form, ZERO draws.
//!      `E_out = primary_flag == 2 ? energy + awr/(awr+1) * E_in : energy`
//!      (mirrors `EnergyDistribution::DiscretePhoton::sample`).
//!    - `CONTINUOUS_TABULAR`: the full File-5/6 Law-1 continuous-tabular law
//!      ([`sample_continuous_tabular_eout`], up to two draws).
//!    - `NONE`: leaves `E_out = 0` (the product is unsamplable -- the CPU
//!      `has_valid_distribution` guard skips it; our data never tags a photon
//!      product `NONE`).
//!
//! Every draw threads the PCG-32 `state`, so the kernel and its
//! [`sample_photon_kinematics_cpu`] twin agree bit-for-bit on the integer /
//! discrete paths and to within a few ULPs where a `sqrt` is involved (the
//! LinLin angle inversion and the continuous-tabular quadratic inversion).
//! `gpu_photon_kinematics_matches_cpu` pins that on real hardware.
//!
//! The building blocks ([`pick_energy_bracket`], [`invert_angle_cdf`],
//! [`sample_continuous_tabular_eout`], [`draw_uniform`]) are the existing,
//! separately-tested shared helpers -- this slice only sequences them in the
//! CPU's draw order; it does not reimplement any sampling math.

use crate::common::pcg32::{draw_uniform, draw_uniform_cpu, expand_seed};
use crate::common::sampling::angle_cdf_invert::{invert_angle_cdf, invert_angle_cdf_cpu};
use crate::common::sampling::energy_bracket::{pick_energy_bracket, pick_energy_bracket_cpu};
use crate::common::sampling::eout_continuous_tabular::{
    sample_continuous_tabular_eout, sample_continuous_tabular_eout_cpu,
};
use crate::neutron::xs::photon_production::{
    PHOTON_EOUT_KIND_CONTINUOUS_TABULAR, PHOTON_EOUT_KIND_DISCRETE,
};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Result of [`sample_photon_kinematics`]: the sampled outgoing photon energy
/// (`e_out`, eV), the scattering cosine (`mu`, in `[-1, 1]`), and the advanced
/// PCG-32 `state`.
#[derive(CubeType)]
pub struct PhotonKin {
    pub e_out: f64,
    pub mu: f64,
    pub state: u64,
}

/// `#[cube]` photon-kinematics sampler: given a SELECTED photon `product_idx`
/// (slice S2) and the incident neutron energy `e_in`, sample `(E_out, mu)`.
/// See the module docs for the exact draw order. `seed` is the incoming PCG-32
/// state, threaded through every draw; the advanced state is returned so the
/// caller can keep sampling. The photon-angle rows are read via tight CSR
/// (issue #104): a product's incident-energy rows start at the global ae-row
/// `pa_ae_offset[product]`, and each row's mu points start at
/// `pa_mu_offset[ae_row]` (no `MAX_PHOTON_ANGLE_*` stride). The continuous-
/// tabular outgoing points are read via the tight CSR `ct_ae_offset` /
/// `ct_x_offset`.
// `manual_clamp`: cubecl's `#[cube]` does not lower `f64::clamp` to a SPIR-V
// op, so the `[-1, 1]` guard is written as paired `if` branches (matching the
// `invert_angle_cdf` convention); the CPU twin keeps the same shape for parity.
#[cube]
#[allow(clippy::too_many_arguments, clippy::manual_clamp)]
pub fn sample_photon_kinematics(
    product_idx: u32,
    e_in: f64,
    prod_eout_kind: &[u32],
    prod_line_energy: &[f64],
    prod_primary_flag: &[i32],
    prod_awr: &[f64],
    prod_dist_slot: &[u32],
    pa_n_energies: &[u32],
    pa_ae_offset: &[u32],
    pa_mu_offset: &[u32],
    pa_energy_grid: &[f64],
    pa_n_mu: &[u32],
    pa_mu: &[f64],
    pa_cdf: &[f64],
    pa_pdf: &[f64],
    pa_interp: &[u32],
    ct_ae_offset: &[u32],
    ct_x_offset: &[u32],
    ct_energy_grid: &[f64],
    ct_n_x: &[u32],
    ct_x: &[f64],
    ct_cdf: &[f64],
    ct_p: &[f64],
    ct_interp: &[u32],
    ct_n_discrete: &[u32],
    ct_n_eout: &[u32],
    ct_hist: &[u32],
    seed: u64,
) -> PhotonKin {
    let p = product_idx;
    let mut state = seed;

    // ----------------------- (1) angle FIRST -----------------------
    // `mu` is assigned in both arms of the dispatch below; declaring it via
    // the `if`-expression keeps cubecl's required initialisation without a
    // dead literal.
    let mut mu = if pa_n_energies[p as usize] == 0u32 {
        // Empty angular table -> isotropic (one draw), mirroring
        // `AngleDistribution::sample`'s empty path.
        let d = draw_uniform(state);
        state = d.state;
        2.0 * d.xi - 1.0
    } else {
        let n_ae = pa_n_energies[p as usize];
        // Find the incident-energy bin (i, r) the same way the CPU
        // `AngleDistribution::sample` does (clamp below/above, else
        // lower-bound index + linear fraction). Tight CSR (issue #104): this
        // product's rows start at the global ae-row `pa_ae_offset[p]`.
        let ae_off = pa_ae_offset[p as usize];
        let e_first = pa_energy_grid[ae_off as usize];
        let e_last = pa_energy_grid[(ae_off + n_ae - 1u32) as usize];
        let mut i_ab = 0u32;
        let mut r_ab = 0.0_f64;
        if e_in < e_first {
            i_ab = 0u32;
            r_ab = 0.0;
        } else if e_in > e_last {
            if n_ae >= 2u32 {
                i_ab = n_ae - 2u32;
            }
            r_ab = 1.0;
        } else {
            // Lower-bound walk: the largest `k + 1` whose grid energy is
            // <= e_in, clamped to [0, n-2] (matches the CPU twin and
            // `find_energy_index`). No early break -- cubecl prefers a
            // branch-free loop that scans every point.
            let mut idx = 0u32;
            let mut k = 0u32;
            while k + 1u32 < n_ae {
                let e_k1 = pa_energy_grid[(ae_off + k + 1u32) as usize];
                if e_k1 <= e_in {
                    idx = k + 1u32;
                }
                k += 1u32;
            }
            // Clamp to the top bracket [0, n-2] (n_ae >= 2 for our photon
            // angle rows). Written as a select to avoid an
            // `a > b { a = b }` assignment.
            i_ab = if idx + 2u32 <= n_ae { idx } else { n_ae - 2u32 };
            let e_i = pa_energy_grid[(ae_off + i_ab) as usize];
            let e_i1 = pa_energy_grid[(ae_off + i_ab + 1u32) as usize];
            let de = e_i1 - e_i;
            if de > 0.0 {
                r_ab = (e_in - e_i) / de;
            }
        }

        // Stochastic bracket pick (one draw): bin in {i, i+1}.
        let bp = pick_energy_bracket(r_ab, i_ab, n_ae, state);
        state = bp.state;
        let bin = bp.bin;

        // Invert the chosen row's tabulated angle CDF (one draw when
        // n_mu >= 2). Default mu = the row's lower endpoint (the closest
        // analogue of the CPU single-point `Tabulated::sample` return).
        // Tight CSR (issue #104): the row's mu points start at
        // `pa_mu_offset[row]`.
        let row = ae_off + bin;
        let mu_off = pa_mu_offset[row as usize];
        let n_mu = pa_n_mu[row as usize];
        let mu_default = pa_mu[mu_off as usize];
        let interp = pa_interp[row as usize];
        let asamp = invert_angle_cdf(
            mu_default, mu_off, n_mu, interp, pa_mu, pa_cdf, pa_pdf, state,
        );
        state = asamp.state;
        asamp.mu
    };
    // Final clamp (invert_angle_cdf already clamps, but the isotropic and
    // single-point-default paths do not; mirrors the CPU `mu_clamped`).
    if mu < -1.0 {
        mu = -1.0;
    }
    if mu > 1.0 {
        mu = 1.0;
    }

    // ---------------------- (2) energy SECOND ----------------------
    let mut e_out = 0.0_f64;
    let kind = prod_eout_kind[p as usize];
    if kind == PHOTON_EOUT_KIND_DISCRETE {
        // Closed form, ZERO draws (EnergyDistribution::DiscretePhoton).
        let line = prod_line_energy[p as usize];
        let pflag = prod_primary_flag[p as usize];
        if pflag == 2i32 {
            let awr = prod_awr[p as usize];
            e_out = line + awr / (awr + 1.0) * e_in;
        } else {
            e_out = line;
        }
    } else if kind == PHOTON_EOUT_KIND_CONTINUOUS_TABULAR {
        let slot = prod_dist_slot[p as usize];
        // Tight CSR (issue #104): the slot's incident-energy rows start at the
        // global ae-row `ct_ae_offset[slot]`; the per-row (x, cdf, p) points are
        // read from the full arrays via `ct_x_offset[eg_off_e + bin]` inside the
        // shared sampler. No fixed per-axis stride.
        let eg_off_e = ct_ae_offset[slot as usize];
        let n_eout = ct_n_eout[slot as usize];
        let hist_outer = ct_hist[slot as usize];
        let es = sample_continuous_tabular_eout(
            // e_default = 0.0: an empty (n_x == 0) row drops the photon instead
            // of emitting the incident neutron energy (issue #175). The second
            // arg is the incident energy used for bracketing.
            0.0,
            e_in,
            eg_off_e,
            n_eout,
            hist_outer,
            ct_x_offset,
            ct_energy_grid,
            ct_n_x,
            ct_x,
            ct_cdf,
            ct_p,
            ct_interp,
            ct_n_discrete,
            state,
        );
        state = es.state;
        e_out = es.e_cm;
    }
    // kind == NONE: e_out stays 0 (unsamplable product).

    PhotonKin { e_out, mu, state }
}

/// CPU twin of [`sample_photon_kinematics`]. Identical structure and identical
/// 32-bit-PCG draws (via the `_cpu` helper twins), so it matches the `#[cube]`
/// kernel bit-for-bit on the integer / discrete paths and to within a few ULPs
/// where a `sqrt` is involved. Returns `(e_out, mu, state)`.
#[allow(clippy::too_many_arguments, clippy::manual_clamp)]
pub fn sample_photon_kinematics_cpu(
    product_idx: u32,
    e_in: f64,
    prod_eout_kind: &[u32],
    prod_line_energy: &[f64],
    prod_primary_flag: &[i32],
    prod_awr: &[f64],
    prod_dist_slot: &[u32],
    pa_n_energies: &[u32],
    pa_ae_offset: &[u32],
    pa_mu_offset: &[u32],
    pa_energy_grid: &[f64],
    pa_n_mu: &[u32],
    pa_mu: &[f64],
    pa_cdf: &[f64],
    pa_pdf: &[f64],
    pa_interp: &[u32],
    ct_ae_offset: &[u32],
    ct_x_offset: &[u32],
    ct_energy_grid: &[f64],
    ct_n_x: &[u32],
    ct_x: &[f64],
    ct_cdf: &[f64],
    ct_p: &[f64],
    ct_interp: &[u32],
    ct_n_discrete: &[u32],
    ct_n_eout: &[u32],
    ct_hist: &[u32],
    seed: u64,
) -> (f64, f64, u64) {
    let p = product_idx;
    let mut state = seed;

    // (1) angle first.
    let mut mu;
    let n_ae = pa_n_energies[p as usize];
    if n_ae == 0 {
        let (xi, st) = draw_uniform_cpu(state);
        state = st;
        mu = 2.0 * xi - 1.0;
    } else {
        // Tight CSR (issue #104): this product's rows start at the global
        // ae-row `pa_ae_offset[p]`.
        let ae_off = pa_ae_offset[p as usize];
        let e_first = pa_energy_grid[ae_off as usize];
        let e_last = pa_energy_grid[(ae_off + n_ae - 1) as usize];
        let (i_ab, r_ab) = if e_in < e_first {
            (0u32, 0.0_f64)
        } else if e_in > e_last {
            (n_ae.saturating_sub(2), 1.0)
        } else {
            // Lower-bound index: largest k with energy_grid[k] <= e_in,
            // clamped to [0, n-2] (matches `find_energy_index`).
            let mut idx = 0u32;
            let mut k = 0u32;
            while k + 1 < n_ae {
                let e_k1 = pa_energy_grid[(ae_off + k + 1) as usize];
                if e_k1 <= e_in {
                    idx = k + 1;
                }
                k += 1;
            }
            let idx = idx.min(n_ae.saturating_sub(2));
            let e_i = pa_energy_grid[(ae_off + idx) as usize];
            let e_i1 = pa_energy_grid[(ae_off + idx + 1) as usize];
            let de = e_i1 - e_i;
            let r = if de > 0.0 { (e_in - e_i) / de } else { 0.0 };
            (idx, r)
        };

        let (bin, st) = pick_energy_bracket_cpu(r_ab, i_ab, n_ae, state);
        state = st;

        // Tight CSR (issue #104): the row's mu points start at
        // `pa_mu_offset[row]`.
        let row = ae_off + bin;
        let mu_off = pa_mu_offset[row as usize];
        let n_mu = pa_n_mu[row as usize];
        let mu_default = pa_mu[mu_off as usize];
        let interp = pa_interp[row as usize];
        let (m, st) = invert_angle_cdf_cpu(
            mu_default, mu_off, n_mu, interp, pa_mu, pa_cdf, pa_pdf, state,
        );
        state = st;
        mu = m;
    }
    if mu < -1.0 {
        mu = -1.0;
    }
    if mu > 1.0 {
        mu = 1.0;
    }

    // (2) energy second.
    let mut e_out = 0.0_f64;
    let kind = prod_eout_kind[p as usize];
    if kind == PHOTON_EOUT_KIND_DISCRETE {
        let line = prod_line_energy[p as usize];
        let pflag = prod_primary_flag[p as usize];
        e_out = if pflag == 2 {
            let awr = prod_awr[p as usize];
            line + awr / (awr + 1.0) * e_in
        } else {
            line
        };
    } else if kind == PHOTON_EOUT_KIND_CONTINUOUS_TABULAR {
        let slot = prod_dist_slot[p as usize];
        let eg_off_e = ct_ae_offset[slot as usize];
        let n_eout = ct_n_eout[slot as usize];
        let hist_outer = ct_hist[slot as usize];
        let (e, st) = sample_continuous_tabular_eout_cpu(
            // e_default = 0.0 (see the kernel twin above; issue #175).
            0.0,
            e_in,
            eg_off_e,
            n_eout,
            hist_outer,
            ct_x_offset,
            ct_energy_grid,
            ct_n_x,
            ct_x,
            ct_cdf,
            ct_p,
            ct_interp,
            ct_n_discrete,
            state,
        );
        state = st;
        e_out = e;
    }

    (e_out, mu, state)
}

/// Test/validation kernel: one photon-kinematics sample per thread, each from
/// its own `(product_idx, e_in, seed)`; each per-history u32 seed is expanded
/// to the 64-bit PCG state in-kernel via [`expand_seed`]. The packed table
/// buffers are shared.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn photon_kinematics_kernel(
    product_idx: &[u32],
    e_in: &[f64],
    prod_eout_kind: &[u32],
    prod_line_energy: &[f64],
    prod_primary_flag: &[i32],
    prod_awr: &[f64],
    prod_dist_slot: &[u32],
    pa_n_energies: &[u32],
    pa_ae_offset: &[u32],
    pa_mu_offset: &[u32],
    pa_energy_grid: &[f64],
    pa_n_mu: &[u32],
    pa_mu: &[f64],
    pa_cdf: &[f64],
    pa_pdf: &[f64],
    pa_interp: &[u32],
    ct_ae_offset: &[u32],
    ct_x_offset: &[u32],
    ct_energy_grid: &[f64],
    ct_n_x: &[u32],
    ct_x: &[f64],
    ct_cdf: &[f64],
    ct_p: &[f64],
    ct_interp: &[u32],
    ct_n_discrete: &[u32],
    ct_n_eout: &[u32],
    ct_hist: &[u32],
    seeds: &[u32],
    out_e: &mut [f64],
    out_mu: &mut [f64],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = sample_photon_kinematics(
        product_idx[ABSOLUTE_POS],
        e_in[ABSOLUTE_POS],
        prod_eout_kind,
        prod_line_energy,
        prod_primary_flag,
        prod_awr,
        prod_dist_slot,
        pa_n_energies,
        pa_ae_offset,
        pa_mu_offset,
        pa_energy_grid,
        pa_n_mu,
        pa_mu,
        pa_cdf,
        pa_pdf,
        pa_interp,
        ct_ae_offset,
        ct_x_offset,
        ct_energy_grid,
        ct_n_x,
        ct_x,
        ct_cdf,
        ct_p,
        ct_interp,
        ct_n_discrete,
        ct_n_eout,
        ct_hist,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_mu[ABSOLUTE_POS] = r.mu;
    out_state[ABSOLUTE_POS] = r.state;
}

/// Run the photon-kinematics kernel for each sample. The packed table buffers
/// are one material's; `product_idx[k]` / `e_in[k]` / `seeds[k]` drive sample
/// `k`. Returns `(e_out, mu, advanced_state)` per sample. For test/validation
/// use; the photon-angle rows are read via the tight CSR `pa_ae_offset` /
/// `pa_mu_offset` and the eout continuous-tabular points via the tight CSR
/// `ct_ae_offset` / `ct_x_offset` (issue #104).
#[allow(clippy::too_many_arguments)]
pub fn run_photon_kinematics(
    ctx: &GpuContext,
    product_idx: &[u32],
    e_in: &[f64],
    prod_eout_kind: &[u32],
    prod_line_energy: &[f64],
    prod_primary_flag: &[i32],
    prod_awr: &[f64],
    prod_dist_slot: &[u32],
    pa_n_energies: &[u32],
    pa_ae_offset: &[u32],
    pa_mu_offset: &[u32],
    pa_energy_grid: &[f64],
    pa_n_mu: &[u32],
    pa_mu: &[f64],
    pa_cdf: &[f64],
    pa_pdf: &[f64],
    pa_interp: &[u32],
    ct_ae_offset: &[u32],
    ct_x_offset: &[u32],
    ct_energy_grid: &[f64],
    ct_n_x: &[u32],
    ct_x: &[f64],
    ct_cdf: &[f64],
    ct_p: &[f64],
    ct_interp: &[u32],
    ct_n_discrete: &[u32],
    ct_n_eout: &[u32],
    ct_hist: &[u32],
    seeds: &[u32],
) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    let n = seeds.len();
    assert_eq!(product_idx.len(), n);
    assert_eq!(e_in.len(), n);
    let client = ctx.client();

    let pidx_h = client.create_from_slice(bytemuck::cast_slice(product_idx));
    let ein_h = client.create_from_slice(bytemuck::cast_slice(e_in));
    let kind_h = client.create_from_slice(bytemuck::cast_slice(prod_eout_kind));
    let line_h = client.create_from_slice(bytemuck::cast_slice(prod_line_energy));
    let pflag_h = client.create_from_slice(bytemuck::cast_slice(prod_primary_flag));
    let awr_h = client.create_from_slice(bytemuck::cast_slice(prod_awr));
    let slot_h = client.create_from_slice(bytemuck::cast_slice(prod_dist_slot));
    let pane_h = client.create_from_slice(bytemuck::cast_slice(pa_n_energies));
    let paaeoff_h = client.create_from_slice(bytemuck::cast_slice(pa_ae_offset));
    let pamuoff_h = client.create_from_slice(bytemuck::cast_slice(pa_mu_offset));
    let paeg_h = client.create_from_slice(bytemuck::cast_slice(pa_energy_grid));
    let panmu_h = client.create_from_slice(bytemuck::cast_slice(pa_n_mu));
    let pamu_h = client.create_from_slice(bytemuck::cast_slice(pa_mu));
    let pacdf_h = client.create_from_slice(bytemuck::cast_slice(pa_cdf));
    let papdf_h = client.create_from_slice(bytemuck::cast_slice(pa_pdf));
    let pai_h = client.create_from_slice(bytemuck::cast_slice(pa_interp));
    let ctaeoff_h = client.create_from_slice(bytemuck::cast_slice(ct_ae_offset));
    let ctxoff_h = client.create_from_slice(bytemuck::cast_slice(ct_x_offset));
    let cteg_h = client.create_from_slice(bytemuck::cast_slice(ct_energy_grid));
    let ctnx_h = client.create_from_slice(bytemuck::cast_slice(ct_n_x));
    let ctx_h = client.create_from_slice(bytemuck::cast_slice(ct_x));
    let ctcdf_h = client.create_from_slice(bytemuck::cast_slice(ct_cdf));
    let ctp_h = client.create_from_slice(bytemuck::cast_slice(ct_p));
    let cti_h = client.create_from_slice(bytemuck::cast_slice(ct_interp));
    let ctnd_h = client.create_from_slice(bytemuck::cast_slice(ct_n_discrete));
    let ctne_h = client.create_from_slice(bytemuck::cast_slice(ct_n_eout));
    let cth_h = client.create_from_slice(bytemuck::cast_slice(ct_hist));
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));

    let oute_h = client.empty(n * core::mem::size_of::<f64>());
    let outmu_h = client.empty(n * core::mem::size_of::<f64>());
    // One u64 state output per sample (== per seed).
    let outs_h = client.empty(n * core::mem::size_of::<u64>());

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        photon_kinematics_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(pidx_h, product_idx.len()),
            BufferArg::from_raw_parts(ein_h, e_in.len()),
            BufferArg::from_raw_parts(kind_h, prod_eout_kind.len()),
            BufferArg::from_raw_parts(line_h, prod_line_energy.len()),
            BufferArg::from_raw_parts(pflag_h, prod_primary_flag.len()),
            BufferArg::from_raw_parts(awr_h, prod_awr.len()),
            BufferArg::from_raw_parts(slot_h, prod_dist_slot.len()),
            BufferArg::from_raw_parts(pane_h, pa_n_energies.len()),
            BufferArg::from_raw_parts(paaeoff_h, pa_ae_offset.len()),
            BufferArg::from_raw_parts(pamuoff_h, pa_mu_offset.len()),
            BufferArg::from_raw_parts(paeg_h, pa_energy_grid.len()),
            BufferArg::from_raw_parts(panmu_h, pa_n_mu.len()),
            BufferArg::from_raw_parts(pamu_h, pa_mu.len()),
            BufferArg::from_raw_parts(pacdf_h, pa_cdf.len()),
            BufferArg::from_raw_parts(papdf_h, pa_pdf.len()),
            BufferArg::from_raw_parts(pai_h, pa_interp.len()),
            BufferArg::from_raw_parts(ctaeoff_h, ct_ae_offset.len()),
            BufferArg::from_raw_parts(ctxoff_h, ct_x_offset.len()),
            BufferArg::from_raw_parts(cteg_h, ct_energy_grid.len()),
            BufferArg::from_raw_parts(ctnx_h, ct_n_x.len()),
            BufferArg::from_raw_parts(ctx_h, ct_x.len()),
            BufferArg::from_raw_parts(ctcdf_h, ct_cdf.len()),
            BufferArg::from_raw_parts(ctp_h, ct_p.len()),
            BufferArg::from_raw_parts(cti_h, ct_interp.len()),
            BufferArg::from_raw_parts(ctnd_h, ct_n_discrete.len()),
            BufferArg::from_raw_parts(ctne_h, ct_n_eout.len()),
            BufferArg::from_raw_parts(cth_h, ct_hist.len()),
            BufferArg::from_raw_parts(seed_h, seeds.len()),
            BufferArg::from_raw_parts(oute_h.clone(), n),
            BufferArg::from_raw_parts(outmu_h.clone(), n),
            BufferArg::from_raw_parts(outs_h.clone(), n),
        );
    }
    let e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oute_h).unwrap()).to_vec();
    let mu = bytemuck::cast_slice::<u8, f64>(&client.read_one(outmu_h).unwrap()).to_vec();
    let st = bytemuck::cast_slice::<u8, u64>(&client.read_one(outs_h).unwrap()).to_vec();
    (e, mu, st)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neutron::xs::constants::{ANGLE_INTERP_HISTOGRAM, ANGLE_INTERP_LINLIN};
    use crate::neutron::xs::photon_production::{
        extract_photon_production_xs, PHOTON_DIST_SLOT_NONE, PHOTON_EOUT_KIND_NONE,
    };
    use crate::neutron::xs::photon_production::{
        PHOTON_EOUT_KIND_CONTINUOUS_TABULAR, PHOTON_EOUT_KIND_DISCRETE,
    };
    use crate::{GpuContext, GpuInitError};
    use yamc_nuclide::nuclide::Nuclide;
    // Local synthetic-fixture dimensions for the continuous-tabular photon
    // table. The kernel reads via CSR offsets, so these are just the
    // test's own padded-buffer strides (no dependency on the deleted
    // deleted per-axis GPU caps, issue #104).
    const CT_AE: usize = 128;
    const CT_X: usize = 512;

    /// A synthetic packed photon table exercising every kinematics path:
    ///   product 0: DISCRETE, primary_flag != 2 (fixed line), 2-point LinLin angle.
    ///   product 1: DISCRETE, primary_flag == 2 (E_in-dependent), 2-point LinLin angle.
    ///   product 2: CONTINUOUS_TABULAR (slot 0), multi-point LinLin angle.
    /// Returns the full set of buffers the sampler reads.
    #[allow(clippy::type_complexity)]
    struct SynthTable {
        prod_eout_kind: Vec<u32>,
        prod_line_energy: Vec<f64>,
        prod_primary_flag: Vec<i32>,
        prod_awr: Vec<f64>,
        prod_dist_slot: Vec<u32>,
        pa_n_energies: Vec<u32>,
        pa_ae_offset: Vec<u32>,
        pa_mu_offset: Vec<u32>,
        pa_energy_grid: Vec<f64>,
        pa_n_mu: Vec<u32>,
        pa_mu: Vec<f64>,
        pa_cdf: Vec<f64>,
        pa_pdf: Vec<f64>,
        pa_interp: Vec<u32>,
        ct_ae_offset: Vec<u32>,
        ct_x_offset: Vec<u32>,
        ct_energy_grid: Vec<f64>,
        ct_n_x: Vec<u32>,
        ct_x: Vec<f64>,
        ct_cdf: Vec<f64>,
        ct_p: Vec<f64>,
        ct_interp: Vec<u32>,
        ct_n_discrete: Vec<u32>,
        ct_n_eout: Vec<u32>,
        ct_hist: Vec<u32>,
    }

    fn synth_table() -> SynthTable {
        let n_product = 3usize;
        let n_ct = 1usize;

        let prod_eout_kind = vec![
            PHOTON_EOUT_KIND_DISCRETE,
            PHOTON_EOUT_KIND_DISCRETE,
            PHOTON_EOUT_KIND_CONTINUOUS_TABULAR,
        ];
        let prod_line_energy = vec![1.0e6, 5.0e5, 0.0];
        let prod_primary_flag = vec![0i32, 2i32, 0i32];
        let prod_awr = vec![0.0, 55.45, 0.0];
        let prod_dist_slot = vec![PHOTON_DIST_SLOT_NONE, PHOTON_DIST_SLOT_NONE, 0u32];

        // ---- per-product angle buffers (tight CSR, issue #104) ----
        // `pa_n_energies` stays per-product; the per-row arrays
        // (`pa_energy_grid` / `pa_n_mu` / `pa_interp`) are concatenated tight,
        // and the per-point arrays (`pa_mu` / `pa_cdf` / `pa_pdf`) tighter
        // still. `pa_ae_offset` (one per product) and `pa_mu_offset` (one per
        // ae-row) are the running prefix sums, exactly as the extractor builds
        // them.
        let mut pa_n_energies = vec![0u32; n_product];
        let mut pa_ae_offset: Vec<u32> = Vec::new();
        let mut pa_mu_offset: Vec<u32> = Vec::new();
        let mut pa_energy_grid: Vec<f64> = Vec::new();
        let mut pa_n_mu: Vec<u32> = Vec::new();
        let mut pa_mu: Vec<f64> = Vec::new();
        let mut pa_cdf: Vec<f64> = Vec::new();
        let mut pa_pdf: Vec<f64> = Vec::new();
        let mut pa_interp: Vec<u32> = Vec::new();

        // Append one ae-row's mu points tight, recording its mu base offset.
        let push_row = |pa_ae_first: bool,
                        p: usize,
                        e_in: f64,
                        xs: &[f64],
                        cs: &[f64],
                        ps: &[f64],
                        pa_n_energies: &mut [u32],
                        pa_ae_offset: &mut Vec<u32>,
                        pa_mu_offset: &mut Vec<u32>,
                        pa_energy_grid: &mut Vec<f64>,
                        pa_n_mu: &mut Vec<u32>,
                        pa_mu: &mut Vec<f64>,
                        pa_cdf: &mut Vec<f64>,
                        pa_pdf: &mut Vec<f64>,
                        pa_interp: &mut Vec<u32>| {
            if pa_ae_first {
                pa_n_energies[p] = 2;
                pa_ae_offset.push(pa_n_mu.len() as u32);
            }
            pa_mu_offset.push(pa_mu.len() as u32);
            pa_energy_grid.push(e_in);
            pa_n_mu.push(xs.len() as u32);
            pa_interp.push(ANGLE_INTERP_LINLIN);
            pa_mu.extend_from_slice(xs);
            pa_cdf.extend_from_slice(cs);
            pa_pdf.extend_from_slice(ps);
        };

        // Products 0 and 1: two incident energies, each a 2-point LinLin
        // (isotropic-like) row.
        let two_xs = [-1.0_f64, 1.0];
        let two_cs = [0.0_f64, 1.0];
        let two_ps = [0.5_f64, 0.5];
        for p in [0usize, 1usize] {
            for (k, &e_in) in [1.0e6_f64, 1.0e7].iter().enumerate() {
                push_row(
                    k == 0,
                    p,
                    e_in,
                    &two_xs,
                    &two_cs,
                    &two_ps,
                    &mut pa_n_energies,
                    &mut pa_ae_offset,
                    &mut pa_mu_offset,
                    &mut pa_energy_grid,
                    &mut pa_n_mu,
                    &mut pa_mu,
                    &mut pa_cdf,
                    &mut pa_pdf,
                    &mut pa_interp,
                );
            }
        }

        // Product 2: two incident energies, each a 4-point LinLin angle
        // (a genuinely non-isotropic, rising-mu spectrum) so the LinLin
        // quadratic inversion is exercised.
        let four_xs = [-1.0_f64, -0.2, 0.5, 1.0];
        let four_cs = [0.0_f64, 0.3, 0.75, 1.0];
        let mut four_ps = [0.0_f64; 4];
        for k in 0..3 {
            four_ps[k] = (four_cs[k + 1] - four_cs[k]) / (four_xs[k + 1] - four_xs[k]);
        }
        four_ps[3] = four_ps[2];
        for (k, &e_in) in [1.0e6_f64, 1.0e7].iter().enumerate() {
            push_row(
                k == 0,
                2,
                e_in,
                &four_xs,
                &four_cs,
                &four_ps,
                &mut pa_n_energies,
                &mut pa_ae_offset,
                &mut pa_mu_offset,
                &mut pa_energy_grid,
                &mut pa_n_mu,
                &mut pa_mu,
                &mut pa_cdf,
                &mut pa_pdf,
                &mut pa_interp,
            );
        }

        // ---- continuous-tabular eout slot 0 (slot-major, CT_AE x CT_X) ----
        let mut ct_energy_grid = vec![0.0_f64; n_ct * CT_AE];
        let mut ct_n_x = vec![0u32; n_ct * CT_AE];
        let mut ct_x = vec![0.0_f64; n_ct * CT_AE * CT_X];
        let mut ct_cdf = vec![0.0_f64; n_ct * CT_AE * CT_X];
        let mut ct_p = vec![0.0_f64; n_ct * CT_AE * CT_X];
        let mut ct_interp = vec![ANGLE_INTERP_HISTOGRAM; n_ct * CT_AE];
        let ct_n_discrete = vec![0u32; n_ct * CT_AE];
        let ct_n_eout = vec![2u32];
        let ct_hist = vec![0u32];

        // Two incident energies (1 and 2 MeV), each a 4-point LinLin spectrum.
        ct_energy_grid[0] = 1.0e6;
        ct_energy_grid[1] = 2.0e6;
        for (ae, scale) in [(0usize, 1.0_f64), (1usize, 1.4_f64)] {
            ct_n_x[ae] = 4;
            ct_interp[ae] = ANGLE_INTERP_LINLIN;
            let off = ae * CT_X;
            let xs = [0.1e6 * scale, 0.4e6 * scale, 0.8e6 * scale, 1.0e6 * scale];
            let cs = [0.0, 0.35, 0.8, 1.0];
            ct_x[off..off + 4].copy_from_slice(&xs);
            ct_cdf[off..off + 4].copy_from_slice(&cs);
            for k in 0..3 {
                let dxk = xs[k + 1] - xs[k];
                ct_p[off + k] = if dxk > 0.0 {
                    (cs[k + 1] - cs[k]) / dxk
                } else {
                    0.0
                };
            }
            ct_p[off + 3] = ct_p[off + 2];
        }

        // Tight CSR offsets (issue #104) describing this fixture's PADDED
        // layout: each slot owns CT_AE rows, each row CT_X points wide. The
        // sampler reads the same padded data through these offsets, so the
        // values stay byte-identical.
        let ct_ae_offset: Vec<u32> = (0..n_ct).map(|s| (s * CT_AE) as u32).collect();
        let ct_x_offset: Vec<u32> = (0..n_ct * CT_AE).map(|r| (r * CT_X) as u32).collect();

        SynthTable {
            prod_eout_kind,
            prod_line_energy,
            prod_primary_flag,
            prod_awr,
            prod_dist_slot,
            pa_n_energies,
            pa_ae_offset,
            pa_mu_offset,
            pa_energy_grid,
            pa_n_mu,
            pa_mu,
            pa_cdf,
            pa_pdf,
            pa_interp,
            ct_ae_offset,
            ct_x_offset,
            ct_energy_grid,
            ct_n_x,
            ct_x,
            ct_cdf,
            ct_p,
            ct_interp,
            ct_n_discrete,
            ct_n_eout,
            ct_hist,
        }
    }

    /// GPU photon-kinematics must match the CPU twin: the PCG state advances
    /// bit-for-bit (pure u64 integer math through every draw), so draw counts
    /// agree; e_out is exact on the discrete paths and within 1e-9 rel on the
    /// continuous-tabular path (one sqrt); mu agrees to ~1e-12 and stays in
    /// [-1, 1]. Sweeps discrete-fixed, discrete-E_in-dependent, and
    /// continuous-tabular products; a 2-point and a multi-point LinLin angle
    /// row; and several e_in spanning below / inside / above the grids, across
    /// many seeds.
    #[test]
    fn gpu_photon_kinematics_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let t = synth_table();

        // (product_idx, incident energies): below / inside / above the grids.
        let energies = [3.0e5_f64, 1.0e6, 1.5e6, 5.0e6, 1.0e7, 2.0e7];
        let n_seeds = 200u32;
        let mut product_idx = Vec::new();
        let mut e_in = Vec::new();
        let mut seeds = Vec::new();
        for p in 0..t.prod_eout_kind.len() as u32 {
            for &e in &energies {
                for s in 0..n_seeds {
                    product_idx.push(p);
                    e_in.push(e);
                    seeds.push((p * 7919 + s).wrapping_mul(2_654_435_761) ^ (e as u32));
                }
            }
        }
        let n = seeds.len();

        // CPU reference.
        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_mu = vec![0.0_f64; n];
        let mut cpu_state = vec![0u64; n];
        for k in 0..n {
            let (e, mu, st) = sample_photon_kinematics_cpu(
                product_idx[k],
                e_in[k],
                &t.prod_eout_kind,
                &t.prod_line_energy,
                &t.prod_primary_flag,
                &t.prod_awr,
                &t.prod_dist_slot,
                &t.pa_n_energies,
                &t.pa_ae_offset,
                &t.pa_mu_offset,
                &t.pa_energy_grid,
                &t.pa_n_mu,
                &t.pa_mu,
                &t.pa_cdf,
                &t.pa_pdf,
                &t.pa_interp,
                &t.ct_ae_offset,
                &t.ct_x_offset,
                &t.ct_energy_grid,
                &t.ct_n_x,
                &t.ct_x,
                &t.ct_cdf,
                &t.ct_p,
                &t.ct_interp,
                &t.ct_n_discrete,
                &t.ct_n_eout,
                &t.ct_hist,
                crate::common::rng::expand_seed(seeds[k]),
            );
            cpu_e[k] = e;
            cpu_mu[k] = mu;
            cpu_state[k] = st;
        }

        let (gpu_e, gpu_mu, gpu_state) = run_photon_kinematics(
            &ctx,
            &product_idx,
            &e_in,
            &t.prod_eout_kind,
            &t.prod_line_energy,
            &t.prod_primary_flag,
            &t.prod_awr,
            &t.prod_dist_slot,
            &t.pa_n_energies,
            &t.pa_ae_offset,
            &t.pa_mu_offset,
            &t.pa_energy_grid,
            &t.pa_n_mu,
            &t.pa_mu,
            &t.pa_cdf,
            &t.pa_pdf,
            &t.pa_interp,
            &t.ct_ae_offset,
            &t.ct_x_offset,
            &t.ct_energy_grid,
            &t.ct_n_x,
            &t.ct_x,
            &t.ct_cdf,
            &t.ct_p,
            &t.ct_interp,
            &t.ct_n_discrete,
            &t.ct_n_eout,
            &t.ct_hist,
            &seeds,
        );
        assert_eq!(gpu_e.len(), n);

        for k in 0..n {
            assert_eq!(
                gpu_state[k], cpu_state[k],
                "sample {k}: PCG state diverged (gpu {} cpu {}, product {}, e_in {}, seed {})",
                gpu_state[k], cpu_state[k], product_idx[k], e_in[k], seeds[k]
            );
            // mu within ~1e-12, in [-1, 1].
            assert!(
                (gpu_mu[k] - cpu_mu[k]).abs() <= 1e-12 + 1e-12 * cpu_mu[k].abs(),
                "sample {k}: mu diverged (gpu {} cpu {}, product {}, e_in {})",
                gpu_mu[k],
                cpu_mu[k],
                product_idx[k],
                e_in[k]
            );
            assert!(
                (-1.0..=1.0).contains(&gpu_mu[k]),
                "sample {k}: mu out of [-1, 1]: {}",
                gpu_mu[k]
            );
            // e_out parity. A fixed discrete line (primary_flag != 2) is a
            // pure buffer copy -> BIT-EXACT. The primary_flag == 2 line
            // (E_out = line + awr/(awr+1) * E_in) involves an f64 divide +
            // multiply, which -- like the sqrt paths -- can round one ULP
            // differently across CPU and GPU, so it is checked to 1e-9 rel,
            // same as the continuous-tabular path.
            let pidx = product_idx[k] as usize;
            let kind = t.prod_eout_kind[pidx];
            if kind == PHOTON_EOUT_KIND_DISCRETE && t.prod_primary_flag[pidx] != 2 {
                assert_eq!(
                    gpu_e[k].to_bits(),
                    cpu_e[k].to_bits(),
                    "sample {k}: fixed discrete e_out not bit-exact (gpu {} cpu {})",
                    gpu_e[k],
                    cpu_e[k]
                );
            } else {
                let tol = 1e-9 * cpu_e[k].abs().max(1.0);
                assert!(
                    (gpu_e[k] - cpu_e[k]).abs() <= tol,
                    "sample {k}: e_out diverged (gpu {} cpu {}, tol {tol}, kind {kind}, e_in {})",
                    gpu_e[k],
                    cpu_e[k],
                    e_in[k]
                );
            }
        }

        // Sanity on the discrete closed forms (CPU side): product 0 is a fixed
        // line; product 1 scales with e_in.
        let awr = t.prod_awr[1];
        for &e in &energies {
            let (e0, _, _) = sample_photon_kinematics_cpu(
                0,
                e,
                &t.prod_eout_kind,
                &t.prod_line_energy,
                &t.prod_primary_flag,
                &t.prod_awr,
                &t.prod_dist_slot,
                &t.pa_n_energies,
                &t.pa_ae_offset,
                &t.pa_mu_offset,
                &t.pa_energy_grid,
                &t.pa_n_mu,
                &t.pa_mu,
                &t.pa_cdf,
                &t.pa_pdf,
                &t.pa_interp,
                &t.ct_ae_offset,
                &t.ct_x_offset,
                &t.ct_energy_grid,
                &t.ct_n_x,
                &t.ct_x,
                &t.ct_cdf,
                &t.ct_p,
                &t.ct_interp,
                &t.ct_n_discrete,
                &t.ct_n_eout,
                &t.ct_hist,
                crate::common::rng::expand_seed(0xABCD_1234),
            );
            assert_eq!(e0, 1.0e6, "product 0 must be the fixed line");
            let (e1, _, _) = sample_photon_kinematics_cpu(
                1,
                e,
                &t.prod_eout_kind,
                &t.prod_line_energy,
                &t.prod_primary_flag,
                &t.prod_awr,
                &t.prod_dist_slot,
                &t.pa_n_energies,
                &t.pa_ae_offset,
                &t.pa_mu_offset,
                &t.pa_energy_grid,
                &t.pa_n_mu,
                &t.pa_mu,
                &t.pa_cdf,
                &t.pa_pdf,
                &t.pa_interp,
                &t.ct_ae_offset,
                &t.ct_x_offset,
                &t.ct_energy_grid,
                &t.ct_n_x,
                &t.ct_x,
                &t.ct_cdf,
                &t.ct_p,
                &t.ct_interp,
                &t.ct_n_discrete,
                &t.ct_n_eout,
                &t.ct_hist,
                crate::common::rng::expand_seed(0xABCD_1234),
            );
            let expected = 5.0e5 + awr / (awr + 1.0) * e;
            assert!(
                (e1 - expected).abs() <= 1e-6 * expected,
                "product 1 (primary_flag==2) e_out {e1} != {expected} at e_in {e}"
            );
        }

        println!(
            "photon kinematics: {n} samples, GPU state bit-exact, discrete e_out bit-exact, ct e_out + mu within tol"
        );
    }

    // ---- real-data extraction parity (Fe56) ----

    fn td(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("yamc")
            .join("tests")
            .join(name)
    }

    fn load(name: &str) -> Option<Nuclide> {
        if !td(name).exists() {
            eprintln!("Skipping: {name} not found");
            return None;
        }
        Some(
            yamc_nuclide::nuclide_loader::load_nuclide(td(name), &yamc_nuclide::LoadScope::full())
                .unwrap_or_else(|e| panic!("Failed to load {name}: {e}")),
        )
    }

    /// The packed S3 photon distribution buffers must reproduce the source
    /// nuclide data slot-for-slot: every discrete product carries
    /// kind=DISCRETE with the right line params; every continuous-tabular
    /// product's eout slot matches the source `ContinuousTabular`
    /// (energies, per-bin n_x, x, cdf, interp); and every product's packed
    /// angle row matches the source `AngleDistribution` (energies, n_mu, mu).
    /// Built on the nuclide's own grid (identity interpolation) so the only
    /// transform is the documented CDF renormalisation. Skips gracefully if
    /// `Fe56.arrow` is absent.
    #[test]
    fn fe56_photon_distribution_buffers_match_cpu() {
        use yamc_nuclide::particle_type::ParticleType;
        use yamc_nuclide::reaction_product::{
            AngleEnergyDistribution, EnergyDistribution, TabulatedProbability,
        };

        let Some(nuclide) = load("Fe56.arrow") else {
            return;
        };
        let temp_idx = nuclide.get_temp_idx("294").expect("294");
        let grid = &nuclide.fast_xs[temp_idx];
        let log_energy_grid: Vec<f64> = grid.energy.iter().map(|e| e.ln()).collect();
        let pp = extract_photon_production_xs(&[(&nuclide, 1.0)], "294", &log_energy_grid).unwrap();

        // Buffer-shape invariants. Tight CSR (issue #104): one ae-row base per
        // product; the per-row arrays are sized to the total ae-rows, and the
        // per-point arrays to the total mu-points.
        assert_eq!(pp.pa_n_energies.len(), pp.n_product);
        assert_eq!(pp.pa_ae_offset.len(), pp.n_product);
        let pa_ae_rows: usize = pp.pa_n_energies.iter().map(|&n| n as usize).sum();
        assert_eq!(pp.pa_energy_grid.len(), pa_ae_rows);
        assert_eq!(pp.pa_n_mu.len(), pa_ae_rows);
        assert_eq!(pp.pa_interp.len(), pa_ae_rows);
        assert_eq!(pp.pa_mu_offset.len(), pa_ae_rows);
        let pa_mu_points: usize = pp.pa_n_mu.iter().map(|&n| n as usize).sum();
        assert_eq!(pp.pa_mu.len(), pa_mu_points);
        assert_eq!(pp.pa_cdf.len(), pa_mu_points);
        assert_eq!(pp.pa_pdf.len(), pa_mu_points);
        assert_eq!(pp.ct_n_eout.len(), pp.n_continuous);
        assert_eq!(pp.ct_hist.len(), pp.n_continuous);
        // Tight CSR (issue #104): one ae-row base per continuous slot; the
        // per-row arrays are sized to the total ae-rows, and the per-point
        // arrays to the total x-points.
        assert_eq!(pp.ct_ae_offset.len(), pp.n_continuous);
        let ct_ae_rows: usize = pp.ct_n_eout.iter().map(|&n| n as usize).sum();
        assert_eq!(pp.ct_energy_grid.len(), ct_ae_rows);
        assert_eq!(pp.ct_n_x.len(), ct_ae_rows);
        assert_eq!(pp.ct_interp.len(), ct_ae_rows);
        assert_eq!(pp.ct_n_discrete.len(), ct_ae_rows);
        assert_eq!(pp.ct_x_offset.len(), ct_ae_rows);
        let ct_x_points: usize = pp.ct_n_x.iter().map(|&n| n as usize).sum();
        assert_eq!(pp.ct_x.len(), ct_x_points);
        assert_eq!(pp.ct_cdf.len(), ct_x_points);
        assert_eq!(pp.ct_p.len(), ct_x_points);

        // Walk the CPU products in the same order the extractor does.
        let reactions = &nuclide.reactions[temp_idx];
        let mut mts: Vec<i32> = reactions
            .iter()
            .filter_map(|(&mt, r)| {
                r.products
                    .iter()
                    .any(|p| p.is_particle_type(&ParticleType::Photon))
                    .then_some(mt)
            })
            .collect();
        mts.sort_unstable();

        let mut p = 0usize;
        let mut n_disc = 0usize;
        let mut n_ct = 0usize;
        for &mt in &mts {
            let rxn = reactions.get(&mt).unwrap();
            for product in &rxn.products {
                if !product.is_particle_type(&ParticleType::Photon) {
                    continue;
                }
                let AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, energy } =
                    &product.distribution[0]
                else {
                    panic!("Fe56 photon product {p} not UncorrelatedAngleEnergy");
                };

                // ---- angle row parity. Tight CSR (issue #104): this product's
                // rows start at the global ae-row `pa_ae_offset[p]`; each row's
                // mu points start at `pa_mu_offset[ae]`. Full resolution: every
                // incident energy and every mu point is kept (no subsampling),
                // so the only transform is the documented CDF renormalisation.
                let n_e = angle.energy.len();
                assert_eq!(pp.pa_n_energies[p], n_e as u32);
                let ae_base = pp.pa_ae_offset[p] as usize;
                for a in 0..n_e {
                    let ae = ae_base + a;
                    assert_eq!(pp.pa_energy_grid[ae], angle.energy[a]);
                    let tab = &angle.mu[a];
                    let m_in = tab.x.len();
                    assert_eq!(pp.pa_n_mu[ae], m_in as u32);
                    let off = pp.pa_mu_offset[ae] as usize;
                    let cdf_max = tab.c.last().copied().unwrap_or(0.0);
                    for j in 0..m_in {
                        assert_eq!(pp.pa_mu[off + j], tab.x[j], "product {p} mu[{j}]");
                        // CDF is renormalised to end at 1.0.
                        let want = if cdf_max > 0.0 {
                            tab.c[j] / cdf_max
                        } else {
                            j as f64 / (m_in - 1).max(1) as f64
                        };
                        assert!(
                            (pp.pa_cdf[off + j] - want).abs() <= 1e-12 + 1e-12 * want.abs(),
                            "product {p} cdf[{j}]: {} != {want}",
                            pp.pa_cdf[off + j]
                        );
                    }
                }

                // ---- eout parity ----
                match energy {
                    Some(EnergyDistribution::DiscretePhoton {
                        primary_flag,
                        energy: line_e,
                        atomic_weight_ratio,
                    }) => {
                        assert_eq!(pp.prod_eout_kind[p], PHOTON_EOUT_KIND_DISCRETE);
                        assert_eq!(pp.prod_line_energy[p], *line_e);
                        assert_eq!(pp.prod_primary_flag[p], *primary_flag);
                        assert_eq!(pp.prod_awr[p], *atomic_weight_ratio);
                        assert_eq!(pp.prod_dist_slot[p], PHOTON_DIST_SLOT_NONE);
                        n_disc += 1;
                    }
                    Some(EnergyDistribution::ContinuousTabular {
                        energy: ct_energy,
                        energy_out,
                        ..
                    }) => {
                        assert_eq!(pp.prod_eout_kind[p], PHOTON_EOUT_KIND_CONTINUOUS_TABULAR);
                        let slot = pp.prod_dist_slot[p];
                        assert_ne!(slot, PHOTON_DIST_SLOT_NONE);
                        let s = slot as usize;
                        let n_ae = ct_energy.len();
                        assert!(
                            n_ae <= CT_AE,
                            "product {p} ct has {n_ae} E_in > CT_AE {CT_AE}"
                        );
                        assert_eq!(pp.ct_n_eout[s], n_ae as u32);
                        // Tight CSR (issue #104): the slot's rows start at the
                        // global ae-row `ct_ae_offset[s]`; each row's x-points
                        // start at `ct_x_offset[eg]`.
                        let eg_base = pp.ct_ae_offset[s] as usize;
                        for a in 0..n_ae {
                            let eg = eg_base + a;
                            assert_eq!(pp.ct_energy_grid[eg], ct_energy[a]);
                            let TabulatedProbability::Tabulated { x, c, .. } = &energy_out[a];
                            let m_in = x.len();
                            assert!(
                                m_in <= CT_X,
                                "product {p} ct E_in {a} has {m_in} x > CT_X {CT_X}"
                            );
                            assert_eq!(pp.ct_n_x[eg], m_in as u32);
                            let xoff = pp.ct_x_offset[eg] as usize;
                            let cdf_max = c.last().copied().unwrap_or(0.0);
                            for j in 0..m_in {
                                assert_eq!(pp.ct_x[xoff + j], x[j], "product {p} ct x[{j}]");
                                let want = if cdf_max > 0.0 {
                                    c[j] / cdf_max
                                } else {
                                    j as f64 / (m_in - 1).max(1) as f64
                                };
                                assert!(
                                    (pp.ct_cdf[xoff + j] - want).abs()
                                        <= 1e-12 + 1e-12 * want.abs(),
                                    "product {p} ct cdf[{j}]: {} != {want}",
                                    pp.ct_cdf[xoff + j]
                                );
                            }
                        }
                        n_ct += 1;
                    }
                    other => panic!(
                        "Fe56 photon product {p} unexpected energy variant: {:?}",
                        other.as_ref().map(|e| e.distribution_name())
                    ),
                }
                p += 1;
            }
        }
        assert_eq!(p, pp.n_product);
        assert!(pp
            .prod_eout_kind
            .iter()
            .all(|&k| k != PHOTON_EOUT_KIND_NONE));
        println!(
            "Fe56 S3 buffers: n_product={}, discrete={n_disc}, continuous={n_ct} (n_continuous={}); angle + ct slots match source",
            pp.n_product, pp.n_continuous
        );
    }
}
