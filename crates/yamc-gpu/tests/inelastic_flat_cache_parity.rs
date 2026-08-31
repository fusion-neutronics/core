//! Bit-identity of the CPU transport's per-collision inelastic flat cache
//! against the GPU host extraction (issue #111, stream unification).
//!
//! `yamc_physics::gpu::flat::inelastic_flat::InelasticFlatCache` gives the CPU
//! transport ONE reaction's flat kinematics arrays (single slot, zero-based
//! CSR); `yamc_gpu::neutron::xs::extract_per_nuclide_inelastic` builds the same
//! arrays for every MT slot of a nuclide concatenated into the GPU's per-slab
//! buffers. Both call the same `eout_extract` layer, so the cached arrays MUST
//! be byte-for-byte the slot's rows of the GPU buffers, otherwise a CPU
//! collision routed through the cache would sample different data than the
//! kernel and the matched-stream harness would diverge.
//!
//! Two levels of check per (nuclide, MT):
//!   1. every cached array equals the GPU slab's rows for that slot, compared
//!      by `f64::to_bits()` / `u32` equality (no tolerance), and every cached
//!      CSR offset equals the GPU offset minus the slot's base;
//!   2. driving `sample_inelastic_kinematics` from the cache (slab 0, slot 0)
//!      and from the GPU slab arrays (slab 0, slot `s`) with the same seed
//!      returns bit-identical `(mu, e_out, ok)` AND leaves the PCG state
//!      identical, which pins the field-name mapping the cache's
//!      `sample_kinematics` forwarding relies on.
//!
//! Skips cleanly when the Arrow fixtures are absent.

use std::path::{Path, PathBuf};

use yamc_gpu::neutron::xs::{
    extract_per_nuclide_inelastic, PerNuclideInelastic, EOUT_KIND_LEVEL_INELASTIC,
    MT_INELASTIC_COUNT, MT_SLOTS,
};
use yamc_nuclide::nuclide::Nuclide;
use yamc_physics::gpu::flat::inelastic_dispatch::sample_inelastic_kinematics;
use yamc_physics::gpu::flat::inelastic_flat::{InelasticFlat, InelasticFlatCache};
use yamc_rng::expand_seed;

/// Test arrow datasets live in `crates/yamc/tests/`.
fn td(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
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
            .unwrap_or_else(|e| panic!("failed to load {name}: {e:?}")),
    )
}

/// Zero-based CSR starts, the same running sum `translate.rs`'s
/// `push_*_csr_offsets` build (the slab base is 0 for a single nuclide).
fn csr(counts: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(counts.len());
    let mut acc = 0u32;
    for &n in counts {
        out.push(acc);
        acc += n;
    }
    out
}

fn sum(counts: &[u32]) -> usize {
    counts.iter().map(|&n| n as usize).sum()
}

fn assert_bits_f64(cache: &[f64], gpu: &[f64], label: &str) {
    assert_eq!(
        cache.len(),
        gpu.len(),
        "{label}: length {} (cache) vs {} (gpu)",
        cache.len(),
        gpu.len()
    );
    for (i, (a, b)) in cache.iter().zip(gpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}[{i}]: cache {a:.17e} vs gpu {b:.17e}"
        );
    }
}

fn assert_eq_u32(cache: &[u32], gpu: &[u32], label: &str) {
    assert_eq!(
        cache.len(),
        gpu.len(),
        "{label}: length {} (cache) vs {} (gpu)",
        cache.len(),
        gpu.len()
    );
    for (i, (a, b)) in cache.iter().zip(gpu).enumerate() {
        assert_eq!(a, b, "{label}[{i}]: cache {a} vs gpu {b}");
    }
}

/// The CSR start of row / point `idx`, or 0 when the family is empty for this
/// nuclide (the slot then spans zero entries, so any in-range base works).
fn base_at(offsets: &[u32], idx: usize) -> u32 {
    offsets.get(idx).copied().unwrap_or(0)
}

/// `offsets` rebased to the slot's own first row / point, i.e. what a
/// single-slot bundle stores.
fn rebased(offsets: &[u32], base: u32) -> Vec<u32> {
    offsets.iter().map(|&o| o - base).collect()
}

/// The GPU slab's CSR offset arrays, rebuilt exactly as `translate.rs` does
/// for a one-nuclide material (all bases 0).
struct SlabCsr {
    angle_ae: Vec<u32>,
    angle_mu: Vec<u32>,
    eout_ae: Vec<u32>,
    eout_x: Vec<u32>,
    corr_ae: Vec<u32>,
    corr_x: Vec<u32>,
    corr_mu: Vec<u32>,
    km_ae: Vec<u32>,
    km_x: Vec<u32>,
    evap_ae: Vec<u32>,
    evap_theta: Vec<u32>,
    maxwell_ae: Vec<u32>,
    watt_ae: Vec<u32>,
}

impl SlabCsr {
    fn new(p: &PerNuclideInelastic) -> Self {
        let evap_theta_counts: Vec<u32> = p
            .evap_n_energies
            .iter()
            .zip(&p.evap_n_components)
            .map(|(&n_e, &n_c)| n_e * n_c)
            .collect();
        Self {
            angle_ae: csr(&p.angle_n_energies),
            angle_mu: csr(&p.angle_n_mu),
            eout_ae: csr(&p.eout_n_energies),
            eout_x: csr(&p.eout_n_x),
            corr_ae: csr(&p.corr_n_energies),
            corr_x: csr(&p.corr_n_x),
            corr_mu: csr(&p.corr_n_mu),
            km_ae: csr(&p.km_n_energies),
            km_x: csr(&p.km_n_x),
            evap_ae: csr(&p.evap_n_energies),
            evap_theta: csr(&evap_theta_counts),
            maxwell_ae: csr(&p.maxwell_n_energies),
            watt_ae: csr(&p.watt_n_energies),
        }
    }
}

/// Level 1: every cached array is the GPU slab's rows for slot `s`.
fn assert_arrays_match(flat: &InelasticFlat, p: &PerNuclideInelastic, c: &SlabCsr, s: usize) {
    let mt = MT_SLOTS[s];
    let tag = |name: &str| format!("MT {mt} {name}");

    // Slice B: angular table. Rows `[ae, ae + n_ae)`, mu points reached
    // through the per-row offsets.
    let ae = c.angle_ae[s] as usize;
    let n_ae = p.angle_n_energies[s] as usize;
    let mu_base = base_at(&c.angle_mu, ae);
    let n_mu = sum(&p.angle_n_mu[ae..ae + n_ae]);
    let mu = mu_base as usize;
    assert_eq_u32(
        &flat.angle_n_energies,
        &p.angle_n_energies[s..s + 1],
        &tag("angle_n_energies"),
    );
    assert_eq_u32(&flat.angle_ae_offset, &[0], &tag("angle_ae_offset"));
    assert_bits_f64(
        &flat.angle_energy_grid,
        &p.angle_energy_grid[ae..ae + n_ae],
        &tag("angle_energy_grid"),
    );
    assert_eq_u32(
        &flat.angle_n_mu,
        &p.angle_n_mu[ae..ae + n_ae],
        &tag("angle_n_mu"),
    );
    assert_eq_u32(
        &flat.angle_mu_offset,
        &rebased(&c.angle_mu[ae..ae + n_ae], mu_base),
        &tag("angle_mu_offset"),
    );
    assert_bits_f64(&flat.angle_mu, &p.angle_mu[mu..mu + n_mu], &tag("angle_mu"));
    assert_bits_f64(
        &flat.angle_cdf,
        &p.angle_cdf[mu..mu + n_mu],
        &tag("angle_cdf"),
    );
    assert_bits_f64(
        &flat.angle_pdf,
        &p.angle_pdf[mu..mu + n_mu],
        &tag("angle_pdf"),
    );
    assert_eq_u32(
        &flat.angle_interp,
        &p.angle_interp[ae..ae + n_ae],
        &tag("angle_interp"),
    );
    assert_eq_u32(
        &flat.scatter_in_cm_per_mt,
        &p.scatter_in_cm[s..s + 1],
        &tag("scatter_in_cm"),
    );

    // Slice C: outgoing-energy table.
    let e_ae = c.eout_ae[s] as usize;
    let e_n = p.eout_n_energies[s] as usize;
    let e_x_base = base_at(&c.eout_x, e_ae);
    let e_nx = sum(&p.eout_n_x[e_ae..e_ae + e_n]);
    let e_x = e_x_base as usize;
    assert_eq_u32(&flat.eout_kind, &p.eout_kind[s..s + 1], &tag("eout_kind"));
    assert_eq_u32(
        &flat.eout_n_energies,
        &p.eout_n_energies[s..s + 1],
        &tag("eout_n_energies"),
    );
    assert_eq_u32(
        &flat.eout_histogram_interp,
        &p.eout_histogram_interp[s..s + 1],
        &tag("eout_histogram_interp"),
    );
    assert_eq_u32(&flat.eout_ae_offset, &[0], &tag("eout_ae_offset"));
    assert_bits_f64(
        &flat.eout_energy_grid,
        &p.eout_energy_grid[e_ae..e_ae + e_n],
        &tag("eout_energy_grid"),
    );
    assert_eq_u32(
        &flat.eout_n_x,
        &p.eout_n_x[e_ae..e_ae + e_n],
        &tag("eout_n_x"),
    );
    assert_eq_u32(
        &flat.eout_x_offset,
        &rebased(&c.eout_x[e_ae..e_ae + e_n], e_x_base),
        &tag("eout_x_offset"),
    );
    assert_bits_f64(&flat.eout_x, &p.eout_x[e_x..e_x + e_nx], &tag("eout_x"));
    assert_bits_f64(&flat.eout_p, &p.eout_p[e_x..e_x + e_nx], &tag("eout_p"));
    assert_bits_f64(
        &flat.eout_cdf,
        &p.eout_cdf[e_x..e_x + e_nx],
        &tag("eout_cdf"),
    );
    assert_eq_u32(
        &flat.eout_interp,
        &p.eout_interp[e_ae..e_ae + e_n],
        &tag("eout_interp"),
    );
    assert_eq_u32(
        &flat.eout_n_discrete,
        &p.eout_n_discrete[e_ae..e_ae + e_n],
        &tag("eout_n_discrete"),
    );

    // Slice D: correlated angle-energy (three CSR levels).
    let c_ae = c.corr_ae[s] as usize;
    let c_n = p.corr_n_energies[s] as usize;
    let c_x_base = base_at(&c.corr_x, c_ae);
    let c_nx = sum(&p.corr_n_x[c_ae..c_ae + c_n]);
    let c_x = c_x_base as usize;
    let c_mu_base = base_at(&c.corr_mu, c_x);
    let c_nmu = sum(&p.corr_n_mu[c_x..c_x + c_nx]);
    let c_mu = c_mu_base as usize;
    assert_eq_u32(
        &flat.corr_n_energies,
        &p.corr_n_energies[s..s + 1],
        &tag("corr_n_energies"),
    );
    assert_eq_u32(
        &flat.corr_n_components,
        &p.corr_n_components[s..s + 1],
        &tag("corr_n_components"),
    );
    assert_eq_u32(&flat.corr_ae_offset, &[0], &tag("corr_ae_offset"));
    assert_bits_f64(
        &flat.corr_energy_grid,
        &p.corr_energy_grid[c_ae..c_ae + c_n],
        &tag("corr_energy_grid"),
    );
    assert_eq_u32(
        &flat.corr_n_x,
        &p.corr_n_x[c_ae..c_ae + c_n],
        &tag("corr_n_x"),
    );
    assert_eq_u32(
        &flat.corr_x_offset,
        &rebased(&c.corr_x[c_ae..c_ae + c_n], c_x_base),
        &tag("corr_x_offset"),
    );
    assert_bits_f64(&flat.corr_x, &p.corr_x[c_x..c_x + c_nx], &tag("corr_x"));
    assert_bits_f64(
        &flat.corr_cdf,
        &p.corr_cdf[c_x..c_x + c_nx],
        &tag("corr_cdf"),
    );
    assert_bits_f64(&flat.corr_p, &p.corr_p[c_x..c_x + c_nx], &tag("corr_p"));
    assert_eq_u32(
        &flat.corr_interp,
        &p.corr_interp[c_ae..c_ae + c_n],
        &tag("corr_interp"),
    );
    assert_eq_u32(
        &flat.corr_n_discrete,
        &p.corr_n_discrete[c_ae..c_ae + c_n],
        &tag("corr_n_discrete"),
    );
    assert_eq_u32(
        &flat.corr_n_mu,
        &p.corr_n_mu[c_x..c_x + c_nx],
        &tag("corr_n_mu"),
    );
    assert_eq_u32(
        &flat.corr_mu_offset,
        &rebased(&c.corr_mu[c_x..c_x + c_nx], c_mu_base),
        &tag("corr_mu_offset"),
    );
    assert_bits_f64(
        &flat.corr_mu,
        &p.corr_mu[c_mu..c_mu + c_nmu],
        &tag("corr_mu"),
    );
    assert_bits_f64(
        &flat.corr_mu_cdf,
        &p.corr_mu_cdf[c_mu..c_mu + c_nmu],
        &tag("corr_mu_cdf"),
    );
    assert_bits_f64(
        &flat.corr_mu_pdf,
        &p.corr_mu_pdf[c_mu..c_mu + c_nmu],
        &tag("corr_mu_pdf"),
    );
    assert_eq_u32(
        &flat.corr_mu_interp,
        &p.corr_mu_interp[c_x..c_x + c_nx],
        &tag("corr_mu_interp"),
    );

    // Slice E: Kalbach-Mann.
    let k_ae = c.km_ae[s] as usize;
    let k_n = p.km_n_energies[s] as usize;
    let k_x_base = base_at(&c.km_x, k_ae);
    let k_nx = sum(&p.km_n_x[k_ae..k_ae + k_n]);
    let k_x = k_x_base as usize;
    assert_eq_u32(
        &flat.km_n_energies,
        &p.km_n_energies[s..s + 1],
        &tag("km_n_energies"),
    );
    assert_eq_u32(&flat.km_ae_offset, &[0], &tag("km_ae_offset"));
    assert_bits_f64(
        &flat.km_energy_grid,
        &p.km_energy_grid[k_ae..k_ae + k_n],
        &tag("km_energy_grid"),
    );
    assert_eq_u32(
        &flat.km_interp,
        &p.km_interp[k_ae..k_ae + k_n],
        &tag("km_interp"),
    );
    assert_eq_u32(
        &flat.km_n_discrete,
        &p.km_n_discrete[k_ae..k_ae + k_n],
        &tag("km_n_discrete"),
    );
    assert_eq_u32(&flat.km_n_x, &p.km_n_x[k_ae..k_ae + k_n], &tag("km_n_x"));
    assert_eq_u32(
        &flat.km_x_offset,
        &rebased(&c.km_x[k_ae..k_ae + k_n], k_x_base),
        &tag("km_x_offset"),
    );
    assert_bits_f64(&flat.km_x, &p.km_x[k_x..k_x + k_nx], &tag("km_x"));
    assert_bits_f64(&flat.km_p, &p.km_p[k_x..k_x + k_nx], &tag("km_p"));
    assert_bits_f64(&flat.km_c, &p.km_c[k_x..k_x + k_nx], &tag("km_c"));
    assert_bits_f64(&flat.km_r, &p.km_r[k_x..k_x + k_nx], &tag("km_r"));
    assert_bits_f64(&flat.km_a, &p.km_a[k_x..k_x + k_nx], &tag("km_a"));

    // Slice F: Evaporation (component-major theta) / n-body / Maxwell / Watt.
    let v_ae = c.evap_ae[s] as usize;
    let v_n = p.evap_n_energies[s] as usize;
    let v_theta = c.evap_theta[s] as usize;
    let v_ntheta = v_n * p.evap_n_components[s] as usize;
    assert_eq_u32(
        &flat.evap_n_energies,
        &p.evap_n_energies[s..s + 1],
        &tag("evap_n_energies"),
    );
    assert_eq_u32(
        &flat.evap_n_components,
        &p.evap_n_components[s..s + 1],
        &tag("evap_n_components"),
    );
    assert_eq_u32(&flat.evap_ae_offset, &[0], &tag("evap_ae_offset"));
    assert_eq_u32(&flat.evap_theta_offset, &[0], &tag("evap_theta_offset"));
    assert_bits_f64(
        &flat.evap_energy_grid,
        &p.evap_energy_grid[v_ae..v_ae + v_n],
        &tag("evap_energy_grid"),
    );
    assert_bits_f64(
        &flat.evap_theta,
        &p.evap_theta[v_theta..v_theta + v_ntheta],
        &tag("evap_theta"),
    );
    assert_bits_f64(&flat.evap_u, &p.evap_u[v_ae..v_ae + v_n], &tag("evap_u"));

    assert_eq_u32(
        &flat.nbps_n_bodies,
        &p.nbps_n_bodies[s..s + 1],
        &tag("nbps_n_bodies"),
    );
    assert_bits_f64(
        &flat.nbps_total_mass,
        &p.nbps_total_mass[s..s + 1],
        &tag("nbps_total_mass"),
    );

    let m_ae = c.maxwell_ae[s] as usize;
    let m_n = p.maxwell_n_energies[s] as usize;
    assert_eq_u32(
        &flat.maxwell_n_energies,
        &p.maxwell_n_energies[s..s + 1],
        &tag("maxwell_n_energies"),
    );
    assert_eq_u32(&flat.maxwell_ae_offset, &[0], &tag("maxwell_ae_offset"));
    assert_bits_f64(
        &flat.maxwell_energy_grid,
        &p.maxwell_energy_grid[m_ae..m_ae + m_n],
        &tag("maxwell_energy_grid"),
    );
    assert_bits_f64(
        &flat.maxwell_theta,
        &p.maxwell_theta[m_ae..m_ae + m_n],
        &tag("maxwell_theta"),
    );
    assert_bits_f64(&flat.maxwell_u, &p.maxwell_u[s..s + 1], &tag("maxwell_u"));

    let w_ae = c.watt_ae[s] as usize;
    let w_n = p.watt_n_energies[s] as usize;
    assert_eq_u32(
        &flat.watt_n_energies,
        &p.watt_n_energies[s..s + 1],
        &tag("watt_n_energies"),
    );
    assert_eq_u32(&flat.watt_ae_offset, &[0], &tag("watt_ae_offset"));
    assert_bits_f64(
        &flat.watt_energy_grid,
        &p.watt_energy_grid[w_ae..w_ae + w_n],
        &tag("watt_energy_grid"),
    );
    assert_bits_f64(&flat.watt_a, &p.watt_a[w_ae..w_ae + w_n], &tag("watt_a"));
    assert_bits_f64(&flat.watt_b, &p.watt_b[w_ae..w_ae + w_n], &tag("watt_b"));
    assert_bits_f64(&flat.watt_u, &p.watt_u[s..s + 1], &tag("watt_u"));

    // Q value: the cache carries the reaction's own Q (what the CPU transport
    // has always used, including `scatter_inelastic_level`), while the GPU
    // stores a peak-cross-section-weighted average over the material's
    // nuclides. For ONE nuclide that average is `(w * q) / w`, which is `q`
    // only up to a single rounding, so this is a near-equality check rather
    // than a bit check. Skipped where the slot has no nonzero cross section on
    // the extraction grid (`w == 0`, the average is left at 0).
    if p.permt_n_stored[s] > 0 {
        let gpu_q = p.q_inelastic_per_mt[s];
        assert!(
            (flat.q_value - gpu_q).abs() <= 4.0 * f64::EPSILON * flat.q_value.abs(),
            "MT {mt} q_value: cache {:.17e} vs gpu {gpu_q:.17e}",
            flat.q_value
        );
    }
}

/// Level 2: the dispatcher driven from the cache (slot 0) and from the GPU
/// slab (slot `s`) is bit-identical in result AND in RNG state consumed.
#[allow(clippy::too_many_arguments)]
fn assert_sampling_matches(
    flat: &InelasticFlat,
    p: &PerNuclideInelastic,
    c: &SlabCsr,
    s: usize,
    target_mass: f64,
) {
    let mt = MT_SLOTS[s];
    // The Q used for the closed-form CM energy is supplied by the caller on
    // both sides, so this compares the ARRAY mapping only (the Q value itself
    // is asserted in `assert_arrays_match`).
    let q = flat.q_value;
    let threshold = (target_mass + 1.0) / target_mass * q.abs();
    let mass_ratio = (target_mass / (target_mass + 1.0)).powi(2);
    for (i, &e_in) in [1.0e5_f64, 1.0e6, 5.0e6, 1.4e7].iter().enumerate() {
        let e_cm_closed = mass_ratio * (e_in - threshold);
        let xi3 = 0.125 + 0.2 * i as f64;
        let mut state_cache = expand_seed(0x5EED_0000 + i as u32);
        let mut state_gpu = state_cache;

        let from_cache =
            flat.sample_kinematics(e_in, target_mass, e_cm_closed, xi3, &mut state_cache);
        let from_gpu = sample_inelastic_kinematics(
            e_in,
            target_mass,
            e_cm_closed,
            xi3,
            0,
            s,
            &mut state_gpu,
            &p.angle_n_energies,
            &c.angle_ae,
            &p.angle_energy_grid,
            &p.angle_n_mu,
            &c.angle_mu,
            &p.angle_mu,
            &p.angle_cdf,
            &p.angle_pdf,
            &p.angle_interp,
            &p.eout_kind,
            &p.eout_n_energies,
            &c.eout_ae,
            &p.eout_energy_grid,
            &p.eout_n_x,
            &c.eout_x,
            &p.eout_x,
            &p.eout_cdf,
            &p.eout_histogram_interp,
            &p.eout_p,
            &p.eout_interp,
            &p.eout_n_discrete,
            &p.corr_n_energies,
            &p.corr_n_components,
            &c.corr_ae,
            &p.corr_energy_grid,
            &p.corr_n_x,
            &c.corr_x,
            &p.corr_x,
            &p.corr_cdf,
            &p.corr_p,
            &p.corr_interp,
            &p.corr_n_discrete,
            &p.corr_n_mu,
            &c.corr_mu,
            &p.corr_mu,
            &p.corr_mu_cdf,
            &p.corr_mu_pdf,
            &p.corr_mu_interp,
            &p.scatter_in_cm,
            &p.km_n_energies,
            &c.km_ae,
            &p.km_energy_grid,
            &p.km_interp,
            &p.km_n_discrete,
            &p.km_n_x,
            &c.km_x,
            &p.km_x,
            &p.km_p,
            &p.km_c,
            &p.km_r,
            &p.km_a,
            &p.evap_n_energies,
            &p.evap_n_components,
            &c.evap_ae,
            &c.evap_theta,
            &p.evap_energy_grid,
            &p.evap_theta,
            &p.evap_u,
            &p.nbps_n_bodies,
            &p.nbps_total_mass,
            &p.maxwell_n_energies,
            &c.maxwell_ae,
            &p.maxwell_energy_grid,
            &p.maxwell_theta,
            &p.maxwell_u,
            &p.watt_n_energies,
            &c.watt_ae,
            &p.watt_energy_grid,
            &p.watt_a,
            &p.watt_b,
            &p.watt_u,
            q,
        );
        assert_eq!(
            from_cache.0.to_bits(),
            from_gpu.0.to_bits(),
            "MT {mt} @ {e_in:.3e} eV: mu {:.17e} (cache) vs {:.17e} (gpu)",
            from_cache.0,
            from_gpu.0
        );
        assert_eq!(
            from_cache.1.to_bits(),
            from_gpu.1.to_bits(),
            "MT {mt} @ {e_in:.3e} eV: e_out {:.17e} (cache) vs {:.17e} (gpu)",
            from_cache.1,
            from_gpu.1
        );
        assert_eq!(from_cache.2, from_gpu.2, "MT {mt} @ {e_in:.3e} eV: ok flag");
        assert_eq!(
            state_cache, state_gpu,
            "MT {mt} @ {e_in:.3e} eV: PCG state diverged (different draw count)"
        );
    }
}

/// Extract `fixture`'s nuclide through both paths and compare every MT slot.
/// Returns the `eout_kind -> MTs` map that was exercised (`None` when the
/// fixture is absent), so callers can assert the comparison was not vacuous.
fn check_fixture(fixture: &str) -> Option<std::collections::BTreeMap<u32, Vec<i32>>> {
    let nuclide = load(fixture)?;
    let temperature = nuclide
        .loaded_temperatures
        .first()
        .expect("fixture must carry at least one temperature")
        .clone();
    let temp_idx = nuclide
        .get_temp_idx(&temperature)
        .expect("temperature index");
    let target_mass = nuclide
        .atomic_weight_ratio
        .expect("fixture must carry an atomic weight ratio");

    // Log grid spanning thermal to 20 MeV. Only the XS / Q columns depend on
    // it; the distribution buffers under test do not.
    let n_grid = 512usize;
    let (lo, hi) = (1.0e-5_f64.ln(), 2.0e7_f64.ln());
    let grid: Vec<f64> = (0..n_grid)
        .map(|i| (lo + (hi - lo) * i as f64 / (n_grid - 1) as f64).exp())
        .collect();

    let pool = extract_per_nuclide_inelastic(&[(&nuclide, 1.0)], &temperature, &grid, &grid)
        .expect("per-nuclide inelastic extraction");
    assert_eq!(pool.n_nuclides, 1);
    assert_eq!(pool.eout_kind.len(), MT_INELASTIC_COUNT);
    let slab = SlabCsr::new(&pool);

    let cache = InelasticFlatCache::default();
    let reactions = &nuclide.reactions[temp_idx];
    let mut n_slots = 0usize;
    let mut kinds: std::collections::BTreeMap<u32, Vec<i32>> = std::collections::BTreeMap::new();
    for (s, &mt) in MT_SLOTS.iter().enumerate() {
        let Some(reaction) = reactions.get(&mt) else {
            continue;
        };
        let flat = cache.get_or_build(&nuclide, reaction);
        assert_arrays_match(&flat, &pool, &slab, s);
        assert_sampling_matches(&flat, &pool, &slab, s, target_mass);
        n_slots += 1;
        kinds.entry(pool.eout_kind[s]).or_default().push(mt);
        // Second lookup must hand back the SAME cached allocation, not a
        // rebuild (this is what makes the per-collision cost a map hit).
        let again = cache.get_or_build(&nuclide, reaction);
        assert!(
            std::sync::Arc::ptr_eq(&flat, &again),
            "MT {mt}: cache miss on second lookup"
        );
    }
    eprintln!("{fixture}: {n_slots} MT slots compared");
    for (kind, mts) in &kinds {
        eprintln!("  eout_kind {kind}: {} MTs {mts:?}", mts.len());
    }
    assert!(
        n_slots > 0,
        "{fixture}: no inelastic MT slots compared (vacuous test)"
    );
    assert_eq!(cache.len(), n_slots, "one cache entry per (nuclide, MT)");
    Some(kinds)
}

/// A non-`LevelInelastic` law must be among the ones compared, otherwise the
/// fixture only exercised the closed-form-Q path and the continuum arrays went
/// unchecked.
fn assert_continuum_covered(kinds: &std::collections::BTreeMap<u32, Vec<i32>>, fixture: &str) {
    assert!(
        kinds.keys().any(|&k| k != EOUT_KIND_LEVEL_INELASTIC),
        "{fixture}: only level-inelastic slots compared (vacuous test)"
    );
}

#[test]
fn fe56_cache_matches_gpu_extraction() {
    // Fe56: 39 discrete levels (closed-form Q) plus correlated MT 91 / 16 / 5.
    let Some(kinds) = check_fixture("Fe56.arrow") else {
        return;
    };
    assert_continuum_covered(&kinds, "Fe56.arrow");
}

#[test]
fn li6_cache_matches_gpu_extraction() {
    // Li6: correlated MT 32 and n-body phase space MT 41.
    let Some(kinds) = check_fixture("Li6.arrow") else {
        return;
    };
    assert_continuum_covered(&kinds, "Li6.arrow");
}

/// The cache key is (nuclide identity, MT), not MT alone: two nuclides that
/// both carry MT 51 must get their own bundle, otherwise a multi-isotope
/// material would sample one isotope's levels for another.
#[test]
fn cache_key_separates_nuclides_carrying_the_same_mt() {
    let (Some(fe56), Some(fe54)) = (load("Fe56.arrow"), load("Fe54.arrow")) else {
        return;
    };
    let level = |n: &Nuclide| {
        let temperature = n.loaded_temperatures.first().expect("temperature").clone();
        let idx = n.get_temp_idx(&temperature).expect("temperature index");
        n.reactions[idx].get(&51).expect("MT 51").clone()
    };
    let (r56, r54) = (level(&fe56), level(&fe54));

    let cache = InelasticFlatCache::default();
    let f56 = cache.get_or_build(&fe56, &r56);
    let f54 = cache.get_or_build(&fe54, &r54);
    assert_eq!(cache.len(), 2, "same MT on two nuclides must not collide");
    assert!(!std::sync::Arc::ptr_eq(&f56, &f54));
    assert!(
        f56.q_value != f54.q_value || f56.angle_energy_grid != f54.angle_energy_grid,
        "Fe56 and Fe54 MT 51 should differ; the key may be collapsing them"
    );
    // Each still resolves to its own nuclide's data on a repeat lookup.
    assert!(std::sync::Arc::ptr_eq(
        &f56,
        &cache.get_or_build(&fe56, &r56)
    ));
    assert!(std::sync::Arc::ptr_eq(
        &f54,
        &cache.get_or_build(&fe54, &r54)
    ));
}

/// Widen the law coverage past what Fe56 / Li6 carry: the remaining neutron
/// fixtures bring the evaporation, Kalbach-Mann and continuous-tabular
/// extractors under the same bit-identity contract. Each fixture is skipped if
/// absent; the union must cover more than the two laws above.
#[test]
fn other_fixtures_cache_matches_gpu_extraction() {
    let mut union: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for fixture in [
        "H2.arrow",
        "Be9.arrow",
        "C12.arrow",
        "O16.arrow",
        "Al27.arrow",
        "Cr52.arrow",
        "Fe54.arrow",
        "Fe57.arrow",
        "Fe58.arrow",
        "Co58.arrow",
        "Li7.arrow",
        "Pb208.arrow",
    ] {
        if let Some(kinds) = check_fixture(fixture) {
            union.extend(kinds.keys().copied());
        }
    }
    if union.is_empty() {
        return; // no fixtures present
    }
    eprintln!("union of eout kinds across the other fixtures: {union:?}");
    assert!(
        union.len() > 2,
        "expected several E_out laws across the fixture set, saw {union:?}"
    );
}
