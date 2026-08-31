//! On-device discrete photon-line energy sampling.
//!
//! A discrete photon spectrum is a set of fixed line energies, each emitted
//! with probability proportional to its intensity. This is how D1S decay
//! photons are emitted (a parent nuclide's gamma lines) and how ENDF
//! discrete-photon production spectra are represented. At emission the
//! transport must pick line `i` with probability
//! `intensity_i / Sum_j intensity_j` and return `energy_i`.
//!
//! This slice (phase 3a of the coupled neutron->photon / D1S work) proves
//! the **line-selection walk** in isolation, the photon-production analogue
//! of the neutron [`crate::neutron::nuclide_select`] walk. Given a
//! spectrum's per-line energies + intensities and a PCG seed, it picks a
//! line. The walk is pure integer-PCG + f64 add/mul (no FMA, no
//! transcendentals), so the GPU kernel and the
//! [`sample_discrete_photon_energy_cpu`] 32-bit-PCG twin agree
//! **bit-for-bit** -- `gpu_discrete_spectrum_matches_cpu` pins that on real
//! hardware.
//!
//! Faithful to `yamc_physics`'s `sample_discrete_energy`: cumulative sum of
//! intensities with the `xi <= cumulative` comparison (note: `<=`, unlike
//! the neutron nuclide walk's strict `<`), and a fall-back to the last line.
//! The unit draw uses the kernel's 32-bit PCG output function (the parity
//! reference is the 32-bit twin, not the 64-bit `rand`-crate production
//! sampler).
//!
//! Deferred to later slices: continuous photon energy laws (P3b), the
//! reaction/product selection walk and direction sampling (P3c), and wiring
//! into a production photon-production kernel that appends to the device
//! particle bank (P4 / D1S P5).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// CPU twin of the GPU discrete-spectrum walk. `energies`/`intensities` are
/// one spectrum's per-line energies and (unnormalised) intensities; `seed`
/// is the PCG state. Returns the sampled line energy. Uses the kernel's
/// 32-bit PCG output function (single draw, no state advance) so it matches
/// the `#[cube]` kernel bit-for-bit. Mirrors `sample_discrete_energy`:
/// empty / non-positive total returns `0.0`, the walk uses `xi <= cumulative`,
/// and the fall-back is the last line.
pub fn sample_discrete_photon_energy_cpu(energies: &[f64], intensities: &[f64], seed: u32) -> f64 {
    let mut total = 0.0_f64;
    for &v in intensities {
        total += v;
    }
    if total <= 0.0 || energies.is_empty() {
        return 0.0;
    }
    // One PCG-32 draw: uniform in (0, 1], scaled by the intensity total.
    let s = crate::common::rng::expand_seed(seed);
    let r = pcg_out(s);
    let xi = (r as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;

    let mut cumulative = 0.0_f64;
    for (i, &intensity) in intensities.iter().enumerate() {
        cumulative += intensity;
        if xi <= cumulative {
            return energies[i];
        }
    }
    // Floating-point edge case: xi == total picks the last line.
    energies[energies.len() - 1]
}

/// Test/validation kernel: one spectrum sample per thread. `energies_flat`
/// and `intensities_flat` pack each sample's per-line data contiguously;
/// `offsets`/`counts` slice them per sample; `seeds` is the PCG state per
/// sample. Writes the sampled line energy. The walk is inlined here -- it is
/// the same block production code will inline at the emission site.
#[cube(launch_unchecked)]
fn discrete_spectrum_kernel(
    energies_flat: &[f64],
    intensities_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
    out: &mut [f64],
) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let base = offsets[ABSOLUTE_POS] as usize;
    let count = counts[ABSOLUTE_POS];

    let mut total = 0.0_f64;
    let mut i = 0u32;
    while i < count {
        total += intensities_flat[base + i as usize];
        i += 1u32;
    }

    if total <= 0.0 || count == 0u32 {
        out[ABSOLUTE_POS] = 0.0;
    } else {
        // One PCG-32 draw from the seed (output function only; no advance).
        let s = expand_seed(seeds[ABSOLUTE_POS]);
        let r = pcg_out(s);
        let xi = (r as f64 + 1.0) * (1.0 / 4_294_967_297.0) * total;

        // First line whose cumulative intensity reaches xi (xi <= cumulative);
        // fall back to the last line. A `found` flag avoids an early break.
        let mut cumulative = 0.0_f64;
        let mut chosen = count - 1u32;
        let mut found = 0u32;
        let mut j = 0u32;
        while j < count {
            cumulative += intensities_flat[base + j as usize];
            if found == 0u32 && xi <= cumulative {
                chosen = j;
                found = 1u32;
            }
            j += 1u32;
        }
        out[ABSOLUTE_POS] = energies_flat[base + chosen as usize];
    }
}

/// Run the discrete-spectrum kernel for each sample (`offsets[k]`,
/// `counts[k]`, `seeds[k]` slicing `energies_flat`/`intensities_flat`).
/// Returns the sampled per-sample line energy. For test/validation use.
pub fn run_discrete_spectrum(
    ctx: &GpuContext,
    energies_flat: &[f64],
    intensities_flat: &[f64],
    offsets: &[u32],
    counts: &[u32],
    seeds: &[u32],
) -> Vec<f64> {
    let n = seeds.len();
    assert_eq!(offsets.len(), n);
    assert_eq!(counts.len(), n);
    assert_eq!(energies_flat.len(), intensities_flat.len());
    let client = ctx.client();
    let en_h = client.create_from_slice(bytemuck::cast_slice(energies_flat));
    let int_h = client.create_from_slice(bytemuck::cast_slice(intensities_flat));
    let off_h = client.create_from_slice(bytemuck::cast_slice(offsets));
    let cnt_h = client.create_from_slice(bytemuck::cast_slice(counts));
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    // One f64 output (energy) per sample (== per seed).
    let out_h = client.empty(n * core::mem::size_of::<f64>());

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        discrete_spectrum_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(en_h, energies_flat.len()),
            BufferArg::from_raw_parts(int_h, intensities_flat.len()),
            BufferArg::from_raw_parts(off_h, offsets.len()),
            BufferArg::from_raw_parts(cnt_h, counts.len()),
            BufferArg::from_raw_parts(seed_h, seeds.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice::<u8, f64>(&client.read_one(out_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// GPU discrete-spectrum sampling must match the 32-bit-PCG CPU twin
    /// bit-for-bit (the walk is integer-PCG + f64 add/mul, no FMA / no
    /// transcendentals). Sweeps representative line spectra (single line,
    /// two equal, skewed, with zero-intensity lines, wide dynamic range)
    /// across many seeds, checks the returned energy is always a real line,
    /// and spot-checks selection-frequency sanity.
    #[test]
    fn gpu_discrete_spectrum_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Representative spectra: (line energies [eV], intensities).
        let spectra: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.332e6], vec![1.0]),                               // single line
            (vec![5.11e5, 1.274e6], vec![1.0, 1.0]),                  // two equal
            (vec![8.46e5, 1.811e6, 2.113e6], vec![0.99, 0.27, 0.14]), // Mn56-like, skewed
            (vec![1.0e5, 5.0e5, 1.0e6, 2.0e6], vec![3.0, 0.0, 1.0, 0.0]), // zero-intensity lines
            (
                vec![1.0e4, 1.0e5, 5.0e5, 1.0e6, 1.5e6, 3.0e6],
                vec![1.0e-3, 1.0e2, 1.0, 50.0, 0.5, 7.0],
            ), // wide dynamic range
        ];

        // Flatten the distinct spectra once; samples reference them by
        // offset, each with its own seed.
        let mut en_flat: Vec<f64> = Vec::new();
        let mut int_flat: Vec<f64> = Vec::new();
        let mut sp_off: Vec<u32> = Vec::new();
        let mut sp_cnt: Vec<u32> = Vec::new();
        for (en, int) in &spectra {
            assert_eq!(en.len(), int.len());
            sp_off.push(en_flat.len() as u32);
            sp_cnt.push(en.len() as u32);
            en_flat.extend_from_slice(en);
            int_flat.extend_from_slice(int);
        }

        let n_seeds = 200u32;
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        let mut seeds = Vec::new();
        for (si, _) in spectra.iter().enumerate() {
            for s in 0..n_seeds {
                offsets.push(sp_off[si]);
                counts.push(sp_cnt[si]);
                // Decorrelate seeds per spectrum+sample with the kernel's
                // standard seed banding.
                seeds.push((si as u32 * 9_973 + s).wrapping_mul(2_654_435_761));
            }
        }

        let gpu = run_discrete_spectrum(&ctx, &en_flat, &int_flat, &offsets, &counts, &seeds);
        assert_eq!(gpu.len(), seeds.len());

        for (k, (&off, &cnt)) in offsets.iter().zip(counts.iter()).enumerate() {
            let en = &en_flat[off as usize..(off + cnt) as usize];
            let int = &int_flat[off as usize..(off + cnt) as usize];
            let cpu = sample_discrete_photon_energy_cpu(en, int, seeds[k]);
            assert_eq!(
                gpu[k], cpu,
                "GPU/CPU discrete-line energy differs at sample {k}: gpu {} cpu {} (en {en:?}, int {int:?}, seed {})",
                gpu[k], cpu, seeds[k]
            );
            // The returned energy must be one of this spectrum's lines, and
            // never a zero-intensity line.
            let line = en
                .iter()
                .position(|&e| e == gpu[k])
                .unwrap_or_else(|| panic!("sample {k}: energy {} is not a line of {en:?}", gpu[k]));
            assert!(
                int[line] > 0.0,
                "sample {k}: picked zero-intensity line {line} (int {int:?})"
            );
        }

        // Sanity: for the skewed Mn56-like spectrum the most intense line
        // (#0, intensity 0.99) must be picked most often over the sweep.
        let skew_si = 2usize;
        let (en, int) = &spectra[skew_si];
        let mut hist = vec![0u32; en.len()];
        for s in 0..n_seeds {
            let seed = (skew_si as u32 * 9_973 + s).wrapping_mul(2_654_435_761);
            let e = sample_discrete_photon_energy_cpu(en, int, seed);
            let line = en.iter().position(|&x| x == e).unwrap();
            hist[line] += 1;
        }
        assert!(
            hist[0] > hist[1] && hist[0] > hist[2],
            "most-intense line should be picked most often, got {hist:?}"
        );
        println!(
            "discrete spectrum: {} samples, GPU==CPU bit-exact; skewed histogram {hist:?}",
            seeds.len()
        );
    }
}
