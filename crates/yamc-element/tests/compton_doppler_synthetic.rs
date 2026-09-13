//! The OpenMC 4036 / Kaltiaisenaho Compton Doppler procedure, piece by piece
//! (fusion-neutronics/core#22), on a synthetic two-shell element so no data
//! files are needed.
//!
//! An integration test rather than a unit test in `photon.rs` on purpose: the
//! synthetic element publishes its own Compton momentum grid through
//! `set_compton_profile_pz`, which is process-wide and first-write-wins, so
//! sharing a test binary with the tests that load the real Fe grid would make
//! whichever ran second sample against the wrong grid. Here it owns the
//! process. The two pure-function tests mirror OpenMC's
//! `tests/cpp_unit_tests/test_photon.cpp`.

use rand::SeedableRng;
use yamc_element::photon::{
    compton_energy_ratio, compton_profile_pz, compton_profile_tail_integral,
    finalize_compton_profiles, invert_compton_profile_tail, set_compton_profile_pz,
    PhotonInteraction, FINE_STRUCTURE, MASS_ELECTRON_EV,
};
use yamc_nuclide::reaction_product::Tabulated1D;
/// Two-shell element with Gaussian-ish profiles on a coarse grid, wide
/// enough that the tail matters, and binding energies of a K and an outer
/// shell.
fn synthetic_element() -> PhotonInteraction {
    let pz: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    set_compton_profile_pz(pz.clone());
    let pz = compton_profile_pz();
    let widths = [3.0_f64, 0.8];
    let mut profile_pdf: Vec<Vec<f64>> = widths
        .iter()
        .map(|w| pz.iter().map(|&p| (-p * p / (2.0 * w * w)).exp()).collect())
        .collect();
    let mut profile_cdf: Vec<Vec<f64>> = profile_pdf
        .iter()
        .map(|row| {
            let mut cdf = vec![0.0; row.len()];
            for i in 1..row.len() {
                cdf[i] = cdf[i - 1] + 0.5 * (row[i - 1] + row[i]) * (pz[i] - pz[i - 1]);
            }
            cdf
        })
        .collect();
    let (profile_tail_slope, profile_negative_mass) =
        finalize_compton_profiles(&mut profile_pdf, &mut profile_cdf, &pz).unwrap();
    PhotonInteraction {
        name: "Xx".into(),
        index: 0,
        atomic_number: 10,
        energy: vec![1.0e3, 1.0e8],
        coherent_xs: vec![0.0; 2],
        incoherent_xs: vec![0.0; 2],
        photoelectric_total_xs: vec![0.0; 2],
        pair_production_total_xs: vec![0.0; 2],
        pair_production_nuclear_xs: vec![0.0; 2],
        pair_production_electron_xs: vec![0.0; 2],
        coherent_int_form_factor: Tabulated1D::Tabulated1D {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            breakpoints: vec![2],
            interpolation: vec![2],
        },
        incoherent_form_factor: Tabulated1D::Tabulated1D {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            breakpoints: vec![2],
            interpolation: vec![2],
        },
        electron_pdf: vec![0.2, 0.8],
        binding_energy: vec![7.1e3, 20.0],
        profile_pdf,
        profile_cdf,
        profile_tail_slope,
        profile_negative_mass,
        shells: Vec::new(),
        cross_sections: Vec::new(),
        subshell_radiative_energy: Vec::new(),
        compton_radiative_energy: 0.0,
        compton_relax_map: Vec::new(),
        has_atomic_relaxation: false,
        dcs: Vec::new(),
        stopping_power_radiative: Vec::new(),
        ionization_energy: Vec::new(),
        n_electrons: Vec::new(),
        mean_excitation_energy: 0.0,
        ttb_electron_energy: Vec::new(),
        ttb_photon_energy: Vec::new(),
    }
}

#[test]
fn exponential_tail_integrates_and_inverts() {
    // Mirrors OpenMC's "Compton profile exponential tail" case.
    let (pz_last, profile_last, slope) = (5.0, 0.1, -0.2);
    assert_eq!(
        compton_profile_tail_integral(pz_last, pz_last, profile_last, slope),
        0.0
    );
    let integral = compton_profile_tail_integral(7.0, pz_last, profile_last, slope);
    let expected = profile_last * (2.0 * slope).exp_m1() / slope;
    assert!((integral - expected).abs() <= 1e-14 * expected.abs());
    let pz = invert_compton_profile_tail(integral, pz_last, profile_last, slope);
    assert!((pz - 7.0).abs() < 1e-13, "{pz}");
    let total_tail = -profile_last / slope;
    let far = compton_profile_tail_integral(200.0, pz_last, profile_last, slope);
    assert!((far - total_tail).abs() <= 1e-14 * total_tail);
}

#[test]
fn energy_root_follows_the_sign_of_the_momentum() {
    // Mirrors OpenMC's "Compton energy root follows signed electron
    // momentum" case, plus the branch ordering across a sweep.
    let (alpha, mu) = (1.0, 0.0);
    let free = 1.0 / (1.0 + alpha * (1.0 - mu));
    assert!((compton_energy_ratio(alpha, mu, 0.0).unwrap() - free).abs() < 1e-15);
    let neg = compton_energy_ratio(alpha, mu, -10.0).unwrap();
    let pos = compton_energy_ratio(alpha, mu, 10.0).unwrap();
    assert!(neg > 0.0 && neg < free, "{neg} vs {free}");
    assert!(pos > free, "{pos} vs {free}");
    for &pz in &[-100.0, -30.0, -3.0, -0.1, 0.1, 3.0, 30.0] {
        let r = compton_energy_ratio(0.3, -0.7, pz).unwrap();
        let f = 1.0 / (1.0 + 0.3 * 1.7);
        assert_eq!(pz < 0.0, r < f, "pz {pz}: ratio {r}, free {f}");
    }
}

#[test]
fn finalisation_normalises_each_half_profile_to_one_half() {
    let el = synthetic_element();
    let pz = compton_profile_pz();
    for i in 0..2 {
        let slope = el.profile_tail_slope[i];
        assert!(slope < 0.0 && slope.is_finite());
        let n = pz.len();
        let tail = -el.profile_pdf[i][n - 1] / slope;
        let total = el.profile_cdf[i][n - 1] + tail;
        assert!((total - 0.5).abs() < 1e-12, "shell {i}: {total}");
        // K_i is monotone, K_i(0) = 0, and saturates at 1/2 far out.
        let mut prev = 0.0;
        for k in 0..200 {
            let x = k as f64 * 0.25;
            let c = el.compton_profile_cdf(i, x);
            assert!(c >= prev - 1e-15, "shell {i} not monotone at {x}");
            prev = c;
        }
        assert_eq!(el.compton_profile_cdf(i, 0.0), 0.0);
        assert!((el.compton_profile_cdf(i, 1.0e4) - 0.5).abs() < 1e-12);
        // The negative-branch mass is K_i(1/alpha), essentially all of it here.
        assert!(
            (el.profile_negative_mass[i] - el.compton_profile_cdf(i, FINE_STRUCTURE)).abs() < 1e-15
        );
    }
}

#[test]
fn profile_cdf_inversion_round_trips_inside_and_past_the_grid() {
    let el = synthetic_element();
    for i in 0..2 {
        for k in 1..60 {
            // Targets from well inside the grid out into the tail.
            let c = 0.5 * (k as f64 / 60.0).powi(2);
            let pz = el.invert_compton_profile_cdf(i, c);
            let back = el.compton_profile_cdf(i, pz);
            assert!(
                (back - c).abs() < 1e-10,
                "shell {i}: c {c} -> pz {pz} -> {back}"
            );
        }
    }
}

#[test]
fn sampled_energy_respects_the_binding_limit_and_broadens_around_the_free_electron_line() {
    let el = synthetic_element();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x0C0);
    for &(e_in, mu) in &[(1.0e5_f64, -0.8_f64), (5.0e4, 0.3), (2.0e6, -0.99)] {
        let alpha = e_in / MASS_ELECTRON_EV;
        let e_kn = alpha / (1.0 + alpha * (1.0 - mu)) * MASS_ELECTRON_EV;
        let n = 20_000;
        let (mut below, mut above, mut sum) = (0usize, 0usize, 0.0);
        for _ in 0..n {
            let (e_out, shell) = el.compton_doppler(alpha, mu, &mut rng);
            assert!(shell >= 0 && (shell as usize) < 2);
            let e_b = el.binding_energy[shell as usize];
            assert!(
                e_out > 0.0 && e_out <= e_in - e_b + 1e-9 * e_in,
                "{e_out} vs {}",
                e_in - e_b
            );
            if e_out < e_kn {
                below += 1;
            } else {
                above += 1;
            }
            sum += e_out;
        }
        // Both branches are populated (the old sampler only ever produced
        // one side of the free-electron line for pz_max < 0) and the
        // broadened mean sits within a few percent of the free-electron
        // energy.
        assert!(
            below > n / 20 && above > n / 20,
            "E {e_in} mu {mu}: {below} below, {above} above"
        );
        let mean = sum / n as f64;
        assert!(
            (mean / e_kn - 1.0).abs() < 0.05,
            "E {e_in} mu {mu}: mean {mean} vs KN {e_kn}"
        );
    }
}

#[test]
fn closed_shells_are_not_selected_and_forward_scattering_favours_accessible_mass() {
    let el = synthetic_element();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11);
    // Below the K binding energy only the outer shell is open.
    let alpha = 5.0e3 / MASS_ELECTRON_EV;
    for _ in 0..2_000 {
        let (_e, shell) = el.compton_doppler(alpha, -0.5, &mut rng);
        assert_eq!(shell, 1, "a closed shell was selected");
    }
    // Near-forward scattering at 100 keV: the K shell's pz_max is deep in
    // the negative tail, so it must be picked far less often than its 20%
    // occupancy, while at backscatter the two shells go by occupancy.
    let alpha = 1.0e5 / MASS_ELECTRON_EV;
    let k_fraction = |mu: f64, rng: &mut rand::rngs::StdRng| {
        let n = 20_000;
        let k = (0..n)
            .filter(|_| el.compton_doppler(alpha, mu, rng).1 == 0)
            .count();
        k as f64 / n as f64
    };
    let forward = k_fraction(0.999, &mut rng);
    let back = k_fraction(-0.9, &mut rng);
    assert!(forward < 0.05, "forward K fraction {forward}");
    assert!((back - 0.2).abs() < 0.03, "backscatter K fraction {back}");
}
