//! Pure sampling helpers used by reaction-product code.
//!
//! Lives here (not in yamc's `interaction` module) so reaction product
//! sampling can stand on its own without depending on transport-engine
//! types (`Particle`, etc.). yamc re-exports these under
//! `yamc::interaction` so existing call sites keep resolving.

use rand::{Rng, RngExt};

/// Sample energy from a Maxwell-Boltzmann fission spectrum (ENDF File 5,
/// Law 7) at temperature `theta`, using the standard three-uniform
/// rejection trick.
///
/// # Arguments
/// * `theta` - Maxwell distribution temperature parameter (same units as the
///   returned energy)
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled energy in same units as theta
#[inline]
pub fn sample_maxwell_spectrum(theta: f64, rng: &mut impl Rng) -> f64 {
    let r1: f64 = rng.random::<f64>();
    let r2: f64 = rng.random::<f64>();
    let r3: f64 = rng.random::<f64>();

    let c = (std::f64::consts::FRAC_PI_2 * r3).cos();
    -theta * (r1.ln() + r2.ln() * c * c)
}

/// Sample from Watt fission spectrum (ENDF File 5, Law 11)
///
/// The Watt distribution is: p(E) ~ exp(-E/a) * sinh(sqrt(b*E))
/// (Watt, Phys. Rev. 87, 1037, 1952)
///
/// Sampling uses the Watt-Maxwell relationship:
///   E = w + a²b/4 + U(-1,1) * sqrt(a²b*w)
/// where w is sampled from Maxwell(a).
///
/// # Arguments
/// * `a` - Watt parameter a (in same units as desired output energy)
/// * `b` - Watt parameter b (in inverse units of energy, e.g., 1/eV)
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled energy in same units as parameter a
#[inline]
pub fn sample_watt_spectrum_params(a: f64, b: f64, rng: &mut impl Rng) -> f64 {
    // First sample from Maxwell distribution with temperature a
    let w = sample_maxwell_spectrum(a, rng);

    // Apply Watt correction
    // E = w + a²b/4 + U(-1,1) * sqrt(a²b*w)
    let a2b = a * a * b;
    let u = 2.0 * rng.random::<f64>() - 1.0; // U(-1, 1)
    w + 0.25 * a2b + u * (a2b * w).sqrt()
}
