//! Secondary particles emitted in a nuclear reaction.

use crate::angle_energy::AngleEnergy;
use crate::function::{Polynomial, Tabulated1D};

/// Whether a particle leaves at once or follows the decay of a precursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmissionMode {
    #[default]
    Prompt,
    /// Emitted from the decay of a reaction product, as delayed neutrons are.
    Delayed,
    /// The yield covers prompt and delayed sources together.
    Total,
}

impl EmissionMode {
    /// The name the Python package uses, e.g. `"prompt"`.
    pub fn name(self) -> &'static str {
        match self {
            EmissionMode::Prompt => "prompt",
            EmissionMode::Delayed => "delayed",
            EmissionMode::Total => "total",
        }
    }
}

/// A yield, which the format gives either as a polynomial in incident energy
/// or as a tabulation against it.
#[derive(Debug, Clone, PartialEq)]
pub enum Yield {
    Polynomial(Polynomial),
    Tabulated(Tabulated1D),
}

impl Default for Yield {
    /// One particle per reaction, the constant the Python reader defaults to.
    fn default() -> Self {
        Yield::Polynomial(Polynomial::new(vec![1.0]))
    }
}

impl Yield {
    /// The yield at an incident energy in eV.
    pub fn eval(&self, energy: f64) -> f64 {
        match self {
            Yield::Polynomial(p) => p.eval(energy),
            Yield::Tabulated(t) => t.eval(energy),
        }
    }
}

/// A secondary particle emitted in a nuclear reaction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Product {
    /// The particle type, `"neutron"` unless said otherwise.
    pub name: String,
    pub yield_: Yield,
    /// Decay rate in inverse seconds. Zero for a prompt particle.
    pub decay_rate: f64,
    /// The angle and energy of the product. More than one when the emission is
    /// split between representations, chosen by `applicability`.
    pub distribution: Vec<AngleEnergy>,
    /// The probability of sampling each distribution, as a function of
    /// incident energy. Empty when there is only one.
    pub applicability: Vec<Tabulated1D>,
    pub emission_mode: EmissionMode,
}

impl Product {
    /// A prompt particle of the given type, with unit yield.
    pub fn new(name: &str) -> Product {
        Product {
            name: name.to_string(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_product_is_one_prompt_particle() {
        let p = Product::new("neutron");
        assert_eq!(p.name, "neutron");
        assert_eq!(p.emission_mode, EmissionMode::Prompt);
        assert_eq!(p.decay_rate, 0.0);
        assert!(p.distribution.is_empty());
        assert!(p.applicability.is_empty());
        // The default yield is the constant 1, whatever the energy.
        assert_eq!(p.yield_.eval(0.0253), 1.0);
        assert_eq!(p.yield_.eval(2.0e7), 1.0);
    }

    #[test]
    fn a_tabulated_yield_is_interpolated() {
        let y = Yield::Tabulated(Tabulated1D::new(vec![0.0, 10.0], vec![2.0, 4.0]));
        assert_eq!(y.eval(5.0), 3.0);
    }
}
