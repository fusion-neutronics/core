//! Reaction-product yield (multiplicity) as a function of energy.

use serde::{Deserialize, Serialize};

/// Yield (multiplicity) of reaction product as function of energy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Yield {
    #[serde(rename = "Polynomial")]
    Polynomial { coefficients: Vec<f64> },
    #[serde(rename = "Tabulated")]
    Tabulated { x: Vec<f64>, y: Vec<f64> },
    #[serde(rename = "Tabulated1D")]
    Tabulated1D {
        x: Vec<f64>,
        y: Vec<f64>,
        breakpoints: Option<Vec<usize>>,
        interpolation: Option<Vec<u32>>,
    },
}

impl Yield {
    /// Evaluate yield at given energy
    pub fn evaluate(&self, energy: f64) -> f64 {
        match self {
            Yield::Polynomial { coefficients } => {
                if coefficients.is_empty() {
                    return 1.0;
                }
                coefficients
                    .iter()
                    .enumerate()
                    .fold(0.0, |acc, (i, &coeff)| acc + coeff * energy.powi(i as i32))
            }
            Yield::Tabulated { x, y } => {
                if x.is_empty() || y.is_empty() {
                    return 1.0;
                }
                if energy <= x[0] {
                    return y[0];
                }
                if energy >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }
                for i in 0..x.len() - 1 {
                    if energy >= x[i] && energy <= x[i + 1] {
                        let f = (energy - x[i]) / (x[i + 1] - x[i]);
                        return y[i] + f * (y[i + 1] - y[i]);
                    }
                }
                1.0
            }
            Yield::Tabulated1D { x, y, .. } => {
                if x.is_empty() || y.is_empty() {
                    return 1.0;
                }
                if energy <= x[0] {
                    return y[0];
                }
                if energy >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }
                for i in 0..x.len() - 1 {
                    if energy >= x[i] && energy <= x[i + 1] {
                        let f = (energy - x[i]) / (x[i + 1] - x[i]);
                        return y[i] + f * (y[i + 1] - y[i]);
                    }
                }
                1.0
            }
        }
    }

    /// Convert yield to tabulated form (energy, values)
    /// Used for extracting nu-bar data when total_nu is missing
    pub fn to_tabulated(&self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Yield::Polynomial { coefficients } => {
                // For polynomial yield, create a tabulated version over typical energy range
                // This handles constant yields (single coefficient) and energy-dependent yields
                if coefficients.is_empty() {
                    return (vec![1e-5, 2e7], vec![1.0, 1.0]);
                }
                if coefficients.len() == 1 {
                    // Constant yield
                    let val = coefficients[0];
                    return (vec![1e-5, 2e7], vec![val, val]);
                }
                // Create tabulated version at several energy points
                let energies: Vec<f64> = (0..100)
                    .map(|i| 1e-5 * (2e7 / 1e-5_f64).powf(i as f64 / 99.0))
                    .collect();
                let values: Vec<f64> = energies.iter().map(|&e| self.evaluate(e)).collect();
                (energies, values)
            }
            Yield::Tabulated { x, y } => (x.clone(), y.clone()),
            Yield::Tabulated1D { x, y, .. } => (x.clone(), y.clone()),
        }
    }
}
