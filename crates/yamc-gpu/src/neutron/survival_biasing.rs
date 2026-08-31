//! Survival-biasing (implicit-capture) runtime parameters for the neutron
//! transport kernel.
//!
//! Mirrors the gate-flag idiom of [`super::CoupledPhotonInputs`] /
//! [`super::DecayPhotonInputs`]: a single small buffer carries a runtime
//! enable flag plus the weight parameters, so the kernel stays
//! byte-identical to today's analog run when the flag is `0` (the
//! reaction-selection block then takes the same analog absorption-kill
//! branch with the same RNG schedule). When the flag is `1` the kernel
//! never terminates on capture: at every collision the scatter/fission
//! selection is renormalised over the scatter+fission mass and the
//! particle weight is multiplied by `(sigma_e + sigma_i + sigma_f) /
//! sigma_t`, the GPU twin of the CPU's `weight *= xs.scatter / xs.total`.
//!
//! The buffer holds three `f64` values:
//!   - `[0]` enable flag: `1.0` on, `0.0` off (read as `!= 0.0`).
//!   - `[1]` `weight_cutoff` (Russian-roulette trigger: a still-alive
//!     post-collision particle below this weight is rouletted).
//!   - `[2]` `weight_survive` (roulette survivor weight: survivors
//!     continue at this weight with survival probability
//!     `weight / weight_survive`).
//!
//! PR1 implemented implicit capture (reading only the enable flag); PR2
//! adds the weight-cutoff Russian roulette that consumes `weight_cutoff`
//! / `weight_survive` after each collision (delta-tracking aside, this is
//! the standard analog VR roulette).

/// Packed survival-biasing parameters, ready to upload as one `&[f64]`
/// kernel buffer. A single buffer keeps the kernel's storage-descriptor
/// budget growth to one binding.
#[derive(Debug, Clone)]
pub struct SurvivalBiasingInputs {
    /// `[enable, weight_cutoff, weight_survive]`. Always length 3.
    pub params: Vec<f64>,
}

impl SurvivalBiasingInputs {
    /// Survival-biasing OFF: the gate flag is `0.0`, so the kernel and the
    /// CPU twin take the analog capture-kill branch unchanged. The two
    /// weight slots are zero (never read while off).
    pub fn off() -> Self {
        Self {
            params: vec![0.0, 0.0, 0.0],
        }
    }

    /// Survival-biasing ON: implicit capture plus the weight-cutoff
    /// Russian roulette. `weight_cutoff` is the post-collision weight
    /// below which a still-alive particle is rouletted; `weight_survive`
    /// is the survivor weight.
    pub fn on(weight_cutoff: f64, weight_survive: f64) -> Self {
        Self {
            params: vec![1.0, weight_cutoff, weight_survive],
        }
    }

    /// `true` when survival biasing is enabled (gate flag `!= 0.0`).
    pub fn enabled(&self) -> bool {
        self.params.first().copied().unwrap_or(0.0) != 0.0
    }

    /// Weight-cutoff roulette trigger (`params[1]`). A still-alive
    /// post-collision particle whose weight falls below this is rouletted.
    pub fn weight_cutoff(&self) -> f64 {
        self.params.get(1).copied().unwrap_or(0.0)
    }

    /// Roulette survivor weight (`params[2]`). Survivors continue at this
    /// weight; survival probability is `weight / weight_survive`.
    pub fn weight_survive(&self) -> f64 {
        self.params.get(2).copied().unwrap_or(0.0)
    }
}

impl Default for SurvivalBiasingInputs {
    fn default() -> Self {
        Self::off()
    }
}
