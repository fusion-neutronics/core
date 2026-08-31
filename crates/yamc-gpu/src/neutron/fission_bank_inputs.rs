//! Runtime gate for the device fission particle bank (issue #78).
//!
//! Mirrors the gate-flag idiom of [`super::survival_biasing::SurvivalBiasingInputs`]
//! and [`CoupledPhotonInputs`](super::transport::CoupledPhotonInputs): a single
//! one-element buffer carries an enable flag so the kernel stays byte-identical
//! to today's analog run when the flag is `0`.
//!
//! # Why this struct lives in an always-built module
//!
//! The plain input struct is referenced by the always-compiled
//! `translate.rs` / `dispatch.rs` layers, which build on macOS too (where the
//! cubecl kernel module [`super::transport`] is
//! `#[cfg(not(target_os = "macos"))]`-gated). Placing the struct here -- next
//! to [`super::survival_biasing`] / [`super::nuclide_select_inputs`] -- keeps it
//! reachable on every target. Several past PRs regressed the macOS wheel build
//! by putting such a struct inside the gated kernel module.
//!
//! # The two transport modes the flag selects
//!
//! - flag `0` (OFF): the kernel's fission branch keeps the legacy
//!   `weight *= nu_bar` + `FISSION_WEIGHT_CAP` weight-cap terminator. A run
//!   with no fissile material never enters the fission branch at all, so it is
//!   byte-identical to a pre-bank run regardless of this flag.
//! - flag `1` (ON): the fission branch instead stochastically rounds `nu_bar`
//!   to an integer `N`, continues ONE chi-sampled progeny in the current walk
//!   (weight unchanged), and appends the other `N - 1` chi-sampled progeny to
//!   the shared device particle bank tagged as neutrons. The host then drains
//!   the banked neutrons and transports them as a fission source in a second
//!   pass, folding into the same tallies -- the GPU twin of the CPU
//!   `sample_fission_neutrons` -> `bank_secondary` fission chain.

/// Runtime gate for the device fission bank, ready to upload as one `&[u32]`
/// kernel buffer. A single buffer keeps the kernel's storage-descriptor budget
/// growth to one binding.
#[derive(Debug, Clone)]
pub struct FissionBankInputs {
    /// `[enable]`. Always length 1. `1` enables the fission-bank branching
    /// path; `0` keeps the legacy `weight *= nu_bar` + cap terminator.
    pub enabled: Vec<u32>,
}

impl FissionBankInputs {
    /// Fission-bank OFF: the gate flag is `0`, so the kernel and the CPU twin
    /// take the legacy `weight *= nu_bar` + cap fission branch unchanged.
    pub fn off() -> Self {
        Self {
            enabled: vec![0u32],
        }
    }

    /// Fission-bank ON: the kernel branches the fission chain into the device
    /// bank instead of multiplying weight.
    pub fn on() -> Self {
        Self {
            enabled: vec![1u32],
        }
    }

    /// `true` when the fission bank is enabled (gate flag `== 1`).
    pub fn is_on(&self) -> bool {
        self.enabled.first().copied().unwrap_or(0) == 1
    }
}
