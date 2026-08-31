use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Build Vose alias tables for O(1) weighted sampling over `weights`.
///
/// `weights` are normalized internally (only their relative values matter), so
/// callers may pass unnormalized probabilities. Returns `(alias, prob_scaled)`
/// indexed by bin. Callers must ensure `weights` is non-empty with a positive
/// sum.
///
/// Vose's improvement to Walker's alias method:
/// - Walker, A.J. (1977). "An Efficient Method for Generating Discrete Random
///   Variables with General Distributions", ACM TOMS 3(3), 253-256.
/// - Vose, M.D. (1991). "A Linear Algorithm for Generating Random Numbers with
///   a Given Distribution", IEEE Trans. Softw. Eng. 17(9), 972-975.
fn build_alias_tables(weights: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = weights.len();
    let total: f64 = weights.iter().sum();
    // Normalize and scale by n.
    let scaled: Vec<f64> = weights.iter().map(|&p| p / total * n as f64).collect();

    let mut alias = vec![0usize; n];
    let mut prob_scaled = scaled.clone();

    // Separate into small and large based on the 1.0 threshold.
    let mut small: Vec<usize> = Vec::new();
    let mut large: Vec<usize> = Vec::new();
    for (i, &p) in scaled.iter().enumerate() {
        if p < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }

    while let (Some(j), Some(k)) = (small.pop(), large.pop()) {
        alias[j] = k;
        prob_scaled[k] += prob_scaled[j] - 1.0;
        if prob_scaled[k] < 1.0 {
            small.push(k);
        } else {
            large.push(k);
        }
    }

    (alias, prob_scaled)
}

/// Discrete energy distribution using the alias method for O(1) sampling.
///
/// Uses Vose's improvement to Walker's alias method:
/// - Walker, A.J. (1977). "An Efficient Method for Generating Discrete Random
///   Variables with General Distributions", ACM TOMS 3(3), 253-256.
/// - Vose, M.D. (1991). "A Linear Algorithm for Generating Random Numbers with
///   a Given Distribution", IEEE Trans. Softw. Eng. 17(9), 972-975.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Discrete {
    // Energy values
    energies: Vec<f64>,

    // Probability for each energy
    probabilities: Vec<f64>,

    // Alias table for O(1) sampling (from DiscreteIndex)
    alias: Vec<usize>,

    // Normalized probabilities scaled by n
    prob_scaled: Vec<f64>,
}

impl Discrete {
    /// Create a new discrete distribution with energies and probabilities
    /// Probabilities will be automatically normalized
    pub fn new(energies: Vec<f64>, probabilities: Vec<f64>) -> Result<Self, String> {
        if energies.is_empty() {
            return Err("Energies vector cannot be empty".to_string());
        }
        if energies.len() != probabilities.len() {
            return Err(format!(
                "Energies and probabilities must have same length (got {} and {})",
                energies.len(),
                probabilities.len()
            ));
        }
        if probabilities.iter().any(|&p| p < 0.0) {
            return Err("Probabilities cannot be negative".to_string());
        }
        if probabilities.iter().all(|&p| p == 0.0) {
            return Err("At least one probability must be non-zero".to_string());
        }

        let mut dist = Self {
            energies,
            probabilities,
            alias: Vec::new(),
            prob_scaled: Vec::new(),
        };

        dist.init_alias();
        Ok(dist)
    }

    /// Initialize alias tables using Vose's algorithm.
    fn init_alias(&mut self) {
        let (alias, prob_scaled) = build_alias_tables(&self.probabilities);
        self.alias = alias;
        self.prob_scaled = prob_scaled;
    }

    /// Sample an energy value using O(1) alias method
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let n = self.energies.len();

        if n == 1 {
            return self.energies[0];
        }

        // Sample bin using alias method
        let u: f64 = rng.random::<f64>();
        let bin = (u * n as f64).floor() as usize;
        let bin = bin.min(n - 1); // Safety check

        let xi: f64 = rng.random::<f64>();
        let selected_bin = if xi < self.prob_scaled[bin] {
            bin
        } else {
            self.alias[bin]
        };

        self.energies[selected_bin]
    }

    /// Get the energies in this distribution
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Get the probabilities in this distribution
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
}

/// Histogram (piecewise-constant) energy distribution.
///
/// `boundaries` are `n + 1` strictly ascending bin edges in eV; `probabilities`
/// are `n` per-bin weights interpreted as probability MASS and normalized
/// internally (only their relative values matter, so a raw multigroup flux may
/// be passed directly). Sampling draws a bin in proportion to its mass via the
/// O(1) alias method, then a uniform energy within that bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    // Bin edges (eV), length n + 1, strictly ascending.
    boundaries: Vec<f64>,

    // Per-bin probability mass (as supplied; normalized in the alias tables).
    probabilities: Vec<f64>,

    // Alias table over bins for O(1) sampling.
    alias: Vec<usize>,

    // Normalized probabilities scaled by n.
    prob_scaled: Vec<f64>,
}

impl Histogram {
    /// Create a histogram energy distribution. `probabilities` are normalized
    /// internally; only their relative values matter.
    pub fn new(boundaries: Vec<f64>, probabilities: Vec<f64>) -> Result<Self, String> {
        if probabilities.is_empty() {
            return Err("Histogram must have at least one bin".to_string());
        }
        if boundaries.len() != probabilities.len() + 1 {
            return Err(format!(
                "Histogram needs one more boundary than probabilities (got {} boundaries and {} probabilities)",
                boundaries.len(),
                probabilities.len()
            ));
        }
        // Ahead of the ascending check, which a NaN passes: every comparison
        // against a NaN is false, so `w[1] <= w[0]` says nothing about one.
        // Such a boundary used to reach the multigroup collapse, where it
        // produced a NaN group average, a NaN reaction rate, and an inventory
        // of NaNs with no error anywhere along the way (issue #576).
        if let Some(i) = boundaries.iter().position(|e| !e.is_finite()) {
            return Err(format!(
                "Histogram boundary {i} is {}, not a finite energy",
                boundaries[i]
            ));
        }
        if boundaries.windows(2).any(|w| w[1] <= w[0]) {
            return Err("Histogram boundaries must be strictly ascending".to_string());
        }
        // Same hole, same reason: `p < 0.0` is false for a NaN.
        if let Some(i) = probabilities.iter().position(|p| !p.is_finite()) {
            return Err(format!(
                "Histogram probability {i} is {}, not a finite value",
                probabilities[i]
            ));
        }
        if probabilities.iter().any(|&p| p < 0.0) {
            return Err("Histogram probabilities cannot be negative".to_string());
        }
        if probabilities.iter().all(|&p| p == 0.0) {
            return Err("At least one Histogram probability must be non-zero".to_string());
        }

        let (alias, prob_scaled) = build_alias_tables(&probabilities);
        Ok(Self {
            boundaries,
            probabilities,
            alias,
            prob_scaled,
        })
    }

    /// Sample an energy: pick a bin by mass (alias method), then draw uniformly
    /// within that bin.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let n = self.probabilities.len();
        let bin = if n == 1 {
            0
        } else {
            let u: f64 = rng.random::<f64>();
            let b = ((u * n as f64).floor() as usize).min(n - 1);
            let xi: f64 = rng.random::<f64>();
            if xi < self.prob_scaled[b] {
                b
            } else {
                self.alias[b]
            }
        };
        let lo = self.boundaries[bin];
        let hi = self.boundaries[bin + 1];
        lo + rng.random::<f64>() * (hi - lo)
    }

    /// The bin boundaries (eV), length `n + 1`.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// The per-bin probabilities, as supplied (before normalization).
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
}

/// Uniform energy distribution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Uniform {
    // Lower bound
    a: f64,

    // Upper bound
    b: f64,
}

impl Uniform {
    /// Create a new uniform distribution with lower and upper bounds
    pub fn new(a: f64, b: f64) -> Result<Self, String> {
        if a >= b {
            return Err(format!(
                "Lower bound must be less than upper bound (got a={a}, b={b})"
            ));
        }

        Ok(Self { a, b })
    }

    /// Sample an energy value uniformly between a and b
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // a + rand * (b - a)
        self.a + rng.random::<f64>() * (self.b - self.a)
    }

    /// Get the lower bound
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the upper bound
    pub fn b(&self) -> f64 {
        self.b
    }
}

/// Normal (Gaussian) energy distribution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Normal {
    /// Mean energy in eV
    mean_val: f64,
    /// Standard deviation in eV
    std_dev: f64,
}

impl Normal {
    /// Create a new normal distribution with mean and standard deviation
    pub fn new(mean_val: f64, std_dev: f64) -> Result<Self, String> {
        if std_dev <= 0.0 {
            return Err(format!(
                "Standard deviation must be positive (got {})",
                std_dev
            ));
        }
        Ok(Self { mean_val, std_dev })
    }

    /// Sample an energy value using Marsaglia's polar method
    /// Uses Marsaglia's polar method
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let x: f64 = 2.0 * rng.random::<f64>() - 1.0;
            let y: f64 = 2.0 * rng.random::<f64>() - 1.0;
            let r2 = x * x + y * y;
            if r2 > 0.0 && r2 < 1.0 {
                let z = (-2.0 * r2.ln() / r2).sqrt();
                return self.mean_val + self.std_dev * z * x;
            }
        }
    }

    /// Get the mean energy
    pub fn mean_val(&self) -> f64 {
        self.mean_val
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// Fusion reactant types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionReactants {
    /// D(d,n)³He reaction
    DD,
    /// T(d,n)α reaction
    DT,
}

/// Return a Gaussian energy distribution for fusion neutron emission.
///
/// Computes the mean energy and spectral width of the neutron energy spectrum
/// from thermonuclear fusion reactions in a plasma with Maxwellian ion velocity
/// distributions, using relativistic interpolation formulas from Ballabio et al.
///
/// Reference: L. Ballabio et al., Nucl. Fusion 38 (1998) 1723
///            doi:10.1088/0029-5515/38/11/310
///
/// # Arguments
/// * `ion_temp` - Ion temperature in eV (must be between 0 and 100 keV)
/// * `reactants` - Fusion reactant type (DD or DT)
pub fn fusion_neutron_spectrum(
    ion_temp: f64,
    reactants: FusionReactants,
) -> Result<Normal, String> {
    if !(0.0..=100.0e3).contains(&ion_temp) {
        return Err(format!(
            "Ion temperature must be between 0 and 100 keV (got {} eV)",
            ion_temp
        ));
    }

    // Atomic masses in amu (AME2020)
    const M_NEUTRON: f64 = 1.008_664_915_95;
    const M_DEUTERON: f64 = 2.014_101_778_12;
    const M_TRITON: f64 = 3.016_049_281_99;
    const M_HE3: f64 = 3.016_029_322_65;
    const M_ALPHA: f64 = 4.002_603_254_13;
    // 1 amu in eV/c²
    const EV_PER_AMU: f64 = 931_494_103.72;

    // Ballabio Table III/IV coefficients and Q-value derived quantities
    struct BallabioCoeffs {
        e_n: f64, // Neutron energy at zero ion temperature (eV)
        w0: f64,  // FWHM coefficient (keV^{1/2})
        // Low-T (Table III) peak shift coefficients
        a1: f64,
        a2: f64,
        a3: f64,
        a4: f64,
        // Low-T (Table III) width correction coefficients
        b1: f64,
        b2: f64,
        b3: f64,
        b4: f64,
        // High-T (Table IV) peak shift coefficients
        a5: f64,
        a6: f64,
        // High-T (Table IV) width correction coefficients
        b5: f64,
        b6: f64,
    }

    let coeffs = match reactants {
        FusionReactants::DD => {
            let q = (M_DEUTERON + M_DEUTERON - M_HE3 - M_NEUTRON) * EV_PER_AMU;
            let e_n = M_HE3 / (M_HE3 + M_NEUTRON) * q;
            BallabioCoeffs {
                e_n,
                w0: 82.542,
                a1: 4.69515,
                a2: -0.040729,
                a3: 0.47,
                a4: 0.81844,
                b1: 1.7013e-3,
                b2: 0.16888,
                b3: 0.49,
                b4: 7.9460e-4,
                a5: 18.225,
                a6: 2.1525,
                b5: 8.4619e-3,
                b6: 8.3241e-4,
            }
        }
        FusionReactants::DT => {
            let q = (M_DEUTERON + M_TRITON - M_ALPHA - M_NEUTRON) * EV_PER_AMU;
            let e_n = M_ALPHA / (M_ALPHA + M_NEUTRON) * q;
            BallabioCoeffs {
                e_n,
                w0: 177.259,
                a1: 5.30509,
                a2: 2.4736e-3,
                a3: 1.84,
                a4: 1.3818,
                b1: 5.1068e-4,
                b2: 7.6223e-3,
                b3: 1.78,
                b4: 8.7691e-5,
                a5: 37.771,
                a6: 0.92181,
                b5: 2.0199e-3,
                b6: 5.9501e-5,
            }
        }
    };

    // Ion temperature in keV
    let t = ion_temp * 1.0e-3;

    // Handle zero temperature case
    if t <= 0.0 {
        return Err("Ion temperature must be positive".to_string());
    }

    let (delta_e, delta_w) = if t <= 40.0 {
        // Low-temperature interpolation (Table III, 0 < T_i <= 40 keV)
        let de =
            coeffs.a1 / (1.0 + coeffs.a2 * t.powf(coeffs.a3)) * t.powf(2.0 / 3.0) + coeffs.a4 * t;
        let dw =
            coeffs.b1 / (1.0 + coeffs.b2 * t.powf(coeffs.b3)) * t.powf(2.0 / 3.0) + coeffs.b4 * t;
        (de, dw)
    } else {
        // High-temperature interpolation (Table IV, 40 < T_i < 100 keV)
        let de = coeffs.a5 + coeffs.a6 * t;
        let dw = coeffs.b5 + coeffs.b6 * t;
        (de, dw)
    };

    // FWHM in eV (w0 and delta_e are in keV units, convert to eV)
    let fwhm = coeffs.w0 * (1.0 + delta_w) * t.sqrt() * 1.0e3;

    // Convert FWHM to standard deviation: σ = FWHM / (2√(2 ln 2))
    let sigma = fwhm / (2.0 * (2.0_f64.ln() * 2.0).sqrt());

    // Mean energy: E_0 + thermal peak shift (convert delta_e from keV to eV)
    let mean = coeffs.e_n + delta_e * 1.0e3;

    Normal::new(mean, sigma)
}
