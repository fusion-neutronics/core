use rand::{Rng, RngExt};

/// Angular distribution types - simplified enum approach
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AngularDistribution {
    Isotropic,
    Monodirectional {
        // Serialized as `direction`; `reference_uvw` is accepted as a
        // backward-compatible alias for model JSON written before the rename.
        #[serde(alias = "reference_uvw")]
        direction: [f64; 3],
    },
}

impl AngularDistribution {
    /// Create a new monodirectional distribution
    pub fn new_monodirectional(u: f64, v: f64, w: f64) -> Self {
        // Normalize the direction vector
        let mag = (u * u + v * v + w * w).sqrt();
        if mag == 0.0 {
            panic!("Direction vector cannot be zero");
        }
        Self::Monodirectional {
            direction: [u / mag, v / mag, w / mag],
        }
    }

    /// Create a new isotropic distribution
    pub fn new_isotropic() -> Self {
        Self::Isotropic
    }

    /// Sample a direction from this distribution
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        match self {
            AngularDistribution::Isotropic => {
                // Sample isotropic direction
                let xi1: f64 = rng.random::<f64>();
                let xi2: f64 = rng.random::<f64>();

                // Convert to spherical coordinates
                let mu = 2.0 * xi1 - 1.0; // cosine of polar angle
                let phi = 2.0 * std::f64::consts::PI * xi2; // azimuthal angle

                // Convert to Cartesian coordinates
                let sqrt_one_minus_mu2 = (1.0 - mu * mu).sqrt();
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();

                [
                    sqrt_one_minus_mu2 * cos_phi,
                    sqrt_one_minus_mu2 * sin_phi,
                    mu,
                ]
            }
            AngularDistribution::Monodirectional { direction } => *direction,
        }
    }
}
