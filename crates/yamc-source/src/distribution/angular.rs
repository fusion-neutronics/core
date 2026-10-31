use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Angular distribution types - simplified enum approach
///
/// A `Monodirectional` direction is a unit vector: [`new_monodirectional`]
/// normalizes the vector it is given, and refuses one that points nowhere.
/// A model file is checked the same way as it is read, so a direction that
/// points nowhere fails the load rather than the transport.
///
/// [`new_monodirectional`]: AngularDistribution::new_monodirectional
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "AngularDistributionData")]
pub enum AngularDistribution {
    Isotropic,
    Monodirectional { direction: [f64; 3] },
}

/// An [`AngularDistribution`] as a model file spells it, checked before it
/// becomes one: a file edited by hand can hold a direction no constructor
/// would have made.
#[derive(Deserialize)]
enum AngularDistributionData {
    Isotropic,
    Monodirectional {
        // Serialized as `direction`; `reference_uvw` is accepted as a
        // backward-compatible alias for model JSON written before the rename.
        #[serde(alias = "reference_uvw")]
        direction: [f64; 3],
    },
}

impl TryFrom<AngularDistributionData> for AngularDistribution {
    type Error = String;

    /// Refuses a direction that points nowhere: the zero vector, or one with
    /// a NaN or infinite component. Any other is kept exactly as written, so
    /// a model saved and loaded again is the same model, bit for bit.
    fn try_from(data: AngularDistributionData) -> Result<Self, String> {
        Ok(match data {
            AngularDistributionData::Isotropic => Self::Isotropic,
            AngularDistributionData::Monodirectional { direction } => {
                check_direction(direction)?;
                Self::Monodirectional { direction }
            }
        })
    }
}

/// Why `direction` cannot be the direction of a monodirectional source, if it
/// cannot: a component that is NaN or infinite, or the zero vector, which
/// points nowhere.
fn check_direction(direction: [f64; 3]) -> Result<(), String> {
    if let Some(i) = direction.iter().position(|c| !c.is_finite()) {
        let what = if direction[i].is_nan() {
            "a NaN"
        } else {
            "an infinite"
        };
        return Err(format!(
            "Monodirectional direction {direction:?} has {what} component ({} = {}). Every \
             component must be a finite number.",
            ["u", "v", "w"][i],
            direction[i]
        ));
    }
    // `-0.0 == 0.0`, so a vector of signed zeros is caught too.
    if direction.iter().all(|&c| c == 0.0) {
        return Err(format!(
            "Monodirectional direction {direction:?} is the zero vector, which points \
             nowhere. Give the direction the particles travel in as a nonzero vector, \
             for example (0, 0, 1) for +z."
        ));
    }
    Ok(())
}

/// `direction` scaled to unit length, once [`check_direction`] accepts it.
fn unit_vector(direction: [f64; 3]) -> Result<[f64; 3], String> {
    check_direction(direction)?;
    let [u, v, w] = direction;
    let squares = u * u + v * v + w * w;
    if squares.is_normal() {
        let mag = squares.sqrt();
        return Ok([u / mag, v / mag, w / mag]);
    }
    // The sum of squares overflowed (a component beyond about 1e154), which
    // would make the direction zero or NaN, or underflowed (every component
    // below about 1e-154), which would call a nonzero vector zero or lose its
    // precision. Scaled by its largest component first, the vector has
    // components in [-1, 1], one of them +-1, and squares safely.
    let largest = u.abs().max(v.abs()).max(w.abs());
    let [u, v, w] = [u / largest, v / largest, w / largest];
    let mag = (u * u + v * v + w * w).sqrt();
    Ok([u / mag, v / mag, w / mag])
}

impl AngularDistribution {
    /// A monodirectional distribution along `(u, v, w)`, normalized to unit
    /// length.
    ///
    /// Refuses a vector that points nowhere: the zero vector, or one with a
    /// NaN or infinite component. A vector too large or too small to square
    /// in `f64` is still normalized correctly.
    pub fn new_monodirectional(u: f64, v: f64, w: f64) -> Result<Self, String> {
        Ok(Self::Monodirectional {
            direction: unit_vector([u, v, w])?,
        })
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
