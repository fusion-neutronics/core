use crate::distribution::energy::{Discrete, Normal, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Why `xyz` cannot be the point `what` names (a source position, a ring
/// origin), if it cannot: a coordinate that is NaN or infinite.
pub fn check_position(what: &str, xyz: [f64; 3]) -> Result<(), String> {
    if let Some(i) = xyz.iter().position(|c| !c.is_finite()) {
        let kind = if xyz[i].is_nan() {
            "a NaN"
        } else {
            "an infinite"
        };
        return Err(format!(
            "{what} {xyz:?} has {kind} coordinate ({} = {}). Every coordinate must be a \
             finite number of cm.",
            ["x", "y", "z"][i],
            xyz[i]
        ));
    }
    Ok(())
}

/// Point spatial distribution - a fixed position in 3D space
///
/// [`Point::new`] takes the position as given, for positions computed in
/// code. One given by a user is checked where it enters, with
/// [`check_position`]: in the Python source constructors, and as a model file
/// is read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PointData")]
pub struct Point {
    // Position coordinates
    xyz: [f64; 3],
}

#[derive(Deserialize)]
struct PointData {
    xyz: [f64; 3],
}

impl TryFrom<PointData> for Point {
    type Error = String;

    fn try_from(data: PointData) -> Result<Self, String> {
        check_position("position", data.xyz)?;
        Ok(Point::new(data.xyz))
    }
}

impl Point {
    /// Create a new point at the specified position
    pub fn new(xyz: [f64; 3]) -> Self {
        Self { xyz }
    }

    /// Sample a position from this distribution
    /// For Point, this always returns the same position (deterministic)
    pub fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> [f64; 3] {
        self.xyz
    }

    /// Get the position of this point
    pub fn position(&self) -> [f64; 3] {
        self.xyz
    }
}

impl Default for Point {
    fn default() -> Self {
        Self {
            xyz: [0.0, 0.0, 0.0],
        }
    }
}

/// A univariate distribution used for cylindrical coordinate components.
/// Reuses the same distribution types as energy distributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Univariate {
    Discrete(Discrete),
    Uniform(Uniform),
    Normal(Normal),
}

impl Univariate {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Univariate::Discrete(d) => d.sample(rng),
            Univariate::Uniform(u) => u.sample(rng),
            Univariate::Normal(n) => n.sample(rng),
        }
    }
}

/// Cylindrical independent spatial distribution -- samples r, phi, z from
/// independent univariate distributions in a cylindrical coordinate system.
///
/// The cylindrical axis is always Z: positions are computed as
/// `x = origin.x + r*cos(phi)`, `y = origin.y + r*sin(phi)`,
/// `z = origin.z + z_sample`.
///
/// A model file is read through [`CylindricalRing::new`], so it is checked as
/// a constructed one is; `r`, `phi` and `z` are checked by their own types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CylindricalRingData")]
pub struct CylindricalRing {
    /// Distribution of r-coordinates (radial distance from axis).
    pub r: Univariate,
    /// Distribution of phi-coordinates (azimuthal angle in radians).
    pub phi: Univariate,
    /// Distribution of z-coordinates (axial position).
    pub z: Univariate,
    /// Origin of the cylindrical reference frame [x, y, z] in cm.
    pub origin: [f64; 3],
}

#[derive(Deserialize)]
struct CylindricalRingData {
    r: Univariate,
    phi: Univariate,
    z: Univariate,
    origin: [f64; 3],
}

impl TryFrom<CylindricalRingData> for CylindricalRing {
    type Error = String;

    fn try_from(data: CylindricalRingData) -> Result<Self, String> {
        CylindricalRing::new(data.r, data.phi, data.z, data.origin)
    }
}

impl CylindricalRing {
    /// A ring about `origin`, refused when `origin` has a NaN or infinite
    /// coordinate (see [`check_position`]).
    pub fn new(
        r: Univariate,
        phi: Univariate,
        z: Univariate,
        origin: [f64; 3],
    ) -> Result<Self, String> {
        check_position("CylindricalRing origin", origin)?;
        Ok(Self { r, phi, z, origin })
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        let r_val = self.r.sample(rng);
        let phi_val = self.phi.sample(rng);
        let z_val = self.z.sample(rng);
        let (sin_phi, cos_phi) = phi_val.sin_cos();
        [
            self.origin[0] + r_val * cos_phi,
            self.origin[1] + r_val * sin_phi,
            self.origin[2] + z_val,
        ]
    }
}
