use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub lower_left: [f64; 3],
    pub upper_right: [f64; 3],
}

impl BoundingBox {
    pub fn new(lower_left: [f64; 3], upper_right: [f64; 3]) -> Self {
        BoundingBox {
            lower_left,
            upper_right,
        }
    }

    /// Geometric center of the bounding box.
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.lower_left[0] + self.upper_right[0]),
            0.5 * (self.lower_left[1] + self.upper_right[1]),
            0.5 * (self.lower_left[2] + self.upper_right[2]),
        ]
    }

    /// Extent of the bounding box along each axis.
    pub fn width(&self) -> [f64; 3] {
        [
            self.upper_right[0] - self.lower_left[0],
            self.upper_right[1] - self.lower_left[1],
            self.upper_right[2] - self.lower_left[2],
        ]
    }

    /// Compute the volume of the bounding box (product of widths).
    pub fn volume(&self) -> f64 {
        let w = self.width();
        w[0] * w[1] * w[2]
    }

    /// Check if all corners are finite (no infinite extents).
    pub fn is_finite(&self) -> bool {
        self.lower_left.iter().all(|v| v.is_finite())
            && self.upper_right.iter().all(|v| v.is_finite())
    }

    pub fn expand_to_include(&mut self, other: &BoundingBox) {
        for i in 0..3 {
            self.lower_left[i] = self.lower_left[i].min(other.lower_left[i]);
            self.upper_right[i] = self.upper_right[i].max(other.upper_right[i]);
        }
    }

    /// The canonical empty box (`lower = +inf`, `upper = -inf`): encloses no
    /// point, and is the identity for [`union`](Self::union) and the absorbing
    /// element for [`intersection`](Self::intersection). Also what a CSG region
    /// with no finite extent (unbounded / genuinely empty) reports.
    pub fn empty() -> Self {
        BoundingBox::new([f64::INFINITY; 3], [f64::NEG_INFINITY; 3])
    }

    /// Tightest box contained in BOTH boxes (per-axis `max` of lowers, `min` of
    /// uppers). If the result is inverted on any axis (the boxes are disjoint),
    /// returns the canonical [`empty`](Self::empty) box so downstream
    /// `is_finite` / union / intersection stay well-defined. This is the AABB of
    /// a CSG `Intersection`.
    pub fn intersection(&self, other: &BoundingBox) -> Self {
        let mut lower = [0.0; 3];
        let mut upper = [0.0; 3];
        for i in 0..3 {
            lower[i] = self.lower_left[i].max(other.lower_left[i]);
            upper[i] = self.upper_right[i].min(other.upper_right[i]);
        }
        if lower[0] > upper[0] || lower[1] > upper[1] || lower[2] > upper[2] {
            return BoundingBox::empty();
        }
        BoundingBox::new(lower, upper)
    }

    /// Smallest box ENCLOSING both boxes (per-axis `min` of lowers, `max` of
    /// uppers). This is the (conservative) AABB of a CSG `Union`: a union may be
    /// disconnected, so its exact hull is the enclosing box of the parts. The
    /// [`empty`](Self::empty) box is the identity (`empty ∪ b == b`).
    pub fn union(&self, other: &BoundingBox) -> Self {
        let mut lower = [0.0; 3];
        let mut upper = [0.0; 3];
        for i in 0..3 {
            lower[i] = self.lower_left[i].min(other.lower_left[i]);
            upper[i] = self.upper_right[i].max(other.upper_right[i]);
        }
        BoundingBox::new(lower, upper)
    }
}
