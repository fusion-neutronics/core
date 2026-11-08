/// Surface filter: the surfaces a surface-current tally scores crossings on.
///
/// A surface tally scores at the moment a particle crosses one of these
/// surfaces, rather than along tracks or at collisions. With one surface it
/// is a gate; with several the tally gains a surface-bin dimension, one bin
/// per surface in the order given.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SurfaceFilter {
    /// The surface ids this filter bins over. Always at least one element.
    pub surface_ids: Vec<usize>,
}

impl SurfaceFilter {
    /// Create a `SurfaceFilter` from a list of surface ids (one bin per id).
    ///
    /// # Panics
    /// Panics if `surface_ids` is empty.
    pub fn from_ids(surface_ids: Vec<usize>) -> Self {
        assert!(
            !surface_ids.is_empty(),
            "SurfaceFilter requires at least one surface id",
        );
        Self { surface_ids }
    }

    /// Number of surface bins this filter produces.
    pub fn num_bins(&self) -> usize {
        self.surface_ids.len()
    }

    /// The bin of a crossed surface, or `None` if it is not one of these.
    pub fn get_bin(&self, surface_id: usize) -> Option<usize> {
        self.surface_ids.iter().position(|&id| id == surface_id)
    }
}

/// Cosine filter: bins surface crossings by the cosine of the angle between
/// the particle's direction and the surface normal.
///
/// The normal points to the surface's positive side (`Surface.above`), so a
/// crossing towards that side has a positive cosine and one towards the
/// negative side a negative cosine. For a sphere the normal points outwards:
/// `[cos(θ), 1.0]` keeps particles leaving within an angle θ of the radial
/// direction, which is what a detector viewing only part of the surface sees.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CosineFilter {
    /// Ascending bin edges, within `[-1, 1]`.
    pub edges: Vec<f64>,
}

impl CosineFilter {
    /// Create a `CosineFilter` from ascending edges within `[-1, 1]`.
    pub fn new(edges: Vec<f64>) -> Result<Self, String> {
        if edges.len() < 2 {
            return Err("cosine_bins needs at least 2 edges".to_string());
        }
        if edges.windows(2).any(|w| w[1] <= w[0]) {
            return Err("cosine_bins must be strictly ascending".to_string());
        }
        if edges[0] < -1.0 || edges[edges.len() - 1] > 1.0 {
            return Err(format!(
                "cosine_bins must lie within [-1, 1], got {} to {}",
                edges[0],
                edges[edges.len() - 1]
            ));
        }
        Ok(Self { edges })
    }

    /// Number of cosine bins.
    pub fn num_bins(&self) -> usize {
        self.edges.len() - 1
    }

    /// The bin a cosine falls in, or `None` outside the edges. The top edge
    /// is inclusive so a crossing exactly along the normal (cosine 1) counts.
    pub fn get_bin(&self, mu: f64) -> Option<usize> {
        let last = self.edges.len() - 1;
        if mu < self.edges[0] || mu > self.edges[last] {
            return None;
        }
        if mu == self.edges[last] {
            return Some(last - 1);
        }
        Some(self.edges.partition_point(|&e| e <= mu) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_filter_bins_in_order() {
        let f = SurfaceFilter::from_ids(vec![7, 3]);
        assert_eq!(f.num_bins(), 2);
        assert_eq!(f.get_bin(3), Some(1));
        assert_eq!(f.get_bin(4), None);
    }

    #[test]
    fn cosine_filter_bins_and_edges() {
        let f = CosineFilter::new(vec![-1.0, 0.0, 0.5, 1.0]).unwrap();
        assert_eq!(f.num_bins(), 3);
        assert_eq!(f.get_bin(-1.0), Some(0));
        assert_eq!(f.get_bin(-0.2), Some(0));
        assert_eq!(f.get_bin(0.0), Some(1));
        assert_eq!(f.get_bin(0.7), Some(2));
        assert_eq!(f.get_bin(1.0), Some(2));
        let partial = CosineFilter::new(vec![0.9, 1.0]).unwrap();
        assert_eq!(partial.get_bin(0.5), None);
    }

    #[test]
    fn cosine_filter_refuses_bad_edges() {
        assert!(CosineFilter::new(vec![0.5]).is_err());
        assert!(CosineFilter::new(vec![0.5, 0.2]).is_err());
        assert!(CosineFilter::new(vec![-1.5, 1.0]).is_err());
    }
}
