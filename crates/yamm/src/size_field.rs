/// A regular grid of target edge lengths and metric data over UV space.
///
/// Stores first-fundamental-form metric tensor components and curvature-based
/// target edge lengths.  The CDT refinement loop interpolates this grid to
/// determine local targets and to measure UV edges in 3D.
///
/// # Anisotropic size field (optional)
///
/// For surfaces with anisotropic curvature (e.g. an ellipsoid where curvature
/// differs between the equator and the poles), the mesh should be finer in the
/// high-curvature direction and coarser in the low-curvature direction.  The
/// optional fields `target_h1`, `target_h2`, and `curvature_angle` store
/// per-grid-point principal-curvature sizing information.  When all three are
/// `Some`, the anisotropic edge-length query [`edge_length_anisotropic`] can
/// measure an edge's length relative to the directional target sizes.
///
/// [`edge_length_anisotropic`]: SizeField::edge_length_anisotropic
#[derive(Clone, Debug)]
pub struct SizeField {
    /// Grid points in U direction.
    pub nu: usize,
    /// Grid points in V direction.
    pub nv: usize,
    /// UV domain bounds.
    pub u_min: f64,
    pub u_max: f64,
    pub v_min: f64,
    pub v_max: f64,
    /// Target 3D edge length at each grid point (row-major, length = nu * nv).
    pub target_h: Vec<f64>,
    /// First fundamental form E = |∂S/∂u|² at each grid point.
    pub metric_e: Vec<f64>,
    /// First fundamental form F = (∂S/∂u · ∂S/∂v) at each grid point.
    pub metric_f: Vec<f64>,
    /// First fundamental form G = |∂S/∂v|² at each grid point.
    pub metric_g: Vec<f64>,

    // ── Anisotropic curvature sizing (optional) ──
    /// Target size in the max-curvature (κ₁) direction at each grid point.
    ///
    /// When present, edges aligned with the max-curvature direction should
    /// have 3D length close to this value.  Must have length `nu * nv`.
    pub target_h1: Option<Vec<f64>>,
    /// Target size in the min-curvature (κ₂) direction at each grid point.
    ///
    /// When present, edges aligned with the min-curvature direction should
    /// have 3D length close to this value.  Must have length `nu * nv`.
    pub target_h2: Option<Vec<f64>>,
    /// Angle (radians) of the max-curvature direction relative to the ∂S/∂u
    /// tangent vector at each grid point.
    ///
    /// The max-curvature direction in the UV tangent plane is:
    ///   d₁ = cos(θ) · ∂S/∂u + sin(θ) · ∂S/∂v
    /// and the min-curvature direction is orthogonal to it (θ + π/2).
    pub curvature_angle: Option<Vec<f64>>,
}

impl SizeField {
    /// Bilinearly interpolate a grid value at (u, v).
    pub fn interp_pub(&self, grid: &[f64], u: f64, v: f64) -> f64 {
        self.interp(grid, u, v)
    }

    /// Bilinearly interpolate a grid value at (u, v).
    fn interp(&self, grid: &[f64], u: f64, v: f64) -> f64 {
        let fu = ((u - self.u_min) / (self.u_max - self.u_min) * (self.nu - 1) as f64)
            .clamp(0.0, (self.nu - 1) as f64);
        let fv = ((v - self.v_min) / (self.v_max - self.v_min) * (self.nv - 1) as f64)
            .clamp(0.0, (self.nv - 1) as f64);

        let iu = fu as usize;
        let iv = fv as usize;
        let iu1 = (iu + 1).min(self.nu - 1);
        let iv1 = (iv + 1).min(self.nv - 1);
        let su = fu - iu as f64;
        let sv = fv - iv as f64;

        let v00 = grid[iv * self.nu + iu];
        let v10 = grid[iv * self.nu + iu1];
        let v01 = grid[iv1 * self.nu + iu];
        let v11 = grid[iv1 * self.nu + iu1];

        (1.0 - su) * (1.0 - sv) * v00
            + su * (1.0 - sv) * v10
            + (1.0 - su) * sv * v01
            + su * sv * v11
    }

    /// Interpolate the target 3D edge length at a UV point.
    pub fn target_h_at(&self, u: f64, v: f64) -> f64 {
        self.interp(&self.target_h, u, v)
    }

    /// Interpolate the metric tensor (E, F, G) at a UV point.
    pub fn metric_at(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let e = self.interp(&self.metric_e, u, v);
        let f = self.interp(&self.metric_f, u, v);
        let g = self.interp(&self.metric_g, u, v);
        (e, f, g)
    }

    /// Compute the 3D length of a UV edge using the metric tensor at its midpoint.
    ///
    /// For a UV displacement (du, dv), the 3D length² = E·du² + 2F·du·dv + G·dv².
    pub fn edge_length_3d(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let du = b[0] - a[0];
        let dv = b[1] - a[1];
        let um = (a[0] + b[0]) * 0.5;
        let vm = (a[1] + b[1]) * 0.5;

        let e = self.interp(&self.metric_e, um, vm);
        let f = self.interp(&self.metric_f, um, vm);
        let g = self.interp(&self.metric_g, um, vm);

        let len_sq = e * du * du + 2.0 * f * du * dv + g * dv * dv;
        if len_sq > 0.0 {
            len_sq.sqrt()
        } else {
            0.0
        }
    }

    /// Whether the anisotropic curvature fields are fully populated.
    pub fn has_anisotropic_field(&self) -> bool {
        self.target_h1.is_some() && self.target_h2.is_some() && self.curvature_angle.is_some()
    }

    /// Compute the anisotropic edge-length ratios for a UV edge.
    ///
    /// Returns `(ratio_h1, ratio_h2)` where each ratio is the edge's 3D
    /// length projected onto the corresponding principal curvature direction,
    /// divided by that direction's target size.  A ratio of 1.0 means the
    /// edge is exactly the desired length in that principal direction.
    ///
    /// # How it works
    ///
    /// 1. Evaluate the metric tensor (E, F, G) and the curvature angle θ at
    ///    the edge midpoint.
    /// 2. Define the principal curvature directions in UV parameter space:
    ///    d₁ = (cos θ, sin θ)  (max curvature direction)
    ///    d₂ = (−sin θ, cos θ) (min curvature direction)
    /// 3. Project the edge's UV displacement onto d₁ and d₂ using the
    ///    metric inner product:
    ///    ⟨(a_u, a_v), (b_u, b_v)⟩ = E·a_u·b_u + F·(a_u·b_v + a_v·b_u) + G·a_v·b_v
    /// 4. Normalise by the metric norm of each direction to get true 3D
    ///    projected lengths, then divide by `target_h1` / `target_h2`.
    ///
    /// Returns `None` if the anisotropic fields are not populated.
    pub fn edge_length_anisotropic(&self, a: [f64; 2], b: [f64; 2]) -> Option<(f64, f64)> {
        let h1_grid = self.target_h1.as_ref()?;
        let h2_grid = self.target_h2.as_ref()?;
        let angle_grid = self.curvature_angle.as_ref()?;

        let du = b[0] - a[0];
        let dv = b[1] - a[1];
        let um = (a[0] + b[0]) * 0.5;
        let vm = (a[1] + b[1]) * 0.5;

        // Metric tensor at midpoint
        let e = self.interp(&self.metric_e, um, vm);
        let f = self.interp(&self.metric_f, um, vm);
        let g = self.interp(&self.metric_g, um, vm);

        // Curvature angle and target sizes at midpoint
        let theta = self.interp(angle_grid, um, vm);
        let h1 = self.interp(h1_grid, um, vm).max(1e-30);
        let h2 = self.interp(h2_grid, um, vm).max(1e-30);

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // d₁ = (cos θ, sin θ), d₂ = (−sin θ, cos θ), edge = (du, dv)
        //
        // ⟨edge, d₁⟩ = E·du·cos θ + F·(du·sin θ + dv·cos θ) + G·dv·sin θ
        let dot1 = e * du * cos_t + f * (du * sin_t + dv * cos_t) + g * dv * sin_t;

        // ⟨edge, d₂⟩ = E·du·(−sin θ) + F·(du·cos θ + dv·(−sin θ)) + G·dv·cos θ
        let dot2 = -e * du * sin_t + f * (du * cos_t - dv * sin_t) + g * dv * cos_t;

        // |d₁|² = E cos²θ + 2F cos θ sin θ + G sin²θ
        let norm1_sq = e * cos_t * cos_t + 2.0 * f * cos_t * sin_t + g * sin_t * sin_t;
        // |d₂|² = E sin²θ − 2F sin θ cos θ + G cos²θ
        let norm2_sq = e * sin_t * sin_t - 2.0 * f * sin_t * cos_t + g * cos_t * cos_t;

        // Projected 3D length = |⟨edge, dᵢ⟩| / |dᵢ|
        let proj_len1 = if norm1_sq > 1e-30 {
            dot1.abs() / norm1_sq.sqrt()
        } else {
            0.0
        };
        let proj_len2 = if norm2_sq > 1e-30 {
            dot2.abs() / norm2_sq.sqrt()
        } else {
            0.0
        };

        Some((proj_len1 / h1, proj_len2 / h2))
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_field(val: f64) -> SizeField {
        // 3x3 grid, unit square, uniform values
        SizeField {
            nu: 3,
            nv: 3,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![val; 9],
            metric_e: vec![1.0; 9],
            metric_f: vec![0.0; 9],
            metric_g: vec![1.0; 9],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        }
    }

    #[test]
    fn interp_at_grid_point() {
        let sf = uniform_field(2.0);
        assert!((sf.target_h_at(0.0, 0.0) - 2.0).abs() < 1e-12);
        assert!((sf.target_h_at(1.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((sf.target_h_at(0.5, 0.5) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn interp_linear() {
        let mut sf = uniform_field(0.0);
        // Set corner values: f(u,v) = u + v
        for iv in 0..3 {
            for iu in 0..3 {
                sf.target_h[iv * 3 + iu] = iu as f64 * 0.5 + iv as f64 * 0.5;
            }
        }
        assert!((sf.target_h_at(0.5, 0.5) - 1.0).abs() < 1e-12);
        assert!((sf.target_h_at(0.25, 0.75) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn edge_length_euclidean() {
        // With identity metric (E=1, F=0, G=1), 3D length = UV length
        let sf = uniform_field(1.0);
        let len = sf.edge_length_3d([0.0, 0.0], [3.0, 4.0]);
        assert!((len - 5.0).abs() < 1e-10);
    }

    #[test]
    fn edge_length_scaled() {
        // With E=4, F=0, G=9: length of (1,0) = 2, length of (0,1) = 3
        let sf = SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![4.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![9.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };
        assert!((sf.edge_length_3d([0.0, 0.0], [1.0, 0.0]) - 2.0).abs() < 1e-10);
        assert!((sf.edge_length_3d([0.0, 0.0], [0.0, 1.0]) - 3.0).abs() < 1e-10);
    }

    // ── Anisotropic field tests ───────────────────────────

    #[test]
    fn has_anisotropic_field_false_by_default() {
        let sf = uniform_field(1.0);
        assert!(!sf.has_anisotropic_field());
    }

    #[test]
    fn has_anisotropic_field_true_when_set() {
        let mut sf = uniform_field(1.0);
        sf.target_h1 = Some(vec![0.5; 9]);
        sf.target_h2 = Some(vec![2.0; 9]);
        sf.curvature_angle = Some(vec![0.0; 9]);
        assert!(sf.has_anisotropic_field());
    }

    #[test]
    fn has_anisotropic_field_partial_is_false() {
        let mut sf = uniform_field(1.0);
        sf.target_h1 = Some(vec![0.5; 9]);
        // target_h2 and curvature_angle still None
        assert!(!sf.has_anisotropic_field());
    }

    #[test]
    fn edge_length_anisotropic_returns_none_without_fields() {
        let sf = uniform_field(1.0);
        assert!(sf.edge_length_anisotropic([0.0, 0.0], [1.0, 0.0]).is_none());
    }

    /// Build an anisotropic size field with identity metric (E=1, F=0, G=1),
    /// curvature angle = 0 (d₁ aligned with du), and specified h1, h2.
    fn anisotropic_field(h1: f64, h2: f64) -> SizeField {
        SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: Some(vec![h1; 4]),
            target_h2: Some(vec![h2; 4]),
            curvature_angle: Some(vec![0.0; 4]),
        }
    }

    #[test]
    fn anisotropic_edge_along_d1_direction() {
        // d₁ = (1, 0) when θ=0.  With identity metric, an edge along u of
        // length 1.0 should give ratio_h1 = 1.0/h1, ratio_h2 = 0.0/h2 = 0.
        let sf = anisotropic_field(0.5, 2.0);
        let (r1, r2) = sf.edge_length_anisotropic([0.0, 0.5], [1.0, 0.5]).unwrap();

        // Edge is purely along d₁: projected length onto d₁ = 1.0, onto d₂ = 0.0
        assert!(
            (r1 - 1.0 / 0.5).abs() < 1e-10,
            "ratio_h1 should be 2.0, got {r1}"
        );
        assert!(
            r2.abs() < 1e-10,
            "ratio_h2 should be ~0 for edge along d₁, got {r2}"
        );
    }

    #[test]
    fn anisotropic_edge_along_d2_direction() {
        // d₂ = (0, 1) when θ=0.  An edge along v should project entirely
        // onto d₂.
        let sf = anisotropic_field(0.5, 2.0);
        let (r1, r2) = sf.edge_length_anisotropic([0.5, 0.0], [0.5, 1.0]).unwrap();

        assert!(
            r1.abs() < 1e-10,
            "ratio_h1 should be ~0 for edge along d₂, got {r1}"
        );
        assert!(
            (r2 - 1.0 / 2.0).abs() < 1e-10,
            "ratio_h2 should be 0.5, got {r2}"
        );
    }

    #[test]
    fn anisotropic_edge_diagonal_splits_evenly() {
        // With identity metric, θ=0, and equal target sizes h1=h2=1.0,
        // a diagonal edge of UV length sqrt(2) should project equally:
        //   projection onto d₁ = 1.0, projection onto d₂ = 1.0
        //   ratio_h1 = 1.0/1.0 = 1.0, ratio_h2 = 1.0/1.0 = 1.0
        let sf = anisotropic_field(1.0, 1.0);
        let (r1, r2) = sf.edge_length_anisotropic([0.0, 0.0], [1.0, 1.0]).unwrap();

        assert!((r1 - 1.0).abs() < 1e-10, "ratio_h1 should be 1.0, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-10, "ratio_h2 should be 1.0, got {r2}");
    }

    #[test]
    fn anisotropic_with_rotated_curvature() {
        // Rotate curvature direction by π/2: now d₁ = (0, 1), d₂ = (1, 0)
        // An edge along u should project entirely onto d₂.
        let sf = SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: Some(vec![0.5; 4]),
            target_h2: Some(vec![2.0; 4]),
            curvature_angle: Some(vec![std::f64::consts::FRAC_PI_2; 4]),
        };

        let (r1, r2) = sf.edge_length_anisotropic([0.0, 0.5], [1.0, 0.5]).unwrap();

        // d₁ = (cos(π/2), sin(π/2)) = (0, 1), d₂ = (−sin(π/2), cos(π/2)) = (−1, 0)
        // Edge along u: projection onto d₁ = 0, projection onto d₂ = 1.0
        assert!(
            r1.abs() < 1e-10,
            "ratio_h1 should be ~0 for edge perpendicular to d₁, got {r1}"
        );
        assert!(
            (r2 - 1.0 / 2.0).abs() < 1e-10,
            "ratio_h2 should be 0.5, got {r2}"
        );
    }

    #[test]
    fn anisotropic_strongly_different_targets() {
        // h1=0.5 (fine in max-curvature), h2=2.0 (coarse in min-curvature)
        // Edge of length 1.0 along d₁ gives ratio_h1 = 2.0 (too long, should split)
        // Edge of length 1.0 along d₂ gives ratio_h2 = 0.5 (too short, could collapse)
        let sf = anisotropic_field(0.5, 2.0);

        let (r1_u, _) = sf.edge_length_anisotropic([0.0, 0.5], [1.0, 0.5]).unwrap();
        let (_, r2_v) = sf.edge_length_anisotropic([0.5, 0.0], [0.5, 1.0]).unwrap();

        assert!(
            r1_u > 1.4,
            "edge along d₁ should exceed split threshold, ratio_h1 = {r1_u}"
        );
        assert!(
            r2_v < 0.7,
            "edge along d₂ should be below collapse threshold, ratio_h2 = {r2_v}"
        );
    }

    #[test]
    fn anisotropic_with_scaled_metric() {
        // E=4, G=1, F=0, θ=0, h1=1.0, h2=1.0
        // Edge along u of UV length 1: 3D length = 2.0 (because sqrt(E)=2)
        // Projection onto d₁ (which is du direction) = 2.0
        // ratio_h1 = 2.0/1.0 = 2.0
        let sf = SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![4.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: Some(vec![1.0; 4]),
            target_h2: Some(vec![1.0; 4]),
            curvature_angle: Some(vec![0.0; 4]),
        };

        let (r1, r2) = sf.edge_length_anisotropic([0.0, 0.5], [1.0, 0.5]).unwrap();

        // 3D length of edge = sqrt(E) * du = 2.0
        // Projected onto d₁ = (1,0): dot = E*du*1 = 4, norm_d1 = sqrt(E) = 2
        // proj_len = 4/2 = 2.0
        assert!((r1 - 2.0).abs() < 1e-10, "ratio_h1 should be 2.0, got {r1}");
        assert!(r2.abs() < 1e-10, "ratio_h2 should be ~0, got {r2}");
    }
}
