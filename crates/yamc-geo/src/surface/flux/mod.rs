//! Free-form toroidal surfaces: a boundary given by a double Fourier
//! series (DESC's product convention, see [`series`]) and the level sets
//! of the signed 3-D distance to it.
//!
//! A [`FluxBoundary`] is the boundary itself. A flux surface uses it with
//! offset 0; an offset surface is the set of points at signed distance
//! `offset` (cm) from it, positive outwards, so layers built from one
//! boundary are nested by construction and never self-intersect (past
//! the radius where outward normals cross, the level set is trimmed with
//! a crease, as a true distance offset is).
//!
//! The sign of the distance is decided exactly in the point's own
//! toroidal cut; its magnitude is the 3-D distance to the closest surface
//! point, found by Newton's method and certified to be the global one
//! (see [`distance`]). Ray crossings march with steps that cannot pass a
//! root (see [`ray`]).

mod distance;
mod patches;
mod ray;
mod series;
#[cfg(test)]
mod tests;

use crate::surface::{BoundaryType, Surface, SurfaceKind};
use distance::MAX_POLY;
use patches::Patches;
use serde::{Deserialize, Serialize};
use series::{Series, MAX_M, MAX_N};
use std::f64::consts::TAU;
use std::sync::Arc;

/// A closed toroidal boundary `R(theta, zeta)`, `Z(theta, zeta)` given by
/// Fourier modes `(m, n)` in DESC's product convention (see the module
/// docs), lengths in cm, `zeta` the cylindrical toroidal angle.
///
/// [`FluxBoundary::new`] checks the geometry: `R > 0` everywhere and every
/// toroidal cut a simple closed curve with the same orientation. The
/// distance machinery is rebuilt (and the checks rerun) when a model is
/// loaded; only the modes are stored.
pub struct FluxBoundary {
    num_field_periods: u32,
    r_modes: Vec<[i32; 2]>,
    r_mn: Vec<f64>,
    z_modes: Vec<[i32; 2]>,
    z_mn: Vec<f64>,
    engine: Engine,
}

impl std::fmt::Debug for FluxBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FluxBoundary")
            .field("num_field_periods", &self.num_field_periods)
            .field("r_modes", &self.r_modes.len())
            .field("z_modes", &self.z_modes.len())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct FluxBoundaryRef<'a> {
    num_field_periods: u32,
    r_modes: &'a [[i32; 2]],
    r_mn: &'a [f64],
    z_modes: &'a [[i32; 2]],
    z_mn: &'a [f64],
}

#[derive(Deserialize)]
struct FluxBoundaryData {
    num_field_periods: u32,
    r_modes: Vec<[i32; 2]>,
    r_mn: Vec<f64>,
    z_modes: Vec<[i32; 2]>,
    z_mn: Vec<f64>,
}

impl Serialize for FluxBoundary {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FluxBoundaryRef {
            num_field_periods: self.num_field_periods,
            r_modes: &self.r_modes,
            r_mn: &self.r_mn,
            z_modes: &self.z_modes,
            z_mn: &self.z_mn,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FluxBoundary {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let d = FluxBoundaryData::deserialize(deserializer)?;
        Self::new(d.num_field_periods, d.r_modes, d.r_mn, d.z_modes, d.z_mn)
            .map_err(serde::de::Error::custom)
    }
}

impl FluxBoundary {
    /// Build and check a boundary. `r_modes` / `z_modes` are `(m, n)` pairs
    /// matching the coefficients `r_mn` / `z_mn` (cm).
    pub fn new(
        num_field_periods: u32,
        r_modes: Vec<[i32; 2]>,
        r_mn: Vec<f64>,
        z_modes: Vec<[i32; 2]>,
        z_mn: Vec<f64>,
    ) -> Result<Self, String> {
        if num_field_periods == 0 {
            return Err("num_field_periods must be at least 1".to_string());
        }
        for (name, modes, values) in [("r", &r_modes, &r_mn), ("z", &z_modes, &z_mn)] {
            if modes.is_empty() || modes.len() != values.len() {
                return Err(format!(
                    "{name}_modes and {name}_mn must be non-empty and the same length (got {} \
                     and {})",
                    modes.len(),
                    values.len()
                ));
            }
            for &[m, n] in modes.iter() {
                if m.unsigned_abs() as usize > MAX_M || n.unsigned_abs() as usize > MAX_N {
                    return Err(format!(
                        "{name}_modes has (m, n) = ({m}, {n}); |m| up to {MAX_M} and |n| up to \
                         {MAX_N} are supported"
                    ));
                }
            }
            let mut sorted = modes.clone();
            sorted.sort_unstable();
            if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
                return Err(format!("{name}_modes lists {:?} more than once", w[0]));
            }
            if let Some(v) = values.iter().find(|v| !v.is_finite()) {
                return Err(format!("{name}_mn has a non-finite coefficient {v}"));
            }
        }
        let series = Series::new(num_field_periods, &r_modes, &r_mn, &z_modes, &z_mn);
        let engine = Engine::new(series)?;
        Ok(Self {
            num_field_periods,
            r_modes,
            r_mn,
            z_modes,
            z_mn,
            engine,
        })
    }

    pub fn num_field_periods(&self) -> u32 {
        self.num_field_periods
    }

    pub fn r_modes(&self) -> &[[i32; 2]] {
        &self.r_modes
    }

    pub fn r_mn(&self) -> &[f64] {
        &self.r_mn
    }

    pub fn z_modes(&self) -> &[[i32; 2]] {
        &self.z_modes
    }

    pub fn z_mn(&self) -> &[f64] {
        &self.z_mn
    }

    /// Signed distance (cm) from `point` to the boundary, positive outside.
    pub fn signed_distance(&self, point: [f64; 3]) -> f64 {
        self.engine.signed(point).d
    }

    /// Signed distance (cm) and the `(theta, zeta)` of the closest
    /// boundary point.
    pub fn closest_point(&self, point: [f64; 3]) -> (f64, f64, f64) {
        let s = self.engine.signed(point);
        (s.d, s.theta, s.zeta)
    }

    /// The boundary point `(R cos zeta, R sin zeta, Z)` at `(theta, zeta)`.
    pub fn point(&self, theta: f64, zeta: f64) -> [f64; 3] {
        self.engine.series.point(theta, zeta).x
    }

    /// Outward unit normal of the boundary at `(theta, zeta)`.
    pub fn normal(&self, theta: f64, zeta: f64) -> [f64; 3] {
        self.engine.normal(&self.engine.series.point(theta, zeta))
    }

    /// Largest inward offset that keeps the layer free of folds: inside
    /// the smallest radius of curvature of the convex parts.
    pub fn inner_reach(&self) -> f64 {
        if self.engine.patches.max_convex > 0.0 {
            1.0 / self.engine.patches.max_convex
        } else {
            f64::INFINITY
        }
    }

    /// Refuse an offset whose level set would fold inwards.
    pub fn check_offset(&self, offset: f64) -> Result<(), String> {
        if !offset.is_finite() {
            return Err(format!("the offset distance must be finite (got {offset})"));
        }
        let reach = self.inner_reach();
        if offset < 0.0 && -offset >= 0.9 * reach {
            return Err(format!(
                "an inward offset of {:.6} cm reaches the boundary's inner fold distance \
                 (about {reach:.6} cm), where the offset layer would crease or vanish; describe \
                 surfaces that deep inside the plasma with FluxSurface(shape, rho < 1) instead",
                -offset
            ));
        }
        Ok(())
    }

    /// `D(point) - offset`: the sign is exact; the magnitude is exact near
    /// the level set and otherwise a rigorous lower bound.
    pub(crate) fn evaluate(&self, point: [f64; 3], offset: f64) -> f64 {
        let e = &self.engine;
        let (cut, lo, hi) = e.bounds(point);
        if lo - offset > e.margin {
            return lo - offset;
        }
        if hi - offset < -e.margin {
            return hi - offset;
        }
        e.signed_from(point, &cut).d - offset
    }

    pub(crate) fn distance(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        offset: f64,
        on_surface: Option<f64>,
        vacuum: bool,
    ) -> Option<f64> {
        self.engine
            .ray(point, direction, offset, on_surface, vacuum)
    }

    /// Box containing the points within `offset` of the boundary (for a
    /// positive offset) or inside it.
    pub(crate) fn bounding_box(&self, offset: f64) -> ([f64; 3], [f64; 3]) {
        let mut bx = self.engine.patches.extents;
        bx.inflate(offset.max(0.0));
        (bx.lo, bx.hi)
    }
}

/// The distance machinery derived from the modes.
pub(super) struct Engine {
    series: Series,
    patches: Patches,
    /// +1 when the cuts run counter-clockwise in the `(R, Z)` plane.
    orientation_in: f64,
    /// Sign making `orientation_3d * (X_theta x X_zeta)` the outward normal.
    orientation_3d: f64,
    n_poly: usize,
    dtheta_poly: f64,
    /// `cos(m theta_k)` / `sin(m theta_k)` on the side-test polyline,
    /// row `k`, column `m`.
    poly_cos: Vec<f64>,
    poly_sin: Vec<f64>,
    /// Floating-point allowance (cm), far below the 1e-8 cm crossing nudge.
    margin: f64,
    /// Newton stops when the closest point moves less than this (cm).
    newton_tolerance: f64,
    /// Largest curvature of either sign anywhere on the surface.
    kappa_all: f64,
}

impl Engine {
    fn new(series: Series) -> Result<Self, String> {
        let scale = series.bounds.r + series.bounds.z;
        if !(scale.is_finite() && scale > 0.0) {
            return Err("the boundary has no extent: every coefficient is zero".to_string());
        }
        let margin = 1e-12 * scale;
        let m_cols = series.m_max + 1;
        let n_poly = (16 * m_cols).clamp(64, MAX_POLY);
        let dtheta_poly = TAU / n_poly as f64;
        let mut poly_cos = vec![0.0; n_poly * m_cols];
        let mut poly_sin = vec![0.0; n_poly * m_cols];
        for k in 0..n_poly {
            series::harmonics(
                k as f64 * dtheta_poly,
                series.m_max,
                &mut poly_cos[k * m_cols..(k + 1) * m_cols],
                &mut poly_sin[k * m_cols..(k + 1) * m_cols],
            );
        }
        let orientation_in = check_cuts(&series)?;
        let orientation_3d = -orientation_in;
        let nt = (8 * m_cols).max(32);
        let nz = series.nfp as usize * (4 * (series.n_max + 1)).max(8);
        let patches = Patches::new(&series, nt, nz, orientation_3d, margin);
        if patches.min_radius() <= 0.0 {
            return Err(
                "the boundary comes within its sampling tolerance of the z axis (R <= 0 \
                 somewhere); check the coefficients (they should be in cm)"
                    .to_string(),
            );
        }
        let kappa_all = patches.max_convex.max(patches.max_concave);
        Ok(Self {
            newton_tolerance: 1e-11 * scale.max(1.0) / 1000.0_f64.max(1.0),
            series,
            patches,
            orientation_in,
            orientation_3d,
            n_poly,
            dtheta_poly,
            poly_cos,
            poly_sin,
            margin,
            kappa_all,
        })
    }
}

/// Check that every toroidal cut over a field period is a simple closed
/// curve and that all run the same way round; return that orientation
/// (+1 counter-clockwise in `(R, Z)`).
fn check_cuts(series: &Series) -> Result<f64, String> {
    let n = (32 * (series.m_max + 1)).clamp(128, 1024);
    let cuts = 8 * (series.n_max + 1);
    let mut orientation = 0.0;
    for c in 0..cuts {
        let zeta = c as f64 * TAU / (series.nfp * cuts as f64);
        let cut = series.collapse(zeta, 1);
        let points: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let p = cut.curve(k as f64 * TAU / n as f64);
                (p[0], p[1])
            })
            .collect();
        let area: f64 = (0..n)
            .map(|k| {
                let (a, b) = (points[k], points[(k + 1) % n]);
                a.0 * b.1 - b.0 * a.1
            })
            .sum::<f64>()
            / 2.0;
        if area == 0.0 || !area.is_finite() {
            return Err(format!(
                "the cut at zeta = {zeta:.6} rad encloses no area: the boundary is degenerate"
            ));
        }
        let sign = area.signum();
        if orientation == 0.0 {
            orientation = sign;
        } else if sign != orientation {
            return Err(format!(
                "the cut at zeta = {zeta:.6} rad runs the other way round from the first cut: \
                 the boundary turns inside out"
            ));
        }
        if let Some((a, b)) = self_intersection(&points) {
            return Err(format!(
                "the cut at zeta = {zeta:.6} rad crosses itself (between theta = {:.6} and {:.6} \
                 rad): every cut must be a simple closed curve",
                a as f64 * TAU / n as f64,
                b as f64 * TAU / n as f64
            ));
        }
    }
    Ok(orientation)
}

/// First pair of non-adjacent segments of a closed polyline that cross.
fn self_intersection(points: &[(f64, f64)]) -> Option<(usize, usize)> {
    let n = points.len();
    let orient = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        for j in i + 2..n {
            if (j + 1) % n == i {
                continue;
            }
            let (c, d) = (points[j], points[(j + 1) % n]);
            if a.0.max(b.0) < c.0.min(d.0)
                || c.0.max(d.0) < a.0.min(b.0)
                || a.1.max(b.1) < c.1.min(d.1)
                || c.1.max(d.1) < a.1.min(b.1)
            {
                continue;
            }
            let (o1, o2) = (orient(a, b, c), orient(a, b, d));
            let (o3, o4) = (orient(c, d, a), orient(c, d, b));
            if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
                return Some((i, j));
            }
        }
    }
    None
}

impl Surface {
    /// A flux surface: the boundary itself (offset 0).
    pub fn new_flux(
        boundary: Arc<FluxBoundary>,
        surface_id: Option<usize>,
        boundary_type: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Flux {
                boundary,
                offset: 0.0,
            },
            boundary: boundary_type.unwrap_or_default(),
            name: None,
        }
    }

    /// The level set at signed distance `distance` (cm, positive outwards)
    /// beyond this flux or offset surface. Distances are measured from the
    /// underlying flux surface, so offsets of offsets add.
    pub fn offset(
        &self,
        distance: f64,
        surface_id: Option<usize>,
        boundary_type: Option<BoundaryType>,
    ) -> Result<Self, String> {
        let SurfaceKind::Flux { boundary, offset } = &self.kind else {
            return Err(format!(
                "only flux surfaces (FluxSurface or OffsetSurface) can be offset, not a {}",
                self.kind.name()
            ));
        };
        let total = offset + distance;
        boundary.check_offset(total)?;
        Ok(Surface {
            surface_id,
            kind: SurfaceKind::Flux {
                boundary: Arc::clone(boundary),
                offset: total,
            },
            boundary: boundary_type.unwrap_or_default(),
            name: None,
        })
    }
}
