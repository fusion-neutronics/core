//! Free-form toroidal surfaces: a boundary given by a double Fourier
//! series (DESC's product convention, see [`series`]) and the level sets
//! of the signed 3-D distance to it.
//!
//! A [`FluxBoundary`] is the boundary itself. A flux surface uses it with
//! offset 0; an offset surface is the set of points at signed distance
//! `offset` (cm) from it, positive outwards, so layers built from one
//! boundary are nested by construction and never self-intersect (past
//! the radius where outward normals cross, the level set is trimmed with
//! a crease, as a true distance offset is). How far out layers stay free
//! of creases is estimated for a warning (see [`crease`]).
//!
//! The sign of the distance is decided exactly in the point's own
//! toroidal cut; its magnitude is the 3-D distance to the closest surface
//! point, found by Newton's method and certified to be the global one
//! (see [`distance`]). Ray crossings march with steps that cannot pass a
//! root (see [`ray`]).

mod crease;
mod distance;
mod patches;
mod ray;
mod series;
#[cfg(test)]
mod tests;

use crate::surface::search::{along_unit, Trace};
use crate::surface::{BoundaryType, Surface, SurfaceKind};
use distance::MAX_POLY;
use patches::Patches;
use serde::{Deserialize, Serialize};
use series::{Series, MAX_M, MAX_N};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

/// Most field periods supported (the patch grid keeps at least 8 rows per
/// period and at most 2048 rows).
const MAX_FIELD_PERIODS: u32 = 256;

/// A closed toroidal boundary `R(theta, zeta)`, `Z(theta, zeta)` given by
/// Fourier modes `(m, n)` in DESC's product convention (see the module
/// docs), lengths in cm, `zeta` the cylindrical toroidal angle.
///
/// [`FluxBoundary::new`] checks the geometry: `R > 0` everywhere, and the
/// toroidal cuts on a grid of angles simple closed curves with the same
/// orientation. The cut check samples (a loop smaller than its polyline
/// spacing, or a self-intersection only between the checked cuts, can get
/// through); it catches badly wrong coefficients, not every degenerate
/// shape. The distance machinery is rebuilt (and the checks rerun) when a
/// model is loaded; only the modes are stored.
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

/// A boundary's modes, as a map key.
type ModesKey = (u32, Vec<[i32; 2]>, Vec<u64>, Vec<[i32; 2]>, Vec<u64>);

/// Boundaries built while loading, by their modes: a saved model holds a
/// copy of a boundary for every surface built on it, and they all share
/// one boundary (and its distance machinery, built once) again on load.
static LOADED: OnceLock<Mutex<HashMap<ModesKey, Weak<FluxBoundary>>>> = OnceLock::new();

/// Deserialize a boundary, sharing it with every other live boundary
/// loaded with the same modes (for `SurfaceKind::Flux`).
pub(crate) fn deserialize_shared<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Arc<FluxBoundary>, D::Error> {
    let d = FluxBoundaryData::deserialize(deserializer)?;
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let key = (
        d.num_field_periods,
        d.r_modes.clone(),
        bits(&d.r_mn),
        d.z_modes.clone(),
        bits(&d.z_mn),
    );
    let mut loaded = LOADED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(boundary) = loaded.get(&key).and_then(Weak::upgrade) {
        return Ok(boundary);
    }
    let boundary = Arc::new(
        FluxBoundary::new(d.num_field_periods, d.r_modes, d.r_mn, d.z_modes, d.z_mn)
            .map_err(serde::de::Error::custom)?,
    );
    loaded.retain(|_, b| b.strong_count() > 0);
    loaded.insert(key, Arc::downgrade(&boundary));
    Ok(boundary)
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
        if !(1..=MAX_FIELD_PERIODS).contains(&num_field_periods) {
            return Err(format!(
                "num_field_periods must be between 1 and {MAX_FIELD_PERIODS} (got \
                 {num_field_periods})"
            ));
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

    /// Depth inside the boundary up to which the distance to it is smooth
    /// (cm): the smaller of the tightest convex radius of curvature and
    /// half the thinnest part, both bounded conservatively.
    pub fn inner_reach(&self) -> f64 {
        self.engine.reach_in
    }

    /// Refuse an offset whose level set would fold inwards.
    pub fn check_offset(&self, offset: f64) -> Result<(), String> {
        if !offset.is_finite() {
            return Err(format!("the offset distance must be finite (got {offset})"));
        }
        let reach = self.inner_reach();
        if offset < 0.0 && -offset >= 0.9 * reach {
            return Err(format!(
                "an inward offset of {:.6} cm comes too close to the boundary's inner reach \
                 (about {reach:.6} cm: its tightest convex radius of curvature or half its \
                 thinnest part), beyond which the offset layer would crease or vanish; describe \
                 surfaces that deep inside the plasma with FluxSurface(shape, rho < 1) instead",
                -offset
            ));
        }
        Ok(())
    }

    /// Distance outwards (cm) up to which offsets of the boundary have no
    /// creases: the smaller of its tightest concave radius of curvature and
    /// half its narrowest gap outside (across the hole, or a fold). It is
    /// estimated on first use by sampling and refining, which can only
    /// miss features finer than its grid (see the `crease` module).
    pub fn crease_free_distance(&self) -> f64 {
        self.engine.crease_free_distance()
    }

    /// The warning for a layer `offset` (cm) out, if that is beyond the
    /// crease-free distance.
    pub fn crease_warning(&self, offset: f64) -> Option<String> {
        // The march's reach bounds the crease-free distance from below, so
        // a layer within it needs no estimate.
        if offset <= self.engine.reach_out {
            return None;
        }
        let free = self.crease_free_distance();
        (offset > free).then(|| {
            format!(
                "An outward offset of {offset:.2} cm is beyond the crease-free distance of the \
                 flux surface (about {free:.2} cm: the smaller of its tightest concave radius of \
                 curvature and half its narrowest gap outside, such as across the central hole). \
                 The layer is still exactly {offset:.2} cm from the flux surface everywhere, but it \
                 has creases (sharp edges) where offset fronts from different parts of the flux \
                 surface meet."
            )
        })
    }

    /// `D(point) - offset`: the sign is exact; the magnitude is exact near
    /// the level set and otherwise a rigorous lower bound.
    pub(crate) fn evaluate(&self, point: [f64; 3], offset: f64) -> f64 {
        if !point.iter().all(|v| v.is_finite()) {
            return f64::NAN;
        }
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

    /// Ray parameter `t` of the first crossing of the level set at
    /// `offset` along `point + t * direction`, if before `t_max`; none for
    /// a zero or non-finite direction or a non-finite point. The search's
    /// answer, with whether it ran out of evaluations (see
    /// [`Engine::trace_ray`]).
    pub(super) fn trace(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        offset: f64,
        on_surface: Option<f64>,
        vacuum: bool,
        t_max: f64,
    ) -> Trace {
        // The march measures arc length, so it runs on the unit direction.
        along_unit(point, direction, t_max, |unit, t_max| {
            self.engine
                .trace_ray(point, unit, offset, on_surface, vacuum, t_max)
        })
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
    /// The polyline's vertices `(R, Z)` when the boundary is axisymmetric,
    /// so every cut is the same curve; empty otherwise.
    poly_fixed: Vec<(f64, f64)>,
    /// Floating-point allowance (cm), far below the 1e-8 cm crossing nudge.
    margin: f64,
    /// Newton stops when the tangential part of `x - X` is below this
    /// (cm); the distance is then exact to second order in it.
    newton_tolerance: f64,
    /// Largest convex / concave curvature anywhere on the surface.
    kappa_convex: f64,
    kappa_concave: f64,
    /// Distances inside / outside up to which the distance to the surface
    /// is smooth, so its second derivative is bounded by the curvatures.
    reach_in: f64,
    reach_out: f64,
    /// The outer reach estimated sharply, for the warning about creased
    /// layers (see [`crease`]); computed on first use.
    crease_free: OnceLock<f64>,
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
        let poly_fixed = if series.n_max == 0 {
            let cut = series.collapse(0.0, 1);
            (0..n_poly)
                .map(|k| {
                    cut.curve_from_table(
                        &poly_cos[k * m_cols..(k + 1) * m_cols],
                        &poly_sin[k * m_cols..(k + 1) * m_cols],
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let orientation_in = check_cuts(&series)?;
        let orientation_3d = -orientation_in;
        // Patches about as long toroidally as poloidally, so their bounds
        // stay tight: the poloidal count follows the mode content, and the
        // toroidal one the ratio of the toroidal to the poloidal extent.
        let b = &series.bounds;
        let nt = (8 * m_cols).clamp(48, 256);
        let poloidal = (b.r_t + b.z_t).max(margin);
        let toroidal = b.r + b.r_z + b.z_z;
        let nfp = series.nfp as usize;
        let per_period = ((toroidal / poloidal * nt as f64) / nfp as f64).ceil() as usize;
        let nz = nfp
            * per_period
                .max(4 * (series.n_max + 1))
                .clamp(8, 2048 / nfp.max(1));
        let patches = Patches::new(&series, nt, nz, orientation_3d, margin);
        if patches.min_radius() <= 0.0 {
            return Err(
                "the boundary comes within its sampling tolerance of the z axis (R <= 0 \
                 somewhere); check the coefficients (they should be in cm)"
                    .to_string(),
            );
        }
        let (kappa_convex, kappa_concave) = (patches.max_convex, patches.max_concave);
        // The rows of one field period see every gap there is. An
        // axisymmetric boundary is the same at every zeta, so its first row
        // already does, and searching all of them would only fill the gap
        // search's piece budget (leaving pairs too coarse to bound).
        let rows = if series.n_max == 0 { 1 } else { nz / nfp };
        let (gap_in, gap_out) = patches.gaps(&series, rows, orientation_3d, margin);
        let reach_in = (1.0 / kappa_convex).min(0.5 * gap_in);
        let reach_out = (1.0 / kappa_concave).min(0.5 * gap_out);
        Ok(Self {
            newton_tolerance: 1e-10 * (scale / 1000.0).max(1.0),
            series,
            patches,
            orientation_in,
            orientation_3d,
            n_poly,
            dtheta_poly,
            poly_cos,
            poly_sin,
            poly_fixed,
            margin,
            kappa_convex,
            kappa_concave,
            reach_in,
            reach_out,
            crease_free: OnceLock::new(),
        })
    }
}

/// Check, on a grid of cuts and polylines, that the toroidal cuts over a
/// field period are simple closed curves that all run the same way round;
/// return that orientation (+1 counter-clockwise in `(R, Z)`).
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
        let (boundary, offset) = match &self.kind {
            SurfaceKind::Flux { boundary, offset } => (boundary, *offset),
            SurfaceKind::Implicit { .. } => {
                return Err(
                    "an ImplicitSurface cannot be offset: a layer of uniform thickness needs \
                     its formula to be a signed distance, which is not tracked yet. For a \
                     formula f that is one, the surface d cm outside it is the formula \
                     \"(f) - d\""
                        .to_string(),
                );
            }
            _ => {
                return Err(format!(
                    "only flux surfaces (FluxSurface or OffsetSurface) can be offset, not a {}",
                    self.kind.name()
                ));
            }
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

    /// For a flux layer beyond its flux surface's crease-free distance, a
    /// warning saying so (see [`FluxBoundary::crease_warning`]).
    pub fn crease_warning(&self) -> Option<String> {
        match &self.kind {
            SurfaceKind::Flux { boundary, offset } => boundary.crease_warning(*offset),
            _ => None,
        }
    }
}
