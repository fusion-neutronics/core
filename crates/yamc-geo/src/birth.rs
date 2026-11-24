//! Where a particle with no cell yet is located: a short step along its
//! flight, so a particle born exactly on a surface starts in the cell it is
//! heading into.
//!
//! Shared by the CPU geometry (`GeometryKind::find_cell_index_along` in
//! `yamc`) and the GPU transport kernels (`yamc-gpu`), so both backends put
//! a birth on a surface in the same cell.

/// How far along its flight a particle with no cell yet is located, unless
/// that does not move the point off the surface it sits on (see
/// [`birth_step`]).
///
/// Equal to the distance below which mesh ray-fire ignores a hit, so that
/// the cell found and the first crossing agree: a particle within it of a
/// facet ahead is put past the facet, which ray-fire then ignores, and one
/// farther away is put before it, where ray-fire sees the crossing. CSG
/// re-locates spatially after every crossing, so any nudge this small does.
pub const BIRTH_NUDGE: f64 = 1e-10;

/// Units in the last place each coordinate a birth step changes is moved
/// by at least, to within a factor of 2 (see [`birth_step`]).
pub const BIRTH_STEP_ULPS: f64 = 16.0;

/// Longest birth step (cm). Only a direction within about 1e-6 of parallel
/// to a surface far from the origin would need longer to move off it.
pub const MAX_BIRTH_STEP: f64 = 1e-6;

/// How far along `direction` from `point` a particle with no cell yet is
/// located: [`BIRTH_NUDGE`], or longer when that would not move every
/// coordinate the direction changes by [`BIRTH_STEP_ULPS`] units in the
/// last place.
///
/// Far from the origin the spacing of doubles is coarse, and a direction
/// grazing a surface there moves the coordinate across it by less than half
/// that spacing in [`BIRTH_NUDGE`]: the point rounds back onto the surface,
/// where a CSG point is in no cell and a mesh point is put on one fixed side.
/// The longer step still crosses the surface by only a few units in the last
/// place, and is capped at [`MAX_BIRTH_STEP`]. It can be longer than the
/// distance below which mesh ray-fire ignores a hit, which only matters for
/// a particle born within the step of a second facet ahead (at an edge).
pub fn birth_step(point: [f64; 3], direction: [f64; 3]) -> f64 {
    let mut step = BIRTH_NUDGE;
    for (x, u) in point.iter().zip(direction) {
        let u = u.abs();
        if u > 0.0 && x.is_finite() {
            // |x| * EPSILON is the spacing of doubles at x, to within a
            // factor of 2 above it, and needs no bit manipulation.
            step = step.max(BIRTH_STEP_ULPS * x.abs() * f64::EPSILON / u);
        }
    }
    step.min(MAX_BIRTH_STEP)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(u: [f64; 3]) -> [f64; 3] {
        let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [u[0] / n, u[1] / n, u[2] / n]
    }

    #[test]
    fn the_birth_step_is_the_nudge_where_the_nudge_moves_every_coordinate() {
        let u = unit([0.3, -0.5, 0.8]);
        assert_eq!(birth_step([1.0, 2.0, -3.0], u), BIRTH_NUDGE);
        assert_eq!(birth_step([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), BIRTH_NUDGE);
    }

    #[test]
    fn the_birth_step_grows_until_a_grazing_coordinate_moves_and_is_capped() {
        let p = [500.0, 500.0, 2200.0];
        let u = unit([-0.95, 0.31, -1e-4]);
        let step = birth_step(p, u);
        assert!(step > BIRTH_NUDGE && step < MAX_BIRTH_STEP, "{step}");
        assert!(p[2] + step * u[2] < p[2]);
        assert_eq!(birth_step(p, unit([1.0, 0.0, 1e-15])), MAX_BIRTH_STEP);
    }
}
