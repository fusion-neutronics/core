//! Closest point on triangle -- 7-region algorithm from Geometric Tools.
//!
//! Reference: "Distance Between Point and Triangle in 3D" by David Eberly.
//! https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

use crate::query::intersect::{add, dot, length, scale, sub};

/// Find the closest point on a triangle (v0, v1, v2) to a query point.
///
/// Returns `(closest_point, distance)`.
pub fn closest_point_on_triangle(
    point: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> ([f64; 3], f64) {
    let sv = sub(v1, v0); // edge v0→v1
    let tv = sub(v2, v0); // edge v0→v2
    let pv = sub(v0, point);

    let ss = dot(sv, sv);
    let st = dot(sv, tv);
    let tt = dot(tv, tv);
    let sp = dot(sv, pv);
    let tp = dot(tv, pv);
    let det = ss * tt - st * st;

    let mut s = st * tp - tt * sp;
    let mut t = st * sp - ss * tp;

    if s + t <= det {
        if s < 0.0 {
            if t < 0.0 {
                // Region 4: closest to v0 or edges
                if sp < 0.0 {
                    t = 0.0;
                    s = if -sp >= ss { 1.0 } else { -sp / ss };
                } else {
                    s = 0.0;
                    t = if tp >= 0.0 {
                        0.0
                    } else if -tp >= tt {
                        1.0
                    } else {
                        -tp / tt
                    };
                }
            } else {
                // Region 3: edge v0-v2
                s = 0.0;
                t = if tp >= 0.0 {
                    0.0
                } else if -tp >= tt {
                    1.0
                } else {
                    -tp / tt
                };
            }
        } else if t < 0.0 {
            // Region 5: edge v0-v1
            t = 0.0;
            s = if sp >= 0.0 {
                0.0
            } else if -sp >= ss {
                1.0
            } else {
                -sp / ss
            };
        } else {
            // Region 0: interior
            let inv_det = 1.0 / det;
            s *= inv_det;
            t *= inv_det;
        }
    } else if s < 0.0 {
        // Region 2: edge v0-v2 or edge v1-v2
        let tmp0 = st + sp;
        let tmp1 = tt + tp;
        if tmp1 > tmp0 {
            let numer = tmp1 - tmp0;
            let denom = ss - 2.0 * st + tt;
            s = if numer >= denom { 1.0 } else { numer / denom };
            t = 1.0 - s;
        } else {
            s = 0.0;
            t = if tmp1 <= 0.0 {
                1.0
            } else if tp >= 0.0 {
                0.0
            } else {
                -tp / tt
            };
        }
    } else if t < 0.0 {
        // Region 6: edge v0-v1 or edge v1-v2
        let tmp0 = st + tp;
        let tmp1 = ss + sp;
        if tmp1 > tmp0 {
            let numer = tmp1 - tmp0;
            let denom = ss - 2.0 * st + tt;
            t = if numer >= denom { 1.0 } else { numer / denom };
            s = 1.0 - t;
        } else {
            t = 0.0;
            s = if tmp1 <= 0.0 {
                1.0
            } else if sp >= 0.0 {
                0.0
            } else {
                -sp / ss
            };
        }
    } else {
        // Region 1: edge v1-v2
        let numer = (tt + tp) - (st + sp);
        if numer <= 0.0 {
            s = 0.0;
            t = 1.0;
        } else {
            let denom = ss - 2.0 * st + tt;
            s = if numer >= denom { 1.0 } else { numer / denom };
            t = 1.0 - s;
        }
    }

    let closest = add(v0, add(scale(sv, s), scale(tv, t)));
    let dist = length(sub(closest, point));
    (closest, dist)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_point_interior() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // Point directly above triangle interior
        let (cp, dist) = closest_point_on_triangle([0.25, 0.25, 1.0], v0, v1, v2);
        assert!((cp[0] - 0.25).abs() < 1e-10);
        assert!((cp[1] - 0.25).abs() < 1e-10);
        assert!((cp[2] - 0.0).abs() < 1e-10);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_closest_point_vertex() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // Point closest to v0
        let (cp, _dist) = closest_point_on_triangle([-1.0, -1.0, 0.0], v0, v1, v2);
        assert!((cp[0] - 0.0).abs() < 1e-10);
        assert!((cp[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_closest_point_edge() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // Point below edge v0-v1
        let (cp, _dist) = closest_point_on_triangle([0.5, -1.0, 0.0], v0, v1, v2);
        assert!((cp[0] - 0.5).abs() < 1e-10);
        assert!((cp[1] - 0.0).abs() < 1e-10);
    }
}
