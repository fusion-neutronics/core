//! CSG region expression tree (`RegionExpr`) and its halfspace leaves, plus
//! the direct recursive containment test.

use crate::surface::Surface;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HalfspaceType {
    Above(Arc<Surface>),
    Below(Arc<Surface>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionExpr {
    Halfspace(HalfspaceType),
    Union(Box<RegionExpr>, Box<RegionExpr>),
    Intersection(Box<RegionExpr>, Box<RegionExpr>),
    Complement(Box<RegionExpr>),
}

impl RegionExpr {
    pub fn evaluate_contains(&self, point: (f64, f64, f64)) -> bool {
        match self {
            RegionExpr::Halfspace(hs) => match hs {
                HalfspaceType::Above(surf) => surf.evaluate(point) > 0.0,
                HalfspaceType::Below(surf) => surf.evaluate(point) < 0.0,
            },
            RegionExpr::Union(a, b) => a.evaluate_contains(point) || b.evaluate_contains(point),
            RegionExpr::Intersection(a, b) => {
                a.evaluate_contains(point) && b.evaluate_contains(point)
            }
            RegionExpr::Complement(inner) => !inner.evaluate_contains(point),
        }
    }
}
