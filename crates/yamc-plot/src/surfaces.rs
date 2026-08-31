//! Walk a [`CsgGeometry`]'s region trees, collect every unique surface,
//! and produce a display-name table the viewer uses for hover tooltips.
//!
//! The viewer's hover handler does the actual nearest-surface lookup
//! in JS (cheap per-mouse-move; see `viewer.js::surfacesAtPoint`); this
//! module is just responsible for the *name* one of the listed surfaces
//! should show as.
//!
//! Display-name fallback (matching the design discussion):
//!
//! 1. `surface.name` if set (`"inner"`, `"outer_blanket"`),
//! 2. else `"surface {surface_id}"` if `surface_id` is set,
//! 3. else `"{Kind} #{i}"` where `i` is a per-kind sequential index.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_geo::geometry::CsgGeometry;
use yamc_geo::region::RegionExpr;
use yamc_geo::surface::{Surface, SurfaceKind};

/// One row of the surface table the viewer sees.
pub struct SurfaceTableEntry {
    pub surface: Arc<Surface>,
    pub display_name: String,
}

/// Collect every unique [`Surface`] referenced by any cell's region.
///
/// Deduplication uses serialized-content equality, **not** [`Arc::ptr_eq`].
/// `Arc::ptr_eq` only works when the same `Arc<Surface>` was cloned --
/// after a JSON round-trip (which the wasm `plot_html` path goes
/// through, and which is generally how shared surfaces propagate
/// between cells in deserialised models) every cell holds its own
/// freshly-allocated Arc, and ptr-equality fails. Content equality
/// catches both cases.
///
/// Stable ordering: first-seen wins, scanning cells in their
/// declaration order.
pub fn build_surface_table(csg: &CsgGeometry) -> Vec<SurfaceTableEntry> {
    let mut out: Vec<SurfaceTableEntry> = Vec::new();
    let mut next_kind_index: HashMap<&'static str, usize> = HashMap::new();
    let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    for cell in &csg.cells {
        let mut local: Vec<(Arc<Surface>, bool)> = Vec::new();
        cell.region.collect_surfaces_with_sense(&mut local);
        for (surf, _sense) in local {
            // Stable key: the surface's serialised form. Covers the
            // Arc-shared case (same key) and the post-round-trip case
            // (separate Arcs, identical content).
            let key = serde_json::to_string(&*surf).unwrap_or_default();
            if !seen_keys.insert(key) {
                continue;
            }
            let display = display_name_for(&surf, &mut next_kind_index);
            out.push(SurfaceTableEntry {
                surface: surf,
                display_name: display,
            });
        }
    }
    out
}

fn display_name_for(surf: &Surface, kind_counters: &mut HashMap<&'static str, usize>) -> String {
    if let Some(n) = surf.name.as_ref() {
        return n.clone();
    }
    if let Some(id) = surf.surface_id {
        return format!("surface {id}");
    }
    let kind = surface_kind_name(&surf.kind);
    let idx = kind_counters.entry(kind).or_insert(0);
    *idx += 1;
    format!("{kind} #{}", *idx)
}

fn surface_kind_name(kind: &SurfaceKind) -> &'static str {
    match kind {
        SurfaceKind::Plane { .. } => "Plane",
        SurfaceKind::Sphere { .. } => "Sphere",
        SurfaceKind::Cylinder { .. } => "Cylinder",
        SurfaceKind::ZTorus { .. } => "ZTorus",
        SurfaceKind::XTorus { .. } => "XTorus",
        SurfaceKind::YTorus { .. } => "YTorus",
        SurfaceKind::Quadric { .. } => "Quadric",
        SurfaceKind::Cone { .. } => "Cone",
    }
}

/// Walk a single `RegionExpr` collecting surfaces (used by the
/// `surfaces_near` point query -- for cell-local search rather than the
/// full geometry-wide table). Senses are dropped; the caller uses
/// `Surface::evaluate` and compares `|f|` against the tolerance.
pub fn collect_region_surfaces(expr: &RegionExpr, out: &mut Vec<Arc<Surface>>) {
    match expr {
        RegionExpr::Halfspace(hs) => {
            let surf = match hs {
                yamc_geo::region::HalfspaceType::Above(s) => s.clone(),
                yamc_geo::region::HalfspaceType::Below(s) => s.clone(),
            };
            if !out.iter().any(|e| Arc::ptr_eq(e, &surf)) {
                out.push(surf);
            }
        }
        RegionExpr::Union(a, b) | RegionExpr::Intersection(a, b) => {
            collect_region_surfaces(a, out);
            collect_region_surfaces(b, out);
        }
        RegionExpr::Complement(inner) => collect_region_surfaces(inner, out),
    }
}
