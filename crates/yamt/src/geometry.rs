//! Public API: `MeshGeometry` -- the main entry point for mesh-based
//! geometric queries. Analogous to DAGMC's `DagMC` class.

use crate::accel::bvh::Bvh;
use crate::mesh::contacts::Contacts;
use crate::mesh::topology::MeshTopology;
use crate::query::{
    closest, element_walk, intersect, point_in_volume, ray_fire, ray_history::RayHistory,
};
use crate::types::*;

/// Error type for mesh geometry operations.
#[derive(Debug)]
pub enum MeshError {
    Io(std::io::Error),
    Parse(String),
    /// A tetrahedron in the input mesh is negatively oriented, which the
    /// element walk cannot work with. Raised by
    /// [`crate::mesh::topology::validate_tet_orientation`]; see there for why
    /// the mesh is rejected instead of repaired.
    NegativeTetOrientation {
        /// Index into the mesh's tetrahedron array.
        tet_index: usize,
        /// Its signed volume (negative, in cm³).
        signed_volume: f64,
    },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Io(e) => write!(f, "I/O error: {e}"),
            MeshError::Parse(msg) => write!(f, "Parse error: {msg}"),
            MeshError::NegativeTetOrientation {
                tet_index,
                signed_volume,
            } => write!(
                f,
                "tetrahedron {tet_index} is negatively oriented (signed volume \
                 {signed_volume:.6e} cm^3). Tet face normals are read off a fixed vertex \
                 ordering that points outward only for positively oriented tets, so a \
                 negative tet makes the element walk leave through an entry face and \
                 unstructured track-length tallies read about 33 percent low. \
                 Fix the mesh at its source: regenerate it with a writer that emits \
                 positively oriented tets (yamc's own mesher, yamm, guarantees this at its \
                 output boundary), or have that writer swap the last two vertices of every \
                 tet whose signed volume is negative."
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// Precomputed triangle data for cache-friendly traversal.
///
/// Stores v0, edge1 (v1-v0), edge2 (v2-v0) for each primitive,
/// eliminating the multi-level vertex indirection chain and
/// redundant edge computation from the inner loop.
#[derive(Clone)]
pub(crate) struct PrecomputedTriData {
    pub(crate) v0: Vec<[f64; 3]>,
    pub(crate) edge1: Vec<[f64; 3]>,
    pub(crate) edge2: Vec<[f64; 3]>,
}

/// Per-volume surface BVH with associated data.
#[derive(Clone)]
pub(crate) struct SurfaceBvhData {
    pub(crate) bvh: Bvh,
    pub(crate) tri_ids: Vec<TriangleId>,
    pub(crate) surf_ids: Vec<SurfaceId>,
    pub(crate) precomputed: PrecomputedTriData,
    /// Per primitive: `+1` when the stored winding (edge1 x edge2) points out
    /// of this BVH's volume, `-1` when it points in, `0` for a facet with no
    /// real volume on either side (the graveyard), which is always accepted.
    /// Lets discontinuous tracking cull facets the ray would ENTER the volume
    /// through.
    pub(crate) exit_sign: Vec<i8>,
}

impl SurfaceBvhData {
    /// True when a ray along `direction` crossing primitive `i` leaves this
    /// BVH's volume there. Graveyard facets (sign 0) always count.
    #[inline]
    fn leaves(&self, i: usize, direction: [f64; 3]) -> bool {
        let sign = self.exit_sign[i];
        if sign == 0 {
            return true;
        }
        let n = intersect::cross(self.precomputed.edge1[i], self.precomputed.edge2[i]);
        f64::from(sign) * intersect::dot(direction, n) > 0.0
    }
}

/// What lies across a crossing found by [`MeshGeometry::next_crossing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Beyond {
    /// Known now: read off the surface topology, or found by re-locating
    /// in place. `None` when the topology records no volume there.
    Volume(Option<VolumeId>),
    /// Left a solid through a triangle facing the implicit complement: what
    /// is beyond is found by [`MeshGeometry::relocate`], and only if the
    /// particle takes the crossing (a collision before it makes the query
    /// wasted work).
    Relocate(TriangleId),
}

/// The exit facet relocation found for the solid it placed a particle in.
///
/// Relocation asks a candidate solid for the first of its facets ahead of the
/// crossing point; when that facet is one the ray leaves through, it is also
/// exactly what the next [`MeshGeometry::next_crossing`] in that solid, from a
/// point a little further along the same ray, would find. Keeping it saves
/// that query on every crossing between touching solids.
#[derive(Clone, Copy)]
struct ExitAhead {
    /// Address of the geometry it belongs to.
    geometry: usize,
    volume: VolumeId,
    /// The relocation point and direction the facet was found from.
    from: [f64; 3],
    direction: [f64; 3],
    /// Distance from `from` to the facet, and the facet's primitive index in
    /// `volume`'s BVH.
    distance: f64,
    prim: u32,
}

thread_local! {
    /// One per transport thread: a thread tracks one particle at a time, so
    /// the next query after a relocation on this thread is that particle's.
    /// The entry is only ever used for a query that provably asks the same
    /// question (same geometry, volume and direction, origin on the same ray
    /// short of the facet), so a stale entry can only be ignored, not misused.
    static EXIT_AHEAD: std::cell::Cell<Option<ExitAhead>> = const { std::cell::Cell::new(None) };
}

/// Minimum hit distance (cm) for discontinuous tracking's ray fire, as in
/// [`ray_fire`].
const RAY_FIRE_MIN_T: f64 = 1e-10;

/// How far (cm) relocation looks behind a crossing point, so that a facet
/// coincident with the one just crossed is found. Far above the round-off in
/// a hit point (about 1e-12 cm at metre-scale coordinates) and far below any
/// modelled feature.
const LOOKBACK: f64 = 1e-7;

/// Contact tolerance for a non-imprinted mesh whose file does not record one:
/// a ten-thousandth of the diagonal of the box around its solids (a graveyard
/// added around them does not count), enough to cover facetings that agree to
/// a few significant figures without filling a gap anyone modelled on purpose.
fn default_contact_tolerance(topo: &MeshTopology) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for b in topo.volume_aabbs.iter().take(topo.num_volumes as usize) {
        for k in 0..3 {
            lo[k] = lo[k].min(b[k]);
            hi[k] = hi[k].max(b[k + 3]);
        }
    }
    let d = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    if d.is_finite() && d > 0.0 {
        1e-4 * d
    } else {
        1e-4
    }
}

/// Complete mesh geometry with acceleration structures, ready for queries.
///
/// This is the primary public API of the crate. It mirrors the DAGMC
/// `DagMC` class for mesh-based particle transport.
#[derive(Clone)]
pub struct MeshGeometry {
    /// The underlying mesh topology.
    pub topology: MeshTopology,

    /// Per-volume surface BVH + mapping arrays.
    surface_bvhs: Vec<SurfaceBvhData>,

    /// Per-volume element BVH + mapping arrays.
    /// `element_bvhs[vol_id] = (bvh, tet_ids)`
    element_bvhs: Vec<(Bvh, Vec<TetrahedronId>)>,

    /// Discontinuous tracking, for a mesh whose solids were meshed one by one
    /// without imprinting (see [`MeshGeometry::next_crossing`]).
    discontinuous: bool,
    /// Distance (cm) within which two solids' surfaces count as touching.
    /// Meaningful only when `discontinuous` is set.
    contact_tolerance: f64,
    /// Per triangle, the other solids within `contact_tolerance` of it, plus,
    /// for a solid that takes an overlap from another, that other solid.
    /// Highest rank first. Empty unless `discontinuous` is set.
    contacts: Contacts,

    /// Overlap resolution for a discontinuous mesh with recorded overlaps
    /// (see [`MeshTopology::overlaps`]); empty otherwise. `ranks[v]` orders
    /// the solids: where several contain a point, the highest rank's cell it
    /// is.
    ranks: Vec<u32>,
    /// Per volume, the facets of the higher-ranked solids overlapping it,
    /// through which a particle in it enters one of them. `None` for a solid
    /// no higher-ranked solid overlaps.
    overlap_bvhs: Vec<Option<OverlapBvh>>,
}

/// Facets of higher-ranked solids that a particle in a lower-ranked solid can
/// enter them through, with the solid each belongs to.
#[derive(Clone)]
struct OverlapBvh {
    /// `exit_sign` is set so that [`SurfaceBvhData::leaves`] is true when the
    /// ray ENTERS the owning solid.
    data: SurfaceBvhData,
    /// The solid each primitive belongs to.
    enters: Vec<VolumeId>,
}

impl MeshGeometry {
    /// Load from an Arrow IPC mesh file and build all acceleration structures.
    #[cfg(feature = "arrow")]
    pub fn from_arrow(path: &std::path::Path) -> Result<Self, MeshError> {
        let data = crate::io::arrow::read_arrow_mesh(path)?;
        let topo = crate::io::arrow::build_topology(data)?;
        Ok(Self::from_topology(topo))
    }

    /// Build from an existing topology (builds BVH trees and computes volumes).
    pub fn from_topology(mut topo: MeshTopology) -> Self {
        #[cfg(feature = "simd")]
        {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(crate::accel::simd::print_simd_info);
        }

        let surface_bvhs = build_surface_bvhs(&topo);
        let element_bvhs = build_element_bvhs(&topo);

        // Precompute analytical volumes if not already set (e.g. from Arrow metadata)
        if topo.volume_measures.is_empty() && topo.num_volumes > 0 {
            // We need the MeshGeometry built first to call measure_volume,
            // but measure_volume only uses topology data, so we can compute
            // directly here using the same divergence-theorem logic.
            let mut measures = Vec::with_capacity(topo.num_volumes as usize);
            for vol_id in 0..topo.num_volumes {
                let mut total = 0.0;
                for &(surf_id, sense) in &topo.volume_surfaces[vol_id as usize] {
                    let range = &topo.surface_tri_ranges[surf_id as usize];
                    for i in range.start..range.end {
                        let tri_id = topo.surface_tri_indices[i as usize];
                        let tri = &topo.triangles[tri_id as usize];
                        let v0 = topo.vertices[tri[0] as usize];
                        let v1 = topo.vertices[tri[1] as usize];
                        let v2 = topo.vertices[tri[2] as usize];
                        let contrib =
                            crate::query::intersect::triangle_volume_contribution(v0, v1, v2);
                        match sense {
                            Sense::Forward => total += contrib,
                            Sense::Reverse => total -= contrib,
                        }
                    }
                }
                measures.push((total / 6.0).abs());
            }
            topo.volume_measures = measures;
        }

        let mut geometry = Self {
            topology: topo,
            surface_bvhs,
            element_bvhs,
            discontinuous: false,
            contact_tolerance: 0.0,
            contacts: Contacts::default(),
            ranks: Vec::new(),
            overlap_bvhs: Vec::new(),
        };
        if geometry.topology.imprinted == Some(false) {
            let tol = geometry
                .topology
                .contact_tolerance
                .unwrap_or_else(|| default_contact_tolerance(&geometry.topology));
            geometry.set_contact_tolerance(Some(tol));
        }
        geometry
    }

    /// Switch discontinuous tracking on with the given contact tolerance (cm),
    /// or off with `None`.
    ///
    /// Two solids whose surfaces come within the contact tolerance of each
    /// other count as touching there: a particle leaving one crosses an
    /// overlap or a gap thinner than the tolerance straight into the other.
    /// It has to be at least the deepest overlap between touching solids
    /// (deeper overlaps are not found), and gaps narrower than it are filled.
    ///
    /// On is right for any mesh whose touching solids do not share a surface,
    /// which is what meshing without imprinting produces; the file written by
    /// such a mesher selects it on its own. Off is the imprinted (conformal)
    /// tracking, where the surface topology alone says what is across every
    /// surface. A partly imprinted mesh is fine on: shared surfaces still hand
    /// over by topology.
    pub fn set_contact_tolerance(&mut self, tolerance: Option<f64>) {
        match tolerance {
            Some(tol) => {
                self.discontinuous = true;
                self.contact_tolerance = tol;
                let topo = &self.topology;
                if topo.overlaps.is_empty() {
                    self.contacts = Contacts::build(topo, tol);
                    self.ranks = Vec::new();
                    self.overlap_bvhs = Vec::new();
                } else {
                    let ranks =
                        crate::mesh::topology::overlap_ranks(topo.num_volumes, &topo.overlaps)
                            .expect("build_topology refuses cyclic overlaps");
                    self.contacts = Contacts::build(topo, tol).with_overlaps(topo, &ranks);
                    self.overlap_bvhs = build_overlap_bvhs(topo);
                    self.ranks = ranks;
                }
            }
            None => {
                self.discontinuous = false;
                self.contact_tolerance = 0.0;
                self.contacts = Contacts::default();
                self.ranks = Vec::new();
                self.overlap_bvhs = Vec::new();
            }
        }
    }

    /// The contact tolerance (cm) when tracking is discontinuous, `None` for
    /// imprinted (conformal) tracking.
    pub fn contact_tolerance(&self) -> Option<f64> {
        self.discontinuous.then_some(self.contact_tolerance)
    }

    /// True when tracking is discontinuous (see [`MeshGeometry::next_crossing`]).
    pub fn is_discontinuous(&self) -> bool {
        self.discontinuous
    }

    /// True when some solids touch without sharing a surface, while the mesh
    /// is tracked conformally: some solid's surface, facing the implicit
    /// complement, has another solid right behind it. Conformal tracking
    /// reads the far side of such a surface as the implicit complement and
    /// skips the other solid, without losing particles. A mesh made without
    /// imprinting is like that throughout; an imprinted one where the imprint
    /// failed to share some faces is like that in places.
    ///
    /// Probes, for up to `max_probes` triangles spread over every surface
    /// facing the implicit complement, a point just outside the solid along
    /// the outward normal, and asks whether any other solid contains it.
    /// Edge-only contact (two solids meeting along a line, as imprinted solids
    /// do at a shared edge) does not trigger it.
    pub fn touches_without_shared_surface(&self, max_probes: usize) -> bool {
        let topo = &self.topology;
        if self.discontinuous || topo.num_volumes < 2 {
            return false;
        }
        let ic = topo.implicit_complement;
        let b = topo.global_aabb;
        let diag = ((b[3] - b[0]).powi(2) + (b[4] - b[1]).powi(2) + (b[5] - b[2]).powi(2)).sqrt();
        let eps = (1e-6 * diag).max(1e-9);
        let candidates: Vec<(SurfaceId, VolumeId, Sense)> = (0..topo.num_surfaces)
            .filter_map(|s| match topo.surface_volumes[s as usize] {
                (Some(v), Some(o)) if o == ic && v != ic => Some((s, v, Sense::Forward)),
                (Some(o), Some(v)) if o == ic && v != ic => Some((s, v, Sense::Reverse)),
                _ => None,
            })
            .collect();
        let total: usize = candidates
            .iter()
            .map(|&(s, _, _)| topo.surface_tri_ranges[s as usize].len())
            .sum();
        if total == 0 {
            return false;
        }
        let stride = total.div_ceil(max_probes.max(1));
        let mut k = 0usize;
        for &(s, owner, sense) in &candidates {
            for i in topo.surface_tri_ranges[s as usize].clone() {
                k += 1;
                if !k.is_multiple_of(stride) {
                    continue;
                }
                let tri = topo.triangles[topo.surface_tri_indices[i as usize] as usize];
                let (a, bb, c) = (
                    topo.vertices[tri[0] as usize],
                    topo.vertices[tri[1] as usize],
                    topo.vertices[tri[2] as usize],
                );
                let n = intersect::normalize(intersect::cross(
                    intersect::sub(bb, a),
                    intersect::sub(c, a),
                ));
                let out = match sense {
                    Sense::Forward => 1.0,
                    Sense::Reverse => -1.0,
                };
                let p = [
                    (a[0] + bb[0] + c[0]) / 3.0 + out * eps * n[0],
                    (a[1] + bb[1] + c[1]) / 3.0 + out * eps * n[1],
                    (a[2] + bb[2] + c[2]) / 3.0 + out * eps * n[2],
                ];
                for v in 0..topo.num_volumes {
                    if v == owner {
                        continue;
                    }
                    let vb = &topo.volume_aabbs[v as usize];
                    if (0..3).any(|j| p[j] < vb[j] || p[j] > vb[j + 3]) {
                        continue;
                    }
                    if self.point_in_volume(v, p) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The per-triangle contact lists discontinuous tracking relocates with.
    pub fn contacts(&self) -> &Contacts {
        &self.contacts
    }

    /// Next boundary crossing for discontinuous tracking, with what is on the
    /// other side: `(distance, surface, beyond)`.
    ///
    /// Only facets the ray LEAVES the current cell through count. From inside
    /// a solid, a facet the ray would enter it through can only be the far
    /// side of a coincident or gap sliver the particle is standing in; from
    /// the implicit complement, a facet the ray would leave a solid through
    /// belongs to a solid it already overlaps. Neither is a boundary of the
    /// cell being tracked.
    ///
    /// A shared (imprinted) surface hands over by topology, as conformal
    /// tracking does. Leaving a solid through a surface facing the implicit
    /// complement needs relocating (see [`MeshGeometry::relocate`]), which is
    /// left to the caller to do once the crossing is taken. The implicit
    /// complement enters the solid whose facet it crossed.
    ///
    /// A particle can be outside the solid it is tracked in: relocation gives a
    /// gap thinner than the contact tolerance to the solid ahead, and the
    /// particle may collide in that gap. Then the solid's first facet ahead is
    /// one the ray would enter it through. Within the contact tolerance that is
    /// the far side of the gap, and the particle carries on into the solid.
    /// Beyond it, or with no facet ahead at all, the particle has left the gap
    /// heading away from the solid (across a hole in it, say), and is
    /// re-located on the spot: a crossing at distance 0 into wherever it is.
    pub fn next_crossing(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
    ) -> Option<(f64, SurfaceId, Beyond)> {
        let sd = self.surface_bvhs.get(volume as usize)?;
        let ic = self.topology.implicit_complement;
        let first_facet = |exits_only: bool| {
            sd.bvh.ray_traverse_tris(
                origin,
                direction,
                &sd.precomputed.v0,
                &sd.precomputed.edge1,
                &sd.precomputed.edge2,
                RAY_FIRE_MIN_T,
                |prim, _t| !exits_only || sd.leaves(prim as usize, direction),
            )
        };
        let hit = match self.exit_ahead(volume, origin, direction) {
            Some(hit) => Some(hit),
            None if volume == ic => first_facet(true),
            // Inside the solid, the nearest of its facets is an exit: one query.
            None => match first_facet(false) {
                Some((prim, t)) if sd.leaves(prim as usize, direction) => Some((prim, t)),
                Some((_, t)) if t > self.contact_tolerance => {
                    let here = self.find_volume(origin);
                    if here != volume {
                        return Some((0.0, SurfaceId::MAX, Beyond::Volume(Some(here))));
                    }
                    first_facet(true)
                }
                _ => first_facet(true),
            },
        };
        // A higher-ranked solid overlapping this one takes over where the ray
        // enters it, if that comes before this solid's own exit. The query
        // starts LOOKBACK behind the particle: an entry coincident with the
        // crossing just taken (a part flush with its host's face, entered from
        // outside; a part touching the one just left) is then just behind the
        // particle, and is taken at distance 0. Only from inside this solid
        // (an exit ahead): a particle outside it is re-located below.
        if let (Some(Some(ob)), Some((_, limit))) = (self.overlap_bvhs.get(volume as usize), hit) {
            let o = [
                origin[0] - LOOKBACK * direction[0],
                origin[1] - LOOKBACK * direction[1],
                origin[2] - LOOKBACK * direction[2],
            ];
            let entry = ob.data.bvh.ray_traverse_tris(
                o,
                direction,
                &ob.data.precomputed.v0,
                &ob.data.precomputed.edge1,
                &ob.data.precomputed.edge2,
                0.0,
                |prim, t| t - LOOKBACK < limit && ob.data.leaves(prim as usize, direction),
            );
            if let Some((prim, t)) = entry {
                let p = prim as usize;
                return Some((
                    (t - LOOKBACK).max(0.0),
                    ob.data.surf_ids[p],
                    Beyond::Volume(Some(ob.enters[p])),
                ));
            }
        }
        let Some((prim, t)) = hit else {
            if volume != ic {
                let here = self.find_volume(origin);
                if here != volume {
                    return Some((0.0, SurfaceId::MAX, Beyond::Volume(Some(here))));
                }
            }
            return None;
        };
        let surf = sd.surf_ids[prim as usize];
        let (fwd, rev) = self.topology.surface_volumes[surf as usize];
        let other = if fwd == Some(volume) { rev } else { fwd };
        let beyond = if volume == ic || other != Some(ic) {
            Beyond::Volume(other)
        } else {
            Beyond::Relocate(sd.tri_ids[prim as usize])
        };
        Some((t, surf, beyond))
    }

    /// The cell a particle is in after leaving solid `from` through triangle
    /// `tri` at `x`, travelling along `u`.
    ///
    /// Only the solids touching `tri` (its contacts) can contain `x`, and each
    /// is asked separately, on its own facets: `x` is inside it when the first
    /// of its facets the ray from `x` meets is one the ray LEAVES it through.
    /// That holds for an overlap sliver of any depth, and a third solid
    /// overlapping the candidate further along the ray cannot confuse it (the
    /// first facet of ANY solid ahead can be that third solid's entry, which
    /// would read as "in a gap"). Failing that, the candidate whose first
    /// facet is the nearest entry within the contact tolerance takes the
    /// particle: the two solids count as touching there, and the thin gap
    /// between their facetings goes to the solid ahead rather than to the
    /// implicit complement. When that nearest solid is one the writer recorded
    /// as separated from `from` in the CAD (see [`MeshTopology::separated`]),
    /// the gap is real and stays the implicit complement, even if a solid that
    /// does touch `from` lies just beyond it. With no solid near, `x` is in a
    /// real gap too.
    ///
    /// When several candidates contain `x` (a sliver claimed by more than one
    /// solid), the lowest volume id wins.
    pub fn relocate(&self, from: VolumeId, tri: TriangleId, x: [f64; 3], u: [f64; 3]) -> VolumeId {
        // A candidate containing `x` wins outright. Failing that, the touching
        // solid whose entry is NEAREST: with a layer thinner than the contact
        // tolerance (a foil, an armour tile), the solid beyond it is within
        // the tolerance too, and taking the first candidate found would skip
        // the layer.
        let mut nearest: Option<(f64, VolumeId)> = None;
        for &k in self.contacts.of(tri) {
            if k == from {
                continue;
            }
            match self.first_facet(k, x, u) {
                Some((t, true, prim)) => {
                    EXIT_AHEAD.set(Some(ExitAhead {
                        geometry: self as *const Self as usize,
                        volume: k,
                        from: x,
                        direction: u,
                        distance: t,
                        prim,
                    }));
                    return k;
                }
                Some((t, false, _))
                    if t <= LOOKBACK.max(self.contact_tolerance)
                        && nearest.is_none_or(|(best, _)| t < best) =>
                {
                    nearest = Some((t, k));
                }
                _ => {}
            }
        }
        match nearest {
            // The nearest solid ahead is apart from this one in the CAD: the
            // gap in front of the particle is real, whatever touches beyond.
            Some((t, k)) if t > LOOKBACK && self.separated(from, k) => {
                self.topology.implicit_complement
            }
            Some((_, k)) => k,
            None => self.topology.implicit_complement,
        }
    }

    /// The exit facet the last relocation on this thread found, if this query
    /// asks for it: same geometry, volume and direction, and an origin on the
    /// same ray, past the relocation point but short of the facet. Taken, so
    /// it answers one query at most.
    fn exit_ahead(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
    ) -> Option<(u32, f64)> {
        let e = EXIT_AHEAD.take()?;
        // The cached exit is the first of this solid's OWN facets; a solid
        // with overlap entries has more to check, so it always queries.
        if matches!(self.overlap_bvhs.get(volume as usize), Some(Some(_))) {
            return None;
        }
        if e.geometry != self as *const Self as usize
            || e.volume != volume
            || e.direction != direction
        {
            return None;
        }
        let d = intersect::sub(origin, e.from);
        let s = intersect::dot(d, direction);
        let off = intersect::sub(d, intersect::scale(direction, s));
        let t = e.distance - s;
        // The transport steps 1e-8 past a crossing; anything else is a
        // different query (a collision moved the particle, say).
        if !(0.0..=1e-6).contains(&s) || intersect::dot(off, off) > 1e-24 || t <= RAY_FIRE_MIN_T {
            return None;
        }
        Some((e.prim, t))
    }

    /// True when solids `a` and `b` come within the contact tolerance in the
    /// mesh but do not touch in the CAD (recorded by the writer): the gap
    /// between them is real, and relocation does not fill it.
    fn separated(&self, a: VolumeId, b: VolumeId) -> bool {
        let pair = (a.min(b), a.max(b));
        !self.topology.separated.is_empty() && self.topology.separated.contains(&pair)
    }

    /// First facet of `volume` met by a ray from `x` along `u`, looking
    /// [`LOOKBACK`] behind `x` so a facet coincident with `x` is not missed:
    /// `(distance from x, whether the ray leaves the volume there, primitive
    /// index in the volume's BVH)`. A facet
    /// the ray leaves through within [`LOOKBACK`] of `x` is skipped, as that
    /// is the volume ending exactly where the particle is.
    fn first_facet(&self, volume: VolumeId, x: [f64; 3], u: [f64; 3]) -> Option<(f64, bool, u32)> {
        let sd = &self.surface_bvhs[volume as usize];
        let o = [
            x[0] - LOOKBACK * u[0],
            x[1] - LOOKBACK * u[1],
            x[2] - LOOKBACK * u[2],
        ];
        let (prim, tp) = sd.bvh.ray_traverse_tris(
            o,
            u,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            0.0,
            |prim, tp| !(sd.leaves(prim as usize, u) && tp <= 2.0 * LOOKBACK),
        )?;
        Some((tp - LOOKBACK, sd.leaves(prim as usize, u), prim))
    }

    /// Fire a ray within a volume. Returns `(distance, surface_id)` of
    /// the next surface crossing, or `None`.
    pub fn ray_fire(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        history: Option<&RayHistory>,
    ) -> Option<(f64, SurfaceId)> {
        if (volume as usize) >= self.surface_bvhs.len() {
            return None;
        }
        let sd = &self.surface_bvhs[volume as usize];
        ray_fire::ray_fire(
            &sd.bvh,
            &sd.tri_ids,
            &sd.surf_ids,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            origin,
            direction,
            history,
        )
        .map(|r| (r.distance, r.surface_id))
    }

    /// Fire a ray and also return the hit triangle ID (for ray history).
    pub fn ray_fire_detailed(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        history: Option<&RayHistory>,
    ) -> Option<ray_fire::RayFireResult> {
        if (volume as usize) >= self.surface_bvhs.len() {
            return None;
        }
        let sd = &self.surface_bvhs[volume as usize];
        ray_fire::ray_fire(
            &sd.bvh,
            &sd.tri_ids,
            &sd.surf_ids,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            origin,
            direction,
            history,
        )
    }

    /// Nearest surface NOT bounding `volume` that intersects the open
    /// ray segment `(eps, t_max - eps)`, or `None`.
    ///
    /// Spatial cross-check for adjacency tracking: a chord of a volume
    /// (from a point inside it to its nearest own boundary) cannot pass
    /// through any foreign surface when the geometry is valid, so a hit
    /// is a geometry error (overlapping or self-intersecting volumes).
    /// Uses the same early-exit nearest-hit queries as tracking; shared
    /// surfaces (which also bound `volume`) are not flagged.
    pub fn segment_blocked(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> Option<(f64, SurfaceId)> {
        // Overlap slivers between independently meshed solids put foreign
        // facets inside a volume by construction; the check does not apply.
        if self.discontinuous {
            return None;
        }
        let eps = 1e-6_f64.max(t_max * 1e-9);
        let t_hi = t_max - eps;
        for w in 0..self.topology.num_volumes {
            if w == volume {
                continue;
            }
            let bb = &self.topology.volume_aabbs[w as usize];
            if !intersect::segment_hits_aabb(origin, direction, t_max, bb) {
                continue;
            }
            let sd = &self.surface_bvhs[w as usize];
            let mut blocked: Option<(f64, SurfaceId)> = None;
            sd.bvh.ray_traverse_upto(origin, direction, t_hi, |prim| {
                let idx = prim as usize;
                let v0 = sd.precomputed.v0[idx];
                let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
                let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
                let t = intersect::ray_triangle_intersect(origin, direction, v0, v1, v2)?;
                if t <= eps || t >= t_hi {
                    return None;
                }
                let sid = sd.surf_ids[idx];
                // A surface bounding the current volume is not foreign.
                // surface_volumes stores the implicit-complement side as
                // None, so when `volume` IS the implicit complement a
                // None side counts as bounding it (grazing re-hits of a
                // just-crossed curved surface land here).
                let ic = self.topology.implicit_complement;
                let (fwd, rev) = self.topology.surface_volumes[sid as usize];
                let bounds_current = |side: Option<VolumeId>| match side {
                    Some(v) => v == volume,
                    None => volume == ic,
                };
                if bounds_current(fwd) || bounds_current(rev) {
                    return None;
                }
                blocked = Some((t, sid));
                // Any-hit: collapse the remaining search immediately.
                Some(f64::MIN_POSITIVE)
            });
            if blocked.is_some() {
                return blocked;
            }
        }
        None
    }

    /// Test if a point is inside a volume.
    pub fn point_in_volume(&self, volume: VolumeId, point: [f64; 3]) -> bool {
        if (volume as usize) >= self.surface_bvhs.len() {
            return false;
        }
        let sd = &self.surface_bvhs[volume as usize];
        point_in_volume::point_in_volume(
            &sd.bvh,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            point,
        )
    }

    /// Find which volume contains a point. Where recorded overlaps make
    /// several contain it, the highest-ranked one.
    pub fn find_volume(&self, point: [f64; 3]) -> VolumeId {
        if self.ranks.is_empty() {
            point_in_volume::find_volume(&self.topology, &self.surface_bvhs, point)
        } else {
            point_in_volume::find_volume_ranked(
                &self.topology,
                &self.surface_bvhs,
                point,
                &self.ranks,
            )
        }
    }

    /// Get the volume on the other side of a surface.
    pub fn next_volume(&self, surface: SurfaceId, current: VolumeId) -> Option<VolumeId> {
        if (surface as usize) >= self.topology.surface_volumes.len() {
            return None;
        }
        let (fwd, rev) = self.topology.surface_volumes[surface as usize];
        if fwd == Some(current) {
            rev
        } else if rev == Some(current) {
            fwd
        } else {
            None
        }
    }

    /// Closest distance from a point to any surface of a volume.
    pub fn closest_to_surface(&self, volume: VolumeId, point: [f64; 3]) -> f64 {
        if (volume as usize) >= self.surface_bvhs.len() {
            return f64::MAX;
        }
        let sd = &self.surface_bvhs[volume as usize];
        let result = sd.bvh.closest_point_traverse(point, |bvh_prim_idx| {
            let idx = bvh_prim_idx as usize;
            let v0 = sd.precomputed.v0[idx];
            let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
            let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
            let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
            dist * dist
        });
        match result {
            Some((_, dist_sq)) => dist_sq.sqrt(),
            None => f64::MAX,
        }
    }

    /// Surface normal at a point (from the closest triangle on the surface).
    ///
    /// Note: This uses a per-surface BVH if the surface is part of a volume.
    /// For surfaces not associated with volumes, it falls back to linear search.
    pub fn surface_normal(&self, surface: SurfaceId, point: [f64; 3]) -> [f64; 3] {
        if (surface as usize) >= self.topology.surface_tri_ranges.len() {
            return [0.0, 0.0, 1.0];
        }

        // Try to find a volume that has this surface to use its BVH
        for sd in &self.surface_bvhs {
            let result = sd.bvh.closest_point_traverse(point, |bvh_prim_idx| {
                let idx = bvh_prim_idx as usize;
                if sd.surf_ids[idx] != surface {
                    return f64::MAX;
                }
                let v0 = sd.precomputed.v0[idx];
                let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
                let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
                let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
                dist * dist
            });
            if let Some((prim_idx, _)) = result {
                let idx = prim_idx as usize;
                if sd.surf_ids[idx] == surface {
                    let e1 = sd.precomputed.edge1[idx];
                    let e2 = sd.precomputed.edge2[idx];
                    return intersect::normalize(intersect::cross(e1, e2));
                }
            }
        }

        // Fallback: linear scan
        let range = &self.topology.surface_tri_ranges[surface as usize];
        let mut best_dist = f64::MAX;
        let mut best_normal = [0.0, 0.0, 1.0];

        for i in range.start..range.end {
            let tri_id = self.topology.surface_tri_indices[i as usize];
            let tri = &self.topology.triangles[tri_id as usize];
            let v0 = self.topology.vertices[tri[0] as usize];
            let v1 = self.topology.vertices[tri[1] as usize];
            let v2 = self.topology.vertices[tri[2] as usize];
            let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
            if dist < best_dist {
                best_dist = dist;
                best_normal = intersect::triangle_normal(v0, v1, v2);
            }
        }

        best_normal
    }

    /// Boundary condition for a surface.
    pub fn boundary_condition(&self, surface: SurfaceId) -> BoundaryCondition {
        self.topology
            .physical_data
            .surface_bcs
            .get(&surface)
            .copied()
            .unwrap_or(BoundaryCondition::Transmission)
    }

    /// Material name for a volume.
    pub fn material_name(&self, volume: VolumeId) -> Option<&str> {
        self.topology
            .physical_data
            .volume_materials
            .get(&volume)
            .and_then(|m| m.material_name.as_deref())
    }

    // -- Element walking (tet mesh overlay) ---------------------------------

    /// Find which tetrahedron contains a point, within a volume.
    pub fn find_element(&self, volume: VolumeId, point: [f64; 3]) -> Option<TetrahedronId> {
        if (volume as usize) >= self.element_bvhs.len() {
            return None;
        }
        let (ref bvh, ref tet_ids) = self.element_bvhs[volume as usize];
        element_walk::find_element(&self.topology, bvh, tet_ids, point)
    }

    /// Walk a ray through the tet mesh, returning (tet_id, path_length) segments.
    pub fn walk_elements(
        &self,
        start_element: TetrahedronId,
        origin: [f64; 3],
        direction: [f64; 3],
        max_distance: f64,
    ) -> Vec<(TetrahedronId, f64)> {
        element_walk::walk_elements(
            &self.topology,
            start_element,
            origin,
            direction,
            max_distance,
        )
    }

    /// Compute segments from start to end through the tet mesh of a volume.
    pub fn segments(
        &self,
        volume: VolumeId,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Vec<(TetrahedronId, f64)> {
        if (volume as usize) >= self.element_bvhs.len() {
            return Vec::new();
        }
        let (ref bvh, ref tet_ids) = self.element_bvhs[volume as usize];
        element_walk::segments(&self.topology, bvh, tet_ids, volume, start, end)
    }

    // -- Measurements -------------------------------------------------------

    /// Compute the volume of a mesh volume using the divergence theorem.
    pub fn measure_volume(&self, volume: VolumeId) -> f64 {
        if (volume as usize) >= self.topology.volume_surfaces.len() {
            return 0.0;
        }
        let mut total = 0.0;
        for &(surf_id, sense) in &self.topology.volume_surfaces[volume as usize] {
            let range = &self.topology.surface_tri_ranges[surf_id as usize];
            for i in range.start..range.end {
                let tri_id = self.topology.surface_tri_indices[i as usize];
                let tri = &self.topology.triangles[tri_id as usize];
                let v0 = self.topology.vertices[tri[0] as usize];
                let v1 = self.topology.vertices[tri[1] as usize];
                let v2 = self.topology.vertices[tri[2] as usize];
                let contrib = intersect::triangle_volume_contribution(v0, v1, v2);
                match sense {
                    Sense::Forward => total += contrib,
                    Sense::Reverse => total -= contrib,
                }
            }
        }
        (total / 6.0).abs()
    }

    /// Compute the area of a mesh surface.
    pub fn measure_surface_area(&self, surface: SurfaceId) -> f64 {
        if (surface as usize) >= self.topology.surface_tri_ranges.len() {
            return 0.0;
        }
        let range = &self.topology.surface_tri_ranges[surface as usize];
        let mut total = 0.0;
        for i in range.start..range.end {
            let tri_id = self.topology.surface_tri_indices[i as usize];
            let tri = &self.topology.triangles[tri_id as usize];
            let v0 = self.topology.vertices[tri[0] as usize];
            let v1 = self.topology.vertices[tri[1] as usize];
            let v2 = self.topology.vertices[tri[2] as usize];
            total += intersect::triangle_area(v0, v1, v2);
        }
        total
    }

    /// Compute the volume of a tetrahedron.
    pub fn tet_volume(&self, tet: TetrahedronId) -> f64 {
        if (tet as usize) >= self.topology.tetrahedra.len() {
            return 0.0;
        }
        let verts = &self.topology.tetrahedra[tet as usize];
        let v0 = self.topology.vertices[verts[0] as usize];
        let v1 = self.topology.vertices[verts[1] as usize];
        let v2 = self.topology.vertices[verts[2] as usize];
        let v3 = self.topology.vertices[verts[3] as usize];
        intersect::tet_volume(v0, v1, v2, v3)
    }

    /// Number of (non-implicit-complement) volumes.
    pub fn num_volumes(&self) -> u32 {
        self.topology.num_volumes
    }

    /// Number of surfaces.
    pub fn num_surfaces(&self) -> u32 {
        self.topology.num_surfaces
    }

    /// The implicit complement volume ID.
    pub fn implicit_complement(&self) -> VolumeId {
        self.topology.implicit_complement
    }
}

// ---------------------------------------------------------------------------
// BVH building helpers
// ---------------------------------------------------------------------------

/// Build per-volume surface BVHs with precomputed triangle data.
///
/// For each volume, collect all triangles on all its surfaces, build
/// one BVH for ray firing, and precompute edge vectors for each triangle.
fn build_surface_bvhs(topo: &MeshTopology) -> Vec<SurfaceBvhData> {
    // Build BVH for each real volume plus the implicit complement.
    // The complement entry (at index num_volumes) was appended to
    // volume_surfaces by the topology builder, so indexing is correct.
    let iter = 0..topo.num_volumes + 1;
    let mapper = |vol_id: u32| {
        let mut tri_ids: Vec<TriangleId> = Vec::new();
        let mut surf_ids: Vec<SurfaceId> = Vec::new();
        let mut aabbs: Vec<[f64; 6]> = Vec::new();
        let mut exit_sign: Vec<i8> = Vec::new();
        let ic = topo.implicit_complement;
        let real = |v: Option<VolumeId>| matches!(v, Some(v) if v != ic);

        for &(surf_id, sense) in &topo.volume_surfaces[vol_id as usize] {
            let (fwd, rev) = topo.surface_volumes[surf_id as usize];
            let sign: i8 = if !real(fwd) && !real(rev) {
                0
            } else {
                match sense {
                    Sense::Forward => 1,
                    Sense::Reverse => -1,
                }
            };
            let range = &topo.surface_tri_ranges[surf_id as usize];
            for i in range.start..range.end {
                let tri_id = topo.surface_tri_indices[i as usize];
                tri_ids.push(tri_id);
                surf_ids.push(surf_id);
                aabbs.push(topo.triangle_aabbs[tri_id as usize]);
                exit_sign.push(sign);
            }
        }

        // Precompute v0, edge1, edge2 for each triangle
        let n = tri_ids.len();
        let mut v0s = Vec::with_capacity(n);
        let mut edge1s = Vec::with_capacity(n);
        let mut edge2s = Vec::with_capacity(n);
        for &tri_id in &tri_ids {
            let tri = &topo.triangles[tri_id as usize];
            let va = topo.vertices[tri[0] as usize];
            let vb = topo.vertices[tri[1] as usize];
            let vc = topo.vertices[tri[2] as usize];
            v0s.push(va);
            edge1s.push(intersect::sub(vb, va));
            edge2s.push(intersect::sub(vc, va));
        }

        let bvh = Bvh::build(&aabbs);
        SurfaceBvhData {
            bvh,
            tri_ids,
            surf_ids,
            precomputed: PrecomputedTriData {
                v0: v0s,
                edge1: edge1s,
                edge2: edge2s,
            },
            exit_sign,
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        iter.into_par_iter().map(mapper).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        iter.map(mapper).collect()
    }
}

/// Per volume, the facets of the higher-ranked solids overlapping it (see
/// [`MeshTopology::overlaps`]), oriented so that crossing one INTO its solid
/// counts. `None` for a volume no higher-ranked solid overlaps.
fn build_overlap_bvhs(topo: &MeshTopology) -> Vec<Option<OverlapBvh>> {
    let n = topo.num_volumes as usize;
    let mut higher: Vec<Vec<VolumeId>> = vec![Vec::new(); n];
    for &(low, high) in &topo.overlaps {
        if !higher[low as usize].contains(&high) {
            higher[low as usize].push(high);
        }
    }
    let mapper = |vol: usize| -> Option<OverlapBvh> {
        if higher[vol].is_empty() {
            return None;
        }
        let mut tri_ids = Vec::new();
        let mut surf_ids = Vec::new();
        let mut aabbs = Vec::new();
        let mut exit_sign = Vec::new();
        let mut enters = Vec::new();
        for &j in &higher[vol] {
            for &(surf_id, sense) in &topo.volume_surfaces[j as usize] {
                // Inverted: `leaves` then means the ray enters `j`.
                let sign: i8 = match sense {
                    Sense::Forward => -1,
                    Sense::Reverse => 1,
                };
                let range = &topo.surface_tri_ranges[surf_id as usize];
                for i in range.start..range.end {
                    let tri_id = topo.surface_tri_indices[i as usize];
                    tri_ids.push(tri_id);
                    surf_ids.push(surf_id);
                    aabbs.push(topo.triangle_aabbs[tri_id as usize]);
                    exit_sign.push(sign);
                    enters.push(j);
                }
            }
        }
        let mut v0 = Vec::with_capacity(tri_ids.len());
        let mut edge1 = Vec::with_capacity(tri_ids.len());
        let mut edge2 = Vec::with_capacity(tri_ids.len());
        for &tri_id in &tri_ids {
            let tri = &topo.triangles[tri_id as usize];
            let a = topo.vertices[tri[0] as usize];
            v0.push(a);
            edge1.push(intersect::sub(topo.vertices[tri[1] as usize], a));
            edge2.push(intersect::sub(topo.vertices[tri[2] as usize], a));
        }
        Some(OverlapBvh {
            data: SurfaceBvhData {
                bvh: Bvh::build(&aabbs),
                tri_ids,
                surf_ids,
                precomputed: PrecomputedTriData { v0, edge1, edge2 },
                exit_sign,
            },
            enters,
        })
    };
    (0..n).map(mapper).collect()
}

/// Build per-volume element BVHs (for point-in-tet queries).
fn build_element_bvhs(topo: &MeshTopology) -> Vec<(Bvh, Vec<TetrahedronId>)> {
    let iter = 0..topo.num_volumes;
    let mapper = |vol_id: u32| {
        let range = &topo.volume_tet_ranges[vol_id as usize];
        let mut tet_ids: Vec<TetrahedronId> = Vec::new();
        let mut aabbs: Vec<[f64; 6]> = Vec::new();

        for i in range.start..range.end {
            let tet_id = topo.volume_tet_indices[i as usize];
            tet_ids.push(tet_id);
            aabbs.push(topo.tet_aabbs[tet_id as usize]);
        }

        let bvh = Bvh::build(&aabbs);
        (bvh, tet_ids)
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        iter.into_par_iter().map(mapper).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        iter.map(mapper).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ----------------------------- Tests -----------------------------

#[cfg(all(test, feature = "arrow"))]
mod tests {
    use super::*;

    fn cube_geometry() -> MeshGeometry {
        MeshGeometry::from_arrow(std::path::Path::new("tests/data/cube.arrow")).unwrap()
    }

    fn two_region_geometry() -> MeshGeometry {
        MeshGeometry::from_arrow(std::path::Path::new("tests/data/two_region_tets.arrow")).unwrap()
    }

    #[test]
    fn test_point_outside_with_two_crossings_ahead() {
        // The parity ray from a point BELOW the cube crosses two faces
        // (bottom and top): even count, outside. Routing the parity
        // enumeration through the nearest-hit-pruned ray_traverse dropped
        // the second hit and misclassified this as inside,
        // which put every mesh-transport history whose source was not in
        // the first tested volume into the wrong birth cell.
        let geom = cube_geometry();
        assert!(!geom.point_in_volume(0, [0.5, 0.5, -1.0]));
        assert!(!geom.point_in_volume(0, [0.5, -5.0, 0.5]));
        assert_eq!(
            geom.find_volume([0.5, 0.5, -1.0]),
            geom.topology.implicit_complement
        );
    }

    #[test]
    fn test_cube_point_in_volume() {
        let geom = cube_geometry();
        // Center of cube should be in volume 0
        assert!(geom.point_in_volume(0, [0.5, 0.5, 0.5]));
        // Outside the cube
        assert!(!geom.point_in_volume(0, [2.0, 0.5, 0.5]));
    }

    #[test]
    fn test_cube_find_volume() {
        let geom = cube_geometry();
        assert_eq!(geom.find_volume([0.5, 0.5, 0.5]), 0);
        assert_eq!(
            geom.find_volume([2.0, 2.0, 2.0]),
            geom.implicit_complement()
        );
    }

    #[test]
    fn test_cube_ray_fire() {
        let geom = cube_geometry();
        // Ray from center in +x direction should hit a surface at distance ~0.5
        let result = geom.ray_fire(0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], None);
        assert!(result.is_some());
        let (dist, _surf) = result.unwrap();
        assert!((dist - 0.5).abs() < 0.1, "Expected ~0.5, got {dist}");
    }

    #[test]
    fn test_cube_next_volume() {
        let geom = cube_geometry();
        // Fire ray and cross surface
        if let Some((_, surf)) = geom.ray_fire(0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], None) {
            let next = geom.next_volume(surf, 0);
            assert!(next.is_some());
            // For the cube, the other side is the implicit complement
            assert_eq!(next.unwrap(), geom.implicit_complement());
        }
    }

    #[test]
    fn test_cube_boundary_condition() {
        let geom = cube_geometry();
        for surf_id in 0..geom.num_surfaces() {
            assert_eq!(geom.boundary_condition(surf_id), BoundaryCondition::Vacuum);
        }
    }

    #[test]
    fn test_cube_material() {
        let geom = cube_geometry();
        assert_eq!(geom.material_name(0), Some("water"));
    }

    #[test]
    fn test_cube_measure_volume() {
        let geom = cube_geometry();
        let vol = geom.measure_volume(0);
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_cube_volume_measures_precomputed() {
        let geom = cube_geometry();
        // volume_measures should be populated by from_topology()
        assert_eq!(geom.topology.volume_measures.len(), 1);
        let vol = geom.topology.volume_measures[0];
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Precomputed volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_two_region_volume_measures() {
        let geom = two_region_geometry();
        assert_eq!(geom.topology.volume_measures.len(), 2);
        // Each half-cube is 0.5 x 1 x 1 = 0.5 cm³
        for &vol in &geom.topology.volume_measures {
            assert!(
                (vol - 0.5).abs() < 0.01,
                "Half-cube volume should be ~0.5, got {vol}"
            );
        }
    }

    #[test]
    fn test_cube_surface_normal() {
        let geom = cube_geometry();
        // Get normal at a point on the +x face (x=1.0)
        // This is approximate -- depends on which surface is the +x face
        let _normal = geom.surface_normal(0, [1.0, 0.5, 0.5]);
        // Just verify it returns a unit vector
        let n = geom.surface_normal(0, [0.5, 0.5, 0.0]);
        let len = intersect::length(n);
        assert!((len - 1.0).abs() < 1e-6 || len < 1e-10);
    }

    #[test]
    fn test_cube_closest_to_surface() {
        let geom = cube_geometry();
        let dist = geom.closest_to_surface(0, [0.5, 0.5, 0.5]);
        // Center of unit cube → closest surface is 0.5 away
        assert!(
            (dist - 0.5).abs() < 0.1,
            "Distance from center to nearest face should be ~0.5, got {dist}"
        );
    }

    #[test]
    fn test_two_region_materials() {
        let geom = two_region_geometry();
        assert_eq!(geom.num_volumes(), 2);
        assert_eq!(geom.material_name(0), Some("fuel"));
        assert_eq!(geom.material_name(1), Some("moderator"));
    }

    #[test]
    fn test_cube_find_element() {
        let geom = cube_geometry();
        let tet = geom.find_element(0, [0.5, 0.5, 0.5]);
        assert!(tet.is_some(), "Should find a tet at cube center");
    }

    #[test]
    fn test_cube_tet_volume() {
        let geom = cube_geometry();
        // Sum of all tet volumes should equal the cube volume (~1.0)
        let mut total = 0.0;
        for tet_id in 0..geom.topology.tetrahedra.len() as u32 {
            total += geom.tet_volume(tet_id);
        }
        assert!(
            (total - 1.0).abs() < 0.1,
            "Sum of tet volumes should be ~1.0, got {total}"
        );
    }
}
