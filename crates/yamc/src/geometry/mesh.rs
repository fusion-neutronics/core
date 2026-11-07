//! Mesh-based geometry backend using `yamt`.
//!
//! Wraps a `yamt::MeshGeometry` to provide the same interface that the
//! transport loop expects. Each mesh volume maps to a `Cell` (with
//! material from physical group names). Ray-surface queries use the
//! mesh BVH instead of CSG evaluation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::geo::BoundingBox;
use crate::geo::{BoundaryType, Surface, SurfaceKind};
use crate::geo::{HalfspaceType, Region, RegionExpr};
use crate::geometry::backend::BoundaryHit;
use crate::geometry::cell::Cell;
use yamc_materials::material::Material;

/// Mesh-based geometry backend.
///
/// Each mesh volume becomes a [`Cell`] with the material looked up from
/// physical group names (`mat:<name>`). The implicit complement (unmeshed
/// space between volumes) is also represented as a cell -- void by default,
/// or with a user-supplied material for air/coolant interactions.
///
/// Ray-fire and point-in-volume queries are delegated to the underlying
/// [`yamt::MeshGeometry`].
// `MeshGeometry` wraps a `yamt::MeshGeometry` (BVH + triangulated surfaces)
// that has no full JSON form. For the model *fingerprint* (combine_results)
// it only needs a STABLE, identity-capturing summary -- not a round-trippable
// encoding. `serialize` emits that summary (counts, AABB, per-volume measures
// and material names): the same mesh fingerprints identically, geometrically
// different meshes differ. `deserialize` still fails loudly -- a fingerprint
// summary cannot reconstruct a mesh, and no path round-trips a Mesh-variant
// model from JSON.
impl serde::Serialize for MeshGeometry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let topo = &self.mesh.topology;
        let volume_materials: Vec<Option<&str>> = (0..topo.num_volumes)
            .map(|v| self.mesh.material_name(v))
            .collect();
        let mut st = serializer.serialize_struct("MeshGeometry", 5)?;
        st.serialize_field("kind", "mesh")?;
        st.serialize_field("num_volumes", &topo.num_volumes)?;
        st.serialize_field("global_aabb", &topo.global_aabb)?;
        st.serialize_field("volume_measures", &topo.volume_measures)?;
        st.serialize_field("volume_materials", &volume_materials)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for MeshGeometry {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "MeshGeometry cannot be deserialized from JSON -- it serializes only a \
             lossy fingerprint summary, not a reconstructable form",
        ))
    }
}

#[derive(Clone)]
pub struct MeshGeometry {
    /// The underlying mesh geometry (BVH-accelerated). Shared, not copied,
    /// when the geometry is cloned: it is never changed after construction,
    /// and it holds nearly all of the memory.
    pub mesh: Arc<yamt::MeshGeometry>,
    /// Synthetic cells: one per mesh volume + one for the implicit complement.
    pub cells: Vec<Cell>,
    /// Flat material store. `Cell.material_idx` is a slot index into this Vec.
    pub materials: Vec<Arc<Material>>,
    /// Volume ID for each cell index: `volume_ids[cell_index] = VolumeId`.
    volume_ids: Vec<yamt::VolumeId>,
    /// Vacuum box `[xmin, ymin, zmin, xmax, ymax, zmax]` added around the
    /// mesh by the model before transport (see [`Self::add_graveyard`]),
    /// or `None`. Separate from a `graveyard_offset` box, which is part of
    /// the mesh itself.
    graveyard: Option<[f64; 6]>,
}

/// Offset (cm) of the vacuum box a model adds around a whole-model mesh
/// geometry that has no vacuum boundary and a void implicit complement.
/// The box is the mesh's bounding box grown by this on every side. In a
/// void complement the offset changes no result; it only decides where
/// escaping particles end, so it is a fixed margin that leaves room for
/// sources and tally meshes that sit a little outside the meshed parts.
pub const DEFAULT_GRAVEYARD_OFFSET: f64 = 10.0;

/// Give every material in `materials`, and the implicit complement material,
/// an id: the ones without draw a fresh id from the process-wide allocator.
///
/// In name order, with the implicit complement last, so the ids do not depend
/// on `HashMap` iteration order, which differs from run to run. A material
/// reached through several names (or also used as the implicit complement)
/// is one material and gets one id.
fn assign_ids(
    materials: &mut HashMap<String, Arc<Material>>,
    implicit_complement_material: &mut Option<Arc<Material>>,
) -> Result<(), String> {
    let mut identified: HashMap<*const Material, Arc<Material>> = HashMap::new();
    let mut identify = |material: &mut Arc<Material>| -> Result<(), String> {
        if material.get_material_id().is_some() {
            return Ok(());
        }
        let original = Arc::as_ptr(material);
        if let Some(done) = identified.get(&original) {
            *material = Arc::clone(done);
            return Ok(());
        }
        Arc::make_mut(material).assign_material_id(None)?;
        identified.insert(original, Arc::clone(material));
        Ok(())
    };
    let mut by_name: Vec<(&String, &mut Arc<Material>)> = materials.iter_mut().collect();
    by_name.sort_unstable_by(|a, b| a.0.cmp(b.0));
    for (_, material) in by_name {
        identify(material)?;
    }
    if let Some(material) = implicit_complement_material {
        identify(material)?;
    }
    Ok(())
}

/// Comma-separated quoted names, cut to the first 20 with a count of the rest.
fn name_list(names: &[&str]) -> String {
    const SHOWN: usize = 20;
    let mut s = names
        .iter()
        .take(SHOWN)
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > SHOWN {
        s += &format!(", ... ({} in total)", names.len());
    }
    s
}

/// Levenshtein distance between two strings, by characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let next = (diag + usize::from(ca != cb))
                .min(row[j] + 1)
                .min(row[j + 1] + 1);
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

/// Closest volume name to `key`: the name left after stripping a `mat:`
/// prefix, else a case-insensitive match, else the nearest name within a
/// third of the key's length in edits.
fn closest_name<'a>(key: &str, names: &[&'a str]) -> Option<&'a str> {
    let bare = key.strip_prefix("mat:").unwrap_or(key);
    names
        .iter()
        .find(|n| n.eq_ignore_ascii_case(bare))
        .copied()
        .or_else(|| {
            names
                .iter()
                .map(|n| (edit_distance(&bare.to_lowercase(), &n.to_lowercase()), *n))
                .filter(|&(d, _)| d <= bare.chars().count().div_ceil(3))
                .min()
                .map(|(_, n)| n)
        })
}

/// Fail when a key of `materials` names no volume, so a typo or a `mat:`
/// prefix cannot silently leave volumes void.
fn check_material_keys(
    materials: &HashMap<String, Arc<Material>>,
    names: &[Option<&str>],
) -> Result<(), String> {
    let mut available: Vec<&str> = names.iter().flatten().copied().collect();
    available.sort_unstable();
    available.dedup();
    let mut unmatched: Vec<&str> = materials
        .keys()
        .map(String::as_str)
        .filter(|k| available.binary_search(k).is_err())
        .collect();
    if unmatched.is_empty() {
        return Ok(());
    }
    unmatched.sort_unstable();
    let hints: Vec<String> = unmatched
        .iter()
        .filter_map(|k| closest_name(k, &available).map(|n| format!("'{n}' for '{k}'")))
        .collect();
    let hint = if hints.is_empty() {
        String::new()
    } else {
        format!(" Did you mean {}?", hints.join(", "))
    };
    Err(format!(
        "materials keys match no volume: {}. Volume names are: {}.{hint}",
        name_list(&unmatched),
        if available.is_empty() {
            "(none)".to_string()
        } else {
            name_list(&available)
        }
    ))
}

impl MeshGeometry {
    /// Build a `MeshGeometry` from a loaded mesh and a material map.
    ///
    /// # Arguments
    /// * `mesh` -- already-loaded `yamt::MeshGeometry`
    /// * `materials` -- mapping from volume name (e.g. `"fuel"`) to YAMC
    ///   `Material`. A volume's name is its `mat:<name>` physical group
    ///   name without the `mat:` prefix, so keys are bare names.
    /// * `implicit_complement_material` -- optional material for the implicit
    ///   complement (unmeshed space). `None` means void (free-streaming,
    ///   `dist_collision = INFINITY`). `Some(mat)` assigns a real material
    ///   for air/coolant interactions in the gaps between mesh volumes.
    ///
    /// A material without an id gets a fresh one from the process-wide
    /// allocator, in name order, so the ids do not depend on the map's
    /// iteration order.
    ///
    /// Volumes whose name has no key in `materials` are void, reported in
    /// one summary line.
    ///
    /// # Errors
    /// Returns an error naming every key of `materials` that matches no
    /// volume name, with the available names and a close match when one exists.
    pub fn new(
        mesh: yamt::MeshGeometry,
        materials: &HashMap<String, Arc<Material>>,
        mut implicit_complement_material: Option<Arc<Material>>,
    ) -> Result<Self, String> {
        let num_volumes = mesh.topology.num_volumes;
        let names: Vec<Option<&str>> = (0..num_volumes).map(|v| mesh.material_name(v)).collect();
        check_material_keys(materials, &names)?;

        let mut cells = Vec::with_capacity(num_volumes as usize + 1);
        let mut volume_ids = Vec::with_capacity(num_volumes as usize + 1);

        let mut materials = materials.clone();
        assign_ids(&mut materials, &mut implicit_complement_material)?;

        // Dummy region for mesh cells -- CSG operations are never evaluated.
        let dummy_region = make_dummy_region();

        let mut flat_materials: Vec<Arc<Material>> = Vec::new();
        let mut void_names: Vec<&str> = Vec::new();

        for vol_id in 0..num_volumes {
            let mat_name = names[vol_id as usize];

            // Use precomputed volume from topology (divergence theorem)
            let vol_measure = mesh.topology.volume_measures.get(vol_id as usize).copied();

            let material_idx = match mat_name {
                Some(name) => match materials.get(name) {
                    Some(mat) => {
                        // Propagate volume to material for transmutation
                        let per_volume = if let Some(vol) = vol_measure {
                            let mut mat_with_vol = (**mat).clone();
                            mat_with_vol.volume = Some(vol);
                            Arc::new(mat_with_vol)
                        } else {
                            Arc::clone(mat)
                        };
                        let idx = flat_materials.len() as u32;
                        flat_materials.push(per_volume);
                        Some(idx)
                    }
                    None => {
                        void_names.push(name);
                        None
                    }
                },
                None => None, // Void volume (no material)
            };

            let cell = Cell {
                cell_id: Some(vol_id),
                name: mat_name.map(|n| format!("mesh_vol_{vol_id}_{n}")),
                flat_region: dummy_region.flatten(),
                region: dummy_region.clone(),
                material_idx,
                volume: vol_measure,
                fill_role: None,
                cached_surfaces: Vec::new(), // Not used -- mesh handles boundaries
                has_free_form: false,
            };

            cells.push(cell);
            volume_ids.push(vol_id);
        }

        if !void_names.is_empty() {
            void_names.sort_unstable();
            void_names.dedup();
            eprintln!(
                "Warning: {} named volume(s) have no entry in materials and are void: {}",
                void_names.len(),
                name_list(&void_names)
            );
        }

        // Implicit complement cell -- the last cell. Void (material_idx: None)
        // unless the user supplied a material for air/coolant interactions.
        let ic_vol = mesh.topology.implicit_complement;
        let ic_material_idx = implicit_complement_material.map(|mat| {
            let idx = flat_materials.len() as u32;
            flat_materials.push(mat);
            idx
        });
        let ic_cell = Cell {
            cell_id: Some(ic_vol),
            name: Some("implicit_complement".to_string()),
            flat_region: dummy_region.flatten(),
            region: dummy_region,
            material_idx: ic_material_idx,
            volume: None,
            fill_role: None,
            cached_surfaces: Vec::new(),
            has_free_form: false,
        };
        cells.push(ic_cell);
        volume_ids.push(ic_vol);

        Ok(MeshGeometry {
            mesh: Arc::new(mesh),
            cells,
            materials: flat_materials,
            volume_ids,
            graveyard: None,
        })
    }

    /// True when the mesh has a surface with a vacuum boundary condition
    /// (a `graveyard_offset` box or a `boundary:vacuum` group) or a vacuum
    /// box added by [`Self::add_graveyard`].
    pub fn has_vacuum_boundary(&self) -> bool {
        self.graveyard.is_some()
            || self
                .mesh
                .topology
                .physical_data
                .surface_bcs
                .values()
                .any(|&bc| bc == yamt::BoundaryCondition::Vacuum)
    }

    /// True when the implicit complement carries a material.
    pub fn implicit_complement_has_material(&self) -> bool {
        self.cells.last().is_some_and(|c| c.material_idx.is_some())
    }

    /// The vacuum box [`Self::add_graveyard`] would add with `offset`: the
    /// mesh's bounding box grown by `offset` on every side, as
    /// `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub fn graveyard_box(&self, offset: f64) -> [f64; 6] {
        let bb = &self.mesh.topology.global_aabb;
        [
            bb[0] - offset,
            bb[1] - offset,
            bb[2] - offset,
            bb[3] + offset,
            bb[4] + offset,
            bb[5] + offset,
        ]
    }

    /// Close the geometry with a vacuum box `offset` cm outside the mesh's
    /// bounding box, returning the box. The box bounds the implicit
    /// complement as a `graveyard_offset` box of the same offset does: a
    /// particle in the complement that reaches it leaks. It is added
    /// analytically rather than as triangles, so the mesh is not rebuilt.
    pub fn add_graveyard(&mut self, offset: f64) -> [f64; 6] {
        let bb = self.graveyard_box(offset);
        self.graveyard = Some(bb);
        bb
    }

    /// The box particles are transported in: the vacuum box from
    /// [`Self::add_graveyard`] when one was added, else the mesh's bounding
    /// box (which holds a `graveyard_offset` box when there is one).
    pub fn transport_extent(&self) -> [f64; 6] {
        self.graveyard.unwrap_or(self.mesh.topology.global_aabb)
    }

    /// Build from an Arrow IPC file produced by cad_to_yamc.
    pub fn from_arrow(
        path: &std::path::Path,
        materials: &HashMap<String, Arc<Material>>,
    ) -> Result<Self, String> {
        let mesh = yamt::MeshGeometry::from_arrow(path)
            .map_err(|e| format!("Failed to load Arrow mesh: {e}"))?;
        Self::new(mesh, materials, None)
    }

    /// True when a surface that does not bound `cell_index`'s volume
    /// intersects the open segment from `origin` towards the accepted
    /// crossing at distance `t_max`. Used by the spatial tracking
    /// verification path: for valid geometry no foreign
    /// surface can lie strictly inside the current volume, so a hit
    /// means the mesh volumes overlap or self-intersect and adjacency
    /// tracking is about to go wrong. Works for the implicit complement
    /// cell as well (foreign then means any surface not bounding it).
    pub(crate) fn crossing_blocked(
        &self,
        cell_index: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> bool {
        let vol = self.volume_ids[cell_index];
        self.mesh
            .segment_blocked(vol, origin, direction, t_max)
            .is_some()
    }

    /// Find the cell containing a point.
    ///
    /// Delegates to [`yamt::MeshGeometry::try_find_volume_from`] and maps the
    /// result to a cell index. Points in the implicit complement return the
    /// complement cell (last cell). `None` for a point the mesh cannot place
    /// (see [`MeshGeometry::locate_cell`]).
    pub fn find_cell_index(&self, point: (f64, f64, f64)) -> Option<usize> {
        self.locate_cell(point, usize::MAX)
    }

    /// The cell containing `point`, asking `hint` (the cell the particle was
    /// last in) first. A point in no volume, inside the mesh's bounding box or
    /// outside it, is in the implicit complement: leaving the geometry is
    /// found along the flight ([`MeshGeometry::distance_to_exit`]).
    ///
    /// `None` for a point that is not a number, and for one the mesh cannot
    /// place: no volume is known to contain it and one that might cannot say,
    /// because a crack, fold or flipped facet near it breaks every ray it is
    /// tested along. Putting that point in the implicit complement would
    /// carry the particle on as void from inside a solid, to leave the
    /// geometry later as an ordinary leak; `None` makes it a lost particle.
    pub fn locate_cell(&self, point: (f64, f64, f64), hint: usize) -> Option<usize> {
        let p = [point.0, point.1, point.2];
        if !p.iter().all(|v| v.is_finite()) {
            return None;
        }
        let hint = self.volume_ids.get(hint).copied().unwrap_or(u32::MAX);
        self.mesh
            .try_find_volume_from(hint, p)
            .map(|vol| self.cell_of_volume(vol))
    }

    /// Distance along a flight to where it leaves the geometry, if that is
    /// within `max_distance`: the first vacuum-boundary triangle it meets
    /// (see [`yamt::MeshGeometry::distance_to_vacuum`]), or where it leaves
    /// [`Self::transport_extent`] for good, beyond which there is nothing to
    /// meet: the vacuum box added by [`Self::add_graveyard`], which surface
    /// tracking stops at too, or else the mesh's bounding box.
    /// A flight from a point that is not a number leaves at once.
    pub(crate) fn distance_to_exit(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        max_distance: f64,
    ) -> Option<f64> {
        if !origin.iter().chain(&direction).all(|v| v.is_finite()) {
            return Some(0.0);
        }
        let leaves_box = box_exit(origin, direction, &self.transport_extent());
        let vacuum = self
            .mesh
            .distance_to_vacuum(origin, direction, max_distance.min(leaves_box));
        let exit = vacuum.unwrap_or(leaves_box);
        (exit <= max_distance).then_some(exit)
    }

    fn cell_of_volume(&self, vol: u32) -> usize {
        if vol == self.mesh.topology.implicit_complement {
            // The complement cell is the last cell.
            self.cells.len() - 1
        } else {
            // volume_ids[i] == i for 0..num_volumes, so vol == cell_index
            vol as usize
        }
    }

    /// Find the closest boundary crossing from a position along a direction.
    pub fn closest_boundary(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
    ) -> Option<BoundaryHit> {
        let vol_id = self.volume_ids[cell_index];
        let ic_cell_index = self.cells.len() - 1;

        let hit = if self.mesh.is_discontinuous() {
            let ic = self.mesh.topology.implicit_complement;
            self.mesh.next_crossing(vol_id, position, direction).map(
                |(distance, surface_id, beyond)| {
                    let boundary = match self.mesh.boundary_condition(surface_id) {
                        yamt::BoundaryCondition::Vacuum => BoundaryType::Vacuum,
                        _ => BoundaryType::Transmission,
                    };
                    let (next_cell_index, relocate_triangle) = match beyond {
                        _ if boundary == BoundaryType::Vacuum => (None, None),
                        yamt::Beyond::Volume(v) => (
                            v.map(|v| if v == ic { ic_cell_index } else { v as usize }),
                            None,
                        ),
                        yamt::Beyond::Relocate(tri) => (None, Some(tri)),
                    };
                    BoundaryHit {
                        distance,
                        boundary,
                        next_cell_index,
                        surface_id: None,
                        relocate_triangle,
                    }
                },
            )
        } else {
            let ahead = self.mesh.ray_fire_leaving(vol_id, position, direction);
            if let Some((_, _, false)) = ahead {
                // The nearest facet ahead is one the ray would enter this
                // volume through, and the particle is not inside it short of
                // that facet: it left through a crack. Crossing would put it
                // in a cell it is not in, so it is reported as lost (as no
                // surface ahead is), rather than tracked on silently.
                return None;
            }
            ahead.map(|(distance, surface_id, _)| {
                let bc = self.mesh.boundary_condition(surface_id);
                let boundary = match bc {
                    yamt::BoundaryCondition::Vacuum => BoundaryType::Vacuum,
                    _ => BoundaryType::Transmission,
                };

                // Use topology to determine the next cell (O(1) lookup).
                let next_cell_index = if boundary == BoundaryType::Vacuum {
                    None // Particle dies at vacuum boundary
                } else {
                    self.mesh.next_volume(surface_id, vol_id).map(|next_vol| {
                        if next_vol == self.mesh.topology.implicit_complement {
                            // Entering the implicit complement -- particle
                            // free-streams (void) or scatters (if material set).
                            ic_cell_index
                        } else {
                            next_vol as usize
                        }
                    })
                };

                BoundaryHit {
                    distance,
                    boundary,
                    next_cell_index,
                    surface_id: None, // Mesh surfaces don't have user-facing IDs
                    relocate_triangle: None,
                }
            })
        };
        // The added vacuum box bounds the implicit complement: its first
        // crossing ahead ends the particle unless a mesh surface is nearer.
        // It applies to both trackings, conformal and discontinuous.
        if cell_index == ic_cell_index
            && let Some(bb) = &self.graveyard
            && let Some(distance) = box_crossing(position, direction, bb)
            && hit.as_ref().is_none_or(|h| distance < h.distance)
        {
            return Some(BoundaryHit {
                distance,
                boundary: BoundaryType::Vacuum,
                next_cell_index: None,
                surface_id: None,
                relocate_triangle: None,
            });
        }
        hit
    }

    /// Fill in the cell beyond a crossing that [`MeshGeometry::closest_boundary`]
    /// deferred: the particle leaves its solid at `position + distance *
    /// direction` through `hit.relocate_triangle`, and is relocated there.
    pub(crate) fn resolve_crossing(
        &self,
        cell_index: usize,
        hit: &mut BoundaryHit,
        position: [f64; 3],
        direction: [f64; 3],
    ) {
        let Some(tri) = hit.relocate_triangle.take() else {
            return;
        };
        let x = [
            position[0] + hit.distance * direction[0],
            position[1] + hit.distance * direction[1],
            position[2] + hit.distance * direction[2],
        ];
        let v = self
            .mesh
            .relocate(self.volume_ids[cell_index], tri, x, direction);
        hit.next_cell_index = Some(if v == self.mesh.topology.implicit_complement {
            self.cells.len() - 1
        } else {
            v as usize
        });
    }

    /// Compute the bounding box of the mesh geometry.
    pub fn bounding_box(&self) -> BoundingBox {
        let bb = &self.mesh.topology.global_aabb;
        BoundingBox::new([bb[0], bb[1], bb[2]], [bb[3], bb[4], bb[5]])
    }

    /// Compute the bounding box of volumes matching a specific material ID.
    ///
    /// Returns `None` if no volumes match the given material.
    /// Only considers real volumes (excludes implicit complement).
    pub fn bounding_box_for_material(&self, material_id: u32) -> Option<BoundingBox> {
        let num_real = self.mesh.topology.num_volumes as usize;
        let mut result: Option<BoundingBox> = None;
        for (i, cell) in self.cells[..num_real].iter().enumerate() {
            let matches = cell
                .material_idx
                .and_then(|mi| self.materials.get(mi as usize))
                .and_then(|m| m.get_material_id())
                .map(|id| id == material_id)
                .unwrap_or(false);
            if matches {
                let aabb = &self.mesh.topology.volume_aabbs[i];
                let cell_bbox =
                    BoundingBox::new([aabb[0], aabb[1], aabb[2]], [aabb[3], aabb[4], aabb[5]]);
                match &mut result {
                    None => result = Some(cell_bbox),
                    Some(b) => b.expand_to_include(&cell_bbox),
                }
            }
        }
        result
    }
}

/// Create a dummy CSG region for mesh cells.
///
/// The region is never evaluated during mesh transport (containment and
/// boundary queries go through the mesh backend). We create a simple
/// "inside an infinite sphere" region to satisfy the `Cell` constructor.
/// Distance along a ray to where it leaves the box `[lo.., hi..]` for good: its
/// exit from the box, or 0 when the box is not ahead at all.
fn box_exit(origin: [f64; 3], direction: [f64; 3], bb: &[f64; 6]) -> f64 {
    let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        if direction[k] == 0.0 {
            if origin[k] < bb[k] || origin[k] > bb[k + 3] {
                return 0.0;
            }
            continue;
        }
        let (a, b) = (
            (bb[k] - origin[k]) / direction[k],
            (bb[k + 3] - origin[k]) / direction[k],
        );
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    if far < near.max(0.0) {
        0.0
    } else {
        far
    }
}

/// Distance along the ray to the first crossing of the box surface ahead:
/// where it leaves the box from inside, where it enters from outside, or
/// `None` when it never meets the box. The analytic twin of a ray fired at
/// a `graveyard_offset` box's triangles.
fn box_crossing(origin: [f64; 3], direction: [f64; 3], bb: &[f64; 6]) -> Option<f64> {
    let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        if direction[k] == 0.0 {
            if origin[k] < bb[k] || origin[k] > bb[k + 3] {
                return None;
            }
            continue;
        }
        let (a, b) = (
            (bb[k] - origin[k]) / direction[k],
            (bb[k + 3] - origin[k]) / direction[k],
        );
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    if far < near.max(0.0) {
        None
    } else if near > 0.0 {
        Some(near)
    } else {
        Some(far)
    }
}

fn make_dummy_region() -> Region {
    let big_sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1e30,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    Region {
        expr: RegionExpr::Halfspace(HalfspaceType::Below(big_sphere)),
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE_ARROW: &str = "../yamt/tests/data/cube.arrow";

    fn cube_mesh() -> yamt::MeshGeometry {
        yamt::MeshGeometry::from_arrow(std::path::Path::new(CUBE_ARROW)).unwrap()
    }

    #[test]
    fn test_dagmc_geometry_cube() {
        // Load the cube mesh from yamt test data
        let mesh = cube_mesh();

        // Create a material map -- the cube has mat:water
        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);

        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        // Should have cells (at least 1 for the water volume)
        assert!(!geom.cells.is_empty());

        // Point at center should be inside the water volume
        let idx = geom.find_cell_index((0.5, 0.5, 0.5));
        assert!(idx.is_some(), "Point at center should be found");
        let cell = &geom.cells[idx.unwrap()];
        assert!(cell.material_idx.is_some());

        // Point outside the mesh volumes should return the complement cell
        let outside = geom.find_cell_index((10.0, 10.0, 10.0));
        assert!(
            outside.is_some(),
            "Point outside mesh should be complement cell"
        );
        let ic_cell = &geom.cells[outside.unwrap()];
        assert!(
            ic_cell.material_idx.is_none(),
            "Complement cell should be void"
        );
    }

    /// Issue #503: a model, a mesh fill and a tally each take a clone of the
    /// geometry, so a clone must share the mesh rather than copy it, while
    /// the parts a model changes (cells, materials, the vacuum box) stay its
    /// own.
    #[test]
    fn clone_shares_the_mesh() {
        let mut geom = MeshGeometry::new(cube_mesh(), &HashMap::new(), None).unwrap();
        let copy = geom.clone();
        assert!(Arc::ptr_eq(&geom.mesh, &copy.mesh));
        geom.add_graveyard(DEFAULT_GRAVEYARD_OFFSET);
        assert_ne!(geom.transport_extent(), copy.transport_extent());
        assert_eq!(copy.transport_extent(), copy.mesh.topology.global_aabb);
    }

    #[test]
    fn test_dagmc_geometry_closest_boundary() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);

        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        let idx = geom.find_cell_index((0.5, 0.5, 0.5)).unwrap();

        // Fire ray in +z direction -- should hit the z=1 face
        let hit = geom.closest_boundary(idx, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!(
            h.distance > 0.0 && h.distance < 1.0,
            "Distance should be ~0.5, got {}",
            h.distance
        );
        // Outer face should be vacuum (cube.arrow has boundary:vacuum)
        assert_eq!(h.boundary, BoundaryType::Vacuum);
    }

    #[test]
    fn test_dagmc_geometry_missing_material_treated_as_void() {
        let mesh = cube_mesh();

        // Empty material map -- the unassigned "water" volume is void
        let materials = HashMap::new();
        let result = MeshGeometry::new(mesh, &materials, None);
        assert!(result.is_ok());
        let geom = result.unwrap();
        // Volume cell should have no material (void)
        assert!(geom.cells[0].material_idx.is_none());
    }

    fn water_map(keys: &[&str]) -> HashMap<String, Arc<Material>> {
        let water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        let water = Arc::new(water);
        keys.iter()
            .map(|k| (k.to_string(), Arc::clone(&water)))
            .collect()
    }

    #[test]
    fn test_unmatched_material_keys_are_rejected() {
        let err = MeshGeometry::new(
            cube_mesh(),
            &water_map(&["mat:water", "wtaer", "zzz"]),
            None,
        )
        .err()
        .unwrap();
        assert!(err.contains("'mat:water', 'wtaer', 'zzz'"), "{err}");
        assert!(err.contains("Volume names are: 'water'"), "{err}");
        assert!(
            err.contains("Did you mean 'water' for 'mat:water', 'water' for 'wtaer'?"),
            "{err}"
        );
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("frist_wall", "first_wall"), 2);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_mesh_cell_volume_populated() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let cell = &geom.cells[0];

        // Cell volume should be populated (not None)
        assert!(cell.volume.is_some(), "Cell volume should be set from mesh");
        let vol = cell.volume.unwrap();
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube cell volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_mesh_material_volume_propagated() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let cell = &geom.cells[0];
        let mat = geom
            .materials
            .get(cell.material_idx.expect("cell has material") as usize)
            .expect("material slot populated");

        // Material volume should also be set (for transmutation)
        assert!(
            mat.volume.is_some(),
            "Material volume should be propagated from mesh"
        );
        let vol = mat.volume.unwrap();
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube material volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_bounding_box_for_material() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        // Should find bounding box for material_id=1 (water)
        let bb = geom.bounding_box_for_material(1);
        assert!(bb.is_some(), "Should find bbox for water material");
        let bb = bb.unwrap();
        // Cube is [0,1]^3
        for i in 0..3 {
            assert!(bb.lower_left[i] <= 0.01);
            assert!(bb.upper_right[i] >= 0.99);
        }

        // Should return None for non-existent material
        let bb_none = geom.bounding_box_for_material(999);
        assert!(bb_none.is_none(), "Should return None for unknown material");
    }

    #[test]
    fn test_dagmc_geometry_bounding_box() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let bb = geom.bounding_box();

        // Cube is [0,1]^3
        for i in 0..3 {
            assert!(bb.lower_left[i] <= 0.01);
            assert!(bb.upper_right[i] >= 0.99);
        }
    }

    #[test]
    fn test_mesh_geometry_serializes_for_fingerprint() {
        // Regression: a Mesh-variant model must fingerprint (combine_results /
        // simulate_transport), which serializes the geometry to JSON. This
        // previously errored ("MeshGeometry is not yet JSON-serializable"),
        // breaking ALL mesh-geometry transport.
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.set_material_id(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        let v1 = serde_json::to_value(&geom).expect("mesh geometry should serialize");
        let v2 = serde_json::to_value(&geom).expect("serialize again");
        assert_eq!(v1, v2, "fingerprint summary must be deterministic");
        assert_eq!(v1["kind"], "mesh");
        assert_eq!(v1["num_volumes"], 1);
        assert!(v1["volume_materials"].is_array());
    }

    #[test]
    fn materials_without_ids_get_them_in_name_order() {
        // The map's iteration order differs from process to process; the ids
        // must not. The allocator only counts up, so name order shows as
        // increasing ids whatever else this process allocates meanwhile.
        let bare = || Arc::new(Material::new(HashMap::new(), "atom", "sum", None).unwrap());
        let shared = bare();
        let mut explicit = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        explicit.set_material_id(7);
        let explicit = Arc::new(explicit);
        let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let mut materials: HashMap<String, Arc<Material>> =
            names.iter().map(|n| (n.to_string(), bare())).collect();
        materials.insert("shared_1".into(), Arc::clone(&shared));
        materials.insert("shared_2".into(), Arc::clone(&shared));
        materials.insert("explicit".into(), explicit);

        let mut with_ids = materials.clone();
        let mut complement = Some(shared);
        assign_ids(&mut with_ids, &mut complement).unwrap();
        let id = |name: &str| with_ids[name].get_material_id().unwrap();

        let in_name_order: Vec<u32> = names.iter().map(|n| id(n)).collect();
        assert!(
            in_name_order.windows(2).all(|w| w[0] < w[1]),
            "{in_name_order:?}"
        );
        assert!(id("h") < id("shared_1"));
        // One material under two names, and as the complement: one id.
        assert_eq!(id("shared_1"), id("shared_2"));
        assert_eq!(complement.unwrap().get_material_id(), Some(id("shared_1")));
        assert_eq!(id("explicit"), 7);
        // The caller's materials are left as they were.
        assert!(materials["a"].get_material_id().is_none());
    }
}
