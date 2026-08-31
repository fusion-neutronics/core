//! Regular Cartesian mesh for spatial tallying
//!
//! A RegularRectangularMesh divides 3D space into a regular grid of voxels (3D cells).
//! This enables track-length or collision-based tallying on a spatial mesh.
//!
//! Includes performance optimizations for track-length scoring.

/// Voxel `(ix, iy, iz)` maps to the flat index `(iz * ny + iy) * nx + ix`,
/// so bins along X are contiguous. This is the only layout.
///
/// A Morton (Z-order) alternative existed until issue #337. The idea was that
/// interleaving the index bits would put spatially-near voxels near each other
/// in memory and so be kinder to the prefetcher. Measured on both backends it
/// was never faster: at a 64^3 power-of-two shape, where Morton's power-of-two
/// padding costs nothing and the curve should have been pure upside, it was
/// 3.2% SLOWER on CPU and within noise on GPU. At the non-power-of-two shapes
/// that real shielding and first-wall maps use, the padding (2.1x at 100^3,
/// 33.6x at 50x50x200) made it 1.5x to 2.5x slower on CPU, and on GPU the
/// 50x50x200 case could not allocate at all. Track-length scoring walks voxels
/// one axis-step at a time, so the row-major stride is already contiguous
/// along the direction of travel far more often than the curve assumed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegularRectangularMesh {
    /// Lower-left corner of the mesh [x, y, z]
    lower_left: [f64; 3],
    /// Upper-right corner of the mesh [x, y, z]
    upper_right: [f64; 3],
    /// Number of bins along each axis [nx, ny, nz]
    shape: [usize; 3],
    /// Width of each voxel in each dimension [dx, dy, dz]
    width: [f64; 3],
    /// Inverse width (1.0 / width) for performance - multiply instead of divide
    inv_width: [f64; 3],
    /// Pre-calculated volume of a single voxel (constant for RegularRectangularMesh)
    voxel_volume: f64,
}

impl RegularRectangularMesh {
    /// Create a new RegularRectangularMesh
    ///
    /// # Arguments
    /// * `lower_left` - Lower-left corner coordinates [x_min, y_min, z_min]
    /// * `upper_right` - Upper-right corner coordinates [x_max, y_max, z_max]
    /// * `shape` - Number of bins along each axis [nx, ny, nz]
    ///
    /// # Panics
    /// Panics if any `shape` entry is 0 or if lower_left >= upper_right in any axis.
    /// For a non-panicking variant that validates user input, see
    /// [`RegularRectangularMesh::try_new`].
    pub fn new(lower_left: [f64; 3], upper_right: [f64; 3], shape: [usize; 3]) -> Self {
        Self::try_new(lower_left, upper_right, shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible constructor for [`RegularRectangularMesh`].
    ///
    /// Validates the same conditions as [`RegularRectangularMesh::new`] but
    /// returns an [`Err`] with a descriptive message instead of panicking, so
    /// callers handling untrusted input (e.g. the Python bindings) can surface
    /// a proper error.
    ///
    /// # Errors
    /// Returns `Err` if any `shape` entry is 0 or if `lower_left >= upper_right`
    /// in any axis.
    pub fn try_new(
        lower_left: [f64; 3],
        upper_right: [f64; 3],
        shape: [usize; 3],
    ) -> Result<Self, String> {
        // Validate inputs
        for i in 0..3 {
            if shape[i] == 0 {
                return Err("Mesh shape must be positive in all directions".to_string());
            }
            if upper_right[i] <= lower_left[i] {
                return Err(
                    "Mesh upper_right must be greater than lower_left in all directions"
                        .to_string(),
                );
            }
        }

        // Calculate voxel widths
        let mut width = [0.0; 3];
        let mut inv_width = [0.0; 3];
        for i in 0..3 {
            width[i] = (upper_right[i] - lower_left[i]) / shape[i] as f64;
            inv_width[i] = 1.0 / width[i];
        }

        // Calculate voxel volume
        let voxel_volume = width[0] * width[1] * width[2];

        Ok(RegularRectangularMesh {
            lower_left,
            upper_right,
            shape,
            width,
            inv_width,
            voxel_volume,
        })
    }

    /// Get the flat bin index for a position in 3D space
    ///
    /// Uses Z-major ordering: bin = (iz * ny + iy) * nx + ix
    ///
    /// # Arguments
    /// * `position` - 3D position [x, y, z]
    ///
    /// # Returns
    /// * `Some(bin_index)` if position is within mesh bounds
    /// * `None` if position is outside mesh bounds
    pub fn get_bin(&self, position: [f64; 3]) -> Option<usize> {
        let mut indices = [0_usize; 3];

        for i in 0..3 {
            // Check if position is within mesh bounds
            if position[i] < self.lower_left[i] || position[i] >= self.upper_right[i] {
                return None;
            }

            // Calculate index in this dimension
            // Formula: ceil((x - x_min) / dx) - 1
            // Using inv_width for performance (multiply instead of divide)
            let idx_float = (position[i] - self.lower_left[i]) * self.inv_width[i];
            indices[i] = idx_float.floor() as usize;

            // Clamp to valid range (handles floating point edge cases)
            if indices[i] >= self.shape[i] {
                indices[i] = self.shape[i] - 1;
            }
        }

        // Flatten to single index using Z-major ordering
        Some(self.get_bin_from_indices(indices))
    }

    /// Convert 3D indices to the flat bin index `(iz * ny + iy) * nx + ix`.
    ///
    /// Exposed at crate visibility so slice extractors (e.g.
    /// `Tally::extract_mesh_slice`) share one definition of the mapping
    /// rather than open-coding it.
    #[inline]
    pub(crate) fn get_bin_from_indices(&self, indices: [usize; 3]) -> usize {
        let [ix, iy, iz] = indices;
        let [nx, ny, _nz] = self.shape;
        (iz * ny + iy) * nx + ix
    }

    /// Number of voxels, `nx * ny * nz`. Every index returned by `get_bin` /
    /// `bins_crossed` is `< num_voxels()`, so this is also the length the
    /// tally's flat scoring buffers need.
    #[inline]
    pub fn num_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Get the volume of a voxel
    ///
    /// For RegularRectangularMesh, all voxels have the same volume (constant)
    #[inline]
    pub fn get_voxel_volume(&self, _bin: usize) -> f64 {
        self.voxel_volume
    }

    /// Get the lower-left corner
    #[inline]
    pub fn lower_left(&self) -> [f64; 3] {
        self.lower_left
    }

    /// Get the upper-right corner
    #[inline]
    pub fn upper_right(&self) -> [f64; 3] {
        self.upper_right
    }

    /// Get the bin counts along each axis `[nx, ny, nz]`.
    #[inline]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Get the voxel widths [dx, dy, dz]
    #[inline]
    pub fn width(&self) -> [f64; 3] {
        self.width
    }

    /// Calculate which mesh bins are crossed by a particle track and the length
    /// in each. Returns a [`BinsCrossedIter`] that lazily walks the track
    /// without heap-allocating a `Vec<MeshCrossing>`. Hot-path scoring code
    /// should use this; the older `bins_crossed` method (returning a `Vec`)
    /// remains for tests and code that needs a fully-materialised list.
    #[allow(clippy::needless_range_loop)]
    pub fn bins_crossed_iter(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        direction: [f64; 3],
    ) -> BinsCrossedIter<'_> {
        const TINY_BIT: f64 = 1e-8;

        let total_length =
            ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();

        // Precompute reciprocal direction components used in
        // distance_to_boundary (one f64 div per axis saved per crossing).
        // Zero/near-zero direction components yield 0.0 here; the
        // distance_to_boundary axis-parallel shortcut runs before the
        // multiplication so the bogus value is never consumed.
        const FP_PRECISION: f64 = 1e-14;
        let inv_direction = [
            if direction[0].abs() < FP_PRECISION {
                0.0
            } else {
                1.0 / direction[0]
            },
            if direction[1].abs() < FP_PRECISION {
                0.0
            } else {
                1.0 / direction[1]
            },
            if direction[2].abs() < FP_PRECISION {
                0.0
            } else {
                1.0 / direction[2]
            },
        ];
        // Amanatides-Woo per-axis t_delta and step. t_delta is the
        // distance along the track to cross one voxel in that axis;
        // step is the per-axis index increment (+1, -1, or 0 for
        // axis-parallel).  Both are constant for a given track, so
        // computing them once here lets the per-crossing update in
        // `Iterator::next` skip the full `distance_to_boundary` call.
        let mut step = [0i8; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for i in 0..3 {
            if direction[i].abs() < FP_PRECISION {
                continue;
            }
            step[i] = if direction[i] > 0.0 { 1 } else { -1 };
            t_delta[i] = self.width[i] * inv_direction[i].abs();
        }

        let finished_iter = BinsCrossedIter {
            mesh: self,
            t_delta,
            step,
            total_length,
            indices: [0; 3],
            distances: [MeshDistance {
                next_index: 0,
                distance: f64::INFINITY,
            }; 3],
            traveled: 0.0,
            finished: true,
        };

        // Early exit for very short tracks.
        if total_length < 2.0 * TINY_BIT {
            return finished_iter;
        }

        // Check if start position is in mesh.
        let mut indices = [0_usize; 3];
        let mut in_mesh = true;
        for i in 0..3 {
            if r0[i] < self.lower_left[i] || r0[i] >= self.upper_right[i] {
                in_mesh = false;
                break;
            }
            let idx_float = (r0[i] - self.lower_left[i]) * self.inv_width[i];
            indices[i] = idx_float.floor() as usize;
            if indices[i] >= self.shape[i] {
                indices[i] = self.shape[i] - 1;
            }
        }

        let mut traveled = 0.0;
        let mut current_pos = r0;

        if !in_mesh {
            // Find where the ray enters the mesh bounding box (slab test).
            let mut t_enter = f64::NEG_INFINITY;
            let mut t_exit = f64::INFINITY;
            for i in 0..3 {
                if direction[i].abs() < 1e-14 {
                    if r0[i] < self.lower_left[i] || r0[i] >= self.upper_right[i] {
                        return finished_iter; // misses mesh entirely
                    }
                } else {
                    let inv_d = 1.0 / direction[i];
                    let t1 = (self.lower_left[i] - r0[i]) * inv_d;
                    let t2 = (self.upper_right[i] - r0[i]) * inv_d;
                    let (t_near, t_far) = if inv_d > 0.0 { (t1, t2) } else { (t2, t1) };
                    t_enter = t_enter.max(t_near);
                    t_exit = t_exit.min(t_far);
                }
            }
            if t_enter >= t_exit || t_exit <= 0.0 || t_enter >= total_length {
                return finished_iter;
            }
            let t_start = t_enter.max(0.0) + TINY_BIT;
            traveled = t_start;
            for i in 0..3 {
                current_pos[i] = r0[i] + direction[i] * t_start;
            }
            in_mesh = true;
            for i in 0..3 {
                if current_pos[i] < self.lower_left[i] || current_pos[i] >= self.upper_right[i] {
                    in_mesh = false;
                    break;
                }
                let idx_float = (current_pos[i] - self.lower_left[i]) * self.inv_width[i];
                indices[i] = idx_float.floor() as usize;
                if indices[i] >= self.shape[i] {
                    indices[i] = self.shape[i] - 1;
                }
            }
            if !in_mesh {
                return finished_iter;
            }
        }

        let distances = [
            self.distance_to_boundary(
                indices,
                0,
                current_pos,
                direction[0],
                inv_direction[0],
                traveled,
            ),
            self.distance_to_boundary(
                indices,
                1,
                current_pos,
                direction[1],
                inv_direction[1],
                traveled,
            ),
            self.distance_to_boundary(
                indices,
                2,
                current_pos,
                direction[2],
                inv_direction[2],
                traveled,
            ),
        ];

        BinsCrossedIter {
            mesh: self,
            t_delta,
            step,
            total_length,
            indices,
            distances,
            traveled,
            finished: false,
        }
    }

    /// Materialised version of [`bins_crossed_iter`]. Retained for backwards
    /// compatibility and for tests that want a `Vec`. Hot-path code should
    /// use the iterator form directly to avoid the heap allocation.
    pub fn bins_crossed(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        direction: [f64; 3],
    ) -> Vec<MeshCrossing> {
        self.bins_crossed_iter(r0, r1, direction).collect()
    }

    /// Calculate distance to the next grid boundary in a given dimension.
    /// Takes the precomputed `inv_direction_dim = 1.0 / direction[dim]`
    /// so the hot DDA path avoids one f64 division per crossing -- the
    /// reciprocal is the same for every crossing along a track, so the
    /// caller computes it once at iter creation.
    #[inline(always)]
    fn distance_to_boundary(
        &self,
        indices: [usize; 3],
        dim: usize,
        position: [f64; 3],
        direction_dim: f64,
        inv_direction_dim: f64,
        traveled: f64,
    ) -> MeshDistance {
        const FP_PRECISION: f64 = 1e-14;

        let mut result = MeshDistance {
            next_index: indices[dim],
            distance: f64::INFINITY,
        };

        // If direction is parallel to grid in this dimension, no crossing
        if direction_dim.abs() < FP_PRECISION {
            return result;
        }

        // Determine which surface we're moving toward
        let moving_positive = direction_dim > 0.0;

        if moving_positive && indices[dim] < self.shape[dim] {
            // Moving toward max surface
            result.next_index = indices[dim] + 1;
            let boundary = self.lower_left[dim] + (indices[dim] + 1) as f64 * self.width[dim];
            result.distance = traveled + (boundary - position[dim]) * inv_direction_dim;
        } else if !moving_positive {
            // Moving toward min surface
            let boundary = self.lower_left[dim] + indices[dim] as f64 * self.width[dim];
            result.distance = traveled + (boundary - position[dim]) * inv_direction_dim;
            if indices[dim] > 0 {
                result.next_index = indices[dim] - 1;
            } else {
                // Exiting mesh through lower boundary -- use sentinel that
                // triggers the `>= dimension` exit check in bins_crossed.
                result.next_index = self.shape[dim];
            }
        }

        result
    }
}

/// Represents a mesh crossing for track-length tallying
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeshCrossing {
    /// The bin index that was crossed
    pub bin: usize,
    /// Fraction of total track length spent in this bin
    pub length_fraction: f64,
}

/// Helper struct for tracking distance to mesh boundaries
#[derive(Debug, Clone, Copy)]
struct MeshDistance {
    /// Index of next voxel in this dimension
    next_index: usize,
    /// Distance along track to reach the boundary
    distance: f64,
}

/// Lazy iterator over the mesh bins crossed by a single particle track.
///
/// Yields one [`MeshCrossing`] per voxel intersected by the segment
/// `r0 → r0 + direction * total_length`. The DDA state machine is identical
/// to the original `bins_crossed` implementation, but no `Vec` is allocated:
/// the per-event hot path in `score_track_length_slow_with` consumes this
/// directly and pays nothing to the global allocator.
pub struct BinsCrossedIter<'a> {
    mesh: &'a RegularRectangularMesh,
    /// Amanatides-Woo "t_delta" per axis: the distance along the track
    /// covered by moving exactly one voxel in axis i. Computed once at
    /// iter creation as `width[i] / |direction[i]|` (or `INFINITY` for
    /// axis-parallel directions). The per-crossing update for the
    /// axis that was just crossed is then `distances[min_dim] +=
    /// t_delta[min_dim]`, eliminating the per-crossing
    /// `distance_to_boundary` call (which would otherwise require a
    /// division, a `current_pos` recomputation, and three `f64` muls).
    t_delta: [f64; 3],
    /// Sign of `direction[i]`: +1, -1, or 0 (axis-parallel; t_delta is
    /// INFINITY in that case so the axis is never picked as `min_dim`).
    /// Used to advance the per-axis index on each crossing.
    step: [i8; 3],
    total_length: f64,
    indices: [usize; 3],
    distances: [MeshDistance; 3],
    traveled: f64,
    finished: bool,
}

impl<'a> Iterator for BinsCrossedIter<'a> {
    type Item = MeshCrossing;

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn next(&mut self) -> Option<MeshCrossing> {
        const TINY_BIT: f64 = 1e-8;

        loop {
            if self.finished {
                return None;
            }

            // Find which face will be crossed first.
            let mut min_dim = 0;
            let mut min_dist = self.distances[0].distance;
            for i in 1..3 {
                if self.distances[i].distance < min_dist {
                    min_dist = self.distances[i].distance;
                    min_dim = i;
                }
            }

            let length_in_voxel = if min_dist >= self.total_length {
                self.total_length - self.traveled
            } else {
                min_dist - self.traveled
            };

            // Stage a crossing if the slice through this voxel is non-trivial.
            let crossing = if length_in_voxel > TINY_BIT {
                Some(MeshCrossing {
                    bin: self.mesh.get_bin_from_indices(self.indices),
                    length_fraction: length_in_voxel / self.total_length,
                })
            } else {
                None
            };

            // End of track reached?
            if min_dist >= self.total_length {
                self.finished = true;
                return crossing;
            }

            // Advance to the next voxel.
            self.traveled = min_dist;
            self.indices[min_dim] = self.distances[min_dim].next_index;

            if self.indices[min_dim] >= self.mesh.shape[min_dim] {
                self.finished = true;
                return crossing;
            }

            // Amanatides-Woo update: the next boundary in `min_dim` is
            // exactly one voxel further along, which is `t_delta[min_dim]`
            // additional distance along the track. No division, no
            // re-derivation of the current position -- both are subsumed
            // by the precomputed `t_delta` / `step` arrays.
            self.distances[min_dim].distance += self.t_delta[min_dim];
            let step_dim = self.step[min_dim];
            self.distances[min_dim].next_index = if step_dim > 0 {
                self.indices[min_dim] + 1
            } else if self.indices[min_dim] > 0 {
                // step_dim < 0 (parallel axes never appear here because
                // their t_delta is INFINITY, so they're never `min_dim`).
                self.indices[min_dim] - 1
            } else {
                // step_dim < 0 and we're at index 0 -- next step exits the
                // mesh through the lower boundary. Use the same `shape[dim]`
                // sentinel that the original `distance_to_boundary` used
                // so the outer `>= shape` check on the next iter trips.
                self.mesh.shape[min_dim]
            };

            if let Some(c) = crossing {
                return Some(c);
            }
            // Slice was below TINY_BIT -- keep walking without yielding.
        }
    }
}

// =====================================================================
// Cylindrical mesh
// =====================================================================

/// Two pi, used as the upper bound / wrap point for the azimuthal grid.
const TWO_PI: f64 = std::f64::consts::TAU;
/// Distances below this along a track are treated as zero (matches the
/// Cartesian DDA's `TINY_BIT`). Used both to nudge index lookups across a
/// freshly crossed boundary and to discard degenerate (grazing) slices.
const CYL_TINY: f64 = 1e-8;
/// Direction/denominator components below this are treated as zero.
const CYL_FP: f64 = 1e-14;
/// A candidate boundary crossing only counts if it lies more than this far
/// ahead of the current cursor. Prevents re-selecting the surface the track
/// is sitting on (which would stall the walk) while staying far below any
/// macroscopic mesh spacing.
const CYL_COINCIDENT: f64 = 1e-10;

/// A cylindrical `(r, φ, z)` mesh for spatial tallying.
///
/// Coordinates are taken relative to `origin`:
/// `ρ = hypot(x − x₀, y − y₀)`, `φ = atan2(y − y₀, x − x₀) ∈ [0, 2π)`,
/// `z = z − z₀`. The grids are stored explicitly and may be non-uniform, so
/// the radial/azimuthal/axial spacing is arbitrary (sorted, strictly
/// increasing).
///
/// Unlike [`RegularRectangularMesh`], element volumes are **not** constant:
/// `V = ½·(r_o² − r_i²)·(φ_o − φ_i)·(z_o − z_i)`, so outer radial rings are
/// larger than inner ones. Use [`CylindricalMesh::get_voxel_volume`] per bin.
///
/// `φ` may span any sub-interval of `[0, 2π]` (e.g. a `[0, π]` half or a
/// `[π, 2π]` sector). The azimuthal seam only wraps (last bin → first bin)
/// when the grid covers the full `[0, 2π]`; otherwise the φ ends are hard
/// mesh boundaries.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CylindricalMesh {
    /// Cylinder axis location `[x₀, y₀, z₀]`; the axis is parallel to +z.
    origin: [f64; 3],
    /// Radial grid edges, length `nr + 1`, `r_grid[0] >= 0`, increasing.
    r_grid: Vec<f64>,
    /// Azimuthal grid edges in radians, length `nphi + 1`, within `[0, 2π]`.
    phi_grid: Vec<f64>,
    /// Axial grid edges (offsets from `origin.z`), length `nz + 1`.
    z_grid: Vec<f64>,
    /// `[nr, nphi, nz]` cell counts.
    shape: [usize; 3],
    /// True when `phi_grid` spans exactly `[0, 2π]` (enables seam wrap).
    full_phi: bool,
    /// Precomputed `r_grid[i]^2` (radial quadratic + volume).
    r_grid_sq: Vec<f64>,
}

impl CylindricalMesh {
    /// Create a cylindrical mesh from explicit grids.
    ///
    /// # Panics
    /// Panics if any grid has fewer than two points or is not strictly
    /// increasing, if `r_grid[0] < 0`, or if `phi_grid` leaves `[0, 2π]`.
    /// For a non-panicking variant that validates user input, see
    /// [`CylindricalMesh::try_new`].
    pub fn new(origin: [f64; 3], r_grid: Vec<f64>, phi_grid: Vec<f64>, z_grid: Vec<f64>) -> Self {
        Self::try_new(origin, r_grid, phi_grid, z_grid).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible constructor for [`CylindricalMesh`] from explicit grids.
    ///
    /// Validates the same conditions as [`CylindricalMesh::new`] but returns an
    /// [`Err`] with a descriptive message instead of panicking, so callers
    /// handling untrusted input (e.g. the Python bindings) can surface a proper
    /// error.
    ///
    /// # Errors
    /// Returns `Err` if any grid has fewer than two points or is not strictly
    /// increasing, if `r_grid[0] < 0`, or if `phi_grid` leaves `[0, 2π]`.
    pub fn try_new(
        origin: [f64; 3],
        r_grid: Vec<f64>,
        phi_grid: Vec<f64>,
        z_grid: Vec<f64>,
    ) -> Result<Self, String> {
        for (name, g) in [("r", &r_grid), ("phi", &phi_grid), ("z", &z_grid)] {
            if g.len() < 2 {
                return Err(format!(
                    "Cylindrical mesh {name}-grid must have at least 2 points"
                ));
            }
            if !g.windows(2).all(|w| w[1] > w[0]) {
                return Err(format!(
                    "Cylindrical mesh {name}-grid must be strictly increasing"
                ));
            }
        }
        if r_grid[0] < 0.0 {
            return Err("Cylindrical mesh r-grid must start at r >= 0".to_string());
        }
        if !(phi_grid[0] >= -CYL_FP && *phi_grid.last().unwrap() <= TWO_PI + CYL_FP) {
            return Err("Cylindrical mesh phi-grid must lie within [0, 2*pi]".to_string());
        }

        let full_phi =
            phi_grid[0].abs() < 1e-9 && (*phi_grid.last().unwrap() - TWO_PI).abs() < 1e-9;
        let r_grid_sq = r_grid.iter().map(|r| r * r).collect();
        let shape = [r_grid.len() - 1, phi_grid.len() - 1, z_grid.len() - 1];

        Ok(CylindricalMesh {
            origin,
            r_grid,
            phi_grid,
            z_grid,
            shape,
            full_phi,
            r_grid_sq,
        })
    }

    /// Build a cylindrical mesh with uniform spacing from bounds + counts.
    ///
    /// `r_bounds = (r_min, r_max)`, `phi_bounds = (φ_min, φ_max)` in radians,
    /// `z_bounds = (z_min, z_max)`, `shape = [nr, nphi, nz]`. This is the
    /// constructor the Python `RegularCylindricalMesh` wraps.
    pub fn uniform(
        origin: [f64; 3],
        r_bounds: (f64, f64),
        phi_bounds: (f64, f64),
        z_bounds: (f64, f64),
        shape: [usize; 3],
    ) -> Self {
        Self::try_uniform(origin, r_bounds, phi_bounds, z_bounds, shape)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible variant of [`CylindricalMesh::uniform`].
    ///
    /// Builds the uniform grids then validates them via
    /// [`CylindricalMesh::try_new`], returning an [`Err`] with a descriptive
    /// message instead of panicking. This is the constructor the Python
    /// `RegularCylindricalMesh` wraps.
    ///
    /// # Errors
    /// Returns `Err` if the resulting grids violate the conditions documented on
    /// [`CylindricalMesh::try_new`].
    pub fn try_uniform(
        origin: [f64; 3],
        r_bounds: (f64, f64),
        phi_bounds: (f64, f64),
        z_bounds: (f64, f64),
        shape: [usize; 3],
    ) -> Result<Self, String> {
        let linspace = |lo: f64, hi: f64, n: usize| -> Vec<f64> {
            (0..=n)
                .map(|i| lo + (hi - lo) * i as f64 / n as f64)
                .collect()
        };
        CylindricalMesh::try_new(
            origin,
            linspace(r_bounds.0, r_bounds.1, shape[0]),
            linspace(phi_bounds.0, phi_bounds.1, shape[1]),
            linspace(z_bounds.0, z_bounds.1, shape[2]),
        )
    }

    /// `[nr, nphi, nz]` cell counts.
    #[inline]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Cylinder origin `[x₀, y₀, z₀]`.
    #[inline]
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Radial grid edges (length `nr + 1`).
    #[inline]
    pub fn r_grid(&self) -> &[f64] {
        &self.r_grid
    }

    /// Azimuthal grid edges in radians (length `nphi + 1`).
    #[inline]
    pub fn phi_grid(&self) -> &[f64] {
        &self.phi_grid
    }

    /// Axial grid edges (length `nz + 1`).
    #[inline]
    pub fn z_grid(&self) -> &[f64] {
        &self.z_grid
    }

    /// True when `phi_grid` spans exactly `[0, 2π]`, so the azimuthal seam
    /// wraps (last φ bin adjacent to the first). Otherwise the φ ends are hard
    /// mesh boundaries. Exposed so the GPU translate layer can reproduce the
    /// seam-wrap behaviour in the kernel descriptor (issue #234).
    #[inline]
    pub fn full_phi(&self) -> bool {
        self.full_phi
    }

    /// Total number of cells (`nr * nphi * nz`).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Flat bin index from `(ir, iphi, iz)`, row-major with `ir` innermost
    /// (contiguous): `(iz * nphi + iphi) * nr + ir`.
    #[inline]
    fn bin_from_indices(&self, ir: usize, iphi: usize, iz: usize) -> usize {
        (iz * self.shape[1] + iphi) * self.shape[0] + ir
    }

    /// Recover `(ir, iphi, iz)` from a flat bin index.
    #[inline]
    fn indices_from_bin(&self, bin: usize) -> [usize; 3] {
        let nr = self.shape[0];
        let nphi = self.shape[1];
        let ir = bin % nr;
        let iphi = (bin / nr) % nphi;
        let iz = bin / (nr * nphi);
        [ir, iphi, iz]
    }

    /// Volume of a single cell. Non-constant: outer radial rings are larger.
    #[inline]
    pub fn get_voxel_volume(&self, bin: usize) -> f64 {
        let [ir, iphi, iz] = self.indices_from_bin(bin);
        0.5 * (self.r_grid_sq[ir + 1] - self.r_grid_sq[ir])
            * (self.phi_grid[iphi + 1] - self.phi_grid[iphi])
            * (self.z_grid[iz + 1] - self.z_grid[iz])
    }

    /// Locate the `(ir, iphi, iz)` cell of an absolute position, or `None`
    /// when the position lies outside the mesh. Used by the collision
    /// estimator.
    pub fn get_bin(&self, position: [f64; 3]) -> Option<usize> {
        let x = position[0] - self.origin[0];
        let y = position[1] - self.origin[1];
        let zc = position[2] - self.origin[2];
        self.indices_local(x, y, zc)
            .map(|[ir, iphi, iz]| self.bin_from_indices(ir, iphi, iz))
    }

    /// Per-axis bracketed cell of a local-frame point, or `None` if outside.
    fn indices_local(&self, x: f64, y: f64, zc: f64) -> Option<[usize; 3]> {
        let rho = x.hypot(y);
        let ir = bracket(&self.r_grid, rho)?;
        let iz = bracket(&self.z_grid, zc)?;
        let iphi = if rho < CYL_FP {
            // On the axis φ is undefined; the point sits at the shared corner
            // of every φ cell. Bin it to the first φ cell (only reachable when
            // the innermost ring touches r = 0).
            0
        } else {
            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += TWO_PI;
            }
            // atan2 can return a value a hair above the last edge for full-2π
            // grids; clamp it back so it brackets into the last cell.
            if self.full_phi && phi >= self.phi_grid[self.shape[1]] {
                phi = self.phi_grid[self.shape[1]] - CYL_FP;
            }
            bracket(&self.phi_grid, phi)?
        };
        Some([ir, iphi, iz])
    }

    /// Materialise the mesh cells crossed by the track `r0 → r1` and the
    /// fraction of the track length spent in each, for track-length scoring.
    ///
    /// `direction` is the unit direction of travel (matching the convention of
    /// [`RegularRectangularMesh::bins_crossed`]). A track that passes through
    /// the central `r < r_min` hole produces two groups of crossings with an
    /// untallied gap between them.
    pub fn bins_crossed(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        direction: [f64; 3],
    ) -> Vec<MeshCrossing> {
        let total =
            ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();
        let mut out = Vec::new();

        if total < 2.0 * CYL_TINY {
            // Degenerate track: attribute the whole thing to the midpoint cell.
            let mid = [
                0.5 * (r0[0] + r1[0]),
                0.5 * (r0[1] + r1[1]),
                0.5 * (r0[2] + r1[2]),
            ];
            if let Some(bin) = self.get_bin(mid) {
                out.push(MeshCrossing {
                    bin,
                    length_fraction: 1.0,
                });
            }
            return out;
        }

        // Local frame; `p` is the start relative to origin, `d` the unit dir.
        let p = [
            r0[0] - self.origin[0],
            r0[1] - self.origin[1],
            r0[2] - self.origin[2],
        ];
        let d = direction;
        // Radial quadratic coefficients: ρ²(t) = a·t² + 2b·t + c.
        let a = d[0] * d[0] + d[1] * d[1];
        let b = p[0] * d[0] + p[1] * d[1];
        let c = p[0] * p[0] + p[1] * p[1];

        let point = |l: f64| [p[0] + l * d[0], p[1] + l * d[1], p[2] + l * d[2]];

        let mut l = 0.0_f64;
        let max_iter = 4 * (self.shape[0] + self.shape[1] + self.shape[2]) + 32;
        let mut iter = 0;

        while l < total {
            iter += 1;
            if iter > max_iter {
                break; // defensive: should never trigger for a valid track
            }

            // Locate the current cell from a point nudged just past the cursor
            // so that, immediately after a crossing, we are inside the new cell
            // rather than ambiguously on the boundary.
            let probe = point((l + CYL_TINY).min(total));
            let pr = probe[0].hypot(probe[1]);
            match self.indices_local(probe[0], probe[1], probe[2]) {
                Some([ir, iphi, iz]) => {
                    // Distance to leave this cell = nearest wall crossing > l.
                    let dr = self.radial_next(a, b, c, ir, l);
                    let dphi = self.phi_next(p, d, iphi, l);
                    let dz = self.z_next(p[2], d[2], iz, l);
                    // The radial perigee t* = -b/a is a forced event: the radial
                    // index turns around there, and a track passing through (or
                    // very near) the axis flips φ by π without crossing any φ
                    // wall. Stopping here lets the walk re-locate the new cell.
                    let d_peri = if a >= CYL_FP {
                        let t_star = -b / a;
                        if t_star > l + CYL_COINCIDENT {
                            t_star
                        } else {
                            f64::INFINITY
                        }
                    } else {
                        f64::INFINITY
                    };
                    let dmin = dr.min(dphi).min(dz).min(d_peri).min(total);

                    let seg = dmin - l;
                    if seg > CYL_TINY {
                        out.push(MeshCrossing {
                            bin: self.bin_from_indices(ir, iphi, iz),
                            length_fraction: seg / total,
                        });
                    }
                    if dmin >= total {
                        break;
                    }
                    l = dmin;
                }
                None => {
                    // Outside the mesh (initial entry, or re-entry after the
                    // radial hole / a φ wedge). Jump to the next boundary that
                    // lands us inside; the skipped span is not tallied.
                    let _ = pr;
                    match self.first_entry(p, d, a, b, c, l, total) {
                        Some(te) if te < total => l = te,
                        _ => break,
                    }
                }
            }
        }

        out
    }

    /// Smallest radial-shell crossing strictly ahead of `l`, considering both
    /// walls of cell `ir`. `∞` when the track makes no further radial crossing
    /// (e.g. parallel to z, or it never reaches either shell again).
    fn radial_next(&self, a: f64, b: f64, c: f64, ir: usize, l: f64) -> f64 {
        let inner = self.shell_crossing(a, b, c, self.r_grid_sq[ir], self.r_grid[ir], l);
        let outer = self.shell_crossing(a, b, c, self.r_grid_sq[ir + 1], self.r_grid[ir + 1], l);
        inner.min(outer)
    }

    /// First root `> l` of `a·t² + 2b·t + (c − R²) = 0` (a ray–cylinder
    /// intersection -- the same quadratic the CSG cylinder surface solves).
    /// `∞` when there is no such crossing.
    fn shell_crossing(&self, a: f64, b: f64, c: f64, r_sq: f64, r: f64, l: f64) -> f64 {
        if r <= 0.0 || a < CYL_FP {
            // r = 0 is the axis (no inner wall for the core cell); a ≈ 0 means
            // the track is parallel to z, so ρ is constant -- no radial crossing.
            return f64::INFINITY;
        }
        let pn = b / a;
        let disc = pn * pn - (c - r_sq) / a;
        if disc < 0.0 {
            return f64::INFINITY;
        }
        let sq = disc.sqrt();
        let t1 = -pn - sq; // smaller root
        let t2 = -pn + sq; // larger root
        let thresh = l + CYL_COINCIDENT;
        if t1 > thresh {
            t1
        } else if t2 > thresh {
            t2
        } else {
            f64::INFINITY
        }
    }

    /// Smallest azimuthal-wall crossing strictly ahead of `l`, considering both
    /// walls of cell `iphi`.
    fn phi_next(&self, p: [f64; 3], d: [f64; 3], iphi: usize, l: f64) -> f64 {
        let lo = self.phi_crossing(p, d, self.phi_grid[iphi], l);
        let hi = self.phi_crossing(p, d, self.phi_grid[iphi + 1], l);
        lo.min(hi)
    }

    /// Distance `> l` at which the track crosses the half-plane at angle `phi`
    /// (a plane through the z-axis -- a CSG plane). `∞` if it does not, or if
    /// the crossing is on the antipodal (`phi + π`) half-plane.
    fn phi_crossing(&self, p: [f64; 3], d: [f64; 3], phi: f64, l: f64) -> f64 {
        let (s, co) = phi.sin_cos();
        let denom = d[0] * s - d[1] * co;
        if denom.abs() < CYL_FP {
            return f64::INFINITY; // track parallel to this half-plane
        }
        let t = -(p[0] * s - p[1] * co) / denom;
        if t <= l + CYL_COINCIDENT {
            return f64::INFINITY;
        }
        // Reject the φ+π half-plane that shares the same full line.
        let x = p[0] + t * d[0];
        let y = p[1] + t * d[1];
        if co * x + s * y > 0.0 {
            t
        } else {
            f64::INFINITY
        }
    }

    /// Distance `> l` to the next axial plane bounding cell `iz`.
    fn z_next(&self, pz: f64, dz: f64, iz: usize, l: f64) -> f64 {
        if dz.abs() < CYL_FP {
            return f64::INFINITY;
        }
        let plane = if dz > 0.0 {
            self.z_grid[iz + 1]
        } else {
            self.z_grid[iz]
        };
        let t = (plane - pz) / dz;
        if t > l + CYL_COINCIDENT {
            t
        } else {
            f64::INFINITY
        }
    }

    /// Smallest distance in `(l, total]` at which the track first lies inside
    /// the mesh, by testing every bounding-surface crossing and keeping the
    /// nearest one whose interior side is in-mesh. Handles initial entry and
    /// re-entry through the central hole / a φ wedge.
    #[allow(clippy::too_many_arguments)]
    fn first_entry(
        &self,
        p: [f64; 3],
        d: [f64; 3],
        a: f64,
        b: f64,
        c: f64,
        l: f64,
        total: f64,
    ) -> Option<f64> {
        let mut best = f64::INFINITY;
        let mut consider = |t: f64, this: &Self| {
            if t > l + CYL_COINCIDENT && t < best && t <= total {
                let probe = [
                    p[0] + (t + CYL_TINY) * d[0],
                    p[1] + (t + CYL_TINY) * d[1],
                    p[2] + (t + CYL_TINY) * d[2],
                ];
                if this.indices_local(probe[0], probe[1], probe[2]).is_some() {
                    best = t;
                }
            }
        };

        // Both radial shells bounding the mesh (inner only if r_min > 0).
        let nr = self.shape[0];
        for &(r_sq, r) in &[
            (self.r_grid_sq[0], self.r_grid[0]),
            (self.r_grid_sq[nr], self.r_grid[nr]),
        ] {
            if r > 0.0 && a >= CYL_FP {
                let pn = b / a;
                let disc = pn * pn - (c - r_sq) / a;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    consider(-pn - sq, self);
                    consider(-pn + sq, self);
                }
            }
        }

        // φ end walls (only meaningful when the mesh is a partial sector).
        if !self.full_phi {
            let nphi = self.shape[1];
            for &phi in &[self.phi_grid[0], self.phi_grid[nphi]] {
                let t = self.phi_crossing(p, d, phi, l);
                if t.is_finite() {
                    consider(t, self);
                }
            }
        }

        // z end planes.
        if d[2].abs() >= CYL_FP {
            let nz = self.shape[2];
            for &zp in &[self.z_grid[0], self.z_grid[nz]] {
                consider((zp - p[2]) / d[2], self);
            }
        }

        if best.is_finite() {
            Some(best)
        } else {
            None
        }
    }
}

/// Bracket `v` into a cell of a sorted, strictly increasing grid: returns the
/// index `i` with `grid[i] <= v < grid[i+1]`, or `None` when `v` is outside
/// `[grid[0], grid[last])`.
#[inline]
fn bracket(grid: &[f64], v: f64) -> Option<usize> {
    if v < grid[0] || v >= grid[grid.len() - 1] {
        return None;
    }
    // partition_point counts the leading edges `<= v`; subtract one for the
    // owning cell. `v >= grid[0]` guarantees the count is at least one.
    Some(grid.partition_point(|&g| g <= v) - 1)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_rectangular_rejects_bad_bounds() {
        // upper_right <= lower_left on an axis is an error, not a panic.
        let err = RegularRectangularMesh::try_new([0.0, 0.0, 0.0], [-1.0, 1.0, 1.0], [1, 1, 1])
            .unwrap_err();
        assert_eq!(
            err,
            "Mesh upper_right must be greater than lower_left in all directions"
        );
    }

    #[test]
    fn try_new_rectangular_rejects_zero_shape() {
        let err = RegularRectangularMesh::try_new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0, 1, 1])
            .unwrap_err();
        assert_eq!(err, "Mesh shape must be positive in all directions");
    }

    #[test]
    fn try_new_rectangular_accepts_valid_input() {
        let mesh = RegularRectangularMesh::try_new([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [4, 4, 4]);
        assert!(mesh.is_ok());
    }

    #[test]
    fn try_new_cylindrical_rejects_short_grid() {
        let err = CylindricalMesh::try_new(
            [0.0, 0.0, 0.0],
            vec![1.0],
            vec![0.0, TWO_PI],
            vec![0.0, 1.0],
        )
        .unwrap_err();
        assert_eq!(err, "Cylindrical mesh r-grid must have at least 2 points");
    }

    #[test]
    fn try_new_cylindrical_rejects_non_increasing_grid() {
        let err = CylindricalMesh::try_new(
            [0.0, 0.0, 0.0],
            vec![0.0, 2.0, 1.0],
            vec![0.0, TWO_PI],
            vec![0.0, 1.0],
        )
        .unwrap_err();
        assert_eq!(err, "Cylindrical mesh r-grid must be strictly increasing");
    }

    #[test]
    fn try_new_cylindrical_rejects_phi_out_of_range() {
        let err = CylindricalMesh::try_new(
            [0.0, 0.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, TWO_PI + 1.0],
            vec![0.0, 1.0],
        )
        .unwrap_err();
        assert_eq!(err, "Cylindrical mesh phi-grid must lie within [0, 2*pi]");
    }

    #[test]
    fn try_uniform_cylindrical_accepts_valid_input() {
        let mesh = CylindricalMesh::try_uniform(
            [0.0, 0.0, 0.0],
            (0.0, 10.0),
            (0.0, TWO_PI),
            (-5.0, 5.0),
            [10, 16, 20],
        );
        assert!(mesh.is_ok());
    }

    #[test]
    fn num_voxels_is_the_product_of_the_shape() {
        let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [4, 4, 4]);
        assert_eq!(m.num_voxels(), 64);
        // Non-power-of-two shapes are exact too: there is no padding to a
        // power of two now that the Morton layout is gone (issue #337).
        let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [5, 5, 5]);
        assert_eq!(m.num_voxels(), 125);
        let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [50, 50, 200]);
        assert_eq!(m.num_voxels(), 500_000);
    }

    /// Every index the mesh can produce must be addressable within
    /// `num_voxels()`, which is what sizes the tally's scoring buffers.
    #[test]
    fn every_bin_index_is_within_num_voxels() {
        let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [5, 5, 5]);
        let mut seen = std::collections::HashSet::new();
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    let pos = [ix as f64 + 0.5, iy as f64 + 0.5, iz as f64 + 0.5];
                    let bin = m.get_bin(pos).expect("inside the mesh");
                    assert!(bin < m.num_voxels(), "bin {bin} >= {}", m.num_voxels());
                    assert!(seen.insert(bin), "bin {bin} produced twice");
                }
            }
        }
        // The mapping is a bijection onto 0..num_voxels, so no slot is wasted.
        assert_eq!(seen.len(), m.num_voxels());
    }

    #[test]
    fn get_bin_indices_match_row_major_formula() {
        let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [4, 4, 4]);
        // (ix, iy, iz) = (1, 2, 3) → (iz*ny + iy)*nx + ix = (3*4+2)*4+1 = 57.
        let pos = [1.5, 2.5, 3.5]; // → indices (1, 2, 3)
        assert_eq!(m.get_bin(pos), Some(57));
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod cylindrical_tests {
    use super::*;

    fn frac_sum(c: &[MeshCrossing]) -> f64 {
        c.iter().map(|m| m.length_fraction).sum()
    }

    /// Full-2π mesh, two radial rings, single z layer.
    fn full_mesh() -> CylindricalMesh {
        CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, TWO_PI],
            vec![0.0, 10.0],
        )
    }

    #[test]
    fn shape_and_num_bins() {
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, std::f64::consts::PI, TWO_PI],
            vec![0.0, 5.0, 10.0],
        );
        assert_eq!(m.shape(), [2, 2, 2]);
        assert_eq!(m.num_bins(), 8);
        assert!(m.full_phi);
    }

    #[test]
    fn partial_phi_not_full() {
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 2.0],
            vec![std::f64::consts::PI, TWO_PI],
            vec![0.0, 1.0],
        );
        assert!(!m.full_phi);
    }

    #[test]
    fn get_bin_known_points() {
        let m = full_mesh();
        // (1.5, 0, 5): rho = 1.5 -> ir = 1; phi = 0 -> iphi = 0; iz = 0.
        assert_eq!(m.get_bin([1.5, 0.0, 5.0]), Some(1));
        // (0.5, 0, 5): rho = 0.5 -> ir = 0.
        assert_eq!(m.get_bin([0.5, 0.0, 5.0]), Some(0));
        // Outside radially / axially.
        assert_eq!(m.get_bin([3.0, 0.0, 5.0]), None);
        assert_eq!(m.get_bin([0.5, 0.0, -1.0]), None);
    }

    #[test]
    fn get_bin_phi_quadrants() {
        // Four φ sectors; check a point in each maps to the right bin.
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 2.0],
            vec![
                0.0,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                3.0 * std::f64::consts::FRAC_PI_2,
                TWO_PI,
            ],
            vec![0.0, 1.0],
        );
        assert_eq!(m.shape(), [1, 4, 1]);
        assert_eq!(m.get_bin([1.0, 0.1, 0.5]), Some(0)); // φ ≈ 0
        assert_eq!(m.get_bin([-0.1, 1.0, 0.5]), Some(1)); // φ ≈ π/2
        assert_eq!(m.get_bin([-1.0, -0.1, 0.5]), Some(2)); // φ ≈ π
        assert_eq!(m.get_bin([0.1, -1.0, 0.5]), Some(3)); // φ ≈ 3π/2
    }

    #[test]
    fn volume_matches_annulus_formula() {
        let m = full_mesh();
        // Inner ring r∈[0,1]: ½(1-0)(2π)(10) = 10π. Outer r∈[1,2]: ½(4-1)(2π)(10)=30π.
        assert!((m.get_voxel_volume(0) - 10.0 * std::f64::consts::PI).abs() < 1e-9);
        assert!((m.get_voxel_volume(1) - 30.0 * std::f64::consts::PI).abs() < 1e-9);
        // Sum over all bins == full cylinder volume π r_max² h = π·4·10 = 40π.
        let total: f64 = (0..m.num_bins()).map(|b| m.get_voxel_volume(b)).sum();
        assert!((total - 40.0 * std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn diametral_track_through_axis() {
        // Track along the x-axis from (-3,0,5) to (3,0,5) through a single
        // azimuthal bin. In-mesh span is x∈[-2,2] (length 4 of total 6).
        let m = full_mesh();
        let cr = m.bins_crossed([-3.0, 0.0, 5.0], [3.0, 0.0, 5.0], [1.0, 0.0, 0.0]);
        // bin1 (outer) twice at 1/6 each, bin0 (inner) once at 2/6.
        let total: f64 = frac_sum(&cr);
        assert!((total - 4.0 / 6.0).abs() < 1e-6, "sum was {total}");
        let bin0: f64 = cr
            .iter()
            .filter(|c| c.bin == 0)
            .map(|c| c.length_fraction)
            .sum();
        let bin1: f64 = cr
            .iter()
            .filter(|c| c.bin == 1)
            .map(|c| c.length_fraction)
            .sum();
        assert!((bin0 - 2.0 / 6.0).abs() < 1e-6, "bin0 {bin0}");
        assert!((bin1 - 2.0 / 6.0).abs() < 1e-6, "bin1 {bin1}");
    }

    #[test]
    fn interior_track_fractions_sum_to_one() {
        // A track that starts and ends inside the mesh: fractions sum to 1.
        let m = full_mesh();
        let cr = m.bins_crossed([-1.5, -0.3, 2.0], [1.5, 0.7, 8.0], unit([3.0, 1.0, 6.0]));
        let s = frac_sum(&cr);
        assert!((s - 1.0).abs() < 1e-6, "interior track sum {s}");
    }

    #[test]
    fn axis_parallel_track_only_z_crossings() {
        // Track parallel to z at fixed (r, φ): crosses only z layers.
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 2.0],
            vec![0.0, TWO_PI],
            vec![0.0, 1.0, 2.0, 3.0],
        );
        let cr = m.bins_crossed([0.5, 0.0, -1.0], [0.5, 0.0, 4.0], [0.0, 0.0, 1.0]);
        // z span inside mesh is [0,3] of total 5 -> sum 3/5; three z cells.
        let s = frac_sum(&cr);
        assert!((s - 3.0 / 5.0).abs() < 1e-6, "sum {s}");
        assert_eq!(cr.len(), 3);
        assert_eq!(cr[0].bin, 0);
        assert_eq!(cr[1].bin, 1);
        assert_eq!(cr[2].bin, 2);
    }

    #[test]
    fn radial_hole_gives_two_groups() {
        // r_min > 0: a diametral track enters, leaves through the inner hole,
        // re-enters on the far side. Expect two tallied groups + untallied gap.
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![1.0, 2.0], // hole of radius 1
            vec![0.0, TWO_PI],
            vec![0.0, 1.0],
        );
        let cr = m.bins_crossed([-3.0, 0.0, 0.5], [3.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
        // In-mesh: x∈[-2,-1] and x∈[1,2], each length 1 of total 6.
        let s = frac_sum(&cr);
        assert!((s - 2.0 / 6.0).abs() < 1e-6, "hole track sum {s}");
        // All in the single bin 0, but as two separate crossings.
        assert_eq!(cr.len(), 2, "expected two groups across the hole");
    }

    #[test]
    fn partial_phi_wedge_no_wrap() {
        // A 90° wedge [0, π/2]: a track must not wrap across the open ends.
        let m = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 2.0],
            vec![0.0, std::f64::consts::FRAC_PI_2],
            vec![0.0, 1.0],
        );
        assert!(!m.full_phi);
        // Track crossing the wedge stays finite and sums to <= 1.
        let cr = m.bins_crossed([0.5, -1.0, 0.5], [0.5, 1.0, 0.5], [0.0, 1.0, 0.0]);
        let s = frac_sum(&cr);
        assert!(s > 0.0 && s <= 1.0 + 1e-6, "wedge sum {s}");
    }

    #[test]
    fn matches_substepping_reference() {
        // Cross-check several tracks against a fine midpoint-substepping
        // reference (used only in the test, never in production).
        let m = CylindricalMesh::new(
            [0.5, -0.5, 0.0],
            vec![0.0, 0.7, 1.4, 2.1],
            vec![0.0, std::f64::consts::PI, TWO_PI],
            vec![-1.0, 0.0, 1.0, 2.0],
        );
        let tracks = [
            ([-2.0, -2.0, -0.5], [2.0, 2.0, 1.5]),
            ([0.4, -0.4, -2.0], [0.6, -0.6, 2.5]),
            ([-3.0, 0.2, 0.3], [3.0, 0.1, 0.4]),
        ];
        for (r0, r1) in tracks {
            let dir = unit([r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]);
            let cr = m.bins_crossed(r0, r1, dir);
            let mut got = std::collections::HashMap::new();
            for c in &cr {
                *got.entry(c.bin).or_insert(0.0) += c.length_fraction;
            }
            let reference = substep_reference(&m, r0, r1, 200_000);
            for (bin, ref_frac) in &reference {
                let g = got.get(bin).copied().unwrap_or(0.0);
                assert!(
                    (g - ref_frac).abs() < 2e-3,
                    "bin {bin}: dda {g} vs ref {ref_frac} for track {r0:?}->{r1:?}"
                );
            }
        }
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// Reference: split the track into many segments and bin each midpoint.
    fn substep_reference(
        m: &CylindricalMesh,
        r0: [f64; 3],
        r1: [f64; 3],
        n: usize,
    ) -> std::collections::HashMap<usize, f64> {
        let mut acc = std::collections::HashMap::new();
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let mid = [
                r0[0] + t * (r1[0] - r0[0]),
                r0[1] + t * (r1[1] - r0[1]),
                r0[2] + t * (r1[2] - r0[2]),
            ];
            if let Some(bin) = m.get_bin(mid) {
                *acc.entry(bin).or_insert(0.0) += 1.0 / n as f64;
            }
        }
        acc
    }
}
