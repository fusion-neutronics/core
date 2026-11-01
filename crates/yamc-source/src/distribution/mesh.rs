//! A source spread over the elements of a mesh: a strength per element, and
//! uniform positions within each.
//!
//! The elements are numbered as the matching mesh tally numbers its bins, so
//! a per-element rate (an R2S decay photon rate per voxel, an activation
//! result per tet) comes straight from a tally's flat `mean`.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// The elements of a mesh, in the order its tallies number their bins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeshElements {
    /// Voxels of a regular rectangular mesh; bin `(iz * ny + iy) * nx + ix`.
    Rectangular {
        lower_left: [f64; 3],
        width: [f64; 3],
        shape: [usize; 3],
    },
    /// Cells of a cylindrical mesh with its axis along +z through `origin`;
    /// bin `(iz * nphi + iphi) * nr + ir`. `z` edges are offsets from
    /// `origin[2]`, `phi` edges are radians.
    Cylindrical {
        origin: [f64; 3],
        r: Vec<f64>,
        phi: Vec<f64>,
        z: Vec<f64>,
    },
    /// Tetrahedra as their four corners; bin = tetrahedron id.
    Tetrahedra(Vec<[[f64; 3]; 4]>),
}

impl MeshElements {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Rectangular { shape, .. } => shape.iter().product(),
            Self::Cylindrical { r, phi, z, .. } => (r.len() - 1) * (phi.len() - 1) * (z.len() - 1),
            Self::Tetrahedra(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Volume of element `i` in cm³.
    pub fn volume(&self, i: usize) -> f64 {
        match self {
            Self::Rectangular { width, .. } => width.iter().product(),
            Self::Cylindrical { r, phi, z, .. } => {
                let [ir, ip, iz] = cyl_indices(r.len() - 1, phi.len() - 1, i);
                0.5 * (r[ir + 1].powi(2) - r[ir].powi(2))
                    * (phi[ip + 1] - phi[ip])
                    * (z[iz + 1] - z[iz])
            }
            Self::Tetrahedra(t) => {
                let [a, b, c, d] = t[i];
                let e = |p: [f64; 3]| [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let (u, v, w) = (e(b), e(c), e(d));
                (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0]))
                    .abs()
                    / 6.0
            }
        }
    }

    /// A uniform point in element `i`.
    pub fn sample<R: Rng + ?Sized>(&self, i: usize, rng: &mut R) -> [f64; 3] {
        match self {
            Self::Rectangular {
                lower_left,
                width,
                shape,
            } => {
                let idx = [
                    i % shape[0],
                    (i / shape[0]) % shape[1],
                    i / (shape[0] * shape[1]),
                ];
                std::array::from_fn(|k| {
                    lower_left[k] + width[k] * (idx[k] as f64 + rng.random::<f64>())
                })
            }
            Self::Cylindrical { origin, r, phi, z } => {
                let [ir, ip, iz] = cyl_indices(r.len() - 1, phi.len() - 1, i);
                // Uniform in volume: r² is uniform between the ring's edges.
                let (r0, r1) = (r[ir] * r[ir], r[ir + 1] * r[ir + 1]);
                let rad = (r0 + (r1 - r0) * rng.random::<f64>()).sqrt();
                let p = phi[ip] + (phi[ip + 1] - phi[ip]) * rng.random::<f64>();
                let zz = z[iz] + (z[iz + 1] - z[iz]) * rng.random::<f64>();
                let (s, c) = p.sin_cos();
                [origin[0] + rad * c, origin[1] + rad * s, origin[2] + zz]
            }
            Self::Tetrahedra(t) => {
                // Sorted uniforms split [0, 1] into four uniform barycentric weights.
                let mut u: [f64; 3] = std::array::from_fn(|_| rng.random::<f64>());
                u.sort_unstable_by(f64::total_cmp);
                let w = [u[0], u[1] - u[0], u[2] - u[1], 1.0 - u[2]];
                let v = &t[i];
                std::array::from_fn(|k| (0..4).map(|j| w[j] * v[j][k]).sum())
            }
        }
    }

    /// The elements `keep` names, renumbered from zero, for tetrahedra (whose
    /// corners are stored); structured meshes compute theirs from the bin.
    fn retain_tetrahedra(&mut self, keep: &[u32]) {
        if let Self::Tetrahedra(t) = self {
            *t = keep.iter().map(|&i| t[i as usize]).collect();
        }
    }
}

fn cyl_indices(nr: usize, nphi: usize, i: usize) -> [usize; 3] {
    [i % nr, (i / nr) % nphi, i / (nr * nphi)]
}

/// Where in each element a mesh source may emit: only points in these cells,
/// or in cells filled with these materials. Lets an element straddling void
/// or several cells (an R2S voxel) emit only where its strength comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceDomain {
    /// Cell ids.
    Cells(Vec<u32>),
    /// Material ids.
    Materials(Vec<u32>),
}

/// Draws within one element before it is declared to have no point in the
/// domain: enough for a part of 0.01% of it to be found with certainty.
pub const MAX_DOMAIN_TRIES: usize = 100_000;

/// Positions spread over a mesh: element `i` is chosen in proportion to its
/// strength, then a point uniform within it.
///
/// Only elements with a positive strength are kept, so a mesh that is mostly
/// cold costs memory and sampling time for its emitting elements alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MeshDistributionData")]
pub struct MeshDistribution {
    /// The mesh; for tetrahedra, only the emitting ones.
    elements: MeshElements,
    /// Number of elements in the whole mesh.
    num_elements: usize,
    /// Mesh bin of each emitting element.
    bins: Vec<u32>,
    /// Cumulative strength through each emitting element.
    cumulative: Vec<f64>,
    /// Where within its element a point may be born; anywhere when `None`.
    domain: Option<SourceDomain>,
}

#[derive(Deserialize)]
struct MeshDistributionData {
    elements: MeshElements,
    num_elements: usize,
    bins: Vec<u32>,
    cumulative: Vec<f64>,
    #[serde(default)]
    domain: Option<SourceDomain>,
}

impl TryFrom<MeshDistributionData> for MeshDistribution {
    type Error = String;

    fn try_from(d: MeshDistributionData) -> Result<Self, String> {
        let stored = match d.elements {
            MeshElements::Tetrahedra(_) => d.bins.len(),
            _ => d.num_elements,
        };
        let ok = d.elements.len() == stored
            && d.bins.len() == d.cumulative.len()
            && d.bins.windows(2).all(|w| w[0] < w[1])
            && d.bins
                .last()
                .is_some_and(|&b| (b as usize) < d.num_elements)
            && d.cumulative.windows(2).all(|w| w[0] < w[1])
            && d.cumulative.first().is_some_and(|&c| c > 0.0)
            && d.cumulative.last().is_some_and(|c| c.is_finite())
            && !d.domain.as_ref().is_some_and(SourceDomain::is_empty);
        if !ok {
            return Err("the mesh source in the model file is inconsistent".into());
        }
        Ok(Self {
            elements: d.elements,
            num_elements: d.num_elements,
            bins: d.bins,
            cumulative: d.cumulative,
            domain: d.domain,
        })
    }
}

impl MeshDistribution {
    /// Spread over `elements` with `strengths` per element (in bin order), or
    /// in proportion to element volume (uniform over the mesh) when `None`.
    ///
    /// Refuses a strength list of the wrong length, a strength that is NaN,
    /// infinite or negative, and strengths that sum to zero.
    pub fn new(mut elements: MeshElements, strengths: Option<&[f64]>) -> Result<Self, String> {
        let n = elements.len();
        if n == 0 {
            return Err("the mesh has no elements to place a source in".into());
        }
        if u32::try_from(n).is_err() {
            return Err(format!(
                "the mesh has {n} elements, more than a source can index"
            ));
        }
        let strengths: Vec<f64> = match strengths {
            Some(s) if s.len() != n => {
                return Err(format!(
                    "strengths has {} values but the mesh has {n} elements; give one per \
                     element, in the order a tally on the mesh numbers its bins",
                    s.len()
                ));
            }
            Some(s) => s.to_vec(),
            None => (0..n).map(|i| elements.volume(i)).collect(),
        };
        if let Some(i) = strengths.iter().position(|s| !s.is_finite() || *s < 0.0) {
            return Err(format!(
                "strengths[{i}] is {}; every strength must be a finite number >= 0",
                strengths[i]
            ));
        }
        let (mut bins, mut cumulative, mut total) = (Vec::new(), Vec::new(), 0.0);
        for (i, &s) in strengths.iter().enumerate().filter(|(_, s)| **s > 0.0) {
            total += s;
            bins.push(i as u32);
            cumulative.push(total);
        }
        if bins.is_empty() {
            return Err("every strength is zero, so the mesh source never emits".into());
        }
        elements.retain_tetrahedra(&bins);
        Ok(Self {
            elements,
            num_elements: n,
            bins,
            cumulative,
            domain: None,
        })
    }

    /// Sum of the element strengths.
    pub fn total(&self) -> f64 {
        *self.cumulative.last().unwrap()
    }

    /// Number of elements in the mesh.
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Number of elements with a positive strength.
    pub fn num_emitting(&self) -> usize {
        self.bins.len()
    }

    /// The mesh elements (for tetrahedra, only the emitting ones).
    pub fn elements(&self) -> &MeshElements {
        &self.elements
    }

    /// Births limited to `domain` within each element, each element keeping
    /// its strength. Refuses an empty cell or material list.
    pub fn with_domain(mut self, domain: SourceDomain) -> Result<Self, String> {
        if domain.is_empty() {
            return Err("a mesh source's cells or materials must not be empty".into());
        }
        self.domain = Some(domain);
        Ok(self)
    }

    /// Where within its element a point may be born; anywhere when `None`.
    pub fn domain(&self) -> Option<&SourceDomain> {
        self.domain.as_ref()
    }

    /// The `elements` index of the `k`th emitting element.
    fn index(&self, k: usize) -> usize {
        match self.elements {
            MeshElements::Tetrahedra(_) => k,
            _ => self.bins[k] as usize,
        }
    }

    /// The `k` of an emitting element chosen by strength.
    fn pick<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let t = rng.random::<f64>() * self.total();
        let k = self.cumulative.partition_point(|&c| c <= t);
        k.min(self.bins.len() - 1)
    }

    /// A point anywhere in an element chosen by strength, ignoring the domain.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        let i = self.index(self.pick(rng));
        self.elements.sample(i, rng)
    }

    /// A point in an element chosen by strength, redrawn within that element
    /// until `inside` accepts it. Redrawing in the same element keeps each
    /// element's strength its own. `Err` with the element's mesh bin after
    /// [`MAX_DOMAIN_TRIES`] misses.
    pub fn sample_within<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Result<[f64; 3], u32> {
        let k = self.pick(rng);
        let i = self.index(k);
        (0..MAX_DOMAIN_TRIES)
            .map(|_| self.elements.sample(i, rng))
            .find(|&p| inside(p))
            .ok_or(self.bins[k])
    }

    /// The mesh bin of the first emitting element with no point `inside`
    /// accepts in [`MAX_DOMAIN_TRIES`] draws, if any: its strength could
    /// never be emitted.
    pub fn unreachable_bin<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Option<u32> {
        (0..self.bins.len())
            .find(|&k| {
                let i = self.index(k);
                !(0..MAX_DOMAIN_TRIES).any(|_| inside(self.elements.sample(i, rng)))
            })
            .map(|k| self.bins[k])
    }
}

impl SourceDomain {
    fn is_empty(&self) -> bool {
        match self {
            Self::Cells(v) | Self::Materials(v) => v.is_empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const N: usize = 200_000;

    /// Fraction of `N` samples landing in each element, by `locate`.
    fn fractions(d: &MeshDistribution, n: usize, locate: impl Fn([f64; 3]) -> usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = vec![0usize; n];
        for _ in 0..N {
            counts[locate(d.sample(&mut rng))] += 1;
        }
        counts.iter().map(|&c| c as f64 / N as f64).collect()
    }

    fn close(got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.005, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn rectangular_follows_strengths_in_tally_bin_order() {
        let elements = MeshElements::Rectangular {
            lower_left: [0.0; 3],
            width: [1.0, 2.0, 3.0],
            shape: [2, 2, 1],
        };
        let d = MeshDistribution::new(elements, Some(&[1.0, 0.0, 2.0, 1.0])).unwrap();
        // (iz * ny + iy) * nx + ix
        let f = fractions(&d, 4, |p| {
            assert!((0.0..3.0).contains(&p[2]));
            (p[1] / 2.0) as usize * 2 + p[0] as usize
        });
        close(&f, &[0.25, 0.0, 0.5, 0.25]);
        assert_eq!((d.num_elements(), d.num_emitting(), d.total()), (4, 3, 4.0));
    }

    #[test]
    fn cylindrical_is_uniform_in_volume() {
        // Two rings, one sector each, default strengths: the outer ring has
        // three times the inner's volume, and r² is uniform within each.
        let elements = MeshElements::Cylindrical {
            origin: [1.0, 2.0, 3.0],
            r: vec![0.0, 1.0, 2.0],
            phi: vec![0.0, std::f64::consts::PI],
            z: vec![0.0, 1.0],
        };
        let d = MeshDistribution::new(elements, None).unwrap();
        let f = fractions(&d, 4, |p| {
            let (x, y) = (p[0] - 1.0, p[1] - 2.0);
            assert!(y >= 0.0 && (3.0..4.0).contains(&p[2]));
            // Quarters of r² over [0, 4].
            ((x * x + y * y) as usize).min(3)
        });
        close(&f, &[0.25; 4]);
    }

    #[test]
    fn tetrahedron_points_are_uniform_inside() {
        // The unit corner tet, split by x + y + z at the plane halving it
        // by volume: (1/2)^(1/3) of the edge.
        let unit = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let shifted = unit.map(|p| [p[0] + 5.0, p[1], p[2]]);
        let d = MeshDistribution::new(
            MeshElements::Tetrahedra(vec![shifted, unit, unit]),
            Some(&[0.0, 1.0, 0.0]),
        )
        .unwrap();
        assert_eq!((d.num_elements(), d.num_emitting()), (3, 1));
        let cut = 0.5f64.cbrt();
        let f = fractions(&d, 2, |p| {
            assert!(p.iter().all(|&c| c >= 0.0) && p.iter().sum::<f64>() <= 1.0);
            usize::from(p.iter().sum::<f64>() > cut)
        });
        close(&f, &[0.5, 0.5]);
        assert!((MeshElements::Tetrahedra(vec![unit]).volume(0) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn bad_strengths_are_refused() {
        let e = || MeshElements::Rectangular {
            lower_left: [0.0; 3],
            width: [1.0; 3],
            shape: [2, 1, 1],
        };
        for (s, msg) in [
            (vec![1.0], "has 1 values but the mesh has 2"),
            (vec![1.0, -1.0], "strengths[1] is -1"),
            (vec![f64::NAN, 1.0], "strengths[0] is NaN"),
            (vec![0.0, 0.0], "never emits"),
        ] {
            let err = MeshDistribution::new(e(), Some(&s)).unwrap_err();
            assert!(err.contains(msg), "{err}");
        }
    }

    #[test]
    fn domain_redraws_within_the_chosen_element() {
        // Two voxels, strengths 3:1; only x < 0.5 and 1.0 < x < 1.25 are
        // inside, so each voxel keeps its share in its inside part.
        let d = MeshDistribution::new(
            MeshElements::Rectangular {
                lower_left: [0.0; 3],
                width: [1.0; 3],
                shape: [2, 1, 1],
            },
            Some(&[3.0, 1.0]),
        )
        .unwrap()
        .with_domain(SourceDomain::Materials(vec![1]))
        .unwrap();
        let inside = |p: [f64; 3]| p[0] < 0.5 || (1.0..1.25).contains(&p[0]);
        let mut rng = StdRng::seed_from_u64(3);
        let mut first = 0usize;
        for _ in 0..N {
            let p = d.sample_within(&mut rng, inside).unwrap();
            assert!(inside(p));
            first += usize::from(p[0] < 1.0);
        }
        assert!((first as f64 / N as f64 - 0.75).abs() < 0.005);
        assert_eq!(d.unreachable_bin(&mut rng, inside), None);
        // The second voxel has nothing inside x < 0.5.
        let left = |p: [f64; 3]| p[0] < 0.5;
        assert_eq!(d.unreachable_bin(&mut rng, left), Some(1));
        assert!(MeshDistribution::new(
            MeshElements::Tetrahedra(vec![[
                [0.0; 3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]]),
            None
        )
        .unwrap()
        .with_domain(SourceDomain::Cells(vec![]))
        .is_err());
    }

    #[test]
    fn model_file_round_trip() {
        let d = MeshDistribution::new(
            MeshElements::Rectangular {
                lower_left: [0.0; 3],
                width: [1.0; 3],
                shape: [3, 1, 1],
            },
            Some(&[0.0, 2.0, 1.0]),
        )
        .unwrap()
        .with_domain(SourceDomain::Cells(vec![4, 7]))
        .unwrap();
        let back: MeshDistribution =
            serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(back, d);
        let broken =
            r#"{"elements":{"Tetrahedra":[]},"num_elements":1,"bins":[0],"cumulative":[1.0]}"#;
        assert!(serde_json::from_str::<MeshDistribution>(broken).is_err());
    }
}
