//! Errors returned by the GPU translation layer.
//!
//! Every variant represents a feature of yamc's transport that the GPU
//! kernel doesn't support. The `compute='auto'` dispatch will fall back
//! to CPU when one of these fires; `compute='gpu'` (explicit) will
//! propagate it as a Python exception with the variant's display string.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum GpuTranslateError {
    /// Mesh geometry isn't supported by the GPU kernel -- only CSG.
    MeshGeometryUnsupported,
    /// A CSG cell is filled by a mesh body (issue #232); the kernel has
    /// no mesh-in-cell tracking.
    MeshFillUnsupported,
    /// A cell uses a region that the GPU kernel can't represent.
    /// Examples: nested boolean operations beyond an AABB, infinite
    /// extents, regions whose bounding box can't be derived.
    CellRegionUnsupported {
        cell_id: Option<u32>,
        reason: String,
    },
    /// A source that emits something other than a neutron reached the
    /// NEUTRON translation. Photon sources have their own kernel and are
    /// routed to it by the dispatch (all-photon models to the photon pass,
    /// neutron+photon models to the mixed pass) before they get here.
    NonNeutronSource { particle_type: String },
    /// A photon source reached the NEUTRON translation. Defensive: the
    /// dispatch routes photon-source models to the photon kernel first.
    /// Coupled neutron->photon transport (`transport_secondary_photons`) and
    /// D1S decay photons both run on the GPU, so neither is rejected here.
    PhotonSourceOnNeutronPath,
    /// A material extracted XS that didn't include both elastic and
    /// absorption -- usually means the source data is missing the MT.
    MissingReactionData { material: String, mt: i32 },
    /// A nuclide carries neutron-emitting MTs the kernel has no slot for, at a
    /// cross section large enough to bias the answer (issue #106). Those
    /// channels fall into the GPU's derived absorption, so it would kill
    /// neutrons the CPU scatters. Refused rather than run quietly wrong.
    UnslottedScatterMts { material: String, detail: String },
    /// The translation found no usable temperature on a nuclide.
    NoLoadedTemperature { nuclide: String, material: String },
    /// `Model.sources` is empty.
    NoSources,
    /// `n_particles == 0`.
    NoParticles,
}

impl fmt::Display for GpuTranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeshGeometryUnsupported => write!(
                f,
                "mesh geometry is not supported by the GPU kernel -- use CSG geometry"
            ),
            Self::MeshFillUnsupported => write!(
                f,
                "mesh-filled cells are not supported by the GPU kernel -- run with compute='cpu'"
            ),
            Self::CellRegionUnsupported { cell_id, reason } => match cell_id {
                Some(id) => write!(f, "cell {id}: region not supported on GPU ({reason})"),
                None => write!(f, "cell: region not supported on GPU ({reason})"),
            },
            Self::NonNeutronSource { particle_type } => write!(
                f,
                "source emits `{particle_type}`; the GPU neutron kernel only handles \
                 neutron sources"
            ),
            Self::PhotonSourceOnNeutronPath => write!(
                f,
                "a photon source reached the GPU neutron translation; photon sources \
                 belong to the photon kernel"
            ),
            Self::MissingReactionData { material, mt } => write!(
                f,
                "material `{material}`: nuclide is missing reaction MT={mt} \
                 (need elastic=2 and capture=102 for the GPU kernel)"
            ),
            Self::UnslottedScatterMts { material, detail } => {
                write!(f, "material `{material}`: {detail}")
            }
            Self::NoLoadedTemperature { nuclide, material } => write!(
                f,
                "material `{material}`: nuclide `{nuclide}` has no loaded temperature data"
            ),
            Self::NoSources => write!(f, "model has no sources"),
            Self::NoParticles => write!(f, "n_particles must be > 0"),
        }
    }
}

impl std::error::Error for GpuTranslateError {}
