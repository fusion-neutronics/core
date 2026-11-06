//! Flux surfaces of plasma shapes, as CSG surfaces.

use crate::geo::{BoundaryType, FluxBoundary, Surface};
use std::sync::Arc;
use yamc_source::stellarator::StellaratorShape;
use yamc_source::tokamak::TokamakShape;

/// A plasma shape to take flux surfaces from: the same shapes that drive
/// the stellarator and tokamak sources.
#[derive(Debug, Clone, Copy)]
pub enum PlasmaShape<'a> {
    Stellarator(&'a StellaratorShape),
    Tokamak(&'a TokamakShape),
}

/// The flux surface `rho` of a plasma shape as a CSG surface whose inside
/// (`below`) is the plasma within `rho`. `rho` is in `(0, 1]`; surfaces
/// beyond the plasma are offsets of the boundary (`Surface::offset`).
pub fn flux_surface(
    shape: PlasmaShape<'_>,
    rho: f64,
    surface_id: Option<usize>,
    boundary: Option<BoundaryType>,
) -> Result<Surface, String> {
    if !(rho > 0.0 && rho <= 1.0) {
        return Err(format!(
            "rho must be in (0, 1] (got {rho}); a surface outside the plasma is an \
             OffsetSurface of the rho = 1 flux surface"
        ));
    }
    let (num_field_periods, (r_modes, r_mn, z_modes, z_mn)) = match shape {
        PlasmaShape::Stellarator(shape) => (shape.num_field_periods(), shape.surface_modes(rho)),
        // Axisymmetric: one field period, with toroidal mode 0 alone.
        PlasmaShape::Tokamak(shape) => (1, shape.surface_modes(rho)),
    };
    let flux = FluxBoundary::new(num_field_periods, r_modes, r_mn, z_modes, z_mn)?;
    Ok(Surface::new_flux(Arc::new(flux), surface_id, boundary))
}
