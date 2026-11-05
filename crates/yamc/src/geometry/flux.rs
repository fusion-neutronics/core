//! Flux surfaces of plasma shapes, as CSG surfaces.

use crate::geo::{BoundaryType, FluxBoundary, Surface};
use std::sync::Arc;
use yamc_source::stellarator::StellaratorShape;

/// The flux surface `rho` of a stellarator plasma shape (the same shape
/// that drives the stellarator source), as a CSG surface whose inside
/// (`below`) is the plasma within `rho`. `rho` is in `(0, 1]`; surfaces
/// beyond the plasma are offsets of the boundary (`Surface::offset`).
pub fn stellarator_flux_surface(
    shape: &StellaratorShape,
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
    let (r_modes, r_mn, z_modes, z_mn) = shape.surface_modes(rho);
    let flux = FluxBoundary::new(shape.num_field_periods(), r_modes, r_mn, z_modes, z_mn)?;
    Ok(Surface::new_flux(Arc::new(flux), surface_id, boundary))
}
