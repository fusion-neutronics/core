//! Spatial, energy, and angular distributions that a `Source` composes from.

pub mod angular;
pub mod energy;
pub mod mesh;
pub mod spatial;

/// Refuse a NaN or infinite `value` for the parameter `what`, saying which
/// and what it was.
///
/// The numbers a source is built from pass through here, so a NaN or an
/// infinity is refused where it is given, in the constructor or as a model
/// file is read, rather than surfacing as a NaN tally or a lost particle in
/// the middle of a run.
pub fn check_finite(what: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{what} must be a finite number (got {value})"))
    }
}
