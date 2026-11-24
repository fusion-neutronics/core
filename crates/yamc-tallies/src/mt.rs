use std::fmt;

/// A validated ENDF MT reaction number (1..=999).
///
/// Wraps a `u16` and guarantees the value is in the ENDF range.
/// Use `Mt::new()` for compile-time constants (panics on invalid)
/// or `Mt::try_new()` / `TryFrom<i32>` for runtime-validated construction.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Mt(u16);

impl Mt {
    // Well-known MT constants used in the codebase
    pub const TOTAL: Mt = Mt(1);
    pub const ELASTIC: Mt = Mt(2);
    pub const INELASTIC: Mt = Mt(4);
    pub const FISSION: Mt = Mt(18);
    pub const ABSORPTION: Mt = Mt(27);
    pub const H1_PRODUCTION: Mt = Mt(203);
    pub const H2_PRODUCTION: Mt = Mt(204);
    pub const H3_PRODUCTION: Mt = Mt(205);
    pub const HE3_PRODUCTION: Mt = Mt(206);
    pub const HE4_PRODUCTION: Mt = Mt(207);
    pub const HEATING: Mt = Mt(301);
    pub const DAMAGE_ENERGY: Mt = Mt(444);
    pub const COHERENT: Mt = Mt(502);
    pub const INCOHERENT: Mt = Mt(504);
    pub const PAIR_PRODUCTION: Mt = Mt(516);
    pub const PHOTOELECTRIC: Mt = Mt(522);
    pub const HEATING_LOCAL: Mt = Mt(901);

    /// Create a new Mt, panicking if value is out of range.
    /// Intended for compile-time constants and cases where the value is known-valid.
    pub const fn new(value: u16) -> Self {
        assert!(
            value >= 1 && value <= 999,
            "MT number must be in range 1..=999"
        );
        Mt(value)
    }

    /// Fallible constructor returning `None` for out-of-range values.
    pub const fn try_new(value: u16) -> Option<Self> {
        if value >= 1 && value <= 999 {
            Some(Mt(value))
        } else {
            None
        }
    }

    /// Convert to `i32` for the existing `lookup_xs_by_mt` API.
    #[inline]
    pub fn as_i32(self) -> i32 {
        self.0 as i32
    }

    /// Get the raw `u16` value.
    #[inline]
    pub fn as_u16(self) -> u16 {
        self.0
    }
}

impl TryFrom<i32> for Mt {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if (1..=999).contains(&value) {
            Ok(Mt(value as u16))
        } else {
            Err(format!("MT number {value} out of range (must be 1..=999)"))
        }
    }
}

impl fmt::Display for Mt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
