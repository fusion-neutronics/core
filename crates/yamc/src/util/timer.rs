//! Wall-clock timer with a wasm32-safe fallback.
//!
//! On native, `std::time::Instant::now()` is a real monotonic clock.
//! On `wasm32-unknown-unknown` it has no clock source and panics at
//! runtime. The transport loop uses three unconditional wall-clock
//! measurements (for the user-visible "Time: X s (data load: Y s,
//! transport: Z s)" summary), so it needs a no-panic fallback to be
//! callable from the browser.
//!
//! The wasm32 stub always reports 0.0 seconds. That makes the timing
//! summary line meaningless under wasm, but the transport itself runs
//! without panicking. A future browser host can swap in a real timer
//! by wrapping `performance.now()` via web-sys.

#[cfg(not(target_arch = "wasm32"))]
pub struct Timer {
    inner: std::time::Instant,
}

#[cfg(target_arch = "wasm32")]
pub struct Timer;

impl Timer {
    /// Start a new timer.
    #[inline]
    pub fn start() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Timer {
                inner: std::time::Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Timer
        }
    }

    /// Seconds elapsed since [`Timer::start`]. Always 0.0 on wasm32.
    #[inline]
    pub fn elapsed_secs(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.elapsed().as_secs_f64()
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }
}
