//! GPU compute infrastructure for yamc.
//!
//! Targets the Vulkan backend through `cubecl-wgpu` and **requires** f64
//! shader support (`SHADER_F64`). yamc's transport math is f64 throughout
//! -- shielding/dose work depends on it, so a GPU path running in f32 is
//! not in scope. Adapters that don't expose f64 are rejected at init.
//!
//! On Apple Silicon, the WebGPU spec, and any browser/WASM target, this
//! crate has no usable f64 path. On macOS specifically the cubecl/wgpu
//! deps are dropped at build time so the crate compiles to a stub: the
//! public API still exists but `GpuContext::new()` returns `NoF64Adapter`
//! and `list_vulkan_f64_adapters()` is empty. Callers should treat that
//! as the cue to fall back to CPU.
//!
//! # Example
//!
//! ```no_run
//! match yamc_gpu::GpuContext::new() {
//!     Ok(ctx) => println!("GPU ready: {}", ctx.adapter_info()),
//!     Err(e) => println!("GPU unavailable: {e}"),
//! }
//! ```

pub mod common;
pub mod neutron;
pub mod photon;

pub use common::particle::GpuParticle;
pub use common::rng::GpuRng;
pub use neutron::xs::{
    extract_material_xs, extract_per_nuclide_inelastic, extract_per_nuclide_macro_total_xs,
    extract_score_xs_per_mt, extract_xs_from_nuclide, GpuNuclideXs, NuclideXsError,
    PerNuclideInelastic, PerNuclideMacroXs,
};
pub use photon::xs::atomic_relaxation_xs::{
    extract_atomic_relaxation_for_gpu, GpuAtomicRelaxation, MAX_AR_SHELLS, MAX_AR_TRANS,
};
pub use photon::xs::bremsstrahlung_xs::{
    extract_bremsstrahlung_for_gpu, GpuBremsstrahlung, MaterialTtb,
};
pub use photon::xs::compton_doppler_xs::{
    extract_compton_doppler_for_gpu, GpuComptonDoppler, MAX_COMPTON_PZ, MAX_COMPTON_SHELLS,
};
pub use photon::xs::incoherent_form_factor_xs::{
    extract_incoherent_form_factor_for_gpu, GpuIncoherentFormFactor, MAX_INCOHERENT_FF,
};
pub use photon::xs::pair_production_xs::{extract_pair_production_for_gpu, GpuPairProduction};
pub use photon::xs::photon_xs::{extract_photon_material_xs, GpuPhotonXs, MAX_RAYLEIGH_FF};

#[cfg(not(target_os = "macos"))]
use cubecl::{client::ComputeClient, Runtime};
#[cfg(not(target_os = "macos"))]
use cubecl_wgpu::{init_setup, RuntimeOptions, Vulkan, WgpuDevice};
use std::fmt;
#[cfg(not(target_os = "macos"))]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use thiserror::Error;

#[cfg(not(target_os = "macos"))]
pub use cubecl_wgpu::WgpuRuntime;

/// Subset of the underlying adapter info we expose for diagnostics.
/// `backend` and `device_type` are the `Debug` rendering of the wgpu
/// enums (e.g. `"Vulkan"`, `"DiscreteGpu"`) so the type stays portable
/// to platforms where the cubecl/wgpu deps are not built.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
}

#[cfg(not(target_os = "macos"))]
impl From<wgpu::AdapterInfo> for AdapterInfo {
    fn from(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            vendor: info.vendor,
        }
    }
}

impl fmt::Display for AdapterInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} \"{}\" ({}, vendor 0x{:x})",
            self.backend, self.name, self.device_type, self.vendor
        )
    }
}

/// Errors returned by `GpuContext::new`.
#[derive(Debug, Clone, Error)]
pub enum GpuInitError {
    /// No Vulkan adapter exposing `SHADER_F64` was found. Hits on:
    /// - macOS / Apple Silicon (no hardware f64; cubecl deps not built),
    /// - hosts without a Vulkan driver,
    /// - WebGPU/WASM targets,
    /// - some integrated GPUs that omit the optional feature.
    #[error("no GPU with floating point 64 bit compute available")]
    NoF64Adapter,
}

/// Errors from [`GpuContext::with_device`] when an explicit adapter is
/// requested. Kept separate from `GpuInitError` so the common `new()` path
/// (and its many exhaustive `match`es) stays a single-variant enum.
#[derive(Debug, Clone, Error)]
pub enum GpuSelectError {
    /// No usable GPU at all -- same condition as [`GpuInitError::NoF64Adapter`].
    #[error("no GPU with floating point 64 bit compute available")]
    Unavailable,

    /// `with_device(Some(name))` was given a name matching no f64-capable
    /// Vulkan adapter. The message lists the valid names (those returned by
    /// [`list_vulkan_f64_adapters`]). A user-input error, distinct from
    /// `Unavailable` so callers can surface it differently (e.g. a Python
    /// `ValueError` rather than a `RuntimeError`).
    #[error("{0}")]
    AdapterNotFound(String),
}

impl From<GpuInitError> for GpuSelectError {
    fn from(e: GpuInitError) -> Self {
        match e {
            GpuInitError::NoF64Adapter => GpuSelectError::Unavailable,
        }
    }
}

/// A live cubecl-Vulkan setup with verified f64 support. Held by callers
/// for the lifetime of any GPU work.
///
/// Constructing a `GpuContext` is idempotent -- cubecl registers compute
/// clients globally per device, so subsequent `new()` calls reuse the
/// already-initialized client rather than panicking.
///
/// On macOS this struct is uninhabited at runtime: `new()` always
/// returns `NoF64Adapter`, so the `device`/`client` accessors are gated
/// out and any caller code that walks past the `Result::Ok` branch
/// becomes dead.
#[derive(Debug, Clone)]
pub struct GpuContext {
    adapter_info: AdapterInfo,
    #[cfg(not(target_os = "macos"))]
    device: WgpuDevice,
    /// Per-shader-stage storage-buffer descriptor binding limit
    /// reported by the adapter. The `multi_cell_transport` kernel
    /// has a fixed (large) number of bindings; if this value drops
    /// below that count on a host's driver the kernel will be
    /// truncated silently -- descriptors past the limit get aliased
    /// or zeroed, producing wrong results that are easy to mistake
    /// for physics bugs (we hit this exact failure mode going
    /// 78 → 91 bindings while adding inelastic distributions).
    /// Stored here so launchers can panic clearly with a meaningful
    /// message rather than corrupting tallies.
    pub max_storage_buffers_per_stage: u32,
}

impl GpuContext {
    /// Initialize a GPU context on the host's preferred Vulkan adapter,
    /// requiring f64 shader support. Errors with `NoF64Adapter` if no
    /// adapter qualifies -- the caller is expected to fall back to CPU.
    ///
    /// Equivalent to `with_device(None)`, but with the simpler
    /// `GpuInitError` since auto-selection can only fail with "no GPU".
    pub fn new() -> Result<Self, GpuInitError> {
        Self::with_device(None).map_err(|e| match e {
            GpuSelectError::Unavailable => GpuInitError::NoF64Adapter,
            GpuSelectError::AdapterNotFound(_) => {
                unreachable!("auto-select (None) never resolves a named adapter")
            }
        })
    }

    /// Initialize a GPU context, optionally pinned to a specific adapter.
    ///
    /// - `None` auto-selects: cubecl/wgpu's `HighPerformance` preference,
    ///   which prefers a discrete GPU over an integrated one.
    /// - `Some(name)` selects the f64-capable adapter whose name matches
    ///   `name` (exact, case-insensitive; the names are those returned by
    ///   [`list_vulkan_f64_adapters`]). Errors with `AdapterNotFound` if no
    ///   adapter matches -- it never panics, even on a single-GPU host.
    ///
    /// Results are cached per resolved device, so repeated calls (and a mix
    /// of auto + named selections across a process) are cheap clones and
    /// never re-run cubecl's one-time per-device client init.
    pub fn with_device(selector: Option<&str>) -> Result<Self, GpuSelectError> {
        context_for(selector)
    }

    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    #[cfg(not(target_os = "macos"))]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Hand back a cubecl `ComputeClient` for this context's device.
    /// This is the entry point for submitting `#[cube]` kernels -- it's
    /// the type cubecl's launch helpers want as `&self`. The client is
    /// `Arc`-backed internally, so this is a cheap lookup-and-clone.
    #[cfg(not(target_os = "macos"))]
    pub fn client(&self) -> ComputeClient<WgpuRuntime> {
        WgpuRuntime::client(&self.device)
    }
}

/// Per-resolved-device cache of build outcomes, keyed by `device_key`.
#[cfg(not(target_os = "macos"))]
type ContextCache = OnceLock<Mutex<HashMap<(u8, usize), Result<GpuContext, GpuInitError>>>>;

/// Resolve a selector to a device, then return (or build and cache) its
/// context. cubecl registers a compute client per device, so `init_setup`
/// must run at most once per device; we key a cache on the resolved device
/// so repeated `new()`/`with_device` calls are cheap clones. Distinct
/// devices coexist (cubecl IDs them separately), so an interactive session
/// can switch cards.
#[cfg(not(target_os = "macos"))]
fn context_for(selector: Option<&str>) -> Result<GpuContext, GpuSelectError> {
    // The cache holds per-device *build* outcomes (`GpuInitError`); name
    // resolution errors (`GpuSelectError::AdapterNotFound`) are per-selector
    // and returned directly, never cached.
    static CACHE: ContextCache = OnceLock::new();

    let device = resolve_device(selector)?;
    let key = device_key(&device);

    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("GPU context cache mutex poisoned");
    let built = if let Some(existing) = map.get(&key) {
        existing.clone()
    } else {
        let result = build_context(device);
        map.insert(key, result.clone());
        result
    };
    built.map_err(GpuSelectError::from)
}

/// Turn a user selector into the `WgpuDevice` cubecl needs, validating that
/// the chosen adapter actually exists so cubecl's `select_from_adapter_list`
/// never panics.
#[cfg(not(target_os = "macos"))]
fn resolve_device(selector: Option<&str>) -> Result<WgpuDevice, GpuSelectError> {
    match selector {
        // Auto: wgpu's `HighPerformance` preference already prefers a discrete
        // GPU over an integrated one. Pre-flight enumeration so `init_setup`
        // doesn't panic on a host with no f64 Vulkan adapter at all.
        None => {
            if list_vulkan_f64_adapters().is_empty() {
                return Err(GpuSelectError::Unavailable);
            }
            Ok(WgpuDevice::default())
        }
        Some(name) => {
            let adapters: Vec<(String, String, bool)> = enumerate_vulkan_adapters()
                .into_iter()
                .map(|(info, is_f64)| (info.name, info.device_type, is_f64))
                .collect();
            resolve_named_device(name, &adapters)
        }
    }
}

/// Translate a user-supplied adapter name into a `WgpuDevice`. cubecl can't
/// select an adapter by name -- only by (device-type, per-type index) -- so
/// we match the name against our own enumeration and compute that index the
/// same way cubecl's `select_from_adapter_list` does (counting adapters of
/// the same `device_type` in enumeration order). `adapters` is the full
/// Vulkan adapter list: `(name, device_type, is_f64)`, where `device_type`
/// is the `Debug` form of `wgpu::DeviceType` ("DiscreteGpu", "Cpu", ...).
///
/// Pure (no GPU calls) so it's unit-testable on any host.
#[cfg(not(target_os = "macos"))]
fn resolve_named_device(
    selector: &str,
    adapters: &[(String, String, bool)],
) -> Result<WgpuDevice, GpuSelectError> {
    let sel = selector.trim();
    // Exact (case-insensitive) name match among f64-capable adapters; the
    // first wins if two identical cards share a name (they're interchangeable
    // and the name-based API can't tell them apart).
    let matched = adapters
        .iter()
        .position(|(name, _ty, is_f64)| *is_f64 && name.trim().eq_ignore_ascii_case(sel));
    let Some(idx) = matched else {
        let available: Vec<&str> = adapters
            .iter()
            .filter(|(_, _, is_f64)| *is_f64)
            .map(|(name, _, _)| name.as_str())
            .collect();
        return Err(GpuSelectError::AdapterNotFound(format!(
            "no f64-capable GPU adapter named {sel:?}; available: {available:?}"
        )));
    };

    let device_type = &adapters[idx].1;
    let type_index = adapters[..idx]
        .iter()
        .filter(|(_, ty, _)| ty == device_type)
        .count();
    let device = match device_type.as_str() {
        "DiscreteGpu" => WgpuDevice::DiscreteGpu(type_index),
        "IntegratedGpu" => WgpuDevice::IntegratedGpu(type_index),
        "VirtualGpu" => WgpuDevice::VirtualGpu(type_index),
        "Cpu" => WgpuDevice::Cpu,
        other => {
            return Err(GpuSelectError::AdapterNotFound(format!(
                "adapter {sel:?} has unsupported device type {other:?}; \
                 use compute='gpu' to auto-select"
            )));
        }
    };
    Ok(device)
}

/// Stable cache key for a resolved device. We only ever construct the
/// variants below; everything else (incl. `DefaultDevice`) shares the auto
/// slot.
#[cfg(not(target_os = "macos"))]
fn device_key(device: &WgpuDevice) -> (u8, usize) {
    match device {
        WgpuDevice::DiscreteGpu(i) => (0, *i),
        WgpuDevice::IntegratedGpu(i) => (1, *i),
        WgpuDevice::VirtualGpu(i) => (2, *i),
        WgpuDevice::Cpu => (3, 0),
        _ => (4, 0),
    }
}

/// Run cubecl's one-time per-device init and verify f64. Callers pre-validate
/// that `device` exists, so `init_setup` won't panic.
#[cfg(not(target_os = "macos"))]
fn build_context(device: WgpuDevice) -> Result<GpuContext, GpuInitError> {
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());

    // Even a pre-validated adapter is re-checked: the auto path lets cubecl
    // pick, and on an iGPU+dGPU mix it could land on a non-f64 adapter.
    if !setup
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F64)
    {
        return Err(GpuInitError::NoF64Adapter);
    }

    let adapter_info: AdapterInfo = setup.adapter.get_info().into();
    let max_storage_buffers_per_stage = setup.adapter.limits().max_storage_buffers_per_shader_stage;
    Ok(GpuContext {
        device,
        adapter_info,
        max_storage_buffers_per_stage,
    })
}

/// macOS stub: the cubecl/wgpu deps aren't built on macOS, so there is
/// nothing to enumerate. Every call returns `NoF64Adapter`, whatever the
/// selector.
#[cfg(target_os = "macos")]
fn context_for(_selector: Option<&str>) -> Result<GpuContext, GpuSelectError> {
    Err(GpuSelectError::Unavailable)
}

/// Enumerate every Vulkan adapter on the host that exposes `SHADER_F64`.
/// Returns an empty `Vec` on hosts without a Vulkan driver or without an
/// f64-capable adapter -- never an error path. Useful for diagnostics
/// (`--list-adapters`-style output). Always empty on macOS.
#[cfg(not(target_os = "macos"))]
pub fn list_vulkan_f64_adapters() -> Vec<AdapterInfo> {
    enumerate_vulkan_adapters()
        .into_iter()
        .filter(|(_, is_f64)| *is_f64)
        .map(|(info, _)| info)
        .collect()
}

/// Enumerate every Vulkan adapter on the host (f64 or not), in the driver's
/// order, each tagged with whether it exposes `SHADER_F64`. Shared by
/// `list_vulkan_f64_adapters` (which filters to f64) and name resolution
/// (which needs the full list to reproduce cubecl's per-type indexing).
#[cfg(not(target_os = "macos"))]
fn enumerate_vulkan_adapters() -> Vec<(AdapterInfo, bool)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    adapters
        .into_iter()
        .map(|a| {
            let is_f64 = a.features().contains(wgpu::Features::SHADER_F64);
            (a.get_info().into(), is_f64)
        })
        .collect()
}

#[cfg(target_os = "macos")]
pub fn list_vulkan_f64_adapters() -> Vec<AdapterInfo> {
    Vec::new()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_vulkan_f64_adapters_does_not_panic() {
        // Cheap smoke test: enumeration should always return cleanly,
        // even on headless CI without a Vulkan driver (empty Vec).
        let _ = list_vulkan_f64_adapters();
    }

    #[test]
    fn create_gpu_context_or_fail_cleanly() {
        // On hosts with a Vulkan driver exposing SHADER_F64 (most Linux
        // workstations + datacenter GPUs), construction should succeed.
        // Hosts without -- including macOS, headless CI, browser contexts
        // -- return NoF64Adapter cleanly. Both are fine here.
        match GpuContext::new() {
            Ok(ctx) => println!("GPU context ready: {ctx:?}"),
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping (expected on Mac / headless CI)");
            }
        }
    }

    // Touches `client()`, which is gated to non-macOS. macOS gets the
    // adjacent NoF64Adapter coverage from `create_gpu_context_or_fail_cleanly`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_context_yields_compute_client() {
        // Smoke test: once we have a context, asking for its compute
        // client must not panic. This is the entry point every kernel
        // launch will go through, so a regression here breaks all GPU
        // work. Skip cleanly on hosts without a usable GPU.
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let _client = ctx.client();
    }

    #[test]
    fn gpu_context_new_is_idempotent() {
        // Repeated calls must not panic -- the underlying cubecl client
        // is registered globally per device, so re-initializing the same
        // device would otherwise blow up. Verifies the OnceLock cache.
        for _ in 0..3 {
            match GpuContext::new() {
                Ok(_) | Err(GpuInitError::NoF64Adapter) => {}
            }
        }
    }

    // Pure name->WgpuDevice translation. No GPU needed, so these run on any
    // host (incl. headless CI). `resolve_named_device` is the f64-and-name
    // matching cubecl can't do itself.
    #[cfg(not(target_os = "macos"))]
    mod resolve {
        use crate::{resolve_named_device, GpuSelectError};
        use cubecl_wgpu::WgpuDevice;

        fn laptop() -> Vec<(String, String, bool)> {
            vec![
                (
                    "NVIDIA RTX A500 Laptop GPU".into(),
                    "DiscreteGpu".into(),
                    true,
                ),
                (
                    "llvmpipe (LLVM 20.1.2, 256 bits)".into(),
                    "Cpu".into(),
                    true,
                ),
            ]
        }

        #[test]
        fn exact_name_matches_discrete() {
            let d = resolve_named_device("NVIDIA RTX A500 Laptop GPU", &laptop()).unwrap();
            assert!(matches!(d, WgpuDevice::DiscreteGpu(0)));
        }

        #[test]
        fn match_is_case_insensitive_and_trimmed() {
            let d = resolve_named_device("  nvidia rtx a500 laptop gpu  ", &laptop()).unwrap();
            assert!(matches!(d, WgpuDevice::DiscreteGpu(0)));
        }

        #[test]
        fn substring_is_not_a_match() {
            // Exact-only: a partial name is rejected, never fuzzily matched.
            let err = resolve_named_device("NVIDIA", &laptop()).unwrap_err();
            assert!(matches!(err, GpuSelectError::AdapterNotFound(_)));
        }

        #[test]
        fn unknown_name_lists_available() {
            match resolve_named_device("nonsuch", &laptop()).unwrap_err() {
                GpuSelectError::AdapterNotFound(msg) => {
                    assert!(msg.contains("NVIDIA RTX A500 Laptop GPU"), "msg: {msg}");
                    assert!(msg.contains("llvmpipe"), "msg: {msg}");
                }
                other => panic!("expected AdapterNotFound, got {other:?}"),
            }
        }

        #[test]
        fn cpu_adapter_resolves_to_cpu() {
            let d = resolve_named_device("llvmpipe (LLVM 20.1.2, 256 bits)", &laptop()).unwrap();
            assert!(matches!(d, WgpuDevice::Cpu));
        }

        #[test]
        fn per_type_index_counts_same_type_including_non_f64() {
            // Two discrete GPUs; only the second exposes f64. cubecl indexes
            // within the device-type list (which includes the non-f64 card),
            // so the usable card is DiscreteGpu(1), not (0).
            let adapters = vec![
                ("Old Discrete".into(), "DiscreteGpu".into(), false),
                ("New Discrete".into(), "DiscreteGpu".into(), true),
            ];
            let d = resolve_named_device("New Discrete", &adapters).unwrap();
            assert!(matches!(d, WgpuDevice::DiscreteGpu(1)));
        }

        #[test]
        fn integrated_resolves_to_integrated() {
            let adapters = vec![
                ("NVIDIA".into(), "DiscreteGpu".into(), true),
                ("Intel iGPU".into(), "IntegratedGpu".into(), true),
            ];
            let d = resolve_named_device("Intel iGPU", &adapters).unwrap();
            assert!(matches!(d, WgpuDevice::IntegratedGpu(0)));
        }

        #[test]
        fn duplicate_name_takes_first() {
            let adapters = vec![
                ("Twin GPU".into(), "DiscreteGpu".into(), true),
                ("Twin GPU".into(), "DiscreteGpu".into(), true),
            ];
            let d = resolve_named_device("Twin GPU", &adapters).unwrap();
            assert!(matches!(d, WgpuDevice::DiscreteGpu(0)));
        }
    }
}
